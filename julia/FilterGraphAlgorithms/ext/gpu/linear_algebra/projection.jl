const PROJECTION_WORKGROUP_SIZE = 256
const COOPERATIVE_PROJECTION_DEFAULT_LANES = 128
const PROJECTION_VALID = Int32(0)

# These fixed binary trees are unrolled for complete power-of-two tiles. A
# device accepting another launch size does not make that reduction correct.
function _require_reduction_lanes(lanes::Integer)
    lanes in (32, 64, 128, 256) || throw(
        ArgumentError("cooperative reduction lanes must be 32, 64, 128 or 256"),
    )
    return nothing
end

@inline _lane_count(::Val{Lanes}) where {Lanes} = Lanes

function _failure_code(node_index::Int32, failure::Enum)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))
    node_index <= (typemax(Int32) >> 8) || throw(
        ArgumentError("captured node index is too large to encode"),
    )
    return (node_index << 8) | Int32(failure)
end

@kernel function _identity_projection!(output, frame_status, input)
    index = @index(Global, Linear)
    if index <= length(output) && @inbounds(iszero(frame_status[1]))
        # Copy rather than multiply so identity execution preserves the input's
        # zero sign and subnormal bits, as on CPU. Nonfinite intermediates also
        # pass through: the existing final validation must observe them.
        @inbounds output[index] = input[index]
    end
end

@kernel function _clear_projection_scratch!(scratch)
    index = @index(Global, Linear)
    index <= length(scratch) && @inbounds(scratch[index] = zero(eltype(scratch)))
end

@kernel function _projection!(output, frame_status, projection, input)
    row = @index(Global, Linear)
    if row <= length(output) && @inbounds(iszero(frame_status[1]))
        value = zero(eltype(output))
        for column in axes(projection, 2)
            @inbounds input_value = input[column]
            @inbounds value = muladd(projection[row, column], input_value, value)
        end
        @inbounds output[row] = value
    end
end

@inline function _reduce_sum!(
    local_sums,
    local_index::Int,
    stride::Int,
)
    if local_index <= stride
        @inbounds local_sums[local_index] += local_sums[local_index + stride]
    end
    return nothing
end

@inline _projection_coefficient(matrix, column, row) = @inbounds matrix[column, row]
@inline _projection_coefficient(matrix::NamedTuple, column, row) =
    @inbounds Float32(matrix.coefficients[column, row]) * matrix.scales[row]

@kernel function _transposed_projection!(
    output,
    frame_status,
    transposed_projection,
    input,
    ::Val{Lanes},
) where {Lanes}
    local_index = @index(Local, Linear)
    _, row = @index(Group, NTuple)
    local_sums = @localmem eltype(output) (Lanes,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    value = zero(eltype(output))
    column = Int(local_index)
    while frame_valid && column <= length(input)
        @inbounds value = muladd(
            _projection_coefficient(transposed_projection, column, row),
            input[column],
            value,
        )
        column += Lanes
    end
    @inbounds local_sums[local_index] = value
    @synchronize

    if Lanes == 256
        _reduce_sum!(local_sums, Int(local_index), 128)
        @synchronize
    end
    if Lanes >= 128
        _reduce_sum!(local_sums, Int(local_index), 64)
        @synchronize
    end
    if Lanes >= 64
        _reduce_sum!(local_sums, Int(local_index), 32)
        @synchronize
    end
    _reduce_sum!(local_sums, Int(local_index), 16)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 8)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 4)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 2)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 1)
    # Lane one is the only subsequent reader of its own final result.

    if frame_valid && local_index == 1
        @inbounds output[row] = local_sums[1]
    end
end

# Dense projection matrices are transposed once during cold preparation so each
# output dot product streams contiguous matrix values.  Extensions may select a
# vendor implementation without changing the algorithm adapters or their
# public configuration.
_projection_backend(::Any) = Val(:ka)

struct IdentityProjection{Launch}
    copy!::Launch
end

# Borrowing internal scratch is independently qualified from validation reuse.
# Backend-specific cold checks may repair incomplete array-library alias traits
# without changing Base or weakening another backend's conservative default.
_capture_arrays_overlap(left, right) = Base.mightalias(left, right)
_projection_borrowing(device) = Val(false)
_capture_projection_input(device, matrix, input, output) =
    _capture_projection_input(_projection_borrowing(device), device, matrix, input, output)
_capture_projection_input(::Val{false}, device, matrix, input, output) =
    _capture_zeros(device, eltype(input), size(matrix, 1))

function _capture_projection_input(::Val{true}, device, matrix, input, output)
    return _capture_projection_input(
        FilterGraphAlgorithms._cpu_projection_execution(matrix), device, matrix, input, output,
    )
end

_capture_projection_input(execution, device, matrix, input, output) =
    _capture_zeros(device, eltype(input), size(matrix, 1))

function _capture_projection_input(
    ::FilterGraphAlgorithms.IdentityProjection, device, matrix, input, output,
)
    # Keep the private intermediate when callers request an in-place result:
    # a subsequent dense projection cannot read and write the same storage safely.
    return _capture_arrays_overlap(input, output) ?
           _capture_zeros(device, eltype(input), size(matrix, 1)) : input
end

struct KAProjection{Launch,Matrix,Lanes}
    launch!::Launch
    matrix::Matrix
    lanes::Lanes
end

function _prepare_projection(
    ::Val{:ka},
    device,
    matrix,
    output,
    input,
    lanes::Integer,
)
    size(matrix) == (length(output), length(input)) || throw(
        DimensionMismatch(
            "projection size $(size(matrix)) does not match " *
            "$(length(output)) outputs and $(length(input)) inputs",
        ),
    )
    _require_reduction_lanes(lanes)
    # Mixed element types retain the original arithmetic fallback.
    execution = eltype(matrix) === eltype(output) === eltype(input) ?
        FilterGraphAlgorithms._cpu_projection_execution(matrix) :
        FilterGraphAlgorithms.DenseProjection()
    return _prepare_ka_projection(execution, device, matrix, output, input, lanes)
end

_prepare_ka_projection(execution, device, matrix, output, input, lanes) =
    _prepare_projection_execution(execution, device, matrix, output, input)

function _prepare_ka_projection(
    ::FilterGraphAlgorithms.DenseProjection, device, matrix, output, input, lanes,
)
    backend = KernelAbstractions.get_backend(output)
    return KAProjection(
        _transposed_projection!(backend, (lanes, 1)),
        _capture_array(device, permutedims(matrix)),
        Val(lanes),
    )
end

function _prepare_projection(
    device, matrix::FilterGraphAlgorithms._ScaledFloat16Reconstructor, output, input, lanes::Integer,
)
    size(matrix) == (length(output), length(input)) ||
        throw(DimensionMismatch("scaled reconstructor dimensions"))
    _require_reduction_lanes(lanes)
    backend = KernelAbstractions.get_backend(output)
    storage = _capture_transposed_coefficients(device, matrix, axes(matrix, 2))
    return KAProjection(_transposed_projection!(backend, (lanes, 1)), storage, Val(lanes))
end

function _capture_transposed_coefficients(device, matrix, columns)
    return _capture_array(device, permutedims(view(matrix, :, columns)))
end

function _capture_transposed_coefficients(device, matrix::FilterGraphAlgorithms._ScaledFloat16Reconstructor, columns)
    return (coefficients=_capture_array(device, permutedims(view(matrix.coefficients, :, columns))),
        scales=_capture_array(device, matrix.scales))
end

function _prepare_projection(
    device,
    matrix,
    output,
    input,
    lanes::Integer,
)
    return _prepare_projection(
        _projection_backend(device),
        device,
        matrix,
        output,
        input,
        lanes,
    )
end

function _enqueue_projection!(
    projection::KAProjection,
    output,
    frame_status,
    input,
)
    projection.launch!(
        output,
        frame_status,
        projection.matrix,
        input,
        projection.lanes;
        ndrange=(
            _lane_count(projection.lanes),
            length(output),
        ),
    )
    return nothing
end

_projection_matrix(projection::KAProjection) =
    projection.matrix

# Projections retain their output validator unless a backend proves an exact identity.
_identity_validation(projection, validate!) = validate!

struct SelectionProjection{Launch, Columns, Collective}
    gather!::Launch
    columns::Columns
    all_lanes::Collective
end

# All hardware lanes must reach the workgroup collective, including lanes
# outside the final output tile. Compute indices from group/local coordinates
# as required by unsafe_indices, with explicit bounds checks below.
@kernel unsafe_indices=true function _native_selection_projection!(
        output, frame_status, columns, input, all_lanes
)
    local_index = @index(Local, Linear)
    group_index = @index(Group, Linear)
    index = (group_index - 1) * PROJECTION_WORKGROUP_SIZE + local_index
    frame_valid = @inbounds iszero(frame_status[1])
    finite = true
    input_index = local_index
    while frame_valid && input_index <= length(input)
        @inbounds finite &= isfinite(input[input_index])
        input_index += PROJECTION_WORKGROUP_SIZE
    end
    # A selection may consume unchecked arithmetic scratch. Preserve dense
    # multiplication's rejection of nonfinite values even in unselected
    # columns, without adding a launch or changing final failure attribution.
    finite = all_lanes(finite)
    if frame_valid && index <= length(output)
        @inbounds output[index] = finite ? input[columns[index]] : eltype(output)(NaN)
    end
end

function _prepare_projection_execution(
        ::FilterGraphAlgorithms.IdentityProjection,
        _device,
        _matrix,
        output,
        input
)
    output === input && return IdentityProjection(nothing)
    _require_capture_workgroup(_device, PROJECTION_WORKGROUP_SIZE)
    backend = KernelAbstractions.get_backend(output)
    return IdentityProjection(
        _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE),
    )
end

function _prepare_projection_execution(
        execution::FilterGraphAlgorithms.SelectionProjection,
        device,
        _matrix,
        output,
        _input
)
    backend = KernelAbstractions.get_backend(output)
    return SelectionProjection(
        _selection_launch(device, backend),
        _capture_array(device, execution.columns),
        _projection_all(device),
    )
end

function _enqueue_projection!(
        projection::IdentityProjection,
        output,
        frame_status,
        input
)
    projection.copy!(output, frame_status, input; ndrange = length(output))
    return nothing
end

_enqueue_projection!(::IdentityProjection{Nothing}, output, frame_status, input) =
    nothing

_projection_matrix(::IdentityProjection) = nothing

function _enqueue_projection!(
        projection::SelectionProjection,
        output,
        frame_status,
        input
)
    projection.gather!(
        output, frame_status, projection.columns, input, projection.all_lanes;
        # CPU lowering also needs valid indices for every reduction lane.
        ndrange = cld(length(output), PROJECTION_WORKGROUP_SIZE) * PROJECTION_WORKGROUP_SIZE,
    )
    return nothing
end

_projection_matrix(::SelectionProjection) = nothing

_projection_all(device) = nothing
function _selection_launch(device, backend)
    _require_capture_workgroup(device, PROJECTION_WORKGROUP_SIZE;
        shared_memory=PROJECTION_WORKGROUP_SIZE * sizeof(Bool))
    return _selection_projection!(backend, PROJECTION_WORKGROUP_SIZE)
end

# Every physical lane participates, including padded lanes in the final tile.
@kernel unsafe_indices=true function _selection_projection!(
    output, frame_status, columns, input, _all_lanes,
)
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    flags = @localmem Bool (PROJECTION_WORKGROUP_SIZE,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    finite = true
    column = lane
    while frame_valid && column <= length(input)
        @inbounds finite &= isfinite(input[column])
        column += PROJECTION_WORKGROUP_SIZE
    end
    @inbounds flags[lane] = finite
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds flags[lane] &= flags[lane + stride]
        end
        @synchronize
    end
    # Materialize in the publication region: KA 0.9 CPU lowering does not carry
    # this derived lane index across the nested synchronized reduction loop.
    index = (group - 1) * PROJECTION_WORKGROUP_SIZE + lane
    if frame_valid && index <= length(output)
        @inbounds output[index] = flags[1] ? input[columns[index]] : eltype(output)(NaN)
    end
end

# The enclosing adapter still gates alias metadata on validation reuse.
FilterGraphAlgorithms.alias_input(::IdentityProjection, ::Val{1}) = 1

_coefficient_bound(matrix) = _supported_bound_arithmetic() ?
    Float64(maximum(abs, matrix)) : Inf

function _projection_bound(maximum_coefficient::Float64, count::Int, input::_FiniteBound)
    _supported_bound_arithmetic() || return nothing
    isfinite(maximum_coefficient) && maximum_coefficient >= 0 || return nothing
    # Product/reduction rounding allowance is < 4 for this extent, using eps
    # rather than half-eps as the rounding unit. Larger extents remain checked.
    0 < count <= (1 << 20) || return nothing
    count_exponent = ndigits(count - 1; base=2)
    count == 1 && (count_exponent = 0)
    product_bound = ldexp(maximum_coefficient * input.magnitude, count_exponent + 2)
    underflow_bound = ldexp(Float64(floatmin(Float32)), count_exponent + 2)
    magnitude = _power_bound(max(product_bound, underflow_bound))
    isfinite(magnitude) && magnitude < Float64(floatmax(Float32)) || return nothing
    return _FiniteBound(magnitude)
end

_projection_bound(projection, input::_FiniteBound, count) = nothing
_projection_bound(::IdentityProjection, input::_FiniteBound, count) = input
_projection_bound(::SelectionProjection, input::_FiniteBound, count) = input
