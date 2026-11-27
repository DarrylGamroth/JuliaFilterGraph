_projection_borrowing(::CUDA.CuDevice) = Val(true)

# A workgroup-wide collective, including all warps and padded output lanes.
struct _ProjectionAll end
@inline (::_ProjectionAll)(finite) = CUDA.sync_threads_and(finite)
_projection_all(::CUDA.CuDevice) = _ProjectionAll()
function _selection_launch(device::CUDA.CuDevice, backend)
    _require_capture_workgroup(device, PROJECTION_WORKGROUP_SIZE)
    return _native_selection_projection!(backend, PROJECTION_WORKGROUP_SIZE)
end

_projection_backend(::CUDA.CuDevice) = Val(:cublas)

struct CUBLASProjection{Matrix, Scalar}
    matrix::Matrix
    alpha::Scalar
    beta::Scalar
    maxabs::Float64
end

function _prepare_projection(
        ::Val{:cublas},
        device::CUDA.CuDevice,
        matrix,
        output,
        input,
        _lanes::Integer
)
    size(matrix) == (length(output), length(input)) || throw(
        DimensionMismatch(
        "projection size $(size(matrix)) does not match " *
        "$(length(output)) outputs and $(length(input)) inputs",
    ),
    )
    element_types_match = eltype(matrix) === eltype(output) &&
                          eltype(output) === eltype(input)
    element_types_match || throw(
        ArgumentError("captured cuBLAS projection element types must match"),
    )
    return _prepare_projection_execution(
        FilterGraphAlgorithms._cpu_projection_execution(matrix),
        device,
        matrix,
        output,
        input
    )
end

function _prepare_projection_execution(
        ::FilterGraphAlgorithms.DenseProjection,
        device::CUDA.CuDevice,
        matrix,
        output,
        _input
)
    return _prepare_cublas_projection(device, matrix, eltype(output))
end

function _prepare_cublas_projection(device, matrix, ::Type{T}) where {T}
    return CUBLASProjection(
        _capture_array(device, permutedims(matrix)),
        _capture_array(device, fill(one(T), 1)),
        _capture_array(device, fill(zero(T), 1)),
        _coefficient_bound(matrix)
    )
end

function _enqueue_projection!(
        projection::CUBLASProjection,
        output,
        _frame_status,
        input
)
    # cuBLAS cannot predicate GEMV on the device-resident frame status.  The
    # result is only candidate storage: the graph tail gates publication and
    # state adoption on frame_status, and beta=0 overwrites it on the next
    # valid frame.  Device-resident scalars are required for repeatable graph
    # replay with CUDA.jl's device pointer mode.
    CUDA.CUBLAS.gemv!(
        'T',
        projection.alpha,
        projection.matrix,
        input,
        projection.beta,
        output
    )
    return nothing
end

_projection_matrix(projection::CUBLASProjection) = projection.matrix

struct PupilProjection{Projection, Rows, Output, Expand}
    projection::Projection
    row_map::Rows
    packed_output::Output
    expand!::Expand
end

struct PupilRowPacking{Projection<:CUBLASProjection}
    dense::Projection
    active::Vector{Int}
end

@kernel unsafe_indices=true function _expand_pupil_projection!(
        output, frame_status, row_map, packed_output, input
)
    lane = @index(Local, Linear)
    finite = true
    # This projection is prepared only for the pupil reconstructor, after its
    # retained input validator and ordering kernel. A successful status proves
    # every ordered input finite. On rejection, ordering may retain arbitrary
    # old scratch: inspect every original column, including zero coefficients,
    # to preserve dense GEMV's finite/nonfinite zero-row observations.
    if @inbounds !iszero(frame_status[1])
        column = lane
        while column <= length(input)
            @inbounds finite &= isfinite(input[column])
            column += PROJECTION_WORKGROUP_SIZE
        end
    end
    finite = CUDA.sync_threads_and(finite)
    row = lane
    while row <= length(output)
        @inbounds begin
            packed_row = row_map[row]
            output[row] = iszero(packed_row) ?
                          (finite ? zero(eltype(output)) : eltype(output)(NaN)) :
                          packed_output[packed_row]
        end
        row += PROJECTION_WORKGROUP_SIZE
    end
end

function _prepare_pupil_projection(
        device::CUDA.CuDevice, matrix::Matrix{Float32},
        output::AbstractArray{Float32}, input::AbstractArray{Float32}, lanes::Integer,
)
    # The single expansion workgroup writes at most two output values per lane.
    if size(matrix) == (length(output), length(input)) &&
            length(output) <= 2 * PROJECTION_WORKGROUP_SIZE
        active = FilterGraphAlgorithms._packable_pupil_rows(matrix)
        if active !== nothing
            # At n*eps(Float32) ≤ 1/8, even 2n rounds amplify the absolute
            # products by at most 4/3. Float64 addition error and Float32
            # subnormals lost to host DAZ add less than 1e-9 to the row bound.
            # A bound of 1/2 leaves both GEMV layouts below Float32 overflow.
            columns = size(matrix, 2)
            if columns * eps(Float32) <= 0.125f0 &&
                    FilterGraphAlgorithms._pupil_reconstruction_row_bound(matrix, active) <= 0.5
                packed = _prepare_cublas_projection(device, matrix[active, :], Float32)
                return _expand_packed_pupil(device, packed, active, output)
            end
            dense = _prepare_cublas_projection(device, matrix, Float32)
            return PupilRowPacking(dense, active)
        end
    end
    return _prepare_projection(device, matrix, output, input, lanes)
end

function _expand_packed_pupil(device, projection::CUBLASProjection, active, output)
    row_map = zeros(Int32, length(output))
    row_map[active] .= Int32.(eachindex(active))
    packed_output = _capture_zeros(device, Float32, length(active))
    backend = KernelAbstractions.get_backend(output)
    return PupilProjection(
        projection, _capture_array(device, row_map), packed_output,
        _expand_pupil_projection!(backend, PROJECTION_WORKGROUP_SIZE),
    )
end

# A graph can prove a tighter input bound before it captures any work.
function _prepare_bounded_pupil_projection(projection::PupilRowPacking, output)
    dense, active = projection.dense, projection.active
    # The retained matrix already stores logical rows as contiguous columns.
    packed = CUBLASProjection(dense.matrix[:, active], dense.alpha, dense.beta, dense.maxabs)
    return _expand_packed_pupil(CUDA.device(dense.matrix), packed, active, output)
end

_enqueue_projection!(projection::PupilRowPacking, output, frame_status, input) =
    _enqueue_projection!(projection.dense, output, frame_status, input)

function _enqueue_projection!(
        projection::PupilProjection, output, frame_status, input
)
    _enqueue_projection!(
        projection.projection, projection.packed_output, frame_status, input,
    )
    # Like the original cuBLAS GEMV, expansion writes disposable candidates even
    # on an already rejected frame. Publication remains the graph's operation.
    projection.expand!(
        output, frame_status, projection.row_map, projection.packed_output, input;
        ndrange = PROJECTION_WORKGROUP_SIZE,
    )
    return nothing
end

_projection_matrix(projection::PupilProjection) =
    _projection_matrix(projection.projection)
_projection_matrix(projection::PupilRowPacking) = _projection_matrix(projection.dense)


_projection_bound(projection::CUBLASProjection, input::_FiniteBound, count) =
    CUDA.CUDACore.math_mode() == CUDA.CUDACore.FAST_MATH ? nothing :
    _projection_bound(projection.maxabs, count, input)
_projection_bound(projection::PupilProjection, input::_FiniteBound, count) =
    _projection_bound(projection.projection, input, count)
_projection_bound(projection::PupilRowPacking, input::_FiniteBound, count) =
    _projection_bound(projection.dense, input, count)

# An exact identity of a proven finite buffer cannot overflow.
_identity_validation(::IdentityProjection, validate!) = _FiniteInput()
