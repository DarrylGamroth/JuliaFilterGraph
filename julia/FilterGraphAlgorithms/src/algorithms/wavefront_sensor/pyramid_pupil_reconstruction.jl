function _packable_pupil_rows(matrix::Matrix{Float32})
    isempty(matrix) && return nothing
    rows, columns = size(matrix)
    # Expansion still touches every output row. Omit at least half the rows
    # and save at least 512 KiB of coefficient reads per multiplication.
    minimum_saved_coefficients = (1 << 19) ÷ sizeof(Float32)
    rows_to_omit = cld(minimum_saved_coefficients, columns)
    maximum_active = min(rows ÷ 2, rows - rows_to_omit)
    maximum_active > 0 || return nothing
    active = Int[]
    for row in 1:rows
        nonzero = false
        negative_zero = false
        for column in 1:columns
            bits = reinterpret(UInt32, matrix[row, column])
            if !iszero(bits & 0x7fffffff)
                nonzero = true
                break
            end
            negative_zero |= bits == 0x80000000
        end
        if nonzero
            push!(active, row)
            length(active) <= maximum_active || return nothing
        elseif negative_zero
            return nothing
        end
    end
    return isempty(active) ? nothing : active
end

function _pupil_reconstruction_row_bound(matrix::Matrix{Float32}, rows)
    bound = 0.0
    for row in rows
        total = 0.0
        for column in axes(matrix, 2)
            total += abs(Float64(matrix[row, column]))
        end
        bound = max(bound, total)
    end
    return bound
end

_prepare_pyramid_pupil_reconstructor(plan::DenseReconstructorPlan, _) = plan

function _prepare_pyramid_pupil_reconstructor(
    plan::DenseReconstructorPlan{Float32,Matrix{Float32}},
    ::Union{Nothing,_PackedPupilReconstruction{Float32}},
)
    matrix = plan.matrix
    columns = size(matrix, 2)
    # Packing can select another vendor GEMV kernel/reduction order. Restrict
    # it to inputs for which products and every partial absolute sum fit with
    # eightfold headroom. Extreme finite inputs use the original matrix and
    # original leading dimension, retaining its overflow/failure observations.
    rounding = columns * Float64(eps(Float32))
    rounding < 0.25 || return plan
    active = _packable_pupil_rows(matrix)
    isnothing(active) && return plan
    row_bound = _pupil_reconstruction_row_bound(matrix, active)
    limit = Float32((Float64(floatmax(Float32)) / 8 / max(1.0, row_bound)) * (1 - rounding))
    execution = _PackedPupilReconstruction(active, matrix[active, :], limit)
    return DenseReconstructorPlan{Float32,Matrix{Float32}}(matrix, execution)
end

function _prepare_row_major_pupil_reconstructor(plan::DenseReconstructorPlan{Float32,Matrix{Float32}})
    # Bound the difference between the two reduction orders by the absolute
    # pupil tolerance (1e-7), including near cancellation. eps(Float32)
    # allows for a rounded product and addition per term.
    rounding_error = size(plan.matrix, 2) * Float64(eps(Float32))
    rounding_error < 1 || return plan
    gamma = rounding_error / (1 - rounding_error)
    coefficients = permutedims(plan.matrix)
    magnitude = get_zero_subnormals() ? abs ∘ _float32_coefficient_as_float64 : abs ∘ Float64
    row_bound = maximum(row -> sum(magnitude, row), eachcol(coefficients))
    limit = prevfloat(Float32(1e-7 / (2 * gamma * row_bound)))
    execution = _RowMajorPupilReconstruction(coefficients, limit)
    return DenseReconstructorPlan{Float32,Matrix{Float32}}(plan.matrix, execution)
end

_prepare_pyramid_pupil_reconstructor(
    plan::DenseReconstructorPlan{Float32,Matrix{Float32}}, ::_RowMajorPupilReconstruction,
) = _prepare_row_major_pupil_reconstructor(plan)

function _replace_pyramid_pupil_reconstructor(plan, values)
    replacement = _replace_dense_reconstructor_storage(plan, values)
    return _prepare_pyramid_pupil_reconstructor(replacement, plan.pupil_execution)
end

_pupil_reconstruction_input_limit(::Nothing) = Inf32
_pupil_reconstruction_input_limit(execution::_PackedPupilReconstruction) = execution.safe_input_limit
_pupil_reconstruction_input_limit(execution::_RowMajorPupilReconstruction) = execution.safe_input_limit

_validate_pupil_reconstruction_storage(reconstructed, workspace, plan, ::Nothing) = nothing

function _validate_pupil_reconstruction_storage(
    reconstructed, workspace, plan, execution::_PackedPupilReconstruction,
)
    scratch = workspace.pupil_output
    length(scratch) >= length(execution.active_rows) || throw(
        DimensionMismatch("pupil reconstruction Workspace has an incompatible extent"),
    )
    Base.require_one_based_indexing(reconstructed, scratch)
    (Base.mightalias(reconstructed, scratch) ||
     Base.mightalias(workspace.ordered_input, scratch) ||
     Base.mightalias(reconstructed, execution.coefficients) ||
     Base.mightalias(reconstructed, execution.active_rows) ||
     Base.mightalias(scratch, plan.matrix) ||
     Base.mightalias(scratch, execution.coefficients) ||
     Base.mightalias(scratch, execution.active_rows) ||
     Base.mightalias(workspace.ordered_input, plan.matrix) ||
     Base.mightalias(workspace.ordered_input, execution.coefficients) ||
     Base.mightalias(workspace.ordered_input, execution.active_rows)) && throw(
        ArgumentError("pupil reconstruction output and scratch must not alias preparation storage"),
    )
    return nothing
end

function _validate_pupil_reconstruction_storage(
    reconstructed, workspace, plan, execution::_RowMajorPupilReconstruction,
)
    Base.require_one_based_indexing(reconstructed, workspace.ordered_input)
    (Base.mightalias(reconstructed, execution.coefficients) ||
     Base.mightalias(workspace.ordered_input, plan.matrix) ||
     Base.mightalias(workspace.ordered_input, execution.coefficients)) && throw(
        ArgumentError("pupil reconstruction output and scratch must not alias preparation storage"),
    )
    return nothing
end

function _multiply_pyramid_pupil_reconstructor!(
    reconstructed::AbstractVector,
    workspace,
    plan,
    ::Union{Nothing,_RowMajorPupilReconstruction},
    safe,
)
    return _multiply_dense_reconstructor!(reconstructed, workspace, plan)
end

function _multiply_pyramid_pupil_reconstructor!(
    reconstructed, workspace, plan, execution::_PackedPupilReconstruction, safe,
)
    scratch = workspace.pupil_output
    # Zero-row expansion and the overflow bound are qualified for the default
    # host arithmetic mode. Check at execution: the caller can change it after
    # preparation, including modes that give a dense zero row a negative sign.
    (safe && !get_zero_subnormals() && rounding(Float32) === RoundNearest) ||
        return _multiply_dense_reconstructor!(reconstructed, workspace, plan)
    # The view borrows fixed Workspace storage; all original input columns and
    # stored active-row coefficients participate in the vendor GEMV unchanged.
    mul!(@view(scratch[1:length(execution.active_rows)]),
        execution.coefficients, workspace.ordered_input)
    fill!(reconstructed, zero(eltype(reconstructed)))
    for row in eachindex(execution.active_rows)
        reconstructed[execution.active_rows[row]] = scratch[row]
    end
    for value in reconstructed
        isfinite(value) || return DenseReconstructionNonFiniteOutput
    end
    return nothing
end

# Each packed column holds a complete reconstructor row. Reuse each input
# load for four rows so the compiler can form independent SIMD accumulators.
# The checked pupil path supplies one-based, disjoint storage with matching extents.
function _multiply_row_major_pupil_reconstructor!(output, packed, input)
    columns, rows = size(packed)
    blocked_rows = rows - rem(rows, 4)
    @inbounds for row in 1:4:blocked_rows
        dot0 = dot1 = dot2 = dot3 = 0f0
        @simd for column in 1:columns
            measurement = input[column]
            dot0 = muladd(packed[column, row], measurement, dot0)
            dot1 = muladd(packed[column, row + 1], measurement, dot1)
            dot2 = muladd(packed[column, row + 2], measurement, dot2)
            dot3 = muladd(packed[column, row + 3], measurement, dot3)
        end
        output[row] = dot0
        output[row + 1] = dot1
        output[row + 2] = dot2
        output[row + 3] = dot3
    end
    @inbounds for row in blocked_rows + 1:rows
        dot = 0f0
        @simd for column in 1:columns
            dot = muladd(packed[column, row], input[column], dot)
        end
        output[row] = dot
    end
    return output
end

function _multiply_pyramid_pupil_reconstructor!(
    reconstructed::Vector{Float32}, workspace, plan, execution::_RowMajorPupilReconstruction, safe,
)
    (safe && !get_zero_subnormals() && rounding(Float32) === RoundNearest) ||
        return _multiply_dense_reconstructor!(reconstructed, workspace, plan)
    _multiply_row_major_pupil_reconstructor!(
        reconstructed, execution.coefficients, workspace.ordered_input,
    )
    for value in reconstructed
        isfinite(value) || return DenseReconstructionNonFiniteOutput
    end
    return nothing
end

function _process_pyramid_pupil_reconstruction!(
    reconstructed::AbstractVector{T},
    workspace::DenseReconstructorWorkspace{T},
    plan::DenseReconstructorPlan{T},
    reconstruction_pixels::AbstractMatrix{T},
) where {T}
    _validate_dense_reconstruction_storage(
        reconstructed,
        workspace,
        plan,
        reconstruction_pixels,
    )
    _validate_pupil_reconstruction_storage(
        reconstructed, workspace, plan, plan.pupil_execution,
    )
    Base.mightalias(reconstruction_pixels, workspace.pupil_output) && throw(
        ArgumentError("reconstruction input must not alias pupil Workspace scratch"),
    )
    pixel_count = size(plan.matrix, 2)
    rem(pixel_count, PYRAMID_PUPIL_COUNT) == 0 || throw(
        DimensionMismatch("pyramid reconstructor must contain four pupil blocks"),
    )
    selected_pixel_count = div(pixel_count, PYRAMID_PUPIL_COUNT)
    size(reconstruction_pixels) ==
        (PYRAMID_PUPIL_COUNT, selected_pixel_count) || throw(
        DimensionMismatch("pyramid reconstruction input has an incompatible shape"),
    )

    input_index = 1
    safe = true
    input_limit = _pupil_reconstruction_input_limit(plan.pupil_execution)
    for pupil in 1:PYRAMID_PUPIL_COUNT
        for pixel in 1:selected_pixel_count
            value = reconstruction_pixels[pupil, pixel]
            isfinite(value) || return DenseReconstructionNonFiniteInput
            safe &= abs(value) <= input_limit
            workspace.ordered_input[input_index] = value
            input_index += 1
        end
    end
    return _multiply_pyramid_pupil_reconstructor!(
        reconstructed, workspace, plan, plan.pupil_execution, safe,
    )
end

"""
Map selected pixels from four pyramid pupils through a dense reconstructor
to a Float32 reconstructed vector. Matrix coefficients define the conversion
from pupil pixel units to output units.
"""
@algorithm PyramidPupilReconstructorF32 begin
    label = "pwfs-pupil-reconstructor-f32"
    failure = DenseReconstructionFailure
    operation = DenseF32Operation()

    """
    Set selected pixel count per pupil, reconstructed count, and output schema.
    The optional initial dense matrix has one column per selected pixel across
    the four pupils and can be replaced through the reconstructor parameter port.
    Set `coefficient_storage = "fp16"` for row-scaled Float16 storage with Float32 arithmetic.
    Set `coefficient_storage = "fp32-row-major"` to opt in to row-major CPU SIMD.
    This retains a second Float32 matrix when the accumulation length permits a
    bound on the difference between reduction orders. SIMD requires input
    magnitudes that bound this difference to 1e-7 per reconstructed coordinate,
    a plain Float32 output vector, and round-to-nearest arithmetic with subnormal
    flushing disabled. Other calls use the ordinary Float32 matrix.
    """
    @configuration PyramidPupilReconstructorF32Configuration begin
        reconstructed_count::Int => graph_rebuild
        selected_pixel_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        reconstructed_schema::String => graph_rebuild
        (coefficient_storage::String = "fp32") => graph_rebuild
    end

    ports(configuration) = (
        input(
            :reconstruction_pixels,
            schema_element_type(PYRAMID_RECONSTRUCTION_PIXELS_V1),
            (PYRAMID_PUPIL_COUNT, configuration.selected_pixel_count),
            PYRAMID_RECONSTRUCTION_PIXELS_V1;
            name="reconstruction-pixels",
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:reconstruction_pixels,
        ),
        parameter(
            :reconstructor,
            schema_element_type(PWFS_RECONSTRUCTOR_V1),
            (
                configuration.reconstructed_count,
                Base.Checked.checked_mul(
                    configuration.selected_pixel_count,
                    PYRAMID_PUPIL_COUNT,
                ),
            ),
            PWFS_RECONSTRUCTOR_V1;
            replace=_replace_pyramid_pupil_reconstructor,
        ),
    )

    prepare(configuration) = begin
        reconstruction_pixel_count = Base.Checked.checked_mul(
            configuration.selected_pixel_count,
            PYRAMID_PUPIL_COUNT,
        )
        plan = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            reconstruction_pixel_count,
            configuration.initial_reconstructor,
        )
        storage = configuration.coefficient_storage
        if storage == "fp32-row-major"
            plan = _prepare_row_major_pupil_reconstructor(plan)
        elseif storage == "fp32" || storage == "fp16"
            plan = _prepare_pyramid_pupil_reconstructor(
                _configure_dense_reconstructor_storage(plan, storage), nothing,
            )
        else
            throw(ArgumentError("coefficient_storage must be fp32, fp16, or fp32-row-major"))
        end
        # Replacement preserves the Workspace even when the active row count
        # changes. Reserve the full logical extent once during preparation.
        workspace = _prepared_dense_reconstructor_workspace(
            zeros(Float32, reconstruction_pixel_count),
            zeros(Float32, configuration.reconstructed_count),
        )
        (plan=plan, workspace=workspace)
    end

    process!(plan, workspace) = begin
        _process_pyramid_pupil_reconstruction!(
            reconstructed,
            workspace,
            plan,
            reconstruction_pixels,
        )
    end
end
