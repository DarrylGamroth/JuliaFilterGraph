# Private CPU pupil execution metadata. The original matrix remains authoritative
# for direct checked calls, GPU preparation, and exceptional finite inputs.
struct _PackedPupilReconstruction{T}
    active_rows::Vector{Int}
    coefficients::Matrix{T}
    safe_input_limit::T
end

struct _RowMajorPupilReconstruction
    coefficients::Matrix{Float32}
    safe_input_limit::Float32
end

"""
    DenseReconstructorPlan(matrix)

Prepare a finite, nonempty dense reconstruction matrix. Rows are reconstructed
coordinates and columns follow the semantic ordering declared by the input
Port. The matrix is copied into Julia column-major storage during preparation;
its mathematical row and column order is unchanged.
"""
struct DenseReconstructorPlan{
    T<:Union{Float32,Float64},
    M<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    matrix::M
    # Keep the Plan type stable across dense/packed parameter replacement.
    pupil_execution::Union{Nothing,_PackedPupilReconstruction{T},_RowMajorPupilReconstruction}

    DenseReconstructorPlan{T,M}(
        matrix::M,
        pupil_execution::Union{Nothing,_PackedPupilReconstruction{T},_RowMajorPupilReconstruction}=nothing,
    ) where {
        T<:Union{Float32,Float64},
        M<:AbstractMatrix{T},
    } = new{T,M}(matrix, pupil_execution)
end

function _prepared_dense_reconstructor_plan(matrix::M) where {
    T<:Union{Float32,Float64},
    M<:AbstractMatrix{T},
}
    return DenseReconstructorPlan{T,M}(matrix)
end

function DenseReconstructorPlan(
    matrix::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(matrix)
    isempty(matrix) &&
        throw(ArgumentError("reconstructor matrix must be nonempty"))
    all(isfinite, matrix) ||
        throw(ArgumentError("reconstructor coefficients must be finite"))
    return _prepared_dense_reconstructor_plan(Matrix(matrix))
end

function DenseReconstructorPlan(
    ::Type{T},
    rows::Int,
    columns::Int,
    row_major_values::Union{Nothing,AbstractVector},
) where {T<:Union{Float32,Float64}}
    _validate_dense_reconstructor_dimensions(rows, columns, T)
    matrix = zeros(T, rows, columns)
    if !isnothing(row_major_values)
        Base.require_one_based_indexing(row_major_values)
        length(row_major_values) == Base.Checked.checked_mul(rows, columns) ||
            throw(
                DimensionMismatch(
                    "initial reconstructor extent does not match its dimensions",
                ),
            )
        for row in 1:rows, column in 1:columns
            matrix[row, column] =
                row_major_values[(row - 1) * columns + column]
        end
    end
    return DenseReconstructorPlan(matrix)
end

# Decode bits rather than converting a Float32 subnormal under a caller's DAZ
# mode. Packing is deterministic even if arithmetic modes change before use.
function _float32_coefficient_as_float64(value::Float32)
    bits = reinterpret(UInt32, value)
    exponent = Int((bits >> 23) & 0xff)
    significand = bits & 0x007fffff
    exponent == 0 || (significand |= 0x00800000)
    result = ldexp(Float64(significand), exponent == 0 ? -149 : exponent - 150)
    return iszero(bits & 0x80000000) ? result : -result
end
# The parameter port remains Float32. This matrix owns the packed coefficients
# and presents their decoded values to the ordinary incremental CPU path.
struct _ScaledFloat16Reconstructor <: AbstractMatrix{Float32}
    coefficients::Matrix{Float16}
    scales::Vector{Float32}
end

Base.size(matrix::_ScaledFloat16Reconstructor) = size(matrix.coefficients)
Base.dataids(matrix::_ScaledFloat16Reconstructor) =
    (Base.dataids(matrix.coefficients)..., Base.dataids(matrix.scales)...)

@inline function Base.getindex(matrix::_ScaledFloat16Reconstructor, row::Int, column::Int)
    @boundscheck checkbounds(matrix, row, column)
    @inbounds return Float32(matrix.coefficients[row, column]) * matrix.scales[row]
end

function _scaled_fp16_reconstructor_plan(matrix::AbstractMatrix{Float32})
    Base.require_one_based_indexing(matrix)
    isempty(matrix) && throw(ArgumentError("reconstructor matrix must be nonempty"))
    all(isfinite, matrix) || throw(ArgumentError("reconstructor coefficients must be finite"))
    coefficients = Matrix{Float16}(undef, size(matrix))
    scales = Vector{Float32}(undef, size(matrix, 1))
    for row in axes(matrix, 1)
        peak = maximum(abs ∘ _float32_coefficient_as_float64, view(matrix, row, :))
        shift = iszero(peak) ? 0 : exponent(peak)
        scales[row] = Float32(ldexp(1.0, shift), RoundNearest)
        for column in axes(matrix, 2)
            coefficient = Float16(ldexp(_float32_coefficient_as_float64(matrix[row, column]), -shift), RoundNearest)
            abs(ldexp(Float64(coefficient), shift)) <= Float64(floatmax(Float32)) ||
                throw(ArgumentError("scaled FP16 reconstructor coefficient exceeds Float32 range"))
            coefficients[row, column] = coefficient
        end
    end
    return _prepared_dense_reconstructor_plan(_ScaledFloat16Reconstructor(coefficients, scales))
end

# Stream each packed column contiguously, widening coefficients for Float32 FMA.
function LinearAlgebra.mul!(output::AbstractVector{Float32}, matrix::_ScaledFloat16Reconstructor,
        input::AbstractVector{Float32})
    Base.require_one_based_indexing(output, input)
    length(output) == size(matrix, 1) && length(input) == size(matrix, 2) ||
        throw(DimensionMismatch("scaled reconstructor dimensions"))
    fill!(output, 0f0)
    for column in axes(matrix, 2)
        @inbounds measurement = input[column]
        @inbounds @simd for row in axes(matrix, 1)
            coefficient = Float32(matrix.coefficients[row, column]) * matrix.scales[row]
            output[row] = fma(coefficient, measurement, output[row])
        end
    end
    return output
end

_replace_dense_reconstructor_storage(plan, values) = DenseReconstructorPlan(values)
_replace_dense_reconstructor_storage(
    ::DenseReconstructorPlan{Float32,_ScaledFloat16Reconstructor}, values,
) = _scaled_fp16_reconstructor_plan(values)

function _configure_dense_reconstructor_storage(plan, storage)
    storage == "fp32" && return plan
    storage == "fp16" && return _scaled_fp16_reconstructor_plan(plan.matrix)
    throw(ArgumentError("coefficient_storage must be fp32 or fp16"))
end

"""
Fixed input-ordering scratch for one dense reconstruction Algorithm instance.

The scratch bridges FGN's semantic row-major vectors to Julia matrix Ports.
It is disposable: reconstruction has no committed mathematical state.
"""
struct DenseReconstructorWorkspace{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
}
    ordered_input::V
    pupil_output::V
end

function _prepared_dense_reconstructor_workspace(values::V, pupil_output::V=similar(values, 0)) where {
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
}
    return DenseReconstructorWorkspace{T,V}(values, pupil_output)
end

function DenseReconstructorWorkspace(
    plan::DenseReconstructorPlan{T},
) where {T}
    return _prepared_dense_reconstructor_workspace(
        zeros(T, size(plan.matrix, 2)),
        zeros(T, isnothing(plan.pupil_execution) ? 0 : size(plan.matrix, 1)),
    )
end

commit!(workspace::DenseReconstructorWorkspace) = workspace
discard!(workspace::DenseReconstructorWorkspace) = workspace

function reset!(workspace::DenseReconstructorWorkspace)
    fill!(workspace.ordered_input, zero(eltype(workspace.ordered_input)))
    fill!(workspace.pupil_output, zero(eltype(workspace.pupil_output)))
    return workspace
end

"""Expected numerical rejection from a dense wavefront reconstruction."""
@enum DenseReconstructionFailure::UInt8 begin
    DenseReconstructionNonFiniteInput = 1
    DenseReconstructionNonFiniteOutput = 2
end

@doc "A dense reconstruction input contains a non-finite value." DenseReconstructionNonFiniteInput
@doc "Dense matrix reconstruction produced a non-finite output coordinate." DenseReconstructionNonFiniteOutput

function _validate_dense_reconstructor_dimensions(
    rows::Int,
    columns::Int,
    ::Type{T},
) where {T}
    rows > 0 || throw(ArgumentError("reconstructed extent must be positive"))
    columns > 0 || throw(ArgumentError("reconstructor input extent must be positive"))
    element_count = Base.Checked.checked_mul(rows, columns)
    byte_count = Base.Checked.checked_mul(element_count, sizeof(T))
    UInt64(byte_count) <= UInt64(typemax(UInt32)) || throw(
        ArgumentError("reconstructor byte extent must fit UInt32"),
    )
    return nothing
end

function _validate_dense_reconstruction_storage(
    reconstructed::AbstractVector{T},
    workspace::DenseReconstructorWorkspace{T},
    plan::DenseReconstructorPlan{T},
    input::AbstractMatrix{T},
) where {T}
    matrix = plan.matrix
    length(reconstructed) == size(matrix, 1) || throw(
        DimensionMismatch("reconstructed output has an incompatible extent"),
    )
    length(workspace.ordered_input) == size(matrix, 2) || throw(
        DimensionMismatch("reconstruction Workspace has an incompatible extent"),
    )
    Base.mightalias(reconstructed, input) && throw(
        ArgumentError("reconstructed output must not alias reconstruction input"),
    )
    Base.mightalias(reconstructed, matrix) && throw(
        ArgumentError("reconstructed output must not alias the prepared matrix"),
    )
    Base.mightalias(reconstructed, workspace.ordered_input) && throw(
        ArgumentError("reconstructed output must not alias Workspace scratch"),
    )
    Base.mightalias(input, workspace.ordered_input) && throw(
        ArgumentError("reconstruction input must not alias Workspace scratch"),
    )
    return nothing
end

function _multiply_dense_reconstructor!(
    reconstructed::AbstractVector{T},
    workspace::DenseReconstructorWorkspace{T},
    plan::DenseReconstructorPlan{T},
) where {T}
    mul!(reconstructed, plan.matrix, workspace.ordered_input)
    for value in reconstructed
        isfinite(value) || return DenseReconstructionNonFiniteOutput
    end
    return nothing
end

"""Apply a prepared dense reconstruction matrix to an ordered vector."""
function _multiply_dense_reconstructor!(
    reconstructed::AbstractVector{T},
    plan::DenseReconstructorPlan{T},
    input::AbstractVector{T},
    ::Val{CheckInput}=Val(true),
) where {T,CheckInput}
    if CheckInput
        for value in input
            isfinite(value) || return DenseReconstructionNonFiniteInput
        end
    end
    mul!(reconstructed, plan.matrix, input)
    for value in reconstructed
        isfinite(value) || return DenseReconstructionNonFiniteOutput
    end
    return nothing
end
