"""
    VectorLinearCombinationPlan(extent, left_scale, right_scale)

Prepare `output[i] = left_scale * left[i] + right_scale * right[i]` for
equal-length Float32 vectors. Both scales are finite and dimensionless; the
vectors share one unit, coordinate order and schema. No conversion is inferred.
On the CPU, the right product is rounded first, then the left product and sum are fused.
"""
struct VectorLinearCombinationPlan <: AbstractAlgorithmPlan
    extent::Int
    left_scale::Float32
    right_scale::Float32

    function VectorLinearCombinationPlan(
        extent::Int, left_scale::Float32, right_scale::Float32,
    )
        0 < extent <= typemax(UInt32) ÷ sizeof(Float32) || throw(
            ArgumentError("vector byte extent must be positive and fit UInt32"),
        )
        isfinite(left_scale) && isfinite(right_scale) || throw(
            ArgumentError("vector combination scales must be finite"),
        )
        return new(extent, left_scale, right_scale)
    end
end

"""Fixed disposable output scratch for a vector linear combination."""
struct VectorLinearCombinationWorkspace
    candidate_output::Vector{Float32}
end

VectorLinearCombinationWorkspace(plan::VectorLinearCombinationPlan) =
    VectorLinearCombinationWorkspace(zeros(Float32, plan.extent))

commit!(workspace::VectorLinearCombinationWorkspace) = workspace
discard!(workspace::VectorLinearCombinationWorkspace) = workspace

function reset!(workspace::VectorLinearCombinationWorkspace)
    fill!(workspace.candidate_output, 0.0f0)
    return workspace
end

"""Expected numerical rejection from a vector linear combination."""
@enum VectorLinearCombinationFailure::UInt8 begin
    VectorLinearCombinationNonFiniteInput = 1
    VectorLinearCombinationNonFiniteOutput = 2
end

@doc "Either input contains a non-finite value, including a zero-scaled input." VectorLinearCombinationNonFiniteInput
@doc "A scaled sum produced a non-finite output value." VectorLinearCombinationNonFiniteOutput

function process!(
    output::AbstractVector{Float32},
    workspace::VectorLinearCombinationWorkspace,
    plan::VectorLinearCombinationPlan,
    left::AbstractVector{Float32},
    right::AbstractVector{Float32},
)
    return _process_vector_combination!(output, workspace, plan, left, right, Val(true))
end

# For finite inputs, one term is zero and the other has magnitude no greater
# than its input. This also bounds the rounded right product before the FMA.
_finite_combination_result(plan::VectorLinearCombinationPlan) =
    (iszero(plan.right_scale) && abs(plan.left_scale) <= 1f0) ||
    (iszero(plan.left_scale) && abs(plan.right_scale) <= 1f0)

function _process_vector_combination!(
    output::AbstractVector{Float32},
    workspace::VectorLinearCombinationWorkspace,
    plan::VectorLinearCombinationPlan,
    left::AbstractVector{Float32},
    right::AbstractVector{Float32},
    ::Val{CheckFinite},
) where {CheckFinite}
    candidate = workspace.candidate_output
    Base.require_one_based_indexing(output, left, right)
    length(output) == length(left) == length(right) == length(candidate) == plan.extent ||
        throw(DimensionMismatch("vector combination extents must match the Plan"))
    (Base.mightalias(output, candidate) || Base.mightalias(left, candidate) ||
     Base.mightalias(right, candidate)) && throw(
        ArgumentError("vector combination inputs and output must not alias Workspace scratch"),
    )
    chunk_size = 256
    for first_index in 1:chunk_size:plan.extent
        last_index = min(first_index + chunk_size - 1, plan.extent)
        finite_chunk = true
        # All four vectors have one-based axes and the checked plan extent.
        @inbounds for index in first_index:last_index
            value = fma(plan.left_scale, left[index], plan.right_scale * right[index])
            candidate[index] = value
            CheckFinite && (finite_chunk &= isfinite(value))
        end
        if CheckFinite && !finite_chunk
            # A nonfinite operand cannot give a finite FMA result, even at zero
            # scale. Earlier chunks therefore had finite inputs. Check the
            # remaining inputs so a later bad operand takes priority over overflow.
            for start in first_index:chunk_size:plan.extent
                finite_inputs = true
                @inbounds for index in start:min(start + chunk_size - 1, plan.extent)
                    finite_inputs &= isfinite(left[index]) & isfinite(right[index])
                end
                finite_inputs || return VectorLinearCombinationNonFiniteInput
            end
            return VectorLinearCombinationNonFiniteOutput
        end
    end
    copyto!(output, candidate)
    return nothing
end

"""
Combine two Float32 vectors by `output = left_scale * left + right_scale * right`.
All three ports use the configured schema and coordinate order. Multiplication
and addition use Float32 arithmetic, with the left product and final sum fused.
Every input and result must be finite;
rejection preserves caller output. Scratch staging permits overlapping caller
input and output views. This operation retains no mathematical history.
"""
@algorithm VectorLinearCombinationF32 begin
    label = "vector-linear-combination-f32"
    failure = VectorLinearCombinationFailure

    """
    Set a positive vector extent, one schema shared by both inputs and output,
    and finite dimensionless left/right scales. Extents fit UInt32 byte counts.
    """
    @configuration VectorLinearCombinationF32Configuration begin
        extent::Int => graph_rebuild
        left_scale::Float32 => construction
        right_scale::Float32 => construction
        schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:left, Float32, (configuration.extent,), configuration.schema),
        input(:right, Float32, (configuration.extent,), configuration.schema),
        output(:output, Float32, (configuration.extent,), configuration.schema;
            metadata_from=:left),
    )

    prepare(configuration) = begin
        plan = VectorLinearCombinationPlan(
            configuration.extent, configuration.left_scale, configuration.right_scale,
        )
        (plan=plan, workspace=VectorLinearCombinationWorkspace(plan))
    end

    process!(plan, workspace) = process!(output, workspace, plan, left, right)
end

# The CPU shared stage selects this entry only for proven finite input buffers
# and a construction-time plan satisfying _finite_combination_result.
function _process_finite_combination!(output, prepared::PreparedAlgorithm{VectorLinearCombinationF32},
        left, right, _left_metadata, _right_metadata)
    return _process_vector_combination!(output, prepared.workspace, prepared.plan, left, right, Val(false))
end

# An offset is replaceable scientific parameter data, separate from scratch.
struct VectorOffsetPlan <: AbstractAlgorithmPlan
    combination::VectorLinearCombinationPlan
    offset::Vector{Float32}
end
function _replace_vector_offset(plan::VectorOffsetPlan, values::AbstractVector{Float32})
    Base.require_one_based_indexing(values)
    length(values) == plan.combination.extent || throw(DimensionMismatch("offset extent differs from requested vector"))
    all(isfinite, values) || throw(ArgumentError("offset must be finite"))
    return VectorOffsetPlan(plan.combination, copy(values))
end

function _process_vector_offset!(output::AbstractVector{Float32}, current_offset::AbstractVector{Float32},
    workspace::VectorLinearCombinationWorkspace, plan::VectorOffsetPlan, requested::AbstractVector{Float32})
    Base.require_one_based_indexing(current_offset)
    length(current_offset) == plan.combination.extent ||
        throw(DimensionMismatch("current offset extent differs from requested vector"))
    (Base.mightalias(current_offset, output) || Base.mightalias(current_offset, requested) ||
     Base.mightalias(current_offset, plan.offset) || Base.mightalias(current_offset, workspace.candidate_output) ||
     Base.mightalias(output, plan.offset)) &&
        throw(ArgumentError("current offset and command outputs must not alias the adopted offset or each other"))
    status = process!(output, workspace, plan.combination, requested, plan.offset)
    status === nothing && copyto!(current_offset, plan.offset)
    return status
end

"""
Add a replaceable, same-schema offset to a Float32 vector. The current-offset
output reports the adopted parameter on successful frames and inherits requested
metadata. Rejection preserves both outputs; reset retains the adopted parameter.
"""
@algorithm VectorOffsetF32 begin
    label = "vector-offset-f32"
    failure = VectorLinearCombinationFailure
    @configuration VectorOffsetF32Configuration begin
        extent::Int => graph_rebuild
        schema::String => graph_rebuild
    end
    ports(configuration) = (
        input(:requested, Float32, (configuration.extent,), configuration.schema),
        output(:output, Float32, (configuration.extent,), configuration.schema;
            metadata_from=:requested),
        output(:current_offset, Float32, (configuration.extent,), configuration.schema;
            name="current-offset", metadata_from=:requested),
        parameter(:offset, Float32, (configuration.extent,), configuration.schema;
            replace=(plan, values) -> _replace_vector_offset(plan, values)),
    )
    prepare(configuration) = begin
        combination = VectorLinearCombinationPlan(configuration.extent, 1f0, 1f0)
        (plan=VectorOffsetPlan(combination, zeros(Float32, configuration.extent)),
            workspace=VectorLinearCombinationWorkspace(combination))
    end
    process!(plan, workspace) = _process_vector_offset!(output, current_offset, workspace, plan, requested)
end

# Preserve the numerical single-output API for callers that do not observe offsets.
function process!(output::AbstractVector, prepared::PreparedAlgorithm{VectorOffsetF32}, requested::AbstractVector)
    return process!(output, prepared.workspace, prepared.plan.combination, requested, prepared.plan.offset)
end
