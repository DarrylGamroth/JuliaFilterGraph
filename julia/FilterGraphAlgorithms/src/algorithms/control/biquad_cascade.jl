"""
    BiquadCascadePlan(coefficients::AbstractArray{T, 3}, extent::Integer)

Prepare a CPU temporal cascade of second-order sections for `extent` independent
elements, with `T` equal to `Float32` or `Float64`. Copy coefficients into owned
dense storage of shape `(5, section_count, coefficient_set_count)`. The extent
and section count must be positive. One coefficient set broadcasts to every
element; otherwise there must be exactly `extent` sets.

Each section stores `(b0, b1, b2, a1, a2)`, with normalized `a0 = 1`, and has
transfer function `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²)/(1 + a1*z⁻¹ + a2*z⁻²)`.
The numerator includes any section gain. All coefficients must be finite, and
each denominator must satisfy the strict Schur conditions `abs(a2) < 1` and
`abs(a1) < 1 + a2`, evaluated in `T`. Poles on the unit circle are rejected.
These checks establish section stability in the prepared precision; they do not
qualify stability of a feedback system or prevent finite arithmetic overflow.

Sections use transposed direct form II (DF2T), with two states per element:

```text
y      = b0*x + s1
s1_new = b1*x - a1*y + s2
s2_new = b2*x - a2*y
```

Each section's output feeds the next section. No high-order polynomial is
formed. Treat the owned coefficient array as read-only after preparation. A
mode-first prepared copy supplies per-mode SIMD loads. Rebuild the plan to change
coefficients. CPU evaluation uses explicit FMA for the output and feedback
products while preserving the grouping shown above; intermediate rounding and
overflow-cancellation results can differ from separate multiplication/addition.

Alternatively, pass a tuple or vector of `UnitRange{Int} => Matrix{T}` pairs
and an extent. Each range assigns one conventional `(5, section_count)`
coefficient matrix to its modes. Ranges may be unsorted, but must be nonempty,
within `1:extent`, and cover every mode exactly once. Matrices must have five
rows, at least one section, one-based indexing, and the same `T` (`Float32` or
`Float64`). Shorter cascades are padded with identity sections. Coefficients
are copied and checked using the same finiteness and strict stability rules as
the dense constructor.
"""
struct BiquadCascadePlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    coefficients::Array{T,3}
    # Mode-first copy gives contiguous coefficient lanes across SIMD modes.
    mode_coefficients::Array{T,3}
    extent::Int

    function BiquadCascadePlan(
        coefficients::AbstractArray{T,3},
        extent::Integer,
    ) where {T<:Union{Float32,Float64}}
        Base.require_one_based_indexing(coefficients)
        extent > 0 || throw(ArgumentError("extent must be positive"))
        size(coefficients, 1) == 5 ||
            throw(DimensionMismatch("biquad coefficients must have first extent 5"))
        size(coefficients, 2) > 0 ||
            throw(ArgumentError("biquad section count must be positive"))
        size(coefficients, 3) in (1, extent) || throw(
            DimensionMismatch("biquad coefficient set count must be 1 or extent"),
        )
        owned = Array{T,3}(coefficients)
        all(isfinite, owned) ||
            throw(ArgumentError("biquad coefficients must be finite"))
        for set in axes(owned, 3), section in axes(owned, 2)
            a1 = owned[4, section, set]
            a2 = owned[5, section, set]
            (abs(a2) < one(T) && abs(a1) < one(T) + a2) || throw(
                ArgumentError("biquad denominators must be strictly Schur stable"),
            )
        end
        return new{T}(owned, permutedims(owned, (3, 1, 2)), Int(extent))
    end
end

function BiquadCascadePlan(regions::Union{Tuple,AbstractVector}, extent::Integer)
    extent > 0 || throw(ArgumentError("extent must be positive"))
    mode_count = Int(extent)
    isempty(regions) && throw(ArgumentError("biquad mode regions must be nonempty"))

    coefficient_type = nothing
    max_sections = 0
    covered = falses(mode_count)
    for region in regions
        region isa Pair || throw(ArgumentError("each biquad mode region must be a pair"))
        modes, coefficients = region
        modes isa UnitRange{Int} || throw(ArgumentError("biquad mode ranges must be UnitRange{Int}"))
        isempty(modes) && throw(ArgumentError("biquad mode ranges must be nonempty"))
        first(modes) >= 1 && last(modes) <= mode_count ||
            throw(ArgumentError("biquad mode ranges must lie within 1:extent"))
        coefficients isa Matrix{<:Union{Float32,Float64}} ||
            throw(ArgumentError("region coefficients must be a Matrix{Float32} or Matrix{Float64}"))
        Base.require_one_based_indexing(coefficients)
        size(coefficients, 1) == 5 ||
            throw(DimensionMismatch("biquad region coefficients must have five rows"))
        size(coefficients, 2) > 0 ||
            throw(ArgumentError("biquad region cascades must contain at least one section"))
        T = eltype(coefficients)
        coefficient_type === nothing || T === coefficient_type ||
            throw(ArgumentError("all biquad regions must use the same coefficient type"))
        coefficient_type = T
        max_sections = max(max_sections, size(coefficients, 2))
        for mode in modes
            covered[mode] && throw(ArgumentError("biquad mode regions must not overlap"))
            covered[mode] = true
        end
    end
    all(covered) || throw(ArgumentError("biquad mode regions must cover every mode"))

    T = coefficient_type
    bank = zeros(T, 5, max_sections, mode_count)
    bank[1, :, :] .= one(T)
    for region in regions
        modes, coefficients = region
        for mode in modes, section in axes(coefficients, 2), coefficient in axes(coefficients, 1)
            bank[coefficient, section, mode] = coefficients[coefficient, section]
        end
    end
    return BiquadCascadePlan(bank, extent)
end

"""
    BiquadCascadeWorkspace(plan)

Allocate zero committed and candidate DF2T state of shape
`(2, plan.extent, section_count)`, and a candidate output vector. The workspace
owns these CPU buffers. Processing stages a complete sample without changing
committed state; [`commit!`](@ref) adopts it for this Algorithm. A later Graph
Algorithm rejecting the sample does not roll back this local adoption.
"""
mutable struct BiquadCascadeWorkspace{T<:Union{Float32,Float64}}
    committed_state::Array{T,3}
    candidate_state::Array{T,3}
    candidate_output::Vector{T}
end

function BiquadCascadeWorkspace(plan::BiquadCascadePlan{T}) where {T}
    dimensions = (2, plan.extent, size(plan.coefficients, 2))
    return BiquadCascadeWorkspace{T}(
        zeros(T, dimensions),
        zeros(T, dimensions),
        zeros(T, plan.extent),
    )
end

"""
    biquad_cascade_state(workspace::BiquadCascadeWorkspace)

Return committed DF2T state in `(state, element, section)` order as a read-only
borrowed reference to workspace storage. Commit can swap this buffer, and later
operations can reuse it; this reference is not a retained snapshot. Use `copy`
on the analysis or control path when a snapshot is needed, and serialize access
with workspace use.
"""
biquad_cascade_state(workspace::BiquadCascadeWorkspace) = workspace.committed_state

"""Adopt a successfully staged biquad sample by swapping the state buffers."""
function commit!(workspace::BiquadCascadeWorkspace)
    workspace.committed_state, workspace.candidate_state =
        workspace.candidate_state, workspace.committed_state
    return workspace
end

"""Restore candidate biquad state from committed state and clear staged output."""
function discard!(workspace::BiquadCascadeWorkspace{T}) where {T}
    copyto!(workspace.candidate_state, workspace.committed_state)
    fill!(workspace.candidate_output, zero(T))
    return workspace
end

"""Zero committed state, candidate state, and staged output for a biquad cascade."""
function reset!(workspace::BiquadCascadeWorkspace{T}) where {T}
    fill!(workspace.committed_state, zero(T))
    fill!(workspace.candidate_state, zero(T))
    fill!(workspace.candidate_output, zero(T))
    return workspace
end

_plan_adopted!(workspace::BiquadCascadeWorkspace) = reset!(workspace)

recover!(workspace::BiquadCascadeWorkspace) = reset!(workspace)

"""Expected numerical rejection from a temporal biquad cascade."""
@enum BiquadCascadeFailure::UInt8 begin
    BiquadCascadeNonFiniteInput = 1
    BiquadCascadeNonFiniteOutput = 2
end

@doc "Reject a sample containing a nonfinite input value." BiquadCascadeNonFiniteInput
@doc "Reject a sample producing nonfinite section output or DF2T state." BiquadCascadeNonFiniteOutput

function _validate_biquad_cascade_buffers(
    output::AbstractVector{T},
    workspace::BiquadCascadeWorkspace{T},
    plan::BiquadCascadePlan{T},
    input::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(output, input)
    length(output) == length(input) == plan.extent || throw(
        DimensionMismatch("biquad input and output lengths must match plan extent"),
    )
    coefficients = plan.coefficients
    size(plan.mode_coefficients) == (size(coefficients, 3), 5, size(coefficients, 2)) ||
        throw(DimensionMismatch("biquad prepared coefficients have incompatible dimensions"))
    size(coefficients, 1) == 5 && size(coefficients, 2) > 0 &&
        size(coefficients, 3) in (1, plan.extent) || throw(
        DimensionMismatch("biquad plan coefficients have incompatible dimensions"),
    )
    state_dimensions = (2, plan.extent, size(coefficients, 2))
    size(workspace.committed_state) == state_dimensions &&
        size(workspace.candidate_state) == state_dimensions &&
        length(workspace.candidate_output) == plan.extent || throw(
        DimensionMismatch("biquad workspace has incompatible dimensions"),
    )
    buffers = (
        coefficients,
        plan.mode_coefficients,
        workspace.committed_state,
        workspace.candidate_state,
    )
    staged_output = workspace.candidate_output
    (Base.mightalias(input, staged_output) || Base.mightalias(output, staged_output)) && throw(
        ArgumentError("biquad input and output must not alias plan or workspace"),
    )
    for buffer in buffers
        (Base.mightalias(input, buffer) || Base.mightalias(output, buffer)) && throw(
            ArgumentError("biquad input and output must not alias plan or workspace"),
        )
        Base.mightalias(staged_output, buffer) && throw(
            ArgumentError("biquad plan and workspace buffers must not alias"),
        )
    end
    for first in 1:length(buffers), second in (first + 1):length(buffers)
        Base.mightalias(buffers[first], buffers[second]) && throw(
            ArgumentError("biquad plan and workspace buffers must not alias"),
        )
    end
    return nothing
end

# Traverse adjacent independent modes within each section. A bounded mode block
# reuses staged output and the same coefficient region across its sections while
# committed/candidate state retain the public (state, element, section) layout.
# Buffer validation proves all dense indices and disjoint workspace ownership;
# each SIMD iteration owns a different mode. FMA preserves the DF2T grouping:
# round b1*x and b2*x before the feedback operation, then add s2 separately.
function _stage_biquad_cascade!(
    workspace::BiquadCascadeWorkspace{T},
    plan::BiquadCascadePlan{T},
    ::Val{Shared},
) where {T,Shared}
    coefficients = plan.coefficients
    mode_coefficients = plan.mode_coefficients
    committed = workspace.committed_state
    candidate = workspace.candidate_state
    values = workspace.candidate_output
    for first_element in 1:4096:plan.extent
        last_element = min(first_element + 4095, plan.extent)
        for section in axes(coefficients, 2)
            state_offset = 2 * plan.extent * (section - 1)
            valid = true
            @inbounds @simd ivdep for element in first_element:last_element
                value = values[element]
                b0 = Shared ? coefficients[1, section, 1] :
                     mode_coefficients[element, 1, section]
                b1 = Shared ? coefficients[2, section, 1] :
                     mode_coefficients[element, 2, section]
                b2 = Shared ? coefficients[3, section, 1] :
                     mode_coefficients[element, 3, section]
                a1 = Shared ? coefficients[4, section, 1] :
                     mode_coefficients[element, 4, section]
                a2 = Shared ? coefficients[5, section, 1] :
                     mode_coefficients[element, 5, section]
                state_index = state_offset + 2 * element - 1
                s1 = committed[state_index]
                s2 = committed[state_index + 1]
                section_output = fma(b0, value, s1)
                next_s1 = fma(-a1, section_output, b1 * value) + s2
                next_s2 = fma(-a2, section_output, b2 * value)
                valid &= isfinite(section_output) & isfinite(next_s1) & isfinite(next_s2)
                candidate[state_index] = next_s1
                candidate[state_index + 1] = next_s2
                values[element] = section_output
            end
            valid || return false
        end
    end
    return true
end

# The caller discards candidate state and output on numerical rejection.
function _process_biquad_cascade!(
    output::AbstractVector{T},
    workspace::BiquadCascadeWorkspace{T},
    plan::BiquadCascadePlan{T},
    input::AbstractVector{T},
)::Union{Nothing,BiquadCascadeFailure} where {T}
    _validate_biquad_cascade_buffers(output, workspace, plan, input)
    for value in input
        isfinite(value) || return BiquadCascadeNonFiniteInput
    end

    # Staging also preserves arbitrary input/output overlap. The caller vectors
    # retain their checked generic indexing; only owned dense buffers enter SIMD.
    copyto!(workspace.candidate_output, input)
    shared = size(plan.coefficients, 3) == 1
    valid = shared ? _stage_biquad_cascade!(workspace, plan, Val(true)) :
            _stage_biquad_cascade!(workspace, plan, Val(false))
    valid || return BiquadCascadeNonFiniteOutput
    copyto!(output, workspace.candidate_output)
    return nothing
end

"""
    process!(output, workspace::BiquadCascadeWorkspace, plan::BiquadCascadePlan, input)

Stage one temporal sample in the plan's `Float32` or `Float64` precision. Input
and output must be vectors of `plan.extent` elements with one-based indexing;
ordinary views and overlapping input/output vectors are supported. Neither may
alias plan coefficients or workspace buffers. Dimensions and buffer ownership
are checked before processing.

Return `output` after successful staging. Call [`commit!`](@ref) to adopt state,
or [`discard!`](@ref) to reject it. Expected numerical rejection returns
[`BiquadCascadeNonFiniteInput`](@ref) or [`BiquadCascadeNonFiniteOutput`](@ref),
preserves caller output and committed state, and restores candidate state.
Nonfinite intermediate section output or state rejects the complete sample.
Output is copied from staging only after all sections and elements succeed.
"""
function process!(
    output::AbstractVector{T},
    workspace::BiquadCascadeWorkspace{T},
    plan::BiquadCascadePlan{T},
    input::AbstractVector{T},
) where {T}
    result = _process_biquad_cascade!(output, workspace, plan, input)
    result === nothing && return output
    # Keep the wrapper small enough to inline and avoid allocating views.
    @noinline discard!(workspace)
    return result
end

function _validate_biquad_cascade_configuration(
    extent::Int,
    section_count::Int,
    per_mode::Bool,
)
    extent > 0 || throw(ArgumentError("extent must be positive"))
    section_count > 0 || throw(ArgumentError("section_count must be positive"))
    max_elements = div(UInt128(typemax(UInt32)), UInt128(sizeof(Float32)))
    UInt128(extent) <= max_elements ||
        throw(ArgumentError("biquad input and output byte extents must fit UInt32"))
    coefficient_elements = UInt128(5) * UInt128(section_count) *
                           UInt128(per_mode ? extent : 1)
    coefficient_elements <= max_elements ||
        throw(ArgumentError("biquad coefficient byte extent must fit UInt32"))
    state_elements = UInt128(2) * UInt128(extent) * UInt128(section_count)
    state_elements <= div(UInt128(typemax(Int)), UInt128(sizeof(Float32))) ||
        throw(ArgumentError("biquad state byte extent must fit Int"))
    return nothing
end

"""
    BiquadCascadeF32

Temporal biquad cascade declaration with `:input` and `:output` vector
Ports and a `:coefficients` Parameter Port using `BIQUAD_COEFFICIENTS_V1`.
Output inherits input metadata. Coefficients have shape
`(5, section_count, per_mode ? extent : 1)` and follow [`BiquadCascadePlan`](@ref).
Replacement validates and copies a complete coefficient bank before adoption;
successful replacement resets all temporal history to zero. Invalid replacement
preserves the active plan and history. State advances on successful local
Algorithm adoption. Loaded CUDA and AMDGPU extensions qualify complete captured
execution with fixed device storage; other devices retain `NoCapture()`.
"""
@algorithm BiquadCascadeF32 begin
    label = "biquad-cascade-f32"
    failure = BiquadCascadeFailure

    """
        BiquadCascadeF32Configuration(; extent, section_count, per_mode,
                                      initial_coefficients, input_schema, output_schema)

    Configure [`BiquadCascadeF32`](@ref). Positive `extent` and `section_count`, the
    `per_mode` choice, and input/output schema strings require a Graph rebuild.
    `per_mode=false` broadcasts one coefficient bank; `true` selects a bank for
    each element. `initial_coefficients` is a construction-only `Vector{Float32}`
    or `nothing`. A vector stores each `(b0,b1,b2,a1,a2)` tuple contiguously, then
    sections, then coefficient sets, in Julia column-major order. `nothing` makes
    every section an identity. Explicit initial values must meet the same finite
    and strictly stable checks as replacement coefficients. Prepared storage is
    owned and does not retain the configuration vector.
    """
    @configuration BiquadCascadeF32Configuration begin
        extent::Int => graph_rebuild
        section_count::Int => graph_rebuild
        per_mode::Bool => graph_rebuild
        initial_coefficients::Union{Nothing,Vector{Float32}} => construction
        input_schema::String => graph_rebuild
        output_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), configuration.input_schema),
        output(
            :output,
            Float32,
            (configuration.extent,),
            configuration.output_schema;
            metadata_from=:input,
        ),
        parameter(
            :coefficients,
            schema_element_type(BIQUAD_COEFFICIENTS_V1),
            (5, configuration.section_count, configuration.per_mode ? configuration.extent : 1),
            BIQUAD_COEFFICIENTS_V1;
            replace=(plan, values) -> BiquadCascadePlan(values, plan.extent),
        ),
    )

    prepare(configuration) = begin
        _validate_biquad_cascade_configuration(
            configuration.extent,
            configuration.section_count,
            configuration.per_mode,
        )
        dimensions = (
            5,
            configuration.section_count,
            configuration.per_mode ? configuration.extent : 1,
        )
        coefficients = if isnothing(configuration.initial_coefficients)
            identity = zeros(Float32, dimensions)
            for set in axes(identity, 3), section in axes(identity, 2)
                identity[1, section, set] = 1.0f0
            end
            identity
        else
            length(configuration.initial_coefficients) == prod(dimensions) || throw(
                DimensionMismatch("initial biquad coefficients have incompatible length"),
            )
            reshape(configuration.initial_coefficients, dimensions)
        end
        plan = BiquadCascadePlan(coefficients, configuration.extent)
        (plan=plan, workspace=BiquadCascadeWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_biquad_cascade!(output, workspace, plan, input)
    end
end
