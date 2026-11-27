"""
    LeakyIntegratorPlan(gain, pole; closed=true)

Prepare the elementwise recurrence `y[k] = pole * y[k-1] + gain * x[k]`.
Both coefficients must be finite and `pole` must be in `[0, 1]`. With
`closed=false`, each successful sample publishes the committed state unchanged;
input validation remains active. Closing resumes the recurrence without reset.
On the CPU, the pole product is rounded first; the gain product and sum use one fused
multiply-add rounding. This can retain a finite cancellation when the separate
gain product would overflow.
"""
struct LeakyIntegratorPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    gain::T
    pole::T
    closed::Bool

    function LeakyIntegratorPlan(gain::T, pole::T; closed::Bool=true) where {T<:Union{Float32,Float64}}
        isfinite(gain) || throw(ArgumentError("gain must be finite"))
        isfinite(pole) && zero(T) <= pole <= one(T) ||
            throw(ArgumentError("pole must be finite and in [0, 1]"))
        return new{T}(gain, pole, closed)
    end
end

"""
    LeakyIntegratorWorkspace(state)
    LeakyIntegratorWorkspace(T, state_length)

Own the initial, committed, and candidate mathematical state of one leaky
integrator. The one-based input state is materialized into owned dense vectors.
Processing writes the candidate without changing committed state; the owner
advances state with [`commit!`](@ref) after this Algorithm succeeds. A later
Graph Algorithm rejecting the sample does not undo this local adoption.
"""
mutable struct LeakyIntegratorWorkspace{T,V<:AbstractVector{T}}
    initial_state::V
    committed_state::V
    candidate_state::V
end

function LeakyIntegratorWorkspace(state::AbstractVector{T}) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(state)
    isempty(state) && throw(ArgumentError("state length must be positive"))
    initial = Vector{T}(state)
    committed = copy(initial)
    candidate = copy(initial)
    return LeakyIntegratorWorkspace{T,typeof(committed)}(initial, committed, candidate)
end

function LeakyIntegratorWorkspace(
    ::Type{T},
    state_length::Integer,
) where {T<:Union{Float32,Float64}}
    state_length > 0 || throw(ArgumentError("state length must be positive"))
    return LeakyIntegratorWorkspace(zeros(T, state_length))
end

"""
    leaky_integrator_state(workspace::LeakyIntegratorWorkspace)

Return the committed `y[k-1]` state as a read-only borrowed reference to
workspace storage. Commit can swap this buffer, and later operations can reuse
it; this reference is not a retained snapshot. Use `copy` on the analysis or
control path when a snapshot is needed, and serialize access with workspace use.
"""
leaky_integrator_state(workspace::LeakyIntegratorWorkspace) = workspace.committed_state

function commit!(workspace::LeakyIntegratorWorkspace)
    workspace.committed_state, workspace.candidate_state =
        workspace.candidate_state, workspace.committed_state
    return workspace
end

function discard!(workspace::LeakyIntegratorWorkspace)
    copyto!(workspace.candidate_state, workspace.committed_state)
    return workspace
end

function reset!(workspace::LeakyIntegratorWorkspace)
    copyto!(workspace.committed_state, workspace.initial_state)
    copyto!(workspace.candidate_state, workspace.initial_state)
    return workspace
end

# Protective recovery clears the correction without changing the Reset baseline.
function recover!(workspace::LeakyIntegratorWorkspace{T,Vector{T}}) where {T}
    fill!(workspace.committed_state, zero(T))
    fill!(workspace.candidate_state, zero(T))
    return workspace
end

"""Expected numerical rejection from the leaky-integrator recurrence."""
@enum LeakyIntegratorFailure::UInt8 begin
    LeakyIntegratorNonFiniteInput = 1
    LeakyIntegratorNonFiniteOutput = 2
end

@doc "The leaky integrator input contains a non-finite component." LeakyIntegratorNonFiniteInput
@doc "The leaky recurrence produced a non-finite output component." LeakyIntegratorNonFiniteOutput

# The caller discards candidate state on numerical rejection.
function _process_leaky_integrator!(
    output::AbstractVector{T},
    workspace::LeakyIntegratorWorkspace{T,Vector{T}},
    plan::LeakyIntegratorPlan{T},
    input::AbstractVector{T},
)::Union{Nothing,LeakyIntegratorFailure} where {T}
    input_axes = axes(input)
    (
        input_axes == axes(output) &&
        input_axes == axes(workspace.committed_state) &&
        input_axes == axes(workspace.candidate_state)
    ) ||
        throw(DimensionMismatch("input, output, and state axes must match"))
    buffers = (workspace.initial_state, workspace.committed_state, workspace.candidate_state)
    for buffer in buffers
        (Base.mightalias(input, buffer) || Base.mightalias(output, buffer)) && throw(
            ArgumentError("leaky-integrator input and output must not alias workspace storage"),
        )
    end
    for first in 1:length(buffers), second in (first + 1):length(buffers)
        Base.mightalias(buffers[first], buffers[second]) && throw(
            ArgumentError("leaky-integrator workspace buffers must not alias"),
        )
    end

    for value in input
        isfinite(value) || return LeakyIntegratorNonFiniteInput
    end

    values = if plan.closed
        # The buffers are disjoint and all axes match. Independent element
        # arithmetic can vectorize without reordering a reduction.
        @inbounds @simd for index in eachindex(
            input,
            workspace.committed_state,
            workspace.candidate_state,
        )
            state_term = plan.pole * workspace.committed_state[index]
            value = fma(plan.gain, input[index], state_term)
            workspace.candidate_state[index] = value
        end
        workspace.candidate_state
    else
        workspace.committed_state
    end
    for value in values
        isfinite(value) || return LeakyIntegratorNonFiniteOutput
    end
    plan.closed || copyto!(workspace.candidate_state, workspace.committed_state)
    for index in eachindex(output, workspace.candidate_state)
        output[index] = workspace.candidate_state[index]
    end

    return nothing
end

"""
    process!(output::AbstractVector{T}, workspace::LeakyIntegratorWorkspace{T,Vector{T}}, plan::LeakyIntegratorPlan{T}, input::AbstractVector{T}) where {T}

Stage one leaky-integrator sample. Input and output axes must match the dense
workspace. Input and output may overlap each other, but neither may alias
workspace storage. Numerical rejection preserves caller output and committed
state, and restores candidate state. Call `commit!` after success to adopt it.
"""
function process!(
    output::AbstractVector{T},
    workspace::LeakyIntegratorWorkspace{T,Vector{T}},
    plan::LeakyIntegratorPlan{T},
    input::AbstractVector{T},
) where {T}
    result = _process_leaky_integrator!(output, workspace, plan, input)
    result === nothing && return output
    discard!(workspace)
    return result
end

const LEAKY_INTEGRATOR_GAIN_PROPERTY = UInt32(0)
const LEAKY_INTEGRATOR_POLE_PROPERTY = UInt32(1)
const LEAKY_INTEGRATOR_CLOSED_PROPERTY = UInt32(2)

function property_schema(::Type{LeakyIntegratorPlan{T}}) where {T}
    return (
        runtime_property(
            LEAKY_INTEGRATOR_GAIN_PROPERTY,
            :gain,
            T;
            default=one(T),
            description="Integrator input gain g",
            unit="1",
            constraint=PropertyRange(-floatmax(T), floatmax(T)),
        ),
        runtime_property(
            LEAKY_INTEGRATOR_POLE_PROPERTY,
            :pole,
            T;
            default=zero(T),
            description="Discrete-time integrator pole p",
            unit="1",
            constraint=PropertyRange(zero(T), one(T)),
        ),
        runtime_property(
            LEAKY_INTEGRATOR_CLOSED_PROPERTY,
            :closed,
            Bool;
            default=true,
            description="Close integration; false holds committed state",
            unit="1",
        ),
    )
end

property_values(plan::LeakyIntegratorPlan) = (gain=plan.gain, pole=plan.pole, closed=plan.closed)

function replace_properties(
    ::LeakyIntegratorPlan{T},
    values::NamedTuple{(:gain,:pole,:closed),Tuple{T,T,Bool}},
) where {T}
    return LeakyIntegratorPlan(values.gain, values.pole; closed=values.closed)
end

"""
Float32 leaky integrator with committed state `y[k-1]` and candidate output
`y[k] = pole * y[k-1] + gain * input[k]`. The output/state units follow the
configured schemas; the gain determines conversion from input units.
"""
@algorithm LeakyIntegratorF32 begin
    label = "leaky-integrator-f32"
    failure = LeakyIntegratorFailure

    """
    Set positive vector `extent`, finite `initial_state`, and versioned input and
    output schema identifiers. `gain` and `pole` are runtime properties of the
    prepared plan; a rejected sample does not advance committed state.
    """
    @configuration LeakyIntegratorF32Configuration begin
        extent::Int => graph_rebuild
        initial_state::Float32 => construction
        input_schema::String => graph_rebuild
        output_schema::String => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            configuration.input_schema,
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            configuration.output_schema;
            metadata_from=:input,
        ),
    )

    prepare(configuration) = (
        plan=LeakyIntegratorPlan(1.0f0, 0.0f0),
        workspace=LeakyIntegratorWorkspace(
            fill(configuration.initial_state, configuration.extent),
        ),
    )

    process!(plan, workspace) = begin
        return _process_leaky_integrator!(output, workspace, plan, input)
    end
end
