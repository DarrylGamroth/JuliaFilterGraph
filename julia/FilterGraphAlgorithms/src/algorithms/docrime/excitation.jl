"""Expected numerical rejection from deterministic DO-CRIME excitation."""
@enum DocrimeExcitationFailure::UInt8 begin
    DocrimeExcitationNonFiniteCalculation = 1
end

@doc "Deterministic DO-CRIME excitation calculation produced a non-finite value." DocrimeExcitationNonFiniteCalculation

"""
    DocrimeExcitationPlan(amplitudes, seed)

Prepare deterministic continuous DO-CRIME excitation. Every amplitude must be
finite and positive. One processed vector contains an independent SplitMix64
draw in `[-amplitude, amplitude)` for each corrector coordinate.
"""
struct DocrimeExcitationPlan{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
    Backend,
} <: AbstractAlgorithmPlan
    amplitudes::V
    seed::UInt64
    backend::Backend
end

"""
    DocrimeBinaryExcitationPlan(amplitudes, seed)

Prepare deterministic binary DO-CRIME excitation. Every coordinate is exactly
its positive or negative prepared amplitude.
"""
struct DocrimeBinaryExcitationPlan{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
    Backend,
} <: AbstractAlgorithmPlan
    amplitudes::V
    seed::UInt64
    backend::Backend
end

function _prepared_docrime_excitation_plan(
    ::Type{DocrimeExcitationPlan},
    amplitudes::V,
    seed::UInt64,
    backend::Backend,
) where {T<:Union{Float32,Float64},V<:AbstractVector{T},Backend}
    return DocrimeExcitationPlan{T,V,Backend}(amplitudes, seed, backend)
end

function _prepared_docrime_excitation_plan(
    ::Type{DocrimeBinaryExcitationPlan},
    amplitudes::V,
    seed::UInt64,
    backend::Backend,
) where {T<:Union{Float32,Float64},V<:AbstractVector{T},Backend}
    return DocrimeBinaryExcitationPlan{T,V,Backend}(amplitudes, seed, backend)
end

function _docrime_seed(seed::Integer)
    0 <= seed <= typemax(UInt64) ||
        throw(ArgumentError("DO-CRIME excitation seed must fit UInt64"))
    return UInt64(seed)
end

function _docrime_amplitudes(amplitudes::AbstractVector{T}) where {T}
    prepared = Vector{T}(amplitudes)
    isempty(prepared) &&
        throw(ArgumentError("DO-CRIME excitation amplitudes must be nonempty"))
    for (index, amplitude) in pairs(prepared)
        isfinite(amplitude) && amplitude > zero(T) || throw(
            ArgumentError(
                "DO-CRIME excitation amplitude at corrector coordinate " *
                "$(index - 1) must be finite and positive",
            ),
        )
    end
    return prepared
end

function DocrimeExcitationPlan(
    amplitudes::AbstractVector{T},
    seed::Integer,
) where {T<:Union{Float32,Float64}}
    return _prepared_docrime_excitation_plan(
        DocrimeExcitationPlan,
        _docrime_amplitudes(amplitudes),
        _docrime_seed(seed),
        nothing,
    )
end

function DocrimeBinaryExcitationPlan(
    amplitudes::AbstractVector{T},
    seed::Integer,
) where {T<:Union{Float32,Float64}}
    return _prepared_docrime_excitation_plan(
        DocrimeBinaryExcitationPlan,
        _docrime_amplitudes(amplitudes),
        _docrime_seed(seed),
        nothing,
    )
end

"""
    docrime_excitation_amplitudes(plan::Union{DocrimeExcitationPlan, DocrimeBinaryExcitationPlan})

Return a copy of the prepared per-coordinate excitation amplitudes.
"""
docrime_excitation_amplitudes(
    plan::Union{DocrimeExcitationPlan,DocrimeBinaryExcitationPlan},
) = copy(plan.amplitudes)

"""
    docrime_excitation_seed(plan::Union{DocrimeExcitationPlan, DocrimeBinaryExcitationPlan})

Return the initial SplitMix64 state for a DO-CRIME excitation Plan.
"""
docrime_excitation_seed(
    plan::Union{DocrimeExcitationPlan,DocrimeBinaryExcitationPlan},
) = plan.seed

"""
    prepare_docrime_excitation_plan(device, plan)

Prepare immutable DO-CRIME excitation amplitudes and kernels for an optional
accelerator backend.

Backend extensions implement this boundary for their concrete device type.
Preparation returns a new storage specialization of the continuous or binary
Plan; it does not mutate the CPU Plan. Unsupported backends have no fallback
method and therefore cannot silently execute on the CPU.
"""
function prepare_docrime_excitation_plan end

"""
    DocrimeExcitationWorkspace(plan)

Own committed and candidate SplitMix64 state plus fixed staged excitation
storage for one continuous or binary DO-CRIME source.
"""
mutable struct DocrimeExcitationWorkspace{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
}
    initial_random_state::UInt64
    committed_random_state::UInt64
    candidate_random_state::UInt64
    staged_excitation::V
end

function _prepared_docrime_excitation_workspace(
    ::Type{T},
    initial_random_state::UInt64,
    committed_random_state::UInt64,
    candidate_random_state::UInt64,
    staged_excitation::V,
) where {T<:Union{Float32,Float64},V<:AbstractVector{T}}
    return DocrimeExcitationWorkspace{T,V}(
        initial_random_state,
        committed_random_state,
        candidate_random_state,
        staged_excitation,
    )
end

function DocrimeExcitationWorkspace(
    plan::Union{
        DocrimeExcitationPlan{T,Vector{T},Nothing},
        DocrimeBinaryExcitationPlan{T,Vector{T},Nothing},
    },
) where {T}
    staged = similar(plan.amplitudes)
    fill!(staged, zero(T))
    return _prepared_docrime_excitation_workspace(
        T,
        plan.seed,
        plan.seed,
        plan.seed,
        staged,
    )
end

"""
    docrime_excitation_state(workspace::DocrimeExcitationWorkspace)

Return the committed SplitMix64 state of a DO-CRIME excitation Workspace as a
scalar value. This value is a snapshot, independent of later workspace changes.
"""
docrime_excitation_state(workspace::DocrimeExcitationWorkspace) =
    workspace.committed_random_state

function commit!(workspace::DocrimeExcitationWorkspace)
    workspace.committed_random_state = workspace.candidate_random_state
    return workspace
end

function discard!(workspace::DocrimeExcitationWorkspace)
    workspace.candidate_random_state = workspace.committed_random_state
    return workspace
end

function reset!(workspace::DocrimeExcitationWorkspace)
    workspace.committed_random_state = workspace.initial_random_state
    workspace.candidate_random_state = workspace.initial_random_state
    return workspace
end

_plan_adopted!(workspace::DocrimeExcitationWorkspace) = discard!(workspace)

function _validate_docrime_execution(
    output::AbstractVector,
    workspace::DocrimeExcitationWorkspace,
    plan::Union{DocrimeExcitationPlan,DocrimeBinaryExcitationPlan},
)
    Base.require_one_based_indexing(
        output,
        workspace.staged_excitation,
        plan.amplitudes,
    )
    output_axes = axes(output)
    output_axes == axes(plan.amplitudes) || throw(
        DimensionMismatch("DO-CRIME excitation output and Plan axes must match"),
    )
    output_axes == axes(workspace.staged_excitation) || throw(
        DimensionMismatch("DO-CRIME excitation Workspace has an incompatible extent"),
    )
    return nothing
end

function process!(
    output::AbstractVector{T},
    workspace::DocrimeExcitationWorkspace{T,Vector{T}},
    plan::DocrimeExcitationPlan{T,Vector{T},Nothing},
) where {T}
    _validate_docrime_execution(output, workspace, plan)
    random_state = workspace.committed_random_state
    for index in eachindex(workspace.staged_excitation, plan.amplitudes)
        draw_index = UInt64(index - 1)
        unit = splitmix64_unit_at(T, random_state, draw_index)
        staged = plan.amplitudes[index] * (unit + unit - one(T))
        isfinite(staged) || return DocrimeExcitationNonFiniteCalculation
        workspace.staged_excitation[index] = staged
    end
    copyto!(output, workspace.staged_excitation)
    workspace.candidate_random_state =
        splitmix64_advance(random_state, UInt64(length(plan.amplitudes)))
    return nothing
end

function process!(
    output::AbstractVector{T},
    workspace::DocrimeExcitationWorkspace{T,Vector{T}},
    plan::DocrimeBinaryExcitationPlan{T,Vector{T},Nothing},
) where {T}
    _validate_docrime_execution(output, workspace, plan)
    random_state = workspace.committed_random_state
    for index in eachindex(workspace.staged_excitation, plan.amplitudes)
        draw_index = UInt64(index - 1)
        workspace.staged_excitation[index] =
            iszero(splitmix64_at(random_state, draw_index) >> 63) ?
            -plan.amplitudes[index] : plan.amplitudes[index]
    end
    copyto!(output, workspace.staged_excitation)
    workspace.candidate_random_state =
        splitmix64_advance(random_state, UInt64(length(plan.amplitudes)))
    return nothing
end

function _replace_docrime_amplitudes(
    plan::DocrimeExcitationPlan{T,Vector{T},Nothing},
    values,
) where {T}
    return DocrimeExcitationPlan(values, plan.seed)
end

function _replace_docrime_amplitudes(
    plan::DocrimeBinaryExcitationPlan{T,Vector{T},Nothing},
    values,
) where {T}
    return DocrimeBinaryExcitationPlan(values, plan.seed)
end

function _validate_docrime_extent(amplitudes::AbstractVector)
    isempty(amplitudes) &&
        throw(ArgumentError("DO-CRIME excitation amplitudes must be nonempty"))
    length(amplitudes) <= div(typemax(UInt32), sizeof(Float32)) || throw(
        ArgumentError("DO-CRIME excitation byte extent must fit UInt32"),
    )
    return nothing
end

"""
Generate deterministic continuous DO-CRIME excitation, one Float32 value per
corrector coordinate in `[-amplitude, amplitude)`. The output has the same
command units as the configured positive amplitudes.
"""
@algorithm DocrimeExcitationF32 begin
    label = "docrime-excitation-f32"
    failure = DocrimeExcitationFailure

    """
    Set a nonempty vector of finite positive per-coordinate `amplitudes` and a
    `UInt64` SplitMix64 `seed`. The amplitude vector fixes the output extent and
    can be replaced through the amplitude parameter port.
    """
    @configuration DocrimeExcitationF32Configuration begin
        amplitudes::Vector{Float32} => graph_rebuild
        seed::UInt64 => construction
    end

    ports(configuration) = (
        output(
            :excitation,
            schema_element_type(DOCRIME_EXCITATION_V1),
            (length(configuration.amplitudes),),
            DOCRIME_EXCITATION_V1,
        ),
        parameter(
            :amplitudes,
            schema_element_type(DOCRIME_EXCITATION_AMPLITUDE_V1),
            (length(configuration.amplitudes),),
            DOCRIME_EXCITATION_AMPLITUDE_V1;
            replace=(plan, values) ->
                _replace_docrime_amplitudes(plan, values),
        ),
    )

    prepare(configuration) = begin
        _validate_docrime_extent(configuration.amplitudes)
        plan = DocrimeExcitationPlan(
            configuration.amplitudes,
            configuration.seed,
        )
        (plan=plan, workspace=DocrimeExcitationWorkspace(plan))
    end

    process!(plan, workspace) = process!(excitation, workspace, plan)
end


"""
Generate deterministic binary DO-CRIME excitation: each Float32 corrector
coordinate is exactly its positive or negative configured amplitude. Output
and amplitude share command units.
"""
@algorithm DocrimeBinaryExcitationF32 begin
    label = "docrime-binary-excitation-f32"
    failure = DocrimeExcitationFailure

    """
    Set a nonempty vector of finite positive per-coordinate `amplitudes` and a
    `UInt64` SplitMix64 `seed`. The amplitude vector fixes the output extent and
    can be replaced through the amplitude parameter port.
    """
    @configuration DocrimeBinaryExcitationF32Configuration begin
        amplitudes::Vector{Float32} => graph_rebuild
        seed::UInt64 => construction
    end

    ports(configuration) = (
        output(
            :excitation,
            schema_element_type(DOCRIME_EXCITATION_V1),
            (length(configuration.amplitudes),),
            DOCRIME_EXCITATION_V1,
        ),
        parameter(
            :amplitudes,
            schema_element_type(DOCRIME_EXCITATION_AMPLITUDE_V1),
            (length(configuration.amplitudes),),
            DOCRIME_EXCITATION_AMPLITUDE_V1;
            replace=(plan, values) ->
                _replace_docrime_amplitudes(plan, values),
        ),
    )

    prepare(configuration) = begin
        _validate_docrime_extent(configuration.amplitudes)
        plan = DocrimeBinaryExcitationPlan(
            configuration.amplitudes,
            configuration.seed,
        )
        (plan=plan, workspace=DocrimeExcitationWorkspace(plan))
    end

    process!(plan, workspace) = process!(excitation, workspace, plan)
end
