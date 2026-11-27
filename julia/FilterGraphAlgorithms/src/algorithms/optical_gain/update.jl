"""Inclusive acceptance bounds for one measured optical-gain dither mode."""
struct OpticalGainUpdateCriteria{T<:AbstractFloat}
    minimum_command_amplitude::T
    minimum_response_amplitude::T
    minimum_optical_gain::T
    maximum_optical_gain::T
    maximum_absolute_phase_lag::T
end

function _optical_gain_update_value(values::AbstractDict, name::String)
    haskey(values, name) && return values[name]
    symbol = Symbol(name)
    haskey(values, symbol) && return values[symbol]
    throw(ArgumentError("optical-gain update value requires '$name'"))
end

function Base.convert(
    ::Type{OpticalGainUpdateCriteria{T}},
    values::AbstractDict,
) where {T<:AbstractFloat}
    allowed = Set((
        "minimum_command_amplitude",
        "minimum_response_amplitude",
        "minimum_optical_gain",
        "maximum_optical_gain",
        "maximum_absolute_phase_lag",
    ))
    all(key -> String(key) in allowed, keys(values)) || throw(
        ArgumentError("optical-gain update criterion has an unknown field"),
    )
    return OpticalGainUpdateCriteria(
        convert(T, _optical_gain_update_value(values, "minimum_command_amplitude")),
        convert(T, _optical_gain_update_value(values, "minimum_response_amplitude")),
        convert(T, _optical_gain_update_value(values, "minimum_optical_gain")),
        convert(T, _optical_gain_update_value(values, "maximum_optical_gain")),
        convert(T, _optical_gain_update_value(values, "maximum_absolute_phase_lag")),
    )
end

"""
    OpticalGainAssignment{T}()
    OpticalGainAssignment(dither_mode, update_fraction)

Assign one reconstructed mode either to hold its committed physical optical
gain or to filter accepted estimates from a zero-based dither-mode index.
"""
struct OpticalGainAssignment{T<:AbstractFloat}
    has_dither_mode::Bool
    dither_mode::Int
    update_fraction::T
end

OpticalGainAssignment{T}() where {T<:AbstractFloat} =
    OpticalGainAssignment{T}(false, 0, zero(T))

function OpticalGainAssignment(
    dither_mode::Integer,
    update_fraction::T,
) where {T<:AbstractFloat}
    return OpticalGainAssignment{T}(true, Int(dither_mode), update_fraction)
end

"""
    optical_gain_assignment_dither_mode(assignment::OpticalGainAssignment)

Return the assigned zero-based dither mode, or `nothing` for hold.
"""
optical_gain_assignment_dither_mode(assignment::OpticalGainAssignment) =
    assignment.has_dither_mode ? assignment.dither_mode : nothing

"""
    optical_gain_assignment_update_fraction(assignment::OpticalGainAssignment)

Return the assigned update fraction, or `nothing` for hold.
"""
optical_gain_assignment_update_fraction(assignment::OpticalGainAssignment) =
    assignment.has_dither_mode ? assignment.update_fraction : nothing

function Base.convert(
    ::Type{OpticalGainAssignment{T}},
    values::AbstractDict,
) where {T<:AbstractFloat}
    kind = String(_optical_gain_update_value(values, "kind"))
    if kind == "hold"
        length(values) == 1 || throw(
            ArgumentError("hold optical-gain assignment accepts only 'kind'"),
        )
        return OpticalGainAssignment{T}()
    elseif kind == "dither-mode"
        allowed = Set(("kind", "dither_mode", "update_fraction"))
        all(key -> String(key) in allowed, keys(values)) || throw(
            ArgumentError("dither-mode optical-gain assignment has an unknown field"),
        )
        length(values) == 3 || throw(
            ArgumentError(
                "dither-mode optical-gain assignment requires dither_mode and update_fraction",
            ),
        )
        return OpticalGainAssignment(
            convert(Int, _optical_gain_update_value(values, "dither_mode")),
            convert(T, _optical_gain_update_value(values, "update_fraction")),
        )
    end
    throw(ArgumentError("unknown optical-gain assignment kind '$kind'"))
end

"""
    OpticalGainUpdatePlan(criteria, assignments)

Prepare per-dither acceptance criteria and per-reconstructed-mode physical
optical-gain assignments. Mode indices in assignments are zero-based, matching
the FGN configuration contract.
"""
struct OpticalGainUpdatePlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    criteria::Vector{OpticalGainUpdateCriteria{T}}
    assignments::Vector{OpticalGainAssignment{T}}

    function OpticalGainUpdatePlan{T}(
        criteria::Vector{OpticalGainUpdateCriteria{T}},
        assignments::Vector{OpticalGainAssignment{T}},
        ::Val{:validated},
    ) where {T<:AbstractFloat}
        return new{T}(criteria, assignments)
    end
end

function OpticalGainUpdatePlan(
    criteria::AbstractVector{OpticalGainUpdateCriteria{T}},
    assignments::AbstractVector{OpticalGainAssignment{T}},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(criteria, assignments)
    isempty(criteria) && throw(
        ArgumentError("optical-gain update criteria must be nonempty"),
    )
    isempty(assignments) && throw(
        ArgumentError("optical-gain reconstructed-mode assignments must be nonempty"),
    )
    prepared_criteria = Vector{OpticalGainUpdateCriteria{T}}(criteria)
    prepared_assignments = Vector{OpticalGainAssignment{T}}(assignments)
    pi_value = T(pi)
    for (mode, criterion) in pairs(prepared_criteria)
        (isfinite(criterion.minimum_command_amplitude) &&
         criterion.minimum_command_amplitude >= zero(T) &&
         isfinite(criterion.minimum_response_amplitude) &&
         criterion.minimum_response_amplitude >= zero(T)) || throw(
            ArgumentError(
                "optical-gain amplitude threshold for dither mode $(mode - 1) must be finite and nonnegative",
            ),
        )
        (isfinite(criterion.minimum_optical_gain) &&
         criterion.minimum_optical_gain > zero(T) &&
         isfinite(one(T) / criterion.minimum_optical_gain) &&
         isfinite(criterion.maximum_optical_gain) &&
         criterion.maximum_optical_gain >= criterion.minimum_optical_gain) || throw(
            ArgumentError(
                "physical optical-gain bounds for dither mode $(mode - 1) must be finite, positive, invertible, and ordered",
            ),
        )
        (isfinite(criterion.maximum_absolute_phase_lag) &&
         zero(T) <= criterion.maximum_absolute_phase_lag <= pi_value) || throw(
            ArgumentError(
                "maximum absolute phase lag for dither mode $(mode - 1) must be in [0, pi]",
            ),
        )
    end
    for (mode, assignment) in pairs(prepared_assignments)
        assignment.has_dither_mode || continue
        0 <= assignment.dither_mode < length(prepared_criteria) || throw(
            ArgumentError(
                "reconstructed mode $(mode - 1) names unavailable dither mode $(assignment.dither_mode)",
            ),
        )
        (isfinite(assignment.update_fraction) &&
         zero(T) < assignment.update_fraction <= one(T)) || throw(
            ArgumentError(
                "optical-gain update fraction for reconstructed mode $(mode - 1) must be in (0, 1]",
            ),
        )
    end
    return OpticalGainUpdatePlan{T}(
        prepared_criteria,
        prepared_assignments,
        Val(:validated),
    )
end

"""
    optical_gain_update_criteria(plan::OpticalGainUpdatePlan)

Return a copy of the prepared dither-mode acceptance criteria.
"""
optical_gain_update_criteria(plan::OpticalGainUpdatePlan) = copy(plan.criteria)

"""
    optical_gain_update_assignments(plan::OpticalGainUpdatePlan)

Return a copy of the prepared reconstructed-mode assignments.
"""
optical_gain_update_assignments(plan::OpticalGainUpdatePlan) = copy(plan.assignments)

"""
    OpticalGainUpdateWorkspace(plan, initial_optical_gains)

Own initial, committed, and candidate physical optical gains plus fixed output
staging. A complete estimate changes committed gains when this Algorithm
commits.
"""
mutable struct OpticalGainUpdateWorkspace{T<:AbstractFloat}
    initial_optical_gains::Vector{T}
    committed_optical_gains::Vector{T}
    candidate_optical_gains::Vector{T}
    accepted_dither_modes::Vector{Bool}
    updated_reconstructed_modes::Vector{Bool}
    candidate_ready::Bool
end

function OpticalGainUpdateWorkspace(
    plan::OpticalGainUpdatePlan{T},
    initial_optical_gains::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(initial_optical_gains)
    reconstructed_modes = length(plan.assignments)
    length(initial_optical_gains) == reconstructed_modes || throw(
        DimensionMismatch(
            "initial physical optical-gain extent does not match reconstructed modes",
        ),
    )
    initial = Vector{T}(undef, reconstructed_modes)
    @inbounds for mode in 1:reconstructed_modes
        gain = initial_optical_gains[mode]
        isfinite(gain) && gain > zero(T) || throw(
            ArgumentError(
                "initial physical optical gain for reconstructed mode $(mode - 1) must be finite and positive",
            ),
        )
        assignment = plan.assignments[mode]
        if assignment.has_dither_mode
            criterion = plan.criteria[assignment.dither_mode + 1]
            criterion.minimum_optical_gain <= gain <= criterion.maximum_optical_gain || throw(
                ArgumentError(
                    "initial physical optical gain for reconstructed mode $(mode - 1) is outside dither mode $(assignment.dither_mode) bounds",
                ),
            )
        end
        initial[mode] = gain
    end
    return OpticalGainUpdateWorkspace{T}(
        initial,
        copy(initial),
        copy(initial),
        falses(length(plan.criteria)),
        falses(reconstructed_modes),
        false,
    )
end

"""Return a copy of the committed reconstructed-mode physical optical gains."""
optical_gains(workspace::OpticalGainUpdateWorkspace) =
    copy(workspace.committed_optical_gains)

function commit!(workspace::OpticalGainUpdateWorkspace)
    workspace.candidate_ready || return workspace
    workspace.committed_optical_gains, workspace.candidate_optical_gains =
        workspace.candidate_optical_gains, workspace.committed_optical_gains
    workspace.candidate_ready = false
    return workspace
end

function discard!(workspace::OpticalGainUpdateWorkspace)
    workspace.candidate_ready = false
    return workspace
end

function reset!(workspace::OpticalGainUpdateWorkspace)
    copyto!(workspace.committed_optical_gains, workspace.initial_optical_gains)
    copyto!(workspace.candidate_optical_gains, workspace.initial_optical_gains)
    fill!(workspace.accepted_dither_modes, false)
    fill!(workspace.updated_reconstructed_modes, false)
    workspace.candidate_ready = false
    return workspace
end

"""Expected numerical rejection from bounded physical optical-gain update."""
@enum OpticalGainUpdateFailure::UInt8 begin
    OpticalGainUpdateNonFiniteInput = 1
    OpticalGainUpdateNonFiniteOutput = 2
end

@doc "A completed dither-mode estimate field is non-finite."
OpticalGainUpdateNonFiniteInput
@doc "Filtering an accepted estimate produced a non-finite physical gain."
OpticalGainUpdateNonFiniteOutput

@inline _optical_gain_update_any_alias(::AbstractArray, ::Tuple{}) = false

@inline function _optical_gain_update_any_alias(
    values::AbstractArray,
    remaining::Tuple,
)
    return Base.mightalias(values, first(remaining)) ||
           _optical_gain_update_any_alias(values, Base.tail(remaining))
end

@inline _optical_gain_update_external_aliases_owned(::Tuple{}, ::Tuple) = false

@inline function _optical_gain_update_external_aliases_owned(
    external::Tuple,
    owned::Tuple,
)
    return _optical_gain_update_any_alias(first(external), owned) ||
           _optical_gain_update_external_aliases_owned(Base.tail(external), owned)
end

@inline _optical_gain_update_extents_are_valid(::Tuple{}, ::Int) = true

@inline function _optical_gain_update_extents_are_valid(
    arrays::Tuple,
    expected::Int,
)
    return length(first(arrays)) == expected &&
           _optical_gain_update_extents_are_valid(Base.tail(arrays), expected)
end

function _check_optical_gain_update_storage(
    optical_gains_output::AbstractVector,
    accepted_dither_modes::AbstractVector,
    updated_reconstructed_modes::AbstractVector,
    workspace::OpticalGainUpdateWorkspace,
    plan::OpticalGainUpdatePlan,
    command_amplitudes::AbstractVector,
    response_amplitudes::AbstractVector,
    optical_gain_estimates::AbstractVector,
    phase_lags::AbstractVector,
    valid::AbstractVector,
    estimate_complete::AbstractVector,
)
    Base.require_one_based_indexing(
        optical_gains_output,
        accepted_dither_modes,
        updated_reconstructed_modes,
        command_amplitudes,
        response_amplitudes,
        optical_gain_estimates,
        phase_lags,
        valid,
        estimate_complete,
    )
    dither_modes = length(plan.criteria)
    reconstructed_modes = length(plan.assignments)
    _optical_gain_update_extents_are_valid(
        (
            command_amplitudes,
            response_amplitudes,
            optical_gain_estimates,
            phase_lags,
            valid,
        ),
        dither_modes,
    ) || throw(DimensionMismatch("optical-gain update input extent is invalid"))
    length(estimate_complete) == 1 || throw(
        DimensionMismatch("optical-gain estimate-complete input extent is invalid"),
    )
    length(optical_gains_output) == reconstructed_modes || throw(
        DimensionMismatch("physical optical-gain output extent is invalid"),
    )
    length(accepted_dither_modes) == dither_modes || throw(
        DimensionMismatch("accepted dither-mode output extent is invalid"),
    )
    length(updated_reconstructed_modes) == reconstructed_modes || throw(
        DimensionMismatch("updated reconstructed-mode output extent is invalid"),
    )
    length(workspace.initial_optical_gains) == reconstructed_modes &&
        length(workspace.committed_optical_gains) == reconstructed_modes &&
        length(workspace.candidate_optical_gains) == reconstructed_modes &&
        length(workspace.accepted_dither_modes) == dither_modes &&
        length(workspace.updated_reconstructed_modes) == reconstructed_modes || throw(
        DimensionMismatch("optical-gain update Workspace extent is invalid"),
    )
    Base.mightalias(accepted_dither_modes, updated_reconstructed_modes) && throw(
        ArgumentError("optical-gain update outputs must not overlap"),
    )
    external = (
        optical_gains_output,
        accepted_dither_modes,
        updated_reconstructed_modes,
        command_amplitudes,
        response_amplitudes,
        optical_gain_estimates,
        phase_lags,
        valid,
        estimate_complete,
    )
    owned = (
        workspace.initial_optical_gains,
        workspace.committed_optical_gains,
        workspace.candidate_optical_gains,
        workspace.accepted_dither_modes,
        workspace.updated_reconstructed_modes,
        plan.criteria,
        plan.assignments,
    )
    _optical_gain_update_external_aliases_owned(external, owned) && throw(
        ArgumentError(
            "optical-gain update inputs and outputs must not overlap prepared or Workspace storage",
        ),
    )
    return nothing
end

@inline function _accept_optical_gain(criterion, command, response, estimate, phase, valid)
    return valid &&
        command >= criterion.minimum_command_amplitude &&
        response >= criterion.minimum_response_amplitude &&
        criterion.minimum_optical_gain <= estimate <= criterion.maximum_optical_gain &&
        abs(phase) <= criterion.maximum_absolute_phase_lag
end

@inline function _update_optical_gain(current::T, estimate::T, fraction::T) where {T}
    if fraction == one(T)
        return estimate
    elseif estimate >= current
        return min(current + fraction * (estimate - current), estimate)
    else
        return max(current - fraction * (current - estimate), estimate)
    end
end

function process!(
    optical_gains_output::AbstractVector{T},
    accepted_dither_modes::AbstractVector{Bool},
    updated_reconstructed_modes::AbstractVector{Bool},
    workspace::OpticalGainUpdateWorkspace{T},
    plan::OpticalGainUpdatePlan{T},
    command_amplitudes::AbstractVector{T},
    response_amplitudes::AbstractVector{T},
    optical_gain_estimates::AbstractVector{T},
    phase_lags::AbstractVector{T},
    valid::AbstractVector{Bool},
    estimate_complete::AbstractVector{Bool},
) where {T<:AbstractFloat}
    _check_optical_gain_update_storage(
        optical_gains_output,
        accepted_dither_modes,
        updated_reconstructed_modes,
        workspace,
        plan,
        command_amplitudes,
        response_amplitudes,
        optical_gain_estimates,
        phase_lags,
        valid,
        estimate_complete,
    )
    workspace.candidate_ready = false
    if !estimate_complete[1]
        copyto!(optical_gains_output, workspace.committed_optical_gains)
        fill!(accepted_dither_modes, false)
        fill!(updated_reconstructed_modes, false)
        return nothing
    end

    dither_modes = length(plan.criteria)
    @inbounds for mode in 1:dither_modes
        (isfinite(command_amplitudes[mode]) &&
         isfinite(response_amplitudes[mode]) &&
         isfinite(optical_gain_estimates[mode]) &&
         isfinite(phase_lags[mode])) || return OpticalGainUpdateNonFiniteInput
    end

    @inbounds for mode in 1:dither_modes
        criterion = plan.criteria[mode]
        workspace.accepted_dither_modes[mode] =
            _accept_optical_gain(criterion, command_amplitudes[mode], response_amplitudes[mode],
                optical_gain_estimates[mode], phase_lags[mode], valid[mode])
    end

    copyto!(workspace.candidate_optical_gains, workspace.committed_optical_gains)
    fill!(workspace.updated_reconstructed_modes, false)
    @inbounds for mode in eachindex(plan.assignments)
        assignment = plan.assignments[mode]
        assignment.has_dither_mode || continue
        dither_mode = assignment.dither_mode + 1
        workspace.accepted_dither_modes[dither_mode] || continue
        current = workspace.committed_optical_gains[mode]
        estimate = optical_gain_estimates[dither_mode]
        fraction = assignment.update_fraction
        next = _update_optical_gain(current, estimate, fraction)
        isfinite(next) && next > zero(T) || return OpticalGainUpdateNonFiniteOutput
        workspace.candidate_optical_gains[mode] = next
        workspace.updated_reconstructed_modes[mode] = true
    end

    copyto!(optical_gains_output, workspace.candidate_optical_gains)
    copyto!(accepted_dither_modes, workspace.accepted_dither_modes)
    copyto!(updated_reconstructed_modes, workspace.updated_reconstructed_modes)
    workspace.candidate_ready = true
    return nothing
end

"""
Float32 acceptance and bounded filtering of completed physical optical-gain
estimates. Graph commit owns adoption; estimation and reciprocal correction
remain separate Nodes.
"""
@algorithm OpticalGainUpdateF32 begin
    label = "optical-gain-update-f32"
    failure = OpticalGainUpdateFailure

    """Configuration for [`OpticalGainUpdateF32`](@ref)."""
    @configuration OpticalGainUpdateF32Configuration begin
        criteria::Vector{OpticalGainUpdateCriteria{Float32}} => graph_rebuild
        assignments::Vector{OpticalGainAssignment{Float32}} => graph_rebuild
        initial_optical_gains::Vector{Float32} => construction
    end

    ports(configuration) = (
        input(
            :command_amplitudes,
            schema_element_type(OPTICAL_GAIN_COMMAND_AMPLITUDE_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_COMMAND_AMPLITUDE_V1;
            name="command-amplitudes",
        ),
        input(
            :response_amplitudes,
            schema_element_type(OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1;
            name="response-amplitudes",
        ),
        input(
            :optical_gain_estimates,
            schema_element_type(OPTICAL_GAIN_DITHER_ESTIMATE_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_DITHER_ESTIMATE_V1;
            name="optical-gain-estimates",
        ),
        input(
            :phase_lags,
            schema_element_type(OPTICAL_GAIN_PHASE_LAG_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_PHASE_LAG_V1;
            name="phase-lags",
        ),
        input(
            :valid,
            schema_element_type(OPTICAL_GAIN_ESTIMATE_VALIDITY_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        ),
        input(
            :estimate_complete,
            schema_element_type(OPTICAL_GAIN_ESTIMATE_COMPLETE_V1),
            (1,),
            OPTICAL_GAIN_ESTIMATE_COMPLETE_V1;
            name="estimate-complete",
        ),
        output(
            :optical_gains,
            schema_element_type(PHYSICAL_OPTICAL_GAIN_V1),
            (length(configuration.assignments),),
            PHYSICAL_OPTICAL_GAIN_V1;
            name="optical-gains",
            metadata_from=:estimate_complete,
        ),
        output(
            :accepted_dither_modes,
            schema_element_type(OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1),
            (length(configuration.criteria),),
            OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1;
            name="accepted-dither-modes",
            metadata_from=:estimate_complete,
        ),
        output(
            :updated_reconstructed_modes,
            schema_element_type(OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1),
            (length(configuration.assignments),),
            OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1;
            name="updated-reconstructed-modes",
            metadata_from=:estimate_complete,
        ),
    )

    prepare(configuration) = begin
        plan = OpticalGainUpdatePlan(
            configuration.criteria,
            configuration.assignments,
        )
        (
            plan=plan,
            workspace=OpticalGainUpdateWorkspace(
                plan,
                configuration.initial_optical_gains,
            ),
        )
    end

    process!(plan, workspace) = begin
        process!(
            optical_gains,
            accepted_dither_modes,
            updated_reconstructed_modes,
            workspace,
            plan,
            command_amplitudes,
            response_amplitudes,
            optical_gain_estimates,
            phase_lags,
            valid,
            estimate_complete,
        )
    end
end
