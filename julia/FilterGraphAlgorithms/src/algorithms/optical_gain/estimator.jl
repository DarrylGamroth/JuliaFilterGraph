"""
    OpticalGainEstimatorPlan(schedule, calibration_responses)

Prepare synchronous demodulation of realized modal dither commands and raw
reconstructed responses. `calibration_responses` contains one finite positive
response per prepared dither mode. Reciprocal correction remains a separate
Algorithm.
"""
struct OpticalGainEstimatorPlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    schedule::OpticalGainDitherSchedule{T}
    calibration_responses::Vector{T}
    two_over_window::T

    function OpticalGainEstimatorPlan(
        schedule::OpticalGainDitherSchedule{T},
        calibration_responses::AbstractVector{T},
    ) where {T<:AbstractFloat}
        Base.require_one_based_indexing(calibration_responses)
        mode_count = length(schedule.modes)
        length(calibration_responses) == mode_count || throw(
            DimensionMismatch(
                "optical-gain calibration-response extent does not match the dither schedule",
            ),
        )
        responses = Vector{T}(undef, mode_count)
        @inbounds for mode in 1:mode_count
            response = calibration_responses[mode]
            isfinite(response) && response > zero(T) || throw(
                ArgumentError(
                    "optical-gain calibration response for mode $(mode - 1) must be finite and positive",
                ),
            )
            responses[mode] = response
        end
        window = T(schedule.frames_per_estimate)
        two_over_window = (one(T) + one(T)) / window
        isfinite(window) && isfinite(two_over_window) &&
            two_over_window > zero(T) || throw(
            ArgumentError(
                "optical-gain estimate-window extent is not representable by the scalar type",
            ),
        )
        return new{T}(schedule, responses, two_over_window)
    end
end

"""
    optical_gain_calibration_responses(plan::OpticalGainEstimatorPlan)

Return a copy of the prepared optical-gain calibration responses.
"""
optical_gain_calibration_responses(plan::OpticalGainEstimatorPlan) =
    copy(plan.calibration_responses)

"""
    OpticalGainEstimatorWorkspace(plan)

Own committed and candidate synchronous quadrature history plus fixed output
staging. One sample enters committed history when this Algorithm commits.
"""
mutable struct OpticalGainEstimatorWorkspace{T<:AbstractFloat}
    committed_frame::Int
    candidate_frame::Int
    committed_command_cosines::Vector{T}
    committed_command_sines::Vector{T}
    committed_response_cosines::Vector{T}
    committed_response_sines::Vector{T}
    candidate_command_cosines::Vector{T}
    candidate_command_sines::Vector{T}
    candidate_response_cosines::Vector{T}
    candidate_response_sines::Vector{T}
    staged_command_amplitudes::Vector{T}
    staged_response_amplitudes::Vector{T}
    staged_optical_gains::Vector{T}
    staged_phase_lags::Vector{T}
    staged_valid::Vector{Bool}
    candidate_ready::Bool
end

function OpticalGainEstimatorWorkspace(plan::OpticalGainEstimatorPlan{T}) where {T}
    mode_count = length(plan.schedule.modes)
    floats = ntuple(_ -> zeros(T, mode_count), Val(12))
    return OpticalGainEstimatorWorkspace{T}(
        0,
        0,
        floats...,
        falses(mode_count),
        false,
    )
end

"""
    optical_gain_estimator_frame(workspace::OpticalGainEstimatorWorkspace)

Return the committed zero-based frame expected by the estimator.
"""
optical_gain_estimator_frame(workspace::OpticalGainEstimatorWorkspace) =
    workspace.committed_frame

function commit!(workspace::OpticalGainEstimatorWorkspace)
    workspace.candidate_ready || return workspace
    workspace.committed_command_cosines,
    workspace.candidate_command_cosines =
        workspace.candidate_command_cosines,
        workspace.committed_command_cosines
    workspace.committed_command_sines,
    workspace.candidate_command_sines =
        workspace.candidate_command_sines,
        workspace.committed_command_sines
    workspace.committed_response_cosines,
    workspace.candidate_response_cosines =
        workspace.candidate_response_cosines,
        workspace.committed_response_cosines
    workspace.committed_response_sines,
    workspace.candidate_response_sines =
        workspace.candidate_response_sines,
        workspace.committed_response_sines
    workspace.committed_frame = workspace.candidate_frame
    workspace.candidate_ready = false
    return workspace
end

function discard!(workspace::OpticalGainEstimatorWorkspace)
    workspace.candidate_ready = false
    return workspace
end

function reset!(workspace::OpticalGainEstimatorWorkspace{T}) where {T}
    workspace.committed_frame = 0
    workspace.candidate_frame = 0
    for values in (
        workspace.committed_command_cosines,
        workspace.committed_command_sines,
        workspace.committed_response_cosines,
        workspace.committed_response_sines,
        workspace.candidate_command_cosines,
        workspace.candidate_command_sines,
        workspace.candidate_response_cosines,
        workspace.candidate_response_sines,
        workspace.staged_command_amplitudes,
        workspace.staged_response_amplitudes,
        workspace.staged_optical_gains,
        workspace.staged_phase_lags,
    )
        fill!(values, zero(T))
    end
    fill!(workspace.staged_valid, false)
    workspace.candidate_ready = false
    return workspace
end

"""Expected rejection from synchronous optical-gain estimation."""
@enum OpticalGainEstimatorFailure::UInt8 begin
    OpticalGainEstimatorWindowFrameOutOfRange = 1
    OpticalGainEstimatorWindowFrameMismatch = 2
    OpticalGainEstimatorPhaseIndexMismatch = 3
    OpticalGainEstimatorWindowCompletionMismatch = 4
    OpticalGainEstimatorNonFiniteCommand = 5
    OpticalGainEstimatorNonFiniteResponse = 6
    OpticalGainEstimatorNonFiniteAccumulation = 7
    OpticalGainEstimatorNonFiniteEstimate = 8
end

@doc "The supplied dither-window frame cannot be represented by `Int`."
OpticalGainEstimatorWindowFrameOutOfRange
@doc "The supplied dither-window frame is not the next estimator frame."
OpticalGainEstimatorWindowFrameMismatch
@doc "A supplied modal phase disagrees with the prepared dither schedule."
OpticalGainEstimatorPhaseIndexMismatch
@doc "The supplied completion flag disagrees with the prepared dither schedule."
OpticalGainEstimatorWindowCompletionMismatch
@doc "A realized modal dither command is non-finite."
OpticalGainEstimatorNonFiniteCommand
@doc "A raw reconstructed modal response is non-finite."
OpticalGainEstimatorNonFiniteResponse
@doc "Synchronous quadrature accumulation produced a non-finite value."
OpticalGainEstimatorNonFiniteAccumulation
@doc "Window completion produced a non-finite amplitude or phase."
OpticalGainEstimatorNonFiniteEstimate

@inline _optical_gain_estimator_any_alias(::AbstractArray, ::Tuple{}) = false

@inline function _optical_gain_estimator_any_alias(
    values::AbstractArray,
    remaining::Tuple,
)
    return Base.mightalias(values, first(remaining)) ||
           _optical_gain_estimator_any_alias(values, Base.tail(remaining))
end

@inline _optical_gain_estimator_outputs_overlap(::Tuple{}) = false

@inline function _optical_gain_estimator_outputs_overlap(outputs::Tuple)
    return _optical_gain_estimator_any_alias(first(outputs), Base.tail(outputs)) ||
           _optical_gain_estimator_outputs_overlap(Base.tail(outputs))
end

@inline _optical_gain_estimator_external_aliases_owned(::Tuple{}, ::Tuple) = false

@inline function _optical_gain_estimator_external_aliases_owned(
    external::Tuple,
    owned::Tuple,
)
    return _optical_gain_estimator_any_alias(first(external), owned) ||
           _optical_gain_estimator_external_aliases_owned(
        Base.tail(external),
        owned,
    )
end

@inline _optical_gain_estimator_extents_are_valid(::Tuple{}, ::Int) = true

@inline function _optical_gain_estimator_extents_are_valid(
    arrays::Tuple,
    expected::Int,
)
    return length(first(arrays)) == expected &&
           _optical_gain_estimator_extents_are_valid(
        Base.tail(arrays),
        expected,
    )
end

function _check_optical_gain_estimator_storage(
    command_amplitudes::AbstractVector,
    response_amplitudes::AbstractVector,
    optical_gains::AbstractVector,
    phase_lags::AbstractVector,
    valid::AbstractVector,
    estimate_complete::AbstractVector,
    workspace::OpticalGainEstimatorWorkspace,
    plan::OpticalGainEstimatorPlan,
    realized_command_modes::AbstractVector,
    reconstructed_response_modes::AbstractVector,
    window_frame::AbstractVector,
    phase_indices::AbstractVector,
    completes_window::AbstractVector,
)
    Base.require_one_based_indexing(
        command_amplitudes,
        response_amplitudes,
        optical_gains,
        phase_lags,
        valid,
        estimate_complete,
        realized_command_modes,
        reconstructed_response_modes,
        window_frame,
        phase_indices,
        completes_window,
    )
    mode_count = length(plan.schedule.modes)
    length(command_amplitudes) == mode_count || throw(
        DimensionMismatch("optical-gain command-amplitude output extent is invalid"),
    )
    length(response_amplitudes) == mode_count || throw(
        DimensionMismatch("optical-gain response-amplitude output extent is invalid"),
    )
    length(optical_gains) == mode_count || throw(
        DimensionMismatch("optical-gain output extent is invalid"),
    )
    length(phase_lags) == mode_count || throw(
        DimensionMismatch("optical-gain phase-lag output extent is invalid"),
    )
    length(valid) == mode_count || throw(
        DimensionMismatch("optical-gain validity output extent is invalid"),
    )
    length(realized_command_modes) == mode_count || throw(
        DimensionMismatch("optical-gain realized-command input extent is invalid"),
    )
    length(reconstructed_response_modes) == mode_count || throw(
        DimensionMismatch("optical-gain reconstructed-response input extent is invalid"),
    )
    length(phase_indices) == mode_count || throw(
        DimensionMismatch("optical-gain phase-index input extent is invalid"),
    )
    length(estimate_complete) == 1 || throw(
        DimensionMismatch("optical-gain estimate-complete output extent is invalid"),
    )
    length(window_frame) == 1 || throw(
        DimensionMismatch("optical-gain window-frame input extent is invalid"),
    )
    length(completes_window) == 1 || throw(
        DimensionMismatch("optical-gain completes-window input extent is invalid"),
    )

    workspace_arrays = (
        workspace.committed_command_cosines,
        workspace.committed_command_sines,
        workspace.committed_response_cosines,
        workspace.committed_response_sines,
        workspace.candidate_command_cosines,
        workspace.candidate_command_sines,
        workspace.candidate_response_cosines,
        workspace.candidate_response_sines,
        workspace.staged_command_amplitudes,
        workspace.staged_response_amplitudes,
        workspace.staged_optical_gains,
        workspace.staged_phase_lags,
        workspace.staged_valid,
    )
    _optical_gain_estimator_extents_are_valid(workspace_arrays, mode_count) || throw(
        DimensionMismatch("optical-gain estimator Workspace extent is invalid"),
    )
    0 <= workspace.committed_frame < plan.schedule.frames_per_estimate || throw(
        ArgumentError("optical-gain estimator Workspace frame is invalid"),
    )

    outputs = (
        command_amplitudes,
        response_amplitudes,
        optical_gains,
        phase_lags,
        valid,
        estimate_complete,
    )
    _optical_gain_estimator_outputs_overlap(outputs) && throw(
        ArgumentError("optical-gain estimator outputs must not overlap"),
    )
    external = (
        outputs...,
        realized_command_modes,
        reconstructed_response_modes,
        window_frame,
        phase_indices,
        completes_window,
    )
    owned = (
        workspace_arrays...,
        plan.calibration_responses,
        plan.schedule.modes,
        plan.schedule.offsets,
        plan.schedule.cosines,
        plan.schedule.sines,
    )
    _optical_gain_estimator_external_aliases_owned(external, owned) && throw(
        ArgumentError(
            "optical-gain estimator inputs and outputs must not overlap prepared or Workspace storage",
        ),
    )
    return nothing
end

@inline function _optical_gain_estimate(command_cosine::T, command_sine::T,
    response_cosine::T, response_sine::T, calibration::T) where {T}
    pi_value = acos(-one(T))
    two_pi = pi_value + pi_value
    command_amplitude = hypot(command_cosine, command_sine)
    response_amplitude = hypot(response_cosine, response_sine)
    isfinite(command_amplitude) && isfinite(response_amplitude) ||
        return command_amplitude, response_amplitude, zero(T), oftype(command_amplitude, NaN), false
    denominator = calibration * command_amplitude
    optical_gain = if command_amplitude > zero(T) &&
                      (!isfinite(denominator) || denominator < floatmin(T))
        calibration_fraction, calibration_exponent = frexp(calibration)
        command_fraction, command_exponent = frexp(command_amplitude)
        response_fraction, response_exponent = frexp(response_amplitude)
        ldexp(response_fraction / (calibration_fraction * command_fraction),
              response_exponent - calibration_exponent - command_exponent)
    else
        response_amplitude / denominator
    end
    accepted = command_amplitude > zero(T) &&
               isfinite(optical_gain) && optical_gain > zero(T)
    phase_lag = if accepted
        phase = atan(response_sine, response_cosine) -
                atan(command_sine, command_cosine)
        if phase > pi_value
            phase - two_pi
        elseif phase < -pi_value
            phase + two_pi
        else
            phase
        end
    else
        zero(T)
    end
    isfinite(phase_lag) || return command_amplitude, response_amplitude, zero(T), oftype(command_amplitude, NaN), false
    return command_amplitude, response_amplitude, accepted ? optical_gain : zero(T), phase_lag, accepted
end

function process!(
    command_amplitudes::AbstractVector{T},
    response_amplitudes::AbstractVector{T},
    optical_gains::AbstractVector{T},
    phase_lags::AbstractVector{T},
    valid::AbstractVector{Bool},
    estimate_complete::AbstractVector{Bool},
    workspace::OpticalGainEstimatorWorkspace{T},
    plan::OpticalGainEstimatorPlan{T},
    realized_command_modes::AbstractVector{T},
    reconstructed_response_modes::AbstractVector{T},
    window_frame::AbstractVector{UInt64},
    phase_indices::AbstractVector{UInt64},
    completes_window::AbstractVector{Bool},
) where {T<:AbstractFloat}
    _check_optical_gain_estimator_storage(
        command_amplitudes,
        response_amplitudes,
        optical_gains,
        phase_lags,
        valid,
        estimate_complete,
        workspace,
        plan,
        realized_command_modes,
        reconstructed_response_modes,
        window_frame,
        phase_indices,
        completes_window,
    )
    workspace.candidate_ready = false
    supplied_frame = window_frame[1]
    supplied_frame <= UInt64(typemax(Int)) ||
        return OpticalGainEstimatorWindowFrameOutOfRange
    frame = workspace.committed_frame
    Int(supplied_frame) == frame ||
        return OpticalGainEstimatorWindowFrameMismatch
    mode_count = length(plan.schedule.modes)
    @inbounds for mode in 1:mode_count
        expected_phase = frame % plan.schedule.modes[mode].frames_per_period
        phase_indices[mode] == UInt64(expected_phase) ||
            return OpticalGainEstimatorPhaseIndexMismatch
    end
    completes = frame + 1 == plan.schedule.frames_per_estimate
    completes_window[1] == completes ||
        return OpticalGainEstimatorWindowCompletionMismatch
    @inbounds for mode in 1:mode_count
        isfinite(realized_command_modes[mode]) ||
            return OpticalGainEstimatorNonFiniteCommand
        isfinite(reconstructed_response_modes[mode]) ||
            return OpticalGainEstimatorNonFiniteResponse
    end

    @inbounds for mode in 1:mode_count
        period = plan.schedule.modes[mode].frames_per_period
        phase = frame % period
        cosine = plan.schedule.cosines[plan.schedule.offsets[mode] + phase]
        sine = plan.schedule.sines[plan.schedule.offsets[mode] + phase]
        workspace.candidate_command_cosines[mode] =
            workspace.committed_command_cosines[mode] +
            realized_command_modes[mode] * cosine
        workspace.candidate_command_sines[mode] =
            workspace.committed_command_sines[mode] +
            realized_command_modes[mode] * sine
        workspace.candidate_response_cosines[mode] =
            workspace.committed_response_cosines[mode] +
            reconstructed_response_modes[mode] * cosine
        workspace.candidate_response_sines[mode] =
            workspace.committed_response_sines[mode] +
            reconstructed_response_modes[mode] * sine
        (isfinite(workspace.candidate_command_cosines[mode]) &&
         isfinite(workspace.candidate_command_sines[mode]) &&
         isfinite(workspace.candidate_response_cosines[mode]) &&
         isfinite(workspace.candidate_response_sines[mode])) ||
            return OpticalGainEstimatorNonFiniteAccumulation
    end

    if completes
        @inbounds for mode in 1:mode_count
            command_cosine = workspace.candidate_command_cosines[mode] *
                             plan.two_over_window
            command_sine = workspace.candidate_command_sines[mode] *
                           plan.two_over_window
            response_cosine = workspace.candidate_response_cosines[mode] *
                              plan.two_over_window
            response_sine = workspace.candidate_response_sines[mode] *
                            plan.two_over_window
            command_amplitude, response_amplitude, optical_gain, phase_lag, accepted =
                _optical_gain_estimate(command_cosine, command_sine, response_cosine, response_sine,
                    plan.calibration_responses[mode])
            isfinite(phase_lag) || return OpticalGainEstimatorNonFiniteEstimate
            workspace.staged_command_amplitudes[mode] = command_amplitude
            workspace.staged_response_amplitudes[mode] = response_amplitude
            workspace.staged_optical_gains[mode] = optical_gain
            workspace.staged_phase_lags[mode] = phase_lag
            workspace.staged_valid[mode] = accepted
        end
        copyto!(command_amplitudes, workspace.staged_command_amplitudes)
        copyto!(response_amplitudes, workspace.staged_response_amplitudes)
        copyto!(optical_gains, workspace.staged_optical_gains)
        copyto!(phase_lags, workspace.staged_phase_lags)
        copyto!(valid, workspace.staged_valid)
        estimate_complete[1] = true
        fill!(workspace.candidate_command_cosines, zero(T))
        fill!(workspace.candidate_command_sines, zero(T))
        fill!(workspace.candidate_response_cosines, zero(T))
        fill!(workspace.candidate_response_sines, zero(T))
        workspace.candidate_frame = 0
    else
        estimate_complete[1] = false
        workspace.candidate_frame = frame + 1
    end
    workspace.candidate_ready = true
    return nothing
end

function _replace_optical_gain_calibration_responses(
    plan::OpticalGainEstimatorPlan,
    values,
)
    return OpticalGainEstimatorPlan(plan.schedule, values)
end

"""
Float32 synchronous modal optical-gain estimator. The Node validates and
demodulates one shared dither schedule; Graph commit owns accumulation advance.
"""
@algorithm OpticalGainEstimatorF32 begin
    label = "optical-gain-estimator-f32"
    failure = OpticalGainEstimatorFailure

    """Configuration for [`OpticalGainEstimatorF32`](@ref)."""
    @configuration OpticalGainEstimatorF32Configuration begin
        modes::Vector{OpticalGainDitherMode{Float32}} => graph_rebuild
        initial_calibration_responses::Vector{Float32} => construction
    end

    ports(configuration) = (
            input(
                :realized_command_modes,
                schema_element_type(OPTICAL_GAIN_DITHER_COMMAND_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_DITHER_COMMAND_V1;
                name="realized-command-modes",
            ),
            input(
                :reconstructed_response_modes,
                schema_element_type(OPTICAL_GAIN_DITHER_RESPONSE_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_DITHER_RESPONSE_V1;
                name="reconstructed-response-modes",
            ),
            input(
                :window_frame,
                schema_element_type(OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1),
                (1,),
                OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1;
                name="window-frame",
            ),
            input(
                :phase_indices,
                schema_element_type(OPTICAL_GAIN_DITHER_PHASE_INDICES_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_DITHER_PHASE_INDICES_V1;
                name="phase-indices",
            ),
            input(
                :completes_window,
                schema_element_type(OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1),
                (1,),
                OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1;
                name="completes-window",
            ),
            output(
                :command_amplitudes,
                schema_element_type(OPTICAL_GAIN_COMMAND_AMPLITUDE_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_COMMAND_AMPLITUDE_V1;
                name="command-amplitudes",
                metadata_from=:reconstructed_response_modes,
            ),
            output(
                :response_amplitudes,
                schema_element_type(OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1;
                name="response-amplitudes",
                metadata_from=:reconstructed_response_modes,
            ),
            output(
                :optical_gains,
                schema_element_type(OPTICAL_GAIN_DITHER_ESTIMATE_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_DITHER_ESTIMATE_V1;
                name="optical-gains",
                metadata_from=:reconstructed_response_modes,
            ),
            output(
                :phase_lags,
                schema_element_type(OPTICAL_GAIN_PHASE_LAG_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_PHASE_LAG_V1;
                name="phase-lags",
                metadata_from=:reconstructed_response_modes,
            ),
            output(
                :valid,
                schema_element_type(OPTICAL_GAIN_ESTIMATE_VALIDITY_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_ESTIMATE_VALIDITY_V1;
                metadata_from=:reconstructed_response_modes,
            ),
            output(
                :estimate_complete,
                schema_element_type(OPTICAL_GAIN_ESTIMATE_COMPLETE_V1),
                (1,),
                OPTICAL_GAIN_ESTIMATE_COMPLETE_V1;
                name="estimate-complete",
                metadata_from=:reconstructed_response_modes,
            ),
            parameter(
                :calibration_responses,
                schema_element_type(OPTICAL_GAIN_CALIBRATION_RESPONSE_V1),
                (length(configuration.modes),),
                OPTICAL_GAIN_CALIBRATION_RESPONSE_V1;
                name="calibration-responses",
                replace=_replace_optical_gain_calibration_responses,
            ),
    )

    prepare(configuration) = begin
        schedule = OpticalGainDitherSchedule(configuration.modes)
        plan = OpticalGainEstimatorPlan(
            schedule,
            configuration.initial_calibration_responses,
        )
        (plan=plan, workspace=OpticalGainEstimatorWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(
            command_amplitudes,
            response_amplitudes,
            optical_gains,
            phase_lags,
            valid,
            estimate_complete,
            workspace,
            plan,
            realized_command_modes,
            reconstructed_response_modes,
            window_frame,
            phase_indices,
            completes_window,
        )
    end
end
