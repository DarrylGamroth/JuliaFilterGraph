import FilterGraphAlgorithms: OpticalGainEstimatorF32, OpticalGainEstimatorWindowFrameOutOfRange,
    OpticalGainEstimatorWindowFrameMismatch, OpticalGainEstimatorPhaseIndexMismatch,
    OpticalGainEstimatorWindowCompletionMismatch, OpticalGainEstimatorNonFiniteCommand,
    OpticalGainEstimatorNonFiniteResponse, OpticalGainEstimatorNonFiniteAccumulation,
    OpticalGainEstimatorNonFiniteEstimate, _optical_gain_estimate

@kernel function _admit_optical_gain_estimate!(status, frame, command, response, window_frame,
        phase_indices, completes_window, periods, window, failures)
    if @inbounds iszero(status[1])
        @inbounds begin
            supplied = window_frame[1]
            expected = frame[1]
            if supplied > UInt64(typemax(Int))
                status[1] = failures[1]
            elseif supplied != UInt64(expected)
                status[1] = failures[2]
            else
                for i in eachindex(periods)
                    if phase_indices[i] != UInt64(expected % periods[i])
                        status[1] = failures[3]
                        break
                    end
                end
                if iszero(status[1])
                    if completes_window[1] != (expected + 1 == window)
                        status[1] = failures[4]
                    else
                        for i in eachindex(command)
                            if !isfinite(command[i])
                                status[1] = failures[5]
                                break
                            elseif !isfinite(response[i])
                                status[1] = failures[6]
                                break
                            end
                        end
                    end
                end
            end
        end
    end
end

@kernel function _stage_optical_gain_estimate!(accumulation, estimates, accepted, status,
        committed, command, response, phase_indices, completes_window,
        offsets, cosines, sines, calibrations, two_over_window)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            index = offsets[i] + Int(phase_indices[i])
            cosine, sine = cosines[index], sines[index]
            cc = committed[1, i] + command[i] * cosine
            cs = committed[2, i] + command[i] * sine
            rc = committed[3, i] + response[i] * cosine
            rs = committed[4, i] + response[i] * sine
            accumulation[1, i] = cc
            accumulation[2, i] = cs
            accumulation[3, i] = rc
            accumulation[4, i] = rs
            if completes_window[1]
                a, b, gain, phase, valid = _optical_gain_estimate(
                    cc * two_over_window, cs * two_over_window,
                    rc * two_over_window, rs * two_over_window, calibrations[i])
                estimates[1, i] = a
                estimates[2, i] = b
                estimates[3, i] = gain
                estimates[4, i] = phase
                accepted[i] = valid
            end
        end
    end
end

@kernel function _check_optical_gain_estimates!(status, results, completes_window, failure)
    if @inbounds iszero(status[1]) && completes_window[1]
        @inbounds for i in axes(results, 2)
            if !isfinite(results[4, i])
                status[1] = failure
                break
            end
        end
    end
end

@kernel function _finish_optical_gain_estimate!(command, response, optical_gains, phase_lags,
        valid, complete, status, committed, frame, accumulation, estimates, accepted, completes_window)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            finishes = completes_window[1]
            if finishes
                command[i] = estimates[1, i]
                response[i] = estimates[2, i]
                optical_gains[i] = estimates[3, i]
                phase_lags[i] = estimates[4, i]
                valid[i] = accepted[i]
            end
            for quadrature in 1:4
                committed[quadrature, i] = finishes ? 0f0 : accumulation[quadrature, i]
            end
            if i == 1
                complete[1] = finishes
                frame[1] = finishes ? 0 : frame[1] + 1
            end
        end
    end
end

capture_support(::Type{OpticalGainEstimatorF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{OpticalGainEstimatorF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:all)
    plan, ws = prepared.plan, prepared.workspace
    FilterGraphAlgorithms._check_optical_gain_estimator_storage(outputs..., ws, plan, inputs...)
    schedule = plan.schedule
    modes = length(schedule.modes)
    frame = _capture_array(device, [ws.committed_frame])
    committed = _capture_array(device, permutedims(hcat(ws.committed_command_cosines,
        ws.committed_command_sines, ws.committed_response_cosines, ws.committed_response_sines)))
    accumulation = similar(committed)
    estimates = similar(committed)
    accepted = _capture_zeros(device, Bool, modes)
    periods = _capture_array(device, map(mode -> mode.frames_per_period, schedule.modes))
    offsets = _capture_array(device, schedule.offsets)
    cosines = _capture_array(device, schedule.cosines)
    sines = _capture_array(device, schedule.sines)
    calibrations = _capture_array(device, plan.calibration_responses)
    failures = _control_failures(node_index, (OpticalGainEstimatorWindowFrameOutOfRange,
        OpticalGainEstimatorWindowFrameMismatch, OpticalGainEstimatorPhaseIndexMismatch,
        OpticalGainEstimatorWindowCompletionMismatch, OpticalGainEstimatorNonFiniteCommand,
        OpticalGainEstimatorNonFiniteResponse, OpticalGainEstimatorNonFiniteAccumulation,
        OpticalGainEstimatorNonFiniteEstimate))
    codes = map(first, failures)
    command, response, window_frame, phases, completion = inputs
    completes_window = similar(completion)
    kernels = (
        _control_kernel(_admit_optical_gain_estimate!, (status, frame, command, response,
            window_frame, phases, completion, periods, schedule.frames_per_estimate, codes), 1; lanes=1),
        _control_copy(completes_window, status, completion),
        _control_kernel(_stage_optical_gain_estimate!, (accumulation, estimates, accepted,
            status, committed, command, response, phases, completes_window, offsets, cosines,
            sines, calibrations, plan.two_over_window), modes),
        _control_validate(status, accumulation, codes[7]),
        _control_kernel(_check_optical_gain_estimates!,
            (status, estimates, completes_window, codes[8]), 1; lanes=1),
        _control_kernel(_finish_optical_gain_estimate!, (outputs..., status, committed,
            frame, accumulation, estimates, accepted, completes_window), modes),
    )
    return ControlCapture(kernels, failures, ((committed, nothing), (frame, nothing)))
end
