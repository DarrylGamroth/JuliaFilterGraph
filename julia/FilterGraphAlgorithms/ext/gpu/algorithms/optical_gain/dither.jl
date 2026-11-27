import FilterGraphAlgorithms: OpticalGainDitherF32

@kernel function _optical_gain_dither!(command, window_frame, phase_indices, completes_window,
        status, phases, frame, modes, offsets, cosines, frames_per_estimate)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            phase = phases[i]
            mode = modes[i]
            command[i] = mode.amplitude * cosines[offsets[i] + phase]
            phase_indices[i] = UInt64(phase)
            phases[i] = phase + 1 == mode.frames_per_period ? 0 : phase + 1
            if i == 1
                current = frame[1]
                completes = current + 1 == frames_per_estimate
                window_frame[1] = UInt64(current)
                completes_window[1] = completes
                frame[1] = completes ? 0 : current + 1
            end
        end
    end
end

capture_support(::Type{OpticalGainDitherF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{OpticalGainDitherF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status)
    FilterGraphAlgorithms._check_optical_gain_dither_storage(outputs..., prepared.workspace, prepared.plan)
    schedule, workspace = prepared.plan.schedule, prepared.workspace
    phases = _capture_array(device, workspace.committed_phases)
    frame = _capture_array(device, [workspace.committed_frame])
    kernel = _control_kernel(_optical_gain_dither!,
        (outputs..., status, phases, frame, _capture_array(device, schedule.modes),
            _capture_array(device, schedule.offsets), _capture_array(device, schedule.cosines),
            schedule.frames_per_estimate), length(phases))
    return ControlCapture((kernel,), (), ((phases, nothing), (frame, nothing)))
end
