import FilterGraphAlgorithms: OpticalGainUpdateF32, OpticalGainUpdateNonFiniteInput,
    OpticalGainUpdateNonFiniteOutput, _accept_optical_gain, _update_optical_gain

@kernel function _stage_accepted_optical_gains!(accepted, finite, status, criteria,
        command, response, gains, phase, valid, complete)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            accepted[i] = complete[1] &&
                _accept_optical_gain(criteria[i], command[i], response[i], gains[i], phase[i], valid[i])
            finite[i] = !complete[1] || (isfinite(command[i]) && isfinite(response[i]) &&
                isfinite(gains[i]) && isfinite(phase[i])) ? 0f0 : NaN32
        end
    end
end

@kernel function _stage_optical_gain_update!(candidate, updated, status, current,
        assignments, accepted, estimates)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            assignment = assignments[i]
            j = assignment.dither_mode + 1
            adopt = assignment.has_dither_mode && accepted[j]
            next = adopt ? _update_optical_gain(current[i], estimates[j], assignment.update_fraction) : current[i]
            candidate[i] = next > 0f0 ? next : NaN32
            updated[i] = adopt
        end
    end
end

@kernel function _finish_optical_gain_update!(gains, accepted_modes, updated_modes,
        status, current, candidate, accepted, updated)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        if i <= length(current)
            @inbounds begin
                value = candidate[i]
                current[i] = value
                gains[i] = value
                updated_modes[i] = updated[i]
            end
        end
        if i <= length(accepted)
            @inbounds accepted_modes[i] = accepted[i]
        end
    end
end

capture_support(::Type{OpticalGainUpdateF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{OpticalGainUpdateF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:all)
    gains, accepted_modes, updated_modes = outputs
    plan, ws = prepared.plan, prepared.workspace
    FilterGraphAlgorithms._check_optical_gain_update_storage(outputs..., ws, plan, inputs...)
    current = _capture_array(device, ws.committed_optical_gains)
    initial = _capture_array(device, ws.initial_optical_gains)
    candidate = similar(current)
    accepted = similar(accepted_modes)
    updated = similar(updated_modes)
    finite = _capture_zeros(device, Float32, length(accepted))
    criteria = _capture_array(device, plan.criteria)
    assignments = _capture_array(device, plan.assignments)
    failures = _control_failures(node_index, (OpticalGainUpdateNonFiniteInput, OpticalGainUpdateNonFiniteOutput))
    input_failure, output_failure = map(first, failures)
    kernels = (
        _control_kernel(_stage_accepted_optical_gains!,
            (accepted, finite, status, criteria, inputs...), length(accepted)),
        _control_validate(status, finite, input_failure),
        _control_kernel(_stage_optical_gain_update!,
            (candidate, updated, status, current, assignments, accepted, inputs[3]), length(current)),
        _control_validate(status, candidate, output_failure),
        _control_kernel(_finish_optical_gain_update!,
            (gains, accepted_modes, updated_modes, status, current, candidate, accepted, updated),
            max(length(current), length(accepted))),
    )
    return ControlCapture(kernels, failures, ((current, initial),))
end
