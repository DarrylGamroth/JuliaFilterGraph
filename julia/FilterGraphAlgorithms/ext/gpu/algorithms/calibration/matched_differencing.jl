@kernel function _calibration_difference!(difference, candidate, status, history, primed, input)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            value = input[index]
            candidate[index] = value
            difference[index] = primed[1] ? value - history[index] : zero(value)
        end
    end
end

@kernel function _calibration_prime!(filtered, primed, status)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds begin
            filtered[1] = primed[1]
            primed[1] = true
        end
    end
end

capture_support(::Type{MatchedCalibrationDifferencingF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{MatchedCalibrationDifferencingF32},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    measurements, corrector = inputs
    filtered_measurements, filtered_corrector, filtered = outputs
    ws = prepared.workspace
    FilterGraphAlgorithms._matched_workspace_extent_is_valid(ws, prepared.plan) ||
        throw(DimensionMismatch("matched calibration Workspace extent is invalid"))
    measurement_history = _capture_array(device, ws.committed_measurements)
    corrector_history = _capture_array(device, ws.committed_corrector_sample)
    primed = _capture_array(device, Bool[ws.committed_primed])
    candidates = map(similar, (measurement_history, corrector_history))
    differences = map(similar, candidates)
    histories = (measurement_history, corrector_history)
    data_outputs = (filtered_measurements, filtered_corrector)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    difference! = _calibration_difference!(backend, PROJECTION_WORKGROUP_SIZE)
    prime! = _calibration_prime!(backend, 1)
    failures = _calibration_failures(node, MatchedCalibrationNonFiniteMeasurements,
        MatchedCalibrationNonFiniteCorrectorSample, MatchedCalibrationNonFiniteDifference)
    enqueue = function ()
        for (value, (code, _)) in zip(inputs, failures)
            _calibration_validate!(validate, status, value, code)
        end
        for (difference, candidate, history, input) in zip(differences, candidates, histories, inputs)
            difference!(difference, candidate, status, history, primed, input; ndrange=length(input))
        end
        for difference in differences
            _calibration_validate!(validate, status, difference, first(failures[3]))
        end
        for (history, candidate, output, difference) in zip(histories, candidates, data_outputs, differences)
            _calibration_publish!(publish, status, history, candidate)
            _calibration_publish!(publish, status, output, difference)
        end
        prime!(filtered, primed, status; ndrange=1)
        return nothing
    end
    # History is never read unprimed, and both histories are replaced together.
    return CalibrationTransition(enqueue, () -> fill!(primed, false), failures)
end
