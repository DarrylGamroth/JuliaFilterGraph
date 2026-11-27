@kernel function _scc_calibration_response!(staged, status, push, push_reference, pull, pull_reference)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds staged[index] = (push[index] - push_reference[index]) -
                                  (pull[index] - pull_reference[index])
    end
end

capture_support(::Type{SccPushPullResponseF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{SccPushPullResponseF32},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    output = only(outputs)
    staged = similar(output)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    response! = _scc_calibration_response!(backend, PROJECTION_WORKGROUP_SIZE)
    failures = _calibration_failures(node, SccPushPullNonFinitePushFringed,
        SccPushPullNonFinitePushUnfringed, SccPushPullNonFinitePullFringed,
        SccPushPullNonFinitePullUnfringed, SccPushPullNonFiniteCalculation)
    enqueue = function ()
        for (value, (code, _)) in zip(inputs, failures)
            _calibration_validate!(validate, status, value, code)
        end
        response!(staged, status, inputs...; ndrange=length(output))
        _calibration_validate!(validate, status, staged, first(failures[5]))
        _calibration_publish!(publish, status, output, staged)
    end
    return CalibrationTransition(enqueue, () -> nothing, failures)
end
