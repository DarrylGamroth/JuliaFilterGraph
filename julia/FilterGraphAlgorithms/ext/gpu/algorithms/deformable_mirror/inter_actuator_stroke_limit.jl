import FilterGraphAlgorithms: InterActuatorStrokeLimitF32, InterActuatorStrokeLimitNonFiniteInput,
    _inter_actuator_stroke_gain, _inter_actuator_product, _dm_widen

@kernel function _stroke_gain!(gain, status, requested, topology)
    if @inbounds iszero(status[1])
        @inbounds gain[1] = _inter_actuator_stroke_gain(topology, requested, Val(false))
    end
end

@inline _pdm_stroke_product(value, gain) =
    _inter_actuator_product(_dm_widen(value, Val(false)), gain, Val(false))

capture_support(::Type{InterActuatorStrokeLimitF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{InterActuatorStrokeLimitF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:exact)
    output, requested = only(outputs), only(inputs)
    plan = prepared.plan
    topology = (first_indices=_capture_array(device, plan.first_indices),
        second_indices=_capture_array(device, plan.second_indices),
        max_differences=_capture_array(device, plan.max_differences))
    gain = _capture_zeros(device, Float64, 1)
    failures = _control_failures(node_index, (InterActuatorStrokeLimitNonFiniteInput,))
    kernels = (
        _control_validate(status, requested, first(only(failures))),
        _control_kernel(_stroke_gain!, (gain, status, requested, topology), 1; lanes=1),
        _control_kernel(_apply_pdm_limit!,
            (output, status, requested, gain, _pdm_stroke_product), length(output)),
    )
    return ControlCapture(kernels, failures)
end
