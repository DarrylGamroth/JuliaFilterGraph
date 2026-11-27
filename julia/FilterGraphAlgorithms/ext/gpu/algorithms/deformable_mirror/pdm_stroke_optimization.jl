import FilterGraphAlgorithms: PdmStrokeOptimizationF32, PdmStrokeOptimizationNonFiniteInput,
    PdmStrokeOptimizationNonFiniteOutput

capture_support(::Type{PdmStrokeOptimizationF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{PdmStrokeOptimizationF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:all)
    plan = prepared.plan
    output, requested = only(outputs), only(inputs)
    centered = similar(output)
    coefficients = _capture_zeros(device, Float32, plan.null_mode_count)
    preferred = _capture_array(device, plan.preferred_command)
    failures = _control_failures(node_index, (PdmStrokeOptimizationNonFiniteInput,
        PdmStrokeOptimizationNonFiniteOutput))
    input_failure, output_failure = map(first, failures)
    kernels = (
        _control_validate(status, requested, input_failure),
        _control_combine(centered, status, requested, preferred, -),
        _control_validate(status, centered, output_failure),
        _control_projection(device, coefficients, status, plan.command_to_null_modes, centered),
        _control_projection(device, centered, status, plan.null_modes_to_command, coefficients),
        _control_combine(centered, status, requested, centered, -),
        _control_validate(status, centered, output_failure),
        _control_copy(output, status, centered),
    )
    return ControlCapture(kernels, failures)
end
