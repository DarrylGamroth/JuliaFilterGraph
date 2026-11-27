import FilterGraphAlgorithms: PhysicalOffsetReferenceF32

capture_support(::Type{PhysicalOffsetReferenceF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{PhysicalOffsetReferenceF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status)
    plan = prepared.plan
    compensated, physical = outputs
    residual = only(inputs)
    correction = _capture_array(device, plan.correction)
    offset = _capture_array(device, plan.offset)
    candidate = similar(compensated)
    failures = _control_failures(node_index, (VectorLinearCombinationNonFiniteInput,
        VectorLinearCombinationNonFiniteOutput))
    input_failure, output_failure = map(first, failures)
    kernels = (
        _control_validate(status, residual, input_failure),
        _control_kernel(_vector_combination_candidate!,
            (candidate, residual, correction, 1f0, 1f0, status), length(candidate)),
        _control_validate(status, candidate, output_failure),
        _control_kernel(_publish_vector_combination!,
            (compensated, candidate, physical, offset, status), max(length(compensated), length(physical))),
    )
    return ControlCapture(kernels, failures)
end
