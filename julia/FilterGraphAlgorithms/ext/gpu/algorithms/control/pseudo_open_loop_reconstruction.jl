import FilterGraphAlgorithms: PseudoOpenLoopReconstructionF32, PseudoOpenLoopNonFiniteResidualMeasurements,
    PseudoOpenLoopNonFiniteCorrectorShape, PseudoOpenLoopNonFiniteOutput

capture_support(::Type{PseudoOpenLoopReconstructionF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{PseudoOpenLoopReconstructionF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:all)
    plan = prepared.plan
    measurements, residual_error = outputs
    residual, corrector = inputs
    predicted = similar(measurements)
    staged_measurements = similar(measurements)
    estimate = similar(residual_error)
    corrector_in_error = similar(residual_error)
    staged_error = similar(residual_error)
    failures = _control_failures(node_index, (PseudoOpenLoopNonFiniteResidualMeasurements,
        PseudoOpenLoopNonFiniteCorrectorShape, PseudoOpenLoopNonFiniteOutput))
    residual_failure, corrector_failure, output_failure = map(first, failures)
    kernels = (
        _control_validate(status, residual, residual_failure),
        _control_validate(status, corrector, corrector_failure),
        _control_projection(device, predicted, status, plan.interaction_matrix, corrector),
        _control_combine(staged_measurements, status, residual, predicted, +),
        _control_validate(status, staged_measurements, output_failure),
        _control_projection(device, estimate, status, plan.reconstructor, staged_measurements),
        _control_projection(device, corrector_in_error, status, plan.corrector_to_error, corrector),
        _control_combine(staged_error, status, estimate, corrector_in_error, -),
        _control_validate(status, staged_error, output_failure),
        _control_copy(measurements, status, staged_measurements),
        _control_copy(residual_error, status, staged_error),
    )
    return ControlCapture(kernels, failures)
end
