import FilterGraphAlgorithms:
    PreparedAlgorithm,
    SccWeightedRegularizedReconstructionInvalidMeasurement,
    SccWeightedRegularizedReconstructionInvalidWeight,
    SccWeightedRegularizedReconstructionNotConverged,
    SccWeightedRegularizedReconstructionPlan,
    SccWeightedRegularizedReconstructionSingularSystem,
    SccWeightedRegularizedReconstructorF32,
    complete!,
    enqueue!,
    preflight,
    reset!

include("weighted_regularized_steps.jl")

include("weighted_regularized_kernels.jl")

struct SccWeightedRegularized{
    Validate,
    Rhs,
    Start,
    Measurement,
    Regularization,
    Operator,
    Alpha,
    Update,
    Residual,
    Precondition,
    Beta,
    Direction,
    Finish,
    Verify,
    Publish,
    V,
    S,
    A,
    Execution,
}
    validate!::Validate
    rhs_diagonal!::Rhs
    start!::Start
    measurement_product!::Measurement
    regularization_product!::Regularization
    operator!::Operator
    alpha!::Alpha
    update!::Update
    residual!::Residual
    precondition!::Precondition
    beta!::Beta
    direction!::Direction
    finish!::Finish
    verify!::Verify
    publish!::Publish
    output::V
    frame_status::S
    measurements::V
    weights::V
    interaction::A
    regularization_operator::A
    candidate::V
    right_hand_side::V
    residual_vector::V
    preconditioned::V
    direction_vector::V
    operator_direction::V
    diagonal::V
    measurement_product_vector::V
    regularization_product_vector::V
    solve_state::S
    rho::V
    tolerance::V
    rhs_scale::V
    alpha_value::V
    beta_value::V
    regularization::Float32
    iterations::Int
    relative_tolerance::Float32
    minimum_diagonal::Float32
    invalid_weight_failure::Int32
    invalid_measurement_failure::Int32
    singular_failure::Int32
    not_converged_failure::Int32
    execution::Execution
end


_prepare_scc_execution(output, interaction, regularization) = nothing
_scc_adjoint_layout(captured, execution) = (captured.interaction, captured.regularization_operator)

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        SccWeightedRegularizedReconstructorF32,
        <:SccWeightedRegularizedReconstructionPlan,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 1 || throw(ArgumentError(
        "SccWeightedRegularizedReconstructorF32 capture requires one output",
    ))
    length(inputs) == 2 || throw(ArgumentError(
        "SccWeightedRegularizedReconstructorF32 capture requires two inputs",
    ))
    output = only(outputs)
    measurements, weights = inputs
    for (port, array) in (
        (:measurements, measurements),
        (:weights, weights),
        (Symbol("controller-residual-error"), output),
    )
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, port), array,
        )
    end
    for (name, array) in (
        ("SCC weighted measurements input", measurements),
        ("SCC weighted weights input", weights),
        ("SCC weighted controller-residual output", output),
        ("captured frame status", frame_status),
    )
        _require_capture_array(name, array, device)
    end

    plan = prepared.plan
    interaction = _capture_array(device, plan.interaction_matrix)
    regularization_operator =
        _capture_array(device, plan.regularization_operator)
    controller_count = size(plan.interaction_matrix, 2)
    measurement_count = size(plan.interaction_matrix, 1)
    penalty_count = size(plan.regularization_operator, 1)
    candidate = _capture_zeros(device, Float32, controller_count)
    right_hand_side = _capture_zeros(device, Float32, controller_count)
    residual = _capture_zeros(device, Float32, controller_count)
    preconditioned = _capture_zeros(device, Float32, controller_count)
    direction = _capture_zeros(device, Float32, controller_count)
    operator_direction = _capture_zeros(device, Float32, controller_count)
    diagonal = _capture_zeros(device, Float32, controller_count)
    measurement_product = _capture_zeros(device, Float32, measurement_count)
    regularization_product = _capture_zeros(device, Float32, penalty_count)
    solve_state = _capture_zeros(device, Int32, 1)
    rho = _capture_zeros(device, Float32, 1)
    tolerance = _capture_zeros(device, Float32, 1)
    rhs_scale = _capture_zeros(device, Float32, 1)
    alpha_value = _capture_zeros(device, Float32, 1)
    beta_value = _capture_zeros(device, Float32, 1)
    backend = KernelAbstractions.get_backend(output)
    execution = _prepare_scc_execution(output, interaction, regularization_operator)
    return SccWeightedRegularized(
        _validate_scc_weighted_samples!(backend, 1),
        _scc_weighted_rhs_diagonal!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _start_scc_weighted!(backend, 1),
        _scc_weighted_measurement_product!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_weighted_regularization_product!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_weighted_operator!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_weighted_alpha!(backend, 1),
        _update_scc_weighted!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _check_scc_weighted_residual!(backend, 1),
        _precondition_scc_weighted!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_weighted_beta!(backend, 1),
        _update_scc_weighted_direction!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _finish_scc_weighted!(backend, 1),
        _verify_scc_weighted!(backend, 1),
        _publish_scc_weighted!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        output,
        frame_status,
        measurements,
        weights,
        interaction,
        regularization_operator,
        candidate,
        right_hand_side,
        residual,
        preconditioned,
        direction,
        operator_direction,
        diagonal,
        measurement_product,
        regularization_product,
        solve_state,
        rho,
        tolerance,
        rhs_scale,
        alpha_value,
        beta_value,
        plan.regularization,
        plan.iterations,
        plan.relative_tolerance,
        plan.minimum_diagonal,
        _failure_code(
            node_index,
            SccWeightedRegularizedReconstructionInvalidWeight,
        ),
        _failure_code(
            node_index,
            SccWeightedRegularizedReconstructionInvalidMeasurement,
        ),
        _failure_code(
            node_index,
            SccWeightedRegularizedReconstructionSingularSystem,
        ),
        _failure_code(
            node_index,
            SccWeightedRegularizedReconstructionNotConverged,
        ),
        execution,
    )
end

preflight(::SccWeightedRegularized) = nothing

enqueue!(captured::SccWeightedRegularized) = _enqueue_scc!(captured, captured.execution)

function _enqueue_scc_operator!(captured, interaction, regularization_operator, vector)
    captured.measurement_product!(
        captured.measurement_product_vector,
        captured.frame_status,
        captured.solve_state,
        captured.interaction,
        captured.weights,
        vector;
        ndrange=length(captured.measurements),
    )
    if !iszero(captured.regularization)
        captured.regularization_product!(
            captured.regularization_product_vector,
            captured.frame_status,
            captured.solve_state,
            captured.regularization_operator,
            vector;
            ndrange=length(captured.regularization_product_vector),
        )
    end
    captured.operator!(
        captured.operator_direction,
        captured.frame_status,
        captured.solve_state,
        interaction,
        regularization_operator,
        captured.measurement_product_vector,
        captured.regularization_product_vector,
        captured.regularization;
        ndrange=length(captured.output),
    )
    return nothing
end

function _enqueue_scc!(captured::SccWeightedRegularized, execution)
    interaction, regularization_operator = _scc_adjoint_layout(captured, execution)
    captured.validate!(
        captured.frame_status,
        captured.solve_state,
        captured.measurements,
        captured.weights,
        captured.invalid_weight_failure,
        captured.invalid_measurement_failure;
        ndrange=1,
    )
    captured.rhs_diagonal!(
        captured.right_hand_side,
        captured.diagonal,
        captured.frame_status,
        captured.measurements,
        captured.weights,
        interaction,
        regularization_operator,
        captured.regularization,
        captured.minimum_diagonal,
        captured.singular_failure;
        ndrange=length(captured.output),
    )
    captured.start!(
        captured.frame_status,
        captured.solve_state,
        captured.rho,
        captured.tolerance,
        captured.rhs_scale,
        captured.candidate,
        captured.right_hand_side,
        captured.residual_vector,
        captured.preconditioned,
        captured.direction_vector,
        captured.diagonal,
        captured.relative_tolerance,
        captured.singular_failure;
        ndrange=1,
    )
    for _ in 1:captured.iterations
        _enqueue_scc_operator!(captured, interaction, regularization_operator,
            captured.direction_vector)
        captured.alpha!(
            captured.frame_status,
            captured.solve_state,
            captured.alpha_value,
            captured.rho,
            captured.direction_vector,
            captured.operator_direction,
            captured.singular_failure;
            ndrange=1,
        )
        captured.update!(
            captured.candidate,
            captured.residual_vector,
            captured.frame_status,
            captured.solve_state,
            captured.alpha_value,
            captured.direction_vector,
            captured.operator_direction;
            ndrange=length(captured.output),
        )
        captured.residual!(
            captured.frame_status,
            captured.solve_state,
            captured.residual_vector,
            captured.tolerance,
            captured.singular_failure;
            ndrange=1,
        )
        captured.precondition!(
            captured.preconditioned,
            captured.frame_status,
            captured.solve_state,
            captured.residual_vector,
            captured.diagonal;
            ndrange=length(captured.output),
        )
        captured.beta!(
            captured.frame_status,
            captured.solve_state,
            captured.rho,
            captured.beta_value,
            captured.residual_vector,
            captured.preconditioned,
            captured.singular_failure;
            ndrange=1,
        )
        captured.direction!(
            captured.direction_vector,
            captured.frame_status,
            captured.solve_state,
            captured.beta_value,
            captured.preconditioned;
            ndrange=length(captured.output),
        )
    end
    captured.finish!(
        captured.frame_status,
        captured.solve_state,
        captured.candidate,
        captured.rhs_scale,
        captured.not_converged_failure,
        captured.singular_failure;
        ndrange=1,
    )
    _enqueue_scc_operator!(captured, interaction, regularization_operator, captured.candidate)
    captured.verify!(captured.frame_status, captured.residual_vector, captured.right_hand_side,
        captured.operator_direction, captured.rhs_scale, captured.tolerance,
        captured.not_converged_failure, captured.singular_failure; ndrange=1)
    captured.publish!(
        captured.output,
        captured.frame_status,
        captured.candidate;
        ndrange=length(captured.output),
    )
    return nothing
end

function complete!(
    captured::SccWeightedRegularized,
    status::Int32,
)
    status == captured.invalid_weight_failure &&
        return SccWeightedRegularizedReconstructionInvalidWeight
    status == captured.invalid_measurement_failure &&
        return SccWeightedRegularizedReconstructionInvalidMeasurement
    status == captured.singular_failure &&
        return SccWeightedRegularizedReconstructionSingularSystem
    status == captured.not_converged_failure &&
        return SccWeightedRegularizedReconstructionNotConverged
    return nothing
end

reset!(captured::SccWeightedRegularized) = captured
