# Staged launch boundaries for the shared weighted SCC scientific steps.

@kernel function _validate_scc_weighted_samples!(
    frame_status,
    solve_state,
    measurements,
    weights,
    invalid_weight_failure,
    invalid_measurement_failure,
)
    index = @index(Global, Linear)
    _scc_input!(
        frame_status, solve_state, measurements, weights, invalid_weight_failure,
        invalid_measurement_failure, index,
    )
end

@kernel function _scc_weighted_rhs_diagonal!(
    right_hand_side,
    diagonal,
    frame_status,
    measurements,
    weights,
    interaction,
    regularization_operator,
    regularization,
    minimum_diagonal,
    singular_failure,
)
    controller = @index(Global, Linear)
    _scc_rhs!(
        right_hand_side, diagonal, frame_status, measurements, weights, interaction,
        regularization_operator, regularization, minimum_diagonal, singular_failure,
        controller,
    )
end

@kernel function _start_scc_weighted!(
    frame_status,
    solve_state,
    rho_storage,
    tolerance_storage,
    rhs_scale_storage,
    candidate,
    right_hand_side,
    residual,
    preconditioned,
    direction,
    diagonal,
    relative_tolerance,
    singular_failure,
)
    index = @index(Global, Linear)
    _scc_start!(
        frame_status, solve_state, rho_storage, tolerance_storage, rhs_scale_storage,
        candidate, right_hand_side, residual, preconditioned, direction, diagonal,
        relative_tolerance, singular_failure, index,
    )
end

@kernel function _scc_weighted_measurement_product!(
    measurement_product,
    frame_status,
    solve_state,
    interaction,
    weights,
    direction,
)
    measurement = @index(Global, Linear)
    _scc_forward!(
        measurement_product, frame_status, solve_state, interaction, weights, direction,
        measurement,
    )
end

@kernel function _scc_weighted_regularization_product!(
    regularization_product,
    frame_status,
    solve_state,
    regularization_operator,
    direction,
)
    penalty = @index(Global, Linear)
    _scc_penalty!(
        regularization_product, frame_status, solve_state, regularization_operator, direction,
        penalty,
    )
end

@kernel function _scc_weighted_operator!(
    operator_direction,
    frame_status,
    solve_state,
    interaction,
    regularization_operator,
    measurement_product,
    regularization_product,
    regularization,
)
    controller = @index(Global, Linear)
    _scc_adjoint!(
        operator_direction, frame_status, solve_state, interaction, regularization_operator,
        measurement_product, regularization_product, regularization, controller,
    )
end

@kernel function _scc_weighted_alpha!(
    frame_status,
    solve_state,
    alpha_storage,
    rho_storage,
    direction,
    operator_direction,
    singular_failure,
)
    index = @index(Global, Linear)
    _scc_alpha!(
        frame_status, solve_state, alpha_storage, rho_storage, direction, operator_direction,
        singular_failure, index,
    )
end

@kernel function _update_scc_weighted!(
    candidate,
    residual,
    frame_status,
    solve_state,
    alpha_storage,
    direction,
    operator_direction,
)
    controller = @index(Global, Linear)
    _scc_update!(
        candidate, residual, frame_status, solve_state, alpha_storage, direction,
        operator_direction, controller,
    )
end

@kernel function _check_scc_weighted_residual!(
    frame_status,
    solve_state,
    residual,
    tolerance_storage,
    singular_failure,
)
    index = @index(Global, Linear)
    _scc_residual!(frame_status, solve_state, residual, tolerance_storage, singular_failure, index)
end

@kernel function _precondition_scc_weighted!(
    preconditioned,
    frame_status,
    solve_state,
    residual,
    diagonal,
)
    controller = @index(Global, Linear)
    _scc_precondition!(preconditioned, frame_status, solve_state, residual, diagonal, controller)
end

@kernel function _scc_weighted_beta!(
    frame_status,
    solve_state,
    rho_storage,
    beta_storage,
    residual,
    preconditioned,
    singular_failure,
)
    index = @index(Global, Linear)
    _scc_beta!(
        frame_status, solve_state, rho_storage, beta_storage, residual, preconditioned,
        singular_failure, index,
    )
end

@kernel function _update_scc_weighted_direction!(
    direction,
    frame_status,
    solve_state,
    beta_storage,
    preconditioned,
)
    controller = @index(Global, Linear)
    _scc_direction!(direction, frame_status, solve_state, beta_storage, preconditioned, controller)
end

@kernel function _finish_scc_weighted!(
    frame_status,
    solve_state,
    candidate,
    rhs_scale_storage,
    not_converged_failure,
    singular_failure,
)
    index = @index(Global, Linear)
    _scc_finish!(
        frame_status, solve_state, candidate, rhs_scale_storage, not_converged_failure,
        singular_failure, index,
    )
end

@kernel function _verify_scc_weighted!(
    frame_status, residual, right_hand_side, operator_candidate, rhs_scale_storage,
    tolerance_storage, not_converged_failure, singular_failure,
)
    index = @index(Global, Linear)
    _scc_verify!(frame_status, residual, right_hand_side, operator_candidate,
        rhs_scale_storage, tolerance_storage, not_converged_failure, singular_failure, index)
end

@kernel function _publish_scc_weighted!(
    output,
    frame_status,
    candidate,
)
    controller = @index(Global, Linear)
    _scc_publish!(output, frame_status, candidate, controller)
end
