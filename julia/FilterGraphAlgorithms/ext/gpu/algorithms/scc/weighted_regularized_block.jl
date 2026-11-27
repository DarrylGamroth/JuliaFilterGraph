# One workgroup owns bounded solves. Larger systems retain staged execution.
# Keep the existing 256-element storage/workgroup bound and at least 32 lanes;
# these are execution limits, independent of a backend's wavefront width.
const SCC_BLOCK_MAX_EXTENT = 256
const SCC_BLOCK_MIN_LANES = 32
struct _SccBlock{K,A,B}
    solve!::K
    interaction::A
    regularization::A
    lanes::Int
    barrier::B
end

struct _SccStages{A}
    interaction::A
    regularization::A
end

function _prepare_scc_execution(barrier, device, output, interaction, regularization)
    extent = max(size(interaction)..., size(regularization)...)
    if extent <= SCC_BLOCK_MAX_EXTENT
        # Each direction has its own prepared physical layout; no per-frame transpose.
        adjoint_interaction = transpose(permutedims(interaction))
        adjoint_regularization = transpose(permutedims(regularization))
        backend = KernelAbstractions.get_backend(output)
        lanes = max(SCC_BLOCK_MIN_LANES, nextpow(2, extent))
        _require_capture_workgroup(device, lanes)
        return _SccBlock(_scc_solve_block!(backend, lanes),
            adjoint_interaction, adjoint_regularization, lanes, barrier)
    end
    # The general solver retains its original layout: an extra transpose copy
    # regressed early-converged large solves in the interleaved screen.
    return _SccStages(interaction, regularization)
end

_scc_adjoint_layout(captured, execution::Union{_SccBlock,_SccStages}) =
    (execution.interaction, execution.regularization)

function _enqueue_scc!(captured::SccWeightedRegularized, execution::_SccBlock)
    state = (
        frame_status=captured.frame_status,
        solve_state=captured.solve_state,
        measurements=captured.measurements,
        weights=captured.weights,
        invalid_weight_failure=captured.invalid_weight_failure,
        invalid_measurement_failure=captured.invalid_measurement_failure,
        right_hand_side=captured.right_hand_side,
        diagonal=captured.diagonal,
        interaction=captured.interaction,
        regularization_operator=captured.regularization_operator,
        regularization=captured.regularization,
        minimum_diagonal=captured.minimum_diagonal,
        singular_failure=captured.singular_failure,
        rho=captured.rho,
        tolerance=captured.tolerance,
        rhs_scale=captured.rhs_scale,
        candidate=captured.candidate,
        residual_vector=captured.residual_vector,
        preconditioned=captured.preconditioned,
        direction_vector=captured.direction_vector,
        relative_tolerance=captured.relative_tolerance,
        measurement_product_vector=captured.measurement_product_vector,
        regularization_product_vector=captured.regularization_product_vector,
        operator_direction=captured.operator_direction,
        alpha_value=captured.alpha_value,
        beta_value=captured.beta_value,
        not_converged_failure=captured.not_converged_failure,
        output=captured.output,
        iterations=captured.iterations,
        adjoint_interaction=execution.interaction,
        adjoint_regularization=execution.regularization,
    )
    execution.solve!(state, execution.barrier; ndrange=execution.lanes)
    return nothing
end

@kernel unsafe_indices=true function _scc_solve_block!(state, barrier)
    lane = Int(@index(Local, Linear))
    regularize = !iszero(state.regularization)
    _scc_input!(
        state.frame_status, state.solve_state, state.measurements, state.weights,
        state.invalid_weight_failure, state.invalid_measurement_failure, lane,
    )
    barrier()
    _scc_rhs!(
        state.right_hand_side, state.diagonal, state.frame_status, state.measurements,
        state.weights, state.adjoint_interaction, state.adjoint_regularization,
        state.regularization, state.minimum_diagonal, state.singular_failure, lane,
    )
    barrier()
    _scc_start!(
        state.frame_status, state.solve_state, state.rho, state.tolerance, state.rhs_scale,
        state.candidate, state.right_hand_side, state.residual_vector, state.preconditioned,
        state.direction_vector, state.diagonal, state.relative_tolerance,
        state.singular_failure, lane,
    )
    barrier()
    for _ in 1:state.iterations
        # The preceding join makes both flags uniform across the entire block.
        if @inbounds(!iszero(state.frame_status[1]) || !iszero(state.solve_state[1]))
            break
        end
        _scc_forward!(
            state.measurement_product_vector, state.frame_status, state.solve_state,
            state.interaction, state.weights, state.direction_vector, lane,
        )
        if regularize
            _scc_penalty!(
                state.regularization_product_vector, state.frame_status, state.solve_state,
                state.regularization_operator, state.direction_vector, lane,
            )
        end
        barrier()
        _scc_adjoint!(
            state.operator_direction, state.frame_status, state.solve_state,
            state.adjoint_interaction, state.adjoint_regularization,
            state.measurement_product_vector, state.regularization_product_vector,
            state.regularization, lane,
        )
        barrier()
        _scc_alpha!(
            state.frame_status, state.solve_state, state.alpha_value, state.rho,
            state.direction_vector, state.operator_direction, state.singular_failure, lane,
        )
        barrier()
        _scc_update!(
            state.candidate, state.residual_vector, state.frame_status, state.solve_state,
            state.alpha_value, state.direction_vector, state.operator_direction, lane,
        )
        barrier()
        _scc_residual!(
            state.frame_status, state.solve_state, state.residual_vector, state.tolerance,
            state.singular_failure, lane,
        )
        barrier()
        _scc_precondition!(
            state.preconditioned, state.frame_status, state.solve_state, state.residual_vector,
            state.diagonal, lane,
        )
        barrier()
        _scc_beta!(
            state.frame_status, state.solve_state, state.rho, state.beta_value,
            state.residual_vector, state.preconditioned, state.singular_failure, lane,
        )
        barrier()
        _scc_direction!(
            state.direction_vector, state.frame_status, state.solve_state, state.beta_value,
            state.preconditioned, lane,
        )
        barrier()
    end
    _scc_finish!(
        state.frame_status, state.solve_state, state.candidate, state.rhs_scale,
        state.not_converged_failure, state.singular_failure, lane,
    )
    barrier()
    _scc_forward!(
        state.measurement_product_vector, state.frame_status, state.solve_state,
        state.interaction, state.weights, state.candidate, lane,
    )
    if regularize
        _scc_penalty!(
            state.regularization_product_vector, state.frame_status, state.solve_state,
            state.regularization_operator, state.candidate, lane,
        )
    end
    barrier()
    _scc_adjoint!(
        state.operator_direction, state.frame_status, state.solve_state,
        state.adjoint_interaction, state.adjoint_regularization,
        state.measurement_product_vector, state.regularization_product_vector,
        state.regularization, lane,
    )
    barrier()
    _scc_verify!(
        state.frame_status, state.residual_vector, state.right_hand_side,
        state.operator_direction, state.rhs_scale, state.tolerance,
        state.not_converged_failure, state.singular_failure, lane,
    )
    barrier()
    _scc_publish!(state.output, state.frame_status, state.candidate, lane)
end
