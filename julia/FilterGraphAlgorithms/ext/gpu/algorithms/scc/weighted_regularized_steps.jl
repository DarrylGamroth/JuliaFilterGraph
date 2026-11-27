# Device-callable scientific steps shared by staged kernels and bounded GPU execution.
# Lane order, scaling, checks and candidate publication match the checked recurrence.

@inline function _scc_input!(
    frame_status, solve_state, measurements, weights, invalid_weight_failure,
    invalid_measurement_failure, index,
)
    if index == 1
        @inbounds solve_state[1] = Int32(0)
        if @inbounds(iszero(frame_status[1]))
            for measurement in 1:length(measurements)
                @inbounds weight = weights[measurement]
                if !isfinite(weight) || weight < zero(eltype(weights))
                    @inbounds frame_status[1] = invalid_weight_failure
                    break
                elseif !iszero(weight) && @inbounds(!isfinite(measurements[measurement]))
                    @inbounds frame_status[1] = invalid_measurement_failure
                    break
                end
            end
        end
    end
    return nothing
end

@inline function _scc_rhs!(
    right_hand_side, diagonal, frame_status, measurements, weights, interaction,
    regularization_operator, regularization, minimum_diagonal, singular_failure, controller,
)
    frame_valid =
        @inbounds iszero(KernelAbstractions.@atomic frame_status[1])
    if controller <= length(right_hand_side) && frame_valid
        rhs_value = zero(eltype(right_hand_side))
        diagonal_value = zero(eltype(diagonal))
        for measurement in 1:length(measurements)
            @inbounds weight = weights[measurement]
            if !iszero(weight)
                @inbounds coefficient = interaction[measurement, controller]
                @inbounds rhs_value = muladd(
                    coefficient,
                    weight * measurements[measurement],
                    rhs_value,
                )
                diagonal_value = muladd(
                    weight * coefficient,
                    coefficient,
                    diagonal_value,
                )
            end
        end
        if !iszero(regularization)
            for penalty in axes(regularization_operator, 1)
                @inbounds coefficient = regularization_operator[penalty, controller]
                diagonal_value = muladd(
                    regularization * coefficient,
                    coefficient,
                    diagonal_value,
                )
            end
        end
        if !isfinite(diagonal_value) || diagonal_value < minimum_diagonal
            @inbounds KernelAbstractions.@atomic frame_status[1] =
                singular_failure
        else
            @inbounds begin
                right_hand_side[controller] = rhs_value
                diagonal[controller] = diagonal_value
            end
        end
    end
    return nothing
end

@inline function _scc_start!(
    frame_status, solve_state, rho_storage, tolerance_storage, rhs_scale_storage, candidate,
    right_hand_side, residual, preconditioned, direction, diagonal, relative_tolerance,
    singular_failure, index,
)
    if index == 1 && @inbounds(iszero(frame_status[1]))
        T = eltype(right_hand_side)
        rhs_scale = zero(T)
        for controller in eachindex(right_hand_side)
            @inbounds value = right_hand_side[controller]
            if !isfinite(value)
                @inbounds frame_status[1] = singular_failure
                break
            end
            rhs_scale = max(rhs_scale, abs(value))
        end
        @inbounds rhs_scale_storage[1] = rhs_scale
        if @inbounds(iszero(frame_status[1]))
            for controller in eachindex(right_hand_side)
                @inbounds begin
                    candidate[controller] = zero(T)
                    value = iszero(rhs_scale) ? zero(T) : right_hand_side[controller] / rhs_scale
                    residual[controller] = value
                    preconditioned[controller] = value / diagonal[controller]
                    direction[controller] = preconditioned[controller]
                end
            end
            if iszero(rhs_scale)
                @inbounds solve_state[1] = Int32(1)
            else
                norm_rhs = @inline FilterGraphAlgorithms._scc_norm(residual)
                rho = zero(T)
                for controller in eachindex(residual)
                    @inbounds rho = muladd(residual[controller], preconditioned[controller], rho)
                end
                if !isfinite(rho) || rho <= zero(T)
                    @inbounds frame_status[1] = singular_failure
                else
                    @inbounds begin
                        rho_storage[1] = rho
                        tolerance_storage[1] = relative_tolerance * norm_rhs
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _scc_forward!(
    measurement_product, frame_status, solve_state, interaction, weights, direction,
    measurement,
)
    if measurement <= length(measurement_product) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        @inbounds weight = weights[measurement]
        value = zero(eltype(measurement_product))
        # Publish zero to the adjoint scratch without evaluating the masked row.
        if !iszero(weight)
            for controller in eachindex(direction)
                @inbounds value = muladd(
                    interaction[measurement, controller],
                    direction[controller],
                    value,
                )
            end
            value = weight * value
        end
        @inbounds measurement_product[measurement] = value
    end
    return nothing
end

@inline function _scc_penalty!(
    regularization_product, frame_status, solve_state, regularization_operator, direction,
    penalty,
)
    if penalty <= length(regularization_product) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        value = zero(eltype(regularization_product))
        for controller in eachindex(direction)
            @inbounds value = muladd(
                regularization_operator[penalty, controller],
                direction[controller],
                value,
            )
        end
        @inbounds regularization_product[penalty] = value
    end
    return nothing
end

@inline function _scc_adjoint!(
    operator_direction, frame_status, solve_state, interaction, regularization_operator,
    measurement_product, regularization_product, regularization, controller,
)
    if controller <= length(operator_direction) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        value = zero(eltype(operator_direction))
        for measurement in eachindex(measurement_product)
            @inbounds value = muladd(
                interaction[measurement, controller],
                measurement_product[measurement],
                value,
            )
        end
        if !iszero(regularization)
            regularized = zero(eltype(operator_direction))
            for penalty in eachindex(regularization_product)
                @inbounds regularized = muladd(
                    regularization_operator[penalty, controller],
                    regularization_product[penalty],
                    regularized,
                )
            end
            value = muladd(regularization, regularized, value)
        end
        @inbounds operator_direction[controller] = value
    end
    return nothing
end

@inline function _scc_alpha!(
    frame_status, solve_state, alpha_storage, rho_storage, direction, operator_direction,
    singular_failure, index,
)
    if index == 1 &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        denominator = zero(eltype(direction))
        for controller in 1:length(direction)
            @inbounds denominator = muladd(
                direction[controller],
                operator_direction[controller],
                denominator,
            )
        end
        if !isfinite(denominator) || denominator <= zero(denominator)
            @inbounds frame_status[1] = singular_failure
        else
            @inbounds alpha_storage[1] = rho_storage[1] / denominator
        end
    end
    return nothing
end

@inline function _scc_update!(
    candidate, residual, frame_status, solve_state, alpha_storage, direction,
    operator_direction, controller,
)
    if controller <= length(candidate) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        @inbounds alpha = alpha_storage[1]
        @inbounds begin
            candidate[controller] =
                muladd(alpha, direction[controller], candidate[controller])
            residual[controller] = muladd(
                -alpha,
                operator_direction[controller],
                residual[controller],
            )
        end
    end
    return nothing
end

@inline function _scc_residual!(
    frame_status, solve_state, residual, tolerance_storage, singular_failure, index,
)
    if index == 1 &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        residual_norm = @inline FilterGraphAlgorithms._scc_norm(residual)
        if !isfinite(residual_norm)
            @inbounds frame_status[1] = singular_failure
        elseif @inbounds(residual_norm <= tolerance_storage[1])
            @inbounds solve_state[1] = Int32(1)
        end
    end
    return nothing
end

@inline function _scc_precondition!(
    preconditioned, frame_status, solve_state, residual, diagonal, controller,
)
    if controller <= length(preconditioned) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        @inbounds preconditioned[controller] =
            residual[controller] / diagonal[controller]
    end
    return nothing
end

@inline function _scc_beta!(
    frame_status, solve_state, rho_storage, beta_storage, residual, preconditioned,
    singular_failure, index,
)
    if index == 1 &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        next_rho = zero(eltype(residual))
        for controller in 1:length(residual)
            @inbounds next_rho = muladd(
                residual[controller],
                preconditioned[controller],
                next_rho,
            )
        end
        if !isfinite(next_rho) || next_rho <= zero(next_rho)
            @inbounds frame_status[1] = singular_failure
        else
            @inbounds begin
                beta_storage[1] = next_rho / rho_storage[1]
                rho_storage[1] = next_rho
            end
        end
    end
    return nothing
end

@inline function _scc_direction!(
    direction, frame_status, solve_state, beta_storage, preconditioned, controller,
)
    if controller <= length(direction) &&
       @inbounds(iszero(frame_status[1]) && iszero(solve_state[1]))
        @inbounds direction[controller] = muladd(
            beta_storage[1],
            direction[controller],
            preconditioned[controller],
        )
    end
    return nothing
end

@inline function _scc_finish!(
    frame_status, solve_state, candidate, rhs_scale_storage, not_converged_failure,
    singular_failure, index,
)
    if index == 1 && @inbounds(iszero(frame_status[1]))
        if @inbounds(iszero(solve_state[1]))
            @inbounds frame_status[1] = not_converged_failure
        else
            @inbounds rhs_scale = rhs_scale_storage[1]
            failure = FilterGraphAlgorithms._rescale_scc_candidate!(candidate, rhs_scale)
            if failure !== nothing
                @inbounds frame_status[1] = singular_failure
            else
                # Enable one more application of the operator to the candidate.
                @inbounds solve_state[1] = Int32(0)
            end
        end
    end
    return nothing
end

@inline function _scc_verify!(
    frame_status, residual, right_hand_side, operator_candidate, rhs_scale_storage,
    tolerance_storage, not_converged_failure, singular_failure, index,
)
    if index == 1 && @inbounds(iszero(frame_status[1]))
        @inbounds rhs_scale = rhs_scale_storage[1]
        if !iszero(rhs_scale)
            for controller in eachindex(residual)
                @inbounds residual[controller] =
                    (right_hand_side[controller] - operator_candidate[controller]) / rhs_scale
            end
            residual_norm = @inline FilterGraphAlgorithms._scc_norm(residual)
            if !isfinite(residual_norm)
                @inbounds frame_status[1] = singular_failure
            elseif @inbounds residual_norm > tolerance_storage[1]
                @inbounds frame_status[1] = not_converged_failure
            end
        end
    end
    return nothing
end

@inline function _scc_publish!(output, frame_status, candidate, controller)
    if controller <= length(output) && @inbounds(iszero(frame_status[1]))
        @inbounds output[controller] = candidate[controller]
    end
    return nothing
end
