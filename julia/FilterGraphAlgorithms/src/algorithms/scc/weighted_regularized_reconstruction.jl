"""Weighted regularized SCC reconstruction recognized by execution adapters."""
struct SccWeightedRegularizedReconstructionF32Operation <: AlgorithmOperation end

"""
Prepared interaction matrix, regularization operator, and bounded conjugate-
gradient policy for dynamic weighted regularized SCC reconstruction.

The assembled right-hand side is normalized before the PCG recurrence and the
solution is rescaled after convergence. This prevents measurement amplitude
alone from underflowing/overflowing the recurrence's scalar products. Assembly,
operator application, and the rescaled solution must still be representable in
the prepared precision. The relative residual is checked against the rescaled
solution before publishing it. Zero-weight measurement rows and a regularization
operator with zero regularization contribute nothing to the system.
"""
struct SccWeightedRegularizedReconstructionPlan{
    T<:Union{Float32,Float64},
    H<:AbstractMatrix{T},
    L<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    interaction_matrix::H
    regularization_operator::L
    regularization::T
    iterations::Int
    relative_tolerance::T
    minimum_diagonal::T
end

function _replace_scc_weighted_regularized_plan(
    plan::SccWeightedRegularizedReconstructionPlan;
    interaction_matrix=plan.interaction_matrix,
    regularization_operator=plan.regularization_operator,
)
    return SccWeightedRegularizedReconstructionPlan(
        interaction_matrix,
        regularization_operator;
        regularization=plan.regularization,
        iterations=plan.iterations,
        relative_tolerance=plan.relative_tolerance,
        minimum_diagonal=plan.minimum_diagonal,
    )
end

function SccWeightedRegularizedReconstructionPlan(
    interaction_matrix::AbstractMatrix{T},
    regularization_operator::AbstractMatrix{T};
    regularization::Real,
    iterations::Integer,
    relative_tolerance::Real,
    minimum_diagonal::Real,
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(interaction_matrix, regularization_operator)
    isempty(interaction_matrix) &&
        throw(ArgumentError("SCC interaction matrix must be nonempty"))
    size(regularization_operator, 1) > 0 ||
        throw(ArgumentError("SCC regularization operator must contain a row"))
    size(regularization_operator, 2) == size(interaction_matrix, 2) || throw(
        DimensionMismatch("SCC interaction and regularization controller extents differ"),
    )
    all(isfinite, interaction_matrix) ||
        throw(ArgumentError("SCC interaction matrix must be finite"))
    all(isfinite, regularization_operator) ||
        throw(ArgumentError("SCC regularization operator must be finite"))
    lambda = T(regularization)
    isfinite(lambda) && lambda >= zero(T) ||
        throw(ArgumentError("SCC regularization must be finite and nonnegative"))
    iteration_count = Int(iterations)
    iteration_count > 0 ||
        throw(ArgumentError("SCC solver iterations must be positive"))
    tolerance = T(relative_tolerance)
    isfinite(tolerance) && tolerance > zero(T) ||
        throw(ArgumentError("SCC relative tolerance must be finite and positive"))
    diagonal = T(minimum_diagonal)
    isfinite(diagonal) && diagonal > zero(T) ||
        throw(ArgumentError("SCC minimum diagonal must be finite and positive"))
    return SccWeightedRegularizedReconstructionPlan(
        Matrix(interaction_matrix),
        Matrix(regularization_operator),
        lambda,
        iteration_count,
        tolerance,
        diagonal,
    )
end

"""Reusable vectors for one bounded weighted regularized SCC solve."""
struct SccWeightedRegularizedReconstructionWorkspace{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
}
    candidate::V
    right_hand_side::V
    residual::V
    preconditioned_residual::V
    direction::V
    operator_direction::V
    diagonal::V
    measurement_product::V
    regularization_product::V
end

function SccWeightedRegularizedReconstructionWorkspace(
    plan::SccWeightedRegularizedReconstructionPlan{T},
) where {T}
    controller_count = size(plan.interaction_matrix, 2)
    measurement_count = size(plan.interaction_matrix, 1)
    regularization_count = size(plan.regularization_operator, 1)
    return SccWeightedRegularizedReconstructionWorkspace(
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, controller_count),
        zeros(T, measurement_count),
        zeros(T, regularization_count),
    )
end

commit!(workspace::SccWeightedRegularizedReconstructionWorkspace) = workspace
discard!(workspace::SccWeightedRegularizedReconstructionWorkspace) = workspace

function reset!(workspace::SccWeightedRegularizedReconstructionWorkspace)
    for values in (
        workspace.candidate,
        workspace.right_hand_side,
        workspace.residual,
        workspace.preconditioned_residual,
        workspace.direction,
        workspace.operator_direction,
        workspace.diagonal,
        workspace.measurement_product,
        workspace.regularization_product,
    )
        fill!(values, zero(eltype(values)))
    end
    return workspace
end

"""Expected disposition from dynamic weighted regularized SCC reconstruction."""
@enum SccWeightedRegularizedReconstructionFailure::UInt8 begin
    SccWeightedRegularizedReconstructionInvalidWeight = 1
    SccWeightedRegularizedReconstructionInvalidMeasurement = 2
    SccWeightedRegularizedReconstructionSingularSystem = 3
    SccWeightedRegularizedReconstructionNotConverged = 4
end

@doc "A supplied SCC reconstruction weight is invalid." SccWeightedRegularizedReconstructionInvalidWeight
@doc "A supplied SCC reconstruction measurement is invalid." SccWeightedRegularizedReconstructionInvalidMeasurement
@doc "The weighted regularized SCC solve broke down: a preconditioner diagonal was invalid or below its configured minimum, a conjugate-gradient inner product was nonpositive or non-finite, or a right-hand-side or residual norm was non-finite. This status does not by itself prove mathematical singularity." SccWeightedRegularizedReconstructionSingularSystem
@doc "The weighted regularized SCC iterative solve did not meet its convergence criterion." SccWeightedRegularizedReconstructionNotConverged

function _validate_scc_weighted_reconstruction_storage(
    controller_residual_error::AbstractVector{T},
    workspace::SccWeightedRegularizedReconstructionWorkspace{T},
    plan::SccWeightedRegularizedReconstructionPlan{T},
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(controller_residual_error, measurements, weights)
    interaction = plan.interaction_matrix
    length(controller_residual_error) == size(interaction, 2) || throw(
        DimensionMismatch("SCC controller-residual-error extent is invalid"),
    )
    length(measurements) == size(interaction, 1) && length(weights) == length(measurements) ||
        throw(DimensionMismatch("SCC measurement and weight extents are invalid"))
    Base.mightalias(controller_residual_error, measurements) && throw(
        ArgumentError("SCC weighted-reconstruction output must not alias measurements"),
    )
    Base.mightalias(controller_residual_error, weights) && throw(
        ArgumentError("SCC weighted-reconstruction output must not alias weights"),
    )
    return nothing
end

function _validate_scc_weighted_samples(
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
) where {T}
    @inbounds for measurement in eachindex(measurements, weights)
        weight = weights[measurement]
        isfinite(weight) && weight >= zero(T) ||
            return SccWeightedRegularizedReconstructionInvalidWeight
        iszero(weight) && continue
        isfinite(measurements[measurement]) ||
            return SccWeightedRegularizedReconstructionInvalidMeasurement
    end
    return nothing
end

function _scc_weighted_rhs_and_diagonal!(
    workspace::SccWeightedRegularizedReconstructionWorkspace{T},
    plan::SccWeightedRegularizedReconstructionPlan{T},
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
    controller_range=eachindex(workspace.right_hand_side),
) where {T}
    interaction = plan.interaction_matrix
    regularization_operator = plan.regularization_operator
    regularization = plan.regularization
    rhs = workspace.right_hand_side
    diagonal = workspace.diagonal
    @inbounds for controller in controller_range
        rhs_value = zero(T)
        diagonal_value = zero(T)
        for measurement in eachindex(measurements, weights)
            weight = weights[measurement]
            iszero(weight) && continue
            coefficient = interaction[measurement, controller]
            rhs_value = muladd(
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
        if !iszero(regularization)
            for penalty in axes(regularization_operator, 1)
                coefficient = regularization_operator[penalty, controller]
                diagonal_value = muladd(
                    regularization * coefficient,
                    coefficient,
                    diagonal_value,
                )
            end
        end
        isfinite(diagonal_value) && diagonal_value >= plan.minimum_diagonal ||
            return SccWeightedRegularizedReconstructionSingularSystem
        rhs[controller] = rhs_value
        diagonal[controller] = diagonal_value
    end
    return nothing
end

function _scc_weighted_operator!(
    output::AbstractVector{T},
    workspace::SccWeightedRegularizedReconstructionWorkspace{T},
    plan::SccWeightedRegularizedReconstructionPlan{T},
    weights::AbstractVector{T},
    input::AbstractVector{T},
    measurement_range=eachindex(workspace.measurement_product),
    regularization_range=eachindex(workspace.regularization_product),
    controller_range=eachindex(output),
) where {T}
    interaction = plan.interaction_matrix
    regularization_operator = plan.regularization_operator
    regularization = plan.regularization
    measurement_product = workspace.measurement_product
    regularization_product = workspace.regularization_product
    @inbounds for measurement in measurement_range
        weight = weights[measurement]
        # The unweighted row product can overflow even when its weight is zero.
        if iszero(weight)
            measurement_product[measurement] = zero(T)
            continue
        end
        value = zero(T)
        for controller in eachindex(input)
            value = muladd(interaction[measurement, controller], input[controller], value)
        end
        measurement_product[measurement] = weight * value
    end
    if !iszero(regularization)
        @inbounds for penalty in regularization_range
            value = zero(T)
            for controller in eachindex(input)
                value = muladd(
                    regularization_operator[penalty, controller],
                    input[controller],
                    value,
                )
            end
            regularization_product[penalty] = value
        end
    end
    @inbounds for controller in controller_range
        value = zero(T)
        for measurement in eachindex(measurement_product)
            value = muladd(
                interaction[measurement, controller],
                measurement_product[measurement],
                value,
            )
        end
        if !iszero(regularization)
            regularized = zero(T)
            for penalty in eachindex(regularization_product)
                regularized = muladd(
                    regularization_operator[penalty, controller],
                    regularization_product[penalty],
                    regularized,
                )
            end
            value = muladd(regularization, regularized, value)
        end
        output[controller] = value
    end
    return nothing
end

function _scc_dot(left::AbstractVector{T}, right::AbstractVector{T}) where {T}
    value = zero(T)
    @inbounds for index in eachindex(left, right)
        value = muladd(left[index], right[index], value)
    end
    return value
end

function _scc_norm(values::AbstractVector{T}) where {T}
    scale = zero(T)
    sum_squares = one(T)
    for value in values
        magnitude = abs(value)
        isfinite(magnitude) || return magnitude
        iszero(magnitude) && continue
        if scale < magnitude
            ratio = scale / magnitude
            sum_squares = one(T) + sum_squares * ratio * ratio
            scale = magnitude
        else
            ratio = magnitude / scale
            sum_squares += ratio * ratio
        end
    end
    return scale * sqrt(sum_squares)
end

function _rescale_scc_candidate!(candidate::AbstractVector{T}, scale::T) where {T}
    for index in eachindex(candidate)
        value = candidate[index] * scale
        isfinite(value) || return SccWeightedRegularizedReconstructionSingularSystem
        candidate[index] = value
    end
    return nothing
end

function _solve_scc_weighted_regularized!(
    workspace::SccWeightedRegularizedReconstructionWorkspace{T},
    plan::SccWeightedRegularizedReconstructionPlan{T},
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
) where {T}
    failure = _validate_scc_weighted_samples(measurements, weights)
    isnothing(failure) || return failure
    failure = _scc_weighted_rhs_and_diagonal!(workspace, plan, measurements, weights)
    isnothing(failure) || return failure
    candidate = workspace.candidate
    rhs = workspace.right_hand_side
    residual = workspace.residual
    preconditioned = workspace.preconditioned_residual
    direction = workspace.direction
    operator_direction = workspace.operator_direction
    diagonal = workspace.diagonal
    fill!(candidate, zero(T))
    rhs_scale = zero(T)
    for value in rhs
        isfinite(value) || return SccWeightedRegularizedReconstructionSingularSystem
        rhs_scale = max(rhs_scale, abs(value))
    end
    iszero(rhs_scale) && return nothing
    @inbounds for controller in eachindex(rhs)
        residual[controller] = rhs[controller] / rhs_scale
        preconditioned[controller] = residual[controller] / diagonal[controller]
        direction[controller] = preconditioned[controller]
    end
    norm_rhs = _scc_norm(residual)
    rho = _scc_dot(residual, preconditioned)
    isfinite(rho) && rho > zero(T) ||
        return SccWeightedRegularizedReconstructionSingularSystem
    tolerance = plan.relative_tolerance * norm_rhs
    for _ in 1:plan.iterations
        _scc_weighted_operator!(
            operator_direction,
            workspace,
            plan,
            weights,
            direction,
        )
        denominator = _scc_dot(direction, operator_direction)
        isfinite(denominator) && denominator > zero(T) ||
            return SccWeightedRegularizedReconstructionSingularSystem
        alpha = rho / denominator
        @inbounds for controller in eachindex(candidate)
            candidate[controller] = muladd(alpha, direction[controller], candidate[controller])
            residual[controller] = muladd(
                -alpha,
                operator_direction[controller],
                residual[controller],
            )
        end
        residual_norm = _scc_norm(residual)
        isfinite(residual_norm) ||
            return SccWeightedRegularizedReconstructionSingularSystem
        if residual_norm <= tolerance
            failure = _rescale_scc_candidate!(candidate, rhs_scale)
            isnothing(failure) || return failure
            # Verify the returned coordinates, including rounding on rescaling.
            _scc_weighted_operator!(operator_direction, workspace, plan, weights, candidate)
            @inbounds for controller in eachindex(rhs)
                residual[controller] = (rhs[controller] - operator_direction[controller]) / rhs_scale
            end
            verified_norm = _scc_norm(residual)
            isfinite(verified_norm) ||
                return SccWeightedRegularizedReconstructionSingularSystem
            return verified_norm <= tolerance ? nothing :
                SccWeightedRegularizedReconstructionNotConverged
        end
        @inbounds for controller in eachindex(preconditioned)
            preconditioned[controller] = residual[controller] / diagonal[controller]
        end
        next_rho = _scc_dot(residual, preconditioned)
        isfinite(next_rho) && next_rho > zero(T) ||
            return SccWeightedRegularizedReconstructionSingularSystem
        beta = next_rho / rho
        @inbounds for controller in eachindex(direction)
            direction[controller] = muladd(beta, direction[controller], preconditioned[controller])
        end
        rho = next_rho
    end
    return SccWeightedRegularizedReconstructionNotConverged
end

function process!(
    controller_residual_error::AbstractVector{T},
    workspace::SccWeightedRegularizedReconstructionWorkspace{T},
    plan::SccWeightedRegularizedReconstructionPlan{T},
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    _validate_scc_weighted_reconstruction_storage(
        controller_residual_error,
        workspace,
        plan,
        measurements,
        weights,
    )
    failure = _solve_scc_weighted_regularized!(workspace, plan, measurements, weights)
    isnothing(failure) || return failure
    copyto!(controller_residual_error, workspace.candidate)
    return nothing
end

"""
Dynamic weighted regularized SCC reconstruction (the method called 9.5 in the
exploration note).
"""
@algorithm SccWeightedRegularizedReconstructorF32 begin
    label = "scc-weighted-regularized-reconstructor-f32"
    failure = SccWeightedRegularizedReconstructionFailure
    operation = SccWeightedRegularizedReconstructionF32Operation()

    """
    Configuration for [`SccWeightedRegularizedReconstructorF32`](@ref).
    """
    @configuration SccWeightedRegularizedReconstructorF32Configuration begin
        measurement_count::Int => graph_rebuild
        controller_coordinate_count::Int => graph_rebuild
        regularization_row_count::Int => graph_rebuild
        measurement_schema::String => graph_rebuild
        initial_interaction_matrix::Union{Nothing,Vector{Float32}} => construction
        initial_regularization_operator::Union{Nothing,Vector{Float32}} => construction
        regularization::Float32 => construction
        iterations::Int => construction
        relative_tolerance::Float32 => construction
        minimum_diagonal::Float32 => construction
    end

    ports(configuration) = (
        input(
            :measurements,
            Float32,
            (configuration.measurement_count,),
            configuration.measurement_schema,
        ),
        input(
            :weights,
            schema_element_type(SCC_MEASUREMENT_WEIGHTS_V1),
            (configuration.measurement_count,),
            SCC_MEASUREMENT_WEIGHTS_V1,
        ),
        output(
            :controller_residual_error,
            schema_element_type(CONTROLLER_RESIDUAL_ERROR_V1),
            (configuration.controller_coordinate_count,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            name="controller-residual-error",
            metadata_from=:measurements,
        ),
        parameter(
            :interaction_matrix,
            schema_element_type(SCC_INTERACTION_MATRIX_V1),
            (
                configuration.measurement_count,
                configuration.controller_coordinate_count,
            ),
            SCC_INTERACTION_MATRIX_V1;
            name="interaction-matrix",
            replace=(plan, values) -> _replace_scc_weighted_regularized_plan(
                plan;
                interaction_matrix=values,
            ),
        ),
        parameter(
            :regularization_operator,
            schema_element_type(SCC_REGULARIZATION_OPERATOR_V1),
            (
                configuration.regularization_row_count,
                configuration.controller_coordinate_count,
            ),
            SCC_REGULARIZATION_OPERATOR_V1;
            name="regularization-operator",
            replace=(plan, values) -> _replace_scc_weighted_regularized_plan(
                plan;
                regularization_operator=values,
            ),
        ),
    )

    prepare(configuration) = begin
        interaction = zeros(
            Float32,
            configuration.measurement_count,
            configuration.controller_coordinate_count,
        )
        if !isnothing(configuration.initial_interaction_matrix)
            length(configuration.initial_interaction_matrix) == length(interaction) ||
                throw(DimensionMismatch(
                    "initial SCC interaction-matrix extent is invalid",
                ))
            for measurement in axes(interaction, 1), controller in axes(interaction, 2)
                source = (measurement - 1) * size(interaction, 2) + controller
                interaction[measurement, controller] =
                    configuration.initial_interaction_matrix[source]
            end
        end
        regularization_operator = zeros(
            Float32,
            configuration.regularization_row_count,
            configuration.controller_coordinate_count,
        )
        if !isnothing(configuration.initial_regularization_operator)
            length(configuration.initial_regularization_operator) ==
            length(regularization_operator) || throw(DimensionMismatch(
                "initial SCC regularization-operator extent is invalid",
            ))
            for penalty in axes(regularization_operator, 1),
                controller in axes(regularization_operator, 2)
                source = (penalty - 1) * size(regularization_operator, 2) + controller
                regularization_operator[penalty, controller] =
                    configuration.initial_regularization_operator[source]
            end
        end
        plan = SccWeightedRegularizedReconstructionPlan(
            interaction,
            regularization_operator;
            regularization=configuration.regularization,
            iterations=configuration.iterations,
            relative_tolerance=configuration.relative_tolerance,
            minimum_diagonal=configuration.minimum_diagonal,
        )
        (plan=plan, workspace=SccWeightedRegularizedReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(controller_residual_error, workspace, plan, measurements, weights)
    end
end
