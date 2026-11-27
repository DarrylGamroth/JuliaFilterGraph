function _reconstructor_relative_matrix_error(
    reference::AbstractMatrix{T},
    comparison::AbstractMatrix{T},
) where {T<:AbstractFloat}
    scale = maximum(abs, reference; init=zero(T))
    iszero(scale) && return iszero(norm(comparison)) ? zero(T) : T(Inf)
    scaled_reference = reference ./ scale
    scaled_comparison = comparison ./ scale
    return T(norm(scaled_reference - scaled_comparison) / norm(scaled_reference))
end

function _reconstructor_relative_matrix_residual(
    reference::AbstractMatrix{T},
    approximation::AbstractMatrix{T},
) where {T<:AbstractFloat}
    reference_norm = norm(reference)
    iszero(reference_norm) && return zero(T)
    return T(norm(reference - approximation) / reference_norm)
end

function _reconstructor_tikhonov_gain(
    singular_value::T,
    tikhonov_scale::T,
    numerical_rank_threshold::T,
) where {T<:AbstractFloat}
    iszero(singular_value) && return zero(T)
    if iszero(tikhonov_scale)
        singular_value <= numerical_rank_threshold && return zero(T)
        return inv(singular_value)
    end

    if singular_value >= tikhonov_scale
        ratio = tikhonov_scale / singular_value
        return inv(singular_value) / (one(T) + ratio * ratio)
    end
    ratio = singular_value / tikhonov_scale
    return (ratio / tikhonov_scale) / (one(T) + ratio * ratio)
end

function _reconstructor_spd_factor(
    matrix::AbstractMatrix{T},
    validation_tolerance::T,
    matrix_name::AbstractString,
) where {T<:AbstractFloat}
    symmetry_error = _reconstructor_relative_matrix_error(
        matrix,
        transpose(matrix),
    )
    symmetry_error <= validation_tolerance ||
        throw(
            ArgumentError(
                "$matrix_name must be symmetric within validation_tolerance",
            ),
        )
    symmetric_matrix = (matrix + transpose(matrix)) / T(2)
    return cholesky(Symmetric(symmetric_matrix)), symmetry_error
end

function _reconstructor_weighting(
    noise_covariance::AbstractMatrix{T},
    stroke_metric::AbstractMatrix{T},
    validation_tolerance::T;
    noise_covariance_name::AbstractString,
    stroke_metric_name::AbstractString,
) where {T<:AbstractFloat}
    noise_factor, noise_symmetry_error = _reconstructor_spd_factor(
        noise_covariance,
        validation_tolerance,
        noise_covariance_name,
    )
    stroke_factor, stroke_symmetry_error = _reconstructor_spd_factor(
        stroke_metric,
        validation_tolerance,
        stroke_metric_name,
    )
    noise_extent = size(noise_covariance, 1)
    stroke_extent = size(stroke_metric, 1)
    noise_whitener = noise_factor.L \ Matrix{T}(I, noise_extent, noise_extent)
    stroke_inverse_sqrt =
        stroke_factor.U \ Matrix{T}(I, stroke_extent, stroke_extent)
    return (
        noise_factor=noise_factor,
        stroke_factor=stroke_factor,
        noise_whitener=noise_whitener,
        stroke_inverse_sqrt=stroke_inverse_sqrt,
        noise_symmetry_error=noise_symmetry_error,
        stroke_symmetry_error=stroke_symmetry_error,
    )
end

function _reconstructor_weighted_svd(
    whitened_operator::AbstractMatrix{T},
    tikhonov_scale::T,
    operator_name::AbstractString,
) where {T<:AbstractFloat}
    all(isfinite, whitened_operator) ||
        throw(ArgumentError("$operator_name contains non-finite values"))
    factors = svd(whitened_operator; full=false)
    singular_values = factors.S
    all(isfinite, singular_values) ||
        throw(
            ArgumentError(
                "$operator_name SVD produced non-finite singular values",
            ),
        )
    left_singular_vectors = factors.U
    right_singular_vectors = Matrix(transpose(factors.Vt))
    largest_singular_value = first(singular_values)
    numerical_rank_threshold =
        T(max(size(whitened_operator)...)) *
        eps(T) *
        largest_singular_value
    numerical_rank = count(
        singular_value -> singular_value > numerical_rank_threshold,
        singular_values,
    )
    tikhonov_gains = similar(singular_values)
    for index in eachindex(tikhonov_gains, singular_values)
        tikhonov_gains[index] = _reconstructor_tikhonov_gain(
            singular_values[index],
            tikhonov_scale,
            numerical_rank_threshold,
        )
    end
    return (
        left_singular_vectors=left_singular_vectors,
        singular_values=singular_values,
        right_singular_vectors=right_singular_vectors,
        largest_singular_value=largest_singular_value,
        numerical_rank=numerical_rank,
        tikhonov_gains=tikhonov_gains,
    )
end
