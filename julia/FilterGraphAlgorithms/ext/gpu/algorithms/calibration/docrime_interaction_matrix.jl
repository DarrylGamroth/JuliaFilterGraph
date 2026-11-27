const DocrimeInteractionMatrix = Union{
    DocrimeOpenLoopInteractionMatrixF32,DocrimeClosedLoopInteractionMatrixF32,
}

@kernel function _docrime_interaction_check_range!(status, count, failure)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]) && count[1] > UInt64(typemax(Int)))
        @inbounds status[1] = failure
    end
end

@kernel function _docrime_interaction_normalize!(covariance, status, input, count)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds n = count[1]
        value = n < 2 ? 0.0f0 : @inbounds(input[index] * inv(Float32(n - 1)))
        @inbounds covariance[index] = value
    end
end

@inline function _docrime_interaction_conditioning!(factor, normalized, original, count, criteria, failures)
    T = eltype(normalized)
    zero_value = zero(T)
    rows = size(normalized, 1)
    # Equality must be tested on the original sums, before normalization rounds them.
    @inbounds for column in 1:rows, row in 1:(column - 1)
        original[row, column] == original[column, row] || return failures[1]
    end
    if count < 2
        return DocrimeInteractionMatrixDiagnostics(
            DocrimeInteractionMatrixInsufficientSamples, Int(count),
            zero_value, zero_value, zero_value, zero_value, 0, 0)
    end
    _docrime_count_representable(count - one(count)) || return failures[2]

    minimum_variance = @inbounds normalized[1, 1]
    maximum_variance = minimum_variance
    insufficient = 0
    for value in normalized
        isfinite(value) || return failures[3]
    end
    @inbounds for row in 1:rows
        variance = normalized[row, row]
        minimum_variance = min(minimum_variance, variance)
        maximum_variance = max(maximum_variance, variance)
        if iszero(insufficient) && variance < criteria.minimum_corrector_variance
            insufficient = row
        end
    end
    if !iszero(insufficient)
        return DocrimeInteractionMatrixDiagnostics(
            DocrimeInteractionMatrixInsufficientCorrectorVariance, Int(count),
            minimum_variance, maximum_variance, zero_value, zero_value, insufficient, 0)
    end

    maximum_correlation = zero_value
    maximum_row = maximum_column = 0
    @inbounds for row in 1:rows
        row_deviation = sqrt(normalized[row, row])
        for column in (row + 1):rows
            correlation = (abs(normalized[row, column]) / row_deviation) /
                          sqrt(normalized[column, column])
            isfinite(correlation) || return failures[4]
            if correlation > maximum_correlation
                maximum_correlation = correlation
                maximum_row, maximum_column = row, column
            end
        end
    end
    if maximum_correlation > criteria.maximum_absolute_corrector_correlation
        return DocrimeInteractionMatrixDiagnostics(
            DocrimeInteractionMatrixExcessiveCorrectorCorrelation, Int(count),
            minimum_variance, maximum_variance, maximum_correlation, zero_value,
            maximum_row, maximum_column)
    end

    # Row-major Cholesky preserves the CPU's rejection order and pivot diagnostics.
    # Every lower-triangle value is defined before it is read.
    minimum_pivot = one(T)
    @inbounds for row in 1:rows, column in 1:row
        factor_sum = zero_value
        for index in 1:(column - 1)
            factor_sum += factor[row, index] * factor[column, index]
        end
        coefficient = normalized[row, column] - factor_sum
        isfinite(coefficient) || return failures[5]
        if row == column
            if coefficient <= zero_value
                return DocrimeInteractionMatrixDiagnostics(
                    DocrimeInteractionMatrixNonPositiveDefinite, Int(count),
                    minimum_variance, maximum_variance, maximum_correlation, zero_value, row, 0)
            end
            pivot_ratio = coefficient / maximum_variance
            isfinite(pivot_ratio) || return failures[5]
            minimum_pivot = min(minimum_pivot, pivot_ratio)
            if pivot_ratio < criteria.minimum_factorization_pivot_ratio
                return DocrimeInteractionMatrixDiagnostics(
                    DocrimeInteractionMatrixInsufficientFactorizationPivot, Int(count),
                    minimum_variance, maximum_variance, maximum_correlation, minimum_pivot, row, 0)
            end
            value = sqrt(coefficient)
        else
            value = coefficient / factor[column, column]
        end
        isfinite(value) || return failures[5]
        factor[row, column] = value
    end
    return DocrimeInteractionMatrixDiagnostics(
        DocrimeInteractionMatrixValid, Int(count), minimum_variance, maximum_variance,
        maximum_correlation, minimum_pivot, 0, 0)
end

@kernel function _docrime_interaction_condition!(
    diagnostics, status, factor, covariance, corrector, count, criteria, failures,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds result = _docrime_interaction_conditioning!(
            factor, covariance, corrector, count[1], criteria, failures)
        @inbounds if result isa Int32
            status[1] = result
        else
            diagnostics[1] = result
        end
    end
end

@kernel function _docrime_interaction_solve!(staged, status, diagnostics, factor, cross, response_scale)
    measurement = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds report = diagnostics[1]
        columns = size(staged, 2)
        if report.status === DocrimeInteractionMatrixValid
            normalization = inv(Float32(report.sample_count - 1))
            @inbounds for row in 1:columns
                value = cross[measurement, row] * normalization * response_scale
                for column in 1:(row - 1)
                    value -= factor[row, column] * staged[measurement, column]
                end
                staged[measurement, row] = value / factor[row, row]
            end
            @inbounds for row in columns:-1:1
                value = staged[measurement, row]
                for column in (row + 1):columns
                    value -= factor[column, row] * staged[measurement, column]
                end
                staged[measurement, row] = value / factor[row, row]
            end
        else
            @inbounds for column in 1:columns
                staged[measurement, column] = 0.0f0
            end
        end
    end
end

@kernel function _publish_docrime_interaction_diagnostics!(
    result, coordinates, sample_count, variance, correlation, pivot, status, diagnostics,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds begin
            report = diagnostics[1]
            result[1] = UInt8(report.status)
            coordinates[1], coordinates[2] = FilterGraphAlgorithms._docrime_status_coordinates(report)
            sample_count[1] = UInt64(report.sample_count)
            variance[1] = report.minimum_corrector_variance
            variance[2] = report.maximum_corrector_variance
            correlation[1] = report.maximum_absolute_corrector_correlation
            pivot[1] = report.minimum_factorization_pivot_ratio
        end
    end
end

capture_support(::Type{<:DocrimeInteractionMatrix}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{<:DocrimeInteractionMatrix},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    cross, corrector, count = inputs
    matrix = first(outputs)
    staged = similar(matrix)
    covariance = similar(corrector)
    factor = similar(corrector)
    diagnostics = similar(status, DocrimeInteractionMatrixDiagnostics{Float32}, 1)
    plan = prepared.plan.calibration_plan
    failures = _calibration_failures(node, DocrimeInteractionMatrixNonFiniteCrossCovariance,
        DocrimeInteractionMatrixNonFiniteCorrectorCovariance,
        DocrimeInteractionMatrixNonSymmetricCorrectorCovariance,
        DocrimeInteractionMatrixSampleCountNotRepresentable,
        DocrimeInteractionMatrixSampleCountOutOfRange,
        DocrimeInteractionMatrixNonFiniteNormalization, DocrimeInteractionMatrixNonFiniteCorrelation,
        DocrimeInteractionMatrixNonFiniteFactorization, DocrimeInteractionMatrixNonFiniteSolve)
    cross_failure, corrector_failure, symmetry_failure, inexact_failure, range_failure,
        normalization_failure, correlation_failure, factor_failure, solve_failure = map(first, failures)
    conditioning_codes = (symmetry_failure, inexact_failure, normalization_failure,
        correlation_failure, factor_failure)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    check_range! = _docrime_interaction_check_range!(backend, 1)
    normalize! = _docrime_interaction_normalize!(backend, PROJECTION_WORKGROUP_SIZE)
    condition! = _docrime_interaction_condition!(backend, 1)
    solve! = _docrime_interaction_solve!(backend, PROJECTION_WORKGROUP_SIZE)
    diagnostics! = _publish_docrime_interaction_diagnostics!(backend, 1)
    enqueue = function ()
        check_range!(status, count, range_failure; ndrange=1)
        _calibration_validate!(validate, status, cross, cross_failure)
        _calibration_validate!(validate, status, corrector, corrector_failure)
        normalize!(covariance, status, corrector, count; ndrange=length(corrector))
        condition!(diagnostics, status, factor, covariance, corrector, count,
            plan.criteria, conditioning_codes; ndrange=1)
        solve!(staged, status, diagnostics, factor, cross, plan.response_scale; ndrange=size(staged, 1))
        _calibration_validate!(validate, status, staged, solve_failure)
        _calibration_publish!(publish, status, matrix, staged)
        diagnostics!(Base.tail(outputs)..., status, diagnostics; ndrange=1)
        return nothing
    end
    return CalibrationTransition(enqueue, () -> nothing, failures)
end
