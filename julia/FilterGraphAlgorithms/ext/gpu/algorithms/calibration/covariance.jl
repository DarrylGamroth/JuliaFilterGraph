# The covariance family always uses Float32. The count can have at most 24
# significant bits after trailing zeros are removed, including above 2^24.
@inline _docrime_count_representable(count::UInt64) =
    leading_zeros(count) + trailing_zeros(count) >= 40

@kernel function _docrime_covariance_check_limit!(status, count, limit, failure)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]) && count[1] >= limit)
        @inbounds status[1] = failure
    end
end

@kernel function _docrime_covariance_check_count!(status, count, overflow, inexact)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds current = count[1]
        @inbounds if current == typemax(UInt64)
            status[1] = overflow
        elseif !_docrime_count_representable(current + one(current))
            status[1] = inexact
        end
    end
end

@kernel function _docrime_covariance_mean!(candidate, delta, status, committed, sample, count)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            n = Float32(count[1] + one(UInt64))
            difference = sample[index] - committed[index]
            delta[index] = difference
            candidate[index] = committed[index] + difference / n
        end
    end
end

@kernel function _docrime_covariance_sum!(staged, status, committed, left_delta, right_delta, count, symmetric)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        rows = size(staged, 1)
        row, column = (index - 1) % rows + 1, (index - 1) ÷ rows + 1
        if !symmetric || row <= column
            @inbounds begin
                n = Float32(count[1] + one(UInt64))
                weight = (n - 1.0f0) / n
                value = fma(weight * left_delta[row], right_delta[column], committed[row, column])
                staged[row, column] = value
                if symmetric && row != column
                    staged[column, row] = value
                end
            end
        end
    end
end

@kernel function _docrime_covariance_advance!(published_count, complete, status, count, limit)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds begin
            next_count = count[1] + one(UInt64)
            count[1] = published_count[1] = next_count
            complete[1] = limit !== nothing && next_count == limit
        end
    end
end

capture_support(::Type{DocrimeCovarianceF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{DocrimeCovarianceF32},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    workspace = prepared.workspace
    FilterGraphAlgorithms._docrime_covariance_workspace_extent_is_valid(workspace, prepared.plan) ||
        throw(DimensionMismatch("DO-CRIME covariance Workspace extent is invalid"))
    committed = map(array -> _capture_array(device, array), (
        workspace.committed_measurement_mean, workspace.committed_corrector_sample_mean,
        workspace.committed_cross_covariance_sum, workspace.committed_corrector_covariance_sum,
    ))
    candidate = map(similar, committed)
    deltas = map(similar, inputs)
    count = _capture_array(device, UInt64[workspace.committed_sample_count])
    limit = prepared.plan.sample_limit
    cross, corrector, output_count, complete_cycle = outputs
    failures = _calibration_failures(node, DocrimeCovarianceNonFiniteMeasurements,
        DocrimeCovarianceNonFiniteCorrectorSample, DocrimeCovarianceNonFiniteCalculation,
        DocrimeCovarianceSampleCountOverflow, DocrimeCovarianceSampleCountNotRepresentable,
        DocrimeCovarianceSampleLimitReached)
    _, _, calculation_failure, overflow_failure, inexact_failure, limit_failure = map(first, failures)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    check_limit! = _docrime_covariance_check_limit!(backend, 1)
    check_count! = _docrime_covariance_check_count!(backend, 1)
    mean! = _docrime_covariance_mean!(backend, PROJECTION_WORKGROUP_SIZE)
    sum! = _docrime_covariance_sum!(backend, PROJECTION_WORKGROUP_SIZE)
    advance! = _docrime_covariance_advance!(backend, 1)
    enqueue = function ()
        if limit !== nothing
            check_limit!(status, count, limit, limit_failure; ndrange=1)
        end
        for (sample, (code, _)) in zip(inputs, failures)
            _calibration_validate!(validate, status, sample, code)
        end
        check_count!(status, count, overflow_failure, inexact_failure; ndrange=1)
        for index in 1:2
            mean!(candidate[index], deltas[index], status, committed[index], inputs[index],
                count; ndrange=length(inputs[index]))
            _calibration_validate!(validate, status, deltas[index], calculation_failure)
            _calibration_validate!(validate, status, candidate[index], calculation_failure)
        end
        sum!(candidate[3], status, committed[3], deltas[1], deltas[2], count, false; ndrange=length(cross))
        sum!(candidate[4], status, committed[4], deltas[2], deltas[2], count, true; ndrange=length(corrector))
        for staging in (candidate[3], candidate[4])
            _calibration_validate!(validate, status, staging, calculation_failure)
        end
        _calibration_publish!(publish, status, cross, candidate[3])
        _calibration_publish!(publish, status, corrector, candidate[4])
        for (value, staging) in zip(committed, candidate)
            _calibration_publish!(publish, status, value, staging)
        end
        advance!(output_count, complete_cycle, status, count, limit; ndrange=1)
        return nothing
    end
    reset = function ()
        for values in (committed..., count)
            fill!(values, zero(eltype(values)))
        end
        return nothing
    end
    return CalibrationTransition(enqueue, reset, failures)
end
