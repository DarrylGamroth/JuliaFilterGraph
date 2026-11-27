@kernel function _calibration_stability_columns!(squares, status, first_window, second_window)
    column = @index(Global, Linear)
    if @inbounds iszero(status[1])
        T = eltype(squares)
        first_squares = zero(T)
        second_squares = zero(T)
        difference_squares = zero(T)
        @inbounds for row in axes(first_window, 1)
            first_value = first_window[row, column]
            second_value = second_window[row, column]
            first_squares += abs2(first_value)
            second_squares += abs2(second_value)
            difference_squares += abs2(first_value - second_value)
        end
        @inbounds begin
            squares[1, column] = first_squares
            squares[2, column] = second_squares
            squares[3, column] = difference_squares
        end
    end
end

@inline function _calibration_stability_metrics(squares, criteria)
    T = eltype(squares)
    minimum_first = T(Inf)
    minimum_second = T(Inf)
    maximum_relative = zero(T)
    maximum_coordinate = 1
    insufficient_coordinate = 0
    total_first = zero(T)
    total_second = zero(T)
    total_difference = zero(T)
    @inbounds for coordinate in axes(squares, 2)
        first_squares = squares[1, coordinate]
        second_squares = squares[2, coordinate]
        difference_squares = squares[3, coordinate]
        total_first += first_squares
        total_second += second_squares
        total_difference += difference_squares
        first_norm = sqrt(first_squares)
        second_norm = sqrt(second_squares)
        minimum_first = min(minimum_first, first_norm)
        minimum_second = min(minimum_second, second_norm)
        if first_norm < criteria.minimum_column_norm || second_norm < criteria.minimum_column_norm
            iszero(insufficient_coordinate) && (insufficient_coordinate = coordinate)
        else
            relative = T(2) * sqrt(difference_squares) / (first_norm + second_norm)
            isfinite(relative) || return nothing
            if relative > maximum_relative
                maximum_relative = relative
                maximum_coordinate = coordinate
            end
        end
    end
    all(isfinite, (total_first, total_second, total_difference)) || return nothing
    total_norm = sqrt(total_first) + sqrt(total_second)
    frobenius = iszero(total_norm) ? zero(T) : T(2) * sqrt(total_difference) / total_norm
    isfinite(frobenius) || return nothing

    result, coordinate = if !iszero(insufficient_coordinate)
        (InteractionMatrixStabilityInsufficientColumnNorm, insufficient_coordinate)
    elseif maximum_relative > criteria.maximum_relative_column_difference
        (InteractionMatrixStabilityExcessiveRelativeColumnDifference, maximum_coordinate)
    elseif frobenius > criteria.maximum_relative_frobenius_difference
        (InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference, 0)
    else
        (InteractionMatrixStabilityValid, 0)
    end
    return (
        UInt8(result), UInt64(max(coordinate - 1, 0)), minimum_first, minimum_second,
        maximum_relative, UInt64(maximum_coordinate - 1), frobenius,
    )
end

@kernel function _publish_calibration_stability!(
    result, coordinate, norms, maximum_column, column_coordinate,
    frobenius, status, squares, criteria, failure,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        metrics = _calibration_stability_metrics(squares, criteria)
        @inbounds if metrics === nothing
            status[1] = failure
        else
            result[1], coordinate[1], norms[1], norms[2], maximum_column[1],
            column_coordinate[1], frobenius[1] = metrics
        end
    end
end

capture_support(::Type{InteractionMatrixStabilityF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{InteractionMatrixStabilityF32},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    plan = prepared.plan
    squares = similar(first(inputs), (3, plan.corrector_coordinate_count))
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    columns! = _calibration_stability_columns!(backend, PROJECTION_WORKGROUP_SIZE)
    publish! = _publish_calibration_stability!(backend, 1)
    failures = _calibration_failures(node, InteractionMatrixStabilityNonFiniteFirstWindow,
        InteractionMatrixStabilityNonFiniteSecondWindow, InteractionMatrixStabilityNonFiniteCalculation)
    enqueue = function ()
        for (value, (code, _)) in zip(inputs, failures)
            _calibration_validate!(validate, status, value, code)
        end
        columns!(squares, status, inputs...; ndrange=plan.corrector_coordinate_count)
        publish!(outputs..., status, squares, plan.criteria, first(failures[3]); ndrange=1)
        return nothing
    end
    return CalibrationTransition(enqueue, () -> nothing, failures)
end
