const StructuredCalibration = Union{ZonalInteractionMatrixF32,HadamardInteractionMatrixF32}

@inline function _calibration_coefficient(responses, measurement, corrector, ::Nothing)
    row = 2 * corrector - 1
    return @inbounds responses[row, measurement] - responses[row + 1, measurement]
end

@inline function _calibration_coefficient(responses, measurement, corrector, order)
    accumulation = zero(eltype(responses))
    for pattern in 0:(order - 1)
        row = 2 * pattern + 1
        @inbounds difference = responses[row, measurement] - responses[row + 1, measurement]
        accumulation += iseven(count_ones(pattern & corrector)) ? difference : -difference
    end
    return accumulation
end

@kernel function _structured_calibration_matrix!(staged, status, responses, scales, order)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        measurements = size(staged, 1)
        measurement = (index - 1) % measurements + 1
        corrector = (index - 1) ÷ measurements + 1
        value = _calibration_coefficient(responses, measurement, corrector, order)
        @inbounds staged[index] = value * scales[corrector]
    end
end

capture_support(::Type{<:StructuredCalibration}, ::DeviceType) = SupportsCapture()

_calibration_order(::ZonalInteractionMatrixPlan) = nothing
_calibration_order(plan::HadamardInteractionMatrixPlan) = plan.order

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{<:StructuredCalibration},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    output, responses = only(outputs), only(inputs)
    staged = similar(output)
    scales = _capture_array(device, prepared.plan.calibration_plan.reciprocal_scales)
    order = _calibration_order(prepared.plan)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    estimate! = _structured_calibration_matrix!(backend, PROJECTION_WORKGROUP_SIZE)
    failures = _calibration_failures(node, StructuredNonFiniteResponse, StructuredNonFiniteCalculation)
    enqueue = function ()
        _calibration_validate!(validate, status, responses, first(failures[1]))
        estimate!(staged, status, responses, scales, order; ndrange=length(output))
        _calibration_validate!(validate, status, staged, first(failures[2]))
        _calibration_publish!(publish, status, output, staged)
    end
    return CalibrationTransition(enqueue, () -> nothing, failures)
end
