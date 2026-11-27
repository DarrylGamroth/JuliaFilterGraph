import FilterGraphAlgorithms: PdmCommandPowerLimitF32, PdmCommandPowerLimitNonFiniteInput,
    _pdm_command_power_gain, _pdm_projected_value

@kernel function _pdm_power_gain!(gain, status, requested, max_power, failure)
    if @inbounds iszero(status[1])
        value = _pdm_command_power_gain(requested, max_power, Val(false))
        if isfinite(value)
            @inbounds gain[1] = value
        else
            @inbounds status[1] = failure
        end
    end
end

@kernel function _apply_pdm_limit!(output, status, requested, gain, project)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            value = requested[i]
            factor = gain[1]
            output[i] = factor == 1.0 ? value : project(value, factor)
        end
    end
end

@inline _pdm_power_product(value, gain) = _pdm_projected_value(value, gain, Val(false))

capture_support(::Type{PdmCommandPowerLimitF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{PdmCommandPowerLimitF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:exact)
    output, requested = only(outputs), only(inputs)
    gain = _capture_zeros(device, Float64, 1)
    failures = _control_failures(node_index, (PdmCommandPowerLimitNonFiniteInput,))
    kernels = (
        _control_kernel(_pdm_power_gain!,
            (gain, status, requested, prepared.plan.max_command_power, first(only(failures))), 1; lanes=1),
        _control_kernel(_apply_pdm_limit!,
            (output, status, requested, gain, _pdm_power_product), length(output)),
    )
    return ControlCapture(kernels, failures)
end
