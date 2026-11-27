import FilterGraphAlgorithms: OpticalGainCorrectionF32, OpticalGainCorrectionNonFiniteInput,
    OpticalGainCorrectionNonFiniteOutput

@kernel function _check_optical_gain_correction!(status, input, reciprocals, input_failure, output_failure)
    if @inbounds iszero(status[1])
        # CPU reports the first invalid mode, then input before product
        # failure within that mode.
        @inbounds for i in eachindex(input)
            if !isfinite(input[i])
                status[1] = input_failure
                break
            elseif !isfinite(input[i] * reciprocals[i])
                status[1] = output_failure
                break
            end
        end
    end
end

capture_support(::Type{OpticalGainCorrectionF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{OpticalGainCorrectionF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status)
    output, input = only(outputs), only(inputs)
    reciprocals = _capture_array(device, prepared.plan.reciprocal_corrections)
    failures = _control_failures(node_index, (OpticalGainCorrectionNonFiniteInput,
        OpticalGainCorrectionNonFiniteOutput))
    kernels = (
        _control_kernel(_check_optical_gain_correction!,
            (status, input, reciprocals, map(first, failures)...), 1; lanes=1),
        _control_combine(output, status, input, reciprocals, *),
    )
    return ControlCapture(kernels, failures)
end
