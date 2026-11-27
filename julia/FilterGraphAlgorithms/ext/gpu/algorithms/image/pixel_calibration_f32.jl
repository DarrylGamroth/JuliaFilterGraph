# Record the first failing pixel in the CPU's linear traversal order. Only
# exceptional pixels contend on the error word. The output remains untouched
# unless every lane succeeded.
@kernel function _validate_calibrated_pixels!(candidate, raw, flat, background, first_error, status)
    i = @index(Global, Linear)
    if iszero(@inbounds status[1])
        @inbounds begin
            value = raw[i]
            calibrated = flat[i] * (value - background[i])
            candidate[i] = calibrated
            if !isfinite(value) || !isfinite(calibrated)
                code = (UInt32(i) << 1) | UInt32(isfinite(value))
                KernelAbstractions.@atomic min(first_error[1], code)
            end
        end
    end
end

@kernel function _publish_calibrated_pixels!(out, candidate, first_error)
    i = @index(Global, Linear)
    @inbounds if first_error[1] == typemax(UInt32)
        out[i] = candidate[i]
    end
end

@kernel function _begin_pixel_calibration!(first_error, status)
    i = @index(Global, Linear)
    # A pre-existing graph failure must also suppress publication.
    i == 1 && (@inbounds first_error[1] = iszero(status[1]) ? typemax(UInt32) : UInt32(0))
end

@kernel function _finish_pixel_calibration!(status, first_error, input_failure, output_failure)
    i = @index(Global, Linear)
    if i == 1
        @inbounds begin
            code = first_error[1]
            if iszero(status[1]) && code != typemax(UInt32)
                status[1] = isodd(code) ? output_failure : input_failure
            end
        end
    end
end

struct SignedPixelCalibration{K}
    kernels::K
    input_failure::Int32
    output_failure::Int32
end

capture_support(::Type{PixelCalibrationF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType, p::PreparedAlgorithm{PixelCalibrationF32},
        outputs::Tuple, inputs::Tuple, status, index::Int32)
    _check_capture_ports(device, p, outputs, inputs, status; input_overlap=:all)
    out, raw = only(outputs), only(inputs)
    size(p.plan.flat) == size(p.plan.background) == size(raw) ||
        throw(DimensionMismatch("pixel calibration planes must match the detector"))
    backend = KernelAbstractions.get_backend(out)
    candidate = similar(out)
    error = _capture_zeros(device, UInt32, 1)
    input_failure = _failure_code(index, FilterGraphAlgorithms.PixelCalibrationNonFiniteInput)
    output_failure = _failure_code(index, FilterGraphAlgorithms.PixelCalibrationNonFiniteOutput)
    kernels = (
        CapturedKernel(_begin_pixel_calibration!(backend, 1), (error, status), 1),
        CapturedKernel(_validate_calibrated_pixels!(backend, 128),
            (candidate, raw, _capture_array(device, p.plan.flat), _capture_array(device, p.plan.background),
                error, status), length(raw)),
        CapturedKernel(_finish_pixel_calibration!(backend, 1), (status, error, input_failure, output_failure), 1),
        CapturedKernel(_publish_calibrated_pixels!(backend, 128), (out, candidate, error), length(raw)),
    )
    return SignedPixelCalibration(kernels, input_failure, output_failure)
end

preflight(::SignedPixelCalibration) = nothing
enqueue!(adapter::SignedPixelCalibration) = foreach(enqueue!, adapter.kernels)
function complete!(adapter::SignedPixelCalibration, status::Int32)
    status == adapter.input_failure && return FilterGraphAlgorithms.PixelCalibrationNonFiniteInput
    status == adapter.output_failure && return FilterGraphAlgorithms.PixelCalibrationNonFiniteOutput
    return nothing
end
reset!(adapter::SignedPixelCalibration) = adapter
