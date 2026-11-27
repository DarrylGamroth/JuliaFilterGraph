@kernel function _select_pixels!(output, image, indices, status)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds output[i] = image[indices[i]]
    end
end

@kernel function _pixel_validity!(output, pixels, status)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds output[i] = isfinite(pixels[i])
    end
end

function _prepare_pixel_capture(device,
        prepared::PreparedAlgorithm{PixelSelectionF32}, outputs::Tuple, inputs::Tuple, status)
    _check_capture_ports(device, prepared, outputs, inputs, status)
    _require_capture_workgroup(device, 256)
    plan = prepared.plan
    indices = (plan.selected_rows .+ (plan.selected_columns .- 1) .* plan.image_size[1])
    output, image = only(outputs), only(inputs)
    return CapturedKernel(_select_pixels!(KernelAbstractions.get_backend(output), 256),
        (output, image, _capture_array(device, indices), status), length(output))
end

function _prepare_pixel_capture(device,
        prepared::PreparedAlgorithm{PixelValidityF32}, outputs::Tuple, inputs::Tuple, status)
    _check_capture_ports(device, prepared, outputs, inputs, status)
    _require_capture_workgroup(device, 256)
    output, pixels = only(outputs), only(inputs)
    return CapturedKernel(_pixel_validity!(KernelAbstractions.get_backend(output), 256),
        (output, pixels, status), length(output))
end
