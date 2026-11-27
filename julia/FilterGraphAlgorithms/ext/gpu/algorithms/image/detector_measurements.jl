# The capture owner binds the same on-device arrays throughout replay.
const DetectorMeasurement = Union{RegionExtractionF32, PyramidPupilFluxU16F64,
    WeightedCenterOfGravityF32, ShackHartmannSummaryF32}

capture_support(::Type{D}, ::DeviceType) where {D<:DetectorMeasurement} = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType, p::PreparedAlgorithm{D},
        outputs::Tuple, inputs::Tuple, status, ::Int32) where {D<:DetectorMeasurement}
    _check_capture_ports(device, p, outputs, inputs, status)
    backend = KernelAbstractions.get_backend(first(outputs))
    return _capture_detector_measurement(D, device, backend, p.plan, outputs, inputs, status)
end

@kernel function _extract_detector_regions!(out, image, origins, status)
    i = @index(Global, Linear)
    if iszero(@inbounds status[1])
        region, row, column = Tuple(CartesianIndices(out)[i])
        @inbounds begin
            r, c = origins[region]
            out[i] = image[r + row, c + column]
        end
    end
end

function _capture_detector_measurement(::Type{RegionExtractionF32}, device, backend,
        plan, outputs, inputs, status)
    out = only(outputs)
    length(plan.origins) == size(out, 1) && plan.region_size == Base.tail(size(out)) ||
        throw(DimensionMismatch("detector region geometry must match the output"))
    origins = _device_region_origins(device, plan, only(inputs))
    return CapturedKernel(_extract_detector_regions!(backend, 128),
        (out, only(inputs), origins, status), length(out))
end

function _device_region_origins(device, plan, image)
    rows, columns = size(image)
    height, width = plan.region_size
    rows >= height > 0 && columns >= width > 0 || throw(DimensionMismatch("invalid detector region size"))
    for (r, c) in plan.origins
        0 <= r <= rows - height && 0 <= c <= columns - width ||
            throw(ArgumentError("detector region extends outside input"))
    end
    return _capture_array(device, plan.origins)
end

@kernel function _pupil_flux!(out, raw, origins, region_size, status)
    pupil = @index(Global, Linear)
    if iszero(@inbounds status[1])
        @inbounds begin
            r, c = origins[pupil]
            total = 0.0
            for column in 1:region_size[2], row in 1:region_size[1]
                total += Float64(raw[r + row, c + column])
            end
            out[pupil] = total
        end
    end
end

function _capture_detector_measurement(::Type{PyramidPupilFluxU16F64}, device, backend,
        plan, outputs, inputs, status)
    regions = plan.regions
    length(regions.origins) == length(only(outputs)) || throw(DimensionMismatch("invalid pupil count"))
    origins = _device_region_origins(device, regions, only(inputs))
    return CapturedKernel(_pupil_flux!(backend, 4),
        (only(outputs), only(inputs), origins,
            regions.region_size, status), length(only(outputs)))
end

@kernel function _weighted_centroid!(centroid, flux, valid, pixels, x, y, weights, settings, status)
    i = @index(Global, Linear)
    if i == 1 && iszero(@inbounds status[1])
        FilterGraphAlgorithms._weighted_centroid!(centroid, flux, valid, pixels, x, y, weights,
            settings[1], settings[2], !iszero(settings[4]), settings[5], settings[3])
    end
end

function _capture_detector_measurement(::Type{WeightedCenterOfGravityF32}, device, backend,
        plan, outputs, inputs, status)
    length(plan.weights) == length(plan.x_coordinates) == length(plan.y_coordinates) == length(only(inputs)) ||
        throw(DimensionMismatch("centroid coordinates and weights must match the selected pixels"))
    settings = _capture_array(device, collect(_cog_settings(plan)))
    kernel = CapturedKernel(_weighted_centroid!(backend, 1), (outputs..., only(inputs),
        _capture_array(device, plan.x_coordinates), _capture_array(device, plan.y_coordinates),
        _capture_array(device, plan.weights), settings, status), 1)
    return CapturedCenterOfGravity((kernel,), settings)
end

@kernel function _summarize_subapertures!(summary, slopes, flux, validity, n, status)
    i = @index(Global, Linear)
    if i == 1 && iszero(@inbounds status[1])
        FilterGraphAlgorithms._shack_hartmann_summary!(summary, slopes, flux, validity, n)
    end
end

function _capture_detector_measurement(::Type{ShackHartmannSummaryF32}, device, backend,
        plan, outputs, inputs, status)
    plan.subaperture_count == length(inputs[2]) || throw(DimensionMismatch("invalid subaperture count"))
    return CapturedKernel(_summarize_subapertures!(backend, 1),
        (only(outputs), inputs..., plan.subaperture_count, status), 1)
end
