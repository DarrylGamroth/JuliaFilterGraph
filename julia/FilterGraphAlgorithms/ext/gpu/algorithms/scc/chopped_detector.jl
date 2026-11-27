# The display consumes one complete detector ROI. Acquisition admission uses
# the host metadata; differences, candidate pixels, and retained pixels stay
# on the selected device.
mutable struct DeviceChoppedDetector{D,A}
    device::D
    state::ChoppedPairMetadata
    committed::A
    candidate::A
    difference::A
end

_chopped_detector_pair(plan) = plan.pair
_chopped_detector_pair(plan::ChoppedImageDifferencePlan) = plan

function prepare_algorithm(p::PreparedAlgorithm{ChoppedDetectorDifferenceF32}, device::DeviceType)
    return _with_eager_device(device) do
        shape = first(p.formats).shape
        history = _capture_zeros(device, Float32, shape...)
        ws = DeviceChoppedDetector(device, ChoppedPairMetadata(), history, similar(history), similar(history))
        plan = _chopped_detector_pair(p.plan)
        PreparedAlgorithm(ChoppedDetectorDifferenceF32, plan, ws, p.formats)
    end
end

function FilterGraphAlgorithms.commit!(ws::DeviceChoppedDetector)
    state = ws.state
    if state.candidate_ready
        ws.committed, ws.candidate = ws.candidate, ws.committed
        FilterGraphAlgorithms.commit!(state)
    end
    return ws
end
function FilterGraphAlgorithms.discard!(ws::DeviceChoppedDetector)
    ws.state.candidate_ready = false
    return ws
end
function reset!(ws::DeviceChoppedDetector)
    reset!(ws.state)
    return ws
end

@kernel function _chopped_detector!(difference, candidate, history, image, paired, fringed)
    i = @index(Global, Linear)
    @inbounds begin
        x = image[i]
        value = NaN32
        if paired && isfinite(x) && isfinite(history[i])
            delta = fringed ? x - history[i] : history[i] - x
            isfinite(delta) && (value = delta)
        end
        candidate[i] = x
        difference[i] = value
    end
end

function FilterGraphAlgorithms._process_admitted!(difference::AbstractArray, pair_status::AbstractArray,
        p::PreparedAlgorithm{ChoppedDetectorDifferenceF32,<:ChoppedImageDifferencePlan,<:DeviceChoppedDetector},
        image::AbstractArray, metadata::FilterGraphAlgorithms.AdmittedSampleMetadata)
    ws = p.workspace
    _require_current_device(ws.device)
    _require_capture_array("chopped detector image", image, ws.device)
    _require_capture_array("chopped detector difference", difference, ws.device)
    _require_capture_array("chopped detector status", pair_status, ws.device)
    _capture_arrays_overlap(difference, pair_status) &&
        throw(ArgumentError("chopped detector outputs must be disjoint"))
    for port in (image, difference, pair_status), owned in (ws.committed, ws.candidate, ws.difference)
        _capture_arrays_overlap(port, owned) &&
            throw(ArgumentError("chopped detector ports must be disjoint from retained storage"))
    end
    sample = FilterGraphAlgorithms._sample_metadata(metadata)
    flags = _admit_chopped_sample!(p.plan, ws.state, sample)
    flags isa ChoppedImageDifferenceFailure && return flags
    paired, fringed = flags
    kernel = _chopped_detector!(KernelAbstractions.get_backend(image),128)
    kernel(ws.difference, ws.candidate, ws.committed, image,
        !iszero(paired), !iszero(fringed); ndrange=length(image))
    copyto!(difference, ws.difference)
    fill!(pair_status, UInt8(iszero(paired) ? ChoppedImagePriming : ChoppedImageComplete))
    return AlgorithmProcessComplete(SampleMetadata(sample; trigger_high=nothing,
        chopped_image_kind=nothing, exposure_start=nothing, reference_timestamp=nothing, corrupted=false))
end
