# A bounded Float32 reduction reads samples in detector-address order. The cold
# sorted map preserves overlapping-pupil multiplicity and allows coalesced loads;
# ordinary sums may differ in rounding from the general logical-order fallback.
const _PUPIL_MEAN_LIMIT = 4096

function _pupil_load_map(indices)
    0 < length(indices) <= _PUPIL_MEAN_LIMIT || return nothing
    lo, hi = extrema(indices)
    1 <= lo && hi <= typemax(Int32) || return nothing
    return sort!(Int32.(indices))
end

struct _PupilMean{K,M}
    kernel!::K
    mapping::M
end

function (mean::_PupilMean)(mean_output, candidate_gain, frame_status, image, indices,
        mean_scale, input_failure, mean_failure, gain_failure; ndrange)
    mean.kernel!(mean_output, candidate_gain, frame_status, image, indices, mean_scale,
        input_failure, mean_failure, gain_failure, mean.mapping; ndrange)
    return nothing
end

function _prepare_pupil_mean(device::CUDA.CuDevice, backend, indices)
    _require_capture_workgroup(device, PYRAMID_PUPIL_LANES; warp_width=32)
    mapping = _pupil_load_map(indices)
    if !isnothing(mapping)
        return _PupilMean(_pupil_mean!(backend, PYRAMID_PUPIL_LANES),
            _capture_array(device, mapping))
    end
    return _pyramid_pupil_mean!(backend, PYRAMID_PUPIL_LANES)
end

@inline function _pupil_stripes(sums, magnitudes, failures, lane)
    # Eight 32-lane stripes contribute a sum and an exceptional-frame flag to
    # each first-warp lane. Val fixes tuple lengths in this unrolled tree.
    partials = ntuple(Val(8)) do i
        index = lane + 32(i - 1)
        @inbounds (sum = sums[index], sumabs = magnitudes[index], failure = failures[index])
    end
    four = ntuple(i -> _pupil_pair(partials[i], partials[i + 4]), Val(4))
    two = ntuple(i -> _pupil_pair(four[i], four[i + 2]), Val(2))
    return _pupil_pair(two...)
end

@inline function _pupil_pair(a, b)
    sum = a.sum + b.sum
    failure = max(a.failure, b.failure)
    !isfinite(sum) && (failure = Int32(1))
    return (sum=sum, sumabs=a.sumabs + b.sumabs, failure=failure)
end

@kernel function _pupil_mean!(
        mean_output, candidate_gain, frame_status, image, indices, mean_scale,
        input_failure, mean_failure, gain_failure, mapping
)
    lane = Int(@index(Local, Linear))
    frame_valid = @inbounds iszero(frame_status[1])
    sum = 0.0f0
    sumabs = 0.0f0
    failure = Int32(0)
    entry = lane
    while frame_valid && entry <= length(mapping)
        value = @inbounds image[mapping[entry]]
        if !isfinite(value)
            failure = Int32(1)
        else
            sum += value
            sumabs += abs(value)
            !isfinite(sum) && (failure = Int32(1))
            # Conservatively detect possible serial-prefix overflow, even when
            # a parallel tree cancels; exceptional frames retain failure priority.
            abs(value) >= floatmax(Float32) * mean_scale * 0.5f0 &&
                (failure = Int32(1))
        end
        entry += PYRAMID_PUPIL_LANES
    end
    # One block-wide join, followed by register reductions in the first warp.
    sums = @localmem Float32 (PYRAMID_PUPIL_LANES,)
    magnitudes = @localmem Float32 (PYRAMID_PUPIL_LANES,)
    failures = @localmem Int32 (PYRAMID_PUPIL_LANES,)
    @inbounds sums[lane] = sum
    @inbounds magnitudes[lane] = sumabs
    @inbounds failures[lane] = failure
    @synchronize
    if lane <= 32
        sum, sumabs, failure = _pupil_stripes(sums, magnitudes, failures, lane)
        delta = 16
        while delta > 0
            sum += CUDA.shfl_down_sync(CUDA.FULL_MASK, sum, delta)
            sumabs += CUDA.shfl_down_sync(CUDA.FULL_MASK, sumabs, delta)
            failure = max(failure, CUDA.shfl_down_sync(CUDA.FULL_MASK, failure, delta))
            !isfinite(sum) && (failure = Int32(1))
            delta >>= 1
        end
        if frame_valid && lane == 1
            _pupil_mean_result!(mean_output, candidate_gain, frame_status, image,
                indices, mean_scale, sum, sumabs, failure,
                input_failure, mean_failure, gain_failure)
        end
    end
end
