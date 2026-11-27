# Stored Int32 ranks come from the row algorithm's declared failure enum.
const PYRAMID_ROW_INPUT_FAILURE = Int32(FilterGraphAlgorithms.PyramidPixelRowNonFiniteInput)
const PYRAMID_ROW_MEAN_FAILURE = Int32(FilterGraphAlgorithms.PyramidPixelRowNonFiniteMean)
const PYRAMID_ROW_GAIN_FAILURE = Int32(FilterGraphAlgorithms.PyramidPixelRowNonFiniteNormalizationGain)
const PYRAMID_ROW_OUTPUT_FAILURE = Int32(FilterGraphAlgorithms.PyramidPixelRowNonFiniteOutput)

const PYRAMID_READOUT_LANES = 256

@inline function _pyramid_normalize_readout(input, gain)
    normalized = gain * input
    failure = !isfinite(input) ? PYRAMID_ROW_INPUT_FAILURE :
        !isfinite(normalized) ? PYRAMID_ROW_OUTPUT_FAILURE : Int32(0)
    return normalized, failure
end

# Scientific readout measurements and pyramid normalization state. The engine
# supplies immutable per-ordinal packets, validated mappings and stream order.

@kernel function _pyramid_readout_pixels!(
        packet, flags, detector, indices, columns, gain)
    entry = @index(Global, Linear)
    input = @inbounds detector[indices[entry]]
    # Input must be checked even on the initial frame, whose gain is zero.
    normalized, failure = _pyramid_normalize_readout(input, @inbounds gain[1])
    @inbounds flags[entry] = failure
    @inbounds packet[entry + 1] = eltype(packet)(normalized, UInt32(columns[entry] -
                                                                    1))
end

@kernel function _finish_pyramid_readout!(
        packet, status, flags, detector, pupil_indices, mean_scale,
        candidate_gain, mean_output, terminal::Bool, flag_count=length(flags), measurement_count=flag_count
)
    # One 256-lane group uses strided F32 partials and a fixed reduction tree.
    # Cancellation and overflow boundaries can differ from serial CPU order.
    # Nonfinite input takes priority over arithmetic overflow within the mean.
    lane = @index(Local, Linear)
    first_failures = @localmem UInt32 (PYRAMID_READOUT_LANES,)
    sums = @localmem Float32 (PYRAMID_READOUT_LANES,)
    mean_failures = @localmem Int32 (PYRAMID_READOUT_LANES,)
    first_failure = UInt32(0)
    entry = Int(lane)
    while entry <= flag_count
        if !iszero(@inbounds flags[entry])
            first_failure = UInt32(entry)
            break
        end
        entry += PYRAMID_READOUT_LANES
    end
    sum = 0.0f0
    mean_failure = Int32(0)
    if terminal
        index = Int(lane)
        while index <= length(pupil_indices)
            input = @inbounds detector[pupil_indices[index]]
            if !isfinite(input)
                mean_failure = PYRAMID_ROW_INPUT_FAILURE
            else
                sum += input
                if !isfinite(sum) && mean_failure != PYRAMID_ROW_INPUT_FAILURE
                    mean_failure = PYRAMID_ROW_MEAN_FAILURE
                end
            end
            index += PYRAMID_READOUT_LANES
        end
    end
    @inbounds begin
        first_failures[lane] = first_failure
        sums[lane] = sum
        mean_failures[lane] = mean_failure
    end
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds begin
                first = first_failures[lane]
                other = first_failures[lane + stride]
                if iszero(first) || (!iszero(other) && other < first)
                    first_failures[lane] = other
                end
                sum = sums[lane] + sums[lane + stride]
                sums[lane] = sum
                failure = mean_failures[lane]
                other_failure = mean_failures[lane + stride]
                mean_failures[lane] = failure == PYRAMID_ROW_INPUT_FAILURE || other_failure == PYRAMID_ROW_INPUT_FAILURE ?
                                      PYRAMID_ROW_INPUT_FAILURE :
                                      failure == PYRAMID_ROW_MEAN_FAILURE || other_failure == PYRAMID_ROW_MEAN_FAILURE ||
                                      !isfinite(sum) ?
                                      PYRAMID_ROW_MEAN_FAILURE : Int32(0)
            end
        end
        @synchronize
    end
    if lane == 1
        @inbounds begin
            # Measurement failures retain scientific completion-group priority.
            first = first_failures[1]
            failure = iszero(first) ? Int32(0) : flags[Int(first)]
            packet[1] = eltype(packet)(0.0f0, UInt32(measurement_count))
            if terminal && iszero(failure)
                failure = mean_failures[1]
                mean = sums[1] * mean_scale
                gain = mean > 0.0f0 ? inv(mean) : 0.0f0
                if iszero(failure)
                    if !isfinite(mean)
                        failure = PYRAMID_ROW_MEAN_FAILURE
                    elseif !isfinite(gain)
                        failure = PYRAMID_ROW_GAIN_FAILURE
                    else
                        mean_output[1] = mean
                        candidate_gain[1] = gain
                    end
                end
            end
            status[1] = failure
        end
    end
end
