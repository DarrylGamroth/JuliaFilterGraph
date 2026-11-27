@kernel function _cuda_readout_merge_status!(status, local_status)
    if @inbounds(iszero(status[1]) && !iszero(local_status[1]))
        @inbounds status[1] = (Int32(1) << 8) | local_status[1]
    end
end

# One workgroup writes each reconstructed coordinate. Both kernels execute on
# the ordered compute stream; WFS writes only its own immutable block packet.
@kernel function _cuda_readout_reconstruct!(candidate, matrix, packet, status, failures,
        first::Bool, ::Val{Lanes}) where {Lanes}
    lane = @index(Local, Linear)
    _, row = @index(Group, NTuple)
    sums = @localmem Float32 (Lanes,)
    valid = @inbounds iszero(status[1])
    count = Int(@inbounds packet[1].column)
    value = 0.0f0
    entry = lane
    while valid && entry <= count
        measurement = @inbounds packet[entry + 1]
        @inbounds value = muladd(matrix[Int(measurement.column) + 1, row], measurement.value, value)
        entry += Lanes
    end
    @inbounds sums[lane] = value
    @synchronize
    stride = Lanes >> 1
    while stride >= 1
        if lane <= stride
            @inbounds sums[lane] += sums[lane + stride]
        end
        @synchronize
        stride >>= 1
    end
    if lane == 1
        @inbounds begin
            result = first ? sums[1] : candidate[row] + sums[1]
            failures[row] = valid && !isfinite(result) ? Int32(1) : Int32(0)
            valid && (candidate[row] = result)
        end
    end
end

@kernel function _cuda_readout_reconstruction_status!(status, failures)
    # Only this workgroup writes global status. No atomics or cross-stream
    # status accesses are needed, and partial values never reach the host.
    lane = @index(Local, Linear)
    flags = @localmem Int32 (256,)
    failure = Int32(0)
    index = lane
    while index <= length(failures)
        @inbounds failure = max(failure, failures[index])
        index += 256
    end
    @inbounds flags[lane] = failure
    @synchronize
    stride = 128
    while stride >= 1
        if lane <= stride
            @inbounds flags[lane] = max(flags[lane], flags[lane + stride])
        end
        @synchronize
        stride >>= 1
    end
    if lane == 1 && @inbounds(iszero(status[1]) && !iszero(flags[1]))
        @inbounds status[1] = (Int32(2) << 8) |
                              Int32(FilterGraphAlgorithms.IncrementalDenseReconstructionNonFiniteOutput)
    end
end

function _enqueue_cuda_readout_reconstruction!(
        wfs, candidate, matrix, frame_status, failures, ordinal, lanes)
    terminal = ordinal == length(wfs.storage.packets)
    empty_block = _empty_raw_pyramid_block(wfs, ordinal)
    if empty_block && ordinal == 1
        fill!(candidate, zero(eltype(candidate)))
    end
    # Raw pixels were certified finite at preparation. A nonterminal block
    # with no selected pixels contributes neither a failure nor a term.
    empty_block && !terminal && return nothing
    backend = KernelAbstractions.get_backend(candidate)
    _cuda_readout_merge_status!(backend, 1)(frame_status.value, wfs.storage.statuses[ordinal]; ndrange = 1)
    terminal && _enqueue_cuda_readout_terminal!(wfs, frame_status.value)
    empty_block && return nothing
    _cuda_readout_reconstruct!(backend, (lanes, 1))(
        candidate, matrix, wfs.storage.packets[ordinal], frame_status.value,
        failures, ordinal == 1, Val(lanes); ndrange = (lanes, length(candidate))
    )
    _cuda_readout_reconstruction_status!(backend, 256)(frame_status.value, failures; ndrange = 256)
    return nothing
end
