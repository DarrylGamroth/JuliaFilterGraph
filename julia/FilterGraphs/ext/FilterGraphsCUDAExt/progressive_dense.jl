import FilterGraphAlgorithms

# One retained packet per declared nonempty block; there is no frame queue.
struct CUDADenseMeasurement
    value::Float32
    column::UInt32
end

@kernel function _cuda_dense_region!(
    candidate,
    matrix,
    packet,
    first::Bool,
    ::Val{Lanes},
) where {Lanes}
    lane = @index(Local, Linear)
    _, row = @index(Group, NTuple)
    sums = @localmem Float32 (Lanes,)
    count = Int(@inbounds packet[1].column)
    value = 0.0f0
    entry = lane
    while entry <= count
        measurement = @inbounds packet[entry+1]
        @inbounds value =
            muladd(matrix[Int(measurement.column)+1, row], measurement.value, value)
        entry += Lanes
    end
    @inbounds sums[lane] = value
    @synchronize
    stride = Lanes >> 1
    while stride >= 1
        if lane <= stride
            @inbounds sums[lane] += sums[lane+stride]
        end
        @synchronize
        stride >>= 1
    end
    if lane == 1
        @inbounds candidate[row] = first ? sums[1] : candidate[row] + sums[1]
    end
end

mutable struct CUDAProgressiveDense{M,V,D,G,E}
    context::CUDACapturedContext
    input_stream::CUDA.CuStream
    matrix::M
    candidate::V
    host_output::Vector{Float32}
    packets::Vector{Vector{CUDADenseMeasurement}}
    device_packets::Vector{D}
    packet_pointers::Vector{Ptr{CUDADenseMeasurement}}
    device_pointers::Vector{CUDA.CuPtr{CUDADenseMeasurement}}
    output_pointer::CUDA.CuPtr{Float32}
    graphs::Vector{G}
    executables::Vector{E}
    input_events::Vector{CUDA.CuEvent}
    measurement_limit::Float32
    lanes::Int
    submitted::Int
    pending::Bool
    faulted::Bool
    block_submissions::UInt64
    terminal_waits::UInt64
end

function FilterGraphs._prepare_gpu_dense_executor(
    prepared,
    device::CUDA.CuDevice,
    lanes,
)
    context = _prepare_captured_context(device)
    return FilterGraphs._with_captured_context(context) do
        capacity = prepared.formats[1].shape[1]
        blocks = Int(prepared.formats[1].rate[1])
        zero_entry = CUDADenseMeasurement(0.0f0, UInt32(0))
        packets = [CUDA.pin(fill(zero_entry, capacity + 1)) for _ = 1:blocks]
        # Initialize on the device: preparation must not start DMA from a
        # pinned host packet before a rooted owner exists.
        device_packets =
            [CUDA.CuArray{CUDADenseMeasurement}(undef, capacity + 1) for _ = 1:blocks]
        foreach(packet -> fill!(packet, zero_entry), device_packets)
        matrix = CUDA.CuArray(permutedims(prepared.plan.matrix))
        candidate = CUDA.zeros(Float32, size(prepared.plan.matrix, 1))
        host_output = CUDA.pin(zeros(Float32, length(candidate)))
        input_stream = CUDA.CuStream()
        input_events = [CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ = 1:blocks]
        kernel! = _cuda_dense_region!(KernelAbstractions.get_backend(candidate), (lanes, 1))
        for index = 1:blocks
            kernel!(
                candidate,
                matrix,
                device_packets[index],
                index == 1,
                Val(lanes);
                ndrange = (lanes, length(candidate)),
            )
        end
        CUDA.synchronize(context.stream; blocking = true)
        graphs = map(1:blocks) do index
            _capture_cuda_graph() do
                kernel!(
                    candidate,
                    matrix,
                    device_packets[index],
                    index == 1,
                    Val(lanes);
                    ndrange = (lanes, length(candidate)),
                )
            end
        end
        # Initialization and uploads complete before the independent input
        # stream can use cached pointers, bypassing CUDA array tracking.
        executables = map(graph -> _instantiate_cuda_graph(context, graph), graphs)
        return CUDAProgressiveDense(
            context,
            input_stream,
            matrix,
            candidate,
            host_output,
            packets,
            device_packets,
            pointer.(packets),
            pointer.(device_packets),
            pointer(candidate),
            graphs,
            executables,
            input_events,
            FilterGraphs._gpu_dense_measurement_limit(prepared.plan.matrix),
            Int(lanes),
            0,
            false,
            false,
            UInt64(0),
            UInt64(0),
        )
    end
end

function FilterGraphs._submit_gpu_dense_block!(
    executor::CUDAProgressiveDense,
    measurements,
    columns,
    count,
)
    iszero(count) && return true
    executor.submitted < length(executor.packets) || return false
    index = executor.submitted + 1
    packet = executor.packets[index]
    @inbounds packet[1] = CUDADenseMeasurement(0.0f0, UInt32(count))
    for entry = 1:count
        @inbounds packet[entry+1] =
            CUDADenseMeasurement(measurements[entry], columns[entry])
    end
    previous = _activate_captured_context(executor.context)
    executor.pending = true
    try
        # Slots are immutable until frame completion. The host can release its
        # borrowed input after packing; CUDA events order DMA before GPU reads.
        GC.@preserve executor begin
            CUDA.unsafe_copyto!(
                executor.device_pointers[index],
                executor.packet_pointers[index],
                count + 1;
                stream = executor.input_stream,
                async = true,
            )
        end
        CUDA.CUDACore.cuEventRecord(executor.input_events[index], executor.input_stream)
        CUDA.wait(executor.input_events[index], executor.context.stream)
        CUDA.launch(executor.executables[index], executor.context.stream)
        executor.submitted = index
        executor.block_submissions += UInt64(1)
    catch
        executor.faulted = true
        rethrow()
    finally
        _restore_captured_context!(executor.context, previous)
    end
    return true
end

function FilterGraphs._drain_gpu_dense!(executor::CUDAProgressiveDense)
    executor.pending || return nothing
    previous = _activate_captured_context(executor.context)
    failure = nothing
    try
        for stream in (executor.input_stream, executor.context.stream)
            try
                CUDA.synchronize(stream; blocking = true)
            catch error
                executor.faulted = true
                isnothing(failure) && (failure = error)
            end
        end
        isnothing(failure) || throw(failure)
        executor.pending = false
    finally
        _restore_captured_context!(executor.context, previous)
    end
    return nothing
end

function FilterGraphs._clear_gpu_dense!(executor::CUDAProgressiveDense)
    FilterGraphs._drain_gpu_dense!(executor)
    executor.submitted = 0
    return nothing
end

function FilterGraphs._finish_gpu_dense!(output, executor::CUDAProgressiveDense)
    iszero(executor.submitted) && return nothing
    previous = _activate_captured_context(executor.context)
    try
        GC.@preserve executor begin
            CUDA.unsafe_copyto!(
                pointer(executor.host_output),
                executor.output_pointer,
                length(output);
                stream = executor.context.stream,
                async = true,
            )
        end
        CUDA.synchronize(executor.context.stream; blocking = true)
        executor.pending = false
        executor.terminal_waits += UInt64(1)
        copyto!(output, executor.host_output)
    catch
        executor.faulted = true
        rethrow()
    finally
        _restore_captured_context!(executor.context, previous)
    end
    return nothing
end

function FilterGraphs._replay_gpu_dense!(output, executor::CUDAProgressiveDense, matrix)
    FilterGraphs._drain_gpu_dense!(executor)
    fill!(output, 0.0f0)
    # Recompute from the original entries, not the rounded GPU prefix. This
    # preserves serial arithmetic when a later block exceeds the certificate.
    for index = 1:executor.submitted
        packet = executor.packets[index]
        count = Int(packet[1].column)
        for entry = 1:count
            measurement = packet[entry+1]
            FilterGraphAlgorithms._accumulate_incremental_dense_reconstruction_rows!(
                output,
                matrix,
                Int(measurement.column) + 1,
                measurement.value,
                eachindex(output),
            )
        end
    end
    return nothing
end

function FilterGraphs._gpu_dense_counters(executor::CUDAProgressiveDense)
    return (
        measurement_limit = executor.measurement_limit,
        region_capacity = length(executor.packets),
        block_submissions = executor.block_submissions,
        terminal_waits = executor.terminal_waits,
    )
end
