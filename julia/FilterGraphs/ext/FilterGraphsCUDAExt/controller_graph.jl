@kernel function _pack_controller_command!(words, command, status)
    index = @index(Global, Linear)
    if index <= length(command)
        @inbounds words[index] = reinterpret(UInt32, command[index])
    elseif index == length(command) + 1
        @inbounds words[index] = reinterpret(UInt32, status[1])
    end
end

# The native graph records an external event after packing the validated
# command. Its feedback suffix continues on the compute stream while a second
# stream copies the packed command and prefix status to retained pinned storage.
struct CUDAControllerExecution{G,E,D,H,C,F,OH,OD,MD,MS,K}
    graph::G
    executable::E
    copy_stream::CUDA.CuStream
    ready::CUDA.CuEvent
    command_done::CUDA.CuEvent
    words::D
    host_words::H
    command_host::C
    frame_status::F
    other_host::OH
    other_device::OD
    command_metadata::MD
    command_metadata_sources::MS
    pack!::K
end

function _drain_cuda_controller_streams!(context, copy_stream)
    failure = nothing
    for stream in (context.stream, copy_stream)
        try
            CUDA.synchronize(stream; blocking=true)
        catch error
            isnothing(failure) && (failure = error)
        end
    end
    isnothing(failure) || throw(failure)
    return nothing
end

function _quarantine_cuda_controller_storage!(owner, roots, failure, backtrace)
    retained = FilterGraphs.CapturedStorageQuarantine(
        roots, CapturedException(failure, backtrace),
    )
    lock(FilterGraphs._captured_storage_quarantine_lock) do
        # Preserve the first borrowed inputs if a later recovery also fails.
        get!(FilterGraphs._captured_storage_quarantine, owner, retained)
    end
    return false
end

function _drain_cuda_controller_storage!(context, copy_stream, owner, roots=owner)
    try
        # Activation or restoration itself may have failed. Select the retained
        # runtime context directly rather than relying on context.active.
        CUDA.context!(context.runtime_context) do
            if isnothing(copy_stream)
                _synchronize_captured_graph!(context)
            else
                _drain_cuda_controller_streams!(context, copy_stream)
            end
        end
    catch failure
        return _quarantine_cuda_controller_storage!(
            owner, roots, failure, catch_backtrace(),
        )
    end
    return true
end

function FilterGraphs._prepare_controller_execution(
    g::FilterGraphs.PreparedCapturedGraph{<:CUDACapturedContext}, key::Val{K}, cut::Int,
) where {K}
    # Cold roots cover failures before an execution owner can be returned,
    # including errors while restoring the caller's CUDA context.
    roots = Any[g]
    copy_stream = nothing
    try
        return FilterGraphs._with_captured_context(g.context) do
            copy_stream = CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING)
            push!(roots, copy_stream)
            ready = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
            command_done = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
            push!(roots, ready, command_done)
            command = FilterGraphs._controller_buffer(g.device_outputs, key)
            format = (element_type=UInt32, shape=(length(command) + 1,))
            words = _allocate_captured_buffer(g.context, format)
            push!(roots, words)
            host_words = _allocate_captured_host_buffer(g.context, format)
            command_host = view(reinterpret(Float32, host_words), 1:length(command))
            outcome = g.executable.frame_status
            status = CapturedFrameStatus(
                outcome.value, outcome.host,
                _clear_captured_frame_status!(KernelAbstractions.get_backend(words), 1),
            )
            pack! = _pack_controller_command!(KernelAbstractions.get_backend(words), 256)
            push!(roots, host_words, command_host, status, pack!)
            # Compile only the new packing kernel. Scientific adapters were
            # warmed by captured preparation and reset by controller preparation.
            pack!(words, command, status.value; ndrange=length(words))
            _synchronize_captured_graph!(g.context)
            prefix, suffix = g.nodes[1:cut], g.nodes[(cut + 1):end]
            graph = FilterGraphs._with_captured_workspace(g.context) do
                _capture_cuda_graph() do
                    _enqueue_captured_frame!(status, prefix)
                    pack!(words, command, status.value; ndrange=length(words))
                    CUDA.CUDACore.cuEventRecordWithFlags(
                        ready, g.context.stream, CUDA.CUDACore.CU_EVENT_RECORD_EXTERNAL,
                    )
                    FilterGraphs._enqueue_captured_nodes!(suffix)
                    FilterGraphs._enqueue_captured_feedback!(g.executable.feedback)
                end
            end
            push!(roots, graph)
            executable = _instantiate_cuda_graph(g.context, graph)
            push!(roots, executable)
            index = findfirst(==(K), keys(g.output_formats))
            execution = CUDAControllerExecution(
                graph, executable, copy_stream, ready, command_done, words, host_words,
                command_host, status,
                FilterGraphs._controller_without_command(g.host_outputs, key),
                FilterGraphs._controller_without_command(g.device_outputs, key),
                (FilterGraphs._controller_buffer(g.published_metadata, key),),
                (g.output_metadata_sources[index],), pack!,
            )
            push!(roots, execution)
            return execution
        end
    catch
        _drain_cuda_controller_storage!(g.context, copy_stream, roots)
        rethrow()
    end
end

function _enqueue_cuda_controller_command_copy!(execution::CUDAControllerExecution)
    CUDA.wait(execution.ready, execution.copy_stream)
    words, host_words = execution.words, execution.host_words
    GC.@preserve execution words host_words begin
        CUDA.CUDACore.cuMemcpyDtoHAsync_v2(
            pointer(host_words), pointer(words), sizeof(host_words), execution.copy_stream,
        )
    end
    CUDA.CUDACore.cuEventRecord(execution.command_done, execution.copy_stream)
    return nothing
end

function _execute_cuda_controller_frame!(
    outputs, owner, inputs, metadata, execution::CUDAControllerExecution,
)
    g = owner.graph
    FilterGraphs._transfer_captured_inputs!(
        g.context, g.device_inputs, inputs, g.executable.host_inputs,
    )
    CUDA.launch(execution.executable, g.context.stream)
    _enqueue_cuda_controller_command_copy!(execution)
    FilterGraphs._enqueue_captured_frame_status_copy!(g.context, execution.frame_status)
    FilterGraphs._transfer_captured_outputs!(
        g.context, execution.other_host, execution.other_device,
    )
    CUDA.CUDACore.cuEventSynchronize(execution.command_done)
    if iszero(reinterpret(Int32, execution.host_words[end]))
        # Mark before the first caller-owned write or callback can escape.
        owner.command_published = true
        command = FilterGraphs._controller_buffer(outputs, owner.command)
        copyto!(command, execution.command_host)
        FilterGraphs._publish_captured_metadata!(
            execution.command_metadata, execution.command_metadata_sources, metadata,
        )
        FilterGraphs._publish_controller_command!(owner, command)
    end
    _drain_cuda_controller_streams!(g.context, execution.copy_stream)
    status = FilterGraphs._read_captured_frame_status(execution.frame_status)
    return FilterGraphs._complete_captured_nodes!(g.nodes, status, g.rejection)
end

function FilterGraphs._execute_controller!(
    outputs, owner, inputs, metadata, execution::CUDAControllerExecution,
)
    g = owner.graph
    FilterGraphs._throw_unavailable_captured_graph(g)
    FilterGraphs._clear_rejection!(g.rejection)
    preflight = FilterGraphs._preflight_captured_nodes(g.nodes, g.rejection)
    if preflight !== nothing
        FilterGraphs._rejection_inputs!(g.rejection, metadata)
        return preflight
    end
    g.state = FilterGraphs.CapturedGraphRunning
    result = try
        previous = _activate_captured_context(g.context)
        result = try
            GC.@preserve owner inputs outputs execution begin
                _execute_cuda_controller_frame!(outputs, owner, inputs, metadata, execution)
            end
        finally
            _restore_captured_context!(g.context, previous)
        end
        if result !== nothing
            FilterGraphs._rejection_inputs!(g.rejection, metadata)
        else
            owner.command_published || error("successful controller frame without command publication")
            FilterGraphs._copy_captured_outputs!(
                values(FilterGraphs._controller_without_command(outputs, owner.command)),
                values(execution.other_host),
            )
            FilterGraphs._publish_captured_metadata!(
                values(g.published_metadata), g.output_metadata_sources, metadata,
            )
        end
        g.state = FilterGraphs.CapturedGraphReady
        result
    catch error
        fault = FilterGraphs._fault_captured_graph!(g, :controller_execution, error)
        _drain_cuda_controller_storage!(
            g.context, execution.copy_stream, owner, (owner, inputs, outputs),
        )
        throw(fault)
    end
    if result !== nothing
        owner.command_published && throw(FilterGraphs.ControllerFault(result))
        return result
    end
    return g.availability
end

function FilterGraphs._reset_controller_execution!(owner, execution::CUDAControllerExecution)
    g = owner.graph
    try
        CUDA.context!(g.context.runtime_context) do
            _drain_cuda_controller_streams!(g.context, execution.copy_stream)
        end
        # A previous restoration failure can leave the context token active.
        # Complete that restoration before the ordinary reset activates it.
        g.context.active && _restore_captured_context!(g.context, nothing)
        FilterGraphs.reset!(g)
    catch error
        fault = FilterGraphs._fault_captured_graph!(g, :controller_reset, error)
        _drain_cuda_controller_storage!(g.context, execution.copy_stream, owner)
        throw(fault)
    end
    FilterGraphs._release_captured_storage!(owner)
    return nothing
end
