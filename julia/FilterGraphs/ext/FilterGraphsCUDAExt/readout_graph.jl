mutable struct CUDAReadoutGraph{W, M, V, F, D, N, O, H, S, MS, G, E, HB, DP, FB}
    context::CUDACapturedContext
    input_stream::CUDA.CuStream
    wfs_stream::CUDA.CuStream
    host_blocks::HB
    device_blocks::Vector{D}
    device_pointers::DP
    input_events::Vector{CUDA.CuEvent}
    wfs_events::Vector{CUDA.CuEvent}
    wfs::W
    matrix::M
    candidate::V
    failures::F
    nodes::N
    device_outputs::O
    host_outputs::H
    frame_status::S
    metadata_sources::MS
    wfs_graphs::Vector{G}
    wfs_executables::Vector{E}
    reconstruction_graphs::Vector{G}
    reconstruction_executables::Vector{E}
    suffix_graph::G
    suffix_executable::E
    feedback::FB
    block_count::Int
    pending::Bool
    faulted::Bool
    block_submissions::UInt64
    terminal_waits::UInt64
end

function _capture_cuda_readout_stages(
        wfs, context, candidate, matrix, status, failures, nodes, feedback, lanes)
    for ordinal in eachindex(wfs.storage.packets)
        _enqueue_cuda_readout_wfs!(wfs, ordinal)
        _enqueue_cuda_readout_reconstruction!(
            wfs, candidate, matrix, status, failures, ordinal, lanes)
    end
    FilterGraphs._enqueue_captured_nodes!(nodes)
    CUDA.synchronize(context.stream; blocking = true)
    _reset_cuda_readout_wfs!(wfs)
    FilterGraphs._reset_captured_nodes!(nodes)
    status.clear!(status.value; ndrange = 1)
    CUDA.synchronize(context.stream; blocking = true)

    # Record on the preparation stream so CUDA array tracking cannot request
    # cross-stream synchronization during capture. Replay uses explicit events
    # and launches these native graphs on the retained WFS stream.
    wfs_graphs = map(eachindex(wfs.storage.packets)) do ordinal
        _capture_cuda_graph() do
            _enqueue_cuda_readout_wfs!(wfs, ordinal)
        end
    end
    reconstruction_graphs = map(eachindex(wfs.storage.packets)) do ordinal
        _capture_cuda_graph() do
            ordinal == 1 && status.clear!(status.value; ndrange = 1)
            _enqueue_cuda_readout_reconstruction!(
                wfs, candidate, matrix, status, failures, ordinal, lanes)
        end
    end
    suffix_graph = _capture_cuda_graph() do
        FilterGraphs._enqueue_captured_nodes!(nodes)
        FilterGraphs._enqueue_captured_feedback!(feedback)
    end
    return wfs_graphs, reconstruction_graphs, suffix_graph
end

function FilterGraphs._prepare_gpu_readout_backend(graph, device::CUDA.CuDevice, lanes, ports, calibration, feedback)
    context = _prepare_captured_context(device)
    return FilterGraphs._with_captured_context(context) do
        # Cold lifetime roots only. No DMA reads pinned input during preparation.
        # A failed drain quarantines these objects when no owner can be returned.
        roots = Any[context]
        try
            readout = FilterGraphs._gpu_readout_plan(graph.nodes[1].prepared.plan)
            shape = first(values(FilterGraphs._readout_input_formats(graph, ports, calibration))).shape
            T = isnothing(calibration) ? Float32 : UInt16
            host_blocks = [CUDA.pin(zeros(T, shape)) for _ in 1:readout.block_count]
            device_blocks = [CUDA.zeros(T, shape) for _ in 1:readout.block_count]
            push!(roots, host_blocks, device_blocks)
            device_pointers = map(pointer, device_blocks)
            status = _prepare_captured_frame_status(context)
            push!(roots, status)
            coefficients = if isnothing(calibration)
                nothing
            else
                for (flat, background) in zip(calibration.flat, calibration.background)
                    isfinite(flat * -background) && isfinite(flat * (65535f0-background)) ||
                        throw(ArgumentError("raw readout calibration can overflow Float32"))
                end
                (; flat=CUDA.CuArray(calibration.flat), background=CUDA.CuArray(calibration.background))
            end
            push!(roots, coefficients)
            wfs = _prepare_cuda_readout_wfs(graph.nodes[1].prepared, device_blocks, coefficients)
            push!(roots, wfs)
            device_inputs = FilterGraphs._captured_external_buffers(context,
                graph.input_formats)
            push!(roots, device_inputs)
            matrix = CUDA.CuArray(permutedims(graph.nodes[2].prepared.plan.matrix))
            node_outputs = FilterGraphs._captured_node_output_buffers(context, graph.nodes)
            push!(roots, matrix, node_outputs)
            candidate = node_outputs[2][1]
            failures = CUDA.zeros(Int32, length(candidate))
            push!(roots, failures)
            nodes = FilterGraphs._prepare_captured_nodes(
                graph.nodes[3:end], node_outputs[3:end], graph.input_routes[3:end],
                device_inputs, node_outputs, device, status.value, Int32(3),
                FilterGraphs.CapturedWorkgroupProfile(); roots,
            )
            nodes = FilterGraphs._prepare_captured_validation(
                device, nodes, graph.input_routes[3:end], node_outputs[3:end],
                device_inputs, node_outputs; indices=3:length(node_outputs),
            )
            push!(roots, nodes)
            FilterGraphs._require_captured_methods(graph.nodes[3:end], device, nodes)
            preflight = FilterGraphs._preflight_captured_nodes(nodes)
            isnothing(preflight) ||
                throw(FilterGraphs.GraphConfigurationError("captured readout suffix is not configured: $preflight"))
            all_outputs = FilterGraphs._captured_external_outputs(graph, node_outputs)
            bindings = FilterGraphs._prepare_captured_feedback(device, feedback, device_inputs,
                all_outputs, status)
            FilterGraphs._reset_captured_feedback!(bindings)
            push!(roots, bindings, all_outputs)
            device_outputs = FilterGraphs._select_captured_ports(all_outputs, ports.outputs)
            formats = FilterGraphs._select_captured_ports(graph.output_formats, ports.outputs)
            host_outputs = map(format -> _allocate_captured_host_buffer(context, format), formats)
            push!(roots, host_outputs)
            input_stream, wfs_stream = CUDA.CuStream(), CUDA.CuStream()
            wfs_graphs, reconstruction_graphs, suffix_graph =
                FilterGraphs._with_captured_workspace(context) do
                    _capture_cuda_readout_stages(
                        wfs, context, candidate, matrix, status, failures, nodes, bindings, lanes
                    )
                end
            wfs_executables = map(graph -> _instantiate_cuda_graph(context, graph), wfs_graphs)
            reconstruction_executables = map(
                graph -> _instantiate_cuda_graph(context, graph), reconstruction_graphs)
            suffix_executable = _instantiate_cuda_graph(context, suffix_graph)
            return CUDAReadoutGraph(
                context, input_stream, wfs_stream, host_blocks, device_blocks, device_pointers,
                [CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ in 1:readout.block_count],
                [CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ in 1:readout.block_count],
                wfs, matrix, candidate, failures, nodes, device_outputs, host_outputs, status,
                ports.metadata_sources,
                wfs_graphs, wfs_executables, reconstruction_graphs, reconstruction_executables,
                suffix_graph, suffix_executable, bindings, readout.block_count, false, false, UInt64(0), UInt64(0)
            )
        catch
            FilterGraphs._drain_captured_storage!(context, roots)
            rethrow()
        end
    end
end

function FilterGraphs._submit_gpu_readout!(owner::CUDAReadoutGraph, block, ordinal)
    try
        copyto!(owner.host_blocks[ordinal], block)
        FilterGraphs._with_captured_context(owner.context) do
            owner.pending = true
            GC.@preserve owner begin
                CUDA.unsafe_copyto!(
                    owner.device_pointers[ordinal], pointer(owner.host_blocks[ordinal]),
                    length(block); stream = owner.input_stream, async = true)
            end
            CUDA.CUDACore.cuEventRecord(owner.input_events[ordinal], owner.input_stream)
            CUDA.wait(owner.input_events[ordinal], owner.wfs_stream)
            CUDA.launch(owner.wfs_executables[ordinal], owner.wfs_stream)
            CUDA.CUDACore.cuEventRecord(owner.wfs_events[ordinal], owner.wfs_stream)
            CUDA.wait(owner.wfs_events[ordinal], owner.context.stream)
            CUDA.launch(owner.reconstruction_executables[ordinal], owner.context.stream)
            owner.block_submissions += UInt64(1)
        end
    catch
        # The frame already admitted this block. A later block cannot replace it.
        owner.faulted = true
        rethrow()
    end
    return nothing
end

function FilterGraphs._finish_gpu_readout!(owner::CUDAReadoutGraph)
    try
        return FilterGraphs._with_captured_context(owner.context) do
            CUDA.launch(owner.suffix_executable, owner.context.stream)
            FilterGraphs._enqueue_captured_frame_status_copy!(owner.context, owner.frame_status)
            FilterGraphs._transfer_captured_outputs!(owner.context, owner.host_outputs, owner.device_outputs)
            CUDA.synchronize(owner.context.stream; blocking = true)
            owner.pending = false
            owner.terminal_waits += UInt64(1)
            return owner.frame_status.host[1]
        end
    catch
        owner.faulted = true
        rethrow()
    end
end

function FilterGraphs._drain_gpu_readout!(owner::CUDAReadoutGraph)
    owner.pending || return nothing
    previous = _activate_captured_context(owner.context)
    failure = nothing
    try
        for stream in (owner.input_stream, owner.wfs_stream, owner.context.stream)
            try
                CUDA.synchronize(stream; blocking = true)
            catch error
                owner.faulted = true
                isnothing(failure) && (failure = error)
            end
        end
        isnothing(failure) || throw(failure)
        owner.pending = false
    finally
        _restore_captured_context!(owner.context, previous)
    end
    return nothing
end

function FilterGraphs._reset_gpu_readout!(owner::CUDAReadoutGraph)
    FilterGraphs._drain_gpu_readout!(owner)
    FilterGraphs._with_captured_context(owner.context) do
        owner.pending = true
        try
            _reset_cuda_readout_wfs!(owner.wfs)
            FilterGraphs._reset_captured_nodes!(owner.nodes)
            FilterGraphs._reset_captured_feedback!(owner.feedback)
            owner.frame_status.clear!(owner.frame_status.value; ndrange = 1)
            fill!(owner.candidate, 0.0f0)
            CUDA.synchronize(owner.context.stream; blocking = true)
            owner.pending = false
        catch
            owner.faulted = true
            rethrow()
        end
    end
    return nothing
end

function FilterGraphs._gpu_readout_failure(owner::CUDAReadoutGraph, status::Int32)
    iszero(status) && return nothing
    node = status >> 8
    local_status = status & Int32(0xff)
    node == 1 && return _cuda_readout_failure(owner.wfs, local_status)
    node == 2 &&
        return FilterGraphAlgorithms.IncrementalDenseReconstructionFailure(local_status)
    return FilterGraphs._complete_captured_nodes!(owner.nodes, status)
end

function FilterGraphs._gpu_readout_counters(owner::CUDAReadoutGraph)
    return (;
        region_capacity = owner.block_count,
        block_submissions = owner.block_submissions,
        terminal_waits = owner.terminal_waits,
        captured_suffix_nodes = length(owner.nodes)
    )
end
