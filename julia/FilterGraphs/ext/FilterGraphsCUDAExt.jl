module FilterGraphsCUDAExt

using CUDA
using FilterGraphs
using KernelAbstractions

function FilterGraphs._prepare_captured_aliases(
    ::CUDA.CuDevice, nodes, routes, inputs, outputs, protected,
)
    return FilterGraphs._captured_alias_outputs(nodes, routes, inputs, outputs, protected)
end

function FilterGraphs._prepare_captured_validation(
    ::CUDA.CuDevice, nodes, routes, outputs, device_inputs, node_outputs;
    indices=eachindex(nodes),
)
    refined = FilterGraphs._reuse_captured_validation(
        nodes, routes, outputs, device_inputs, node_outputs,
    )
    # SCC residual is ineligible for either CUDA fork branch. An adjacent
    # reconstruction therefore runs after it and uses the same status storage.
    refined = _reuse_scc_residual_validation(refined, indices, routes, outputs, device_inputs, node_outputs)
    return _reuse_captured_bounds(refined, routes, outputs, device_inputs, node_outputs)
end

FilterGraphs._prepare_captured_fusion(::CUDA.CuDevice, nodes, routes) =
    _fuse_captured_scc_pixels(nodes, routes)

import FilterGraphs:
    _activate_captured_context,
    _allocate_captured_buffer,
    _allocate_captured_host_buffer,
    CapturedFrameOutcome,
    CapturedFrameStatus,
    _copy_from_captured!,
    _copy_to_captured!,
    _enqueue_captured_frame!,
    _launch_captured_graph!,
    _prepare_captured_pipeline_backend,
    _prepare_captured_pipeline_events,
    _enqueue_captured_pipeline_input!,
    _captured_pipeline_input_done,
    _enqueue_captured_pipeline_graph!,
    _captured_pipeline_graph_done,
    _enqueue_captured_pipeline_output!,
    _captured_pipeline_output_done,
    _drain_captured_pipeline_backend!,
    _prepare_captured_context,
    _prepare_captured_frame_status,
    _record_captured_graph,
    _restore_captured_context!,
    _synchronize_captured_graph!

# Newer capture APIs select a dedicated stream unless one is supplied. Keep
# capture on the stream where preparation installed the retained cuBLAS workspace.
const _CAPTURE_SUPPORTS_STREAM = hasmethod(CUDA.capture, Tuple{Function}, (:stream,))

mutable struct CUDACapturedContext
    device::CUDA.CuDevice
    runtime_context::CUDA.CuContext
    stream::CUDA.CuStream
    # Native graphs retain these addresses; all compute uses the ordered stream.
    workspace::CUDA.CuArray{UInt8,1,CUDA.DeviceMemory}
    # CuStream is mutable, so returning it in a per-frame restoration token
    # allocates. Capacity-one ownership makes these retained slots sufficient.
    previous_context::Union{Nothing,CUDA.CuContext}
    previous_stream::CUDA.CuStream
    active::Bool
end

@kernel function _clear_captured_frame_status!(status)
    @inbounds status[1] = Int32(0)
end

function _prepare_captured_frame_status(context::CUDACapturedContext)
    status = CUDA.zeros(Int32, 1)
    backend = KernelAbstractions.get_backend(status)
    return CapturedFrameStatus(
        status,
        _allocate_captured_host_buffer(
            context,
            (element_type=Int32, shape=(1,)),
        ),
        _clear_captured_frame_status!(backend, 1),
    )
end

struct CUDACapturedExecutable{G,E,F,I,B}
    graph::G
    executable::E
    frame_status::F
    host_inputs::I
    feedback::B
end

function _prepare_captured_context(device::CUDA.CuDevice)
    CUDA.functional() || throw(
        ArgumentError("CUDA is not functional on this host"),
    )
    return CUDA.device!(device) do
        CUDACapturedContext(
            device,
            CUDA.context(device),
            CUDA.CuStream(flags=_CAPTURE_SUPPORTS_STREAM ?
                CUDA.STREAM_NON_BLOCKING : CUDA.STREAM_DEFAULT),
            CUDA.CuArray{UInt8,1,CUDA.DeviceMemory}(
                undef, CUDA.capability(device) >= v"9.0" ? 32 << 20 : 4 << 20,
            ),
            nothing,
            CUDA.stream(),
            false,
        )
    end
end

function FilterGraphs._with_captured_workspace(f, context::CUDACapturedContext,
        workspace=context.workspace)
    # Obtain the task-local handle after activating the retained stream: CUDA's
    # handle cache resets workspace when it updates the handle's stream.
    handle = CUDA.CUBLAS.handle()
    UInt(pointer(workspace)) % 256 == 0 ||
        error("captured cuBLAS workspace must be 256-byte aligned")
    GC.@preserve workspace begin
        CUDA.CUBLAS.cublasSetWorkspace_v2(handle, pointer(workspace), sizeof(workspace))
        try
            # A conditional cuBLAS body captures on its own stream. Carry the
            # same workspace lease through the stream change and restoration.
            ext = Base.get_extension(FilterGraphs.FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
            if ext !== nothing && isdefined(ext, :_with_cuda_conditional_workspace)
                return ext._with_cuda_conditional_workspace(f, workspace)
            end
            return f()
        finally
            # SetStream restores default workspace even for the same stream.
            # Keep CUDA's task-local stream cache consistent and leave other
            # callers free to use its library-owned workspace after preparation.
            CUDA.CUBLAS.cublasSetStream_v2(handle, context.stream)
        end
    end
end

function _require_cuda_graph_fixed_storage(graph)
    count = Ref{Csize_t}(0)
    CUDA.CUDACore.cuGraphGetNodes(graph, C_NULL, count)
    iszero(count[]) && return graph
    nodes = Vector{CUDA.CUDACore.CUgraphNode}(undef, count[])
    CUDA.CUDACore.cuGraphGetNodes(graph, nodes, count)
    for node in nodes
        kind = Ref{CUDA.CUDACore.CUgraphNodeType}()
        CUDA.CUDACore.cuGraphNodeGetType(node, kind)
        if kind[] == CUDA.CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC ||
                kind[] == CUDA.CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE
            throw(FilterGraphs.GraphConfigurationError(
                "captured CUDA graph contains a device allocation or free node",
            ))
        elseif kind[] == CUDA.CUDACore.CU_GRAPH_NODE_TYPE_GRAPH
            child = Ref{CUDA.CUDACore.CUgraph}(C_NULL)
            CUDA.CUDACore.cuGraphChildGraphNodeGetGraph(node, child)
            _require_cuda_graph_fixed_storage(child[])
        end
    end
    return graph
end

function _capture_cuda_graph(f)
    graph = _CAPTURE_SUPPORTS_STREAM ?
        CUDA.capture(f; stream=CUDA.stream()) : CUDA.capture(f)
    return _require_cuda_graph_fixed_storage(graph)
end

function _instantiate_cuda_graph(context::CUDACapturedContext, graph)
    executable = CUDA.instantiate(graph)
    GC.@preserve context graph executable begin
        try
            # Prepare GPU graph resources without executing stateful Algorithms.
            CUDA.CUDACore.cuGraphUpload(executable, context.stream)
            _synchronize_captured_graph!(context)
        catch
            # Upload can remain in flight after an API failure. Retain the
            # executable and its graph if completing the stream also fails.
            FilterGraphs._drain_captured_storage!(context, (context, graph, executable))
            rethrow()
        end
    end
    return executable
end

@inline function _activate_captured_context(
    context::CUDACapturedContext,
)
    context.active && error("captured CUDA context is already active")
    CUDA.CUDACore.isvalid(context.runtime_context) || throw(
        ArgumentError("the prepared CUDA context is no longer valid"),
    )
    context.previous_context = CUDA.context!(context.runtime_context)
    context.previous_stream = CUDA.stream()
    CUDA.stream!(context.stream)
    context.active = true
    return nothing
end

@inline function _restore_captured_context!(
    context::CUDACapturedContext,
    ::Nothing,
)
    context.active || error("captured CUDA context is not active")
    CUDA.stream!(context.previous_stream)
    previous_context = context.previous_context
    if previous_context !== nothing &&
            previous_context != context.runtime_context &&
            CUDA.CUDACore.isvalid(previous_context)
        CUDA.context!(previous_context)
    end
    context.previous_context = nothing
    context.active = false
    return nothing
end

function _allocate_captured_buffer(context::CUDACapturedContext, format)
    return CUDA.zeros(format.element_type, format.shape)
end

function _allocate_captured_host_buffer(context::CUDACapturedContext, format)
    buffer = zeros(format.element_type, format.shape)
    CUDA.pin(buffer)
    return buffer
end

@inline function _copy_to_captured!(
    context::CUDACapturedContext,
    destination,
    source,
)
    copyto!(destination, source)
    return nothing
end

function _copy_to_captured!(
    context::CUDACapturedContext, destination::CUDA.CuArray{T}, source::Array{T},
) where {T}
    isempty(source) && return nothing
    if isbitstype(T) && CUDA.context(destination) == context.runtime_context
        # The frame keeps admitted inputs and staging alive through completion
        # or drain. Avoid the generic pinned-source copy's intermediate wait.
        GC.@preserve destination source begin
            CUDA.CUDACore.cuMemcpyHtoDAsync_v2(
                pointer(destination), pointer(source), sizeof(source), context.stream,
            )
        end
    else
        copyto!(destination, source)
    end
    return nothing
end

function _copy_to_captured!(
    context::CUDACapturedContext, destination::CUDA.CuArray{T}, source::CUDA.CuArray{T},
) where {T}
    isempty(source) && return nothing
    if isbitstype(T) && CUDA.context(source) == context.runtime_context
        # Fixed destination storage belongs to this context. Avoid the generic
        # pointer copy's two allocating device queries for a same-context copy.
        GC.@preserve destination source begin
            CUDA.CUDACore.cuMemcpyDtoDAsync_v2(
                pointer(destination), pointer(source), sizeof(source), context.stream,
            )
        end
    else
        copyto!(destination, source)
    end
    return nothing
end

@inline function _copy_from_captured!(
    context::CUDACapturedContext,
    destination,
    source,
)
    GC.@preserve destination source begin
        CUDA.unsafe_copyto!(
            pointer(destination),
            pointer(source),
            length(destination);
            stream=context.stream,
            async=true,
        )
    end
    return nothing
end

function _record_captured_graph(context::CUDACapturedContext, frame_status, nodes::Tuple)
    return _record_captured_graph(context, frame_status, nodes, ())
end

function _record_captured_graph(context::CUDACapturedContext, frame_status, nodes::Tuple, feedback::Tuple)
    graph = _capture_cuda_graph() do
        _enqueue_captured_frame!(frame_status, nodes)
        FilterGraphs._enqueue_captured_feedback!(feedback)
    end
    outcome = CapturedFrameOutcome(frame_status.value, frame_status.host)
    return CUDACapturedExecutable(graph, _instantiate_cuda_graph(context, graph), outcome, nothing, feedback)
end

@inline function _launch_captured_graph!(
    context::CUDACapturedContext,
    captured::CUDACapturedExecutable,
)
    CUDA.launch(captured.executable, context.stream)
    return nothing
end

@inline function _synchronize_captured_graph!(context::CUDACapturedContext)
    CUDA.synchronize(context.stream; blocking=true)
    return nothing
end

struct CUDACapturedPipelineBackend{I,O,S}
    input_stream::CUDA.CuStream
    output_stream::CUDA.CuStream
    shared_inputs::I
    shared_outputs::O
    status::S
end

struct CUDACapturedPipelineEvents{I,O}
    input::CUDA.CuEvent
    graph::CUDA.CuEvent
    output::CUDA.CuEvent
    device_inputs::I
    device_outputs::O
end

function _prepare_captured_pipeline_backend(
    graph::FilterGraphs.PreparedCapturedGraph{<:CUDACapturedContext},
)
    isempty(graph.executable.feedback) || throw(FilterGraphs.GraphConfigurationError(
        "captured pipelines are not qualified for resident feedback",
    ))
    return CUDACapturedPipelineBackend(
        CUDA.CuStream(),
        CUDA.CuStream(),
        map(pointer, graph.device_inputs),
        map(pointer, graph.device_outputs),
        pointer(graph.executable.frame_status.value),
    )
end

function _prepare_captured_pipeline_events(
    ::CUDACapturedContext,
    device_inputs,
    device_outputs,
)
    return CUDACapturedPipelineEvents(
        CUDA.CuEvent(),
        CUDA.CuEvent(),
        CUDA.CuEvent(),
        map(pointer, device_inputs),
        map(pointer, device_outputs),
    )
end

function _enqueue_captured_pipeline_input!(
    backend::CUDACapturedPipelineBackend,
    context::CUDACapturedContext,
    slot,
)
    CUDA.stream!(backend.input_stream)
    try
        GC.@preserve slot begin
            for (device, host) in zip(
                values(slot.events.device_inputs),
                values(slot.host_inputs),
            )
                CUDA.unsafe_copyto!(
                    device,
                    pointer(host),
                    length(host);
                    stream=backend.input_stream,
                    async=true,
                )
            end
        end
        # These events are private DMA ordering markers. Native records avoid
        # allocating SDK epoch metadata; the matching stream waits stay intact.
        CUDA.CUDACore.cuEventRecord(slot.events.input, backend.input_stream)
    finally
        CUDA.stream!(context.stream)
    end
    return nothing
end

_captured_pipeline_input_done(events::CUDACapturedPipelineEvents) =
    CUDA.isdone(events.input)

function _enqueue_captured_pipeline_graph!(
    backend::CUDACapturedPipelineBackend,
    context::CUDACapturedContext,
    graph,
    slot,
)
    CUDA.wait(slot.events.input, context.stream)
    # All buffers were allocated in this retained context during preparation.
    # The generic pointer copy queries both devices on each call, allocating
    # two RefValues; this same-context copy needs no peer-device discovery.
    GC.@preserve graph slot begin
        for (destination, source, array) in zip(
            values(backend.shared_inputs),
            values(slot.events.device_inputs),
            values(graph.device_inputs),
        )
            CUDA.CUDACore.cuMemcpyDtoDAsync_v2(
                destination,
                source,
                sizeof(array),
                context.stream,
            )
        end
        _launch_captured_graph!(context, graph.executable)
        for (destination, source, array) in zip(
            values(slot.events.device_outputs),
            values(backend.shared_outputs),
            values(graph.device_outputs),
        )
            CUDA.CUDACore.cuMemcpyDtoDAsync_v2(
                destination,
                source,
                sizeof(array),
                context.stream,
            )
        end
        CUDA.unsafe_copyto!(
            pointer(slot.host_status),
            backend.status,
            1;
            stream=context.stream,
            async=true,
        )
    end
    CUDA.CUDACore.cuEventRecord(slot.events.graph, context.stream)
    return nothing
end

_captured_pipeline_graph_done(events::CUDACapturedPipelineEvents) =
    CUDA.isdone(events.graph)

function _enqueue_captured_pipeline_output!(
    backend::CUDACapturedPipelineBackend,
    context::CUDACapturedContext,
    slot,
)
    CUDA.stream!(backend.output_stream)
    try
        CUDA.wait(slot.events.graph, backend.output_stream)
        GC.@preserve slot begin
            for (host, device) in zip(
                values(slot.host_outputs),
                values(slot.events.device_outputs),
            )
                CUDA.unsafe_copyto!(
                    pointer(host),
                    device,
                    length(host);
                    stream=backend.output_stream,
                    async=true,
                )
            end
        end
        CUDA.CUDACore.cuEventRecord(slot.events.output, backend.output_stream)
    finally
        CUDA.stream!(context.stream)
    end
    return nothing
end

_captured_pipeline_output_done(events::CUDACapturedPipelineEvents) =
    CUDA.isdone(events.output)

function _drain_captured_pipeline_backend!(
    backend::CUDACapturedPipelineBackend,
    context::CUDACapturedContext,
)
    failure = nothing
    for stream in (backend.input_stream, context.stream, backend.output_stream)
        try
            CUDA.synchronize(stream; blocking=true)
        catch error
            isnothing(failure) && (failure = error)
        end
    end
    isnothing(failure) || throw(failure)
    return nothing
end

include("FilterGraphsCUDAExt/captured_inputs.jl")
include("FilterGraphsCUDAExt/captured_feedback.jl")
include("FilterGraphsCUDAExt/captured_layouts.jl")
include("FilterGraphsCUDAExt/captured_bounds.jl")
include("FilterGraphsCUDAExt/captured_dependencies.jl")
include("FilterGraphsCUDAExt/progressive_dense.jl")
include("FilterGraphsCUDAExt/readout_wfs.jl")
include("FilterGraphsCUDAExt/readout_reconstruction.jl")
include("FilterGraphsCUDAExt/readout_graph.jl")
include("FilterGraphsCUDAExt/controller_graph.jl")
include("FilterGraphsCUDAExt/composition.jl")
include("captured_scc_pixels.jl")

end
