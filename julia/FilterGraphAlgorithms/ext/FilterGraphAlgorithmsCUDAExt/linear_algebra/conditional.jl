using CUDA: CUDACore
const _cuda_conditional_workspace =
    Base.ScopedValues.ScopedValue{Union{Nothing,CUDA.CuArray}}(nothing)

function _with_cuda_conditional_workspace(f, workspace)
    return Base.ScopedValues.with(f, _cuda_conditional_workspace => workspace)
end

_supports_cuda_conditional() = pkgversion(CUDA) >= v"6.5.1" &&
    CUDA.driver_version() >= v"12.4" && CUDA.runtime_version() >= v"12.4"

function cuda_conditional_handle()
    _supports_cuda_conditional() || return nothing
    stream = CUDA.stream()
    CUDACore.current_capture(stream) === nothing && return nothing
    status = Ref{CUDACore.CUstreamCaptureStatus}()
    graph = Ref{CUDACore.CUgraph}(C_NULL)
    CUDACore.cuStreamGetCaptureInfo_v3(
        stream, status, C_NULL, graph, C_NULL, C_NULL, C_NULL
    )
    status[] == CUDACore.CU_STREAM_CAPTURE_STATUS_NONE && return nothing
    status[] == CUDACore.CU_STREAM_CAPTURE_STATUS_ACTIVE ||
        throw(CUDA.CaptureError("CUDA conditional requires an active stream capture"))

    handle = Ref{CUDACore.CUgraphConditionalHandle}()
    result = CUDACore.unchecked_cuGraphConditionalHandleCreate(
        handle, graph[], CUDA.context(), 0, CUDACore.CU_GRAPH_COND_ASSIGN_DEFAULT
    )
    result == CUDACore.CUDA_ERROR_NOT_SUPPORTED && return nothing
    result == CUDACore.CUDA_SUCCESS || CUDACore.throw_api_error(result)
    return handle[]
end

@inline function set_cuda_conditional!(handle::UInt64, value::Cuint)
    ccall("extern cudaGraphSetConditional", llvmcall, Cvoid,
        (UInt64, Cuint), handle, value)
    return nothing
end

function _set_cuda_conditional!(handle, fallback)
    @inbounds set_cuda_conditional!(handle, Cuint(!iszero(fallback[1])))
    return nothing
end

function enqueue_cuda_conditional!(handle::UInt64, fallback::CUDA.CuVector{Int32})
    CUDA.@cuda threads=1 _set_cuda_conditional!(handle, fallback)
    return nothing
end

function _bind_cuda_conditional_workspace!(workspace)
    handle = CUDA.CUBLAS.handle()
    GC.@preserve workspace CUDA.CUBLAS.cublasSetWorkspace_v2(
        handle, pointer(workspace), sizeof(workspace)
    )
    return nothing
end

function capture_cuda_if!(f, handle::UInt64)
    stream = CUDA.stream()
    ctx = CUDA.context()
    parent = CUDACore.current_capture(stream)
    parent === nothing &&
        throw(CUDA.CaptureError("CUDA conditional requires a CUDA.jl capture"))

    status = Ref{CUDACore.CUstreamCaptureStatus}()
    graph = Ref{CUDACore.CUgraph}(C_NULL)
    dependencies = Ref{Ptr{CUDACore.CUgraphNode}}()
    edges = Ref{Ptr{CUDACore.CUgraphEdgeData}}()
    count = Ref{Csize_t}(0)
    node = Ref{CUDACore.CUgraphNode}()
    params = Ref(CUDACore.CUgraphNodeParams(
        ntuple(Returns(UInt8(0)), Val(sizeof(CUDACore.CUgraphNodeParams)))
    ))
    GC.@preserve params begin
        ptr = Base.unsafe_convert(Ptr{CUDACore.CUgraphNodeParams}, params)
        ptr.type = CUDACore.CU_GRAPH_NODE_TYPE_CONDITIONAL
        ptr.conditional = CUDACore.CUDA_CONDITIONAL_NODE_PARAMS(
            handle, CUDACore.CU_GRAPH_COND_TYPE_IF, Cuint(1), C_NULL, ctx.handle
        )
    end
    CUDACore.cuStreamGetCaptureInfo_v3(
        stream, status, C_NULL, graph, dependencies, edges, count
    )
    status[] == CUDACore.CU_STREAM_CAPTURE_STATUS_ACTIVE ||
        throw(CUDA.CaptureError("CUDA conditional requires an active stream capture"))
    CUDACore.cuGraphAddNode_v2(
        node, graph[], dependencies[], edges[], count[], params
    )

    # CUDA owns the conditional body handle. CUDA.jl puts its operations on a
    # separate capture stream and collects their memory leases in this wrapper.
    body = unsafe_wrap(CUDA.CuGraph, unsafe_load(params[].conditional.phGraph_out); ctx)
    workspace = _cuda_conditional_workspace[]
    try
        CUDACore.capture_stream(
            body, CUDACore.CUgraphNode[];
            mode=CUDACore.STREAM_CAPTURE_MODE_RELAXED, throw_error=true
        ) do
            workspace === nothing || _bind_cuda_conditional_workspace!(workspace)
            f()
        end
    finally
        for managed in body.memory
            CUDACore.record!(parent, managed)
            CUDACore.unlease!(managed)
        end
        # cuBLAS resets the workspace when the task-local handle changes streams.
        workspace === nothing || _bind_cuda_conditional_workspace!(workspace)
    end
    # Subsequent parent operations wait for the IF node, including its body.
    CUDACore.cuStreamUpdateCaptureDependencies_v2(
        stream, node, C_NULL, 1, CUDACore.CU_STREAM_SET_CAPTURE_DEPENDENCIES
    )
    return nothing
end
