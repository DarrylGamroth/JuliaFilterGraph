function FilterGraphs._prepare_captured_inputs(
    context::CUDACapturedContext, captured::CUDACapturedExecutable, device_inputs,
)
    host_inputs = map(device_inputs) do device
        _allocate_captured_host_buffer(
            context, (element_type=eltype(device), shape=size(device)),
        )
    end
    return CUDACapturedExecutable(
        captured.graph, captured.executable, captured.frame_status, host_inputs, captured.feedback,
    )
end

# CPU views need packing; GPU views retain their existing device copy protocol.
_captured_host_parent(::AbstractArray) = false
_captured_host_parent(::Array) = true
_captured_host_parent(input::SubArray) = _captured_host_parent(parent(input))

@inline _copy_to_captured!(context::CUDACapturedContext, device, input, staging) =
    _copy_to_captured!(context, device, input)

@inline function _copy_to_captured!(
    context::CUDACapturedContext, device, input::SubArray, staging,
)
    if _captured_host_parent(input)
        copyto!(staging, input)
        _copy_to_captured!(context, device, staging)
    else
        _copy_to_captured!(context, device, input)
    end
    return nothing
end

@inline FilterGraphs._copy_to_captured_inputs!(
    ::CUDACapturedContext, ::Tuple{}, ::Tuple{}, ::Tuple{},
) = nothing

@inline function FilterGraphs._copy_to_captured_inputs!(
    context::CUDACapturedContext, device::Tuple, inputs::Tuple, staging::Tuple,
)
    destination, input = first(device), first(inputs)
    destination === input || _copy_to_captured!(context, destination, input, first(staging))
    return FilterGraphs._copy_to_captured_inputs!(
        context, Base.tail(device), Base.tail(inputs), Base.tail(staging),
    )
end

@noinline function FilterGraphs._transfer_captured_inputs!(
    context::CUDACapturedContext, device_inputs, inputs, host_inputs,
)
    FilterGraphs._copy_to_captured_inputs!(
        context, values(device_inputs), values(inputs), values(host_inputs),
    )
    return nothing
end

@inline function FilterGraphs._execute_captured_frame!(
    graph::FilterGraphs.PreparedCapturedGraph{<:CUDACapturedContext}, inputs, host_outputs,
)
    GC.@preserve graph inputs begin
        FilterGraphs._transfer_captured_inputs!(
            graph.context, graph.device_inputs, inputs, graph.executable.host_inputs,
        )
        status = FilterGraphs._launch_and_complete_captured_frame!(
            graph.context, graph.executable, host_outputs, graph.device_outputs,
        )
        return FilterGraphs._complete_captured_nodes!(graph.nodes, status, graph.rejection)
    end
end
