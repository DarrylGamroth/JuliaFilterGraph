module FilterGraphsAMDGPUExt

using AMDGPU
using FilterGraphs
using KernelAbstractions

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
    _prepare_captured_context,
    _prepare_captured_frame_status,
    _record_captured_graph,
    _restore_captured_context!,
    _synchronize_captured_graph!

mutable struct HIPCapturedHostPins
    device::AMDGPU.HIPDevice
    buffers::Vector{Any}
end

function _release_captured_host_pins!(pins::HIPCapturedHostPins)
    AMDGPU.device!(pins.device) do
        for buffer in pins.buffers
            GC.@preserve buffer AMDGPU.Mem.unpin(pointer(buffer))
        end
    end
    empty!(pins.buffers)
    return nothing
end

mutable struct HIPCapturedContext
    device::AMDGPU.HIPDevice
    stream::AMDGPU.HIPStream
    host_pins::HIPCapturedHostPins
end

@kernel function _clear_captured_frame_status!(status)
    @inbounds status[1] = Int32(0)
end

function _prepare_captured_frame_status(context::HIPCapturedContext)
    status = AMDGPU.zeros(Int32, 1)
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

@kernel function _commit_captured_feedback!(destination, source, status)
    index = @index(Global, Linear)
    if index <= length(destination) && @inbounds(iszero(status[1]))
        @inbounds destination[index] = source[index]
    end
end

struct HIPCapturedFeedback{D,S,F,K}
    destination::D
    source::S
    frame_status::F
    commit!::K
end

FilterGraphs._require_captured_feedback_support(::AMDGPU.HIPDevice, feedback) = nothing

function FilterGraphs._prepare_captured_feedback(
    ::AMDGPU.HIPDevice, feedback, device_inputs, device_outputs, frame_status,
)
    return map(keys(feedback)) do input
        destination = device_inputs[input]
        source = device_outputs[feedback[input]]
        commit! = _commit_captured_feedback!(KernelAbstractions.get_backend(destination), 256)
        HIPCapturedFeedback(destination, source, frame_status.value, commit!)
    end
end

function FilterGraphs._enqueue_captured_feedback!(feedback::HIPCapturedFeedback)
    feedback.commit!(feedback.destination, feedback.source, feedback.frame_status;
        ndrange=length(feedback.destination))
    return nothing
end

function FilterGraphs._reset_captured_feedback!(feedback::HIPCapturedFeedback)
    fill!(feedback.destination, zero(eltype(feedback.destination)))
    return nothing
end

struct HIPCapturedExecutable{G,E,F,B}
    graph::G
    executable::E
    frame_status::F
    feedback::B
end

FilterGraphs._reset_captured_feedback!(executable::HIPCapturedExecutable) =
    FilterGraphs._reset_captured_feedback!(executable.feedback)

function _prepare_captured_context(device::AMDGPU.HIPDevice)
    AMDGPU.functional() || throw(
        ArgumentError("AMDGPU is not functional on this host"),
    )
    return AMDGPU.device!(device) do
        pins = HIPCapturedHostPins(device, Any[])
        finalizer(_release_captured_host_pins!, pins)
        HIPCapturedContext(device, AMDGPU.HIPStream(), pins)
    end
end

@inline function _activate_captured_context(
    context::HIPCapturedContext,
)
    # Explicit restoration lets the frame closure inline without the nested
    # device!/stream! do-block allocations in AMDGPU 2.8.
    old_device = AMDGPU.device()
    switched_device = old_device != context.device
    switched_device && AMDGPU.device!(context.device)
    old_stream = AMDGPU.stream()
    AMDGPU.stream!(context.stream)
    return (; old_device, old_stream, switched_device)
end

@inline function _restore_captured_context!(
    context::HIPCapturedContext,
    previous,
)
    AMDGPU.stream!(previous.old_stream)
    previous.switched_device && AMDGPU.device!(previous.old_device)
    return nothing
end

function _allocate_captured_buffer(context::HIPCapturedContext, format)
    return AMDGPU.zeros(format.element_type, format.shape)
end

function _allocate_captured_host_buffer(context::HIPCapturedContext, format)
    buffer = zeros(format.element_type, format.shape)
    GC.@preserve buffer begin
        host_pointer = pointer(buffer)
        AMDGPU.Mem.pin(host_pointer, sizeof(buffer))
        try
            push!(context.host_pins.buffers, buffer)
        catch
            AMDGPU.Mem.unpin(host_pointer)
            rethrow()
        end
    end
    return buffer
end

@inline function _copy_to_captured!(
    context::HIPCapturedContext,
    destination,
    source,
)
    copyto!(destination, source)
    return nothing
end

@inline function _copy_from_captured!(
    context::HIPCapturedContext,
    destination,
    source,
)
    GC.@preserve destination source begin
        copyto!(destination, 1, source, 1, length(destination); async=true)
    end
    return nothing
end

@inline function _copy_from_captured!(
    context::HIPCapturedContext,
    destination::Array{T},
    source::AMDGPU.ROCArray{T},
) where {T}
    # Preparation pins these retained host destinations. Their device sources
    # belong to this owner's stream, and publication follows the final wait.
    # The known direction avoids pointer queries and hipMemcpyWithStream's
    # intermediate host waits; the existing failure drain retains both ends.
    if isbitstype(T)
        isempty(destination) && return nothing
        GC.@preserve destination source begin
            AMDGPU.HIP.hipMemcpyAsync(
                pointer(destination), pointer(source), sizeof(destination),
                AMDGPU.HIP.hipMemcpyDeviceToHost, context.stream,
            )
        end
    else
        GC.@preserve destination source begin
            copyto!(destination, 1, source, 1, length(destination); async=true)
        end
    end
    return nothing
end

function _record_captured_graph(context::HIPCapturedContext, frame_status, nodes::Tuple)
    return _record_captured_graph(context, frame_status, nodes, ())
end

function _record_captured_graph(context::HIPCapturedContext, frame_status, nodes::Tuple, feedback::Tuple)
    graph = try
        AMDGPU.HIP.capture() do
            _enqueue_captured_frame!(frame_status, nodes)
            FilterGraphs._enqueue_captured_feedback!(feedback)
        end
    catch
        # Some ROCm releases retain a sticky stream-capture error. Consume it
        # so a cold preparation failure does not poison explicit recovery.
        AMDGPU.HIP.clear_last_error()
        rethrow()
    end
    outcome = CapturedFrameOutcome(frame_status.value, frame_status.host)
    return HIPCapturedExecutable(graph, AMDGPU.HIP.instantiate(graph), outcome, feedback)
end

@inline function _launch_captured_graph!(
    context::HIPCapturedContext,
    captured::HIPCapturedExecutable,
)
    AMDGPU.HIP.launch(captured.executable, context.stream)
    return nothing
end

@inline function _synchronize_captured_graph!(context::HIPCapturedContext)
    AMDGPU.synchronize(context.stream; blocking=true)
    return nothing
end

const FG = FilterGraphs
const CapturedContext = HIPCapturedContext
const algorithm_extension = :FilterGraphAlgorithmsAMDGPUExt
include("captured_composition.jl")
include("captured_scc_pixels.jl")

function FG._prepare_captured_validation(
    ::AMDGPU.HIPDevice, nodes, routes, outputs, device_inputs, node_outputs,
)
    return _reuse_scc_residual_validation(nodes, eachindex(nodes), routes, outputs, device_inputs, node_outputs)
end

FG._prepare_captured_fusion(::AMDGPU.HIPDevice, nodes, routes) =
    _fuse_captured_scc_pixels(nodes, routes)

"""Capture each stage of a fresh prepared composition on AMDGPU.

Bind calibration before capture. Camera admission remains on the host, and each
camera stage and the shared DM stage run as native HIP graphs.
"""
function FG.prepare_captured_graph(composition::FG.PreparedComposition, device::AMDGPU.HIPDevice)
    extension = Base.get_extension(FGA, :FilterGraphAlgorithmsAMDGPUExt)
    return _prepare_captured_composition(composition, device, extension, FG._prepare_captured_node)
end

end
