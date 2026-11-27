# The process callback is the only writer. Reset is called only while native
# processing is stopped. Control readers inspect only the atomic failure latch;
# the caller owns publication of any additional observer state.
mutable struct FrameObserver{C}
    const callback::C
    const failed::Threads.Atomic{Bool}
    update_sequence::UInt64
    notified_sequence::UInt64
end

_owner_frame_observer(owner) = nothing
_frame_observer(::Nothing) = nothing
_frame_observer(callback) = FrameObserver(callback, Threads.Atomic{Bool}(false), UInt64(0), UInt64(0))
@inline _frame_observer_failed(::Nothing) = false
@inline _frame_observer_failed(observer::FrameObserver) = observer.failed[]

@inline _frame_observer_event!(::Nothing, phase) = nothing
@inline function _frame_observer_event!(observer::FrameObserver, phase)
    observer.failed[] && return nothing
    observer.update_sequence += UInt64(1)
    try
        observer.callback(phase)
    catch
        observer.failed[] = true
    end
    return nothing
end

@inline _readonly_frame_array(array::BorrowedArray{T,N}) where {T,N} =
    BorrowedArray{T,N,false}(array.data, array.shape, array.length)
@inline _readonly_frame_array(array::RowMajorBorrowedArray{T,N}) where {T,N} =
    RowMajorBorrowedArray{T,N,false}(array.data, array.shape, array.length)

@inline _observe_frame!(::Nothing, owner, buffers) = nothing
@inline function _observe_frame!(observer::FrameObserver, owner, buffers)
    observer.failed[] && return nothing
    observer.update_sequence += UInt64(1)
    try
        bindings = _public_output_bindings(owner)
        outputs = map(_readonly_frame_array, _borrow_buffers(bindings, buffers))
        admitted = FilterGraphs._output_metadata_admitted(owner.graph)
        metadata = NamedTuple{keys(bindings)}(map(name -> getproperty(admitted, name), keys(bindings)))
        observer.callback(Val(:frame), outputs, metadata)
    catch
        # Optional observation cannot reject accepted science. Never retry or
        # retain the exception/borrowed storage, and never log from this path.
        observer.failed[] = true
    end
    return nothing
end

const _FRAME_OBSERVER_STATE = "pipewireao.frame-observer.state"
_frame_observer_property_info(owner) = PipeWireAO.SPA.PropInfo[]
_frame_observer_property_info(::Nothing) = PipeWireAO.SPA.PropInfo[]
function _frame_observer_property_info(::FrameObserver)
    return [PipeWireAO.SPA.PropInfo(_FRAME_OBSERVER_STATE, PipeWireAO.Pod("active");
        description="Optional frame observer state; failure is retained for this owner",
        display_name="Frame observer", group="Frame observation", params=true,
        flags=PipeWireAO.SPA.PROPERTY_READONLY)]
end

_append_frame_observer_properties!(owner, properties) = nothing
_append_frame_observer_properties!(::Nothing, properties) = nothing
function _append_frame_observer_properties!(observer::FrameObserver, properties)
    push!(properties.values, _FRAME_OBSERVER_STATE => PipeWireAO.Pod(
        observer.failed[] ? "failed" : "active"))
    return nothing
end

# These counters coalesce wakeups only; they are never sample identity/evidence.
@inline _frame_observer_pending(::Nothing) = false
@inline _frame_observer_pending(observer::FrameObserver) =
    observer.update_sequence != observer.notified_sequence
@inline _frame_observer_notified!(::Nothing) = nothing
@inline function _frame_observer_notified!(observer::FrameObserver)
    observer.notified_sequence = observer.update_sequence
    return nothing
end
