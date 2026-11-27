@enum CapturedPipelineSlotState::UInt8 begin
    PipelineAvailable = 1
    PipelineInputPending = 2
    PipelineFrameReady = 3
    PipelineGraphRunning = 4
    PipelineOutputPending = 5
    PipelineCommandReady = 6
    PipelineFailureReady = 7
    PipelineQuarantined = 8
end

mutable struct CapturedPipelineSlot{HI,DI,DO,HO,HS,E,M,R,RS}
    host_inputs::HI
    device_inputs::DI
    device_outputs::DO
    host_outputs::HO
    host_status::HS
    events::E
    state::CapturedPipelineSlotState
    generation::UInt64
    sequence::UInt64
    metadata::M
    result::Union{Nothing,R}
    rejection::RS
end

function _captured_pipeline_slot(
    graph::PreparedCapturedGraph, host_inputs, device_inputs, device_outputs,
    host_outputs, host_status, events,
)
    initial_metadata = _captured_output_metadata_admitted(
        graph, map(FilterGraphAlgorithms._admitted_sample_metadata, graph.default_input_metadata),
    )
    return _captured_pipeline_slot(
        graph.result_types, _rejection_state(graph.result_types, graph.input_formats),
        host_inputs, device_inputs, device_outputs, host_outputs, host_status, events, initial_metadata,
    )
end

function _captured_pipeline_slot(
    ::GraphResultTypes{Availability,Failure}, rejection::RS, host_inputs::HI, device_inputs::DI,
    device_outputs::DO, host_outputs::HO, host_status::HS, events::E,
    metadata::M,
) where {Availability,Failure,HI,DI,DO,HO,HS,E,M,RS}
    return CapturedPipelineSlot{HI,DI,DO,HO,HS,E,M,Union{Availability,Failure},RS}(
        host_inputs, device_inputs, device_outputs, host_outputs, host_status,
        events, PipelineAvailable, UInt64(0), UInt64(0), metadata, nothing, rejection,
    )
end

"""
One host-controlled, fixed-capacity captured Graph pipeline.

The retained captured Graph is private to this owner. All calls that mutate
the pipeline are serialized by its lock. Backend streams may continue to work
after those calls return, but no slot is reused until its final output event
and sink copy have completed. The scientific Graph executes in sample order.
"""
mutable struct PreparedCapturedPipeline{G,B,S}
    graph::G
    backend::B
    slots::Vector{S}
    lock::ReentrantLock
    graph_slot::Int
    last_sequence::UInt64
    has_sequence::Bool
    drops::UInt64
    completions::UInt64
    fault::Union{Nothing,CapturedGraphFault}
    closed::Bool
end

# Raw CUDA copies bypass allocation stream tracking. Keep every live owner
# reachable until an explicit close has drained all streams. An abandoned
# owner retains storage instead of allowing pending DMA to access freed memory.
const _captured_pipeline_roots = IdDict{Any,Nothing}()
const _captured_pipeline_roots_lock = ReentrantLock()

_prepare_captured_pipeline_backend(graph) = throw(
    GraphConfigurationError(
        "captured frame pipelining is not qualified for $(typeof(graph.context))",
    ),
)
function _prepare_captured_pipeline_events end
function _enqueue_captured_pipeline_input! end
function _captured_pipeline_input_done end
function _enqueue_captured_pipeline_graph! end
function _captured_pipeline_graph_done end
function _enqueue_captured_pipeline_output! end
function _captured_pipeline_output_done end
function _drain_captured_pipeline_backend! end

"""
    prepare_captured_pipeline(graph, device; capacity=2, workgroups=CapturedWorkgroupProfile())

Prepare a bounded host-staged captured pipeline on a qualified backend. The
current backend implementation is CUDA. Its two or more slots retain pinned
host input/output storage, device input/output snapshots, and completion events.
The scientific Graph and its stateful Algorithm adapters are prepared once.
"""
function prepare_captured_pipeline(
    graph::PreparedGraph,
    device;
    capacity::Integer=2,
    workgroups::CapturedWorkgroupProfile=CapturedWorkgroupProfile(),
)
    count = Int(capacity)
    count >= 2 || throw(ArgumentError("captured pipeline needs at least two slots"))
    captured = prepare_captured_graph(graph, device; workgroups)
    isnothing(captured.camera_metadata) || throw(GraphConfigurationError(
        "captured pipeline does not support host-admitted chopped-image metadata"))
    return _with_captured_context(captured.context) do
        backend = _prepare_captured_pipeline_backend(captured)
        slots = map(1:count) do _
            host_inputs = NamedTuple{keys(captured.input_formats)}(
                map(
                    format -> _allocate_captured_host_buffer(captured.context, format),
                    values(captured.input_formats),
                ),
            )
            device_inputs = _captured_external_buffers(
                captured.context,
                captured.input_formats,
            )
            device_outputs = _captured_external_buffers(
                captured.context,
                captured.output_formats,
            )
            host_outputs = NamedTuple{keys(captured.output_formats)}(
                map(
                    format -> _allocate_captured_host_buffer(captured.context, format),
                    values(captured.output_formats),
                ),
            )
            host_status = _allocate_captured_host_buffer(
                captured.context,
                (element_type=Int32, shape=(1,)),
            )
            _captured_pipeline_slot(
                captured,
                host_inputs,
                device_inputs,
                device_outputs,
                host_outputs,
                host_status,
                _prepare_captured_pipeline_events(
                    captured.context,
                    device_inputs,
                    device_outputs,
                ),
            )
        end
        # Slot allocations and zeroing were enqueued on the retained compute
        # stream. The CUDA backend uses cached raw pointers on its input/output
        # streams, so preparation must finish before the first cross-stream use.
        _synchronize_captured_graph!(captured.context)
        # Ordinary process! and reset! on the retained Graph must reject while
        # the pipeline owns its state. Only the pipeline may launch or reset it.
        captured.state = CapturedGraphRunning
        pipeline = PreparedCapturedPipeline(
            captured,
            backend,
            slots,
            ReentrantLock(),
            0,
            UInt64(0),
            false,
            UInt64(0),
            UInt64(0),
            nothing,
            false,
        )
        lock(_captured_pipeline_roots_lock) do
            _captured_pipeline_roots[pipeline] = nothing
        end
        return pipeline
    end
end

@inline function _require_captured_pipeline!(pipeline::PreparedCapturedPipeline)
    pipeline.closed && throw(InvalidStateException("captured pipeline is closed", :closed))
    isnothing(pipeline.fault) || throw(pipeline.fault)
    return nothing
end

function _fault_captured_pipeline!(pipeline::PreparedCapturedPipeline, phase::Symbol, cause)
    fault = something(pipeline.fault, CapturedGraphFault(phase, cause))
    pipeline.fault = fault
    _fault_captured_graph!(pipeline.graph, phase, cause)
    for slot in pipeline.slots
        slot.state === PipelineAvailable || (slot.state = PipelineQuarantined)
    end
    return fault
end

@inline function _copy_captured_pipeline_inputs!(host_inputs, inputs)
    for (destination, source) in zip(values(host_inputs), values(inputs))
        copyto!(destination, source)
    end
    return nothing
end

"""
    submit_captured_frame!(pipeline, inputs, metadata, sequence)

Copy borrowed inputs into a retained slot before returning and enqueue their
host-to-device transfer. Return `false` when the finite ring is full (drop
newest), otherwise `true`. Sequence numbers must increase even across drops.
The caller must treat a dropped sequence as a scientific discontinuity; the
current captured Algorithms do not consume sample-gap metadata. Inputs and
metadata require exactly the public input Port names, in any field order.
"""
function submit_captured_frame!(
    pipeline::PreparedCapturedPipeline,
    inputs::NamedTuple,
    metadata::NamedTuple,
    sequence::Integer,
)
    lock(pipeline.lock)
    try
        _require_captured_pipeline!(pipeline)
        0 <= sequence <= typemax(UInt64) || throw(
            ArgumentError("captured frame sequence must fit UInt64"),
        )
        sample = UInt64(sequence)
        (!pipeline.has_sequence || sample > pipeline.last_sequence) || throw(
            ArgumentError("captured frame sequence must increase"),
        )
        inputs = _validate_external_arrays(:input, pipeline.graph.input_formats, inputs)
        metadata = _validate_external_metadata(pipeline.graph.input_formats, metadata)
        pipeline.last_sequence = sample
        pipeline.has_sequence = true
        index = findfirst(slot -> slot.state === PipelineAvailable, pipeline.slots)
        if isnothing(index)
            pipeline.drops += UInt64(1)
            return false
        end
        slot = pipeline.slots[index]
        slot.generation == typemax(UInt64) && throw(
            OverflowError("captured pipeline slot generation exhausted"),
        )
        slot.generation += UInt64(1)
        slot.sequence = sample
        slot.metadata = _captured_output_metadata_admitted(
            pipeline.graph, map(FilterGraphAlgorithms._admitted_sample_metadata, metadata),
        )
        slot.result = nothing
        _clear_rejection!(slot.rejection)
        _rejection_inputs!(slot.rejection, metadata)
        slot.state = PipelineInputPending
        try
            _copy_captured_pipeline_inputs!(slot.host_inputs, inputs)
            _with_captured_context(pipeline.graph.context) do
                _enqueue_captured_pipeline_input!(
                    pipeline.backend,
                    pipeline.graph.context,
                    slot,
                )
            end
        catch error
            throw(_fault_captured_pipeline!(pipeline, :input, error))
        end
        return true
    finally
        unlock(pipeline.lock)
    end
end

function submit_captured_frame!(
    pipeline::PreparedCapturedPipeline,
    inputs::NamedTuple,
    sequence::Integer,
)
    return submit_captured_frame!(
        pipeline,
        inputs,
        pipeline.graph.default_input_metadata,
        sequence,
    )
end

function _oldest_captured_pipeline_input(pipeline::PreparedCapturedPipeline)
    index = 0
    sequence = typemax(UInt64)
    for (candidate, slot) in pairs(pipeline.slots)
        slot.state in (PipelineInputPending, PipelineFrameReady) || continue
        if iszero(index) || slot.sequence < sequence
            index = candidate
            sequence = slot.sequence
        end
    end
    return index
end

function _advance_captured_pipeline_graph!(pipeline::PreparedCapturedPipeline)
    index = pipeline.graph_slot
    iszero(index) && return nothing
    slot = pipeline.slots[index]
    _captured_pipeline_graph_done(slot.events) || return nothing
    result = _complete_captured_nodes!(pipeline.graph.nodes, slot.host_status[1], slot.rejection)
    pipeline.graph_slot = 0
    if result !== nothing
        slot.result = result
        slot.state = PipelineFailureReady
        return nothing
    end
    slot.result = pipeline.graph.availability
    _enqueue_captured_pipeline_output!(
        pipeline.backend,
        pipeline.graph.context,
        slot,
    )
    slot.state = PipelineOutputPending
    return nothing
end

function _launch_next_captured_pipeline_frame!(pipeline::PreparedCapturedPipeline)
    iszero(pipeline.graph_slot) || return nothing
    while true
        index = _oldest_captured_pipeline_input(pipeline)
        iszero(index) && return nothing
        slot = pipeline.slots[index]
        slot.state === PipelineFrameReady || return nothing
        result = _preflight_captured_nodes(pipeline.graph.nodes, slot.rejection)
        if result !== nothing
            slot.result = result
            slot.state = PipelineFailureReady
            continue
        end
        _enqueue_captured_pipeline_graph!(
            pipeline.backend,
            pipeline.graph.context,
            pipeline.graph,
            slot,
        )
        slot.state = PipelineGraphRunning
        pipeline.graph_slot = index
        return nothing
    end
end

"""
    advance_captured_pipeline!(pipeline::PreparedCapturedPipeline)

Advance completed GPU events without blocking the host.
"""
function advance_captured_pipeline!(pipeline::PreparedCapturedPipeline)
    lock(pipeline.lock) do
        _require_captured_pipeline!(pipeline)
        try
            _with_captured_context(pipeline.graph.context) do
                for slot in pipeline.slots
                    if slot.state === PipelineInputPending &&
                            _captured_pipeline_input_done(slot.events)
                        slot.state = PipelineFrameReady
                    end
                end
                _advance_captured_pipeline_graph!(pipeline)
                for slot in pipeline.slots
                    if slot.state === PipelineOutputPending &&
                            _captured_pipeline_output_done(slot.events)
                        slot.state = PipelineCommandReady
                    end
                end
                _launch_next_captured_pipeline_frame!(pipeline)
            end
        catch error
            throw(_fault_captured_pipeline!(pipeline, :progress, error))
        end
        return nothing
    end
end

function _oldest_captured_pipeline_result(pipeline::PreparedCapturedPipeline)
    index = 0
    sequence = typemax(UInt64)
    for (candidate, slot) in pairs(pipeline.slots)
        slot.state === PipelineAvailable && continue
        if iszero(index) || slot.sequence < sequence
            index = candidate
            sequence = slot.sequence
        end
    end
    return index
end

@inline function _captured_pipeline_receipt(
    slot::CapturedPipelineSlot{HI,DI,DO,HO,HS,E,NamedTuple{Names,MT},R},
) where {HI,DI,DO,HO,HS,E,Names,MT,R}
    Metadata = NamedTuple{Names,NTuple{length(Names),Union{Nothing,FilterGraphAlgorithms.SampleMetadata}}}
    metadata = slot.state === PipelineCommandReady ?
        Metadata(map(FilterGraphAlgorithms._sample_metadata, slot.metadata)) : nothing
    Receipt = NamedTuple{(:sequence,:result,:metadata),Tuple{UInt64,R,Union{Nothing,Metadata}}}
    return Receipt((slot.sequence, slot.result::R, metadata))
end

_captured_pipeline_receipt(slot, names, ::Val{false}) = _captured_pipeline_receipt(slot)

function _captured_pipeline_receipt(slot, names, ::Val{true})
    receipt = _captured_pipeline_receipt(slot)
    context = _rejection_context(slot.rejection, names)
    return merge(receipt, (; rejection=context))
end

"""
    take_captured_result!(outputs, pipeline; diagnostics=false)

Return `nothing` until the oldest admitted frame has completed. A successful
result copies its retained command into `outputs` before releasing the slot;
an expected failure leaves `outputs` unchanged. The return value carries the
source sequence, ordinary Graph result, and output metadata on success.
With `diagnostics=true`, the receipt additionally contains `rejection`, an
immutable `GraphRejection` for that frame or `nothing`. This opt-in fourth field
does not change default receipt keys or positional destructuring. Diagnostics
are retained per slot and survive delayed consumption and subsequent slot reuse.
"""
Base.@constprop :aggressive function take_captured_result!(
    outputs::NamedTuple, pipeline::PreparedCapturedPipeline; diagnostics::Bool=false,
)
    return _take_captured_result!(outputs, pipeline, Val(diagnostics))
end

function _take_captured_result!(
    outputs::NamedTuple, pipeline::PreparedCapturedPipeline, ::Val{Diagnostics},
) where {Diagnostics}
    lock(pipeline.lock)
    try
        _require_captured_pipeline!(pipeline)
        index = _oldest_captured_pipeline_result(pipeline)
        iszero(index) && return nothing
        slot = pipeline.slots[index]
        slot.state in (PipelineCommandReady, PipelineFailureReady) || return nothing
        if slot.state === PipelineCommandReady
            outputs = _validate_external_arrays(:output, pipeline.graph.output_formats, outputs)
            _validate_external_output_aliases!(
                keys(pipeline.graph.output_formats),
                values(outputs),
            )
            try
                _copy_captured_outputs!(values(outputs), values(slot.host_outputs))
            catch error
                throw(_fault_captured_pipeline!(pipeline, :publication, error))
            end
        end
        receipt = _captured_pipeline_receipt(slot, pipeline.graph.node_names, Val(Diagnostics))
        slot.state = PipelineAvailable
        slot.result = nothing
        pipeline.completions += UInt64(1)
        return receipt
    finally
        unlock(pipeline.lock)
    end
end

"""
    captured_pipeline_counters(pipeline::PreparedCapturedPipeline)

Return bounded pipeline counters and current occupancy.
"""
function captured_pipeline_counters(pipeline::PreparedCapturedPipeline)
    lock(pipeline.lock) do
        return (
            capacity=length(pipeline.slots),
            occupied=count(slot -> slot.state !== PipelineAvailable, pipeline.slots),
            graph_running=!iszero(pipeline.graph_slot),
            drops=pipeline.drops,
            completions=pipeline.completions,
            fault=pipeline.fault,
            closed=pipeline.closed,
        )
    end
end

"""Drain owned streams and reset scientific state at a quiescent boundary."""
function reset!(pipeline::PreparedCapturedPipeline)
    lock(pipeline.lock) do
        pipeline.closed && throw(InvalidStateException("captured pipeline is closed", :closed))
        if isnothing(pipeline.fault)
            all(slot -> slot.state === PipelineAvailable, pipeline.slots) || throw(
                InvalidStateException(
                    "captured pipeline must be quiescent before reset",
                    :busy,
                ),
            )
        end
        try
            _with_captured_context(pipeline.graph.context) do
                _drain_captured_pipeline_backend!(
                    pipeline.backend,
                    pipeline.graph.context,
                )
                _reset_captured_nodes!(pipeline.graph.nodes)
                _synchronize_captured_graph!(pipeline.graph.context)
            end
        catch error
            throw(_fault_captured_pipeline!(pipeline, :reset, error))
        end
        for slot in pipeline.slots
            slot.state = PipelineAvailable
            slot.result = nothing
            _clear_rejection!(slot.rejection)
        end
        pipeline.graph_slot = 0
        pipeline.last_sequence = UInt64(0)
        pipeline.has_sequence = false
        pipeline.graph.fault = nothing
        pipeline.graph.state = CapturedGraphRunning
        pipeline.fault = nothing
        return pipeline
    end
end

"""
    close(pipeline::PreparedCapturedPipeline)

Stop admission and physically drain all owned CUDA streams before releasing
the caller's lifetime obligation for borrowed host/device storage. This is
safe with occupied slots and after a pipeline fault. Use it in `finally`
when an experiment may time out or throw. A failed close retains the owner
and fault so it can be retried after the backend is usable again. An owner
abandoned without `close` remains rooted and leaks its storage until process
exit, preserving DMA safety.
"""
function Base.close(pipeline::PreparedCapturedPipeline)
    lock(pipeline.lock) do
        pipeline.closed && return nothing
        try
            _with_captured_context(pipeline.graph.context) do
                _drain_captured_pipeline_backend!(
                    pipeline.backend,
                    pipeline.graph.context,
                )
            end
        catch error
            throw(_fault_captured_pipeline!(pipeline, :close, error))
        end
        pipeline.closed = true
        pipeline.graph_slot = 0
        for slot in pipeline.slots
            slot.state = PipelineAvailable
            slot.result = nothing
            _clear_rejection!(slot.rejection)
        end
        if isnothing(pipeline.fault)
            pipeline.graph.state = CapturedGraphReady
        end
        lock(_captured_pipeline_roots_lock) do
            delete!(_captured_pipeline_roots, pipeline)
        end
        return nothing
    end
end

Base.isopen(pipeline::PreparedCapturedPipeline) = lock(pipeline.lock) do
    !pipeline.closed
end
