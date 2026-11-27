mutable struct GpuReadoutOwner{G,I,O,IR,OR}
    graph::G
    input_bindings::I
    output_bindings::O
    input_records::IR
    output_records::OR
    join::Nothing
    metadata_fault::Union{Nothing,MetadataPublicationError}
    active::Bool
    process_count::UInt64
end

_public_input_bindings(owner::GpuReadoutOwner) = owner.input_bindings
_metadata_sources(owner::GpuReadoutOwner) = owner.graph.output_metadata_sources
_graph_callback_trace(::GpuReadoutOwner) = nothing
_graph_callback_metrics(::GpuReadoutOwner) = nothing
_preflight_callback_outputs!(::GpuReadoutOwner, inputs, outputs) = nothing
_parameter_adoption_count(::GpuReadoutOwner) = UInt64(0)
_adopt_pending_properties!(::GpuReadoutOwner) = false
_adopt_pending_recovery!(::GpuReadoutOwner) = false
function _prepare_owner(owner::GpuReadoutOwner)
    isnothing(owner.metadata_fault) || throw(owner.metadata_fault)
    owner.active = true
    return nothing
end
function _latch_metadata_fault!(owner::GpuReadoutOwner, error::MetadataPublicationError)
    owner.active = false
    owner.metadata_fault = error
    return nothing
end
_deactivate_owner(owner::GpuReadoutOwner) = (owner.active = false; nothing)
function _reset_owner!(owner::GpuReadoutOwner)
    FilterGraphs.reset!(owner.graph)
    _clear_metadata_snapshots!(owner.input_records)
    _clear_metadata_snapshots!(owner.output_records)
    owner.metadata_fault = nothing
    return nothing
end

function _process_graph_buffers!(
    owner::GpuReadoutOwner, ::Nothing, inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    _set_outputs_unavailable!(values(owner.output_bindings), outputs, 1)
    _all_inputs_available(values(owner.input_bindings), inputs, 1) || return nothing
    input_arrays = _borrow_buffers(owner.input_bindings, inputs)
    metadata = _input_metadata(owner.input_bindings, inputs)
    graph = owner.graph
    sample = only(metadata)
    output_arrays = if sample.present && sample.value.terminal
        _preflight_output_slots!(values(owner.output_records), _metadata_sources(owner), inputs, outputs, 1)
        arrays = _borrow_buffers(owner.output_bindings, outputs)
        _retain_input_metadata!(values(owner.input_records), inputs, values(metadata), 1)
        terminal = FilterGraphAlgorithms._admitted_sample_metadata(
            FilterGraphAlgorithms.SampleMetadata(sample.value; offset=0))
        anticipated = map(index -> iszero(index) ? FilterGraphAlgorithms._NO_SAMPLE_METADATA : terminal,
            _metadata_sources(owner))
        _cache_output_metadata!(values(owner.output_records), values(owner.input_records),
            _metadata_sources(owner), anticipated)
        _reserve_captured_output_metadata!(values(owner.output_records), outputs, 1)
        arrays
    end
    FilterGraphs._assert_gpu_readout_open(graph)
    FilterGraphs._clear_rejection!(graph.rejection)
    result = FilterGraphs._process_gpu_readout!(
        output_arrays, graph, input_arrays, metadata, graph.result_types)
    result isa NamedTuple && any(values(result)) || return nothing
    try
        admitted = map(FilterGraphAlgorithms._admitted_sample_metadata,
            values(FilterGraphs.output_metadata(graph)))
        _cache_output_metadata!(values(owner.output_records), values(owner.input_records),
            _metadata_sources(owner), admitted)
        _finalize_outputs!(values(result), admitted, outputs, 1, values(owner.output_records))
    catch error
        # Terminal CUDA work may have committed feedback. A restart must
        # reset that state before it can accept another camera frame.
        throw(error isa MetadataPublicationError ? error : MetadataPublicationError(error))
    end
    return nothing
end

"""
    PipeWireNode(graph::FilterGraphs.PreparedGpuReadoutGraph; name, rate, kwargs...)

Expose one prepared CUDA row-block owner through PipeWire. Apply calibration
and prepare retained feedback before creating the Node. The caller closes the
Graph after the PipeWire Node has stopped.
"""
function PipeWireNode(graph::FilterGraphs.PreparedGpuReadoutGraph;
    rate=nothing, boundary_layout::Union{Nothing,ArrayLayout}=nothing,
    input_metadata::NamedTuple=NamedTuple(), output_metadata::NamedTuple=NamedTuple(),
    fifo_inputs::Bool=true, kwargs...)
    isopen(graph) || throw(ArgumentError("CUDA readout Graph is closed"))
    inputs = _port_bindings(graph.input_formats, boundary_layout)
    outputs = _port_bindings(graph.output_formats, boundary_layout)
    owner = GpuReadoutOwner(graph, inputs, outputs, _metadata_snapshots(inputs),
        _metadata_snapshots(outputs), nothing, nothing, false, UInt64(0))
    ports = _graph_ports(graph.input_formats, NamedTuple(), graph.output_formats, rate;
        boundary_layout)
    return _pipewire_owned_graph_node(owner; fifo_inputs,
        ports=_metadata_ports(ports, input_metadata, output_metadata), kwargs...)
end
