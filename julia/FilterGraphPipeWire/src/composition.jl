# The composition's flattened checked graph supplies ordinary properties and
# parameters; this adapter publishes only its public camera/telemetry boundary.
struct CompositionBindings{I,O,F,S}
    public_input_bindings::I
    public_output_bindings::O
    fallback_inputs::F
    output_metadata_source_inputs::S
end

function _owner_feedback_bridge(graph, composition::FilterGraphs.PreparedComposition,
                               layout, ::Nothing)
    return CompositionBindings(_port_bindings(composition.input_formats, layout),
        _port_bindings(composition.output_formats, layout),
        FilterGraphs._composition_subset(composition.inputs,keys(composition.input_formats)),
        FilterGraphs.output_metadata_source_inputs(composition))
end
_public_input_bindings(owner, bridge::CompositionBindings) = bridge.public_input_bindings

function _reset_owner!(owner::PreparedGraphOwner, composition::FilterGraphs.PreparedComposition)
    _invalidate_recovery_observation!(owner)
    _invalidate_recovery_branch_readiness!(owner)
    _adopt_pending_recovery!(owner)
    _adopt_pending_properties!(owner)
    _adopt_pending_parameters!(owner)
    return reset!(composition)
end

_composition_borrow_inputs(::Tuple{},::Tuple{},inputs,index) = ()
function _composition_borrow_inputs(bindings::Tuple,fallbacks::Tuple,inputs,index)
    binding = first(bindings)
    # The absent wrapper has the same concrete type as a real camera buffer.
    # Its owned fallback pointer is never read: the presence mask gates the branch.
    array = PipeWireAO.input_available(inputs[index]) ? _borrow_buffer(binding,inputs[index]) :
        _borrow_pointer(binding,pointer(first(fallbacks)),Val(false))
    return (array,_composition_borrow_inputs(Base.tail(bindings),Base.tail(fallbacks),inputs,index+1)...)
end

_composition_input_presence(::Tuple{},inputs,index) = ()
function _composition_input_presence(bindings::Tuple,inputs,index)
    return (PipeWireAO.input_available(inputs[index]),
        _composition_input_presence(Base.tail(bindings),inputs,index+1)...)
end

# Branch stage types differ. Keep the prepared tuple concrete while retaining
# accepted camera records in the warmed callback.
_retain_composition_records!(::Tuple{},accepted,names,inputs,metadata,snapshots,::Val) = nothing
function _retain_composition_records!(branches::Tuple,accepted,names,inputs,metadata,snapshots,::Val{Index}) where {Index}
    if accepted[Index]
        camera = something(findfirst(==(first(branches).camera),names))
        _retain_metadata!(getfield(snapshots,camera),inputs[camera],getfield(metadata,camera))
    end
    return _retain_composition_records!(Base.tail(branches),accepted,names,inputs,metadata,snapshots,Val(Index+1))
end

function _process_graph_buffers!(owner::PreparedGraphOwner,
                                composition::FilterGraphs.PreparedComposition,
                                inputs::NdArrayFilterBuffers{false}, outputs::NdArrayFilterBuffers{true})
    bindings = owner.feedback
    length(inputs) == length(bindings.public_input_bindings) || throw(DimensionMismatch("composition camera count"))
    length(outputs) == length(bindings.public_output_bindings) || throw(DimensionMismatch("composition output count"))
    names = keys(bindings.public_input_bindings)
    presence = NamedTuple{names}(_composition_input_presence(values(bindings.public_input_bindings),inputs,1))
    any(values(presence)) || throw(InvalidStateException("composition callback has no input arrival",:unavailable))
    arrays = NamedTuple{names}(_composition_borrow_inputs(values(bindings.public_input_bindings),
        values(bindings.fallback_inputs),inputs,1))
    metadata = _input_metadata(bindings.public_input_bindings,inputs)
    destinations = _borrow_buffers(bindings.public_output_bindings,outputs)
    _adopt_pending_parameters!(owner)
    _begin_composition_recovery_frame!(owner, presence)
    sequence = composition.sequence
    try
        availability = FilterGraphs._process_composition_admitted!(destinations,composition,arrays,metadata,presence)
        published = FilterGraphs._composition_subset(
            FilterGraphs._output_metadata_admitted(owner.graph),keys(composition.output_formats))
        _retain_composition_records!(composition.branches,composition.accepted,
            keys(bindings.public_input_bindings),inputs,values(metadata),values(owner.input_records),Val(1))
        _cache_output_metadata!(values(owner.output_records),values(owner.input_records),
            bindings.output_metadata_source_inputs,values(published))
        _finalize_outputs!(values(availability),values(published),outputs,1,values(owner.output_records))
        composition.sequence != sequence && _observe_composition_recovery!(owner)
    catch
        composition.failed = true
        _set_outputs_unavailable!(values(bindings.public_output_bindings),outputs,1)
        rethrow()
    end
    return nothing
end

"""Publish a graph composition with independent camera arrivals.

All graphs share one serialized owner and the standard flat parameter/property
surface. Feedback is private and retained by the composition. No common camera
identity, camera-rate ratio or shared trigger is required.
`parameter_ports=false` retains cold-bound parameters without exposing live
Parameter Ports or their sparse-update callback.
"""
function PipeWireNode(composition::FilterGraphs.PreparedComposition;
    name::AbstractString="julia.filter-graph-island", remote=nothing, rate=nothing, connect::Bool=true, run_control::Bool=false,
    boundary_layout=nothing, fifo_inputs::Bool=false,
    parameter_ports::Bool=!FilterGraphs._captured_composition(composition), feedback=nothing,
    channel_units=NamedTuple(), property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[], property_getter=nothing,
    recovery=nothing, ingress::NamedTuple=NamedTuple(), frame_observer=nothing,
    input_metadata::NamedTuple=NamedTuple(),output_metadata::NamedTuple=NamedTuple(),
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME)
    feedback === nothing || throw(ArgumentError("composition owns its private feedback"))
    if FilterGraphs._captured_composition(composition)
        parameter_ports && throw(ArgumentError("Captured composition parameters must be set before capture"))
        recovery === nothing || throw(ArgumentError("Captured composition requires a full reset for recovery"))
    end
    isempty(ingress) && frame_observer === nothing ||
        throw(ArgumentError("composition does not support single-frame ingress or observation"))
    graph = composition.graph
    owner = PreparedGraphOwner(graph, composition, boundary_layout)
    ports = _metadata_ports(_composition_ports(composition,rate;boundary_layout,parameter_ports),input_metadata,output_metadata)
    return _pipewire_graph_node(owner, graph; ports, name, remote, connect,
        independent_inputs=true, fifo_inputs, parameter_ports, run_control, channel_units,
        property_observer, property_info, property_getter,recovery,reference_clock)
end

function _composition_ports(composition, rate; boundary_layout=nothing, parameter_ports::Bool=true)
    ports = NdArrayFilterPort[]
    for branch in composition.branches
        formats = FilterGraphs._composition_subset(composition.input_formats, (branch.camera,))
        # An explicit branch rate is absolute Hz, independent of the graph's
        # relative Algorithm rates and the owner's nominal command-port rate.
        if branch.rate === nothing
            _append_graph_ports!(ports, formats, PipeWireAO.DIRECTION_INPUT, rate; boundary_layout)
        else
            format = getproperty(formats, branch.camera)
            push!(ports, NdArrayFilterPort(String(branch.camera), PipeWireAO.DIRECTION_INPUT,
                NdArrayFormat(_element_type(format.element_type),format.shape;
                    layout=_layout(something(boundary_layout,format.layout)),
                    rate=SPA.Fraction(numerator(branch.rate),denominator(branch.rate)));
                schema=format.schema))
        end
    end
    if parameter_ports
        _append_graph_ports!(ports, composition.graph.parameter_formats, PipeWireAO.DIRECTION_INPUT, nothing;
            parameter=true, boundary_layout)
    end
    for name in keys(composition.output_formats)
        formats = FilterGraphs._composition_subset(composition.output_formats, (name,))
        _append_graph_ports!(ports, formats, PipeWireAO.DIRECTION_OUTPUT, rate;
            boundary_layout)
    end
    return ports
end

function _prepare_cpu_graph!(owner::PreparedGraphOwner, composition::FilterGraphs.PreparedComposition,
                             ::Val{:complete_frame}, required_outputs; samples=nothing)
    samples isa Tuple && !isempty(samples) || throw(ArgumentError(
        "composition preparation requires explicit independent-camera samples"))
    length(unique(required_outputs)) == length(required_outputs) ||
        throw(ArgumentError("required outputs must be distinct"))
    all(name -> haskey(composition.output_formats, name), required_outputs) ||
        throw(ArgumentError("unknown required composition output"))
    outputs = FilterGraphs._external_buffers(composition.output_formats)
    try
        _reset_owner!(owner)
        _prepare_owner(owner)
        for (index, sample) in pairs(samples)
            sample isa NamedTuple && Set(keys(sample)) == Set((:inputs,:metadata,:expected)) ||
                throw(ArgumentError("composition samples require inputs, metadata and expected"))
            availability = process!(outputs, composition, sample.inputs, sample.metadata)
            if sample.expected isa Enum
                rejection = FilterGraphs.rejection_context(composition)
                rejection !== nothing && rejection.reason === sample.expected ||
                    throw(ArgumentError("composition sample did not produce its expected branch rejection"))
            elseif sample.expected isa NamedTuple
                all(name -> haskey(availability, name), keys(sample.expected)) ||
                    throw(ArgumentError("unknown expected composition output"))
                all(getproperty(availability,name) === value for (name,value) in pairs(sample.expected)) ||
                    throw(ArgumentError("composition sample output availability differs"))
            elseif sample.expected === nothing
                all(name -> getproperty(availability,name), required_outputs) ||
                    throw(ArgumentError("composition sample omitted a required output"))
            else
                throw(ArgumentError("composition sample expected must be nothing, branch rejection or availability"))
            end
            index == length(samples) && !all(name -> getproperty(availability,name), required_outputs) &&
                throw(ArgumentError("final composition preparation sample must publish required outputs"))
        end
    finally
        _reset_owner!(owner)
        for array in values(owner.warm_inputs); fill!(array, zero(eltype(array))); end
        for array in values(owner.warm_outputs); fill!(array, zero(eltype(array))); end
        owner.active = false
    end
    return composition
end
