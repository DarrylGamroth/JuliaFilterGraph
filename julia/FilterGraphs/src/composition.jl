struct CompositionStage{N,O,R,A,M,I,X,E}
    nodes::N
    outputs::O
    routes::R
    available::A
    metadata::M
    inflight::I
    indices::X
    execution::E
end

Base.@nospecializeinfer function _composition_select(@nospecialize(values::Tuple), indices)
    # Cold selection must not compile collect over a union of every deeply
    # heterogeneous Node type. Tuple construction restores concrete field types;
    # the selected objects and repeated execution remain unchanged.
    selected = Any[]
    sizehint!(selected, length(indices))
    for index in indices
        push!(selected, values[index])
    end
    return Tuple(selected)
end

function CompositionStage(graph, indices, execution=nothing)
    return CompositionStage(_composition_select(graph.nodes, indices),
        _composition_select(graph.node_outputs, indices), _composition_select(graph.input_routes, indices),
        _composition_select(graph.node_output_available, indices),
        _composition_select(graph.node_output_metadata, indices),
        _composition_select(graph.node_inflight, indices), indices, execution)
end

function _prepare_cpu_shared_stage(graph, indices, inputs)
    stage = CompositionStage(graph, indices)
    FGA = FilterGraphAlgorithms
    isdefined(FGA, :_process_finite_input!) || return stage
    finite = AbstractArray[]
    processes = map(stage.nodes, stage.routes, stage.outputs) do node, routes, outputs
        prepared = node.prepared
        # These producers preserve their inputs and validate every command or
        # feedback output before returning success. Other nodes are barriers.
        qualified = prepared isa FGA.PreparedAlgorithm{<:Union{
            FGA.VectorLinearCombinationF32,FGA.VdmToPdmF32,
            FGA.PdmCommandF32,FGA.DenseVectorReconstructorF32}}
        if !qualified || !_captured_safe_overlap(prepared, outputs,
                _route_inputs(routes, inputs, graph.node_outputs))
            empty!(finite)
            return FGA._process_admitted!
        end
        proven = length(routes) == 1 &&
            _captured_input_finite(only(routes), inputs, graph.node_outputs, finite)
        finite_combination = isdefined(FGA, :_process_finite_combination!) &&
            prepared isa FGA.PreparedAlgorithm{FGA.VectorLinearCombinationF32} &&
            FGA._finite_combination_result(prepared.plan) &&
            all(route -> _captured_input_finite(route, inputs, graph.node_outputs, finite), routes)
        filter!(buffer -> !any(output -> _captured_mightalias(buffer, output), outputs), finite)
        push!(finite, first(outputs))
        prepared isa FGA.PreparedAlgorithm{FGA.PdmCommandF32} && push!(finite, outputs[2])
        finite_combination && return FGA._process_finite_combination!
        if proven && isdefined(FGA, :_process_finite_conditioning!) &&
                prepared isa FGA.PreparedAlgorithm{FGA.PdmCommandF32} &&
                outputs isa NTuple{2,Vector{Float32}} &&
                _route_value(only(routes), inputs, graph.node_outputs) isa Vector{Float32} &&
                FGA._finite_conditioning_result(prepared.plan)
            return FGA._process_finite_conditioning!
        end
        return proven ? FGA._process_finite_input! : FGA._process_admitted!
    end
    all(process -> process === FGA._process_admitted!, processes) && return stage
    return CompositionStage(stage.nodes, stage.outputs, stage.routes, stage.available,
        stage.metadata, stage.inflight, stage.indices, processes)
end

"""A bounded, serialized collection of independent input branches and one shared output DAG.

Each branch has one public input, one private delayed-feedback input, and a
correction output connected to the shared DAG. The shared DAG runs once after
all arrivals in a call have been admitted. Its private feedback outputs replace
the branch feedback slots together. Startup corrections and feedback are zero.
All nodes and parameter names remain globally unique and retain their ordinary
property and parameter contracts. Processing and control calls must be serialized.
"""
mutable struct PreparedComposition{G,B,S,ST,PR,IF,OF,I,M}
    graph::G
    branches::B
    shared_nodes::S
    shared_stage::ST
    public_routes::PR
    input_formats::IF
    output_formats::OF
    inputs::I
    metadata::M
    contributors::Vector{Union{Nothing,FilterGraphAlgorithms.SampleMetadata}}
    last_inputs::Vector{Union{Nothing,FilterGraphAlgorithms.SampleMetadata}}
    accepted::BitVector
    domain::NTuple{16,UInt8}
    generation::UInt64
    sequence::UInt64
    trigger::BitVector
    failed::Bool
end

_composition_subset(values::NamedTuple, names::Tuple) = NamedTuple{names}(
    map(name -> getproperty(values, name), names))
_composition_subset(values::NamedTuple, names) = _composition_subset(values,Tuple(names))

function _composition_endpoint(spec, name, direction)
    endpoints = direction === :input ? spec.inputs : spec.outputs
    matches = filter(endpoint -> endpoint.external_name === name, endpoints)
    length(matches) == 1 || throw(GraphConfigurationError(
        "composition $direction '$name' requires one explicitly named external Port"))
    return only(matches)
end

"""Prepare a `filter.composition` SPA-JSON descriptor referencing ordinary graph files.

`branches` is a nonempty bounded list of objects with `name`, `graph`, `input`,
`correction`, `feedback`, `shared-input`, and `shared-feedback`. `shared` names
the shared graph file. Paths are resolved relative to the descriptor. Names,
formats, closed graph boundaries, and feedback schemas are checked before use.
Only SCC incomplete-pair rejection is recoverable; any other rejection or
exception latches failure until global reset.
"""
function prepare_composition(path::AbstractString; algorithms::Tuple)
    document = _parse_spa_json(read(path, String))
    _reject_unknown_keys(document, ("node.name", "filter.composition"), "composition document")
    descriptor = get(document, "filter.composition", nothing)
    descriptor isa AbstractDict || throw(GraphConfigurationError("filter.composition must be an object"))
    _reject_unknown_keys(descriptor, ("branches", "shared"), "filter.composition")
    rows = get(descriptor, "branches", nothing)
    rows isa AbstractVector && 1 <= length(rows) <= 16 || throw(GraphConfigurationError(
        "composition requires 1:16 independent branches"))
    directory = dirname(abspath(path))
    specs = GraphSpec[]
    names = Symbol[]
    for row in rows
        row isa AbstractDict || throw(GraphConfigurationError("composition branch must be an object"))
        fields = ("name", "graph", "input", "correction", "feedback", "shared-input", "shared-feedback")
        _reject_unknown_keys(row, (fields..., "rate"), "composition branch")
        all(key -> get(row, key, nothing) isa AbstractString && !isempty(row[key]), fields) ||
            throw(GraphConfigurationError("composition branch requires nonempty string fields"))
        name = Symbol(row["name"])
        name in names && throw(GraphConfigurationError("duplicate composition branch '$name'"))
        push!(names, name)
        push!(specs, _load_graph(normpath(joinpath(directory, row["graph"]))))
    end
    shared_path = get(descriptor, "shared", nothing)
    shared_path isa AbstractString && !isempty(shared_path) ||
        throw(GraphConfigurationError("composition shared must name a graph file"))
    shared = _load_graph(normpath(joinpath(directory, shared_path)))
    all(spec -> iszero(spec.cpu_execution.row_workers), (specs..., shared)) ||
        throw(GraphConfigurationError("composition supports ordinary complete-frame graphs only"))
    links = LinkSpec[link for spec in (specs..., shared) for link in spec.links]
    shared_inputs = Symbol.(getindex.(rows, "shared-input"))
    length(unique(shared_inputs)) == length(rows) ||
        throw(GraphConfigurationError("composition shared-input Ports must be distinct"))
    for (row, spec) in zip(rows, specs)
        source = _composition_endpoint(spec, Symbol(row["correction"]), :output)
        destination = _composition_endpoint(shared, Symbol(row["shared-input"]), :input)
        push!(links, LinkSpec(source, destination))
        _composition_endpoint(spec, Symbol(row["input"]), :input)
        _composition_endpoint(spec, Symbol(row["feedback"]), :input)
        _composition_endpoint(shared, Symbol(row["shared-feedback"]), :output)
    end
    inputs = Endpoint[endpoint for spec in specs for endpoint in spec.inputs]
    append!(inputs, filter(endpoint -> !(endpoint.external_name in shared_inputs), shared.inputs))
    merged = GraphSpec(NodeSpec[node for spec in (specs..., shared) for node in spec.nodes], links,
        inputs, Endpoint[endpoint for spec in (specs..., shared) for endpoint in spec.outputs],
        shared.cpu_execution)
    graph = _prepare_graph(merged; algorithms)
    # Preparing each closed graph separately proves that the descriptor has not
    # hidden a missing camera, parameter, or feedback binding by flattening it.
    for (row, spec) in zip(rows, specs)
        checked = _prepare_graph(spec; algorithms)
        Set(keys(checked.input_formats)) == Set(Symbol.((row["input"], row["feedback"]))) ||
            throw(GraphConfigurationError("each composition branch needs exactly input and feedback data Ports"))
    end
    checked_shared = _prepare_graph(shared; algorithms)
    Set(keys(checked_shared.input_formats)) == Set(shared_inputs) ||
        throw(GraphConfigurationError("shared graph data inputs must exactly match branch corrections"))
    any(values(checked_shared.output_conditional)) && throw(GraphConfigurationError(
        "composition shared graph outputs must be unconditional"))
    indices(spec) = Tuple(findall(node -> any(authored -> authored.name === node.name, spec.nodes), graph.nodes))
    input_storage = _external_buffers(graph.input_formats)
    branches = Tuple(map(zip(rows, specs)) do (row, spec)
        camera, feedback = Symbol(row["input"]), Symbol(row["feedback"])
        correction, returned = Symbol(row["correction"]), Symbol(row["shared-feedback"])
        input_format, output_format = getproperty(graph.input_formats, feedback), getproperty(graph.output_formats, returned)
        input_format.element_type === Float32 && length(input_format.shape) == 1 &&
            input_format.schema == FilterGraphAlgorithms.CONTROLLER_CONSTRAINT_FEEDBACK_V1 &&
            input_format.shape == output_format.shape && input_format.schema == output_format.schema &&
            input_format.rate == output_format.rate && output_format.element_type === Float32 || throw(GraphConfigurationError(
                "composition feedback requires identical controller-feedback F32 vector formats"))
        correction_format = getproperty(graph.output_formats, correction)
        correction_format.element_type === Float32 && length(correction_format.shape) == 1 ||
            throw(GraphConfigurationError("composition correction must be an F32 vector"))
        route = getfield(graph.output_routes, findfirst(==(correction), keys(graph.output_formats)))
        raw_rate = get(row, "rate", nothing)
        rate = if raw_rate === nothing
            nothing
        else
            raw_rate isa AbstractVector && length(raw_rate) == 2 &&
                all(value -> value isa Integer && !(value isa Bool) && 0 < value <= typemax(UInt32), raw_rate) ||
                throw(GraphConfigurationError("composition branch rate requires positive UInt32 numerator and denominator"))
            Int64(raw_rate[1]) // Int64(raw_rate[2])
        end
        camera_source = ExternalInput{findfirst(==(camera),keys(graph.input_formats))}()
        returned_route = graph.output_routes[findfirst(==(returned),keys(graph.output_formats))]
        branch_indices = indices(spec)
        stage = CompositionStage(graph, branch_indices)
        first_controller = findfirst(_composition_controller_node, stage.nodes)
        controller_index = first_controller === nothing ? typemax(Int) : branch_indices[first_controller]
        (; name=Symbol(row["name"]), camera, feedback, correction, returned, rate,
            nodes=branch_indices, route, stage, controller_index, camera_source, returned_route,
            camera_buffer=getproperty(input_storage,camera),feedback_buffer=getproperty(input_storage,feedback))
    end)
    public_inputs = Symbol.(getindex.(rows, "input"))
    private_outputs = Symbol.(getindex.(rows, "shared-feedback"))
    length(unique(private_outputs)) == length(rows) || throw(GraphConfigurationError("feedback outputs must be distinct"))
    public_outputs = filter(name -> !(name in private_outputs), keys(graph.output_formats))
    public_routes = Tuple(graph.output_routes[findfirst(==(name),keys(graph.output_formats))] for name in public_outputs)
    return PreparedComposition(graph, branches, indices(shared),
        _prepare_cpu_shared_stage(graph,indices(shared),input_storage),public_routes,
        _composition_subset(graph.input_formats, public_inputs), _composition_subset(graph.output_formats, public_outputs),
        input_storage,
        NamedTuple{keys(graph.input_formats)}(map(_ -> SampleMetadataSlot(), values(graph.input_formats))),
        Union{Nothing,FilterGraphAlgorithms.SampleMetadata}[nothing for _ in rows],
        Union{Nothing,FilterGraphAlgorithms.SampleMetadata}[nothing for _ in rows], falses(length(rows)),
        _composition_domain(),
        UInt64(1), UInt64(0), falses(length(rows)), false)
end

function _composition_domain()
    publication_uuid = UUIDs.uuid4().value
    return ntuple(index -> UInt8((publication_uuid >> (8 * (index-1))) & 0xff), 16)
end

_composition_shared_output(::NodeOutput{N,P}, shared) where {N,P} = N in shared
_composition_shared_output(route, shared) = false
function output_metadata_source_inputs(composition::PreparedComposition)
    graph=composition.graph
    sources=output_metadata_source_inputs(graph)
    return map(keys(composition.output_formats)) do name
        index=findfirst(==(name),keys(graph.output_formats))
        _composition_shared_output(graph.output_routes[index],composition.shared_nodes) && return 0
        source=sources[index]
        iszero(source) && return 0
        input=getfield(keys(graph.input_formats),source)
        return something(findfirst(==(input),keys(composition.input_formats)),0)
    end
end

input_formats(composition::PreparedComposition) = composition.input_formats
output_formats(composition::PreparedComposition) = composition.output_formats
parameter_formats(composition::PreparedComposition) = parameter_formats(composition.graph)
_captured_composition(composition::PreparedComposition) =
    !(composition.shared_stage.execution isa Union{Nothing,Tuple})

function _require_composition_updates(composition)
    _captured_composition(composition) && throw(ArgumentError(
        "Captured composition parameters must be set before capture"))
    return nothing
end

function replace_parameters!(composition::PreparedComposition, updates::Pair...)
    _require_composition_updates(composition)
    replace_parameters!(composition.graph, updates...)
    return composition
end
function replace_properties!(composition::PreparedComposition, updates::Pair...)
    replace_properties!(composition.graph, updates...)
    return composition
end

"""Snapshot of command publication and retained contributors. No camera identity is
invented for a combined command. `sequence` identifies a publication within a
reset `generation`; contributors preserve each branch's own accepted metadata.
"""
composition_observation(composition::PreparedComposition) = (
    generation=composition.generation, sequence=composition.sequence,
    trigger=Tuple(branch.name for (index,branch) in pairs(composition.branches) if composition.trigger[index]),
    identity=FilterGraphAlgorithms.SampleAcquisitionIdentity(composition.domain,composition.generation,composition.sequence),
    contributors=NamedTuple{Tuple(branch.name for branch in composition.branches)}(Tuple(composition.contributors)),
    failed=composition.failed)

function _composition_stage!(composition, stage::CompositionStage, metadata)
    rejected = _composition_stage!(composition, stage, metadata, stage.execution)
    rejected && _rejection_inputs!(composition.graph.rejection, metadata)
    return rejected
end

function _composition_stage!(composition, stage::CompositionStage, metadata,
        processes::Union{Nothing,Tuple})
    graph = composition.graph
    result,deferred = _process_nodes!(stage.nodes,stage.outputs,stage.routes,stage.available,
        stage.metadata,stage.inflight,composition.inputs,metadata,graph.node_outputs,
        graph.node_output_available,graph.node_output_metadata,false,graph.rejection,1,processes)
    if result !== nothing
        graph.rejection.node = stage.indices[graph.rejection.node]
        return true
    end
    deferred && throw(GraphConfigurationError("composition branch deferred complete-frame processing"))
    return false
end

_composition_prime(failure) = failure === FilterGraphAlgorithms.SccPixelResidualIncompletePair
_composition_controller_node(node) =
    node.prepared.workspace isa FilterGraphAlgorithms.ClosedLoopCorrectionWorkspace
_composition_retain!(::NodeOutput{N,P}, graph) where {N,P} = (getfield(getfield(graph.node_output_available, N), P)[] = true)

function _composition_new_input(previous, current)
    previous === nothing && return true
    current === nothing && return true
    before, after = previous.acquisition_identity, current.acquisition_identity
    if previous.has_acquisition_identity && current.has_acquisition_identity
        before.domain == after.domain || return false
        after.generation > before.generation && return true
        after.generation < before.generation && return false
        return after.sequence > before.sequence
    end
    return current.sequence > previous.sequence
end

_composition_camera(::ExternalInput{I},graph) where {I} = getfield(keys(graph.input_formats),I)
_composition_arrived(arrivals,::Nothing,camera) = haskey(arrivals,camera)
_composition_arrived(arrivals,presence::NamedTuple,camera) = getproperty(presence,camera)

_composition_branches!(composition,::Tuple{},arrivals,metadata,presence,index) = nothing
function _composition_branches!(composition,branches::Tuple,arrivals,metadata,presence,index)
    branch = first(branches)
    graph = composition.graph
    camera = _composition_camera(branch.camera_source,graph)
    if _composition_arrived(arrivals,presence,camera)
        sample = getproperty(metadata,camera)
        current = sample.present ? sample.value : nothing
        if !(sample.present && sample.value.corrupted) && _composition_new_input(composition.last_inputs[index],current)
            composition.last_inputs[index] = current
            copyto!(branch.camera_buffer,getproperty(arrivals,camera))
            _store_metadata!(getproperty(composition.metadata,camera),sample)
            admitted = map(_admitted_metadata,composition.metadata)
            if _composition_stage!(composition,branch.stage,admitted)
                reason = graph.rejection.reason
                _composition_prime(reason) || throw(InvalidStateException("composition branch rejected: $reason",:rejected))
                branch.controller_index < graph.rejection.node &&
                    throw(InvalidStateException("composition priming occurred after a controller",:rejected))
                _clear_available!(branch.stage.available)
            else
                _route_available(branch.route,graph.node_output_available) ||
                    throw(InvalidStateException("composition branch omitted its correction",:unavailable))
                composition.accepted[index] = true
            end
        end
    end
    return _composition_branches!(composition,Base.tail(branches),arrivals,metadata,presence,index+1)
end

_composition_retain_branches!(::Tuple{},graph) = nothing
function _composition_retain_branches!(branches::Tuple,graph)
    _composition_retain!(first(branches).route,graph)
    return _composition_retain_branches!(Base.tail(branches),graph)
end

_composition_publish_metadata!(::Tuple{},publication) = nothing
function _composition_publish_metadata!(metadata::Tuple,publication)
    for slot in first(metadata)
        _store_metadata!(slot,publication)
    end
    return _composition_publish_metadata!(Base.tail(metadata),publication)
end

_composition_feedback!(composition,::Tuple{},index) = nothing
function _composition_feedback!(composition,branches::Tuple,index)
    branch = first(branches)
    graph = composition.graph
    _route_available(branch.returned_route,graph.node_output_available) ||
        throw(InvalidStateException("composition shared graph omitted feedback",:unavailable))
    copyto!(branch.feedback_buffer,_route_value(branch.returned_route,(),graph.node_outputs))
    if composition.accepted[index]
        composition.contributors[index] = _metadata_value(_route_metadata_slot(branch.route,graph.node_output_metadata))
    else
        _clear_available!(branch.stage.available)
    end
    return _composition_feedback!(composition,Base.tail(branches),index+1)
end

@inline _composition_trigger_reference(::Tuple{},accepted,metadata,graph,index)=(false,nothing)
@inline function _composition_trigger_reference(branches::Tuple,accepted,metadata,graph,index)
    selected,reference=_composition_trigger_reference(Base.tail(branches),accepted,metadata,graph,index+1)
    selected && return (selected,reference)
    accepted[index] || return (false,nothing)
    camera=_composition_camera(first(branches).camera_source,graph)
    sample=getproperty(metadata,camera)
    return (true,sample.present ? FilterGraphAlgorithms.sample_reference_timestamp(sample.value) : nothing)
end

function _process_composition_admitted!(outputs,composition::PreparedComposition,arrivals,metadata,presence=nothing)
    composition.failed && throw(InvalidStateException("composition failed and requires reset",:failed))
    graph = composition.graph
    _clear_available!(graph.node_output_available)
    _clear_rejection!(graph.rejection)
    fill!(composition.accepted,false)
    try
        _composition_branches!(composition,composition.branches,arrivals,metadata,presence,1)
        if any(composition.accepted)
            _composition_retain_branches!(composition.branches,graph)
            admitted = map(_admitted_metadata,composition.metadata)
            if _composition_stage!(composition,composition.shared_stage,admitted)
                reason = graph.rejection.reason
                throw(InvalidStateException("composition shared graph rejected: $reason",:rejected))
            end
            composition.sequence == typemax(UInt64) && throw(InvalidStateException("composition publication sequence exhausted",:overflow))
            composition.sequence += UInt64(1)
            copyto!(composition.trigger,composition.accepted)
            # Descriptor order defines the final accepted trigger in a batch.
            # This source-associated timestamp does not become a new exposure time.
            _,reference=_composition_trigger_reference(composition.branches,composition.accepted,admitted,graph,1)
            publication = FilterGraphAlgorithms.SampleMetadata(composition.sequence;terminal=true,
                reference_timestamp=reference,acquisition_identity=FilterGraphAlgorithms.SampleAcquisitionIdentity(
                    composition.domain,composition.generation,composition.sequence))
            _composition_publish_metadata!(composition.shared_stage.metadata,publication)
            _composition_feedback!(composition,composition.branches,1)
        end
        _copy_external_metadata!(graph.output_routes,values(graph.published_metadata),graph.node_output_metadata,graph.node_output_available)
        available = _copy_external_outputs!(composition.public_routes,values(outputs),graph.node_outputs,graph.node_output_available)
        return NamedTuple{keys(composition.output_formats)}(available)
    catch
        composition.failed = true
        rethrow()
    end
end

"""Process a partial NamedTuple of independent arrivals. All arrivals in one call
are processed in descriptor order, followed by at most one shared publication.
Available outputs are copied to caller buffers. The composition retains its
own corrections and delayed feedback across calls.
No arrival from an absent camera is synthesized or reintegrated.
"""
function process!(outputs::NamedTuple, composition::PreparedComposition, arrivals::NamedTuple,
                  metadata::NamedTuple=NamedTuple{keys(arrivals)}(map(_ -> nothing, values(arrivals))))
    known_names = keys(composition.input_formats)
    all(name -> name in known_names, keys(arrivals)) ||
        throw(ArgumentError("unknown composition input"))
    keys(metadata) == keys(arrivals) || throw(ArgumentError("composition metadata must match arrivals"))
    formats = _composition_subset(composition.input_formats, keys(arrivals))
    _validate_external_arrays(:input, formats, arrivals)
    outputs = _validate_external_arrays(:output, composition.output_formats, outputs)
    _validate_external_metadata(formats, metadata)
    _validate_external_output_aliases!(keys(outputs), values(outputs))
    admitted = map(FilterGraphAlgorithms._admitted_sample_metadata, metadata)
    return _process_composition_admitted!(outputs, composition, arrivals, admitted)
end

output_metadata(composition::PreparedComposition) = _composition_subset(output_metadata(composition.graph), keys(composition.output_formats))
rejection_context(composition::PreparedComposition) = rejection_context(composition.graph)

_reset_composition_stage!(::Union{Nothing,Tuple}) = nothing

function reset!(composition::PreparedComposition)
    composition.failed = true
    for branch in composition.branches
        _reset_composition_stage!(branch.stage.execution)
    end
    _reset_composition_stage!(composition.shared_stage.execution)
    reset!(composition.graph)
    for buffers in composition.graph.node_outputs, array in buffers
        fill!(array, zero(eltype(array)))
    end
    for buffers in composition.graph.node_output_metadata, slot in buffers
        _store_metadata!(slot, nothing)
    end
    for slot in values(composition.graph.published_metadata); _store_metadata!(slot, nothing); end
    for array in values(composition.inputs); fill!(array, zero(eltype(array))); end
    for slot in values(composition.metadata); _store_metadata!(slot, nothing); end
    fill!(composition.contributors, nothing)
    fill!(composition.last_inputs, nothing)
    fill!(composition.accepted, false)
    composition.generation += UInt64(1)
    composition.sequence = UInt64(0)
    fill!(composition.trigger,false)
    composition.failed = false
    return composition
end
