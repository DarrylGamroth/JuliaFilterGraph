struct ExternalInput{Index} end

struct NodeOutput{NodeIndex,PortIndex} end

struct ParameterInput{NodeIndex,Name} end

# Keep the finite Graph result union explicit after inference stops union-splitting
# heterogeneous Algorithm failure enums in larger DAGs.
struct GraphResultTypes{Availability,Failure} end

mutable struct SampleMetadataSlot
    present::Bool
    value::FilterGraphAlgorithms.SampleMetadata
end

SampleMetadataSlot() =
    SampleMetadataSlot(false, FilterGraphAlgorithms.SampleMetadata(0))

struct PreparedNode{D,P,IN,OUT,PN,IF,OF,PF,OC,MS}
    name::Symbol
    prepared::P
    input_names::IN
    output_names::OUT
    parameter_names::PN
    input_formats::IF
    output_formats::OF
    parameter_formats::PF
    output_conditional::OC
    output_metadata_sources::MS
end

function PreparedNode(
    ::Type{D},
    name::Symbol,
    prepared::P,
    input_names::IN,
    output_names::OUT,
    parameter_names::PN,
    input_formats::IF,
    output_formats::OF,
    parameter_formats::PF,
    output_conditional::OC,
    output_metadata_sources::MS,
) where {D,P,IN,OUT,PN,IF,OF,PF,OC,MS}
    return PreparedNode{D,P,IN,OUT,PN,IF,OF,PF,OC,MS}(
        name,
        prepared,
        input_names,
        output_names,
        parameter_names,
        input_formats,
        output_formats,
        parameter_formats,
        output_conditional,
        output_metadata_sources,
    )
end

"""
    PreparedGraph

One prepared Julia Filter Graph. It owns its Nodes, fixed routes, and internal
buffers.
"""
struct PreparedGraph{N,NF,B,A,M,I,IR,PR,OR,IF,PF,OF,OC,PM,DM,RT,RS}
    nodes::N
    node_failure_types::NF
    node_outputs::B
    node_output_available::A
    node_output_metadata::M
    node_inflight::I
    input_routes::IR
    parameter_routes::PR
    output_routes::OR
    input_formats::IF
    parameter_formats::PF
    output_formats::OF
    output_conditional::OC
    published_metadata::PM
    default_input_metadata::DM
    result_types::RT
    rejection::RS
end

@inline _metadata_source_input(::PreparedGraph, ::ExternalInput{Index}) where {Index} = Index

@inline function _metadata_source_input(
    graph::PreparedGraph,
    ::NodeOutput{NodeIndex,PortIndex},
) where {NodeIndex,PortIndex}
    node = getfield(graph.nodes, NodeIndex)
    input_index = getfield(node.output_metadata_sources, PortIndex)
    iszero(input_index) && return 0
    return _metadata_source_input(
        graph,
        getfield(graph.input_routes, NodeIndex)[input_index],
    )
end

"""
    output_metadata_source_inputs(graph::PreparedGraph)

Return each external output's ultimate external metadata-source input index.
Zero denotes an output without declared input provenance.
"""
function output_metadata_source_inputs(graph::PreparedGraph)
    return map(route -> _metadata_source_input(graph, route), graph.output_routes)
end

function _graph_configuration_error(context::AbstractString, error)
    error isa GraphConfigurationError && return error
    return GraphConfigurationError("$context: $(sprint(showerror, error))")
end

function _node_indices(spec::GraphSpec)
    indices = Dict{Symbol,Int}()
    for (index, node) in pairs(spec.nodes)
        haskey(indices, node.name) && throw(
            GraphConfigurationError("Graph Node name '$(node.name)' is not unique"),
        )
        indices[node.name] = index
    end
    return indices
end

function _topological_order(spec::GraphSpec, node_indices::Dict{Symbol,Int})
    node_count = length(spec.nodes)
    indegree = zeros(Int, node_count)
    for link in spec.links
        source = get(node_indices, link.output.node, 0)
        destination = get(node_indices, link.input.node, 0)
        iszero(source) && throw(
            GraphConfigurationError("Link names unknown output Node '$(link.output.node)'"),
        )
        iszero(destination) && throw(
            GraphConfigurationError("Link names unknown input Node '$(link.input.node)'"),
        )
        indegree[destination] += 1
    end

    order = Int[]
    selected = falses(node_count)
    while length(order) < node_count
        next_node = findfirst(index -> !selected[index] && iszero(indegree[index]), 1:node_count)
        isnothing(next_node) && throw(
            GraphConfigurationError("filter.graph links must form a directed acyclic Graph"),
        )
        selected[next_node] = true
        push!(order, next_node)
        source_name = spec.nodes[next_node].name
        for link in spec.links
            link.output.node === source_name || continue
            indegree[node_indices[link.input.node]] -= 1
        end
    end
    return order
end

function _validate_algorithms(algorithms::Tuple)
    isempty(algorithms) && throw(
        GraphConfigurationError("prepare_graph requires at least one Algorithm declaration"),
    )
    labels = Set{String}()
    for declaration in algorithms
        declaration isa DataType &&
            declaration <: FilterGraphAlgorithms.AbstractAlgorithmDeclaration || throw(
                GraphConfigurationError(
                    "Algorithm set entries must be concrete Algorithm declaration types",
                ),
            )
        label = FilterGraphAlgorithms.algorithm_label(declaration)
        label isa AbstractString && !isempty(label) || throw(
            GraphConfigurationError("Algorithm declaration has an invalid label"),
        )
        normalized = String(label)
        normalized in labels && throw(
            GraphConfigurationError("Algorithm label '$normalized' is not unique"),
        )
        push!(labels, normalized)
    end
    return algorithms
end

function _resolve_algorithm(algorithms::Tuple, label::String)
    for declaration in algorithms
        FilterGraphAlgorithms.algorithm_label(declaration) == label && return declaration
    end
    throw(GraphConfigurationError("No supplied Algorithm has label '$label'"))
end

function _convert_graph_value(value, destination_type::Type, description::AbstractString)
    value isa destination_type && return value
    if destination_type <: Tuple && value isa AbstractVector
        fields = fieldtypes(destination_type)
        length(value) == length(fields) || throw(GraphConfigurationError(
            "$description requires $(length(fields)) entries, received $(length(value))",
        ))
        return ntuple(length(fields)) do index
            _convert_graph_value(value[index], fields[index], "$description entry $index")
        end
    end
    if destination_type isa Union
        converted = nothing
        converted_type = nothing
        for member in Base.uniontypes(destination_type)
            candidate = try
                convert(member, value)
            catch
                continue
            end
            isnothing(converted_type) || throw(
                GraphConfigurationError(
                    "$description is ambiguous between $converted_type and $member",
                ),
            )
            converted = candidate
            converted_type = member
        end
        !isnothing(converted_type) && return converted
    end
    try
        return convert(destination_type, value)
    catch error
        throw(
            GraphConfigurationError(
                "$description requires $destination_type, received $(typeof(value)): " *
                sprint(showerror, error),
            ),
        )
    end
end

function _prepare_configuration(declaration, node::NodeSpec)
    fields = FilterGraphAlgorithms.algorithm_configuration(declaration)
    known_names = Set(String(field.name) for field in fields)
    for name in keys(node.configuration)
        name in known_names || throw(
            GraphConfigurationError(
                "Node '$(node.name)' has unknown configuration field '$name'",
            ),
        )
    end
    values = map(fields) do field
        name = String(field.name)
        if !haskey(node.configuration, name) && hasproperty(field, :default) &&
           field.default !== nothing
            return _convert_graph_value(
                something(field.default), field.julia_type,
                "Node '$(node.name)' default configuration field '$name'",
            )
        end
        haskey(node.configuration, name) || throw(
            GraphConfigurationError(
                "Node '$(node.name)' requires configuration field '$name'",
            ),
        )
        return _convert_graph_value(
            node.configuration[name],
            field.julia_type,
            "Node '$(node.name)' configuration field '$name'",
        )
    end
    configuration_type = FilterGraphAlgorithms.algorithm_configuration_type(declaration)
    try
        return configuration_type(values...)
    catch error
        throw(_graph_configuration_error("Node '$(node.name)' configuration is invalid", error))
    end
end

function _property_info(infos::Tuple{}, name::String)
    throw(GraphConfigurationError("Unknown Node Property '$name'"))
end

function _property_info(infos::Tuple, name::String)
    info = first(infos)
    String(info.name) == name && return info
    return _property_info(Base.tail(infos), name)
end

function _apply_initial_properties!(prepared, node::NodeSpec)
    isempty(node.properties) && return prepared
    prepared.plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan || throw(
        GraphConfigurationError(
            "Node '$(node.name)' declares Properties but its Plan has no Property contract",
        ),
    )
    infos = try
        FilterGraphAlgorithms.property_infos(prepared.plan)
    catch error
        throw(_graph_configuration_error("Node '$(node.name)' Property schema is invalid", error))
    end
    assignments = Any[]
    for (name, value) in node.properties
        info = try
            _property_info(infos, name)
        catch error
            error isa GraphConfigurationError || rethrow()
            throw(
                GraphConfigurationError(
                    "Node '$(node.name)' has unknown Property '$name'",
                ),
            )
        end
        converted = _convert_graph_value(
            value,
            typeof(info.default_value),
            "Node '$(node.name)' Property '$name'",
        )
        push!(assignments, FilterGraphAlgorithms.PropertyAssignment(info.id, converted))
    end
    try
        prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan, Tuple(assignments))
    catch error
        throw(_graph_configuration_error("Node '$(node.name)' Properties are invalid", error))
    end
    return prepared
end

function _prepare_node(declaration, node::NodeSpec)
    configuration = _prepare_configuration(declaration, node)
    prepared = try
        FilterGraphAlgorithms.prepare_algorithm(declaration, configuration)
    catch error
        throw(_graph_configuration_error("Node '$(node.name)' preparation failed", error))
    end
    _apply_initial_properties!(prepared, node)

    ports = FilterGraphAlgorithms.algorithm_ports(declaration)
    formats = FilterGraphAlgorithms.algorithm_formats(prepared)
    length(ports) == length(formats) || throw(
        GraphConfigurationError("Node '$(node.name)' has inconsistent Port metadata"),
    )

    input_names = Symbol[]
    output_names = Symbol[]
    parameter_names = Symbol[]
    input_formats = Any[]
    output_formats = Any[]
    parameter_formats = Any[]
    output_ports = Any[]
    for (port, format) in zip(ports, formats)
        if port.role isa FilterGraphAlgorithms.SparseParameter
            port.direction isa FilterGraphAlgorithms.InputPort || throw(
                GraphConfigurationError(
                    "Node '$(node.name)' has an invalid Parameter Port direction",
                ),
            )
            format.layout isa FilterGraphAlgorithms.ColumnMajorLayout || throw(
                GraphConfigurationError(
                    "Node '$(node.name)' requires a Julia column-major Port layout",
                ),
            )
            push!(parameter_names, port.name)
            push!(parameter_formats, format)
            continue
        end
        port.role isa FilterGraphAlgorithms.FrameData || throw(
            GraphConfigurationError(
                "Node '$(node.name)' has an unsupported Port role",
            ),
        )
        format.layout isa FilterGraphAlgorithms.ColumnMajorLayout || throw(
            GraphConfigurationError(
                "Node '$(node.name)' requires a Julia column-major Port layout",
            ),
        )
        if port.direction isa FilterGraphAlgorithms.InputPort
            push!(input_names, port.name)
            push!(input_formats, format)
        elseif port.direction isa FilterGraphAlgorithms.OutputPort
            push!(output_names, port.name)
            push!(output_formats, format)
            push!(output_ports, port)
        else
            throw(GraphConfigurationError("Node '$(node.name)' has an invalid Port direction"))
        end
    end
    output_metadata_sources = map(output_ports) do port
        isnothing(port.metadata_source) && return 0
        source = findfirst(==(port.metadata_source), input_names)
        isnothing(source) && throw(
            GraphConfigurationError(
                "Node '$(node.name)' output Port '$(port.name)' has an invalid metadata source",
            ),
        )
        return source
    end
    output_conditional = Tuple(port.conditional for port in output_ports)
    return PreparedNode(
        declaration,
        node.name,
        prepared,
        Tuple(input_names),
        Tuple(output_names),
        Tuple(parameter_names),
        Tuple(input_formats),
        Tuple(output_formats),
        Tuple(parameter_formats),
        output_conditional,
        Tuple(output_metadata_sources),
    )
end

_node_failure_type(::PreparedNode{D}) where {D} =
    FilterGraphAlgorithms.algorithm_failure_type(D)

function _graph_result_types(output_formats::NamedTuple, failure_types::Tuple)
    availability_type = NamedTuple{
        keys(output_formats),
        NTuple{length(output_formats),Bool},
    }
    declared_failures = filter(type -> type !== Union{}, failure_types)
    failure_type = Union{declared_failures...}
    return GraphResultTypes{availability_type,failure_type}()
end

_route_conditional(::ExternalInput, node_output_conditional) = false

function _route_conditional(
    ::NodeOutput{NodeIndex,PortIndex},
    node_output_conditional,
) where {NodeIndex,PortIndex}
    return node_output_conditional[NodeIndex][PortIndex]
end

function _external_input_port(node::PreparedNode, endpoint::Endpoint)
    if isnothing(endpoint.port)
        count = length(node.input_names) + length(node.parameter_names)
        count == 1 || throw(
            GraphConfigurationError(
                "Node '$(node.name)' requires an explicit input Port name",
            ),
        )
        return isempty(node.input_names) ? (:parameter, 1) : (:data, 1)
    end

    data_index = findfirst(==(endpoint.port), node.input_names)
    !isnothing(data_index) && return (:data, data_index)
    parameter_index = findfirst(==(endpoint.port), node.parameter_names)
    !isnothing(parameter_index) && return (:parameter, parameter_index)
    throw(
        GraphConfigurationError(
            "Node '$(node.name)' has no input Port '$(endpoint.port)'",
        ),
    )
end

function _port_index(node::PreparedNode, endpoint::Endpoint, direction::Symbol)
    names = direction === :input ? node.input_names : node.output_names
    if isnothing(endpoint.port)
        length(names) == 1 || throw(
            GraphConfigurationError(
                "Node '$(node.name)' requires an explicit $direction Port name",
            ),
        )
        return 1
    end
    index = findfirst(==(endpoint.port), names)
    isnothing(index) && throw(
        GraphConfigurationError(
            "Node '$(node.name)' has no $direction Port '$(endpoint.port)'",
        ),
    )
    return index
end

function _require_compatible_formats(output, input, link::LinkSpec)
    compatible = output.element_type === input.element_type &&
                 typeof(output.layout) === typeof(input.layout) &&
                 output.shape == input.shape &&
                 output.schema == input.schema &&
                 output.rate == input.rate
    compatible || throw(
        GraphConfigurationError(
            "Link '$(link.output.node):$(link.output.port)' -> " *
            "'$(link.input.node):$(link.input.port)' has incompatible Port formats",
        ),
    )
    return nothing
end

function _external_name!(names::Vector{Symbol}, name::Symbol, direction::Symbol)
    name in names && throw(
        GraphConfigurationError(
            "External $direction Port name '$name' is ambiguous; declare unique external names",
        ),
    )
    push!(names, name)
    return name
end

function _prepared_index(
    endpoint::Endpoint,
    node_indices::Dict{Symbol,Int},
    authored_to_prepared::Vector{Int},
)
    authored_index = get(node_indices, endpoint.node, 0)
    iszero(authored_index) && throw(
        GraphConfigurationError("External endpoint names unknown Node '$(endpoint.node)'"),
    )
    return authored_to_prepared[authored_index]
end

# Route validation only needs the Nodes' Port declarations and formats.
# Avoid compiling it for every concrete tuple of prepared Algorithms.
Base.@nospecializeinfer function _prepare_routes(
    spec::GraphSpec,
    @nospecialize(nodes::Tuple),
    node_indices::Dict{Symbol,Int},
    authored_to_prepared::Vector{Int},
)
    input_routes = [Any[nothing for _ in node.input_names] for node in nodes]

    for link in spec.links
        source_index = authored_to_prepared[node_indices[link.output.node]]
        destination_index = authored_to_prepared[node_indices[link.input.node]]
        source = nodes[source_index]
        destination = nodes[destination_index]
        output_index = _port_index(source, link.output, :output)
        input_index = _port_index(destination, link.input, :input)
        isnothing(input_routes[destination_index][input_index]) || throw(
            GraphConfigurationError(
                "Node '$(destination.name)' input Port '$(destination.input_names[input_index])' " *
                "has more than one source",
            ),
        )
        _require_compatible_formats(
            source.output_formats[output_index],
            destination.input_formats[input_index],
            link,
        )
        input_routes[destination_index][input_index] =
            NodeOutput{source_index,output_index}()
    end

    external_input_names = Symbol[]
    external_data_names = Symbol[]
    external_data_formats = Any[]
    external_parameter_names = Symbol[]
    external_parameter_formats = Any[]
    external_parameter_routes = Any[]
    parameter_destinations = Set{Tuple{Int,Symbol}}()
    for endpoint in spec.inputs
        node_index = _prepared_index(endpoint, node_indices, authored_to_prepared)
        node = nodes[node_index]
        role, input_index = _external_input_port(node, endpoint)
        if role === :data
            isnothing(input_routes[node_index][input_index]) || throw(
                GraphConfigurationError(
                    "Node '$(node.name)' input Port '$(node.input_names[input_index])' " *
                    "has more than one source",
                ),
            )
            name = _external_name!(
                external_input_names,
                something(endpoint.external_name, node.input_names[input_index]),
                :input,
            )
            push!(external_data_names, name)
            push!(external_data_formats, node.input_formats[input_index])
            external_index = length(external_data_names)
            input_routes[node_index][input_index] = ExternalInput{external_index}()
        else
            destination = (node_index, node.parameter_names[input_index])
            destination in parameter_destinations && throw(
                GraphConfigurationError("Parameter Port $(node.name):$(destination[2]) has more than one external binding"),
            )
            push!(parameter_destinations, destination)
            name = _external_name!(
                external_input_names,
                something(endpoint.external_name, node.parameter_names[input_index]),
                :input,
            )
            push!(external_parameter_names, name)
            push!(external_parameter_formats, node.parameter_formats[input_index])
            push!(
                external_parameter_routes,
                ParameterInput{node_index,node.parameter_names[input_index]}(),
            )
        end
    end

    for (node_index, node) in pairs(nodes)
        for input_index in eachindex(node.input_names)
            isnothing(input_routes[node_index][input_index]) && throw(
                GraphConfigurationError(
                    "Node '$(node.name)' input Port " *
                    "'$(node.input_names[input_index])' has no source",
                ),
            )
        end
    end

    node_output_conditional = Vector{Vector{Bool}}(undef, length(nodes))
    for (node_index, node) in pairs(nodes)
        skipped_when_input_unavailable = any(
            route -> _route_conditional(route, node_output_conditional),
            input_routes[node_index],
        )
        node_output_conditional[node_index] = map(
            conditional -> conditional || skipped_when_input_unavailable,
            collect(node.output_conditional),
        )
    end

    output_routes = Any[]
    external_output_names = Symbol[]
    external_output_formats = Any[]
    external_output_conditional = Bool[]
    for endpoint in spec.outputs
        node_index = _prepared_index(endpoint, node_indices, authored_to_prepared)
        node = nodes[node_index]
        output_index = _port_index(node, endpoint, :output)
        _external_name!(
            external_output_names,
            something(endpoint.external_name, node.output_names[output_index]),
            :output,
        )
        push!(external_output_formats, node.output_formats[output_index])
        push!(external_output_conditional, node_output_conditional[node_index][output_index])
        push!(output_routes, NodeOutput{node_index,output_index}())
    end

    isempty(external_output_names) && throw(
        GraphConfigurationError("filter.graph requires at least one external output Port"),
    )

    prepared_input_routes = Tuple(Tuple(routes) for routes in input_routes)
    parameter_routes = NamedTuple{Tuple(external_parameter_names)}(
        Tuple(external_parameter_routes),
    )
    input_formats = NamedTuple{Tuple(external_data_names)}(Tuple(external_data_formats))
    parameter_formats = NamedTuple{Tuple(external_parameter_names)}(
        Tuple(external_parameter_formats),
    )
    output_formats = NamedTuple{Tuple(external_output_names)}(Tuple(external_output_formats))
    output_conditional = NamedTuple{Tuple(external_output_names)}(
        Tuple(external_output_conditional),
    )
    return (
        prepared_input_routes,
        parameter_routes,
        Tuple(output_routes),
        input_formats,
        parameter_formats,
        output_formats,
        output_conditional,
    )
end

_allocate_port_buffer(format) = zeros(format.element_type, format.shape)

# Progressive executors may replace an internal Node output with a frame-owned
# window. Ordinary output buffers need no work before a Node call.
@inline _prepare_node_output!(output, routed_metadata) = nothing
@inline _prepare_node_outputs!(::Tuple{}, routed_metadata) = nothing
@inline function _prepare_node_outputs!(outputs::Tuple, routed_metadata)
    _prepare_node_output!(first(outputs), routed_metadata)
    _prepare_node_outputs!(Base.tail(outputs), routed_metadata)
    return nothing
end

function _node_output_buffers(nodes::Tuple)
    return map(node -> map(_allocate_port_buffer, node.output_formats), nodes)
end

function _node_output_availability(nodes::Tuple)
    return map(node -> map(format -> Ref(false), node.output_formats), nodes)
end

function _node_output_metadata(nodes::Tuple)
    return map(node -> map(_ -> SampleMetadataSlot(), node.output_formats), nodes)
end

_node_inflight(nodes::Tuple) = map(_ -> Ref(false), nodes)

_dense_buffer_type(format) = Array{format.element_type,length(format.shape)}

function _external_buffer_type(formats::NamedTuple)
    return NamedTuple{keys(formats),Tuple{map(_dense_buffer_type, values(formats))...}}
end

function _external_buffers(formats::NamedTuple)
    return NamedTuple{keys(formats)}(map(_allocate_port_buffer, values(formats)))
end

function _external_output_metadata(formats::NamedTuple)
    return NamedTuple{keys(formats)}(map(_ -> SampleMetadataSlot(), values(formats)))
end

function _default_input_metadata(formats::NamedTuple)
    return NamedTuple{keys(formats)}(map(_ -> nothing, values(formats)))
end

"""
    prepare_graph(path; algorithms)
    prepare_graph(document::AbstractDict; algorithms)

Read a PipeWire filter-graph SPA-JSON file or accept its dictionary form
directly. Resolve its Node labels
against the explicit Algorithm declaration tuple, validate and prepare the
fixed DAG, allocate Graph-owned buffers, and precompile the dense checked call
shape. No registry or configuration lookup remains in repeated processing.
"""
prepare_graph(path::AbstractString; algorithms::Tuple) =
    _prepare_frame_graph(_load_graph(path), algorithms)

prepare_graph(document::AbstractDict; algorithms::Tuple) =
    _prepare_frame_graph(_graph_spec(document), algorithms)

function _prepare_frame_graph(spec::GraphSpec, algorithms)
    iszero(spec.cpu_execution.row_workers) || throw(
        GraphConfigurationError(
            "filter.graph execution cpu row_workers requires prepare_cpu_progressive_graph",
        ),
    )
    return _prepare_graph(spec; algorithms)
end

function _prepare_graph(spec::GraphSpec; algorithms::Tuple)
    _validate_algorithms(algorithms)
    node_indices = _node_indices(spec)
    order = _topological_order(spec, node_indices)
    authored_to_prepared = zeros(Int, length(order))
    prepared_nodes = Any[]
    for (prepared_index, authored_index) in pairs(order)
        authored_to_prepared[authored_index] = prepared_index
        node = spec.nodes[authored_index]
        declaration = _resolve_algorithm(algorithms, node.label)
        push!(prepared_nodes, _prepare_node(declaration, node))
    end
    nodes = Tuple(prepared_nodes)
    input_routes,
    parameter_routes,
    output_routes,
    input_formats,
    parameter_formats,
    output_formats,
    output_conditional = _prepare_routes(
        spec,
        nodes,
        node_indices,
        authored_to_prepared,
    )
    node_failure_types = map(_node_failure_type, nodes)
    graph = PreparedGraph(
        nodes,
        node_failure_types,
        _node_output_buffers(nodes),
        _node_output_availability(nodes),
        _node_output_metadata(nodes),
        _node_inflight(nodes),
        input_routes,
        parameter_routes,
        output_routes,
        input_formats,
        parameter_formats,
        output_formats,
        output_conditional,
        _external_output_metadata(output_formats),
        _default_input_metadata(input_formats),
        _graph_result_types(output_formats, node_failure_types),
    )

    input_type = _external_buffer_type(input_formats)
    output_type = _external_buffer_type(output_formats)
    precompile(
        process!,
        (output_type, typeof(graph), input_type),
    )
    precompile(
        process!,
        (
            output_type,
            typeof(graph),
            input_type,
            typeof(graph.default_input_metadata),
        ),
    )
    precompile(
        _process_admitted!,
        (
            output_type,
            typeof(graph),
            input_type,
            typeof(graph.default_input_metadata),
        ),
    )
    precompile(
        _process_candidate_admitted!,
        (
            output_type,
            typeof(graph),
            input_type,
            typeof(graph.default_input_metadata),
        ),
    )
    precompile(reset!, (typeof(graph),))
    return graph
end

@inline _route_value(::ExternalInput{Index}, inputs, node_outputs) where {Index} =
    getfield(inputs, Index)

@inline _route_value(::NodeOutput{NodeIndex,PortIndex}, inputs, node_outputs) where {
    NodeIndex,
    PortIndex,
} = getfield(getfield(node_outputs, NodeIndex), PortIndex)

@inline _route_available(::ExternalInput, node_output_available) = true

@inline _route_available(
    ::NodeOutput{NodeIndex,PortIndex},
    node_output_available,
) where {NodeIndex,PortIndex} = getfield(
    getfield(node_output_available, NodeIndex),
    PortIndex,
)[]

@inline _routes_available(::Tuple{}, node_output_available) = true

@inline function _routes_available(routes::Tuple, node_output_available)
    _route_available(first(routes), node_output_available) || return false
    return _routes_available(Base.tail(routes), node_output_available)
end

@inline _route_inputs(::Tuple{}, inputs, node_outputs) = ()

@inline function _route_inputs(routes::Tuple, inputs, node_outputs)
    return (
        _route_value(first(routes), inputs, node_outputs),
        _route_inputs(Base.tail(routes), inputs, node_outputs)...,
    )
end

@inline _metadata_value(slot::SampleMetadataSlot) =
    slot.present ? slot.value : nothing

@inline _admitted_metadata(slot::SampleMetadataSlot) =
    FilterGraphAlgorithms.AdmittedSampleMetadata(slot.present, slot.value)

@inline _route_metadata(
    ::ExternalInput{Index},
    input_metadata,
    node_output_metadata,
) where {Index} = FilterGraphAlgorithms._admitted_sample_metadata(
    getfield(input_metadata, Index),
)

@inline _route_metadata(
    ::NodeOutput{NodeIndex,PortIndex},
    input_metadata,
    node_output_metadata,
) where {NodeIndex,PortIndex} = _admitted_metadata(
    getfield(getfield(node_output_metadata, NodeIndex), PortIndex),
)

@inline _route_metadata_slot(
    ::NodeOutput{NodeIndex,PortIndex},
    node_output_metadata,
) where {NodeIndex,PortIndex} = getfield(
    getfield(node_output_metadata, NodeIndex),
    PortIndex,
)

@inline _route_metadata(::Tuple{}, input_metadata, node_output_metadata) = ()

@inline function _route_metadata(routes::Tuple, input_metadata, node_output_metadata)
    return (
        _route_metadata(first(routes), input_metadata, node_output_metadata),
        _route_metadata(Base.tail(routes), input_metadata, node_output_metadata)...,
    )
end

@inline function _store_metadata!(slot::SampleMetadataSlot, metadata::Nothing)
    slot.present = false
    return nothing
end

@inline function _store_metadata!(
    slot::SampleMetadataSlot,
    metadata::FilterGraphAlgorithms.SampleMetadata,
)
    slot.value = metadata
    slot.present = true
    return nothing
end

@inline function _store_metadata!(
    slot::SampleMetadataSlot,
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    slot.present = metadata.present
    slot.value = metadata.value
    return nothing
end

@inline _publish_node_metadata!(
    ::Tuple{},
    ::Tuple{},
    input_metadata,
    replace::Bool,
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
) = nothing

@inline function _publish_node_metadata!(
    output_metadata::Tuple,
    sources::Tuple,
    input_metadata,
    replace::Bool,
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    source = first(sources)
    value = replace ? metadata : (
        iszero(source) ? FilterGraphAlgorithms._NO_SAMPLE_METADATA :
        getfield(input_metadata, source)
    )
    _store_metadata!(first(output_metadata), value)
    return _publish_node_metadata!(
        Base.tail(output_metadata),
        Base.tail(sources),
        input_metadata,
        replace,
        metadata,
    )
end

@inline function _validate_node_result(::PreparedNode{D}, result) where {D}
    return FilterGraphAlgorithms._validate_process_result(D, result)
end

_node_process(::Nothing) = FilterGraphAlgorithms._process_admitted!
_node_process(processes::Tuple) = first(processes)
_remaining_node_processes(::Nothing) = nothing
_remaining_node_processes(processes::Tuple) = Base.tail(processes)

@inline function _process_nodes!(
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    inputs,
    input_metadata,
    all_node_outputs,
    all_node_output_available,
    all_node_output_metadata,
    deferred::Bool,
    rejection,
    node_index::Int,
    node_processes::Union{Nothing,Tuple}=nothing,
)
    return nothing, deferred
end

@inline function _process_nodes!(
    nodes::Tuple,
    node_outputs::Tuple,
    input_routes::Tuple,
    node_output_available::Tuple,
    node_output_metadata::Tuple,
    node_inflight::Tuple,
    inputs,
    input_metadata,
    all_node_outputs,
    all_node_output_available,
    all_node_output_metadata,
    deferred::Bool,
    rejection,
    node_index::Int,
    node_processes::Union{Nothing,Tuple}=nothing,
)
    node = first(nodes)
    outputs = first(node_outputs)
    routes = first(input_routes)
    if _routes_available(routes, all_node_output_available)
        node_result, routed_metadata = try
            routed_inputs = _route_inputs(routes, inputs, all_node_outputs)
            routed_metadata = _route_metadata(
                routes,
                input_metadata,
                all_node_output_metadata,
            )
            _prepare_node_outputs!(outputs, routed_metadata)
            process_result = _node_process(node_processes)(
                outputs...,
                node.prepared,
                routed_inputs...,
                routed_metadata...,
            )
            _validate_node_result(node, process_result), routed_metadata
        catch
            discard!(node.prepared.workspace)
            first(node_inflight)[] = false
            rethrow()
        end
        if node_result === nothing
            commit!(node.prepared.workspace)
            first(node_inflight)[] = false
            _set_available!(first(node_output_available))
            _publish_node_metadata!(
                first(node_output_metadata),
                node.output_metadata_sources,
                routed_metadata,
                false,
                FilterGraphAlgorithms._NO_SAMPLE_METADATA,
            )
        elseif node_result isa FilterGraphAlgorithms.AlgorithmProcessResult
            if node_result.complete
                commit!(node.prepared.workspace)
                first(node_inflight)[] = false
                _set_available!(first(node_output_available))
                replace_metadata = node_result.has_metadata
                replacement = replace_metadata ?
                    FilterGraphAlgorithms._admitted_sample_metadata(node_result.metadata) :
                    FilterGraphAlgorithms._NO_SAMPLE_METADATA
                _publish_node_metadata!(
                    first(node_output_metadata),
                    node.output_metadata_sources,
                    routed_metadata,
                    replace_metadata,
                    replacement,
                )
            else
                first(node_inflight)[] = true
                deferred = true
            end
        else
            discard!(node.prepared.workspace)
            first(node_inflight)[] = false
            _record_rejection!(rejection, node_index, node_result)
            return node_result, deferred
        end
    end
    return _process_nodes!(
        Base.tail(nodes),
        Base.tail(node_outputs),
        Base.tail(input_routes),
        Base.tail(node_output_available),
        Base.tail(node_output_metadata),
        Base.tail(node_inflight),
        inputs,
        input_metadata,
        all_node_outputs,
        all_node_output_available,
        all_node_output_metadata,
        deferred,
        rejection,
        node_index + 1,
        _remaining_node_processes(node_processes),
    )
end

@inline _set_available!(::Tuple{}) = nothing

@inline function _set_available!(available::Tuple)
    first(available)[] = true
    return _set_available!(Base.tail(available))
end

@inline _clear_available!(::Tuple{}) = nothing

@inline function _clear_available!(available::Tuple)
    _clear_port_available!(first(available))
    return _clear_available!(Base.tail(available))
end

@inline _clear_port_available!(::Tuple{}) = nothing

@inline function _clear_port_available!(available::Tuple)
    first(available)[] = false
    return _clear_port_available!(Base.tail(available))
end

@inline _reset_nodes!(::Tuple{}) = nothing

@inline function _reset_nodes!(nodes::Tuple)
    reset!(first(nodes).prepared)
    return _reset_nodes!(Base.tail(nodes))
end

@inline _discard_inflight_nodes!(::Tuple{}, ::Tuple{}) = nothing

@inline function _discard_inflight_nodes!(nodes::Tuple, inflight::Tuple)
    if first(inflight)[]
        discard!(first(nodes).prepared.workspace)
        first(inflight)[] = false
    end
    return _discard_inflight_nodes!(Base.tail(nodes), Base.tail(inflight))
end

@inline _clear_inflight!(::Tuple{}) = nothing

@inline function _clear_inflight!(inflight::Tuple)
    first(inflight)[] = false
    return _clear_inflight!(Base.tail(inflight))
end

@inline _validate_arrays(context, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

@inline function _validate_arrays(context, names::Tuple, formats::Tuple, arrays::Tuple)
    FilterGraphAlgorithms._validate_declared_array(
        first(formats), first(arrays), (context, first(names)),
    )
    return _validate_arrays(context, Base.tail(names), Base.tail(formats), Base.tail(arrays))
end

function _validate_external_arrays(kind::Symbol, formats::NamedTuple, arrays::NamedTuple)
    context = kind === :input ? "Graph input" : "Graph output"
    ordered = FilterGraphAlgorithms._order_port_values(context, Val(keys(formats)), arrays)
    _validate_arrays(context, keys(formats), values(formats), values(ordered))
    return ordered
end

@inline _validate_external_output_aliases!(::Tuple{}, ::Tuple{}) = nothing

@inline function _validate_external_output_aliases!(names::Tuple, arrays::Tuple)
    name = first(names)
    array = first(arrays)
    _validate_external_output_aliases!(name, array, Base.tail(names), Base.tail(arrays))
    return _validate_external_output_aliases!(Base.tail(names), Base.tail(arrays))
end

@inline _validate_external_output_aliases!(
    name::Symbol,
    array,
    ::Tuple{},
    ::Tuple{},
) = nothing

@inline function _validate_external_output_aliases!(
    name::Symbol,
    array,
    other_names::Tuple,
    other_arrays::Tuple,
)
    _external_output_overlap(array, first(other_arrays)) && throw(
        ArgumentError(
            "Graph external output buffers '$name' and '$(first(other_names))' must not alias",
        ),
    )
    return _validate_external_output_aliases!(
        name,
        array,
        Base.tail(other_names),
        Base.tail(other_arrays),
    )
end

const _ArrayRangeView = SubArray{
    <:Any,<:Any,<:Any,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}},
}
const _ArrayWrapper = Union{
    _ArrayRangeView,Base.ReshapedArray,Base.ReinterpretArray,PermutedDimsArray,
}
const _DenseHostArray = Union{Array,Memory}
const _ArrayOrWrapper = Union{_DenseHostArray,_ArrayWrapper}

@inline _external_output_overlap(a, b) = Base.mightalias(a, b)
@inline _external_output_overlap(a::AbstractArray, b::AbstractArray) =
    _external_parent_overlap(a, b)

# Independently wrapped host arrays can overlap even if their base addresses differ.
@inline function _external_output_overlap(a::_DenseHostArray, b::_DenseHostArray)
    (isempty(a) || isempty(b)) && return false
    (_external_array_storage(a) && _external_array_storage(b)) || return Base.mightalias(a, b)
    (iszero(Base.elsize(a)) || iszero(Base.elsize(b))) && return false
    GC.@preserve a b begin
        return _external_spans_overlap((UInt(pointer(a)), sizeof(a)), (UInt(pointer(b)), sizeof(b)))
    end
end

@inline function _external_output_overlap(a::_ArrayOrWrapper, b::_ArrayOrWrapper)
    (_external_array_storage(a) && _external_array_storage(b)) ||
        return _external_parent_overlap(a, b)
    return _external_array_ranges_overlap(a, b, false)
end

# For other wrappers, keep positives from Base, including shared index arrays.
# An ordinary host array parent bounds the wrapped storage.
@inline _external_parent_overlap(a, b) =
    Base.mightalias(a, b) || _external_array_ranges_overlap(a, b, true)

@inline function _external_array_ranges_overlap(a, b, allow_parent)
    (isempty(a) || isempty(b)) && return false
    GC.@preserve a b begin
        left = _external_array_span(a, allow_parent)
        right = _external_array_span(b, allow_parent)
        (left === nothing || right === nothing) && return false
        span_a, step_a, width_a = left
        span_b, step_b, width_b = right
        (iszero(last(span_a)) || iszero(last(span_b))) && return false
        _external_spans_overlap(span_a, span_b) || return false
        if width_a > 0 && width_a == width_b
            return _external_vector_overlap(span_a, span_b, step_a, step_b, width_a)
        end
        # For shared-parent range views, Base can compare the indices to
        # recognize disjoint slices within overlapping multidimensional bounds.
        if !allow_parent && a isa _ArrayRangeView && b isa _ArrayRangeView && parent(a) === parent(b)
            return Base.mightalias(a, b)
        end
        return true
    end
end

@inline function _external_spans_overlap((pa, na), (pb, nb))
    return pa <= pb ? pb - pa < na : pa - pb < nb
end

@inline function _external_vector_overlap((pa, na), (pb, nb), step_a, step_b, width)
    step_a == step_b == 1 && return true
    # Signed element coordinates can cross the full pointer range. Widen before
    # normalizing negative strides or multiplying strides and Bézout coefficients.
    Wide = widen(Int)
    offset, remainder = divrem(Wide(pb) - Wide(pa), width)
    iszero(remainder) || return true
    sa = max(one(Wide), abs(Wide(step_a)))
    sb = max(one(Wide), abs(Wide(step_b)))
    divisor, coefficient, _ = gcdx(sa, sb)
    iszero(rem(offset, divisor)) || return false
    # Solve x ≡ 0 mod sa and x ≡ offset mod sb in the common interval.
    modulus = div(sb, divisor)
    period = sa * modulus
    first_match = sa * mod(div(offset, divisor) * coefficient, modulus)
    low = max(zero(Wide), offset)
    high = min(div(na - width, width), offset + div(nb - width, width))
    return low + mod(first_match - low, period) <= high
end

@inline _external_array_storage(a) = false
@inline _external_array_storage(a::_ArrayWrapper) = _external_array_storage(parent(a))
# Union element tags can occupy separate storage.
@inline _external_array_storage(a::_DenseHostArray) = !Base.isbitsunion(eltype(a))

@inline _external_array_strides(a) = nothing
@inline _external_array_strides(a::_DenseHostArray) = strides(a)
@inline function _external_array_strides(a::Union{_ArrayRangeView,PermutedDimsArray})
    (a isa StridedArray || _external_array_strides(parent(a)) !== nothing) || return nothing
    return strides(a)
end
@inline function _external_array_strides(a::Union{Base.ReshapedArray,Base.ReinterpretArray})
    a isa StridedArray && return strides(a)
    _external_array_strides(parent(a)) === nothing && return nothing
    try
        return strides(a)
    catch error
        error isa ArgumentError || rethrow()
        return nothing
    end
end

# Span metadata includes the stride and element width only for affine vectors.
@inline _external_array_span(a, allow_parent) = nothing
@inline _external_array_span(a::AbstractArray, allow_parent) =
    allow_parent ? _external_parent_span(a, false) : nothing
@inline function _external_array_span(a::_DenseHostArray, allow_parent)
    _external_array_storage(a) || return nothing
    return (UInt(pointer(a)), sizeof(a)), 1, ndims(a) == 1 ? Base.elsize(a) : 0
end

@inline function _external_parent_span(a, allow_parent)
    layout = _external_array_span(parent(a), allow_parent)
    return layout === nothing ? nothing : (first(layout), 0, 0)
end

@inline function _external_array_span(a::_ArrayWrapper, allow_parent)
    steps = _external_array_storage(a) ? _external_array_strides(a) : nothing
    steps === nothing && return _external_parent_span(a, allow_parent)
    low = high = 0
    for (extent, stride) in zip(size(a), steps)
        offset = (extent - 1) * stride
        low += min(0, offset)
        high += max(0, offset)
    end
    bytes = Base.elsize(a)
    span = (UInt(pointer(a) + low * bytes), (high - low + 1) * bytes)
    step, width = ndims(a) == 1 ? (only(steps), bytes) : (0, 0)
    return span, step, width
end

@inline _validate_metadata_values(::Tuple{}, ::Tuple{}) = nothing

@inline function _validate_metadata_values(names::Tuple, metadata::Tuple)
    value = first(metadata)
    (isnothing(value) || value isa FilterGraphAlgorithms.SampleMetadata) || throw(
        ArgumentError("Graph input metadata Port '$(first(names))' must be SampleMetadata or nothing"),
    )
    return _validate_metadata_values(Base.tail(names), Base.tail(metadata))
end

function _validate_external_metadata(formats::NamedTuple, metadata::NamedTuple)
    ordered = FilterGraphAlgorithms._order_port_values(
        "Graph input metadata", Val(keys(formats)), metadata,
    )
    _validate_metadata_values(keys(formats), values(ordered))
    return ordered
end

@inline _copy_external_outputs!(::Tuple{}, ::Tuple{}, node_outputs, available) = ()

@inline function _copy_external_outputs!(
    routes::Tuple,
    destination::Tuple,
    node_outputs,
    available,
)
    route = first(routes)
    present = _route_available(route, available)
    present && copyto!(first(destination), _route_value(route, destination, node_outputs))
    return (
        present,
        _copy_external_outputs!(
            Base.tail(routes),
            Base.tail(destination),
            node_outputs,
            available,
        )...,
    )
end

@inline _copy_external_metadata!(::Tuple{}, ::Tuple{}, node_metadata, available) = nothing

@inline function _copy_external_metadata!(
    routes::Tuple,
    destination::Tuple,
    node_metadata,
    available,
)
    route = first(routes)
    if _route_available(route, available)
        source = _route_metadata_slot(route, node_metadata)
        _store_metadata!(first(destination), _metadata_value(source))
    end
    return _copy_external_metadata!(
        Base.tail(routes),
        Base.tail(destination),
        node_metadata,
        available,
    )
end

@inline _unavailable_external_outputs(::Tuple{}) = ()

@inline function _unavailable_external_outputs(formats::Tuple)
    return (false, _unavailable_external_outputs(Base.tail(formats))...)
end

@inline _metadata_values(::Tuple{}) = ()

@inline function _metadata_values(slots::Tuple)
    return (_metadata_value(first(slots)), _metadata_values(Base.tail(slots))...)
end

@inline _admitted_metadata_values(::Tuple{}) = ()

@inline function _admitted_metadata_values(slots::Tuple)
    return (
        _admitted_metadata(first(slots)),
        _admitted_metadata_values(Base.tail(slots))...,
    )
end

@inline function _output_metadata_admitted(graph::PreparedGraph)
    return NamedTuple{keys(graph.output_formats)}(
        _admitted_metadata_values(values(graph.published_metadata)),
    )
end

"""
    output_metadata(graph::PreparedGraph)

Return the retained sample metadata for the Graph's external output buffers.
The availability returned by the most recent `process!` call determines which
entries are newly publishable; an unavailable output retains its prior buffer
and metadata.
"""
function output_metadata(graph::PreparedGraph)
    return NamedTuple{keys(graph.output_formats)}(
        _metadata_values(values(graph.published_metadata)),
    )
end

@inline function _discard_inflight!(graph::PreparedGraph)
    _discard_inflight_nodes!(graph.nodes, graph.node_inflight)
    _clear_available!(graph.node_output_available)
    return nothing
end

@inline _publish_cpu_outputs!(outputs::NamedTuple, graph, node_outputs) =
    _copy_external_outputs!(graph.output_routes, values(outputs), node_outputs, graph.node_output_available)

@inline function _process_candidate_admitted!(
    outputs::NamedTuple,
    graph::PreparedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
)
    return _process_candidate_admitted!(
        outputs,
        graph,
        inputs,
        input_metadata,
        graph.result_types,
    )
end

function _process_candidate_admitted!(
    outputs::Union{Nothing,NamedTuple},
    graph::PreparedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
    ::GraphResultTypes{Availability,Failure},
    nodes=graph.nodes,
    node_outputs=graph.node_outputs,
)::Union{
    NamedTuple{(:availability, :terminal),Tuple{Availability,Bool}},
    Failure,
} where {Availability,Failure}
    _clear_available!(graph.node_output_available)
    _clear_rejection!(graph.rejection)

    result, deferred = try
        _process_nodes!(
            nodes,
            node_outputs,
            graph.input_routes,
            graph.node_output_available,
            graph.node_output_metadata,
            graph.node_inflight,
            inputs,
            input_metadata,
            node_outputs,
            graph.node_output_available,
            graph.node_output_metadata,
            false,
            graph.rejection,
            1,
        )
    catch
        _clear_available!(graph.node_output_available)
        rethrow()
    end
    result = result::Union{Nothing,Failure}

    if result !== nothing
        _rejection_inputs!(graph.rejection, input_metadata)
        _clear_available!(graph.node_output_available)
        return result
    end

    deferred && return (
        availability=NamedTuple{keys(graph.output_formats)}(
            _unavailable_external_outputs(values(graph.output_formats)),
        ),
        terminal=false,
    )

    availability = try
        copied = _publish_cpu_outputs!(outputs, graph, node_outputs)
        _copy_external_metadata!(
            graph.output_routes,
            values(graph.published_metadata),
            graph.node_output_metadata,
            graph.node_output_available,
        )
        copied
    catch
        _clear_available!(graph.node_output_available)
        rethrow()
    end
    return (
        availability=NamedTuple{keys(graph.output_formats)}(availability),
        terminal=true,
    )
end

@inline function _process_admitted!(
    outputs::NamedTuple,
    graph::PreparedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
)
    return _process_admitted!(
        outputs,
        graph,
        inputs,
        input_metadata,
        graph.result_types,
    )
end

function _process_admitted!(
    outputs::NamedTuple,
    graph::PreparedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
    result_types::GraphResultTypes{Availability,Failure},
)::Union{Availability,Failure} where {Availability,Failure}
    result = _process_candidate_admitted!(
        outputs,
        graph,
        inputs,
        input_metadata,
        result_types,
    )
    result isa NamedTuple || return result
    return result.availability
end

"""
    process!(outputs, graph::PreparedGraph, inputs[, input_metadata])

Process one Graph cycle and return a Boolean NamedTuple describing which
external output Ports were published. A required consumer of an absent
conditional Port is skipped, while independent ready Nodes continue. If any
Node defers, the Graph retains caller output buffers and output metadata and
returns every external Port unavailable until the logical sample completes.
Each terminal Node commits its Workspace before the next Node runs. Deferred
Nodes retain their candidate Workspace until their terminal completion. An
expected Algorithm failure instead returns its declared enum, discards that
Node's candidate Workspace, and leaves caller outputs unchanged.
`input_metadata` is a NamedTuple keyed like `inputs`; each value is
`SampleMetadata` or `nothing`. Omitting it supplies no metadata. Inputs, outputs,
and metadata must have exactly their declared Port names; field order is
arbitrary. Returned availability and published metadata retain declared order.

A source-only Graph accepts an empty `NamedTuple()` for `inputs`; its host
trigger defines the Graph cycle.
"""
function process!(
    outputs::NamedTuple,
    graph::PreparedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
)
    inputs = _validate_external_arrays(:input, graph.input_formats, inputs)
    outputs = _validate_external_arrays(:output, graph.output_formats, outputs)
    _validate_external_output_aliases!(keys(graph.output_formats), values(outputs))
    input_metadata = _validate_external_metadata(graph.input_formats, input_metadata)
    return _process_admitted!(outputs, graph, inputs, input_metadata)
end

function process!(outputs::NamedTuple, graph::PreparedGraph, inputs::NamedTuple)
    return process!(outputs, graph, inputs, graph.default_input_metadata)
end

"""
    reset!(graph::PreparedGraph)

Restore every Node Workspace to its initial state, discard in-flight work and
clear rejection diagnostics. Retained publication metadata remains unchanged.
Serialize this operation with processing and control updates.
"""
function reset!(graph::PreparedGraph)
    _reset_nodes!(graph.nodes)
    _clear_inflight!(graph.node_inflight)
    _clear_available!(graph.node_output_available)
    _clear_rejection!(graph.rejection)
    return graph
end
