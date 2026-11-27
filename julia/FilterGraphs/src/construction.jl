# Native descriptions use the same keys and prepared representation as SPA-JSON.
# Only structural records are normalized; scientific values keep their Julia types.
function _construction_record(value, context)
    value isa NamedTuple && return Dict{String,Any}(
        String(name) => getfield(value, name) for name in keys(value))
    if value isa AbstractDict
        all(key -> key isa Union{Symbol,AbstractString}, keys(value)) ||
            throw(GraphConfigurationError("$context keys must be strings or Symbols"))
        result = Dict{String,Any}()
        for (key, item) in value
            name = String(key)
            haskey(result, name) && throw(GraphConfigurationError("$context repeats key '$name'"))
            result[name] = item
        end
        return result
    end
    throw(GraphConfigurationError("$context must be a NamedTuple or dictionary"))
end

function _construction_list(value, context)
    value isa Union{Tuple,AbstractVector} ||
        throw(GraphConfigurationError("$context must be a tuple or vector"))
    return value
end

function _construction_name(value, context)
    value isa Union{Symbol,AbstractString} ||
        throw(GraphConfigurationError("$context must be a string or Symbol"))
    name = String(value)
    !isempty(strip(name)) || throw(GraphConfigurationError("$context must be nonempty"))
    return name
end

function _construction_configuration(value)
    value isa Union{NamedTuple,AbstractDict} &&
        return _construction_record(value, "Node config")
    # Generated @configuration values are ordinary structs. Read their fields,
    # not arbitrary property accessors, and let declaration validation own names.
    type = typeof(value)
    isstructtype(type) && !(value isa Union{AbstractArray,AbstractString,Function,Type}) ||
        throw(GraphConfigurationError("Node config must be a record or configuration struct"))
    return Dict{String,Any}(String(name) => getfield(value, name) for name in fieldnames(type))
end

function _construction_endpoints(value, context)
    entries = value isa NamedTuple ? pairs(value) : _construction_list(value, context)
    endpoints = Endpoint[]
    for entry in entries
        if entry isa Pair
            name = _construction_name(first(entry), "$context name")
            endpoint = last(entry)
            endpoint isa AbstractString || throw(GraphConfigurationError(
                "$context endpoint must be a string"))
            parsed = Endpoint(endpoint)
            push!(endpoints, Endpoint(parsed.node, parsed.port, Symbol(name)))
        elseif entry isa AbstractString
            push!(endpoints, Endpoint(entry))
        else
            push!(endpoints, _external_endpoint(_construction_record(entry, context), context))
        end
    end
    return endpoints
end

function _construction_link(value)
    if value isa Pair
        first(value) isa AbstractString && last(value) isa AbstractString ||
            throw(GraphConfigurationError("Graph link endpoints must be strings"))
        return LinkSpec(Endpoint(first(value)), Endpoint(last(value)))
    end
    return _link_spec(_construction_record(value, "Graph link"))
end

function _construction_exports(endpoints, context)
    result = Dict{Symbol,Endpoint}()
    for endpoint in endpoints
        name = endpoint.external_name
        isnothing(name) && throw(GraphConfigurationError(
            "$context requires explicitly named exported Ports"))
        haskey(result, name) && throw(GraphConfigurationError(
            "$context repeats exported Port '$name'"))
        result[name] = endpoint
    end
    return result
end

function _construction_graph(description, prefix, ancestors, depth, boundaries)
    depth <= 64 || throw(GraphConfigurationError("subgraph nesting exceeds 64 levels"))
    description in ancestors && throw(GraphConfigurationError("recursive subgraph description"))
    push!(ancestors, description)
    try
        document = _construction_record(description, "Graph description")
        graph = haskey(document, "filter.graph") ?
            _construction_record(document["filter.graph"], "filter.graph") : document
        _reject_unknown_keys(graph, ("nodes", "links", "inputs", "outputs", "execution"), "filter.graph")
        !isempty(prefix) && haskey(graph, "execution") && throw(GraphConfigurationError(
            "subgraph execution belongs to the outer Graph"))
        authored = _construction_list(get(graph, "nodes", ()), "Graph nodes")
        isempty(authored) && throw(GraphConfigurationError("filter.graph requires Nodes"))
        nodes = NodeSpec[]
        links = LinkSpec[]
        local_names = Set{Symbol}()
        children = Dict{Symbol,Tuple{Dict{Symbol,Endpoint},Dict{Symbol,Endpoint}}}()
        for entry in authored
            node = _construction_record(entry, "Graph Node")
            name = _construction_name(get(node, "name", nothing), "Node name")
            occursin(':', name) && throw(GraphConfigurationError("Node name cannot contain ':'"))
            local_name = Symbol(name)
            local_name in local_names && throw(GraphConfigurationError("duplicate Node '$name'"))
            push!(local_names, local_name)
            if haskey(node, "graph")
                _reject_unknown_keys(node, ("name", "graph"), "subgraph Node")
                child = _construction_graph(node["graph"], prefix * name * ".", ancestors, depth + 1, boundaries)
                push!(boundaries, child)
                children[local_name] = (
                    _construction_exports(child.inputs, "subgraph '$name' inputs"),
                    _construction_exports(child.outputs, "subgraph '$name' outputs"))
                append!(nodes, child.nodes)
                append!(links, child.links)
            else
                node["type"] = get(node, "type", "ndarray")
                node["name"] = prefix * name
                haskey(node, "config") &&
                    (node["config"] = _construction_configuration(node["config"]))
                haskey(node, "props") &&
                    (node["props"] = _construction_record(node["props"], "Node props"))
                push!(nodes, _node_spec(node))
            end
        end
        function resolve(endpoint, direction)
            endpoint.node in local_names || throw(GraphConfigurationError(
                "Graph endpoint names unknown Node '$(endpoint.node)'"))
            if haskey(children, endpoint.node)
                exports = children[endpoint.node][direction === :input ? 1 : 2]
                haskey(exports, endpoint.port) || throw(GraphConfigurationError(
                    "subgraph '$(endpoint.node)' has no $direction Port '$(endpoint.port)'"))
                target = exports[endpoint.port]
                return Endpoint(target.node, target.port, endpoint.external_name)
            end
            return Endpoint(Symbol(prefix * String(endpoint.node)), endpoint.port, endpoint.external_name)
        end
        for entry in _construction_list(get(graph, "links", ()), "Graph links")
            link = _construction_link(entry)
            push!(links, LinkSpec(resolve(link.output, :output), resolve(link.input, :input)))
        end
        inputs = map(endpoint -> resolve(endpoint, :input),
            _construction_endpoints(get(graph, "inputs", ()), "Graph inputs"))
        outputs = map(endpoint -> resolve(endpoint, :output),
            _construction_endpoints(get(graph, "outputs", ()), "Graph outputs"))
        execution = get(graph, "execution", (;))
        execution_record = _construction_record(execution, "Graph execution")
        if haskey(execution_record, "cpu")
            execution_record["cpu"] = _construction_record(execution_record["cpu"], "Graph execution cpu")
        end
        return GraphSpec(nodes, links, inputs, outputs, _cpu_execution_spec(execution_record))
    finally
        delete!(ancestors, description)
    end
end

function _validate_construction_boundaries(graph, boundaries)
    prepared = Dict(node.name => node for node in graph.nodes)
    for spec in boundaries
        indices = _node_indices(spec)
        order = _topological_order(spec, indices)
        authored_to_prepared = zeros(Int, length(order))
        for (index, authored) in pairs(order)
            authored_to_prepared[authored] = index
        end
        nodes = Tuple(prepared[spec.nodes[index].name] for index in order)
        # Check the complete closed component, including unused exports. Reuse
        # route validation without preparing scientific Plans a second time.
        _prepare_routes(spec, nodes, indices, authored_to_prepared)
    end
    return graph
end

"""
    prepare_graph(description::Union{NamedTuple,AbstractDict}; algorithms)

Prepare an ordinary CPU Graph from Julia records using the SPA-JSON field names:
`nodes`, `links`, `inputs`, `outputs`, and optional `execution`. Each Node supplies
`name`, `label`, optional `config` and `props`; `type="ndarray"` is implicit.
Configuration accepts a NamedTuple, dictionary, or ordinary configuration struct.
Links are `"source:output" => "sink:input"` pairs or records with `output`/`input`.
External Ports can be endpoint strings, named pairs, or a NamedTuple of endpoints.

A Node with `name` and `graph` instantiates a reusable description. Its inputs
and outputs must have explicit exported names. Nested Nodes acquire dot-qualified
names; expansion precedes the existing topology and exact-format checks. Each
instance owns independent Workspaces. Scheduling belongs to the outer Graph.

Preparation deep-copies the description, including mutable configuration arrays,
before normalization. Separate preparations cannot alias caller-owned config
storage through this entry point. Custom types follow Julia's `deepcopy` contract.
No resource paths are rebased; Algorithm-specific path interpretation is retained.
Processing, failure, publication, Property/Parameter updates, and accelerator
qualification are those of the existing prepared Graph. Native construction
itself does not establish capture support.
"""
function prepare_graph(description::Union{NamedTuple,AbstractDict}; algorithms::Tuple)
    boundaries = GraphSpec[]
    spec = _construction_graph(deepcopy(description), "", Base.IdSet{Any}(), 1, boundaries)
    iszero(spec.cpu_execution.row_workers) || throw(GraphConfigurationError(
        "filter.graph execution cpu row_workers requires prepare_cpu_progressive_graph"))
    graph = _prepare_graph(spec; algorithms)
    return _validate_construction_boundaries(graph, boundaries)
end
