"""Invalid SPA-JSON Graph configuration detected before preparation."""
struct GraphConfigurationError <: Exception
    message::String
end

Base.showerror(io::IO, error::GraphConfigurationError) = print(io, error.message)

struct Endpoint
    node::Symbol
    port::Union{Nothing,Symbol}
    external_name::Union{Nothing,Symbol}
end

Endpoint(node::Symbol, port::Union{Nothing,Symbol}) = Endpoint(node, port, nothing)

function Endpoint(text::AbstractString)
    parts = split(text, ':'; limit=2)
    if length(parts) == 1
        isempty(parts[1]) && throw(GraphConfigurationError("Graph endpoint node is empty"))
        return Endpoint(Symbol(parts[1]), nothing)
    end
    all(!isempty, parts) || throw(
        GraphConfigurationError("Graph endpoint must be written as node or node:port"),
    )
    return Endpoint(Symbol(parts[1]), Symbol(parts[2]))
end

struct NodeSpec{C<:AbstractDict,P<:AbstractDict}
    name::Symbol
    label::String
    configuration::C
    properties::P
end

# Retain parsed values in containers with a bounded union of their actual types.
_configuration_value(value) = value
function _configuration_value(value::AbstractVector)
    items = map(_configuration_value, value)
    Element = foldl((type, item) -> Union{type,typeof(item)}, items; init=Union{})
    return Vector{Element}(items)
end
_configuration_value(value::AbstractDict) = _configuration_fields(value)
function _configuration_fields(value::AbstractDict)
    items = map(_configuration_value, collect(values(value)))
    Value = foldl((type, item) -> Union{type,typeof(item)}, items; init=Union{})
    return Dict{String,Value}(String(name) => item for (name, item) in zip(keys(value), items))
end

struct LinkSpec
    output::Endpoint
    input::Endpoint
end

"""Deployment-owned CPU scheduling for one prepared Graph."""
struct CpuExecutionSpec
    row_workers::Int
    worker_thread_ids::Union{Nothing,Vector{Int}}
    matrix_layout::Symbol
    configured::Bool
end

struct GraphSpec
    nodes::Vector{NodeSpec}
    links::Vector{LinkSpec}
    inputs::Vector{Endpoint}
    outputs::Vector{Endpoint}
    cpu_execution::CpuExecutionSpec
end

mutable struct SpaJsonParser
    bytes::Vector{UInt8}
    index::Int
end

_spa_done(parser::SpaJsonParser) = parser.index > length(parser.bytes)
_spa_byte(parser::SpaJsonParser) = parser.bytes[parser.index]

function _spa_skip!(parser::SpaJsonParser)
    while !_spa_done(parser)
        byte = _spa_byte(parser)
        if isspace(Char(byte)) || byte == UInt8(',')
            parser.index += 1
        elseif byte == UInt8('#')
            while !_spa_done(parser) && _spa_byte(parser) != UInt8('\n')
                parser.index += 1
            end
        elseif byte == UInt8('/') && parser.index < length(parser.bytes) &&
               parser.bytes[parser.index + 1] == UInt8('/')
            parser.index += 2
            while !_spa_done(parser) && _spa_byte(parser) != UInt8('\n')
                parser.index += 1
            end
        else
            break
        end
    end
    return parser
end

function _spa_string!(parser::SpaJsonParser)
    parser.index += 1
    result = UInt8[]
    while !_spa_done(parser)
        byte = _spa_byte(parser)
        parser.index += 1
        byte == UInt8('"') && return String(result)
        if byte == UInt8('\\')
            _spa_done(parser) && throw(
                GraphConfigurationError("unterminated SPA-JSON escape"),
            )
            escaped = _spa_byte(parser)
            parser.index += 1
            byte = escaped == UInt8('n') ? UInt8('\n') :
                   escaped == UInt8('t') ? UInt8('\t') : escaped
        end
        push!(result, byte)
    end
    throw(GraphConfigurationError("unterminated SPA-JSON string"))
end

function _spa_bare!(parser::SpaJsonParser)
    start = parser.index
    delimiters = (UInt8('{'), UInt8('}'), UInt8('['), UInt8(']'), UInt8('='), UInt8(','))
    while !_spa_done(parser)
        byte = _spa_byte(parser)
        (isspace(Char(byte)) || byte in delimiters) && break
        parser.index += 1
    end
    parser.index == start && throw(GraphConfigurationError("expected SPA-JSON value"))
    token = String(parser.bytes[start:parser.index - 1])
    token == "true" && return true
    token == "false" && return false
    token == "null" && return nothing
    integer = tryparse(Int, token)
    !isnothing(integer) && return integer
    unsigned_integer = tryparse(UInt64, token)
    !isnothing(unsigned_integer) && return unsigned_integer
    floating = tryparse(Float64, token)
    !isnothing(floating) && return floating
    return token
end

function _spa_value!(parser::SpaJsonParser)
    _spa_skip!(parser)
    _spa_done(parser) && throw(GraphConfigurationError("missing SPA-JSON value"))
    byte = _spa_byte(parser)
    byte == UInt8('{') && return _spa_object!(parser)
    byte == UInt8('[') && return _spa_array!(parser)
    byte == UInt8('"') && return _spa_string!(parser)
    return _spa_bare!(parser)
end

function _spa_key!(parser::SpaJsonParser)
    _spa_skip!(parser)
    _spa_done(parser) && throw(GraphConfigurationError("missing SPA-JSON key"))
    value = _spa_byte(parser) == UInt8('"') ? _spa_string!(parser) : _spa_bare!(parser)
    value isa String || throw(GraphConfigurationError("SPA-JSON keys must be strings"))
    return value
end

function _spa_assignment!(parser::SpaJsonParser, object::Dict{String,Any})
    key = _spa_key!(parser)
    _spa_skip!(parser)
    (!_spa_done(parser) && _spa_byte(parser) == UInt8('=')) || throw(
        GraphConfigurationError("SPA-JSON key '$key' requires '='"),
    )
    parser.index += 1
    haskey(object, key) && throw(GraphConfigurationError("duplicate SPA-JSON key '$key'"))
    object[key] = _spa_value!(parser)
    return object
end

function _spa_object!(parser::SpaJsonParser)
    parser.index += 1
    object = Dict{String,Any}()
    while true
        _spa_skip!(parser)
        _spa_done(parser) && throw(GraphConfigurationError("unterminated SPA-JSON object"))
        if _spa_byte(parser) == UInt8('}')
            parser.index += 1
            return object
        end
        _spa_assignment!(parser, object)
    end
end

function _spa_array!(parser::SpaJsonParser)
    parser.index += 1
    values = Any[]
    while true
        _spa_skip!(parser)
        _spa_done(parser) && throw(GraphConfigurationError("unterminated SPA-JSON array"))
        if _spa_byte(parser) == UInt8(']')
            parser.index += 1
            return values
        end
        push!(values, _spa_value!(parser))
    end
end

function _parse_spa_json(text::AbstractString)
    parser = SpaJsonParser(collect(codeunits(text)), 1)
    _spa_skip!(parser)
    document = if !_spa_done(parser) && _spa_byte(parser) == UInt8('{')
        _spa_object!(parser)
    else
        assignments = Dict{String,Any}()
        while true
            _spa_skip!(parser)
            _spa_done(parser) && break
            _spa_assignment!(parser, assignments)
        end
        assignments
    end
    _spa_skip!(parser)
    _spa_done(parser) || throw(GraphConfigurationError("trailing SPA-JSON content"))
    return document
end

function _reject_unknown_keys(table, allowed, description)
    for key in keys(table)
        String(key) in allowed || throw(
            GraphConfigurationError("$description has unknown field '$key'"),
        )
    end
    return table
end

function _required_string(table, key, description)
    value = get(table, key, nothing)
    value isa String && !isempty(value) || throw(
        GraphConfigurationError("$description requires nonempty string '$key'"),
    )
    return value
end

function _node_spec(value)
    value isa AbstractDict || throw(GraphConfigurationError("Graph node must be an object"))
    _reject_unknown_keys(value, ("type", "name", "label", "config", "props"), "Graph node")
    _required_string(value, "type", "Graph node") == "ndarray" || throw(
        GraphConfigurationError("Graph node type must be 'ndarray'"),
    )
    configuration = get(value, "config", Dict{String,Any}())
    properties = get(value, "props", Dict{String,Any}())
    configuration isa AbstractDict || throw(
        GraphConfigurationError("Graph node config must be an object"),
    )
    properties isa AbstractDict || throw(
        GraphConfigurationError("Graph node props must be an object"),
    )
    return NodeSpec(
        Symbol(_required_string(value, "name", "Graph node")),
        _required_string(value, "label", "Graph node"),
        _configuration_fields(configuration),
        _configuration_fields(properties),
    )
end

function _link_spec(value)
    value isa AbstractDict || throw(GraphConfigurationError("Graph link must be an object"))
    _reject_unknown_keys(value, ("output", "input"), "Graph link")
    return LinkSpec(
        Endpoint(_required_string(value, "output", "Graph link")),
        Endpoint(_required_string(value, "input", "Graph link")),
    )
end

_external_endpoint(value::AbstractString, description) = Endpoint(value)

function _external_endpoint(value::AbstractDict, description)
    _reject_unknown_keys(value, ("name", "endpoint"), "$description entry")
    name = _required_string(value, "name", "$description entry")
    endpoint_text = _required_string(value, "endpoint", "$description entry")
    isempty(strip(name)) && throw(
        GraphConfigurationError("$description entry name must not be blank"),
    )
    isempty(strip(endpoint_text)) && throw(
        GraphConfigurationError("$description entry endpoint must not be blank"),
    )
    endpoint = Endpoint(endpoint_text)
    return Endpoint(endpoint.node, endpoint.port, Symbol(name))
end

function _external_endpoint(value, description)
    throw(GraphConfigurationError("$description entries must be endpoint strings or named endpoint objects"))
end

function _endpoint_list(value, description)
    value isa AbstractVector || throw(GraphConfigurationError("$description must be an array"))
    return Endpoint[
        _external_endpoint(item, description)
        for item in value
    ]
end

function _cpu_execution_spec(value)
    value isa AbstractDict || throw(
        GraphConfigurationError("filter.graph execution must be an object"),
    )
    _reject_unknown_keys(value, ("cpu",), "filter.graph execution")
    cpu = get(value, "cpu", Dict{String,Any}())
    cpu isa AbstractDict || throw(
        GraphConfigurationError("filter.graph execution cpu must be an object"),
    )
    _reject_unknown_keys(
        cpu,
        ("row_workers", "worker_thread_ids", "matrix_layout"),
        "filter.graph execution cpu",
    )
    ids = if haskey(cpu, "worker_thread_ids")
        values = cpu["worker_thread_ids"]
        values isa AbstractVector || throw(
            GraphConfigurationError("worker_thread_ids must be an array"),
        )
        all(id -> id isa Integer && !(id isa Bool) && 0 < id <= typemax(Int), values) ||
            throw(GraphConfigurationError("worker_thread_ids must contain positive integers"))
        Int[id for id in values]
    else
        nothing
    end
    raw_workers = get(cpu, "row_workers", isnothing(ids) ? 0 : length(ids))
    raw_workers isa Integer && !(raw_workers isa Bool) &&
        0 <= raw_workers <= typemax(Int) || throw(
        GraphConfigurationError("row_workers must be a nonnegative integer"),
    )
    workers = Int(raw_workers)
    matrix_layout = get(cpu, "matrix_layout", "shared")
    matrix_layout in ("shared", "sharded") || throw(
        GraphConfigurationError("matrix_layout must be shared or sharded"),
    )
    isnothing(ids) || length(ids) == workers || throw(
        GraphConfigurationError("worker_thread_ids count must equal row_workers"),
    )
    return CpuExecutionSpec(
        workers,
        ids,
        Symbol(matrix_layout),
        haskey(cpu, "row_workers") || haskey(cpu, "worker_thread_ids"),
    )
end

function _graph_spec(document::AbstractDict)
    graph = get(document, "filter.graph", document)
    graph isa AbstractDict || throw(GraphConfigurationError("filter.graph must be an object"))
    _reject_unknown_keys(
        graph,
        ("nodes", "links", "inputs", "outputs", "execution"),
        "filter.graph",
    )
    node_values = get(graph, "nodes", Any[])
    link_values = get(graph, "links", Any[])
    node_values isa AbstractVector || throw(
        GraphConfigurationError("filter.graph nodes must be an array"),
    )
    link_values isa AbstractVector || throw(
        GraphConfigurationError("filter.graph links must be an array"),
    )
    isempty(node_values) && throw(GraphConfigurationError("filter.graph requires Nodes"))
    return GraphSpec(
        NodeSpec[_node_spec(value) for value in node_values],
        LinkSpec[_link_spec(value) for value in link_values],
        _endpoint_list(get(graph, "inputs", Any[]), "filter.graph inputs"),
        _endpoint_list(get(graph, "outputs", Any[]), "filter.graph outputs"),
        _cpu_execution_spec(get(graph, "execution", Dict{String,Any}())),
    )
end

_load_graph(path::AbstractString) = _graph_spec(_parse_spa_json(read(path, String)))
