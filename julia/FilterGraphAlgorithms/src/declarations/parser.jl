_declaration_statements(block::Expr) =
    filter(statement -> !(statement isa LineNumberNode), block.args)

function _declaration_assignment(statement, name::Symbol)
    Meta.isexpr(statement, :(=)) || return nothing
    statement.args[1] === name || return nothing
    return statement.args[2]
end

function _declaration_method(statement, name::Symbol, allowed)
    Meta.isexpr(statement, :(=)) || return nothing
    signature = statement.args[1]
    Meta.isexpr(signature, :call) || return nothing
    signature.args[1] === name || return nothing
    length(signature.args) - 1 in allowed || throw(
        ArgumentError("$name must declare one of $(join(allowed, ", ")) argument counts"),
    )
    all(argument -> argument isa Symbol, signature.args[2:end]) ||
        throw(ArgumentError("$name arguments must be untyped names"))
    return (Tuple(signature.args[2:end]), statement.args[2])
end

_unquote(value) = value
_unquote(value::QuoteNode) = value.value

function _quoted_port_name(value)
    name = _unquote(value)
    name isa Symbol || throw(ArgumentError("port bindings must be symbols"))
    return name
end

_canonical_port_name(value) = _quoted_port_name(value)
function _canonical_port_name(value::AbstractString)
    isempty(value) && throw(ArgumentError("port bindings must be symbols"))
    return Symbol(value)
end

function _parse_configuration(statement)
    documentation = nothing
    documentation_source = nothing
    if Meta.isexpr(statement, :macrocall) &&
       statement.args[1] === GlobalRef(Core, Symbol("@doc")) &&
       length(statement.args) == 4
        documentation = statement.args[3]
        documentation_source = statement.args[2]
        statement = statement.args[4]
    end
    Meta.isexpr(statement, :macrocall) || return nothing
    statement.args[1] === Symbol("@configuration") || return nothing
    length(statement.args) == 4 ||
        throw(ArgumentError("@configuration requires a type name and block"))
    name = statement.args[3]
    name isa Symbol || throw(ArgumentError("configuration type name must be an identifier"))
    block = statement.args[4]
    Meta.isexpr(block, :block) ||
        throw(ArgumentError("@configuration requires a block"))
    fields = Tuple{Symbol,Any,Symbol,Any}[]
    for field in _declaration_statements(block)
        Meta.isexpr(field, :call) && field.args[1] === :(=>) ||
            throw(ArgumentError("configuration fields use name::Type => update"))
        typed = field.args[2]
        default = nothing
        if Meta.isexpr(typed, :(=))
            value = typed.args[2]
            (value isa Union{Number,AbstractString,Char,QuoteNode} ||
             value in (:nothing, :missing)) || throw(
                ArgumentError("configuration defaults must be field-independent literals"),
            )
            default = Some(value)
            typed = typed.args[1]
        end
        Meta.isexpr(typed, :(::)) && typed.args[1] isa Symbol ||
            throw(ArgumentError("configuration fields require name::Type"))
        update = field.args[3]
        update in (:construction, :graph_rebuild) ||
            throw(ArgumentError("configuration update must be construction or graph_rebuild"))
        push!(fields, (typed.args[1], typed.args[2], update, default))
    end
    isempty(fields) && throw(ArgumentError("configuration must declare at least one field"))
    return (name, fields, documentation, documentation_source)
end

function _parse_port(expression)
    Meta.isexpr(expression, :call) ||
        throw(ArgumentError("ports must use input(...), output(...), or parameter(...)"))
    kind = expression.args[1]
    kind in (:input, :output, :parameter) ||
        throw(ArgumentError("ports must use input(...), output(...), or parameter(...)"))
    parameters = nothing
    positional = Any[]
    for argument in expression.args[2:end]
        if Meta.isexpr(argument, :parameters)
            parameters === nothing || throw(ArgumentError("duplicate port keyword list"))
            parameters = argument
        else
            push!(positional, argument)
        end
    end
    length(positional) == 4 ||
        throw(ArgumentError("ports require name, element type, shape, and schema"))
    binding = _quoted_port_name(positional[1])
    name = binding
    name_declared = false
    metadata_source = nothing
    sample_binding = nothing
    replacement = nothing
    rate = nothing
    publication_declared = false
    conditional = false
    if parameters !== nothing
        for keyword in parameters.args
            Meta.isexpr(keyword, :kw) ||
                throw(ArgumentError("port options must be keyword assignments"))
            if keyword.args[1] === :name
                name_declared && throw(ArgumentError("name is declared more than once"))
                name_declared = true
                name = _canonical_port_name(keyword.args[2])
            elseif keyword.args[1] === :metadata_from
                metadata_source === nothing ||
                    throw(ArgumentError("metadata_from is declared more than once"))
                metadata_source = _quoted_port_name(keyword.args[2])
            elseif keyword.args[1] === :sample
                sample_binding === nothing ||
                    throw(ArgumentError("sample is declared more than once"))
                sample_binding = _quoted_port_name(keyword.args[2])
            elseif keyword.args[1] === :replace
                replacement === nothing ||
                    throw(ArgumentError("replace is declared more than once"))
                replacement = keyword.args[2]
            elseif keyword.args[1] === :rate
                rate === nothing || throw(ArgumentError("rate is declared more than once"))
                rate = keyword.args[2]
            elseif keyword.args[1] === :publication
                publication_declared &&
                    throw(ArgumentError("publication is declared more than once"))
                publication_declared = true
                publication = _unquote(keyword.args[2])
                publication === :conditional || throw(
                    ArgumentError("publication must be :conditional"),
                )
                conditional = true
            else
                throw(
                    ArgumentError(
                        "port options are name, metadata_from, sample, replace, publication, and rate",
                    ),
                )
            end
        end
    end
    if kind !== :output && metadata_source !== nothing
        throw(ArgumentError("only output ports can declare metadata_from"))
    elseif kind !== :input && sample_binding !== nothing
        throw(ArgumentError("only data input ports can bind sample metadata"))
    elseif kind === :parameter && replacement === nothing
        throw(ArgumentError("parameter ports must declare replace=(plan, values) -> ..."))
    elseif kind === :parameter && rate !== nothing
        throw(ArgumentError("Parameter Ports do not declare a repeated data rate"))
    elseif kind !== :parameter && replacement !== nothing
        throw(ArgumentError("only parameter ports can declare replace"))
    elseif kind !== :output && conditional
        throw(ArgumentError("only output ports can declare conditional publication"))
    end
    return (
        kind=kind,
        binding=binding,
        name=name,
        element_type=positional[2],
        shape=positional[3],
        schema=positional[4],
        metadata_source=metadata_source,
        sample_binding=sample_binding,
        replacement=replacement,
        rate=rate,
        conditional=conditional,
    )
end
