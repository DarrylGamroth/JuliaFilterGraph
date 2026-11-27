"""
    @algorithm Declaration begin ... end

Declare one typed algorithm graph boundary beside ordinary scientific plan and
workspace code. The macro generates typed configuration, static metadata,
native preparation, exact formats, a checked `process!` method, a private
admitted processing method, and reset. A `process!(plan)` declaration returns
its Plan directly from `prepare`, including a NamedTuple Plan. A declaration
with `process!(plan, workspace)` returns `(plan=plan, workspace=workspace)`;
either field order is accepted, with no missing or extra fields. A Workspace
must implement [`commit!`](@ref), [`discard!`](@ref), and [`reset!`](@ref), even
when it contains only reusable scratch storage. Preparation checks that these
methods apply, without calling them. Their bounded, allocation-free behavior
remains the Algorithm author's responsibility. The existing exact
`(plan=plan, workspace=workspace)` bundle is also accepted by one-argument
declarations. The declaration contains no PipeWire, SPA, FGN, JSON, pointer, or
callback concepts. An Algorithm with an expected scientific rejection declares
`failure = ReasonEnum` and returns one of that enum's values from `process!`.
Algorithms without that behavior omit the clause. A data input that needs
transport-neutral sample identity declares `sample=:binding`; that binding is
`nothing` when the native caller supplied no [`SampleMetadata`](@ref). The
first Port argument is its Julia lexical binding. An optional `name="..."`
retains a canonical graph name that is not a natural Julia identifier.
Place an ordinary docstring before `@algorithm` to document the declaration,
and before its nested `@configuration` to document the generated configuration.
A configuration field can declare a literal default as `(name::Type = value) => update`.
Defaults cannot reference other fields or call functions.
Omitting that field uses the default in keyword construction and graph preparation.
"""
macro algorithm(declaration, body)
    declaration isa Symbol || throw(ArgumentError("declaration name must be an identifier"))
    Meta.isexpr(body, :block) ||
        throw(ArgumentError("@algorithm requires a block"))

    label = nothing
    configuration = nothing
    property_policy = nothing
    operation = nothing
    failure_type = nothing
    ports_method = nothing
    prepare_method = nothing
    process_method = nothing

    for statement in _declaration_statements(body)
        if (value = _declaration_assignment(statement, :label)) !== nothing
            label === nothing || throw(ArgumentError("label is declared more than once"))
            label = value
        elseif (value = _declaration_assignment(statement, :properties)) !== nothing
            property_policy === nothing ||
                throw(ArgumentError("properties is declared more than once"))
            property_policy = value
        elseif (value = _declaration_assignment(statement, :operation)) !== nothing
            operation === nothing ||
                throw(ArgumentError("operation is declared more than once"))
            operation = value
        elseif (value = _declaration_assignment(statement, :failure)) !== nothing
            failure_type === nothing ||
                throw(ArgumentError("failure is declared more than once"))
            failure_type = value
        elseif (value = _parse_configuration(statement)) !== nothing
            configuration === nothing ||
                throw(ArgumentError("configuration is declared more than once"))
            configuration = value
        elseif (value = _declaration_method(statement, :ports, (1,))) !== nothing
            ports_method === nothing || throw(ArgumentError("ports is declared more than once"))
            ports_method = value
        elseif (value = _declaration_method(statement, :prepare, (1,))) !== nothing
            prepare_method === nothing ||
                throw(ArgumentError("prepare is declared more than once"))
            prepare_method = value
        elseif (value = _declaration_method(statement, Symbol("process!"), (1, 2))) !== nothing
            process_method === nothing ||
                throw(ArgumentError("process! is declared more than once"))
            process_method = value
        else
            throw(ArgumentError("unknown algorithm declaration statement: $statement"))
        end
    end

    label === nothing && throw(ArgumentError("label is required"))
    configuration === nothing && throw(ArgumentError("@configuration is required"))
    property_policy === nothing && (property_policy = QuoteNode(:none))
    operation === nothing && (operation = :(FilterGraphAlgorithms.GeneralOperation()))
    ports_method === nothing && throw(ArgumentError("ports(configuration) is required"))
    prepare_method === nothing && throw(ArgumentError("prepare(configuration) is required"))
    process_method === nothing &&
        throw(ArgumentError("process!(plan) or process!(plan, workspace) is required"))

    config_name, fields, config_documentation, config_documentation_source = configuration
    field_definitions = [
        field[4] === nothing ? :($(field[1])::$(field[2])) :
        :($(field[1])::$(field[2]) = $(something(field[4]))) for field in fields
    ]
    configuration_definition = quote
        Base.@kwdef struct $config_name
            $(field_definitions...)
        end
    end
    if config_documentation !== nothing
        configuration_definition = Expr(
            :macrocall,
            GlobalRef(Core, Symbol("@doc")),
            config_documentation_source,
            config_documentation,
            configuration_definition,
        )
    end
    configuration_values = [
        :(
            FilterGraphAlgorithms.AlgorithmConfiguration(
                $(QuoteNode(field[1])),
                $(field[2]),
                $(field[3] === :construction ?
                  :(FilterGraphAlgorithms.ConstructionConfiguration()) :
                  :(FilterGraphAlgorithms.GraphRebuildConfiguration())),
                $(field[4] === nothing ? nothing : :(Some($(something(field[4]))))),
            )
        ) for field in fields
    ]

    policy = _unquote(property_policy)
    policy in (:none, :runtime) ||
        throw(ArgumentError("properties must be :none or :runtime"))
    policy_value = policy === :none ?
        :(FilterGraphAlgorithms.NoDeclaredProperties()) :
        :(FilterGraphAlgorithms.RuntimeDeclaredProperties())
    failure_definition = failure_type === nothing ? nothing :
        :(FilterGraphAlgorithms.algorithm_failure_type(::Type{$declaration}) = $failure_type)

    port_arguments, port_body = ports_method
    Meta.isexpr(port_body, :block) &&
        length(_declaration_statements(port_body)) == 1 &&
        (port_body = only(_declaration_statements(port_body)))
    Meta.isexpr(port_body, :tuple) ||
        throw(ArgumentError("ports must return a tuple of input/output declarations"))
    ports = map(_parse_port, port_body.args)
    isempty(ports) && throw(ArgumentError("at least one port is required"))
    bindings = map(port -> port.binding, ports)
    length(unique(bindings)) == length(bindings) || throw(
        ArgumentError("port bindings must be unique"),
    )
    names = map(port -> port.name, ports)
    length(unique(names)) == length(names) || throw(ArgumentError("port names must be unique"))
    input_names = Dict(
        port.binding => port.name for port in ports if port.kind === :input
    )
    for port in ports
        port.kind === :output && port.metadata_source !== nothing &&
            !haskey(input_names, port.metadata_source) &&
            throw(ArgumentError("output metadata source must name an input"))
    end

    static_ports = map(ports) do port
        direction = port.kind === :output ?
            :(FilterGraphAlgorithms.OutputPort()) :
            :(FilterGraphAlgorithms.InputPort())
        role = port.kind === :parameter ?
            :(FilterGraphAlgorithms.SparseParameter()) :
            :(FilterGraphAlgorithms.FrameData())
        return :(
            FilterGraphAlgorithms.AlgorithmPort(
                $(QuoteNode(port.name)),
                $direction,
                $role,
                $(port.element_type),
                $(
                    port.metadata_source === nothing ? nothing :
                    QuoteNode(input_names[port.metadata_source])
                ),
                $(port.conditional),
            )
        )
    end
    formats = [
        :(
            FilterGraphAlgorithms.AlgorithmFormat(
                $(port.element_type),
                $(port.shape),
                $(port.schema),
                ;
                rate=$(port.rate === nothing ? :((1, 1)) : port.rate),
            )
        ) for port in ports
    ]

    format_accessors = [
        :(
            @inline FilterGraphAlgorithms.algorithm_format(
                prepared::FilterGraphAlgorithms.PreparedAlgorithm{$declaration},
                ::Val{$(QuoteNode(port.name))},
            ) = prepared.formats[$index]
        ) for (index, port) in enumerate(ports)
    ]

    prepare_arguments, prepare_body = prepare_method
    prepare_configuration = only(prepare_arguments)
    process_arguments, process_body = process_method
    output_ports = filter(port -> port.kind === :output, ports)
    input_ports = filter(port -> port.kind === :input, ports)
    sample_bindings = Symbol[
        port.sample_binding for port in input_ports if port.sample_binding !== nothing
    ]
    length(unique(sample_bindings)) == length(sample_bindings) ||
        throw(ArgumentError("input sample metadata bindings must be unique"))
    process_names = Set{Symbol}(bindings)
    union!(process_names, process_arguments)
    any(binding -> binding in process_names, sample_bindings) && throw(
        ArgumentError("input sample metadata bindings must not shadow process names"),
    )
    process_signature = Any[
        [:($(port.binding)::Base.AbstractArray) for port in output_ports]...,
        :(prepared::FilterGraphAlgorithms.PreparedAlgorithm{$declaration}),
        [:($(port.binding)::Base.AbstractArray) for port in input_ports]...,
    ]
    process_call = Any[
        [port.binding for port in output_ports]...,
        :prepared,
        [port.binding for port in input_ports]...,
    ]
    input_names_tuple = Tuple(port.binding for port in input_ports)
    sample_arguments = [gensym(Symbol(port.binding, :_sample)) for port in input_ports]
    admitted_signature = Any[
        process_signature...,
        [:(
            $argument::FilterGraphAlgorithms.AdmittedSampleMetadata
        ) for argument in sample_arguments]...,
    ]
    ordered_metadata = gensym(:ordered_metadata)
    checked_metadata_signature = Any[
        process_signature...,
        :(metadata::NamedTuple{$input_names_tuple}),
    ]
    admitted_metadata_values = [
        :(FilterGraphAlgorithms._admitted_sample_metadata(getfield(metadata, $index)))
        for index in eachindex(input_ports)
    ]
    absent_metadata_values = [
        :(FilterGraphAlgorithms._NO_SAMPLE_METADATA) for _ in input_ports
    ]
    validations = Expr[]
    for (index, port) in enumerate(ports)
        port.kind === :parameter && continue
        push!(
            validations,
            :(
                FilterGraphAlgorithms._validate_declared_array(
                    prepared.formats[$index],
                    $(port.binding),
                    ($("Algorithm '$label' $(port.kind)"), $(QuoteNode(port.name))),
                )
            ),
        )
    end
    metadata_validations = [quote
        local value = getfield(metadata, $index)
        (isnothing(value) || value isa FilterGraphAlgorithms.SampleMetadata) || throw(
            ArgumentError($("Algorithm '$label' input metadata Port '$(port.name)' must be SampleMetadata or nothing")),
        )
    end for (index, port) in enumerate(input_ports)]

    parameter_definitions = Expr[]
    for (index, port) in enumerate(ports)
        port.kind === :parameter || continue
        replacement = port.replacement
        push!(
            parameter_definitions,
            quote
                function FilterGraphAlgorithms._prepare_parameter_replacement(
                    prepared::FilterGraphAlgorithms.PreparedAlgorithm{$declaration,P},
                    plan::P,
                    ::Val{$(QuoteNode(port.name))},
                    values::AbstractArray,
                ) where {P}
                    FilterGraphAlgorithms._validate_declared_array(
                        prepared.formats[$index],
                        values,
                        ($("Algorithm '$label' parameter"), $(QuoteNode(port.name))),
                    )
                    local replacement = $replacement
                    local replacement_plan =
                        FilterGraphAlgorithms._replace_prepared_parameter(replacement, plan, values)
                    replacement_plan isa P || throw(
                        ArgumentError(
                            "Parameter Port replacement must preserve Plan type " *
                            string(P),
                        ),
                    )
                    return replacement_plan
                end

                function FilterGraphAlgorithms.replace_parameter!(
                    prepared::FilterGraphAlgorithms.PreparedAlgorithm{$declaration},
                    name::Val{$(QuoteNode(port.name))},
                    values::AbstractArray,
                )
                    local replacement_plan =
                        FilterGraphAlgorithms._prepare_parameter_replacement(
                            prepared,
                            prepared.plan,
                            name,
                            values,
                        )
                    FilterGraphAlgorithms._plan_adopted!(prepared.workspace)
                    prepared.plan = replacement_plan
                    return prepared
                end
            end,
        )
    end

    prepared_plan = gensym(:plan)
    prepared_workspace = gensym(:workspace)
    declared_configuration = gensym(:configuration)
    host_preparation = gensym(:host_preparation)
    port_configuration = only(port_arguments)
    port_binding = if port_configuration == prepare_configuration
        nothing
    else
        :(local $port_configuration = $prepare_configuration)
    end
    process_bindings = if length(process_arguments) == 1
        quote
            local $(only(process_arguments)) = prepared.plan
        end
    else
        quote
            local $(process_arguments[1]) = prepared.plan
            local $(process_arguments[2]) = prepared.workspace
        end
    end
    prepared_values = if length(process_arguments) == 1
        :(FilterGraphAlgorithms._prepared_plan_workspace($prepare_body))
    else
        :(FilterGraphAlgorithms._prepared_plan_workspace($declaration, $prepare_body))
    end
    sample_binding_expressions = Expr[]
    for (index, port) in enumerate(input_ports)
        port.sample_binding === nothing && continue
        push!(
            sample_binding_expressions,
            :(
                local $(port.sample_binding) = FilterGraphAlgorithms._sample_metadata(
                    $(sample_arguments[index]),
                )
            ),
        )
    end

    return esc(quote
        Base.@__doc__ struct $declaration <: FilterGraphAlgorithms.AbstractAlgorithmDeclaration end

        $configuration_definition

        FilterGraphAlgorithms.algorithm_label(::Type{$declaration}) = $label
        FilterGraphAlgorithms.algorithm_configuration(::Type{$declaration}) =
            ($(configuration_values...),)
        FilterGraphAlgorithms.algorithm_configuration_type(::Type{$declaration}) = $config_name
        FilterGraphAlgorithms.algorithm_ports(::Type{$declaration}) = ($(static_ports...),)
        FilterGraphAlgorithms.algorithm_property_policy(::Type{$declaration}) = $policy_value
        function FilterGraphAlgorithms.algorithm_operation(::Type{$declaration})
            local selected = $operation
            selected isa FilterGraphAlgorithms.AlgorithmOperation || throw(
                ArgumentError("operation must be an AlgorithmOperation"),
            )
            return selected
        end
        $failure_definition
        $(format_accessors...)

        function FilterGraphAlgorithms.prepare_algorithm(
            ::Type{$declaration},
            $prepare_configuration::$config_name,
        )
            return FilterGraphAlgorithms._prepare_declared_algorithm(
                $declaration, $prepare_configuration,
            )
        end

        function FilterGraphAlgorithms._prepare_declared_algorithm(
            ::Type{$declaration},
            $declared_configuration::$config_name;
            prepare_host_plan::Bool=true,
        )
            local $host_preparation = prepare_host_plan
            local $prepare_configuration = $declared_configuration
            FilterGraphAlgorithms._validate_failure_type($declaration)
            local $prepared_plan, $prepared_workspace = $prepared_values
            FilterGraphAlgorithms._validate_workspace_lifecycle(
                $declaration,
                $prepared_workspace,
            )
            $port_binding
            local formats = ($(formats...),)
            if $host_preparation
                $prepared_plan = FilterGraphAlgorithms._prepare_host_plan($declaration, $prepared_plan)
            end
            return FilterGraphAlgorithms.PreparedAlgorithm(
                $declaration,
                $prepared_plan,
                $prepared_workspace,
                formats,
            )
        end

        function FilterGraphAlgorithms.process!(
            $(checked_metadata_signature...),
        )
            $(validations...)
            $(metadata_validations...)
            local result = FilterGraphAlgorithms._process_admitted!(
                $(process_call...),
                $(admitted_metadata_values...),
            )
            return FilterGraphAlgorithms._finish_checked_process!(
                $declaration,
                prepared.workspace,
                result,
            )
        end

        function FilterGraphAlgorithms.process!(
            $(process_signature...),
        )
            $(validations...)
            local result = FilterGraphAlgorithms._process_admitted!(
                $(process_call...),
                $(absent_metadata_values...),
            )
            return FilterGraphAlgorithms._finish_checked_process!(
                $declaration,
                prepared.workspace,
                result,
            )
        end

        function FilterGraphAlgorithms.process!(
            $(process_signature...),
            metadata::NamedTuple,
        )
            local $ordered_metadata = FilterGraphAlgorithms._order_port_values(
                $("Algorithm '$label' input metadata"),
                Val($(QuoteNode(input_names_tuple))),
                metadata,
            )
            return FilterGraphAlgorithms.process!($(process_call...), $ordered_metadata)
        end

        function FilterGraphAlgorithms._process_admitted!(
            $(admitted_signature...),
        )
            $process_bindings
            $(sample_binding_expressions...)
            return $process_body
        end

        function FilterGraphAlgorithms.reset!(
            prepared::FilterGraphAlgorithms.PreparedAlgorithm{$declaration},
        )
            FilterGraphAlgorithms.reset!(prepared.workspace)
            return prepared
        end

        $(parameter_definitions...)
    end)
end
