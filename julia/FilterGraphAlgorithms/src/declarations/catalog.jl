"""
    algorithm_description(declaration::Type{<:AbstractAlgorithmDeclaration})
    algorithm_description(prepared::PreparedAlgorithm)

Return an independent metadata snapshot for one Algorithm. Declaration inspection
does not prepare an Algorithm: `formats`, `properties`, and `required_metadata`
are `nothing` until inspecting a prepared instance. `configuration` always
describes declaration fields, rather than retained construction values.

The prepared overload reports exact formats, validated Property descriptors,
and required metadata by input Port. Queries may allocate and execute declaration
metadata methods; they never invoke preparation or processing. Returned nested
mutable metadata is copied; no Plan or Workspace is exposed. Operation markers
do not imply backend support. These are cold inspection APIs.
"""
Base.@nospecializeinfer function algorithm_description(
    @nospecialize(declaration::Type{<:AbstractAlgorithmDeclaration}),
)
    return deepcopy((
        stage=:declaration,
        declaration=declaration,
        label=algorithm_label(declaration),
        configuration_type=algorithm_configuration_type(declaration),
        configuration=algorithm_configuration(declaration),
        ports=algorithm_ports(declaration),
        property_policy=algorithm_property_policy(declaration),
        operation=algorithm_operation(declaration),
        failure_type=algorithm_failure_type(declaration),
        formats=nothing,
        properties=nothing,
        required_metadata=nothing,
    ))
end

_catalog_properties(::RuntimeDeclaredProperties, plan) = property_infos(plan)
_catalog_properties(policy, plan) = ()

function algorithm_description(prepared::PreparedAlgorithm{D}) where {D}
    description = algorithm_description(D)
    properties = _catalog_properties(description.property_policy, prepared.plan)
    inputs = filter(port -> port.direction isa InputPort, description.ports)
    names = map(port -> port.name, inputs)
    required_metadata = NamedTuple{names}(map(inputs) do port
        required_sample_metadata(prepared, port.name)
    end)
    return merge(description, deepcopy((
        stage=:prepared,
        formats=algorithm_formats(prepared),
        properties=properties,
        required_metadata=required_metadata,
    )))
end

"""
    algorithm_catalog(declarations::Tuple=algorithms())

Return declaration metadata snapshots in caller order. An explicit tuple can
include declarations from scientist-owned packages. Duplicate canonical labels
throw `ArgumentError`; no global registry or resource preparation is involved.
See [`algorithm_description`](@ref) for availability and snapshot ownership.
"""
Base.@nospecializeinfer function algorithm_catalog(declarations::Tuple=algorithms())
    @nospecialize declarations
    # Declaration tuples may be large and heterogeneous. Cold iteration avoids
    # compiling a tuple-wide specialization merely to inspect the selection.
    descriptions = Any[]
    labels = Set{String}()
    for declaration in declarations
        description = algorithm_description(declaration)
        description.label in labels &&
            throw(ArgumentError("duplicate algorithm label $(repr(description.label))"))
        push!(labels, description.label)
        push!(descriptions, description)
    end
    return Tuple(descriptions)
end
