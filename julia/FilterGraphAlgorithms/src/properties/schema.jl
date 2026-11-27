"""Marker supertype for a scalar property's update class."""
abstract type PropertyUpdate end

"""A property reported by an algorithm but not accepted in transactions."""
struct ReadOnlyProperty <: PropertyUpdate end

"""A property that may be adopted atomically at a process boundary."""
struct RuntimeProperty <: PropertyUpdate end

"""A property accepted only while constructing an algorithm instance."""
struct ConstructionProperty <: PropertyUpdate end

"""A property whose change requires graph reconstruction."""
struct GraphRebuildProperty <: PropertyUpdate end

"""Types admitted by the transport-independent scalar property contract."""
const PropertyScalar = Union{Bool,Int32,Int64,Float32,Float64,UInt32,String}

"""One admitted value and label for an enumerated scalar property."""
struct PropertyChoice{T<:PropertyScalar}
    value::T
    name::Symbol
    description::String
end

"""Marker supertype for scalar property constraints."""
abstract type PropertyConstraint end

"""A scalar property with no additional value constraint."""
struct NoPropertyConstraint <: PropertyConstraint end

"""Scalar kinds that admit a portable ordered inclusive range."""
const PropertyRangeScalar = Union{Int32,Int64,Float32,Float64,UInt32}

"""The inclusive admitted interval for an ordered scalar property."""
struct PropertyRange{T<:PropertyRangeScalar} <: PropertyConstraint
    minimum::T
    maximum::T
end

"""The complete admitted set for an enumerated scalar property."""
struct PropertyChoices{T<:PropertyScalar,N} <: PropertyConstraint
    choices::NTuple{N,PropertyChoice{T}}
end

"""
    PropertyInfo

Describe one algorithm-local scalar property. An identifier and name remain
stable for the lifetime of the algorithm interface. The concrete value,
update, and constraint types preserve the descriptor without type-erased
storage.
"""
struct PropertyInfo{T<:PropertyScalar,U<:PropertyUpdate,C<:PropertyConstraint}
    id::UInt32
    name::Symbol
    description::String
    unit::String
    update::U
    default_value::T
    constraint::C
end

"""
    PropertySpec

Bind one public property descriptor to the identically named field in an
algorithm's typed property-values named tuple. Algorithms construct specs with
[`runtime_property`](@ref), [`read_only_property`](@ref),
[`construction_property`](@ref), or [`graph_rebuild_property`](@ref).
"""
struct PropertySpec{
    Name,
    T<:PropertyScalar,
    U<:PropertyUpdate,
    C<:PropertyConstraint,
}
    info::PropertyInfo{T,U,C}
end

function _property_spec(
    id::Integer,
    ::Val{Name},
    ::Type{T},
    update::U;
    default::T,
    description::String,
    unit::String,
    constraint::C=NoPropertyConstraint(),
) where {Name,T<:PropertyScalar,U<:PropertyUpdate,C<:PropertyConstraint}
    _validate_property_definition(default, constraint)
    info = PropertyInfo(
        UInt32(id),
        Name,
        description,
        unit,
        update,
        default,
        constraint,
    )
    return PropertySpec{Name,T,U,C}(info)
end

function _property_spec(
    id::Integer,
    name::Symbol,
    type::Type{T},
    update::PropertyUpdate;
    kwargs...,
) where {T<:PropertyScalar}
    return _property_spec(id, Val(name), type, update; kwargs...)
end

"""
    runtime_property(id, name, type; kwargs...)

Declare one runtime-writable scalar property.
"""
runtime_property(id, name, type; kwargs...) =
    _property_spec(id, name, type, RuntimeProperty(); kwargs...)

"""
    read_only_property(id, name, type; kwargs...)

Declare one read-only scalar property.
"""
read_only_property(id, name, type; kwargs...) =
    _property_spec(id, name, type, ReadOnlyProperty(); kwargs...)

"""
    construction_property(id, name, type; kwargs...)

Declare one construction-only scalar property.
"""
construction_property(id, name, type; kwargs...) =
    _property_spec(id, name, type, ConstructionProperty(); kwargs...)

"""
    graph_rebuild_property(id, name, type; kwargs...)

Declare one scalar property whose change requires graph reconstruction.
"""
graph_rebuild_property(id, name, type; kwargs...) =
    _property_spec(id, name, type, GraphRebuildProperty(); kwargs...)

"""A declarative property schema is ambiguous or internally inconsistent."""
struct PropertySchemaError <: Exception
    id::UInt32
    reason::Symbol
    first_index::Int
    second_index::Int
end

PropertySchemaError(id::UInt32, reason::Symbol) =
    PropertySchemaError(id, reason, 0, 0)

function Base.showerror(io::IO, error::PropertySchemaError)
    print(io, "invalid property schema at id ", error.id, ": ", error.reason)
    if error.first_index != 0 || error.second_index != 0
        print(io, " (indices ", error.first_index, " and ", error.second_index, ")")
    end
    return nothing
end

"""
    property_schema(::Type{<:AbstractAlgorithmPlan}) -> Tuple

Return the declarative scalar property specs for an algorithm plan type. The
empty tuple is the default and preserves the base algorithm interface for plans
without properties.
"""
property_schema(::Type{<:AbstractAlgorithmPlan}) = ()
property_schema(plan::AbstractAlgorithmPlan) = property_schema(typeof(plan))

_validate_property_constraint(::PropertyInfo, ::NoPropertyConstraint) = nothing

function _validate_property_constraint(
    info::PropertyInfo{T},
    constraint::PropertyRange{S},
) where {T,S}
    T === S || throw(PropertySchemaError(info.id, :constraint_type))
    minimum = constraint.minimum
    maximum = constraint.maximum
    if !isfinite(minimum) || !isfinite(maximum) || minimum > maximum
        throw(PropertySchemaError(info.id, :invalid_range))
    end
    minimum <= info.default_value <= maximum ||
        throw(PropertySchemaError(info.id, :invalid_default))
    return nothing
end

function _validate_property_constraint(
    info::PropertyInfo{T},
    constraint::PropertyChoices{S},
) where {T,S}
    T === S || throw(PropertySchemaError(info.id, :constraint_type))
    choices = constraint.choices
    isempty(choices) && throw(PropertySchemaError(info.id, :empty_choices))
    any(choice -> choice.value == info.default_value, choices) ||
        throw(PropertySchemaError(info.id, :invalid_default))
    for first_index in eachindex(choices)
        first_choice = choices[first_index]
        first_choice.name === Symbol("") &&
            throw(PropertySchemaError(info.id, :empty_choice_name, first_index, 0))
        for second_index in first_index + 1:lastindex(choices)
            second_choice = choices[second_index]
            first_choice.value == second_choice.value &&
                throw(PropertySchemaError(
                    info.id,
                    :duplicate_choice_value,
                    first_index,
                    second_index,
                ))
            first_choice.name === second_choice.name &&
                throw(PropertySchemaError(
                    info.id,
                    :duplicate_choice_name,
                    first_index,
                    second_index,
                ))
        end
    end
    return nothing
end

function _validate_property_info(spec::PropertySpec)
    info = spec.info
    info.name === Symbol("") && throw(PropertySchemaError(info.id, :empty_name))
    isempty(info.description) &&
        throw(PropertySchemaError(info.id, :empty_description))
    isempty(info.unit) && throw(PropertySchemaError(info.id, :empty_unit))
    _validate_property_constraint(info, info.constraint)
    return nothing
end

_validate_property_against_remaining(::PropertyInfo, ::Tuple{}, ::Int, ::Int) = nothing

function _validate_property_against_remaining(
    info::PropertyInfo,
    remaining::Tuple{PropertySpec,Vararg},
    first_index::Int,
    second_index::Int,
)
    candidate = first(remaining).info
    info.id == candidate.id &&
        throw(PropertySchemaError(info.id, :duplicate_id, first_index, second_index))
    info.name === candidate.name &&
        throw(PropertySchemaError(candidate.id, :duplicate_name, first_index, second_index))
    return _validate_property_against_remaining(
        info,
        Base.tail(remaining),
        first_index,
        second_index + 1,
    )
end

_validate_property_schema(::Tuple{}, ::Int) = nothing

function _validate_property_schema(schema::Tuple{PropertySpec,Vararg}, index::Int)
    spec = first(schema)
    _validate_property_info(spec)
    _validate_property_against_remaining(spec.info, Base.tail(schema), index, index + 1)
    return _validate_property_schema(Base.tail(schema), index + 1)
end

"""
    validate_property_schema(type::Type{<:AbstractAlgorithmPlan})

Validate descriptor metadata and schema-wide identifier/name uniqueness.
"""
function validate_property_schema(type::Type{<:AbstractAlgorithmPlan})
    schema = property_schema(type)
    _validate_property_schema(schema, 1)
    return schema
end

validate_property_schema(plan::AbstractAlgorithmPlan) =
    validate_property_schema(typeof(plan))

"""
    property_infos(type::Type{<:AbstractAlgorithmPlan})

Return the ordered public descriptors derived from a validated schema.
"""
property_infos(type::Type{<:AbstractAlgorithmPlan}) =
    map(spec -> spec.info, validate_property_schema(type))
property_infos(plan::AbstractAlgorithmPlan) = property_infos(typeof(plan))

_validate_property_definition(default, ::NoPropertyConstraint) = default

function _validate_property_definition(default, constraint::PropertyRange)
    constraint.minimum <= constraint.maximum ||
        throw(ArgumentError("property range minimum must not exceed maximum"))
    constraint.minimum <= default <= constraint.maximum ||
        throw(ArgumentError("property default must be inside its range"))
    return default
end

function _validate_property_definition(default, constraint::PropertyChoices)
    choices = constraint.choices
    isempty(choices) && throw(ArgumentError("property choices must not be empty"))
    any(choice -> choice.value == default, choices) ||
        throw(ArgumentError("property default must be one of its choices"))
    for first_index in eachindex(choices)
        for second_index in first_index + 1:lastindex(choices)
            first_choice = choices[first_index]
            second_choice = choices[second_index]
            first_choice.value == second_choice.value &&
                throw(ArgumentError("property choice values must be unique"))
            first_choice.name == second_choice.name &&
                throw(ArgumentError("property choice names must be unique"))
        end
    end
    return default
end
