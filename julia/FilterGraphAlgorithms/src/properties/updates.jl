"""Assign one typed scalar value in a property transaction."""
struct PropertyAssignment{T<:PropertyScalar}
    id::UInt32
    value::T
end

PropertyAssignment(id::Integer, value::T) where {T} =
    PropertyAssignment{T}(UInt32(id), value)

"""An assignment names no property in the selected algorithm interface."""
struct UnknownPropertyError <: Exception
    id::UInt32
end

"""A property transaction assigns one property more than once."""
struct DuplicatePropertyError <: Exception
    id::UInt32
end

"""A property assignment has a different scalar type from its descriptor."""
struct PropertyTypeError <: Exception
    id::UInt32
    expected::DataType
    actual::DataType
end

"""A scalar value violates its descriptor range or choice constraint."""
struct PropertyConstraintError <: Exception
    id::UInt32
end

"""A runtime transaction targets a non-runtime property."""
struct PropertyUpdateClassError <: Exception
    id::UInt32
end

function Base.showerror(io::IO, error::UnknownPropertyError)
    return print(io, "unknown property id ", error.id)
end

function Base.showerror(io::IO, error::DuplicatePropertyError)
    return print(io, "property id ", error.id, " is assigned more than once")
end

function Base.showerror(io::IO, error::PropertyTypeError)
    return print(
        io,
        "property id ",
        error.id,
        " requires ",
        error.expected,
        ", received ",
        error.actual,
    )
end

function Base.showerror(io::IO, error::PropertyConstraintError)
    return print(io, "property id ", error.id, " violates its descriptor constraint")
end

function Base.showerror(io::IO, error::PropertyUpdateClassError)
    return print(io, "property id ", error.id, " is not runtime-writable")
end

"""
    property_values(plan::AbstractAlgorithmPlan)

Return the current typed property values as a schema-named tuple.
"""
function property_values end

"""
    replace_properties(plan, values) -> plan

Construct one replacement immutable plan from a complete validated property-
values tuple. The algorithm's ordinary plan constructor remains authoritative
for cross-property validation and preservation of non-property plan state.
"""
function replace_properties end

function _property_value(
    ::AbstractAlgorithmPlan,
    ::Tuple{},
    id::UInt32,
)
    throw(UnknownPropertyError(id))
end

function _property_value(
    plan::AbstractAlgorithmPlan,
    schema::Tuple{PropertySpec{Name,T,U,C},Vararg},
    id::UInt32,
) where {Name,T,U,C}
    spec = first(schema)
    if spec.info.id == id
        return getproperty(property_values(plan), Name)::T
    end
    return _property_value(plan, Base.tail(schema), id)
end

"""
    property_value(plan::AbstractAlgorithmPlan, id::Integer)

Return the current typed scalar value for an algorithm-local property ID.
"""
function property_value(plan::AbstractAlgorithmPlan, id::Integer)
    property_id = UInt32(id)
    return _property_value(plan, validate_property_schema(plan), property_id)
end

_validate_property_value(::NoPropertyConstraint, value, ::UInt32) = value

function _validate_property_value(
    constraint::PropertyRange,
    value,
    id::UInt32,
)
    constraint.minimum <= value <= constraint.maximum ||
        throw(PropertyConstraintError(id))
    return value
end

function _validate_property_value(
    constraint::PropertyChoices,
    value,
    id::UInt32,
)
    any(choice -> choice.value == value, constraint.choices) ||
        throw(PropertyConstraintError(id))
    return value
end

_assignment_id_present(::UInt32, ::Tuple{}) = false

function _assignment_id_present(
    id::UInt32,
    assignments::Tuple{PropertyAssignment{T},Vararg},
) where {T}
    return first(assignments).id == id ||
           _assignment_id_present(id, Base.tail(assignments))
end

_validate_unique_assignments(::Tuple{}) = nothing

function _validate_unique_assignments(
    assignments::Tuple{PropertyAssignment{T},Vararg},
) where {T}
    assignment = first(assignments)
    remaining = Base.tail(assignments)
    _assignment_id_present(assignment.id, remaining) &&
        throw(DuplicatePropertyError(assignment.id))
    return _validate_unique_assignments(remaining)
end

function _replace_property_value(
    values::NamedTuple,
    spec::PropertySpec,
    assignment::PropertyAssignment,
)
    return _replace_property_value(values, spec, assignment, spec.info.update)
end

function _replace_property_value(
    values::NamedTuple,
    spec::PropertySpec,
    assignment::PropertyAssignment,
    ::RuntimeProperty,
)
    return _replace_typed_property_value(values, spec, assignment)
end

function _replace_property_value(
    ::NamedTuple,
    ::PropertySpec,
    assignment::PropertyAssignment,
    ::PropertyUpdate,
)
    throw(PropertyUpdateClassError(assignment.id))
end

function _replace_typed_property_value(
    values::NamedTuple,
    spec::PropertySpec{Name,T,RuntimeProperty,C},
    assignment::PropertyAssignment{T},
) where {Name,T,C}
    value = _validate_property_value(
        spec.info.constraint,
        assignment.value,
        assignment.id,
    )
    return merge(values, NamedTuple{(Name,)}((value,)))
end

function _replace_typed_property_value(
    ::NamedTuple,
    ::PropertySpec{Name,T,RuntimeProperty,C},
    assignment::PropertyAssignment{S},
) where {Name,T,C,S}
    throw(PropertyTypeError(assignment.id, T, S))
end

function _apply_property_assignment(
    ::NamedTuple,
    ::Tuple{},
    assignment::PropertyAssignment,
)
    throw(UnknownPropertyError(assignment.id))
end

function _apply_property_assignment(
    values::NamedTuple,
    schema::Tuple{PropertySpec{Name,T,U,C},Vararg},
    assignment::PropertyAssignment,
) where {Name,T,U,C}
    spec = first(schema)
    spec.info.id == assignment.id &&
        return _replace_property_value(values, spec, assignment)
    return _apply_property_assignment(values, Base.tail(schema), assignment)
end

_apply_property_assignments(values::NamedTuple, ::Tuple, ::Tuple{}) = values

function _apply_property_assignments(
    values::NamedTuple,
    schema::Tuple,
    assignments::Tuple{PropertyAssignment{T},Vararg},
) where {T}
    updated = _apply_property_assignment(values, schema, first(assignments))
    return _apply_property_assignments(updated, schema, Base.tail(assignments))
end

"""
    prepare_properties(plan, assignments) -> plan

Validate a complete scalar property transaction and return a replacement
immutable plan. Shared infrastructure rejects duplicate, unknown, mistyped,
constrained, and non-runtime assignments before the algorithm's one replacement
constructor performs cross-property validation.
Assignments may be a Tuple or an AbstractVector of PropertyAssignment values.
"""
function prepare_properties(plan::AbstractAlgorithmPlan, assignments::Tuple)
    _validate_unique_assignments(assignments)
    schema = validate_property_schema(plan)
    values = _apply_property_assignments(
        property_values(plan),
        schema,
        assignments,
    )
    return replace_properties(plan, values)
end

# Control-plane callers may have a runtime number of assignments. Iteration
# keeps transaction length out of the call type while using the same schema
# validation and final replacement constructor as the Tuple interface.
function prepare_properties(
    plan::AbstractAlgorithmPlan,
    assignments::AbstractVector{<:PropertyAssignment},
)
    ids = Set{UInt32}()
    for assignment in assignments
        assignment.id in ids && throw(DuplicatePropertyError(assignment.id))
        push!(ids, assignment.id)
    end
    schema = validate_property_schema(plan)
    values = property_values(plan)
    for assignment in assignments
        values = _apply_property_assignment(values, schema, assignment)
    end
    return replace_properties(plan, values)
end
