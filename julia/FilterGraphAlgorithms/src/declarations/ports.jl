"""Marker supertype for packed ndarray storage order."""
abstract type ArrayLayout end

"""The last logical axis is contiguous."""
struct RowMajorLayout <: ArrayLayout end

"""The first logical axis is contiguous, as in an ordinary Julia array."""
struct ColumnMajorLayout <: ArrayLayout end

"""Marker supertype for a declared ndarray port direction."""
abstract type PortDirection end

"""A per-frame or parameter input port."""
struct InputPort <: PortDirection end

"""A per-frame output port."""
struct OutputPort <: PortDirection end

"""Marker supertype for a declared ndarray update role."""
abstract type PortRole end

"""An ndarray value supplied for every complete process call."""
struct FrameData <: PortRole end

"""A sparse ndarray replacement prepared outside processing."""
struct SparseParameter <: PortRole end

"""One statically declared Algorithm Port with its canonical graph name."""
struct AlgorithmPort
    name::Symbol
    direction::PortDirection
    role::PortRole
    element_type::DataType
    metadata_source::Union{Nothing,Symbol}
    conditional::Bool
end

const _DECLARED_ELEMENT_TYPES = Union{
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
}

const _MAX_SCIENTIFIC_SCHEMA_BYTES = 255

function _valid_scientific_schema(schema::AbstractString)
    normalized = String(schema)
    bytes = codeunits(normalized)
    isempty(bytes) && return false
    length(bytes) <= _MAX_SCIENTIFIC_SCHEMA_BYTES || return false
    all(byte -> 0x21 <= byte <= 0x7e, bytes) || return false
    parts = rsplit(normalized, '/'; limit=2)
    length(parts) == 2 || return false
    name, encoded_version = parts
    isempty(name) && return false
    isempty(encoded_version) && return false
    all(isdigit, encoded_version) || return false
    version = tryparse(UInt32, encoded_version)
    return !isnothing(version) && !iszero(version)
end

"""
Exact packed ndarray format prepared for one Port. `rate` is the reduced
positive rational multiplier of the Node's base deployment rate.
"""
struct AlgorithmFormat{N}
    element_type::DataType
    layout::ArrayLayout
    shape::NTuple{N,Int}
    schema::String
    rate::NTuple{2,UInt32}

    function AlgorithmFormat(
        element_type::Type{T},
        shape::NTuple{N,<:Integer},
        schema::AbstractString,
        layout::ArrayLayout=ColumnMajorLayout(),
        ;
        rate::Tuple{<:Integer,<:Integer}=(1, 1),
    ) where {T<:_DECLARED_ELEMENT_TYPES,N}
        1 <= N <= 64 || throw(ArgumentError("ndarray rank must be in 1:64"))
        all(>(0), shape) || throw(ArgumentError("ndarray dimensions must be positive"))
        _valid_scientific_schema(schema) || throw(
            ArgumentError("scientific schema must be a nonempty versioned ASCII identifier"),
        )
        declared_type = schema_element_type(schema)
        isnothing(declared_type) || declared_type === T || throw(ArgumentError(
            "scientific schema $schema requires element type $declared_type, received $T",
        ))
        converted = ntuple(index -> Int(shape[index]), Val(N))
        elements = foldl(converted; init=UInt64(1)) do product, extent
            Base.checked_mul(product, UInt64(extent))
        end
        bytes = Base.checked_mul(elements, UInt64(sizeof(T)))
        bytes <= typemax(UInt32) ||
            throw(ArgumentError("ndarray byte extent must fit UInt32"))
        numerator, denominator = rate
        numerator > 0 && denominator > 0 ||
            throw(ArgumentError("Port rate terms must be positive"))
        divisor = gcd(numerator, denominator)
        numerator = div(numerator, divisor)
        denominator = div(denominator, divisor)
        numerator <= typemax(UInt32) && denominator <= typemax(UInt32) ||
            throw(ArgumentError("reduced Port rate terms must fit UInt32"))
        return new{N}(
            T,
            layout,
            converted,
            String(schema),
            (UInt32(numerator), UInt32(denominator)),
        )
    end
end

# Declared Port names are type parameters, so ordering is fixed before numerical
# execution. Guard the exact key set: a NamedTuple constructor can drop extras.
function _order_port_values(
    context::AbstractString, ::Val{Names}, supplied::NamedTuple{Given},
) where {Names,Given}
    Names === Given && return supplied
    (length(Names) == length(Given) && all(name -> name in Given, Names)) || throw(
        ArgumentError("$context names must be $Names, received $Given"),
    )
    return NamedTuple{Names}(supplied)
end

_declared_array_description(::Nothing) = "declared array"
_declared_array_description(context::Tuple{AbstractString,Symbol}) =
    "$(context[1]) Port '$(context[2])'"

function _validate_declared_array(
    format::AlgorithmFormat, array::AbstractArray, context=nothing,
)
    eltype(array) === format.element_type || throw(
        ArgumentError(
            "$(_declared_array_description(context)) element type " *
            "$(format.element_type) does not match $(eltype(array))",
        ),
    )
    Base.has_offset_axes(array) && throw(
        ArgumentError("$(_declared_array_description(context)) requires one-based indexing"),
    )
    size(array) == format.shape || throw(
        DimensionMismatch(
            "$(_declared_array_description(context)) shape " *
            "$(format.shape) does not match $(size(array))",
        ),
    )
    return array
end

function _validate_declared_array(format::AlgorithmFormat, value, context=nothing)
    throw(ArgumentError(
        "$(_declared_array_description(context)) requires an AbstractArray, received $(typeof(value))",
    ))
end
