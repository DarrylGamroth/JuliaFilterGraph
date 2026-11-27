"""Marker supertype for transport-neutral algorithm graph declarations."""
abstract type AbstractAlgorithmDeclaration end

"""
Mathematical operation shape exposed to execution adapters.

An operation marker contains no worker, device, scheduling, partition, pointer,
or transport policy. Every declaration defaults to [`GeneralOperation`](@ref);
an eligible declaration may select a more specific marker while retaining its
ordinary `process!` body as the scientific reference.
"""
abstract type AlgorithmOperation end

"""An Algorithm that supplies only its ordinary typed `process!` method."""
struct GeneralOperation <: AlgorithmOperation end

"""One Float32 dense operation `output = matrix * input`."""
struct DenseF32Operation <: AlgorithmOperation end

"""Complete-image Float32 Shack--Hartmann measurements over independent regions."""
struct ShackHartmannImageF32Operation <: AlgorithmOperation end

"""Complete-image Float32 weighted Shack--Hartmann centroids over independent regions."""
struct ShackHartmannWeightedCentroidF32Operation <: AlgorithmOperation end

"""Progressive Float32 Shack--Hartmann measurement and reconstruction."""
struct ShackHartmannRowReconstructionF32Operation <: AlgorithmOperation end

"""Progressive Float32 pyramid normalization and reconstruction."""
struct PyramidRowReconstructionF32Operation <: AlgorithmOperation end

"""Marker supertype for construction-configuration update behavior."""
abstract type ConfigurationUpdate end

"""A configuration value used only while preparing one algorithm instance."""
struct ConstructionConfiguration <: ConfigurationUpdate end

"""A configuration value whose change requires graph reconstruction."""
struct GraphRebuildConfiguration <: ConfigurationUpdate end

"""One typed configuration field; `default` is `nothing` when required or `Some(value)`."""
struct AlgorithmConfiguration
    name::Symbol
    julia_type::Type
    update::ConfigurationUpdate
    default::Union{Nothing,Some}
end

AlgorithmConfiguration(name::Symbol, julia_type::Type, update::ConfigurationUpdate) =
    AlgorithmConfiguration(name, julia_type, update, nothing)

"""Marker supertype for a declaration's scalar-property capability."""
abstract type DeclaredProperties end

"""A declaration without a scalar-property surface."""
struct NoDeclaredProperties <: DeclaredProperties end

"""A declaration whose plan implements the portable property interface."""
struct RuntimeDeclaredProperties <: DeclaredProperties end

"""
    algorithm_label(::Type{<:AbstractAlgorithmDeclaration})

Return the stable algorithm label for one declaration type.
"""
function algorithm_label end

"""
    algorithm_configuration(::Type{<:AbstractAlgorithmDeclaration})

Return typed construction fields in declaration order.
"""
function algorithm_configuration end

"""
    algorithm_configuration_type(::Type{<:AbstractAlgorithmDeclaration})

Return the concrete typed configuration constructed for one declaration.
"""
function algorithm_configuration_type end

"""
    algorithm_ports(::Type{<:AbstractAlgorithmDeclaration})

Return static ports in declaration order.
"""
function algorithm_ports end

"""
    algorithm_property_policy(::Type{<:AbstractAlgorithmDeclaration})

Return the declaration's scalar-property capability.
"""
function algorithm_property_policy end

"""
    algorithm_operation(::Type{<:AbstractAlgorithmDeclaration})

Return the transport-neutral mathematical operation shape for a declaration.

Execution adapters may recognize this marker and partition the implementation
without changing the scientific Algorithm. Unrecognized and general operations
must continue through the ordinary serial `process!` implementation.
"""
algorithm_operation(::Type{<:AbstractAlgorithmDeclaration}) = GeneralOperation()

"""
    algorithm_failure_type(::Type{<:AbstractAlgorithmDeclaration})

Return the declared expected-failure enum type, or `Union{}` when absent.
"""
algorithm_failure_type(::Type{<:AbstractAlgorithmDeclaration}) = Union{}
