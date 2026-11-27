"""
    PdmSystemFlatPlan(system_flat)

Prepare an additive physical-deformable-mirror system flat. The flat is copied
into dense Julia storage and must be nonempty and finite.
"""
struct PdmSystemFlatPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    system_flat::Vector{T}

    function PdmSystemFlatPlan(
        system_flat::AbstractVector{T},
    ) where {T<:Union{Float32,Float64}}
        Base.require_one_based_indexing(system_flat)
        isempty(system_flat) && throw(
            ArgumentError("PDM system-flat extent must be nonempty"),
        )
        all(isfinite, system_flat) || throw(
            ArgumentError("PDM system flat must be finite"),
        )
        return new{T}(Vector{T}(system_flat))
    end
end

function _replace_pdm_system_flat(
    plan::PdmSystemFlatPlan{T},
    system_flat::AbstractVector{T},
) where {T}
    length(system_flat) == length(plan.system_flat) || throw(
        DimensionMismatch("PDM system-flat extent is invalid"),
    )
    return PdmSystemFlatPlan(system_flat)
end

"""Expected numerical rejection from PDM system-flat addition."""
@enum PdmSystemFlatFailure::UInt8 begin
    PdmSystemFlatNonFiniteInput = 1
    PdmSystemFlatNonFiniteOutput = 2
end

@doc "The requested PDM command contains a non-finite actuator value."
PdmSystemFlatNonFiniteInput
@doc "PDM system-flat addition produced a non-finite actuator value."
PdmSystemFlatNonFiniteOutput

function _process_pdm_system_flat!(
    requested_with_system_flat::AbstractVector{T},
    plan::PdmSystemFlatPlan{T},
    requested::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(requested_with_system_flat, requested)
    system_flat = plan.system_flat
    extent = length(system_flat)
    length(requested) == extent || throw(
        DimensionMismatch("requested PDM command extent is invalid"),
    )
    length(requested_with_system_flat) == extent || throw(
        DimensionMismatch("system-flat PDM command extent is invalid"),
    )
    requested_with_system_flat === requested ||
        !Base.mightalias(requested_with_system_flat, requested) || throw(
        ArgumentError("partially overlapping requested and system-flat commands are unsupported"),
    )
    Base.mightalias(requested_with_system_flat, system_flat) && throw(
        ArgumentError("system-flat PDM output must not alias the prepared system flat"),
    )

    @inbounds for index in eachindex(requested, system_flat)
        requested_value = requested[index]
        isfinite(requested_value) || return PdmSystemFlatNonFiniteInput
        isfinite(requested_value + system_flat[index]) ||
            return PdmSystemFlatNonFiniteOutput
    end
    @inbounds for index in eachindex(requested_with_system_flat, requested, system_flat)
        requested_with_system_flat[index] = requested[index] + system_flat[index]
    end
    return nothing
end

"""
Float32 addition of a configured physical-DM system flat to a requested command.
"""
@algorithm PdmSystemFlatF32 begin
    label = "pdm-system-flat-f32"
    failure = PdmSystemFlatFailure

    """Configuration for [`PdmSystemFlatF32`](@ref)."""
    @configuration PdmSystemFlatF32Configuration begin
        actuator_count::Int => construction
    end

    ports(configuration) = (
        input(
            :requested,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1,
        ),
        output(
            :requested_with_system_flat,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            name="requested-with-system-flat",
            metadata_from=:requested,
        ),
        parameter(
            :system_flat,
            schema_element_type(PDM_SYSTEM_FLAT_V1),
            (configuration.actuator_count,),
            PDM_SYSTEM_FLAT_V1;
            name="system-flat",
            replace=(plan, values) -> _replace_pdm_system_flat(plan, values),
        ),
    )

    prepare(configuration) = begin
        configuration.actuator_count > 0 || throw(
            ArgumentError("PDM system-flat actuator count must be positive"),
        )
        PdmSystemFlatPlan(zeros(Float32, configuration.actuator_count))
    end

    process!(plan) = begin
        _process_pdm_system_flat!(requested_with_system_flat, plan, requested)
    end
end
