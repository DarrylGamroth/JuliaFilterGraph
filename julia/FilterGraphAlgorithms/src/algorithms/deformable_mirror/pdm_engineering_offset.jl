"""
    PdmEngineeringOffsetPlan(engineering_offset)

Prepare an additive physical-deformable-mirror engineering offset in µm wavefront. The offset is copied
into dense Julia storage and must be nonempty and finite.
"""
struct PdmEngineeringOffsetPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    engineering_offset::Vector{T}

    function PdmEngineeringOffsetPlan(
        engineering_offset::AbstractVector{T},
    ) where {T<:Union{Float32,Float64}}
        Base.require_one_based_indexing(engineering_offset)
        isempty(engineering_offset) && throw(
            ArgumentError("PDM engineering-offset extent must be nonempty"),
        )
        all(isfinite, engineering_offset) || throw(
            ArgumentError("PDM engineering offset must be finite"),
        )
        return new{T}(Vector{T}(engineering_offset))
    end
end

function _replace_pdm_engineering_offset(
    plan::PdmEngineeringOffsetPlan{T},
    engineering_offset::AbstractVector{T},
) where {T}
    length(engineering_offset) == length(plan.engineering_offset) || throw(
        DimensionMismatch("PDM engineering-offset extent is invalid"),
    )
    return PdmEngineeringOffsetPlan(engineering_offset)
end

"""Expected numerical rejection from PDM engineering-offset addition."""
@enum PdmEngineeringOffsetFailure::UInt8 begin
    PdmEngineeringOffsetNonFiniteInput = 1
    PdmEngineeringOffsetNonFiniteOutput = 2
end

@doc "The requested PDM command contains a non-finite actuator value."
PdmEngineeringOffsetNonFiniteInput
@doc "PDM engineering-offset addition produced a non-finite actuator value."
PdmEngineeringOffsetNonFiniteOutput

# Preserve the exact requested value, including signed zero, for a zero offset.
_pdm_offset_sum(requested, offset) = iszero(offset) ? requested : requested + offset

function _process_pdm_engineering_offset!(
    requested_with_offset::AbstractVector{T},
    current_offset::AbstractVector{T},
    plan::PdmEngineeringOffsetPlan{T},
    requested::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(requested_with_offset, current_offset, requested)
    engineering_offset = plan.engineering_offset
    extent = length(engineering_offset)
    length(requested) == extent || throw(
        DimensionMismatch("requested PDM command extent is invalid"),
    )
    length(requested_with_offset) == extent || throw(
        DimensionMismatch("engineering-offset PDM command extent is invalid"),
    )
    length(current_offset) == extent || throw(
        DimensionMismatch("current PDM engineering-offset extent is invalid"),
    )
    (Base.mightalias(current_offset, requested_with_offset) ||
     Base.mightalias(current_offset, requested) ||
     Base.mightalias(current_offset, engineering_offset)) && throw(
        ArgumentError("current offset output must not alias commands or the prepared offset"),
    )
    requested_with_offset === requested ||
        !Base.mightalias(requested_with_offset, requested) || throw(
        ArgumentError("partially overlapping requested and engineering-offset commands are unsupported"),
    )
    Base.mightalias(requested_with_offset, engineering_offset) && throw(
        ArgumentError("engineering-offset PDM output must not alias the prepared engineering offset"),
    )

    @inbounds for index in eachindex(requested, engineering_offset)
        requested_value = requested[index]
        isfinite(requested_value) || return PdmEngineeringOffsetNonFiniteInput
        isfinite(_pdm_offset_sum(requested_value, engineering_offset[index])) ||
            return PdmEngineeringOffsetNonFiniteOutput
    end
    @inbounds for index in eachindex(requested_with_offset, current_offset, requested, engineering_offset)
        requested_with_offset[index] = _pdm_offset_sum(requested[index], engineering_offset[index])
        current_offset[index] = engineering_offset[index]
    end
    return nothing
end

"""
Float32 addition of an adopted physical-DM engineering offset to a requested command.
The current-offset output reports the adopted parameter for each successful frame.
Connect requested-with-offset to the existing PDM command limiter.
"""
@algorithm PdmEngineeringOffsetF32 begin
    label = "pdm-engineering-offset-f32"
    failure = PdmEngineeringOffsetFailure

    """Configuration for [`PdmEngineeringOffsetF32`](@ref)."""
    @configuration PdmEngineeringOffsetF32Configuration begin
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
            :requested_with_offset,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            name="requested-with-offset",
            metadata_from=:requested,
        ),
        output(
            :current_offset,
            schema_element_type(PDM_ENGINEERING_OFFSET_V1),
            (configuration.actuator_count,),
            PDM_ENGINEERING_OFFSET_V1;
            name="current-offset",
            metadata_from=:requested,
        ),
        parameter(
            :engineering_offset,
            schema_element_type(PDM_ENGINEERING_OFFSET_V1),
            (configuration.actuator_count,),
            PDM_ENGINEERING_OFFSET_V1;
            name="engineering-offset",
            replace=(plan, values) -> _replace_pdm_engineering_offset(plan, values),
        ),
    )

    prepare(configuration) = begin
        configuration.actuator_count > 0 || throw(
            ArgumentError("PDM engineering-offset actuator count must be positive"),
        )
        PdmEngineeringOffsetPlan(zeros(Float32, configuration.actuator_count))
    end

    process!(plan) = begin
        _process_pdm_engineering_offset!(requested_with_offset, current_offset, plan, requested)
    end
end
