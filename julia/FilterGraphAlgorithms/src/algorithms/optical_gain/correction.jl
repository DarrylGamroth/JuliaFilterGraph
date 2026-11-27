"""
    OpticalGainCorrectionPlan(optical_gains)

Prepare reciprocal correction for reconstructed modes. `optical_gains` are
physical sensor-response ratios, not already-inverted correction factors. The
Plan owns both the validated gains and their prepared reciprocals.
"""
struct OpticalGainCorrectionPlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    optical_gains::Vector{T}
    reciprocal_corrections::Vector{T}

    function OpticalGainCorrectionPlan(
        optical_gains::AbstractVector{T},
    ) where {T<:AbstractFloat}
        Base.require_one_based_indexing(optical_gains)
        isempty(optical_gains) &&
            throw(ArgumentError("optical-gain correction must be nonempty"))
        gains = Vector{T}(undef, length(optical_gains))
        reciprocals = similar(gains)
        @inbounds for mode in eachindex(optical_gains)
            gain = optical_gains[mode]
            isfinite(gain) && gain > zero(T) || throw(
                ArgumentError(
                    "physical optical gain for mode $(mode - 1) must be finite and positive",
                ),
            )
            reciprocal = one(T) / gain
            isfinite(reciprocal) || throw(
                ArgumentError(
                    "reciprocal optical-gain correction for mode $(mode - 1) is not finite",
                ),
            )
            gains[mode] = gain
            reciprocals[mode] = reciprocal
        end
        return new{T}(gains, reciprocals)
    end
end

"""
    optical_gains(plan::OpticalGainCorrectionPlan)

Return a copy of the prepared physical optical gains.
"""
optical_gains(plan::OpticalGainCorrectionPlan) = copy(plan.optical_gains)

"""
    optical_gain_reciprocal_corrections(plan::OpticalGainCorrectionPlan)

Return a copy of the reciprocal corrections used during processing.
"""
optical_gain_reciprocal_corrections(plan::OpticalGainCorrectionPlan) =
    copy(plan.reciprocal_corrections)

"""Expected numerical rejection from reciprocal optical-gain correction."""
@enum OpticalGainCorrectionFailure::UInt8 begin
    OpticalGainCorrectionNonFiniteInput = 1
    OpticalGainCorrectionNonFiniteOutput = 2
end

@doc "A reconstructed input mode is non-finite."
OpticalGainCorrectionNonFiniteInput
@doc "Applying a reciprocal correction would produce a non-finite mode."
OpticalGainCorrectionNonFiniteOutput

function _process_optical_gain_correction!(
    corrected::AbstractVector{T},
    plan::OpticalGainCorrectionPlan{T},
    reconstructed::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(corrected, reconstructed)
    expected = length(plan.optical_gains)
    length(reconstructed) == expected ||
        throw(DimensionMismatch("reconstructed modal extent is invalid"))
    length(corrected) == expected ||
        throw(DimensionMismatch("corrected modal extent is invalid"))
    Base.mightalias(corrected, reconstructed) && throw(
        ArgumentError("reconstructed and corrected storage must not alias"),
    )
    Base.mightalias(corrected, plan.optical_gains) && throw(
        ArgumentError("corrected output must not alias prepared optical gains"),
    )
    Base.mightalias(corrected, plan.reciprocal_corrections) && throw(
        ArgumentError("corrected output must not alias prepared corrections"),
    )

    @inbounds for mode in eachindex(reconstructed)
        value = reconstructed[mode]
        isfinite(value) || return OpticalGainCorrectionNonFiniteInput
        isfinite(value * plan.reciprocal_corrections[mode]) ||
            return OpticalGainCorrectionNonFiniteOutput
    end
    @inbounds @simd for mode in eachindex(corrected, reconstructed)
        corrected[mode] =
            reconstructed[mode] * plan.reciprocal_corrections[mode]
    end
    return nothing
end

"""
Float32 reciprocal correction of reconstructed modes by prepared physical
optical gain. This Node performs correction only; estimation and bounded gain
adoption remain separate Nodes in the optical-gain Filter Graph.
"""
@algorithm OpticalGainCorrectionF32 begin
    label = "optical-gain-correction-f32"
    failure = OpticalGainCorrectionFailure

    """Configuration for [`OpticalGainCorrectionF32`](@ref)."""
    @configuration OpticalGainCorrectionF32Configuration begin
        optical_gains::Vector{Float32} => construction
    end

    ports(configuration) = (
        input(
            :reconstructed,
            schema_element_type(RECONSTRUCTED_MODES_V1),
            (length(configuration.optical_gains),),
            RECONSTRUCTED_MODES_V1,
        ),
        output(
            :corrected,
            schema_element_type(RECONSTRUCTED_MODES_V1),
            (length(configuration.optical_gains),),
            RECONSTRUCTED_MODES_V1;
            metadata_from=:reconstructed,
        ),
        parameter(
            :optical_gains,
            schema_element_type(PHYSICAL_OPTICAL_GAIN_V1),
            (length(configuration.optical_gains),),
            PHYSICAL_OPTICAL_GAIN_V1;
            name="optical-gains",
            replace=(_plan, values) -> OpticalGainCorrectionPlan(values),
        ),
    )

    prepare(configuration) = OpticalGainCorrectionPlan(
        configuration.optical_gains,
    )

    process!(plan) = begin
        _process_optical_gain_correction!(corrected, plan, reconstructed)
    end
end
