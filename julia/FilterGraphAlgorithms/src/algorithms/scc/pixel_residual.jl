"""
    SccPixelResidualPlan(reference_image, mask, minimum_valid_fraction)

Prepare the reference subtraction and ordered dark-hole pixel selection for
the simplified SCC measurement.

`reference_image` and `mask` use the complete chopped-image pixel order. The
selected residual order is the increasing linear order of the `true` mask
entries and must match the columns of the prepared SCC reconstructor. Invalid
or non-finite selected pixels contribute zero. A complete chopped pair is
accepted only when at least `minimum_valid_fraction` of the selected pixels
remain valid.

Full- and half-dark-hole operation use separate coherent Plans; there is no
runtime dark-hole switch.
"""
struct SccPixelResidualPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    reference_image::Vector{T}
    mask::Vector{Bool}
    selected_indices::Vector{Int}
    minimum_valid_fraction::T
    minimum_valid_count::Int
end

function SccPixelResidualPlan(
    reference_image::AbstractVector{T},
    mask::AbstractVector{Bool},
    minimum_valid_fraction::Real,
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(reference_image, mask)
    isempty(reference_image) &&
        throw(ArgumentError("SCC reference image must be nonempty"))
    length(mask) == length(reference_image) || throw(
        DimensionMismatch("SCC mask and reference image extents must match"),
    )
    all(isfinite, reference_image) ||
        throw(ArgumentError("SCC reference image must be finite"))
    selected_indices = findall(mask)
    isempty(selected_indices) &&
        throw(ArgumentError("SCC mask must select at least one pixel"))
    fraction = T(minimum_valid_fraction)
    isfinite(fraction) && zero(T) < fraction <= one(T) || throw(
        ArgumentError("minimum_valid_fraction must be finite and in (0, 1]"),
    )
    minimum_valid_count = ceil(
        Int,
        Float64(fraction) * length(selected_indices),
    )
    return SccPixelResidualPlan{T}(
        Vector(reference_image),
        Vector(mask),
        selected_indices,
        fraction,
        minimum_valid_count,
    )
end

function _replace_scc_reference_image(
    plan::SccPixelResidualPlan,
    reference_image,
)
    return SccPixelResidualPlan(
        reference_image,
        plan.mask,
        plan.minimum_valid_fraction,
    )
end

function _replace_scc_pixel_mask(plan::SccPixelResidualPlan, mask)
    replacement = SccPixelResidualPlan(
        plan.reference_image,
        mask,
        plan.minimum_valid_fraction,
    )
    length(replacement.selected_indices) == length(plan.selected_indices) ||
        throw(
            DimensionMismatch(
                "replacement SCC mask changes the selected-pixel count",
            ),
        )
    return replacement
end

"""Expected unusable-frame dispositions from SCC pixel-residual formation."""
@enum SccPixelResidualFailure::UInt8 begin
    SccPixelResidualIncompletePair = 1
    SccPixelResidualInvalidPairStatus = 2
    SccPixelResidualInsufficientValidPixels = 3
end

@doc "The current input does not represent a complete fringed/unfringed pair."
SccPixelResidualIncompletePair
@doc "The chopped-image pair status is not a defined `ChoppedImagePairStatus`."
SccPixelResidualInvalidPairStatus
@doc "Too few selected SCC pixels remain valid for the prepared profile."
SccPixelResidualInsufficientValidPixels

function _check_scc_pixel_residual_storage(
    selected_residuals::AbstractVector,
    selected_validity::AbstractVector,
    valid_fraction::AbstractVector,
    plan::SccPixelResidualPlan,
    difference::AbstractVector,
    difference_validity::AbstractVector,
    pair_status::AbstractVector,
)
    Base.require_one_based_indexing(
        selected_residuals,
        selected_validity,
        valid_fraction,
        difference,
        difference_validity,
        pair_status,
    )
    pixel_count = length(plan.reference_image)
    selected_count = length(plan.selected_indices)
    length(difference) == pixel_count ||
        throw(DimensionMismatch("SCC chopped-image difference extent is invalid"))
    length(difference_validity) == pixel_count || throw(
        DimensionMismatch("SCC chopped-image difference validity extent is invalid"),
    )
    length(pair_status) == 1 ||
        throw(DimensionMismatch("SCC chopped-image pair-status extent is invalid"))
    length(selected_residuals) == selected_count ||
        throw(DimensionMismatch("SCC selected-residual extent is invalid"))
    length(selected_validity) == selected_count ||
        throw(DimensionMismatch("SCC selected-validity extent is invalid"))
    length(valid_fraction) == 1 ||
        throw(DimensionMismatch("SCC valid-fraction extent is invalid"))
    Base.mightalias(selected_residuals, difference) && throw(
        ArgumentError("SCC selected residuals must not alias the input difference"),
    )
    Base.mightalias(selected_residuals, plan.reference_image) && throw(
        ArgumentError("SCC selected residuals must not alias the prepared reference"),
    )
    return nothing
end

_can_use_contiguous_scc_residual(
    residuals, validity, difference, difference_validity, reference, indices,
) = false

function _can_use_contiguous_scc_residual(
    residuals::Vector{Float32},
    validity::Vector{Bool},
    difference::Vector{Float32},
    difference_validity::Vector{Bool},
    reference::Vector{Float32},
    indices::Vector{Int},
)
    length(indices) == length(difference) || return false
    # Distinct Vectors can wrap overlapping memory, including across element types.
    _scc_storage_disjoint(residuals, difference) &&
        _scc_storage_disjoint(residuals, difference_validity) &&
        _scc_storage_disjoint(residuals, reference) &&
        _scc_storage_disjoint(residuals, indices) &&
        _scc_storage_disjoint(residuals, validity) &&
        _scc_storage_disjoint(validity, difference) &&
        _scc_storage_disjoint(validity, difference_validity) &&
        _scc_storage_disjoint(validity, reference) &&
        _scc_storage_disjoint(validity, indices) || return false
    # Separate reductions let validity bytes, indices and operand bits use
    # their natural SIMD widths. Reject invalid input before either other scan.
    complete = true
    @inbounds @simd for i in eachindex(difference_validity)
        complete &= difference_validity[i]
    end
    complete || return false
    # The prepared index vector is mutable. A differing index sets a bit.
    mismatch = 0
    @inbounds @simd for i in eachindex(indices)
        mismatch |= indices[i] ⊻ i
    end
    iszero(mismatch) || return false
    # Classify operands without evaluating any floating-point arithmetic.
    eligible = true
    @inbounds @simd for i in eachindex(difference)
        eligible &= _scc_finite_difference_operand(difference[i]) &
            _scc_finite_difference_operand(reference[i])
    end
    return eligible
end

function _process_contiguous_scc_residual!(
    residuals::Vector{Float32},
    validity::Vector{Bool},
    difference::Vector{Float32},
    reference::Vector{Float32},
)
    # The eligibility check establishes disjoint output storage and finite results.
    @inbounds @simd ivdep for i in eachindex(difference)
        residuals[i] = difference[i] - reference[i]
        validity[i] = true
    end
    return length(difference)
end

"""
    process!(selected_residuals, selected_validity, valid_fraction, plan,
             difference, difference_validity, pair_status)

Form one ordered SCC selected-pixel residual vector. `pair_status` must contain
`UInt8(ChoppedImageComplete)`. Upstream-invalid, non-finite, or overflowing
selected pixels are written as zero with false selected validity. The valid
fraction is always calculated for a complete pair; a fraction below the
prepared minimum returns [`SccPixelResidualInsufficientValidPixels`](@ref).
"""
function process!(
    selected_residuals::AbstractVector{T},
    selected_validity::AbstractVector{Bool},
    valid_fraction::AbstractVector{T},
    plan::SccPixelResidualPlan{T},
    difference::AbstractVector{T},
    difference_validity::AbstractVector{Bool},
    pair_status::AbstractVector{UInt8},
) where {T<:Union{Float32,Float64}}
    _check_scc_pixel_residual_storage(
        selected_residuals,
        selected_validity,
        valid_fraction,
        plan,
        difference,
        difference_validity,
        pair_status,
    )
    status = pair_status[1]
    status == UInt8(ChoppedImagePriming) &&
        return SccPixelResidualIncompletePair
    status == UInt8(ChoppedImageComplete) ||
        return SccPixelResidualInvalidPairStatus

    valid_count = 0
    if _can_use_contiguous_scc_residual(
           selected_residuals,
           selected_validity,
           difference,
           difference_validity,
           plan.reference_image,
           plan.selected_indices,
       )
        valid_count = _process_contiguous_scc_residual!(
            selected_residuals,
            selected_validity,
            difference,
            plan.reference_image,
        )
    else
        @inbounds for selected in eachindex(plan.selected_indices)
            pixel = plan.selected_indices[selected]
            valid = difference_validity[pixel]
            value = zero(T)
            if valid
                sample = difference[pixel]
                valid = isfinite(sample)
                if valid
                    value = sample - plan.reference_image[pixel]
                    valid = isfinite(value)
                end
            end
            selected_residuals[selected] = valid ? value : zero(T)
            selected_validity[selected] = valid
            valid_count += valid
        end
    end
    valid_fraction[1] = T(valid_count) / T(length(plan.selected_indices))
    valid_count >= plan.minimum_valid_count ||
        return SccPixelResidualInsufficientValidPixels
    return nothing
end

"""
Float32 simplified SCC pixel-residual formation. The Node subtracts a
prepared reference from a complete chopped-image difference, gathers one
prepared dark-hole profile, publishes selected validity and valid fraction,
and rejects unusable pairs before downstream controller state advances.
"""
@algorithm SccPixelResidualF32 begin
    label = "scc-pixel-residual-f32"
    failure = SccPixelResidualFailure

    """Configuration for [`SccPixelResidualF32`](@ref)."""
    @configuration SccPixelResidualF32Configuration begin
        pixel_count::Int => graph_rebuild
        mask::Vector{Bool} => graph_rebuild
        initial_reference_image::Union{Nothing,Vector{Float32}} => construction
        minimum_valid_fraction::Float32 => construction
    end

    ports(configuration) = (
        input(
            :difference,
            schema_element_type(SCC_CHOPPED_IMAGE_DIFFERENCE_V1),
            (configuration.pixel_count,),
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        ),
        input(
            :difference_validity,
            schema_element_type(SCC_CHOPPED_DIFFERENCE_VALIDITY_V1),
            (configuration.pixel_count,),
            SCC_CHOPPED_DIFFERENCE_VALIDITY_V1;
            name="difference-validity",
        ),
        input(
            :pair_status,
            schema_element_type(SCC_CHOPPED_DIFFERENCE_STATUS_V1),
            (1,),
            SCC_CHOPPED_DIFFERENCE_STATUS_V1;
            name="pair-status",
        ),
        output(
            :selected_residuals,
            schema_element_type(SCC_SELECTED_PIXEL_RESIDUALS_V1),
            (count(configuration.mask),),
            SCC_SELECTED_PIXEL_RESIDUALS_V1;
            name="selected-residuals",
            metadata_from=:difference,
        ),
        output(
            :selected_validity,
            schema_element_type(SCC_SELECTED_PIXEL_VALIDITY_V1),
            (count(configuration.mask),),
            SCC_SELECTED_PIXEL_VALIDITY_V1;
            name="selected-validity",
            metadata_from=:difference,
        ),
        output(
            :valid_fraction,
            schema_element_type(SCC_SELECTED_PIXEL_VALID_FRACTION_V1),
            (1,),
            SCC_SELECTED_PIXEL_VALID_FRACTION_V1;
            name="valid-fraction",
            metadata_from=:difference,
        ),
        parameter(
            :reference_image,
            schema_element_type(SCC_REFERENCE_IMAGE_V1),
            (configuration.pixel_count,),
            SCC_REFERENCE_IMAGE_V1;
            name="reference-image",
            replace=(plan, values) ->
                _replace_scc_reference_image(plan, values),
        ),
        parameter(
            :mask,
            schema_element_type(PIXEL_SELECTION_MASK_V1),
            (configuration.pixel_count,),
            PIXEL_SELECTION_MASK_V1;
            replace=(plan, values) -> _replace_scc_pixel_mask(plan, values),
        ),
    )

    prepare(configuration) = begin
        configuration.pixel_count > 0 ||
            throw(ArgumentError("pixel_count must be positive"))
        length(configuration.mask) == configuration.pixel_count || throw(
            DimensionMismatch("SCC mask extent must equal pixel_count"),
        )
        reference_image = if isnothing(configuration.initial_reference_image)
            zeros(Float32, configuration.pixel_count)
        else
            configuration.initial_reference_image
        end
        SccPixelResidualPlan(
            reference_image,
            configuration.mask,
            configuration.minimum_valid_fraction,
        )
    end

    process!(plan) = begin
        process!(
            selected_residuals,
            selected_validity,
            valid_fraction,
            plan,
            difference,
            difference_validity,
            pair_status,
        )
    end
end
