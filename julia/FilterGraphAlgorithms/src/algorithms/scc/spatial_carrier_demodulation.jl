"""Mask-aware spatial carrier demodulation recognized by execution adapters."""
struct SccSpatialCarrierDemodulationF32Operation <: AlgorithmOperation end

"""
Prepared detector geometry, carrier, FIR, masks, and selected measurement order
for mask-aware SCC spatial carrier demodulation.
"""
struct SccSpatialCarrierDemodulationPlan{
    T<:Union{Float32,Float64},
    R<:AbstractVector{T},
    S<:AbstractVector{Bool},
    V<:AbstractVector{Bool},
    P<:AbstractVector{T},
    C<:AbstractVector{Complex{T}},
    H<:AbstractVector{T},
    I<:AbstractVector{Int},
} <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    reference_difference::R
    static_validity::S
    roi::V
    carrier_phase::P
    carrier_conjugate::C
    fir::H
    selected_indices::I
    minimum_support::T
    minimum_valid_fraction::T
    minimum_valid_count::Int
end

function _scc_spatial_carrier_vector(
    ::Type{T},
    pixel_count::Int,
    values::Union{Nothing,AbstractVector},
    default,
    name::AbstractString,
) where {T}
    if isnothing(values)
        return fill(T(default), pixel_count)
    end
    Base.require_one_based_indexing(values)
    length(values) == pixel_count ||
        throw(DimensionMismatch("$name extent does not match the detector"))
    result = Vector{T}(values)
    all(isfinite, result) || throw(ArgumentError("$name must be finite"))
    return result
end

function _scc_spatial_carrier_validity(
    pixel_count::Int,
    values::Union{Nothing,AbstractVector{Bool}},
)
    isnothing(values) && return fill(true, pixel_count)
    Base.require_one_based_indexing(values)
    length(values) == pixel_count || throw(
        DimensionMismatch("static validity extent does not match the detector"),
    )
    return Vector{Bool}(values)
end

function _scc_spatial_carrier_fir(
    ::Type{T},
    tap_count::Int,
    values::Union{Nothing,AbstractVector},
) where {T}
    tap_count > 0 && isodd(tap_count) ||
        throw(ArgumentError("spatial FIR tap count must be positive and odd"))
    fir = if isnothing(values)
        result = zeros(T, tap_count)
        result[(tap_count + 1) >> 1] = one(T)
        result
    else
        Base.require_one_based_indexing(values)
        length(values) == tap_count ||
            throw(DimensionMismatch("spatial FIR extent does not match its tap count"))
        Vector{T}(values)
    end
    all(isfinite, fir) || throw(ArgumentError("spatial FIR coefficients must be finite"))
    normalization = sum(fir)
    isfinite(normalization) && !iszero(normalization) ||
        throw(ArgumentError("spatial FIR must have a finite nonzero sum"))
    fir ./= normalization
    return fir
end

function _scc_spatial_carrier_indices(roi::AbstractVector{Bool}, pixel_count::Int)
    Base.require_one_based_indexing(roi)
    length(roi) == pixel_count ||
        throw(DimensionMismatch("SCC ROI extent does not match the detector"))
    selected = findall(roi)
    isempty(selected) && throw(ArgumentError("SCC ROI must select at least one pixel"))
    return selected
end

function SccSpatialCarrierDemodulationPlan(
    ::Type{T},
    image_size::NTuple{2,Int},
    roi::AbstractVector{Bool},
    fir_tap_count::Int;
    reference_difference::Union{Nothing,AbstractVector}=nothing,
    static_validity::Union{Nothing,AbstractVector{Bool}}=nothing,
    carrier_phase::Union{Nothing,AbstractVector}=nothing,
    fir::Union{Nothing,AbstractVector}=nothing,
    minimum_support::Real=T(0.8),
    minimum_valid_fraction::Real=one(T),
) where {T<:Union{Float32,Float64}}
    image_rows, image_columns = image_size
    image_rows > 0 && image_columns > 0 ||
        throw(ArgumentError("SCC detector dimensions must be positive"))
    pixel_count = Base.Checked.checked_mul(image_rows, image_columns)
    reference = _scc_spatial_carrier_vector(
        T,
        pixel_count,
        reference_difference,
        zero(T),
        "reference difference",
    )
    validity = _scc_spatial_carrier_validity(pixel_count, static_validity)
    phase = _scc_spatial_carrier_vector(
        T,
        pixel_count,
        carrier_phase,
        zero(T),
        "carrier phase",
    )
    carrier = similar(phase, Complex{T})
    @inbounds for index in eachindex(carrier, phase)
        carrier[index] = cis(-phase[index])
    end
    coefficients = _scc_spatial_carrier_fir(T, fir_tap_count, fir)
    selected = _scc_spatial_carrier_indices(roi, pixel_count)
    prepared_roi = Vector{Bool}(roi)
    support = T(minimum_support)
    isfinite(support) && support > zero(T) ||
        throw(ArgumentError("minimum support must be finite and positive"))
    fraction = T(minimum_valid_fraction)
    isfinite(fraction) && zero(T) < fraction <= one(T) || throw(
        ArgumentError("minimum valid fraction must be finite and in (0, 1]"),
    )
    minimum_count = ceil(Int, fraction * T(length(selected)))
    return SccSpatialCarrierDemodulationPlan(
        image_size,
        reference,
        validity,
        prepared_roi,
        phase,
        carrier,
        coefficients,
        selected,
        support,
        fraction,
        minimum_count,
    )
end

function _replace_scc_spatial_carrier_plan(
    plan::SccSpatialCarrierDemodulationPlan{T};
    reference_difference=plan.reference_difference,
    static_validity=plan.static_validity,
    roi=plan.roi,
    carrier_phase=plan.carrier_phase,
    fir=plan.fir,
) where {T}
    replacement = SccSpatialCarrierDemodulationPlan(
        T,
        plan.image_size,
        roi,
        length(plan.fir);
        reference_difference,
        static_validity,
        carrier_phase,
        fir,
        minimum_support=plan.minimum_support,
        minimum_valid_fraction=plan.minimum_valid_fraction,
    )
    length(replacement.selected_indices) == length(plan.selected_indices) ||
        throw(ArgumentError("replacement SCC ROI must preserve measurement extent"))
    return replacement
end

"""Reusable numerator, support, and candidate-output storage for SCC demodulation."""
struct SccSpatialCarrierDemodulationWorkspace{
    T<:Union{Float32,Float64},
    C<:AbstractVector{Complex{T}},
    R<:AbstractVector{T},
    V<:AbstractVector{Bool},
}
    masked_carrier::C
    horizontal_carrier::C
    mask::R
    horizontal_support::R
    candidate_measurements::R
    candidate_weights::R
    candidate_validity::V
end

function SccSpatialCarrierDemodulationWorkspace(
    plan::SccSpatialCarrierDemodulationPlan{T},
) where {T}
    pixel_count = prod(plan.image_size)
    measurement_count = 2 * length(plan.selected_indices)
    return SccSpatialCarrierDemodulationWorkspace(
        zeros(Complex{T}, pixel_count),
        zeros(Complex{T}, pixel_count),
        zeros(T, pixel_count),
        zeros(T, pixel_count),
        zeros(T, measurement_count),
        zeros(T, measurement_count),
        falses(measurement_count),
    )
end

commit!(workspace::SccSpatialCarrierDemodulationWorkspace) = workspace
discard!(workspace::SccSpatialCarrierDemodulationWorkspace) = workspace

function reset!(workspace::SccSpatialCarrierDemodulationWorkspace)
    fill!(workspace.masked_carrier, zero(eltype(workspace.masked_carrier)))
    fill!(workspace.horizontal_carrier, zero(eltype(workspace.horizontal_carrier)))
    fill!(workspace.mask, zero(eltype(workspace.mask)))
    fill!(workspace.horizontal_support, zero(eltype(workspace.horizontal_support)))
    fill!(workspace.candidate_measurements, zero(eltype(workspace.candidate_measurements)))
    fill!(workspace.candidate_weights, zero(eltype(workspace.candidate_weights)))
    fill!(workspace.candidate_validity, false)
    return workspace
end

"""Expected disposition from mask-aware SCC spatial carrier demodulation."""
@enum SccSpatialCarrierDemodulationFailure::UInt8 begin
    SccSpatialCarrierDemodulationIncompletePair = 1
    SccSpatialCarrierDemodulationInvalidPairStatus = 2
    SccSpatialCarrierDemodulationInsufficientSupport = 3
end

@doc "A spatial-carrier SCC input is not a complete fringed/unfringed pair." SccSpatialCarrierDemodulationIncompletePair
@doc "The SCC pair status is not a defined chopped-image pair status." SccSpatialCarrierDemodulationInvalidPairStatus
@doc "Too few valid selected SCC pixels support carrier demodulation." SccSpatialCarrierDemodulationInsufficientSupport

function _prepare_scc_spatial_carrier_input!(
    workspace::SccSpatialCarrierDemodulationWorkspace{T},
    plan::SccSpatialCarrierDemodulationPlan{T},
    difference::AbstractVector{T},
    difference_validity::AbstractVector{Bool},
) where {T}
    @inbounds for index in eachindex(difference)
        valid = plan.static_validity[index] && difference_validity[index]
        value = zero(Complex{T})
        if valid
            residual = difference[index] - plan.reference_difference[index]
            valid = isfinite(residual)
            valid && (value = residual * plan.carrier_conjugate[index])
        end
        workspace.mask[index] = valid ? one(T) : zero(T)
        workspace.masked_carrier[index] = valid ? value : zero(Complex{T})
    end
    return nothing
end

function _scc_spatial_carrier_horizontal!(
    workspace::SccSpatialCarrierDemodulationWorkspace{T},
    plan::SccSpatialCarrierDemodulationPlan{T},
    output_range=eachindex(workspace.horizontal_carrier),
) where {T}
    rows, columns = plan.image_size
    radius = length(plan.fir) >> 1
    @inbounds for target in output_range
        row = mod1(target, rows)
        column = div(target - 1, rows) + 1
        carrier = zero(Complex{T})
        support = zero(T)
        for tap in -radius:radius
            source_column = clamp(column + tap, 1, columns)
            source = row + (source_column - 1) * rows
            coefficient = plan.fir[tap + radius + 1]
            carrier += coefficient * workspace.masked_carrier[source]
            support += coefficient * workspace.mask[source]
        end
        workspace.horizontal_carrier[target] = carrier
        workspace.horizontal_support[target] = support
    end
    return nothing
end

function _scc_spatial_carrier_selected!(
    workspace::SccSpatialCarrierDemodulationWorkspace{T},
    plan::SccSpatialCarrierDemodulationPlan{T},
    selected_range,
) where {T}
    rows, _ = plan.image_size
    radius = length(plan.fir) >> 1
    selected_count = length(plan.selected_indices)
    @inbounds for selected in selected_range
        pixel = plan.selected_indices[selected]
        row = mod1(pixel, rows)
        column = div(pixel - 1, rows) + 1
        carrier = zero(Complex{T})
        support = zero(T)
        for tap in -radius:radius
            source_row = clamp(row + tap, 1, rows)
            source = source_row + (column - 1) * rows
            coefficient = plan.fir[tap + radius + 1]
            carrier += coefficient * workspace.horizontal_carrier[source]
            support += coefficient * workspace.horizontal_support[source]
        end
        valid = isfinite(support) && support >= plan.minimum_support
        value = zero(Complex{T})
        if valid
            value = carrier / support
            valid = isfinite(real(value)) && isfinite(imag(value))
        end
        real_index = selected
        imaginary_index = selected + selected_count
        weight = valid ? max(support, zero(T)) : zero(T)
        workspace.candidate_measurements[real_index] = valid ? real(value) : zero(T)
        workspace.candidate_measurements[imaginary_index] = valid ? imag(value) : zero(T)
        workspace.candidate_weights[real_index] = weight
        workspace.candidate_weights[imaginary_index] = weight
        workspace.candidate_validity[real_index] = valid
        workspace.candidate_validity[imaginary_index] = valid
    end
    return nothing
end

function _validate_scc_spatial_carrier_storage(
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
    measurement_validity::AbstractVector{Bool},
    valid_fraction::AbstractVector{T},
    workspace::SccSpatialCarrierDemodulationWorkspace{T},
    plan::SccSpatialCarrierDemodulationPlan{T},
    difference::AbstractVector{T},
    difference_validity::AbstractVector{Bool},
    pair_status::AbstractVector{UInt8},
) where {T}
    Base.require_one_based_indexing(
        measurements,
        weights,
        measurement_validity,
        valid_fraction,
        difference,
        difference_validity,
        pair_status,
    )
    pixel_count = prod(plan.image_size)
    measurement_count = 2 * length(plan.selected_indices)
    length(difference) == pixel_count && length(difference_validity) == pixel_count ||
        throw(DimensionMismatch("SCC detector input extent is invalid"))
    length(measurements) == measurement_count && length(weights) == measurement_count &&
        length(measurement_validity) == measurement_count ||
        throw(DimensionMismatch("SCC carrier measurement extent is invalid"))
    length(valid_fraction) == 1 && length(pair_status) == 1 ||
        throw(DimensionMismatch("SCC scalar status extent is invalid"))
    (Base.mightalias(measurements, weights) ||
        Base.mightalias(measurements, measurement_validity) ||
        Base.mightalias(measurements, valid_fraction) ||
        Base.mightalias(weights, measurement_validity) ||
        Base.mightalias(weights, valid_fraction) ||
        Base.mightalias(measurement_validity, valid_fraction)) &&
        throw(ArgumentError("SCC carrier outputs must not overlap"))
    Base.mightalias(measurements, difference) &&
        throw(ArgumentError("SCC carrier output must not alias the difference"))
    Base.mightalias(weights, difference) &&
        throw(ArgumentError("SCC carrier output must not alias the difference"))
    Base.mightalias(measurement_validity, difference) &&
        throw(ArgumentError("SCC carrier output must not alias the difference"))
    Base.mightalias(valid_fraction, difference) &&
        throw(ArgumentError("SCC carrier output must not alias the difference"))
    Base.mightalias(measurements, difference_validity) &&
        throw(ArgumentError("SCC carrier output must not alias input validity"))
    Base.mightalias(weights, difference_validity) &&
        throw(ArgumentError("SCC carrier output must not alias input validity"))
    Base.mightalias(measurement_validity, difference_validity) &&
        throw(ArgumentError("SCC carrier output must not alias input validity"))
    Base.mightalias(valid_fraction, difference_validity) &&
        throw(ArgumentError("SCC carrier output must not alias input validity"))
    return nothing
end

function process!(
    measurements::AbstractVector{T},
    weights::AbstractVector{T},
    measurement_validity::AbstractVector{Bool},
    valid_fraction::AbstractVector{T},
    workspace::SccSpatialCarrierDemodulationWorkspace{T},
    plan::SccSpatialCarrierDemodulationPlan{T},
    difference::AbstractVector{T},
    difference_validity::AbstractVector{Bool},
    pair_status::AbstractVector{UInt8},
) where {T<:Union{Float32,Float64}}
    _validate_scc_spatial_carrier_storage(
        measurements,
        weights,
        measurement_validity,
        valid_fraction,
        workspace,
        plan,
        difference,
        difference_validity,
        pair_status,
    )
    status = pair_status[1]
    status == UInt8(ChoppedImagePriming) &&
        return SccSpatialCarrierDemodulationIncompletePair
    status == UInt8(ChoppedImageComplete) ||
        return SccSpatialCarrierDemodulationInvalidPairStatus
    _prepare_scc_spatial_carrier_input!(workspace, plan, difference, difference_validity)
    _scc_spatial_carrier_horizontal!(workspace, plan)
    _scc_spatial_carrier_selected!(workspace, plan, eachindex(plan.selected_indices))
    valid_count = 0
    @inbounds for selected in eachindex(plan.selected_indices)
        valid_count += workspace.candidate_validity[selected]
    end
    valid_count >= plan.minimum_valid_count ||
        return SccSpatialCarrierDemodulationInsufficientSupport
    copyto!(measurements, workspace.candidate_measurements)
    copyto!(weights, workspace.candidate_weights)
    copyto!(measurement_validity, workspace.candidate_validity)
    valid_fraction[1] = T(valid_count) / T(length(plan.selected_indices))
    return nothing
end

"""
Mask-aware spatial carrier demodulation (the SCC method called 9.2 in the
exploration note).
"""
@algorithm SccSpatialCarrierDemodulationF32 begin
    label = "scc-spatial-carrier-demodulation-f32"
    failure = SccSpatialCarrierDemodulationFailure
    operation = SccSpatialCarrierDemodulationF32Operation()

    """
    Configuration for [`SccSpatialCarrierDemodulationF32`](@ref).
    """
    @configuration SccSpatialCarrierDemodulationF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        roi::Vector{Bool} => graph_rebuild
        fir_tap_count::Int => graph_rebuild
        initial_reference_difference::Union{Nothing,Vector{Float32}} => construction
        initial_static_validity::Union{Nothing,Vector{Bool}} => construction
        initial_carrier_phase::Union{Nothing,Vector{Float32}} => construction
        initial_fir::Union{Nothing,Vector{Float32}} => construction
        minimum_support::Float32 => construction
        minimum_valid_fraction::Float32 => construction
    end

    ports(configuration) = (
        input(
            :difference,
            schema_element_type(SCC_CHOPPED_IMAGE_DIFFERENCE_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        ),
        input(
            :difference_validity,
            schema_element_type(SCC_CHOPPED_DIFFERENCE_VALIDITY_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
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
            :measurements,
            schema_element_type(SCC_SPATIAL_CARRIER_MEASUREMENTS_V1),
            (2 * count(configuration.roi),),
            SCC_SPATIAL_CARRIER_MEASUREMENTS_V1;
            metadata_from=:difference,
        ),
        output(
            :weights,
            schema_element_type(SCC_MEASUREMENT_WEIGHTS_V1),
            (2 * count(configuration.roi),),
            SCC_MEASUREMENT_WEIGHTS_V1;
            metadata_from=:difference,
        ),
        output(
            :measurement_validity,
            schema_element_type(SCC_MEASUREMENT_VALIDITY_V1),
            (2 * count(configuration.roi),),
            SCC_MEASUREMENT_VALIDITY_V1;
            name="measurement-validity",
            metadata_from=:difference,
        ),
        output(
            :valid_fraction,
            schema_element_type(SCC_SPATIAL_CARRIER_VALID_FRACTION_V1),
            (1,),
            SCC_SPATIAL_CARRIER_VALID_FRACTION_V1;
            name="valid-fraction",
            metadata_from=:difference,
        ),
        parameter(
            :reference_difference,
            schema_element_type(SCC_REFERENCE_IMAGE_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
            SCC_REFERENCE_IMAGE_V1;
            name="reference-difference",
            replace=(plan, values) -> _replace_scc_spatial_carrier_plan(
                plan;
                reference_difference=values,
            ),
        ),
        parameter(
            :static_validity,
            schema_element_type(CALIBRATED_PIXEL_VALIDITY_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
            CALIBRATED_PIXEL_VALIDITY_V1;
            name="static-validity",
            replace=(plan, values) -> _replace_scc_spatial_carrier_plan(
                plan;
                static_validity=values,
            ),
        ),
        parameter(
            :carrier_phase,
            schema_element_type(SCC_CARRIER_PHASE_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
            SCC_CARRIER_PHASE_V1;
            name="carrier-phase",
            replace=(plan, values) -> _replace_scc_spatial_carrier_plan(
                plan;
                carrier_phase=values,
            ),
        ),
        parameter(
            :fir,
            schema_element_type(SCC_SPATIAL_FIR_V1),
            (configuration.fir_tap_count,),
            SCC_SPATIAL_FIR_V1;
            replace=(plan, values) ->
                _replace_scc_spatial_carrier_plan(plan; fir=values),
        ),
        parameter(
            :roi,
            schema_element_type(PIXEL_SELECTION_MASK_V1),
            (configuration.image_size[1] * configuration.image_size[2],),
            PIXEL_SELECTION_MASK_V1;
            replace=(plan, values) ->
                _replace_scc_spatial_carrier_plan(plan; roi=values),
        ),
    )

    prepare(configuration) = begin
        plan = SccSpatialCarrierDemodulationPlan(
            Float32,
            configuration.image_size,
            configuration.roi,
            configuration.fir_tap_count;
            reference_difference=configuration.initial_reference_difference,
            static_validity=configuration.initial_static_validity,
            carrier_phase=configuration.initial_carrier_phase,
            fir=configuration.initial_fir,
            minimum_support=configuration.minimum_support,
            minimum_valid_fraction=configuration.minimum_valid_fraction,
        )
        (plan=plan, workspace=SccSpatialCarrierDemodulationWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(
            measurements,
            weights,
            measurement_validity,
            valid_fraction,
            workspace,
            plan,
            difference,
            difference_validity,
            pair_status,
        )
    end
end
