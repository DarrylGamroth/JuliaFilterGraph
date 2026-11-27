"""Normalization policy for a complete calibrated Zernike pupil image."""
@enum ZernikePupilSignalNormalization::UInt8 begin
    ZernikePupilSignalMeanValidFlux = 1
    ZernikePupilSignalIncidenceFlux = 2
end

@doc "Normalize by the current complete frame's mean valid pupil flux."
ZernikePupilSignalMeanValidFlux
@doc "Normalize by an accepted incidence-flux calibration divisor."
ZernikePupilSignalIncidenceFlux

"""Expected numerical or calibration rejection from complete-frame Zernike processing."""
@enum ZernikePupilSignalFailure::UInt8 begin
    ZernikePupilSignalCalibrationMismatch = 1
    ZernikePupilSignalNonFiniteInput = 2
    ZernikePupilSignalInvalidNormalization = 3
    ZernikePupilSignalNonFiniteOutput = 4
end

@doc "The acquired frame calibration signature differs from the active Zernike calibration."
ZernikePupilSignalCalibrationMismatch
@doc "The complete acquired Zernike pupil image contains a non-finite pixel."
ZernikePupilSignalNonFiniteInput
@doc "The computed Zernike normalization divisor is not finite."
ZernikePupilSignalInvalidNormalization
@doc "A calibrated Zernike pupil-signal component is non-finite."
ZernikePupilSignalNonFiniteOutput

"""
Immutable calibration for a complete Zernike pupil detector frame.

The support and reference image share the complete detector-frame extent.  Output
coordinates are the true support entries in Julia column-major order, and each
component is dimensionless: `frame[i, j] / divisor - reference_pupil_image[i, j]`.
"""
struct ZernikePupilSignalPlan{
    T<:AbstractFloat,
    Support<:AbstractMatrix{Bool},
    Offsets<:AbstractVector{Int},
    Reference<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    support::Support
    selected_offsets::Offsets
    reference_pupil_image::Reference
    normalization::ZernikePupilSignalNormalization
    incidence_flux::T
    calibration_signature::UInt64
end

"""Caller-output staging for a stateless complete-frame Zernike operation."""
mutable struct ZernikePupilSignalWorkspace{
    T<:AbstractFloat,
    Signal<:AbstractVector{T},
    Divisor<:AbstractVector{T},
}
    staged_signal::Signal
    staged_divisor::Divisor
end

function _zernike_pupil_signal_normalization(policy::AbstractString)
    policy == "mean-valid-flux" && return ZernikePupilSignalMeanValidFlux
    policy == "incidence-flux" && return ZernikePupilSignalIncidenceFlux
    throw(ArgumentError("unknown Zernike pupil-signal normalization policy: $policy"))
end

function _zernike_pupil_signal_selected_offsets(support::AbstractMatrix{Bool})
    Base.require_one_based_indexing(support)
    offsets = Int[]
    sizehint!(offsets, count(support))
    for (ordinal, index) in enumerate(eachindex(support))
        support[index] && push!(offsets, ordinal - 1)
    end
    isempty(offsets) && throw(ArgumentError("Zernike pupil support must be nonempty"))
    return offsets
end

function ZernikePupilSignalPlan(
    image_size::NTuple{2,Int},
    support::AbstractMatrix{Bool},
    reference_pupil_image::AbstractMatrix{T},
    normalization::ZernikePupilSignalNormalization,
    calibration_signature::UInt64,
    incidence_flux::T=zero(T),
) where {T<:AbstractFloat}
    image_rows, image_columns = image_size
    image_rows > 0 && image_columns > 0 || throw(
        ArgumentError("Zernike pupil image dimensions must be positive"),
    )
    size(support) == image_size || throw(
        DimensionMismatch("Zernike pupil support extent must match image size"),
    )
    size(reference_pupil_image) == image_size || throw(
        DimensionMismatch("Zernike reference pupil image extent must match image size"),
    )
    all(isfinite, reference_pupil_image) || throw(
        ArgumentError("Zernike reference pupil image must be finite"),
    )
    isfinite(incidence_flux) && incidence_flux >= zero(T) || throw(
        ArgumentError("Zernike incidence flux must be finite and nonnegative"),
    )
    selected_offsets = _zernike_pupil_signal_selected_offsets(support)
    return ZernikePupilSignalPlan(
        image_size,
        Matrix(support),
        selected_offsets,
        Matrix(reference_pupil_image),
        normalization,
        incidence_flux,
        calibration_signature,
    )
end

function ZernikePupilSignalWorkspace(plan::ZernikePupilSignalPlan{T}) where {T}
    return ZernikePupilSignalWorkspace(zeros(T, length(plan.selected_offsets)), zeros(T, 1))
end

commit!(workspace::ZernikePupilSignalWorkspace) = workspace
discard!(workspace::ZernikePupilSignalWorkspace) = workspace
reset!(workspace::ZernikePupilSignalWorkspace) = workspace

function _replace_zernike_pupil_signal_support(
    plan::ZernikePupilSignalPlan,
    values::AbstractMatrix{Bool},
)
    size(values) == plan.image_size || throw(
        DimensionMismatch("Zernike pupil support extent is invalid"),
    )
    count(values) == length(plan.selected_offsets) || throw(
        DimensionMismatch("replacement Zernike pupil support must preserve selected extent"),
    )
    return ZernikePupilSignalPlan(
        plan.image_size, values, plan.reference_pupil_image, plan.normalization,
        plan.calibration_signature, plan.incidence_flux,
    )
end

function _replace_zernike_pupil_signal_reference(
    plan::ZernikePupilSignalPlan{T},
    values::AbstractMatrix{T},
) where {T}
    return ZernikePupilSignalPlan(
        plan.image_size, plan.support, values, plan.normalization,
        plan.calibration_signature, plan.incidence_flux,
    )
end

function _replace_zernike_pupil_signal_incidence_flux(
    plan::ZernikePupilSignalPlan{T},
    values::AbstractVector{T},
) where {T}
    length(values) == 1 || throw(DimensionMismatch("Zernike incidence flux extent is invalid"))
    return ZernikePupilSignalPlan(
        plan.image_size, plan.support, plan.reference_pupil_image, plan.normalization,
        plan.calibration_signature, values[1],
    )
end

function _replace_zernike_pupil_signal_calibration_signature(
    plan::ZernikePupilSignalPlan,
    values::AbstractVector{UInt64},
)
    length(values) == 1 || throw(
        DimensionMismatch("Zernike calibration signature extent is invalid"),
    )
    return ZernikePupilSignalPlan(
        plan.image_size, plan.support, plan.reference_pupil_image, plan.normalization,
        values[1], plan.incidence_flux,
    )
end

function _check_zernike_pupil_signal_execution(
    signal::AbstractVector{T},
    divisor::AbstractVector{T},
    workspace::ZernikePupilSignalWorkspace{T},
    plan::ZernikePupilSignalPlan{T},
    image::AbstractMatrix{F},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat,F<:Real}
    Base.require_one_based_indexing(
        signal, divisor, image, frame_calibration_signature,
        workspace.staged_signal, workspace.staged_divisor,
    )
    size(image) == plan.image_size || throw(
        DimensionMismatch("Zernike pupil detector image extent is invalid"),
    )
    length(signal) == length(plan.selected_offsets) || throw(
        DimensionMismatch("Zernike pupil signal output extent is invalid"),
    )
    length(divisor) == 1 || throw(
        DimensionMismatch("Zernike normalization divisor output must contain one value"),
    )
    length(frame_calibration_signature) == 1 || throw(
        DimensionMismatch("Zernike frame calibration signature input must contain one value"),
    )
    length(workspace.staged_signal) == length(signal) &&
        length(workspace.staged_divisor) == 1 || throw(
        DimensionMismatch("Zernike pupil-signal Workspace extent is invalid"),
    )
    (Base.mightalias(signal, divisor) || Base.mightalias(signal, image) ||
     Base.mightalias(divisor, image)) && throw(
        ArgumentError("Zernike pupil-signal caller arrays must not overlap"),
    )
    (Base.mightalias(signal, plan.support) || Base.mightalias(signal, plan.selected_offsets) ||
     Base.mightalias(signal, plan.reference_pupil_image) ||
     Base.mightalias(divisor, plan.support) || Base.mightalias(divisor, plan.selected_offsets) ||
     Base.mightalias(divisor, plan.reference_pupil_image) ||
     Base.mightalias(image, plan.support) || Base.mightalias(image, plan.selected_offsets) ||
     Base.mightalias(image, plan.reference_pupil_image)) && throw(
        ArgumentError("Zernike pupil-signal caller arrays must not alias Plan storage"),
    )
    (Base.mightalias(signal, workspace.staged_signal) ||
     Base.mightalias(signal, workspace.staged_divisor) ||
     Base.mightalias(divisor, workspace.staged_signal) ||
     Base.mightalias(divisor, workspace.staged_divisor) ||
     Base.mightalias(image, workspace.staged_signal) ||
     Base.mightalias(image, workspace.staged_divisor)) && throw(
        ArgumentError("Zernike pupil-signal caller arrays must not alias Workspace storage"),
    )
    return nothing
end

function _stage_zernike_pupil_signal!(
    workspace::ZernikePupilSignalWorkspace{T},
    plan::ZernikePupilSignalPlan{T},
    image::AbstractMatrix{F},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat,F<:Real}
    frame_calibration_signature[1] == plan.calibration_signature ||
        return ZernikePupilSignalCalibrationMismatch
    @inbounds for value in image
        isfinite(value) || return ZernikePupilSignalNonFiniteInput
    end
    divisor = if plan.normalization === ZernikePupilSignalMeanValidFlux
        total = zero(T)
        @inbounds for offset in plan.selected_offsets
            total += T(image[offset + 1])
        end
        max(total / T(length(plan.selected_offsets)), eps(T))
    else
        plan.incidence_flux
    end
    isfinite(divisor) || return ZernikePupilSignalInvalidNormalization
    if divisor <= eps(T)
        fill!(workspace.staged_signal, zero(T))
        workspace.staged_divisor[1] = divisor
        return nothing
    end
    @inbounds for (index, offset) in enumerate(plan.selected_offsets)
        value = T(image[offset + 1]) / divisor - plan.reference_pupil_image[offset + 1]
        isfinite(value) || return ZernikePupilSignalNonFiniteOutput
        workspace.staged_signal[index] = value
    end
    workspace.staged_divisor[1] = divisor
    return nothing
end

function _publish_zernike_pupil_signal!(
    signal::AbstractVector{T}, divisor::AbstractVector{T},
    workspace::ZernikePupilSignalWorkspace{T},
) where {T<:AbstractFloat}
    copyto!(signal, workspace.staged_signal)
    divisor[1] = workspace.staged_divisor[1]
    return nothing
end

"""Process one complete calibrated Zernike pupil frame with caller-owned outputs."""
function process!(
    signal::AbstractVector{T}, normalization_divisor::AbstractVector{T},
    workspace::ZernikePupilSignalWorkspace{T}, plan::ZernikePupilSignalPlan{T},
    image::AbstractMatrix{F}, frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat,F<:Real}
    _check_zernike_pupil_signal_execution(
        signal, normalization_divisor, workspace, plan, image, frame_calibration_signature,
    )
    result = _stage_zernike_pupil_signal!(workspace, plan, image, frame_calibration_signature)
    result isa ZernikePupilSignalFailure && return result
    return _publish_zernike_pupil_signal!(signal, normalization_divisor, workspace)
end

function _configured_zernike_pupil_signal_support(
    values::Union{Nothing,Vector{Bool}}, rows::Int, columns::Int,
)
    pixels = Base.checked_mul(rows, columns)
    isnothing(values) && return trues(rows, columns)
    length(values) == pixels || throw(
        DimensionMismatch("zernike_pupil_support must contain $pixels column-major values"),
    )
    return reshape(copy(values), rows, columns)
end

function _configured_zernike_pupil_signal_selected_count(configuration)
    return count(_configured_zernike_pupil_signal_support(
        configuration.zernike_pupil_support,
        configuration.image_size[1],
        configuration.image_size[2],
    ))
end

"""Float32 complete-frame calibrated Zernike normalized pupil signal."""
@algorithm ZernikePupilSignalF32 begin
    label = "zernike-pupil-signal-f32"
    failure = ZernikePupilSignalFailure

    """Configuration for ZernikePupilSignalF32."""
    @configuration ZernikePupilSignalF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        zernike_pupil_support::Union{Nothing,Vector{Bool}} => construction
        initial_reference_pupil_image::Union{Nothing,Vector{Float32}} => construction
        normalization_policy::String => graph_rebuild
        incidence_flux::Float32 => construction
        initial_calibration_signature::UInt64 => construction
    end

    ports(configuration) = (
        input(
            :image, schema_element_type(CALIBRATED_PIXELS_V1), configuration.image_size,
            CALIBRATED_PIXELS_V1,
        ),
        input(
            :frame_calibration_signature, schema_element_type(ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1), (1,),
            ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1;
            name="frame-calibration-signature",
        ),
        output(
            :signal, schema_element_type(ZERNIKE_PUPIL_SIGNAL_V1), (_configured_zernike_pupil_signal_selected_count(configuration),),
            ZERNIKE_PUPIL_SIGNAL_V1; metadata_from=:image,
        ),
        output(
            :normalization_divisor, schema_element_type(ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1), (1,),
            ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1;
            name="normalization-divisor", metadata_from=:image,
        ),
        parameter(
            :zernike_pupil_support, schema_element_type(ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1),
            configuration.image_size,
            ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1;
            name="zernike-pupil-support",
            replace=(plan, values) -> _replace_zernike_pupil_signal_support(plan, values),
        ),
        parameter(
            :reference_pupil_image, schema_element_type(ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1),
            configuration.image_size,
            ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1;
            name="reference-pupil-image",
            replace=(plan, values) -> _replace_zernike_pupil_signal_reference(plan, values),
        ),
        parameter(
            :incidence_flux, schema_element_type(ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1), (1,), ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1;
            name="incidence-flux",
            replace=(plan, values) -> _replace_zernike_pupil_signal_incidence_flux(plan, values),
        ),
        parameter(
            :calibration_signature, schema_element_type(ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1), (1,),
            ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1;
            name="calibration-signature",
            replace=(plan, values) ->
                _replace_zernike_pupil_signal_calibration_signature(plan, values),
        ),
    )

    prepare(configuration) = begin
        support = _configured_zernike_pupil_signal_support(
            configuration.zernike_pupil_support,
            configuration.image_size[1],
            configuration.image_size[2],
        )
        reference = if isnothing(configuration.initial_reference_pupil_image)
            zeros(Float32, configuration.image_size[1], configuration.image_size[2])
        else
            reshape(
                copy(configuration.initial_reference_pupil_image),
                configuration.image_size[1], configuration.image_size[2],
            )
        end
        plan = ZernikePupilSignalPlan(
            configuration.image_size, support, reference,
            _zernike_pupil_signal_normalization(configuration.normalization_policy),
            configuration.initial_calibration_signature, configuration.incidence_flux,
        )
        (plan=plan, workspace=ZernikePupilSignalWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(signal, normalization_divisor, workspace, plan, image, frame_calibration_signature)
    end
end


"""
    replace_zernike_pupil_signal_calibration!(prepared, support, reference, incidence_flux, signature)

Validate a complete Zernike calibration replacement and publish its Plan once.
The support must retain the selected-coordinate extent declared for the graph.
"""
function replace_zernike_pupil_signal_calibration!(
    prepared::PreparedAlgorithm{ZernikePupilSignalF32,P},
    support::AbstractMatrix{Bool},
    reference_pupil_image::AbstractMatrix{T},
    incidence_flux::AbstractVector{T},
    calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat,P<:ZernikePupilSignalPlan{T}}
    plan = prepared.plan
    length(incidence_flux) == 1 || throw(
        DimensionMismatch("Zernike incidence flux extent is invalid"),
    )
    length(calibration_signature) == 1 || throw(
        DimensionMismatch("Zernike calibration signature extent is invalid"),
    )
    replacement = _replace_zernike_pupil_signal_support(plan, support)
    replacement = _replace_zernike_pupil_signal_reference(replacement, reference_pupil_image)
    replacement = _replace_zernike_pupil_signal_incidence_flux(replacement, incidence_flux)
    replacement = _replace_zernike_pupil_signal_calibration_signature(
        replacement, calibration_signature,
    )
    prepared.plan = replacement
    return prepared
end
