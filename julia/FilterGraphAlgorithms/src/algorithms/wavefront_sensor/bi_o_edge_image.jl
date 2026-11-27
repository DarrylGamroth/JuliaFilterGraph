"""Normalization policy for a complete-frame calibrated Bi-O-edge signal."""
@enum BiOEdgeImageNormalization::UInt8 begin
    BiOEdgeImageMeanValidFlux = 1
    BiOEdgeImageIncidenceFlux = 2
end

@doc "Normalize by the current frame's mean valid four-pupil flux."
BiOEdgeImageMeanValidFlux
@doc "Normalize by the finite incidence-flux calibration divisor greater than `eps(T)`."
BiOEdgeImageIncidenceFlux

"""Expected numerical or calibration rejection from complete-frame Bi-O-edge processing."""
@enum BiOEdgeImageFailure::UInt8 begin
    BiOEdgeImageCalibrationMismatch = 1
    BiOEdgeImageNonFiniteInput = 2
    BiOEdgeImageInvalidNormalization = 3
    BiOEdgeImageNonFiniteOutput = 4
end

@doc "The frame calibration signature differs from the active Bi-O-edge calibration signature."
BiOEdgeImageCalibrationMismatch
@doc "A configured Bi-O-edge pupil contains a non-finite input pixel."
BiOEdgeImageNonFiniteInput
@doc "The current-frame Bi-O-edge normalization divisor is invalid."
BiOEdgeImageInvalidNormalization
@doc "A calibrated Bi-O-edge differential signal component is non-finite."
BiOEdgeImageNonFiniteOutput

const BI_O_EDGE_IMAGE_NORMALIZATION_LANES = 256
const BI_O_EDGE_PUPIL_COUNT = 4

"""
Immutable numerical calibration for one complete-frame Bi-O-edge image.

Pupil origins are zero-based `(row, column)` coordinates ordered q1 top-left,
q2 bottom-left, q3 bottom-right, and q4 top-right. Selected support coordinates
use Julia column-major order. The output at each selected coordinate is
`gain * (differential / normalization - reference)` in X/Y columns.
"""
struct BiOEdgeImagePlan{
    T<:AbstractFloat,
    Support<:AbstractMatrix{Bool},
    Offsets<:AbstractVector{Int},
    Reference<:AbstractMatrix{T},
    Gain<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    pupil_size::NTuple{2,Int}
    origins::NTuple{BI_O_EDGE_PUPIL_COUNT,NTuple{2,Int}}
    support::Support
    selected_offsets::Offsets
    reference_signal::Reference
    optical_gain::Gain
    normalization::BiOEdgeImageNormalization
    incidence_flux::T
    calibration_signature::UInt64
end

"""Caller-output staging for a stateless complete-frame Bi-O-edge operation."""
mutable struct BiOEdgeImageWorkspace{
    T<:AbstractFloat,
    Signal<:AbstractMatrix{T},
    Divisor<:AbstractVector{T},
    Partials<:AbstractVector{T},
}
    staged_signal::Signal
    staged_divisor::Divisor
    normalization_partials::Partials
end

function _bi_o_edge_image_normalization(policy::AbstractString)
    policy == "mean-valid-flux" && return BiOEdgeImageMeanValidFlux
    policy == "incidence-flux" && return BiOEdgeImageIncidenceFlux
    throw(ArgumentError("unknown Bi-O-edge normalization policy: $policy"))
end

function _bi_o_edge_image_origins(values::AbstractVector{<:AbstractVector{<:Integer}})
    Base.require_one_based_indexing(values)
    length(values) == BI_O_EDGE_PUPIL_COUNT || throw(
        DimensionMismatch("pupil_origins must contain q1, q2, q3, and q4"),
    )
    return ntuple(Val(BI_O_EDGE_PUPIL_COUNT)) do index
        origin = values[index]
        Base.require_one_based_indexing(origin)
        length(origin) == 2 || throw(
            DimensionMismatch("each pupil origin must contain row and column"),
        )
        row, column = origin
        row >= 0 && column >= 0 || throw(ArgumentError("pupil origins must be nonnegative"))
        (Int(row), Int(column))
    end
end

function _bi_o_edge_image_selected_offsets(support::AbstractMatrix{Bool})
    Base.require_one_based_indexing(support)
    offsets = Int[]
    sizehint!(offsets, count(support))
    for (ordinal, index) in enumerate(eachindex(support))
        support[index] && push!(offsets, ordinal - 1)
    end
    isempty(offsets) && throw(ArgumentError("Bi-O-edge pupil support must be nonempty"))
    return offsets
end

function BiOEdgeImagePlan(
    image_size::NTuple{2,Int},
    pupil_size::NTuple{2,Int},
    origins::NTuple{BI_O_EDGE_PUPIL_COUNT,NTuple{2,Int}},
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    optical_gain::AbstractMatrix{T},
    normalization::BiOEdgeImageNormalization,
    calibration_signature::UInt64,
    incidence_flux::T=zero(T),
) where {T<:AbstractFloat}
    image_rows, image_columns = image_size
    pupil_rows, pupil_columns = pupil_size
    image_rows > 0 && image_columns > 0 || throw(ArgumentError("image dimensions must be positive"))
    pupil_rows > 0 && pupil_columns > 0 || throw(ArgumentError("pupil dimensions must be positive"))
    size(support) == pupil_size || throw(
        DimensionMismatch("Bi-O-edge support extent must match pupil size"),
    )
    selected_offsets = _bi_o_edge_image_selected_offsets(support)
    selected = length(selected_offsets)
    size(reference_signal) == (selected, 2) || throw(
        DimensionMismatch("Bi-O-edge reference signal extent is invalid"),
    )
    size(optical_gain) == (selected, 2) || throw(
        DimensionMismatch("Bi-O-edge optical gain extent is invalid"),
    )
    all(isfinite, reference_signal) || throw(
        ArgumentError("Bi-O-edge reference signal must be finite"),
    )
    all(isfinite, optical_gain) || throw(
        ArgumentError("Bi-O-edge optical gain must be finite"),
    )
    normalization === BiOEdgeImageIncidenceFlux &&
        !(isfinite(incidence_flux) && incidence_flux > eps(T)) &&
        throw(ArgumentError("incidence flux must be finite and greater than eps(T)"))
    maximum_origin_row = image_rows - pupil_rows
    maximum_origin_column = image_columns - pupil_columns
    for (row, column) in origins
        0 <= row <= maximum_origin_row && 0 <= column <= maximum_origin_column ||
            throw(DimensionMismatch("pupil region exceeds detector image"))
    end
    return BiOEdgeImagePlan(
        image_size,
        pupil_size,
        origins,
        Matrix(support),
        selected_offsets,
        Matrix(reference_signal),
        Matrix(optical_gain),
        normalization,
        incidence_flux,
        calibration_signature,
    )
end

function BiOEdgeImageWorkspace(plan::BiOEdgeImagePlan{T}) where {T}
    return BiOEdgeImageWorkspace(
        zeros(T, size(plan.reference_signal)),
        zeros(T, 1),
        zeros(T, BI_O_EDGE_IMAGE_NORMALIZATION_LANES),
    )
end

commit!(workspace::BiOEdgeImageWorkspace) = workspace
discard!(workspace::BiOEdgeImageWorkspace) = workspace
reset!(workspace::BiOEdgeImageWorkspace) = workspace

function _replace_bi_o_edge_image_origins(
    plan::BiOEdgeImagePlan,
    values::AbstractMatrix{UInt32},
)
    size(values) == (BI_O_EDGE_PUPIL_COUNT, 2) || throw(
        DimensionMismatch("pupil origin extent is invalid"),
    )
    origins = ntuple(Val(BI_O_EDGE_PUPIL_COUNT)) do index
        row, column = values[index, 1], values[index, 2]
        row <= typemax(Int) && column <= typemax(Int) || throw(
            ArgumentError("pupil origin cannot be represented by Int"),
        )
        (Int(row), Int(column))
    end
    return BiOEdgeImagePlan(
        plan.image_size, plan.pupil_size, origins, plan.support,
        plan.reference_signal, plan.optical_gain, plan.normalization,
        plan.calibration_signature, plan.incidence_flux,
    )
end

function _replace_bi_o_edge_image_support(
    plan::BiOEdgeImagePlan,
    values::AbstractMatrix{Bool},
)
    size(values) == plan.pupil_size || throw(
        DimensionMismatch("Bi-O-edge support extent is invalid"),
    )
    count(values) == length(plan.selected_offsets) || throw(
        DimensionMismatch("replacement Bi-O-edge support must preserve selected extent"),
    )
    return BiOEdgeImagePlan(
        plan.image_size, plan.pupil_size, plan.origins, values,
        plan.reference_signal, plan.optical_gain, plan.normalization,
        plan.calibration_signature, plan.incidence_flux,
    )
end

function _replace_bi_o_edge_image_reference(
    plan::BiOEdgeImagePlan{T},
    values::AbstractMatrix{T},
) where {T}
    return BiOEdgeImagePlan(
        plan.image_size, plan.pupil_size, plan.origins, plan.support,
        values, plan.optical_gain, plan.normalization, plan.calibration_signature,
        plan.incidence_flux,
    )
end

function _replace_bi_o_edge_image_gain(
    plan::BiOEdgeImagePlan{T},
    values::AbstractMatrix{T},
) where {T}
    return BiOEdgeImagePlan(
        plan.image_size, plan.pupil_size, plan.origins, plan.support,
        plan.reference_signal, values, plan.normalization, plan.calibration_signature,
        plan.incidence_flux,
    )
end

function _replace_bi_o_edge_image_calibration_signature(
    plan::BiOEdgeImagePlan,
    values::AbstractVector{UInt64},
)
    length(values) == 1 || throw(DimensionMismatch("calibration signature extent is invalid"))
    return BiOEdgeImagePlan(
        plan.image_size, plan.pupil_size, plan.origins, plan.support,
        plan.reference_signal, plan.optical_gain, plan.normalization,
        values[1], plan.incidence_flux,
    )
end

function _check_bi_o_edge_image_execution(
    signal::AbstractMatrix{T},
    divisor::AbstractVector{T},
    workspace::BiOEdgeImageWorkspace{T},
    plan::BiOEdgeImagePlan{T},
    image::AbstractMatrix{T},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T}
    Base.require_one_based_indexing(
        signal, divisor, image, frame_calibration_signature,
        workspace.staged_signal, workspace.staged_divisor, workspace.normalization_partials,
    )
    size(image) == plan.image_size || throw(
        DimensionMismatch("Bi-O-edge detector image extent is invalid"),
    )
    size(signal) == size(plan.reference_signal) || throw(
        DimensionMismatch("Bi-O-edge signal output extent is invalid"),
    )
    length(divisor) == 1 || throw(
        DimensionMismatch("Bi-O-edge normalization divisor output must contain one value"),
    )
    length(frame_calibration_signature) == 1 || throw(
        DimensionMismatch(
            "Bi-O-edge frame calibration signature input must contain one value",
        ),
    )
    size(workspace.staged_signal) == size(signal) &&
        length(workspace.staged_divisor) == 1 &&
        length(workspace.normalization_partials) == BI_O_EDGE_IMAGE_NORMALIZATION_LANES ||
        throw(DimensionMismatch("Bi-O-edge image Workspace extent is invalid"))
    (Base.mightalias(signal, divisor) || Base.mightalias(signal, image) ||
     Base.mightalias(divisor, image)) &&
        throw(ArgumentError("Bi-O-edge image caller arrays must not overlap"))
    (Base.mightalias(signal, plan.support) || Base.mightalias(signal, plan.selected_offsets) ||
     Base.mightalias(signal, plan.reference_signal) || Base.mightalias(signal, plan.optical_gain) ||
     Base.mightalias(divisor, plan.support) || Base.mightalias(divisor, plan.selected_offsets) ||
     Base.mightalias(divisor, plan.reference_signal) || Base.mightalias(divisor, plan.optical_gain) ||
     Base.mightalias(image, plan.support) || Base.mightalias(image, plan.selected_offsets) ||
     Base.mightalias(image, plan.reference_signal) || Base.mightalias(image, plan.optical_gain)) &&
        throw(ArgumentError("Bi-O-edge image caller arrays must not alias Plan storage"))
    (Base.mightalias(signal, workspace.staged_signal) ||
     Base.mightalias(signal, workspace.staged_divisor) ||
     Base.mightalias(divisor, workspace.staged_signal) ||
     Base.mightalias(divisor, workspace.staged_divisor) ||
     Base.mightalias(divisor, workspace.normalization_partials) ||
     Base.mightalias(image, workspace.staged_signal) ||
     Base.mightalias(image, workspace.staged_divisor) ||
     Base.mightalias(image, workspace.normalization_partials) ||
     Base.mightalias(signal, workspace.normalization_partials)) &&
        throw(ArgumentError("Bi-O-edge image caller arrays must not alias Workspace storage"))
    return nothing
end

function _bi_o_edge_image_mean_valid_flux!(
    partials::AbstractVector{T},
    plan::BiOEdgeImagePlan{T},
    image::AbstractMatrix{T},
) where {T}
    fill!(partials, zero(T))
    pupil_rows = plan.pupil_size[1]
    selected_count = length(plan.selected_offsets)
    q1, q2, q3, q4 = plan.origins
    @inbounds for lane in 1:BI_O_EDGE_IMAGE_NORMALIZATION_LANES
        selected = lane
        lane_sum = zero(T)
        while selected <= selected_count
            offset = plan.selected_offsets[selected]
            row = rem(offset, pupil_rows) + 1
            column = div(offset, pupil_rows) + 1
            coordinate_flux = image[q1[1] + row, q1[2] + column] +
                image[q2[1] + row, q2[2] + column] +
                image[q3[1] + row, q3[2] + column] +
                image[q4[1] + row, q4[2] + column]
            lane_sum += coordinate_flux
            selected += BI_O_EDGE_IMAGE_NORMALIZATION_LANES
        end
        partials[lane] = lane_sum
    end
    stride = div(BI_O_EDGE_IMAGE_NORMALIZATION_LANES, 2)
    @inbounds while stride > 0
        for lane in 1:stride
            partials[lane] += partials[lane + stride]
        end
        stride = div(stride, 2)
    end
    return partials[1] / T(selected_count)
end

function _stage_bi_o_edge_image!(
    workspace::BiOEdgeImageWorkspace{T},
    plan::BiOEdgeImagePlan{T},
    image::AbstractMatrix{T},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T}
    frame_calibration_signature[1] == plan.calibration_signature ||
        return BiOEdgeImageCalibrationMismatch
    pupil_rows, pupil_columns = plan.pupil_size
    q1, q2, q3, q4 = plan.origins
    @inbounds for (origin_row, origin_column) in plan.origins
        for column in 1:pupil_columns, row in 1:pupil_rows
            isfinite(image[origin_row + row, origin_column + column]) ||
                return BiOEdgeImageNonFiniteInput
        end
    end
    divisor = plan.normalization === BiOEdgeImageMeanValidFlux ?
        _bi_o_edge_image_mean_valid_flux!(workspace.normalization_partials, plan, image) :
        plan.incidence_flux
    isfinite(divisor) && divisor > eps(T) || return BiOEdgeImageInvalidNormalization
    @inbounds for (index, offset) in enumerate(plan.selected_offsets)
        row = rem(offset, pupil_rows) + 1
        column = div(offset, pupil_rows) + 1
        value1 = image[q1[1] + row, q1[2] + column]
        value2 = image[q2[1] + row, q2[2] + column]
        value3 = image[q3[1] + row, q3[2] + column]
        value4 = image[q4[1] + row, q4[2] + column]
        normalized_x = _normalized_pupil_difference(value1, value2, value4, value3, divisor)
        normalized_y = _normalized_pupil_difference(value1, value4, value2, value3, divisor)
        output_x = plan.optical_gain[index, 1] * (normalized_x - plan.reference_signal[index, 1])
        output_y = plan.optical_gain[index, 2] * (normalized_y - plan.reference_signal[index, 2])
        isfinite(output_x) && isfinite(output_y) || return BiOEdgeImageNonFiniteOutput
        workspace.staged_signal[index, 1] = output_x
        workspace.staged_signal[index, 2] = output_y
    end
    workspace.staged_divisor[1] = divisor
    return nothing
end

function _publish_bi_o_edge_image!(
    signal::AbstractMatrix{T}, divisor::AbstractVector{T}, workspace::BiOEdgeImageWorkspace{T},
) where {T}
    copyto!(signal, workspace.staged_signal)
    divisor[1] = workspace.staged_divisor[1]
    return nothing
end

"""Process one complete calibrated Bi-O-edge frame with caller-owned outputs."""
function process!(
    signal::AbstractMatrix{T}, normalization_divisor::AbstractVector{T},
    workspace::BiOEdgeImageWorkspace{T}, plan::BiOEdgeImagePlan{T},
    image::AbstractMatrix{T}, frame_calibration_signature::AbstractVector{UInt64},
) where {T}
    _check_bi_o_edge_image_execution(
        signal, normalization_divisor, workspace, plan, image, frame_calibration_signature,
    )
    result = _stage_bi_o_edge_image!(workspace, plan, image, frame_calibration_signature)
    result isa BiOEdgeImageFailure && return result
    return _publish_bi_o_edge_image!(signal, normalization_divisor, workspace)
end

function _configured_bi_o_edge_image_support(
    values::Union{Nothing,Vector{Bool}}, rows::Int, columns::Int,
)
    pixels = Base.checked_mul(rows, columns)
    isnothing(values) && return trues(rows, columns)
    length(values) == pixels || throw(
        DimensionMismatch("bi_o_edge_support must contain $pixels column-major values"),
    )
    return reshape(copy(values), rows, columns)
end

function _configured_bi_o_edge_image_selected_count(configuration)
    return count(_configured_bi_o_edge_image_support(
        configuration.bi_o_edge_support,
        configuration.pupil_size[1],
        configuration.pupil_size[2],
    ))
end

"""Float32 complete-frame calibrated Bi-O-edge differential signal."""
@algorithm BiOEdgeImageF32 begin
    label = "bi-o-edge-image-f32"
    failure = BiOEdgeImageFailure

    """Configuration for BiOEdgeImageF32."""
    @configuration BiOEdgeImageF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        bi_o_edge_support::Union{Nothing,Vector{Bool}} => construction
        initial_reference_signal::Union{Nothing,Vector{Float32}} => construction
        initial_optical_gain::Union{Nothing,Vector{Float32}} => construction
        normalization_policy::String => graph_rebuild
        incidence_flux::Float32 => construction
        initial_calibration_signature::UInt64 => construction
    end

    ports(configuration) = (
        input(
            :image,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.image_size,
            CALIBRATED_PIXELS_V1,
        ),
        input(
            :frame_calibration_signature,
            schema_element_type(BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1),
            (1,),
            BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1;
            name="frame-calibration-signature",
        ),
        output(
            :signal,
            schema_element_type(BI_O_EDGE_SIGNAL_V1),
            (_configured_bi_o_edge_image_selected_count(configuration), 2),
            BI_O_EDGE_SIGNAL_V1;
            metadata_from=:image,
        ),
        output(
            :normalization_divisor,
            schema_element_type(BI_O_EDGE_NORMALIZATION_DIVISOR_V1),
            (1,),
            BI_O_EDGE_NORMALIZATION_DIVISOR_V1;
            name="normalization-divisor",
            metadata_from=:image,
        ),
        parameter(
            :pupil_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (BI_O_EDGE_PUPIL_COUNT, 2),
            REGION_ORIGINS_V1;
            name="pupil-origins",
            replace=(plan, values) -> _replace_bi_o_edge_image_origins(plan, values),
        ),
        parameter(
            :bi_o_edge_support,
            schema_element_type(BI_O_EDGE_SUPPORT_V1),
            configuration.pupil_size,
            BI_O_EDGE_SUPPORT_V1;
            name="bi-o-edge-support",
            replace=(plan, values) -> _replace_bi_o_edge_image_support(plan, values),
        ),
        parameter(
            :reference_signal,
            schema_element_type(BI_O_EDGE_REFERENCE_SIGNAL_V1),
            (_configured_bi_o_edge_image_selected_count(configuration), 2),
            BI_O_EDGE_REFERENCE_SIGNAL_V1;
            name="reference-signal",
            replace=(plan, values) -> _replace_bi_o_edge_image_reference(plan, values),
        ),
        parameter(
            :optical_gain,
            schema_element_type(BI_O_EDGE_OPTICAL_GAIN_V1),
            (_configured_bi_o_edge_image_selected_count(configuration), 2),
            BI_O_EDGE_OPTICAL_GAIN_V1;
            name="optical-gain",
            replace=(plan, values) -> _replace_bi_o_edge_image_gain(plan, values),
        ),
        parameter(
            :calibration_signature,
            schema_element_type(BI_O_EDGE_CALIBRATION_SIGNATURE_V1),
            (1,),
            BI_O_EDGE_CALIBRATION_SIGNATURE_V1;
            name="calibration-signature",
            replace=(plan, values) ->
                _replace_bi_o_edge_image_calibration_signature(plan, values),
        ),
    )

    prepare(configuration) = begin
        support = _configured_bi_o_edge_image_support(
            configuration.bi_o_edge_support,
            configuration.pupil_size[1],
            configuration.pupil_size[2],
        )
        selected = count(support)
        reference_values = if isnothing(configuration.initial_reference_signal)
            zeros(Float32, selected, 2)
        else
            reshape(copy(configuration.initial_reference_signal), selected, 2)
        end
        gain_values = if isnothing(configuration.initial_optical_gain)
            ones(Float32, selected, 2)
        else
            reshape(copy(configuration.initial_optical_gain), selected, 2)
        end
        plan = BiOEdgeImagePlan(
            configuration.image_size,
            configuration.pupil_size,
            _bi_o_edge_image_origins(configuration.pupil_origins), support,
            reference_values, gain_values,
            _bi_o_edge_image_normalization(configuration.normalization_policy),
            configuration.initial_calibration_signature, configuration.incidence_flux,
        )
        (plan=plan, workspace=BiOEdgeImageWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(signal, normalization_divisor, workspace, plan, image, frame_calibration_signature)
    end
end
