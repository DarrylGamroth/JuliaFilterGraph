# Keep ordinary pupil arithmetic in its native precision and grouping.
@inline function _normalized_pupil_difference(a::T, b::T, c::T, d::T, divisor::T) where {T}
    raw = a - b + c - d
    isfinite(raw) && return raw / divisor
    return _wide_pupil_difference(a, b, c, d, divisor)
end

@inline _wide_pupil_difference(a::Float32, b::Float32, c::Float32, d::Float32, divisor::Float32) =
    Float32((Float64(a) - Float64(b) + Float64(c) - Float64(d)) / Float64(divisor))

@inline function _wide_pupil_difference(a::Float64, b::Float64, c::Float64, d::Float64, divisor::Float64)
    scaled = 0.25 * a - 0.25 * b + 0.25 * c - 0.25 * d
    return (scaled / divisor) * 4.0
end

"""Normalization policy for a complete-frame calibrated pyramid signal."""
@enum PyramidImageNormalization::UInt8 begin
    PyramidImageMeanValidFlux = 1
    PyramidImageIncidenceFlux = 2
end

@doc "Normalize by the current frame's mean valid four-pupil flux."
PyramidImageMeanValidFlux
@doc "Normalize by the finite positive incidence-flux calibration divisor."
PyramidImageIncidenceFlux

"""Expected numerical rejection from complete-frame pyramid processing."""
@enum PyramidImageFailure::UInt8 begin
    PyramidImageNonFiniteInput = 1
    PyramidImageInvalidNormalization = 2
    PyramidImageNonFiniteOutput = 3
end

@doc "A configured pyramid pupil contains a non-finite input pixel."
PyramidImageNonFiniteInput
@doc "The current-frame pyramid normalization divisor is invalid."
PyramidImageInvalidNormalization
@doc "A calibrated pyramid differential signal component is non-finite."
PyramidImageNonFiniteOutput

const PYRAMID_IMAGE_NORMALIZATION_LANES = 256

"""
Immutable numerical calibration for one complete-frame pyramid image.

Pupil origins are zero-based `(row, column)` coordinates ordered q1 top-left,
q2 bottom-left, q3 bottom-right, and q4 top-right. Selected support coordinates
use Julia column-major order. The output at each selected coordinate is
`gain * (differential / normalization - reference)` in X/Y columns.
"""
struct PyramidImagePlan{
    T<:AbstractFloat,
    Support<:AbstractMatrix{Bool},
    Offsets<:AbstractVector{Int},
    Reference<:AbstractMatrix{T},
    Gain<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    pupil_size::NTuple{2,Int}
    origins::NTuple{PYRAMID_PUPIL_COUNT,NTuple{2,Int}}
    support::Support
    selected_offsets::Offsets
    reference_signal::Reference
    optical_gain::Gain
    normalization::PyramidImageNormalization
    incidence_flux::T
end

"""Caller-output staging for a stateless complete-frame pyramid operation."""
mutable struct PyramidImageWorkspace{
    T<:AbstractFloat,
    Signal<:AbstractMatrix{T},
    Divisor<:AbstractVector{T},
    Partials<:AbstractVector{T},
}
    staged_signal::Signal
    staged_divisor::Divisor
    normalization_partials::Partials
end

function _pyramid_image_normalization(policy::AbstractString)
    policy == "mean-valid-flux" && return PyramidImageMeanValidFlux
    policy == "incidence-flux" && return PyramidImageIncidenceFlux
    throw(ArgumentError("unknown pyramid normalization policy: $policy"))
end

function _pyramid_image_origins(values::AbstractVector{<:AbstractVector{<:Integer}})
    Base.require_one_based_indexing(values)
    length(values) == PYRAMID_PUPIL_COUNT || throw(
        DimensionMismatch("pupil_origins must contain q1, q2, q3, and q4"),
    )
    return ntuple(Val(PYRAMID_PUPIL_COUNT)) do index
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

function _pyramid_image_selected_offsets(support::AbstractMatrix{Bool})
    Base.require_one_based_indexing(support)
    offsets = Int[]
    sizehint!(offsets, count(support))
    # Julia's linear traversal is the required FGA column-major support order.
    for (ordinal, index) in enumerate(eachindex(support))
        support[index] && push!(offsets, ordinal - 1)
    end
    isempty(offsets) && throw(ArgumentError("pyramid I4Q support must be nonempty"))
    return offsets
end

function PyramidImagePlan(
    image_size::NTuple{2,Int},
    pupil_size::NTuple{2,Int},
    origins::NTuple{PYRAMID_PUPIL_COUNT,NTuple{2,Int}},
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    optical_gain::AbstractMatrix{T},
    normalization::PyramidImageNormalization,
    incidence_flux::T=zero(T),
) where {T<:AbstractFloat}
    image_rows, image_columns = image_size
    pupil_rows, pupil_columns = pupil_size
    image_rows > 0 && image_columns > 0 || throw(ArgumentError("image dimensions must be positive"))
    pupil_rows > 0 && pupil_columns > 0 || throw(ArgumentError("pupil dimensions must be positive"))
    size(support) == pupil_size || throw(
        DimensionMismatch("I4Q support extent must match pupil size"),
    )
    selected_offsets = _pyramid_image_selected_offsets(support)
    selected = length(selected_offsets)
    size(reference_signal) == (selected, 2) || throw(
        DimensionMismatch("pyramid reference signal extent is invalid"),
    )
    size(optical_gain) == (selected, 2) || throw(
        DimensionMismatch("pyramid optical gain extent is invalid"),
    )
    all(isfinite, reference_signal) || throw(
        ArgumentError("pyramid reference signal must be finite"),
    )
    all(isfinite, optical_gain) || throw(
        ArgumentError("pyramid optical gain must be finite"),
    )
    normalization === PyramidImageIncidenceFlux &&
        !(isfinite(incidence_flux) && incidence_flux > zero(T)) &&
        throw(ArgumentError("incidence flux must be finite and positive"))
    maximum_origin_row = image_rows - pupil_rows
    maximum_origin_column = image_columns - pupil_columns
    for (row, column) in origins
        0 <= row <= maximum_origin_row && 0 <= column <= maximum_origin_column ||
            throw(DimensionMismatch("pupil region exceeds detector image"))
    end
    return PyramidImagePlan(
        image_size,
        pupil_size,
        origins,
        Matrix(support),
        selected_offsets,
        Matrix(reference_signal),
        Matrix(optical_gain),
        normalization,
        incidence_flux,
    )
end

function PyramidImageWorkspace(plan::PyramidImagePlan{T}) where {T}
    return PyramidImageWorkspace(
        zeros(T, size(plan.reference_signal)),
        zeros(T, 1),
        zeros(T, PYRAMID_IMAGE_NORMALIZATION_LANES),
    )
end

commit!(workspace::PyramidImageWorkspace) = workspace
discard!(workspace::PyramidImageWorkspace) = workspace
reset!(workspace::PyramidImageWorkspace) = workspace

function _replace_pyramid_image_origins(
    plan::PyramidImagePlan,
    values::AbstractMatrix{UInt32},
)
    size(values) == (PYRAMID_PUPIL_COUNT, 2) || throw(
        DimensionMismatch("pupil origin extent is invalid"),
    )
    origins = ntuple(Val(PYRAMID_PUPIL_COUNT)) do index
        row, column = values[index, 1], values[index, 2]
        row <= typemax(Int) && column <= typemax(Int) || throw(
            ArgumentError("pupil origin cannot be represented by Int"),
        )
        (Int(row), Int(column))
    end
    return PyramidImagePlan(
        plan.image_size,
        plan.pupil_size,
        origins,
        plan.support,
        plan.reference_signal,
        plan.optical_gain,
        plan.normalization,
        plan.incidence_flux,
    )
end

function _replace_pyramid_image_support(
    plan::PyramidImagePlan,
    values::AbstractMatrix{Bool},
)
    size(values) == plan.pupil_size || throw(
        DimensionMismatch("pyramid I4Q support extent is invalid"),
    )
    count(values) == length(plan.selected_offsets) || throw(
        DimensionMismatch(
            "replacement I4Q support must preserve selected extent",
        ),
    )
    return PyramidImagePlan(
        plan.image_size,
        plan.pupil_size,
        plan.origins,
        values,
        plan.reference_signal,
        plan.optical_gain,
        plan.normalization,
        plan.incidence_flux,
    )
end

function _replace_pyramid_image_reference(
    plan::PyramidImagePlan{T},
    values::AbstractMatrix{T},
) where {T}
    return PyramidImagePlan(
        plan.image_size,
        plan.pupil_size,
        plan.origins,
        plan.support,
        values,
        plan.optical_gain,
        plan.normalization,
        plan.incidence_flux,
    )
end

function _replace_pyramid_image_gain(
    plan::PyramidImagePlan{T},
    values::AbstractMatrix{T},
) where {T}
    return PyramidImagePlan(
        plan.image_size,
        plan.pupil_size,
        plan.origins,
        plan.support,
        plan.reference_signal,
        values,
        plan.normalization,
        plan.incidence_flux,
    )
end

function _check_pyramid_image_execution(
    signal::AbstractMatrix{T},
    divisor::AbstractVector{T},
    workspace::PyramidImageWorkspace{T},
    plan::PyramidImagePlan{T},
    image::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(
        signal,
        divisor,
        image,
        workspace.staged_signal,
        workspace.staged_divisor,
        workspace.normalization_partials,
    )
    size(image) == plan.image_size || throw(
        DimensionMismatch("pyramid detector image extent is invalid"),
    )
    size(signal) == size(plan.reference_signal) || throw(
        DimensionMismatch("pyramid signal output extent is invalid"),
    )
    length(divisor) == 1 || throw(
        DimensionMismatch(
            "pyramid normalization divisor output must contain one value",
        ),
    )
    size(workspace.staged_signal) == size(signal) &&
        length(workspace.staged_divisor) == 1 &&
        length(workspace.normalization_partials) ==
            PYRAMID_IMAGE_NORMALIZATION_LANES || throw(
        DimensionMismatch("pyramid image Workspace extent is invalid"),
    )
    (Base.mightalias(signal, divisor) ||
     Base.mightalias(signal, image) ||
     Base.mightalias(divisor, image)) &&
        throw(ArgumentError("pyramid image caller arrays must not overlap"))
    (Base.mightalias(signal, plan.support) ||
     Base.mightalias(signal, plan.selected_offsets) ||
     Base.mightalias(signal, plan.reference_signal) ||
     Base.mightalias(signal, plan.optical_gain) ||
     Base.mightalias(divisor, plan.support) ||
     Base.mightalias(divisor, plan.selected_offsets) ||
     Base.mightalias(divisor, plan.reference_signal) ||
     Base.mightalias(divisor, plan.optical_gain) ||
     Base.mightalias(image, plan.support) ||
     Base.mightalias(image, plan.selected_offsets) ||
     Base.mightalias(image, plan.reference_signal) ||
     Base.mightalias(image, plan.optical_gain)) &&
        throw(ArgumentError("pyramid image caller arrays must not alias Plan storage"))
    (Base.mightalias(signal, workspace.staged_signal) ||
     Base.mightalias(signal, workspace.staged_divisor) ||
     Base.mightalias(divisor, workspace.staged_signal) ||
     Base.mightalias(divisor, workspace.staged_divisor) ||
     Base.mightalias(divisor, workspace.normalization_partials) ||
     Base.mightalias(image, workspace.staged_signal) ||
     Base.mightalias(image, workspace.staged_divisor) ||
     Base.mightalias(image, workspace.normalization_partials) ||
     Base.mightalias(signal, workspace.normalization_partials)) &&
        throw(ArgumentError("pyramid image caller arrays must not alias Workspace storage"))
    return nothing
end

function _pyramid_image_mean_valid_flux!(
    partials::AbstractVector{T},
    plan::PyramidImagePlan{T},
    image::AbstractMatrix{T},
) where {T}
    # This lane-stride accumulation and 128→1 tree are a numerical contract
    # shared with captured CUDA/HIP execution. Keep the operation order aligned
    # so finite/overflow dispositions cannot diverge between CPU and GPU.
    fill!(partials, zero(T))
    pupil_rows = plan.pupil_size[1]
    selected_count = length(plan.selected_offsets)
    q1, q2, q3, q4 = plan.origins
    @inbounds for first_selected in 1:PYRAMID_IMAGE_NORMALIZATION_LANES:selected_count
        chunk_count = min(PYRAMID_IMAGE_NORMALIZATION_LANES,
            selected_count - first_selected + 1)
        @simd for lane in 1:chunk_count
            selected = first_selected + lane - 1
            offset = plan.selected_offsets[selected]
            row = rem(offset, pupil_rows) + 1
            column = div(offset, pupil_rows) + 1
            coordinate_flux =
                image[q1[1] + row, q1[2] + column] +
                image[q2[1] + row, q2[2] + column] +
                image[q3[1] + row, q3[2] + column] +
                image[q4[1] + row, q4[2] + column]
            partials[lane] += coordinate_flux
        end
    end
    stride = div(PYRAMID_IMAGE_NORMALIZATION_LANES, 2)
    @inbounds while stride > 0
        for lane in 1:stride
            partials[lane] += partials[lane + stride]
        end
        stride = div(stride, 2)
    end
    return partials[1] / T(selected_count)
end

function _stage_pyramid_image!(
    workspace::PyramidImageWorkspace{T},
    plan::PyramidImagePlan{T},
    image::AbstractMatrix{T},
) where {T}
    pupil_rows, pupil_columns = plan.pupil_size
    q1, q2, q3, q4 = plan.origins
    @inbounds for (origin_row, origin_column) in plan.origins
        for column in 1:pupil_columns, row in 1:pupil_rows
            pixel = image[origin_row + row, origin_column + column]
            isfinite(pixel) || return PyramidImageNonFiniteInput
        end
    end
    if plan.normalization === PyramidImageMeanValidFlux
        divisor = _pyramid_image_mean_valid_flux!(
            workspace.normalization_partials,
            plan,
            image,
        )
    else
        divisor = plan.incidence_flux
    end
    isfinite(divisor) && divisor > zero(T) || return PyramidImageInvalidNormalization
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
        isfinite(output_x) && isfinite(output_y) || return PyramidImageNonFiniteOutput
        workspace.staged_signal[index, 1] = output_x
        workspace.staged_signal[index, 2] = output_y
    end
    workspace.staged_divisor[1] = divisor
    return nothing
end

function _publish_pyramid_image!(
    signal::AbstractMatrix{T},
    divisor::AbstractVector{T},
    workspace::PyramidImageWorkspace{T},
) where {T}
    copyto!(signal, workspace.staged_signal)
    divisor[1] = workspace.staged_divisor[1]
    return nothing
end

"""Process one complete calibrated pyramid frame with caller-owned outputs."""
function process!(
    signal::AbstractMatrix{T},
    normalization_divisor::AbstractVector{T},
    workspace::PyramidImageWorkspace{T},
    plan::PyramidImagePlan{T},
    image::AbstractMatrix{T},
) where {T}
    _check_pyramid_image_execution(
        signal,
        normalization_divisor,
        workspace,
        plan,
        image,
    )
    result = _stage_pyramid_image!(workspace, plan, image)
    result isa PyramidImageFailure && return result
    return _publish_pyramid_image!(signal, normalization_divisor, workspace)
end

function _configured_pyramid_image_support(
    values::Union{Nothing,Vector{Bool}},
    rows::Int,
    columns::Int,
)
    pixels = Base.checked_mul(rows, columns)
    isnothing(values) && return trues(rows, columns)
    length(values) == pixels || throw(
        DimensionMismatch(
            "i4q_support must contain $pixels column-major values",
        ),
    )
    return reshape(copy(values), rows, columns)
end

"""Float32 complete-frame calibrated pyramid I4Q differential signal."""
@algorithm PyramidImageF32 begin
    label = "pyramid-image-f32"
    failure = PyramidImageFailure

    """Configuration for PyramidImageF32."""
    @configuration PyramidImageF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        i4q_support::Union{Nothing,Vector{Bool}} => construction
        initial_reference_signal::Union{Nothing,Vector{Float32}} => construction
        initial_optical_gain::Union{Nothing,Vector{Float32}} => construction
        normalization_policy::String => graph_rebuild
        incidence_flux::Float32 => construction
    end

    ports(configuration) = (
        input(
            :image,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.image_size,
            CALIBRATED_PIXELS_V1,
        ),
        output(
            :signal,
            schema_element_type(PYRAMID_I4Q_SIGNAL_V1),
            (
                count(
                    _configured_pyramid_image_support(
                        configuration.i4q_support,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
                2,
            ),
            PYRAMID_I4Q_SIGNAL_V1;
            metadata_from=:image,
        ),
        output(
            :normalization_divisor,
            schema_element_type(PYRAMID_NORMALIZATION_DIVISOR_V1),
            (1,),
            PYRAMID_NORMALIZATION_DIVISOR_V1;
            name="normalization-divisor",
            metadata_from=:image,
        ),
        parameter(
            :pupil_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (PYRAMID_PUPIL_COUNT, 2),
            REGION_ORIGINS_V1;
            name="pupil-origins",
            replace=(plan, values) ->
                _replace_pyramid_image_origins(plan, values),
        ),
        parameter(
            :i4q_support,
            schema_element_type(PYRAMID_I4Q_SUPPORT_V1),
            configuration.pupil_size,
            PYRAMID_I4Q_SUPPORT_V1;
            name="i4q-support",
            replace=(plan, values) ->
                _replace_pyramid_image_support(plan, values),
        ),
        parameter(
            :reference_signal,
            schema_element_type(PYRAMID_REFERENCE_SIGNAL_V1),
            (
                count(
                    _configured_pyramid_image_support(
                        configuration.i4q_support,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
                2,
            ),
            PYRAMID_REFERENCE_SIGNAL_V1;
            name="reference-signal",
            replace=(plan, values) ->
                _replace_pyramid_image_reference(plan, values),
        ),
        parameter(
            :optical_gain,
            schema_element_type(PYRAMID_OPTICAL_GAIN_V1),
            (
                count(
                    _configured_pyramid_image_support(
                        configuration.i4q_support,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
                2,
            ),
            PYRAMID_OPTICAL_GAIN_V1;
            name="optical-gain",
            replace=(plan, values) ->
                _replace_pyramid_image_gain(plan, values),
        ),
    )

    prepare(configuration) = begin
        support = _configured_pyramid_image_support(
            configuration.i4q_support,
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
        plan = PyramidImagePlan(
            configuration.image_size,
            configuration.pupil_size,
            _pyramid_image_origins(configuration.pupil_origins),
            support,
            reference_values,
            gain_values,
            _pyramid_image_normalization(configuration.normalization_policy),
            configuration.incidence_flux,
        )
        (plan=plan, workspace=PyramidImageWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(signal, normalization_divisor, workspace, plan, image)
    end
end
