"""
    PyramidPixelImagePlan(image_size, pupil_size, pupil_origins, pupil_mask)

Prepare four detector-image pupil regions and shared previous-frame pyramid
normalization as one image-backed scientific operation. Image and pupil sizes
use `(rows, columns)`. Pupil origins are zero-based `(row, column)` pairs, and
the Boolean mask is supplied in row-major pupil-pixel order.

The prepared operation reads pupil pixels directly from each detector image;
it does not materialize an intermediate four-pupil slab. It prepares selected
image indices in mask order for canonical storage and the dense parent of a
logical transpose. Plan storage is logically read-only; replace pupil origins
or masks through parameters to rebuild both lookups. Full-pupil reductions
may reassociate floating-point additions; numerical accuracy and previous-frame
normalization semantics are preserved, without a cross-backend bitwise guarantee.
"""
struct PyramidPixelImagePlan <: AbstractAlgorithmPlan
    pupil_regions::RegionExtractionPlan
    normalization::PyramidPixelPlan{
        Float32,
        Vector{Bool},
        Vector{Int},
        Nothing,
    }
    selected_indices::Matrix{Int}
    transposed_indices::Matrix{Int}
    transposed_run_starts::Vector{Int}
    ordered_sum::Bool

    function PyramidPixelImagePlan(
        pupil_regions::RegionExtractionPlan,
        normalization::PyramidPixelPlan{
            Float32,
            Vector{Bool},
            Vector{Int},
            Nothing,
        },
        ordered_sum::Bool=false,
    )
        length(pupil_regions.origins) == PYRAMID_PUPIL_COUNT || throw(
            DimensionMismatch("pyramid image geometry must contain four pupils"),
        )
        region_pixels = Base.checked_mul(pupil_regions.region_size...)
        region_pixels == pyramid_pupil_pixel_count(normalization) || throw(
            DimensionMismatch(
                "pupil region and normalization mask pixel extents must match",
            ),
        )
        selected_indices, transposed_indices =
            _pyramid_image_selected_indices(pupil_regions, normalization)
        run_starts = _pyramid_image_transposed_run_starts(transposed_indices)
        return new(pupil_regions, normalization, selected_indices,
            transposed_indices, run_starts, ordered_sum)
    end

    # Layout-only Graph preparation retains the canonical summation order
    # without rebuilding the detached Plan's pixel lookups.
    function PyramidPixelImagePlan(plan::PyramidPixelImagePlan, ordered_sum::Bool)
        return new(plan.pupil_regions, plan.normalization, plan.selected_indices,
            plan.transposed_indices, plan.transposed_run_starts, ordered_sum)
    end
end

# Each pupil has the same selected-offset pattern; only its image base changes.
# The final sentinel makes each run's length a subtraction at execution time.
function _pyramid_image_transposed_run_starts(indices::Matrix{Int})
    selected_count = size(indices, 1)
    starts = Int[1]
    for selected in 2:selected_count
        indices[selected, 1] == indices[selected - 1, 1] + 1 ||
            push!(starts, selected)
    end
    push!(starts, selected_count + 1)
    return starts
end

# Retain the mask's scientific row-major order while addressing Julia image
# storage directly. Both owned lookups retain that order; the transposed lookup
# addresses the dense parent without per-selected-pixel coordinate division.
function _pyramid_image_selected_indices(regions, normalization)
    offsets = normalization.selected_pixel_offsets
    indices = Matrix{Int}(undef, length(offsets), PYRAMID_PUPIL_COUNT)
    transposed_indices = similar(indices)
    linear = LinearIndices(regions.image_size)
    transposed_linear = LinearIndices(reverse(regions.image_size))
    columns = regions.region_size[2]
    for pupil in eachindex(regions.origins)
        first_row, first_column = regions.origins[pupil]
        for selected in eachindex(offsets)
            row, column = divrem(offsets[selected], columns)
            indices[selected, pupil] = linear[first_row + row + 1, first_column + column + 1]
            transposed_indices[selected, pupil] =
                transposed_linear[first_column + column + 1, first_row + row + 1]
        end
    end
    return indices, transposed_indices
end

@inline _pyramid_image_storage(plan::PyramidPixelImagePlan, image::AbstractMatrix{Float32}) =
    (image, plan.selected_indices)
@inline _pyramid_image_storage(
    plan::PyramidPixelImagePlan, image::Transpose{Float32,Matrix{Float32}},
) = (parent(image), plan.transposed_indices)

function PyramidPixelImagePlan(
    image_size::NTuple{2,Int},
    pupil_size::NTuple{2,Int},
    pupil_origins,
    pupil_mask::AbstractVector{Bool},
)
    regions = RegionExtractionPlan(image_size, pupil_origins, pupil_size)
    normalization = PyramidPixelPlan{Float32}(pupil_mask)
    return PyramidPixelImagePlan(regions, normalization)
end

function _configured_pyramid_image_origins(
    values::AbstractVector{<:AbstractVector{<:Integer}},
)
    Base.require_one_based_indexing(values)
    length(values) == PYRAMID_PUPIL_COUNT || throw(
        DimensionMismatch("pupil_origins must contain four coordinate pairs"),
    )
    return ntuple(Val(PYRAMID_PUPIL_COUNT)) do pupil
        origin = values[pupil]
        Base.require_one_based_indexing(origin)
        length(origin) == 2 || throw(
            DimensionMismatch("each pupil origin must contain row and column"),
        )
        (origin[1], origin[2])
    end
end

function _replace_pyramid_image_origins(
    plan::PyramidPixelImagePlan,
    values::AbstractMatrix{UInt32},
)
    origins = ntuple(Val(PYRAMID_PUPIL_COUNT)) do pupil
        row = values[pupil, 1]
        column = values[pupil, 2]
        row <= typemax(Int) && column <= typemax(Int) || throw(
            ArgumentError("pupil origin cannot be represented by Int"),
        )
        (Int(row), Int(column))
    end
    regions = RegionExtractionPlan(
        plan.pupil_regions.image_size,
        origins,
        plan.pupil_regions.region_size,
    )
    return PyramidPixelImagePlan(regions, plan.normalization, plan.ordered_sum)
end

function _replace_pyramid_image_mask(
    plan::PyramidPixelImagePlan,
    values::AbstractMatrix{Bool},
)
    rows, columns = size(values)
    row_major_mask = Vector{Bool}(undef, Base.checked_mul(rows, columns))
    for row in axes(values, 1), column in axes(values, 2)
        row_major_mask[(row - 1) * columns + column] = values[row, column]
    end
    normalization = PyramidPixelPlan{Float32}(row_major_mask)
    pyramid_selected_pixel_count(normalization) ==
        pyramid_selected_pixel_count(plan.normalization) || throw(
        DimensionMismatch(
            "replacement pupil mask must preserve selected-pixel extent",
        ),
    )
    return PyramidPixelImagePlan(plan.pupil_regions, normalization, plan.ordered_sum)
end

"""Expected numerical rejection from image-backed pyramid processing."""
@enum PyramidPixelImageFailure::UInt8 begin
    PyramidPixelImageNonFiniteInput = 1
    PyramidPixelImageNonFiniteMean = 2
    PyramidPixelImageNonFiniteNormalizationGain = 3
    PyramidPixelImageNonFiniteOutput = 4
end

@doc "A detector pupil contains a non-finite input pixel."
PyramidPixelImageNonFiniteInput
@doc "The full-pupil mean calculation produced a non-finite value."
PyramidPixelImageNonFiniteMean
@doc "The current-frame normalization gain is non-finite."
PyramidPixelImageNonFiniteNormalizationGain
@doc "Previous-frame normalization produced a non-finite output pixel."
PyramidPixelImageNonFiniteOutput

include("pyramid_pupil_sum.jl")

function _check_pyramid_pixel_image_execution(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    workspace::PyramidPixelWorkspace,
    plan::PyramidPixelImagePlan,
    image::AbstractMatrix{Float32},
)
    Base.require_one_based_indexing(
        reconstruction_pixels,
        mean_pupil_intensity,
        image,
    )
    size(image) == plan.pupil_regions.image_size || throw(
        DimensionMismatch("pyramid detector-image extent is invalid"),
    )
    selected = pyramid_selected_pixel_count(plan.normalization)
    size(reconstruction_pixels) == (PYRAMID_PUPIL_COUNT, selected) || throw(
        DimensionMismatch("pyramid reconstruction-pixel extent is invalid"),
    )
    length(mean_pupil_intensity) == 1 || throw(
        DimensionMismatch("mean-pupil-intensity output must contain one value"),
    )
    _check_pyramid_pixel_workspace(workspace, plan.normalization)
    _pyramid_pixel_aliases_workspace(reconstruction_pixels, workspace) && throw(
        ArgumentError("reconstruction pixels must not alias Workspace storage"),
    )
    _pyramid_pixel_aliases_workspace(mean_pupil_intensity, workspace) && throw(
        ArgumentError("mean pupil intensity must not alias Workspace storage"),
    )
    _pyramid_pixel_aliases_workspace(image, workspace) && throw(
        ArgumentError("detector image must not alias Workspace storage"),
    )
    Base.mightalias(reconstruction_pixels, mean_pupil_intensity) && throw(
        ArgumentError("pyramid image outputs must not overlap"),
    )
    for indices in (plan.selected_indices, plan.transposed_indices)
        (_pyramid_pixel_aliases_workspace(indices, workspace) ||
         Base.mightalias(reconstruction_pixels, indices) ||
         Base.mightalias(mean_pupil_intensity, indices)) && throw(
            ArgumentError("selected pupil indices must not alias writable execution storage"),
        )
    end
    run_starts = plan.transposed_run_starts
    (_pyramid_pixel_aliases_workspace(run_starts, workspace) ||
     Base.mightalias(reconstruction_pixels, run_starts) ||
     Base.mightalias(mean_pupil_intensity, run_starts)) && throw(
        ArgumentError("selected pupil runs must not alias writable execution storage"),
    )
    return nothing
end

function _stage_pyramid_selected_pixels_indexed!(
    workspace::PyramidPixelWorkspace{Float32},
    plan::PyramidPixelImagePlan,
    image::AbstractMatrix{Float32},
    gain::Float32,
)
    selected_count = pyramid_selected_pixel_count(plan.normalization)
    storage, indices = _pyramid_image_storage(plan, image)
    valid = true
    @inbounds for pupil in axes(indices, 2)
        @simd for selected in axes(indices, 1)
            value = gain * storage[indices[selected, pupil]]
            valid &= isfinite(value)
            workspace.staged_pixels[
                (pupil - 1) * selected_count + selected
            ] = value
        end
    end
    return valid
end

_stage_pyramid_selected_pixels!(
    workspace::PyramidPixelWorkspace{Float32},
    plan::PyramidPixelImagePlan,
    image::AbstractMatrix{Float32},
    gain::Float32,
) = _stage_pyramid_selected_pixels_indexed!(workspace, plan, image, gain)

@inline function _pyramid_image_use_transposed_runs(plan::PyramidPixelImagePlan)
    selected_count = pyramid_selected_pixel_count(plan.normalization)
    run_count = length(plan.transposed_run_starts) - 1
    return selected_count >= 1024 && div(selected_count, run_count) >= 16
end

function _stage_pyramid_selected_pixels!(
    workspace::PyramidPixelWorkspace{Float32},
    plan::PyramidPixelImagePlan,
    image::Transpose{Float32,Matrix{Float32}},
    gain::Float32,
)
    selected_count = pyramid_selected_pixel_count(plan.normalization)
    storage = parent(image)
    indices = plan.transposed_indices
    starts = plan.transposed_run_starts
    # Run overhead wins only for sufficiently long selected streams. The
    # threshold excludes fragmented and small masks measured on native layout.
    if !_pyramid_image_use_transposed_runs(plan)
        return _stage_pyramid_selected_pixels_indexed!(workspace, plan, image, gain)
    end
    valid = true
    @inbounds for pupil in axes(indices, 2)
        for run in 1:length(starts)-1
            first = starts[run]
            source = indices[first, pupil]
            destination = (pupil - 1) * selected_count + first
            run_valid = true
            @simd for offset in 0:(starts[run + 1] - first - 1)
                value = gain * storage[source + offset]
                run_valid &= isfinite(value)
                workspace.staged_pixels[destination + offset] = value
            end
            valid &= run_valid
        end
    end
    return valid
end

@inline function _reject_pyramid_pixel_image!(
    workspace::PyramidPixelWorkspace,
    failure::PyramidPixelImageFailure,
)
    workspace.candidate_ready = false
    workspace.status[1] = Int32(failure)
    workspace.completion_status[1] = Int32(failure)
    return failure
end

function _stage_pyramid_pixel_image!(
    workspace::PyramidPixelWorkspace{Float32},
    plan::PyramidPixelImagePlan,
    image::AbstractMatrix{Float32},
)
    workspace.candidate_ready = false
    workspace.status[1] = Int32(PyramidPixelValid)
    workspace.completion_status[1] = Int32(PyramidPixelValid)
    normalization = plan.normalization
    sum_intensity, failure = _pyramid_image_sum(plan, image)
    isnothing(failure) || return _reject_pyramid_pixel_image!(workspace, failure)

    mean = sum_intensity * normalization.mean_scale
    isfinite(mean) || return _reject_pyramid_pixel_image!(
        workspace,
        PyramidPixelImageNonFiniteMean,
    )
    candidate_gain = mean > 0.0f0 ? inv(mean) : 0.0f0
    isfinite(candidate_gain) || return _reject_pyramid_pixel_image!(
        workspace,
        PyramidPixelImageNonFiniteNormalizationGain,
    )
    workspace.staged_mean_pupil_intensity[1] = mean
    workspace.staged_reciprocal_current_frame_mean[1] = candidate_gain

    gain = workspace.reciprocal_previous_frame_mean[1]
    valid = _stage_pyramid_selected_pixels!(workspace, plan, image, gain)
    # Candidates are disposable on failure. No caller output or previous gain
    # changes until every selected result has passed the same finite check.
    valid || return _reject_pyramid_pixel_image!(workspace, PyramidPixelImageNonFiniteOutput)
    return nothing
end

function _publish_pyramid_pixel_image!(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    workspace::PyramidPixelWorkspace,
    plan::PyramidPixelImagePlan,
)
    selected_count = pyramid_selected_pixel_count(plan.normalization)
    @inbounds for pupil in 1:PYRAMID_PUPIL_COUNT
        for selected in 1:selected_count
            reconstruction_pixels[pupil, selected] = workspace.staged_pixels[
                (pupil - 1) * selected_count + selected
            ]
        end
    end
    mean_pupil_intensity[1] = workspace.staged_mean_pupil_intensity[1]
    workspace.candidate_ready = true
    return nothing
end

# Dense Julia outputs have four adjacent pupils per selected pixel. Checked
# callers prove the (4, selected_count) shape and disjoint workspace/output
# storage. Each SIMD iteration owns four distinct output cells; the four
# source planes are read-only. Native row-major and generic outputs retain
# the pupil-major method above.
function _publish_pyramid_pixel_image!(
    reconstruction_pixels::Matrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    workspace::PyramidPixelWorkspace{Float32,Vector{Float32}},
    plan::PyramidPixelImagePlan,
)
    selected_count = pyramid_selected_pixel_count(plan.normalization)
    staged = workspace.staged_pixels
    @inbounds @simd ivdep for selected in 1:selected_count
        reconstruction_pixels[4selected - 3] = staged[selected]
        reconstruction_pixels[4selected - 2] = staged[selected_count + selected]
        reconstruction_pixels[4selected - 1] = staged[2selected_count + selected]
        reconstruction_pixels[4selected] = staged[3selected_count + selected]
    end
    mean_pupil_intensity[1] = workspace.staged_mean_pupil_intensity[1]
    workspace.candidate_ready = true
    return nothing
end

@inline function _finish_pyramid_pupil_image!(
    ::Nothing,
    reconstruction_pixels,
    mean_pupil_intensity,
    workspace,
    plan,
)
    _publish_pyramid_pixel_image!(
        reconstruction_pixels,
        mean_pupil_intensity,
        workspace,
        plan,
    )
    return nothing
end

@inline _finish_pyramid_pupil_image!(
    failure::PyramidPixelImageFailure,
    reconstruction_pixels,
    mean_pupil_intensity,
    workspace,
    plan,
) = failure

"""Float32 image-backed pyramid pupil normalization."""
@algorithm PyramidPupilImageF32 begin
    label = "pyramid-pupil-image-f32"
    failure = PyramidPixelImageFailure

    """Configuration for PyramidPupilImageF32."""
    @configuration PyramidPupilImageF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
    end

    ports(configuration) = (
        input(
            :image,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.image_size,
            CALIBRATED_PIXELS_V1,
        ),
        output(
            :reconstruction_pixels,
            schema_element_type(PYRAMID_RECONSTRUCTION_PIXELS_V1),
            (
                PYRAMID_PUPIL_COUNT,
                _configured_pyramid_selected_pixel_count(
                    configuration.pupil_mask,
                    configuration.pupil_size[1],
                    configuration.pupil_size[2],
                ),
            ),
            PYRAMID_RECONSTRUCTION_PIXELS_V1;
            name="reconstruction-pixels",
            metadata_from=:image,
        ),
        output(
            :mean_pupil_intensity,
            schema_element_type(PYRAMID_MEAN_PUPIL_INTENSITY_V1),
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1;
            name="mean-pupil-intensity",
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
            :pupil_mask,
            schema_element_type(PYRAMID_PUPIL_MASK_V1),
            configuration.pupil_size,
            PYRAMID_PUPIL_MASK_V1;
            name="pupil-mask",
            replace=(plan, values) -> _replace_pyramid_image_mask(plan, values),
        ),
    )

    prepare(configuration) = begin
        pupil_pixels = Base.checked_mul(
            configuration.pupil_size[1],
            configuration.pupil_size[2],
        )
        isnothing(configuration.pupil_mask) ||
            length(configuration.pupil_mask) == pupil_pixels || throw(
            DimensionMismatch(
                "pupil_mask must contain $pupil_pixels row-major values",
            ),
        )
        pupil_mask = isnothing(configuration.pupil_mask) ?
                     trues(pupil_pixels) : configuration.pupil_mask
        plan = PyramidPixelImagePlan(
            configuration.image_size,
            configuration.pupil_size,
            _configured_pyramid_image_origins(configuration.pupil_origins),
            pupil_mask,
        )
        (plan=plan, workspace=PyramidPixelWorkspace(plan.normalization))
    end

    process!(plan, workspace) = begin
        _check_pyramid_pixel_image_execution(
            reconstruction_pixels,
            mean_pupil_intensity,
            workspace,
            plan,
            image,
        )
        result = _stage_pyramid_pixel_image!(workspace, plan, image)
        _finish_pyramid_pupil_image!(
            result,
            reconstruction_pixels,
            mean_pupil_intensity,
            workspace,
            plan,
        )
    end
end
