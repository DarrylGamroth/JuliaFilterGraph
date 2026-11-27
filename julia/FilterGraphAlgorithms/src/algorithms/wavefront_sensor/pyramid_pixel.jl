"""Number of pupil images produced by a pyramid wavefront sensor."""
const PYRAMID_PUPIL_COUNT = 4

"""
    PyramidPixelStatus

Classify the numerical result of one pyramid pupil-pixel operation.

The status is retained as execution scratch so accelerator implementations can
decide whether to commit on the selected device. A successful public call has
`PyramidPixelValid` status. Numerical failure is reported as a
[`PyramidPixelNumericalError`](@ref).
"""
@enum PyramidPixelStatus::Int32 begin
    PyramidPixelValid = 0
    PyramidPixelNonFiniteInput = 1
    PyramidPixelNonFiniteMean = 2
    PyramidPixelNonFiniteNormalizationGain = 3
    PyramidPixelNonFiniteOutput = 4
end

@doc "The pyramid pupil input contains a non-finite pixel."
PyramidPixelNonFiniteInput
@doc "The full-pupil mean calculation produced a non-finite value."
PyramidPixelNonFiniteMean
@doc "The current-frame normalization gain is non-finite."
PyramidPixelNonFiniteNormalizationGain
@doc "Previous-frame normalization produced a non-finite output pixel."
PyramidPixelNonFiniteOutput
@doc "Pyramid pupil-pixel processing completed without a numerical failure."
PyramidPixelValid

"""Numerical failure from pyramid pupil-pixel processing."""
struct PyramidPixelNumericalError <: Exception
    status::PyramidPixelStatus
end

function Base.showerror(io::IO, error::PyramidPixelNumericalError)
    message = if error.status == PyramidPixelNonFiniteInput
        "pyramid pupil input contains a non-finite pixel"
    elseif error.status == PyramidPixelNonFiniteMean
        "pyramid full-pupil mean intensity is non-finite"
    elseif error.status == PyramidPixelNonFiniteNormalizationGain
        "pyramid current-frame normalization gain is non-finite"
    elseif error.status == PyramidPixelNonFiniteOutput
        "normalized pyramid output contains a non-finite pixel"
    else
        "pyramid pupil-pixel operation has an invalid failure status"
    end
    return print(io, message)
end

"""
    PyramidPixelPlan(mask)
    PyramidPixelPlan{T}(mask)

Prepare one shared pyramid pupil mask for previous-frame normalization.

The plan copies a nonempty logical mask and requires at least one selected
pixel. `T` is an `AbstractFloat` and defaults to `Float32`. The mask selects
reconstruction pixels but does not change the full-pupil mean. Selected pixel
offsets retain mask order and are prepared once for direct CPU or accelerator
gathering.
"""
struct PyramidPixelPlan{
    T<:AbstractFloat,
    Mask<:AbstractVector{Bool},
    Offsets<:AbstractVector{Int},
    Backend,
} <: AbstractAlgorithmPlan
    mask::Mask
    selected_pixel_offsets::Offsets
    output_pixel_count::Int
    mean_scale::T
    backend::Backend
end

function _prepared_pyramid_pixel_plan(
    ::Type{T},
    mask::Mask,
    selected_pixel_offsets::Offsets,
    output_pixel_count::Int,
    mean_scale::T,
    backend::Backend,
) where {
    T<:AbstractFloat,
    Mask<:AbstractVector{Bool},
    Offsets<:AbstractVector{Int},
    Backend,
}
    return PyramidPixelPlan{T,Mask,Offsets,Backend}(
        mask,
        selected_pixel_offsets,
        output_pixel_count,
        mean_scale,
        backend,
    )
end

function PyramidPixelPlan{T}(
    mask::AbstractVector{Bool},
) where {T<:AbstractFloat}
    prepared_mask = Vector{Bool}(mask)
    isempty(prepared_mask) &&
        throw(ArgumentError("pyramid pupil mask must be nonempty"))

    selected_pixel_offsets = Int[]
    sizehint!(selected_pixel_offsets, count(prepared_mask))
    for (pixel, selected) in enumerate(prepared_mask)
        selected && push!(selected_pixel_offsets, pixel - 1)
    end
    isempty(selected_pixel_offsets) &&
        throw(
            ArgumentError(
                "pyramid pupil mask must select reconstruction pixels",
            ),
        )

    output_pixel_count = Base.Checked.checked_mul(
        length(selected_pixel_offsets),
        PYRAMID_PUPIL_COUNT,
    )
    sample_count = Base.Checked.checked_mul(
        length(prepared_mask),
        PYRAMID_PUPIL_COUNT,
    )
    scalar_sample_count = T(sample_count)
    isfinite(scalar_sample_count) && scalar_sample_count > zero(T) ||
        throw(
            ArgumentError(
                "full-pupil sample count is not representable by $T",
            ),
        )
    mean_scale = inv(scalar_sample_count)
    isfinite(mean_scale) ||
        throw(
            ArgumentError(
                "full-pupil mean scale is not representable by $T",
            ),
        )

    return _prepared_pyramid_pixel_plan(
        T,
        prepared_mask,
        selected_pixel_offsets,
        output_pixel_count,
        mean_scale,
        nothing,
    )
end

PyramidPixelPlan(mask::AbstractVector{Bool}) = PyramidPixelPlan{Float32}(mask)

"""
    pyramid_pupil_mask(plan::PyramidPixelPlan)

Return a copy of the shared pyramid pupil mask in logical pixel order.
"""
pyramid_pupil_mask(plan::PyramidPixelPlan) = copy(plan.mask)

"""
    pyramid_pupil_pixel_count(plan::PyramidPixelPlan)

Return the full pixel extent of one pyramid pupil image.
"""
pyramid_pupil_pixel_count(plan::PyramidPixelPlan) = length(plan.mask)

"""
    pyramid_selected_pixel_count(plan::PyramidPixelPlan)

Return the number of selected pixels in one pyramid pupil image.
"""
pyramid_selected_pixel_count(plan::PyramidPixelPlan) =
    length(plan.selected_pixel_offsets)

"""
    pyramid_output_pixel_count(plan::PyramidPixelPlan)

Return the selected pixel extent across all four pyramid pupils.
"""
pyramid_output_pixel_count(plan::PyramidPixelPlan) = plan.output_pixel_count

"""
    prepare_pyramid_pixel_plan(device, plan)

Prepare immutable pyramid-pixel artifacts for an optional accelerator backend.

Backend extensions implement this preparation boundary for their concrete
device type. Preparation returns a new storage specialization of
[`PyramidPixelPlan`](@ref); it does not mutate the CPU plan. Device selection,
artifact transfer, kernel preparation, and first-use compilation happen here,
outside repeated [`process!`](@ref) calls. Unsupported backends have no
fallback method and therefore cannot silently execute on the CPU.
"""
function prepare_pyramid_pixel_plan end

"""
    PyramidPixelOutput(pixels, mean_pupil_intensity)

Group caller-owned outputs for one pyramid pupil-pixel frame.

`pixels` receives dimensionless normalized values in pupil-major order.
`mean_pupil_intensity` is a one-element vector that receives the current-frame
mean over every pixel in all four pupils, including masked pixels. The two
arrays are published coherently only after complete numerical success.
"""
struct PyramidPixelOutput{
    T<:AbstractFloat,
    Pixels<:AbstractVector{T},
    Mean<:AbstractVector{T},
}
    pixels::Pixels
    mean_pupil_intensity::Mean
end

const _HostPyramidVector{T} =
    Union{Vector{T},SubArray{T,1,<:Array}}
const _HostPyramidMatrix{T} =
    Union{Matrix{T},SubArray{T,2,<:Array}}

"""
    PyramidPixelWorkspace(plan)

Allocate committed normalization history and staged execution storage.

A new workspace starts with zero previous-frame gain, so its first successful
frame publishes zero normalized pixels and stages the measured reciprocal mean
for the next frame. The owning checked operation or Graph calls [`commit!`](@ref)
only after terminal success. Staged pixels, mean, next gain, and numerical
status allow output and history to remain unchanged on failure.
"""
mutable struct PyramidPixelWorkspace{
    T<:AbstractFloat,
    Values<:AbstractVector{T},
    Status<:AbstractVector{Int32},
    CompletionStatus<:AbstractVector{Int32},
}
    reciprocal_previous_frame_mean::Values
    staged_pixels::Values
    staged_mean_pupil_intensity::Values
    staged_reciprocal_current_frame_mean::Values
    status::Status
    completion_status::CompletionStatus
    candidate_ready::Bool
end

function _prepared_pyramid_pixel_workspace(
    ::Type{T},
    reciprocal_previous_frame_mean::Values,
    staged_pixels::Values,
    staged_mean_pupil_intensity::Values,
    staged_reciprocal_current_frame_mean::Values,
    status::Status,
    completion_status::CompletionStatus,
    candidate_ready::Bool=false,
) where {
    T<:AbstractFloat,
    Values<:AbstractVector{T},
    Status<:AbstractVector{Int32},
    CompletionStatus<:AbstractVector{Int32},
}
    return PyramidPixelWorkspace{T,Values,Status,CompletionStatus}(
        reciprocal_previous_frame_mean,
        staged_pixels,
        staged_mean_pupil_intensity,
        staged_reciprocal_current_frame_mean,
        status,
        completion_status,
        candidate_ready,
    )
end

function PyramidPixelWorkspace(
    plan::PyramidPixelPlan{T,Vector{Bool},Vector{Int},Nothing},
) where {T<:AbstractFloat}
    return _prepared_pyramid_pixel_workspace(
        T,
        zeros(T, 1),
        zeros(T, pyramid_output_pixel_count(plan)),
        zeros(T, 1),
        zeros(T, 1),
        zeros(Int32, 1),
        zeros(Int32, 1),
    )
end

function commit!(workspace::PyramidPixelWorkspace)
    if workspace.candidate_ready
        copyto!(
            workspace.reciprocal_previous_frame_mean,
            workspace.staged_reciprocal_current_frame_mean,
        )
        workspace.candidate_ready = false
    end
    return workspace
end

function discard!(workspace::PyramidPixelWorkspace)
    workspace.candidate_ready = false
    return workspace
end

function reset!(workspace::PyramidPixelWorkspace{T}) where {T}
    fill!(workspace.reciprocal_previous_frame_mean, zero(T))
    fill!(workspace.staged_pixels, zero(T))
    fill!(workspace.staged_mean_pupil_intensity, zero(T))
    fill!(workspace.staged_reciprocal_current_frame_mean, zero(T))
    fill!(workspace.status, Int32(PyramidPixelValid))
    fill!(workspace.completion_status, Int32(PyramidPixelValid))
    workspace.candidate_ready = false
    return workspace
end

_plan_adopted!(workspace::PyramidPixelWorkspace) = discard!(workspace)

"""
    pyramid_normalization_gain(workspace::PyramidPixelWorkspace{ T, Vector{T}, Vector{Int32}, Vector{Int32}, }) where {T<:AbstractFloat}

Return the previous-frame normalization gain committed by a CPU workspace.
"""
function pyramid_normalization_gain(
    workspace::PyramidPixelWorkspace{
        T,
        Vector{T},
        Vector{Int32},
        Vector{Int32},
    },
) where {T<:AbstractFloat}
    return workspace.reciprocal_previous_frame_mean[1]
end

function _check_pyramid_pixel_workspace(
    workspace::PyramidPixelWorkspace,
    plan::PyramidPixelPlan,
)
    length(workspace.reciprocal_previous_frame_mean) == 1 ||
        throw(DimensionMismatch("workspace committed gain must have length 1"))
    length(workspace.staged_pixels) == pyramid_output_pixel_count(plan) ||
        throw(
            DimensionMismatch(
                "workspace staged pixels must have length $(pyramid_output_pixel_count(plan))",
            ),
        )
    length(workspace.staged_mean_pupil_intensity) == 1 ||
        throw(DimensionMismatch("workspace staged mean must have length 1"))
    length(workspace.staged_reciprocal_current_frame_mean) == 1 ||
        throw(DimensionMismatch("workspace staged gain must have length 1"))
    length(workspace.status) == 1 ||
        throw(DimensionMismatch("workspace status must have length 1"))
    length(workspace.completion_status) == 1 ||
        throw(DimensionMismatch("workspace completion status must have length 1"))
    return nothing
end

@inline function _pyramid_pixel_aliases_workspace(
    array::AbstractArray,
    workspace::PyramidPixelWorkspace,
)
    Base.mightalias(
        array,
        workspace.reciprocal_previous_frame_mean,
    ) && return true
    Base.mightalias(array, workspace.staged_pixels) && return true
    Base.mightalias(
        array,
        workspace.staged_mean_pupil_intensity,
    ) && return true
    Base.mightalias(
        array,
        workspace.staged_reciprocal_current_frame_mean,
    ) && return true
    Base.mightalias(array, workspace.status) && return true
    return Base.mightalias(array, workspace.completion_status)
end

function _check_pyramid_pixel_execution(
    output::PyramidPixelOutput{T},
    workspace::PyramidPixelWorkspace{T},
    plan::PyramidPixelPlan{T},
    input::AbstractMatrix{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(
        output.pixels,
        output.mean_pupil_intensity,
        input,
    )
    expected_input = (pyramid_pupil_pixel_count(plan), PYRAMID_PUPIL_COUNT)
    size(input) == expected_input ||
        throw(
            DimensionMismatch(
                "pyramid pupil input must have shape $expected_input",
            ),
        )
    length(output.pixels) == pyramid_output_pixel_count(plan) ||
        throw(
            DimensionMismatch(
                "pyramid pixel output must have length $(pyramid_output_pixel_count(plan))",
            ),
        )
    length(output.mean_pupil_intensity) == 1 ||
        throw(
            DimensionMismatch(
                "mean pupil intensity output must have length 1",
            ),
        )
    _check_pyramid_pixel_workspace(workspace, plan)
    _pyramid_pixel_aliases_workspace(output.pixels, workspace) &&
        throw(ArgumentError("output pixels must not alias Workspace storage"))
    _pyramid_pixel_aliases_workspace(
        output.mean_pupil_intensity,
        workspace,
    ) && throw(ArgumentError("mean output must not alias Workspace storage"))
    _pyramid_pixel_aliases_workspace(input, workspace) &&
        throw(ArgumentError("pyramid input must not alias Workspace storage"))
    Base.mightalias(
        output.pixels,
        output.mean_pupil_intensity,
    ) && throw(ArgumentError("pyramid pixel and mean outputs must not alias"))
    return nothing
end

@noinline function _throw_pyramid_pixel_numerical_error(
    workspace::PyramidPixelWorkspace,
    status::PyramidPixelStatus,
)
    workspace.candidate_ready = false
    workspace.status[1] = Int32(status)
    workspace.completion_status[1] = Int32(status)
    throw(PyramidPixelNumericalError(status))
end

function _publish_pyramid_pixels!(
    output::AbstractVector,
    staged::AbstractVector,
)
    for (output_index, staged_index) in zip(axes(output, 1), axes(staged, 1))
        @inbounds output[output_index] = staged[staged_index]
    end
    return output
end

"""
    process!(output, workspace, plan::PyramidPixelPlan, input)

Normalize one complete four-pupil pyramid-WFS frame on the CPU.

`input` has logical shape `(pixels_per_pupil, 4)`; each column is one pupil in
pupil-major order. The current frame is multiplied by the reciprocal full-
pupil mean committed by the previous successful frame. The mask applies only
to output selection. The current full-pupil mean and its reciprocal are staged.
The owner calls [`commit!`](@ref) only after the complete sample succeeds.

CPU execution accepts strided host matrices and vectors, including padded and
noncontiguous views. It performs bounded work without heap allocation after
warmup, preserves caller output and committed gain on numerical failure, and
does not support progressive processing.
"""
function process!(
    output::PyramidPixelOutput{
        T,
        <:_HostPyramidVector{T},
        <:_HostPyramidVector{T},
    },
    workspace::PyramidPixelWorkspace{
        T,
        Vector{T},
        Vector{Int32},
        Vector{Int32},
    },
    plan::PyramidPixelPlan{
        T,
        Vector{Bool},
        Vector{Int},
        Nothing,
    },
    input::_HostPyramidMatrix{T},
) where {T<:AbstractFloat}
    _check_pyramid_pixel_execution(output, workspace, plan, input)
    workspace.candidate_ready = false
    workspace.status[1] = Int32(PyramidPixelValid)
    workspace.completion_status[1] = Int32(PyramidPixelValid)

    sum_intensity = zero(T)
    for pupil_axis in axes(input, 2)
        for pixel_axis in axes(input, 1)
            @inbounds value = input[pixel_axis, pupil_axis]
            isfinite(value) ||
                _throw_pyramid_pixel_numerical_error(
                    workspace,
                    PyramidPixelNonFiniteInput,
                )
            sum_intensity += value
            isfinite(sum_intensity) ||
                _throw_pyramid_pixel_numerical_error(
                    workspace,
                    PyramidPixelNonFiniteMean,
                )
        end
    end

    mean_pupil_intensity = sum_intensity * plan.mean_scale
    isfinite(mean_pupil_intensity) ||
        _throw_pyramid_pixel_numerical_error(
            workspace,
            PyramidPixelNonFiniteMean,
        )
    reciprocal_current_frame_mean = if mean_pupil_intensity > zero(T)
        inv(mean_pupil_intensity)
    else
        zero(T)
    end
    isfinite(reciprocal_current_frame_mean) ||
        _throw_pyramid_pixel_numerical_error(
            workspace,
            PyramidPixelNonFiniteNormalizationGain,
        )
    workspace.staged_mean_pupil_intensity[1] = mean_pupil_intensity
    workspace.staged_reciprocal_current_frame_mean[1] =
        reciprocal_current_frame_mean

    gain = workspace.reciprocal_previous_frame_mean[1]
    input_pixel_start = first(axes(input, 1))
    input_pupil_start = first(axes(input, 2))
    staged_pixel_start = first(axes(workspace.staged_pixels, 1))
    selected_pixel_count = pyramid_selected_pixel_count(plan)
    for pupil_offset in 0:(PYRAMID_PUPIL_COUNT - 1)
        pupil_axis = input_pupil_start + pupil_offset
        output_offset = pupil_offset * selected_pixel_count
        for selected_pixel in eachindex(plan.selected_pixel_offsets)
            input_pixel =
                input_pixel_start + plan.selected_pixel_offsets[selected_pixel]
            value = gain * input[input_pixel, pupil_axis]
            isfinite(value) ||
                _throw_pyramid_pixel_numerical_error(
                    workspace,
                    PyramidPixelNonFiniteOutput,
                )
            @inbounds workspace.staged_pixels[
                staged_pixel_start + output_offset + selected_pixel - 1
            ] = value
        end
    end

    _publish_pyramid_pixels!(output.pixels, workspace.staged_pixels)
    output.mean_pupil_intensity[first(axes(output.mean_pupil_intensity, 1))] =
        workspace.staged_mean_pupil_intensity[1]
    workspace.candidate_ready = true
    return output
end
