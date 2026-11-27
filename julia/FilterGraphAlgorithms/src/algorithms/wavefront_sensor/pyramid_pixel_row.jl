"""
    PyramidPixelRowPlan(...)

Prepare fixed detector geometry and a shared pyramid pupil mask for contiguous
row-block processing. Detector and pupil origins are zero based. The logical
mask axes are `(pupil row, pupil column)`.
"""
struct PyramidPixelRowPlan <: AbstractAlgorithmPlan
    image_rows::Int
    image_columns::Int
    block_rows::Int
    pupil_rows::Int
    pupil_columns::Int
    pupil_origins::NTuple{PYRAMID_PUPIL_COUNT,NTuple{2,Int}}
    pupil_mask::Matrix{Bool}
    selected_pixel_count::Int
    mean_scale::Float32
end

function _prepared_pyramid_pixel_row_plan(
    image_rows::Int,
    image_columns::Int,
    block_rows::Int,
    pupil_rows::Int,
    pupil_columns::Int,
    pupil_origins::NTuple{PYRAMID_PUPIL_COUNT,NTuple{2,Int}},
    pupil_mask::AbstractMatrix{Bool};
    selected_pixel_count::Union{Nothing,Int}=nothing,
)
    image_rows > 0 || throw(ArgumentError("image_size first dimension must be positive"))
    image_columns > 0 || throw(ArgumentError("image_size second dimension must be positive"))
    0 < block_rows < image_rows || throw(
        ArgumentError("row_block_rows must be positive and smaller than the image row count"),
    )
    rem(image_rows, block_rows) == 0 || throw(
        ArgumentError("row_block_rows must divide the image row count"),
    )
    pupil_rows > 0 || throw(ArgumentError("pupil_size first dimension must be positive"))
    pupil_columns > 0 || throw(ArgumentError("pupil_size second dimension must be positive"))
    size(pupil_mask) == (pupil_rows, pupil_columns) || throw(
        DimensionMismatch("pupil mask shape must be ($pupil_rows, $pupil_columns)"),
    )
    for (row, column) in pupil_origins
        row >= 0 && column >= 0 || throw(
            ArgumentError("pupil origins must be zero-based nonnegative coordinates"),
        )
        row <= image_rows - pupil_rows && column <= image_columns - pupil_columns || throw(
            ArgumentError("pupil region exceeds the detector image"),
        )
    end

    selected = count(pupil_mask)
    selected > 0 || throw(ArgumentError("pupil mask must select at least one pixel"))
    if !isnothing(selected_pixel_count)
        selected == selected_pixel_count || throw(
            DimensionMismatch(
                "replacement pupil mask must retain $selected_pixel_count selected pixels",
            ),
        )
    end
    sample_count = Base.checked_mul(
        Base.checked_mul(pupil_rows, pupil_columns),
        PYRAMID_PUPIL_COUNT,
    )
    scalar_sample_count = Float32(sample_count)
    isfinite(scalar_sample_count) || throw(
        ArgumentError("pyramid sample count is not representable by Float32"),
    )
    mean_scale = inv(scalar_sample_count)
    isfinite(mean_scale) || throw(
        ArgumentError("pyramid mean scale is not representable by Float32"),
    )
    return PyramidPixelRowPlan(
        image_rows,
        image_columns,
        block_rows,
        pupil_rows,
        pupil_columns,
        pupil_origins,
        Matrix(pupil_mask),
        selected,
        mean_scale,
    )
end

function _configured_pyramid_origins(
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

function _configured_pyramid_selected_pixel_count(
    values::Union{Nothing,AbstractVector{Bool}},
    rows::Int,
    columns::Int,
)
    isnothing(values) && return Base.checked_mul(rows, columns)
    return count(values)
end

function _configured_pyramid_mask(
    values::Union{Nothing,AbstractVector{Bool}},
    rows::Int,
    columns::Int,
)
    count = Base.checked_mul(rows, columns)
    isnothing(values) && return trues(rows, columns)
    Base.require_one_based_indexing(values)
    length(values) == count || throw(
        DimensionMismatch("pupil_mask must contain $count row-major values"),
    )
    mask = Matrix{Bool}(undef, rows, columns)
    for row in 1:rows, column in 1:columns
        mask[row, column] = values[(row - 1) * columns + column]
    end
    return mask
end

function _replace_pyramid_row_origins(plan::PyramidPixelRowPlan, values)
    origins = ntuple(Val(PYRAMID_PUPIL_COUNT)) do pupil
        (Int(values[pupil, 1]), Int(values[pupil, 2]))
    end
    return _prepared_pyramid_pixel_row_plan(
        plan.image_rows,
        plan.image_columns,
        plan.block_rows,
        plan.pupil_rows,
        plan.pupil_columns,
        origins,
        plan.pupil_mask;
        selected_pixel_count=plan.selected_pixel_count,
    )
end

function _replace_pyramid_row_mask(plan::PyramidPixelRowPlan, values)
    return _prepared_pyramid_pixel_row_plan(
        plan.image_rows,
        plan.image_columns,
        plan.block_rows,
        plan.pupil_rows,
        plan.pupil_columns,
        plan.pupil_origins,
        values;
        selected_pixel_count=plan.selected_pixel_count,
    )
end

"""Mutable progressive and candidate state for row-block pyramid processing."""
mutable struct PyramidPixelRowWorkspace{
    T<:Union{Float32,Float64},
    Detector<:AbstractMatrix{T},
    Pixels<:AbstractMatrix{T},
    ProcessedRows<:AbstractVector{Int},
    ProcessedSelected<:AbstractVector{Int},
}
    detector::Detector
    staged_reconstruction_pixels::Pixels
    committed_gain::T
    candidate_gain::T
    frame::RowBlockFrame
    processed_rows::ProcessedRows
    processed_selected::ProcessedSelected
    processed_pupil_count::Int
    summed_pupil::Int
    summed_rows::Int
    sum::T
    terminal_candidate::Bool
end

function _prepared_pyramid_pixel_row_workspace(
    detector::Detector,
    staged_reconstruction_pixels::Pixels,
    processed_rows::ProcessedRows,
    processed_selected::ProcessedSelected,
) where {
    T<:Union{Float32,Float64},
    Detector<:AbstractMatrix{T},
    Pixels<:AbstractMatrix{T},
    ProcessedRows<:AbstractVector{Int},
    ProcessedSelected<:AbstractVector{Int},
}
    return PyramidPixelRowWorkspace{
        T,
        Detector,
        Pixels,
        ProcessedRows,
        ProcessedSelected,
    }(
        detector,
        staged_reconstruction_pixels,
        zero(T),
        zero(T),
        RowBlockFrame(),
        processed_rows,
        processed_selected,
        0,
        1,
        0,
        zero(T),
        false,
    )
end

function PyramidPixelRowWorkspace(plan::PyramidPixelRowPlan)
    return _prepared_pyramid_pixel_row_workspace(
        zeros(Float32, plan.image_rows, plan.image_columns),
        zeros(Float32, PYRAMID_PUPIL_COUNT, plan.selected_pixel_count),
        zeros(Int, PYRAMID_PUPIL_COUNT),
        zeros(Int, PYRAMID_PUPIL_COUNT),
    )
end

function _clear_pyramid_row_partial!(workspace::PyramidPixelRowWorkspace)
    fill!(workspace.processed_rows, 0)
    fill!(workspace.processed_selected, 0)
    workspace.processed_pupil_count = 0
    workspace.summed_pupil = 1
    workspace.summed_rows = 0
    workspace.sum = zero(typeof(workspace.sum))
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_pyramid_row!(workspace::PyramidPixelRowWorkspace)
    _abandon_row_block_frame!(workspace.frame)
    workspace.candidate_gain = workspace.committed_gain
    return _clear_pyramid_row_partial!(workspace)
end

function commit!(workspace::PyramidPixelRowWorkspace)
    if workspace.terminal_candidate
        workspace.committed_gain = workspace.candidate_gain
        _complete_row_block_frame!(workspace.frame)
        _clear_pyramid_row_partial!(workspace)
    end
    workspace.candidate_gain = workspace.committed_gain
    return workspace
end

discard!(workspace::PyramidPixelRowWorkspace) = _abandon_pyramid_row!(workspace)

function reset!(workspace::PyramidPixelRowWorkspace)
    workspace.committed_gain = zero(typeof(workspace.committed_gain))
    workspace.candidate_gain = zero(typeof(workspace.candidate_gain))
    _reset_row_block_frame!(workspace.frame)
    return _clear_pyramid_row_partial!(workspace)
end

_plan_adopted!(workspace::PyramidPixelRowWorkspace) =
    _abandon_pyramid_row!(workspace)

"""
    pyramid_row_normalization_gain(workspace::PyramidPixelRowWorkspace)

Return the previous-complete-frame normalization gain.
"""
pyramid_row_normalization_gain(workspace::PyramidPixelRowWorkspace) =
    workspace.committed_gain

"""Expected rejection of one row-block pyramid call."""
@enum PyramidPixelRowFailure::UInt8 begin
    PyramidPixelRowMetadataMissing = 1
    PyramidPixelRowOffsetOverflow = 2
    PyramidPixelRowNonFiniteInput = 3
    PyramidPixelRowNonFiniteMean = 4
    PyramidPixelRowNonFiniteNormalizationGain = 5
    PyramidPixelRowNonFiniteOutput = 6
end

@doc "The row block did not carry required sample metadata."
PyramidPixelRowMetadataMissing
@doc "The row-block offset cannot be represented by the Julia target."
PyramidPixelRowOffsetOverflow
@doc "A pyramid pupil contains a non-finite input pixel."
PyramidPixelRowNonFiniteInput
@doc "The full-pupil mean calculation produced a non-finite value."
PyramidPixelRowNonFiniteMean
@doc "The current-frame normalization gain is non-finite."
PyramidPixelRowNonFiniteNormalizationGain
@doc "Previous-frame normalization produced a non-finite output pixel."
PyramidPixelRowNonFiniteOutput

function _process_ready_pyramid_rows!(
    plan::PyramidPixelRowPlan,
    workspace::PyramidPixelRowWorkspace,
    completed_rows::Int,
)
    gain = workspace.committed_gain
    for pupil in 1:PYRAMID_PUPIL_COUNT
        first_detector_row, first_detector_column = plan.pupil_origins[pupil]
        available_rows = min(
            max(completed_rows - first_detector_row, 0),
            plan.pupil_rows,
        )
        first_pupil_row = workspace.processed_rows[pupil]
        available_rows <= first_pupil_row && continue
        output_index = workspace.processed_selected[pupil] + 1
        for pupil_row in (first_pupil_row + 1):available_rows
            detector_row = first_detector_row + pupil_row
            for pupil_column in 1:plan.pupil_columns
                detector_column = first_detector_column + pupil_column
                input = workspace.detector[detector_row, detector_column]
                isfinite(input) || return PyramidPixelRowNonFiniteInput
                if plan.pupil_mask[pupil_row, pupil_column]
                    value = gain * input
                    isfinite(value) || return PyramidPixelRowNonFiniteOutput
                    workspace.staged_reconstruction_pixels[
                        pupil,
                        output_index,
                    ] = value
                    output_index += 1
                end
            end
        end
        workspace.processed_rows[pupil] = available_rows
        workspace.processed_selected[pupil] = output_index - 1
        available_rows == plan.pupil_rows &&
            (workspace.processed_pupil_count += 1)
    end

    while workspace.summed_pupil <= PYRAMID_PUPIL_COUNT
        pupil = workspace.summed_pupil
        first_detector_row, first_detector_column = plan.pupil_origins[pupil]
        while workspace.summed_rows < workspace.processed_rows[pupil]
            pupil_row = workspace.summed_rows + 1
            detector_row = first_detector_row + pupil_row
            for pupil_column in 1:plan.pupil_columns
                detector_column = first_detector_column + pupil_column
                workspace.sum +=
                    workspace.detector[detector_row, detector_column]
                isfinite(workspace.sum) || return PyramidPixelRowNonFiniteMean
            end
            workspace.summed_rows += 1
        end
        workspace.summed_rows < plan.pupil_rows && break
        workspace.summed_pupil += 1
        workspace.summed_rows = 0
    end
    return nothing
end

function _stage_pyramid_pixel_row!(
    plan::PyramidPixelRowPlan,
    workspace::PyramidPixelRowWorkspace,
    row_block::AbstractMatrix{Float32},
    sample::SampleMetadata,
)
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return PyramidPixelRowOffsetOverflow
    first_row = Int(sample.offset)
    accepted, completed_rows = _accept_row_block!(
        workspace.frame,
        sample,
        first_row,
        plan.block_rows,
        plan.image_rows,
    )
    if !accepted
        workspace.candidate_gain = workspace.committed_gain
        _clear_pyramid_row_partial!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(first_row) && _clear_pyramid_row_partial!(workspace)

    for column in 1:plan.image_columns, row in 1:plan.block_rows
        workspace.detector[first_row + row, column] = row_block[row, column]
    end
    status = _process_ready_pyramid_rows!(plan, workspace, completed_rows)
    isnothing(status) || return status
    sample.terminal || return AlgorithmProcessDeferred()

    if workspace.processed_pupil_count != PYRAMID_PUPIL_COUNT ||
       workspace.summed_pupil != PYRAMID_PUPIL_COUNT + 1
        _abandon_pyramid_row!(workspace)
        return AlgorithmProcessDeferred()
    end
    mean = workspace.sum * plan.mean_scale
    isfinite(mean) || return PyramidPixelRowNonFiniteMean
    candidate_gain = mean > 0.0f0 ? inv(mean) : 0.0f0
    isfinite(candidate_gain) || return PyramidPixelRowNonFiniteNormalizationGain

    workspace.candidate_gain = candidate_gain
    workspace.terminal_candidate = true
    return AlgorithmProcessComplete(
        SampleMetadata(
            sample;
            offset=0,
            terminal=true,
            discontinuity=workspace.frame.current_discontinuity,
        ),
    )
end

function _publish_pyramid_pixel_row!(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    plan::PyramidPixelRowPlan,
    workspace::PyramidPixelRowWorkspace,
)
    copyto!(reconstruction_pixels, workspace.staged_reconstruction_pixels)
    mean_pupil_intensity[1] = workspace.sum * plan.mean_scale
    return nothing
end

"""
Normalize successive pyramid pupil row blocks and publish selected
reconstruction pixels plus mean pupil intensity. Normalization uses the
previous complete frame's gain while the current frame's mean is collected.
"""
@algorithm PyramidPixelRowF32 begin
    label = "pyramid-pixel-row-f32"
    failure = PyramidPixelRowFailure

    """
    Set image and row-block geometry, four pupil rectangles and an optional
    Boolean pupil mask. Rows and columns are pixel counts; `row_block_rows` fixes
    the number of readout blocks per complete image.
    """
    @configuration PyramidPixelRowF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
    end

    ports(configuration) = (
        input(
            :row_block,
            schema_element_type(CALIBRATED_PIXEL_ROW_BLOCK_V1),
            (configuration.row_block_rows, configuration.image_size[2]),
            CALIBRATED_PIXEL_ROW_BLOCK_V1;
            name="row-block",
            sample=:row_sample,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
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
            metadata_from=:row_block,
            publication=:conditional,
        ),
        output(
            :mean_pupil_intensity,
            schema_element_type(PYRAMID_MEAN_PUPIL_INTENSITY_V1),
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1;
            name="mean-pupil-intensity",
            metadata_from=:row_block,
            publication=:conditional,
        ),
        parameter(
            :pupil_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (PYRAMID_PUPIL_COUNT, 2),
            REGION_ORIGINS_V1;
            name="pupil-origins",
            replace=(plan, values) -> _replace_pyramid_row_origins(plan, values),
        ),
        parameter(
            :pupil_mask,
            schema_element_type(PYRAMID_PUPIL_MASK_V1),
            configuration.pupil_size,
            PYRAMID_PUPIL_MASK_V1;
            name="pupil-mask",
            replace=(plan, values) -> _replace_pyramid_row_mask(plan, values),
        ),
    )

    prepare(configuration) = begin
        plan = _prepared_pyramid_pixel_row_plan(
            configuration.image_size[1],
            configuration.image_size[2],
            configuration.row_block_rows,
            configuration.pupil_size[1],
            configuration.pupil_size[2],
            _configured_pyramid_origins(configuration.pupil_origins),
            _configured_pyramid_mask(
                configuration.pupil_mask,
                configuration.pupil_size[1],
                configuration.pupil_size[2],
            ),
        )
        (plan=plan, workspace=PyramidPixelRowWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(row_sample) && return PyramidPixelRowMetadataMissing
        result = _stage_pyramid_pixel_row!(
            plan,
            workspace,
            row_block,
            row_sample,
        )
        if result isa AlgorithmProcessResult && result.complete
            _publish_pyramid_pixel_row!(
                reconstruction_pixels,
                mean_pupil_intensity,
                plan,
                workspace,
            )
        end
        result
    end
end
