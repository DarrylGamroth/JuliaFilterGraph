"""
Row-block pyramid sensing that publishes newly immutable normalized pixels.

Each block contains aligned `measurements` and zero-based reconstructor
`columns`; `count[1]` selects the valid prefix. Columns use the same pupil-major
order as `PyramidPixelF32` and the existing pyramid reconstructors. Detector
readout order therefore does not change scientific reconstruction order.
"""
mutable struct PyramidMeasurementBlockWorkspace{
    Pixels<:PyramidPixelRowWorkspace,
    PublishedSelected<:AbstractVector{Int},
}
    pixels::Pixels
    published_selected::PublishedSelected
end

function PyramidMeasurementBlockWorkspace(plan::PyramidPixelRowPlan)
    pixels = PyramidPixelRowWorkspace(plan)
    published_selected = zeros(Int, PYRAMID_PUPIL_COUNT)
    return PyramidMeasurementBlockWorkspace{
        typeof(pixels),
        typeof(published_selected),
    }(pixels, published_selected)
end

function _clear_pyramid_measurement_block!(
    workspace::PyramidMeasurementBlockWorkspace,
)
    fill!(workspace.published_selected, 0)
    return workspace
end

function commit!(workspace::PyramidMeasurementBlockWorkspace)
    terminal_candidate = workspace.pixels.terminal_candidate
    commit!(workspace.pixels)
    terminal_candidate && _clear_pyramid_measurement_block!(workspace)
    return workspace
end

function discard!(workspace::PyramidMeasurementBlockWorkspace)
    discard!(workspace.pixels)
    return _clear_pyramid_measurement_block!(workspace)
end

function reset!(workspace::PyramidMeasurementBlockWorkspace)
    reset!(workspace.pixels)
    return _clear_pyramid_measurement_block!(workspace)
end

function _plan_adopted!(workspace::PyramidMeasurementBlockWorkspace)
    _plan_adopted!(workspace.pixels)
    return _clear_pyramid_measurement_block!(workspace)
end

function _pyramid_measurement_block_aliases_workspace(
    array,
    workspace::PyramidMeasurementBlockWorkspace,
)
    pixels = workspace.pixels
    return Base.mightalias(array, pixels.detector) ||
        Base.mightalias(array, pixels.staged_reconstruction_pixels) ||
        Base.mightalias(array, pixels.processed_rows) ||
        Base.mightalias(array, pixels.processed_selected) ||
        Base.mightalias(array, workspace.published_selected)
end

function _validate_pyramid_measurement_block_output(
    output,
    row_block,
    workspace::PyramidMeasurementBlockWorkspace,
    plan::PyramidPixelRowPlan,
)
    Base.mightalias(output, row_block) && throw(
        ArgumentError("measurement-block output must not alias the row input"),
    )
    Base.mightalias(output, plan.pupil_mask) && throw(
        ArgumentError("measurement-block output must not alias the Plan"),
    )
    _pyramid_measurement_block_aliases_workspace(output, workspace) && throw(
        ArgumentError("measurement-block output must not alias Workspace storage"),
    )
    return nothing
end

function _validate_pyramid_measurement_block_storage(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::PyramidMeasurementBlockWorkspace,
    plan::PyramidPixelRowPlan,
    row_block::AbstractMatrix{Float32},
)
    Base.require_one_based_indexing(measurements, columns, count, row_block)
    capacity = Base.Checked.checked_mul(
        PYRAMID_PUPIL_COUNT,
        plan.selected_pixel_count,
    )
    length(measurements) == capacity || throw(
        DimensionMismatch("measurement block has an incompatible capacity"),
    )
    length(columns) == capacity || throw(
        DimensionMismatch("column block has an incompatible capacity"),
    )
    length(count) == 1 || throw(
        DimensionMismatch("measurement count must contain one value"),
    )
    size(row_block) == (plan.block_rows, plan.image_columns) || throw(
        DimensionMismatch("row block has an incompatible detector shape"),
    )
    size(workspace.pixels.detector) == (plan.image_rows, plan.image_columns) ||
        throw(
            DimensionMismatch(
                "pyramid measurement-block Workspace has an invalid detector",
            ),
        )
    Base.mightalias(measurements, columns) && throw(
        ArgumentError("measurement and column outputs must not alias"),
    )
    Base.mightalias(measurements, count) && throw(
        ArgumentError("measurement and count outputs must not alias"),
    )
    Base.mightalias(columns, count) && throw(
        ArgumentError("column and count outputs must not alias"),
    )
    _validate_pyramid_measurement_block_output(
        measurements,
        row_block,
        workspace,
        plan,
    )
    _validate_pyramid_measurement_block_output(
        columns,
        row_block,
        workspace,
        plan,
    )
    _validate_pyramid_measurement_block_output(
        count,
        row_block,
        workspace,
        plan,
    )
    _pyramid_measurement_block_aliases_workspace(row_block, workspace) && throw(
        ArgumentError("row input must not alias Workspace storage"),
    )
    return nothing
end

function _publish_pyramid_measurement_block!(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::PyramidMeasurementBlockWorkspace,
    plan::PyramidPixelRowPlan,
)
    block_count = 0
    selected = plan.selected_pixel_count
    pixels = workspace.pixels
    @inbounds for pupil in 1:PYRAMID_PUPIL_COUNT
        first_pixel = workspace.published_selected[pupil] + 1
        last_pixel = pixels.processed_selected[pupil]
        for pixel in first_pixel:last_pixel
            block_count += 1
            measurements[block_count] =
                pixels.staged_reconstruction_pixels[pupil, pixel]
            columns[block_count] = UInt32((pupil - 1) * selected + pixel - 1)
        end
        workspace.published_selected[pupil] = last_pixel
    end
    count[1] = UInt32(block_count)
    return nothing
end

function _process_pyramid_measurement_block!(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::PyramidMeasurementBlockWorkspace,
    plan::PyramidPixelRowPlan,
    row_block::AbstractMatrix{Float32},
    sample::SampleMetadata,
)
    _validate_pyramid_measurement_block_storage(
        measurements,
        columns,
        count,
        workspace,
        plan,
        row_block,
    )
    iszero(sample.offset) && _clear_pyramid_measurement_block!(workspace)
    result = _stage_pyramid_pixel_row!(plan, workspace.pixels, row_block, sample)
    if result isa PyramidPixelRowFailure
        discard!(workspace)
        return result
    end
    if !workspace.pixels.frame.active
        _clear_pyramid_measurement_block!(workspace)
        return result
    end

    _publish_pyramid_measurement_block!(
        measurements,
        columns,
        count,
        workspace,
        plan,
    )
    discontinuity = sample.discontinuity ||
        (iszero(sample.offset) && workspace.pixels.frame.current_discontinuity)
    return AlgorithmProcessComplete(
        SampleMetadata(sample; discontinuity=discontinuity),
    )
end

"""
From successive pyramid pixel row blocks, publish sparse measurement values,
their zero-based dense-reconstructor columns, and the number of valid entries
in each block. Values follow the pyramid pixel normalization contract.
"""
@algorithm PyramidMeasurementBlockF32 begin
    label = "pyramid-measurement-block-f32"
    failure = PyramidPixelRowFailure

    """
    Set detector/row-block geometry, four pupil origins and optional pupil mask,
    plus versioned measurement, column-index and count schemas. Pupil dimensions
    and mask fix each block's sparse output capacity.
    """
    @configuration PyramidMeasurementBlockF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
        measurements_schema::String => graph_rebuild
        column_indices_schema::String => graph_rebuild
        measurement_count_schema::String => graph_rebuild
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
            :measurements,
            Float32,
            (
                Base.Checked.checked_mul(
                    PYRAMID_PUPIL_COUNT,
                    _configured_pyramid_selected_pixel_count(
                        configuration.pupil_mask,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
            ),
            configuration.measurements_schema;
            metadata_from=:row_block,
            publication=:conditional,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
        ),
        output(
            :columns,
            UInt32,
            (
                Base.Checked.checked_mul(
                    PYRAMID_PUPIL_COUNT,
                    _configured_pyramid_selected_pixel_count(
                        configuration.pupil_mask,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
            ),
            configuration.column_indices_schema;
            metadata_from=:row_block,
            publication=:conditional,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
        ),
        output(
            :count,
            UInt32,
            (1,),
            configuration.measurement_count_schema;
            metadata_from=:row_block,
            publication=:conditional,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
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
        (plan=plan, workspace=PyramidMeasurementBlockWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(row_sample) && return PyramidPixelRowMetadataMissing
        _process_pyramid_measurement_block!(
            measurements,
            columns,
            count,
            workspace,
            plan,
            row_block,
            row_sample,
        )
    end
end
