"""
    PixelCalibrationPlan(flat, background)

Prepare complete detector-plane flat and background calibration arrays. Both
arrays use Julia's logical `(row, column)` axes, must be nonempty and equal in
shape, and must contain only finite `Float32` values.
"""
struct PixelCalibrationPlan
    flat::Matrix{Float32}
    background::Matrix{Float32}

    function PixelCalibrationPlan(
        flat::AbstractMatrix{Float32},
        background::AbstractMatrix{Float32},
    )
        isempty(flat) && throw(ArgumentError("pixel calibration must be nonempty"))
        size(flat) == size(background) || throw(
            DimensionMismatch("flat and background calibration shapes must match"),
        )
        all(isfinite, flat) || throw(ArgumentError("flat calibration must be finite"))
        all(isfinite, background) ||
            throw(ArgumentError("background calibration must be finite"))
        return new(Matrix(flat), Matrix(background))
    end
end

@inline function _calibrated_pixel(
    raw::UInt16,
    flat::Float32,
    background::Float32,
)
    return flat * (Float32(raw) - background)
end

function _row_major_calibration_plane(
    values::Union{Nothing,AbstractVector{T}},
    rows::Int,
    columns::Int,
    default::T,
) where {T<:AbstractFloat}
    count = Base.checked_mul(rows, columns)
    if isnothing(values)
        return fill(default, rows, columns)
    end
    Base.require_one_based_indexing(values)
    length(values) == count || throw(
        DimensionMismatch("calibration plane must contain $count values"),
    )
    plane = Matrix{T}(undef, rows, columns)
    for row in 1:rows, column in 1:columns
        plane[row, column] = values[(row - 1) * columns + column]
    end
    return plane
end

function _replace_pixel_flat(plan::PixelCalibrationPlan, values)
    return PixelCalibrationPlan(values, plan.background)
end

function _replace_pixel_background(plan::PixelCalibrationPlan, values)
    return PixelCalibrationPlan(plan.flat, values)
end

function _process_pixel_calibration!(
    calibrated::AbstractMatrix{Float32},
    plan::PixelCalibrationPlan,
    raw::AbstractMatrix{UInt16},
)
    Base.require_one_based_indexing(calibrated, raw)
    size(raw) == size(plan.flat) ||
        throw(DimensionMismatch("raw detector shape is incompatible"))
    size(calibrated) == size(plan.flat) ||
        throw(DimensionMismatch("calibrated detector shape is incompatible"))
    Base.mightalias(calibrated, plan.flat) &&
        throw(ArgumentError("calibrated output must not alias flat calibration"))
    Base.mightalias(calibrated, plan.background) && throw(
        ArgumentError("calibrated output must not alias background calibration"),
    )

    rows, columns = size(raw)
    @inbounds for column in 1:columns, row in 1:rows
        calibrated[row, column] = _calibrated_pixel(
            raw[row, column],
            plan.flat[row, column],
            plan.background[row, column],
        )
    end
    return nothing
end

"""
Calibrate complete `UInt16` detector pixels to `Float32` by
`calibrated = flat * (Float32(raw) - background)` at each detector pixel.
The flat coefficient converts raw-count differences to calibrated-pixel units.
"""
@algorithm PixelCalibrationU16F32 begin
    label = "pixel-calibration-u16-f32"

    """
    Set complete-image rows/columns and optional initial flat/background arrays.
    The arrays have one coefficient per image pixel and may be replaced at the
    matching parameter ports.
    """
    @configuration PixelCalibrationU16F32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        initial_flat::Union{Nothing,Vector{Float32}} => construction
        initial_background::Union{Nothing,Vector{Float32}} => construction
    end

    ports(configuration) = (
        input(
            :raw,
            schema_element_type(RAW_DETECTOR_PIXELS_V1),
            configuration.image_size,
            RAW_DETECTOR_PIXELS_V1,
        ),
        output(
            :calibrated,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.image_size,
            CALIBRATED_PIXELS_V1;
            metadata_from=:raw,
        ),
        parameter(
            :flat,
            schema_element_type(FLAT_CALIBRATION_V1),
            configuration.image_size,
            FLAT_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_flat(plan, values),
        ),
        parameter(
            :background,
            schema_element_type(BACKGROUND_CALIBRATION_V1),
            configuration.image_size,
            BACKGROUND_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_background(plan, values),
        ),
    )

    prepare(configuration) = begin
        rows = configuration.image_size[1]
        columns = configuration.image_size[2]
        rows > 0 || throw(ArgumentError("image_size first dimension must be positive"))
        columns > 0 || throw(ArgumentError("image_size second dimension must be positive"))
        max_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
        encoded_rows = UInt64(rows)
        UInt64(columns) <= div(max_elements, encoded_rows) || throw(
            ArgumentError("detector image byte extent must fit UInt32"),
        )
        PixelCalibrationPlan(
            _row_major_calibration_plane(
                configuration.initial_flat,
                rows,
                columns,
                1.0f0,
            ),
            _row_major_calibration_plane(
                configuration.initial_background,
                rows,
                columns,
                0.0f0,
            ),
        )
    end

    process!(plan) = begin
        _process_pixel_calibration!(calibrated, plan, raw)
    end
end


"""Mutable publication state for row-block pixel calibration."""
mutable struct PixelCalibrationRowWorkspace
    pending_discontinuity::Bool
    candidate_pending_discontinuity::Bool
end

PixelCalibrationRowWorkspace() = PixelCalibrationRowWorkspace(false, false)

function commit!(workspace::PixelCalibrationRowWorkspace)
    workspace.pending_discontinuity = workspace.candidate_pending_discontinuity
    return workspace
end

function discard!(workspace::PixelCalibrationRowWorkspace)
    workspace.candidate_pending_discontinuity = workspace.pending_discontinuity
    return workspace
end

function reset!(workspace::PixelCalibrationRowWorkspace)
    workspace.pending_discontinuity = true
    workspace.candidate_pending_discontinuity = true
    return workspace
end

function _plan_adopted!(workspace::PixelCalibrationRowWorkspace)
    workspace.pending_discontinuity = true
    workspace.candidate_pending_discontinuity = true
    return workspace
end

"""Expected rejection of malformed row-block sample identity."""
@enum PixelCalibrationRowFailure::UInt8 begin
    PixelCalibrationRowMetadataMissing = 1
    PixelCalibrationRowOffsetMisaligned = 2
    PixelCalibrationRowOutOfBounds = 3
    PixelCalibrationRowTerminalMismatch = 4
end

@doc "A detector row block lacks metadata needed to select calibration coefficients." PixelCalibrationRowMetadataMissing
@doc "The row-block offset does not align with the configured block height." PixelCalibrationRowOffsetMisaligned
@doc "The row-block offset addresses rows outside the configured image." PixelCalibrationRowOutOfBounds
@doc "The row-block terminal marker disagrees with the configured image boundary." PixelCalibrationRowTerminalMismatch

"""
Calibrate a `UInt16` detector row block to `Float32`, selecting flat and
background coefficients by its zero-based row-block sample offset. Publishes
calibrated pixels only for a valid block within the configured image.
"""
@algorithm PixelCalibrationRowU16F32 begin
    label = "pixel-calibration-row-u16-f32"
    failure = PixelCalibrationRowFailure

    """
    Set image rows/columns, rows per readout block, and optional initial
    full-image flat/background arrays. Image rows must divide into row blocks;
    the sample offset identifies each block's image rows.
    """
    @configuration PixelCalibrationRowU16F32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
        initial_flat::Union{Nothing,Vector{Float32}} => construction
        initial_background::Union{Nothing,Vector{Float32}} => construction
    end

    ports(configuration) = (
        input(
            :raw,
            schema_element_type(RAW_PIXEL_ROW_BLOCK_V1),
            (configuration.row_block_rows, configuration.image_size[2]),
            RAW_PIXEL_ROW_BLOCK_V1;
            sample=:raw_sample,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
        ),
        output(
            :calibrated,
            schema_element_type(CALIBRATED_PIXEL_ROW_BLOCK_V1),
            (configuration.row_block_rows, configuration.image_size[2]),
            CALIBRATED_PIXEL_ROW_BLOCK_V1;
            metadata_from=:raw,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
        ),
        parameter(
            :flat,
            schema_element_type(FLAT_CALIBRATION_V1),
            configuration.image_size,
            FLAT_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_flat(plan, values),
        ),
        parameter(
            :background,
            schema_element_type(BACKGROUND_CALIBRATION_V1),
            configuration.image_size,
            BACKGROUND_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_background(plan, values),
        ),
    )

    prepare(configuration) = begin
        rows = configuration.image_size[1]
        columns = configuration.image_size[2]
        block_rows = configuration.row_block_rows
        rows > 0 || throw(ArgumentError("image_size first dimension must be positive"))
        columns > 0 || throw(ArgumentError("image_size second dimension must be positive"))
        0 < block_rows < rows && rem(rows, block_rows) == 0 || throw(
            ArgumentError(
                "row_block_rows must be positive, smaller than the image row count, and divide it",
            ),
        )
        plan = PixelCalibrationPlan(
            _row_major_calibration_plane(
                configuration.initial_flat,
                rows,
                columns,
                1.0f0,
            ),
            _row_major_calibration_plane(
                configuration.initial_background,
                rows,
                columns,
                0.0f0,
            ),
        )
        (plan=plan, workspace=PixelCalibrationRowWorkspace())
    end

    process!(plan, workspace) = begin
        isnothing(raw_sample) && return PixelCalibrationRowMetadataMissing
        first_row = Int(raw_sample.offset)
        block_rows = size(raw, 1)
        image_rows, image_columns = size(plan.flat)
        rem(first_row, block_rows) == 0 ||
            return PixelCalibrationRowOffsetMisaligned
        0 <= first_row <= image_rows - block_rows ||
            return PixelCalibrationRowOutOfBounds
        raw_sample.terminal == (first_row + block_rows == image_rows) ||
            return PixelCalibrationRowTerminalMismatch

        Base.require_one_based_indexing(calibrated, raw)
        size(raw, 2) == image_columns && size(calibrated) == size(raw) ||
            throw(DimensionMismatch("row-block calibration shape differs from the detector"))
        @inbounds for column in 1:image_columns, row in 1:block_rows
            detector_row = first_row + row
            calibrated[row, column] = _calibrated_pixel(
                raw[row, column],
                plan.flat[detector_row, column],
                plan.background[detector_row, column],
            )
        end
        discontinuity =
            raw_sample.discontinuity || workspace.candidate_pending_discontinuity
        workspace.candidate_pending_discontinuity = false
        return AlgorithmProcessComplete(
            SampleMetadata(raw_sample; discontinuity=discontinuity),
        )
    end
end
