"""
One detector-readout region in zero-based detector `(row, column)` coordinates.

`line_step` and `sample_step` describe acquisition movement through detector
coordinates. They are independent of Julia array strides, so a mapping may
reverse or transpose acquisition while every block retains logical shape
`(region, line, sample)`.
"""
struct DetectorReadoutRegion
    origin::NTuple{2,Int}
    line_step::NTuple{2,Int}
    sample_step::NTuple{2,Int}

    function DetectorReadoutRegion(origin, line_step, sample_step)
        prepared_origin = _detector_coordinate_pair(origin, "origin")
        prepared_line_step = _detector_step_pair(line_step, "line step")
        prepared_sample_step = _detector_step_pair(sample_step, "sample step")
        prepared_line_step == (0, 0) &&
            throw(ArgumentError("detector readout line step must not be zero"))
        prepared_sample_step == (0, 0) &&
            throw(ArgumentError("detector readout sample step must not be zero"))
        return new(prepared_origin, prepared_line_step, prepared_sample_step)
    end
end

function _detector_coordinate_pair(value, description)
    length(value) == 2 || throw(
        ArgumentError("detector readout $description needs two coordinates"),
    )
    pair = (Int(value[1]), Int(value[2]))
    all(>=(0), pair) || throw(
        ArgumentError("detector readout $description must be nonnegative"),
    )
    return pair
end

function _detector_step_pair(value, description)
    length(value) == 2 || throw(
        ArgumentError("detector readout $description needs two coordinates"),
    )
    return (Int(value[1]), Int(value[2]))
end

function _checked_detector_coordinate(
    origin::Int,
    line::Int,
    line_step::Int,
    sample::Int,
    sample_step::Int,
)
    line_offset = Base.Checked.checked_mul(line, line_step)
    sample_offset = Base.Checked.checked_mul(sample, sample_step)
    return Base.Checked.checked_add(
        Base.Checked.checked_add(origin, line_offset),
        sample_offset,
    )
end

"""
Validated acquisition mapping for detector readout blocks.

Preparation proves that the declared regions cover every detector coordinate
exactly once and precomputes the coordinate of each acquisition-ordered value.
The repeated path therefore performs no coordinate arithmetic or mapping
discovery.
"""
struct DetectorReadoutPlan <: AbstractAlgorithmPlan
    detector_size::NTuple{2,Int}
    region_lines::Int
    block_lines::Int
    region_samples::Int
    regions::Vector{DetectorReadoutRegion}
    block_count::Int
    block_elements::Int
    detector_rows::Vector{Int}
    detector_columns::Vector{Int}
    arrival_ordinals::Matrix{Int}
end

function DetectorReadoutPlan(
    detector_size,
    region_lines::Integer,
    block_lines::Integer,
    region_samples::Integer,
    regions,
)
    prepared_detector_size = _detector_coordinate_pair(detector_size, "detector size")
    all(>(0), prepared_detector_size) ||
        throw(ArgumentError("detector readout extents must be positive"))
    prepared_region_lines = Int(region_lines)
    prepared_block_lines = Int(block_lines)
    prepared_region_samples = Int(region_samples)
    prepared_region_lines > 0 &&
        prepared_block_lines > 0 &&
        prepared_region_samples > 0 ||
        throw(ArgumentError("detector readout extents must be positive"))
    prepared_block_lines <= prepared_region_lines &&
        rem(prepared_region_lines, prepared_block_lines) == 0 || throw(
        ArgumentError("readout block lines must divide the lines in each region"),
    )
    prepared_regions = Vector{DetectorReadoutRegion}(regions)
    isempty(prepared_regions) &&
        throw(ArgumentError("at least one detector readout region is required"))

    block_count = div(prepared_region_lines, prepared_block_lines)
    block_elements = Base.Checked.checked_mul(
        length(prepared_regions),
        Base.Checked.checked_mul(prepared_block_lines, prepared_region_samples),
    )
    detector_elements = Base.Checked.checked_mul(prepared_detector_size...)
    readout_elements = Base.Checked.checked_mul(block_count, block_elements)
    readout_elements == detector_elements || throw(
        DimensionMismatch(
            "detector readout regions contain $readout_elements pixels; " *
            "detector contains $detector_elements",
        ),
    )

    detector_rows = Vector{Int}(undef, detector_elements)
    detector_columns = Vector{Int}(undef, detector_elements)
    arrival_ordinals = fill(-1, prepared_detector_size)
    acquisition_index = 1
    rows, columns = prepared_detector_size
    for ordinal in 0:(block_count - 1)
        acquisition_line = Base.Checked.checked_mul(ordinal, prepared_block_lines)
        for (region_index, region) in enumerate(prepared_regions)
            for local_line in 0:(prepared_block_lines - 1)
                line = Base.Checked.checked_add(acquisition_line, local_line)
                for sample in 0:(prepared_region_samples - 1)
                    row = _checked_detector_coordinate(
                        region.origin[1],
                        line,
                        region.line_step[1],
                        sample,
                        region.sample_step[1],
                    )
                    column = _checked_detector_coordinate(
                        region.origin[2],
                        line,
                        region.line_step[2],
                        sample,
                        region.sample_step[2],
                    )
                    0 <= row < rows && 0 <= column < columns || throw(
                        ArgumentError(
                            "detector readout region $(region_index - 1) exceeds " *
                            "the detector image",
                        ),
                    )
                    detector_row = row + 1
                    detector_column = column + 1
                    arrival_ordinals[detector_row, detector_column] == -1 || throw(
                        ArgumentError(
                            "detector coordinate ($row, $column) is mapped more than once",
                        ),
                    )
                    detector_rows[acquisition_index] = detector_row
                    detector_columns[acquisition_index] = detector_column
                    arrival_ordinals[detector_row, detector_column] = ordinal
                    acquisition_index += 1
                end
            end
        end
    end
    missing = findfirst(==(-1), arrival_ordinals)
    if !isnothing(missing)
        row = missing[1] - 1
        column = missing[2] - 1
        throw(ArgumentError("detector coordinate ($row, $column) is not mapped"))
    end
    return DetectorReadoutPlan(
        prepared_detector_size,
        prepared_region_lines,
        prepared_block_lines,
        prepared_region_samples,
        prepared_regions,
        block_count,
        block_elements,
        detector_rows,
        detector_columns,
        arrival_ordinals,
    )
end

"""Return the logical detector `(rows, columns)` extent."""
detector_readout_detector_size(plan::DetectorReadoutPlan) = plan.detector_size

"""Return the number of simultaneously delivered acquisition regions."""
detector_readout_region_count(plan::DetectorReadoutPlan) = length(plan.regions)

"""Return the number of blocks in one complete detector frame."""
detector_readout_block_count(plan::DetectorReadoutPlan) = plan.block_count

"""Return the logical `(region, line, sample)` extent of one block."""
detector_readout_block_shape(plan::DetectorReadoutPlan) =
    (length(plan.regions), plan.block_lines, plan.region_samples)

"""Return a detector coordinate's zero-based acquisition ordinal, if valid."""
function detector_readout_arrival_ordinal(
    plan::DetectorReadoutPlan,
    row::Integer,
    column::Integer,
)
    rows, columns = plan.detector_size
    0 <= row < rows && 0 <= column < columns || return nothing
    return plan.arrival_ordinals[Int(row) + 1, Int(column) + 1]
end

function _prepare_detector_readout(
    detector_rows,
    detector_columns,
    region_lines,
    block_lines,
    region_samples,
    origins,
    line_steps,
    sample_steps,
)
    length(origins) == length(line_steps) == length(sample_steps) || throw(
        ArgumentError("detector readout region geometry counts must match"),
    )
    regions = map(eachindex(origins)) do index
        DetectorReadoutRegion(origins[index], line_steps[index], sample_steps[index])
    end
    return DetectorReadoutPlan(
        (detector_rows, detector_columns),
        region_lines,
        block_lines,
        region_samples,
        regions,
    )
end

# DetectorReadoutPlan stores the map in acquisition order. Arrange it once
# in the device array's linear order so each work item reads its own pixel.
function _readout_detector_indices(plan::DetectorReadoutPlan, ::Type{T}=Int) where {T<:Integer}
    shape = detector_readout_block_shape(plan)
    linear = LinearIndices(shape)
    detector = LinearIndices(plan.detector_size)
    result = Vector{Vector{T}}(undef, plan.block_count)
    mapping = 1
    for ordinal in eachindex(result)
        indices = Vector{T}(undef, plan.block_elements)
        for region in 1:shape[1], line in 1:shape[2], sample in 1:shape[3]
            index = detector[plan.detector_rows[mapping], plan.detector_columns[mapping]]
            indices[linear[region, line, sample]] = index
            mapping += 1
        end
        result[ordinal] = indices
    end
    return result
end

function _pack_detector_readout(
    plan::DetectorReadoutPlan,
    detector::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(detector)
    size(detector) == plan.detector_size || throw(
        DimensionMismatch("detector plane shape is incompatible with the readout mapping"),
    )
    packed = Vector{T}(undef, length(plan.detector_rows))
    @inbounds for index in eachindex(packed)
        packed[index] = detector[
            plan.detector_rows[index],
            plan.detector_columns[index],
        ]
    end
    return packed
end

function _scatter_detector_readout_block!(
    detector::AbstractMatrix{T},
    plan::DetectorReadoutPlan,
    block::AbstractArray{T,3},
    ordinal::Int,
) where {T}
    Base.require_one_based_indexing(detector, block)
    size(detector) == plan.detector_size || throw(
        DimensionMismatch("detector workspace shape is incompatible"),
    )
    size(block) == detector_readout_block_shape(plan) || throw(
        DimensionMismatch("detector readout block shape is incompatible"),
    )
    0 <= ordinal < plan.block_count ||
        throw(BoundsError(Base.OneTo(plan.block_count), ordinal + 1))
    mapping_index = ordinal * plan.block_elements + 1
    @inbounds for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        detector[
            plan.detector_rows[mapping_index],
            plan.detector_columns[mapping_index],
        ] = block[region, line, sample]
        mapping_index += 1
    end
    return detector
end

"""Prepared U16-to-F32 calibration in detector acquisition order."""
struct DetectorReadoutCalibrationPlan <: AbstractAlgorithmPlan
    readout::DetectorReadoutPlan
    flat::Vector{Float32}
    background::Vector{Float32}
end

function DetectorReadoutCalibrationPlan(
    readout::DetectorReadoutPlan,
    flat::AbstractMatrix{Float32},
    background::AbstractMatrix{Float32},
)
    size(flat) == readout.detector_size ||
        throw(DimensionMismatch("flat calibration shape is incompatible"))
    size(background) == readout.detector_size ||
        throw(DimensionMismatch("background calibration shape is incompatible"))
    all(isfinite, flat) || throw(ArgumentError("flat calibration must be finite"))
    all(isfinite, background) ||
        throw(ArgumentError("background calibration must be finite"))
    return DetectorReadoutCalibrationPlan(
        readout,
        _pack_detector_readout(readout, flat),
        _pack_detector_readout(readout, background),
    )
end

function _replace_detector_readout_flat(plan::DetectorReadoutCalibrationPlan, values)
    return DetectorReadoutCalibrationPlan(
        plan.readout,
        values,
        _detector_plane(plan.readout, plan.background),
    )
end

function _replace_detector_readout_background(plan::DetectorReadoutCalibrationPlan, values)
    return DetectorReadoutCalibrationPlan(
        plan.readout,
        _detector_plane(plan.readout, plan.flat),
        values,
    )
end

function _detector_plane(
    plan::DetectorReadoutPlan,
    packed::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(packed)
    length(packed) == length(plan.detector_rows) || throw(
        DimensionMismatch("packed detector plane has an invalid extent"),
    )
    detector = Matrix{T}(undef, plan.detector_size)
    @inbounds for index in eachindex(packed)
        detector[plan.detector_rows[index], plan.detector_columns[index]] = packed[index]
    end
    return detector
end

"""Transactional discontinuity state for readout-block calibration."""
mutable struct DetectorReadoutCalibrationWorkspace
    pending_discontinuity::Bool
    candidate_pending_discontinuity::Bool
end

DetectorReadoutCalibrationWorkspace() =
    DetectorReadoutCalibrationWorkspace(false, false)

function commit!(workspace::DetectorReadoutCalibrationWorkspace)
    workspace.pending_discontinuity = workspace.candidate_pending_discontinuity
    return workspace
end

function discard!(workspace::DetectorReadoutCalibrationWorkspace)
    workspace.candidate_pending_discontinuity = workspace.pending_discontinuity
    return workspace
end

function reset!(workspace::DetectorReadoutCalibrationWorkspace)
    workspace.pending_discontinuity = true
    workspace.candidate_pending_discontinuity = true
    return workspace
end

function _plan_adopted!(workspace::DetectorReadoutCalibrationWorkspace)
    workspace.pending_discontinuity = true
    workspace.candidate_pending_discontinuity = true
    return workspace
end

"""Expected rejection of malformed detector-readout calibration metadata."""
@enum PixelCalibrationReadoutFailure::UInt8 begin
    PixelCalibrationReadoutMetadataMissing = 1
    PixelCalibrationReadoutOrdinalRange = 2
    PixelCalibrationReadoutTerminalMismatch = 3
end

@doc "A detector readout block lacks metadata needed for pixel calibration." PixelCalibrationReadoutMetadataMissing
@doc "A readout acquisition ordinal is outside the configured detector sequence." PixelCalibrationReadoutOrdinalRange
@doc "The readout terminal marker disagrees with the configured sequence boundary." PixelCalibrationReadoutTerminalMismatch

"""
Calibrate a `UInt16` detector readout block to `Float32` with flat/background
coefficients addressed through the configured detector-region map. Sample
metadata identifies the block's acquisition position.
"""
@algorithm PixelCalibrationReadoutU16F32 begin
    label = "pixel-calibration-readout-u16-f32"
    failure = PixelCalibrationReadoutFailure

    """
    Set detector geometry, region lines, block lines, samples per region, and
    per-region detector origin/line/sample steps. Optional initial flat and
    background vectors use the full detector's pixel indexing.
    """
    @configuration PixelCalibrationReadoutU16F32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
        initial_flat::Union{Nothing,Vector{Float32}} => construction
        initial_background::Union{Nothing,Vector{Float32}} => construction
    end

    ports(configuration) = (
        input(
            :raw,
            schema_element_type(RAW_PIXEL_READOUT_BLOCK_V1),
            (
                length(configuration.detector_origins),
                configuration.block_lines,
                configuration.region_samples,
            ),
            RAW_PIXEL_READOUT_BLOCK_V1;
            sample=:raw_sample,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        output(
            :calibrated,
            schema_element_type(CALIBRATED_PIXEL_READOUT_BLOCK_V1),
            (
                length(configuration.detector_origins),
                configuration.block_lines,
                configuration.region_samples,
            ),
            CALIBRATED_PIXEL_READOUT_BLOCK_V1;
            metadata_from=:raw,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        parameter(
            :flat,
            schema_element_type(FLAT_CALIBRATION_V1),
            configuration.detector_size,
            FLAT_CALIBRATION_V1;
            replace=(plan, values) -> _replace_detector_readout_flat(plan, values),
        ),
        parameter(
            :background,
            schema_element_type(BACKGROUND_CALIBRATION_V1),
            configuration.detector_size,
            BACKGROUND_CALIBRATION_V1;
            replace=(plan, values) -> _replace_detector_readout_background(plan, values),
        ),
    )

    prepare(configuration) = begin
        readout = _prepare_detector_readout(
            configuration.detector_size[1],
            configuration.detector_size[2],
            configuration.region_lines,
            configuration.block_lines,
            configuration.region_samples,
            configuration.detector_origins,
            configuration.detector_line_steps,
            configuration.detector_sample_steps,
        )
        flat = _row_major_calibration_plane(
            configuration.initial_flat,
            configuration.detector_size[1],
            configuration.detector_size[2],
            1.0f0,
        )
        background = _row_major_calibration_plane(
            configuration.initial_background,
            configuration.detector_size[1],
            configuration.detector_size[2],
            0.0f0,
        )
        (
            plan=DetectorReadoutCalibrationPlan(readout, flat, background),
            workspace=DetectorReadoutCalibrationWorkspace(),
        )
    end

    process!(plan, workspace) = begin
        isnothing(raw_sample) && return PixelCalibrationReadoutMetadataMissing
        ordinal = Int(raw_sample.offset)
        0 <= ordinal < plan.readout.block_count ||
            return PixelCalibrationReadoutOrdinalRange
        raw_sample.terminal == (ordinal + 1 == plan.readout.block_count) ||
            return PixelCalibrationReadoutTerminalMismatch
        Base.mightalias(calibrated, plan.flat) && throw(
            ArgumentError("calibrated output must not alias flat calibration"),
        )
        Base.mightalias(calibrated, plan.background) && throw(
            ArgumentError("calibrated output must not alias background calibration"),
        )

        calibration_index = ordinal * plan.readout.block_elements + 1
        @inbounds for region in axes(raw, 1), line in axes(raw, 2), sample in axes(raw, 3)
            calibrated[region, line, sample] = _calibrated_pixel(
                raw[region, line, sample],
                plan.flat[calibration_index],
                plan.background[calibration_index],
            )
            calibration_index += 1
        end
        discontinuity = raw_sample.discontinuity ||
            workspace.candidate_pending_discontinuity
        workspace.candidate_pending_discontinuity = false
        return AlgorithmProcessComplete(
            SampleMetadata(raw_sample; discontinuity=discontinuity),
        )
    end
end


"""Prepared progressive assembly of calibrated readout blocks."""
struct DetectorFrameAssemblerPlan <: AbstractAlgorithmPlan
    readout::DetectorReadoutPlan
end

mutable struct DetectorReadoutFrame
    active::Bool
    sequence::UInt64
    metadata::SampleMetadata
    expected_ordinal::Int
    pending_discontinuity::Bool
    current_discontinuity::Bool
end

DetectorReadoutFrame() = DetectorReadoutFrame(false, 0, _EMPTY_SAMPLE_METADATA, 0, false, false)

function _abandon_detector_readout!(frame::DetectorReadoutFrame)
    frame.active = false
    frame.sequence = 0
    frame.expected_ordinal = 0
    frame.pending_discontinuity = true
    frame.current_discontinuity = false
    return frame
end

function _complete_detector_readout!(frame::DetectorReadoutFrame)
    frame.active = false
    frame.sequence = 0
    frame.expected_ordinal = 0
    frame.pending_discontinuity = false
    frame.current_discontinuity = false
    return frame
end

function _accept_detector_readout!(
    frame::DetectorReadoutFrame,
    sample::SampleMetadata,
    block_count::Int,
)
    if sample.corrupted
        _abandon_detector_readout!(frame)
        return false
    end
    ordinal = Int(sample.offset)
    if !(0 <= ordinal < block_count)
        _abandon_detector_readout!(frame)
        return false
    end
    if ordinal == 0
        superseded = frame.active
        superseded && _abandon_detector_readout!(frame)
        inherited_discontinuity = frame.pending_discontinuity
        frame.active = true
        frame.sequence = sample.sequence
        frame.metadata=sample
        frame.expected_ordinal = 0
        frame.current_discontinuity =
            inherited_discontinuity || sample.discontinuity || superseded
        frame.pending_discontinuity = false
    else
        if !frame.active
            _abandon_detector_readout!(frame)
            return false
        end
        if !_same_row_block_metadata(frame.metadata,sample)
            _abandon_detector_readout!(frame)
            return false
        end
        if sample.discontinuity
            _abandon_detector_readout!(frame)
            return false
        end
    end
    if ordinal != frame.expected_ordinal
        _abandon_detector_readout!(frame)
        return false
    end
    terminal = ordinal + 1 == block_count
    if sample.terminal != terminal
        _abandon_detector_readout!(frame)
        return false
    end
    frame.expected_ordinal = ordinal + 1
    return true
end

"""Exclusively owned partial detector frame and acquisition identity."""
mutable struct DetectorFrameAssemblerWorkspace{D<:AbstractMatrix{Float32}}
    detector::D
    frame::DetectorReadoutFrame
end

function DetectorFrameAssemblerWorkspace(plan::DetectorFrameAssemblerPlan)
    return DetectorFrameAssemblerWorkspace(
        zeros(Float32, plan.readout.detector_size),
        DetectorReadoutFrame(),
    )
end

commit!(workspace::DetectorFrameAssemblerWorkspace) = workspace

function discard!(workspace::DetectorFrameAssemblerWorkspace)
    _abandon_detector_readout!(workspace.frame)
    return workspace
end

function reset!(workspace::DetectorFrameAssemblerWorkspace)
    _abandon_detector_readout!(workspace.frame)
    fill!(workspace.detector, 0.0f0)
    return workspace
end

_plan_adopted!(workspace::DetectorFrameAssemblerWorkspace) = reset!(workspace)

"""Expected rejection when a frame-assembly input has no sample identity."""
@enum DetectorFrameAssemblerFailure::UInt8 begin
    DetectorFrameAssemblerMetadataMissing = 1
end

@doc "A detector readout block lacks the sample metadata needed to locate it." DetectorFrameAssemblerMetadataMissing

"""
Assemble one detector image from successive Float32 readout blocks using
readout sample offsets and the configured detector-region mappings. The image
is published on completion of the detector frame and retains readout units.
"""
@algorithm DetectorFrameAssemblerF32 begin
    label = "detector-frame-assembler-f32"
    failure = DetectorFrameAssemblerFailure

    """
    Set detector rows/columns, region and block line counts, samples per region,
    and each region's detector origin, line step, and sample step. These dimensions
    and maps determine where a readout block lands in the assembled image.
    """
    @configuration DetectorFrameAssemblerF32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
    end

    ports(configuration) = (
        input(
            :readout_block,
            schema_element_type(CALIBRATED_PIXEL_READOUT_BLOCK_V1),
            (
                length(configuration.detector_origins),
                configuration.block_lines,
                configuration.region_samples,
            ),
            CALIBRATED_PIXEL_READOUT_BLOCK_V1;
            name="readout-block",
            sample=:readout_sample,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        output(
            :detector_frame,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.detector_size,
            CALIBRATED_PIXELS_V1;
            name="detector-frame",
            metadata_from=:readout_block,
            publication=:conditional,
        ),
    )

    prepare(configuration) = begin
        readout = _prepare_detector_readout(
            configuration.detector_size[1],
            configuration.detector_size[2],
            configuration.region_lines,
            configuration.block_lines,
            configuration.region_samples,
            configuration.detector_origins,
            configuration.detector_line_steps,
            configuration.detector_sample_steps,
        )
        plan = DetectorFrameAssemblerPlan(readout)
        (plan=plan, workspace=DetectorFrameAssemblerWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(readout_sample) && return DetectorFrameAssemblerMetadataMissing
        accepted = _accept_detector_readout!(
            workspace.frame,
            readout_sample,
            plan.readout.block_count,
        )
        accepted || return AlgorithmProcessDeferred()
        ordinal = Int(readout_sample.offset)
        _scatter_detector_readout_block!(
            workspace.detector,
            plan.readout,
            readout_block,
            ordinal,
        )
        readout_sample.terminal || return AlgorithmProcessDeferred()

        discontinuity = workspace.frame.current_discontinuity
        copyto!(detector_frame, workspace.detector)
        _complete_detector_readout!(workspace.frame)
        return AlgorithmProcessComplete(
            SampleMetadata(
                readout_sample;
                offset=0,
                terminal=true,
                discontinuity=discontinuity,
                corrupted=false,
            ),
        )
    end
end
