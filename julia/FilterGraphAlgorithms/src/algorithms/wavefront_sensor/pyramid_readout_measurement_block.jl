"""
Detector-readout pyramid sensing that publishes newly ready normalized pixels.

The existing readout reconstructor owns the detector mapping, selected-pixel
completion groups, and previous-frame normalization gain. A zero one-row
reconstructor lets this declaration reuse those scientific rules while the
Graph's incremental dense reconstructor owns the actual MVM.
"""
struct PyramidReadoutMeasurementBlockPlan{
    Source<:PyramidReadoutReconstructionPlan,
} <: AbstractAlgorithmPlan
    source::Source
end

"""
Mutable scratch for [`PyramidReadoutMeasurementBlockF32`](@ref). Owns its
readout normalization workspace and terminal pixel, mean-intensity, and
reconstruction buffers; callers must not share a prepared workspace across
concurrent process calls.
"""
mutable struct PyramidReadoutMeasurementBlockWorkspace
    source::PyramidReadoutReconstructionWorkspace
    terminal_pixels::Matrix{Float32}
    terminal_mean::Vector{Float32}
    terminal_reconstructed::Vector{Float32}
end

function PyramidReadoutMeasurementBlockWorkspace(
    plan::PyramidReadoutMeasurementBlockPlan,
)
    selected = length(plan.source.normalization.selected_pixel_offsets)
    return PyramidReadoutMeasurementBlockWorkspace(
        PyramidReadoutReconstructionWorkspace(plan.source),
        zeros(Float32, PYRAMID_PUPIL_COUNT, selected),
        zeros(Float32, 1),
        zeros(Float32, 1),
    )
end

function commit!(workspace::PyramidReadoutMeasurementBlockWorkspace)
    commit!(workspace.source)
    return workspace
end

function discard!(workspace::PyramidReadoutMeasurementBlockWorkspace)
    discard!(workspace.source)
    return workspace
end

function reset!(workspace::PyramidReadoutMeasurementBlockWorkspace)
    reset!(workspace.source)
    return workspace
end

function _plan_adopted!(workspace::PyramidReadoutMeasurementBlockWorkspace)
    _abandon_pyramid_readout!(workspace.source)
    return workspace
end

function _validate_pyramid_readout_measurement_block_output(
    output,
    readout_block,
    workspace::PyramidReadoutMeasurementBlockWorkspace,
)
    Base.mightalias(output, readout_block) && throw(
        ArgumentError("pyramid measurement outputs must not alias input"),
    )
    _pyramid_readout_aliases_workspace(output, workspace.source) && throw(
        ArgumentError("pyramid measurement outputs must not alias Workspace"),
    )
    return nothing
end

function _validate_pyramid_readout_measurement_block(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::PyramidReadoutMeasurementBlockWorkspace,
    plan::PyramidReadoutMeasurementBlockPlan,
    readout_block::AbstractArray{Float32,3},
)
    Base.require_one_based_indexing(measurements, columns, count, readout_block)
    capacity = plan.source.normalization.output_pixel_count
    length(measurements) == capacity || throw(
        DimensionMismatch("pyramid measurement block has an invalid capacity"),
    )
    length(columns) == capacity || throw(
        DimensionMismatch("pyramid column block has an invalid capacity"),
    )
    length(count) == 1 || throw(
        DimensionMismatch("pyramid measurement count must have length one"),
    )
    size(readout_block) == detector_readout_block_shape(plan.source.readout) || throw(
        DimensionMismatch("pyramid readout block has an invalid shape"),
    )
    _validate_pyramid_readout_measurement_block_output(measurements, readout_block, workspace)
    _validate_pyramid_readout_measurement_block_output(columns, readout_block, workspace)
    _validate_pyramid_readout_measurement_block_output(count, readout_block, workspace)
    Base.mightalias(measurements, columns) && throw(
        ArgumentError("pyramid measurement outputs must not alias one another"),
    )
    Base.mightalias(measurements, count) && throw(
        ArgumentError("pyramid measurement outputs must not alias one another"),
    )
    Base.mightalias(columns, count) && throw(
        ArgumentError("pyramid measurement outputs must not alias one another"),
    )
    return nothing
end

function _process_pyramid_readout_measurement_block!(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::PyramidReadoutMeasurementBlockWorkspace,
    plan::PyramidReadoutMeasurementBlockPlan,
    readout_block::AbstractArray{Float32,3},
    sample::SampleMetadata,
)
    _validate_pyramid_readout_measurement_block(
        measurements,
        columns,
        count,
        workspace,
        plan,
        readout_block,
    )
    result = _process_pyramid_readout_reconstruction!(
        workspace.terminal_pixels,
        workspace.terminal_mean,
        workspace.terminal_reconstructed,
        workspace.source,
        plan.source,
        readout_block,
        sample,
    )
    result isa PyramidReadoutReconstructionFailure && return result
    workspace.source.frame.active || return result

    ordinal = Int(sample.offset)
    ready = plan.source.completion_groups[ordinal + 1]
    selected = length(plan.source.normalization.selected_pixel_offsets)
    @inbounds for (index, canonical_pixel) in enumerate(ready)
        pupil = div(canonical_pixel - 1, selected) + 1
        pixel = rem(canonical_pixel - 1, selected) + 1
        measurements[index] =
            workspace.source.candidate_reconstruction_pixels[pupil, pixel]
        columns[index] = UInt32(canonical_pixel - 1)
    end
    count[1] = UInt32(length(ready))
    discontinuity = sample.discontinuity ||
        (iszero(sample.offset) && workspace.source.frame.current_discontinuity)
    return AlgorithmProcessComplete(
        SampleMetadata(sample; discontinuity),
    )
end

"""
Decode detector readout blocks, normalize pyramid pupil pixels, and publish
sparse measurement values with zero-based columns and valid count for a
separate incremental dense reconstructor. The previous complete frame's gain
is used while the current mean is accumulated.
"""
@algorithm PyramidReadoutMeasurementBlockF32 begin
    label = "pyramid-readout-measurement-block-f32"
    failure = PyramidReadoutReconstructionFailure

    """
    Set detector-region readout mapping, four pupil rectangles, optional pupil
    mask, and versioned measurement/column/count schemas. Block and pupil
    dimensions determine each published sparse block's capacity.
    """
    @configuration PyramidReadoutMeasurementBlockF32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
        measurements_schema::String => graph_rebuild
        column_indices_schema::String => graph_rebuild
        measurement_count_schema::String => graph_rebuild
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
            metadata_from=:readout_block,
            publication=:conditional,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
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
            metadata_from=:readout_block,
            publication=:conditional,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        output(
            :count,
            UInt32,
            (1,),
            configuration.measurement_count_schema;
            metadata_from=:readout_block,
            publication=:conditional,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
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
        origins = _configured_pyramid_origins(configuration.pupil_origins)
        regions = RegionExtractionPlan(
            configuration.detector_size,
            origins,
            configuration.pupil_size,
        )
        mask = _configured_pyramid_mask(
            configuration.pupil_mask,
            configuration.pupil_size[1],
            configuration.pupil_size[2],
        )
        row_major_mask = Bool[
            mask[row, column] for row in axes(mask, 1) for column in axes(mask, 2)
        ]
        normalization = PyramidPixelPlan{Float32}(row_major_mask)
        dummy = DenseReconstructorPlan(
            Float32,
            1,
            normalization.output_pixel_count,
            nothing,
        )
        source = PyramidReadoutReconstructionPlan(
            regions,
            normalization,
            dummy,
            readout,
        )
        plan = PyramidReadoutMeasurementBlockPlan(source)
        (plan=plan, workspace=PyramidReadoutMeasurementBlockWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(readout_sample) && return PyramidReadoutMetadataMissing
        _process_pyramid_readout_measurement_block!(
            measurements,
            columns,
            count,
            workspace,
            plan,
            readout_block,
            readout_sample,
        )
    end
end
