"""
Prepared detector-readout Shack-Hartmann measurements for incremental dense
reconstruction.

Every readout block publishes the slopes whose last detector pixel arrived in
that block.  `measurements` and zero-based `columns` are aligned; `count[1]`
selects their valid prefix.  Scientific subaperture order defines the paired
X/Y MVM column identities, independently of detector acquisition order.
"""
struct ShackHartmannReadoutMeasurementBlockPlan <: AbstractAlgorithmPlan
    sensing::ShackHartmannImagePlan
    readout::DetectorReadoutPlan
    subaperture_completion_ordinals::Vector{Int}
    completion_groups::Vector{Vector{Int}}

    function ShackHartmannReadoutMeasurementBlockPlan(
        sensing::ShackHartmannImagePlan,
        readout::DetectorReadoutPlan,
    )
        sensing.regions.image_size == readout.detector_size || throw(
            DimensionMismatch(
                "detector readout and Shack-Hartmann image extents must match",
            ),
        )
        completion_ordinals, completion_groups =
            _shack_hartmann_readout_completion_groups(sensing, readout)
        return new(
            sensing,
            readout,
            completion_ordinals,
            completion_groups,
        )
    end
end

"""Progressive detector and measurement state for one readout frame."""
mutable struct ShackHartmannReadoutMeasurementBlockWorkspace
    detector::Matrix{Float32}
    slopes::Matrix{Float32}
    flux::Vector{Float32}
    validity::Vector{Bool}
    frame::DetectorReadoutFrame
    measured_subapertures::Int
    terminal_candidate::Bool
end

function ShackHartmannReadoutMeasurementBlockWorkspace(
    plan::ShackHartmannReadoutMeasurementBlockPlan,
)
    subaperture_count = length(plan.sensing.regions.origins)
    return ShackHartmannReadoutMeasurementBlockWorkspace(
        zeros(Float32, plan.readout.detector_size),
        zeros(Float32, subaperture_count, 2),
        zeros(Float32, subaperture_count),
        fill(false, subaperture_count),
        DetectorReadoutFrame(),
        0,
        false,
    )
end

function _clear_shack_hartmann_readout_measurement_block!(
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
)
    workspace.measured_subapertures = 0
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_shack_hartmann_readout_measurement_block!(
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
)
    _abandon_detector_readout!(workspace.frame)
    return _clear_shack_hartmann_readout_measurement_block!(workspace)
end

function commit!(workspace::ShackHartmannReadoutMeasurementBlockWorkspace)
    if workspace.terminal_candidate
        _complete_detector_readout!(workspace.frame)
        _clear_shack_hartmann_readout_measurement_block!(workspace)
    end
    return workspace
end

discard!(workspace::ShackHartmannReadoutMeasurementBlockWorkspace) =
    _abandon_shack_hartmann_readout_measurement_block!(workspace)

function reset!(workspace::ShackHartmannReadoutMeasurementBlockWorkspace)
    fill!(workspace.detector, 0.0f0)
    fill!(workspace.slopes, 0.0f0)
    fill!(workspace.flux, 0.0f0)
    fill!(workspace.validity, false)
    return _abandon_shack_hartmann_readout_measurement_block!(workspace)
end

_plan_adopted!(workspace::ShackHartmannReadoutMeasurementBlockWorkspace) =
    _abandon_shack_hartmann_readout_measurement_block!(workspace)

"""Expected rejection from detector-readout Shack-Hartmann measurements."""
@enum ShackHartmannReadoutMeasurementBlockFailure::UInt8 begin
    ShackHartmannReadoutMeasurementBlockMetadataMissing = 1
    ShackHartmannReadoutMeasurementBlockOffsetOverflow = 2
end

@doc "A Shack-Hartmann detector readout block lacks required sample metadata." ShackHartmannReadoutMeasurementBlockMetadataMissing
@doc "A Shack-Hartmann readout offset cannot be represented or advanced safely." ShackHartmannReadoutMeasurementBlockOffsetOverflow

function _replace_shack_hartmann_readout_measurement_block_sensing(
    plan::ShackHartmannReadoutMeasurementBlockPlan,
    sensing::ShackHartmannImagePlan,
)
    return ShackHartmannReadoutMeasurementBlockPlan(sensing, plan.readout)
end

function _replace_shack_hartmann_readout_measurement_block_origins(plan, values)
    return _replace_shack_hartmann_readout_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_origins(plan.sensing, values),
    )
end

function _replace_shack_hartmann_readout_measurement_block_coordinates(plan, values)
    return _replace_shack_hartmann_readout_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_coordinates(plan.sensing, values),
    )
end

function _replace_shack_hartmann_readout_measurement_block_reference_slopes(plan, values)
    return _replace_shack_hartmann_readout_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_reference_slopes(plan.sensing, values),
    )
end

function _replace_shack_hartmann_readout_measurement_block_thresholds(plan, values)
    return _replace_shack_hartmann_readout_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_thresholds(plan.sensing, values),
    )
end

function _replace_shack_hartmann_readout_measurement_block_active(plan, values)
    return _replace_shack_hartmann_readout_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_active(plan.sensing, values),
    )
end

@inline function _shack_hartmann_readout_measurement_block_aliases_workspace(
    array,
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
)
    return Base.mightalias(array, workspace.detector) ||
        Base.mightalias(array, workspace.slopes) ||
        Base.mightalias(array, workspace.flux) ||
        Base.mightalias(array, workspace.validity)
end

@inline function _validate_shack_hartmann_readout_measurement_block_output(
    output,
    readout_block,
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
    plan::ShackHartmannReadoutMeasurementBlockPlan,
)
    Base.mightalias(output, readout_block) && throw(
        ArgumentError("measurement-block output must not alias the readout input"),
    )
    _shack_hartmann_aliases_plan(output, plan.sensing) && throw(
        ArgumentError("measurement-block output must not alias sensing calibration"),
    )
    _shack_hartmann_readout_measurement_block_aliases_workspace(output, workspace) &&
        throw(ArgumentError("measurement-block output must not alias Workspace storage"))
    return nothing
end

function _validate_shack_hartmann_readout_measurement_block_storage(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
    plan::ShackHartmannReadoutMeasurementBlockPlan,
    readout_block::AbstractArray{Float32,3},
)
    Base.require_one_based_indexing(measurements, columns, count, readout_block)
    capacity = 2 * length(plan.sensing.regions.origins)
    length(measurements) == capacity || throw(
        DimensionMismatch("measurement block has an incompatible capacity"),
    )
    length(columns) == capacity || throw(
        DimensionMismatch("column block has an incompatible capacity"),
    )
    length(count) == 1 || throw(
        DimensionMismatch("measurement count must contain one value"),
    )
    size(readout_block) == detector_readout_block_shape(plan.readout) || throw(
        DimensionMismatch("readout block has an incompatible detector shape"),
    )
    size(workspace.detector) == plan.readout.detector_size || throw(
        DimensionMismatch("measurement-block Workspace has an invalid detector"),
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
    _validate_shack_hartmann_readout_measurement_block_output(
        measurements,
        readout_block,
        workspace,
        plan,
    )
    _validate_shack_hartmann_readout_measurement_block_output(
        columns,
        readout_block,
        workspace,
        plan,
    )
    _validate_shack_hartmann_readout_measurement_block_output(
        count,
        readout_block,
        workspace,
        plan,
    )
    _shack_hartmann_readout_measurement_block_aliases_workspace(
        readout_block,
        workspace,
    ) && throw(ArgumentError("readout input must not alias Workspace storage"))
    return nothing
end

function _process_shack_hartmann_readout_measurement_block!(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::ShackHartmannReadoutMeasurementBlockWorkspace,
    plan::ShackHartmannReadoutMeasurementBlockPlan,
    readout_block::AbstractArray{Float32,3},
    sample::SampleMetadata,
)
    _validate_shack_hartmann_readout_measurement_block_storage(
        measurements,
        columns,
        count,
        workspace,
        plan,
        readout_block,
    )
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return ShackHartmannReadoutMeasurementBlockOffsetOverflow
    ordinal = Int(sample.offset)
    accepted = _accept_detector_readout!(
        workspace.frame,
        sample,
        plan.readout.block_count,
    )
    if !accepted
        _clear_shack_hartmann_readout_measurement_block!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(ordinal) && _clear_shack_hartmann_readout_measurement_block!(workspace)

    _scatter_detector_readout_block!(
        workspace.detector,
        plan.readout,
        readout_block,
        ordinal,
    )
    block_count = 0
    @inbounds for subaperture in plan.completion_groups[ordinal + 1]
        _measure_shack_hartmann_range!(
            workspace.slopes,
            workspace.flux,
            workspace.validity,
            plan.sensing,
            workspace.detector,
            subaperture:subaperture,
        )
        x_column = 2 * subaperture - 1
        block_count += 1
        measurements[block_count] = workspace.slopes[subaperture, 1]
        columns[block_count] = UInt32(x_column - 1)
        block_count += 1
        measurements[block_count] = workspace.slopes[subaperture, 2]
        columns[block_count] = UInt32(x_column)
        workspace.measured_subapertures += 1
    end
    count[1] = UInt32(block_count)

    if sample.terminal
        workspace.measured_subapertures == length(plan.sensing.regions.origins) ||
            error("prepared readout readiness did not complete every subaperture")
        workspace.terminal_candidate = true
    end
    discontinuity = sample.discontinuity ||
        (iszero(ordinal) && workspace.frame.current_discontinuity)
    return AlgorithmProcessComplete(SampleMetadata(sample; discontinuity=discontinuity))
end

"""
Convert Shack-Hartmann detector readout blocks into sparse centroid
measurements with zero-based columns and valid count. Detector-region mapping
and sample offset locate each readout block.
"""
@algorithm ShackHartmannReadoutMeasurementBlockF32 begin
    label = "shwfs-readout-measurement-block-f32"
    failure = ShackHartmannReadoutMeasurementBlockFailure

    """
    Set detector-region and subaperture geometry, optional origins, coordinate
    scale, thresholds, reference slopes and active mask, plus versioned sparse
    measurement/column/count schemas.
    """
    @configuration ShackHartmannReadoutMeasurementBlockF32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
        subaperture_size::NTuple{2,Int} => graph_rebuild
        subaperture_count::Int => graph_rebuild
        initial_subaperture_origins::Union{Nothing,Vector{Vector{Int}}} => construction
        coordinate_scale::Float32 => construction
        pixel_threshold::Float32 => construction
        flux_threshold::Float32 => construction
        reference_slopes::Union{Nothing,Vector{Float32}} => construction
        active::Union{Nothing,Vector{Bool}} => construction
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
            (Base.Checked.checked_mul(configuration.subaperture_count, 2),),
            configuration.measurements_schema;
            metadata_from=:readout_block,
            publication=:conditional,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        output(
            :columns,
            UInt32,
            (Base.Checked.checked_mul(configuration.subaperture_count, 2),),
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
        parameter(
            :subaperture_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (configuration.subaperture_count, 2),
            REGION_ORIGINS_V1;
            name="subaperture-origins",
            replace=(plan, values) ->
                _replace_shack_hartmann_readout_measurement_block_origins(plan, values),
        ),
        parameter(
            :coordinates,
            schema_element_type(SHACK_HARTMANN_COORDINATES_V1),
            (
                Base.Checked.checked_mul(
                    configuration.subaperture_size[1],
                    configuration.subaperture_size[2],
                ),
                2,
            ),
            SHACK_HARTMANN_COORDINATES_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_readout_measurement_block_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_readout_measurement_block_reference_slopes(
                    plan,
                    values,
                ),
        ),
        parameter(
            :thresholds,
            schema_element_type(SHACK_HARTMANN_THRESHOLDS_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_THRESHOLDS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_readout_measurement_block_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_readout_measurement_block_active(plan, values),
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
        origins = _initial_shack_hartmann_origins(
            configuration.detector_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        sensing = ShackHartmannImagePlan(
            configuration.detector_size,
            configuration.subaperture_size,
            origins,
            configuration.coordinate_scale,
            configuration.pixel_threshold,
            configuration.flux_threshold;
            reference_slopes=configuration.reference_slopes,
            active=configuration.active,
        )
        plan = ShackHartmannReadoutMeasurementBlockPlan(sensing, readout)
        (plan=plan, workspace=ShackHartmannReadoutMeasurementBlockWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(readout_sample) &&
            return ShackHartmannReadoutMeasurementBlockMetadataMissing
        _process_shack_hartmann_readout_measurement_block!(
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
