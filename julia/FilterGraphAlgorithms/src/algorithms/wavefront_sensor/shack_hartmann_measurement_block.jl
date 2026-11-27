"""
Prepared row-block Shack-Hartmann sensing that publishes newly immutable
pair-interleaved slopes as a measurement block.

Each block contains aligned `measurements` and zero-based reconstructor
`columns`; `count[1]` selects the valid prefix. Detector-row order may differ
from scientific subaperture order. The prepared completion schedule therefore
waits for the greatest ready scientific prefix before publishing a block.
"""
struct ShackHartmannMeasurementBlockPlan{
    Sensing<:ShackHartmannImagePlan,
    Schedule<:AbstractVector{Int},
} <: AbstractAlgorithmPlan
    sensing::Sensing
    completion_schedule::Schedule
    block_rows::Int

    function ShackHartmannMeasurementBlockPlan(
        sensing::ShackHartmannImagePlan,
        block_rows::Int,
    )
        image_rows = sensing.regions.image_size[1]
        0 < block_rows < image_rows || throw(
            ArgumentError(
                "row_block_rows must be positive and smaller than the image row count",
            ),
        )
        rem(image_rows, block_rows) == 0 || throw(
            ArgumentError("row_block_rows must divide the image row count"),
        )
        schedule = _shack_hartmann_measurement_block_completion_schedule(sensing)
        return new{typeof(sensing),typeof(schedule)}(
            sensing,
            schedule,
            block_rows,
        )
    end
end

function _shack_hartmann_measurement_block_completion_schedule(
    sensing::ShackHartmannImagePlan,
)
    region_rows = sensing.regions.region_size[1]
    schedule = collect(eachindex(sensing.regions.origins))
    sort!(schedule; by=subaperture -> (
        sensing.regions.origins[subaperture][1] + region_rows,
        subaperture,
    ))
    return schedule
end

@inline _shack_hartmann_measurement_block_sensing(
    plan::ShackHartmannMeasurementBlockPlan,
) = plan.sensing

@inline _shack_hartmann_measurement_block_aliases_plan(
    array,
    plan::ShackHartmannMeasurementBlockPlan,
) = _shack_hartmann_aliases_plan(array, plan.sensing)

function _measure_shack_hartmann_measurement_block_range!(
    slopes::AbstractMatrix{Float32},
    flux::AbstractVector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannMeasurementBlockPlan,
    image::AbstractMatrix{Float32},
    subapertures,
)
    return _measure_shack_hartmann_range!(
        slopes,
        flux,
        validity,
        plan.sensing,
        image,
        subapertures,
    )
end

"""Progressive detector and slope state for one Shack-Hartmann frame."""
mutable struct ShackHartmannMeasurementBlockWorkspace{
    T<:Union{Float32,Float64},
    Detector<:AbstractMatrix{T},
    Slopes<:AbstractMatrix{T},
    Flux<:AbstractVector{T},
    Validity<:AbstractVector{Bool},
    Ready<:AbstractVector{Bool},
}
    detector::Detector
    slopes::Slopes
    flux::Flux
    validity::Validity
    ready::Ready
    frame::RowBlockFrame
    schedule_cursor::Int
    published_subapertures::Int
    terminal_candidate::Bool
end

function _prepared_shack_hartmann_measurement_block_workspace(
    detector::Detector,
    slopes::Slopes,
    flux::Flux,
    validity::Validity,
    ready::Ready,
) where {
    T<:Union{Float32,Float64},
    Detector<:AbstractMatrix{T},
    Slopes<:AbstractMatrix{T},
    Flux<:AbstractVector{T},
    Validity<:AbstractVector{Bool},
    Ready<:AbstractVector{Bool},
}
    return ShackHartmannMeasurementBlockWorkspace{
        T,
        Detector,
        Slopes,
        Flux,
        Validity,
        Ready,
    }(
        detector,
        slopes,
        flux,
        validity,
        ready,
        RowBlockFrame(),
        0,
        0,
        false,
    )
end

function ShackHartmannMeasurementBlockWorkspace(
    plan::ShackHartmannMeasurementBlockPlan,
)
    image_rows, image_columns = plan.sensing.regions.image_size
    subaperture_count = length(plan.sensing.regions.origins)
    return _prepared_shack_hartmann_measurement_block_workspace(
        zeros(Float32, image_rows, image_columns),
        zeros(Float32, subaperture_count, 2),
        zeros(Float32, subaperture_count),
        fill(false, subaperture_count),
        fill(false, subaperture_count),
    )
end

function _clear_shack_hartmann_measurement_block!(
    workspace::ShackHartmannMeasurementBlockWorkspace,
)
    fill!(workspace.ready, false)
    workspace.schedule_cursor = 0
    workspace.published_subapertures = 0
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_shack_hartmann_measurement_block!(
    workspace::ShackHartmannMeasurementBlockWorkspace,
)
    _abandon_row_block_frame!(workspace.frame)
    return _clear_shack_hartmann_measurement_block!(workspace)
end

function commit!(workspace::ShackHartmannMeasurementBlockWorkspace)
    if workspace.terminal_candidate
        _complete_row_block_frame!(workspace.frame)
        _clear_shack_hartmann_measurement_block!(workspace)
    end
    return workspace
end

discard!(workspace::ShackHartmannMeasurementBlockWorkspace) =
    _abandon_shack_hartmann_measurement_block!(workspace)

function reset!(workspace::ShackHartmannMeasurementBlockWorkspace)
    fill!(workspace.detector, zero(eltype(workspace.detector)))
    fill!(workspace.slopes, zero(eltype(workspace.slopes)))
    fill!(workspace.flux, zero(eltype(workspace.flux)))
    fill!(workspace.validity, false)
    _reset_row_block_frame!(workspace.frame)
    return _clear_shack_hartmann_measurement_block!(workspace)
end

_plan_adopted!(workspace::ShackHartmannMeasurementBlockWorkspace) =
    _abandon_shack_hartmann_measurement_block!(workspace)

"""Expected rejection from Shack-Hartmann measurement-block production."""
@enum ShackHartmannMeasurementBlockFailure::UInt8 begin
    ShackHartmannMeasurementBlockMetadataMissing = 1
    ShackHartmannMeasurementBlockOffsetOverflow = 2
end

@doc "A Shack-Hartmann row block lacks the sample metadata needed to locate it." ShackHartmannMeasurementBlockMetadataMissing
@doc "A Shack-Hartmann row-block offset cannot be represented or advanced safely." ShackHartmannMeasurementBlockOffsetOverflow

function _replace_shack_hartmann_measurement_block_sensing(
    plan::ShackHartmannMeasurementBlockPlan,
    sensing::ShackHartmannImagePlan,
)
    return ShackHartmannMeasurementBlockPlan(sensing, plan.block_rows)
end

function _replace_shack_hartmann_measurement_block_origins(plan, values)
    return _replace_shack_hartmann_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_origins(plan.sensing, values),
    )
end

function _replace_shack_hartmann_measurement_block_coordinates(plan, values)
    return _replace_shack_hartmann_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_coordinates(plan.sensing, values),
    )
end

function _replace_shack_hartmann_measurement_block_reference_slopes(plan, values)
    return _replace_shack_hartmann_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_reference_slopes(plan.sensing, values),
    )
end

function _replace_shack_hartmann_measurement_block_thresholds(plan, values)
    return _replace_shack_hartmann_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_thresholds(plan.sensing, values),
    )
end

function _replace_shack_hartmann_measurement_block_active(plan, values)
    return _replace_shack_hartmann_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_active(plan.sensing, values),
    )
end

function _validate_shack_hartmann_measurement_block_storage(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::ShackHartmannMeasurementBlockWorkspace,
    plan,
    row_block::AbstractMatrix{Float32},
)
    Base.require_one_based_indexing(measurements, columns, count, row_block)
    sensing = _shack_hartmann_measurement_block_sensing(plan)
    capacity = 2 * length(sensing.regions.origins)
    length(measurements) == capacity || throw(
        DimensionMismatch("measurement block has an incompatible capacity"),
    )
    length(columns) == capacity || throw(
        DimensionMismatch("column block has an incompatible capacity"),
    )
    length(count) == 1 || throw(
        DimensionMismatch("measurement count must contain one value"),
    )
    image_rows, image_columns = sensing.regions.image_size
    size(row_block) == (plan.block_rows, image_columns) || throw(
        DimensionMismatch("row block has an incompatible detector shape"),
    )
    size(workspace.detector) == (image_rows, image_columns) || throw(
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
    _validate_shack_hartmann_measurement_block_output(
        measurements,
        row_block,
        workspace,
        plan,
    )
    _validate_shack_hartmann_measurement_block_output(
        columns,
        row_block,
        workspace,
        plan,
    )
    _validate_shack_hartmann_measurement_block_output(
        count,
        row_block,
        workspace,
        plan,
    )
    _shack_hartmann_measurement_block_aliases_workspace(row_block, workspace) &&
        throw(ArgumentError("row input must not alias Workspace storage"))
    return nothing
end

@inline function _validate_shack_hartmann_measurement_block_output(
    output,
    row_block,
    workspace::ShackHartmannMeasurementBlockWorkspace,
    plan,
)
    Base.mightalias(output, row_block) && throw(
        ArgumentError("measurement-block output must not alias the row input"),
    )
    _shack_hartmann_measurement_block_aliases_plan(output, plan) && throw(
        ArgumentError("measurement-block output must not alias the Plan"),
    )
    _shack_hartmann_measurement_block_aliases_workspace(output, workspace) &&
        throw(ArgumentError("measurement-block output must not alias Workspace storage"))
    return nothing
end

@inline function _shack_hartmann_measurement_block_aliases_workspace(
    array,
    workspace::ShackHartmannMeasurementBlockWorkspace,
)
    return Base.mightalias(array, workspace.detector) ||
        Base.mightalias(array, workspace.slopes) ||
        Base.mightalias(array, workspace.flux) ||
        Base.mightalias(array, workspace.validity) ||
        Base.mightalias(array, workspace.ready)
end

function _ready_shack_hartmann_measurement_prefix!(
    workspace::ShackHartmannMeasurementBlockWorkspace,
    plan,
    completed_rows::Int,
)
    sensing = _shack_hartmann_measurement_block_sensing(plan)
    origins = sensing.regions.origins
    region_rows = sensing.regions.region_size[1]
    while workspace.schedule_cursor < length(plan.completion_schedule)
        cursor = workspace.schedule_cursor + 1
        subaperture = plan.completion_schedule[cursor]
        completion_row = origins[subaperture][1] + region_rows
        completion_row <= completed_rows || break
        workspace.ready[subaperture] = true
        workspace.schedule_cursor = cursor
    end

    first_subaperture = workspace.published_subapertures + 1
    last_subaperture = workspace.published_subapertures
    while last_subaperture < length(workspace.ready) &&
          workspace.ready[last_subaperture + 1]
        last_subaperture += 1
    end
    return first_subaperture:last_subaperture
end

function _process_shack_hartmann_measurement_block!(
    measurements::AbstractVector{Float32},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::ShackHartmannMeasurementBlockWorkspace,
    plan,
    row_block::AbstractMatrix{Float32},
    sample::SampleMetadata,
)
    _validate_shack_hartmann_measurement_block_storage(
        measurements,
        columns,
        count,
        workspace,
        plan,
        row_block,
    )
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return ShackHartmannMeasurementBlockOffsetOverflow
    first_row = Int(sample.offset)
    sensing = _shack_hartmann_measurement_block_sensing(plan)
    image_rows, image_columns = sensing.regions.image_size
    accepted, completed_rows = _accept_row_block!(
        workspace.frame,
        sample,
        first_row,
        plan.block_rows,
        image_rows,
    )
    if !accepted
        _clear_shack_hartmann_measurement_block!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(first_row) && _clear_shack_hartmann_measurement_block!(workspace)

    for column in 1:image_columns, row in 1:plan.block_rows
        workspace.detector[first_row + row, column] = row_block[row, column]
    end
    ready = _ready_shack_hartmann_measurement_prefix!(
        workspace,
        plan,
        completed_rows,
    )
    block_count = 0
    if !isempty(ready)
        _measure_shack_hartmann_measurement_block_range!(
            workspace.slopes,
            workspace.flux,
            workspace.validity,
            plan,
            workspace.detector,
            ready,
        )
        @inbounds for subaperture in ready
            first_column = 2 * subaperture - 1
            block_count += 1
            measurements[block_count] = workspace.slopes[subaperture, 1]
            columns[block_count] = UInt32(first_column - 1)
            block_count += 1
            measurements[block_count] = workspace.slopes[subaperture, 2]
            columns[block_count] = UInt32(first_column)
        end
        workspace.published_subapertures = last(ready)
    end
    count[1] = UInt32(block_count)

    if sample.terminal
        if workspace.published_subapertures != length(sensing.regions.origins)
            _abandon_shack_hartmann_measurement_block!(workspace)
            return AlgorithmProcessDeferred()
        end
        workspace.terminal_candidate = true
    end
    discontinuity = sample.discontinuity ||
        (iszero(first_row) && workspace.frame.current_discontinuity)
    return AlgorithmProcessComplete(SampleMetadata(sample; discontinuity=discontinuity))
end

"""
Convert Shack-Hartmann row blocks to sparse centroid measurements, zero-based
dense-reconstructor columns, and valid count. A block is positioned by sample
metadata; centroid offsets are measured in scaled pixel coordinates.
"""
@algorithm ShackHartmannMeasurementBlockF32 begin
    label = "shack-hartmann-measurement-block-f32"
    failure = ShackHartmannMeasurementBlockFailure

    """
    Set image/row-block/subaperture geometry, centroids' origins and thresholds,
    reference slopes and active mask, and versioned sparse output schemas. The
    number of row blocks derives from image rows and block rows.
    """
    @configuration ShackHartmannMeasurementBlockF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
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
            (Base.Checked.checked_mul(configuration.subaperture_count, 2),),
            configuration.measurements_schema;
            metadata_from=:row_block,
            publication=:conditional,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
        ),
        output(
            :columns,
            UInt32,
            (Base.Checked.checked_mul(configuration.subaperture_count, 2),),
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
            :subaperture_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (configuration.subaperture_count, 2),
            REGION_ORIGINS_V1;
            name="subaperture-origins",
            replace=(plan, values) ->
                _replace_shack_hartmann_measurement_block_origins(plan, values),
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
                _replace_shack_hartmann_measurement_block_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_measurement_block_reference_slopes(plan, values),
        ),
        parameter(
            :thresholds,
            schema_element_type(SHACK_HARTMANN_THRESHOLDS_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_THRESHOLDS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_measurement_block_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_measurement_block_active(plan, values),
        ),
    )

    prepare(configuration) = begin
        origins = _initial_shack_hartmann_origins(
            configuration.image_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        sensing = ShackHartmannImagePlan(
            configuration.image_size,
            configuration.subaperture_size,
            origins,
            configuration.coordinate_scale,
            configuration.pixel_threshold,
            configuration.flux_threshold;
            reference_slopes=configuration.reference_slopes,
            active=configuration.active,
        )
        plan = ShackHartmannMeasurementBlockPlan(sensing, configuration.row_block_rows)
        (plan=plan, workspace=ShackHartmannMeasurementBlockWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(row_sample) && return ShackHartmannMeasurementBlockMetadataMissing
        _process_shack_hartmann_measurement_block!(
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
