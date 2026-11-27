"""
Prepared weighted row-block Shack-Hartmann sensing that publishes newly
immutable pair-interleaved slopes as a measurement block.

The weighted complete-image plan owns finite, nonnegative per-lenslet weights.
Detector-row order may differ from scientific subaperture order, so publication
waits for the greatest ready scientific prefix.
"""
struct ShackHartmannWeightedMeasurementBlockPlan{
    Sensing<:ShackHartmannWeightedCentroidPlan,
    Schedule<:AbstractVector{Int},
} <: AbstractAlgorithmPlan
    sensing::Sensing
    completion_schedule::Schedule
    block_rows::Int

    function ShackHartmannWeightedMeasurementBlockPlan(
        sensing::ShackHartmannWeightedCentroidPlan,
        block_rows::Int;
        pack_weights::Bool=true,
    )
        image_sensing = sensing.sensing
        image_rows = image_sensing.regions.image_size[1]
        0 < block_rows < image_rows || throw(
            ArgumentError(
                "row_block_rows must be positive and smaller than the image row count",
            ),
        )
        rem(image_rows, block_rows) == 0 || throw(
            ArgumentError("row_block_rows must divide the image row count"),
        )
        schedule = _shack_hartmann_measurement_block_completion_schedule(
            image_sensing,
        )
        sensing = ShackHartmannWeightedCentroidPlan(image_sensing, sensing; pack_weights)
        return new{typeof(sensing),typeof(schedule)}(
            sensing,
            schedule,
            block_rows,
        )
    end

    function ShackHartmannWeightedMeasurementBlockPlan(
        plan::ShackHartmannWeightedMeasurementBlockPlan;
        pack_weights::Bool,
    )
        calibration = plan.sensing
        sensing = ShackHartmannWeightedCentroidPlan(calibration.sensing, calibration; pack_weights)
        return new{typeof(sensing),typeof(plan.completion_schedule)}(
            sensing,
            plan.completion_schedule,
            plan.block_rows,
        )
    end
end

@inline _shack_hartmann_measurement_block_sensing(
    plan::ShackHartmannWeightedMeasurementBlockPlan,
) = plan.sensing.sensing

@inline _shack_hartmann_measurement_block_aliases_plan(
    array,
    plan::ShackHartmannWeightedMeasurementBlockPlan,
) = _shack_hartmann_weighted_centroid_aliases_plan(array, plan.sensing)

function _measure_shack_hartmann_measurement_block_range!(
    slopes::AbstractMatrix{Float32},
    flux::AbstractVector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannWeightedMeasurementBlockPlan,
    image::AbstractMatrix{Float32},
    subapertures,
)
    return _measure_shack_hartmann_weighted_centroid_range!(
        slopes,
        flux,
        validity,
        plan.sensing,
        image,
        subapertures,
    )
end

function ShackHartmannMeasurementBlockWorkspace(
    plan::ShackHartmannWeightedMeasurementBlockPlan,
)
    image_rows, image_columns = plan.sensing.sensing.regions.image_size
    subaperture_count = length(plan.sensing.sensing.regions.origins)
    return _prepared_shack_hartmann_measurement_block_workspace(
        zeros(Float32, image_rows, image_columns),
        zeros(Float32, subaperture_count, 2),
        zeros(Float32, subaperture_count),
        fill(false, subaperture_count),
        fill(false, subaperture_count),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_sensing(
    plan::ShackHartmannWeightedMeasurementBlockPlan,
    sensing::ShackHartmannWeightedCentroidPlan,
)
    return ShackHartmannWeightedMeasurementBlockPlan(
        sensing,
        plan.block_rows;
        pack_weights=!isnothing(sensing.packed_weights),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_origins(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_origins(plan.sensing, values),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_coordinates(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_coordinates(plan.sensing, values),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_reference_slopes(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_reference_slopes(plan.sensing, values),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_thresholds(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_thresholds(plan.sensing, values),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_active(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_active(plan.sensing, values),
    )
end

function _replace_shack_hartmann_weighted_measurement_block_weights(plan, values)
    return _replace_shack_hartmann_weighted_measurement_block_sensing(
        plan,
        _replace_shack_hartmann_weighted_centroid_weights(plan.sensing, values),
    )
end

"""
Convert weighted Shack-Hartmann row blocks into sparse centroid measurements,
zero-based dense-reconstructor columns, and valid count. Configured weights
modify the per-subaperture centroid accumulation.
"""
@algorithm ShackHartmannWeightedMeasurementBlockF32 begin
    label = "shack-hartmann-weighted-measurement-block-f32"
    failure = ShackHartmannMeasurementBlockFailure

    """
    Set image/row-block/subaperture geometry, optional origins, coordinate scale,
    thresholds, reference slopes and active mask, per-subaperture weight arrays,
    and versioned sparse output schemas.
    """
    @configuration ShackHartmannWeightedMeasurementBlockF32Configuration begin
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
        initial_weights::Vector{Vector{Float32}} => construction
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
                _replace_shack_hartmann_weighted_measurement_block_origins(plan, values),
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
                _replace_shack_hartmann_weighted_measurement_block_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_measurement_block_reference_slopes(
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
                _replace_shack_hartmann_weighted_measurement_block_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_measurement_block_active(plan, values),
        ),
        parameter(
            :weights,
            schema_element_type(SHACK_HARTMANN_WEIGHTS_V1),
            (
                configuration.subaperture_count,
                Base.Checked.checked_mul(
                    configuration.subaperture_size[1],
                    configuration.subaperture_size[2],
                ),
            ),
            SHACK_HARTMANN_WEIGHTS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_measurement_block_weights(plan, values),
        ),
    )

    prepare(configuration) = begin
        origins = _initial_shack_hartmann_origins(
            configuration.image_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        image_sensing = ShackHartmannImagePlan(
            configuration.image_size,
            configuration.subaperture_size,
            origins,
            configuration.coordinate_scale,
            configuration.pixel_threshold,
            configuration.flux_threshold;
            reference_slopes=configuration.reference_slopes,
            active=configuration.active,
        )
        weights = _configured_shack_hartmann_weights(
            configuration.initial_weights,
            length(image_sensing.regions.origins),
            length(image_sensing.x_coordinates),
        )
        sensing = ShackHartmannWeightedCentroidPlan(image_sensing, weights; pack_weights=false)
        plan = ShackHartmannWeightedMeasurementBlockPlan(
            sensing,
            configuration.row_block_rows;
            pack_weights=false,
        )
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

# Host row-block processing uses scalar ranges in the dense image workspace.
_prepare_host_plan(::Type{ShackHartmannWeightedMeasurementBlockF32}, plan::ShackHartmannWeightedMeasurementBlockPlan) =
    ShackHartmannWeightedMeasurementBlockPlan(plan; pack_weights=true)
