"""
Prepared calibrated per-pixel weighted center-of-gravity measurement for one
selected detector region.

The prepared dimensionless weight at each selected pixel multiplies its
calibrated value before flux and coordinate moments are accumulated. Weights
are finite and nonnegative, but are deliberately not normalized. Pixel
thresholding remains in calibrated-pixel units, while the flux threshold
applies to the weighted flux Σ(wᵢ·Iᵢ). A non-finite pixel makes the measurement
invalid even when its prepared weight is zero. CPU X/Y moments use Float32
fused multiply-add after the weighted pixel has rounded to Float32; flux uses
ordinary ordered addition.
"""
struct WeightedCenterOfGravityPlan <: AbstractAlgorithmPlan
    x_coordinates::Vector{Float32}
    y_coordinates::Vector{Float32}
    weights::Vector{Float32}
    reference_x::Float32
    reference_y::Float32
    pixel_threshold_enabled::Bool
    pixel_threshold::Float32
    flux_threshold::Float32
end

function WeightedCenterOfGravityPlan(
    x_coordinates,
    y_coordinates,
    weights,
    reference_centroid,
    pixel_threshold::Union{Nothing,Float32},
    flux_threshold::Float32,
)
    center_of_gravity = CenterOfGravityPlan(
        x_coordinates,
        y_coordinates,
        reference_centroid,
        pixel_threshold,
        flux_threshold,
    )
    length(weights) == length(center_of_gravity.x_coordinates) || throw(
        DimensionMismatch("center-of-gravity weight extent does not match pixel count"),
    )
    prepared_weights = collect(Float32, weights)
    all(weight -> isfinite(weight) && weight >= 0.0f0, prepared_weights) || throw(
        ArgumentError("center-of-gravity weights must be finite and nonnegative"),
    )
    return WeightedCenterOfGravityPlan(
        center_of_gravity.x_coordinates,
        center_of_gravity.y_coordinates,
        prepared_weights,
        center_of_gravity.reference_x,
        center_of_gravity.reference_y,
        center_of_gravity.pixel_threshold_enabled,
        center_of_gravity.pixel_threshold,
        center_of_gravity.flux_threshold,
    )
end

function _replace_weighted_center_of_gravity_coordinates(
    plan::WeightedCenterOfGravityPlan,
    values,
)
    return WeightedCenterOfGravityPlan(
        view(values, :, 1),
        view(values, :, 2),
        plan.weights,
        (plan.reference_x, plan.reference_y),
        plan.pixel_threshold_enabled ? plan.pixel_threshold : nothing,
        plan.flux_threshold,
    )
end

function _replace_weighted_center_of_gravity_weights(
    plan::WeightedCenterOfGravityPlan,
    values,
)
    return WeightedCenterOfGravityPlan(
        plan.x_coordinates,
        plan.y_coordinates,
        values,
        (plan.reference_x, plan.reference_y),
        plan.pixel_threshold_enabled ? plan.pixel_threshold : nothing,
        plan.flux_threshold,
    )
end

@inline function _weighted_center_of_gravity_aliases_plan(
    array,
    plan::WeightedCenterOfGravityPlan,
)
    return Base.mightalias(array, plan.x_coordinates) ||
        Base.mightalias(array, plan.y_coordinates) ||
        Base.mightalias(array, plan.weights)
end

property_schema(::Type{WeightedCenterOfGravityPlan}) =
    property_schema(CenterOfGravityPlan)

function property_values(plan::WeightedCenterOfGravityPlan)
    return NamedTuple{_COG_PROPERTY_NAMES}((
        plan.reference_x,
        plan.reference_y,
        plan.pixel_threshold_enabled,
        plan.pixel_threshold,
        plan.flux_threshold,
    ))
end

function replace_properties(plan::WeightedCenterOfGravityPlan, values::NamedTuple)
    reference = (
        getproperty(values, _COG_PROPERTY_NAMES[1]),
        getproperty(values, _COG_PROPERTY_NAMES[2]),
    )
    return WeightedCenterOfGravityPlan(
        plan.x_coordinates,
        plan.y_coordinates,
        plan.weights,
        reference,
        getproperty(values, _COG_PROPERTY_NAMES[3]) ?
            getproperty(values, _COG_PROPERTY_NAMES[4]) : nothing,
        getproperty(values, _COG_PROPERTY_NAMES[5]),
    )
end

"""
Compute a weighted center of gravity for one selected pixel vector. Publishes
reference-relative X/Y offsets in pixel-coordinate units, weighted flux in
input-pixel units (weights are dimensionless), and validity.
"""
@algorithm WeightedCenterOfGravityF32 begin
    label = "weighted-center-of-gravity-f32"
    properties = :runtime

    """
    Set pixel count, one initial `(X, Y)` coordinate pair and one initial weight
    per selected pixel. Reference and threshold values are runtime properties;
    coordinates and weights also have parameter ports.
    """
    @configuration WeightedCenterOfGravityF32Configuration begin
        pixel_count::Int => graph_rebuild
        initial_coordinates::Vector{Vector{Float32}} => construction
        initial_weights::Vector{Float32} => construction
    end

    ports(configuration) = (
        input(
            :pixels,
            schema_element_type(CENTER_OF_GRAVITY_PIXELS_V1),
            (configuration.pixel_count,),
            CENTER_OF_GRAVITY_PIXELS_V1,
        ),
        output(
            :centroid_offset,
            schema_element_type(CENTER_OF_GRAVITY_OFFSET_V1),
            (2,),
            CENTER_OF_GRAVITY_OFFSET_V1;
            name="centroid-offset",
            metadata_from=:pixels,
        ),
        output(
            :flux,
            schema_element_type(WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1),
            (1,),
            WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1;
            metadata_from=:pixels,
        ),
        output(
            :validity,
            schema_element_type(CENTER_OF_GRAVITY_VALIDITY_V1),
            (1,),
            CENTER_OF_GRAVITY_VALIDITY_V1;
            metadata_from=:pixels,
        ),
        parameter(
            :coordinates,
            schema_element_type(CENTER_OF_GRAVITY_COORDINATES_V1),
            (configuration.pixel_count, 2),
            CENTER_OF_GRAVITY_COORDINATES_V1;
            replace=(plan, values) ->
                _replace_weighted_center_of_gravity_coordinates(plan, values),
        ),
        parameter(
            :weights,
            schema_element_type(CENTER_OF_GRAVITY_WEIGHTS_V1),
            (configuration.pixel_count,),
            CENTER_OF_GRAVITY_WEIGHTS_V1;
            replace=(plan, values) ->
                _replace_weighted_center_of_gravity_weights(plan, values),
        ),
    )

    prepare(configuration) = begin
        configuration.pixel_count > 0 ||
            throw(ArgumentError("pixel_count must be positive"))
        x, y = _configured_center_of_gravity_coordinates(
            configuration.initial_coordinates,
            configuration.pixel_count,
        )
        WeightedCenterOfGravityPlan(
            x,
            y,
            configuration.initial_weights,
            (0.0f0, 0.0f0),
            nothing,
            0.0f0,
        )
    end

    process!(plan) = begin
        (Base.mightalias(centroid_offset, pixels) ||
            _weighted_center_of_gravity_aliases_plan(centroid_offset, plan)) && throw(
            ArgumentError("centroid-offset output must not alias input or prepared calibration"),
        )
        (Base.mightalias(flux, pixels) ||
            _weighted_center_of_gravity_aliases_plan(flux, plan)) &&
            throw(ArgumentError("flux output must not alias input or prepared calibration"))
        Base.mightalias(centroid_offset, flux) && throw(
            ArgumentError("centroid-offset and flux outputs must not alias"),
        )
        (Base.mightalias(validity, pixels) ||
            _weighted_center_of_gravity_aliases_plan(validity, plan)) && throw(
            ArgumentError("validity output must not alias input or prepared calibration"),
        )
        Base.mightalias(validity, centroid_offset) && throw(
            ArgumentError("validity and centroid-offset outputs must not alias"),
        )
        Base.mightalias(validity, flux) && throw(
            ArgumentError("validity and flux outputs must not alias"),
        )

        _weighted_centroid!(centroid_offset, flux, validity, pixels,
            plan.x_coordinates, plan.y_coordinates, plan.weights,
            plan.reference_x, plan.reference_y, plan.pixel_threshold_enabled,
            plan.pixel_threshold, plan.flux_threshold)
    end
end

function _weighted_centroid!(centroid_offset, flux, validity, pixels, x_coordinates,
        y_coordinates, weights, reference_x, reference_y, pixel_threshold_enabled,
        pixel_threshold, flux_threshold)
    x_moment = 0.0f0
    y_moment = 0.0f0
    measured_flux = 0.0f0
    for pixel in eachindex(
        pixels,
        x_coordinates,
        y_coordinates,
        weights,
    )
        value = pixels[pixel]
        selected = !pixel_threshold_enabled || value >= pixel_threshold
        if !isfinite(value) || selected
            weighted_value = value * weights[pixel]
            x_moment = fma(weighted_value, x_coordinates[pixel], x_moment)
            y_moment = fma(weighted_value, y_coordinates[pixel], y_moment)
            measured_flux += weighted_value
        end
    end

    x_offset, y_offset, measured_flux, valid = _finish_center_of_gravity(
        x_moment,
        y_moment,
        measured_flux,
        reference_x,
        reference_y,
        flux_threshold,
    )
    centroid_offset[1] = x_offset
    centroid_offset[2] = y_offset
    flux[1] = measured_flux
    validity[1] = valid
    return nothing
end
