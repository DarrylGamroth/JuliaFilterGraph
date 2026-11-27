const _COG_REFERENCE_X_PROPERTY = UInt32(0)
const _COG_REFERENCE_Y_PROPERTY = UInt32(1)
const _COG_PIXEL_THRESHOLD_ENABLED_PROPERTY = UInt32(2)
const _COG_PIXEL_THRESHOLD_PROPERTY = UInt32(3)
const _COG_FLUX_THRESHOLD_PROPERTY = UInt32(4)

const _COG_PROPERTY_NAMES = (
    Symbol("reference-centroid-x"),
    Symbol("reference-centroid-y"),
    Symbol("pixel-threshold-enabled"),
    Symbol("pixel-threshold"),
    Symbol("flux-threshold"),
)

"""
Prepared center-of-gravity measurement for one selected detector region.

X follows increasing detector column and Y follows increasing detector row.
Coordinates use X then Y selected-pixel order. An optional inclusive pixel
threshold excludes finite values below the threshold. Non-finite pixels remain
in the accumulation so the measurement becomes explicitly invalid rather than
silently treating them as zero.
"""
struct CenterOfGravityPlan <: AbstractAlgorithmPlan
    x_coordinates::Vector{Float32}
    y_coordinates::Vector{Float32}
    reference_x::Float32
    reference_y::Float32
    pixel_threshold_enabled::Bool
    pixel_threshold::Float32
    flux_threshold::Float32
end

function CenterOfGravityPlan(
    x_coordinates,
    y_coordinates,
    reference_centroid,
    pixel_threshold::Union{Nothing,Float32},
    flux_threshold::Float32,
)
    isempty(x_coordinates) &&
        throw(ArgumentError("selected region must contain pixels"))
    length(x_coordinates) == length(y_coordinates) || throw(
        DimensionMismatch("center-of-gravity coordinate extents do not match"),
    )
    prepared_x_coordinates = collect(Float32, x_coordinates)
    prepared_y_coordinates = collect(Float32, y_coordinates)
    (
        all(isfinite, prepared_x_coordinates) &&
        all(isfinite, prepared_y_coordinates)
    ) || throw(
        ArgumentError("center-of-gravity coordinates must be finite"),
    )
    length(reference_centroid) == 2 || throw(
        DimensionMismatch("reference centroid must contain X and Y"),
    )
    reference_x = Float32(reference_centroid[1])
    reference_y = Float32(reference_centroid[2])
    isfinite(reference_x) && isfinite(reference_y) ||
        throw(ArgumentError("reference centroid must be finite"))
    isnothing(pixel_threshold) || isfinite(pixel_threshold) || throw(
        ArgumentError("pixel threshold must be finite when supplied"),
    )
    isfinite(flux_threshold) && flux_threshold >= 0.0f0 || throw(
        ArgumentError("flux threshold must be finite and nonnegative"),
    )
    return CenterOfGravityPlan(
        prepared_x_coordinates,
        prepared_y_coordinates,
        reference_x,
        reference_y,
        !isnothing(pixel_threshold),
        something(pixel_threshold, 0.0f0),
        flux_threshold,
    )
end

function _configured_center_of_gravity_coordinates(
    values::AbstractVector{<:AbstractVector{T}},
    pixel_count::Int,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(values)
    length(values) == pixel_count || throw(
        DimensionMismatch("initial coordinate count does not match pixel_count"),
    )
    x = Vector{T}(undef, pixel_count)
    y = Vector{T}(undef, pixel_count)
    for pixel in eachindex(values)
        coordinate = values[pixel]
        Base.require_one_based_indexing(coordinate)
        length(coordinate) == 2 || throw(
            DimensionMismatch("each initial coordinate must contain X and Y"),
        )
        x[pixel] = coordinate[1]
        y[pixel] = coordinate[2]
    end
    return x, y
end

function _replace_center_of_gravity_coordinates(plan::CenterOfGravityPlan, values)
    return CenterOfGravityPlan(
        view(values, :, 1),
        view(values, :, 2),
        (plan.reference_x, plan.reference_y),
        plan.pixel_threshold_enabled ? plan.pixel_threshold : nothing,
        plan.flux_threshold,
    )
end

@inline function _finish_center_of_gravity(
    x_moment::Float32,
    y_moment::Float32,
    measured_flux::Float32,
    reference_x::Float32,
    reference_y::Float32,
    flux_threshold::Float32,
)
    valid = isfinite(measured_flux) &&
        measured_flux > 0.0f0 &&
        measured_flux >= flux_threshold
    x_offset = 0.0f0
    y_offset = 0.0f0
    if valid
        x_offset = x_moment / measured_flux - reference_x
        y_offset = y_moment / measured_flux - reference_y
        valid = isfinite(x_offset) && isfinite(y_offset)
        if !valid
            x_offset = 0.0f0
            y_offset = 0.0f0
        end
    end
    return x_offset, y_offset, measured_flux, valid
end

function property_schema(::Type{CenterOfGravityPlan})
    limit = floatmax(Float32)
    return (
        runtime_property(
            _COG_REFERENCE_X_PROPERTY,
            _COG_PROPERTY_NAMES[1],
            Float32;
            default=0.0f0,
            description="Reference centroid X coordinate",
            unit="px",
            constraint=PropertyRange(-limit, limit),
        ),
        runtime_property(
            _COG_REFERENCE_Y_PROPERTY,
            _COG_PROPERTY_NAMES[2],
            Float32;
            default=0.0f0,
            description="Reference centroid Y coordinate",
            unit="px",
            constraint=PropertyRange(-limit, limit),
        ),
        runtime_property(
            _COG_PIXEL_THRESHOLD_ENABLED_PROPERTY,
            _COG_PROPERTY_NAMES[3],
            Bool;
            default=false,
            description="Whether the inclusive calibrated-pixel threshold is active",
            unit="1",
        ),
        runtime_property(
            _COG_PIXEL_THRESHOLD_PROPERTY,
            _COG_PROPERTY_NAMES[4],
            Float32;
            default=0.0f0,
            description="Inclusive calibrated-pixel threshold",
            unit="PDE",
            constraint=PropertyRange(-limit, limit),
        ),
        runtime_property(
            _COG_FLUX_THRESHOLD_PROPERTY,
            _COG_PROPERTY_NAMES[5],
            Float32;
            default=0.0f0,
            description="Minimum positive selected-region flux",
            unit="PDE",
            constraint=PropertyRange(0.0f0, limit),
        ),
    )
end

function property_values(plan::CenterOfGravityPlan)
    return NamedTuple{_COG_PROPERTY_NAMES}((
        plan.reference_x,
        plan.reference_y,
        plan.pixel_threshold_enabled,
        plan.pixel_threshold,
        plan.flux_threshold,
    ))
end

function replace_properties(plan::CenterOfGravityPlan, values::NamedTuple)
    reference = (
        getproperty(values, _COG_PROPERTY_NAMES[1]),
        getproperty(values, _COG_PROPERTY_NAMES[2]),
    )
    return CenterOfGravityPlan(
        plan.x_coordinates,
        plan.y_coordinates,
        reference,
        getproperty(values, _COG_PROPERTY_NAMES[3]) ?
            getproperty(values, _COG_PROPERTY_NAMES[4]) : nothing,
        getproperty(values, _COG_PROPERTY_NAMES[5]),
    )
end

@inline _center_of_gravity_aliases_plan(array, plan::CenterOfGravityPlan) =
    Base.mightalias(array, plan.x_coordinates) || Base.mightalias(array, plan.y_coordinates)

"""
Float32 center-of-gravity measurement of one selected pixel vector. Publishes
X/Y centroid offsets relative to the configured reference (pixel-coordinate
units), summed flux (input-pixel units), and a validity flag. A non-finite or
insufficient-flux region is invalid instead of contributing a centroid.
"""
@algorithm CenterOfGravityF32 begin
    label = "center-of-gravity-f32"
    properties = :runtime

    """
    Set `pixel_count` and one initial `(X, Y)` coordinate pair per selected pixel.
    `initial_coordinates=nothing` reserves zero coordinates for startup replacement
    through the typed coordinates parameter port.
    The reference centroid and pixel/flux thresholds are runtime properties of the
    prepared plan; coordinates can also be replaced through its parameter port.
    """
    @configuration CenterOfGravityF32Configuration begin
        pixel_count::Int => graph_rebuild
        initial_coordinates::Union{Nothing,Vector{Vector{Float32}}} => construction
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
            schema_element_type(CENTER_OF_GRAVITY_FLUX_V1),
            (1,),
            CENTER_OF_GRAVITY_FLUX_V1;
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
            replace=(plan, values) -> _replace_center_of_gravity_coordinates(plan, values),
        ),
    )

    prepare(configuration) = begin
        configuration.pixel_count > 0 ||
            throw(ArgumentError("pixel_count must be positive"))
        x, y = isnothing(configuration.initial_coordinates) ?
            (zeros(Float32, configuration.pixel_count), zeros(Float32, configuration.pixel_count)) :
            _configured_center_of_gravity_coordinates(configuration.initial_coordinates, configuration.pixel_count)
        CenterOfGravityPlan(x, y, (0.0f0, 0.0f0), nothing, 0.0f0)
    end

    process!(plan) = begin
        (_center_of_gravity_aliases_plan(centroid_offset, plan) ||
            _center_of_gravity_aliases_plan(flux, plan) ||
            _center_of_gravity_aliases_plan(validity, plan)) && throw(
            ArgumentError("center-of-gravity output must not alias prepared coordinates"),
        )
        Base.mightalias(centroid_offset, pixels) && throw(
            ArgumentError("centroid-offset output must not alias input pixels"),
        )
        Base.mightalias(flux, pixels) &&
            throw(ArgumentError("flux output must not alias input pixels"))
        Base.mightalias(centroid_offset, flux) && throw(
            ArgumentError("centroid-offset and flux outputs must not alias"),
        )
        Base.mightalias(validity, pixels) && throw(
            ArgumentError("validity output must not alias input pixels"),
        )
        Base.mightalias(validity, centroid_offset) && throw(
            ArgumentError("validity and centroid-offset outputs must not alias"),
        )
        Base.mightalias(validity, flux) && throw(
            ArgumentError("validity and flux outputs must not alias"),
        )

        x_moment = 0.0f0
        y_moment = 0.0f0
        measured_flux = 0.0f0
        @inbounds for pixel in eachindex(pixels, plan.x_coordinates, plan.y_coordinates)
            value = pixels[pixel]
            selected = !plan.pixel_threshold_enabled || value >= plan.pixel_threshold
            if !isfinite(value) || selected
                x_moment += value * plan.x_coordinates[pixel]
                y_moment += value * plan.y_coordinates[pixel]
                measured_flux += value
            end
        end

        x_offset, y_offset, measured_flux, valid = _finish_center_of_gravity(
            x_moment,
            y_moment,
            measured_flux,
            plan.reference_x,
            plan.reference_y,
            plan.flux_threshold,
        )
        centroid_offset[1] = x_offset
        centroid_offset[2] = y_offset
        flux[1] = measured_flux
        validity[1] = valid
        nothing
    end
end
