"""
Prepared detector-image geometry and centroid calibration for a Shack-Hartmann
wavefront sensor.

Each rectangular subaperture is measured directly from borrowed image rows; no
region slab is materialized. Coordinates are stored in row-major pixel order,
with positive X following detector columns and positive Y following detector
rows. Pixel thresholds are absolute, inclusive calibrated-pixel thresholds.
The published slope is the flux-normalized coordinate moment minus the
prepared reference slope, so coordinate scale and reference slope must already
use the requested output units. The scalar construction scale is positive;
signed or anisotropic calibrations use the coordinate and reference parameter
ports directly.

The output has shape `(subaperture, X/Y)`. A non-finite selected pixel or
insufficient selected flux produces positive zero slopes and `false` validity.
An inactive but otherwise valid subaperture also produces positive zero slopes,
while its flux and validity continue to describe the numerical measurement.
This operation is stateless: it has no reset, commit, or discard transition.
"""
struct ShackHartmannImagePlan <: AbstractAlgorithmPlan
    regions::RegionExtractionPlan
    x_coordinates::Vector{Float32}
    y_coordinates::Vector{Float32}
    reference_slopes::Vector{Float32}
    pixel_thresholds::Vector{Float32}
    flux_thresholds::Vector{Float32}
    active::Vector{Bool}
    # The SIMD loop walks detector rows within each column.
    _column_coordinates::NTuple{2,Vector{Float32}}

    function ShackHartmannImagePlan(
        regions::RegionExtractionPlan,
        x_coordinates::Vector{Float32},
        y_coordinates::Vector{Float32},
        reference_slopes::Vector{Float32},
        pixel_thresholds::Vector{Float32},
        flux_thresholds::Vector{Float32},
        active::Vector{Bool},
    )
        subaperture_count = length(regions.origins)
        subaperture_count > 0 ||
            throw(ArgumentError("at least one subaperture is required"))
        pixel_count = Base.Checked.checked_mul(regions.region_size...)
        pixel_count > 0 ||
            throw(ArgumentError("each subaperture must contain pixels"))
        length(x_coordinates) == pixel_count &&
            length(y_coordinates) == pixel_count || throw(
            DimensionMismatch(
                "coordinate extent must match one rectangular subaperture",
            ),
        )
        length(reference_slopes) ==
            Base.Checked.checked_mul(subaperture_count, 2) || throw(
            DimensionMismatch(
                "reference slopes must contain one X/Y pair per subaperture",
            ),
        )
        length(pixel_thresholds) == subaperture_count &&
            length(flux_thresholds) == subaperture_count &&
            length(active) == subaperture_count || throw(
            DimensionMismatch(
                "threshold and active extents must match subapertures",
            ),
        )
        all(isfinite, x_coordinates) && all(isfinite, y_coordinates) || throw(
            ArgumentError("Shack-Hartmann coordinates must be finite"),
        )
        all(isfinite, reference_slopes) || throw(
            ArgumentError("Shack-Hartmann reference slopes must be finite"),
        )
        all(value -> isfinite(value) && value >= 0.0f0, pixel_thresholds) ||
            throw(
                ArgumentError(
                    "Shack-Hartmann pixel thresholds must be finite and nonnegative",
                ),
            )
        all(value -> isfinite(value) && value >= 0.0f0, flux_thresholds) ||
            throw(
                ArgumentError(
                    "Shack-Hartmann flux thresholds must be finite and nonnegative",
                ),
            )
        rows, columns = regions.region_size
        column_coordinates = if rows >= 16 && subaperture_count > 1
            x = Vector{Float32}(undef, pixel_count)
            y = similar(x)
            for column in 1:columns, row in 1:rows
                source = (row - 1) * columns + column
                destination = (column - 1) * rows + row
                x[destination] = x_coordinates[source]
                y[destination] = y_coordinates[source]
            end
            (x, y)
        else
            (x_coordinates, y_coordinates)
        end
        return new(
            regions,
            x_coordinates,
            y_coordinates,
            reference_slopes,
            pixel_thresholds,
            flux_thresholds,
            active,
            column_coordinates,
        )
    end
end

function ShackHartmannImagePlan(
    image_size::NTuple{2,Int},
    subaperture_size::NTuple{2,Int},
    origins,
    coordinate_scale::Float32,
    pixel_threshold::Float32,
    flux_threshold::Float32;
    reference_slopes=nothing,
    active=nothing,
)
    isfinite(coordinate_scale) && coordinate_scale > 0.0f0 || throw(
        ArgumentError("coordinate scale must be finite and positive"),
    )
    regions = RegionExtractionPlan(image_size, origins, subaperture_size)
    subaperture_rows, subaperture_columns = subaperture_size
    pixel_count = Base.Checked.checked_mul(
        subaperture_rows,
        subaperture_columns,
    )
    x_coordinates = Vector{Float32}(undef, pixel_count)
    y_coordinates = Vector{Float32}(undef, pixel_count)
    x_center = (Float32(subaperture_columns) - 1.0f0) * 0.5f0
    y_center = (Float32(subaperture_rows) - 1.0f0) * 0.5f0
    pixel = 1
    for row in 0:(subaperture_rows - 1)
        for column in 0:(subaperture_columns - 1)
            x_coordinates[pixel] =
                (Float32(column) - x_center) * coordinate_scale
            y_coordinates[pixel] =
                (Float32(row) - y_center) * coordinate_scale
            pixel += 1
        end
    end

    subaperture_count = length(regions.origins)
    references = isnothing(reference_slopes) ?
        zeros(Float32, Base.Checked.checked_mul(subaperture_count, 2)) :
        Vector{Float32}(reference_slopes)
    selected = isnothing(active) ?
        fill(true, subaperture_count) : Vector{Bool}(active)
    return ShackHartmannImagePlan(
        regions,
        x_coordinates,
        y_coordinates,
        references,
        fill(pixel_threshold, subaperture_count),
        fill(flux_threshold, subaperture_count),
        selected,
    )
end

function _initial_shack_hartmann_origins(
    image_size::NTuple{2,Int},
    subaperture_size::NTuple{2,Int},
    subaperture_count::Int,
    explicit,
)
    subaperture_count > 0 ||
        throw(ArgumentError("subaperture_count must be positive"))
    if !isnothing(explicit)
        length(explicit) == subaperture_count || throw(
            DimensionMismatch(
                "initial subaperture origin count does not match subaperture_count",
            ),
        )
        origins = Vector{NTuple{2,Int}}(undef, subaperture_count)
        for subaperture in eachindex(explicit)
            origin = explicit[subaperture]
            length(origin) == 2 || throw(
                DimensionMismatch(
                    "each initial subaperture origin must contain row and column",
                ),
            )
            origins[subaperture] = (Int(origin[1]), Int(origin[2]))
        end
        return origins
    end

    all(>(0), image_size) && all(>(0), subaperture_size) || throw(
        ArgumentError("image and subaperture extents must be positive"),
    )
    grid_rows = div(image_size[1], subaperture_size[1])
    grid_columns = div(image_size[2], subaperture_size[2])
    capacity = Base.Checked.checked_mul(grid_rows, grid_columns)
    subaperture_count <= capacity || throw(
        ArgumentError("regular initial subaperture grid does not fit the image"),
    )
    origins = Vector{NTuple{2,Int}}(undef, subaperture_count)
    for subaperture in 0:(subaperture_count - 1)
        origins[subaperture + 1] = (
            div(subaperture, grid_columns) * subaperture_size[1],
            rem(subaperture, grid_columns) * subaperture_size[2],
        )
    end
    return origins
end

function _replace_shack_hartmann_origins(
    plan::ShackHartmannImagePlan,
    values,
)
    origins = Vector{NTuple{2,Int}}(undef, length(plan.regions.origins))
    for subaperture in eachindex(origins)
        origins[subaperture] = (
            Int(values[subaperture, 1]),
            Int(values[subaperture, 2]),
        )
    end
    regions = RegionExtractionPlan(
        plan.regions.image_size,
        origins,
        plan.regions.region_size,
    )
    return ShackHartmannImagePlan(
        regions,
        plan.x_coordinates,
        plan.y_coordinates,
        plan.reference_slopes,
        plan.pixel_thresholds,
        plan.flux_thresholds,
        plan.active,
    )
end

function _replace_shack_hartmann_coordinates(
    plan::ShackHartmannImagePlan,
    values,
)
    pixel_count = length(plan.x_coordinates)
    x_coordinates = Vector{Float32}(undef, pixel_count)
    y_coordinates = Vector{Float32}(undef, pixel_count)
    for pixel in 1:pixel_count
        x_coordinates[pixel] = values[pixel, 1]
        y_coordinates[pixel] = values[pixel, 2]
    end
    return ShackHartmannImagePlan(
        plan.regions,
        x_coordinates,
        y_coordinates,
        plan.reference_slopes,
        plan.pixel_thresholds,
        plan.flux_thresholds,
        plan.active,
    )
end

function _replace_shack_hartmann_reference_slopes(
    plan::ShackHartmannImagePlan,
    values,
)
    subaperture_count = length(plan.regions.origins)
    references = Vector{Float32}(undef, 2 * subaperture_count)
    for subaperture in 1:subaperture_count
        reference = 2 * subaperture - 1
        references[reference] = values[subaperture, 1]
        references[reference + 1] = values[subaperture, 2]
    end
    return ShackHartmannImagePlan(
        plan.regions,
        plan.x_coordinates,
        plan.y_coordinates,
        references,
        plan.pixel_thresholds,
        plan.flux_thresholds,
        plan.active,
    )
end

function _replace_shack_hartmann_thresholds(
    plan::ShackHartmannImagePlan,
    values,
)
    subaperture_count = length(plan.regions.origins)
    pixel_thresholds = Vector{Float32}(undef, subaperture_count)
    flux_thresholds = Vector{Float32}(undef, subaperture_count)
    for subaperture in 1:subaperture_count
        pixel_thresholds[subaperture] = values[subaperture, 1]
        flux_thresholds[subaperture] = values[subaperture, 2]
    end
    return ShackHartmannImagePlan(
        plan.regions,
        plan.x_coordinates,
        plan.y_coordinates,
        plan.reference_slopes,
        pixel_thresholds,
        flux_thresholds,
        plan.active,
    )
end

function _replace_shack_hartmann_active(
    plan::ShackHartmannImagePlan,
    values,
)
    return ShackHartmannImagePlan(
        plan.regions,
        plan.x_coordinates,
        plan.y_coordinates,
        plan.reference_slopes,
        plan.pixel_thresholds,
        plan.flux_thresholds,
        Vector{Bool}(values),
    )
end

@inline function _shack_hartmann_aliases_plan(array, plan::ShackHartmannImagePlan)
    return Base.mightalias(array, plan.x_coordinates) ||
        Base.mightalias(array, plan.y_coordinates) ||
        Base.mightalias(array, plan.reference_slopes) ||
        Base.mightalias(array, plan.pixel_thresholds) ||
        Base.mightalias(array, plan.flux_thresholds) ||
        Base.mightalias(array, plan.active) ||
        Base.mightalias(array, plan._column_coordinates[1]) ||
        Base.mightalias(array, plan._column_coordinates[2])
end

# Keep the serial pixel order for generic image storage and exceptional arithmetic.
@inline function _shack_hartmann_moments_serial(plan, image, subaperture)
    subaperture_rows, subaperture_columns = plan.regions.region_size
    first_row, first_column = plan.regions.origins[subaperture]
    x_moment = 0.0f0
    y_moment = 0.0f0
    measured_flux = 0.0f0
    coordinate = 1
    pixel_threshold = plan.pixel_thresholds[subaperture]
    @inbounds for row in 1:subaperture_rows
        for column in 1:subaperture_columns
            value = image[first_row + row, first_column + column]
            if !isfinite(value) || value >= pixel_threshold
                x_moment += value * plan.x_coordinates[coordinate]
                y_moment += value * plan.y_coordinates[coordinate]
                measured_flux += value
            end
            coordinate += 1
        end
    end
    return x_moment, y_moment, measured_flux
end

# Constructor callers retain the public coordinate vectors. Test their current
# bits once per range so direct edits (including signed zero) take effect.
function _shack_hartmann_coordinates_match(plan, coordinates)
    rows, columns = plan.regions.region_size
    length(coordinates[1]) == length(coordinates[2]) == length(plan.x_coordinates) || return false
    changed = zero(UInt32)
    @inbounds for column in 1:columns
        @simd for row in 1:rows
            pixel = (row - 1) * columns + column
            packed = (column - 1) * rows + row
            changed |= reinterpret(UInt32, plan.x_coordinates[pixel]) ⊻
                       reinterpret(UInt32, coordinates[1][packed])
            changed |= reinterpret(UInt32, plan.y_coordinates[pixel]) ⊻
                       reinterpret(UInt32, coordinates[2][packed])
        end
    end
    return iszero(changed)
end

@inline _shack_hartmann_moments(plan, image::AbstractMatrix, subaperture, limit, flux_error, packed::Val) =
    _shack_hartmann_moments_serial(plan, image, subaperture)

@inline function _shack_hartmann_moments(
    plan, image::StridedMatrix, subaperture, limit, flux_error, ::Val{Packed},
) where {Packed}
    if limit == 0.0f0
        return _shack_hartmann_moments_serial(plan, image, subaperture)
    end
    subaperture_rows, subaperture_columns = plan.regions.region_size
    first_row, first_column = plan.regions.origins[subaperture]
    pixel_threshold = plan.pixel_thresholds[subaperture]
    x_coordinates, y_coordinates = Packed ?
        plan._column_coordinates : (plan.x_coordinates, plan.y_coordinates)
    x_moment = 0.0f0
    y_moment = 0.0f0
    measured_flux = 0.0f0
    @inbounds for column in 1:subaperture_columns
        @simd for row in 1:subaperture_rows
            coordinate = Packed ? (column - 1) * subaperture_rows + row :
                                  (row - 1) * subaperture_columns + column
            value = image[first_row + row, first_column + column]
            value = ifelse(!isfinite(value) || value >= pixel_threshold, value, 0.0f0)
            x_moment += value * x_coordinates[coordinate]
            y_moment += value * y_coordinates[coordinate]
            measured_flux += value
        end
    end
    reference = 2 * subaperture - 1
    # The selected finite pixels are nonnegative. Bounding their flux times the
    # largest coordinate bounds every serial partial moment, even with cancellation.
    # Near a flux threshold, preserve the serial inclusive validity decision.
    if !isfinite(x_moment) || !isfinite(y_moment) ||
       !isfinite(measured_flux) || measured_flux > limit ||
       abs(measured_flux - plan.flux_thresholds[subaperture]) <= flux_error * measured_flux ||
       abs(plan.reference_slopes[reference]) > 0.25f0 * floatmax(Float32) ||
       abs(plan.reference_slopes[reference + 1]) > 0.25f0 * floatmax(Float32)
        return _shack_hartmann_moments_serial(plan, image, subaperture)
    end
    return x_moment, y_moment, measured_flux
end

_shack_hartmann_reduction_bounds(plan, image::AbstractMatrix, pixel_count) = (0.0f0, 0.0f0)

function _shack_hartmann_reduction_bounds(plan, image::StridedMatrix{Float32}, pixel_count)
    # Short windows retain the serial accumulation order.
    if plan.regions.region_size[1] < 16 || stride(image, 1) != 1 ||
       get_zero_subnormals() || rounding(Float32) !== RoundNearest
        return 0.0f0, 0.0f0
    end
    max_coordinate = max(maximum(abs, plan.x_coordinates), maximum(abs, plan.y_coordinates))
    # A conservative positive-sum rounding bound covers both accumulation orders.
    flux_error = 2.0f0 * Float32(pixel_count) * eps(Float32)
    if flux_error < 0.25f0 && max_coordinate <= 0.25f0 * floatmax(Float32)
        return 0.25f0 * floatmax(Float32) / max(1.0f0, max_coordinate), flux_error
    end
    return 0.0f0, 0.0f0
end

function _measure_shack_hartmann_range!(
    slopes::AbstractMatrix{Float32},
    flux::AbstractVector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannImagePlan,
    image::AbstractMatrix{Float32},
    subapertures::AbstractUnitRange{Int},
)
    isempty(subapertures) && return nothing
    pixel_count = prod(plan.regions.region_size)
    length(plan.x_coordinates) == length(plan.y_coordinates) == pixel_count ||
        throw(DimensionMismatch("coordinate extent must match one rectangular subaperture"))
    limit, flux_error = _shack_hartmann_reduction_bounds(plan, image, pixel_count)
    if !iszero(limit) && length(subapertures) > 1 &&
       _shack_hartmann_coordinates_match(plan, plan._column_coordinates)
        return _shack_hartmann_centroids!(
            slopes, flux, validity, plan, image, subapertures, limit, flux_error, Val(true),
        )
    end
    return _shack_hartmann_centroids!(
        slopes, flux, validity, plan, image, subapertures, limit, flux_error, Val(false),
    )
end

function _shack_hartmann_centroids!(
    slopes, flux, validity, plan, image, subapertures, limit, flux_error, packed::Val,
)
    @inbounds for subaperture in subapertures
        x_moment, y_moment, measured_flux = _shack_hartmann_moments(
            plan, image, subaperture, limit, flux_error, packed,
        )
        reference = 2 * subaperture - 1
        x_offset, y_offset, measured_flux, valid = _finish_center_of_gravity(
            x_moment,
            y_moment,
            measured_flux,
            plan.reference_slopes[reference],
            plan.reference_slopes[reference + 1],
            plan.flux_thresholds[subaperture],
        )
        if valid && plan.active[subaperture]
            slopes[subaperture, 1] = x_offset
            slopes[subaperture, 2] = y_offset
        else
            slopes[subaperture, 1] = 0.0f0
            slopes[subaperture, 2] = 0.0f0
        end
        flux[subaperture] = measured_flux
        validity[subaperture] = valid
    end
    return nothing
end

"""
Measure Shack-Hartmann spot centroids from a complete Float32 detector image.
Publishes X/Y slopes relative to reference positions (scaled pixel-coordinate
units), summed subaperture flux (input-pixel units), and validity per lenslet.
"""
@algorithm ShackHartmannImageF32 begin
    label = "shack-hartmann-image-f32"
    operation = ShackHartmannImageF32Operation()

    """
    Set detector/subaperture geometry and count, optional origins, pixel-coordinate
    scale, pixel and flux thresholds, reference slopes, and active mask. Optional
    arrays have one entry or pair per subaperture as defined by their parameter
    ports.
    """
    @configuration ShackHartmannImageF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        image_schema::String => graph_rebuild
        subaperture_size::NTuple{2,Int} => graph_rebuild
        subaperture_count::Int => graph_rebuild
        initial_subaperture_origins::Union{Nothing,Vector{Vector{Int}}} => construction
        coordinate_scale::Float32 => construction
        pixel_threshold::Float32 => construction
        flux_threshold::Float32 => construction
        reference_slopes::Union{Nothing,Vector{Float32}} => construction
        active::Union{Nothing,Vector{Bool}} => construction
    end

    ports(configuration) = (
        input(
            :image,
            Float32,
            configuration.image_size,
            configuration.image_schema,
        ),
        output(
            :slopes,
            schema_element_type(SHACK_HARTMANN_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_SLOPES_V1;
            metadata_from=:image,
        ),
        output(
            :flux,
            schema_element_type(SHACK_HARTMANN_FLUX_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_FLUX_V1;
            metadata_from=:image,
        ),
        output(
            :validity,
            schema_element_type(SHACK_HARTMANN_VALIDITY_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_VALIDITY_V1;
            metadata_from=:image,
        ),
        parameter(
            :subaperture_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (configuration.subaperture_count, 2),
            REGION_ORIGINS_V1;
            name="subaperture-origins",
            replace=(plan, values) -> _replace_shack_hartmann_origins(plan, values),
        ),
        parameter(
            :coordinates,
            schema_element_type(SHACK_HARTMANN_COORDINATES_V1),
            (
                configuration.subaperture_size[1] * configuration.subaperture_size[2],
                2,
            ),
            SHACK_HARTMANN_COORDINATES_V1;
            replace=(plan, values) -> _replace_shack_hartmann_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_reference_slopes(plan, values),
        ),
        parameter(
            :thresholds,
            schema_element_type(SHACK_HARTMANN_THRESHOLDS_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_THRESHOLDS_V1;
            replace=(plan, values) -> _replace_shack_hartmann_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) -> _replace_shack_hartmann_active(plan, values),
        ),
    )

    prepare(configuration) = begin
        origins = _initial_shack_hartmann_origins(
            configuration.image_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        ShackHartmannImagePlan(
            configuration.image_size,
            configuration.subaperture_size,
            origins,
            configuration.coordinate_scale,
            configuration.pixel_threshold,
            configuration.flux_threshold;
            reference_slopes=configuration.reference_slopes,
            active=configuration.active,
        )
    end

    process!(plan) = begin
        (Base.mightalias(slopes, image) || _shack_hartmann_aliases_plan(slopes, plan)) &&
            throw(ArgumentError("slope output aliases input or prepared calibration"))
        (Base.mightalias(flux, image) || _shack_hartmann_aliases_plan(flux, plan)) &&
            throw(ArgumentError("flux output aliases input or prepared calibration"))
        (Base.mightalias(validity, image) ||
            _shack_hartmann_aliases_plan(validity, plan)) && throw(
            ArgumentError("validity output aliases input or prepared calibration"),
        )
        Base.mightalias(slopes, flux) &&
            throw(ArgumentError("slope and flux outputs must not alias"))
        Base.mightalias(slopes, validity) &&
            throw(ArgumentError("slope and validity outputs must not alias"))
        Base.mightalias(flux, validity) &&
            throw(ArgumentError("flux and validity outputs must not alias"))

        _measure_shack_hartmann_range!(
            slopes,
            flux,
            validity,
            plan,
            image,
            eachindex(plan.regions.origins),
        )
    end
end
