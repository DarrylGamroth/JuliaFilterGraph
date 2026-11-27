function _shack_hartmann_weighted_simd(sensing::ShackHartmannImagePlan)
    return length(sensing.regions.origins) >= 4 &&
        length(sensing.x_coordinates) <= 64 &&
        all(>(1), sensing.regions.region_size)
end

function _check_shack_hartmann_weight_shape(sensing, weights)
    size(weights) == (length(sensing.regions.origins), length(sensing.x_coordinates)) ||
        throw(DimensionMismatch(
            "Shack-Hartmann weight shape must be (subaperture_count, pixel_count)",
        ))
    return nothing
end

# 4,096 Float32 coefficients retain at least 16 KiB of packed payload.
_pack_shack_hartmann_weights(weights) = length(weights) >= 4096 ? permutedims(weights) : nothing

"""
Prepared complete-image weighted centroid measurement for a Shack-Hartmann
wavefront sensor.

Each weight row belongs to one subaperture and follows the FGA row-major
selected-pixel order used by [`ShackHartmannImagePlan`](@ref). A finite,
nonnegative dimensionless weight multiplies each selected calibrated pixel
before its flux and coordinate moments are accumulated. CPU X/Y moments use Float32
fused multiply-add in row-major pixel order; weighted pixels are rounded first
and flux retains ordinary ordered addition. Pixel thresholds remain
absolute, inclusive calibrated-pixel thresholds, while the flux threshold
applies to the weighted flux Σ(wᵢ·Iᵢ). A non-finite image pixel invalidates its
subaperture even when its weight is zero.

Weights are owned, logically read-only calibration. CPU plans for scalar
traversal also pack calibrations of at least 4,096 coefficients with contiguous
pixels per subaperture. Complete-image plans with SIMD-eligible geometry use
the canonical weights for blocks and scalar fallbacks. Weight replacement
preserves the prepared layouts.

The CPU-only keyword `simd_width` accepts 0 (automatic, currently four lanes),
4, 8, or 16. It selects independent lenslets per block, not an instruction-set
requirement. LLVM lowers each width for the target CPU. Wider blocks retain
four-lane and scalar tails; wrappers and larger windows retain scalar traversal.
GPU kernels do not use this CPU tuning option.
"""
struct ShackHartmannWeightedCentroidPlan <: AbstractAlgorithmPlan
    sensing::ShackHartmannImagePlan
    weights::Matrix{Float32}
    packed_weights::Union{Nothing,Matrix{Float32}}
    simd_width::Int

    function ShackHartmannWeightedCentroidPlan(
        sensing::ShackHartmannImagePlan,
        weights::AbstractMatrix{Float32};
        simd_width::Integer=0,
        pack_weights::Bool=!_shack_hartmann_weighted_simd(sensing),
    )
        simd_width in (0, 4, 8, 16) || throw(
            ArgumentError("SIMD width must be 0 (automatic), 4, 8, or 16"),
        )
        Base.require_one_based_indexing(weights)
        _check_shack_hartmann_weight_shape(sensing, weights)
        prepared_weights = Matrix{Float32}(weights)
        all(weight -> isfinite(weight) && weight >= 0.0f0, prepared_weights) || throw(
            ArgumentError("Shack-Hartmann weights must be finite and nonnegative"),
        )
        packed_weights = pack_weights ? _pack_shack_hartmann_weights(prepared_weights) : nothing
        return new(sensing, prepared_weights, packed_weights, simd_width == 0 ? 4 : Int(simd_width))
    end

    function ShackHartmannWeightedCentroidPlan(
        sensing::ShackHartmannImagePlan,
        calibration::ShackHartmannWeightedCentroidPlan;
        pack_weights::Bool=!isnothing(calibration.packed_weights),
    )
        _check_shack_hartmann_weight_shape(sensing, calibration.weights)
        packed_weights = if !pack_weights
            nothing
        elseif isnothing(calibration.packed_weights)
            _pack_shack_hartmann_weights(calibration.weights)
        else
            calibration.packed_weights
        end
        return new(sensing, calibration.weights, packed_weights, calibration.simd_width)
    end
end

function _configured_shack_hartmann_weights(
    values::Vector{Vector{Float32}},
    subaperture_count::Int,
    pixel_count::Int,
)
    length(values) == subaperture_count || throw(
        DimensionMismatch("initial Shack-Hartmann weight count must match subapertures"),
    )
    weights = Matrix{Float32}(undef, subaperture_count, pixel_count)
    for subaperture in 1:subaperture_count
        row = values[subaperture]
        length(row) == pixel_count || throw(
            DimensionMismatch(
                "each initial Shack-Hartmann weight row must match subaperture pixels",
            ),
        )
        for pixel in 1:pixel_count
            weights[subaperture, pixel] = row[pixel]
        end
    end
    return weights
end

function _replace_shack_hartmann_weighted_centroid_origins(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        _replace_shack_hartmann_origins(plan.sensing, values),
        plan,
    )
end

function _replace_shack_hartmann_weighted_centroid_coordinates(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        _replace_shack_hartmann_coordinates(plan.sensing, values),
        plan,
    )
end

function _replace_shack_hartmann_weighted_centroid_reference_slopes(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        _replace_shack_hartmann_reference_slopes(plan.sensing, values),
        plan,
    )
end

function _replace_shack_hartmann_weighted_centroid_thresholds(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        _replace_shack_hartmann_thresholds(plan.sensing, values),
        plan,
    )
end

function _replace_shack_hartmann_weighted_centroid_active(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        _replace_shack_hartmann_active(plan.sensing, values),
        plan,
    )
end

function _replace_shack_hartmann_weighted_centroid_weights(
    plan::ShackHartmannWeightedCentroidPlan,
    values,
)
    return ShackHartmannWeightedCentroidPlan(
        plan.sensing,
        values;
        simd_width=plan.simd_width,
        pack_weights=!isnothing(plan.packed_weights),
    )
end

@inline function _shack_hartmann_weighted_centroid_aliases_plan(
    array,
    plan::ShackHartmannWeightedCentroidPlan,
)
    return _shack_hartmann_aliases_plan(array, plan.sensing) ||
        Base.mightalias(array, plan.weights) ||
        (!isnothing(plan.packed_weights) && Base.mightalias(array, plan.packed_weights))
end

# Dense images benefit from contiguous per-lenslet weights on scalar traversal.
# Image views retain the original calibration layout.
_shack_hartmann_packed_weights(plan, ::Matrix{Float32}) = plan.packed_weights
_shack_hartmann_packed_weights(plan, ::AbstractMatrix{Float32}) = nothing

function _measure_shack_hartmann_weighted_centroid_range!(
    slopes::AbstractMatrix{Float32},
    flux::AbstractVector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannWeightedCentroidPlan,
    image::AbstractMatrix{Float32},
    subapertures,
    packed_weights::Union{Nothing,Matrix{Float32}} = _shack_hartmann_packed_weights(plan, image),
)
    sensing = plan.sensing
    subaperture_rows, subaperture_columns = sensing.regions.region_size
    for subaperture in subapertures
        first_row, first_column = sensing.regions.origins[subaperture]
        x_moment = 0.0f0
        y_moment = 0.0f0
        measured_flux = 0.0f0
        coordinate = 1
        pixel_threshold = sensing.pixel_thresholds[subaperture]
        for row in 1:subaperture_rows
            for column in 1:subaperture_columns
                value = image[first_row + row, first_column + column]
                if !isfinite(value) || value >= pixel_threshold
                    weight = isnothing(packed_weights) ?
                        plan.weights[subaperture, coordinate] :
                        packed_weights[coordinate, subaperture]
                    weighted_value = weight * value
                    x_moment = fma(weighted_value, sensing.x_coordinates[coordinate], x_moment)
                    y_moment = fma(weighted_value, sensing.y_coordinates[coordinate], y_moment)
                    measured_flux += weighted_value
                end
                coordinate += 1
            end
        end
        reference = 2 * subaperture - 1
        x_offset, y_offset, measured_flux, valid = _finish_center_of_gravity(
            x_moment,
            y_moment,
            measured_flux,
            sensing.reference_slopes[reference],
            sensing.reference_slopes[reference + 1],
            sensing.flux_thresholds[subaperture],
        )
        if valid && sensing.active[subaperture]
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

# Checked callers prove image bounds; each SIMD lane follows one lenslet.
function _shack_hartmann_weighted_moments_block(plan, image::Matrix{Float32}, weights::Matrix{Float32}, first, ::Val{W}=Val(4)) where {W}
    origins = @inbounds ntuple(lane -> plan.regions.origins[first + lane - 1], Val(W))
    thresholds = @inbounds ntuple(lane -> plan.pixel_thresholds[first + lane - 1], Val(W))
    mx = my = flux = zero(Vec{W,Float32})
    coordinate = 1
    rows, columns = plan.regions.region_size
    @inbounds for row in 1:rows, column in 1:columns
        x, y = plan.x_coordinates[coordinate], plan.y_coordinates[coordinate]
        pixels = let coordinate=coordinate
            ntuple(Val(W)) do lane
                @inline
                r, c = origins[lane]
                value = @inbounds image[r + row, c + column]
                weighted = (@inbounds weights[first + lane - 1, coordinate]) * value
                (weighted, !isfinite(value) | (value >= thresholds[lane]))
            end
        end
        weighted_pixels = Vec(ntuple(lane -> pixels[lane][1], Val(W)))
        # Full-width flags give LLVM one packed mask for all three updates.
        selected = Vec(ntuple(lane -> Int32(pixels[lane][2]), Val(W))) == Int32(1)
        mx = ifelse(selected, fma(weighted_pixels, x, mx), mx)
        my = ifelse(selected, fma(weighted_pixels, y, my), my)
        flux = ifelse(selected, flux + weighted_pixels, flux)
        coordinate += 1
    end
    return let mx=mx, my=my, flux=flux
        ntuple(lane -> (mx[lane], my[lane], flux[lane]), Val(W))
    end
end

@inline function _store_shack_hartmann_weighted_moments!(slopes, flux, validity, plan, subaperture, moments)
    reference = 2 * subaperture - 1
    x_offset, y_offset, measured_flux, valid = _finish_center_of_gravity(
        moments..., plan.reference_slopes[reference],
        plan.reference_slopes[reference + 1], plan.flux_thresholds[subaperture],
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
    return nothing
end

# Returns the first remaining subaperture; the established range helper handles tails.
function _measure_shack_hartmann_weighted_blocks!(slopes::Matrix{Float32}, flux::Vector{Float32},
    validity, plan, weights, image::Matrix{Float32}, subapertures, ::Val{W}=Val(4)) where {W}
    @inbounds for start in first(subapertures):W:(last(subapertures) - W + 1)
        moments = _shack_hartmann_weighted_moments_block(plan, image, weights, start, Val(W))
        for lane in 1:W
            _store_shack_hartmann_weighted_moments!(
                slopes, flux, validity, plan, start + lane - 1, moments[lane],
            )
        end
    end
    first_remaining = first(subapertures) + W * div(length(subapertures), W)
    if W != 4
        return _measure_shack_hartmann_weighted_blocks!(
            slopes, flux, validity, plan, weights, image,
            first_remaining:last(subapertures), Val(4),
        )
    end
    return first_remaining
end

function _measure_shack_hartmann_weighted_centroid!(
    slopes::AbstractMatrix{Float32},
    flux::AbstractVector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannWeightedCentroidPlan,
    image::AbstractMatrix{Float32},
)
    return _measure_shack_hartmann_weighted_centroid_range!(
        slopes, flux, validity, plan, image, eachindex(plan.sensing.regions.origins),
    )
end

function _measure_shack_hartmann_weighted_centroid!(
    slopes::Matrix{Float32},
    flux::Vector{Float32},
    validity::AbstractVector{Bool},
    plan::ShackHartmannWeightedCentroidPlan,
    image::Matrix{Float32},
)
    subapertures = eachindex(plan.sensing.regions.origins)
    if _shack_hartmann_weighted_simd(plan.sensing)
        width = plan.simd_width == 16 ? Val(16) :
            plan.simd_width == 8 ? Val(8) : Val(4)
        first = _measure_shack_hartmann_weighted_blocks!(
            slopes, flux, validity, plan.sensing, plan.weights, image, subapertures, width,
        )
        subapertures = first:last(subapertures)
        isempty(subapertures) && return nothing
    end
    return _measure_shack_hartmann_weighted_centroid_range!(
        slopes,
        flux,
        validity,
        plan,
        image,
        subapertures,
    )
end

"""
Measure complete-image Shack-Hartmann centroids with per-pixel weights.
Publishes reference-relative X/Y slopes, flux, and validity for each active
subaperture; weights modify the centroid accumulation.
"""
@algorithm ShackHartmannWeightedCentroidF32 begin
    label = "shack-hartmann-weighted-centroid-f32"
    operation = ShackHartmannWeightedCentroidF32Operation()

    """
    Set image/subaperture geometry and count, optional origins, coordinate scale,
    thresholds, reference slopes and active mask, plus one initial weight array
    per subaperture. Pixel and flux thresholds are in input-pixel units.
    Optional CPU `simd_width` is 0 (automatic, currently 4), 4, 8, or 16.
    It selects lenslets per block; wider widths require workload qualification.
    """
    @configuration ShackHartmannWeightedCentroidF32Configuration begin
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
        initial_weights::Vector{Vector{Float32}} => construction
        (simd_width::Int = 0) => construction
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
            schema_element_type(WEIGHTED_SHACK_HARTMANN_FLUX_V1),
            (configuration.subaperture_count,),
            WEIGHTED_SHACK_HARTMANN_FLUX_V1;
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
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_origins(plan, values),
        ),
        parameter(
            :coordinates,
            schema_element_type(SHACK_HARTMANN_COORDINATES_V1),
            (
                configuration.subaperture_size[1] * configuration.subaperture_size[2],
                2,
            ),
            SHACK_HARTMANN_COORDINATES_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_reference_slopes(plan, values),
        ),
        parameter(
            :thresholds,
            schema_element_type(SHACK_HARTMANN_THRESHOLDS_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_THRESHOLDS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_active(plan, values),
        ),
        parameter(
            :weights,
            schema_element_type(SHACK_HARTMANN_WEIGHTS_V1),
            (
                configuration.subaperture_count,
                configuration.subaperture_size[1] * configuration.subaperture_size[2],
            ),
            SHACK_HARTMANN_WEIGHTS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_weighted_centroid_weights(plan, values),
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
        weights = _configured_shack_hartmann_weights(
            configuration.initial_weights,
            length(sensing.regions.origins),
            length(sensing.x_coordinates),
        )
        ShackHartmannWeightedCentroidPlan(sensing, weights; simd_width=configuration.simd_width)
    end

    process!(plan) = begin
        Base.require_one_based_indexing(image, slopes, flux, validity)
        (Base.mightalias(slopes, image) ||
            _shack_hartmann_weighted_centroid_aliases_plan(slopes, plan)) && throw(
            ArgumentError("slope output aliases input or prepared calibration"),
        )
        (Base.mightalias(flux, image) ||
            _shack_hartmann_weighted_centroid_aliases_plan(flux, plan)) && throw(
            ArgumentError("flux output aliases input or prepared calibration"),
        )
        (Base.mightalias(validity, image) ||
            _shack_hartmann_weighted_centroid_aliases_plan(validity, plan)) && throw(
            ArgumentError("validity output aliases input or prepared calibration"),
        )
        Base.mightalias(slopes, flux) &&
            throw(ArgumentError("slope and flux outputs must not alias"))
        Base.mightalias(slopes, validity) &&
            throw(ArgumentError("slope and validity outputs must not alias"))
        Base.mightalias(flux, validity) &&
            throw(ArgumentError("flux and validity outputs must not alias"))

        _measure_shack_hartmann_weighted_centroid!(
            slopes,
            flux,
            validity,
            plan,
            image,
        )
    end
end

# Preserve the original positional configuration constructor; keyword callers
# receive the same automatic setting through the declared default.
ShackHartmannWeightedCentroidF32Configuration(values::Vararg{Any,11}) =
    ShackHartmannWeightedCentroidF32Configuration(values..., 0)
