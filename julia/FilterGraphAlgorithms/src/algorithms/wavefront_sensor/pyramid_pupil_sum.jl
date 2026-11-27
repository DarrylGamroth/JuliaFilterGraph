# Only bounded, ordinary host arithmetic uses the reassociated reduction.
const _PYRAMID_IMAGE_SUM_LIMIT = 4096

"""
    pyramid_geometry_requires_ordered_sum(plan::PyramidPixelImagePlan)

Return whether the dense CPU pupil geometry requires the canonical Float32 sum
in pupil, row, and column order. A one-row pupil has only one term per column.
For taller pupils the sample limit counts every pixel in all four regions,
including masked pixels and overlapping samples.

This geometry check is stable across frames and arithmetic modes. Within the
geometry eligible for reassociation, the plan, arithmetic mode, or pixel values
can still require an ordered scan on a particular call.
"""
function pyramid_geometry_requires_ordered_sum(plan::PyramidPixelImagePlan)
    rows, _ = plan.pupil_regions.region_size
    count = PYRAMID_PUPIL_COUNT * pyramid_pupil_pixel_count(plan.normalization)
    return rows == 1 || count > _PYRAMID_IMAGE_SUM_LIMIT
end

@inline function _pyramid_sum_needs_serial(total, sumabs, mean_scale)
    # Keep the ordered gain decision when cancellation can change the sign,
    # or when rounding can make a tiny mean or its reciprocal under/overflow.
    return abs(total) <= 0.01 * sumabs ||
           abs(total) * Float64(mean_scale) < floatmin(Float32)
end

function _serial_pyramid_image_sum(plan::PyramidPixelImagePlan, image)
    regions = plan.pupil_regions
    rows, columns = regions.region_size
    total = 0.0f0
    @inbounds for pupil in eachindex(regions.origins)
        first_row, first_column = regions.origins[pupil]
        for row in 1:rows, column in 1:columns
            value = image[first_row + row, first_column + column]
            isfinite(value) || return total, PyramidPixelImageNonFiniteInput
            total += value
            isfinite(total) || return total, PyramidPixelImageNonFiniteMean
        end
    end
    return total, nothing
end

_pyramid_image_sum(plan::PyramidPixelImagePlan, image::AbstractMatrix{Float32}) =
    _serial_pyramid_image_sum(plan, image)

function _pyramid_image_sum(plan::PyramidPixelImagePlan, image::StridedMatrix{Float32})
    regions = plan.pupil_regions
    rows, columns = regions.region_size
    if plan.ordered_sum || pyramid_geometry_requires_ordered_sum(plan) ||
       get_zero_subnormals() || rounding(Float32) !== RoundNearest
        return _serial_pyramid_image_sum(plan, image)
    end
    total = 0.0
    sumabs = 0.0
    valid = true
    limit = floatmax(Float32) * plan.normalization.mean_scale * 0.5f0
    @inbounds for pupil in eachindex(regions.origins)
        first_row, first_column = regions.origins[pupil]
        for column in 1:columns
            partial = 0.0
            column_abs = 0.0
            column_valid = true
            @simd for row in 1:rows
                value = image[first_row + row, first_column + column]
                partial += Float64(value)
                column_abs += abs(Float64(value))
                # This finite bound also rejects NaN/Inf. Near the bound, rescan
                # in logical order to preserve input versus overflow priority.
                column_valid &= abs(value) < limit
            end
            total += partial
            sumabs += column_abs
            valid &= column_valid
        end
    end
    sum = Float32(total)
    if !valid || !isfinite(sum) || _pyramid_sum_needs_serial(total, sumabs, plan.normalization.mean_scale)
        return _serial_pyramid_image_sum(plan, image)
    end
    return sum, nothing
end

# With at most 2^14 terms, the ordered Float32 sum's rounding error
# is below 0.001 × sumabs, within the 1% cancellation guard.
const _PYRAMID_TRANSPOSED_IMAGE_SUM_LIMIT = 1 << 14

# Logical columns are contiguous within each row of the transposed image.
# Reduce contiguous samples in Float64 before converting the full sum.
function _pyramid_image_sum(
    plan::PyramidPixelImagePlan,
    image::Transpose{Float32,Matrix{Float32}},
)
    regions = plan.pupil_regions
    rows, columns = regions.region_size
    count = PYRAMID_PUPIL_COUNT * length(plan.normalization.mask)
    if plan.ordered_sum || columns == 1 || count > _PYRAMID_TRANSPOSED_IMAGE_SUM_LIMIT ||
       get_zero_subnormals() || rounding(Float32) !== RoundNearest
        return _serial_pyramid_image_sum(plan, image)
    end
    total = 0.0
    sumabs = 0.0
    valid = true
    limit = floatmax(Float32) * plan.normalization.mean_scale * 0.5f0
    @inbounds for pupil in eachindex(regions.origins)
        first_row, first_column = regions.origins[pupil]
        for row in 1:rows
            partial = 0.0
            row_abs = 0.0
            row_valid = true
            @simd for column in 1:columns
                value = image[first_row + row, first_column + column]
                partial += Float64(value)
                row_abs += abs(Float64(value))
                row_valid &= abs(value) < limit
            end
            total += partial
            sumabs += row_abs
            valid &= row_valid
        end
    end
    sum = Float32(total)
    if !valid || !isfinite(sum) || _pyramid_sum_needs_serial(total, sumabs, plan.normalization.mean_scale)
        return _serial_pyramid_image_sum(plan, image)
    end
    return sum, nothing
end
