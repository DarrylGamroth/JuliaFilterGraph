const COG_PIXELS_PER_LANE = 8

@kernel unsafe_indices=true function _cog_partials!(partials, pixels, x, y, settings, status)
    lane = @index(Local, Linear)
    block = @index(Group, Linear)
    mx = my = total = ax = ay = absolute_flux = 0f0
    if @inbounds iszero(status[1])
        first_pixel = (block - 1) * 32 * COG_PIXELS_PER_LANE + lane
        @inbounds for pixel in first_pixel:32:min(length(pixels), first_pixel + 32 * (COG_PIXELS_PER_LANE - 1))
            value = pixels[pixel]
            if !isfinite(value) || iszero(settings[4]) || value >= settings[5]
                vx = value * x[pixel]
                vy = value * y[pixel]
                mx += vx
                my += vy
                total += value
                ax += abs(vx)
                ay += abs(vy)
                absolute_flux += abs(value)
            end
        end
    end
    mx, my, total = _warp_moments(mx, my, total)
    ax, ay, absolute_flux = _warp_moments(ax, ay, absolute_flux)
    if lane == 1
        @inbounds begin
            partials[1, block] = mx
            partials[2, block] = my
            partials[3, block] = total
            partials[4, block] = ax
            partials[5, block] = ay
            partials[6, block] = absolute_flux
        end
    end
end

_require_cog_workgroup(device::CUDA.CuDevice) = _require_capture_workgroup(device, COG_TILE; warp_width=32)

capture_support(::Type{CenterOfGravityF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{CenterOfGravityF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_center_of_gravity(device, prepared, outputs, inputs, status; ordered=true)
end

function _prepare_parallel_center_of_gravity(device::CUDA.CuDevice, backend, exact,
        outputs, pixels, x, y, settings, status, fallback)
    blocks = cld(length(pixels), 32 * COG_PIXELS_PER_LANE)
    partial = _capture_zeros(device, Float32, 6, blocks)
    reduce = CapturedKernel(_cog_partials!(backend, 32),
        (partial, pixels, x, y, settings, status), blocks * 32)
    finish = CapturedKernel(_cog_finish!(backend, 32),
        (outputs..., partial, settings, status, fallback, length(pixels)), 32)
    return CapturedCenterOfGravity((reduce, finish, exact), settings)
end

# If all intermediates are comfortably finite, the standard Float32 summation
# bound γₙΣ|term| applies to either addition order. For n ≤ 65,536 the two
# orders together, including rounding of the computed absolute sum, are
# bounded by 2n*eps(Float32)*absolute_flux plus the subnormal allowance.
# Use the ordered kernel if this interval touches either flux boundary or
# if the ordered moments, centroid, or reference subtraction could overflow.
@inline function _cog_requires_ordered(mx, my, total, ax, ay, absolute_flux, settings, count)
    limit = Float32(0x1p126)
    if !(ax < limit && ay < limit && absolute_flux < limit &&
            isfinite(mx) && isfinite(my) && isfinite(total))
        return true
    end
    iszero(absolute_flux) && return false
    margin = 2f0 * count * eps(Float32) * absolute_flux + count * floatmin(Float32)
    threshold = @inbounds settings[3]
    if abs(total) <= margin || abs(total - threshold) <= margin
        return true
    end
    if total > 0f0 && total >= threshold
        minimum_flux = total - margin
        return !(2f0 * ax / minimum_flux < limit && 2f0 * ay / minimum_flux < limit &&
            @inbounds(abs(settings[1]) < limit && abs(settings[2]) < limit))
    end
    return false
end

@kernel unsafe_indices=true function _cog_finish!(
        centroid, flux, validity, partials, settings, status, fallback, count)
    lane = @index(Local, Linear)
    active = @inbounds iszero(status[1])
    mx = my = total = ax = ay = absolute_flux = 0f0
    if active
        @inbounds for block in lane:32:size(partials, 2)
            mx += partials[1, block]
            my += partials[2, block]
            total += partials[3, block]
            ax += partials[4, block]
            ay += partials[5, block]
            absolute_flux += partials[6, block]
        end
    end
    mx, my, total = _warp_moments(mx, my, total)
    ax, ay, absolute_flux = _warp_moments(ax, ay, absolute_flux)
    if lane == 1
        ordered = active && _cog_requires_ordered(mx, my, total, ax, ay, absolute_flux, settings, count)
        @inbounds fallback[1] = ordered
        if active && !ordered
            @inbounds begin
                dx, dy, measured, valid = FilterGraphAlgorithms._finish_center_of_gravity(
                    mx, my, total, settings[1], settings[2], settings[3])
                centroid[1] = dx
                centroid[2] = dy
                flux[1] = measured
                validity[1] = valid
            end
        end
    end
end
