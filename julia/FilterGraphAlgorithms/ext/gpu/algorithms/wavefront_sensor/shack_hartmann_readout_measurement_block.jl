# The serial-per-subaperture readout uses a launch tile, not a reduction tree.
const SHACK_HARTMANN_READOUT_LANES = 128

@kernel function _shack_hartmann_readout!(
        packet, status, detector, ready, origins, x_coordinates, y_coordinates,
        reference_slopes, pixel_thresholds, flux_thresholds, active,
        region_rows::Int, region_columns::Int, weights=nothing
)
    entry = @index(Global, Linear)
    if entry == 1
        @inbounds status[1] = Int32(0)
        @inbounds packet[1] = eltype(packet)(0.0f0, UInt32(2 * length(ready)))
    end
    if entry <= length(ready)
        @inbounds begin
            subaperture = Int(ready[entry])
            first_row = Int(origins[1, subaperture])
            first_column = Int(origins[2, subaperture])
            x_moment = 0.0f0
            y_moment = 0.0f0
            flux = 0.0f0
            coordinate = 1
            threshold = pixel_thresholds[subaperture]
            for row in 1:region_rows, column in 1:region_columns

                value = detector[first_row + row, first_column + column]
                if !isfinite(value) || value >= threshold
                    if weights === nothing
                        x_moment += value * x_coordinates[coordinate]
                        y_moment += value * y_coordinates[coordinate]
                        flux += value
                    else
                        weighted = weights[subaperture, coordinate] * value
                        x_moment = fma(weighted, x_coordinates[coordinate], x_moment)
                        y_moment = fma(weighted, y_coordinates[coordinate], y_moment)
                        flux += weighted
                    end
                end
                coordinate += 1
            end
            reference = 2 * subaperture - 1
            x, y, _, valid = FilterGraphAlgorithms._finish_center_of_gravity(
                x_moment, y_moment, flux,
                reference_slopes[reference], reference_slopes[reference + 1],
                flux_thresholds[subaperture]
            )
            if !valid || !active[subaperture]
                x = 0.0f0
                y = 0.0f0
            end
            packet[2 * entry] = eltype(packet)(x, UInt32(reference - 1))
            packet[2 * entry + 1] = eltype(packet)(y, UInt32(reference))
        end
    end
end

@kernel function _shack_hartmann_readout_cooperative!(
        packet, status, detector, ready, origins, x_coordinates, y_coordinates,
        reference_slopes, pixel_thresholds, flux_thresholds, active,
        region_rows::Int, region_columns::Int, reduction
)
    lane = @index(Local, Linear)
    _, entry = @index(Group, NTuple)
    x_moments = @localmem Float32 (32,)
    y_moments = @localmem Float32 (32,)
    fluxes = @localmem Float32 (32,)
    if entry == 1 && lane == 1
        @inbounds status[1] = Int32(0)
        @inbounds packet[1] = eltype(packet)(0.0f0, UInt32(2 * length(ready)))
    end
    x_moment = @private Float32 (1,)
    x_moment[1] = 0.0f0
    y_moment = @private Float32 (1,)
    y_moment[1] = 0.0f0
    flux = @private Float32 (1,)
    flux[1] = 0.0f0
    if entry <= length(ready)
        @inbounds begin
            subaperture = Int(ready[entry])
            first_row = Int(origins[1, subaperture])
            first_column = Int(origins[2, subaperture])
            threshold = pixel_thresholds[subaperture]
            # Lanes follow the detector's contiguous row dimension; coordinates
            # retain the CPU Plan's row-then-column scientific ordering.
            pixel = Int(lane)
            while pixel <= region_rows * region_columns
                row = rem(pixel - 1, region_rows) + 1
                column = div(pixel - 1, region_rows) + 1
                coordinate = (row - 1) * region_columns + column
                value = detector[first_row + row, first_column + column]
                if !isfinite(value) || value >= threshold
                    x_moment[1] += value * x_coordinates[coordinate]
                    y_moment[1] += value * y_coordinates[coordinate]
                    flux[1] += value
                end
                pixel += 32
            end
        end
    end
    if reduction === nothing
        @inbounds begin
            x_moments[lane] = x_moment[1]
            y_moments[lane] = y_moment[1]
            fluxes[lane] = flux[1]
        end
        @synchronize
        for stride in (16, 8, 4, 2, 1)
            _reduce_shack_hartmann_moments!(
                x_moments, y_moments, fluxes, Int(lane), stride
            )
            @synchronize
        end
        if lane == 1
            @inbounds begin
                x_moment[1] = x_moments[1]
                y_moment[1] = y_moments[1]
                flux[1] = fluxes[1]
            end
        end
    else
        x_moment[1], y_moment[1], flux[1] = reduction(x_moment[1], y_moment[1], flux[1])
    end
    if entry <= length(ready) && lane == 1
        @inbounds begin
            subaperture = Int(ready[entry])
            reference = 2 * subaperture - 1
            x, y, _, valid = FilterGraphAlgorithms._finish_center_of_gravity(
                x_moment[1], y_moment[1], flux[1],
                reference_slopes[reference], reference_slopes[reference + 1],
                flux_thresholds[subaperture]
            )
            if !valid || !active[subaperture]
                x = 0.0f0
                y = 0.0f0
            end
            packet[2 * entry] = eltype(packet)(x, UInt32(reference - 1))
            packet[2 * entry + 1] = eltype(packet)(y, UInt32(reference))
        end
    end
end

_shack_hartmann_readout_kernel(backend) = _ShackHartmannKernel(
    _shack_hartmann_readout_cooperative!(backend, 32), nothing,
)
