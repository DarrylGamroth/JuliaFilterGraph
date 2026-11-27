function _summarize_shack_hartmann!(summary, slopes, flux, validity, subaperture_count::Int)
    (Base.mightalias(summary, slopes) || Base.mightalias(summary, flux) ||
     Base.mightalias(summary, validity)) && throw(
        ArgumentError("Shack-Hartmann summary output must not alias its inputs"),
    )
    return _shack_hartmann_summary!(summary, slopes, flux, validity, subaperture_count)
end

function _shack_hartmann_summary!(summary, slopes, flux, validity, subaperture_count)
    valid_count = 0
    finite = true
    flux_sum = 0.0
    x_sum = 0.0
    y_sum = 0.0
    x_square_sum = 0.0
    y_square_sum = 0.0
    for subaperture in 1:subaperture_count
        if validity[subaperture]
            valid_count += 1
            f = Float64(flux[subaperture])
            x = Float64(slopes[subaperture, 1])
            y = Float64(slopes[subaperture, 2])
            finite &= isfinite(f) && isfinite(x) && isfinite(y)
            flux_sum += f
            x_sum += x
            y_sum += y
            x_square_sum += x * x
            y_square_sum += y * y
        end
    end
    summary[1] = Float32(subaperture_count)
    summary[2] = Float32(valid_count)
    if valid_count == 0 || !finite
        for field in 3:8
            summary[field] = NaN32
        end
    else
        rms_x = sqrt(x_square_sum / valid_count)
        rms_y = sqrt(y_square_sum / valid_count)
        summary[3] = Float32(flux_sum / valid_count)
        summary[4] = Float32(rms_x)
        summary[5] = Float32(rms_y)
        summary[6] = Float32(hypot(rms_x, rms_y))
        summary[7] = Float32(x_sum / valid_count)
        summary[8] = Float32(y_sum / valid_count)
    end
    return nothing
end

"""
Summarize one complete Shack-Hartmann frame without modifying its inputs.
The eight Float32 fields are total subaperture count, valid count, mean valid
flux, X slope RMS, Y slope RMS, hypot(X RMS, Y RMS), mean X slope, and mean Y
slope. Accumulation uses Float64; RMS is uncentered. Invalid rows are ignored.
With no valid rows, or any nonfinite flux or slope in a valid row, the six
aggregate fields are NaN and processing still succeeds. Counts follow validity.

Connect all three inputs to outputs from the same measurement frame. Summary
metadata follows `slopes`. Flux retains its input units; slopes retain their
detector-coordinate units (detector pixels for pixel-coordinate measurements).
Mean slopes are not calibrated physical tip, tilt, or focus estimates. This
diagnostic performs no coordinate, reference, optical, or calibration conversion.
"""
@algorithm ShackHartmannSummaryF32 begin
    label = "shack-hartmann-summary-f32"

    """
    Fix the subaperture count in 1:2^24 for [`ShackHartmannSummaryF32`](@ref).
    This limit keeps both Float32 counts exact. The count bounds each processing
    call; changing it requires Graph rebuilding.
    """
    @configuration ShackHartmannSummaryF32Configuration begin
        subaperture_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :slopes,
            schema_element_type(SHACK_HARTMANN_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_SLOPES_V1,
        ),
        input(
            :flux,
            schema_element_type(SHACK_HARTMANN_FLUX_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_FLUX_V1,
        ),
        input(
            :validity,
            schema_element_type(SHACK_HARTMANN_VALIDITY_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_VALIDITY_V1,
        ),
        output(
            :summary,
            schema_element_type(SHACK_HARTMANN_SUMMARY_V1),
            (8,),
            SHACK_HARTMANN_SUMMARY_V1;
            metadata_from=:slopes,
        ),
    )

    prepare(configuration) = begin
        0 < configuration.subaperture_count <= 2^24 || throw(
            ArgumentError("subaperture_count must be in 1:2^24 for exact Float32 counts"),
        )
        (subaperture_count=configuration.subaperture_count,)
    end

    process!(plan) = _summarize_shack_hartmann!(
        summary, slopes, flux, validity, plan.subaperture_count,
    )
end
