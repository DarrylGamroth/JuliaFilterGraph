"""Expected rejection from signed Float32 detector calibration."""
@enum PixelCalibrationFailure::UInt8 begin
    PixelCalibrationNonFiniteInput = 1
    PixelCalibrationNonFiniteOutput = 2
end
@doc "Reject nonfinite detector samples." PixelCalibrationNonFiniteInput
@doc "Reject nonfinite calibrated samples." PixelCalibrationNonFiniteOutput

"""Replaceable scratch for finite Float32 pixel calibration."""
struct PixelCalibrationWorkspace
    calibrated::Matrix{Float32}
end

commit!(workspace::PixelCalibrationWorkspace) = workspace
discard!(workspace::PixelCalibrationWorkspace) = workspace
function reset!(workspace::PixelCalibrationWorkspace)
    fill!(workspace.calibrated, 0f0)
    return workspace
end

# The checked caller proves that these dense candidate writes are disjoint
# from raw and both coefficient planes. Validate each tile while it is local.
function _stage_pixel_calibration_f32!(
    candidate::Matrix{Float32}, plan::PixelCalibrationPlan, raw::Matrix{Float32},
)
    failure = nothing
    for first in 1:256:length(raw)
        last = min(first + 255, length(raw))
        if isnothing(failure)
            valid = true
            @inbounds @simd ivdep for i in first:last
                value = raw[i]
                result = plan.flat[i] * (value - plan.background[i])
                candidate[i] = result
                valid &= isfinite(value) & isfinite(result)
            end
            if !valid
                # Preserve input-before-result precedence at the first failure.
                @inbounds for i in first:last
                    if !isfinite(raw[i])
                        failure = PixelCalibrationNonFiniteInput
                        break
                    elseif !isfinite(candidate[i])
                        failure = PixelCalibrationNonFiniteOutput
                        break
                    end
                end
            end
        else
            @inbounds @simd ivdep for i in first:last
                candidate[i] = plan.flat[i] * (raw[i] - plan.background[i])
            end
        end
        # Finish the candidate even after rejection, retaining the original
        # scratch contents while leaving caller output unpublished.
    end
    return failure
end

function _stage_pixel_calibration_f32!(
    candidate::Matrix{Float32}, plan::PixelCalibrationPlan, raw::AbstractMatrix{Float32},
)
    for i in eachindex(raw, plan.flat, plan.background, candidate)
        candidate[i] = plan.flat[i] * (raw[i] - plan.background[i])
    end
    # Preserve the first input/result failure in logical iteration order.
    for i in eachindex(raw, candidate)
        isfinite(raw[i]) || return PixelCalibrationNonFiniteInput
        isfinite(candidate[i]) || return PixelCalibrationNonFiniteOutput
    end
    return nothing
end

function _process_pixel_calibration_f32!(
    calibrated::AbstractMatrix{Float32}, workspace::PixelCalibrationWorkspace,
    plan::PixelCalibrationPlan, raw::AbstractMatrix{Float32},
)
    Base.require_one_based_indexing(calibrated, raw)
    expected = size(plan.flat)
    size(raw) == size(calibrated) == size(workspace.calibrated) == expected ||
        throw(DimensionMismatch("pixel calibration arrays have incompatible shapes"))
    for owned in (plan.flat, plan.background, workspace.calibrated)
        Base.mightalias(calibrated, owned) &&
            throw(ArgumentError("calibrated output must not alias calibration or scratch"))
    end
    Base.mightalias(raw, workspace.calibrated) &&
        throw(ArgumentError("raw input must not alias calibration scratch"))
    candidate = workspace.calibrated
    if Base.mightalias(candidate, plan.flat) || Base.mightalias(candidate, plan.background)
        # Manually supplied scratch may overlap a calibration plane. Retain the
        # original partial-write behavior for that storage arrangement.
        for i in eachindex(raw, plan.flat, plan.background, candidate)
            value = raw[i]
            isfinite(value) || return PixelCalibrationNonFiniteInput
            result = plan.flat[i] * (value - plan.background[i])
            isfinite(result) || return PixelCalibrationNonFiniteOutput
            candidate[i] = result
        end
    else
        failure = _stage_pixel_calibration_f32!(candidate, plan, raw)
        isnothing(failure) || return failure
    end
    copyto!(calibrated, workspace.calibrated)
    return nothing
end

"""
Calibrate signed Float32 detector samples as `flat * (raw - background)`.
The declared input schema establishes the raw quantity; the flat converts it
to calibrated-pixel units. Negative read noise is preserved. No ADC, clipping,
gain, or unit conversion is inferred. Numerical failure preserves caller output.
"""
@algorithm PixelCalibrationF32 begin
    label = "pixel-calibration-f32"
    failure = PixelCalibrationFailure

    """Configure logical `[row,column]` image size, raw schema, and optional row-major flat/background arrays."""
    @configuration PixelCalibrationF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        input_schema::String => graph_rebuild
        initial_flat::Union{Nothing,Vector{Float32}} => construction
        initial_background::Union{Nothing,Vector{Float32}} => construction
    end

    ports(configuration) = (
        input(:raw, Float32, configuration.image_size, configuration.input_schema),
        output(:calibrated, Float32, configuration.image_size, CALIBRATED_PIXELS_V1;
            metadata_from=:raw),
        parameter(:flat, Float32, configuration.image_size, FLAT_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_flat(plan, values)),
        parameter(:background, Float32, configuration.image_size, BACKGROUND_CALIBRATION_V1;
            replace=(plan, values) -> _replace_pixel_background(plan, values)),
    )

    prepare(configuration) = begin
        rows, columns = configuration.image_size
        rows > 0 && columns > 0 || throw(ArgumentError("image size must be positive"))
        UInt128(rows) * UInt128(columns) * sizeof(Float32) <= typemax(UInt32) ||
            throw(ArgumentError("detector byte extent must fit UInt32"))
        isempty(strip(configuration.input_schema)) && throw(ArgumentError("input schema must not be empty"))
        plan = PixelCalibrationPlan(
            _row_major_calibration_plane(configuration.initial_flat, rows, columns, 1f0),
            _row_major_calibration_plane(configuration.initial_background, rows, columns, 0f0))
        (plan=plan, workspace=PixelCalibrationWorkspace(zeros(Float32, rows, columns)))
    end

    process!(plan, workspace) =
        _process_pixel_calibration_f32!(calibrated, workspace, plan, raw)
end
