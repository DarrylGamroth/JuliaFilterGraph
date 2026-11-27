import FilterGraphAlgorithms:
    PixelCalibrationPlan,
    PixelCalibrationU16F32,
    PreparedAlgorithm,
    _calibrated_pixel,
    complete!,
    enqueue!,
    preflight,
    reset!

@kernel function _pixel_calibration!(output, flat, background, input)
    index = @index(Global, Linear)
    if index <= length(output)
        @inbounds output[index] = _calibrated_pixel(
            input[index],
            flat[index],
            background[index],
        )
    end
end

struct PixelCalibration{Calibrate,O,F,B,I,Bounds}
    calibrate!::Calibrate
    output::O
    flat::F
    background::B
    input::I
    bounds::Bounds
end

_capture_pixel_bounds(device, plan) = nothing

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{PixelCalibrationU16F32,<:PixelCalibrationPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 1 || throw(
        ArgumentError("PixelCalibrationU16F32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("PixelCalibrationU16F32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :raw),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :calibrated),
        output,
    )
    _require_capture_array("raw detector input", input, device)
    _require_capture_array("calibrated detector output", output, device)
    _require_capture_array("captured frame status", frame_status, device)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))

    flat = _capture_array(device, prepared.plan.flat)
    background = _capture_array(device, prepared.plan.background)
    backend = KernelAbstractions.get_backend(output)
    return PixelCalibration(
        _pixel_calibration!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        output,
        flat,
        background,
        input,
        _capture_pixel_bounds(device, prepared.plan),
    )
end

preflight(::PixelCalibration) = nothing

function enqueue!(captured::PixelCalibration)
    captured.calibrate!(
        captured.output,
        captured.flat,
        captured.background,
        captured.input;
        ndrange=length(captured.output),
    )
    return nothing
end

complete!(::PixelCalibration, ::Int32) = nothing
reset!(captured::PixelCalibration) = captured

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs

preserves_inputs(c::PixelCalibration) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.output))

# Finite calibration coefficients do not bound overflow; this adapter has no
# finite_output guarantee.

# Cold certificates for fixed Float32 calibration. No bound calculation runs
# during replay, and an unavailable certificate retains the checked adapter.
struct _PixelBounds
    magnitude::Float64
    quantum_exponent::Union{Nothing,Int}
end

const _CaptureBound = Union{_PixelBounds,_FiniteBound}

# The least significant set bit gives the exact dyadic quantum of a nonzero
# Float32 value. Zero imposes no lattice constraint.
function _dyadic_exponent(value::Float32)
    bits = reinterpret(UInt32, value) & 0x7fffffff
    iszero(bits) && return nothing
    isfinite(value) || return nothing
    encoded = Int(bits >> 23)
    fraction = bits & 0x007fffff
    mantissa = iszero(encoded) ? fraction : fraction | 0x00800000
    return (iszero(encoded) ? -149 : encoded - 150) + trailing_zeros(mantissa)
end

function _pixel_bounds(flat, background)
    _supported_bound_arithmetic() || return nothing
    magnitude = 0.0
    quantum = nothing
    for (coefficient, offset) in zip(flat, background)
        all(isfinite, (coefficient, offset)) || return nothing
        lower = FilterGraphAlgorithms._calibrated_pixel(UInt16(0), coefficient, offset)
        upper = FilterGraphAlgorithms._calibrated_pixel(typemax(UInt16), coefficient, offset)
        all(isfinite, (lower, upper)) || return nothing
        magnitude = max(magnitude, abs(Float64(lower)), abs(Float64(upper)))
        iszero(coefficient) && continue
        offset_quantum = _dyadic_exponent(offset)
        difference_quantum = isnothing(offset_quantum) ? 0 : min(0, offset_quantum)
        product_quantum = max(-149, difference_quantum + _dyadic_exponent(coefficient))
        quantum = isnothing(quantum) ? product_quantum : min(quantum, product_quantum)
    end
    return _PixelBounds(magnitude, quantum)
end


_capture_output_bound(c::PixelCalibration, ::Val{1}) = c.bounds
