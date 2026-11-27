# Shared Float32 certificate representation and arithmetic assumptions used by
# calibration, pupil normalization, and projection bounds.

struct _FiniteBound
    magnitude::Float64
end

_supported_bound_arithmetic() = !get_zero_subnormals() && rounding(Float32) == RoundNearest

# Strict power-of-two envelopes avoid inward-rounded Float64 certificate sums.
# Each supported operand below is a Float32 value times a power of two, so its
# Float64 representation is exact. The extra exponent covers Float32 rounding.
_power_bound(value::Float64) = iszero(value) ? 0.0 : ldexp(1.0, exponent(value) + 1)
