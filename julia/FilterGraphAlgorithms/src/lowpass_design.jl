"""
    design_lowpass(; sample_rate_hz, passband_hz, stopband_hz,
        passband_loss_db=1.0, stopband_attenuation_db=40.0,
        update_period=1, response_points=4097, tolerance_db=1e-3)

Prepare Butterworth low-pass coefficients using the optional DSP.jl extension.
Load `DSP` before calling this function. Design runs during preparation only;
the resulting coefficients work with `BiquadCascadePlan` without DSP at runtime.

Frequencies are in Hz at the input sample rate. Require finite positive
`sample_rate_hz`, `0 < passband_hz < stopband_hz < sample_rate_hz/2`, positive
finite passband loss, and finite stopband attenuation greater than passband loss.
`update_period` must be a positive integer; `stopband_hz` must be at or below
`sample_rate_hz/(2*update_period)`. This condition places the stopband edge at
or below the slower update's Nyquist frequency; it does not prove alias rejection
for a particular signal. `response_points` is an integer at least two.

Return a named tuple containing owned `coefficients::Matrix{Float32}` of shape
`(5, section_count)` with rows `(b0, b1, b2, a1, a2)` and normalized `a0 = 1`,
the Butterworth `order`, `cutoff_hz`, and measured `passband_loss_db` (maximum
loss), `passband_gain_db` (maximum gain), and `stopband_attenuation_db` (minimum
attenuation). Any overall SOS gain is included in the numerator coefficients.

Check finiteness and strict section stability after rounding to Float32. Measure
the rounded cascade in ComplexF64 on `response_points` evenly spaced frequencies
in each of the passband and stopband, including DC, both band edges, and Nyquist.
Reject a design whose maximum passband gain exceeds `tolerance_db`, maximum
passband loss exceeds its specification plus `tolerance_db`, or minimum stopband
attenuation falls below its specification minus `tolerance_db`. The finite,
nonnegative tolerance is a numerical acceptance tolerance, not a design margin.
These sampled magnitude and section stability checks do not guarantee continuous
band behavior, feedback phase margin, or absence of arithmetic overflow.
"""
function design_lowpass end
