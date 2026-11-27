module FilterGraphAlgorithmsDSPExt

using FilterGraphAlgorithms: FilterGraphAlgorithms, BiquadCascadePlan
import DSP
import FilterGraphAlgorithms: design_lowpass

function _response_db(coefficients, frequency_hz, sample_rate_hz)
    z = cis(-2π * (frequency_hz / sample_rate_hz))
    response = one(ComplexF64)
    for section in axes(coefficients, 2)
        b0, b1, b2, a1, a2 = Float64.(coefficients[:, section])
        response *= (b0 + z * (b1 + z * b2)) / (1 + z * (a1 + z * a2))
    end
    return 20log10(abs(response))
end

function _check_response(coefficients::Matrix{Float32}; sample_rate_hz,
    passband_hz, stopband_hz, passband_loss_db, stopband_attenuation_db,
    response_points, tolerance_db)
    BiquadCascadePlan(reshape(coefficients, 5, size(coefficients, 2), 1), 1)
    passband = [_response_db(coefficients, f, sample_rate_hz)
        for f in range(0, passband_hz; length=response_points)]
    stopband = [_response_db(coefficients, f, sample_rate_hz)
        for f in range(stopband_hz, sample_rate_hz / 2; length=response_points)]
    all(isfinite, passband) && all(x -> isfinite(x) || x == -Inf, stopband) ||
        throw(ArgumentError("rounded low-pass response is nonfinite"))
    gain = maximum(passband)
    loss = -minimum(passband)
    attenuation = -maximum(stopband)
    gain <= tolerance_db || throw(ArgumentError("rounded low-pass exceeds passband gain tolerance"))
    loss <= passband_loss_db + tolerance_db ||
        throw(ArgumentError("rounded low-pass exceeds passband loss specification"))
    attenuation >= stopband_attenuation_db - tolerance_db ||
        throw(ArgumentError("rounded low-pass fails stopband attenuation specification"))
    return (; passband_loss_db=loss, passband_gain_db=gain,
        stopband_attenuation_db=attenuation)
end

function design_lowpass(; sample_rate_hz, passband_hz, stopband_hz,
    passband_loss_db=1.0, stopband_attenuation_db=40.0, update_period=1,
    response_points=4097, tolerance_db=1e-3)
    fs, fp, fst, loss, attenuation, tolerance =
        Float64.((sample_rate_hz, passband_hz, stopband_hz,
            passband_loss_db, stopband_attenuation_db, tolerance_db))
    isfinite(fs) && fs > 0 || throw(ArgumentError("sample_rate_hz must be finite and positive"))
    isfinite(fp) && isfinite(fst) && 0 < fp < fst < fs / 2 ||
        throw(ArgumentError("require 0 < passband_hz < stopband_hz < sample_rate_hz/2"))
    isfinite(loss) && loss > 0 && isfinite(attenuation) && attenuation > loss ||
        throw(ArgumentError("require finite 0 < passband_loss_db < stopband_attenuation_db"))
    update_period isa Integer && update_period > 0 ||
        throw(ArgumentError("update_period must be a positive integer"))
    fst <= (fs / 2) / update_period ||
        throw(ArgumentError("stopband_hz exceeds the slower update Nyquist frequency"))
    response_points isa Integer && response_points >= 2 ||
        throw(ArgumentError("response_points must be an integer at least two"))
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("tolerance_db must be finite and nonnegative"))
    order, cutoff = DSP.buttord(fp / (fs / 2), fst / (fs / 2), loss, attenuation)
    sos = DSP.SecondOrderSections(DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.Butterworth(order)))
    coefficients = Matrix{Float32}(undef, 5, length(sos.biquads))
    for (section, biquad) in enumerate(sos.biquads)
        gain = section == 1 ? sos.g : one(sos.g)
        coefficients[:, section] .= (gain * biquad.b0, gain * biquad.b1,
            gain * biquad.b2, biquad.a1, biquad.a2)
    end
    measured = _check_response(coefficients; sample_rate_hz=fs, passband_hz=fp,
        stopband_hz=fst, passband_loss_db=loss, stopband_attenuation_db=attenuation,
        response_points, tolerance_db=tolerance)
    return (; coefficients, order, cutoff_hz=cutoff * (fs / 2), measured...)
end

end
