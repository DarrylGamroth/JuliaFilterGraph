module LowpassDesignTests

using Test
using FilterGraphAlgorithms
import DSP
import FilterGraphAlgorithms: process!, commit!

const EXT = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsDSPExt)
const SPEC = (; sample_rate_hz=1000.0, passband_hz=40.0, stopband_hz=100.0)

@testset "optional low-pass design" begin
    @test EXT !== nothing
    @test Base.isexported(FilterGraphAlgorithms, :design_lowpass)
    design = design_lowpass(; SPEC...)
    coefficients = design.coefficients
    scaled = design_lowpass(sample_rate_hz=1e308, passband_hz=4e306, stopband_hz=1e307)
    @test scaled.coefficients ≈ coefficients
    @test coefficients isa Matrix{Float32}
    @test size(coefficients) == (5, cld(design.order, 2))
    @test design.passband_loss_db <= 1.001
    @test design.passband_gain_db <= 0.001
    @test design.stopband_attenuation_db >= 39.999
    @test all(isfinite, coefficients)
    plan = BiquadCascadePlan(reshape(coefficients, 5, :, 1), 1)
    @test plan.coefficients !== coefficients
    original = copy(plan.coefficients)
    coefficients[1] *= 2
    @test plan.coefficients == original
    coefficients[1] /= 2
    @test design_lowpass(; SPEC...).coefficients !== coefficients

    order, cutoff = DSP.buttord(0.08, 0.2, 1.0, 40.0)
    reference = DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.Butterworth(order))
    sos = DSP.SecondOrderSections(reference)
    @test design.order == order
    @test design.cutoff_hz ≈ 500cutoff
    @test coefficients[:, 1] ≈ Float32[sos.g*sos.biquads[1].b0,
        sos.g*sos.biquads[1].b1, sos.g*sos.biquads[1].b2,
        sos.biquads[1].a1, sos.biquads[1].a2]
    frequencies = sort!(unique!([collect(range(0, 500; length=10001)); 40; 100]))
    measured = [EXT._response_db(coefficients, f, 1000.0) for f in frequencies]
    expected = 20log10.(abs.(DSP.freqresp(reference, 2π .* frequencies ./ 1000)))
    # Compare linear magnitude, including the zero at Nyquist.
    @test 10 .^ (measured ./ 20) ≈ 10 .^ (expected ./ 20) rtol=3e-5 atol=2e-6
    @test maximum(measured[frequencies .<= 40]) <= 0.001
    @test minimum(measured[frequencies .<= 40]) >= -1.001
    @test maximum(measured[frequencies .>= 100]) <= -39.999

    impulse = zeros(Float32, 256)
    impulse[1] = 1
    workspace = BiquadCascadeWorkspace(plan)
    actual = similar(impulse)
    input, output = zeros(Float32, 1), zeros(Float32, 1)
    for frame in eachindex(impulse)
        input[1] = impulse[frame]
        @test process!(output, workspace, plan, input) === output
        actual[frame] = output[1]
        commit!(workspace)
    end
    rounded_sos = DSP.SecondOrderSections([
        DSP.Biquad(coefficients[:, section]...) for section in axes(coefficients, 2)], 1f0)
    @test actual ≈ DSP.filt(rounded_sos, impulse) rtol=2e-5 atol=2e-7
    for period in (1, 2, 4)
        @test design_lowpass(; SPEC..., update_period=period).coefficients == coefficients
    end
end

@testset "low-pass parameter and rounded-response rejection" begin
    for invalid in ((;sample_rate_hz=0), (;sample_rate_hz=Inf),
        (;sample_rate_hz=NaN), (;passband_hz=0), (;passband_hz=100),
        (;passband_hz=NaN), (;stopband_hz=500), (;stopband_hz=Inf),
        (;passband_loss_db=0), (;passband_loss_db=Inf),
        (;stopband_attenuation_db=1), (;stopband_attenuation_db=NaN),
        (;update_period=0), (;update_period=1.5), (;update_period=6),
        (;response_points=1), (;response_points=2.5),
        (;tolerance_db=-1), (;tolerance_db=Inf))
        @test_throws ArgumentError design_lowpass(; merge(SPEC, invalid)...)
    end
    @test design_lowpass(; SPEC..., update_period=5, response_points=2).order > 0
    coefficients = design_lowpass(; SPEC...).coefficients
    check(c) = EXT._check_response(c; SPEC..., passband_loss_db=1.0,
        stopband_attenuation_db=40.0, response_points=4097, tolerance_db=1e-3)
    identity = reshape(Float32[1, 0, 0, 0, 0], 5, 1)
    @test_throws ArgumentError check(identity)
    @test_throws ArgumentError design_lowpass(sample_rate_hz=1000,
        passband_hz=1e-6, stopband_hz=1e-5)
    amplified = copy(coefficients)
    amplified[1:3, 1] .*= 2
    @test_throws ArgumentError check(amplified)
    attenuated = copy(coefficients)
    attenuated[1:3, 1] .*= 0.1f0
    @test_throws ArgumentError check(attenuated)
    unstable = copy(coefficients)
    unstable[5, 1] = 1f0
    @test_throws ArgumentError check(unstable)
    nonfinite = copy(coefficients)
    nonfinite[1] = Inf32
    @test_throws ArgumentError check(nonfinite)
end

end
