module PixelCalibrationF32Tests
using Test, FilterGraphAlgorithms
import FilterGraphAlgorithms: process!, replace_parameter!

function prepare_pixels(; size=(2,3), schema=DETECTOR_INPUT_EQUIVALENT_ELECTRONS_V1)
    prepare_algorithm(PixelCalibrationF32, PixelCalibrationF32Configuration(
        image_size=size, input_schema=schema, initial_flat=nothing, initial_background=nothing))
end

@testset "signed Float32 detector calibration" begin
    p = prepare_pixels()
    raw = Float32[-0.5 1 2; 3 4 5]
    output = zeros(Float32, 2, 3)
    @test process!(output, p, raw) === nothing
    @test output == raw
    @test first(algorithm_formats(p)).schema == DETECTOR_INPUT_EQUIVALENT_ELECTRONS_V1
    @test algorithm_formats(p)[2].schema == CALIBRATED_PIXELS_V1
    replace_parameter!(p, :background, fill(0.5f0, 2, 3))
    replace_parameter!(p, :flat, Float32[1 2 3; 4 5 6])
    @test process!(output, p, raw) === nothing
    @test output == Float32[1 2 3; 4 5 6] .* (raw .- 0.5f0)
    for value in (NaN32, Inf32, -Inf32)
        bad = copy(raw); bad[end] = value
        fill!(output, 77f0)
        @test process!(output, p, bad) === PixelCalibrationNonFiniteInput
        @test all(==(77f0), output)
    end
    raw[end] = floatmax(Float32)
    @test process!(output, p, raw) === PixelCalibrationNonFiniteOutput
    @test all(==(77f0), output)
    @test_throws DimensionMismatch process!(zeros(Float32,3,2), p, raw)
    @test_throws ArgumentError prepare_pixels(;size=(0,3))
    @test_throws ArgumentError prepare_pixels(;schema=" ")
    raw[end] = 5f0
    storage = fill(100f0,4,6)
    target = @view storage[1:2:4,1:2:6]
    @test process!(target,p,raw) === nothing
    @test target == Float32[1 2 3; 4 5 6] .* (raw .- 0.5f0)
    @test all(==(100f0), storage[2:2:4,:])
    measure(output,p,raw) = @allocated process!(output,p,raw)
    measure(output,p,raw)
    @test measure(output,p,raw) == 0
end

@testset "Float32 calibration retains logical failure precedence" begin
    prepared = prepare_pixels()
    replace_parameter!(prepared, :flat, fill(2f0, 2, 3))
    output = fill(77f0, 2, 3)
    raw = ones(Float32, 2, 3)
    # Column-major iteration reaches [2,1] before [1,2].
    raw[2, 1] = floatmax(Float32)
    raw[1, 2] = NaN32
    @test process!(output, prepared, raw) === PixelCalibrationNonFiniteOutput
    @test all(==(77f0), output)
    raw[2, 1] = NaN32
    raw[1, 2] = floatmax(Float32)
    @test process!(output, prepared, raw) === PixelCalibrationNonFiniteInput
    @test all(==(77f0), output)

    storage = fill(NaN32, 4, 6)
    noncontiguous_raw = @view storage[1:2:4, 1:2:6]
    noncontiguous_raw .= Float32[-0.5 1 2; 3 4 5]
    @test process!(output, prepared, noncontiguous_raw) === nothing
    @test output == 2f0 .* noncontiguous_raw
end

@testset "Float32 calibration retains aliased calibration scratch behavior" begin
    plan = FilterGraphAlgorithms.PixelCalibrationPlan(fill(2f0, 2, 3), zeros(Float32, 2, 3))
    workspace = FilterGraphAlgorithms.PixelCalibrationWorkspace(plan.flat)
    output = fill(77f0, 2, 3)
    raw = ones(Float32, 2, 3)
    raw[1] = 3f0
    raw[2] = NaN32
    @test FilterGraphAlgorithms._process_pixel_calibration_f32!(
        output, workspace, plan, raw,
    ) === PixelCalibrationNonFiniteInput
    @test all(==(77f0), output)
    @test plan.flat[1] == 6f0
    @test all(==(2f0), plan.flat[2:end])
    fill!(raw, 1f0)
    retained_flat = copy(plan.flat)
    @test FilterGraphAlgorithms._process_pixel_calibration_f32!(
        output, workspace, plan, raw,
    ) === nothing
    @test output == retained_flat
end

@testset "Float32 calibration retains rejected scratch across tile boundaries" begin
    prepared = prepare_pixels(; size=(1025, 1))
    replace_parameter!(prepared, :flat, fill(2f0, 1025, 1))
    replace_parameter!(prepared, :background, fill(0.25f0, 1025, 1))
    raw = fill(3f0, 1025, 1)
    output = fill(77f0, size(raw))
    bits(a) = reinterpret(UInt32, vec(a))
    for position in (1, 255, 256, 257, 511, 512, 513, 1025)
        for value in (NaN32, Inf32, -Inf32, floatmax(Float32))
            raw[position] = value
            expected = 2f0 .* (raw .- 0.25f0)
            failure = isfinite(value) ? PixelCalibrationNonFiniteOutput : PixelCalibrationNonFiniteInput
            @test process!(output, prepared, raw) === failure
            @test all(==(77f0), output)
            @test bits(prepared.workspace.calibrated) == bits(expected)
            raw[position] = 3f0
        end
    end
    raw[256] = floatmax(Float32)
    raw[257] = NaN32
    @test process!(output, prepared, raw) === PixelCalibrationNonFiniteOutput
    @test bits(prepared.workspace.calibrated) == bits(2f0 .* (raw .- 0.25f0))
    raw[256] = NaN32
    raw[257] = floatmax(Float32)
    @test process!(output, prepared, raw) === PixelCalibrationNonFiniteInput
    @test bits(prepared.workspace.calibrated) == bits(2f0 .* (raw .- 0.25f0))
    fill!(raw, 3f0)
    @test process!(output, prepared, raw) === nothing
    @test all(==(5.5f0), output)
    measure() = @allocated process!(output, prepared, raw)
    measure()
    @test measure() == 0
end
@testset "Float32 calibration exact bits, tails, and admitted aliases" begin
    bits(a) = reinterpret(UInt32, vec(a))
    # Include signed zeros, subnormals, adjacent normals, and extreme finite
    # values; a fused multiply-add or precomputed flat*background is not an oracle.
    corpus = reinterpret(Float32, UInt32[0, 0x80000000, 1, 0x80000001,
        0x007fffff, 0x00800000, 0x3f7fffff, 0x3f800001,
        0xbf800001, 0x7f7fffff, 0xff7fffff])
    for count in (1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025)
        raw = reshape([corpus[mod1(i, length(corpus))] for i in 1:count], count, 1)
        flat = reshape(Float32[isodd(i) ? 0.5f0 : 1.0000001f0 for i in 1:count], count, 1)
        background = reshape(Float32[isodd(i) ? -0.0f0 : 0.125f0 for i in 1:count], count, 1)
        plan = FilterGraphAlgorithms.PixelCalibrationPlan(flat, background)
        workspace = FilterGraphAlgorithms.PixelCalibrationWorkspace(zeros(Float32, count, 1))
        output = fill(77f0, count, 1)
        expected = similar(raw)
        failure = nothing
        for i in eachindex(raw)
            expected[i] = flat[i] * (raw[i] - background[i])
            if isnothing(failure)
                failure = !isfinite(raw[i]) ? PixelCalibrationNonFiniteInput :
                    !isfinite(expected[i]) ? PixelCalibrationNonFiniteOutput : nothing
            end
        end
        @test FilterGraphAlgorithms._process_pixel_calibration_f32!(output, workspace, plan, raw) === failure
        @test bits(workspace.calibrated) == bits(expected)
        @test isnothing(failure) ? bits(output) == bits(expected) : all(==(77f0), output)
        # Output may alias raw: publication occurs after staging and validation.
        inplace = copy(raw)
        @test FilterGraphAlgorithms._process_pixel_calibration_f32!(inplace, workspace, plan, inplace) === failure
        @test bits(inplace) == bits(isnothing(failure) ? expected : raw)
        @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration_f32!(output, workspace, plan, workspace.calibrated)
        @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration_f32!(plan.flat, workspace, plan, raw)
        @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration_f32!(plan.background, workspace, plan, raw)
        @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration_f32!(workspace.calibrated, workspace, plan, raw)
    end
end

end
