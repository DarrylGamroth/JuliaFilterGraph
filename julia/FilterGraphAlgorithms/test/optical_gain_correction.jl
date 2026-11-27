module OpticalGainCorrectionTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

function correction_prepared(gains::Vector{Float32}=Float32[1, 0.5, 0.25])
    return prepare_algorithm(
        OpticalGainCorrectionF32,
        OpticalGainCorrectionF32Configuration(optical_gains=gains),
    )
end

correction_allocation_bytes(output, prepared, input) =
    @allocated process!(output, prepared, input)

@testset "optical-gain correction preserves the frozen FGN contract" begin
    @test algorithm_label(OpticalGainCorrectionF32) ==
          "optical-gain-correction-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        OpticalGainCorrectionF32,
    ) === OpticalGainCorrectionFailure
    configuration = algorithm_configuration(OpticalGainCorrectionF32)
    @test map(field -> field.name, configuration) == (:optical_gains,)
    @test only(configuration).update isa ConstructionConfiguration

    ports = algorithm_ports(OpticalGainCorrectionF32)
    @test map(port -> port.name, ports) ==
          (:reconstructed, :corrected, Symbol("optical-gains"))
    @test ports[1].role isa FilterGraphAlgorithms.FrameData
    @test ports[2].role isa FilterGraphAlgorithms.FrameData
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    @test ports[2].metadata_source === :reconstructed

    prepared = correction_prepared()
    @test getproperty.(algorithm_formats(prepared), :shape) ==
          ((3,), (3,), (3,))
    @test getproperty.(algorithm_formats(prepared), :schema) == (
        RECONSTRUCTED_MODES_V1,
        RECONSTRUCTED_MODES_V1,
        PHYSICAL_OPTICAL_GAIN_V1,
    )
    @test optical_gains(prepared.plan) == Float32[1, 0.5, 0.25]
    @test optical_gain_reciprocal_corrections(prepared.plan) ==
          Float32[1, 2, 4]
end

@testset "correction divides by physical optical gain" begin
    prepared = correction_prepared()
    corrected = zeros(Float32, 3)
    @test process!(corrected, prepared, Float32[2, -2, 0.125]) === nothing
    @test corrected == Float32[2, -4, 0.5]

    replacement = Float32[0.5, 0.25, 0.125]
    @test replace_parameter!(prepared, "optical-gains", replacement) === prepared
    replacement .= 1
    @test optical_gains(prepared.plan) == Float32[0.5, 0.25, 0.125]
    @test process!(corrected, prepared, Float32[1, 1, 1]) === nothing
    @test corrected == Float32[2, 4, 8]
end

@testset "invalid gains and samples preserve caller-owned output" begin
    @test_throws ArgumentError OpticalGainCorrectionPlan(Float32[])
    for gain in (0.0f0, -1.0f0, NaN32, Inf32)
        @test_throws ArgumentError OpticalGainCorrectionPlan(Float32[gain])
    end
    @test_throws ArgumentError OpticalGainCorrectionPlan(
        Float32[reinterpret(Float32, UInt32(1))],
    )

    prepared = correction_prepared(Float32[1, 0.5])
    corrected = Float32[7, 8]
    @test process!(corrected, prepared, Float32[1, NaN]) ===
          OpticalGainCorrectionNonFiniteInput
    @test corrected == Float32[7, 8]
    @test_throws DimensionMismatch process!(corrected, prepared, Float32[1])
    @test corrected == Float32[7, 8]
    @test_throws ArgumentError process!(corrected, prepared, corrected)

    large_plan = OpticalGainCorrectionPlan(Float64[1, 0.5])
    large_output = Float64[7, 8]
    @test FilterGraphAlgorithms._process_optical_gain_correction!(
        large_output,
        large_plan,
        Float64[1, floatmax(Float64)],
    ) === OpticalGainCorrectionNonFiniteOutput
    @test large_output == Float64[7, 8]

    original_plan = prepared.plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "optical-gains",
        Float32[1, NaN],
    )
    @test prepared.plan === original_plan
end

@testset "logical views and warmed outcomes obey the array contract" begin
    prepared = correction_prepared(Float32[1, 0.5])
    input_storage = Float32[99, 2, -2, 99]
    output_storage = fill(71.0f0, 4)
    input = @view input_storage[2:3]
    output = @view output_storage[1:2:3]
    @test process!(output, prepared, input) === nothing
    @test output == Float32[2, -4]

    valid = Float32[2, -2]
    invalid = Float32[2, NaN]
    process!(output, prepared, valid)
    process!(output, prepared, invalid)
    @test correction_allocation_bytes(output, prepared, valid) == 0
    @test correction_allocation_bytes(output, prepared, invalid) == 0
end

@testset "optical-gain correction public API exports" begin
    for name in (
        :PHYSICAL_OPTICAL_GAIN_V1,
        :RECONSTRUCTED_MODES_V1,
        :OpticalGainCorrectionF32,
        :OpticalGainCorrectionF32Configuration,
        :OpticalGainCorrectionFailure,
        :OpticalGainCorrectionPlan,
        :optical_gain_reciprocal_corrections,
        :optical_gains,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
