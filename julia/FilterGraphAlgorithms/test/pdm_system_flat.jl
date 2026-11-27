module PdmSystemFlatTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

function system_flat_prepared(actuator_count::Int)
    return prepare_algorithm(
        PdmSystemFlatF32,
        PdmSystemFlatF32Configuration(actuator_count=actuator_count),
    )
end

system_flat_allocation_bytes(output, prepared, input) =
    @allocated process!(output, prepared, input)

@testset "PDM system-flat declaration and addition preserve the FGN contract" begin
    @test algorithm_label(PdmSystemFlatF32) == "pdm-system-flat-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PdmSystemFlatF32) ===
          PdmSystemFlatFailure
    @test map(field -> field.name, algorithm_configuration(PdmSystemFlatF32)) ==
          (:actuator_count,)
    @test first(algorithm_configuration(PdmSystemFlatF32)).update isa
          ConstructionConfiguration

    ports = algorithm_ports(PdmSystemFlatF32)
    @test map(port -> port.name, ports) == (
        :requested,
        Symbol("requested-with-system-flat"),
        Symbol("system-flat"),
    )
    @test ports[2].metadata_source === :requested

    prepared = system_flat_prepared(3)
    @test getproperty.(algorithm_formats(prepared), :schema) == (
        REQUESTED_PDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
        PDM_SYSTEM_FLAT_V1,
    )
    @test process!(zeros(Float32, 3), prepared, Float32[1, 2, 3]) === nothing
    output = zeros(Float32, 3)
    @test process!(output, prepared, Float32[1, 2, 3]) === nothing
    @test output == Float32[1, 2, 3]

    @test replace_parameter!(prepared, "system-flat", Float32[0.5, -2, 3]) ===
          prepared
    @test process!(output, prepared, Float32[1, 2, 3]) === nothing
    @test output == Float32[1.5, 0, 6]
end

@testset "PDM system-flat rejects invalid values without modifying output" begin
    supplied = Float32[1, 2]
    owned = PdmSystemFlatPlan(supplied)
    supplied[1] = 9
    @test owned.system_flat == Float32[1, 2]
    @test_throws ArgumentError PdmSystemFlatPlan(Float32[])
    @test_throws ArgumentError PdmSystemFlatPlan(Float32[NaN])

    prepared = system_flat_prepared(2)
    original_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "system-flat",
        Float32[1],
    )
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "system-flat",
        Float32[NaN, 0],
    )
    @test prepared.plan === original_plan

    output = fill(71.0f0, 2)
    @test process!(output, prepared, Float32[NaN, 0]) === PdmSystemFlatNonFiniteInput
    @test output == fill(71.0f0, 2)

    replace_parameter!(
        prepared,
        "system-flat",
        Float32[floatmax(Float32), 0],
    )
    @test process!(output, prepared, Float32[floatmax(Float32), 0]) ===
          PdmSystemFlatNonFiniteOutput
    @test output == fill(71.0f0, 2)
end

@testset "PDM system-flat processing is allocation-free after warmup" begin
    prepared = system_flat_prepared(2)
    replace_parameter!(prepared, "system-flat", Float32[0.5, -0.25])
    output = zeros(Float32, 2)
    input = Float32[1, 2]
    process!(output, prepared, input)
    @test system_flat_allocation_bytes(output, prepared, input) == 0
end

end
