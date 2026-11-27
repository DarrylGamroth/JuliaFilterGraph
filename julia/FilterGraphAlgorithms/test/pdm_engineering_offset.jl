module PdmEngineeringOffsetTests

using FilterGraphAlgorithms
using Test
import FilterGraphAlgorithms: process!

prepared_offset(count) = prepare_algorithm(
    PdmEngineeringOffsetF32,
    PdmEngineeringOffsetF32Configuration(actuator_count=count),
)
allocation_bytes(output, current, prepared, requested) =
    @allocated process!(output, current, prepared, requested)

@testset "Engineering offset declaration and adopted telemetry" begin
    @test algorithm_label(PdmEngineeringOffsetF32) == "pdm-engineering-offset-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PdmEngineeringOffsetF32) ===
          PdmEngineeringOffsetFailure
    @test map(field -> field.name, algorithm_configuration(PdmEngineeringOffsetF32)) ==
          (:actuator_count,)
    ports = algorithm_ports(PdmEngineeringOffsetF32)
    @test map(port -> port.name, ports) == (
        :requested, Symbol("requested-with-offset"), Symbol("current-offset"),
        Symbol("engineering-offset"),
    )
    @test ports[2].metadata_source === :requested
    @test ports[3].metadata_source === :requested
    @test ports[4].direction isa FilterGraphAlgorithms.InputPort
    @test ports[4].role isa FilterGraphAlgorithms.SparseParameter
    prepared = prepared_offset(3)
    @test getproperty.(algorithm_formats(prepared), :schema) == (
        REQUESTED_PDM_COMMAND_V1, REQUESTED_PDM_COMMAND_V1,
        PDM_ENGINEERING_OFFSET_V1, PDM_ENGINEERING_OFFSET_V1,
    )
    @test all(format -> format.shape == (3,), algorithm_formats(prepared))
    requested = Float32[1, 2, 3]
    output = zeros(Float32, 3)
    current = fill(9.0f0, 3)
    @test process!(output, current, prepared, requested) === nothing
    @test output == requested
    @test current == zeros(Float32, 3)
    offset = Float32[0.5, -2, 3]
    @test replace_parameter!(prepared, "engineering-offset", offset) === prepared
    offset[1] = 99
    @test process!(output, current, prepared, requested) === nothing
    @test output == Float32[1.5, 0, 6]
    @test current == Float32[0.5, -2, 3]
    replace_parameter!(prepared, "engineering-offset", Float32[-1, 0, 1])
    @test process!(output, current, prepared, requested) === nothing
    @test current == Float32[-1, 0, 1]
    @test output == Float32[0, 2, 4]
    @test_throws ArgumentError prepared_offset(0)
end

@testset "Zero offset preserves requested bits and limiter behavior" begin
    requested = Float32[-0.0, 0.0, nextfloat(0.0f0), -0.45, 0.6]
    prepared = prepared_offset(length(requested))
    output = similar(requested)
    current = similar(requested)
    @test process!(output, current, prepared, requested) === nothing
    @test reinterpret(UInt32, output) == reinterpret(UInt32, requested)
    replace_parameter!(prepared, "engineering-offset", fill(-0.0f0, length(requested)))
    @test process!(output, current, prepared, requested) === nothing
    @test reinterpret(UInt32, output) == reinterpret(UInt32, requested)
    @test all(signbit, current)

    limiter = PdmCommandPlan(fill(-0.25f0, 5), fill(0.25f0, 5))
    first_workspace = PdmCommandWorkspace(limiter, zeros(Float32, 5))
    second_workspace = PdmCommandWorkspace(limiter, zeros(Float32, 5))
    direct = zeros(Float32, 5)
    composed = similar(direct)
    first_feedback = similar(direct)
    second_feedback = similar(direct)
    process!(direct, first_feedback, first_workspace, limiter, requested)
    process!(composed, second_feedback, second_workspace, limiter, output)
    @test reinterpret(UInt32, direct) == reinterpret(UInt32, composed)
    @test reinterpret(UInt32, first_feedback) == reinterpret(UInt32, second_feedback)

    replace_parameter!(prepared, "engineering-offset", Float32[0, 0, 0, 0.5, -0.5])
    process!(output, current, prepared, requested)
    process!(composed, second_feedback, second_workspace, limiter, output)
    @test composed[4:5] ≈ Float32[0.05, 0.1]
end

@testset "Engineering offset saturation feeds back the composed requested command" begin
    requested = Float32[0.125, -0.125]
    offset = Float32[0.5, -0.5]
    prepared = prepared_offset(2)
    limiter = PdmCommandPlan(fill(-0.25f0, 2), fill(0.25f0, 2))
    workspace = PdmCommandWorkspace(limiter, zeros(Float32, 2))
    demanded = zeros(Float32, 2)
    feedback = similar(demanded)
    @test process!(demanded, feedback, workspace, limiter, requested) === nothing
    @test demanded == requested
    @test feedback == zeros(Float32, 2)

    replace_parameter!(prepared, "engineering-offset", offset)
    composed_requested = similar(requested)
    current_offset = similar(requested)
    @test process!(composed_requested, current_offset, prepared, requested) === nothing
    @test composed_requested == requested + offset
    @test current_offset == offset
    @test process!(demanded, feedback, workspace, limiter, composed_requested) === nothing
    @test demanded == Float32[0.25, -0.25]
    @test feedback == composed_requested - demanded
    @test feedback == Float32[0.375, -0.375]
end

@testset "Invalid replacement and frame are failure atomic" begin
    @test_throws ArgumentError PdmEngineeringOffsetPlan(Float32[])
    @test_throws ArgumentError PdmEngineeringOffsetPlan(Float32[Inf])
    prepared = prepared_offset(3)
    original = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(prepared, "engineering-offset", Float32[1])
    for value in (NaN32, Inf32, -Inf32)
        @test_throws ArgumentError replace_parameter!(prepared, "engineering-offset", Float32[0, 0, value])
        @test prepared.plan === original
    end
    output = fill(71.0f0, 3)
    current = fill(72.0f0, 3)
    for value in (NaN32, Inf32, -Inf32)
        @test process!(output, current, prepared, Float32[1, 2, value]) ===
              PdmEngineeringOffsetNonFiniteInput
        @test output == fill(71.0f0, 3)
        @test current == fill(72.0f0, 3)
    end
    replace_parameter!(prepared, "engineering-offset", Float32[0, 0, floatmax(Float32)])
    @test process!(output, current, prepared, Float32[1, 2, floatmax(Float32)]) ===
          PdmEngineeringOffsetNonFiniteOutput
    @test output == fill(71.0f0, 3)
    @test current == fill(72.0f0, 3)
    @test_throws DimensionMismatch process!(output, current[1:2], prepared, zeros(Float32, 3))
    @test_throws DimensionMismatch process!(output[1:2], current, prepared, zeros(Float32, 3))
    @test_throws DimensionMismatch process!(output, current, prepared, zeros(Float32, 2))
    @test_throws ArgumentError process!(output, output, prepared, zeros(Float32, 3))
    @test_throws ArgumentError process!(output, current, prepared, @view output[1:3])
    @test output == fill(71.0f0, 3)
    @test current == fill(72.0f0, 3)
end

@testset "Logical views and exact input alias are supported" begin
    prepared = prepared_offset(2)
    replace_parameter!(prepared, "engineering-offset", Float32[0.5, -0.25])
    storage = Float32[1, 99, 2, 99]
    output_storage = fill(71.0f0, 4)
    current_storage = fill(72.0f0, 4)
    input = @view storage[1:2:3]
    output = @view output_storage[2:2:4]
    current = @view current_storage[1:2:3]
    @test process!(output, current, prepared, input) === nothing
    @test output == Float32[1.5, 1.75]
    @test current == Float32[0.5, -0.25]
    @test output_storage[1:2:3] == fill(71.0f0, 2)
    @test current_storage[2:2:4] == fill(72.0f0, 2)
    @test process!(input, current, prepared, input) === nothing
    @test input == Float32[1.5, 1.75]
end

@testset "Warmed engineering offset frames allocate zero bytes" begin
    prepared = prepared_offset(277)
    requested = fill(1.0f0, 277)
    output = similar(requested)
    current = similar(requested)
    for offset in (zeros(Float32, 277), fill(0.5f0, 277))
        replace_parameter!(prepared, "engineering-offset", offset)
        process!(output, current, prepared, requested)
        @test allocation_bytes(output, current, prepared, requested) == 0
    end
    requested[end] = NaN32
    process!(output, current, prepared, requested)
    @test allocation_bytes(output, current, prepared, requested) == 0
    requested[end] = floatmax(Float32)
    replace_parameter!(prepared, "engineering-offset", fill(floatmax(Float32), 277))
    process!(output, current, prepared, requested)
    @test allocation_bytes(output, current, prepared, requested) == 0
end

end
