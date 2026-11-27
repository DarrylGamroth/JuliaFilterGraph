module DenseVectorReconstructionTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

const INPUT_SCHEMA = "org.example.ao.ordered-input/1"
const OUTPUT_SCHEMA = "org.example.ao.reconstructed-output/1"
const RECONSTRUCTOR_SCHEMA = "org.example.ao.dense-reconstructor/1"

function prepared_fixture(; reconstructor=Float32[1, 10, 100, -1, -10, -100], coefficient_storage="fp32")
    return prepare_algorithm(
        DenseVectorReconstructorF32,
        DenseVectorReconstructorF32Configuration(
            input_count=3,
            output_count=2,
            initial_reconstructor=reconstructor,
            coefficient_storage=coefficient_storage,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            reconstructor_schema=RECONSTRUCTOR_SCHEMA,
        ),
    )
end

function process_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

@testset "dense vector reconstruction declares explicit ordered-vector formats" begin
    prepared = prepared_fixture()

    @test algorithm_label(DenseVectorReconstructorF32) ==
          "dense-vector-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(DenseVectorReconstructorF32) ===
          DenseReconstructionFailure
    @test algorithm_operation(DenseVectorReconstructorF32) isa DenseF32Operation
    @test capture_support(DenseVectorReconstructorF32, nothing) isa NoCapture
    fields = algorithm_configuration(DenseVectorReconstructorF32)
    @test map(field -> field.name, fields) == (
        :input_count,
        :output_count,
        :initial_reconstructor,
        :input_schema,
        :output_schema,
        :reconstructor_schema,
        :coefficient_storage,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        (fields[1], fields[2], fields[4], fields[5], fields[6], fields[7]),
    )
    @test fields[3].update isa ConstructionConfiguration

    ports = algorithm_ports(DenseVectorReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("ordered-input"),
        :output,
        :reconstructor,
    )
    @test ports[2].metadata_source === Symbol("ordered-input")
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3,), (2,), (2, 3))
    @test getproperty.(formats, :schema) == (
        INPUT_SCHEMA,
        OUTPUT_SCHEMA,
        RECONSTRUCTOR_SCHEMA,
    )
    @test prepared.plan.matrix == Float32[1 10 100; -1 -10 -100]
end

@testset "dense vector reconstruction preserves declared order and replacement atomicity" begin
    prepared = prepared_fixture()
    input = Float32[2, 3, 4]
    output = zeros(Float32, 2)

    @test process!(output, prepared, input) === nothing
    @test output == Float32[432, -432]

    replacement = Float32[2 3 5; -2 4 1]
    @test replace_parameter!(prepared, "reconstructor", replacement) === prepared
    @test process!(output, prepared, input) === nothing
    @test output == Float32[33, 12]

    accepted_plan = prepared.plan
    accepted_output = copy(output)
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :reconstructor,
        zeros(Float32, 2, 2),
    )
    @test prepared.plan === accepted_plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :reconstructor,
        Float32[1 NaN 0; 0 1 0],
    )
    @test prepared.plan === accepted_plan
    @test process!(output, prepared, input) === nothing
    @test output == accepted_output
end

@testset "dense vector reconstruction keeps FP16 storage across parameter replacement" begin
    prepared = prepared_fixture(; coefficient_storage="fp16", reconstructor=nothing)
    @test prepared.plan.matrix.coefficients isa Matrix{Float16}
    input = Float32[2, 3, 4]
    output = zeros(Float32, 2)
    @test process!(output, prepared, input) === nothing
    @test all(iszero, output)

    replacement = Float32[2 3 5; -2 4 1]
    @test replace_parameter!(prepared, "reconstructor", replacement) === prepared
    @test prepared.plan.matrix.coefficients isa Matrix{Float16}
    @test process!(output, prepared, input) === nothing
    @test output == Float32[33, 12]
    @test process_allocation_bytes(output, prepared, input) == 0
    @test_throws ArgumentError prepared_fixture(; coefficient_storage="bf16")
end

@testset "dense vector reconstruction rejects nonfinite input atomically" begin
    prepared = prepared_fixture()
    input = Float32[2, NaN, 4]
    output = fill(71.0f0, 2)

    @test process!(output, prepared, input) === DenseReconstructionNonFiniteInput
    @test output == fill(71.0f0, 2)
end

@testset "dense vector reconstruction accepts views with zero warmed host allocations" begin
    prepared = prepared_fixture()
    input_storage = Float32[2, 0, 3, 0, 4, 0]
    output_storage = fill(71.0f0, 4)
    input = @view input_storage[1:2:5]
    output = @view output_storage[1:2:3]

    @test process!(output, prepared, input) === nothing
    @test output == Float32[432, -432]
    @test output_storage[2:2:4] == fill(71.0f0, 2)
    @test process_allocation_bytes(output, prepared, input) == 0
    @test @inferred(Union{Nothing,DenseReconstructionFailure}, process!(output, prepared, input)) ===
          nothing

    @test_throws ArgumentError process!(
        @view(input_storage[1:2]),
        prepared,
        @view(input_storage[1:3]),
    )
    @test_throws DimensionMismatch process!(output, prepared, zeros(Float32, 2))
end

@testset "dense vector reconstruction validates preparation" begin
    @test_throws ArgumentError prepare_algorithm(
        DenseVectorReconstructorF32,
        DenseVectorReconstructorF32Configuration(
            input_count=0,
            output_count=2,
            initial_reconstructor=nothing,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            reconstructor_schema=RECONSTRUCTOR_SCHEMA,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        DenseVectorReconstructorF32,
        DenseVectorReconstructorF32Configuration(
            input_count=3,
            output_count=2,
            initial_reconstructor=nothing,
            input_schema="invalid-schema",
            output_schema=OUTPUT_SCHEMA,
            reconstructor_schema=RECONSTRUCTOR_SCHEMA,
        ),
    )
end

@testset "dense vector reconstruction public API exports" begin
    for name in (
        :DenseVectorReconstructorF32,
        :DenseVectorReconstructorF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
