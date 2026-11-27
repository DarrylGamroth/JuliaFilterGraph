module SccReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "scc_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_scc_reconstructor()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        SccReconstructorF32,
        SccReconstructorF32Configuration(
            controller_coordinate_count=configuration["controller_coordinate_count"],
            selected_pixel_count=configuration["selected_pixel_count"],
            initial_reconstructor=Float32.(
                configuration["initial_reconstructor"],
            ),
        ),
    )
end

function scc_reconstruction_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

@testset "SCC reconstructor declares the pixel-to-controller boundary" begin
    @test algorithm_label(SccReconstructorF32) == "scc-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(SccReconstructorF32) ===
          DenseReconstructionFailure
    @test map(field -> field.name, algorithm_configuration(SccReconstructorF32)) == (
        :controller_coordinate_count,
        :selected_pixel_count,
        :initial_reconstructor,
        :coefficient_storage,
    )
    ports = algorithm_ports(SccReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("selected-residuals"),
        Symbol("controller-residual-error"),
        :reconstructor,
    )
    @test ports[2].metadata_source === Symbol("selected-residuals")
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter

    formats = algorithm_formats(prepared_scc_reconstructor())
    @test getproperty.(formats, :shape) == ((4,), (2,), (2, 4))
    @test getproperty.(formats, :schema) == (
        SCC_SELECTED_PIXEL_RESIDUALS_V1,
        CONTROLLER_RESIDUAL_ERROR_V1,
        SCC_RECONSTRUCTOR_V1,
    )
end

@testset "SCC reconstruction matches an independent row-oriented oracle" begin
    prepared = prepared_scc_reconstructor()
    sample = FIXTURE["sample"]
    input = Float32.(sample["selected_residuals"])
    output = zeros(Float32, 2)
    @test process!(output, prepared, input) === nothing
    @test output == Float32.(sample["controller_residual_error"])

    replacement = FIXTURE["replacement"]
    matrix = row_major_matrix(Float32, replacement["reconstructor"], 2, 4)
    @test replace_parameter!(prepared, :reconstructor, matrix) === prepared
    @test process!(output, prepared, input) === nothing
    @test output == Float32.(replacement["controller_residual_error"])

    legacy_selected_pixel_to_output = transpose(matrix)
    @test output == transpose(legacy_selected_pixel_to_output) * input
end

@testset "SCC reconstructor rejects invalid inputs and supports views" begin
    prepared = prepared_scc_reconstructor()
    output_storage = fill(71.0f0, 4)
    output = @view output_storage[1:2:3]
    input_storage = Float32[2, 99, 5, 99, 0, 99, 0]
    input = @view input_storage[1:2:7]
    @test process!(output, prepared, input) === nothing
    @test output == Float32[12, 0.5]
    @test output_storage[2:2:4] == fill(71.0f0, 2)
    @test scc_reconstruction_allocation_bytes(output, prepared, input) == 0

    fill!(output, 71.0f0)
    input[2] = NaN32
    @test process!(output, prepared, input) ===
          DenseReconstructionNonFiniteInput
    @test output == fill(71.0f0, 2)
    @test scc_reconstruction_allocation_bytes(output, prepared, input) == 0

    @test_throws ArgumentError process!(
        @view(input_storage[1:2]),
        prepared,
        @view(input_storage[1:4]),
    )
    @test_throws DimensionMismatch process!(
        output,
        prepared,
        zeros(Float32, 3),
    )
end

@testset "SCC reconstructor preparation is validated" begin
    @test_throws ArgumentError prepare_algorithm(
        SccReconstructorF32,
        SccReconstructorF32Configuration(
            controller_coordinate_count=0,
            selected_pixel_count=4,
            initial_reconstructor=nothing,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        SccReconstructorF32,
        SccReconstructorF32Configuration(
            controller_coordinate_count=2,
            selected_pixel_count=4,
            initial_reconstructor=zeros(Float32, 7),
        ),
    )
    prepared = prepared_scc_reconstructor()
    original_plan = prepared.plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :reconstructor,
        Float32[1 NaN 0 0; 0 1 0 0],
    )
    @test prepared.plan === original_plan

    overflowing = prepare_algorithm(
        SccReconstructorF32,
        SccReconstructorF32Configuration(
            controller_coordinate_count=2,
            selected_pixel_count=2,
            initial_reconstructor=nothing,
        ),
    )
    replace_parameter!(
        overflowing,
        :reconstructor,
        fill(floatmax(Float32), 2, 2),
    )
    output = zeros(Float32, 2)
    input = Float32[2, 2]
    @test process!(output, overflowing, input) ===
          DenseReconstructionNonFiniteOutput
    @test scc_reconstruction_allocation_bytes(output, overflowing, input) == 0
end

@testset "SCC reconstructor public API exports" begin
    for name in (
        :SccReconstructorF32,
        :SccReconstructorF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
