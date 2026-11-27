module PyramidReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_fixture()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PyramidReconstructorF32,
        PyramidReconstructorF32Configuration(
            reconstructed_count=configuration["reconstructed_count"],
            selected_coordinate_count=configuration["selected_coordinate_count"],
            initial_reconstructor=Float32.(
                configuration["initial_reconstructor"],
            ),
            reconstructed_schema=configuration["reconstructed_schema"],
        ),
    )
end

function checked_allocation_bytes(reconstructed, prepared, pixels)
    return @allocated process!(reconstructed, prepared, pixels)
end

function admitted_allocation_bytes(reconstructed, prepared, pixels)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        reconstructed,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "pyramid I4Q reconstruction declaration" begin
    prepared = prepared_fixture()
    configuration = FIXTURE["configuration"]

    @test algorithm_label(PyramidReconstructorF32) == "pwfs-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PyramidReconstructorF32) ===
          DenseReconstructionFailure
    fields = algorithm_configuration(PyramidReconstructorF32)
    @test map(field -> field.name, fields) == (
        :reconstructed_count,
        :selected_coordinate_count,
        :initial_reconstructor,
        :reconstructed_schema,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        (fields[1], fields[2], fields[4]),
    )
    @test fields[3].update isa ConstructionConfiguration

    ports = algorithm_ports(PyramidReconstructorF32)
    @test map(port -> port.name, ports) == (
        :signal,
        :reconstructed,
        :reconstructor,
    )
    @test ports[2].metadata_source === :signal
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2, 2), (3,), (3, 4))
    @test getproperty.(formats, :schema) == (
        PYRAMID_I4Q_SIGNAL_V1,
        configuration["reconstructed_schema"],
        PYRAMID_I4Q_RECONSTRUCTOR_V1,
    )
end

@testset "packed pyramid I4Q signals match the frozen oracle" begin
    configuration = FIXTURE["configuration"]
    pixels = row_major_matrix(
        Float32,
        FIXTURE["sample"]["signal"],
        configuration["selected_coordinate_count"],
        2,
    )
    reconstructed = zeros(Float32, configuration["reconstructed_count"])
    prepared = prepared_fixture()
    @test process!(reconstructed, prepared, pixels) === nothing
    @test reconstructed == Float32.(FIXTURE["sample"]["reconstructed"])

    replacement = FIXTURE["replacement"]
    matrix = row_major_matrix(
        Float32,
        replacement["reconstructor"],
        configuration["reconstructed_count"],
        2 * configuration["selected_coordinate_count"],
    )
    @test replace_parameter!(prepared, :reconstructor, matrix) === prepared
    @test process!(reconstructed, prepared, pixels) === nothing
    @test reconstructed == Float32.(replacement["reconstructed"])
end

@testset "pyramid rejection, views, and warmed allocation contract" begin
    prepared = prepared_fixture()
    pixels = row_major_matrix(
        Float32,
        FIXTURE["sample"]["signal"],
        2,
        2,
    )
    reconstructed = fill(29.0f0, 3)
    pixels[2, 2] = Inf32
    @test process!(reconstructed, prepared, pixels) ===
          DenseReconstructionNonFiniteInput
    @test reconstructed == fill(29.0f0, 3)

    prepared = prepared_fixture()
    pixel_storage = zeros(Float32, 4, 2)
    pixel_view = @view pixel_storage[1:2:end, :]
    pixel_view .= row_major_matrix(
        Float32,
        FIXTURE["sample"]["signal"],
        2,
        2,
    )
    output_storage = zeros(Float32, 6)
    output_view = @view output_storage[1:2:end]
    process!(output_view, prepared, pixel_view)
    @test output_view == Float32.(FIXTURE["sample"]["reconstructed"])
    @test checked_allocation_bytes(output_view, prepared, pixel_view) == 0
    @test admitted_allocation_bytes(output_view, prepared, pixel_view) == 0

    alias_storage = zeros(Float32, 4)
    aliased_pixels = reshape(alias_storage, 2, 2)
    @test_throws ArgumentError process!(
        @view(alias_storage[1:3]),
        prepared,
        aliased_pixels,
    )
end

@testset "invalid pyramid preparation is rejected" begin
    schema = FIXTURE["configuration"]["reconstructed_schema"]
    @test_throws ArgumentError prepare_algorithm(
        PyramidReconstructorF32,
        PyramidReconstructorF32Configuration(
            reconstructed_count=3,
            selected_coordinate_count=0,
            initial_reconstructor=nothing,
            reconstructed_schema=schema,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        PyramidReconstructorF32,
        PyramidReconstructorF32Configuration(
            reconstructed_count=3,
            selected_coordinate_count=2,
            initial_reconstructor=zeros(Float32, 11),
            reconstructed_schema=schema,
        ),
    )
end

end
