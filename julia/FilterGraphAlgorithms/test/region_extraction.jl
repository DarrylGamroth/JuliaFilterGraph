module RegionExtractionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "region_extraction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function region_major_values(regions)
    values = eltype(regions)[]
    for region in axes(regions, 1), row in axes(regions, 2), column in axes(regions, 3)
        push!(values, regions[region, row, column])
    end
    return values
end

function extraction_prepared()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        RegionExtractionF32,
        RegionExtractionF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            region_size=(configuration["region_rows"], configuration["region_columns"]),
            origins=configuration["origins"],
        ),
    )
end

extraction_allocation_bytes(output, prepared, image) =
    @allocated process!(output, prepared, image)

extraction_admitted_allocation_bytes(output, prepared, image) = @allocated(
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
)

@testset "region extraction preserves FGN geometry and order" begin
    prepared = extraction_prepared()
    fixture = FIXTURE
    configuration = fixture["configuration"]
    image = row_major_matrix(Float32, fixture["sample"]["image"], 4, 5)
    regions = fill(-1.0f0, 2, 2, 2)

    @test algorithm_label(RegionExtractionF32) == "region-extraction-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(RegionExtractionF32) === Union{}
    @test map(field -> field.name, algorithm_configuration(RegionExtractionF32)) ==
          (:image_size, :region_size, :origins)
    ports = algorithm_ports(RegionExtractionF32)
    @test map(port -> port.name, ports) == (:calibrated, :regions, :origins)
    @test ports[2].metadata_source === :calibrated
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((4, 5), (2, 2, 2), (2, 2))
    @test getproperty.(formats, :schema) ==
          (CALIBRATED_PIXELS_V1, REGION_PIXELS_V1, REGION_ORIGINS_V1)

    @test process!(regions, prepared, image) === nothing
    @test region_major_values(regions) == Float32.(fixture["sample"]["regions"])

    replacement_values = fixture["replacement"]["origins"]
    replacement = Matrix{UInt32}(undef, 2, 2)
    for region in 1:2, coordinate in 1:2
        replacement[region, coordinate] = replacement_values[region][coordinate]
    end
    @test replace_parameter!(prepared, "origins", replacement) === prepared
    @test process!(regions, prepared, image) === nothing
    @test region_major_values(regions) ==
          Float32.(fixture["replacement"]["regions"])
end

@testset "region extraction supports padded CPU views" begin
    prepared = extraction_prepared()
    image_storage = fill(NaN32, 8, 9)
    image = @view image_storage[2:2:8, 1:2:9]
    image .= row_major_matrix(Float32, FIXTURE["sample"]["image"], 4, 5)
    output_storage = fill(91.0f0, 4, 4, 4)
    regions = @view output_storage[1:2:3, 2:2:4, 1:2:3]

    @test process!(regions, prepared, image) === nothing
    @test region_major_values(regions) ==
          Float32.(FIXTURE["sample"]["regions"])

    aliased_image = row_major_matrix(Float32, FIXTURE["sample"]["image"], 4, 5)
    aliased_regions = reshape(@view(vec(aliased_image)[1:8]), 2, 2, 2)
    @test_throws ArgumentError process!(aliased_regions, prepared, aliased_image)
end

@testset "region geometry and replacement reject before adoption" begin
    @test_throws ArgumentError RegionExtractionPlan((2, 4), [], (1, 1))
    @test_throws ArgumentError RegionExtractionPlan((2, 4), [(1, 3)], (2, 1))
    @test_throws DimensionMismatch prepare_algorithm(
        RegionExtractionF32,
        RegionExtractionF32Configuration(
            image_size=(2, 4),
            region_size=(1, 1),
            origins=[[0]],
        ),
    )

    prepared = extraction_prepared()
    original = prepared.plan
    invalid = UInt32[0 1; 3 4]
    @test_throws ArgumentError replace_parameter!(prepared, :origins, invalid)
    @test prepared.plan === original
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :origins,
        zeros(UInt32, 1, 4),
    )
    @test prepared.plan === original
end

@testset "region-extraction checked and admitted calls allocate no heap" begin
    prepared = extraction_prepared()
    image = row_major_matrix(Float32, FIXTURE["sample"]["image"], 4, 5)
    output = zeros(Float32, 2, 2, 2)
    process!(output, prepared, image)
    @test extraction_allocation_bytes(output, prepared, image) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test extraction_admitted_allocation_bytes(output, prepared, image) == 0
end

end
