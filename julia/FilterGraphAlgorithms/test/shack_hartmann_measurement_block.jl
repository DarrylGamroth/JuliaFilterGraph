module ShackHartmannMeasurementBlockTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_row_reconstruction.toml"),
)
const MEASUREMENTS_SCHEMA = "org.example.ao.shack-hartmann-measurement-block/1"
const COLUMNS_SCHEMA = "org.example.ao.reconstructor-column-indices/1"
const COUNT_SCHEMA = "org.example.ao.measurement-block-count/1"

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_fixture()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        ShackHartmannMeasurementBlockF32,
        ShackHartmannMeasurementBlockF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            row_block_rows=configuration["row_block_rows"],
            subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            subaperture_count=configuration["subaperture_count"],
            initial_subaperture_origins=[
                Int.(origin) for
                origin in configuration["initial_subaperture_origins"]
            ],
            coordinate_scale=Float32(configuration["coordinate_scale"]),
            pixel_threshold=Float32(configuration["pixel_threshold"]),
            flux_threshold=Float32(configuration["flux_threshold"]),
            reference_slopes=nothing,
            active=nothing,
            measurements_schema=MEASUREMENTS_SCHEMA,
            column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA,
        ),
    )
end

function fixture_image()
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
end

function process_allocation_bytes(outputs, prepared, block, metadata)
    return @allocated process!(outputs..., prepared, block, metadata)
end

@testset "Shack-Hartmann measurement prepared storage is concrete" begin
    prepared = prepared_fixture()
    @test isconcretetype(typeof(prepared.plan))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    @test prepared.workspace isa ShackHartmannMeasurementBlockWorkspace{
        Float32,
        Matrix{Float32},
        Matrix{Float32},
        Vector{Float32},
        Vector{Bool},
        Vector{Bool},
    }
end

@testset "Shack-Hartmann measurement blocks follow scientific column order" begin
    prepared = prepared_fixture()
    @test algorithm_label(ShackHartmannMeasurementBlockF32) ==
          "shack-hartmann-measurement-block-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannMeasurementBlockF32,
    ) === ShackHartmannMeasurementBlockFailure
    @test map(port -> port.name, algorithm_ports(ShackHartmannMeasurementBlockF32)) == (
        Symbol("row-block"),
        :measurements,
        :columns,
        :count,
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
    )

    image = fixture_image()
    measurements = fill(-1.0f0, 8)
    columns = fill(typemax(UInt32), 8)
    count = fill(typemax(UInt32), 1)
    collected = zeros(Float32, 8)
    configuration = FIXTURE["configuration"]
    for first_row in FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + 1), :]
        terminal = first_row + 1 == configuration["image_rows"]
        sample = SampleMetadata(21, first_row; terminal=terminal)
        result = process!(
            measurements,
            columns,
            count,
            prepared,
            block,
            (row_block=sample,),
        )
        @test result == AlgorithmProcessComplete(sample)
        block_count = Int(count[1])
        for index in 1:block_count
            column = Int(columns[index]) + 1
            @test iszero(collected[column])
            collected[column] = measurements[index]
        end
        if first_row == 1
            @test block_count == 4
            @test columns[1:4] == UInt32[0, 1, 2, 3]
        elseif first_row == 4
            @test block_count == 4
            @test columns[1:4] == UInt32[4, 5, 6, 7]
        else
            @test iszero(block_count)
        end
    end
    expected_slopes = reduce(
        vcat,
        (Float32.(pair) for pair in FIXTURE["expected"]["slopes"]),
    )
    @test collected ≈ expected_slopes
    @test !prepared.workspace.frame.active
end

@testset "Shack-Hartmann measurement-block lifecycle and hot path" begin
    prepared = prepared_fixture()
    image = fixture_image()
    outputs = (zeros(Float32, 8), zeros(UInt32, 8), zeros(UInt32, 1))
    first = @view image[1:1, :]
    second = @view image[2:2, :]
    first_metadata = (row_block=SampleMetadata(30, 0),)
    second_metadata = (row_block=SampleMetadata(30, 1),)

    first_result = @inferred Union{
        AlgorithmProcessResult,
        ShackHartmannMeasurementBlockFailure,
    } process!(outputs..., prepared, first, first_metadata)
    @test first_result == AlgorithmProcessComplete(first_metadata.row_block)
    @test process_allocation_bytes(outputs, prepared, second, second_metadata) == 0
    discard!(prepared.workspace)
    @test !prepared.workspace.frame.active

    reset!(prepared.workspace)
    process!(outputs..., prepared, first, (row_block=SampleMetadata(31, 0),))
    @test prepared.workspace.frame.active
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
end

end
