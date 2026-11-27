module PyramidMeasurementBlockTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_row_reconstruction.toml"),
)
const MEASUREMENTS_SCHEMA = "org.example.ao.pyramid-measurement-block/1"
const COLUMNS_SCHEMA = "org.example.ao.reconstructor-column-indices/1"
const COUNT_SCHEMA = "org.example.ao.measurement-block-count/1"

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_fixture()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PyramidMeasurementBlockF32,
        PyramidMeasurementBlockF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            row_block_rows=configuration["row_block_rows"],
            pupil_size=(configuration["pupil_rows"], configuration["pupil_columns"]),
            pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
            pupil_mask=Bool.(configuration["pupil_mask"]),
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

function process_frame!(outputs, prepared, image, sequence)
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    for first_row in FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + block_rows), :]
        terminal = first_row + block_rows == configuration["image_rows"]
        result = process!(
            outputs...,
            prepared,
            block,
            (row_block=SampleMetadata(sequence, first_row; terminal=terminal),),
        )
        @test result.complete
    end
    return nothing
end

function process_allocation_bytes(
    measurements,
    columns,
    count,
    prepared,
    block,
    metadata,
)
    return @allocated process!(
        measurements,
        columns,
        count,
        prepared,
        block,
        metadata,
    )
end

@testset "pyramid measurement prepared storage is concrete" begin
    prepared = prepared_fixture()
    @test isconcretetype(typeof(prepared.plan))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    @test typeof(prepared.workspace) === PyramidMeasurementBlockWorkspace{
        PyramidPixelRowWorkspace{
            Float32,
            Matrix{Float32},
            Matrix{Float32},
            Vector{Int},
            Vector{Int},
        },
        Vector{Int},
    }
end

@testset "pyramid measurement blocks preserve pupil-major columns" begin
    prepared = prepared_fixture()
    @test algorithm_label(PyramidMeasurementBlockF32) ==
          "pyramid-measurement-block-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PyramidMeasurementBlockF32,
    ) === PyramidPixelRowFailure
    @test map(port -> port.name, algorithm_ports(PyramidMeasurementBlockF32)) == (
        Symbol("row-block"),
        :measurements,
        :columns,
        :count,
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
    )

    image = fixture_image()
    outputs = (zeros(Float32, 12), zeros(UInt32, 12), zeros(UInt32, 1))
    process_frame!(outputs, prepared, image, 10)

    collected = zeros(Float32, 12)
    seen = fill(false, 12)
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    for first_row in FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + block_rows), :]
        terminal = first_row + block_rows == configuration["image_rows"]
        sample = SampleMetadata(11, first_row; terminal=terminal)
        result = process!(outputs..., prepared, block, (row_block=sample,))
        @test result == AlgorithmProcessComplete(sample)
        block_count = Int(outputs[3][1])
        for index in 1:block_count
            column = Int(outputs[2][index]) + 1
            @test !seen[column]
            seen[column] = true
            collected[column] = outputs[1][index]
        end
        if iszero(first_row)
            @test outputs[2][1:block_count] == UInt32[3, 4, 5, 9, 10, 11]
        elseif first_row == 6
            @test outputs[2][1:block_count] == UInt32[0, 1, 2, 6, 7, 8]
        else
            @test iszero(block_count)
        end
    end
    expected_pixels = row_major_matrix(
        Float32,
        FIXTURE["frames"][2]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        3,
    )
    expected = [expected_pixels[pupil, pixel] for pupil in 1:4 for pixel in 1:3]
    @test all(seen)
    @test collected == expected
    @test !prepared.workspace.pixels.frame.active
end

@testset "pyramid measurement-block lifecycle and hot path" begin
    prepared = prepared_fixture()
    image = fixture_image()
    outputs = (zeros(Float32, 12), zeros(UInt32, 12), zeros(UInt32, 1))
    block_rows = FIXTURE["configuration"]["row_block_rows"]
    first = @view image[1:block_rows, :]
    second = @view image[(block_rows + 1):(2 * block_rows), :]

    first_metadata = (row_block=SampleMetadata(30, 0),)
    second_metadata = (row_block=SampleMetadata(30, block_rows),)
    first_result = @inferred Union{
        AlgorithmProcessResult,
        PyramidPixelRowFailure,
    } process!(outputs..., prepared, first, first_metadata)
    @test first_result == AlgorithmProcessComplete(first_metadata.row_block)
    @test process_allocation_bytes(
        outputs...,
        prepared,
        second,
        second_metadata,
    ) == 0
    discard!(prepared.workspace)
    @test !prepared.workspace.pixels.frame.active

    reset!(prepared.workspace)
    process!(outputs..., prepared, first, (row_block=SampleMetadata(31, 0),))
    @test prepared.workspace.pixels.frame.active
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
end

end
