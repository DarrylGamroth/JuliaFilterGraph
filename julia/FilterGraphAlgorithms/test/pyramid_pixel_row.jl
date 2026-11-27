module PyramidPixelRowTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_pixel_row.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function row_process_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    prepared,
    row_block,
    metadata,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        row_block,
        metadata,
    )
end

function row_process_without_metadata_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    prepared,
    row_block,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        row_block,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

function prepared_fixture()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PyramidPixelRowF32,
        PyramidPixelRowF32Configuration(
            image_size=(configuration["image_size"][1], configuration["image_size"][2]),
            row_block_rows=configuration["row_block_rows"],
            pupil_size=(configuration["pupil_size"][1], configuration["pupil_size"][2]),
            pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
            pupil_mask=Bool.(configuration["pupil_mask"]),
        ),
    )
end

function process_fixture_frame!(
    reconstruction_pixels,
    mean_pupil_intensity,
    prepared,
    image,
    frame,
)
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    offsets = FIXTURE["block_offsets"]
    result = nothing
    for first_row in offsets
        row_block = @view image[(first_row + 1):(first_row + block_rows), :]
        sample = SampleMetadata(
            frame["sequence"],
            first_row;
            timestamp_nanoseconds=1000 + first_row,
            terminal=first_row + block_rows == configuration["image_size"][1],
        )
        result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            prepared,
            row_block,
            (row_block=sample,),
        )
        if sample.terminal
            expected_metadata = SampleMetadata(
                sample;
                offset=0,
                terminal=true,
                discontinuity=frame["expected_discontinuity"],
            )
            @test result == AlgorithmProcessComplete(expected_metadata)
        else
            @test result == AlgorithmProcessDeferred()
        end
    end
    return result
end

@testset "row-block pyramid normalization matches the repository fixture" begin
    prepared = prepared_fixture()
    @test typeof(prepared.workspace) === PyramidPixelRowWorkspace{
        Float32,
        Matrix{Float32},
        Matrix{Float32},
        Vector{Int},
        Vector{Int},
    }
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    configuration = FIXTURE["configuration"]
    selected = count(configuration["pupil_mask"])
    image = row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, selected)
    mean_pupil_intensity = Float32[-1]

    @test algorithm_label(PyramidPixelRowF32) == "pyramid-pixel-row-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PyramidPixelRowF32) ===
          PyramidPixelRowFailure
    ports = algorithm_ports(PyramidPixelRowF32)
    @test map(port -> port.name, ports) == (
        Symbol("row-block"),
        Symbol("reconstruction-pixels"),
        Symbol("mean-pupil-intensity"),
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
    )
    @test ports[2].metadata_source === Symbol("row-block")
    @test ports[2].conditional
    @test ports[3].conditional
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[4:5],
    )

    formats = algorithm_formats(prepared)
    @test formats[1].shape == (2, 8)
    @test formats[1].rate == (0x00000004, 0x00000001)
    @test formats[2].shape == (PYRAMID_PUPIL_COUNT, selected)
    @test formats[2].rate == (0x00000001, 0x00000001)
    @test formats[4].shape == (PYRAMID_PUPIL_COUNT, 2)
    @test formats[5].shape == (2, 2)

    for frame in FIXTURE["frames"]
        process_fixture_frame!(
            reconstruction_pixels,
            mean_pupil_intensity,
            prepared,
            image,
            frame,
        )
        expected = row_major_matrix(
            Float32,
            frame["expected_reconstruction_pixels"],
            PYRAMID_PUPIL_COUNT,
            selected,
        )
        @test reconstruction_pixels == expected
        @test mean_pupil_intensity == Float32[frame["expected_mean_pupil_intensity"]]
    end
    @test pyramid_row_normalization_gain(prepared.workspace) == inv(31.5f0)
end

@testset "plan adoption abandons a partial frame and preserves gain" begin
    prepared = prepared_fixture()
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    image = row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    selected = count(configuration["pupil_mask"])
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, selected)
    mean_pupil_intensity = Float32[-1]
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        image,
        FIXTURE["frames"][1],
    )
    committed_gain = pyramid_row_normalization_gain(prepared.workspace)

    first_block = @view image[1:block_rows, :]
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        first_block,
        (row_block=SampleMetadata(20, 0),),
    ) == AlgorithmProcessDeferred()

    replacement = row_major_matrix(
        Bool,
        FIXTURE["replacement"]["pupil_mask"],
        configuration["pupil_size"][1],
        configuration["pupil_size"][2],
    )
    @test replace_parameter!(prepared, "pupil-mask", replacement) === prepared
    @test pyramid_row_normalization_gain(prepared.workspace) == committed_gain

    for first_row in (block_rows):(block_rows):(configuration["image_size"][1] - block_rows)
        row_block = @view image[(first_row + 1):(first_row + block_rows), :]
        result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            prepared,
            row_block,
            (
                row_block=SampleMetadata(
                    20,
                    first_row;
                    terminal=first_row + block_rows == configuration["image_size"][1],
                ),
            ),
        )
        @test result == AlgorithmProcessDeferred()
    end

    replacement_frame = Dict(
        "sequence" => 21,
        "expected_discontinuity" => FIXTURE["replacement"]["expected_discontinuity"],
    )
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        image,
        replacement_frame,
    )
    expected = row_major_matrix(
        Float32,
        FIXTURE["replacement"]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reconstruction_pixels == expected
    @test mean_pupil_intensity ==
          Float32[FIXTURE["replacement"]["expected_mean_pupil_intensity"]]
end

@testset "row recovery is transactional and allocation-free" begin
    prepared = prepared_fixture()
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    image = row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    reconstruction_pixels = fill(-3.0f0, PYRAMID_PUPIL_COUNT, 3)
    mean_pupil_intensity = Float32[-4]
    first_block = @view image[1:block_rows, :]
    second_block = @view image[3:4, :]
    third_block = @view image[5:6, :]
    last_block = @view image[7:8, :]
    retained_pixels = copy(reconstruction_pixels)
    retained_mean = copy(mean_pupil_intensity)

    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        first_block,
    ) === PyramidPixelRowMetadataMissing
    @test reconstruction_pixels == retained_pixels
    @test mean_pupil_intensity == retained_mean

    first_metadata = (row_block=SampleMetadata(30, 0),)
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        first_block,
        first_metadata,
    )
    gap_metadata = (row_block=SampleMetadata(30, 6; terminal=true),)
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        last_block,
        gap_metadata,
    ) == AlgorithmProcessDeferred()
    @test reconstruction_pixels == retained_pixels
    @test mean_pupil_intensity == retained_mean

    @test row_process_without_metadata_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        first_block,
    ) == 0
    @test row_process_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        first_block,
        (row_block=SampleMetadata(31, 0),),
    ) == 0
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        second_block,
        (row_block=SampleMetadata(31, 2),),
    )
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        third_block,
        (row_block=SampleMetadata(31, 4),),
    )
    @test row_process_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        prepared,
        last_block,
        (row_block=SampleMetadata(31, 6; terminal=true),),
    ) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0
end

@testset "invalid row pyramid preparation is rejected" begin
    configuration = FIXTURE["configuration"]
    @test_throws ArgumentError prepare_algorithm(
        PyramidPixelRowF32,
        PyramidPixelRowF32Configuration(
            image_size=(8, 8),
            row_block_rows=3,
            pupil_size=(2, 2),
            pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
            pupil_mask=Bool.(configuration["pupil_mask"]),
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PyramidPixelRowF32,
        PyramidPixelRowF32Configuration(
            image_size=(8, 8),
            row_block_rows=2,
            pupil_size=(2, 2),
            pupil_origins=[[7, 0], [0, 0], [6, 4], [0, 4]],
            pupil_mask=Bool.(configuration["pupil_mask"]),
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PyramidPixelRowF32,
        PyramidPixelRowF32Configuration(
            image_size=(8, 8),
            row_block_rows=2,
            pupil_size=(2, 2),
            pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
            pupil_mask=falses(4),
        ),
    )
end

end
