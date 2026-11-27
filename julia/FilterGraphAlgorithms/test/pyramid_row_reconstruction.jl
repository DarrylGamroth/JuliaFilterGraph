module PyramidRowReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_row_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_configuration(; reconstructor=nothing)
    configuration = FIXTURE["configuration"]
    values = isnothing(reconstructor) ?
        configuration["initial_reconstructor"] : reconstructor
    return PyramidRowReconstructorF32Configuration(
        image_size=(configuration["image_rows"], configuration["image_columns"]),
        row_block_rows=configuration["row_block_rows"],
        pupil_size=(configuration["pupil_rows"], configuration["pupil_columns"]),
        pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
        pupil_mask=Bool.(configuration["pupil_mask"]),
        reconstructed_count=configuration["reconstructed_count"],
        initial_reconstructor=Float32.(values),
        reconstructed_schema=configuration["reconstructed_schema"],
    )
end

fixture_prepared(; kwargs...) = prepare_algorithm(
    PyramidRowReconstructorF32,
    fixture_configuration(; kwargs...),
)

function fixture_image()
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
end

function process_fixture_frame!(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    image,
    sequence;
    discontinuity=false,
)
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    result = nothing
    for first_row in FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + block_rows), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=1000 + first_row,
            terminal=first_row + block_rows == configuration["image_rows"],
            discontinuity=discontinuity && iszero(first_row),
        )
        result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            block,
            (row_block=sample,),
        )
    end
    return result
end

function row_reconstruction_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    row_block,
    metadata,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        row_block,
        metadata,
    )
end

function row_reconstruction_without_metadata_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    row_block,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        row_block,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "row pyramid declaration preserves the frozen FGN contract" begin
    prepared = fixture_prepared()
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    fields = algorithm_configuration(PyramidRowReconstructorF32)

    @test algorithm_label(PyramidRowReconstructorF32) ==
          "pwfs-row-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PyramidRowReconstructorF32,
    ) === PyramidRowReconstructionFailure
    @test map(field -> field.name, fields) == (
        :image_size,
        :row_block_rows,
        :pupil_size,
        :pupil_origins,
        :pupil_mask,
        :reconstructed_count,
        :initial_reconstructor,
        :reconstructed_schema,
    )
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:4])
    @test fields[5].update isa ConstructionConfiguration
    @test fields[6].update isa GraphRebuildConfiguration
    @test fields[7].update isa ConstructionConfiguration
    @test fields[8].update isa GraphRebuildConfiguration

    ports = algorithm_ports(PyramidRowReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("row-block"),
        Symbol("reconstruction-pixels"),
        Symbol("mean-pupil-intensity"),
        :reconstructed,
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
        :reconstructor,
    )
    @test all(port -> port.conditional, ports[2:4])
    @test all(
        port -> port.metadata_source === Symbol("row-block"),
        ports[2:4],
    )
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[5:7],
    )

    formats = algorithm_formats(prepared)
    @test formats[1].shape == (2, 8)
    @test formats[1].rate == (0x00000004, 0x00000001)
    @test formats[1].schema == CALIBRATED_PIXEL_ROW_BLOCK_V1
    @test formats[2].shape == (PYRAMID_PUPIL_COUNT, 3)
    @test formats[2].schema == PYRAMID_RECONSTRUCTION_PIXELS_V1
    @test formats[3].shape == (1,)
    @test formats[3].schema == PYRAMID_MEAN_PUPIL_INTENSITY_V1
    @test formats[4].shape == (5,)
    @test formats[5].shape == (PYRAMID_PUPIL_COUNT, 2)
    @test formats[6].shape == (2, 2)
    @test formats[7].shape == (5, 12)
    @test formats[7].schema == PWFS_RECONSTRUCTOR_V1
end

@testset "progressive row reconstruction matches the pinned Rust oracle" begin
    prepared = fixture_prepared()
    image = fixture_image()
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, 3)
    mean_pupil_intensity = Float32[-1]
    reconstructed = fill(-1.0f0, 5)

    for frame in FIXTURE["frames"]
        retained_pixels = copy(reconstruction_pixels)
        retained_mean = copy(mean_pupil_intensity)
        retained_reconstructed = copy(reconstructed)
        block_rows = FIXTURE["configuration"]["row_block_rows"]
        for first_row in FIXTURE["block_offsets"]
            block = @view image[(first_row + 1):(first_row + block_rows), :]
            terminal = first_row + block_rows ==
                FIXTURE["configuration"]["image_rows"]
            result = process!(
                reconstruction_pixels,
                mean_pupil_intensity,
                reconstructed,
                prepared,
                block,
                (
                    row_block=SampleMetadata(
                        frame["sequence"],
                        first_row;
                        terminal=terminal,
                    ),
                ),
            )
            if terminal
                @test result.complete
            else
                @test result == AlgorithmProcessDeferred()
                @test reconstruction_pixels == retained_pixels
                @test mean_pupil_intensity == retained_mean
                @test reconstructed == retained_reconstructed
            end
            if frame["sequence"] == 11 && iszero(first_row)
                @test prepared.workspace.accumulated_selected == [0, 3, 0, 3]
                @test any(!iszero, prepared.workspace.candidate_reconstructed)
            end
        end

        expected_pixels = row_major_matrix(
            Float32,
            frame["expected_reconstruction_pixels"],
            PYRAMID_PUPIL_COUNT,
            3,
        )
        @test reconstruction_pixels == expected_pixels
        @test mean_pupil_intensity ==
              Float32[frame["expected_mean_pupil_intensity"]]
        expected_reconstructed = Float32.(frame["expected_reconstructed"])
        if haskey(frame, "expected_reconstructed_bits")
            @test reinterpret(UInt32, expected_reconstructed) ==
                  UInt32.(frame["expected_reconstructed_bits"])
        end
        @test reconstructed ≈ expected_reconstructed atol =
            4.0f-8 rtol = 0
    end
    @test pyramid_row_normalization_gain(prepared.workspace.pixels) == inv(31.5f0)
end

@testset "parameter adoption abandons the partial composite frame" begin
    prepared = fixture_prepared()
    image = fixture_image()
    reconstruction_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 3)
    mean_pupil_intensity = zeros(Float32, 1)
    reconstructed = zeros(Float32, 5)
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        image,
        30,
    )
    committed_gain = pyramid_row_normalization_gain(prepared.workspace.pixels)

    first_block = @view image[1:2, :]
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        first_block,
        (row_block=SampleMetadata(31, 0),),
    ) == AlgorithmProcessDeferred()
    replacement = zeros(Float32, 5, 12)
    @test replace_parameter!(prepared, "reconstructor", replacement) === prepared
    @test pyramid_row_normalization_gain(prepared.workspace.pixels) == committed_gain
    @test !prepared.workspace.pixels.frame.active

    result = process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        image,
        32,
    )
    @test result.metadata.discontinuity
    @test iszero(reconstructed)

    origins = UInt32[6 0; 0 0; 6 4; 0 4]
    mask = Bool[true false; true true]
    @test replace_parameter!(prepared, "pupil-origins", origins) === prepared
    @test replace_parameter!(prepared, "pupil-mask", mask) === prepared
end

@testset "row faults retain outputs and mark recovery discontinuous" begin
    prepared = fixture_prepared()
    image = fixture_image()
    reconstruction_pixels = fill(7.0f0, PYRAMID_PUPIL_COUNT, 3)
    mean_pupil_intensity = Float32[8]
    reconstructed = fill(9.0f0, 5)
    retained_pixels = copy(reconstruction_pixels)
    retained_mean = copy(mean_pupil_intensity)
    retained_reconstructed = copy(reconstructed)
    first_block = @view image[1:2, :]

    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        first_block,
    ) === PyramidRowMetadataMissing
    @test reconstruction_pixels == retained_pixels
    @test mean_pupil_intensity == retained_mean
    @test reconstructed == retained_reconstructed

    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        first_block,
        (row_block=SampleMetadata(40, 0),),
    ) == AlgorithmProcessDeferred()
    last_block = @view image[7:8, :]
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        last_block,
        (row_block=SampleMetadata(40, 6; terminal=true),),
    ) == AlgorithmProcessDeferred()
    recovery = process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        image,
        41,
    )
    @test recovery.metadata.discontinuity

    invalid = copy(first_block)
    invalid[1, 1] = Inf32
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        invalid,
        (row_block=SampleMetadata(42, 0),),
    ) === PyramidRowNonFiniteInput
    @test !prepared.workspace.pixels.frame.active

    overflowing = fixture_prepared(reconstructor=fill(floatmax(Float32), 60))
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        image,
        43,
    )
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        first_block,
        (row_block=SampleMetadata(44, 0),),
    ) === PyramidRowNonFiniteReconstruction
    @test !overflowing.workspace.pixels.frame.active
end

@testset "row views and every warmed outcome allocate no Julia heap" begin
    prepared = fixture_prepared()
    image = fixture_image()
    blocks = ntuple(4) do index
        first_row = 2 * (index - 1)
        storage = zeros(Float32, 4, 16)
        block = @view storage[1:2:3, 1:2:15]
        block .= @view image[(first_row + 1):(first_row + 2), :]
        block
    end
    pixel_storage = zeros(Float32, 8, 6)
    reconstruction_pixels = @view pixel_storage[1:2:7, 1:2:5]
    mean_storage = zeros(Float32, 2)
    mean_pupil_intensity = @view mean_storage[1:2:1]
    reconstructed_storage = zeros(Float32, 10)
    reconstructed = @view reconstructed_storage[1:2:9]

    reset!(prepared)
    first_metadata = (row_block=SampleMetadata(50, 0),)
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        first_metadata,
    )
    discard!(prepared.workspace)
    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (row_block=SampleMetadata(51, 0),),
    ) == 0
    for index in 2:3
        process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            blocks[index],
            (row_block=SampleMetadata(51, 2 * (index - 1)),),
        )
    end
    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[4],
        (row_block=SampleMetadata(51, 6; terminal=true),),
    ) == 0
    for index in 1:3
        process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            blocks[index],
            (row_block=SampleMetadata(52, 2 * (index - 1)),),
        )
    end
    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[4],
        (row_block=SampleMetadata(52, 6; terminal=true),),
    ) == 0
    @test reconstructed ≈
          Float32.(FIXTURE["frames"][2]["expected_reconstructed"]) atol = 4.0f-8

    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[2],
        (row_block=SampleMetadata(53, 2),),
    ) == 0
    @test row_reconstruction_without_metadata_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
    ) == 0

    invalid = copy(blocks[1])
    invalid[1, 1] = Inf32
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (row_block=SampleMetadata(54, 0),),
    )
    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        invalid,
        (row_block=SampleMetadata(54, 0),),
    ) == 0

    overflowing = fixture_prepared(reconstructor=fill(floatmax(Float32), 60))
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        image,
        55,
    )
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        blocks[1],
        (row_block=SampleMetadata(56, 0),),
    )
    @test row_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        blocks[1],
        (row_block=SampleMetadata(57, 0),),
    ) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0

    aliased_reconstructed = @view vec(reconstruction_pixels)[1:5]
    @test_throws ArgumentError process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        aliased_reconstructed,
        prepared,
        blocks[1],
        (row_block=SampleMetadata(58, 0),),
    )
    workspace_output = reshape(
        @view(prepared.workspace.pixels.detector[1:12]),
        PYRAMID_PUPIL_COUNT,
        3,
    )
    @test_throws ArgumentError process!(
        workspace_output,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (row_block=SampleMetadata(58, 0),),
    )
end

end
