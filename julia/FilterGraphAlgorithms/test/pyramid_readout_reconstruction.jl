module PyramidReadoutReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_readout_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function readout_block(::Type{T}, values, regions, lines, samples) where {T}
    block = Array{T}(undef, regions, lines, samples)
    source = 1
    for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        block[region, line, sample] = T(values[source])
        source += 1
    end
    return block
end

function fixture_configuration(;
    reconstructor=FIXTURE["configuration"]["initial_reconstructor"],
    pupil_origins=FIXTURE["configuration"]["pupil_origins"],
    pupil_mask=FIXTURE["configuration"]["pupil_mask"],
)
    configuration = FIXTURE["configuration"]
    return PyramidReadoutReconstructorF32Configuration(
        detector_size=(configuration["detector_rows"], configuration["detector_columns"]),
        region_lines=configuration["region_lines"],
        block_lines=configuration["block_lines"],
        region_samples=configuration["region_samples"],
        detector_origins=[Int.(value) for value in configuration["detector_origins"]],
        detector_line_steps=[
            Int.(value) for value in configuration["detector_line_steps"]
        ],
        detector_sample_steps=[
            Int.(value) for value in configuration["detector_sample_steps"]
        ],
        pupil_size=(configuration["pupil_rows"], configuration["pupil_columns"]),
        pupil_origins=[Int.(value) for value in pupil_origins],
        pupil_mask=Bool.(pupil_mask),
        reconstructed_count=configuration["reconstructed_count"],
        initial_reconstructor=Float32.(reconstructor),
        reconstructed_schema=configuration["reconstructed_schema"],
    )
end

fixture_prepared(; kwargs...) = prepare_algorithm(
    PyramidReadoutReconstructorF32,
    fixture_configuration(; kwargs...),
)

function eight_region_configuration()
    origins = Vector{Int}[]
    line_steps = Vector{Int}[]
    sample_steps = Vector{Int}[]
    for tile_row in 0:1, tile_column in 0:3
        reverse_lines = isodd(tile_row + tile_column)
        reverse_samples = isodd(tile_column)
        first_row = tile_row * 4
        first_column = tile_column * 2
        push!(
            origins,
            [
                reverse_lines ? first_row + 3 : first_row,
                reverse_samples ? first_column + 1 : first_column,
            ],
        )
        push!(line_steps, [reverse_lines ? -1 : 1, 0])
        push!(sample_steps, [0, reverse_samples ? -1 : 1])
    end
    reconstructor = Float32[
        ((index - 1) % 11 - 5) * 0.03125 for index in 1:36
    ]
    return PyramidReadoutReconstructorF32Configuration(
        detector_size=(8, 8),
        region_lines=4,
        block_lines=2,
        region_samples=2,
        detector_origins=origins,
        detector_line_steps=line_steps,
        detector_sample_steps=sample_steps,
        pupil_size=(2, 2),
        pupil_origins=[[6, 6], [0, 0], [4, 2], [2, 4]],
        pupil_mask=Bool[true, false, true, true],
        reconstructed_count=3,
        initial_reconstructor=reconstructor,
        reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
    )
end

function fixture_blocks(frame)
    configuration = FIXTURE["configuration"]
    regions = length(configuration["detector_origins"])
    block_elements = regions * configuration["block_lines"] *
        configuration["region_samples"]
    packed = frame["packed"]
    return ntuple(div(length(packed), block_elements)) do index
        first = (index - 1) * block_elements + 1
        readout_block(
            Float32,
            @view(packed[first:(first + block_elements - 1)]),
            regions,
            configuration["block_lines"],
            configuration["region_samples"],
        )
    end
end

function fixture_detector(frame)
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        frame["detector"],
        configuration["detector_rows"],
        configuration["detector_columns"],
    )
end

function process_fixture_frame!(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    frame;
    sequence=frame["sequence"],
    discontinuity=false,
)
    blocks = fixture_blocks(frame)
    result = nothing
    for (index, block) in enumerate(blocks)
        ordinal = index - 1
        result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            block,
            (
                readout_block=SampleMetadata(
                    sequence,
                    ordinal;
                    timestamp_nanoseconds=7000 + ordinal,
                    terminal=index == length(blocks),
                    discontinuity=discontinuity && isone(index),
                ),
            ),
        )
    end
    return result
end

function readout_reconstruction_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    block,
    metadata,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        block,
        metadata,
    )
end

function readout_reconstruction_without_metadata_allocation_bytes(
    reconstruction_pixels,
    mean_pupil_intensity,
    reconstructed,
    prepared,
    block,
)
    return @allocated process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        block,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "readout pyramid declaration preserves the frozen FGN contract" begin
    prepared = fixture_prepared()
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    fields = algorithm_configuration(PyramidReadoutReconstructorF32)

    @test algorithm_label(PyramidReadoutReconstructorF32) ==
          "pwfs-readout-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PyramidReadoutReconstructorF32,
    ) === PyramidReadoutReconstructionFailure
    @test map(field -> field.name, fields) == (
        :detector_size,
        :region_lines,
        :block_lines,
        :region_samples,
        :detector_origins,
        :detector_line_steps,
        :detector_sample_steps,
        :pupil_size,
        :pupil_origins,
        :pupil_mask,
        :reconstructed_count,
        :initial_reconstructor,
        :reconstructed_schema,
    )
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:9])
    @test fields[10].update isa ConstructionConfiguration
    @test fields[11].update isa GraphRebuildConfiguration
    @test fields[12].update isa ConstructionConfiguration
    @test fields[13].update isa GraphRebuildConfiguration

    ports = algorithm_ports(PyramidReadoutReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("readout-block"),
        Symbol("reconstruction-pixels"),
        Symbol("mean-pupil-intensity"),
        :reconstructed,
    )
    @test all(port -> port.conditional, ports[2:4])
    @test all(
        port -> port.metadata_source === Symbol("readout-block"),
        ports[2:4],
    )

    formats = algorithm_formats(prepared)
    @test length(formats) == 4
    @test formats[1].shape == (2, 1, 4)
    @test formats[1].rate == (0x00000004, 0x00000001)
    @test formats[1].schema == CALIBRATED_PIXEL_READOUT_BLOCK_V1
    @test formats[2].shape == (PYRAMID_PUPIL_COUNT, 1)
    @test formats[2].schema == PYRAMID_RECONSTRUCTION_PIXELS_V1
    @test formats[3].shape == (1,)
    @test formats[3].schema == PYRAMID_MEAN_PUPIL_INTENSITY_V1
    @test formats[4].shape == (3,)
    @test formats[4].schema == CONTROLLER_RESIDUAL_ERROR_V1
    @test prepared.plan.pixel_completion_ordinals ==
          Int.(FIXTURE["completion_ordinals"])
    @test prepared.plan.completion_groups ==
          [[Int(index) + 1 for index in group] for group in FIXTURE["completion_groups"]]
end

@testset "readout pyramid preparation rejects incoherent composition" begin
    origins = FIXTURE["configuration"]["pupil_origins"]
    @test_throws DimensionMismatch fixture_prepared(pupil_origins=origins[1:3])
    @test_throws DimensionMismatch fixture_prepared(pupil_mask=Bool[true])
    @test_throws ArgumentError fixture_prepared(pupil_mask=Bool[false, false])
    @test_throws DimensionMismatch fixture_prepared(reconstructor=zeros(Float32, 11))

    outside = deepcopy(origins)
    outside[1] = [8, 0]
    @test_throws ArgumentError fixture_prepared(pupil_origins=outside)
end

@testset "progressive readout reconstruction matches the pinned Rust oracle" begin
    prepared = fixture_prepared()
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, 1)
    mean_pupil_intensity = Float32[-2]
    reconstructed = fill(-3.0f0, 3)

    for frame in FIXTURE["frames"]
        retained_pixels = copy(reconstruction_pixels)
        retained_mean = copy(mean_pupil_intensity)
        retained_reconstructed = copy(reconstructed)
        blocks = fixture_blocks(frame)
        result = nothing
        for (index, block) in enumerate(blocks)
            ordinal = index - 1
            terminal = index == length(blocks)
            result = process!(
                reconstruction_pixels,
                mean_pupil_intensity,
                reconstructed,
                prepared,
                block,
                (
                    readout_block=SampleMetadata(
                        frame["sequence"],
                        ordinal;
                        timestamp_nanoseconds=7000 + ordinal,
                        terminal=terminal,
                    ),
                ),
            )
            if terminal
                @test result == AlgorithmProcessComplete(
                    SampleMetadata(
                        frame["sequence"],
                        0;
                        timestamp_nanoseconds=7003,
                        terminal=true,
                    ),
                )
            else
                @test result == AlgorithmProcessDeferred()
                @test reconstruction_pixels == retained_pixels
                @test mean_pupil_intensity == retained_mean
                @test reconstructed == retained_reconstructed
            end
            if frame["sequence"] == 2 && isone(index)
                @test prepared.workspace.candidate_reconstruction_pixels[1:2, 1] ==
                      Float32[1.873016, 0.095238104]
                @test any(!iszero, prepared.workspace.candidate_reconstructed)
            end
        end

        expected_pixels = reshape(
            Float32.(frame["expected_reconstruction_pixels"]),
            PYRAMID_PUPIL_COUNT,
            1,
        )
        @test reconstruction_pixels == expected_pixels
        @test reinterpret(UInt32, vec(reconstruction_pixels)) ==
              UInt32.(frame["expected_reconstruction_pixel_bits"])
        @test mean_pupil_intensity ==
              Float32[frame["expected_mean_pupil_intensity"]]
        @test reinterpret(UInt32, mean_pupil_intensity)[1] ==
              UInt32(frame["expected_mean_pupil_intensity_bits"])
        @test reinterpret(UInt32, reconstructed) ==
              UInt32.(frame["expected_reconstructed_bits"])
        @test prepared.workspace.detector == fixture_detector(frame)
        @test reinterpret(
            UInt32,
            Float32[prepared.workspace.committed_gain],
        )[1] == UInt32(frame["expected_committed_gain_bits"])
    end
end

@testset "eight-region readiness preserves multi-pixel scientific order" begin
    prepared = prepare_algorithm(
        PyramidReadoutReconstructorF32,
        eight_region_configuration(),
    )
    normalization = PyramidPixelPlan{Float32}(Bool[true, false, true, true])
    normalization_workspace = PyramidPixelWorkspace(normalization)
    normalized = zeros(Float32, 12)
    expected_mean = zeros(Float32, 1)
    pupil_pixels = zeros(Float32, 4, PYRAMID_PUPIL_COUNT)
    expected_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 3)
    expected_reconstructed = zeros(Float32, 3)
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, 3)
    mean_pupil_intensity = Float32[-1]
    reconstructed = fill(-1.0f0, 3)
    pupil_origins = ((6, 6), (0, 0), (4, 2), (2, 4))

    @test length(prepared.plan.pixel_completion_ordinals) == 12
    @test sort(vcat(prepared.plan.completion_groups...)) == collect(1:12)
    @test all(ordinal -> 0 <= ordinal < 2, prepared.plan.pixel_completion_ordinals)

    for sequence in 1:2
        detector = Matrix{Float32}(undef, 8, 8)
        for row in axes(detector, 1), column in axes(detector, 2)
            detector[row, column] = Float32(
                (row - 1) * size(detector, 2) + column,
            ) + Float32(sequence) * 0.25f0
        end
        for (pupil, (first_row, first_column)) in enumerate(pupil_origins)
            pixel = 1
            for row in 1:2, column in 1:2
                pupil_pixels[pixel, pupil] =
                    detector[first_row + row, first_column + column]
                pixel += 1
            end
        end
        process!(
            PyramidPixelOutput(normalized, expected_mean),
            normalization_workspace,
            normalization,
            pupil_pixels,
        )
        commit!(normalization_workspace)
        source = 1
        for pupil in 1:PYRAMID_PUPIL_COUNT, selected in 1:3
            expected_pixels[pupil, selected] = normalized[source]
            source += 1
        end
        for output in eachindex(expected_reconstructed)
            value = 0.0f0
            for pupil in 1:PYRAMID_PUPIL_COUNT, selected in 1:3
                column = (pupil - 1) * 3 + selected
                value += prepared.plan.reconstructor.matrix[output, column] *
                    expected_pixels[pupil, selected]
            end
            expected_reconstructed[output] = value
        end

        packed = FilterGraphAlgorithms._pack_detector_readout(
            prepared.plan.readout,
            detector,
        )
        for ordinal in 0:1
            first = ordinal * 32 + 1
            block = readout_block(
                Float32,
                @view(packed[first:(first + 31)]),
                8,
                2,
                2,
            )
            result = process!(
                reconstruction_pixels,
                mean_pupil_intensity,
                reconstructed,
                prepared,
                block,
                (
                    readout_block=SampleMetadata(
                        100 + sequence,
                        ordinal;
                        terminal=ordinal == 1,
                    ),
                ),
            )
            ordinal == 0 && @test result == AlgorithmProcessDeferred()
            ordinal == 1 && @test result.complete
        end
        @test reconstruction_pixels == expected_pixels
        @test mean_pupil_intensity == expected_mean
        @test reconstructed ≈ expected_reconstructed atol = 2.0f-6 rtol = 0
    end
end

@testset "nonpositive frame mean is published without becoming a gain" begin
    prepared = fixture_prepared()
    reconstruction_pixels = fill(5.0f0, PYRAMID_PUPIL_COUNT, 1)
    mean_pupil_intensity = Float32[6]
    reconstructed = fill(7.0f0, 3)
    block = fill(-1.0f0, 2, 1, 4)
    result = nothing
    for ordinal in 0:3
        result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            block,
            (
                readout_block=SampleMetadata(
                    20,
                    ordinal;
                    terminal=ordinal == 3,
                ),
            ),
        )
    end
    @test result.complete
    @test reconstruction_pixels == zeros(Float32, PYRAMID_PUPIL_COUNT, 1)
    @test mean_pupil_intensity == Float32[-1]
    @test reconstructed == zeros(Float32, 3)
    @test iszero(prepared.workspace.committed_gain)
end

@testset "readout faults preserve committed state and mark recovery" begin
    prepared = fixture_prepared()
    first_frame, second_frame = FIXTURE["frames"]
    first_blocks = fixture_blocks(first_frame)
    second_blocks = fixture_blocks(second_frame)
    reconstruction_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 1)
    mean_pupil_intensity = zeros(Float32, 1)
    reconstructed = zeros(Float32, 3)
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        first_frame,
    )
    retained_pixels = copy(reconstruction_pixels)
    retained_mean = copy(mean_pupil_intensity)
    retained_reconstructed = copy(reconstructed)
    committed_gain = prepared.workspace.committed_gain

    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        second_blocks[1],
        (readout_block=SampleMetadata(30, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        second_blocks[2],
        (readout_block=SampleMetadata(31, 1),),
    ) == AlgorithmProcessDeferred()
    @test reconstruction_pixels == retained_pixels
    @test mean_pupil_intensity == retained_mean
    @test reconstructed == retained_reconstructed
    @test prepared.workspace.committed_gain == committed_gain

    recovery = process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        second_frame;
        sequence=32,
    )
    @test recovery.metadata.discontinuity

    fill!(reconstruction_pixels, 41.0f0)
    fill!(mean_pupil_intensity, 42.0f0)
    fill!(reconstructed, 43.0f0)
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        second_blocks[1],
    ) === PyramidReadoutMetadataMissing
    @test reconstruction_pixels == fill(41.0f0, PYRAMID_PUPIL_COUNT, 1)
    @test mean_pupil_intensity == Float32[42]
    @test reconstructed == fill(43.0f0, 3)

    if UInt128(typemax(Int)) < UInt128(typemax(UInt32))
        overflowing_offset = UInt32(typemax(Int)) + UInt32(1)
        @test process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            second_blocks[1],
            (readout_block=SampleMetadata(33, overflowing_offset),),
        ) === PyramidReadoutOffsetOverflow
    else
        @test_throws ArgumentError SampleMetadata(
            33,
            UInt64(typemax(UInt32)) + UInt64(1),
        )
    end

    invalid = copy(second_blocks[1])
    invalid[1, 1, 1] = Inf32
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        invalid,
        (readout_block=SampleMetadata(34, 0),),
    ) === PyramidReadoutNonFiniteInput
    @test !prepared.workspace.frame.active

    subnormal = fill(nextfloat(0.0f0), 2, 1, 4)
    gain_failure = fixture_prepared()
    gain_result = nothing
    for ordinal in 0:3
        gain_result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            gain_failure,
            subnormal,
            (
                readout_block=SampleMetadata(
                    35,
                    ordinal;
                    terminal=ordinal == 3,
                ),
            ),
        )
    end
    @test gain_result === PyramidReadoutNonFiniteNormalizationGain
    @test iszero(gain_failure.workspace.committed_gain)

    mean_failure = fixture_prepared()
    maximum = fill(floatmax(Float32), 2, 1, 4)
    mean_result = nothing
    for ordinal in 0:3
        mean_result = process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            mean_failure,
            maximum,
            (
                readout_block=SampleMetadata(
                    36,
                    ordinal;
                    terminal=ordinal == 3,
                ),
            ),
        )
    end
    @test mean_result === PyramidReadoutNonFiniteMean

    output_failure = fixture_prepared()
    tiny = fill(1.0f-38, 2, 1, 4)
    for ordinal in 0:3
        process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            output_failure,
            tiny,
            (
                readout_block=SampleMetadata(
                    37,
                    ordinal;
                    terminal=ordinal == 3,
                ),
            ),
        )
    end
    @test isfinite(output_failure.workspace.committed_gain)
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        output_failure,
        maximum,
        (readout_block=SampleMetadata(38, 0),),
    ) === PyramidReadoutNonFiniteOutput

    overflowing = fixture_prepared(reconstructor=fill(floatmax(Float32), 12))
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        first_frame;
        sequence=39,
    )
    @test process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        second_blocks[1],
        (readout_block=SampleMetadata(40, 0),),
    ) === PyramidReadoutNonFiniteReconstruction
    @test !overflowing.workspace.frame.active
end

@testset "readout views and every warmed outcome allocate no Julia heap" begin
    prepared = fixture_prepared()
    frame = FIXTURE["frames"][2]
    plain_blocks = fixture_blocks(frame)
    blocks = ntuple(4) do index
        storage = zeros(Float32, 4, 2, 8)
        block = @view storage[1:2:3, 1:1, 1:2:7]
        block .= plain_blocks[index]
        block
    end
    pixel_storage = zeros(Float32, 8, 2)
    reconstruction_pixels = @view pixel_storage[1:2:7, 1:1]
    mean_storage = zeros(Float32, 2)
    mean_pupil_intensity = @view mean_storage[1:2:1]
    reconstructed_storage = zeros(Float32, 6)
    reconstructed = @view reconstructed_storage[1:2:5]

    reset!(prepared)
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(50, 0),),
    )
    discard!(prepared.workspace)
    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(51, 0),),
    ) == 0
    for index in 2:3
        process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            blocks[index],
            (readout_block=SampleMetadata(51, index - 1),),
        )
    end
    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[4],
        (readout_block=SampleMetadata(51, 3; terminal=true),),
    ) == 0

    for index in 1:3
        process!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            prepared,
            blocks[index],
            (readout_block=SampleMetadata(52, index - 1),),
        )
    end
    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[4],
        (readout_block=SampleMetadata(52, 3; terminal=true),),
    ) == 0
    reference = fixture_prepared()
    reference_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 1)
    reference_mean = zeros(Float32, 1)
    reference_reconstructed = zeros(Float32, 3)
    process_fixture_frame!(
        reference_pixels,
        reference_mean,
        reference_reconstructed,
        reference,
        frame;
        sequence=60,
    )
    process_fixture_frame!(
        reference_pixels,
        reference_mean,
        reference_reconstructed,
        reference,
        frame;
        sequence=61,
    )
    @test reconstruction_pixels == reference_pixels
    @test mean_pupil_intensity == reference_mean
    @test reconstructed == reference_reconstructed

    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[2],
        (readout_block=SampleMetadata(53, 1),),
    ) == 0
    @test readout_reconstruction_without_metadata_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
    ) == 0

    invalid = copy(blocks[1])
    invalid[1, 1, 1] = Inf32
    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        invalid,
        (readout_block=SampleMetadata(54, 0),),
    ) == 0

    overflowing = fixture_prepared(reconstructor=fill(floatmax(Float32), 12))
    process_fixture_frame!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        FIXTURE["frames"][1];
        sequence=55,
    )
    process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        blocks[1],
        (readout_block=SampleMetadata(56, 0),),
    )
    @test readout_reconstruction_allocation_bytes(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        overflowing,
        blocks[1],
        (readout_block=SampleMetadata(57, 0),),
    ) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0

    aliased_reconstructed = @view vec(reconstruction_pixels)[1:3]
    @test_throws ArgumentError process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        aliased_reconstructed,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(58, 0),),
    )
    workspace_output = reshape(
        @view(prepared.workspace.detector[1:4]),
        PYRAMID_PUPIL_COUNT,
        1,
    )
    @test_throws ArgumentError process!(
        workspace_output,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(58, 0),),
    )
    workspace_block = reshape(@view(prepared.workspace.detector[1:8]), 2, 1, 4)
    @test_throws ArgumentError process!(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        prepared,
        workspace_block,
        (readout_block=SampleMetadata(58, 0),),
    )
end

function measurement_fixture()
    config = FIXTURE["configuration"]
    return prepare_algorithm(
        PyramidReadoutMeasurementBlockF32,
        PyramidReadoutMeasurementBlockF32Configuration(
            detector_size=(config["detector_rows"], config["detector_columns"]),
            region_lines=config["region_lines"], block_lines=config["block_lines"],
            region_samples=config["region_samples"],
            detector_origins=[Int.(v) for v in config["detector_origins"]],
            detector_line_steps=[Int.(v) for v in config["detector_line_steps"]],
            detector_sample_steps=[Int.(v) for v in config["detector_sample_steps"]],
            pupil_size=(config["pupil_rows"], config["pupil_columns"]),
            pupil_origins=[Int.(v) for v in config["pupil_origins"]],
            pupil_mask=Bool.(config["pupil_mask"]),
            measurements_schema="org.example.ao.pyramid-readout-measurements/1",
            column_indices_schema="org.example.ao.pyramid-readout-columns/1",
            measurement_count_schema="org.example.ao.pyramid-readout-count/1",
        ),
    )
end

function measurement_frame!(outputs, prepared, blocks, sequence)
    for (index, block) in enumerate(blocks)
        result = process!(outputs..., prepared, block,
            (readout_block=SampleMetadata(sequence,index-1;terminal=index==length(blocks)),))
        result.complete || error("prepared measurement frame did not complete")
    end
    return nothing
end

measurement_frame_bytes(outputs,prepared,blocks,sequence) =
    @allocated measurement_frame!(outputs,prepared,blocks,sequence)

@testset "readout measurement preserves concrete source and allocation-free frame state" begin
    prepared = measurement_fixture()
    outputs = (zeros(Float32,4),zeros(UInt32,4),zeros(UInt32,1))
    @test isconcretetype(fieldtype(typeof(prepared.plan),:source))
    for (sequence,frame) in enumerate(FIXTURE["frames"])
        blocks = fixture_blocks(frame)
        collected = zeros(Float32,4)
        seen = falses(4)
        for (index,block) in enumerate(blocks)
            sample = SampleMetadata(sequence,index-1;terminal=index==length(blocks))
            result = @inferred Union{AlgorithmProcessResult,PyramidReadoutReconstructionFailure} process!(
                outputs...,prepared,block,(readout_block=sample,))
            @test result == AlgorithmProcessComplete(sample)
            for offset=1:Int(outputs[3][1])
                column = Int(outputs[2][offset])+1
                @test !seen[column]
                seen[column]=true
                collected[column]=outputs[1][offset]
            end
        end
        @test all(seen)
        @test collected == Float32.(frame["expected_reconstruction_pixels"])
        @test prepared.workspace.source.committed_gain == Float32(frame["expected_committed_gain"])
    end
    blocks = fixture_blocks(last(FIXTURE["frames"]))
    for sequence=3:8
        measurement_frame!(outputs,prepared,blocks,sequence)
    end
    measurement_frame_bytes(outputs,prepared,blocks,9)
    @test measurement_frame_bytes(outputs,prepared,blocks,10) == 0
    @test measurement_frame_bytes(outputs,prepared,blocks,11) == 0
    @test !prepared.workspace.source.frame.active
end


@testset "readout PWFS grouped FMA preserves finite cancellation" begin
    prepared = fixture_prepared()
    plan = prepared.plan
    values = zeros(Float32, size(plan.reconstructor.matrix))
    values[:, 1] .= 1.0f0 + eps(Float32)
    plan = PyramidReadoutReconstructionPlan(plan.regions, plan.normalization,
        DenseReconstructorPlan(values), plan.readout)
    workspace = PyramidReadoutReconstructionWorkspace(plan)
    fill!(workspace.detector, 1.0f0 - eps(Float32))
    fill!(workspace.candidate_reconstructed, -1)
    workspace.committed_gain = 1
    @test isnothing(FilterGraphAlgorithms._accumulate_pyramid_readout_group!(workspace, plan, [1]))
    @test workspace.candidate_reconstructed == fill(-eps(Float32)^2, size(values, 1))
end

end
