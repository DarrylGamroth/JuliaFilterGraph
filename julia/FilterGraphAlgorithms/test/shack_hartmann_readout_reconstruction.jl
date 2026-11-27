module ShackHartmannReadoutReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_readout_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function pair_matrix(::Type{T}, values) where {T}
    matrix = Matrix{T}(undef, length(values), 2)
    for index in eachindex(values)
        matrix[index, 1] = T(values[index][1])
        matrix[index, 2] = T(values[index][2])
    end
    return matrix
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
    coordinate_scale=FIXTURE["configuration"]["coordinate_scale"],
    reconstructor=FIXTURE["configuration"]["initial_reconstructor"],
)
    configuration = FIXTURE["configuration"]
    return ShackHartmannReadoutReconstructorF32Configuration(
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
        subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
        subaperture_count=configuration["subaperture_count"],
        reconstructed_count=configuration["reconstructed_count"],
        initial_subaperture_origins=[
            Int.(value) for value in configuration["initial_subaperture_origins"]
        ],
        coordinate_scale=Float32(coordinate_scale),
        pixel_threshold=Float32(configuration["pixel_threshold"]),
        flux_threshold=Float32(configuration["flux_threshold"]),
        reference_slopes=nothing,
        active=nothing,
        initial_reconstructor=Float32.(reconstructor),
        reconstructed_schema=configuration["reconstructed_schema"],
    )
end

fixture_prepared(; kwargs...) =
    prepare_algorithm(ShackHartmannReadoutReconstructorF32, fixture_configuration(; kwargs...))

function fixture_image()
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        FIXTURE["image"],
        configuration["detector_rows"],
        configuration["detector_columns"],
    )
end

function fixture_blocks()
    configuration = FIXTURE["configuration"]
    regions = length(configuration["detector_origins"])
    return map(FIXTURE["blocks"]) do block
        readout_block(
            Float32,
            block["values"],
            regions,
            configuration["block_lines"],
            configuration["region_samples"],
        )
    end
end

function process_fixture_frame!(output, prepared, sequence; discontinuity=false)
    result = nothing
    blocks = fixture_blocks()
    for (index, block) in enumerate(blocks)
        ordinal = index - 1
        result = process!(
            output,
            prepared,
            block,
            (
                readout_block=SampleMetadata(
                    sequence,
                    ordinal;
                    timestamp_nanoseconds=4000 + ordinal,
                    terminal=index == length(blocks),
                    discontinuity=discontinuity && isone(index),
                ),
            ),
        )
    end
    return result
end

function readout_allocation_bytes(output, prepared, block, metadata)
    return @allocated process!(output, prepared, block, metadata)
end

function readout_without_metadata_allocation_bytes(output, prepared, block)
    return @allocated process!(output, prepared, block)
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "readout Shack-Hartmann declaration preserves the frozen FGN contract" begin
    prepared = fixture_prepared()
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    fields = algorithm_configuration(ShackHartmannReadoutReconstructorF32)

    @test algorithm_label(ShackHartmannReadoutReconstructorF32) ==
          "shwfs-readout-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannReadoutReconstructorF32,
    ) === ShackHartmannReadoutReconstructionFailure
    @test map(field -> field.name, fields) == (
        :detector_size,
        :region_lines,
        :block_lines,
        :region_samples,
        :detector_origins,
        :detector_line_steps,
        :detector_sample_steps,
        :subaperture_size,
        :subaperture_count,
        :reconstructed_count,
        :initial_subaperture_origins,
        :coordinate_scale,
        :pixel_threshold,
        :flux_threshold,
        :reference_slopes,
        :active,
        :initial_reconstructor,
        :reconstructed_schema,
    )
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:10])
    @test all(field -> field.update isa ConstructionConfiguration, fields[11:17])
    @test fields[18].update isa GraphRebuildConfiguration

    ports = algorithm_ports(ShackHartmannReadoutReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("readout-block"),
        :reconstructed,
    )
    @test ports[2].conditional
    @test ports[2].metadata_source === Symbol("readout-block")
    formats = algorithm_formats(prepared)
    @test formats[1].shape == (2, 1, 6)
    @test formats[1].rate == (0x00000003, 0x00000001)
    @test formats[1].schema == CALIBRATED_PIXEL_READOUT_BLOCK_V1
    @test formats[2].shape == (3,)
    @test prepared.plan.subaperture_completion_ordinals ==
          Int.(FIXTURE["expected"]["completion_ordinals"])
    @test prepared.plan.completion_groups == [Int[], [1, 2], [3, 4]]
end

@testset "readout groups match the frozen Shack-Hartmann reconstruction oracle" begin
    prepared = fixture_prepared()
    output = fill(-99.0f0, 3)
    blocks = fixture_blocks()

    for (index, block) in enumerate(blocks)
        ordinal = index - 1
        terminal = index == length(blocks)
        sample = SampleMetadata(
            41,
            ordinal;
            timestamp_nanoseconds=1000 + ordinal,
            terminal=terminal,
        )
        result = process!(
            output,
            prepared,
            block,
            (readout_block=sample,),
        )
        if terminal
            @test result == AlgorithmProcessComplete(
                SampleMetadata(
                    41,
                    0;
                    timestamp_nanoseconds=1002,
                    terminal=true,
                ),
            )
        else
            @test result == AlgorithmProcessDeferred()
            @test output == fill(-99.0f0, 3)
        end
        index == 1 && @test iszero(prepared.workspace.measured_subapertures)
        index == 2 && @test prepared.workspace.measured_subapertures == 2
    end
    @test output ≈ Float32.(FIXTURE["expected"]["reconstructed"]) atol = 2.0f-6
    @test prepared.workspace.detector == fixture_image()
    @test prepared.workspace.slopes ≈ pair_matrix(
        Float32,
        FIXTURE["expected"]["slopes"],
    )
    @test !prepared.workspace.frame.active
end

@testset "acquisition readiness never reorders scientific slope columns" begin
    origins = [[1, 0], [0, 0], [2, 0], [3, 0], [1, 2], [0, 2], [2, 2], [3, 2]]
    slope_count = 2 * length(origins)
    matrix = Matrix{Float32}(undef, 3, slope_count)
    for row in axes(matrix, 1), column in axes(matrix, 2)
        matrix[row, column] = Float32(row * column) / 32.0f0
    end
    configuration = ShackHartmannReadoutReconstructorF32Configuration(
        detector_size=(4, 4),
        region_lines=2,
        block_lines=1,
        region_samples=4,
        detector_origins=[[0, 0], [3, 0]],
        detector_line_steps=[[1, 0], [-1, 0]],
        detector_sample_steps=[[0, 1], [0, 1]],
        subaperture_size=(1, 2),
        subaperture_count=length(origins),
        reconstructed_count=3,
        initial_subaperture_origins=origins,
        coordinate_scale=1.0f0,
        pixel_threshold=0.0f0,
        flux_threshold=0.0f0,
        reference_slopes=nothing,
        active=nothing,
        initial_reconstructor=vec(permutedims(matrix)),
        reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
    )
    prepared = prepare_algorithm(ShackHartmannReadoutReconstructorF32, configuration)
    @test prepared.plan.subaperture_completion_ordinals == [1, 0, 1, 0, 1, 0, 1, 0]
    @test prepared.plan.completion_groups == [[2, 4, 6, 8], [1, 3, 5, 7]]

    image = row_major_matrix(Float32, 1:16, 4, 4)
    packed = FilterGraphAlgorithms._pack_detector_readout(prepared.plan.readout, image)
    blocks = ntuple(2) do index
        first = (index - 1) * prepared.plan.readout.block_elements + 1
        readout_block(Float32, @view(packed[first:(first + 7)]), 2, 1, 4)
    end
    output = fill(-1.0f0, 3)
    for (index, block) in enumerate(blocks)
        process!(
            output,
            prepared,
            block,
            (readout_block=SampleMetadata(50, index - 1; terminal=index == 2),),
        )
    end

    sensing = prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(4, 4),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(1, 2),
            subaperture_count=length(origins),
            initial_subaperture_origins=origins,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=0.0f0,
            reference_slopes=nothing,
            active=nothing,
        ),
    )
    slopes = zeros(Float32, length(origins), 2)
    process!(slopes, zeros(Float32, length(origins)), falses(length(origins)), sensing, image)
    reconstructor = prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=3,
            subaperture_count=length(origins),
            initial_reconstructor=vec(permutedims(matrix)),
            reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
    expected = zeros(Float32, 3)
    process!(expected, reconstructor, slopes)
    @test output ≈ expected atol = 2.0f-5
end

@testset "readout faults abandon unpublished work and mark recovery" begin
    prepared = fixture_prepared()
    blocks = fixture_blocks()
    output = fill(77.0f0, 3)

    @test process!(
        output,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(60, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        output,
        prepared,
        blocks[2],
        (readout_block=SampleMetadata(61, 1),),
    ) == AlgorithmProcessDeferred()
    @test output == fill(77.0f0, 3)
    result = process_fixture_frame!(output, prepared, 62)
    @test result.complete
    @test result.metadata.discontinuity

    fill!(output, 83.0f0)
    @test process!(output, prepared, blocks[1]) ===
          ShackHartmannReadoutMetadataMissing
    @test output == fill(83.0f0, 3)

    bad = fixture_prepared(
        coordinate_scale=100.0,
        reconstructor=fill(floatmax(Float32), 24),
    )
    @test process!(
        output,
        bad,
        blocks[1],
        (readout_block=SampleMetadata(70, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        output,
        bad,
        blocks[2],
        (readout_block=SampleMetadata(70, 1),),
    ) === ShackHartmannReadoutNonFiniteReconstruction
    @test output == fill(83.0f0, 3)
    @test !bad.workspace.frame.active
end

@testset "readout views and every warmed outcome allocate no Julia heap" begin
    prepared = fixture_prepared()
    plain_blocks = fixture_blocks()
    blocks = ntuple(3) do index
        storage = zeros(Float32, 4, 2, 12)
        block = @view storage[1:2:3, 1:1, 1:2:11]
        block .= plain_blocks[index]
        block
    end
    output_storage = zeros(Float32, 6)
    output = @view output_storage[1:2:5]

    reset!(prepared)
    process!(
        output,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(90, 0),),
    )
    discard!(prepared.workspace)
    first_metadata = (readout_block=SampleMetadata(91, 0),)
    @test readout_allocation_bytes(
        output,
        prepared,
        blocks[1],
        first_metadata,
    ) == 0
    process!(
        output,
        prepared,
        blocks[2],
        (readout_block=SampleMetadata(91, 1),),
    )
    terminal_metadata = (readout_block=SampleMetadata(91, 2; terminal=true),)
    @test readout_allocation_bytes(
        output,
        prepared,
        blocks[3],
        terminal_metadata,
    ) == 0
    @test output ≈ Float32.(FIXTURE["expected"]["reconstructed"]) atol = 2.0f-6

    dropped_metadata = (readout_block=SampleMetadata(92, 1),)
    @test readout_allocation_bytes(
        output,
        prepared,
        blocks[2],
        dropped_metadata,
    ) == 0
    @test readout_without_metadata_allocation_bytes(output, prepared, blocks[1]) == 0

    bad = fixture_prepared(
        coordinate_scale=100.0,
        reconstructor=fill(floatmax(Float32), 24),
    )
    process!(
        output,
        bad,
        blocks[1],
        (readout_block=SampleMetadata(94, 0),),
    )
    process!(
        output,
        bad,
        blocks[2],
        (readout_block=SampleMetadata(94, 1),),
    )
    process!(
        output,
        bad,
        blocks[1],
        (readout_block=SampleMetadata(95, 0),),
    )
    @test readout_allocation_bytes(
        output,
        bad,
        blocks[2],
        (readout_block=SampleMetadata(95, 1),),
    ) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0

    alias_output = @view vec(blocks[1])[1:3]
    @test_throws ArgumentError process!(
        alias_output,
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(93, 0),),
    )
    @test_throws ArgumentError process!(
        @view(prepared.workspace.candidate_reconstructed[1:3]),
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(93, 0),),
    )
    @test_throws ArgumentError process!(
        @view(prepared.plan.reconstructor.matrix[1:3]),
        prepared,
        blocks[1],
        (readout_block=SampleMetadata(93, 0),),
    )
    workspace_block = reshape(@view(prepared.workspace.detector[1:12]), 2, 1, 6)
    @test_throws ArgumentError process!(
        output,
        prepared,
        workspace_block,
        (readout_block=SampleMetadata(93, 0),),
    )
end


@testset "readout SHWFS grouped FMA keeps X then Y" begin
    original = fixture_prepared().plan
    sensing = original.sensing
    pixels = length(sensing.x_coordinates)
    exact_sensing = ShackHartmannImagePlan(sensing.regions,
        fill(1.0f0-eps(Float32), pixels), zeros(Float32, pixels),
        copy(sensing.reference_slopes), copy(sensing.pixel_thresholds),
        copy(sensing.flux_thresholds), copy(sensing.active))
    values = zeros(Float32, size(original.reconstructor.matrix))
    values[:, 1] .= 1.0f0 + eps(Float32)
    plan = ShackHartmannReadoutReconstructionPlan(exact_sensing,
        DenseReconstructorPlan(values), original.readout)
    workspace = ShackHartmannReadoutReconstructionWorkspace(plan)
    fill!(workspace.detector, 1)
    fill!(workspace.candidate_reconstructed, -1)
    @test isnothing(FilterGraphAlgorithms._accumulate_shack_hartmann_readout_group!(workspace, plan, [1]))
    @test workspace.candidate_reconstructed == fill(-eps(Float32)^2, size(values, 1))
    @test workspace.measured_subapertures == 1
end

end
