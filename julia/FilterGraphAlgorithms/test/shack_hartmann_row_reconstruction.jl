module ShackHartmannRowReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_row_reconstruction.toml"),
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

function prepared_fixture(;
    origins=FIXTURE["configuration"]["initial_subaperture_origins"],
    coordinate_scale=FIXTURE["configuration"]["coordinate_scale"],
    reconstructor=FIXTURE["configuration"]["initial_reconstructor"],
)
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        ShackHartmannRowReconstructorF32,
        ShackHartmannRowReconstructorF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            row_block_rows=configuration["row_block_rows"],
            subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            subaperture_count=configuration["subaperture_count"],
            reconstructed_count=configuration["actuator_count"],
            initial_subaperture_origins=[Int.(origin) for origin in origins],
            coordinate_scale=Float32(coordinate_scale),
            pixel_threshold=Float32(configuration["pixel_threshold"]),
            flux_threshold=Float32(configuration["flux_threshold"]),
            reference_slopes=nothing,
            active=nothing,
            initial_reconstructor=Float32.(reconstructor),
            reconstructed_schema=configuration["reconstructed_schema"],
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

function process_frame!(output, prepared, image, sequence; discontinuity=false)
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
        result = process!(output, prepared, block, (row_block=sample,))
    end
    return result
end

function row_allocation_bytes(output, prepared, block, metadata)
    return @allocated process!(output, prepared, block, metadata)
end

function row_without_metadata_allocation_bytes(output, prepared, block)
    return @allocated process!(output, prepared, block)
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "row Shack-Hartmann declaration preserves the FGN contract" begin
    prepared = prepared_fixture()
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    configuration = FIXTURE["configuration"]

    @test algorithm_label(ShackHartmannRowReconstructorF32) ==
          "shwfs-row-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannRowReconstructorF32,
    ) === ShackHartmannRowReconstructionFailure
    fields = algorithm_configuration(ShackHartmannRowReconstructorF32)
    @test map(field -> field.name, fields) == (
        :image_size,
        :row_block_rows,
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
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:5])
    @test all(field -> field.update isa ConstructionConfiguration, fields[6:12])
    @test fields[13].update isa GraphRebuildConfiguration

    ports = algorithm_ports(ShackHartmannRowReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("row-block"),
        :reconstructed,
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
        :reconstructor,
    )
    @test ports[2].conditional
    @test ports[2].metadata_source === Symbol("row-block")
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[3:8],
    )
    formats = algorithm_formats(prepared)
    @test formats[1].shape == (1, 6)
    @test formats[1].rate == (0x00000006, 0x00000001)
    @test formats[2].shape == (3,)
    @test formats[2].rate == (0x00000001, 0x00000001)
    @test formats[8].shape == (3, 8)
    @test formats[8].schema == SHWFS_RECONSTRUCTOR_V1
    @test prepared.plan.completion_schedule == [1, 2, 3, 4]
end

@testset "row prefixes match the frozen FGN reconstruction oracle" begin
    prepared = prepared_fixture()
    image = fixture_image()
    output = fill(-99.0f0, 3)
    configuration = FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]

    for first_row in FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + block_rows), :]
        terminal = first_row + block_rows == configuration["image_rows"]
        sample = SampleMetadata(
            41,
            first_row;
            timestamp_nanoseconds=terminal ? 123_456 : nothing,
            terminal=terminal,
        )
        result = process!(output, prepared, block, (row_block=sample,))
        if terminal
            @test result == AlgorithmProcessComplete(
                SampleMetadata(
                    41,
                    0;
                    timestamp_nanoseconds=123_456,
                    terminal=true,
                ),
            )
        else
            @test result == AlgorithmProcessDeferred()
            @test output == fill(-99.0f0, 3)
        end
        first_row == 1 && @test prepared.workspace.processed_subapertures == 2
        first_row == 4 && @test prepared.workspace.processed_subapertures == 4
    end
    @test output ≈ Float32.(FIXTURE["expected"]["reconstructed"]) atol = 2.0f-6

    full_sensor = prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            image_schema=CALIBRATED_PIXELS_V1,
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
        ),
    )
    slopes = zeros(Float32, 4, 2)
    flux = zeros(Float32, 4)
    validity = fill(false, 4)
    process!(slopes, flux, validity, full_sensor, image)
    @test slopes ≈ pair_matrix(Float32, FIXTURE["expected"]["slopes"])
end

@testset "completion schedule never reorders scientific slopes" begin
    reordered = [[3, 0], [0, 0], [0, 3], [3, 3]]
    prepared = prepared_fixture(origins=reordered)
    @test prepared.plan.completion_schedule == [2, 3, 1, 4]
    image = fixture_image()
    output = zeros(Float32, 3)
    for first_row in 0:1
        process!(
            output,
            prepared,
            @view(image[(first_row + 1):(first_row + 1), :]),
            (row_block=SampleMetadata(50, first_row),),
        )
    end
    @test prepared.workspace.processed_subapertures == 0
    for first_row in 2:5
        process!(
            output,
            prepared,
            @view(image[(first_row + 1):(first_row + 1), :]),
            (row_block=SampleMetadata(50, first_row; terminal=first_row == 5),),
        )
    end

    full = prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(6, 6),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(2, 2),
            subaperture_count=4,
            initial_subaperture_origins=reordered,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=0.0f0,
            reference_slopes=nothing,
            active=nothing,
        ),
    )
    slopes = zeros(Float32, 4, 2)
    process!(slopes, zeros(Float32, 4), falses(4), full, image)
    dense = prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=3,
            subaperture_count=4,
            initial_reconstructor=Float32.(
                FIXTURE["configuration"]["initial_reconstructor"],
            ),
            reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
    expected = zeros(Float32, 3)
    process!(expected, dense, slopes)
    @test output ≈ expected atol = 2.0f-6
end

@testset "row faults abandon the frame without publication" begin
    prepared = prepared_fixture()
    image = fixture_image()
    output = fill(-7.0f0, 3)
    first = @view image[1:1, :]
    third = @view image[3:3, :]
    @test process!(
        output,
        prepared,
        first,
        (row_block=SampleMetadata(60, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        output,
        prepared,
        third,
        (row_block=SampleMetadata(60, 2),),
    ) == AlgorithmProcessDeferred()
    @test output == fill(-7.0f0, 3)
    result = process_frame!(output, prepared, image, 61)
    @test result isa AlgorithmProcessResult
    @test result.complete
    @test result.metadata.discontinuity

    invalid_reconstructor = fill(floatmax(Float32), 3, 8)
    bad = prepared_fixture(coordinate_scale=100.0, reconstructor=vec(
        permutedims(invalid_reconstructor),
    ))
    fill!(output, -11.0f0)
    @test process!(
        output,
        bad,
        first,
        (row_block=SampleMetadata(70, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        output,
        bad,
        @view(image[2:2, :]),
        (row_block=SampleMetadata(70, 1),),
    ) === ShackHartmannRowNonFiniteReconstruction
    @test output == fill(-11.0f0, 3)
    @test !bad.workspace.frame.active
end

@testset "parameter adoption abandons partial work atomically" begin
    prepared = prepared_fixture()
    image = fixture_image()
    output = zeros(Float32, 3)
    process!(
        output,
        prepared,
        @view(image[1:1, :]),
        (row_block=SampleMetadata(80, 0),),
    )
    original = prepared.plan
    invalid = zeros(Float32, 3, 8)
    invalid[1, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :reconstructor,
        invalid,
    )
    @test prepared.plan === original
    @test prepared.workspace.frame.active

    replacement = row_major_matrix(
        Float32,
        FIXTURE["configuration"]["initial_reconstructor"],
        3,
        8,
    )
    @test replace_parameter!(prepared, :reconstructor, replacement) === prepared
    @test prepared.plan !== original
    @test !prepared.workspace.frame.active
    @test process_frame!(output, prepared, image, 81).metadata.discontinuity
end

@testset "row views and every warmed outcome allocate no Julia heap" begin
    prepared = prepared_fixture()
    image_storage = zeros(Float32, 12, 12)
    image = @view image_storage[1:2:end, 1:2:end]
    image .= fixture_image()
    output_storage = zeros(Float32, 6)
    output = @view output_storage[1:2:end]
    first = @view image[1:1, :]

    reset!(prepared)
    process!(
        output,
        prepared,
        first,
        (row_block=SampleMetadata(90, 0),),
    )
    discard!(prepared.workspace)
    deferred_metadata = (row_block=SampleMetadata(91, 0),)
    @test row_allocation_bytes(
        output,
        prepared,
        first,
        deferred_metadata,
    ) == 0
    for first_row in 1:4
        process!(
            output,
            prepared,
            @view(image[(first_row + 1):(first_row + 1), :]),
            (row_block=SampleMetadata(91, first_row),),
        )
    end
    terminal_block = @view image[6:6, :]
    terminal_metadata = (row_block=SampleMetadata(91, 5; terminal=true),)
    @test row_allocation_bytes(
        output,
        prepared,
        terminal_block,
        terminal_metadata,
    ) == 0
    @test output ≈ Float32.(FIXTURE["expected"]["reconstructed"]) atol = 2.0f-6

    @test row_without_metadata_allocation_bytes(output, prepared, first) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0

    alias_output = @view vec(first)[1:3]
    @test_throws ArgumentError process!(
        alias_output,
        prepared,
        first,
        (row_block=SampleMetadata(92, 0),),
    )
    @test_throws ArgumentError process!(
        @view(prepared.workspace.candidate_reconstructed[1:3]),
        prepared,
        first,
        (row_block=SampleMetadata(92, 0),),
    )
    @test_throws ArgumentError process!(
        @view(prepared.plan.sensing.reference_slopes[1:3]),
        prepared,
        first,
        (row_block=SampleMetadata(92, 0),),
    )
end

end
