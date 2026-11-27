const PIXEL_PYRAMID_ROW_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "pixel-pyramid-row.conf")
const PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_pixel_row.toml"),
)
const PIXEL_PYRAMID_INPUT_NAMES = (:raw,)
const PIXEL_PYRAMID_OUTPUT_NAMES = (
    Symbol("reconstruction-pixels"),
    Symbol("mean-pupil-intensity"),
)

pixel_pyramid_inputs(raw) = NamedTuple{PIXEL_PYRAMID_INPUT_NAMES}((raw,))
pixel_pyramid_metadata(sample) =
    NamedTuple{PIXEL_PYRAMID_INPUT_NAMES}((sample,))
pixel_pyramid_outputs(reconstruction_pixels, mean_pupil_intensity) =
    NamedTuple{PIXEL_PYRAMID_OUTPUT_NAMES}(
        (reconstruction_pixels, mean_pupil_intensity),
    )
pixel_pyramid_availability(reconstruction_pixels, mean_pupil_intensity) =
    NamedTuple{PIXEL_PYRAMID_OUTPUT_NAMES}(
        (reconstruction_pixels, mean_pupil_intensity),
    )

function pixel_pyramid_row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function process_pixel_pyramid_frame!(outputs, graph, image, sequence)
    configuration = PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE["configuration"]
    block_rows = configuration["row_block_rows"]
    availability = nothing
    for first_row in PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE["block_offsets"]
        row_block = @view image[(first_row + 1):(first_row + block_rows), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=3000 + first_row,
            terminal=first_row + block_rows == configuration["image_size"][1],
        )
        availability = process!(
            outputs,
            graph,
            pixel_pyramid_inputs(row_block),
            pixel_pyramid_metadata(sample),
        )
        expected = sample.terminal ?
            pixel_pyramid_availability(true, true) :
            pixel_pyramid_availability(false, false)
        @test availability == expected
    end
    return availability
end

function pixel_pyramid_allocation_bytes(outputs, graph, input, metadata)
    return @allocated process!(outputs, graph, input, metadata)
end

function parameter_adoption_allocation_bytes(graph, transaction)
    return @allocated FilterGraphs._adopt_parameter_transaction!(
        graph,
        transaction,
    )
end

@testset "production Pixel and Pyramid row Nodes match the repository fixture" begin
    graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    fixture = PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = pixel_pyramid_row_major_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    selected = count(configuration["pupil_mask"])
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, selected)
    mean_pupil_intensity = Float32[-1]
    outputs = pixel_pyramid_outputs(
        reconstruction_pixels,
        mean_pupil_intensity,
    )

    @test map(node -> node.name, graph.nodes) == (:calibrate, :pyramid)
    @test keys(graph.input_formats) == (:raw,)
    @test keys(graph.parameter_formats) == (
        :flat,
        :background,
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
    )
    @test graph.input_formats.raw.schema == RAW_PIXEL_ROW_BLOCK_V1
    @test getproperty(
        graph.output_formats,
        Symbol("reconstruction-pixels"),
    ).schema == PYRAMID_RECONSTRUCTION_PIXELS_V1

    for frame in fixture["frames"]
        process_pixel_pyramid_frame!(outputs, graph, image, frame["sequence"])
        expected = pixel_pyramid_row_major_matrix(
            Float32,
            frame["expected_reconstruction_pixels"],
            PYRAMID_PUPIL_COUNT,
            selected,
        )
        @test reconstruction_pixels == expected
        @test mean_pupil_intensity ==
              Float32[frame["expected_mean_pupil_intensity"]]
        expected_metadata = SampleMetadata(
            frame["sequence"],
            0;
            timestamp_nanoseconds=3006,
            terminal=true,
            discontinuity=frame["expected_discontinuity"],
        )
        @test FilterGraphs.output_metadata(graph) == NamedTuple{
            PIXEL_PYRAMID_OUTPUT_NAMES,
        }(
            (expected_metadata, expected_metadata),
        )
    end
end

@testset "multi-Node Parameter Port replacement preserves one generation" begin
    graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    fixture = PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = pixel_pyramid_row_major_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    selected = count(configuration["pupil_mask"])
    reconstruction_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, selected)
    mean_pupil_intensity = zeros(Float32, 1)
    outputs = pixel_pyramid_outputs(
        reconstruction_pixels,
        mean_pupil_intensity,
    )
    process_pixel_pyramid_frame!(outputs, graph, image, 70)

    first_block = @view image[1:2, :]
    @test process!(
        outputs,
        graph,
        pixel_pyramid_inputs(first_block),
        pixel_pyramid_metadata(SampleMetadata(71, 0)),
    ) == pixel_pyramid_availability(false, false)
    @test_throws DimensionMismatch replace_parameters!(
        graph,
        "flat" => fill(2.0f0, 8, 8),
        "pupil-mask" => falses(1, 4),
    )

    for first_row in 2:2:6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        availability = process!(
            outputs,
            graph,
            pixel_pyramid_inputs(row_block),
            pixel_pyramid_metadata(
                SampleMetadata(71, first_row; terminal=first_row == 6),
            ),
        )
        expected = first_row == 6 ?
            pixel_pyramid_availability(true, true) :
            pixel_pyramid_availability(false, false)
        @test availability == expected
    end
    expected_original = pixel_pyramid_row_major_matrix(
        Float32,
        fixture["frames"][2]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reconstruction_pixels == expected_original
    @test !getproperty(
        FilterGraphs.output_metadata(graph),
        Symbol("reconstruction-pixels"),
    ).discontinuity

    @test process!(
        outputs,
        graph,
        pixel_pyramid_inputs(first_block),
        pixel_pyramid_metadata(SampleMetadata(72, 0)),
    ) == pixel_pyramid_availability(false, false)
    replacement_mask = pixel_pyramid_row_major_matrix(
        Bool,
        fixture["replacement"]["pupil_mask"],
        configuration["pupil_size"][1],
        configuration["pupil_size"][2],
    )
    @test replace_parameters!(
        graph,
        "flat" => fill(2.0f0, 8, 8),
        "pupil-mask" => replacement_mask,
    ) === graph

    for first_row in 2:2:6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        @test process!(
            outputs,
            graph,
            pixel_pyramid_inputs(row_block),
            pixel_pyramid_metadata(
                SampleMetadata(72, first_row; terminal=first_row == 6),
            ),
        ) == pixel_pyramid_availability(false, false)
    end

    process_pixel_pyramid_frame!(outputs, graph, image, 73)
    expected_replacement = 2.0f0 .* pixel_pyramid_row_major_matrix(
        Float32,
        fixture["replacement"]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reconstruction_pixels == expected_replacement
    @test mean_pupil_intensity == Float32[63]
    @test getproperty(
        FilterGraphs.output_metadata(graph),
        Symbol("reconstruction-pixels"),
    ).discontinuity
end

@testset "Parameter replacement does not discard a completed untouched Node" begin
    graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    fixture = PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = pixel_pyramid_row_major_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    outputs = pixel_pyramid_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, count(configuration["pupil_mask"])),
        zeros(Float32, 1),
    )

    process_pixel_pyramid_frame!(outputs, graph, image, 80)
    pyramid_frame = graph.nodes[2].prepared.workspace.frame
    @test !pyramid_frame.active
    @test !pyramid_frame.pending_discontinuity
    @test replace_parameters!(graph, "flat" => fill(2.0f0, 8, 8)) === graph
    @test !pyramid_frame.active
    @test !pyramid_frame.pending_discontinuity
end

@testset "prepared Parameter Port transaction adopts at one Graph boundary" begin
    graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    fixture = PIXEL_PYRAMID_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = pixel_pyramid_row_major_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    outputs = pixel_pyramid_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, count(configuration["pupil_mask"])),
        zeros(Float32, 1),
    )
    first_block = @view image[1:2, :]
    @test process!(
        outputs,
        graph,
        pixel_pyramid_inputs(first_block),
        pixel_pyramid_metadata(SampleMetadata(90, 0)),
    ) == pixel_pyramid_availability(false, false)
    original_plans = map(node -> node.prepared.plan, graph.nodes)
    replacement_mask = pixel_pyramid_row_major_matrix(
        Bool,
        fixture["replacement"]["pupil_mask"],
        configuration["pupil_size"][1],
        configuration["pupil_size"][2],
    )
    transaction = FilterGraphs._prepare_parameter_transaction(
        graph,
        "flat" => fill(2.0f0, 8, 8),
        "pupil-mask" => replacement_mask,
    )
    @test transaction.touched == (true, true)
    @test map(node -> node.prepared.plan, graph.nodes) === original_plans
    @test FilterGraphs._adopt_parameter_transaction!(graph, transaction) === graph
    @test map(node -> node.prepared.plan, graph.nodes) === transaction.plans

    allocation_graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    allocation_transaction = FilterGraphs._prepare_parameter_transaction(
        allocation_graph,
        "flat" => fill(2.0f0, 8, 8),
        "pupil-mask" => replacement_mask,
    )
    @test parameter_adoption_allocation_bytes(
        allocation_graph,
        allocation_transaction,
    ) == 0
end

@testset "linked production row path allocates nothing after warmup" begin
    graph = prepare_graph(
        PIXEL_PYRAMID_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
    )
    image = reshape(UInt16.(1:64), 8, 8)'
    outputs = pixel_pyramid_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, 3),
        zeros(Float32, 1),
    )
    process_pixel_pyramid_frame!(outputs, graph, image, 79)
    blocks = ntuple(4) do block
        first_row = 2 * (block - 1)
        @view image[(first_row + 1):(first_row + 2), :]
    end

    @test pixel_pyramid_allocation_bytes(
        outputs,
        graph,
        pixel_pyramid_inputs(blocks[1]),
        pixel_pyramid_metadata(SampleMetadata(80, 0)),
    ) == 0
    process!(
        outputs,
        graph,
        pixel_pyramid_inputs(blocks[2]),
        pixel_pyramid_metadata(SampleMetadata(80, 2)),
    )
    process!(
        outputs,
        graph,
        pixel_pyramid_inputs(blocks[3]),
        pixel_pyramid_metadata(SampleMetadata(80, 4)),
    )
    @test pixel_pyramid_allocation_bytes(
        outputs,
        graph,
        pixel_pyramid_inputs(blocks[4]),
        pixel_pyramid_metadata(SampleMetadata(80, 6; terminal=true)),
    ) == 0
end
