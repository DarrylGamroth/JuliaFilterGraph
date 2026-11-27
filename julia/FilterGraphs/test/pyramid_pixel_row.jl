@enum PyramidMeanGateFailure::UInt8 begin
    PyramidMeanGateRejected = 1
end

struct PyramidMeanGatePlan
    minimum::Float32
end

@algorithm PyramidMeanGateF32 begin
    label = "pyramid-mean-gate-f32"
    failure = PyramidMeanGateFailure

    @configuration PyramidMeanGateF32Configuration begin
        minimum::Float32 => construction
    end

    ports(configuration) = (
        input(
            :mean_pupil_intensity,
            Float32,
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1,
        ),
        output(
            :accepted_mean,
            Float32,
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1;
            metadata_from=:mean_pupil_intensity,
        ),
    )

    prepare(configuration) = PyramidMeanGatePlan(configuration.minimum)

    process!(plan) = begin
        mean_pupil_intensity[1] < plan.minimum && return PyramidMeanGateRejected
        accepted_mean[1] = mean_pupil_intensity[1]
        return nothing
    end
end

const PYRAMID_PIXEL_ROW_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "pyramid-pixel-row.conf")
const PYRAMID_ROW_BLOCK_PORT = Symbol("row-block")
const PYRAMID_RECONSTRUCTION_PIXELS_PORT = Symbol("reconstruction-pixels")
const PYRAMID_GRAPH_INPUT_NAMES = (PYRAMID_ROW_BLOCK_PORT,)
const PYRAMID_GRAPH_OUTPUT_NAMES = (
    PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    :accepted_mean,
)

pyramid_graph_inputs(row_block) =
    NamedTuple{PYRAMID_GRAPH_INPUT_NAMES}((row_block,))
pyramid_graph_metadata(sample) =
    NamedTuple{PYRAMID_GRAPH_INPUT_NAMES}((sample,))
pyramid_graph_outputs(reconstruction_pixels, accepted_mean) =
    NamedTuple{PYRAMID_GRAPH_OUTPUT_NAMES}((reconstruction_pixels, accepted_mean))
pyramid_graph_availability(reconstruction_pixels, accepted_mean) =
    NamedTuple{PYRAMID_GRAPH_OUTPUT_NAMES}((reconstruction_pixels, accepted_mean))

function pyramid_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function pyramid_graph_without_metadata_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function process_pyramid_graph_frame!(outputs, graph, image, sequence)
    availability = nothing
    for first_row in 0:2:6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=2000 + first_row,
            terminal=first_row == 6,
        )
        availability = process!(
            outputs,
            graph,
            pyramid_graph_inputs(row_block),
            pyramid_graph_metadata(sample),
        )
        expected = first_row == 6 ?
            pyramid_graph_availability(true, true) :
            pyramid_graph_availability(false, false)
        @test availability == expected
    end
    return availability
end

@testset "production pyramid row aggregate is conditional with per-Node state" begin
    graph = prepare_graph(
        PYRAMID_PIXEL_ROW_GRAPH_FIXTURE;
        algorithms=(PyramidPixelRowF32, PyramidMeanGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:pyramid, :gate)
    @test graph.node_failure_types == (PyramidPixelRowFailure, PyramidMeanGateFailure)
    @test graph.output_conditional == NamedTuple{PYRAMID_GRAPH_OUTPUT_NAMES}((true, true))
    input_format = getproperty(graph.input_formats, PYRAMID_ROW_BLOCK_PORT)
    reconstruction_format = getproperty(
        graph.output_formats,
        PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    )
    @test input_format.schema == CALIBRATED_PIXEL_ROW_BLOCK_V1
    @test input_format.rate == (0x00000004, 0x00000001)
    @test keys(graph.parameter_formats) == (
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
    )
    @test getproperty(
        graph.parameter_formats,
        Symbol("pupil-origins"),
    ).schema == REGION_ORIGINS_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("pupil-mask"),
    ).schema == PYRAMID_PUPIL_MASK_V1
    @test reconstruction_format.schema ==
          PYRAMID_RECONSTRUCTION_PIXELS_V1
    @test reconstruction_format.shape == (4, 3)
    @test graph.output_formats.accepted_mean.schema ==
          PYRAMID_MEAN_PUPIL_INTENSITY_V1

    image = permutedims(reshape(Float32.(1:64), 8, 8))
    reconstruction_pixels = fill(-1.0f0, 4, 3)
    accepted_mean = Float32[-2]
    outputs = pyramid_graph_outputs(reconstruction_pixels, accepted_mean)

    process_pyramid_graph_frame!(outputs, graph, image, 40)
    @test reconstruction_pixels == zeros(Float32, 4, 3)
    @test accepted_mean == Float32[31.5]
    expected_metadata = SampleMetadata(
        40,
        0;
        timestamp_nanoseconds=2006,
        terminal=true,
    )
    @test FilterGraphs.output_metadata(graph) == NamedTuple{
        PYRAMID_GRAPH_OUTPUT_NAMES,
    }(
        (expected_metadata, expected_metadata),
    )

    process_pyramid_graph_frame!(outputs, graph, image, 41)
    @test any(!iszero, reconstruction_pixels)
    committed_gain = pyramid_row_normalization_gain(
        graph.nodes[1].prepared.workspace,
    )

    fill!(reconstruction_pixels, 71.0f0)
    fill!(accepted_mean, 72.0f0)
    retained_metadata = FilterGraphs.output_metadata(graph)
    negative_block = fill(-1.0f0, 2, 8)
    for first_row in 0:2:4
        availability = process!(
            outputs,
            graph,
            pyramid_graph_inputs(negative_block),
            pyramid_graph_metadata(SampleMetadata(42, first_row)),
        )
        @test availability == pyramid_graph_availability(false, false)
    end
    rejection = process!(
        outputs,
        graph,
        pyramid_graph_inputs(negative_block),
        pyramid_graph_metadata(SampleMetadata(42, 6; terminal=true)),
    )
    @test rejection === PyramidMeanGateRejected
    @test reconstruction_pixels == fill(71.0f0, 4, 3)
    @test accepted_mean == Float32[72]
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test pyramid_row_normalization_gain(graph.nodes[1].prepared.workspace) !=
          committed_gain
    @test iszero(
        pyramid_row_normalization_gain(graph.nodes[1].prepared.workspace),
    )

    process_pyramid_graph_frame!(outputs, graph, image, 43)
    @test reconstruction_pixels == zeros(Float32, 4, 3)
    next_metadata = FilterGraphs.output_metadata(graph)
    @test !getproperty(
        next_metadata,
        PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    ).discontinuity
    @test !next_metadata.accepted_mean.discontinuity
end

@testset "Graph Parameter Port replacement is failure-atomic" begin
    graph = prepare_graph(
        PYRAMID_PIXEL_ROW_GRAPH_FIXTURE;
        algorithms=(PyramidPixelRowF32, PyramidMeanGateF32),
    )
    image = permutedims(reshape(Float32.(1:64), 8, 8))
    outputs = pyramid_graph_outputs(
        zeros(Float32, 4, 3),
        zeros(Float32, 1),
    )
    reconstruction_pixels = getproperty(
        outputs,
        PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    )

    process_pyramid_graph_frame!(outputs, graph, image, 60)
    first_block = @view image[1:2, :]
    @test process!(
        outputs,
        graph,
        pyramid_graph_inputs(first_block),
        pyramid_graph_metadata(SampleMetadata(61, 0)),
    ) == pyramid_graph_availability(false, false)

    replacement_mask = Bool[false true; true true]
    @test_throws DimensionMismatch replace_parameters!(
        graph,
        "pupil-mask" => replacement_mask,
        "pupil-origins" => zeros(UInt32, 3, 2),
    )

    for first_row in 2:2:6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        availability = process!(
            outputs,
            graph,
            pyramid_graph_inputs(row_block),
            pyramid_graph_metadata(
                SampleMetadata(61, first_row; terminal=first_row == 6),
            ),
        )
        expected = first_row == 6 ?
            pyramid_graph_availability(true, true) :
            pyramid_graph_availability(false, false)
        @test availability == expected
    end
    @test !getproperty(
        FilterGraphs.output_metadata(graph),
        PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    ).discontinuity
    original_pixels = copy(reconstruction_pixels)
    committed_gain = pyramid_row_normalization_gain(
        graph.nodes[1].prepared.workspace,
    )

    @test process!(
        outputs,
        graph,
        pyramid_graph_inputs(first_block),
        pyramid_graph_metadata(SampleMetadata(62, 0)),
    ) == pyramid_graph_availability(false, false)
    @test replace_parameters!(
        graph,
        "pupil-mask" => replacement_mask,
    ) === graph
    @test pyramid_row_normalization_gain(graph.nodes[1].prepared.workspace) ==
          committed_gain

    for first_row in 2:2:6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        @test process!(
            outputs,
            graph,
            pyramid_graph_inputs(row_block),
            pyramid_graph_metadata(
                SampleMetadata(62, first_row; terminal=first_row == 6),
            ),
        ) == pyramid_graph_availability(false, false)
    end

    process_pyramid_graph_frame!(outputs, graph, image, 63)
    @test reconstruction_pixels != original_pixels
    @test getproperty(
        FilterGraphs.output_metadata(graph),
        PYRAMID_RECONSTRUCTION_PIXELS_PORT,
    ).discontinuity

    @test_throws ArgumentError replace_parameters!(graph)
    @test_throws ArgumentError replace_parameters!(
        graph,
        "pupil-mask" => replacement_mask,
        :"pupil-mask" => replacement_mask,
    )
    @test_throws ArgumentError replace_parameters!(
        graph,
        "row-block" => first_block,
    )
    @test_throws ArgumentError replace_parameters!(
        graph,
        "missing" => replacement_mask,
    )
end

@testset "pyramid Graph completion, rejection, and deferral allocate nothing" begin
    graph = prepare_graph(
        PYRAMID_PIXEL_ROW_GRAPH_FIXTURE;
        algorithms=(PyramidPixelRowF32, PyramidMeanGateF32),
    )
    image = permutedims(reshape(Float32.(1:64), 8, 8))
    blocks = ntuple(4) do block
        first_row = 2 * (block - 1)
        @view image[(first_row + 1):(first_row + 2), :]
    end
    outputs = pyramid_graph_outputs(
        zeros(Float32, 4, 3),
        zeros(Float32, 1),
    )

    @test pyramid_graph_allocation_bytes(
        outputs,
        graph,
        pyramid_graph_inputs(blocks[1]),
        pyramid_graph_metadata(SampleMetadata(50, 0)),
    ) == 0
    process!(
        outputs,
        graph,
        pyramid_graph_inputs(blocks[2]),
        pyramid_graph_metadata(SampleMetadata(50, 2)),
    )
    process!(
        outputs,
        graph,
        pyramid_graph_inputs(blocks[3]),
        pyramid_graph_metadata(SampleMetadata(50, 4)),
    )
    @test pyramid_graph_allocation_bytes(
        outputs,
        graph,
        pyramid_graph_inputs(blocks[4]),
        pyramid_graph_metadata(SampleMetadata(50, 6; terminal=true)),
    ) == 0

    negative_block = fill(-1.0f0, 2, 8)
    for first_row in 0:2:4
        process!(
            outputs,
            graph,
            pyramid_graph_inputs(negative_block),
            pyramid_graph_metadata(SampleMetadata(51, first_row)),
        )
    end
    @test pyramid_graph_allocation_bytes(
        outputs,
        graph,
        pyramid_graph_inputs(negative_block),
        pyramid_graph_metadata(SampleMetadata(51, 6; terminal=true)),
    ) == 0
    @test pyramid_graph_without_metadata_allocation_bytes(
        outputs,
        graph,
        pyramid_graph_inputs(blocks[1]),
    ) == 0
end
