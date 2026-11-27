@enum PyramidRowGateFailure::UInt8 begin
    PyramidRowGateRejected = 1
end

struct PyramidRowGatePlan end

@algorithm PyramidRowGateF32 begin
    label = "pyramid-row-gate-f32"
    failure = PyramidRowGateFailure

    @configuration PyramidRowGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :reconstructed,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            sample=:reconstructed_sample,
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            metadata_from=:reconstructed,
        ),
    )

    prepare(_configuration) = PyramidRowGatePlan()

    process!(_plan) = begin
        !isnothing(reconstructed_sample) && isodd(reconstructed_sample.sequence) &&
            return PyramidRowGateRejected
        copyto!(accepted, reconstructed)
        nothing
    end
end

const PYRAMID_ROW_RECONSTRUCTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = pwfs-row-reconstructor-f32
            config = {
                image_size = [8, 8]
                row_block_rows = 2
                pupil_size = [2, 2]
                pupil_origins = [ [6 0] [0 0] [6 4] [0 4] ]
                pupil_mask = [ true false true true ]
                reconstructed_count = 5
                initial_reconstructor = [
                    -0.01875 -0.015625 -0.0125 -0.009375 -0.00625 -0.003125
                     0.0 0.003125 0.00625 0.009375 0.0125 0.015625
                     0.01875 -0.01875 -0.015625 -0.0125 -0.009375 -0.00625
                    -0.003125 0.0 0.003125 0.00625 0.009375 0.0125
                     0.015625 0.01875 -0.01875 -0.015625 -0.0125 -0.009375
                    -0.00625 -0.003125 0.0 0.003125 0.00625 0.009375
                     0.0125 0.015625 0.01875 -0.01875 -0.015625 -0.0125
                    -0.009375 -0.00625 -0.003125 0.0 0.003125 0.00625
                     0.009375 0.0125 0.015625 0.01875 -0.01875 -0.015625
                    -0.0125 -0.009375 -0.00625 -0.003125 0.0 0.003125
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
        {
            type = ndarray
            name = gate
            label = pyramid-row-gate-f32
            config = { extent = 5 }
        }
    ]
    links = [
        { output = "reconstruct:reconstructed" input = "gate:reconstructed" }
    ]
    inputs = [
        "reconstruct:row-block"
        "reconstruct:pupil-origins"
        "reconstruct:pupil-mask"
        "reconstruct:reconstructor"
    ]
    outputs = [
        "reconstruct:reconstruction-pixels"
        "reconstruct:mean-pupil-intensity"
        "gate:accepted"
    ]
}
"""

const PYRAMID_ROW_RECONSTRUCTION_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_row_reconstruction.toml"),
)

const PYRAMID_RECONSTRUCTION_ROW_BLOCK = Symbol("row-block")
const PYRAMID_RECONSTRUCTION_PIXELS = Symbol("reconstruction-pixels")
const PYRAMID_RECONSTRUCTION_MEAN = Symbol("mean-pupil-intensity")
const PYRAMID_ROW_RECONSTRUCTION_INPUT_NAMES =
    (PYRAMID_RECONSTRUCTION_ROW_BLOCK,)
const PYRAMID_ROW_RECONSTRUCTION_OUTPUT_NAMES = (
    PYRAMID_RECONSTRUCTION_PIXELS,
    PYRAMID_RECONSTRUCTION_MEAN,
    :accepted,
)

pyramid_row_reconstruction_inputs(row_block) =
    NamedTuple{PYRAMID_ROW_RECONSTRUCTION_INPUT_NAMES}((row_block,))
pyramid_row_reconstruction_metadata(sample) =
    NamedTuple{PYRAMID_ROW_RECONSTRUCTION_INPUT_NAMES}((sample,))
pyramid_row_reconstruction_outputs(pixels, mean, accepted) =
    NamedTuple{PYRAMID_ROW_RECONSTRUCTION_OUTPUT_NAMES}((pixels, mean, accepted))
pyramid_row_reconstruction_availability(pixels, mean, accepted) =
    NamedTuple{PYRAMID_ROW_RECONSTRUCTION_OUTPUT_NAMES}((pixels, mean, accepted))

function pyramid_row_reconstruction_image()
    configuration = PYRAMID_ROW_RECONSTRUCTION_FIXTURE["configuration"]
    return permutedims(
        reshape(
            Float32.(PYRAMID_ROW_RECONSTRUCTION_FIXTURE["image"]),
            configuration["image_columns"],
            configuration["image_rows"],
        ),
    )
end

function process_pyramid_row_reconstruction_frame!(
    outputs,
    graph,
    image,
    sequence,
)
    availability = nothing
    for first_row in PYRAMID_ROW_RECONSTRUCTION_FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + 2), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=6000 + first_row,
            terminal=first_row == 6,
        )
        availability = process!(
            outputs,
            graph,
            pyramid_row_reconstruction_inputs(block),
            pyramid_row_reconstruction_metadata(sample),
        )
        expected = first_row == 6 ?
            pyramid_row_reconstruction_availability(true, true, true) :
            pyramid_row_reconstruction_availability(false, false, false)
        @test availability == expected
    end
    return availability
end

function pyramid_row_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function pyramid_row_graph_without_metadata_allocation_bytes(
    outputs,
    graph,
    inputs,
)
    return @allocated process!(outputs, graph, inputs)
end

@testset "row pyramid reconstruction is a conditional Graph path" begin
    graph = prepare_graph_text(
        PYRAMID_ROW_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidRowReconstructorF32, PyramidRowGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:reconstruct, :gate)
    @test graph.node_failure_types == (
        PyramidRowReconstructionFailure,
        PyramidRowGateFailure,
    )
    @test keys(graph.input_formats) == PYRAMID_ROW_RECONSTRUCTION_INPUT_NAMES
    @test getproperty(
        graph.input_formats,
        PYRAMID_RECONSTRUCTION_ROW_BLOCK,
    ).rate == (0x00000004, 0x00000001)
    @test graph.output_conditional ==
          pyramid_row_reconstruction_availability(true, true, true)
    @test keys(graph.parameter_formats) == (
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
        :reconstructor,
    )
    @test graph.parameter_formats.reconstructor.shape == (5, 12)

    pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, 3)
    mean = Float32[-2]
    accepted = fill(-3.0f0, 5)
    outputs = pyramid_row_reconstruction_outputs(pixels, mean, accepted)
    image = pyramid_row_reconstruction_image()

    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 100)
    @test pixels == zeros(Float32, PYRAMID_PUPIL_COUNT, 3)
    @test mean == Float32[31.5]
    @test accepted == zeros(Float32, 5)
    expected_metadata = SampleMetadata(
        100,
        0;
        timestamp_nanoseconds=6006,
        terminal=true,
    )
    @test FilterGraphs.output_metadata(graph) == NamedTuple{
        PYRAMID_ROW_RECONSTRUCTION_OUTPUT_NAMES,
    }((expected_metadata, expected_metadata, expected_metadata))

    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 102)
    @test accepted ≈ Float32.(
        PYRAMID_ROW_RECONSTRUCTION_FIXTURE["frames"][2]["expected_reconstructed"],
    ) atol = 4.0f-8 rtol = 0
end

@testset "downstream rejection preserves outputs after progressive reconstruction commits" begin
    graph = prepare_graph_text(
        PYRAMID_ROW_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidRowReconstructorF32, PyramidRowGateF32),
    )
    image = pyramid_row_reconstruction_image()
    outputs = pyramid_row_reconstruction_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, 3),
        zeros(Float32, 1),
        zeros(Float32, 5),
    )
    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 110)
    retained_pixels = copy(getproperty(outputs, PYRAMID_RECONSTRUCTION_PIXELS))
    retained_mean = copy(getproperty(outputs, PYRAMID_RECONSTRUCTION_MEAN))
    retained_accepted = copy(outputs.accepted)
    retained_metadata = FilterGraphs.output_metadata(graph)
    workspace = graph.nodes[1].prepared.workspace
    committed_gain = pyramid_row_normalization_gain(workspace.pixels)

    rejection = nothing
    doubled = 2.0f0 .* image
    for first_row in PYRAMID_ROW_RECONSTRUCTION_FIXTURE["block_offsets"]
        block = @view doubled[(first_row + 1):(first_row + 2), :]
        rejection = process!(
            outputs,
            graph,
            pyramid_row_reconstruction_inputs(block),
            pyramid_row_reconstruction_metadata(
                SampleMetadata(111, first_row; terminal=first_row == 6),
            ),
        )
    end
    @test rejection === PyramidRowGateRejected
    @test getproperty(outputs, PYRAMID_RECONSTRUCTION_PIXELS) == retained_pixels
    @test getproperty(outputs, PYRAMID_RECONSTRUCTION_MEAN) == retained_mean
    @test outputs.accepted == retained_accepted
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test pyramid_row_normalization_gain(workspace.pixels) != committed_gain
    @test !workspace.pixels.frame.active
    @test iszero(workspace.candidate_reconstructed)

    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 112)
    metadata = FilterGraphs.output_metadata(graph)
    @test !getproperty(metadata, PYRAMID_RECONSTRUCTION_PIXELS).discontinuity
    @test !getproperty(metadata, PYRAMID_RECONSTRUCTION_MEAN).discontinuity
    @test !metadata.accepted.discontinuity
end

@testset "row pyramid Parameter adoption abandons one in-flight frame" begin
    graph = prepare_graph_text(
        PYRAMID_ROW_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidRowReconstructorF32, PyramidRowGateF32),
    )
    image = pyramid_row_reconstruction_image()
    outputs = pyramid_row_reconstruction_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, 3),
        zeros(Float32, 1),
        zeros(Float32, 5),
    )
    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 120)
    first_block = @view image[1:2, :]
    @test process!(
        outputs,
        graph,
        pyramid_row_reconstruction_inputs(first_block),
        pyramid_row_reconstruction_metadata(SampleMetadata(121, 0)),
    ) == pyramid_row_reconstruction_availability(false, false, false)

    @test_throws DimensionMismatch replace_parameters!(
        graph,
        "reconstructor" => zeros(Float32, 5, 12),
        "pupil-origins" => zeros(UInt32, 3, 2),
    )
    @test graph.nodes[1].prepared.workspace.pixels.frame.active

    @test replace_parameters!(
        graph,
        "reconstructor" => zeros(Float32, 5, 12),
    ) === graph
    @test !graph.nodes[1].prepared.workspace.pixels.frame.active
    process_pyramid_row_reconstruction_frame!(outputs, graph, image, 122)
    @test iszero(outputs.accepted)
    @test FilterGraphs.output_metadata(graph).accepted.discontinuity
end

@testset "row pyramid Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        PYRAMID_ROW_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidRowReconstructorF32, PyramidRowGateF32),
    )
    image = pyramid_row_reconstruction_image()
    blocks = ntuple(4) do index
        first_row = 2 * (index - 1)
        @view image[(first_row + 1):(first_row + 2), :]
    end
    inputs = map(pyramid_row_reconstruction_inputs, blocks)
    outputs = pyramid_row_reconstruction_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, 3),
        zeros(Float32, 1),
        zeros(Float32, 5),
    )

    @test pyramid_row_graph_allocation_bytes(
        outputs,
        graph,
        inputs[1],
        pyramid_row_reconstruction_metadata(SampleMetadata(130, 0)),
    ) == 0
    for index in 2:3
        process!(
            outputs,
            graph,
            inputs[index],
            pyramid_row_reconstruction_metadata(
                SampleMetadata(130, 2 * (index - 1)),
            ),
        )
    end
    @test pyramid_row_graph_allocation_bytes(
        outputs,
        graph,
        inputs[4],
        pyramid_row_reconstruction_metadata(
            SampleMetadata(130, 6; terminal=true),
        ),
    ) == 0

    for index in 1:3
        process!(
            outputs,
            graph,
            inputs[index],
            pyramid_row_reconstruction_metadata(
                SampleMetadata(131, 2 * (index - 1)),
            ),
        )
    end
    @test pyramid_row_graph_allocation_bytes(
        outputs,
        graph,
        inputs[4],
        pyramid_row_reconstruction_metadata(
            SampleMetadata(131, 6; terminal=true),
        ),
    ) == 0
    @test pyramid_row_graph_allocation_bytes(
        outputs,
        graph,
        inputs[2],
        pyramid_row_reconstruction_metadata(SampleMetadata(132, 2)),
    ) == 0
    @test pyramid_row_graph_without_metadata_allocation_bytes(
        outputs,
        graph,
        inputs[1],
    ) == 0
end
