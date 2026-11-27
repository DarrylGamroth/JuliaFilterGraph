@enum ShackRowGateFailure::UInt8 begin
    ShackRowGateRejected = 1
end

struct ShackRowGatePlan end

@algorithm ShackRowGateF32 begin
    label = "shack-row-gate-f32"
    failure = ShackRowGateFailure

    @configuration ShackRowGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(_configuration) = (
        input(
            :reconstructed,
            Float32,
            (_configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1,
        ),
        output(
            :accepted,
            Float32,
            (_configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            metadata_from=:reconstructed,
        ),
    )

    prepare(_configuration) = ShackRowGatePlan()

    process!(_plan) = begin
        reconstructed[1] < 0.0f0 && return ShackRowGateRejected
        copyto!(accepted, reconstructed)
        nothing
    end
end

const SHACK_ROW_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = shwfs-row-reconstructor-f32
            config = {
                image_size = [6, 6]
                row_block_rows = 1
                subaperture_size = [2, 2]
                subaperture_count = 4
                reconstructed_count = 3
                initial_subaperture_origins = [ [0 0] [0 3] [3 0] [3 3] ]
                coordinate_scale = 1.0
                pixel_threshold = 0.0
                flux_threshold = 0.0
                reference_slopes = [ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ]
                active = [ true true true true ]
                initial_reconstructor = [
                    0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
                    0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16
                    0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
        {
            type = ndarray
            name = gate
            label = shack-row-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "reconstruct:reconstructed" input = "gate:reconstructed" }
    ]
    inputs = [
        "reconstruct:row-block"
        "reconstruct:subaperture-origins"
        "reconstruct:coordinates"
        "reconstruct:reference-slopes"
        "reconstruct:thresholds"
        "reconstruct:active"
        "reconstruct:reconstructor"
    ]
    outputs = [ "gate:accepted" ]
}
"""

const SHACK_ROW_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_row_reconstruction.toml"),
)

const SHACK_ROW_BLOCK_PORT = Symbol("row-block")
const SHACK_ROW_INPUT_NAMES = (SHACK_ROW_BLOCK_PORT,)

shack_row_inputs(block) = NamedTuple{SHACK_ROW_INPUT_NAMES}((block,))
shack_row_metadata(sample) = NamedTuple{SHACK_ROW_INPUT_NAMES}((sample,))

function shack_row_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function shack_row_graph_without_metadata_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function shack_row_image()
    values = Float32.(SHACK_ROW_FIXTURE["image"])
    return permutedims(reshape(values, 6, 6))
end

function process_shack_row_graph_frame!(outputs, graph, image, sequence)
    availability = nothing
    for first_row in 0:5
        block = @view image[(first_row + 1):(first_row + 1), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=3000 + first_row,
            terminal=first_row == 5,
        )
        availability = process!(
            outputs,
            graph,
            shack_row_inputs(block),
            shack_row_metadata(sample),
        )
        expected = first_row == 5 ? (accepted=true,) : (accepted=false,)
        @test availability == expected
    end
    return availability
end

@testset "row Shack-Hartmann reconstruction is a conditional Graph path" begin
    graph = prepare_graph_text(
        SHACK_ROW_GRAPH_TEXT;
        algorithms=(ShackHartmannRowReconstructorF32, ShackRowGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:reconstruct, :gate)
    @test graph.node_failure_types == (
        ShackHartmannRowReconstructionFailure,
        ShackRowGateFailure,
    )
    @test keys(graph.input_formats) == SHACK_ROW_INPUT_NAMES
    @test getproperty(graph.input_formats, SHACK_ROW_BLOCK_PORT).rate ==
          (0x00000006, 0x00000001)
    @test graph.output_conditional == (accepted=true,)
    @test keys(graph.parameter_formats) == (
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
        :reconstructor,
    )

    image = shack_row_image()
    accepted = fill(-1.0f0, 3)
    outputs = (accepted=accepted,)
    process_shack_row_graph_frame!(outputs, graph, image, 100)
    @test accepted ≈ Float32.(
        SHACK_ROW_FIXTURE["expected"]["reconstructed"],
    ) atol = 2.0f-6
    expected_metadata = SampleMetadata(
        100,
        0;
        timestamp_nanoseconds=3005,
        terminal=true,
    )
    @test FilterGraphs.output_metadata(graph).accepted == expected_metadata
end

@testset "downstream rejection preserves outputs after row reconstruction commits" begin
    graph = prepare_graph_text(
        SHACK_ROW_GRAPH_TEXT;
        algorithms=(ShackHartmannRowReconstructorF32, ShackRowGateF32),
    )
    image = shack_row_image()
    accepted = zeros(Float32, 3)
    outputs = (accepted=accepted,)
    process_shack_row_graph_frame!(outputs, graph, image, 110)
    retained = copy(accepted)
    retained_metadata = FilterGraphs.output_metadata(graph)

    positive = permutedims(reshape(
        Float32.(SHACK_ROW_FIXTURE["configuration"]["initial_reconstructor"]),
        8,
        3,
    ))
    @test replace_parameters!(graph, :reconstructor => -positive) === graph
    for first_row in 0:4
        @test process!(
            outputs,
            graph,
            shack_row_inputs(@view(image[(first_row + 1):(first_row + 1), :])),
            shack_row_metadata(SampleMetadata(111, first_row)),
        ) == (accepted=false,)
    end
    rejection = process!(
        outputs,
        graph,
        shack_row_inputs(@view(image[6:6, :])),
        shack_row_metadata(SampleMetadata(111, 5; terminal=true)),
    )
    @test rejection === ShackRowGateRejected
    @test accepted == retained
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test !graph.nodes[1].prepared.workspace.frame.active

    @test replace_parameters!(graph, :reconstructor => positive) === graph
    process_shack_row_graph_frame!(outputs, graph, image, 112)
    @test FilterGraphs.output_metadata(graph).accepted.discontinuity
end

@testset "row Graph success, rejection, and deferral allocate nothing" begin
    graph = prepare_graph_text(
        SHACK_ROW_GRAPH_TEXT;
        algorithms=(ShackHartmannRowReconstructorF32, ShackRowGateF32),
    )
    image = shack_row_image()
    blocks = ntuple(6) do block
        @view image[block:block, :]
    end
    outputs = (accepted=zeros(Float32, 3),)

    @test shack_row_graph_allocation_bytes(
        outputs,
        graph,
        shack_row_inputs(blocks[1]),
        shack_row_metadata(SampleMetadata(120, 0)),
    ) == 0
    for first_row in 1:4
        process!(
            outputs,
            graph,
            shack_row_inputs(blocks[first_row + 1]),
            shack_row_metadata(SampleMetadata(120, first_row)),
        )
    end
    @test shack_row_graph_allocation_bytes(
        outputs,
        graph,
        shack_row_inputs(blocks[6]),
        shack_row_metadata(SampleMetadata(120, 5; terminal=true)),
    ) == 0

    positive = permutedims(reshape(
        Float32.(SHACK_ROW_FIXTURE["configuration"]["initial_reconstructor"]),
        8,
        3,
    ))
    replace_parameters!(graph, :reconstructor => -positive)
    for first_row in 0:4
        process!(
            outputs,
            graph,
            shack_row_inputs(blocks[first_row + 1]),
            shack_row_metadata(SampleMetadata(121, first_row)),
        )
    end
    @test shack_row_graph_allocation_bytes(
        outputs,
        graph,
        shack_row_inputs(blocks[6]),
        shack_row_metadata(SampleMetadata(121, 5; terminal=true)),
    ) == 0
    @test shack_row_graph_without_metadata_allocation_bytes(
        outputs,
        graph,
        shack_row_inputs(blocks[1]),
    ) == 0
end
