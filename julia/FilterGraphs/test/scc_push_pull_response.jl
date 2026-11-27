const SCC_PUSH_PULL_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = response
            label = scc-push-pull-response-f32
            config = { pixel_count = 3 }
        }
    ]
    links = []
    inputs = [
        "response:push-fringed"
        "response:push-unfringed"
        "response:pull-fringed"
        "response:pull-unfringed"
    ]
    outputs = [ "response:response" ]
}
"""

const SCC_PUSH_PULL_INPUT_NAMES = (
    Symbol("push-fringed"),
    Symbol("push-unfringed"),
    Symbol("pull-fringed"),
    Symbol("pull-unfringed"),
)

scc_push_pull_graph_inputs(push_fringed, push_unfringed, pull_fringed, pull_unfringed) =
    NamedTuple{SCC_PUSH_PULL_INPUT_NAMES}(
        (push_fringed, push_unfringed, pull_fringed, pull_unfringed),
    )

scc_push_pull_graph_outputs(response) = (response=response,)

function scc_push_pull_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function scc_push_pull_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "SCC four-image response prepares as one Graph Node" begin
    graph = prepare_graph_text(
        SCC_PUSH_PULL_GRAPH_TEXT;
        algorithms=(SccPushPullResponseF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:response,)
    @test graph.node_failure_types == (SccPushPullResponseFailure,)
    @test keys(graph.input_formats) == SCC_PUSH_PULL_INPUT_NAMES
    @test keys(graph.output_formats) == (:response,)
    @test isempty(graph.parameter_formats)
    @test all(format -> format.shape == (3,), values(graph.input_formats))
    @test all(
        format -> format.schema == CALIBRATED_PIXELS_V1,
        values(graph.input_formats),
    )
    @test graph.output_formats.response.schema == SCC_PUSH_PULL_RESPONSE_V1

    inputs = scc_push_pull_graph_inputs(
        Float32[10, 5, -1],
        Float32[4, 7, -2],
        Float32[3, -1, 8],
        Float32[2, 2, 3],
    )
    response = fill(71.0f0, 3)
    outputs = scc_push_pull_graph_outputs(response)
    metadata = NamedTuple{SCC_PUSH_PULL_INPUT_NAMES}(
        ntuple(
            index -> SampleMetadata(
                41,
                index - 1;
                timestamp_nanoseconds=500 + index,
                terminal=true,
            ),
            4,
        ),
    )

    @test process!(outputs, graph, inputs, metadata) == (response=true,)
    @test response == Float32[5, 1, -4]
    @test FilterGraphs.output_metadata(graph).response ==
          getproperty(metadata, Symbol("push-fringed"))
end

@testset "SCC Graph rejection withholds response and metadata" begin
    graph = prepare_graph_text(
        SCC_PUSH_PULL_GRAPH_TEXT;
        algorithms=(SccPushPullResponseF32,),
    )
    output = fill(71.0f0, 3)
    outputs = scc_push_pull_graph_outputs(output)
    valid = scc_push_pull_graph_inputs(
        Float32[10, 5, -1],
        Float32[4, 7, -2],
        Float32[3, -1, 8],
        Float32[2, 2, 3],
    )
    process!(outputs, graph, valid)
    retained_output = copy(output)
    retained_metadata = FilterGraphs.output_metadata(graph)
    rejected = scc_push_pull_graph_inputs(
        valid[1],
        valid[2],
        Float32[3, NaN, 8],
        valid[4],
    )

    @test process!(outputs, graph, rejected) ===
          SccPushPullNonFinitePullFringed
    @test output == retained_output
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "SCC Graph success and rejection allocate nothing after warmup" begin
    graph = prepare_graph_text(
        SCC_PUSH_PULL_GRAPH_TEXT;
        algorithms=(SccPushPullResponseF32,),
    )
    outputs = scc_push_pull_graph_outputs(zeros(Float32, 3))
    valid = scc_push_pull_graph_inputs(
        Float32[10, 5, -1],
        Float32[4, 7, -2],
        Float32[3, -1, 8],
        Float32[2, 2, 3],
    )
    rejected = scc_push_pull_graph_inputs(
        valid[1],
        valid[2],
        Float32[3, NaN, 8],
        valid[4],
    )
    process!(outputs, graph, valid)
    process!(outputs, graph, rejected)
    FilterGraphs._process_admitted!(
        outputs,
        graph,
        valid,
        graph.default_input_metadata,
    )
    FilterGraphs._process_admitted!(
        outputs,
        graph,
        rejected,
        graph.default_input_metadata,
    )

    @test scc_push_pull_graph_allocation_bytes(outputs, graph, valid) == 0
    @test scc_push_pull_graph_allocation_bytes(outputs, graph, rejected) == 0
    @test scc_push_pull_graph_admitted_allocation_bytes(outputs, graph, valid) == 0
    @test scc_push_pull_graph_admitted_allocation_bytes(
        outputs,
        graph,
        rejected,
    ) == 0
end
