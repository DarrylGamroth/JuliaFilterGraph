const FRAME_DELAY_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = delay
            label = frame-delay-f32
            config = {
                extent = 2
                delay_frames = 2
                initial_value = -1.0
                input_schema = test.signal.f32/1
                output_schema = test.signal.f32/1
            }
        }
        {
            type = ndarray
            name = gate
            label = test-gate-f32
            config = { extent = 2 }
        }
    ]
    links = [
        { output = "delay:output" input = "gate:input" }
    ]
    inputs = [ "delay:input" ]
    outputs = [ "gate:output" ]
}
"""

function prepare_frame_delay_graph()
    return prepare_graph_text(
        FRAME_DELAY_GRAPH;
        algorithms=(FrameDelayF32, TestGateF32),
    )
end

function frame_delay_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

@testset "frame delay advances at its local Graph adoption boundary" begin
    graph = prepare_frame_delay_graph()
    workspace = graph.nodes[1].prepared.workspace
    output = fill(71.0f0, 2)
    outputs = (output=output,)

    first_metadata = (input=SampleMetadata(1; timestamp_nanoseconds=10),)
    @test process!(
        outputs,
        graph,
        (input=Float32[1, 10],),
        first_metadata,
    ) === TestGateRejected
    @test output == fill(71.0f0, 2)
    @test frame_delay_head(workspace) == 2

    @test process!(outputs, graph, (input=Float32[2, 20],)) ===
        TestGateRejected
    @test output == fill(71.0f0, 2)
    @test frame_delay_head(workspace) == 1

    third_metadata = (input=SampleMetadata(3; timestamp_nanoseconds=30),)
    @test process!(
        outputs,
        graph,
        (input=Float32[3, 30],),
        third_metadata,
    ) == (output=true,)
    @test output == Float32[1, 10]
    @test FilterGraphs.output_metadata(graph) == (output=third_metadata.input,)
    @test frame_delay_head(workspace) == 2

    history_before = copy(frame_delay_history(workspace))
    fill!(output, 83.0f0)
    @test process!(outputs, graph, (input=Float32[NaN, 0],)) ===
        FrameDelayNonFiniteInput
    @test output == fill(83.0f0, 2)
    @test frame_delay_history(workspace) == history_before
    @test frame_delay_head(workspace) == 2

    @test process!(outputs, graph, (input=Float32[4, 40],)) == (output=true,)
    @test output == Float32[2, 20]

    @test reset!(graph) === graph
    @test frame_delay_head(workspace) == 1
    @test frame_delay_history(workspace) == fill(-1.0f0, 2, 2)

    valid_inputs = (input=Float32[5, 50],)
    process!(outputs, graph, valid_inputs)
    metadata = (input=SampleMetadata(2; timestamp_nanoseconds=20),)
    process!(outputs, graph, valid_inputs, metadata)
    @test frame_delay_graph_allocation_bytes(
        outputs,
        graph,
        valid_inputs,
        metadata,
    ) == 0
end
