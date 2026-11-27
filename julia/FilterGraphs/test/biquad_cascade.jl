const BIQUAD_CASCADE_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = temporal
            label = biquad-cascade-f32
            config = {
                extent = 2
                section_count = 1
                per_mode = false
                initial_coefficients = [ 0.5 0 0 -0.5 0 ]
                input_schema = test.signal.f32/1
                output_schema = test.signal.f32/1
            }
        }
        { type = ndarray name = gate label = test-gate-f32 config = { extent = 2 } }
    ]
    links = [ { output = "temporal:output" input = "gate:input" } ]
    inputs = [ "temporal:input" "temporal:coefficients" ]
    outputs = [ "gate:output" ]
}
"""

function prepare_biquad_cascade_graph()
    return prepare_graph_text(
        BIQUAD_CASCADE_GRAPH;
        algorithms=(BiquadCascadeF32, TestGateF32),
    )
end

@testset "biquad state adopts locally while Graph publication remains atomic" begin
    graph = prepare_biquad_cascade_graph()
    workspace = graph.nodes[1].prepared.workspace
    output = fill(71.0f0, 2)
    outputs = (output=output,)
    @test process!(outputs, graph, (input=Float32[-2, 4],),
                   (input=SampleMetadata(1),)) === TestGateRejected
    @test output == fill(71.0f0, 2)
    @test any(!iszero, workspace.committed_state)
    metadata = SampleMetadata(2; timestamp_nanoseconds=20)
    @test process!(outputs, graph, (input=Float32[2, 0],),
                   (input=metadata,)) == (output=true,)
    @test output == Float32[0.5, 1]
    @test FilterGraphs.output_metadata(graph) == (output=metadata,)
    state_before = copy(workspace.committed_state)
    @test process!(outputs, graph, (input=Float32[1, NaN],)) === BiquadCascadeNonFiniteInput
    @test output == Float32[0.5, 1]
    @test workspace.committed_state == state_before
    @test FilterGraphs.output_metadata(graph) == (output=metadata,)

    old_plan = graph.nodes[1].prepared.plan
    invalid = reshape(Float32[1, 0, 0, -1, 0], 5, 1, 1)
    @test_throws ArgumentError replace_parameters!(graph, :coefficients => invalid)
    @test graph.nodes[1].prepared.plan === old_plan
    @test workspace.committed_state == state_before
    replacement = reshape(Float32[0.25, 0, 0, -0.5, 0], 5, 1, 1)
    @test replace_parameters!(graph, :coefficients => replacement) === graph
    @test all(iszero, workspace.committed_state)
    @test process!(outputs, graph, (input=Float32[4, 8],)) == (output=true,)
    @test output == Float32[1, 2]
    reset!(graph)
    @test all(iszero, workspace.committed_state)
    input = zeros(Float32, 2)
    graph_allocation_bytes(outputs, graph, (input=input,))
    @test graph_allocation_bytes(outputs, graph, (input=input,)) == 0
    @test output == input
end
