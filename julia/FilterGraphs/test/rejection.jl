@testset "configured rejection provenance" begin
    graph = prepare_graph(GRAPH_FIXTURE; algorithms=(LeakyIntegratorF32, TestGateF32))
    inputs = (input=Float32[-2, 1, 1],)
    outputs = (output=fill(71f0, 3),)
    metadata = (input=SampleMetadata(12),)
    @test FilterGraphs.rejection_context(graph) === nothing
    @test process!(outputs, graph, inputs, metadata) === TestGateRejected
    rejection = FilterGraphs.rejection_context(graph)
    @test rejection isa FilterGraphs.GraphRejection
    @test rejection.node === :gate
    @test rejection.reason === TestGateRejected
    @test rejection.inputs == metadata
    @test outputs.output == fill(71f0, 3)
    @test_throws ArgumentError process!(outputs, graph, (unknown=inputs.input,))
    @test FilterGraphs.rejection_context(graph) == rejection
    reset!(graph)
    @test FilterGraphs.rejection_context(graph) === nothing
    @test process!(outputs, graph, (input=ones(Float32, 3),)) == (output=true,)
    @test rejection.inputs == metadata # a snapshot, unaffected by subsequent calls
    @test FilterGraphs.rejection_context(graph) === nothing
    @test graph_allocation_bytes(outputs, graph, (input=ones(Float32, 3),)) == 0

    text = """
    filter.graph = {
        nodes = [
            { type = ndarray name = first label = leaky-integrator-f32 config = { extent = 1 initial_state = 0.0 input_schema = test.signal.f32/1 output_schema = test.signal.f32/1 } }
            { type = ndarray name = second label = leaky-integrator-f32 config = { extent = 1 initial_state = 0.0 input_schema = test.signal.f32/1 output_schema = test.signal.f32/1 } }
        ]
        links = [ { output = "first:output" input = "second:input" } ]
        inputs = [ "first:input" ] outputs = [ "second:output" ]
    }
    """
    duplicate = prepare_graph_text(text; algorithms=(LeakyIntegratorF32,))
    replace_properties!(duplicate, "first:gain"=>2f0, "second:gain"=>2f0)
    y = (output=zeros(Float32, 1),)
    @test process!(y, duplicate, (input=Float32[floatmax(Float32)],)) === LeakyIntegratorNonFiniteOutput
    @test FilterGraphs.rejection_context(duplicate).node === :first
    reset!(duplicate)
    @test process!(y, duplicate, (input=Float32[floatmax(Float32)/3],)) === LeakyIntegratorNonFiniteOutput
    @test FilterGraphs.rejection_context(duplicate).node === :second
end

@testset "captured per-frame rejection diagnostics" begin
    graph = prepare_graph_text(CAPTURED_PASS_GATE_GRAPH; algorithms=(FastPassF32, CapturedGateF32))
    captured = prepare_captured_graph(graph, TestCapturedDevice())
    outputs = (output=fill(71f0, 3),)
    inputs = (input=fill(-1f0, 3),)
    metadata = (input=SampleMetadata(31),)
    @test process!(outputs, captured, inputs, metadata) === TestGateRejected
    rejected = FilterGraphs.rejection_context(captured)
    @test rejected.node === :gate
    @test rejected.reason === TestGateRejected
    @test rejected.inputs == metadata
    reset!(captured)
    @test FilterGraphs.rejection_context(captured) === nothing

    pipeline = prepare_captured_pipeline(graph, TestCapturedDevice(); capacity=2)
    try
        @test submit_captured_frame!(pipeline, inputs, metadata, 31)
        @test submit_captured_frame!(pipeline, (input=ones(Float32, 3),), 32)
        for _ in 1:5
            advance_captured_pipeline!(pipeline)
        end
        receipt = take_captured_result!(outputs, pipeline; diagnostics=true)
        @test keys(receipt) === (:sequence, :result, :metadata, :rejection)
        @test receipt.sequence == 31
        @test receipt.result === TestGateRejected
        @test receipt.metadata === nothing
        @test receipt.rejection.inputs == metadata
        @test receipt.rejection.node === :gate
        @test outputs.output == fill(71f0, 3)
        ordinary = take_captured_result!(outputs, pipeline)
        @test keys(ordinary) === (:sequence, :result, :metadata)
        @test ordinary.sequence == 32
        @test ordinary.result == (output=true,)
        @test submit_captured_frame!(pipeline, inputs, 33)
        for _ in 1:3
            advance_captured_pipeline!(pipeline)
        end
        reused = take_captured_result!(outputs, pipeline; diagnostics=true)
        @test reused.sequence == 33
        @test reused.rejection.node === :gate
        @test receipt.rejection.inputs == metadata # independent of reused slot
        @test take_captured_result!(outputs, pipeline; diagnostics=true) === nothing
    finally
        close(pipeline)
    end
end

struct ApiPreflightRejection end
FilterGraphAlgorithms.preflight(::ApiPreflightRejection) = TestGateRejected

@testset "preflight rejection records configured index and absent input metadata" begin
    graph = prepare_graph_text(CAPTURED_PASS_GATE_GRAPH; algorithms=(FastPassF32, CapturedGateF32))
    captured = prepare_captured_graph(graph, TestCapturedDevice())
    rejector = FilterGraphs.CapturedNode{CapturedGateF32,ApiPreflightRejection}(ApiPreflightRejection())
    nodes = (first(captured.nodes), rejector)
    @test FilterGraphs._preflight_captured_nodes(nodes, captured.rejection) === TestGateRejected
    @test FilterGraphs.rejection_context(captured).node === :gate
    @test FilterGraphs.rejection_context(captured).inputs == (input=nothing,)
    FilterGraphs._clear_rejection!(captured.rejection)
    @test FilterGraphs.rejection_context(captured) === nothing
end

@testset "rejection snapshots preserve absent and distinct external identities" begin
    graph = prepare_graph_text(SYNCHRONIZED_TRANSACTION_GRAPH;
        algorithms=(SynchronizedSumF32, LeakyIntegratorF32, TestGateF32))
    inputs = (a=fill(-2f0, 2), b=zeros(Float32, 2), c=zeros(Float32, 2))
    outputs = (output=fill(71f0, 2),)
    metadata = (a=SampleMetadata(101), b=nothing, c=SampleMetadata(7))
    @test process!(outputs, graph, inputs, metadata) === TestGateRejected
    snapshot = FilterGraphs.rejection_context(graph)
    @test snapshot.node === :gate
    @test snapshot.reason === TestGateRejected
    @test snapshot.inputs == metadata
    @test outputs.output == fill(71f0, 2)
    reset!(graph)
    @test process!(outputs, graph, inputs) === TestGateRejected
    @test FilterGraphs.rejection_context(graph).inputs == (a=nothing,b=nothing,c=nothing)
    @test snapshot.inputs == metadata
end
