using FilterGraphAlgorithms, FilterGraphs, Test

@enum LoanTestFailure::UInt8 LoanTestRejected=1

@algorithm LoanTestF32 begin
    label = "test-cpu-loan-f32"
    failure = LoanTestFailure
    @configuration LoanTestConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(c) = (
        input(:input, Float32, (c.extent,), "test.loan/1"),
        output(:output, Float32, (c.extent,), "test.loan/1"; metadata_from=:input),
    )
    prepare(c) = nothing
    process!(plan) = begin
        output[1] = input[1]
        input[1] < 0 && return LoanTestRejected
        output[2] = input[2]
        return nothing
    end
end

@testset "CPU loan slot publication" begin
    node = (name=:node, label="test-cpu-loan-f32", config=(; extent=2))
    document = (; nodes=(node,), inputs=(; input="node:input"), outputs=(; output="node:output"))
    algorithms = (LoanTestF32,)
    inputs = ((; input=Float32[1,2]), (; input=Float32[-1,2]))
    outputs = ((; output=zeros(Float32, 2)), (; output=zeros(Float32, 2)),
        (; output=zeros(Float32, 2)))
    owner = prepare_cpu_loans(document; algorithms, input_slots=inputs, output_slots=outputs)
    sample = SampleMetadata(1; terminal=true)
    @test process_cpu_loan!(owner, 1, 3, (; input=sample)) == (; output=true)
    @test outputs[3].output == inputs[1].input
    @test process_cpu_loan!(owner, 2, 2) == LoanTestRejected
    @test outputs[2].output[1] == -1
    @test FilterGraphs.output_metadata(owner).output == sample
    @test outputs[3].output == inputs[1].input
    @test_throws BoundsError process_cpu_loan!(owner, 3, 1)
    @test_throws BoundsError process_cpu_loan!(owner, 1, 4)
    @test_throws ArgumentError prepare_cpu_loans(document; algorithms, input_slots=inputs,
        output_slots=((; output=inputs[1].input),))
    @test_throws ArgumentError prepare_cpu_loans(document; algorithms, input_slots=inputs,
        output_slots=(outputs[1], outputs[1]))
    backing = zeros(Float32, 4)
    GC.@preserve backing begin
        first_view = unsafe_wrap(Array, pointer(backing), 2)
        second_view = unsafe_wrap(Array, pointer(backing, 2), 2)
        @test_throws ArgumentError prepare_cpu_loans(document; algorithms,
            input_slots=((; input=first_view),), output_slots=((; output=second_view),))
        @test_throws ArgumentError prepare_cpu_loans(document; algorithms,
            input_slots=inputs, output_slots=((; output=first_view), (; output=second_view)))
    end
    chain = (; nodes=(node, merge(node, (; name=:next))),
        links=("node:output"=>"next:input",), document.inputs,
        outputs=(; output="node:output"))
    @test_throws GraphConfigurationError prepare_cpu_loans(chain; algorithms,
        input_slots=inputs, output_slots=outputs)
    staged = prepare_graph(document; algorithms)
    sentinel = (; output=fill(42f0, 2))
    @test FilterGraphs.process!(sentinel, staged, inputs[2]) == LoanTestRejected
    @test sentinel.output == fill(42f0, 2)
    FilterGraphs.reset!(owner)
    inputs[1].input .= 7
    @test process_cpu_loan!(owner, 1, 1) == (; output=true)
    @test outputs[1].output == Float32[7,7]
    @test outputs[3].output == Float32[1,2]
end

@testset "CPU loan fixed output storage" begin
    producer = (name=:producer, label="test-cpu-loan-f32", config=(; extent=2))
    consumer = merge(producer, (; name=:consumer))
    document = (; nodes=(producer, consumer), links=("producer:output"=>"consumer:input",),
        inputs=(; input="producer:input"), outputs=(; output="consumer:output"))
    graph = prepare_graph(document; algorithms=(LoanTestF32,))
    fixed = Float32[0, 0, 0, 0]
    inputs = ((; input=Float32[1, 2]),)
    outputs = ((; output=zeros(Float32, 2)), (; output=zeros(Float32, 2)))
    GC.@preserve fixed begin
        intermediate = unsafe_wrap(Array, pointer(fixed), 2)
        overlap = unsafe_wrap(Array, pointer(fixed, 2), 2)
        node_outputs = ((intermediate,), graph.node_outputs[2])
        owner = FilterGraphs._prepare_cpu_loans(graph, inputs, outputs, node_outputs)
        @test process_cpu_loan!(owner, 1, 2) == (; output=true)
        @test intermediate == inputs[1].input
        @test outputs[1].output == zeros(Float32, 2)
        @test outputs[2].output == inputs[1].input
        destination = zeros(Float32, 2)
        inputs[1].input .= 9
        sample = SampleMetadata(7; terminal=true)
        @test FilterGraphs._process_cpu_output_loan!(owner, destination, (; input=sample)) ==
            (; output=true)
        @test intermediate == destination == Float32[9, 9]
        @test outputs[1].output == zeros(Float32, 2)
        @test outputs[2].output == Float32[1, 2]
        @test FilterGraphs.output_metadata(owner).output == sample
        @test_throws ArgumentError FilterGraphs._prepare_cpu_loans(graph,
            ((; input=overlap),), outputs, node_outputs)
        @test_throws ArgumentError FilterGraphs._prepare_cpu_loans(graph,
            inputs, ((; output=overlap),), node_outputs)
    end
end
