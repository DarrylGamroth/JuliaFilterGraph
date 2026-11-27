named_port_bytes(outputs, graph, inputs, metadata) =
    @allocated process!(outputs, graph, inputs, metadata)

struct NamedOffsetVector <: AbstractVector{Float32}
    values::Vector{Float32}
end
Base.size(array::NamedOffsetVector) = size(array.values)
Base.axes(array::NamedOffsetVector) = (0:(length(array.values)-1),)
Base.getindex(array::NamedOffsetVector, index::Int) = array.values[index+1]

@testset "checked Graph Ports match names independently of field order" begin
    graph = metadata_join_graph()
    inputs = (c=fill(30f0, 2), a=fill(10f0, 2), b=fill(20f0, 2))
    outputs = (z=zeros(Float32, 2), y=zeros(Float32, 2), x=zeros(Float32, 2))
    metadata = (b=SampleMetadata(2), c=SampleMetadata(3), a=SampleMetadata(1))
    @test (@inferred process!(outputs, graph, inputs, metadata)) == (x=true,y=true,z=true)
    @test outputs == (z=inputs.c, y=inputs.b, x=inputs.a)
    @test FilterGraphs.output_metadata(graph) ==
        (x=metadata.a,y=metadata.b,z=metadata.c)
    @test named_port_bytes(outputs, graph, inputs, metadata) == 0
    before = deepcopy(outputs)
    for invalid in ((a=inputs.a,b=inputs.b,), merge(inputs,(unknown=inputs.a,)))
        @test_throws ArgumentError process!(outputs, graph, invalid, metadata)
        @test outputs == before
    end
    @test_throws ArgumentError process!(merge(outputs,(unknown=outputs.x,)), graph, inputs, metadata)
    @test_throws ArgumentError process!(outputs, graph, inputs, (a=nothing,b=nothing,))
    @test_throws ArgumentError process!(outputs, graph, inputs, merge(metadata,(unknown=nothing,)))
    @test outputs == before
    alias = (z=outputs.x, y=outputs.y, x=outputs.x)
    @test_throws ArgumentError process!(alias, graph, inputs, metadata)
    @test outputs == before

    @test_throws r"Graph input Port 'a'.*AbstractArray" process!(outputs, graph,
        merge(inputs,(a=1f0,)),metadata)
    @test_throws r"Graph input Port 'b'.*element type" process!(outputs, graph,
        merge(inputs,(b=ones(Float64,2),)),metadata)
    @test_throws r"Graph input Port 'a'.*shape" process!(outputs, graph,
        merge(inputs,(a=ones(Float32,3),)),metadata)
    @test_throws r"Graph input Port 'c'.*one-based" process!(outputs, graph,
        merge(inputs,(c=NamedOffsetVector(ones(Float32,2)),)),metadata)
    @test_throws r"Graph output Port 'y'.*shape" process!(
        merge(outputs,(y=zeros(Float32,3),)), graph, inputs, metadata)
    @test_throws r"metadata Port 'b'" process!(outputs, graph, inputs, merge(metadata,(b=1,)))
end

@testset "declared Algorithm errors name the operation and Port" begin
    prepared = prepare_algorithm(MetadataJoinF32, MetadataJoinConfiguration(2))
    a,b,c = fill(1f0,2),fill(2f0,2),fill(3f0,2)
    x,y,z = zeros(Float32,2),zeros(Float32,2),zeros(Float32,2)
    metadata = (c=SampleMetadata(3),a=SampleMetadata(1),b=nothing)
    @test (@inferred process!(x,y,z,prepared,a,b,c,metadata)) === nothing
    @test (x,y,z) == (a,b,c)
    @test_throws ArgumentError process!(x,y,z,prepared,a,b,c,merge(metadata,(extra=nothing,)))
    @test_throws r"Algorithm 'test-metadata-join-f32' input Port 'a'.*element type" process!(
        x,y,z,prepared,Float64.(a),b,c)
end

@testset "captured Ports and pipeline slots preserve named routing" begin
    graph = captured_feedback_graph()
    captured = prepare_captured_graph(graph, FeedbackTestDevice(); outputs=(:checked,:sum))
    inputs = (drive=ones(Float32,3),feedback=zeros(Float32,3))
    outputs = (sum=zeros(Float32,3),checked=zeros(Float32,3))
    metadata = (drive=SampleMetadata(11),feedback=SampleMetadata(7))
    @test process!(outputs,captured,inputs,metadata) == (checked=true,sum=true)
    @test outputs.sum == outputs.checked == ones(Float32,3)
    @test FilterGraphs.output_metadata(captured) ==
        (checked=metadata.drive,sum=metadata.drive)

    # The mock pipeline uses the same retained stream protocol as a captured
    # owner, and keeps different metadata for each delayed receipt.
    pipeline = prepare_captured_pipeline(graph, FeedbackTestDevice(); capacity=2)
    try
        bad = merge(inputs,(extra=inputs.drive,))
        @test_throws ArgumentError submit_captured_frame!(pipeline,bad,metadata,1)
        @test !pipeline.has_sequence
        for sequence in 1:2
            m = (drive=SampleMetadata(sequence),feedback=nothing)
            @test submit_captured_frame!(pipeline,inputs,m,sequence)
        end
        for _ in 1:6
            advance_captured_pipeline!(pipeline)
        end
        # The checked Graph exposes (:sum,Symbol("next-feedback"),:checked); order the sink
        # differently to ensure publication follows names rather than positions.
        sink = (checked=zeros(Float32,3),var"next-feedback"=zeros(Float32,3),sum=zeros(Float32,3))
        for sequence in 1:2
            receipt = take_captured_result!(sink,pipeline)
            @test receipt.sequence == sequence
            @test sink.sum == sink.checked == ones(Float32,3)
            @test sink.var"next-feedback" == fill(0.5f0,3)
            @test receipt.metadata.checked == SampleMetadata(sequence)
        end
    finally
        close(pipeline)
    end
end

named_default_port_bytes(outputs, graph, inputs) = @allocated process!(outputs, graph, inputs)
named_prepared_port_bytes(outputs, prepared, inputs, metadata) =
    @allocated process!(outputs..., prepared, inputs..., metadata)
named_latest_cycle_bytes(outputs, latest, primary, held, sequence) = @allocated begin
    FilterGraphs.accept_held!(latest, held, SampleMetadata(sequence))
    process!(outputs, latest, primary, SampleMetadata(sequence))
end

@testset "named boundaries retain default and direct prepared allocation budgets" begin
    graph = metadata_join_graph()
    inputs = (c=fill(3f0, 2), a=ones(Float32, 2), b=fill(2f0, 2))
    outputs = (z=zeros(Float32, 2), x=zeros(Float32, 2), y=zeros(Float32, 2))
    @test (@inferred process!(outputs, graph, inputs)) == (x=true, y=true, z=true)
    @test outputs == (z=inputs.c, x=inputs.a, y=inputs.b)
    named_default_port_bytes(outputs, graph, inputs)
    @test named_default_port_bytes(outputs, graph, inputs) == 0
    @test FilterGraphs.output_metadata(graph) == (x=nothing, y=nothing, z=nothing)
    @test_throws ArgumentError process!(outputs, graph,
        (unknown=inputs.a, b=inputs.b, c=inputs.c))

    prepared = prepare_algorithm(MetadataJoinF32, MetadataJoinConfiguration(2))
    metadata = (c=SampleMetadata(3), a=SampleMetadata(1), b=SampleMetadata(2))
    arrays = (inputs.a, inputs.b, inputs.c)
    destinations = (outputs.x, outputs.y, outputs.z)
    named_prepared_port_bytes(destinations, prepared, arrays, metadata)
    @test named_prepared_port_bytes(destinations, prepared, arrays, metadata) == 0
end

@testset "join owners publish reordered output Ports with source metadata" begin
    graph = metadata_join_graph()
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph)
    inputs = (a=ones(Float32, 2), b=fill(2f0, 2), c=fill(3f0, 2))
    outputs = (z=zeros(Float32, 2), y=zeros(Float32, 2), x=zeros(Float32, 2))
    metadata = (a=SampleMetadata(1; timestamp_nanoseconds=100),
        b=SampleMetadata(1; timestamp_nanoseconds=200),
        c=SampleMetadata(1; timestamp_nanoseconds=300))
    before = FilterGraphs.synchronized_input_counters(synchronized)
    @test_throws ArgumentError process!((z=outputs.z, y=outputs.y),
        synchronized, :a, inputs.a, metadata.a)
    @test FilterGraphs.synchronized_input_counters(synchronized) == before
    @test process!(outputs, synchronized, :c, inputs.c, metadata.c) == (x=false,y=false,z=false)
    @test process!(outputs, synchronized, :a, inputs.a, metadata.a) == (x=false,y=false,z=false)
    @test process!(outputs, synchronized, :b, inputs.b, metadata.b) == (x=true,y=true,z=true)
    @test outputs == (z=inputs.c, y=inputs.b, x=inputs.a)
    @test FilterGraphs.output_metadata(graph) == (x=metadata.a,y=metadata.b,z=metadata.c)

    graph = prepare_graph_text("""
        filter.graph = {
            nodes = [ { type = ndarray name = join label = test-latest-pair-f32 config = { extent = 2 } } ]
            links = [] inputs = [ "join:a" "join:b" ] outputs = [ "join:x" "join:y" ]
        }
        """; algorithms=(LatestPairF32,))
    latest = FilterGraphs.prepare_latest_hold(graph, :a, :b)
    sink = (y=zeros(Float32,2), x=zeros(Float32,2))
    FilterGraphs.accept_held!(latest, inputs.b, metadata.b)
    before = FilterGraphs.latest_hold_counters(latest)
    @test_throws ArgumentError process!((y=sink.y, extra=sink.x), latest, inputs.a, metadata.a)
    @test FilterGraphs.latest_hold_counters(latest) == before
    @test process!(sink, latest, inputs.a, metadata.a) == (x=true,y=true)
    @test sink == (y=inputs.b, x=inputs.a)
    @test FilterGraphs.output_metadata(graph) == (x=metadata.a,y=metadata.b)
    named_latest_cycle_bytes(sink, latest, inputs.a, inputs.b, 2)
    @test named_latest_cycle_bytes(sink, latest, inputs.a, inputs.b, 3) == 0
end

@testset "reordered pipeline Ports preserve drops and reused slots" begin
    pipeline = prepare_captured_pipeline(captured_feedback_graph(), FeedbackTestDevice(); capacity=2)
    inputs = (drive=ones(Float32,3), feedback=zeros(Float32,3))
    outputs = (checked=zeros(Float32,3), var"next-feedback"=zeros(Float32,3), sum=zeros(Float32,3))
    metadata(sequence) = (drive=SampleMetadata(sequence), feedback=SampleMetadata(100+sequence))
    try
        @test submit_captured_frame!(pipeline, inputs, metadata(1), 1)
        @test submit_captured_frame!(pipeline, inputs, metadata(2), 2)
        @test !submit_captured_frame!(pipeline, inputs, metadata(3), 3)
        @test_throws ArgumentError submit_captured_frame!(pipeline, inputs, metadata(3), 3)
        for _ in 1:6
            advance_captured_pipeline!(pipeline)
        end
        first = take_captured_result!(outputs, pipeline)
        @test first.sequence == 1
        @test first.metadata.checked == metadata(1).drive
        @test submit_captured_frame!(pipeline, inputs, metadata(4), 4)
        for _ in 1:6
            advance_captured_pipeline!(pipeline)
        end
        for sequence in (2,4)
            receipt = take_captured_result!(outputs, pipeline)
            @test receipt.sequence == sequence
            @test receipt.metadata.checked == metadata(sequence).drive
            @test outputs.checked == outputs.sum == ones(Float32,3)
            @test outputs.var"next-feedback" == fill(0.5f0,3)
        end
        @test captured_pipeline_counters(pipeline).drops == 1
        @test captured_pipeline_counters(pipeline).occupied == 0
    finally
        close(pipeline)
    end
end
