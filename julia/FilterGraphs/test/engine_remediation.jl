# Host-only regression coverage for the checked join and captured pipeline paths.
@algorithm MetadataJoinF32 begin
    label = "test-metadata-join-f32"
    @configuration MetadataJoinConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:a, Float32, (configuration.extent,), "test.f32/1"),
        input(:b, Float32, (configuration.extent,), "test.f32/1"),
        input(:c, Float32, (configuration.extent,), "test.f32/1"),
        output(:x, Float32, (configuration.extent,), "test.f32/1"; metadata_from=:a),
        output(:y, Float32, (configuration.extent,), "test.f32/1"; metadata_from=:b),
        output(:z, Float32, (configuration.extent,), "test.f32/1"; metadata_from=:c),
    )
    prepare(configuration) = configuration.extent
    process!(plan) = begin
        copyto!(x, a)
        copyto!(y, b)
        copyto!(z, c)
        nothing
    end
end

@algorithm LatestPairF32 begin
    label = "test-latest-pair-f32"
    @configuration LatestPairConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:a, Float32, (configuration.extent,), "test.f32/1"; rate=(2, 1)),
        input(:b, Float32, (configuration.extent,), "test.f32/1"),
        output(:x, Float32, (configuration.extent,), "test.f32/1"; rate=(2, 1), metadata_from=:a),
        output(:y, Float32, (configuration.extent,), "test.f32/1"; rate=(2, 1), metadata_from=:b),
    )
    prepare(configuration) = configuration.extent
    process!(plan) = begin
        copyto!(x, a)
        copyto!(y, b)
        nothing
    end
end

function metadata_join_graph()
    return prepare_graph_text("""
        filter.graph = {
            nodes = [ { type = ndarray name = join label = test-metadata-join-f32 config = { extent = 2 } } ]
            links = [] inputs = [ "join:a" "join:b" "join:c" ]
            outputs = [ "join:x" "join:y" "join:z" ]
        }
        """; algorithms=(MetadataJoinF32,))
end

join_arrival_bytes(y, join, name, x, metadata) = @allocated process!(y, join, name, x, metadata)

@testset "synchronized discontinuity applies to every retained Port" begin
    graph = metadata_join_graph()
    join = FilterGraphs.prepare_synchronized_inputs(graph)
    inputs = (a=ones(Float32, 2), b=fill(2f0, 2), c=fill(3f0, 2))
    outputs = (x=zeros(Float32, 2), y=zeros(Float32, 2), z=zeros(Float32, 2))
    orders = ((:a, :b, :c), (:a, :c, :b), (:b, :a, :c),
        (:b, :c, :a), (:c, :a, :b), (:c, :b, :a))
    sequence = 0
    for order in orders, flagged in keys(inputs)
        sequence += 1
        metadata = NamedTuple{keys(inputs)}(ntuple(i -> SampleMetadata(sequence;
            timestamp_nanoseconds=100i, discontinuity=keys(inputs)[i] === flagged), 3))
        for name in order
            process!(outputs, join, name, inputs[name], metadata[name])
        end
        published = values(FilterGraphs.output_metadata(graph))
        for i in 1:3
            @test published[i] == SampleMetadata(metadata[i]; discontinuity=true)
        end
        @test outputs == (x=inputs.a, y=inputs.b, z=inputs.c)
    end
    # A duplicate can announce the discontinuity after its Port's first arrival.
    sequence += 1
    process!(outputs, join, :a, inputs.a, SampleMetadata(sequence))
    process!(outputs, join, :a, inputs.a, SampleMetadata(sequence; discontinuity=true))
    process!(outputs, join, :b, inputs.b, SampleMetadata(sequence))
    process!(outputs, join, :c, inputs.c, SampleMetadata(sequence))
    @test all(m -> m.discontinuity, values(FilterGraphs.output_metadata(graph)))
    @test FilterGraphs.synchronized_input_counters(join).duplicate_arrivals == 1
    # Explicit sequence reset and a corrupted late input keep publication atomic.
    before = deepcopy(outputs)
    process!(outputs, join, :a, inputs.a, SampleMetadata(0; discontinuity=true))
    process!(outputs, join, :b, inputs.b, SampleMetadata(0; corrupted=true))
    process!(outputs, join, :c, inputs.c, SampleMetadata(0))
    @test outputs == before
    @test FilterGraphs.synchronized_input_counters(join).corrupted_windows == 1
    for s in 1:3
        m = SampleMetadata(s)
        process!(outputs, join, :a, inputs.a, m)
        process!(outputs, join, :b, inputs.b, SampleMetadata(s; discontinuity=true))
        @test join_arrival_bytes(outputs, join, :c, inputs.c, m) == 0
    end
end

@testset "join output alias admission precedes any state mutation" begin
    graph = metadata_join_graph()
    join = FilterGraphs.prepare_synchronized_inputs(graph)
    storage = fill(71f0, 8)
    x = ones(Float32, 2)
    for second in (view(storage, 1:2), view(storage, 2:3))
        outputs = (x=view(storage, 1:2), y=second, z=view(storage, 5:6))
        before = FilterGraphs.synchronized_input_counters(join)
        @test_throws ArgumentError process!(outputs, join, :a, x, SampleMetadata(1))
        @test FilterGraphs.synchronized_input_counters(join) == before
        @test !join.state.active
        @test storage == fill(71f0, 8)
    end
    disjoint = (x=view(storage, 1:2), y=view(storage, 3:4), z=view(storage, 5:6))
    for name in (:a, :b, :c)
        process!(disjoint, join, name, x, SampleMetadata(1))
    end
    @test all(==(1f0), storage[1:6])

    graph = prepare_graph_text("""
        filter.graph = {
            nodes = [ { type = ndarray name = join label = test-latest-pair-f32 config = { extent = 2 } } ]
            links = [] inputs = [ "join:a" "join:b" ] outputs = [ "join:x" "join:y" ]
        }
        """; algorithms=(LatestPairF32,))
    latest = FilterGraphs.prepare_latest_hold(graph, :a, :b)
    FilterGraphs.accept_held!(latest, fill(2f0, 2), SampleMetadata(1))
    fill!(storage, 71f0)
    for second in (view(storage, 1:2), view(storage, 2:3))
        outputs = (x=view(storage, 1:2), y=second)
        before = FilterGraphs.latest_hold_counters(latest)
        @test_throws ArgumentError process!(outputs, latest, x, SampleMetadata(1))
        @test FilterGraphs.latest_hold_counters(latest) == before
        @test latest.state.fresh
        @test storage == fill(71f0, 8)
    end
    outputs = (x=view(storage, 1:2), y=view(storage, 3:4))
    @test process!(outputs, latest, x, SampleMetadata(1)) == (x=true, y=true)
    @test outputs.x == x
    @test outputs.y == fill(2f0, 2)
end

# Synchronous CPU backend isolates host bookkeeping from device library costs.
FilterGraphs._prepare_captured_pipeline_backend(::PreparedCapturedGraph{TestCapturedContext}) = nothing
FilterGraphs._prepare_captured_pipeline_events(::TestCapturedContext, inputs, outputs) = nothing
FilterGraphs._enqueue_captured_pipeline_input!(::Nothing, ::TestCapturedContext, slot) =
    FilterGraphs._copy_captured_outputs!(values(slot.device_inputs), values(slot.host_inputs))
FilterGraphs._captured_pipeline_input_done(::Nothing) = true
FilterGraphs._captured_pipeline_graph_done(::Nothing) = true
FilterGraphs._captured_pipeline_output_done(::Nothing) = true
function FilterGraphs._enqueue_captured_pipeline_graph!(::Nothing, context::TestCapturedContext, graph, slot)
    FilterGraphs._copy_captured_outputs!(values(graph.device_inputs), values(slot.device_inputs))
    FilterGraphs._launch_captured_graph!(context, graph.executable)
    FilterGraphs._copy_captured_outputs!(values(slot.device_outputs), values(graph.device_outputs))
    slot.host_status[1] = graph.executable.frame_status.value[1]
    nothing
end
FilterGraphs._enqueue_captured_pipeline_output!(::Nothing, ::TestCapturedContext, slot) =
    FilterGraphs._copy_captured_outputs!(values(slot.host_outputs), values(slot.device_outputs))
FilterGraphs._drain_captured_pipeline_backend!(::Nothing, ::TestCapturedContext) = nothing

pipeline_submit_bytes(p, x, m, n) = @allocated submit_captured_frame!(p, x, m, n)
pipeline_advance_bytes(p) = @allocated advance_captured_pipeline!(p)
pipeline_take_bytes(y, p) = @allocated take_captured_result!(y, p)

@testset "captured pipeline typed receipt and zero warmed host bookkeeping" begin
    for text in (CAPTURED_PASS_GRAPH, CAPTURED_PASS_GATE_GRAPH)
        graph = prepare_graph_text(text; algorithms=(FastPassF32, CapturedGateF32))
        pipeline = prepare_captured_pipeline(graph, TestCapturedDevice())
        inputs, outputs = (input=ones(Float32, 3),), (output=fill(71f0, 3),)
        try
            for sequence in 1:4
                metadata = (input=SampleMetadata(sequence),)
                @test pipeline_submit_bytes(pipeline, inputs, metadata, sequence) == 0
                @test pipeline_advance_bytes(pipeline) == 0
                @test pipeline_advance_bytes(pipeline) == 0
                @test pipeline_take_bytes(outputs, pipeline) == 0
                @test outputs.output == inputs.input
            end
            # Empty polling and full/drop are ordinary allocation-free statuses.
            @test pipeline_take_bytes(outputs, pipeline) == 0
            @test take_captured_result!(outputs, pipeline) === nothing
            @test submit_captured_frame!(pipeline, inputs, (input=SampleMetadata(5),), 5)
            @test submit_captured_frame!(pipeline, inputs, 6)
            @test !submit_captured_frame!(pipeline, inputs, 7)
            @test pipeline_submit_bytes(pipeline, inputs, (input=nothing,), 8) == 0
            @test captured_pipeline_counters(pipeline).drops == 2
            for _ in 1:4
                advance_captured_pipeline!(pipeline)
            end
            receipt = take_captured_result!(outputs, pipeline)
            @test receipt.sequence == 5
            expected_metadata = text === CAPTURED_PASS_GRAPH ? SampleMetadata(5) : nothing
            @test receipt.metadata == (output=expected_metadata,)
            @test receipt.result == (output=true,)
            receipt = take_captured_result!(outputs, pipeline)
            @test receipt.sequence == 6
            @test receipt.metadata == (output=nothing,)
            if text === CAPTURED_PASS_GATE_GRAPH
                fill!(inputs.input, -1f0)
                fill!(outputs.output, 71f0)
                for sequence in 9:11
                    metadata = (input=SampleMetadata(sequence),)
                    @test pipeline_submit_bytes(pipeline, inputs, metadata, sequence) == 0
                    @test pipeline_advance_bytes(pipeline) == 0
                    @test pipeline_advance_bytes(pipeline) == 0
                    if sequence == 11
                        receipt = take_captured_result!(outputs, pipeline)
                        @test receipt.result === TestGateRejected
                        @test receipt.metadata === nothing
                    else
                        @test pipeline_take_bytes(outputs, pipeline) == 0
                    end
                    @test outputs.output == fill(71f0, 3)
                end
            end
            inferred = only(Base.return_types(take_captured_result!, Tuple{typeof(outputs),typeof(pipeline)}))
            for type in Base.uniontypes(inferred)
                type === Nothing && continue
                @test isconcretetype(type)
                @test fieldtype(type, :result) !== Any
                @test fieldtype(type, :metadata) !== Any
            end
        finally
            close(pipeline)
        end
    end
end

@testset "exception payloads and parsed Node storage retain concrete types" begin
    for cause in (ErrorException("test"), :arbitrary_thrown_symbol, UInt32(7), nothing)
        fault = CapturedGraphFault(:execution, cause)
        @test fault.cause === cause
        @test fieldtype(typeof(fault), :cause) === typeof(cause)
        handoff = FilterGraphs._cpu_handoff(nothing, 1)
        failure = try
            throw(cause)
        catch error
            FilterGraphs.CpuWorkerFailure(error, catch_backtrace())
        end
        @test !isempty(failure.backtrace)
        FilterGraphs._cpu_publish_completion!(handoff, 1, UInt64(1), failure)
        @test take!(only(handoff.completions)).cause === cause
    end
    graph = metadata_join_graph()
    spec = FilterGraphs._parse_spa_json("config = { values = [1 2 3] }")
    fields = FilterGraphs._configuration_fields(spec["config"])
    @test fields["values"] isa Vector{Int}
    @test graph.nodes[1].prepared.plan == 2
    heterogeneous = FilterGraphs._configuration_fields(
        FilterGraphs._parse_spa_json("array = [1 true 2.5 text] object = { scalar = 1 }"))
    @test eltype(heterogeneous["array"]) == Union{Int,Bool,Float64,String}
    @test heterogeneous["object"] isa AbstractDict
    @test valtype(typeof(heterogeneous)) !== Any
    @test valtype(typeof(heterogeneous["object"])) === Int
    node = FilterGraphs._node_spec(FilterGraphs._parse_spa_json(
        "type = ndarray name = test label = test config = { extent = 2 } props = { gain = 1.0 }"))
    @test fieldtype(typeof(node), :configuration) === Dict{String,Int}
    @test fieldtype(typeof(node), :properties) === Dict{String,Float64}
end
