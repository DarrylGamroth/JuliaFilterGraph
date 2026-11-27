struct LifecycleTestDevice end

# Retain only host bookkeeping. The drain hook models an unsuccessful backend
# wait without GPU work, and counts any attempt to touch execution resources.
mutable struct LifecycleTestBackend
    faulted::Bool
    fail_drain::Bool
    drain_calls::Int
    block_submissions::UInt64
    terminal_waits::UInt64
end

FilterGraphs._prepare_gpu_dense_executor(prepared, ::LifecycleTestDevice, lanes) =
    LifecycleTestBackend(false, false, 0, UInt64(3), UInt64(1))

function FilterGraphs._drain_gpu_dense!(backend::LifecycleTestBackend)
    backend.drain_calls += 1
    if backend.fail_drain
        backend.faulted = true
        error("injected lifecycle drain failure")
    end
    return nothing
end

FilterGraphs._drain_gpu_readout!(backend::LifecycleTestBackend) =
    FilterGraphs._drain_gpu_dense!(backend)

FilterGraphs._gpu_dense_counters(backend::LifecycleTestBackend) = (
    measurement_limit=1.0f0,
    region_capacity=2,
    block_submissions=backend.block_submissions,
    terminal_waits=backend.terminal_waits,
)

FilterGraphs._gpu_readout_counters(backend::LifecycleTestBackend) = (
    region_capacity=2,
    block_submissions=backend.block_submissions,
    terminal_waits=backend.terminal_waits,
    captured_suffix_nodes=0,
)

function lifecycle_gpu_progressive_owner()
    return mktemp() do path, io
        write(io, PROGRESSIVE_SHACK_GRAPH_TEXT)
        flush(io)
        prepare_gpu_progressive_graph(path, LifecycleTestDevice(); algorithms=(
            ShackHartmannMeasurementBlockF32, IncrementalDenseReconstructorF32))
    end
end

function lifecycle_gpu_readout_owner(graph)
    backend = LifecycleTestBackend(false, false, 0, UInt64(3), UInt64(1))
    # This owner exercises host lifecycle admission; no device is prepared.
    return FilterGraphs.PreparedGpuReadoutGraph(
        graph, backend, graph.input_formats, graph.output_formats, graph.result_types,
        (), 1, FilterGraphAlgorithms.DetectorReadoutFrame(),
        NamedTuple(), NamedTuple(), NamedTuple(), nothing, false)
end

@testset "progressive owners retain host diagnostics after close" begin
    cpu = prepare_cpu_progressive_graph_text(PROGRESSIVE_SHACK_GRAPH_TEXT;
        algorithms=(ShackHartmannMeasurementBlockF32, IncrementalDenseReconstructorF32),
        workers=1)
    try
        @test isopen(cpu)
        replace_parameters!(cpu,
            :reconstructor => progressive_reconstructor(PROGRESSIVE_SHACK_FIXTURE))
        process_progressive_shack_frame!(zeros(Float32, 3), cpu, progressive_shack_image(), 801)
        snapshot = FilterGraphs.cpu_progressive_counters(cpu)
        @test !isempty(snapshot)
        close(cpu)
        @test !isopen(cpu)
        @test FilterGraphs.cpu_progressive_counters(cpu) == snapshot
        @test_throws ArgumentError reset!(cpu)
        close(cpu)
    finally
        close(cpu)
    end
end

@testset "GPU host diagnostics survive faults and failed drains" begin
    progressive = lifecycle_gpu_progressive_owner()
    backend = progressive.graph.nodes[2].prepared.workspace.executor
    readout = lifecycle_gpu_readout_owner(progressive.graph)
    for (owner, backend) in ((progressive, backend), (readout, readout.backend))
        try
            @test isopen(owner)
            snapshot = FilterGraphs.gpu_progressive_counters(owner)
            backend.faulted = true
            @test isopen(owner) # Resource lifetime and processing admission differ.
            @test FilterGraphs.gpu_progressive_counters(owner) == snapshot
            @test_throws ErrorException reset!(owner)
            @test backend.drain_calls == 0

            backend.fail_drain = true
            @test_throws ErrorException close(owner)
            @test isopen(owner)
            @test backend.drain_calls == 1
            @test FilterGraphs.gpu_progressive_counters(owner) == snapshot
            @test backend.drain_calls == 1

            backend.fail_drain = false
            close(owner)
            @test !isopen(owner)
            @test FilterGraphs.gpu_progressive_counters(owner) == snapshot
            @test backend.drain_calls == 2
            @test_throws ArgumentError reset!(owner)
            close(owner)
            @test backend.drain_calls == 2
        finally
            backend.fail_drain = false
            close(owner)
        end
    end
end

@testset "fixed-calibration owners reject replacement without changing state" begin
    graph = prepare_graph_text(CAPTURED_PASS_GRAPH; algorithms=(FastPassF32,))
    captured = prepare_captured_graph(graph, TestCapturedDevice())
    pipeline = prepare_captured_pipeline(graph, TestCapturedDevice())
    progressive = lifecycle_gpu_progressive_owner()
    readout = lifecycle_gpu_readout_owner(progressive.graph)
    plans = map(node -> node.prepared.plan, (graph.nodes..., readout.graph.nodes...))
    try
        for owner in (captured, pipeline, readout)
            published = owner isa PreparedCapturedPipeline ? owner.graph : owner
            metadata = FilterGraphs.output_metadata(published)
            failure = try
                replace_parameters!(owner, :reconstructor => ones(Float32, 3, 8))
            catch error
                error
            end
            @test failure isa ArgumentError
            @test occursin("prepare a fresh checked Graph", sprint(showerror, failure))
            @test FilterGraphs.output_metadata(published) == metadata
        end
        @test map(node -> node.prepared.plan, (graph.nodes..., readout.graph.nodes...)) === plans
        @test captured.nodes[1].adapter.adoptions == 0
        @test pipeline.graph.nodes[1].adapter.adoptions == 0
        @test captured_pipeline_counters(pipeline).occupied == 0
        @test readout.backend.drain_calls == 0
        @test isopen(readout)
        @test only(values(FilterGraphs.gpu_progressive_counters(readout))).block_submissions == 3
        inputs, outputs = (input=Float32[1, 2, 3],), (output=zeros(Float32, 3),)
        @test process!(outputs, captured, inputs) == (output=true,)
        @test outputs.output == inputs.input
        @test submit_captured_frame!(pipeline, inputs, 1)
        for _ in 1:4
            advance_captured_pipeline!(pipeline)
        end
        @test take_captured_result!(outputs, pipeline).result == (output=true,)
        @test outputs.output == inputs.input
    finally
        close(pipeline)
        close(readout)
        close(progressive)
    end
end
