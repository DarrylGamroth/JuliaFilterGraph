function check_format_queries(owner, expected_inputs, expected_outputs, expected_parameters)
    @test (@inferred FilterGraphs.input_formats(owner)) === expected_inputs
    @test (@inferred FilterGraphs.output_formats(owner)) === expected_outputs
    @test (@inferred FilterGraphs.parameter_formats(owner)) === expected_parameters
    @test @allocated(FilterGraphs.input_formats(owner)) == 0
    @test @allocated(FilterGraphs.output_formats(owner)) == 0
    @test @allocated(FilterGraphs.parameter_formats(owner)) == 0
end

@testset "logical Graph format queries" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(ShackHartmannReconstructorF32,),
    )
    check_format_queries(graph, graph.input_formats, graph.output_formats, graph.parameter_formats)
    @test keys(FilterGraphs.input_formats(graph)) == (:slopes,)
    @test keys(FilterGraphs.parameter_formats(graph)) == (:reconstructor,)
    @test FilterGraphs.parameter_formats(graph).reconstructor.shape == (3, 4)
    @test FilterGraphs.parameter_formats(graph).reconstructor.schema == SHWFS_RECONSTRUCTOR_V1
    @test FilterGraphs.input_formats(graph).slopes.element_type === Float32
    @test FilterGraphs.input_formats(graph).slopes.rate == (UInt32(1), UInt32(1))
    replace_parameters!(graph, :reconstructor => fill(1.0f0, 3, 4))
    @test FilterGraphs.parameter_formats(graph) === graph.parameter_formats

    cpu = prepare_cpu_progressive_graph(
        GRAPH_FIXTURE; algorithms=(LeakyIntegratorF32, TestGateF32), workers=0,
    )
    try
        check_format_queries(cpu, cpu.graph.input_formats, cpu.graph.output_formats,
            cpu.graph.parameter_formats)
    finally
        close(cpu)
    end
    # Immutable schema remains inspectable after the execution resources close.
    @test FilterGraphs.input_formats(cpu) === cpu.graph.input_formats

    latest = FilterGraphs.prepare_latest_hold(prepare_latest_hold_test_graph(), :primary, :held)
    check_format_queries(latest, latest.graph.input_formats, latest.graph.output_formats,
        latest.graph.parameter_formats)
    @test keys(FilterGraphs.input_formats(latest)) == (:primary, :held)
    @test FilterGraphs.input_formats(latest).primary.rate == (UInt32(2), UInt32(1))
    @test FilterGraphs.input_formats(latest).held.rate == (UInt32(1), UInt32(1))

    synchronized = FilterGraphs.prepare_synchronized_inputs(prepare_synchronized_sum_graph())
    check_format_queries(synchronized, synchronized.graph.input_formats,
        synchronized.graph.output_formats, synchronized.graph.parameter_formats)
    @test keys(FilterGraphs.input_formats(synchronized)) == (:a, :b, :c)
end

@testset "captured format queries preserve feedback and selected outputs" begin
    graph = captured_feedback_graph()
    captured = prepare_captured_graph(graph, FeedbackTestDevice();
        feedback=CAPTURED_FEEDBACK_BINDING, outputs=(:checked, :sum))
    check_format_queries(captured, captured.input_formats, captured.output_formats,
        graph.parameter_formats)
    @test keys(FilterGraphs.input_formats(graph)) == (:feedback, :drive)
    @test keys(FilterGraphs.input_formats(captured)) == (:drive,)
    @test keys(FilterGraphs.output_formats(captured)) == (:checked, :sum)
    @test FilterGraphs.output_formats(captured).sum === graph.output_formats.sum
    @test_throws ArgumentError FilterGraphs.replace_properties!(captured, "sum:gain" => 1.0f0)

    split = prepare_graph_text(CAPTURED_MULTI_OUTPUT_GRAPH; algorithms=(CapturedSplitF32,))
    reordered = prepare_captured_graph(split, TestCapturedDevice(); outputs=(:second, :first))
    @test keys(FilterGraphs.output_formats(reordered)) == (:second, :first)
    @test FilterGraphs.output_formats(reordered).first === split.output_formats.first
    selected = prepare_captured_graph(split, TestCapturedDevice(); outputs=(:second,))
    @test keys(FilterGraphs.output_formats(selected)) == (:second,)

    pipeline_graph = prepare_graph_text(CAPTURED_PASS_GRAPH; algorithms=(FastPassF32,))
    pipeline = prepare_captured_pipeline(pipeline_graph, TestCapturedDevice())
    try
        check_format_queries(pipeline, pipeline.graph.input_formats,
            pipeline.graph.output_formats, pipeline_graph.parameter_formats)
    finally
        close(pipeline)
    end
end

@testset "source-only logical input format is empty" begin
    source = prepare_graph(joinpath(@__DIR__, "data", "examples", "docrime-excitation", "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms())
    @test (@inferred FilterGraphs.input_formats(source)) === NamedTuple()
    @test !isempty(FilterGraphs.output_formats(source))
    @test (@inferred FilterGraphs.parameter_formats(source)) === NamedTuple()
end

@testset "precompiled dense Port types match execution buffers" begin
    formats = (image=FilterGraphAlgorithms.AlgorithmFormat(UInt16, (4, 3), "test.image/1"),
               command=FilterGraphAlgorithms.AlgorithmFormat(Float32, (5,), "test.command/1"))
    @test FilterGraphs._external_buffer_type(formats) === typeof(FilterGraphs._external_buffers(formats))
    @test FilterGraphs._external_buffer_type((;)) === typeof((;))
end
