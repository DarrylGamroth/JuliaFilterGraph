@testset "direct preparation matches standalone lowering" begin
    options = parse_arguments(["--graph", LEAKY, "--check"])
    direct = FilterGraphPipeWire.prepare_service_graph(LEAKY)
    lowered = _prepare_service_graph(options)
    try
        inputs = map(format -> ones(format.element_type, format.shape), FilterGraphs.input_formats(direct))
        first_outputs = map(format -> zeros(format.element_type, format.shape), FilterGraphs.output_formats(direct))
        second_outputs = map(copy, first_outputs)
        @test FilterGraphs.process!(first_outputs, direct, inputs) ==
            FilterGraphs.process!(second_outputs, lowered, inputs)
        @test first_outputs == second_outputs
        expected = map(copy, first_outputs)
        FilterGraphs.reset!(direct)
        FilterGraphs.process!(first_outputs, direct, inputs)
        @test first_outputs == expected
    finally
        close_execution(direct)
        close_execution(lowered)
    end
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY; execution=:unknown)
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY; capture_backend=:unknown)
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY; workers=2)
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY; worker_thread_ids=[3])
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY; matrix_layout=:sharded)
    @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(LEAKY;
        execution=:row_block, capture_backend=:cuda)
    @test FilterGraphPipeWire.algorithm_declarations(; builtins=false) == ()
    @test all(id -> !(id.name in ("CUDA", "AMDGPU")), keys(Base.loaded_modules))
end
