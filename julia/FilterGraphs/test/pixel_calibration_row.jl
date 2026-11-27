const PIXEL_CALIBRATION_ROW_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "pixel-calibration-row.conf")

@testset "production row calibration runs as a Graph Node" begin
    graph = prepare_graph(
        PIXEL_CALIBRATION_ROW_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationRowU16F32,),
    )
    @test graph.nodes[1].input_names == (:raw,)
    @test graph.nodes[1].output_names == (:calibrated,)
    @test graph.nodes[1].parameter_names == (:flat, :background)
    @test keys(graph.input_formats) == (:raw,)
    @test keys(graph.parameter_formats) == (:flat, :background)
    @test graph.node_failure_types == (PixelCalibrationRowFailure,)
    @test graph.output_conditional == (calibrated=false,)
    @test graph.input_formats.raw.schema == RAW_PIXEL_ROW_BLOCK_V1
    @test graph.output_formats.calibrated.schema == CALIBRATED_PIXEL_ROW_BLOCK_V1
    @test graph.input_formats.raw.rate == (0x00000002, 0x00000001)
    @test graph.parameter_formats.flat.schema == FLAT_CALIBRATION_V1
    @test graph.parameter_formats.flat.shape == (4, 3)
    @test graph.parameter_formats.background.schema == BACKGROUND_CALIBRATION_V1

    input = reshape(UInt16.(1:6), 2, 3)
    output = fill(-1.0f0, 2, 3)
    inputs = (raw=input,)
    outputs = (calibrated=output,)
    first_metadata = SampleMetadata(9, 0; timestamp_nanoseconds=100)

    @test process!(outputs, graph, inputs, (raw=first_metadata,)) ==
          (calibrated=true,)
    @test output == Float32.(input)
    @test FilterGraphs.output_metadata(graph) == (calibrated=first_metadata,)

    last_metadata = SampleMetadata(9, 2; timestamp_nanoseconds=200, terminal=true)
    @test process!(outputs, graph, inputs, (raw=last_metadata,)) ==
          (calibrated=true,)
    @test FilterGraphs.output_metadata(graph) == (calibrated=last_metadata,)

    @test replace_parameters!(
        graph,
        :flat => fill(2.0f0, 4, 3),
        :background => fill(1.0f0, 4, 3),
    ) === graph
    replacement_metadata = SampleMetadata(10, 0; timestamp_nanoseconds=300)
    @test process!(outputs, graph, inputs, (raw=replacement_metadata,)) ==
          (calibrated=true,)
    @test output == 2.0f0 .* (Float32.(input) .- 1.0f0)
    expected_replacement_metadata = SampleMetadata(
        replacement_metadata;
        discontinuity=true,
    )
    @test FilterGraphs.output_metadata(graph) ==
          (calibrated=expected_replacement_metadata,)

    fill!(output, 73.0f0)
    @test process!(outputs, graph, inputs) === PixelCalibrationRowMetadataMissing
    @test output == fill(73.0f0, 2, 3)
    @test FilterGraphs.output_metadata(graph) ==
          (calibrated=expected_replacement_metadata,)

    process!(outputs, graph, inputs, (raw=first_metadata,))
    @test @allocated(process!(outputs, graph, inputs, (raw=last_metadata,))) == 0
    @test @allocated(process!(outputs, graph, inputs)) == 0
end
