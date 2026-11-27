const DETECTOR_READOUT_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "detector-readout.conf")
const DETECTOR_READOUT_NUMERICAL_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "detector_readout.toml"),
)
const DETECTOR_READOUT_INPUT_NAMES = (:raw,)
const DETECTOR_READOUT_OUTPUT_NAMES = (Symbol("detector-frame"),)

detector_readout_inputs(raw) =
    NamedTuple{DETECTOR_READOUT_INPUT_NAMES}((raw,))
detector_readout_input_metadata(metadata) =
    NamedTuple{DETECTOR_READOUT_INPUT_NAMES}((metadata,))
detector_readout_outputs(frame) =
    NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((frame,))

function graph_readout_block(::Type{T}, values, regions, lines, samples) where {T}
    block = Array{T}(undef, regions, lines, samples)
    source = 1
    for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        block[region, line, sample] = values[source]
        source += 1
    end
    return block
end

function detector_readout_graph_allocation_bytes(
    outputs,
    graph,
    inputs,
    metadata,
)
    return @allocated process!(outputs, graph, inputs, metadata)
end

@testset "detector readout Algorithms are composable Nodes" begin
    graph = prepare_graph(
        DETECTOR_READOUT_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationReadoutU16F32, DetectorFrameAssemblerF32),
    )
    fixture = DETECTOR_READOUT_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    frame = fill(-1.0f0, 4, 4)
    outputs = detector_readout_outputs(frame)

    @test map(node -> node.name, graph.nodes) == (:calibrate, :assemble)
    @test keys(graph.input_formats) == (:raw,)
    @test keys(graph.parameter_formats) == (:flat, :background)
    @test keys(graph.output_formats) == DETECTOR_READOUT_OUTPUT_NAMES
    @test graph.input_formats.raw.shape == (2, 1, 4)
    @test graph.input_formats.raw.rate == (0x00000002, 0x00000001)
    @test graph.output_formats[1].schema == CALIBRATED_PIXELS_V1
    @test graph.output_conditional == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((true,))

    for block_fixture in fixture["blocks"]
        block = graph_readout_block(
            UInt16,
            block_fixture["raw"],
            length(configuration["detector_origins"]),
            configuration["block_lines"],
            configuration["region_samples"],
        )
        metadata = SampleMetadata(
            block_fixture["sequence"],
            block_fixture["ordinal"];
            timestamp_nanoseconds=block_fixture["timestamp_nanoseconds"],
            terminal=block_fixture["terminal"],
        )
        availability = process!(
            outputs,
            graph,
            detector_readout_inputs(block),
            detector_readout_input_metadata(metadata),
        )
        @test availability == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}(
            (block_fixture["terminal"],),
        )
        !block_fixture["terminal"] && @test frame == fill(-1.0f0, 4, 4)
    end
    @test vec(permutedims(frame)) == Float32.(fixture["expected"]["frame"])
    @test FilterGraphs.output_metadata(graph) == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((
        SampleMetadata(11, 0; timestamp_nanoseconds=200, terminal=true),
    ))
end

@testset "readout rejection abandons an earlier partial frame" begin
    graph = prepare_graph(
        DETECTOR_READOUT_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationReadoutU16F32, DetectorFrameAssemblerF32),
    )
    fixture = DETECTOR_READOUT_NUMERICAL_FIXTURE
    first = graph_readout_block(UInt16, fixture["blocks"][1]["raw"], 2, 1, 4)
    last = graph_readout_block(UInt16, fixture["blocks"][2]["raw"], 2, 1, 4)
    frame = fill(73.0f0, 4, 4)
    outputs = detector_readout_outputs(frame)
    first_inputs = detector_readout_inputs(first)

    @test process!(
        outputs,
        graph,
        first_inputs,
        detector_readout_input_metadata(SampleMetadata(40, 0)),
    ) == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((false,))
    @test process!(outputs, graph, first_inputs) ===
          PixelCalibrationReadoutMetadataMissing
    @test frame == fill(73.0f0, 4, 4)
    @test process!(
        outputs,
        graph,
        first_inputs,
        detector_readout_input_metadata(SampleMetadata(41, 0)),
    ) == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((false,))
    @test process!(
        outputs,
        graph,
        detector_readout_inputs(last),
        detector_readout_input_metadata(SampleMetadata(41, 1; terminal=true)),
    ) == NamedTuple{DETECTOR_READOUT_OUTPUT_NAMES}((true,))
    @test only(values(FilterGraphs.output_metadata(graph))).discontinuity
end

@testset "detector-readout Graph allocates no heap after warmup" begin
    graph = prepare_graph(
        DETECTOR_READOUT_GRAPH_FIXTURE;
        algorithms=(PixelCalibrationReadoutU16F32, DetectorFrameAssemblerF32),
    )
    fixture = DETECTOR_READOUT_NUMERICAL_FIXTURE
    first = graph_readout_block(UInt16, fixture["blocks"][1]["raw"], 2, 1, 4)
    last = graph_readout_block(UInt16, fixture["blocks"][2]["raw"], 2, 1, 4)
    outputs = detector_readout_outputs(zeros(Float32, 4, 4))
    first_inputs = detector_readout_inputs(first)
    last_inputs = detector_readout_inputs(last)
    first_metadata = detector_readout_input_metadata(SampleMetadata(30, 0))
    last_metadata = detector_readout_input_metadata(
        SampleMetadata(30, 1; terminal=true),
    )

    process!(outputs, graph, first_inputs, first_metadata)
    process!(outputs, graph, last_inputs, last_metadata)
    @test detector_readout_graph_allocation_bytes(
        outputs,
        graph,
        first_inputs,
        first_metadata,
    ) == 0
    @test detector_readout_graph_allocation_bytes(
        outputs,
        graph,
        last_inputs,
        last_metadata,
    ) == 0
    process!(outputs, graph, first_inputs)
    @test @allocated(process!(outputs, graph, first_inputs)) == 0
end
