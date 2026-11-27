const BI_O_EDGE_IMAGE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = bi-o-edge
            label = bi-o-edge-image-f32
            config = {
                image_size = [4, 4]
                pupil_size = [2, 2]
                pupil_origins = [ [0 0] [2 0] [2 2] [0 2] ]
                bi_o_edge_support = [ true false true true ]
                initial_reference_signal = [ 0.1 0.3 0.4 -0.1 -0.3 -0.4 ]
                initial_optical_gain = [ 1.5 0.5 2.0 0.25 1.25 0.75 ]
                normalization_policy = mean-valid-flux
                incidence_flux = 0.0
                initial_calibration_signature = 42
            }
        }
    ]
    links = []
    inputs = [
        "bi-o-edge:image"
        "bi-o-edge:frame-calibration-signature"
        "bi-o-edge:pupil-origins"
        "bi-o-edge:bi-o-edge-support"
        "bi-o-edge:reference-signal"
        "bi-o-edge:optical-gain"
        "bi-o-edge:calibration-signature"
    ]
    outputs = [ "bi-o-edge:signal" "bi-o-edge:normalization-divisor" ]
}
"""

const BI_O_EDGE_IMAGE_INPUT_NAMES = (:image, Symbol("frame-calibration-signature"))
const BI_O_EDGE_IMAGE_OUTPUT_NAMES = (:signal, Symbol("normalization-divisor"))

bi_o_edge_image_inputs(image, signature) =
    NamedTuple{BI_O_EDGE_IMAGE_INPUT_NAMES}((image, signature))
bi_o_edge_image_outputs(signal, divisor) =
    NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((signal, divisor))

function prepare_bi_o_edge_image_graph()
    return prepare_graph_text(
        BI_O_EDGE_IMAGE_GRAPH_TEXT;
        algorithms=(BiOEdgeImageF32,),
    )
end

function bi_o_edge_graph_allocation_bytes(outputs, graph, inputs, metadata)
    process!(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

@testset "complete-frame Bi-O-edge graph preserves calibration and frame identity" begin
    graph = prepare_bi_o_edge_image_graph()
    @test map(node -> node.name, graph.nodes) == (Symbol("bi-o-edge"),)
    @test graph.node_failure_types == (BiOEdgeImageFailure,)
    @test graph.input_formats.image.shape == (4, 4)
    @test graph.input_formats.var"frame-calibration-signature".shape == (1,)
    @test graph.output_formats.signal.shape == (3, 2)
    @test keys(graph.parameter_formats) == (
        Symbol("pupil-origins"),
        Symbol("bi-o-edge-support"),
        Symbol("reference-signal"),
        Symbol("optical-gain"),
        Symbol("calibration-signature"),
    )

    image = Float32[
        10 30 5 7
        20 40 6 8
        1 3 9 11
        2 4 10 12
    ]
    signature = UInt64[42]
    signal = fill(-1.0f0, 3, 2)
    divisor = Float32[-1]
    inputs = bi_o_edge_image_inputs(image, signature)
    outputs = bi_o_edge_image_outputs(signal, divisor)
    sample = SampleMetadata(17; timestamp_nanoseconds=2_000, terminal=true)
    metadata = NamedTuple{BI_O_EDGE_IMAGE_INPUT_NAMES}((sample, sample))

    @test process!(outputs, graph, inputs, metadata) ==
          NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((true, true))
    @test divisor ≈ Float32[46.666668] rtol=2f-7
    @test vec(signal) ≈ Float32[
        0.010714274,
        0.09642856,
        0.5714285,
        0.008928573,
        0.77678573,
        0.68571424,
    ] rtol=2f-7
    @test FilterGraphs.output_metadata(graph) == (
        signal=sample,
        var"normalization-divisor"=sample,
    )

    retained = (copy(signal), copy(divisor))
    signature[1] = 43
    @test process!(outputs, graph, inputs, metadata) === BiOEdgeImageCalibrationMismatch
    @test signal == retained[1]
    @test divisor == retained[2]

    original = only(graph.nodes).prepared.plan
    replacement_reference = zeros(Float32, 3, 2)
    replacement_gain = ones(Float32, 3, 2)
    @test replace_parameters!(
        graph,
        Symbol("reference-signal") => replacement_reference,
        Symbol("optical-gain") => replacement_gain,
        Symbol("calibration-signature") => UInt64[43],
    ) === graph
    @test only(graph.nodes).prepared.plan !== original
    @test process!(outputs, graph, inputs, metadata) ==
          NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((true, true))
    replacement_expected = hcat(Float32[5, 23, 32], Float32[-3, 15, 24]) ./ divisor[1]
    @test signal ≈ replacement_expected rtol=2f-7

    retained_plan = only(graph.nodes).prepared.plan
    retained_signal = copy(signal)
    @test_throws DimensionMismatch replace_parameters!(
        graph,
        Symbol("reference-signal") => fill(0.25f0, 3, 2),
        Symbol("optical-gain") => ones(Float32, 2, 2),
        Symbol("calibration-signature") => UInt64[44],
    )
    @test only(graph.nodes).prepared.plan === retained_plan
    @test process!(outputs, graph, inputs, metadata) ==
          NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((true, true))
    @test signal == retained_signal
    @test bi_o_edge_graph_allocation_bytes(outputs, graph, inputs, metadata) == 0
end

@testset "supported Bi-O-edge boundary joins frame and signature by exact sequence" begin
    graph = prepare_bi_o_edge_image_graph()
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph)
    image = Float32[
        10 30 5 7
        20 40 6 8
        1 3 9 11
        2 4 10 12
    ]
    signature = UInt64[42]
    signal = fill(-1.0f0, 3, 2)
    divisor = Float32[-1]
    outputs = bi_o_edge_image_outputs(signal, divisor)

    @test process!(
        outputs,
        synchronized,
        :image,
        image,
        SampleMetadata(20; timestamp_nanoseconds=2_000),
    ) ==
          NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((false, false))
    @test process!(
        outputs,
        synchronized,
        Symbol("frame-calibration-signature"),
        signature,
        SampleMetadata(21; timestamp_nanoseconds=9_000),
    ) == NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((false, false))
    @test signal == fill(-1.0f0, 3, 2)
    @test divisor == Float32[-1]
    @test process!(
        outputs,
        synchronized,
        :image,
        image,
        SampleMetadata(21; timestamp_nanoseconds=2_100),
    ) ==
          NamedTuple{BI_O_EDGE_IMAGE_OUTPUT_NAMES}((true, true))
    @test vec(signal) ≈ Float32[
        0.010714274,
        0.09642856,
        0.5714285,
        0.008928573,
        0.77678573,
        0.68571424,
    ] rtol=2f-7
    @test FilterGraphs.output_metadata(graph).signal.sequence == 21
    @test FilterGraphs.output_metadata(graph).signal.timestamp_nanoseconds == 2_100
    counters = FilterGraphs.synchronized_input_counters(synchronized)
    @test counters.completed_windows == 1
    @test counters.expired_windows == 1
    @test counters.discontinuities == 1

    reset!(synchronized)
    first_metadata = SampleMetadata(22)
    process!(outputs, synchronized, :image, image, first_metadata)
    process!(
        outputs,
        synchronized,
        Symbol("frame-calibration-signature"),
        signature,
        first_metadata,
    )
    reset!(synchronized)
    @test @allocated(
        process!(outputs, synchronized, :image, image, first_metadata)
    ) == 0
    @test @allocated(
        process!(
            outputs,
            synchronized,
            Symbol("frame-calibration-signature"),
            signature,
            first_metadata,
        )
    ) == 0
end
