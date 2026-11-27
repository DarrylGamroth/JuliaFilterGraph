const PYRAMID_IMAGE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = pyramid
            label = pyramid-image-f32
            config = {
                image_size = [4, 4]
                pupil_size = [2, 2]
                pupil_origins = [ [0 0] [2 0] [2 2] [0 2] ]
                i4q_support = [ true false true true ]
                initial_reference_signal = [ 0.0 0.0 0.0 0.0 0.0 0.0 ]
                initial_optical_gain = [ 1.0 1.0 1.0 1.0 1.0 1.0 ]
                normalization_policy = mean-valid-flux
                incidence_flux = 0.0
            }
        }
    ]
    links = []
    inputs = [
        "pyramid:image"
        "pyramid:pupil-origins"
        "pyramid:i4q-support"
        "pyramid:reference-signal"
        "pyramid:optical-gain"
    ]
    outputs = [ "pyramid:signal" "pyramid:normalization-divisor" ]
}
"""

const PYRAMID_IMAGE_INPUT_NAMES = (:image,)
const PYRAMID_IMAGE_OUTPUT_NAMES = (:signal, Symbol("normalization-divisor"))

pyramid_image_inputs(image) = NamedTuple{PYRAMID_IMAGE_INPUT_NAMES}((image,))
pyramid_image_outputs(signal, divisor) =
    NamedTuple{PYRAMID_IMAGE_OUTPUT_NAMES}((signal, divisor))

function pyramid_image_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "complete-frame pyramid image prepares with calibrated signal ports" begin
    graph = prepare_graph_text(
        PYRAMID_IMAGE_GRAPH_TEXT;
        algorithms=(PyramidImageF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:pyramid,)
    @test graph.node_failure_types == (PyramidImageFailure,)
    @test graph.input_formats.image.shape == (4, 4)
    @test graph.output_formats.signal.shape == (3, 2)
    @test graph.output_formats.var"normalization-divisor".shape == (1,)
    @test keys(graph.parameter_formats) ==
          (Symbol("pupil-origins"), Symbol("i4q-support"),
           Symbol("reference-signal"), Symbol("optical-gain"))

    image = permutedims(reshape(Float32.(1:16), 4, 4))
    signal = fill(-1.0f0, 3, 2)
    divisor = Float32[-1]
    outputs = pyramid_image_outputs(signal, divisor)
    inputs = pyramid_image_inputs(image)
    @test process!(outputs, graph, inputs) ==
          NamedTuple{PYRAMID_IMAGE_OUTPUT_NAMES}((true, true))
    @test divisor == Float32[32.0]
    @test signal == Float32[-0.5 -0.125; -0.5 -0.125; -0.5 -0.125]

    retained = (copy(signal), copy(divisor))
    bad = pyramid_image_inputs(fill(Float32(NaN), 4, 4))
    @test process!(outputs, graph, bad) === PyramidImageNonFiniteInput
    @test signal == retained[1]
    @test divisor == retained[2]
end

@testset "complete-frame pyramid image parameters replace atomically" begin
    graph = prepare_graph_text(
        PYRAMID_IMAGE_GRAPH_TEXT;
        algorithms=(PyramidImageF32,),
    )
    original = only(graph.nodes).prepared.plan
    @test replace_parameters!(
        graph,
        "pupil-origins" => UInt32[0 2; 0 0; 2 0; 2 2],
        "i4q-support" => Bool[false true; true true],
    ) === graph
    @test only(graph.nodes).prepared.plan !== original
    retained = only(graph.nodes).prepared.plan
    @test_throws DimensionMismatch replace_parameters!(
        graph, "i4q-support" => Bool[true false; false true],
    )
    @test only(graph.nodes).prepared.plan === retained
end

@testset "complete-frame pyramid image allocates nothing after warmup" begin
    graph = prepare_graph_text(
        PYRAMID_IMAGE_GRAPH_TEXT;
        algorithms=(PyramidImageF32,),
    )
    outputs = pyramid_image_outputs(zeros(Float32, 3, 2), zeros(Float32, 1))
    inputs = pyramid_image_inputs(permutedims(reshape(Float32.(1:16), 4, 4)))
    process!(outputs, graph, inputs)
    @test pyramid_image_allocation_bytes(outputs, graph, inputs) == 0
end
