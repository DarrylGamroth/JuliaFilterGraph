const SHACK_HARTMANN_RECONSTRUCTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = shwfs-reconstructor-f32
            config = {
                reconstructed_count = 3
                subaperture_count = 2
                initial_reconstructor = [
                     1.0  2.0 3.0  4.0
                    -1.0  0.0 2.0 -2.0
                     0.5 -0.5 1.5 -1.5
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
    ]
    links = []
    inputs = [ "reconstruct:slopes" "reconstruct:reconstructor" ]
    outputs = [ "reconstruct:reconstructed" ]
}
"""

const PYRAMID_RECONSTRUCTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = pwfs-reconstructor-f32
            config = {
                reconstructed_count = 3
                selected_coordinate_count = 2
                initial_reconstructor = [
                     1.0 0.0 0.0 0.0
                     0.0 1.0 1.0 0.0
                    -1.0 0.0 0.0 0.5
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
    ]
    links = []
    inputs = [ "reconstruct:signal" "reconstruct:reconstructor" ]
    outputs = [ "reconstruct:reconstructed" ]
}
"""

const DENSE_VECTOR_RECONSTRUCTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = dense-vector-reconstructor-f32
            config = {
                input_count = 3
                output_count = 2
                initial_reconstructor = [
                     1.0  10.0  100.0
                    -1.0 -10.0 -100.0
                ]
                input_schema = org.example.ao.ordered-input/1
                output_schema = org.example.ao.reconstructed-output/1
                reconstructor_schema = org.example.ao.dense-reconstructor/1
            }
        }
    ]
    links = []
    inputs = [ "reconstruct:ordered-input" "reconstruct:reconstructor" ]
    outputs = [ "reconstruct:output" ]
}
"""

const SHACK_HARTMANN_MEASURE_RECONSTRUCT_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = shack-hartmann-image-f32
            config = {
                image_size = [2, 2]
                image_schema = pipewireao/calibrated-pixels-f32/1
                subaperture_size = [1, 2]
                subaperture_count = 2
                initial_subaperture_origins = [ [0 0] [1 0] ]
                coordinate_scale = 1.0
                pixel_threshold = 0.0
                flux_threshold = 0.0
                reference_slopes = [ 0.0 0.0 0.0 0.0 ]
                active = [ true true ]
            }
        }
        {
            type = ndarray
            name = reconstruct
            label = shwfs-reconstructor-f32
            config = {
                reconstructed_count = 1
                subaperture_count = 2
                initial_reconstructor = [ 1.0 0.0 0.0 0.0 ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
    ]
    links = [
        { output = "measure:slopes" input = "reconstruct:slopes" }
    ]
    inputs = [
        "measure:image"
        "measure:subaperture-origins"
        "measure:coordinates"
        "measure:reference-slopes"
        "measure:thresholds"
        "measure:active"
        "reconstruct:reconstructor"
    ]
    outputs = [ "reconstruct:reconstructed" ]
}
"""

function reconstruction_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "complete-frame reconstructors prepare as Graph Nodes" begin
    shack_graph = prepare_graph_text(
        SHACK_HARTMANN_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(ShackHartmannReconstructorF32,),
    )
    slopes = Float32[2 -1; 0.5 4]
    reconstructed = zeros(Float32, 3)
    outputs = (reconstructed=reconstructed,)
    inputs = (slopes=slopes,)
    @test keys(shack_graph.input_formats) == (:slopes,)
    @test keys(shack_graph.parameter_formats) == (:reconstructor,)
    @test shack_graph.parameter_formats.reconstructor.schema ==
          SHWFS_RECONSTRUCTOR_V1
    @test process!(outputs, shack_graph, inputs) == (reconstructed=true,)
    @test reconstructed == Float32[17.5, -9, -3.75]
    @test reconstruction_graph_allocation_bytes(outputs, shack_graph, inputs) == 0

    pyramid_graph = prepare_graph_text(
        PYRAMID_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidReconstructorF32,),
    )
    signal = Float32[1 2; 3 4]
    fill!(reconstructed, 0.0f0)
    input_names = (:signal,)
    pyramid_inputs = NamedTuple{input_names}((signal,))
    @test keys(pyramid_graph.input_formats) == input_names
    @test pyramid_graph.parameter_formats.reconstructor.schema ==
          PYRAMID_I4Q_RECONSTRUCTOR_V1
    @test process!(outputs, pyramid_graph, pyramid_inputs) ==
          (reconstructed=true,)
    @test reconstructed == Float32[1, 5, 1]
    @test reconstruction_graph_allocation_bytes(
        outputs,
        pyramid_graph,
        pyramid_inputs,
    ) == 0
end

@testset "Shack-Hartmann measurement and reconstruction dogfood Graph links" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_MEASURE_RECONSTRUCT_GRAPH_TEXT;
        algorithms=(ShackHartmannImageF32, ShackHartmannReconstructorF32),
    )
    @test map(node -> node.name, graph.nodes) == (:measure, :reconstruct)
    @test keys(graph.input_formats) == (:image,)
    @test keys(graph.output_formats) == (:reconstructed,)

    image = Float32[1 2; 3 4]
    reconstructed = zeros(Float32, 1)
    outputs = (reconstructed=reconstructed,)
    inputs = (image=image,)
    @test process!(outputs, graph, inputs) == (reconstructed=true,)
    @test reconstructed ≈ Float32[1 / 6]
    @test reconstruction_graph_allocation_bytes(outputs, graph, inputs) == 0

    metadata = SampleMetadata(91, 0; timestamp_nanoseconds=9001, terminal=true)
    @test process!(outputs, graph, inputs, (image=metadata,)) ==
          (reconstructed=true,)
    @test FilterGraphs.output_metadata(graph).reconstructed == metadata
end

@testset "Graph reconstructor replacement is failure-atomic" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(ShackHartmannReconstructorF32,),
    )
    original = only(graph.nodes).prepared.plan
    invalid = zeros(Float32, 3, 4)
    invalid[1, 1] = NaN32
    @test_throws ArgumentError replace_parameters!(
        graph,
        :reconstructor => invalid,
    )
    @test only(graph.nodes).prepared.plan === original

    replacement = Float32[
        1 0 0 0
        0 1 0 0
        0 0 1 0
    ]
    @test replace_parameters!(graph, :reconstructor => replacement) === graph
    reconstructed = zeros(Float32, 3)
    @test process!(
        (reconstructed=reconstructed,),
        graph,
        (slopes=Float32[2 -1; 0.5 4],),
    ) == (reconstructed=true,)
    @test reconstructed == Float32[2, -1, 0.5]
end

@testset "dense vector reconstruction preserves Graph metadata and publication" begin
    graph = prepare_graph_text(
        DENSE_VECTOR_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(DenseVectorReconstructorF32,),
    )
    @test keys(graph.input_formats) == (Symbol("ordered-input"),)
    @test keys(graph.output_formats) == (:output,)
    @test graph.input_formats[Symbol("ordered-input")].schema ==
          "org.example.ao.ordered-input/1"
    @test graph.output_formats.output.schema ==
          "org.example.ao.reconstructed-output/1"
    @test graph.parameter_formats.reconstructor.schema ==
          "org.example.ao.dense-reconstructor/1"

    output = fill(71.0f0, 2)
    metadata = SampleMetadata(
        77,
        0;
        timestamp_nanoseconds=7_700,
        terminal=true,
        discontinuity=true,
    )
    input_name = Symbol("ordered-input")
    inputs = NamedTuple{(input_name,)}((Float32[2, 3, 4],))
    input_metadata = NamedTuple{(input_name,)}((metadata,))
    outputs = (output=output,)
    @test process!(outputs, graph, inputs, input_metadata) ==
          (output=true,)
    @test output == Float32[432, -432]
    @test FilterGraphs.output_metadata(graph).output == metadata

    @test replace_parameters!(
        graph,
        :reconstructor => fill(floatmax(Float32), 2, 3),
    ) === graph
    published_output = copy(output)
    published_metadata = FilterGraphs.output_metadata(graph)
    @test process!(
        outputs,
        graph,
        NamedTuple{(input_name,)}((Float32[2, 2, 2],)),
        input_metadata,
    ) === DenseReconstructionNonFiniteOutput
    @test output == published_output
    @test FilterGraphs.output_metadata(graph) == published_metadata
end
