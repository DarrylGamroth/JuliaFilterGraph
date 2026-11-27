const CENTER_OF_GRAVITY_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = center-of-gravity-f32
            config = {
                pixel_count = 3
                initial_coordinates = [ [-1.0 0.0] [0.0 1.0] [1.0 2.0] ]
            }
            props = {
                reference-centroid-x = 0.25
                reference-centroid-y = -0.5
                flux-threshold = 4.0
            }
        }
    ]
    links = []
    inputs = [ "measure:pixels" "measure:coordinates" ]
    outputs = [ "measure:centroid-offset" "measure:flux" "measure:validity" ]
}
"""

const CENTER_OF_GRAVITY_OUTPUT_NAMES = (
    Symbol("centroid-offset"),
    :flux,
    :validity,
)

function center_of_gravity_graph_outputs(offset, flux, validity)
    return NamedTuple{CENTER_OF_GRAVITY_OUTPUT_NAMES}((offset, flux, validity))
end

function center_of_gravity_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "center-of-gravity declaration prepares as a Graph Node" begin
    graph = prepare_graph_text(
        CENTER_OF_GRAVITY_GRAPH_TEXT;
        algorithms=(CenterOfGravityF32,),
    )
    pixels = Float32[0, 0, 4]
    offset = fill(NaN32, 2)
    flux = fill(NaN32, 1)
    validity = fill(false, 1)
    inputs = (pixels=pixels,)
    outputs = center_of_gravity_graph_outputs(offset, flux, validity)

    @test map(node -> node.name, graph.nodes) == (:measure,)
    @test keys(graph.input_formats) == (:pixels,)
    @test keys(graph.parameter_formats) == (:coordinates,)
    @test keys(graph.output_formats) == CENTER_OF_GRAVITY_OUTPUT_NAMES
    @test graph.input_formats.pixels.schema == CENTER_OF_GRAVITY_PIXELS_V1
    @test graph.parameter_formats.coordinates.shape == (3, 2)
    @test graph.parameter_formats.coordinates.schema ==
          CENTER_OF_GRAVITY_COORDINATES_V1
    @test getproperty(
        graph.output_formats,
        Symbol("centroid-offset"),
    ).schema == CENTER_OF_GRAVITY_OFFSET_V1
    @test graph.output_formats.flux.schema == CENTER_OF_GRAVITY_FLUX_V1
    @test graph.output_formats.validity.schema == CENTER_OF_GRAVITY_VALIDITY_V1

    properties = property_values(only(graph.nodes).prepared.plan)
    @test getproperty(properties, Symbol("reference-centroid-x")) === 0.25f0
    @test getproperty(properties, Symbol("reference-centroid-y")) === -0.5f0
    @test getproperty(properties, Symbol("flux-threshold")) === 4.0f0

    @test process!(outputs, graph, inputs) == NamedTuple{CENTER_OF_GRAVITY_OUTPUT_NAMES}(
        (true, true, true),
    )
    @test offset == Float32[0.75, 2.5]
    @test flux == Float32[4]
    @test validity == Bool[true]
    @test center_of_gravity_graph_allocation_bytes(outputs, graph, inputs) == 0

    metadata = SampleMetadata(71, 0; timestamp_nanoseconds=1234, terminal=true)
    @test process!(outputs, graph, inputs, (pixels=metadata,)) ==
          NamedTuple{CENTER_OF_GRAVITY_OUTPUT_NAMES}((true, true, true))
    @test all(==(metadata), values(FilterGraphs.output_metadata(graph)))
end

@testset "center-of-gravity coordinates replace atomically at the Graph boundary" begin
    graph = prepare_graph_text(
        CENTER_OF_GRAVITY_GRAPH_TEXT;
        algorithms=(CenterOfGravityF32,),
    )
    original = only(graph.nodes).prepared.plan
    invalid = Float32[-2 0; 0 NaN; 2 4]
    @test_throws ArgumentError replace_parameters!(graph, :coordinates => invalid)
    @test only(graph.nodes).prepared.plan === original

    replacement = Float32[-2 0; 0 2; 2 4]
    @test replace_parameters!(graph, :coordinates => replacement) === graph
    @test only(graph.nodes).prepared.plan !== original

    offset = zeros(Float32, 2)
    flux = zeros(Float32, 1)
    validity = fill(false, 1)
    outputs = center_of_gravity_graph_outputs(offset, flux, validity)
    @test process!(outputs, graph, (pixels=Float32[0, 0, 4],)) ==
          NamedTuple{CENTER_OF_GRAVITY_OUTPUT_NAMES}((true, true, true))
    @test offset == Float32[1.75, 4.5]
    @test flux == Float32[4]
    @test validity == Bool[true]
end
