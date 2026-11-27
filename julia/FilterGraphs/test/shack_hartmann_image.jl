const SHACK_HARTMANN_IMAGE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = shack-hartmann-image-f32
            config = {
                image_size = [3, 7]
                image_schema = org.example.shwfs-frame/1
                subaperture_size = [2, 3]
                subaperture_count = 2
                initial_subaperture_origins = [ [0 1] [1 3] ]
                coordinate_scale = 1.0
                pixel_threshold = 0.0
                flux_threshold = 1.0
                reference_slopes = [ 0.25 -0.5 -0.75 0.25 ]
                active = [ true true ]
            }
        }
    ]
    links = []
    inputs = [
        "measure:image"
        "measure:subaperture-origins"
        "measure:coordinates"
        "measure:reference-slopes"
        "measure:thresholds"
        "measure:active"
    ]
    outputs = [ "measure:slopes" "measure:flux" "measure:validity" ]
}
"""

const SHACK_HARTMANN_IMAGE_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_image.toml"),
)

function shack_graph_row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function shack_graph_pair_matrix(::Type{T}, values) where {T}
    matrix = Matrix{T}(undef, length(values), 2)
    for index in eachindex(values)
        matrix[index, 1] = T(values[index][1])
        matrix[index, 2] = T(values[index][2])
    end
    return matrix
end

function shack_graph_outputs(slopes, flux, validity)
    return (slopes=slopes, flux=flux, validity=validity)
end

function shack_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "Shack-Hartmann image declaration prepares as a Graph Node" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_IMAGE_GRAPH_TEXT;
        algorithms=(ShackHartmannImageF32,),
    )
    fixture = SHACK_HARTMANN_IMAGE_FIXTURE
    configuration = fixture["configuration"]
    image = shack_graph_row_major_matrix(
        Float32,
        fixture["sample"]["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
    slopes = zeros(Float32, 2, 2)
    flux = zeros(Float32, 2)
    validity = fill(false, 2)
    inputs = (image=image,)
    outputs = shack_graph_outputs(slopes, flux, validity)

    @test map(node -> node.name, graph.nodes) == (:measure,)
    @test keys(graph.input_formats) == (:image,)
    @test keys(graph.parameter_formats) == (
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
    )
    @test keys(graph.output_formats) == (:slopes, :flux, :validity)
    @test graph.input_formats.image.schema == "org.example.shwfs-frame/1"
    @test graph.output_formats.slopes.schema == SHACK_HARTMANN_SLOPES_V1
    @test graph.output_formats.flux.schema == SHACK_HARTMANN_FLUX_V1
    @test graph.output_formats.validity.schema == SHACK_HARTMANN_VALIDITY_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("subaperture-origins"),
    ).schema == REGION_ORIGINS_V1
    @test graph.parameter_formats.coordinates.shape == (6, 2)

    @test process!(outputs, graph, inputs) ==
          (slopes=true, flux=true, validity=true)
    @test slopes == shack_graph_pair_matrix(Float32, fixture["sample"]["slopes"])
    @test flux == Float32.(fixture["sample"]["flux"])
    @test validity == Bool[true, true]
    @test shack_graph_allocation_bytes(outputs, graph, inputs) == 0

    metadata = SampleMetadata(81, 0; timestamp_nanoseconds=5678, terminal=true)
    @test process!(outputs, graph, inputs, (image=metadata,)) ==
          (slopes=true, flux=true, validity=true)
    @test all(==(metadata), values(FilterGraphs.output_metadata(graph)))
end

@testset "Shack-Hartmann calibration adopts as one Parameter transaction" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_IMAGE_GRAPH_TEXT;
        algorithms=(ShackHartmannImageF32,),
    )
    original = only(graph.nodes).prepared.plan
    coordinates = shack_graph_pair_matrix(
        Float32,
        SHACK_HARTMANN_IMAGE_FIXTURE["coordinate_replacement"]["values"],
    )
    invalid_thresholds = Float32[0 0; -1 0]
    @test_throws ArgumentError replace_parameters!(
        graph,
        :coordinates => coordinates,
        :thresholds => invalid_thresholds,
    )
    @test only(graph.nodes).prepared.plan === original

    @test replace_parameters!(
        graph,
        :coordinates => coordinates,
        :active => Bool[false, true],
    ) === graph
    @test only(graph.nodes).prepared.plan !== original

    fixture = SHACK_HARTMANN_IMAGE_FIXTURE
    configuration = fixture["configuration"]
    image = shack_graph_row_major_matrix(
        Float32,
        fixture["sample"]["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
    slopes = zeros(Float32, 2, 2)
    flux = zeros(Float32, 2)
    validity = fill(false, 2)
    @test process!(
        shack_graph_outputs(slopes, flux, validity),
        graph,
        (image=image,),
    ) == (slopes=true, flux=true, validity=true)
    coordinate_slopes = shack_graph_pair_matrix(
        Float32,
        fixture["coordinate_replacement"]["slopes"],
    )
    coordinate_slopes[1, :] .= 0.0f0
    @test slopes == coordinate_slopes
    @test flux == Float32.(fixture["sample"]["flux"])
    @test validity == Bool[true, true]
end
