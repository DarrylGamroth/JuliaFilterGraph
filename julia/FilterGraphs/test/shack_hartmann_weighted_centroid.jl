const SHACK_HARTMANN_WEIGHTED_CENTROID_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = shack-hartmann-weighted-centroid-f32
            config = {
                image_size = [2, 3]
                image_schema = org.example.shwfs-frame/1
                subaperture_size = [2, 3]
                subaperture_count = 1
                initial_subaperture_origins = [ [0 0] ]
                coordinate_scale = 1.0
                pixel_threshold = 2.0
                flux_threshold = 4.0
                reference_slopes = [ 0.0 0.0 ]
                active = [ true ]
                initial_weights = [ [0.0 1.0 2.0 3.0 0.5 1.0] ]
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
        "measure:weights"
    ]
    outputs = [ "measure:slopes" "measure:flux" "measure:validity" ]
}
"""

@testset "weighted Shack-Hartmann declaration parses and prepares as a Graph Node" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_WEIGHTED_CENTROID_GRAPH_TEXT;
        algorithms=(ShackHartmannWeightedCentroidF32,),
    )
    plan = only(graph.nodes).prepared.plan
    @test plan.simd_width == 4
    @test plan.weights isa Matrix{Float32}
    @test size(plan.weights) == (1, 6)
    @test vec(plan.weights) == Float32[0, 1, 2, 3, 0.5, 1]
    @test keys(graph.parameter_formats) == (
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
        :weights,
    )
    @test graph.parameter_formats.weights.shape == (1, 6)
    @test graph.parameter_formats.weights.schema == SHACK_HARTMANN_WEIGHTS_V1
    @test graph.output_formats.flux.schema == WEIGHTED_SHACK_HARTMANN_FLUX_V1

    slopes = zeros(Float32, 1, 2)
    flux = zeros(Float32, 1)
    validity = falses(1)
    @test process!(
        (slopes=slopes, flux=flux, validity=validity),
        graph,
        (image=Float32[1 2 3; 7 8 9],),
    ) == (slopes=true, flux=true, validity=true)
    @test slopes == Float32[-1 / 7 13 / 42]
    @test flux == Float32[42]
    @test validity == Bool[true]
end

const SHACK_HARTMANN_WEIGHTED_BLOCK_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = shack-hartmann-weighted-measurement-block-f32
            config = {
                image_size = [2, 3]
                row_block_rows = 1
                subaperture_size = [2, 3]
                subaperture_count = 1
                initial_subaperture_origins = [ [0 0] ]
                coordinate_scale = 1.0
                pixel_threshold = 2.0
                flux_threshold = 4.0
                reference_slopes = [ 0.0 0.0 ]
                active = [ true ]
                initial_weights = [ [0.0 1.0 2.0 3.0 0.5 1.0] ]
                measurements_schema = org.example.ao.weighted-shack-measurements/1
                column_indices_schema = org.example.ao.reconstructor-columns/1
                measurement_count_schema = org.example.ao.measurement-count/1
            }
        }
    ]
    links = []
    inputs = [ "measure:row-block" "measure:weights" ]
    outputs = [ "measure:measurements" "measure:columns" "measure:count" ]
}
"""

@testset "weighted Shack-Hartmann row blocks prepare and publish as a Graph Node" begin
    graph = prepare_graph_text(
        SHACK_HARTMANN_WEIGHTED_BLOCK_GRAPH_TEXT;
        algorithms=(ShackHartmannWeightedMeasurementBlockF32,),
    )
    @test graph.parameter_formats.weights.schema == SHACK_HARTMANN_WEIGHTS_V1
    measurements = fill(-1.0f0, 2)
    columns = fill(typemax(UInt32), 2)
    count = fill(typemax(UInt32), 1)
    outputs = (; measurements, columns, count)
    image = Float32[1 2 3; 7 8 9]
    input_names = (Symbol("row-block"),)
    for row in 1:2
        sample = SampleMetadata(1, row - 1; terminal=row == 2)
        inputs = NamedTuple{input_names}((@view(image[row:row, :]),))
        metadata = NamedTuple{input_names}((sample,))
        @test process!(outputs, graph, inputs, metadata) == (
            measurements=true,
            columns=true,
            count=true,
        )
        @test count[1] == UInt32(row == 2 ? 2 : 0)
    end
    @test count == UInt32[2]
    @test columns == UInt32[0, 1]
    @test measurements ≈ Float32[-1 / 7, 13 / 42] atol=1.0f-6
end

@testset "weighted Shack-Hartmann graph SIMD width override" begin
    for width in (0, 4, 8, 16)
        text = replace(SHACK_HARTMANN_WEIGHTED_CENTROID_GRAPH_TEXT,
            "initial_weights =" => "simd_width = $width\n                initial_weights =")
        graph = prepare_graph_text(text; algorithms=(ShackHartmannWeightedCentroidF32,))
        @test only(graph.nodes).prepared.plan.simd_width == (width == 0 ? 4 : width)
    end
    for value in ("3", "-1", "32", "invalid")
        text = replace(SHACK_HARTMANN_WEIGHTED_CENTROID_GRAPH_TEXT,
            "initial_weights =" => "simd_width = $value\n                initial_weights =")
        @test_throws GraphConfigurationError prepare_graph_text(text;
            algorithms=(ShackHartmannWeightedCentroidF32,))
    end
end
