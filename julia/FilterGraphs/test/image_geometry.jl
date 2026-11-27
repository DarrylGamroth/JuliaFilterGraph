const PIXEL_SELECTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = select
            label = pixel-selection-f32
            config = {
                image_size = [2, 3]
                mask = [ true false true false true false ]
                input_schema = pipewireao/calibrated-pixels-f32/1
                output_schema = org.calculon.test.selected-pixels/1
            }
        }
    ]
    links = []
    inputs = [ "select:image" "select:mask" ]
    outputs = [ "select:selected-pixels" ]
}
"""

const REGION_EXTRACTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = extract
            label = region-extraction-f32
            config = {
                image_size = [4, 5]
                region_size = [2, 2]
                origins = [ [0 0] [2 3] ]
            }
        }
    ]
    links = []
    inputs = [ "extract:calibrated" "extract:origins" ]
    outputs = [ "extract:regions" ]
}
"""

function geometry_row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function geometry_region_major_values(regions)
    values = eltype(regions)[]
    for region in axes(regions, 1), row in axes(regions, 2), column in axes(regions, 3)
        push!(values, regions[region, row, column])
    end
    return values
end

geometry_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

@testset "image size arrays preserve row and column order" begin
    graph = prepare_graph_text(PIXEL_SELECTION_GRAPH_TEXT; algorithms=(PixelSelectionF32,))
    @test graph.input_formats.image.shape === (2, 3)
    @test only(graph.nodes).prepared.plan.image_size === (2, 3)
    for size_text in ("[]", "[2]", "[2 3 4]", "[2.5 3]", "[0 3]", "[-2 3]")
        @test_throws GraphConfigurationError prepare_graph_text(
            replace(PIXEL_SELECTION_GRAPH_TEXT, "image_size = [2, 3]" => "image_size = $size_text");
            algorithms=(PixelSelectionF32,),
        )
    end
    @test_throws GraphConfigurationError prepare_graph_text(
        replace(PIXEL_SELECTION_GRAPH_TEXT, "image_size = [2, 3]" => "image_rows = 2 image_columns = 3");
        algorithms=(PixelSelectionF32,),
    )
end

@testset "pixel-selection declaration prepares as a Graph Node" begin
    graph = prepare_graph_text(
        PIXEL_SELECTION_GRAPH_TEXT;
        algorithms=(PixelSelectionF32,),
    )
    image = geometry_row_major_matrix(Float32, 0:5, 2, 3)
    selected = fill(-1.0f0, 3)
    inputs = (image=image,)
    outputs = NamedTuple{(Symbol("selected-pixels"),)}((selected,))

    @test keys(graph.input_formats) == (:image,)
    @test keys(graph.parameter_formats) == (:mask,)
    @test graph.parameter_formats.mask.schema == PIXEL_SELECTION_MASK_V1
    @test process!(outputs, graph, inputs) ==
          NamedTuple{keys(outputs)}((true,))
    @test selected == Float32[0, 2, 4]
    @test geometry_graph_allocation_bytes(outputs, graph, inputs) == 0
end

@testset "region-extraction declaration prepares as a Graph Node" begin
    graph = prepare_graph_text(
        REGION_EXTRACTION_GRAPH_TEXT;
        algorithms=(RegionExtractionF32,),
    )
    image = geometry_row_major_matrix(Float32, 0:19, 4, 5)
    regions = fill(-1.0f0, 2, 2, 2)
    inputs = (calibrated=image,)
    outputs = (regions=regions,)

    @test keys(graph.input_formats) == (:calibrated,)
    @test keys(graph.parameter_formats) == (:origins,)
    @test graph.parameter_formats.origins.schema == REGION_ORIGINS_V1
    @test process!(outputs, graph, inputs) == (regions=true,)
    @test geometry_region_major_values(regions) ==
          Float32[0, 1, 5, 6, 13, 14, 18, 19]
    @test geometry_graph_allocation_bytes(outputs, graph, inputs) == 0
end
