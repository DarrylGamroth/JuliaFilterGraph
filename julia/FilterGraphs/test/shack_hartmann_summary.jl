module ShackHartmannSummaryGraphTests

using FilterGraphAlgorithms
using FilterGraphs
using Test
import FilterGraphAlgorithms: process!

function prepare_graph_text(text; algorithms)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        return prepare_graph(path; algorithms)
    end
end

const SUMMARY_GRAPH = """
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
            name = summarize
            label = shack-hartmann-summary-f32
            config = { subaperture_count = 2 }
        }
    ]
    links = [
        { output = "measure:slopes" input = "summarize:slopes" }
        { output = "measure:flux" input = "summarize:flux" }
        { output = "measure:validity" input = "summarize:validity" }
    ]
    inputs = [ "measure:image" ]
    outputs = [ "summarize:summary" "measure:slopes" "measure:flux" "measure:validity" ]
}
"""

function replay_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "complete-frame measurement and summary compile into an ordinary CPU Graph" begin
    graph = prepare_graph_text(SUMMARY_GRAPH; algorithms=algorithms())
    @test keys(graph.input_formats) == (:image,)
    @test graph.output_formats.summary.schema == SHACK_HARTMANN_SUMMARY_V1
    @test graph.output_formats.summary.shape == (8,)
    image = Float32[1 3; 3 1]
    outputs = (summary=zeros(Float32, 8), slopes=zeros(Float32, 2, 2),
        flux=zeros(Float32, 2), validity=falses(2))
    inputs = (image=image,)
    metadata = SampleMetadata(81, 0; timestamp_nanoseconds=5678, terminal=true)
    @test process!(outputs, graph, inputs, (image=metadata,)) ==
        (summary=true, slopes=true, flux=true, validity=true)
    @test outputs.slopes == Float32[0.25 0; -0.25 0]
    @test outputs.flux == Float32[4, 4]
    @test all(outputs.validity)
    @test outputs.summary == Float32[2, 2, 4, 0.25, 0, 0.25, 0, 0]
    @test all(==(metadata), values(FilterGraphs.output_metadata(graph)))
    @test image == Float32[1 3; 3 1]
    @test replay_allocation_bytes(outputs, graph, inputs) == 0

    # A later frame with no valid flux publishes NaN diagnostics successfully.
    fill!(image, 0)
    next_metadata = SampleMetadata(82, 0; timestamp_nanoseconds=9999, terminal=true)
    @test process!(outputs, graph, inputs, (image=next_metadata,)) ==
        (summary=true, slopes=true, flux=true, validity=true)
    @test outputs.summary[1:2] == Float32[2, 0]
    @test all(isnan, outputs.summary[3:8])
    @test all(iszero, outputs.slopes)
    @test !any(outputs.validity)
    @test all(==(next_metadata), values(FilterGraphs.output_metadata(graph)))
    @test replay_allocation_bytes(outputs, graph, inputs) == 0
end

@testset "nonfinite valid diagnostics do not fault a Graph" begin
    text = """
    filter.graph = {
        nodes = [ { type = ndarray name = summarize label = shack-hartmann-summary-f32
            config = { subaperture_count = 2 } } ]
        links = []
        inputs = [ "summarize:slopes" "summarize:flux" "summarize:validity" ]
        outputs = [ "summarize:summary" ]
    }
    """
    graph = prepare_graph_text(text; algorithms=algorithms())
    inputs = (slopes=Float32[NaN 4; 3 4], flux=Float32[1, 2], validity=trues(2))
    outputs = (summary=zeros(Float32, 8),)
    @test process!(outputs, graph, inputs) == (summary=true,)
    @test outputs.summary[1:2] == Float32[2, 2]
    @test all(isnan, outputs.summary[3:8])
    @test replay_allocation_bytes(outputs, graph, inputs) == 0
    inputs.slopes[1, 1] = 3
    @test process!(outputs, graph, inputs) == (summary=true,)
    @test outputs.summary == Float32[2, 2, 1.5, 3, 4, 5, 3, 4]
end

end
