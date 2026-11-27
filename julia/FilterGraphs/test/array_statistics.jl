module ArrayStatisticsGraphTests

using FilterGraphAlgorithms
using FilterGraphs
using Test
import FilterGraphAlgorithms: process!, reset!

const GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = statistics
            label = array-statistics-f32
            config = {
                size = [2 3]
                mode = finite
                window_samples = 2
                input_schema = "example/sample-f32/1"
                mean_schema = "example/mean-f32/1"
                variance_schema = "example/variance-f32/1"
                stddev_schema = "example/stddev-f32/1"
            }
        }
    ]
    links = []
    inputs = [ "statistics:sample" ]
    outputs = [
        "statistics:mean"
        "statistics:variance"
        "statistics:stddev"
        "statistics:sample-count"
        "statistics:complete"
        "statistics:variance-valid"
    ]
}
"""

const OUTPUT_NAMES = (:mean, :variance, :stddev, Symbol("sample-count"), :complete,
    Symbol("variance-valid"))
allocation_bytes(out, graph, input) = @allocated process!(out, graph, input)

@testset "complete-array statistics prepares and publishes as a headless Graph" begin
    graph = mktemp() do path, io
        write(io, GRAPH)
        flush(io)
        prepare_graph(path; algorithms=algorithms())
    end
    @test keys(graph.input_formats) == (:sample,)
    @test keys(graph.output_formats) == OUTPUT_NAMES
    @test map(f -> f.shape, values(graph.output_formats)) ==
        ((2, 3), (2, 3), (2, 3), (1,), (1,), (1,))
    @test graph.output_formats.variance.schema == "example/variance-f32/1"
    @test graph.output_formats.mean.schema != graph.output_formats.variance.schema
    out = NamedTuple{OUTPUT_NAMES}((fill(71f0, 2, 3), fill(72f0, 2, 3),
        fill(73f0, 2, 3), UInt64[74], Bool[false], Bool[false]))
    input = (sample=fill(2f0, 2, 3),)
    available = NamedTuple{OUTPUT_NAMES}((true, true, true, true, true, true))
    metadata = (sample=SampleMetadata(41, 0; timestamp_nanoseconds=1234, terminal=true),)
    @test process!(out, graph, input, metadata) == available
    @test out.mean == input.sample
    @test out.variance == zeros(Float32, 2, 3)
    @test out[4] == UInt64[1]
    @test out.complete == [false]
    @test out[6] == [false]
    @test all(m -> m == metadata.sample, values(FilterGraphs.output_metadata(graph)))

    # Rejected Algorithm data does not publish a partial Graph result or advance history.
    before = deepcopy(out)
    input.sample[end] = NaN32
    @test process!(out, graph, input) === ArrayStatisticsNonFiniteInput
    @test out == before
    @test first(graph.nodes).prepared.workspace.committed.sample_count == 1
    input.sample .= 4
    @test process!(out, graph, input) == available
    @test out.mean == fill(3f0, 2, 3)
    @test out.variance == fill(2f0, 2, 3)
    @test out.stddev ≈ fill(sqrt(2f0), 2, 3)
    @test out[4] == UInt64[2]
    @test out.complete == [true]
    @test out[6] == [true]
    input.sample .= 99
    @test process!(out, graph, input) == available
    @test out.mean == fill(3f0, 2, 3)
    allocation_bytes(out, graph, input)
    @test allocation_bytes(out, graph, input) == 0
    reset!(graph)
    @test process!(out, graph, input) == available
    @test out.mean == input.sample
    @test out[4] == UInt64[1]
    @test out.complete == [false]
end

end
