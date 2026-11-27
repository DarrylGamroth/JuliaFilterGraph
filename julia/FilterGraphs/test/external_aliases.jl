module ExternalAliasTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

import FilterGraphAlgorithms: process!

const NODES = """
nodes = [
    {
        type = ndarray name = first label = dense-vector-reconstructor-f32
        config = {
            input_count = 2 output_count = 1 initial_reconstructor = [ 1.0 2.0 ]
            input_schema = test.ordered-input/1 output_schema = test.output/1
            reconstructor_schema = test.matrix/1
        }
    }
    {
        type = ndarray name = second label = dense-vector-reconstructor-f32
        config = {
            input_count = 2 output_count = 1 initial_reconstructor = [ 3.0 4.0 ]
            input_schema = test.ordered-input/1 output_schema = test.output/1
            reconstructor_schema = test.matrix/1
        }
    }
]
"""

const INPUTS = """
[
    { name = first-input endpoint = "first:ordered-input" }
    { name = second-input endpoint = "second:ordered-input" }
    { name = first-matrix endpoint = "first:reconstructor" }
    { name = second-matrix endpoint = "second:reconstructor" }
]
"""

const OUTPUTS = """
[
    { name = first-output endpoint = "first:output" }
    { name = second-output endpoint = "second:output" }
]
"""

graph_text(; inputs=INPUTS, outputs=OUTPUTS, links="[]") =
    "filter.graph = { $NODES links = $links inputs = $inputs outputs = $outputs }"

function prepare(text)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        prepare_graph(path; algorithms=(DenseVectorReconstructorF32,))
    end
end

@testset "external aliases preserve Port contracts and parameter destinations" begin
    graph = prepare(graph_text())
    @test keys(graph.input_formats) == (Symbol("first-input"), Symbol("second-input"))
    @test keys(graph.parameter_formats) == (Symbol("first-matrix"), Symbol("second-matrix"))
    @test keys(graph.output_formats) == (Symbol("first-output"), Symbol("second-output"))
    @test all(format -> format.shape == (2,) && format.schema == "test.ordered-input/1",
        values(graph.input_formats))
    @test all(format -> format.shape == (1, 2) && format.schema == "test.matrix/1",
        values(graph.parameter_formats))
    @test all(format -> format.shape == (1,) && format.schema == "test.output/1",
        values(graph.output_formats))

    inputs = (var"second-input"=Float32[5, 6], var"first-input"=Float32[1, 2])
    outputs = (var"second-output"=zeros(Float32, 1), var"first-output"=zeros(Float32, 1))
    metadata = (var"second-input"=SampleMetadata(22), var"first-input"=SampleMetadata(11))
    @test process!(outputs, graph, inputs, metadata) == (
        var"first-output"=true, var"second-output"=true,
    )
    @test outputs.var"first-output" == Float32[5]
    @test outputs.var"second-output" == Float32[39]
    @test FilterGraphs.output_metadata(graph) == (
        var"first-output"=metadata.var"first-input",
        var"second-output"=metadata.var"second-input",
    )

    @test replace_parameters!(graph, "first-matrix" => Float32[10 20]) === graph
    @test process!(outputs, graph, inputs) == (var"first-output"=true, var"second-output"=true)
    @test outputs.var"first-output" == Float32[50]
    @test outputs.var"second-output" == Float32[39]
    @test replace_parameters!(graph,
        Symbol("first-matrix") => Float32[-1 -2],
        "second-matrix" => Float32[30 40]) === graph
    @test process!(outputs, graph, inputs) == (var"first-output"=true, var"second-output"=true)
    @test outputs.var"first-output" == Float32[-5]
    @test outputs.var"second-output" == Float32[390]
    @test_throws ArgumentError replace_parameters!(graph, :reconstructor => Float32[1 2])
    @test_throws DimensionMismatch process!(outputs, graph,
        (var"first-input"=zeros(Float32, 3), var"second-input"=inputs.var"second-input"))
end

@testset "external alias validation rejects malformed or ambiguous declarations" begin
    for bad in (
        "{ name = same endpoint = \"first:ordered-input\" unknown = true }",
        "{ endpoint = \"first:ordered-input\" }",
        "{ name = same }",
        "{ name = \"\" endpoint = \"first:ordered-input\" }",
        "{ name = \"  \" endpoint = \"first:ordered-input\" }",
        "{ name = same endpoint = \"\" }",
        "{ name = same endpoint = \"  \" }",
        "{ name = 1 endpoint = \"first:ordered-input\" }",
        "{ name = same endpoint = false }",
        "{ name = same endpoint = {} }",
        "false",
    )
        @test_throws GraphConfigurationError prepare(graph_text(inputs="[ $bad ]"))
        @test_throws GraphConfigurationError prepare(graph_text(outputs="[ $bad ]"))
    end
    @test_throws GraphConfigurationError prepare(graph_text(inputs=replace(
        INPUTS, "name = second-input" => "name = first-input",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(inputs=replace(
        INPUTS, "name = second-matrix" => "name = first-matrix",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(inputs=replace(
        INPUTS, "name = first-matrix" => "name = first-input",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(outputs=replace(
        OUTPUTS, "name = second-output" => "name = first-output",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(outputs=replace(
        OUTPUTS, "second:output" => "missing:output",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(outputs=replace(
        OUTPUTS, "second:output" => "second:missing",
    )))
    @test_throws GraphConfigurationError prepare(graph_text(links="""
        [ { output = { name = alias endpoint = "first:output" } input = "second:ordered-input" } ]
        """))
    @test_throws GraphConfigurationError prepare(graph_text(links="""
        [ { output = "first:output" input = { name = alias endpoint = "second:ordered-input" } } ]
        """))
end

@testset "string endpoints retain Algorithm names alongside aliases" begin
    graph = prepare(graph_text(
        inputs="""[
            "first:ordered-input" "first:reconstructor"
            { name = second-input endpoint = "second:ordered-input" }
            { name = second-matrix endpoint = "second:reconstructor" }
        ]""",
        outputs="""[ "first:output" { name = second-output endpoint = "second:output" } ]""",
    ))
    @test keys(graph.input_formats) == (Symbol("ordered-input"), Symbol("second-input"))
    @test keys(graph.parameter_formats) == (:reconstructor, Symbol("second-matrix"))
    @test keys(graph.output_formats) == (:output, Symbol("second-output"))
    outputs = (output=zeros(Float32, 1), var"second-output"=zeros(Float32, 1))
    @test process!(outputs, graph,
        (var"ordered-input"=Float32[1, 2], var"second-input"=Float32[5, 6])) ==
        (output=true, var"second-output"=true)
    @test outputs.output == Float32[5]
    @test outputs.var"second-output" == Float32[39]
end

@testset "one external binding per parameter destination" begin
    repeated = replace(INPUTS, "second:reconstructor" => "first:reconstructor")
    text = "filter.graph = { " * NODES * " inputs = " * repeated * " outputs = " * OUTPUTS * " }"
    mktemp() do path,io
        write(io,text);close(io)
        @test_throws FilterGraphs.GraphConfigurationError prepare_graph(path;algorithms=algorithms())
    end
end

@testset "independently wrapped external outputs use byte-range alias checks" begin
    storage = zeros(Float32, 4)
    GC.@preserve storage begin
        left = unsafe_wrap(Array, pointer(storage), 2; own=false)
        overlap = unsafe_wrap(Array, pointer(storage, 2), 2; own=false)
        right = unsafe_wrap(Array, pointer(storage, 3), 2; own=false)
        @test_throws ArgumentError FilterGraphs._validate_external_output_aliases!(
            (:left, :overlap), (left, overlap),
        )
        @test isnothing(FilterGraphs._validate_external_output_aliases!(
            (:left, :right), (left, right),
        ))
        memory = unsafe_wrap(Memory{Float32}, pointer(storage), 2; own=false)
        @test_throws ArgumentError FilterGraphs._validate_external_output_aliases!(
            (:left, :overlap), (memory, overlap),
        )
        @test isnothing(FilterGraphs._validate_external_output_aliases!(
            (:left, :right), (memory, right),
        ))
    end
end

@testset "shared-parent wrappers use their storage bounds" begin
    source = view(zeros(Float32, 3, 2), 1:2, :)
    permuted = PermutedDimsArray(source, (2, 1))
    for wrapper in (reshape(source, :), reinterpret(UInt32, source))
        @test_throws ArgumentError FilterGraphs._validate_external_output_aliases!(
            (:permuted, :wrapper), (permuted, wrapper),
        )
    end
end

end
