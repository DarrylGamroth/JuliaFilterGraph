module DocrimeExcitationGraphTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

import FilterGraphAlgorithms: process!, reset!

const SOURCE_FIXTURE = joinpath(@__DIR__, "data", "examples", "docrime-excitation/graph.conf")
const SOURCE_SEED = typemax(UInt64)

@testset "deterministic DO-CRIME source runs as a zero-input Graph" begin
    graph = prepare_graph(
        SOURCE_FIXTURE;
        algorithms=(DocrimeExcitationF32,),
    )
    @test isempty(keys(graph.input_formats))
    @test keys(graph.output_formats) == (:excitation,)
    @test docrime_excitation_seed(only(graph.nodes).prepared.plan) ==
          SOURCE_SEED

    inputs = NamedTuple()
    output = zeros(Float32, 3)
    outputs = (excitation=output,)
    workspace = only(graph.nodes).prepared.workspace

    expected_prepared = prepare_algorithm(
        DocrimeExcitationF32,
        DocrimeExcitationF32Configuration(
            amplitudes=Float32[1, 2, 0.5],
            seed=SOURCE_SEED,
        ),
    )
    expected = similar(output)
    process!(expected, expected_prepared)
    first_expected = copy(expected)

    # Candidate ownership belongs to the Algorithm workspace, not the Graph.
    @test FilterGraphAlgorithms._process_admitted!(output, only(graph.nodes).prepared) ===
          nothing
    @test docrime_excitation_state(workspace) == SOURCE_SEED
    @test FilterGraphAlgorithms.discard!(workspace) === workspace
    @test docrime_excitation_state(workspace) == SOURCE_SEED
    @test FilterGraphAlgorithms._process_admitted!(output, only(graph.nodes).prepared) ===
          nothing
    @test FilterGraphAlgorithms.commit!(workspace) === workspace
    @test docrime_excitation_state(workspace) ==
          FilterGraphAlgorithms.splitmix64_advance(SOURCE_SEED, UInt64(3))
    reset!(graph)

    candidate = process!(outputs, graph, inputs)
    @test candidate == (excitation=true,)
    @test output == expected
    @test docrime_excitation_state(workspace) ==
          FilterGraphAlgorithms.splitmix64_advance(SOURCE_SEED, UInt64(3))

    process!(expected, expected_prepared)
    @test process!(outputs, graph, inputs) == (excitation=true,)
    @test output == expected
    @test FilterGraphs.output_metadata(graph) == (excitation=nothing,)

    reset!(graph)
    @test docrime_excitation_state(workspace) == SOURCE_SEED
    @test process!(outputs, graph, inputs) == (excitation=true,)
    @test output == first_expected
    @test_throws ArgumentError process!(
        outputs,
        graph,
        (invented=Float32[1],),
    )

    process!(outputs, graph, inputs)
    @test @allocated(process!(outputs, graph, inputs)) == 0
end

end
