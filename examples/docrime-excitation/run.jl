module DocrimeExcitationExample

import FilterGraphAlgorithms
import FilterGraphs

"""
    run_example()

Prepare a source-only Graph and return three deterministic samples.
"""
function run_example()
    graph = FilterGraphs.prepare_graph(
        joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms(),
    )
    excitation = zeros(Float32, 3)
    outputs = (excitation=excitation,)
    samples = Vector{Vector{Float32}}(undef, 3)

    for sample_index in eachindex(samples)
        availability = FilterGraphs.process!(
            outputs,
            graph,
            NamedTuple(),
        )
        availability == (excitation=true,) || error(
            "DO-CRIME excitation was not available for sample $sample_index",
        )
        samples[sample_index] = copy(excitation)
    end
    return samples
end

function (@main)(args)
    samples = run_example()
    println("Three deterministic DO-CRIME excitation samples:")
    for sample in samples
        println(sample)
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .DocrimeExcitationExample: main
end
