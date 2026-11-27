# Generated snapshot from examples/modal-biquad-controller/run.jl.
# Regenerate with julia test/update_snapshots.jl examples/modal-biquad-controller/run.jl.
module ModalBiquadControllerExample

import FilterGraphAlgorithms
import FilterGraphs

const MODES = 3
const SECTIONS = 2
const OUTPUT_NAMES = (:output,)

"""
    example_coefficients()

Build distinct stable two-section filters for each of three modes.
"""
function example_coefficients()
    coefficients = Array{Float32}(undef, 5, SECTIONS, MODES)
    for mode in 1:MODES
        scale = Float32(1 / mode)
        coefficients[:, 1, mode] .= (0.7f0 * scale, 0.1f0 * scale, 0.05f0 * scale, -0.2f0, 0.1f0)
        coefficients[:, 2, mode] .= (1.0f0, -0.15f0, 0.04f0, 0.25f0, 0.12f0)
    end
    return coefficients
end

"""
    run_example()

Prepare the modal filter bank and return three controller commands.
"""
function run_example()
    graph = FilterGraphs.prepare_graph(
        joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms(),
    )
    coefficients = example_coefficients()
    FilterGraphs.replace_parameters!(
        graph,
        :coefficients => coefficients,
    )

    output = zeros(Float32, MODES)
    outputs = NamedTuple{OUTPUT_NAMES}((output,))
    samples = (
        Float32[1.0, -0.5, 0.25],
        Float32[0.5, 0.25, -0.75],
        Float32[-0.25, 1.0, 0.5],
    )
    filtered_commands = Vector{Vector{Float32}}(undef, length(samples))
    for index in eachindex(samples)
        availability = FilterGraphs.process!(
            outputs,
            graph,
            (input=samples[index],),
        )
        all(values(availability)) || error("Graph did not publish sample $index")
        filtered_commands[index] = copy(output)
    end

    return filtered_commands
end

function (@main)(args)
    println("Filtered modal controller commands:")
    foreach(println, run_example())
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .ModalBiquadControllerExample: main
end
