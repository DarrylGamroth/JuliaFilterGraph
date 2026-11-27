# Generated snapshot from examples/leaky-controller/run.jl.
# Regenerate with julia test/update_snapshots.jl examples/leaky-controller/run.jl.
module LeakyControllerExample

import FilterGraphAlgorithms
import FilterGraphs

const OUTPUT_NAMES = (:demanded, Symbol("controller-constraint-feedback"))

"""
    run_example()

Process two samples through a leaky controller with projections and published
constraint feedback. The external owner receives the feedback; this integrator
does not adopt it in its recurrence.
"""
function run_example()
    graph = FilterGraphs.prepare_graph(
        joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms(),
    )
    FilterGraphs.replace_parameters!(
        graph,
        "controller-to-vdm" => Float32[
            1 0.5 -0.25
            0 1 0.5
        ],
        "active-to-full" => Float32[
            1 0
            0 1
            0.5 -1
        ],
        "vdm-to-pdm" => Float32[
            1 0 0.2
            0 1 -0.3
        ],
        "full-to-active" => Float32[
            1 0 0.25
            0 1 -0.5
        ],
        "pdm-to-vdm" => Float32[
            1 0
            0 1
            0.25 -0.5
        ],
        "vdm-to-controller" => Float32[
            1 0
            0 1
            0.5 -0.25
        ],
    )
    demanded = zeros(Float32, 2)
    controller_feedback = zeros(Float32, 3)
    outputs = NamedTuple{OUTPUT_NAMES}((demanded, controller_feedback))
    commands = Vector{Vector{Float32}}(undef, 2)
    feedback = Vector{Vector{Float32}}(undef, 2)

    residual_errors = (
        Float32[1, -2, 0.5],
        Float32[0.5, 0.25, -0.75],
    )
    for sample_index in eachindex(residual_errors)
        availability = FilterGraphs.process!(
            outputs,
            graph,
            (input=residual_errors[sample_index],),
        )
        all(values(availability)) || error(
            "control outputs were not available for sample $sample_index",
        )
        commands[sample_index] = copy(demanded)
        feedback[sample_index] = copy(controller_feedback)
    end
    return (demanded_commands=commands, controller_constraint_feedback=feedback)
end

function (@main)(args)
    result = run_example()
    println("Demanded PDM commands:")
    for command in result.demanded_commands
        println(command)
    end
    println("Controller constraint feedback for the host delay boundary:")
    for feedback in result.controller_constraint_feedback
        println(feedback)
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .LeakyControllerExample: main
end
