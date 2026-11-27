module OpticalGainWorkflowExample

import FilterGraphAlgorithms
import FilterGraphs

const INPUT_NAMES = (
    Symbol("reconstructed-response-modes"),
    :reconstructed,
)
const OUTPUT_NAMES = (
    Symbol("commanded-modes"),
    Symbol("estimate-complete"),
    Symbol("optical-gains"),
    Symbol("accepted-dither-modes"),
    Symbol("updated-reconstructed-modes"),
    :corrected,
)

function workflow_inputs(response, reconstructed)
    return NamedTuple{INPUT_NAMES}((Float32[response], Float32[reconstructed]))
end

function workflow_outputs()
    return NamedTuple{OUTPUT_NAMES}(
        (
            zeros(Float32, 1),
            falses(1),
            zeros(Float32, 1),
            falses(1),
            falses(1),
            zeros(Float32, 1),
        ),
    )
end

"""
    run_example()

Run one dither-estimate-update window, then adopt its successfully published
physical gain for reciprocal correction on the following sample.
"""
function run_example()
    graph = FilterGraphs.prepare_graph(
        joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms(),
    )
    outputs = workflow_outputs()
    responses = Float32[0.25, 0, -0.25, 0]
    corrected_before_adoption = Float32[]
    for response in responses
        availability = FilterGraphs.process!(
            outputs,
            graph,
            workflow_inputs(response, 0.75f0),
        )
        all(values(availability)) || error(
            "the optical-gain workflow did not publish every output",
        )
        push!(corrected_before_adoption, only(outputs.corrected))
    end
    only(getproperty(outputs, Symbol("estimate-complete"))) || error(
        "the optical-gain estimate window did not complete",
    )
    only(getproperty(outputs, Symbol("accepted-dither-modes"))) || error(
        "the completed optical-gain estimate was not accepted",
    )
    only(getproperty(outputs, Symbol("updated-reconstructed-modes"))) || error(
        "the reconstructed-mode physical gain was not updated",
    )

    published_gain = copy(getproperty(outputs, Symbol("optical-gains")))
    FilterGraphs.replace_parameters!(
        graph,
        Symbol("optical-gains") => published_gain,
    )
    availability = FilterGraphs.process!(
        outputs,
        graph,
        workflow_inputs(0.25f0, 0.75f0),
    )
    all(values(availability)) || error(
        "the optical-gain workflow did not publish the corrected sample",
    )
    return (
        published_gain=only(published_gain),
        corrected_before_adoption=corrected_before_adoption,
        corrected_after_adoption=only(outputs.corrected),
    )
end

function (@main)(args)
    result = run_example()
    println("Published physical optical gain: ", result.published_gain)
    println(
        "Correction before adoption: ",
        last(result.corrected_before_adoption),
    )
    println("Correction after adoption: ", result.corrected_after_adoption)
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .OpticalGainWorkflowExample: main
end
