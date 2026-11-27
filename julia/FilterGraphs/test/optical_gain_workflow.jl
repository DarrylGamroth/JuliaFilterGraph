const OPTICAL_GAIN_WORKFLOW_CONFIG = joinpath(@__DIR__, "data", "examples", "optical-gain-workflow/graph.conf")

const OPTICAL_GAIN_WORKFLOW_INPUT_NAMES = (
    Symbol("reconstructed-response-modes"),
    :reconstructed,
)
const OPTICAL_GAIN_WORKFLOW_OUTPUT_NAMES = (
    Symbol("commanded-modes"),
    Symbol("estimate-complete"),
    Symbol("optical-gains"),
    Symbol("accepted-dither-modes"),
    Symbol("updated-reconstructed-modes"),
    :corrected,
)

function optical_gain_workflow_inputs(response; reconstructed=0.75f0)
    return NamedTuple{OPTICAL_GAIN_WORKFLOW_INPUT_NAMES}(
        (Float32[response], Float32[reconstructed]),
    )
end

function optical_gain_workflow_outputs()
    return NamedTuple{OPTICAL_GAIN_WORKFLOW_OUTPUT_NAMES}(
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

function optical_gain_workflow_workspaces(graph)
    dither = only(node for node in graph.nodes if node.name === :dither)
    estimate = only(node for node in graph.nodes if node.name === :estimate)
    update = only(node for node in graph.nodes if node.name === :update)
    return (
        dither.prepared.workspace,
        estimate.prepared.workspace,
        update.prepared.workspace,
    )
end

optical_gain_workflow_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

@testset "optical-gain calibration dogfoods one composed Graph" begin
    graph = prepare_graph(
        OPTICAL_GAIN_WORKFLOW_CONFIG;
        algorithms=(
            OpticalGainDitherF32,
            OpticalGainEstimatorF32,
            OpticalGainUpdateF32,
            OpticalGainCorrectionF32,
        ),
    )
    @test keys(graph.input_formats) == OPTICAL_GAIN_WORKFLOW_INPUT_NAMES
    @test keys(graph.parameter_formats) == (
        Symbol("calibration-responses"),
        Symbol("optical-gains"),
    )
    @test keys(graph.output_formats) == OPTICAL_GAIN_WORKFLOW_OUTPUT_NAMES
    @test graph.node_failure_types == (
        Union{},
        OpticalGainEstimatorFailure,
        OpticalGainUpdateFailure,
        OpticalGainCorrectionFailure,
    )
    dither, estimate, update = optical_gain_workflow_workspaces(graph)
    outputs = optical_gain_workflow_outputs()
    availability = NamedTuple{OPTICAL_GAIN_WORKFLOW_OUTPUT_NAMES}(
        (true, true, true, true, true, true),
    )
    responses = Float32[0.25, 0, -0.25, 0]
    for frame in 0:3
        metadata = NamedTuple{OPTICAL_GAIN_WORKFLOW_INPUT_NAMES}(
            (
                SampleMetadata(100 + frame, 0; terminal=true),
                SampleMetadata(200 + frame, 0; terminal=true),
            ),
        )
        @test process!(
            outputs,
            graph,
            optical_gain_workflow_inputs(responses[frame + 1]),
            metadata,
        ) == availability
        @test only(getproperty(outputs, Symbol("estimate-complete"))) == (frame == 3)
        @test only(getproperty(outputs, Symbol("optical-gains"))) ==
              (frame == 3 ? 0.75f0 : 1.0f0)
        @test only(outputs.corrected) == 0.75f0
    end
    @test only(getproperty(outputs, Symbol("accepted-dither-modes")))
    @test only(getproperty(outputs, Symbol("updated-reconstructed-modes")))
    @test optical_gain_dither_phases(dither) == [0]
    @test optical_gain_estimator_frame(estimate) == 0
    @test optical_gains(update) == Float32[0.75]
    published = FilterGraphs.output_metadata(graph)
    @test published[1] === nothing
    @test published[2] == SampleMetadata(103, 0; terminal=true)
    @test all(
        value -> value == SampleMetadata(103, 0; terminal=true),
        values(published)[3:5],
    )
    @test published.corrected == SampleMetadata(203, 0; terminal=true)

    adopted = copy(getproperty(outputs, Symbol("optical-gains")))
    @test replace_parameters!(graph, Symbol("optical-gains") => adopted) === graph
    adopted[1] = 9
    @test process!(outputs, graph, optical_gain_workflow_inputs(0.25f0)) isa NamedTuple
    @test only(outputs.corrected) == 1.0f0
end

@testset "successful workflow Nodes commit before a later rejection" begin
    graph = prepare_graph(
        OPTICAL_GAIN_WORKFLOW_CONFIG;
        algorithms=(
            OpticalGainDitherF32,
            OpticalGainEstimatorF32,
            OpticalGainUpdateF32,
            OpticalGainCorrectionF32,
        ),
    )
    outputs = optical_gain_workflow_outputs()
    dither, estimate, update = optical_gain_workflow_workspaces(graph)
    for response in Float32[0.25, 0, -0.25]
        process!(outputs, graph, optical_gain_workflow_inputs(response))
    end
    @test optical_gain_dither_phases(dither) == [3]
    @test optical_gain_estimator_frame(estimate) == 3
    @test optical_gains(update) == Float32[1]

    process!(outputs, graph, optical_gain_workflow_inputs(0.0f0))
    @test optical_gain_dither_phases(dither) == [0]
    @test optical_gain_estimator_frame(estimate) == 0
    @test optical_gains(update) == Float32[0.75]

    retained = map(copy, outputs)
    @test process!(outputs, graph, optical_gain_workflow_inputs(NaN32)) ===
          OpticalGainEstimatorNonFiniteResponse
    @test outputs == retained
    @test optical_gain_dither_phases(dither) == [1]
    @test optical_gain_estimator_frame(estimate) == 0
    @test optical_gains(update) == Float32[0.75]
end

@testset "composed optical-gain Graph allocates no heap after warmup" begin
    graph = prepare_graph(
        OPTICAL_GAIN_WORKFLOW_CONFIG;
        algorithms=(
            OpticalGainDitherF32,
            OpticalGainEstimatorF32,
            OpticalGainUpdateF32,
            OpticalGainCorrectionF32,
        ),
    )
    outputs = optical_gain_workflow_outputs()
    process!(outputs, graph, optical_gain_workflow_inputs(0.25f0))
    nonterminal = optical_gain_workflow_inputs(0.0f0)
    @test optical_gain_workflow_allocation_bytes(outputs, graph, nonterminal) == 0
    process!(outputs, graph, optical_gain_workflow_inputs(-0.25f0))
    terminal = optical_gain_workflow_inputs(0.0f0)
    @test optical_gain_workflow_allocation_bytes(outputs, graph, terminal) == 0

    rejected = optical_gain_workflow_inputs(NaN32)
    process!(outputs, graph, rejected)
    @test optical_gain_workflow_allocation_bytes(outputs, graph, rejected) == 0

end
