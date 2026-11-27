const OPTICAL_GAIN_ESTIMATOR_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = dither
            label = optical-gain-dither-f32
            config = {
                modes = [
                    {
                        amplitude = 0.5
                        frames_per_period = 4
                        periods_per_estimate = 1
                    }
                ]
            }
        }
        {
            type = ndarray
            name = estimate
            label = optical-gain-estimator-f32
            config = {
                modes = [
                    {
                        amplitude = 0.5
                        frames_per_period = 4
                        periods_per_estimate = 1
                    }
                ]
                initial_calibration_responses = [ 1.0 ]
            }
        }
    ]
    links = [
        {
            output = "dither:commanded-modes"
            input = "estimate:realized-command-modes"
        }
        {
            output = "dither:window-frame"
            input = "estimate:window-frame"
        }
        {
            output = "dither:phase-indices"
            input = "estimate:phase-indices"
        }
        {
            output = "dither:completes-window"
            input = "estimate:completes-window"
        }
    ]
    inputs = [
        "estimate:reconstructed-response-modes"
        "estimate:calibration-responses"
    ]
    outputs = [
        "estimate:command-amplitudes"
        "estimate:response-amplitudes"
        "estimate:optical-gains"
        "estimate:phase-lags"
        "estimate:valid"
        "estimate:estimate-complete"
    ]
}
"""

const OPTICAL_GAIN_ESTIMATOR_OUTPUT_NAMES = (
    Symbol("command-amplitudes"),
    Symbol("response-amplitudes"),
    Symbol("optical-gains"),
    Symbol("phase-lags"),
    :valid,
    Symbol("estimate-complete"),
)
const OPTICAL_GAIN_ESTIMATOR_INPUT_NAMES =
    (Symbol("reconstructed-response-modes"),)

function estimator_graph_outputs()
    return NamedTuple{OPTICAL_GAIN_ESTIMATOR_OUTPUT_NAMES}(
        (
            fill(NaN32, 1),
            fill(NaN32, 1),
            fill(NaN32, 1),
            fill(NaN32, 1),
            falses(1),
            trues(1),
        ),
    )
end

estimator_graph_inputs(response=0.0f0) =
    NamedTuple{OPTICAL_GAIN_ESTIMATOR_INPUT_NAMES}((Float32[response],))

estimator_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function optical_gain_node_workspaces(graph)
    dither = only(node for node in graph.nodes if node.name === :dither)
    estimate = only(node for node in graph.nodes if node.name === :estimate)
    return dither.prepared.workspace, estimate.prepared.workspace
end

@testset "dither and estimator compose as SPA-JSON Graph Nodes" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_ESTIMATOR_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, OpticalGainEstimatorF32),
    )
    @test keys(graph.input_formats) == OPTICAL_GAIN_ESTIMATOR_INPUT_NAMES
    @test keys(graph.output_formats) == OPTICAL_GAIN_ESTIMATOR_OUTPUT_NAMES
    @test keys(graph.parameter_formats) == (Symbol("calibration-responses"),)
    @test graph.input_formats[1].schema == OPTICAL_GAIN_DITHER_RESPONSE_V1
    @test getproperty.(values(graph.output_formats), :schema) == (
        OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        OPTICAL_GAIN_PHASE_LAG_V1,
        OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
    )
    @test graph.node_failure_types == (Union{}, OpticalGainEstimatorFailure)

    outputs = estimator_graph_outputs()
    availability = NamedTuple{OPTICAL_GAIN_ESTIMATOR_OUTPUT_NAMES}(
        (true, true, true, true, true, true),
    )
    responses = Float32[0.25, 0, -0.25, 0]
    for frame in 0:3
        inputs = estimator_graph_inputs(responses[frame + 1])
        metadata = NamedTuple{OPTICAL_GAIN_ESTIMATOR_INPUT_NAMES}(
            (SampleMetadata(100 + frame, 0; terminal=true),),
        )
        @test process!(outputs, graph, inputs, metadata) == availability
        @test getproperty(outputs, Symbol("estimate-complete")) == [frame == 3]
        if frame < 3
            @test getproperty(outputs, Symbol("optical-gains")) == Float32[0]
        end
        @test all(
            value -> value == only(metadata),
            values(FilterGraphs.output_metadata(graph)),
        )
    end
    @test getproperty(outputs, Symbol("command-amplitudes")) ≈ Float32[0.5] atol=2eps(Float32)
    @test getproperty(outputs, Symbol("response-amplitudes")) ≈ Float32[0.25] atol=2eps(Float32)
    @test getproperty(outputs, Symbol("optical-gains")) ≈ Float32[0.5] atol=2eps(Float32)
    @test getproperty(outputs, Symbol("phase-lags")) ≈ Float32[0] atol=2eps(Float32)
    @test outputs.valid == [true]
    dither, estimate = optical_gain_node_workspaces(graph)
    @test optical_gain_dither_phases(dither) == [0]
    @test optical_gain_estimator_frame(estimate) == 0
end

@testset "successful Nodes commit before a later estimator rejection" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_ESTIMATOR_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, OpticalGainEstimatorF32),
    )
    outputs = estimator_graph_outputs()
    inputs = estimator_graph_inputs(0.25f0)
    dither, estimate = optical_gain_node_workspaces(graph)

    @test process!(outputs, graph, inputs) isa NamedTuple
    @test optical_gain_dither_phases(dither) == [1]
    @test optical_gain_estimator_frame(estimate) == 1
    retained = map(copy, outputs)
    rejected = estimator_graph_inputs(NaN32)
    @test process!(outputs, graph, rejected) ===
          OpticalGainEstimatorNonFiniteResponse
    @test outputs == retained
    @test optical_gain_dither_phases(dither) == [2]
    @test optical_gain_estimator_frame(estimate) == 1

    @test process!(outputs, graph, estimator_graph_inputs(0.0f0)) ===
          OpticalGainEstimatorWindowFrameMismatch
    @test outputs == retained
    @test optical_gain_dither_phases(dither) == [3]
    @test optical_gain_estimator_frame(estimate) == 1
    @test reset!(graph) === graph
    @test optical_gain_dither_phases(dither) == [0]
    @test optical_gain_estimator_frame(estimate) == 0
end

@testset "estimator calibration replaces atomically at the Graph boundary" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_ESTIMATOR_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, OpticalGainEstimatorF32),
    )
    outputs = estimator_graph_outputs()
    process!(outputs, graph, estimator_graph_inputs(0.25f0))
    _, estimate = optical_gain_node_workspaces(graph)
    @test optical_gain_estimator_frame(estimate) == 1
    replacement = Float32[0.5]
    @test replace_parameters!(
        graph,
        Symbol("calibration-responses") => replacement,
    ) === graph
    replacement[1] = 9
    estimator_node = only(node for node in graph.nodes if node.name === :estimate)
    @test optical_gain_calibration_responses(estimator_node.prepared.plan) ==
          Float32[0.5]
    @test optical_gain_estimator_frame(estimate) == 1
    active = estimator_node.prepared.plan
    @test_throws ArgumentError replace_parameters!(
        graph,
        Symbol("calibration-responses") => Float32[NaN],
    )
    @test estimator_node.prepared.plan === active
end

@testset "optical-gain estimator Graph outcomes allocate no heap after warmup" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_ESTIMATOR_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, OpticalGainEstimatorF32),
    )
    outputs = estimator_graph_outputs()
    process!(outputs, graph, estimator_graph_inputs(0.25f0))
    valid = estimator_graph_inputs(0.0f0)
    @test estimator_graph_allocation_bytes(outputs, graph, valid) == 0

    rejected = estimator_graph_inputs(NaN32)
    process!(outputs, graph, rejected)
    @test estimator_graph_allocation_bytes(outputs, graph, rejected) == 0

end
