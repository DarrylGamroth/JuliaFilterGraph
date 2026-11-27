const OPTICAL_GAIN_UPDATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = update
            label = optical-gain-update-f32
            config = {
                criteria = [
                    {
                        minimum_command_amplitude = 0.1
                        minimum_response_amplitude = 0.05
                        minimum_optical_gain = 0.2
                        maximum_optical_gain = 1.0
                        maximum_absolute_phase_lag = 0.25
                    }
                ]
                assignments = [
                    {
                        kind = dither-mode
                        dither_mode = 0
                        update_fraction = 0.5
                    }
                    { kind = hold }
                ]
                initial_optical_gains = [ 1.0 3.0 ]
            }
        }
    ]
    links = []
    inputs = [
        "update:command-amplitudes"
        "update:response-amplitudes"
        "update:optical-gain-estimates"
        "update:phase-lags"
        "update:valid"
        "update:estimate-complete"
    ]
    outputs = [
        "update:optical-gains"
        "update:accepted-dither-modes"
        "update:updated-reconstructed-modes"
    ]
}
"""

const OPTICAL_GAIN_UPDATE_INPUT_NAMES = (
    Symbol("command-amplitudes"),
    Symbol("response-amplitudes"),
    Symbol("optical-gain-estimates"),
    Symbol("phase-lags"),
    :valid,
    Symbol("estimate-complete"),
)
const OPTICAL_GAIN_UPDATE_OUTPUT_NAMES = (
    Symbol("optical-gains"),
    Symbol("accepted-dither-modes"),
    Symbol("updated-reconstructed-modes"),
)

function optical_gain_update_graph_inputs(; estimate=0.5f0, complete=true)
    return NamedTuple{OPTICAL_GAIN_UPDATE_INPUT_NAMES}(
        (
            Float32[0.5],
            Float32[0.25],
            Float32[estimate],
            Float32[0],
            trues(1),
            Bool[complete],
        ),
    )
end

function optical_gain_update_graph_outputs()
    return NamedTuple{OPTICAL_GAIN_UPDATE_OUTPUT_NAMES}(
        (fill(NaN32, 2), trues(1), trues(2)),
    )
end

optical_gain_update_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

@testset "bounded optical-gain update is a SPA-JSON Graph Node" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_UPDATE_GRAPH_TEXT;
        algorithms=(OpticalGainUpdateF32,),
    )
    @test keys(graph.input_formats) == OPTICAL_GAIN_UPDATE_INPUT_NAMES
    @test keys(graph.output_formats) == OPTICAL_GAIN_UPDATE_OUTPUT_NAMES
    @test getproperty.(values(graph.input_formats), :schema) == (
        OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        OPTICAL_GAIN_PHASE_LAG_V1,
        OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
    )
    @test getproperty.(values(graph.output_formats), :schema) == (
        PHYSICAL_OPTICAL_GAIN_V1,
        OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1,
        OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
    )
    @test graph.node_failure_types == (OpticalGainUpdateFailure,)
    workspace = only(graph.nodes).prepared.workspace

    outputs = optical_gain_update_graph_outputs()
    incomplete = optical_gain_update_graph_inputs(estimate=NaN32, complete=false)
    @test process!(outputs, graph, incomplete) ==
          NamedTuple{OPTICAL_GAIN_UPDATE_OUTPUT_NAMES}((true, true, true))
    @test getproperty(outputs, Symbol("optical-gains")) == Float32[1, 3]
    @test getproperty(outputs, Symbol("accepted-dither-modes")) == [false]
    @test getproperty(outputs, Symbol("updated-reconstructed-modes")) == [false, false]
    @test optical_gains(workspace) == Float32[1, 3]

    metadata = NamedTuple{OPTICAL_GAIN_UPDATE_INPUT_NAMES}(
        (
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            SampleMetadata(71, 0; terminal=true),
        ),
    )
    @test process!(outputs, graph, optical_gain_update_graph_inputs(), metadata) isa
          NamedTuple
    @test getproperty(outputs, Symbol("optical-gains")) == Float32[0.75, 3]
    @test getproperty(outputs, Symbol("accepted-dither-modes")) == [true]
    @test getproperty(outputs, Symbol("updated-reconstructed-modes")) == [true, false]
    @test optical_gains(workspace) == Float32[0.75, 3]
    @test all(
        value -> value == metadata[6],
        values(FilterGraphs.output_metadata(graph)),
    )
end

@testset "optical-gain update Node commits physical-gain adoption on success" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_UPDATE_GRAPH_TEXT;
        algorithms=(OpticalGainUpdateF32,),
    )
    outputs = optical_gain_update_graph_outputs()
    workspace = only(graph.nodes).prepared.workspace
    inputs = optical_gain_update_graph_inputs()

    @test process!(outputs, graph, inputs) isa NamedTuple
    @test optical_gains(workspace) == Float32[0.75, 3]
    retained = map(copy, outputs)
    @test process!(
        outputs,
        graph,
        optical_gain_update_graph_inputs(estimate=NaN32),
    ) === OpticalGainUpdateNonFiniteInput
    @test outputs == retained
    @test optical_gains(workspace) == Float32[0.75, 3]
    @test reset!(graph) === graph
    @test optical_gains(workspace) == Float32[1, 3]
end

@testset "optical-gain update Graph outcomes allocate no heap after warmup" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_UPDATE_GRAPH_TEXT;
        algorithms=(OpticalGainUpdateF32,),
    )
    outputs = optical_gain_update_graph_outputs()
    complete = optical_gain_update_graph_inputs()
    process!(outputs, graph, complete)
    @test optical_gain_update_graph_allocation_bytes(outputs, graph, complete) == 0

    incomplete = optical_gain_update_graph_inputs(estimate=NaN32, complete=false)
    process!(outputs, graph, incomplete)
    @test optical_gain_update_graph_allocation_bytes(outputs, graph, incomplete) == 0

    rejected = optical_gain_update_graph_inputs(estimate=NaN32)
    process!(outputs, graph, rejected)
    @test optical_gain_update_graph_allocation_bytes(outputs, graph, rejected) == 0

end
