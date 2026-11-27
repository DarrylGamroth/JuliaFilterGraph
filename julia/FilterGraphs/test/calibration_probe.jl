const MODAL_PROBE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = probe
            label = modal-push-pull-f32
            config = {
                mode_count = 2
                corrector_coordinate_count = 2
                initial_positive_commands = [ 1.0 0.0 0.0 2.0 ]
                excitation_schema = "test.signal.f32/1"
            }
        }
    ]
    links = []
    inputs = [ "probe:positive-commands" ]
    outputs = [
        "probe:excitation"
        "probe:pattern-index"
        "probe:polarity"
        "probe:completes-cycle"
    ]
}
"""

const ZONAL_PROBE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = probe
            label = zonal-push-pull-f32
            config = {
                amplitudes = [ 1.0 2.0 ]
                excitation_schema = "test.signal.f32/1"
            }
        }
    ]
    links = []
    inputs = [ "probe:amplitudes" ]
    outputs = [
        "probe:excitation"
        "probe:pattern-index"
        "probe:polarity"
        "probe:completes-cycle"
    ]
}
"""

const HADAMARD_PROBE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = probe
            label = hadamard-push-pull-f32
            config = {
                amplitudes = [ 1.0 2.0 ]
                excitation_schema = "test.signal.f32/1"
            }
        }
    ]
    links = []
    inputs = [ "probe:amplitudes" ]
    outputs = [
        "probe:excitation"
        "probe:pattern-index"
        "probe:polarity"
        "probe:completes-cycle"
    ]
}
"""

const MODAL_PROBE_GATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = probe
            label = modal-push-pull-f32
            config = {
                mode_count = 2
                corrector_coordinate_count = 2
                initial_positive_commands = [ 1.0 0.0 0.0 2.0 ]
                excitation_schema = "test.signal.f32/1"
            }
        }
        {
            type = ndarray
            name = gate
            label = test-gate-f32
            config = { extent = 2 }
        }
    ]
    links = [
        { output = "probe:excitation" input = "gate:input" }
    ]
    inputs = [ "probe:positive-commands" ]
    outputs = [
        "gate:output"
        "probe:pattern-index"
        "probe:polarity"
        "probe:completes-cycle"
    ]
}
"""

const PROBE_OUTPUT_NAMES = (
    :excitation,
    Symbol("pattern-index"),
    :polarity,
    Symbol("completes-cycle"),
)
const PROBE_GATE_OUTPUT_NAMES = (
    :output,
    Symbol("pattern-index"),
    :polarity,
    Symbol("completes-cycle"),
)

function probe_graph_outputs(names=PROBE_OUTPUT_NAMES)
    return NamedTuple{names}(
        (
            zeros(Float32, 2),
            zeros(UInt64, 1),
            zeros(UInt8, 1),
            zeros(Bool, 1),
        ),
    )
end

probe_graph_process_allocation_bytes(outputs, graph) =
    @allocated process!(outputs, graph, NamedTuple())

function probe_graph_admitted_allocation_bytes(outputs, graph)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        NamedTuple(),
        graph.default_input_metadata,
    )
end

function probe_graph_candidate_allocation_bytes(outputs, graph)
    return @allocated FilterGraphs._process_candidate_admitted!(
        outputs,
        graph,
        NamedTuple(),
        graph.default_input_metadata,
    )
end


@testset "structured calibration sources prepare as zero-input Graph Nodes" begin
    cases = (
        (
            text=MODAL_PROBE_GRAPH_TEXT,
            declaration=ModalPushPullF32,
            first=Float32[1, 0],
            second=Float32[-1, -0.0],
        ),
        (
            text=ZONAL_PROBE_GRAPH_TEXT,
            declaration=ZonalPushPullF32,
            first=Float32[1, 0],
            second=Float32[-1, 0],
        ),
        (
            text=HADAMARD_PROBE_GRAPH_TEXT,
            declaration=HadamardPushPullF32,
            first=Float32[1, 2],
            second=Float32[-1, -2],
        ),
    )
    availability = NamedTuple{PROBE_OUTPUT_NAMES}((true, true, true, true))

    for case in cases
        graph = prepare_graph_text(case.text; algorithms=(case.declaration,))
        @test isempty(keys(graph.input_formats))
        @test keys(graph.output_formats) == PROBE_OUTPUT_NAMES
        @test keys(graph.parameter_formats) ==
              (case.declaration === ModalPushPullF32 ?
               (Symbol("positive-commands"),) : (:amplitudes,))
        @test graph.node_failure_types == (Union{},)
        @test map(format -> format.shape, values(graph.output_formats)) ==
              ((2,), (1,), (1,), (1,))
        @test map(format -> format.schema, values(graph.output_formats))[2:4] == (
            CALIBRATION_PATTERN_INDEX_V1,
            CALIBRATION_PROBE_POLARITY_V1,
            CALIBRATION_CYCLE_COMPLETE_V1,
        )

        outputs = probe_graph_outputs()
        @test process!(outputs, graph, NamedTuple()) == availability
        @test outputs.excitation == case.first
        @test getproperty(outputs, Symbol("pattern-index")) == UInt64[0]
        @test outputs.polarity == UInt8[0]
        @test getproperty(outputs, Symbol("completes-cycle")) == [false]
        @test process!(outputs, graph, NamedTuple()) == availability
        @test outputs.excitation == case.second
        @test outputs.polarity == UInt8[1]
        @test FilterGraphs.output_metadata(graph) == NamedTuple{
            PROBE_OUTPUT_NAMES,
        }((nothing, nothing, nothing, nothing))
    end
end

@testset "Graph parameter adoption preserves a modal probe pair" begin
    graph = prepare_graph_text(
        MODAL_PROBE_GRAPH_TEXT;
        algorithms=(ModalPushPullF32,),
    )
    outputs = probe_graph_outputs()
    process!(outputs, graph, NamedTuple())
    @test outputs.excitation == Float32[1, 0]

    @test replace_parameters!(
        graph,
        Symbol("positive-commands") => Float32[3 0; 0 4],
    ) === graph
    @test calibration_probe_next_step(only(graph.nodes).prepared.workspace) ==
          CalibrationProbeStep(0, CalibrationProbeNegative, false)
    process!(outputs, graph, NamedTuple())
    @test outputs.excitation == Float32[-3, -0.0]

    active = only(graph.nodes).prepared.plan
    @test_throws ArgumentError replace_parameters!(
        graph,
        Symbol("positive-commands") => zeros(Float32, 2, 2),
    )
    @test only(graph.nodes).prepared.plan === active
end

@testset "downstream rejection preserves outputs but advances an earlier probe" begin
    graph = prepare_graph_text(
        MODAL_PROBE_GATE_GRAPH_TEXT;
        algorithms=(ModalPushPullF32, TestGateF32),
    )
    workspace = first(graph.nodes).prepared.workspace
    outputs = probe_graph_outputs(PROBE_GATE_OUTPUT_NAMES)
    availability = NamedTuple{PROBE_GATE_OUTPUT_NAMES}((true, true, true, true))

    @test process!(outputs, graph, NamedTuple()) == availability
    @test outputs.output == Float32[1, 0]
    @test calibration_probe_next_step(workspace) ==
          CalibrationProbeStep(0, CalibrationProbeNegative, false)
    retained = map(copy, outputs)
    @test process!(outputs, graph, NamedTuple()) === TestGateRejected
    @test outputs == retained
    @test calibration_probe_next_step(workspace) ==
          CalibrationProbeStep(1, CalibrationProbePositive, false)
end

@testset "structured probe Graph paths allocate no heap after warmup" begin
    for (text, declaration) in (
        (MODAL_PROBE_GRAPH_TEXT, ModalPushPullF32),
        (ZONAL_PROBE_GRAPH_TEXT, ZonalPushPullF32),
        (HADAMARD_PROBE_GRAPH_TEXT, HadamardPushPullF32),
    )
        graph = prepare_graph_text(text; algorithms=(declaration,))
        outputs = probe_graph_outputs()
        process!(outputs, graph, NamedTuple())
        @test probe_graph_process_allocation_bytes(outputs, graph) == 0
        @test probe_graph_admitted_allocation_bytes(outputs, graph) == 0
        @test probe_graph_candidate_allocation_bytes(outputs, graph) == 0
    end

    gate_graph = prepare_graph_text(
        MODAL_PROBE_GATE_GRAPH_TEXT;
        algorithms=(ModalPushPullF32, TestGateF32),
    )
    gate_outputs = probe_graph_outputs(PROBE_GATE_OUTPUT_NAMES)
    process!(gate_outputs, gate_graph, NamedTuple())
    process!(gate_outputs, gate_graph, NamedTuple())
    @test probe_graph_process_allocation_bytes(gate_outputs, gate_graph) == 0
end
