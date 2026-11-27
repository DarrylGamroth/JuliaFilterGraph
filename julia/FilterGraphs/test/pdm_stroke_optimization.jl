const PDM_STROKE_OPTIMIZATION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = optimize
            label = pdm-stroke-optimization-f32
            config = {
                actuator_count = 3
                null_mode_count = 1
                preferred_command = [ 0.0 0.0 0.0 ]
                command_to_null_modes = [ 0.33333334 0.33333334 0.33333334 ]
                null_modes_to_command = [ 1.0 1.0 1.0 ]
            }
        }
    ]
    links = []
    inputs = [
        "optimize:requested"
        "optimize:preferred-command"
        "optimize:command-to-null-modes"
        "optimize:null-modes-to-command"
    ]
    outputs = [ "optimize:optimized" ]
}
"""

const PDM_CONSTRAINT_CHAIN_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = optimize
            label = pdm-stroke-optimization-f32
            config = {
                actuator_count = 3
                null_mode_count = 1
                preferred_command = [ 0.0 0.0 0.0 ]
                command_to_null_modes = [ 0.33333334 0.33333334 0.33333334 ]
                null_modes_to_command = [ 1.0 1.0 1.0 ]
            }
        }
        {
            type = ndarray
            name = power
            label = pdm-command-power-limit-f32
            config = {
                actuator_count = 3
                max_command_power = 25.0
            }
        }
        {
            type = ndarray
            name = stroke
            label = inter-actuator-stroke-limit-f32
            config = {
                actuator_count = 3
                edges = [ [0 1] [1 2] ]
                max_differences = [ 4.0 4.0 ]
            }
        }
        {
            type = ndarray
            name = gate
            label = test-inter-actuator-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "optimize:optimized" input = "power:requested" }
        { output = "power:power-limited" input = "stroke:requested" }
        { output = "stroke:stroke-limited" input = "gate:input" }
    ]
    inputs = [
        "optimize:requested"
        "optimize:preferred-command"
        "optimize:command-to-null-modes"
        "optimize:null-modes-to-command"
        "stroke:max-differences"
    ]
    outputs = [ "gate:accepted" ]
}
"""

function pdm_stroke_optimization_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function pdm_stroke_optimization_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "PDM stroke optimization prepares as one Graph Node" begin
    graph = prepare_graph_text(
        PDM_STROKE_OPTIMIZATION_GRAPH_TEXT;
        algorithms=(PdmStrokeOptimizationF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:optimize,)
    @test graph.node_failure_types == (PdmStrokeOptimizationFailure,)
    @test keys(graph.input_formats) == (:requested,)
    @test keys(graph.output_formats) == (:optimized,)
    @test keys(graph.parameter_formats) == (
        Symbol("preferred-command"),
        Symbol("command-to-null-modes"),
        Symbol("null-modes-to-command"),
    )
    @test graph.input_formats.requested.schema == REQUESTED_PDM_COMMAND_V1
    @test graph.output_formats.optimized.schema == REQUESTED_PDM_COMMAND_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("preferred-command"),
    ).schema == PREFERRED_PDM_COMMAND_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("command-to-null-modes"),
    ).shape == (1, 3)
    @test getproperty(
        graph.parameter_formats,
        Symbol("null-modes-to-command"),
    ).shape == (3, 1)

    optimized = fill(71.0f0, 3)
    outputs = (optimized=optimized,)
    inputs = (requested=Float32[4, 1, -2],)
    @test process!(outputs, graph, inputs) == (optimized=true,)
    @test optimized == Float32[3, 0, -3]

    metadata = SampleMetadata(93, 0; timestamp_nanoseconds=9003, terminal=true)
    @test process!(outputs, graph, inputs, (requested=metadata,)) ==
          (optimized=true,)
    @test FilterGraphs.output_metadata(graph).optimized == metadata

    fill!(optimized, 83.0f0)
    retained_metadata = FilterGraphs.output_metadata(graph)
    @test process!(outputs, graph, (requested=Float32[NaN, 0, 0],)) ===
          PdmStrokeOptimizationNonFiniteInput
    @test optimized == fill(83.0f0, 3)
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "PDM optimization Graph parameters replace atomically" begin
    graph = prepare_graph_text(
        PDM_STROKE_OPTIMIZATION_GRAPH_TEXT;
        algorithms=(PdmStrokeOptimizationF32,),
    )
    original = only(graph.nodes).prepared.plan
    preferred = Float32[1, 1, 1]
    @test replace_parameters!(
        graph,
        Symbol("preferred-command") => preferred,
    ) === graph
    @test only(graph.nodes).prepared.plan !== original
    @test only(graph.nodes).prepared.plan.command_to_null_modes ===
          original.command_to_null_modes
    @test only(graph.nodes).prepared.plan.null_modes_to_command ===
          original.null_modes_to_command

    optimized = zeros(Float32, 3)
    outputs = (optimized=optimized,)
    @test process!(outputs, graph, (requested=preferred,)) == (optimized=true,)
    @test optimized == preferred

    active = only(graph.nodes).prepared.plan
    invalid = Float32[NaN 0 0]
    @test_throws ArgumentError replace_parameters!(
        graph,
        Symbol("command-to-null-modes") => invalid,
    )
    @test only(graph.nodes).prepared.plan === active
    @test process!(outputs, graph, (requested=preferred,)) == (optimized=true,)
    @test optimized == preferred
end

@testset "PDM optimization and physical constraints compose as Nodes" begin
    graph = prepare_graph_text(
        PDM_CONSTRAINT_CHAIN_GRAPH_TEXT;
        algorithms=(
            PdmStrokeOptimizationF32,
            PdmCommandPowerLimitF32,
            InterActuatorStrokeLimitF32,
            InterActuatorGateF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) == (
        :optimize,
        :power,
        :stroke,
        :gate,
    )
    @test graph.node_failure_types == (
        PdmStrokeOptimizationFailure,
        PdmCommandPowerLimitFailure,
        InterActuatorStrokeLimitFailure,
        InterActuatorGateFailure,
    )

    accepted = fill(71.0f0, 3)
    outputs = (accepted=accepted,)
    @test process!(outputs, graph, (requested=Float32[4, 1, -2],)) ==
          (accepted=true,)
    @test accepted == Float32[3, 0, -3]

    retained = copy(accepted)
    @test process!(outputs, graph, (requested=Float32[-4, -1, 2],)) ===
          InterActuatorGateRejected
    @test accepted == retained
end

@testset "PDM optimization Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        PDM_STROKE_OPTIMIZATION_GRAPH_TEXT;
        algorithms=(PdmStrokeOptimizationF32,),
    )
    outputs = (optimized=zeros(Float32, 3),)
    success = (requested=Float32[4, 1, -2],)
    rejection = (requested=Float32[NaN, 0, 0],)

    process!(outputs, graph, success)
    @test pdm_stroke_optimization_graph_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0
    process!(outputs, graph, rejection)
    @test pdm_stroke_optimization_graph_allocation_bytes(
        outputs,
        graph,
        rejection,
    ) == 0
    FilterGraphs._process_admitted!(
        outputs,
        graph,
        success,
        graph.default_input_metadata,
    )
    @test pdm_stroke_optimization_admitted_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0

    chain = prepare_graph_text(
        PDM_CONSTRAINT_CHAIN_GRAPH_TEXT;
        algorithms=(
            PdmStrokeOptimizationF32,
            PdmCommandPowerLimitF32,
            InterActuatorStrokeLimitF32,
            InterActuatorGateF32,
        ),
    )
    chain_outputs = (accepted=zeros(Float32, 3),)
    downstream_rejection = (requested=Float32[-4, -1, 2],)
    process!(chain_outputs, chain, downstream_rejection)
    @test pdm_stroke_optimization_graph_allocation_bytes(
        chain_outputs,
        chain,
        downstream_rejection,
    ) == 0
end
