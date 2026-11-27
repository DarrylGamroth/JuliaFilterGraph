@enum InterActuatorGateFailure::UInt8 begin
    InterActuatorGateRejected = 1
end

struct InterActuatorGatePlan
    extent::Int
end

@algorithm InterActuatorGateF32 begin
    label = "test-inter-actuator-gate-f32"
    failure = InterActuatorGateFailure

    @configuration InterActuatorGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            REQUESTED_PDM_COMMAND_V1,
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            REQUESTED_PDM_COMMAND_V1;
            metadata_from=:input,
        ),
    )

    prepare(configuration) = InterActuatorGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] < 0.0f0 && return InterActuatorGateRejected
        copyto!(accepted, input)
        nothing
    end
end

const INTER_ACTUATOR_STROKE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = stroke
            label = inter-actuator-stroke-limit-f32
            config = {
                actuator_count = 3
                edges = [ [0 1] [1 2] ]
                max_differences = [ 1.0 0.375 ]
            }
        }
    ]
    links = []
    inputs = [ "stroke:requested" "stroke:max-differences" ]
    outputs = [ "stroke:stroke-limited" ]
}
"""

const PDM_LIMIT_CHAIN_GRAPH_TEXT = """
filter.graph = {
    nodes = [
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
                max_differences = [ 1.0 0.375 ]
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
        { output = "power:power-limited" input = "stroke:requested" }
        { output = "stroke:stroke-limited" input = "gate:input" }
    ]
    inputs = [ "power:requested" "stroke:max-differences" ]
    outputs = [ "gate:accepted" ]
}
"""

const INTER_ACTUATOR_STROKE_OUTPUT_NAMES = (Symbol("stroke-limited"),)

function inter_actuator_stroke_outputs(output)
    return NamedTuple{INTER_ACTUATOR_STROKE_OUTPUT_NAMES}((output,))
end

function inter_actuator_stroke_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function inter_actuator_stroke_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "inter-actuator stroke declaration prepares as one Graph Node" begin
    graph = prepare_graph_text(
        INTER_ACTUATOR_STROKE_GRAPH_TEXT;
        algorithms=(InterActuatorStrokeLimitF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:stroke,)
    @test graph.node_failure_types == (InterActuatorStrokeLimitFailure,)
    @test keys(graph.input_formats) == (:requested,)
    @test keys(graph.output_formats) == INTER_ACTUATOR_STROKE_OUTPUT_NAMES
    @test keys(graph.parameter_formats) == (Symbol("max-differences"),)
    @test graph.input_formats.requested.shape == (3,)
    @test graph.input_formats.requested.schema == REQUESTED_PDM_COMMAND_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("max-differences"),
    ).schema == INTER_ACTUATOR_STROKE_LIMIT_V1

    output = fill(71.0f0, 3)
    inputs = (requested=Float32[2, -2, 1],)
    outputs = inter_actuator_stroke_outputs(output)
    availability = NamedTuple{INTER_ACTUATOR_STROKE_OUTPUT_NAMES}((true,))
    @test process!(outputs, graph, inputs) == availability
    @test output == Float32[0.25, -0.25, 0.125]

    metadata = SampleMetadata(92, 0; timestamp_nanoseconds=9002, terminal=true)
    @test process!(outputs, graph, inputs, (requested=metadata,)) == availability
    @test getproperty(
        FilterGraphs.output_metadata(graph),
        Symbol("stroke-limited"),
    ) == metadata

    fill!(output, 83.0f0)
    retained_metadata = FilterGraphs.output_metadata(graph)
    rejected = (requested=Float32[NaN, 0, 0],)
    @test process!(outputs, graph, rejected) ===
          InterActuatorStrokeLimitNonFiniteInput
    @test output == fill(83.0f0, 3)
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "Graph parameter replacement is failure-atomic" begin
    graph = prepare_graph_text(
        INTER_ACTUATOR_STROKE_GRAPH_TEXT;
        algorithms=(InterActuatorStrokeLimitF32,),
    )
    output = zeros(Float32, 3)
    outputs = inter_actuator_stroke_outputs(output)
    inputs = (requested=Float32[2, -2, 1],)
    original = only(graph.nodes).prepared.plan

    replacement_storage = Float32[83, 4, 83, 4]
    replacement = @view replacement_storage[2:2:4]
    @test replace_parameters!(
        graph,
        Symbol("max-differences") => replacement,
    ) === graph
    @test only(graph.nodes).prepared.plan !== original
    @test process!(outputs, graph, inputs) ==
          NamedTuple{INTER_ACTUATOR_STROKE_OUTPUT_NAMES}((true,))
    @test output == inputs.requested

    active = only(graph.nodes).prepared.plan
    @test_throws ArgumentError replace_parameters!(
        graph,
        Symbol("max-differences") => Float32[4, -1],
    )
    @test only(graph.nodes).prepared.plan === active
    @test process!(outputs, graph, inputs) ==
          NamedTuple{INTER_ACTUATOR_STROKE_OUTPUT_NAMES}((true,))
    @test output == inputs.requested
end

@testset "power and stroke limits compose before terminal publication" begin
    graph = prepare_graph_text(
        PDM_LIMIT_CHAIN_GRAPH_TEXT;
        algorithms=(
            PdmCommandPowerLimitF32,
            InterActuatorStrokeLimitF32,
            InterActuatorGateF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) == (:power, :stroke, :gate)
    @test graph.node_failure_types == (
        PdmCommandPowerLimitFailure,
        InterActuatorStrokeLimitFailure,
        InterActuatorGateFailure,
    )

    accepted = fill(71.0f0, 3)
    outputs = (accepted=accepted,)
    @test process!(outputs, graph, (requested=Float32[2, -2, 1],)) ==
          (accepted=true,)
    @test accepted == Float32[0.25, -0.25, 0.125]

    retained = copy(accepted)
    @test process!(outputs, graph, (requested=Float32[-2, 2, 1],)) ===
          InterActuatorGateRejected
    @test accepted == retained
end

@testset "inter-actuator Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        INTER_ACTUATOR_STROKE_GRAPH_TEXT;
        algorithms=(InterActuatorStrokeLimitF32,),
    )
    output = zeros(Float32, 3)
    outputs = inter_actuator_stroke_outputs(output)
    success = (requested=Float32[2, -2, 1],)
    rejection = (requested=Float32[NaN, 0, 0],)

    process!(outputs, graph, success)
    @test inter_actuator_stroke_graph_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0
    process!(outputs, graph, rejection)
    @test inter_actuator_stroke_graph_allocation_bytes(
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
    @test inter_actuator_stroke_admitted_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0

    chain = prepare_graph_text(
        PDM_LIMIT_CHAIN_GRAPH_TEXT;
        algorithms=(
            PdmCommandPowerLimitF32,
            InterActuatorStrokeLimitF32,
            InterActuatorGateF32,
        ),
    )
    chain_outputs = (accepted=zeros(Float32, 3),)
    downstream_rejection = (requested=Float32[-2, 2, 1],)
    process!(chain_outputs, chain, downstream_rejection)
    @test inter_actuator_stroke_graph_allocation_bytes(
        chain_outputs,
        chain,
        downstream_rejection,
    ) == 0
end
