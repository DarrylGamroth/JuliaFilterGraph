@enum PdmPowerGateFailure::UInt8 begin
    PdmPowerGateRejected = 1
end

struct PdmPowerGatePlan
    extent::Int
end

@algorithm PdmPowerGateF32 begin
    label = "test-pdm-power-gate-f32"
    failure = PdmPowerGateFailure

    @configuration PdmPowerGateF32Configuration begin
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

    prepare(configuration) = PdmPowerGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] < 0.0f0 && return PdmPowerGateRejected
        copyto!(accepted, input)
        nothing
    end
end

const PDM_COMMAND_POWER_LIMIT_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = limit
            label = pdm-command-power-limit-f32
            config = {
                actuator_count = 3
                max_command_power = 6.25
            }
        }
    ]
    links = []
    inputs = [ "limit:requested" ]
    outputs = [ "limit:power-limited" ]
}
"""

const PDM_COMMAND_POWER_LIMIT_GATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = limit
            label = pdm-command-power-limit-f32
            config = {
                actuator_count = 3
                max_command_power = 6.25
            }
        }
        {
            type = ndarray
            name = gate
            label = test-pdm-power-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "limit:power-limited" input = "gate:input" }
    ]
    inputs = [ "limit:requested" ]
    outputs = [ "gate:accepted" ]
}
"""

const PDM_COMMAND_POWER_LIMIT_OUTPUT_NAMES = (Symbol("power-limited"),)

function pdm_command_power_limit_outputs(output)
    return NamedTuple{PDM_COMMAND_POWER_LIMIT_OUTPUT_NAMES}((output,))
end

function pdm_command_power_limit_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function pdm_command_power_limit_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "PDM command-power declaration prepares as one Graph Node" begin
    graph = prepare_graph_text(
        PDM_COMMAND_POWER_LIMIT_GRAPH_TEXT;
        algorithms=(PdmCommandPowerLimitF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:limit,)
    @test graph.node_failure_types == (PdmCommandPowerLimitFailure,)
    @test keys(graph.input_formats) == (:requested,)
    @test keys(graph.output_formats) == PDM_COMMAND_POWER_LIMIT_OUTPUT_NAMES
    @test isempty(graph.parameter_formats)
    @test graph.input_formats.requested.shape == (3,)
    @test graph.input_formats.requested.schema == REQUESTED_PDM_COMMAND_V1
    @test getproperty(
        graph.output_formats,
        Symbol("power-limited"),
    ).schema == REQUESTED_PDM_COMMAND_V1

    output = fill(71.0f0, 3)
    inputs = (requested=Float32[3, 4, 0],)
    outputs = pdm_command_power_limit_outputs(output)
    availability = NamedTuple{PDM_COMMAND_POWER_LIMIT_OUTPUT_NAMES}((true,))
    @test process!(outputs, graph, inputs) == availability
    @test output == Float32[1.5, 2, 0]

    metadata = SampleMetadata(91, 0; timestamp_nanoseconds=9001, terminal=true)
    @test process!(outputs, graph, inputs, (requested=metadata,)) == availability
    @test getproperty(
        FilterGraphs.output_metadata(graph),
        Symbol("power-limited"),
    ) == metadata

    fill!(output, 83.0f0)
    retained_metadata = FilterGraphs.output_metadata(graph)
    rejected = (requested=Float32[NaN, 0, 0],)
    @test process!(outputs, graph, rejected) ===
          PdmCommandPowerLimitNonFiniteInput
    @test output == fill(83.0f0, 3)
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "downstream rejection withholds the power-limited command" begin
    graph = prepare_graph_text(
        PDM_COMMAND_POWER_LIMIT_GATE_GRAPH_TEXT;
        algorithms=(PdmCommandPowerLimitF32, PdmPowerGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:limit, :gate)
    @test graph.node_failure_types == (
        PdmCommandPowerLimitFailure,
        PdmPowerGateFailure,
    )

    accepted = fill(71.0f0, 3)
    outputs = (accepted=accepted,)
    @test process!(outputs, graph, (requested=Float32[3, 4, 0],)) ==
          (accepted=true,)
    @test accepted == Float32[1.5, 2, 0]

    retained = copy(accepted)
    @test process!(outputs, graph, (requested=Float32[-3, 4, 0],)) ===
          PdmPowerGateRejected
    @test accepted == retained

    @test process!(outputs, graph, (requested=Float32[NaN, 0, 0],)) ===
          PdmCommandPowerLimitNonFiniteInput
    @test accepted == retained
end

@testset "PDM command-power Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        PDM_COMMAND_POWER_LIMIT_GRAPH_TEXT;
        algorithms=(PdmCommandPowerLimitF32,),
    )
    output = zeros(Float32, 3)
    outputs = pdm_command_power_limit_outputs(output)
    success = (requested=Float32[3, 4, 0],)
    rejection = (requested=Float32[NaN, 0, 0],)

    process!(outputs, graph, success)
    @test pdm_command_power_limit_graph_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0
    process!(outputs, graph, rejection)
    @test pdm_command_power_limit_graph_allocation_bytes(
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
    @test pdm_command_power_limit_admitted_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0

    gate_graph = prepare_graph_text(
        PDM_COMMAND_POWER_LIMIT_GATE_GRAPH_TEXT;
        algorithms=(PdmCommandPowerLimitF32, PdmPowerGateF32),
    )
    gate_outputs = (accepted=zeros(Float32, 3),)
    downstream_rejection = (requested=Float32[-3, 4, 0],)
    process!(gate_outputs, gate_graph, downstream_rejection)
    @test pdm_command_power_limit_graph_allocation_bytes(
        gate_outputs,
        gate_graph,
        downstream_rejection,
    ) == 0
end
