const LEAKY_CONTROLLER_EXAMPLE = joinpath(@__DIR__, "data", "examples", "leaky-controller/graph.conf")

const LEAKY_CONTROLLER_OUTPUT_NAMES = (
    :demanded,
    Symbol("controller-constraint-feedback"),
)

function leaky_controller_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function leaky_controller_outputs(demanded, controller_constraint_feedback)
    return NamedTuple{LEAKY_CONTROLLER_OUTPUT_NAMES}(
        (demanded, controller_constraint_feedback),
    )
end

const LEAKY_CONTROLLER_PROJECTION = Float32[
    1 0.5 -0.25
    0 1 0.5
]

const LEAKY_ACTIVE_TO_FULL_PROJECTION = Float32[
    1 0
    0 1
    0.5 -1
]

const LEAKY_VDM_TO_PDM_PROJECTION = Float32[
    1 0 0.2
    0 1 -0.3
]

const LEAKY_FULL_TO_ACTIVE_PROJECTION = Float32[
    1 0 0.25
    0 1 -0.5
]

const LEAKY_PDM_TO_VDM_PROJECTION = Float32[
    1 0
    0 1
    0.25 -0.5
]

const LEAKY_VDM_TO_CONTROLLER_PROJECTION = Float32[
    1 0
    0 1
    0.5 -0.25
]

function prepare_leaky_pdm_graph()
    graph = prepare_graph(
        LEAKY_CONTROLLER_EXAMPLE;
        algorithms=algorithms(),
    )
    replace_parameters!(
        graph,
        "controller-to-vdm" => LEAKY_CONTROLLER_PROJECTION,
        "active-to-full" => LEAKY_ACTIVE_TO_FULL_PROJECTION,
        "vdm-to-pdm" => LEAKY_VDM_TO_PDM_PROJECTION,
        "full-to-active" => LEAKY_FULL_TO_ACTIVE_PROJECTION,
        "pdm-to-vdm" => LEAKY_PDM_TO_VDM_PROJECTION,
        "vdm-to-controller" => LEAKY_VDM_TO_CONTROLLER_PROJECTION,
    )
    return graph
end

@testset "leaky control law links through command and feedback projections" begin
    graph = prepare_leaky_pdm_graph()
    law = graph.nodes[1]
    workspace = law.prepared.workspace

    @test map(node -> node.name, graph.nodes) == (
        :law,
        :project,
        Symbol("pdm-project"),
        :condition,
        Symbol("feedback-to-vdm"),
        Symbol("feedback-to-controller"),
    )
    @test graph.input_formats.input.schema == CONTROLLER_RESIDUAL_ERROR_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("controller-to-vdm"),
    ).schema == CONTROLLER_TO_VDM_PROJECTION_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("active-to-full"),
    ).schema == ACTIVE_TO_FULL_VDM_PROJECTION_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("vdm-to-pdm"),
    ).schema == VDM_TO_PDM_PROJECTION_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("full-to-active"),
    ).schema == FULL_TO_ACTIVE_VDM_PROJECTION_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("pdm-to-vdm"),
    ).schema == PDM_TO_VDM_PROJECTION_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("vdm-to-controller"),
    ).schema == VDM_TO_CONTROLLER_PROJECTION_V1
    @test getproperty(
        graph.output_formats,
        :demanded,
    ).schema == DEMANDED_PDM_COMMAND_V1
    @test getproperty(
        graph.output_formats,
        Symbol("controller-constraint-feedback"),
    ).schema == CONTROLLER_CONSTRAINT_FEEDBACK_V1
    @test property_values(law.prepared.plan) == (gain=-0.4f0, pole=1.0f0, closed=true)

    demanded = zeros(Float32, 2)
    controller_constraint_feedback = zeros(Float32, 3)
    outputs = leaky_controller_outputs(demanded, controller_constraint_feedback)
    first_inputs = (input=Float32[1, -2, 0.5],)
    second_inputs = (input=Float32[0.5, 0.25, -0.75],)

    @test process!(outputs, graph, first_inputs) == (
        demanded=true,
        var"controller-constraint-feedback"=true,
    )
    @test demanded ≈ Float32[-0.085, 0.5]
    @test controller_constraint_feedback ≈
          Float32[-0.0503125, 0.503125, -0.1509375]
    @test graph.node_outputs[4][2] ≈ Float32[0, 0.4025]
    @test graph.node_outputs[5][1] ≈ Float32[-0.0503125, 0.503125]
    @test leaky_integrator_state(workspace) == Float32[-0.4, 0.8, -0.2]
    command_workspace = graph.nodes[4].prepared.workspace
    @test pdm_command_state(command_workspace) ≈ Float32[-0.085, 0.5]

    @test process!(outputs, graph, second_inputs) == (
        demanded=true,
        var"controller-constraint-feedback"=true,
    )
    @test demanded ≈ Float32[-0.4525, 0.8]
    @test controller_constraint_feedback ≈
          Float32[-0.02703125, 0.2703125, -0.08109375]
    @test graph.node_outputs[4][2] ≈ Float32[0, 0.21625]
    @test graph.node_outputs[5][1] ≈ Float32[-0.02703125, 0.2703125]
    @test leaky_integrator_state(workspace) ≈ Float32[-0.6, 0.7, 0.1]
    @test pdm_command_state(command_workspace) ≈ Float32[-0.4525, 0.8]

    process!(outputs, graph, first_inputs)
    @test leaky_controller_allocation_bytes(outputs, graph, second_inputs) == 0

    @test reset!(graph) === graph
    @test iszero(leaky_integrator_state(workspace))
    @test iszero(pdm_command_state(command_workspace))
end

@testset "terminal feedback rejection preserves earlier Workspace commits" begin
    graph = prepare_leaky_pdm_graph()
    workspace = graph.nodes[1].prepared.workspace
    command_workspace = graph.nodes[4].prepared.workspace
    demanded = zeros(Float32, 2)
    controller_constraint_feedback = zeros(Float32, 3)
    outputs = leaky_controller_outputs(demanded, controller_constraint_feedback)
    first_inputs = (input=Float32[1, -2, 0.5],)
    @test process!(outputs, graph, first_inputs) == (
        demanded=true,
        var"controller-constraint-feedback"=true,
    )
    replace_parameters!(
        graph,
        "vdm-to-controller" => fill(floatmax(Float32), 3, 2),
    )
    fill!(demanded, 71.0f0)
    fill!(controller_constraint_feedback, 72.0f0)
    rejected_inputs = (input=Float32[100, 100, 100],)
    @test process!(outputs, graph, rejected_inputs) ===
          VdmFeedbackToControllerNonFiniteOutput
    @test demanded == fill(71.0f0, 2)
    @test controller_constraint_feedback == fill(72.0f0, 3)
    @test leaky_integrator_state(workspace) == Float32[-40.4, -39.2, -40.2]
    command_state = pdm_command_state(command_workspace)
    # The projection can round one Float32 step differently across CPUs.
    @test command_state[1] ≈ -0.58500004f0 atol=eps(0.585f0) rtol=0
    @test iszero(command_state[2])

    process!(outputs, graph, rejected_inputs)
    @test leaky_controller_allocation_bytes(outputs, graph, rejected_inputs) == 0
end
