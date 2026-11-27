function prepared_coupled_feedback_recovery_graph(; demanded=false)
    port, schema = demanded ? ("demanded", DEMANDED_PDM_COMMAND_V1) :
        ("constraint-feedback", PDM_CONSTRAINT_FEEDBACK_V1)
    additions = """
        { type = ndarray name = feedback-map label = dense-vector-reconstructor-f32
          config = { input_count = 1 output_count = 1 initial_reconstructor = [1]
            input_schema = $schema output_schema = $(CONTROLLER_COMMAND_V1)
            reconstructor_schema = test.recovery.linear-map/1 } }
        { type = ndarray name = realized label = vector-linear-combination-f32
          config = { extent = 1 left_scale = 1.0 right_scale = -1.0 schema = $(CONTROLLER_COMMAND_V1) } }
        { type = ndarray name = remainder label = vector-linear-combination-f32
          config = { extent = 1 left_scale = 1.0 right_scale = -1.0 schema = $(CONTROLLER_COMMAND_V1) } }
        { type = ndarray name = woofer-map label = dense-vector-reconstructor-f32
          config = { input_count = 1 output_count = 1 initial_reconstructor = [1]
            input_schema = $(CONTROLLER_COMMAND_V1) output_schema = $(REQUESTED_PDM_COMMAND_V1)
            reconstructor_schema = test.recovery.linear-map/1 } }
        { type = ndarray name = woofer-offset label = vector-offset-f32
          config = { extent = 1 schema = $(REQUESTED_PDM_COMMAND_V1) } }
        { type = ndarray name = woofer-command label = pdm-command-f32
          config = { actuator_count = 1 command_limits = [-0.8, 0.8] slew_limit = 0.1
            initial_current = [0.6] quantization_origin = null quantization_step = null dither_seed = null } }
    """
    links = """
        { output = "command:$port" input = "feedback-map:ordered-input" }
        { output = "temporal:output" input = "realized:left" }
        { output = "feedback-map:output" input = "realized:right" }
        { output = "correction:correction" input = "remainder:left" }
        { output = "realized:output" input = "remainder:right" }
        { output = "remainder:output" input = "woofer-map:ordered-input" }
        { output = "woofer-map:output" input = "woofer-offset:requested" }
        { output = "woofer-offset:output" input = "woofer-command:requested" }
    """
    function transform(text)
        return replace(text, "nodes = [" => "nodes = [$additions",
            "links = [" => "links = [$links",
            "\"flat:system-flat\"" => "\"flat:system-flat\" \"woofer-offset:offset\"")
    end
    graph = prepared_recovery_graph(; transform_text=transform)
    replace_parameters!(graph, :offset => Float32[0.1])
    return graph
end

@testset "Coupled reference proof requires zero constraint feedback" begin
    graph = prepared_coupled_feedback_recovery_graph()
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command)) # every physical conditioner participates
    scope = prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command, Symbol("woofer-command")), reference_timeout_ms=60000)
    command_index = findfirst(node -> node.name === :command, graph.nodes)
    @test scope.feedback_required[command_index]
    @test count(identity, scope.feedback_required) == 1
    command = recovery_node(graph, :command).prepared
    woofer = recovery_node(graph, Symbol("woofer-command")).prepared
    calibration = recovery_node(graph, Symbol("woofer-map")).prepared.plan.matrix
    generation = stage_recovery!(scope)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    for _ in 1:3
        process_recovery_owner!(owner)
        @test any(!iszero, command.workspace.candidate_constraint_feedback)
        @test !recovery_status(scope).reference_reached
    end
    for _ in 1:12
        process_recovery_owner!(owner)
    end
    @test only(command.workspace.committed_current) ≈ 0.2f0
    @test only(woofer.workspace.committed_current) ≈ 0.1f0
    @test all(iszero, command.workspace.candidate_constraint_feedback)
    @test recovery_status(scope).reference_reached
    @test @allocated(FilterGraphPipeWire._observe_recovery_reference!(owner)) == 0
    @test recovery_node(graph, Symbol("woofer-map")).prepared.plan.matrix === calibration
    # Readiness requires both the last successful current equality and zero
    # feedback from the routed constraint-feedback branch.
    command.workspace.candidate_constraint_feedback .= 0.01f0
    FilterGraphPipeWire._observe_recovery_reference!(owner)
    @test !recovery_status(scope).reference_reached
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
    process_recovery_owner!(owner)
    stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    before = recovery_status(scope).observation_sequence
    process_recovery_owner!(owner)
    @test recovery_status(scope).observation_sequence > before
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    FilterGraphPipeWire._adopt_pending_properties!(owner)
    for _ in 1:3
        process_recovery_owner!(owner)
        @test only(command.workspace.committed_current) ≈ 0.2f0
        @test only(woofer.workspace.committed_current) ≈ 0.1f0
    end

    # A dynamic feedback branch must never feed a temporal filter whose earlier
    # nonzero state could masquerade as the final retained reference.
    temporal = recovery_node(graph, :temporal).prepared.plan
    @test_throws ArgumentError FilterGraphPipeWire._recovery_reference_transform(
        temporal, UInt8(2), true)
end

@testset "Recovery refuses a demanded-command cascade with compatible schemas" begin
    graph = prepared_coupled_feedback_recovery_graph(; demanded=true)
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError("only selected conditioner constraint-feedback may enter another recovery command path") prepare_recovery(
        owner; controller=:correction, participants=(:temporal, :command, Symbol("woofer-command")))
end

@testset "Recovery scope owns one exact delayed-feedback controller" begin
    graph = prepared_recovery_graph()
    without_bridge = FilterGraphPipeWire.PreparedGraphOwner(graph)
    @test_throws ArgumentError prepare_recovery(without_bridge; controller=:correction,
        participants=(:temporal, :command))
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command)).reference_supported
    controller_route = graph.input_routes[1][2]
    @test !FilterGraphPipeWire._recovery_feedback_route_matches(controller_route, 1)
    @test FilterGraphPipeWire._recovery_feedback_route_matches(controller_route, 2)
end
