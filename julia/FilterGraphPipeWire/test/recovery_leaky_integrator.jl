@testset "Recovery of a leaky integrator without constraint feedback" begin
    text = """
        filter.graph = {
            nodes = [
                { type = ndarray name = correction label = leaky-integrator-f32
                  config = { extent = 1 initial_state = 0
                    input_schema = $(CONTROLLER_RESIDUAL_ERROR_V1)
                    output_schema = $(REQUESTED_PDM_COMMAND_V1) }
                  props = { gain = 1.0 pole = 1.0 } }
                { type = ndarray name = command label = pdm-command-f32
                  config = { actuator_count = 1 command_limits = [-1.0, 1.0]
                    slew_limit = 0.25 initial_current = [0]
                    quantization_origin = null quantization_step = null dither_seed = null } }
            ]
            links = [{ output = "correction:output" input = "command:requested" }]
            inputs = ["correction:input"]
            outputs = ["command:demanded"]
        }
        """
    graph = prepare_graph_text(text; algorithms=algorithms())
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing)
    integrator = first(graph.nodes).prepared
    command = last(graph.nodes).prepared.workspace
    inputs, outputs = (input=Float32[0.25],), (demanded=zeros(Float32, 1),)
    function process_frame!()
        @test all(values(FilterGraphs.process!(outputs, graph, inputs)))
        FilterGraphPipeWire._observe_recovery_reference!(owner)
        owner.process_count += UInt64(1)
        return only(outputs.demanded)
    end
    @test process_frame!() == 0.25f0
    @test process_frame!() == 0.5f0
    @test process_frame!() == 0.75f0

    scope = prepare_recovery(owner; controller=:correction, participants=(:command,))
    generation = stage_recovery!(scope)
    @test FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !integrator.plan.closed
    @test iszero(only(integrator.workspace.initial_state))
    @test iszero(only(integrator.workspace.committed_state))
    @test iszero(only(integrator.workspace.candidate_state))
    @test only(command.committed_current) == 0.75f0
    @test !recovery_status(scope).reference_reached
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
    @test process_frame!() == 0.5f0
    @test process_frame!() == 0.25f0
    @test !recovery_status(scope).reference_reached
    @test process_frame!() == 0f0
    @test recovery_status(scope).reference_reached
    stage_recovery_release!(scope, generation)
    @test FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !integrator.plan.closed
    @test recovery_status(scope).close_guarded
    FilterGraphPipeWire._invalidate_recovery_observation!(owner)
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(
        owner, SPA.Props("correction:closed" => true))
    @test process_frame!() == 0f0
    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(
        owner, SPA.Props("correction:closed" => true, "correction:gain" => 0.5f0))
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test !recovery_status(scope).close_guarded
    @test process_frame!() == 0.25f0
    nonzero = prepare_graph_text(replace(text, "initial_state = 0" => "initial_state = 0.25");
        algorithms=algorithms())
    @test_throws ArgumentError prepare_recovery(FilterGraphPipeWire.PreparedGraphOwner(nonzero);
        controller=:correction, participants=(:command,))

    # A second accumulator could retain a command after the selected one is opened.
    second = """
        { type = ndarray name = second label = leaky-integrator-f32
          config = { extent = 1 initial_state = 0
            input_schema = $(REQUESTED_PDM_COMMAND_V1) output_schema = $(REQUESTED_PDM_COMMAND_V1) } }
        """
    cascade = replace(text, "nodes = [" => "nodes = [$second",
        "input = \"command:requested\"" =>
            "input = \"second:input\" } { output = \"second:output\" input = \"command:requested\"")
    two = prepare_graph_text(cascade; algorithms=algorithms())
    @test_throws ArgumentError prepare_recovery(FilterGraphPipeWire.PreparedGraphOwner(two);
        controller=:correction, participants=(:command,))

    auxiliary = """
        { type = ndarray name = auxiliary label = vector-offset-f32
          config = { extent = 1 schema = $(CONTROLLER_CONSTRAINT_FEEDBACK_V1) } }
        """
    with_feedback = replace(text, "nodes = [" => "nodes = [$auxiliary",
        "inputs = [\"correction:input\"]" => "inputs = [\"correction:input\" { name = return-input endpoint = \"auxiliary:requested\" }]",
        "outputs = [\"command:demanded\"]" => "outputs = [\"command:demanded\" \"auxiliary:output\"]")
    bridged = prepare_graph_text(with_feedback; algorithms=algorithms())
    feedback_owner = FilterGraphPipeWire.PreparedGraphOwner(bridged, nothing;
        feedback=Symbol("return-input") => :output)
    @test_throws "leaky integrator recovery requires no feedback bridge" prepare_recovery(
        feedback_owner; controller=:correction, participants=(:command,))
end
