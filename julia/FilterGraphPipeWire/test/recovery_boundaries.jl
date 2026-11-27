function process_recovery_owner!(owner, residual_value=0f0)
    inputs = (var"residual-error"=Float32[residual_value],
        var"constraint-feedback"=owner.feedback.slot)
    outputs = (demanded=zeros(Float32, 1),
        var"controller-constraint-feedback"=owner.feedback.scratch)
    @test all(values(FilterGraphs.process!(outputs, owner.graph, inputs)))
    copyto!(owner.feedback.slot, owner.feedback.scratch)
    FilterGraphPipeWire._observe_recovery_reference!(owner)
    owner.process_count += UInt64(1)
    return outputs.demanded
end

function settled_recovery_owner()
    # Keep this cold fixture from inferring the complete owner and recovery
    # constructor together; the concrete processing methods are tested below.
    graph = Base.invokelatest(prepared_recovery_graph)
    owner = Base.invokelatest(FilterGraphPipeWire.PreparedGraphOwner,graph,nothing;feedback=CALLBACK_FEEDBACK_PAIR)
    scope = prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command), reference_timeout_ms=60000)
    generation = stage_recovery!(scope)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    for _ in 1:8
        process_recovery_owner!(owner)
    end
    @test recovery_status(scope).reference_reached
    return owner, scope, generation
end

@testset "RESET cannot release a recovery against pre-reset evidence" begin
    owner, scope, generation = settled_recovery_owner()
    command = recovery_node(owner.graph, :command).prepared.workspace
    @test only(command.committed_current) ≈ 0.2f0
    stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._reset_owner!(owner)
    @test only(command.committed_current) ≈ 0.6f0
    @test recovery_status(scope).closure_inhibited
    @test recovery_status(scope).status === :release_refused
    @test !recovery_status(scope).reference_reached
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))
end

@testset "Released and release-pending frames publish fresh scope observations" begin
    owner, scope, generation = settled_recovery_owner()
    before = recovery_status(scope).observation_sequence
    stage_recovery_release!(scope, generation)
    process_recovery_owner!(owner) # an already executing frame completes
    @test recovery_status(scope).observation_sequence > before
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    released = recovery_status(scope).observation_sequence
    process_recovery_owner!(owner)
    @test recovery_status(scope).observation_sequence > released
    @test recovery_status(scope).reference_reached
end

@testset "First close rechecks scope evidence at adoption" begin
    owner, scope, generation = settled_recovery_owner()
    stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    process_recovery_owner!(owner)
    gain = recovery_node(owner.graph, :correction).prepared.plan.gain
    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true, "correction:gain" => 0.25f0))
    @test recovery_node(owner.graph, :correction).prepared.plan.gain == gain
    @test !recovery_node(owner.graph, :correction).prepared.plan.closed
    # A parameter mutation between release and close changes the target. Current
    # is still .2 while the next request is .7; close must await a new witness.
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,
        Symbol("system-flat") => Float32[0.7])
    FilterGraphPipeWire._adopt_pending_parameters!(owner)
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))
    for _ in 1:8
        process_recovery_owner!(owner)
    end
    @test recovery_status(scope).reference_reached
    # Model an in-flight frame: requested scratch has changed, but the last
    # published successful observation remains valid. Control staging reads
    # only atomic evidence; scientific arrays are inspected at callback adoption.
    flat_index = findfirst(node -> node.name === :flat, owner.graph.nodes)
    owner.graph.node_outputs[flat_index][1] .= NaN32
    @test FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true)) === nothing
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test !recovery_node(owner.graph, :correction).prepared.plan.closed
    @test recovery_status(scope).closure_inhibited
    @test recovery_status(scope).status === :release_refused
    @test recovery_status(scope).close_guarded
    @test !FilterGraphPipeWire._adopt_pending_properties!(owner)
    process_recovery_owner!(owner)
    stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    process_recovery_owner!(owner)
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test recovery_node(owner.graph, :correction).prepared.plan.closed
    # After the validated first close, ordinary nonzero hold/resume is retained.
    process_recovery_owner!(owner, 0.125f0)
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => false, "correction:gain" => 0.5f0))
    FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test recovery_node(owner.graph, :correction).prepared.plan.gain == 0.5f0
    process_recovery_owner!(owner)
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test recovery_node(owner.graph, :correction).prepared.plan.closed
end

@testset "Recovery admission requires causal temporal histories" begin
    graph = prepared_recovery_graph()
    FilterGraphs.replace_properties!(graph, "correction:pole"=>1f0,
        "correction:anti-windup-gain"=>0f0)
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    for _ in 1:4
        process_recovery_owner!(owner, 0.125f0)
    end
    for _ in 1:8
        process_recovery_owner!(owner)
    end
    @test only(recovery_node(graph, :command).prepared.workspace.committed_current) ≈ 0.7f0
    @test_throws ArgumentError prepare_recovery(owner; controller=:correction, participants=(:command,))
    @test only(recovery_node(graph, :correction).prepared.workspace.committed_output) ≈ 0.5f0
    @test only(recovery_node(graph, :command).prepared.workspace.committed_current) ≈ 0.7f0
end

@testset "RESET after release requires actual post-reset requalification" begin
    owner, scope, generation = settled_recovery_owner()
    stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test recovery_status(scope).close_guarded
    FilterGraphPipeWire._reset_owner!(owner)
    @test recovery_status(scope).close_guarded
    @test !recovery_status(scope).reference_reached
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))
    @test only(process_recovery_owner!(owner)) ≈ 0.5f0 # actual reset current .6, not old .2
    @test !recovery_status(scope).reference_reached
    for _ in 1:8
        process_recovery_owner!(owner)
    end
    @test recovery_status(scope).reference_reached
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test !recovery_status(scope).close_guarded
    @test recovery_node(owner.graph, :correction).prepared.plan.closed
end

@testset "Scope topology qualifies static references and rejects ambiguous targets" begin
    graph = prepared_recovery_graph(; reverse_nodes=true)
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(owner; controller=:correction, participants=(:command,))
    @test prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command)).reference_supported

    modal = """
        { type = ndarray name = modal label = biquad-cascade-f32
          config = { extent = 1 section_count = 1 per_mode = false
            initial_coefficients = [0 1 0 0 0]
            input_schema = $(CONTROLLER_RESIDUAL_ERROR_V1) output_schema = $(CONTROLLER_RESIDUAL_ERROR_V1) } }
    """
    upstream(text) = replace(replace(text,
        "\"correction:residual-error\"" => "\"modal:input\""),
        "nodes = [" => "nodes = [$modal",
        "links = [" => "links = [{ output = \"modal:output\" input = \"correction:residual-error\" }")
    upstream_graph = prepared_recovery_graph(; transform_text=upstream)
    upstream_owner = FilterGraphPipeWire.PreparedGraphOwner(upstream_graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(upstream_owner; controller=:correction,
        participants=(:temporal, :command))
    scope = prepare_recovery(upstream_owner; controller=:correction,
        participants=(:modal, :temporal, :command))
    modal_workspace = recovery_node(upstream_graph, :modal).prepared.workspace
    fill!(modal_workspace.committed_state, 0.5f0)
    stage_recovery!(scope)
    FilterGraphPipeWire._adopt_pending_recovery!(upstream_owner)
    @test all(iszero, modal_workspace.committed_state)

    # A static bias before a temporal filter has an evolving transient even
    # after correction is zero; instantaneous conditioner equality is insufficient.
    biased(text) = replace(text,
        "input_schema = $(CONTROLLER_COMMAND_V1) output_schema = $(CONTROLLER_COMMAND_V1)" =>
            "input_schema = $(REQUESTED_PDM_COMMAND_V1) output_schema = $(REQUESTED_PDM_COMMAND_V1)",
        "output = \"correction:correction\" input = \"temporal:input\"" =>
            "output = \"correction:correction\" input = \"projection:controller-command\"",
        "output = \"temporal:output\" input = \"projection:controller-command\"" =>
            "output = \"flat:requested-with-system-flat\" input = \"temporal:input\"",
        "output = \"flat:requested-with-system-flat\" input = \"command:requested\"" =>
            "output = \"temporal:output\" input = \"command:requested\"")
    biased_graph = prepared_recovery_graph(; transform_text=biased)
    biased_owner = FilterGraphPipeWire.PreparedGraphOwner(biased_graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(biased_owner; controller=:correction,
        participants=(:temporal, :command))

    combine = """
        { type = ndarray name = bypass label = vector-linear-combination-f32
          config = { extent = 1 left_scale = 1.0 right_scale = 1.0 schema = $(REQUESTED_PDM_COMMAND_V1) } }
    """
    bypass(text) = replace(text,
        "nodes = [" => "nodes = [$combine",
        "\"flat:system-flat\"" => "\"flat:system-flat\" \"bypass:right\"",
        "output = \"flat:requested-with-system-flat\" input = \"command:requested\"" =>
            "output = \"flat:requested-with-system-flat\" input = \"bypass:left\" }\n" *
            "{ output = \"bypass:output\" input = \"command:requested\"")
    bypass_graph = prepared_recovery_graph(; transform_text=bypass)
    bypass_owner = FilterGraphPipeWire.PreparedGraphOwner(bypass_graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(bypass_owner; controller=:correction,
        participants=(:temporal, :command))

    engineering = """
        { type = ndarray name = engineering label = pdm-engineering-offset-f32
          config = { actuator_count = 1 } }
    """
    fixed_offsets(text) = replace(text,
        "nodes = [" => "nodes = [$engineering",
        "\"flat:system-flat\"" => "\"flat:system-flat\" \"engineering:engineering-offset\"",
        "output = \"flat:requested-with-system-flat\" input = \"command:requested\"" =>
            "output = \"flat:requested-with-system-flat\" input = \"engineering:requested\" }\n" *
            "{ output = \"engineering:requested-with-offset\" input = \"command:requested\"")
    offset_graph = prepared_recovery_graph(; transform_text=fixed_offsets)
    replace_parameters!(offset_graph, Symbol("engineering-offset")=>Float32[0.1])
    offset_owner = FilterGraphPipeWire.PreparedGraphOwner(offset_graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    offset_scope = prepare_recovery(offset_owner; controller=:correction, participants=(:temporal, :command))
    stage_recovery!(offset_scope)
    FilterGraphPipeWire._adopt_pending_recovery!(offset_owner)
    for _ in 1:8
        @test only(process_recovery_owner!(offset_owner)) >= 0.3f0
    end
    @test only(recovery_node(offset_graph, :command).prepared.workspace.committed_current) ≈ 0.3f0
    @test recovery_status(offset_scope).reference_reached
end
