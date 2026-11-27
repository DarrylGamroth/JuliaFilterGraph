# Uses the callback helpers and CPU feedback graph in feedback_bridge.jl.
function prepared_recovery_graph(; transform_text=identity, reverse_nodes=false)
    additions = """
        { type = ndarray name = temporal label = biquad-cascade-f32
          config = { extent = 1 section_count = 1 per_mode = false
            initial_coefficients = [0 1 0 0 0]
            input_schema = controller.command.v1 output_schema = controller.command.v1 } }
        { type = ndarray name = flat label = pdm-system-flat-f32
          config = { actuator_count = 1 } }
    """
    text = replace(CALLBACK_FEEDBACK_GRAPH,
        "nodes = [" => "nodes = [\n$additions";
        count=1)
    text = replace(text,
        "input_schema = controller.command.v1 output_schema = controller.command.v1" =>
            "input_schema = $(CONTROLLER_COMMAND_V1) output_schema = $(CONTROLLER_COMMAND_V1)",
        "output = \"correction:correction\" input = \"projection:controller-command\"" =>
            "output = \"correction:correction\" input = \"temporal:input\" }\n" *
            "{ output = \"temporal:output\" input = \"projection:controller-command\"",
        "output = \"physical:requested-pdm-command\" input = \"command:requested\"" =>
            "output = \"physical:requested-pdm-command\" input = \"flat:requested\" }\n" *
            "{ output = \"flat:requested-with-system-flat\" input = \"command:requested\"",
        "slew_limit = null" => "slew_limit = 0.1",
        "initial_current = null" => "initial_current = [0.6]",
        "\"projection:controller-to-vdm\"" =>
            "\"temporal:coefficients\" \"flat:system-flat\" \"projection:controller-to-vdm\"")
    graph = if reverse_nodes
        mktemp() do path, io
            write(io, transform_text(text)); flush(io)
            spec = FilterGraphs._load_graph(path)
            reversed = FilterGraphs.GraphSpec(reverse(spec.nodes), spec.links,
                spec.inputs, spec.outputs, spec.cpu_execution)
            FilterGraphs._prepare_graph(reversed; algorithms=algorithms())
        end
    else
        prepare_graph_text(transform_text(text); algorithms=algorithms())
    end
    identity = ones(Float32, 1, 1)
    replace_parameters!(graph,
        Symbol("shape-to-hidden") => zeros(Float32, 1, 1),
        Symbol("hidden-to-shape") => zeros(Float32, 1, 1),
        Symbol("controller-to-vdm") => identity,
        Symbol("active-to-full") => identity,
        Symbol("vdm-to-pdm") => identity,
        Symbol("full-to-active") => identity,
        Symbol("pdm-to-vdm") => identity,
        Symbol("vdm-to-controller") => identity,
        Symbol("system-flat") => Float32[0.2])
    return graph
end

recovery_node(graph, name) = only(filter(node -> node.name === name, graph.nodes))

@testset "Scoped recovery opens, clears histories and retains constrained current" begin
    graph = prepared_recovery_graph()
    node = PipeWireNode(graph; rate=1000//1, connect=false, feedback=CALLBACK_FEEDBACK_PAIR)
    owner = node.owner
    FilterGraphPipeWire._prepare_owner(owner)
    scope = prepare_recovery(node; controller=:correction,
        participants=(:temporal, :command), reference_timeout_ms=60000)
    controller = recovery_node(graph, :correction).prepared
    temporal = recovery_node(graph, :temporal).prepared
    command = recovery_node(graph, :command).prepared
    flat = recovery_node(graph, :flat).prepared.plan
    projection = recovery_node(graph, :projection).prepared.plan
    coefficients = temporal.plan
    original_controller = controller.plan
    initial_current = copy(command.workspace.initial_current)
    residual, demanded = Float32[2], zeros(Float32, 1)
    native_inputs = native_callback_buffer(residual;
        header=BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(1), Int64(0), UInt64(19)),
        acquisition_sequence=UInt64(19))
    native_outputs = native_callback_buffer(demanded; header_available=true, properties_available=true)
    callback = node.filter.callbacks.on_process
    function process_recovery_frame!()
        GC.@preserve residual demanded native_inputs native_outputs begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_inputs)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_outputs)), UInt32(1))
            callback(node.filter, inputs, outputs)
        end
        return only(demanded)
    end
    for _ in 1:6
        process_recovery_frame!()
    end
    @test only(command.workspace.committed_current) ≈ 0.8f0
    @test any(!iszero, controller.workspace.committed_state)
    @test any(!iszero, temporal.workspace.committed_state)
    @test any(!iszero, owner.feedback.slot)
    old_current = copy(command.workspace.committed_current)
    acquisition = native_callback_acquisition(native_inputs)

    # Admission fails without changing any state, even for a partially valid scope.
    @test_throws ArgumentError prepare_recovery(node; controller=:correction, participants=(:temporal, :flat))
    @test_throws ArgumentError prepare_recovery(node; controller=:projection)
    @test_throws ArgumentError prepare_recovery(node; controller=:correction, participants=(:missing,))
    @test_throws ArgumentError prepare_recovery(node; controller=:correction, participants=(:correction,))
    @test command.workspace.committed_current == old_current
    @test controller.plan.closed

    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:gain" => 1f0))
    @test_throws PropertyUpdateBusy stage_recovery!(scope)
    @test recovery_status(scope).status === :idle
    FilterGraphPipeWire._adopt_pending_properties!(owner)
    generation = stage_recovery!(scope)
    @test generation == 1
    @test recovery_status(scope).status === :pending
    @test recovery_status(scope).closure_inhibited
    @test recovery_status(scope).active_generation == 0
    @test controller.plan.closed # Staging itself has no scientific effects.
    @test command.workspace.committed_current == old_current
    @test_throws PropertyUpdateBusy stage_recovery!(scope)
    @test_throws PropertyUpdateBusy FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))
    @test !FilterGraphPipeWire._stage_parameter_transaction!(owner,
        Symbol("system-flat") => Float32[0.7])

    # Inspect exactly the callback adoption boundary before normal slew advances.
    @test FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !controller.plan.closed
    @test all(iszero, controller.workspace.committed_state)
    @test all(iszero, controller.workspace.committed_output)
    @test all(iszero, temporal.workspace.committed_state)
    @test all(iszero, owner.feedback.slot) && all(iszero, owner.feedback.scratch)
    @test !owner.feedback.failed
    @test command.workspace.committed_current == command.workspace.candidate_current == old_current
    @test command.workspace.initial_current == initial_current
    @test recovery_node(graph, :flat).prepared.plan === flat
    @test recovery_node(graph, :projection).prepared.plan === projection
    @test temporal.plan === coefficients
    @test controller.plan.shape_to_hidden === original_controller.shape_to_hidden
    @test controller.plan.hidden_to_shape === original_controller.hidden_to_shape
    @test recovery_status(scope).status === :applied
    @test recovery_status(scope).active_generation == generation
    @test recovery_status(scope).scope_matches
    @test !FilterGraphPipeWire._adopt_pending_recovery!(owner) # consumed exactly once
    @test_throws InvalidStateException stage_recovery!(scope)
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))
    @test_throws InvalidStateException stage_recovery_release!(scope, generation + UInt64(1))
    foreign = prepare_recovery(node; controller=:correction, participants=(:temporal, :command))
    @test_throws InvalidStateException stage_recovery_release!(foreign, generation)

    residual .= 0
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
    @test !recovery_status(scope).reference_reached
    @test process_recovery_frame!() ≈ old_current[1] - 0.1f0
    @test !recovery_status(scope).reference_reached
    @test recovery_status(scope).observation_sequence > 0
    @test all(iszero, controller.workspace.committed_output)
    for _ in 1:8
        process_recovery_frame!()
    end
    @test only(demanded) ≈ 0.2f0
    @test all(iszero, temporal.workspace.committed_state)
    @test all(iszero, owner.feedback.slot)
    @test native_callback_acquisition(native_inputs) == acquisition
    @test native_callback_acquisition(native_outputs) == acquisition

    # Reference observations come only from complete successful publications.
    @test recovery_status(scope).reference_supported
    @test recovery_status(scope).reference_reached
    @test recovery_status(scope).observation_sequence > 0
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:gain" => 0.5f0))
    @test_throws InvalidStateException FilterGraphPipeWire._stage_parameter_transaction!(owner,
        Symbol("system-flat") => Float32[0.7])
    @test recovery_status(scope).closure_inhibited
    @test recovery_status(scope).active_generation == generation
    @test_throws InvalidStateException FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("correction:closed" => true))

    @test stage_recovery_release!(scope, generation) == generation
    @test recovery_status(scope).status === :release_pending
    @test recovery_status(scope).closure_inhibited
    process_recovery_frame!()
    @test recovery_status(scope).status === :released
    @test !recovery_status(scope).closure_inhibited
    @test !controller.plan.closed
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
    FilterGraphPipeWire._stage_property_transaction!(owner, SPA.Props("correction:closed" => true))
    for _ in 1:4
        @test process_recovery_frame!() ≈ 0.2f0
        @test all(iszero, controller.workspace.committed_output)
    end
    @test controller.plan.closed

    # Warmed recovery of this heterogeneous graph reuses all prepared storage.
    generation2 = stage_recovery!(scope)
    @test generation2 == generation + UInt64(1)
    @test @allocated(FilterGraphPipeWire._adopt_pending_recovery!(owner)) == 0
    @test recovery_status(scope).active_generation == generation2
    @test !recovery_status(scope).reference_reached
    @test recovery_status(scope).observation_sequence == 0
    @test @allocated(process_recovery_frame!()) == 0
    @test recovery_status(scope).reference_reached
    successful_sequence = recovery_status(scope).observation_sequence
    residual .= NaN32
    @test_throws InvalidStateException process_recovery_frame!()
    @test !recovery_status(scope).reference_reached
    @test recovery_status(scope).observation_sequence == successful_sequence
    @test_throws InvalidStateException stage_recovery_release!(scope, generation2)
    @test recovery_status(scope).closure_inhibited
    close(node)
end

@testset "Recovery refuses progressive and joined execution" begin
    progressive = prepared_progressive_feedback_graph()
    progressive_node = PipeWireNode(progressive; rate=1000//1, connect=false,
        boundary_layout=RowMajorLayout(), feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(progressive_node; controller=:correction, participants=(:command,))
    close(progressive_node)
    joined = FilterGraphs.prepare_latest_hold(prepared_latest_hold_graph(), :primary, :held)
    joined_owner = FilterGraphPipeWire.PreparedGraphOwner(joined)
    @test_throws ArgumentError prepare_recovery(joined_owner; controller=:correction)
    @test_throws ArgumentError prepare_recovery(nothing; controller=:correction)
end

recovery_property_values(properties) = Dict(name => PipeWireAO.pod_value(value)
    for (name, value) in properties.values)

@testset "Optional native recovery controls share admission and expose adoption" begin
    graph = prepared_recovery_graph()
    events = Tuple{Symbol,Dict{String,Any}}[]
    observer(phase, properties) = push!(events,
        (phase isa Val{:requested} ? :requested : :observed, recovery_property_values(properties)))
    node = PipeWireNode(graph; rate=1000//1, connect=false, feedback=CALLBACK_FEEDBACK_PAIR,
        recovery=(controller=:correction, participants=(:temporal, :command), reference_timeout_ms=60000),
        property_observer=observer)
    FilterGraphPipeWire._prepare_owner(node.owner)
    readback() = recovery_property_values(node.filter.callbacks.on_get_properties(node.filter))
    before = readback()
    @test before["pipewireao-recovery:version"] === Int32(1)
    @test before["pipewireao-recovery:reference-timeout-ms"] === Int32(60000)
    @test before["pipewireao-recovery:controller"] == "correction"
    @test before["pipewireao-recovery:state"] == "idle"
    @test before["pipewireao-recovery:requested-generation"] == 0
    @test before["pipewireao-recovery:active-generation"] == 0
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int64(10), "correction:closed" => false))
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:closure-inhibited" => false))
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int64(0)))
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int32(1)))
    @test readback()["pipewireao-recovery:requested-generation"] == 0
    @test stage_properties!(node, SPA.Props("pipewireao-recovery:request-token" => Int64(10))) === nothing
    @test last(events)[1] === :requested
    @test last(events)[2]["correction:closed"] === false
    @test last(events)[2]["correction:requested-generation"] == 1
    @test last(events)[2]["correction:active-generation"] == 0
    pending = readback()
    @test pending["pipewireao-recovery:request-token"] == 10
    @test pending["pipewireao-recovery:requested-generation"] == 1
    @test pending["pipewireao-recovery:active-generation"] == 0
    @test pending["pipewireao-recovery:state"] == "pending"
    @test_throws PropertyUpdateBusy stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int64(11)))
    @test_throws PropertyUpdateBusy stage_properties!(node, SPA.Props("correction:closed" => true))

    residual, demanded = zeros(Float32, 1), zeros(Float32, 1)
    input = native_callback_buffer(residual; header=BufferHeader(
        SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(1), Int64(0), UInt64(1)))
    output = native_callback_buffer(demanded; header_available=true)
    function frame!()
        GC.@preserve residual demanded input output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output)), UInt32(1))
            node.filter.callbacks.on_process(node.filter, inputs, outputs)
        end
        return nothing
    end
    frame!() # Recovery is adopted by actual native process callback.
    applied = readback()
    @test applied["pipewireao-recovery:state"] == "applied"
    @test applied["pipewireao-recovery:requested-generation"] == 1
    @test applied["pipewireao-recovery:active-generation"] == 1
    @test applied["pipewireao-recovery:recovery-generation"] == 1
    @test applied["correction:closed"] === false
    @test applied["correction:requested-generation"] == 1
    @test applied["correction:active-generation"] == 1
    @test applied["pipewireao-recovery:reference-reached"] === false
    @test_throws InvalidStateException stage_properties!(node, SPA.Props(
        "pipewireao-recovery:release-token" => Int64(1)))
    @test_throws InvalidStateException stage_properties!(node, SPA.Props("correction:gain" => 2f0))
    for _ in 1:8
        frame!()
    end
    reached = readback()
    @test reached["pipewireao-recovery:reference-reached"] === true
    @test reached["pipewireao-recovery:observation-sequence"] > 1
    @test reached["pipewireao-recovery:observation-monotonic-nanoseconds"] > 0
    fresh_timestamp = node.owner.recovery_observation_timestamp_ns[]
    node.owner.recovery_observation_timestamp_ns[] = time_ns() - UInt64(60_000_000_001)
    @test readback()["pipewireao-recovery:reference-reached"] === false
    @test_throws InvalidStateException stage_properties!(node, SPA.Props(
        "pipewireao-recovery:release-token" => Int64(1)))
    @test readback()["pipewireao-recovery:requested-generation"] == 1
    node.owner.recovery_observation_timestamp_ns[] = fresh_timestamp
    @test stage_properties!(node, SPA.Props("pipewireao-recovery:release-token" => Int64(1))) === nothing
    releasing = readback()
    @test releasing["pipewireao-recovery:requested-generation"] == 2
    @test releasing["pipewireao-recovery:active-generation"] == 1
    @test releasing["pipewireao-recovery:recovery-generation"] == 1
    @test releasing["pipewireao-recovery:release-token"] == 1
    @test releasing["pipewireao-recovery:closure-inhibited"] === true
    # Expiry after staging is refused at adoption; successful processing can
    # refresh evidence but cannot replay the refused release.
    node.owner.recovery_observation_timestamp_ns[] = time_ns() - UInt64(60_000_000_001)
    frame!()
    refused = readback()
    @test refused["pipewireao-recovery:state"] == "release_refused"
    @test refused["pipewireao-recovery:active-generation"] == 2
    @test refused["pipewireao-recovery:closure-inhibited"] === true
    @test refused["pipewireao-recovery:reference-reached"] === true
    @test stage_properties!(node, SPA.Props("pipewireao-recovery:release-token" => Int64(1))) === nothing
    frame!()
    released = readback()
    @test released["pipewireao-recovery:active-generation"] == 3
    @test released["pipewireao-recovery:state"] == "released"
    @test released["pipewireao-recovery:closure-inhibited"] === false
    @test released["pipewireao-recovery:close-guarded"] === true
    @test released["correction:closed"] === false
    @test_throws InvalidStateException stage_properties!(node, SPA.Props(
        "pipewireao-recovery:release-token" => Int64(1)))
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int64(10)))
    @test_throws ArgumentError stage_properties!(node, SPA.Props(
        "pipewireao-recovery:request-token" => Int64(9)))
    @test stage_properties!(node, SPA.Props("correction:closed" => true)) === nothing
    frame!()
    @test readback()["correction:closed"] === true
    @test readback()["pipewireao-recovery:close-guarded"] === false
    @test readback()["pipewireao-recovery:reference-reached"] === false
    @test only(demanded) ≈ 0.2f0
    @test @allocated(frame!()) == 0
    close(node)

    ordinary = PipeWireNode(prepared_recovery_graph(); rate=1000//1,
        connect=false, feedback=CALLBACK_FEEDBACK_PAIR)
    @test all(pair -> !startswith(pair.first, "pipewireao-recovery:"),
        ordinary.filter.callbacks.on_get_properties(ordinary.filter).values)
    close(ordinary)
    collision = prepare_graph_text(replace(CALLBACK_FEEDBACK_GRAPH,
        "name = correction" => "name = pipewireao-recovery",
        "correction:" => "pipewireao-recovery:"); algorithms=algorithms())
    @test_throws ArgumentError PipeWireNode(collision; rate=1000//1, connect=false,
        feedback=CALLBACK_FEEDBACK_PAIR,
        recovery=(controller=:"pipewireao-recovery", participants=(:command,)))
end

@testset "Recovery consumes an unexpected application failure without replay" begin
    graph = prepared_recovery_graph()
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    scope = prepare_recovery(owner; controller=:correction, participants=(:temporal, :command))
    generation = stage_recovery!(scope)
    transaction = owner.recovery_transaction
    # Fault injection: violate the prepared selection invariant by selecting a
    # stateless flat workspace, whose unsupported scientific recovery must fail.
    flat_index = findfirst(node -> node.name === :flat, graph.nodes)
    selected = Base.setindex(transaction.touched, true, flat_index)
    owner.recovery_transaction = FilterGraphPipeWire.PreparedPropertyTransaction(transaction.plans, selected)
    @test_throws MethodError FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !recovery_node(graph, :correction).prepared.plan.closed
    @test recovery_status(scope).status === :unknown
    @test recovery_status(scope).closure_inhibited
    @test recovery_status(scope).active_generation == 0
    @test !recovery_status(scope).reference_reached
    @test !FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test_throws InvalidStateException stage_recovery!(scope)
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
end

@testset "Serialized native controls adopt recovery without an input frame" begin
    graph = prepared_recovery_graph()
    node = PipeWireNode(graph; rate=1000//1, connect=false, run_control=true,
        feedback=CALLBACK_FEEDBACK_PAIR,
        recovery=(controller=:correction, participants=(:temporal, :command),
            reference_timeout_ms=60000))
    try
        owner = node.owner
        FilterGraphPipeWire._prepare_owner(owner)
        controller = recovery_node(graph, :correction).prepared
        temporal = recovery_node(graph, :temporal).prepared
        command = recovery_node(graph, :command).prepared
        fill!(controller.workspace.committed_state, 0.75f0)
        fill!(temporal.workspace.committed_state, 0.5f0)
        current = copy(command.workspace.committed_current)
        frames = owner.process_count
        readback() = recovery_property_values(node.filter.callbacks.on_get_properties(node.filter))
        stage_properties!(node, SPA.Props("pipewireao-recovery:request-token" => Int64(1)))
        @test controller.plan.closed
        @test readback()["pipewireao-recovery:state"] == "pending"

        node.filter.callbacks.on_apply_properties(node.filter)
        applied = readback()
        @test applied["pipewireao-recovery:state"] == "applied"
        @test applied["pipewireao-recovery:requested-generation"] ==
            applied["pipewireao-recovery:active-generation"] == 1
        @test !controller.plan.closed
        @test all(iszero, controller.workspace.committed_state)
        @test all(iszero, temporal.workspace.committed_state)
        @test command.workspace.committed_current == current
        @test owner.process_count == frames
        @test !applied["pipewireao-recovery:reference-reached"]
        @test applied["pipewireao-recovery:closure-inhibited"]
        @test_throws InvalidStateException stage_properties!(node,
            SPA.Props("pipewireao-recovery:release-token" => Int64(1)))
        @test_throws InvalidStateException stage_properties!(node,
            SPA.Props("correction:closed" => true))

        # A repeated control callback must not replay the consumed reset.
        fill!(controller.workspace.committed_state, 0.25f0)
        node.filter.callbacks.on_apply_properties(node.filter)
        @test all(==(0.25f0), controller.workspace.committed_state)
        @test readback()["pipewireao-recovery:active-generation"] == 1
    finally
        close(node)
    end
end
