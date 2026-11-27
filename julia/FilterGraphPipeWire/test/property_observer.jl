function observer_values(properties)
    return Dict(name => PipeWireAO.pod_value(
        endswith(name, ":closed") ? Bool :
        endswith(name, "generation") || endswith(name, "sequence") ? Int64 : Float32,
        value) for (name, value) in properties)
end

@testset "serialized scalar Property observer" begin
    events = Tuple{Symbol,SPA.Props}[]
    observer(phase::Val{P}, props::SPA.Props) where {P} = push!(events, (P, props))
    node = PipeWireNode(prepared_control_graph(); connect=false, property_observer=observer)
    callbacks = node.filter.callbacks
    owner = node.owner
    try
        @test first.(events) == [:observed]
        @test observer_values(only(events)[2].values)["closed-loop-correction:closed"] === true
        initial = callbacks.on_get_properties(node.filter)
        @test first.(events) == [:observed, :observed]
        @test last(events)[2] === initial
        empty!(events)
        @test callbacks.on_properties(node.filter,
            SPA.Props("closed-loop-correction:closed" => false)) === nothing
        @test first.(events) == [:observed, :requested]
        snapshot = @atomic :acquire owner.control_snapshot
        pending_snapshot = owner.pending_control_snapshot
        pending_transaction = owner.property_transaction
        property_state, control_state = owner.property_state[], owner.control_state[]
        FilterGraphPipeWire._warm_property_preparation!(owner)
        @test first.(events) == [:observed, :requested]
        @test (@atomic :acquire owner.control_snapshot) === snapshot
        @test owner.pending_control_snapshot === pending_snapshot
        @test owner.property_transaction === pending_transaction
        @test owner.property_state[] == property_state
        @test owner.control_state[] == control_state
        requested = observer_values(last(events)[2].values)
        @test requested["closed-loop-correction:closed"] === false
        @test requested["closed-loop-correction:requested-generation"] == 1
        @test requested["closed-loop-correction:active-generation"] == 0
        # A duplicate/opposite request cannot replace a pending transaction.
        @test_throws FilterGraphAlgorithms.PropertyUpdateBusy callbacks.on_properties(
            node.filter, SPA.Props("closed-loop-correction:closed" => true))
        @test length(events) == 2
        pending = callbacks.on_get_properties(node.filter)
        @test observer_values(pending.values)["closed-loop-correction:closed"] === true
        empty!(events)
        workspace = only(owner.graph.nodes).prepared.workspace
        state_before = deepcopy(map(name -> getfield(workspace, name), fieldnames(typeof(workspace))))
        process_count = owner.process_count
        @test callbacks.on_apply_properties(node.filter) === nothing
        @test owner.process_count == process_count == 0
        @test map(name -> getfield(workspace, name), fieldnames(typeof(workspace))) == state_before
        # A control-only callback is idempotent when no transaction is pending.
        @test callbacks.on_apply_properties(node.filter) === nothing
        @test owner.process_count == 0
        # Adoption and scientific processing do not invoke the observer.
        prepared = only(owner.graph.nodes).prepared
        output = zeros(Float32, 2); state = similar(output)
        FilterGraphAlgorithms.process!(output, state, prepared, ones(Float32, 2), zeros(Float32, 2))
        @test isempty(events)
        applied = callbacks.on_get_properties(node.filter)
        duplicate = callbacks.on_get_properties(node.filter)
        @test length(events) == 2
        @test events[1][2] === applied
        @test events[2][2] === duplicate
        @test applied.values == duplicate.values
        values = observer_values(applied.values)
        @test values["closed-loop-correction:closed"] === false
        @test values["closed-loop-correction:requested-generation"] ==
              values["closed-loop-correction:active-generation"] == 1
    finally
        close(node)
    end

    # Adoption can win the race before requested notification is delivered.
    # The requested facts still describe pre-publication active generations.
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    function immediate_observer(phase::Val{P}, props) where {P}
        if P === :requested
            FilterGraphPipeWire._adopt_pending_properties!(owner)
        end
        push!(events, (P, props))
    end
    immediate = FilterGraphPipeWire.PropertyObserver(immediate_observer)
    empty!(events)
    FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("closed-loop-correction:closed" => false), UInt64(0), immediate)
    @test observer_values(last(events)[2].values)["closed-loop-correction:active-generation"] == 0
    @test only((@atomic :acquire owner.control_snapshot).active_property_generations) == 1

    # An opposite accepted write must first report prior adoption, even when
    # that adoption's native readback event has not yet run.
    empty!(events)
    recorder = FilterGraphPipeWire.PropertyObserver(observer)
    FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("closed-loop-correction:closed" => true), UInt64(0), recorder)
    @test first.(events) == [:observed, :requested]
    applied = observer_values(events[1][2].values)
    desired = observer_values(events[2][2].values)
    @test applied["closed-loop-correction:closed"] === false
    @test applied["closed-loop-correction:active-generation"] == 1
    @test desired["closed-loop-correction:closed"] === true
    @test desired["closed-loop-correction:requested-generation"] == 2
    @test desired["closed-loop-correction:active-generation"] == 1

    failures = Ref(0)
    fail = (phase, props) -> begin
        phase === Val(:requested) || return nothing
        failures[] += 1
        error("observer failed")
    end
    node = PipeWireNode(prepared_control_graph(); connect=false, property_observer=fail)
    try
        callbacks = node.filter.callbacks
        @test_logs (:error, "Property observer disabled after failure") begin
            @test callbacks.on_properties(node.filter,
                SPA.Props("closed-loop-correction:closed" => false)) === nothing
        end
        @test failures[] == 1
        @test callbacks.on_properties.observer.disabled[]
        @test FilterGraphPipeWire._adopt_pending_properties!(node.owner)
        @test observer_values(callbacks.on_get_properties(node.filter).values)["closed-loop-correction:closed"] === false
        @test failures[] == 1
        @test callbacks.on_properties(node.filter,
            SPA.Props("closed-loop-correction:closed" => true)) === nothing
    finally
        close(node)
    end

    # Reject a foreign thread before entering user code; native readback survives.
    node = PipeWireNode(prepared_control_graph(); connect=false, property_observer=fail)
    try
        foreign_observer = node.filter.callbacks.on_get_properties.observer
        foreign_observer.owner_thread += 1
        @test_logs (:error, "Property observer disabled after failure") begin
            @test node.filter.callbacks.on_get_properties(node.filter) isa SPA.Props
        end
        @test failures[] == 1
        @test foreign_observer.disabled[]
    finally
        close(node)
    end

    node = PipeWireNode(prepared_control_graph(); connect=false)
    try
        @test node.filter.callbacks.on_get_properties.observer === nothing
        @test node.filter.callbacks.on_properties.observer === nothing
    finally
        close(node)
    end
end

@testset "Graph processing never invokes the Property observer" begin
    calls = Ref(0)
    node = PipeWireNode(prepared_leaky_graph(); connect=false,
        property_observer=(phase, props) -> (calls[] += 1))
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        callbacks = node.filter.callbacks
        callbacks.on_properties(node.filter, SPA.Props("integrate:closed" => false))
        calls[] = 0
        input = collect(Float32, 1:8)
        output = zeros(Float32, 8)
        native_input = native_callback_buffer(input)
        native_output = native_callback_buffer(output)
        GC.@preserve input output native_input native_output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(1))
            @test callbacks.on_process(node.filter, inputs, outputs) === nothing
            @test calls[] == 0
            callbacks.on_process(node.filter, inputs, outputs)
            @test @allocated(callbacks.on_process(node.filter, inputs, outputs)) == 0
        end
        @test calls[] == 0
        props = callbacks.on_get_properties(node.filter)
        @test observer_values(props.values)["integrate:closed"] === false
        @test calls[] == 1
    finally
        close(node)
    end
end
