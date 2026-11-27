status_info(name="application:ready", value=false; kwargs...) =
    SPA.PropInfo(name, Pod(value); flags=SPA.PROPERTY_READONLY, params=true, kwargs...)

@testset "read-only application Property publication" begin
    observed = Ref{Union{Nothing,SPA.Props}}(nothing)
    getter_calls = Ref(0)
    available = Bool[]
    function observer(phase, props)
        phase === Val(:observed) && (observed[] = props)
        @test all(pair -> pair.first != "application:ready", props.values)
    end
    function getter(props, observer_available)
        getter_calls[] += 1
        push!(available, observer_available)
        @test props === observed[]
        values = observer_values(props.values)
        ready = values["closed-loop-correction:requested-generation"] ==
                values["closed-loop-correction:active-generation"]
        return SPA.Props("application:ready" => ready)
    end
    node = PipeWireNode(prepared_control_graph(); connect=false,
        property_observer=observer, property_info=[status_info()], property_getter=getter)
    try
        @test getter_calls[] == 1 # Also publishes the cold, disconnected snapshot.
        @test all(isconcretetype, fieldtypes(typeof(node.filter.callbacks.on_get_properties.getter)))
        callbacks = node.filter.callbacks
        initial = callbacks.on_get_properties(node.filter)
        @test pod_value(Bool, last(initial.values).second)
        @test length(initial.values) == length(observed[].values) + 1
        @test callbacks.on_get_properties.getter.declarations == (("application:ready" => PipeWireAO.LibPipeWire.SPA_TYPE_Bool),)
        before = @atomic :acquire node.owner.control_snapshot
        @test_throws ArgumentError callbacks.on_properties(node.filter,
            SPA.Props("closed-loop-correction:closed" => false, "application:ready" => true))
        @test (@atomic :acquire node.owner.control_snapshot) === before
        @test node.owner.control_state[] == FilterGraphPipeWire._CONTROL_SLOT_AVAILABLE
        callbacks.on_properties(node.filter, SPA.Props("closed-loop-correction:closed" => false))
        @test getter_calls[] == 2 # Requested events carry Graph facts only.
        @test !pod_value(Bool, last(callbacks.on_get_properties(node.filter).values).second)
        @test FilterGraphPipeWire._adopt_pending_properties!(node.owner)
        @test getter_calls[] == 3
        @test pod_value(Bool, last(callbacks.on_get_properties(node.filter).values).second)
        @test all(available)
        # The getter must never enter user code on a foreign thread.
        @test callbacks.on_get_properties.getter.owner_thread == Threads.threadid()
        foreign = FilterGraphPipeWire.PropertyGetter(getter,
            callbacks.on_get_properties.getter.declarations, Threads.threadid() + 1)
        calls = getter_calls[]
        @test_throws InvalidStateException FilterGraphPipeWire._get_extra_properties(
            foreign, callbacks.on_get_properties.observer, observed[])
        @test getter_calls[] == calls
    finally
        close(node)
    end
end

@testset "Property getter declaration and result validation" begin
    graph = prepared_control_graph()
    getter = (props, available) -> SPA.Props("application:ready" => available)
    @test_throws ArgumentError PipeWireNode(graph; connect=false, property_info=[status_info()])
    @test_throws ArgumentError PipeWireNode(graph; connect=false, property_getter=getter)
    for info in (
        SPA.PropInfo("application:ready", Pod(false)),
        status_info("closed-loop-correction:closed"),
        status_info(FilterGraphPipeWire._CHANNELS_PROPERTY),
        status_info("pipewireao.run-control"),
        status_info("pipewireao.run-control.actual-state"),
        status_info("pipewireao.reset-control.request-token"),
        status_info("application:ready", SPA.Array(Int32[1, 2])),
        status_info("application:ready", false; container=PipeWireAO.LibPipeWire.SPA_TYPE_Array),
    )
        @test_throws ArgumentError PipeWireNode(graph; connect=false,
            property_info=[info], property_getter=getter)
    end
    @test_throws ArgumentError PipeWireNode(graph; connect=false,
        property_info=[status_info(), status_info()], property_getter=getter)
    for result in (
        SPA.Props(), SPA.Props("other" => false),
        SPA.Props("application:ready" => Int32(1)),
        SPA.Props("application:ready" => false, "application:ready" => true),
        nothing,
    )
        @test_throws ArgumentError PipeWireNode(graph; connect=false,
            property_info=[status_info()], property_getter=(props, available) -> result)
    end
    @test_throws ErrorException PipeWireNode(graph; connect=false,
        property_info=[status_info()], property_getter=(props, available) -> error("cold getter failed"))
    # Subsequent construction retains the default property behavior.
    node = PipeWireNode(graph; connect=false, property_info=SPA.PropInfo[], property_getter=nothing)
    try
        @test node.filter.callbacks.on_get_properties.getter === nothing
        @test node.filter.callbacks.on_properties.getter === nothing
    finally
        close(node)
    end
end

@testset "Property getter observes observer failure and propagates its own failure" begin
    available = Bool[]
    observer = (phase, props) -> error("observer failed")
    getter = (props, observer_available) -> begin
        push!(available, observer_available)
        SPA.Props("application:ready" => observer_available)
    end
    node = @test_logs (:error, "Property observer disabled after failure") PipeWireNode(
        prepared_control_graph(); connect=false, property_observer=observer,
        property_info=[status_info()], property_getter=getter)
    try
        @test available == [false]
        @test !pod_value(Bool, last(node.filter.callbacks.on_get_properties(node.filter).values).second)
        @test available == [false, false]
    finally
        close(node)
    end
    failing = Ref(false)
    node = PipeWireNode(prepared_control_graph(); connect=false,
        property_info=[status_info()], property_getter=(props, available) -> begin
            failing[] && error("getter failed")
            SPA.Props("application:ready" => true)
        end)
    try
        failing[] = true
        @test_throws ErrorException node.filter.callbacks.on_get_properties(node.filter)
        destination = Ref{Ptr{PipeWireAO.LibPipeWire.spa_pod}}(C_NULL)
        GC.@preserve destination begin
            @test PipeWireAO._ndarray_filter_get_props(node.filter,
                Base.unsafe_convert(Ptr{Ptr{PipeWireAO.LibPipeWire.spa_pod}}, destination)) < 0
        end
        @test node.filter.callback_error[] isa ErrorException
        @test destination[] == C_NULL
    finally
        close(node)
    end
end

@testset "Graph processing never invokes the Property getter" begin
    calls = Ref(0)
    node = PipeWireNode(prepared_leaky_graph(); connect=false,
        property_info=[status_info()], property_getter=(props, available) -> begin
            calls[] += 1
            SPA.Props("application:ready" => available)
        end)
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        callbacks = node.filter.callbacks
        calls[] = 0
        input = collect(Float32, 1:8)
        output = zeros(Float32, 8)
        native_input = native_callback_buffer(input)
        native_output = native_callback_buffer(output)
        GC.@preserve input output native_input native_output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(1))
            callbacks.on_process(node.filter, inputs, outputs)
            @test @allocated(callbacks.on_process(node.filter, inputs, outputs)) == 0
        end
        @test calls[] == 0
    finally
        close(node)
    end
end

@testset "Joined Graph status publication" begin
    joins = (
        FilterGraphs.prepare_latest_hold(prepared_latest_hold_graph(), :primary, :held),
        FilterGraphs.prepare_synchronized_inputs(prepared_synchronized_graph()),
    )
    for join in joins
        node = PipeWireNode(join; connect=false, rate=(1_000, 1), property_info=[status_info()],
            property_getter=(props, available) -> SPA.Props("application:ready" => available))
        try
            @test pod_value(Bool, last(node.filter.callbacks.on_get_properties(node.filter).values).second)
        finally
            close(node)
        end
    end
end

@testset "Extra Properties retain exact SPA scalar types" begin
    values = (
        "application:ready" => true,
        "application:state" => SPA.Id(UInt32(2)),
        "application:count" => Int32(3),
        "application:sequence" => Int64(4),
        "application:fraction" => 0.5f0,
        "application:elapsed" => 0.25,
        "application:message" => "ready",
    )
    infos = [status_info(name, value) for (name, value) in values]
    # Scalar Choice metadata describes the type; readback remains a scalar.
    infos[3] = status_info("application:count", SPA.Choice(SPA.CHOICE_RANGE, Int32[3, 0, 9]))
    node = PipeWireNode(prepared_control_graph(); connect=false, property_info=infos,
        property_getter=(props, available) -> SPA.Props(reverse(values)))
    try
        extra = node.filter.callbacks.on_get_properties(node.filter).values[end-6:end]
        @test extra == SPA.Props(reverse(values)).values
    finally
        close(node)
    end
    @test_throws ArgumentError PipeWireNode(prepared_control_graph(); connect=false,
        property_info=[status_info(), status_info("application:available")],
        property_getter=(props, available) -> SPA.Props(
            "application:ready" => false, "application:ready" => true))
end
