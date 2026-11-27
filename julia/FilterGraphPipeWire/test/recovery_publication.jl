# Model a native consumer's cached Props: only a notification refreshes the cache.
mutable struct RecoveryPropertyCache
    filter::Any
    properties::Dict{String,Any}
    notifications::Int
end
PipeWireAO.filter_state(::RecoveryPropertyCache) = PipeWireAO.NDARRAY_FILTER_STATE_PAUSED
function PipeWireAO.notify_properties!(cache::RecoveryPropertyCache)
    cache.properties = recovery_property_values(
        cache.filter.callbacks.on_get_properties(cache.filter))
    cache.notifications += 1
    return nothing
end

function recovery_publication_frame!(owner,filter,storage)
    GC.@preserve storage begin
        FilterGraphPipeWire._process_graph_callback!(owner,filter,storage[][5],storage[][6])
    end
    return nothing
end

function recovery_publication_fixture()
    node = Base.invokelatest(PipeWireNode,Base.invokelatest(prepared_recovery_graph); rate=1000//1, connect=false,
        feedback=CALLBACK_FEEDBACK_PAIR,
        recovery=(controller=:correction, participants=(:temporal, :command),
            reference_timeout_ms=60000))
    FilterGraphPipeWire._prepare_owner(node.owner)
    cache = RecoveryPropertyCache(node.filter, Dict{String,Any}(), 0)
    residual, demanded = zeros(Float32, 1), zeros(Float32, 1)
    input = native_callback_buffer(residual; header=BufferHeader(
        SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(1), Int64(0), UInt64(1)))
    output = native_callback_buffer(demanded; header_available=true)
    storage=Ref((residual,demanded,input,output,
        NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input)),UInt32(1)),
        NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output)),UInt32(1))))
    owner=Ref{Any}(node.owner) # This cold cache must not capture the full Node type twice.
    frame!(filter=cache)=Base.invokelatest(recovery_publication_frame!,owner[],filter,storage)
    # Compile the cold notification sink and callback before starting the
    # reference timeout. This disconnected zero-input warmup is not evidence.
    PipeWireAO.notify_properties!(cache)
    frame!()
    stage_properties!(node, SPA.Props("pipewireao-recovery:request-token" => Int64(1)))
    for _ in 1:8
        frame!()
    end
    return Base.inferencebarrier(node),cache,frame!
end

@testset "Stable reference evidence refreshes native cached Props" begin
    node, cache, frame! = recovery_publication_fixture()
    try
        @test cache.properties["pipewireao-recovery:reference-reached"]
        sequence = cache.properties["pipewireao-recovery:observation-sequence"]
        # Same actual callback and same true reference result; no control write
        # or Boolean transition can cause the required cache refresh.
        sleep(0.11)
        frame!()
        @test cache.properties["pipewireao-recovery:reference-reached"]
        @test cache.properties["pipewireao-recovery:observation-sequence"] > sequence
        observation = FilterGraphPipeWire._recovery_observation(node.owner)
        @test cache.properties["pipewireao-recovery:observation-sequence"] == observation.sequence
        @test cache.properties["pipewireao-recovery:observation-monotonic-nanoseconds"] == observation.timestamp_ns
    finally
        close(node)
    end
end

# A notification sink without getter allocations checks the processing-side work.
mutable struct RecoveryNotificationCounter
    notifications::Int
end
PipeWireAO.filter_state(::RecoveryNotificationCounter) = PipeWireAO.NDARRAY_FILTER_STATE_PAUSED
function PipeWireAO.notify_properties!(counter::RecoveryNotificationCounter)
    counter.notifications += 1
    return nothing
end

@testset "Recovery publications coalesce without changing scientific evidence" begin
    node, cache, frame! = recovery_publication_fixture()
    owner = node.owner
    counter = RecoveryNotificationCounter(0)
    notify!(changed, now_ns) = FilterGraphPipeWire._notify_graph_properties!(owner, counter, changed, now_ns)
    try
        @test FilterGraphPipeWire._recovery_properties_interval(owner) == 100_000_000
        allocation_counter = RecoveryNotificationCounter(0)
        frame!(allocation_counter) # warm the full active callback specialization
        frame!(allocation_counter)
        owner.recovery_properties_notified_timestamp_ns -= UInt64(100_000_000)
        notifications = allocation_counter.notifications
        @test @allocated(frame!(allocation_counter)) == 0
        @test allocation_counter.notifications == notifications + 1
        now_ns = owner.recovery_properties_notified_timestamp_ns
        observation = FilterGraphPipeWire._recovery_observation(owner)
        FilterGraphPipeWire._publish_recovery_observation!(owner, observation.generation,
            observation.sequence + UInt64(1), true)
        evidence = FilterGraphPipeWire._recovery_observation(owner)
        notify!(false, now_ns + UInt64(99_999_999))
        @test counter.notifications == 0
        notify!(false, now_ns + UInt64(100_000_000))
        @test counter.notifications == 1
        notify!(false, now_ns + UInt64(200_000_000))
        @test counter.notifications == 1 # no new successful observation
        @test FilterGraphPipeWire._recovery_observation(owner) == evidence
        @test @allocated(notify!(false, now_ns + UInt64(200_000_000))) == 0

        # The minimum configured timeout is 1 ms: quarter-timeout publication
        # gives 250 us, without weakening the age check or per-frame publication.
        owner.recovery_reference_timeout_ns[] = UInt64(1_000_000)
        @test FilterGraphPipeWire._recovery_properties_interval(owner) == 250_000
        FilterGraphPipeWire._publish_recovery_observation!(owner, observation.generation,
            observation.sequence + UInt64(2), true)
        notify!(false, now_ns + UInt64(100_249_999))
        @test counter.notifications == 1
        notify!(false, now_ns + UInt64(100_250_000))
        @test counter.notifications == 2
        owner.recovery_reference_timeout_ns[] = UInt64(60_000_000_000)

        # Negative-frame invalidation retains its sequence and still publishes
        # immediately through the callback's existing Boolean-change path.
        FilterGraphPipeWire._invalidate_recovery_observation!(owner)
        notify!(true, now_ns + UInt64(100_250_001))
        @test counter.notifications == 3
        @test !FilterGraphPipeWire._recovery_observation(owner).reached
        @test @allocated(notify!(true, now_ns + UInt64(100_250_002))) == 0

        # Restore a real successful observation before release. Publication is
        # still needed through the first guarded close, and stops afterwards.
        owner.recovery_properties_notified_timestamp_ns = time_ns()
        frame!()
        stage_properties!(node, SPA.Props("pipewireao-recovery:release-token" => Int64(1)))
        @test !iszero(FilterGraphPipeWire._recovery_properties_interval(owner))
        frame!()
        @test !iszero(FilterGraphPipeWire._recovery_properties_interval(owner))
        stage_properties!(node, SPA.Props("correction:closed" => true))
        frame!()
        @test iszero(FilterGraphPipeWire._recovery_properties_interval(owner))
        observation = FilterGraphPipeWire._recovery_observation(owner)
        frame!()
        @test FilterGraphPipeWire._recovery_observation(owner) == observation
        notifications = cache.notifications
        sleep(0.11)
        frame!()
        @test cache.notifications == notifications
    finally
        close(node)
    end
end
