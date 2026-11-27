@testset "Recovery reference proof expires without new successful frames" begin
    graph = prepared_recovery_graph()
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing; feedback=CALLBACK_FEEDBACK_PAIR)
    @test_throws ArgumentError prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command), reference_timeout_ms=0)
    @test_throws ArgumentError prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command), reference_timeout_ms=60001)
    scope = prepare_recovery(owner; controller=:correction,
        participants=(:temporal, :command), reference_timeout_ms=60000)
    generation = stage_recovery!(scope)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    # Establish actual committed state/reference agreement before injecting age.
    for _ in 1:8
        outputs = (demanded=zeros(Float32, 1),
            var"controller-constraint-feedback"=owner.feedback.scratch)
        inputs = (var"residual-error"=zeros(Float32, 1),
            var"constraint-feedback"=owner.feedback.slot)
        @test all(values(FilterGraphs.process!(outputs, graph, inputs)))
        copyto!(owner.feedback.slot, owner.feedback.scratch)
        FilterGraphPipeWire._observe_recovery_reference!(owner)
        owner.process_count += UInt64(1)
    end
    # Deterministic monotonic-clock injection exercises exact age boundaries;
    # the observation seqlock remains the same one written by successful frames.
    FilterGraphPipeWire._publish_recovery_observation!(owner, generation, UInt64(17), true)
    observation = FilterGraphPipeWire._recovery_observation(owner)
    timestamp = observation.timestamp_ns
    @test FilterGraphPipeWire._recovery_status(scope, timestamp).reference_reached
    @test FilterGraphPipeWire._recovery_status(scope,
        timestamp + scope.reference_timeout_ns).reference_reached
    @test !FilterGraphPipeWire._recovery_status(scope,
        timestamp + scope.reference_timeout_ns + UInt64(1)).reference_reached
    expired = time_ns() - scope.reference_timeout_ns - UInt64(1)
    owner.recovery_observation_timestamp_ns[] = expired
    @test !recovery_status(scope).reference_reached
    @test recovery_status(scope).observation_sequence == 17
    @test recovery_status(scope).observation_monotonic_nanoseconds == expired
    @test_throws InvalidStateException stage_recovery_release!(scope, generation)
    @test recovery_status(scope).closure_inhibited

    FilterGraphPipeWire._publish_recovery_observation!(owner, generation, UInt64(18), true)
    @test stage_recovery_release!(scope, generation) == generation
    owner.recovery_observation_timestamp_ns[] = expired
    @test FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test recovery_status(scope).status === :release_refused
    @test recovery_status(scope).closure_inhibited
    @test !FilterGraphPipeWire._adopt_pending_recovery!(owner)
    # A later explicit release can use a newly successful observation; the old
    # refused release is never replayed when freshness returns.
    FilterGraphPipeWire._publish_recovery_observation!(owner, generation, UInt64(19), true)
    @test !FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test recovery_status(scope).closure_inhibited
    stage_recovery_release!(scope, generation)
    @test FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !recovery_status(scope).closure_inhibited
end
