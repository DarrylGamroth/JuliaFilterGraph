@algorithm SynchronizedSumF32 begin
    label = "test-synchronized-sum-f32"

    @configuration SynchronizedSumF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:a, Float32, (configuration.extent,), "test.signal.f32/1"),
        input(:b, Float32, (configuration.extent,), "test.signal.f32/1"),
        input(:c, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :sum,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:a,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @inbounds @simd for index in eachindex(sum, a, b, c)
            sum[index] = a[index] + b[index] + c[index]
        end
        nothing
    end
end

const SYNCHRONIZED_SUM_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = add
            label = test-synchronized-sum-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "add:a" "add:b" "add:c" ]
    outputs = [ "add:sum" ]
}
"""

const SYNCHRONIZED_TRANSACTION_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = add
            label = test-synchronized-sum-f32
            config = { extent = 2 }
        }
        {
            type = ndarray
            name = integrate
            label = leaky-integrator-f32
            config = {
                extent = 2
                initial_state = 0.0
                input_schema = test.signal.f32/1
                output_schema = test.signal.f32/1
            }
            props = { gain = 1.0 pole = 0.5 }
        }
        {
            type = ndarray
            name = gate
            label = test-gate-f32
            config = { extent = 2 }
        }
    ]
    links = [
        { output = "add:sum" input = "integrate:input" }
        { output = "integrate:output" input = "gate:input" }
    ]
    inputs = [ "add:a" "add:b" "add:c" ]
    outputs = [ "gate:output" ]
}
"""

prepare_synchronized_sum_graph() = prepare_graph_text(
    SYNCHRONIZED_SUM_GRAPH;
    algorithms=(SynchronizedSumF32,),
)

function process_synchronized_window!(outputs, synchronized, sequence, a, b, c)
    metadata = SampleMetadata(sequence)
    process!(outputs, synchronized, :a, a, metadata)
    process!(outputs, synchronized, :b, b, metadata)
    return process!(outputs, synchronized, :c, c, metadata)
end

@testset "equal-rate all-required synchronized inputs" begin
    graph = prepare_synchronized_sum_graph()
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph)
    @test synchronized isa FilterGraphs.PreparedSynchronizedInputs

    a = Float32[10, 20]
    b_storage = Float32[-1, 1, -1, 2]
    b = view(b_storage, 2:2:4)
    c = Float32[100, 200]
    output = fill(-1.0f0, 2)
    outputs = (sum=output,)
    metadata_a = SampleMetadata(10; timestamp_nanoseconds=100)

    @test process!(outputs, synchronized, :b, b, SampleMetadata(10)) == (sum=false,)
    b .= 99.0f0
    @test process!(outputs, synchronized, :a, a, metadata_a) == (sum=false,)
    @test process!(outputs, synchronized, :c, c, SampleMetadata(10)) == (sum=true,)
    @test output == Float32[111, 222]
    @test FilterGraphs.output_metadata(graph) == (sum=metadata_a,)

    @test process!(outputs, synchronized, :a, a, SampleMetadata(10)) == (sum=false,)
    @test process!(outputs, synchronized, :a, a, SampleMetadata(11)) == (sum=false,)
    @test process!(outputs, synchronized, :a, Float32[90, 90], SampleMetadata(11)) ==
          (sum=false,)

    b .= Float32[3, 4]
    @test process!(outputs, synchronized, :b, b, SampleMetadata(12)) == (sum=false,)
    @test process!(outputs, synchronized, :a, a, SampleMetadata(11)) == (sum=false,)
    @test process!(outputs, synchronized, :a, a, SampleMetadata(12)) == (sum=false,)
    @test process!(outputs, synchronized, :c, c, SampleMetadata(12)) == (sum=true,)
    @test output == Float32[113, 224]
    @test FilterGraphs.output_metadata(graph).sum.discontinuity

    @test process!(
        outputs,
        synchronized,
        :a,
        a,
        SampleMetadata(1; discontinuity=true),
    ) == (sum=false,)
    @test process!(outputs, synchronized, :b, b, SampleMetadata(1)) == (sum=false,)
    @test FilterGraphs.expire!(synchronized)
    @test !FilterGraphs.expire!(synchronized)

    fill!(output, -2.0f0)
    @test process!(
        outputs,
        synchronized,
        :a,
        a,
        SampleMetadata(2; corrupted=true),
    ) == (sum=false,)
    @test process!(outputs, synchronized, :b, b, SampleMetadata(2)) == (sum=false,)
    @test process!(outputs, synchronized, :c, c, SampleMetadata(2)) == (sum=false,)
    @test output == fill(-2.0f0, 2)

    @test process_synchronized_window!(outputs, synchronized, 3, a, b, c) == (sum=true,)
    @test output == Float32[113, 224]

    restart = SampleMetadata(4; discontinuity=true)
    @test process!(outputs, synchronized, :a, a, restart) == (sum=false,)
    @test process!(outputs, synchronized, :b, b, restart) == (sum=false,)
    @test process!(outputs, synchronized, :c, c, restart) == (sum=true,)
    @test output == Float32[113, 224]
    @test FilterGraphs.output_metadata(graph).sum.discontinuity
    @test FilterGraphs.synchronized_input_counters(synchronized) == (
        arrivals=21,
        completed_windows=4,
        expired_windows=2,
        corrupted_windows=1,
        rejected_windows=0,
        aborted_windows=0,
        stale_arrivals=2,
        duplicate_arrivals=1,
        discontinuities=3,
        incoherent_arrivals=0,
        domain_changes=0,
    )

    @test_throws ArgumentError process!(
        outputs,
        synchronized,
        :missing,
        a,
        SampleMetadata(4),
    )
    @test_throws ArgumentError process!(
        outputs,
        synchronized,
        :a,
        Float64[1, 2],
        SampleMetadata(4),
    )
    @test_throws DimensionMismatch process!(
        outputs,
        synchronized,
        :a,
        Float32[1],
        SampleMetadata(4),
    )
    @test_throws ArgumentError process!(
        (wrong=output,),
        synchronized,
        :a,
        a,
        SampleMetadata(4),
    )
    @test_throws MethodError process!(outputs, synchronized, :a, a)

    @test reset!(synchronized) === synchronized
    @test !synchronized.state.active
    @test !synchronized.state.has_last_sequence
end

@testset "synchronized input window preserves per-Node commits" begin
    graph = prepare_graph_text(
        SYNCHRONIZED_TRANSACTION_GRAPH;
        algorithms=(SynchronizedSumF32, LeakyIntegratorF32, TestGateF32),
    )
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph)
    workspace = graph.nodes[2].prepared.workspace
    zero = zeros(Float32, 2)
    output = zeros(Float32, 2)
    outputs = (output=output,)

    @test process_synchronized_window!(
        outputs,
        synchronized,
        1,
        Float32[1, 1],
        zero,
        zero,
    ) == (output=true,)
    @test output == Float32[1, 1]
    @test leaky_integrator_state(workspace) == Float32[1, 1]

    fill!(output, 71.0f0)
    @test process_synchronized_window!(
        outputs,
        synchronized,
        2,
        Float32[-2, 1],
        zero,
        zero,
    ) === TestGateRejected
    @test output == fill(71.0f0, 2)
    @test leaky_integrator_state(workspace) == Float32[-1.5, 1.5]

    @test process_synchronized_window!(
        outputs,
        synchronized,
        3,
        Float32[2, 2],
        zero,
        zero,
    ) == (output=true,)
    @test output == Float32[1.25, 2.75]
    @test leaky_integrator_state(workspace) == output

    fill!(output, 83.0f0)
    committed_before_invalid = copy(leaky_integrator_state(workspace))
    process!(outputs, synchronized, :a, Float32[NaN, 0], SampleMetadata(4))
    process!(outputs, synchronized, :b, zero, SampleMetadata(4))
    @test process!(
        outputs,
        synchronized,
        :c,
        zero,
        SampleMetadata(4),
    ) === LeakyIntegratorNonFiniteInput
    @test output == fill(83.0f0, 2)
    @test leaky_integrator_state(workspace) == committed_before_invalid
    @test !synchronized.state.active

    @test process_synchronized_window!(
        outputs,
        synchronized,
        5,
        Float32[1, 1],
        zero,
        zero,
    ) == (output=true,)
    @test output == Float32[1.625, 2.375]

    committed_before_expiry = copy(leaky_integrator_state(workspace))
    @test process!(outputs, synchronized, :a, Float32[9, 9], SampleMetadata(6)) ==
          (output=false,)
    @test leaky_integrator_state(workspace) == committed_before_expiry
    @test FilterGraphs.expire!(synchronized)
    @test leaky_integrator_state(workspace) == committed_before_expiry

    counters = FilterGraphs.synchronized_input_counters(synchronized)
    @test counters.completed_windows == 3
    @test counters.rejected_windows == 2
    @test counters.aborted_windows == 0
end

@testset "synchronized input preparation rejects unsupported Graphs" begin
    one_input = prepare_graph(
        GRAPH_FIXTURE;
        algorithms=(LeakyIntegratorF32, TestGateF32),
    )
    @test_throws ArgumentError FilterGraphs.prepare_synchronized_inputs(one_input)

    mismatched_rates = prepare_latest_hold_test_graph()
    @test_throws ArgumentError FilterGraphs.prepare_synchronized_inputs(mismatched_rates)

    progressive = prepare_graph(
        CONDITIONAL_GRAPH_FIXTURE;
        algorithms=(PairAccumulatorF32, ConditionalConsumerF32, PreviewF32),
    )
    @test_throws ArgumentError FilterGraphs.prepare_synchronized_inputs(progressive)
end

@testset "synchronized input warmed allocation contract" begin
    synchronized = FilterGraphs.prepare_synchronized_inputs(
        prepare_synchronized_sum_graph(),
    )
    a = Float32[1, 2]
    b = Float32[3, 4]
    c = Float32[5, 6]
    outputs = (sum=zeros(Float32, 2),)
    metadata = ntuple(index -> SampleMetadata(index), 8)

    process_synchronized_window!(outputs, synchronized, 1, a, b, c)
    reset!(synchronized)

    @test @allocated(process!(outputs, synchronized, :a, a, metadata[1])) == 0
    @test @allocated(process!(outputs, synchronized, :a, a, metadata[1])) == 0
    @test @allocated(process!(outputs, synchronized, :b, b, metadata[1])) == 0
    @test @allocated(process!(outputs, synchronized, :c, c, metadata[1])) == 0
    @test @allocated(process!(outputs, synchronized, :a, a, metadata[1])) == 0

    @test @allocated(process!(outputs, synchronized, :a, a, metadata[2])) == 0
    @test @allocated(process!(outputs, synchronized, :b, b, metadata[3])) == 0
    @test @allocated(process!(outputs, synchronized, :a, a, metadata[3])) == 0
    corrupt = SampleMetadata(3; corrupted=true)
    @test @allocated(process!(outputs, synchronized, :c, c, corrupt)) == 0

    @test @allocated(process!(outputs, synchronized, :a, a, metadata[4])) == 0
    @test @allocated(FilterGraphs.expire!(synchronized)) == 0
    @test @allocated(FilterGraphs.expire!(synchronized)) == 0
    @test @allocated(FilterGraphs.synchronized_input_counters(synchronized)) == 0
    @test @allocated(reset!(synchronized)) == 0
end

const SYNCHRONIZED_ACQUISITION_DOMAIN = ntuple(index -> UInt8(index), 16)

function synchronized_acquisition_metadata(sequence; generation=1,
        domain=SYNCHRONIZED_ACQUISITION_DOMAIN, timestamp=1000 + sequence,
        corrupted=false, discontinuity=false)
    return SampleMetadata(sequence; timestamp_nanoseconds=timestamp,
        acquisition_identity=SampleAcquisitionIdentity(domain, generation, sequence),
        corrupted, discontinuity)
end

function process_acquisition_window!(outputs, synchronized, metadata, a, b, c)
    process!(outputs, synchronized, :a, a, metadata)
    process!(outputs, synchronized, :b, b, metadata)
    return process!(outputs, synchronized, :c, c, metadata)
end

@testset "acquisition synchronization requires coherent owned metadata" begin
    graph = prepare_synchronized_sum_graph()
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph; match=:acquisition)
    @test_throws ArgumentError FilterGraphs.prepare_synchronized_inputs(graph; match=:unknown)
    a, b, c = Float32[1, 2], Float32[10, 20], Float32[100, 200]
    outputs = (sum=fill(-1.0f0, 2),)
    metadata = synchronized_acquisition_metadata(10)
    process!(outputs, synchronized, :c, c, metadata)
    process!(outputs, synchronized, :a, a, metadata)
    a .= -10
    @test process!(outputs, synchronized, :b, b, metadata) == (sum=true,)
    @test outputs.sum == Float32[111, 222]
    @test FilterGraphs.output_metadata(graph).sum === metadata
    @test all(slot -> slot.value === metadata, values(synchronized.input_metadata))

    # Physical identity matches independently of publication counters and PTS.
    fill!(outputs.sum, -1)
    metadata = synchronized_acquisition_metadata(11)
    process!(outputs, synchronized, :a, a, SampleMetadata(metadata; sequence=100))
    @test process!(outputs, synchronized, :b, b,
        SampleMetadata(metadata; sequence=200, timestamp_nanoseconds=999)) == (sum=false,)
    @test process!(outputs, synchronized, :c, c,
        SampleMetadata(metadata; sequence=300, timestamp_nanoseconds=nothing)) == (sum=true,)
    @test outputs.sum == a + b + c
    @test FilterGraphs.output_metadata(graph).sum.sequence == 100
    @test sample_acquisition_identity(FilterGraphs.output_metadata(graph).sum).sequence == 11
    # An advancing publication counter cannot make the retained acquisition fresh.
    @test process!(outputs, synchronized, :a, a,
        SampleMetadata(metadata; sequence=400)) == (sum=false,)
    @test FilterGraphs.synchronized_input_counters(synchronized).stale_arrivals == 1

    # Missing physical identity never creates a window; Header PTS is optional.
    @test process!(outputs, synchronized, :a, a, SampleMetadata(12;
        timestamp_nanoseconds=1012)) == (sum=false,)
    @test !synchronized.state.active
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(13; timestamp=nothing), a, b, c) == (sum=true,)
    @test FilterGraphs.output_metadata(graph).sum.discontinuity

    # Payload validity belongs to the Algorithm; the join preserves NaN data.
    a[1] = NaN32
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(14), a, b, c) == (sum=true,)
    @test isnan(outputs.sum[1]) && isnan(a[1])
    @test b == Float32[10, 20] && c == Float32[100, 200]
    @test length(synchronized.input_buffers) == 3
    @test all(buffer -> length(buffer) == 2, values(synchronized.input_buffers))
end

@testset "acquisition generations reset once and never rewind" begin
    graph = prepare_graph_text(SYNCHRONIZED_TRANSACTION_GRAPH;
        algorithms=(SynchronizedSumF32, LeakyIntegratorF32, TestGateF32))
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph; match=:acquisition)
    outputs = (output=zeros(Float32, 2),)
    one, zero = ones(Float32, 2), zeros(Float32, 2)
    workspace = graph.nodes[2].prepared.workspace
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(40), one, zero, zero) == (output=true,)
    @test outputs.output == one
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(41), one, zero, zero) == (output=true,)
    @test outputs.output == fill(1.5f0, 2)

    # A gap abandons only the incomplete frame; it does not reset integration.
    process!(outputs, synchronized, :a, one, synchronized_acquisition_metadata(42))
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(45), one, zero, zero) == (output=true,)
    @test outputs.output == fill(1.75f0, 2)
    @test FilterGraphs.synchronized_input_counters(synchronized).expired_windows == 1

    # Corrupt newer-generation metadata retires an old incomplete frame but
    # cannot reset scientific state or permit that old frame to flush later.
    process!(outputs, synchronized, :a, one, synchronized_acquisition_metadata(46))
    @test process!(outputs, synchronized, :a, one,
        synchronized_acquisition_metadata(1; generation=2, corrupted=true)) == (output=false,)
    @test leaky_integrator_state(workspace) == fill(1.75f0, 2)
    @test synchronized.match.generation == 1
    @test !synchronized.state.active
    @test process!(outputs, synchronized, :b, zero,
        synchronized_acquisition_metadata(46)) == (output=false,)
    @test FilterGraphs.synchronized_input_counters(synchronized).expired_windows == 2

    new_epoch = synchronized_acquisition_metadata(1; generation=2)
    @test process!(outputs, synchronized, :a, one, new_epoch) == (output=false,)
    @test leaky_integrator_state(workspace) == zero
    # Both an old generation and a rewound same-generation discontinuity drop.
    @test process!(outputs, synchronized, :c, zero,
        synchronized_acquisition_metadata(46; discontinuity=true)) == (output=false,)
    @test process!(outputs, synchronized, :b, zero,
        synchronized_acquisition_metadata(0; generation=2, discontinuity=true)) == (output=false,)
    @test process!(outputs, synchronized, :b, zero, new_epoch) == (output=false,)
    @test process!(outputs, synchronized, :c, zero, new_epoch) == (output=true,)
    @test outputs.output == one
    @test FilterGraphs.output_metadata(graph).output.acquisition_identity == new_epoch.acquisition_identity
    @test FilterGraphs.output_metadata(graph).output.discontinuity
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(2; generation=2), one, zero, zero) == (output=true,)
    @test outputs.output == fill(1.5f0, 2)
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(1; generation=2, discontinuity=true),
        one, zero, zero) == (output=false,)
    @test leaky_integrator_state(workspace) == fill(1.5f0, 2)
    invalid = Float32[NaN, 0]
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(3; generation=2), invalid, zero, zero) ===
        LeakyIntegratorNonFiniteInput
    @test leaky_integrator_state(workspace) == fill(1.5f0, 2)
    @test isnan(invalid[1])
end

@testset "explicit acquisition reset preserves sequence admission fences" begin
    synchronized = FilterGraphs.prepare_synchronized_inputs(
        prepare_synchronized_sum_graph(); match=:acquisition)
    input = ones(Float32, 2)
    outputs = (sum=zeros(Float32, 2),)
    first = synchronized_acquisition_metadata(10)
    @test process_acquisition_window!(outputs, synchronized, first,
        input, input, input) == (sum=true,)
    reset!(synchronized)
    @test process_acquisition_window!(outputs, synchronized, first,
        input, input, input) == (sum=false,)
    next = synchronized_acquisition_metadata(11)
    process!(outputs, synchronized, :a, input, next)
    reset!(synchronized)
    @test process_acquisition_window!(outputs, synchronized, next,
        input, input, input) == (sum=false,)
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(1; generation=2), input, input, input) == (sum=true,)
end

@testset "acquisition domain replacement permanently closes admission" begin
    synchronized = FilterGraphs.prepare_synchronized_inputs(
        prepare_synchronized_sum_graph(); match=:acquisition)
    inputs = ones(Float32, 2)
    outputs = (sum=fill(-1.0f0, 2),)
    original = synchronized_acquisition_metadata(1)
    process!(outputs, synchronized, :a, inputs, original)
    other_domain = ntuple(index -> UInt8(20 + index), 16)
    @test process!(outputs, synchronized, :b, inputs,
        synchronized_acquisition_metadata(1; domain=other_domain)) == (sum=false,)
    @test process_acquisition_window!(outputs, synchronized, original,
        inputs, inputs, inputs) == (sum=false,)
    reset!(synchronized)
    @test process_acquisition_window!(outputs, synchronized,
        synchronized_acquisition_metadata(2), inputs, inputs, inputs) == (sum=false,)
    @test outputs.sum == fill(-1.0f0, 2)
    @test FilterGraphs.synchronized_input_counters(synchronized).domain_changes == 1
end

function acquisition_synchronization_allocations()
    owner = FilterGraphs.prepare_synchronized_inputs(
        prepare_synchronized_sum_graph(); match=:acquisition)
    outputs = (sum=zeros(Float32, 2),)
    input = ones(Float32, 2)
    first = synchronized_acquisition_metadata(1)
    next = synchronized_acquisition_metadata(2)
    reset = synchronized_acquisition_metadata(1; generation=2)
    stale = synchronized_acquisition_metadata(3)
    second = synchronized_acquisition_metadata(2; generation=2)
    mismatch = synchronized_acquisition_metadata(2; generation=2, timestamp=99)
    missing = SampleMetadata(3)
    rewind = synchronized_acquisition_metadata(0; generation=2, discontinuity=true)
    discontinuous = synchronized_acquisition_metadata(3; generation=2, discontinuity=true)
    corrupt = synchronized_acquisition_metadata(4; generation=2, corrupted=true)
    changed = synchronized_acquisition_metadata(5; generation=2,
        domain=ntuple(index -> UInt8(20 + index), 16))
    return (
        complete=@allocated(process_acquisition_window!(outputs, owner, first, input, input, input)),
        arrival=@allocated(process!(outputs, owner, :a, input, next)),
        duplicate=@allocated(process!(outputs, owner, :a, input, next)),
        generation=@allocated(process!(outputs, owner, :b, input, reset)),
        stale=@allocated(process!(outputs, owner, :a, input, stale)),
        reset_a=@allocated(process!(outputs, owner, :a, input, reset)),
        reset_c=@allocated(process!(outputs, owner, :c, input, reset)),
        rewind=@allocated(process!(outputs, owner, :a, input, rewind)),
        next=@allocated(process!(outputs, owner, :a, input, second)),
        mismatch=@allocated(process!(outputs, owner, :b, input, mismatch)),
        missing=@allocated(process!(outputs, owner, :a, input, missing)),
        discontinuity_a=@allocated(process!(outputs, owner, :a, input, discontinuous)),
        discontinuity_b=@allocated(process!(outputs, owner, :b, input, discontinuous)),
        discontinuity_c=@allocated(process!(outputs, owner, :c, input, discontinuous)),
        corruption=@allocated(process_acquisition_window!(outputs, owner, corrupt, input, input, input)),
        domain_change=@allocated(process!(outputs, owner, :a, input, changed)),
        domain_closed=@allocated(process!(outputs, owner, :a, input, second)),
    )
end

@testset "acquisition synchronization allocates no steady-state memory" begin
    acquisition_synchronization_allocations()
    @test all(iszero, values(acquisition_synchronization_allocations()))
end
