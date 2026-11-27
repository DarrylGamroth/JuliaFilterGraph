# The trait alone must not admit old or incomplete adapters. Keep these legacy
# methods executable so an accidental tagged-to-legacy bridge is observable.
struct TestCaptureProtocolDevice{Kind} end
const TEST_LEGACY_PREPARATIONS = Ref(0)
const TEST_PROTOCOL_WORK = Ref(0)
const TEST_PROTOCOL_WORKGROUPS = Ref{Union{Nothing,CapturedWorkgroupProfile}}(nothing)

FilterGraphs._prepare_captured_context(::TestCaptureProtocolDevice) =
    FilterGraphs._prepare_captured_context(TestCapturedDevice())

for algorithm in (FastPassF32, CapturedUnsupportedPassF32)
    @eval FilterGraphAlgorithms.capture_support(
        ::Type{$algorithm}, ::TestCaptureProtocolDevice,
    ) = SupportsCapture()
end

function FilterGraphAlgorithms.prepare_capture(
        ::CompleteTransition, ::TestCaptureProtocolDevice,
        prepared::FilterGraphAlgorithms.PreparedAlgorithm{FastPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer)
    return FilterGraphAlgorithms.prepare_capture(
        CompleteTransition(), TestCapturedDevice(), prepared, outputs, inputs, status, index)
end

mutable struct TestLegacyCandidateAdapter
    candidate::Int
    committed::Int
end
FilterGraphAlgorithms.enqueue!(adapter::TestLegacyCandidateAdapter) =
    (TEST_PROTOCOL_WORK[] += 1; adapter.candidate += 1; nothing)
legacy_test_adopt!(adapter::TestLegacyCandidateAdapter) =
    (adapter.committed = adapter.candidate; nothing)

struct TestIncompleteCapturedAdapter{Missing} end
for missing in (:enqueue!, :preflight, :complete!, :reset!)
    for hook in (:enqueue!, :preflight, :complete!, :reset!)
        missing == hook && continue
        arguments = Any[:(adapter::TestIncompleteCapturedAdapter{$(QuoteNode(missing))})]
        hook == :complete! && push!(arguments, :(::Int32))
        @eval function FilterGraphAlgorithms.$hook($(arguments...))
            TEST_PROTOCOL_WORK[] += 1
            return nothing
        end
    end
end

function FilterGraphAlgorithms.prepare_capture(
        ::CompleteTransition, ::TestCaptureProtocolDevice{Missing},
        ::FilterGraphAlgorithms.PreparedAlgorithm{CapturedUnsupportedPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer,
    ) where {Missing}
    return TestIncompleteCapturedAdapter{Missing}()
end

# The selected optional-workgroup method must win over the tagged forwarder.
function FilterGraphAlgorithms.prepare_capture(
        ::CompleteTransition, ::TestCaptureProtocolDevice{:workgroups},
        ::FilterGraphAlgorithms.PreparedAlgorithm{CapturedUnsupportedPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer, workgroups)
    TEST_PROTOCOL_WORKGROUPS[] = workgroups
    return TestCapturedPass(only(outputs), only(inputs), Int32(index), 0)
end

struct TestLegacyOnlyDevice{Stateful} end
FilterGraphs._prepare_captured_context(::TestLegacyOnlyDevice) =
    FilterGraphs._prepare_captured_context(TestCapturedDevice())
FilterGraphAlgorithms.capture_support(::Type{FastPassF32}, ::TestLegacyOnlyDevice) = SupportsCapture()
function FilterGraphAlgorithms.prepare_capture(
        ::TestLegacyOnlyDevice{Stateful}, ::FilterGraphAlgorithms.PreparedAlgorithm{FastPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer) where {Stateful}
    TEST_LEGACY_PREPARATIONS[] += 1
    return Stateful ? TestLegacyCandidateAdapter(0, 0) :
        TestCapturedPass(only(outputs), only(inputs), Int32(index), 0)
end
function FilterGraphAlgorithms.prepare_capture(
        ::TestLegacyOnlyDevice{Stateful}, ::FilterGraphAlgorithms.PreparedAlgorithm{FastPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer, workgroups) where {Stateful}
    TEST_LEGACY_PREPARATIONS[] += 1
    return Stateful ? TestLegacyCandidateAdapter(0, 0) :
        TestCapturedPass(only(outputs), only(inputs), Int32(index), 0)
end

struct TestMissingCaptureDevice end
FilterGraphs._prepare_captured_context(::TestMissingCaptureDevice) =
    FilterGraphs._prepare_captured_context(TestCapturedDevice())
FilterGraphAlgorithms.capture_support(::Type{FastPassF32}, ::TestMissingCaptureDevice) = SupportsCapture()

struct TestCaptureRefinementDevice end
FilterGraphs._prepare_captured_context(::TestCaptureRefinementDevice) =
    FilterGraphs._prepare_captured_context(TestCapturedDevice())
FilterGraphAlgorithms.capture_support(::Type{FastPassF32}, ::TestCaptureRefinementDevice) = SupportsCapture()
function FilterGraphAlgorithms.prepare_capture(
        ::CompleteTransition, ::TestCaptureRefinementDevice, prepared::FilterGraphAlgorithms.PreparedAlgorithm{FastPassF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer)
    return FilterGraphAlgorithms.prepare_capture(
        CompleteTransition(), TestCapturedDevice(), prepared, outputs, inputs, status, Int32(index))
end
function FilterGraphs._prepare_captured_validation(
        ::TestCaptureRefinementDevice, nodes, routes, outputs, inputs, node_outputs)
    return (FilterGraphs._captured_node_with_adapter(
        only(nodes), TestIncompleteCapturedAdapter{:enqueue!}()),)
end

@testset "captured protocol rejects legacy and missing methods before execution" begin
    pass = prepare_graph_text(CAPTURED_PASS_GRAPH; algorithms=(FastPassF32,))
    partial = prepare_graph_text(CAPTURED_PARTLY_SUPPORTED_GRAPH;
        algorithms=(FastPassF32, CapturedUnsupportedPassF32))
    for (graph, device, node, algorithm, reason) in (
        (pass, TestLegacyOnlyDevice{true}(), "pass", "fast-pass-f32", "CompleteTransition()"),
        (pass, TestLegacyOnlyDevice{false}(), "pass", "fast-pass-f32", "CompleteTransition()"),
        (pass, TestMissingCaptureDevice(), "pass", "fast-pass-f32", "CompleteTransition()"),
        (pass, TestCaptureRefinementDevice(), "pass", "fast-pass-f32", "missing enqueue!"),
        ((partial, TestCaptureProtocolDevice{hook}(), "gate", "captured-unsupported-pass-f32", "missing $hook")
            for hook in (:enqueue!, :preflight, :complete!, :reset!))...,
    )
        clear_test_capture_events!()
        records_before = TEST_CAPTURE_RECORD_COUNT[]
        TEST_LEGACY_PREPARATIONS[] = 0
        TEST_PROTOCOL_WORK[] = 0
        error = try
            prepare_captured_graph(graph, device)
            nothing
        catch caught
            caught
        end
        @test error isa GraphConfigurationError
        message = sprint(showerror, error)
        @test occursin("Node '$node'", message)
        @test occursin("Algorithm '$algorithm'", message)
        @test occursin(string(typeof(device)), message)
        @test occursin(reason, message)
        @test occursin("prepare_capture(CompleteTransition()", message)
        @test occursin("complete", message)
        @test TEST_LEGACY_PREPARATIONS[] == 0
        @test TEST_PROTOCOL_WORK[] == 0
        @test TEST_CAPTURE_EVENT_COUNT[] == 0
        @test TEST_CAPTURE_RECORD_COUNT[] == records_before
    end

    prepared = pass.nodes[1].prepared
    input = zeros(Float32, 3)
    output = fill(71.0f0, 3)
    status = zeros(Int32, 1)
    for tag in (Val(1), Val(2), Val(3), Val(:unknown)), workgroups in ((), (CapturedWorkgroupProfile(),))
        @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(
            tag, TestLegacyOnlyDevice{true}(), prepared, (output,), (input,), status, Int32(1), workgroups...)
    end
    @test TEST_LEGACY_PREPARATIONS[] == 0
    @test output == fill(71.0f0, 3)
    @test status == Int32[0]

    profile = CapturedWorkgroupProfile(projection_lanes=32)
    captured = prepare_captured_graph(partial, TestCaptureProtocolDevice{:workgroups}(); workgroups=profile)
    @test TEST_PROTOCOL_WORKGROUPS[] === profile
    @test process!((; output), captured, (input=Float32[1, 2, 3],)) == (output=true,)
    @test output == Float32[1, 2, 3]
    @test isempty(Test.detect_ambiguities(FilterGraphAlgorithms))
end

# Wrap the actual delay workspace to detect omitted or duplicated commitment
# through its non-idempotent ring position, including across wraparound.
struct TestCapturedFrameDelay{O,I,S,W,P}
    output::O
    input::I
    status::S
    workspace::W
    plan::P
    node_index::Int32
end
FilterGraphAlgorithms.capture_support(::Type{FrameDelayF32}, ::TestCapturedDevice) = SupportsCapture()
function FilterGraphAlgorithms.prepare_capture(
        ::CompleteTransition, ::TestCapturedDevice, prepared::FilterGraphAlgorithms.PreparedAlgorithm{FrameDelayF32},
        outputs::Tuple, inputs::Tuple, status, index::Integer)
    return TestCapturedFrameDelay(only(outputs), only(inputs), status,
        deepcopy(prepared.workspace), prepared.plan, Int32(index))
end
FilterGraphAlgorithms.preflight(::TestCapturedFrameDelay) = nothing
function FilterGraphAlgorithms.enqueue!(adapter::TestCapturedFrameDelay)
    iszero(adapter.status[1]) || return nothing
    result = FilterGraphAlgorithms._process_frame_delay!(
        adapter.output, adapter.workspace, adapter.plan, adapter.input)
    if isnothing(result)
        FilterGraphAlgorithms.commit!(adapter.workspace)
    else
        adapter.status[1] = adapter.node_index
    end
    return nothing
end
FilterGraphAlgorithms.complete!(adapter::TestCapturedFrameDelay, status::Int32) =
    status == adapter.node_index ? FrameDelayNonFiniteInput : nothing
FilterGraphAlgorithms.reset!(adapter::TestCapturedFrameDelay) =
    (reset!(adapter.workspace); fill!(adapter.output, 0.0f0); nothing)

@testset "captured delay commits once per successful local transition" begin
    text = """
        filter.graph = {
            nodes = [ { type = ndarray name = delay label = frame-delay-f32
                config = { extent = 1 delay_frames = 3 initial_value = -1.0
                    input_schema = test.signal.f32/1 output_schema = test.signal.f32/1 } } ]
            inputs = [ "delay:input" ]
            outputs = [ "delay:output" ]
        }
        """
    graph = prepare_graph_text(text; algorithms=(FrameDelayF32,))
    captured = prepare_captured_graph(graph, TestCapturedDevice())
    adapter = only(captured.nodes).adapter
    history = frame_delay_history(adapter.workspace)
    output = zeros(Float32, 1)
    for frame in 1:8
        @test process!((; output), captured, (input=Float32[frame],)) == (output=true,)
        @test output == Float32[frame <= 3 ? -1 : frame - 3]
        @test frame_delay_head(adapter.workspace) == mod1(frame + 1, 3)
    end
    before = copy(history)
    @test process!((; output), captured, (input=Float32[NaN],)) === FrameDelayNonFiniteInput
    @test history == before
    @test frame_delay_head(adapter.workspace) == 3
    @test reset!(captured) === captured
    @test frame_delay_history(adapter.workspace) === history
    @test history == fill(-1.0f0, 1, 3)
    @test frame_delay_head(adapter.workspace) == 1

    # Direct backend users receive the same complete transition from enqueue!.
    direct_input, direct_output, direct_status = Float32[1], zeros(Float32, 1), zeros(Int32, 1)
    direct = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), TestCapturedDevice(), graph.nodes[1].prepared,
        (direct_output,), (direct_input,), direct_status, Int32(1), CapturedWorkgroupProfile())
    for frame in 1:5
        direct_input[1] = Float32(frame)
        FilterGraphAlgorithms.enqueue!(direct)
        @test direct_output == Float32[frame <= 3 ? -1 : frame - 3]
        @test frame_delay_head(direct.workspace) == mod1(frame + 1, 3)
    end
end
