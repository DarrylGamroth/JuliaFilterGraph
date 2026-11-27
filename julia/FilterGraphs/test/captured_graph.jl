@algorithm CapturedUnsupportedPassF32 begin
    label = "captured-unsupported-pass-f32"

    @configuration CapturedUnsupportedPassF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            rate=(2, 1),
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
            rate=(2, 1),
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = begin
        copyto!(output, input)
        nothing
    end
end

@algorithm CapturedSplitF32 begin
    label = "captured-split-f32"

    @configuration CapturedSplitF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :first,
            Float32,
            (configuration.extent,),
            "test.first.f32/1";
            metadata_from=:input,
        ),
        output(
            :second,
            Float32,
            (configuration.extent,),
            "test.second.f32/1";
            metadata_from=:input,
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = begin
        copyto!(first, input)
        for index in eachindex(second, input)
            second[index] = -input[index]
        end
        nothing
    end
end

@algorithm CapturedGateF32 begin
    label = "captured-gate-f32"
    failure = TestGateFailure

    @configuration CapturedGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            rate=(2, 1),
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            rate=(2, 1),
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = begin
        input[1] <= 0.0f0 && return TestGateRejected
        copyto!(output, input)
        nothing
    end
end

struct TestCapturedDevice end

mutable struct TestCapturedContext
    fail_launch::Bool
    fail_synchronize::Bool
    fail_restore::Bool
    fail_copy_from_at::Int
    active::Bool
    pending_destinations::Vector{Any}
    pending_sources::Vector{Any}
    pending_count::Int
end

struct TestCapturedExecutable{C,F,N}
    capture_status::C
    frame_status::F
    nodes::N
end

mutable struct TestCapturedPass{O,I}
    output::O
    input::I
    node_index::Int32
    adoptions::Int
end

mutable struct TestCapturedSplit{F,S,I}
    first::F
    second::S
    input::I
    node_index::Int32
    adoptions::Int
end

mutable struct TestCapturedGate{O,I,S}
    output::O
    input::I
    frame_status::S
    node_index::Int32
    adoptions::Int
end

const TEST_CAPTURE_EVENT_COUNT = Ref(0)
const TEST_CAPTURE_EVENTS = fill((Int32(0), :none), 32)
const TEST_CAPTURE_RECORD_COUNT = Ref(0)

@inline function record_test_capture_event!(node_index::Int32, phase::Symbol)
    count = TEST_CAPTURE_EVENT_COUNT[] + 1
    TEST_CAPTURE_EVENT_COUNT[] = count
    TEST_CAPTURE_EVENTS[mod1(count, length(TEST_CAPTURE_EVENTS))] = (node_index, phase)
    return nothing
end

function clear_test_capture_events!()
    TEST_CAPTURE_EVENT_COUNT[] = 0
    fill!(TEST_CAPTURE_EVENTS, (Int32(0), :none))
    return nothing
end

struct TestClearCapturedFrameStatus end

function (::TestClearCapturedFrameStatus)(status; ndrange)
    ndrange == 1 || error("test frame status requires one work item")
    status[1] = Int32(0)
    return nothing
end

FilterGraphAlgorithms.capture_support(::Type{FastPassF32}, ::TestCapturedDevice) =
    SupportsCapture()
FilterGraphAlgorithms.capture_support(::Type{CapturedSplitF32}, ::TestCapturedDevice) =
    SupportsCapture()
FilterGraphAlgorithms.capture_support(::Type{CapturedGateF32}, ::TestCapturedDevice) =
    SupportsCapture()

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::TestCapturedDevice,
    ::FilterGraphAlgorithms.PreparedAlgorithm{FastPassF32},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    return TestCapturedPass(only(outputs), only(inputs), node_index, 0)
end

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::TestCapturedDevice,
    ::FilterGraphAlgorithms.PreparedAlgorithm{CapturedGateF32},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    return TestCapturedGate(only(outputs), only(inputs), frame_status, node_index, 0)
end

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::TestCapturedDevice,
    ::FilterGraphAlgorithms.PreparedAlgorithm{CapturedSplitF32},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    return TestCapturedSplit(outputs[1], outputs[2], only(inputs), node_index, 0)
end

FilterGraphAlgorithms.preflight(::TestCapturedPass) = nothing
FilterGraphAlgorithms.preflight(::TestCapturedSplit) = nothing
FilterGraphAlgorithms.preflight(::TestCapturedGate) = nothing

function FilterGraphAlgorithms.enqueue!(captured::TestCapturedPass)
    record_test_capture_event!(captured.node_index, :algorithm)
    copyto!(captured.output, captured.input)
    record_test_capture_event!(captured.node_index, :adoption)
    captured.adoptions += 1
    return nothing
end

function FilterGraphAlgorithms.enqueue!(captured::TestCapturedSplit)
    record_test_capture_event!(captured.node_index, :algorithm)
    copyto!(captured.first, captured.input)
    for index in eachindex(captured.second, captured.input)
        captured.second[index] = -captured.input[index]
    end
    record_test_capture_event!(captured.node_index, :adoption)
    captured.adoptions += 1
    return nothing
end

function FilterGraphAlgorithms.enqueue!(captured::TestCapturedGate)
    iszero(captured.frame_status[1]) || return nothing
    if captured.input[1] <= 0.0f0
        captured.frame_status[1] = captured.node_index
        return nothing
    end
    copyto!(captured.output, captured.input)
    captured.adoptions += 1
    return nothing
end

FilterGraphAlgorithms.complete!(::TestCapturedPass, ::Int32) = nothing
FilterGraphAlgorithms.complete!(::TestCapturedSplit, ::Int32) =
    nothing
function FilterGraphAlgorithms.complete!(
    captured::TestCapturedGate,
    status::Int32,
)
    return status == captured.node_index ? TestGateRejected : nothing
end

function FilterGraphAlgorithms.reset!(captured::TestCapturedPass)
    fill!(captured.output, 0.0f0)
    captured.adoptions = 0
    return captured
end

function FilterGraphAlgorithms.reset!(captured::TestCapturedSplit)
    fill!(captured.first, 0.0f0)
    fill!(captured.second, 0.0f0)
    captured.adoptions = 0
    return captured
end

function FilterGraphAlgorithms.reset!(captured::TestCapturedGate)
    fill!(captured.output, 0.0f0)
    captured.adoptions = 0
    return captured
end

FilterGraphs._prepare_captured_context(::TestCapturedDevice) =
    TestCapturedContext(
        false,
        false,
        false,
        0,
        false,
        Vector{Any}(undef, 8),
        Vector{Any}(undef, 8),
        0,
    )

function FilterGraphs._activate_captured_context(context::TestCapturedContext)
    context.active && error("test captured context is already active")
    context.active = true
    return nothing
end

function FilterGraphs._restore_captured_context!(
    context::TestCapturedContext,
    ::Nothing,
)
    context.active || error("test captured context is not active")
    context.active = false
    context.fail_restore && error("injected captured context restoration failure")
    return nothing
end

function FilterGraphs._allocate_captured_buffer(
    ::TestCapturedContext,
    format,
)
    return zeros(format.element_type, format.shape)
end

FilterGraphs._prepare_captured_frame_status(::TestCapturedContext) =
    FilterGraphs.CapturedFrameStatus(
        zeros(Int32, 1),
        zeros(Int32, 1),
        TestClearCapturedFrameStatus(),
    )

function FilterGraphs._copy_to_captured!(
    ::TestCapturedContext,
    destination,
    source,
)
    copyto!(destination, source)
    return nothing
end

function FilterGraphs._copy_from_captured!(
    context::TestCapturedContext,
    destination,
    source,
)
    index = context.pending_count + 1
    index == context.fail_copy_from_at && error("injected captured D2H submission failure")
    index <= length(context.pending_destinations) || error("test captured copy queue is full")
    context.pending_destinations[index] = destination
    context.pending_sources[index] = source
    context.pending_count = index
    return nothing
end

function FilterGraphs._record_captured_graph(
    ::TestCapturedContext,
    frame_status,
    nodes::Tuple,
)
    TEST_CAPTURE_RECORD_COUNT[] += 1
    return TestCapturedExecutable(
        frame_status,
        FilterGraphs.CapturedFrameOutcome(frame_status.value, frame_status.host),
        nodes,
    )
end

captured_process_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function FilterGraphs._launch_captured_graph!(
    context::TestCapturedContext,
    executable::TestCapturedExecutable,
)
    context.fail_launch && error("injected captured launch failure")
    FilterGraphs._enqueue_captured_frame!(
        executable.capture_status,
        executable.nodes,
    )
    return nothing
end

function FilterGraphs._synchronize_captured_graph!(context::TestCapturedContext)
    if context.fail_synchronize
        error("injected captured completion failure")
    end
    for index in 1:context.pending_count
        copyto!(context.pending_destinations[index], context.pending_sources[index])
    end
    context.pending_count = 0
    return nothing
end

const CAPTURED_PASS_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = pass
            label = fast-pass-f32
            config = { extent = 3 }
        }
    ]
    links = []
    inputs = [ "pass:input" ]
    outputs = [ "pass:output" ]
}
"""

const CAPTURED_UNSUPPORTED_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = gate
            label = test-gate-f32
            config = { extent = 3 }
        }
    ]
    links = []
    inputs = [ "gate:input" ]
    outputs = [ "gate:output" ]
}
"""

const CAPTURED_PARTLY_SUPPORTED_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = first
            label = fast-pass-f32
            config = { extent = 3 }
        }
        {
            type = ndarray
            name = gate
            label = captured-unsupported-pass-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "first:output" input = "gate:input" }
    ]
    inputs = [ "first:input" ]
    outputs = [ "gate:output" ]
}
"""

const CAPTURED_TWO_PASS_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = first
            label = fast-pass-f32
            config = { extent = 3 }
        }
        {
            type = ndarray
            name = second
            label = fast-pass-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "first:output" input = "second:input" }
    ]
    inputs = [ "first:input" ]
    outputs = [ "second:output" ]
}
"""

const CAPTURED_PASS_GATE_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = first
            label = fast-pass-f32
            config = { extent = 3 }
        }
        {
            type = ndarray
            name = gate
            label = captured-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "first:output" input = "gate:input" }
    ]
    inputs = [ "first:input" ]
    outputs = [ "gate:output" ]
}
"""

const CAPTURED_MULTI_OUTPUT_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = split
            label = captured-split-f32
            config = { extent = 3 }
        }
    ]
    links = []
    inputs = [ "split:input" ]
    outputs = [ "split:first" "split:second" ]
}
"""

@testset "captured Graph qualification is explicit and all-or-nothing" begin
    default_workgroups = CapturedWorkgroupProfile()
    @test default_workgroups.projection_lanes == 128
    @test default_workgroups.pyramid_reconstruction_lanes == 256
    @test default_workgroups.shack_hartmann_reconstruction_lanes == 128
    @test_throws ArgumentError CapturedWorkgroupProfile(projection_lanes=16)
    @test_throws ArgumentError CapturedWorkgroupProfile(
        pyramid_reconstruction_lanes=512,
    )

    unsupported = prepare_graph_text(
        CAPTURED_UNSUPPORTED_GRAPH;
        algorithms=(TestGateF32,),
    )
    error = try
        prepare_captured_graph(unsupported, TestCapturedDevice())
        nothing
    catch caught
        caught
    end
    @test error isa GraphConfigurationError
    @test occursin("no qualified captured adapter", sprint(showerror, error))

    partly_supported = prepare_graph_text(
        CAPTURED_PARTLY_SUPPORTED_GRAPH;
        algorithms=(FastPassF32, CapturedUnsupportedPassF32),
    )
    error = try
        prepare_captured_graph(partly_supported, TestCapturedDevice())
        nothing
    catch caught
        caught
    end
    @test error isa GraphConfigurationError
    @test occursin("Node 'gate'", sprint(showerror, error))
    @test occursin("no qualified captured adapter", sprint(showerror, error))

    multi_node = prepare_graph_text(
        CAPTURED_TWO_PASS_GRAPH;
        algorithms=(FastPassF32,),
    )
    workgroups = CapturedWorkgroupProfile(
        projection_lanes=32,
        pyramid_reconstruction_lanes=64,
        shack_hartmann_reconstruction_lanes=256,
    )
    captured = prepare_captured_graph(
        multi_node,
        TestCapturedDevice();
        workgroups,
    )
    @test length(captured.nodes) == 2
    @test captured_workgroup_profile(captured) === workgroups
    @test captured.nodes[2].adapter.input === captured.nodes[1].adapter.output
    @test captured.nodes[1].adapter.adoptions == 0
    @test captured.nodes[2].adapter.adoptions == 0

    clear_test_capture_events!()
    output = zeros(Float32, 3)
    @test process!((; output), captured, (input=Float32[1, 2, 3],)) == (output=true,)
    @test TEST_CAPTURE_EVENT_COUNT[] == 4
    @test TEST_CAPTURE_EVENTS[1:4] == [
        (Int32(1), :algorithm),
        (Int32(1), :adoption),
        (Int32(2), :algorithm),
        (Int32(2), :adoption),
    ]
end

@testset "captured Graph replays fixed storage and publishes after completion" begin
    checked = prepare_graph_text(
        CAPTURED_TWO_PASS_GRAPH;
        algorithms=(FastPassF32,),
    )
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    @test captured isa PreparedCapturedGraph
    @test captured_graph_state(captured) === CapturedGraphReady
    @test captured_graph_fault(captured) === nothing
    @test !captured.context.active

    device_input = captured.device_inputs.input
    device_output = captured.device_outputs.output
    host_output = captured.host_outputs.output
    output = fill(71.0f0, 3)
    inputs = (input=Float32[1, 2, 3],)
    metadata = SampleMetadata(42; timestamp_nanoseconds=1234)

    @test process!((output=output,), captured, inputs, (input=metadata,)) ==
          (output=true,)
    @test captured.nodes[1].adapter.adoptions == 1
    @test captured.nodes[2].adapter.adoptions == 1
    @test output == inputs.input
    @test FilterGraphs.output_metadata(captured) == (output=metadata,)

    inputs.input .= Float32[4, 5, 6]
    @test process!((output=output,), captured, inputs) == (output=true,)
    @test output == inputs.input
    @test captured.nodes[1].adapter.adoptions == 2
    @test captured.nodes[2].adapter.adoptions == 2
    @test captured_process_allocation_bytes((output=output,), captured, inputs) == 0

    @test reset!(captured) === captured
    @test captured.device_inputs.input === device_input
    @test captured.device_outputs.output === device_output
    @test captured.host_outputs.output === host_output
    @test captured.nodes[1].adapter.adoptions == 0
    @test captured.nodes[2].adapter.adoptions == 0
    @test captured_graph_state(captured) === CapturedGraphReady
end

@testset "captured downstream rejection preserves earlier Node adoption" begin
    checked = prepare_graph_text(
        CAPTURED_PASS_GATE_GRAPH;
        algorithms=(FastPassF32, CapturedGateF32),
    )
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    output = fill(71.0f0, 3)

    @test process!((; output), captured, (input=Float32[-1, 2, 3],)) ===
          TestGateRejected
    @test output == fill(71.0f0, 3)
    @test captured.nodes[1].adapter.adoptions == 1
    @test captured.nodes[2].adapter.adoptions == 0
    @test captured_graph_state(captured) === CapturedGraphReady
end

@testset "captured warm declared failure records and retains runtime suppression" begin
    checked = prepare_graph_text(
        CAPTURED_PASS_GATE_GRAPH;
        algorithms=(FastPassF32, CapturedGateF32),
    )
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    output = fill(71.0f0, 3)

    # Retained capture buffers are zero-initialized, so graph preparation has
    # already observed this declared failure before the first user frame.
    @test captured.nodes[1].adapter.adoptions == 0
    @test captured.nodes[2].adapter.adoptions == 0
    @test process!((; output), captured, (input=zeros(Float32, 3),)) ===
          TestGateRejected
    @test output == fill(71.0f0, 3)
    @test captured.nodes[1].adapter.adoptions == 1
    @test captured.nodes[2].adapter.adoptions == 0

    positive = Float32[1, 2, 3]
    @test process!((; output), captured, (; input=positive)) == (output=true,)
    @test output == positive
    @test captured.nodes[1].adapter.adoptions == 2
    @test captured.nodes[2].adapter.adoptions == 1
end

@testset "captured Graph publishes multiple outputs after completion" begin
    checked = prepare_graph_text(
        CAPTURED_MULTI_OUTPUT_GRAPH;
        algorithms=(CapturedSplitF32,),
    )
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    first = fill(71.0f0, 3)
    second = fill(83.0f0, 3)
    outputs = (; first, second)
    input = Float32[1, -2, 3]
    inputs = (; input)
    metadata = SampleMetadata(81; timestamp_nanoseconds=9_001)

    overlapping_storage = fill(91.0f0, 4)
    overlapping_outputs = (
        first=view(overlapping_storage, 1:3),
        second=view(overlapping_storage, 2:4),
    )
    @test Base.mightalias(overlapping_outputs.first, overlapping_outputs.second)
    @test_throws ArgumentError process!(overlapping_outputs, captured, inputs)
    @test overlapping_storage == fill(91.0f0, 4)
    @test captured.nodes[1].adapter.adoptions == 0
    @test captured_graph_state(captured) === CapturedGraphReady

    @test process!(outputs, captured, inputs, (; input=metadata)) ==
          (first=true, second=true)
    @test first == input
    @test second == .-input
    @test FilterGraphs.output_metadata(captured) ==
          (first=metadata, second=metadata)
    @test captured_process_allocation_bytes(outputs, captured, inputs) == 0

    device_first = captured.device_outputs.first
    device_second = captured.device_outputs.second
    host_first = captured.host_outputs.first
    host_second = captured.host_outputs.second
    @test reset!(captured) === captured
    @test captured.device_outputs.first === device_first
    @test captured.device_outputs.second === device_second
    @test captured.host_outputs.first === host_first
    @test captured.host_outputs.second === host_second
    @test captured.nodes[1].adapter.adoptions == 0
end

@testset "captured Graph faults are retained and fail-stop" begin
    checked = prepare_graph_text(
        CAPTURED_PASS_GRAPH;
        algorithms=(FastPassF32,),
    )
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    output = fill(71.0f0, 3)
    inputs = (input=Float32[1, 2, 3],)

    captured.context.fail_synchronize = true
    first_fault = try
        process!((output=output,), captured, inputs)
        nothing
    catch caught
        caught
    end
    @test first_fault isa CapturedGraphFault
    @test first_fault.phase === :execution
    @test captured_graph_state(captured) === CapturedGraphFaulted
    @test captured_graph_fault(captured) === first_fault
    @test !captured.context.active
    @test output == fill(71.0f0, 3)
    @test captured.host_outputs.output == zeros(Float32, 3)
    @test captured.context.pending_count == 2
    drain_failure = FilterGraphs._captured_drain_failure(captured)
    @test drain_failure isa CapturedException
    @test drain_failure.ex isa ErrorException
    @test !isempty(drain_failure.processed_bt)

    captured.context.fail_synchronize = false
    retained_fault = try
        process!((output=output,), captured, inputs)
        nothing
    catch caught
        caught
    end
    @test retained_fault === first_fault

    @test reset!(captured) === captured
    @test captured_graph_state(captured) === CapturedGraphReady
    @test captured_graph_fault(captured) === nothing
    @test FilterGraphs._captured_drain_failure(captured) === nothing
    @test process!((output=output,), captured, inputs) == (output=true,)
    @test output == inputs.input

    partial_copy_checked = prepare_graph_text(
        CAPTURED_MULTI_OUTPUT_GRAPH;
        algorithms=(CapturedSplitF32,),
    )
    partial_copy_faulted = prepare_captured_graph(partial_copy_checked, TestCapturedDevice())
    partial_first = fill(101.0f0, 3)
    partial_second = fill(103.0f0, 3)
    partial_outputs = (; first=partial_first, second=partial_second)
    partial_copy_faulted.context.fail_copy_from_at = 3
    partial_copy_fault = try
        process!(partial_outputs, partial_copy_faulted, inputs)
        nothing
    catch caught
        caught
    end
    @test partial_copy_fault isa CapturedGraphFault
    @test partial_copy_fault.phase === :execution
    @test captured_graph_state(partial_copy_faulted) === CapturedGraphFaulted
    @test partial_first == fill(101.0f0, 3)
    @test partial_second == fill(103.0f0, 3)
    @test partial_copy_faulted.context.pending_count == 0
    @test FilterGraphs._captured_drain_failure(partial_copy_faulted) === nothing
    partial_copy_faulted.context.fail_copy_from_at = 0
    @test reset!(partial_copy_faulted) === partial_copy_faulted
    @test partial_copy_faulted.context.pending_count == 0
    @test process!(partial_outputs, partial_copy_faulted, inputs) ==
          (first=true, second=true)
    @test partial_first == inputs.input
    @test partial_second == .-inputs.input

    captured.state = CapturedGraphRunning
    @test_throws CapturedGraphBusy reset!(captured)
    captured.state = CapturedGraphReady

    launch_faulted = prepare_captured_graph(checked, TestCapturedDevice())
    launch_output = fill(83.0f0, 3)
    launch_faulted.context.fail_launch = true
    launch_fault = try
        process!((output=launch_output,), launch_faulted, inputs)
        nothing
    catch caught
        caught
    end
    @test launch_fault isa CapturedGraphFault
    @test launch_fault.phase === :execution
    @test captured_graph_state(launch_faulted) === CapturedGraphFaulted
    @test captured_graph_fault(launch_faulted) === launch_fault
    @test !launch_faulted.context.active
    @test launch_output == fill(83.0f0, 3)

    restore_faulted = prepare_captured_graph(checked, TestCapturedDevice())
    restore_output = fill(97.0f0, 3)
    restore_faulted.context.fail_restore = true
    restore_fault = try
        process!((output=restore_output,), restore_faulted, inputs)
        nothing
    catch caught
        caught
    end
    @test restore_fault isa CapturedGraphFault
    @test captured_graph_state(restore_faulted) === CapturedGraphFaulted
    @test captured_graph_fault(restore_faulted) === restore_fault
    @test !restore_faulted.context.active
    @test restore_output == fill(97.0f0, 3)
end

@noinline function quarantined_capture_weakrefs(checked)
    captured = prepare_captured_graph(checked, TestCapturedDevice())
    input = Float32[3, 4, 5]
    captured.context.fail_copy_from_at = 2
    captured.context.fail_synchronize = true
    @test_throws CapturedGraphFault process!(
        (output=zeros(Float32, 3),),
        captured,
        (input=input,),
    )
    return WeakRef(captured), WeakRef(input)
end

@noinline function recover_quarantined_capture!(owner_ref)
    captured = owner_ref.value
    @test captured isa PreparedCapturedGraph
    @test captured.context.pending_count == 1
    original_drain_failure = FilterGraphs._captured_drain_failure(captured)
    @test original_drain_failure isa CapturedException

    # Failed reset must not replace the retained input roots with the owner.
    @test_throws CapturedGraphFault reset!(captured)
    @test FilterGraphs._captured_drain_failure(captured) === original_drain_failure
    captured.context.fail_synchronize = false
    captured.context.fail_copy_from_at = 0
    @test reset!(captured) === captured
    @test captured.context.pending_count == 0
    @test FilterGraphs._captured_drain_failure(captured) === nothing
    return nothing
end

@noinline function check_quarantined_capture_roots(owner_ref, input_ref)
    GC.gc(true)
    @test owner_ref.value isa PreparedCapturedGraph
    @test input_ref.value == Float32[3, 4, 5]
    return nothing
end

@testset "failed captured drain retains borrowed inputs until recovery" begin
    checked = prepare_graph_text(CAPTURED_PASS_GRAPH; algorithms=(FastPassF32,))
    owner_ref, input_ref = quarantined_capture_weakrefs(checked)
    check_quarantined_capture_roots(owner_ref, input_ref)
    recover_quarantined_capture!(owner_ref)
    GC.gc(true)
    @test owner_ref.value === nothing
    @test input_ref.value === nothing
end

@testset "in-memory Graph and captured device publication" begin
    document = FilterGraphs._parse_spa_json(CAPTURED_PASS_GATE_GRAPH)
    graph = prepare_graph(document; algorithms=(FastPassF32, CapturedGateF32))
    captured = prepare_captured_graph(graph, TestCapturedDevice())
    # Frame status is the first copy. The next copy would be the image.
    captured.context.fail_copy_from_at = 2
    fill!(captured.host_outputs.output, 71.0f0)
    inputs = (input=Float32[1, 2, 3],)
    metadata = (input=SampleMetadata(4),)
    published = Ref(0)
    result = process_device!(captured, inputs, metadata) do outputs
        published[] += 1
        @test outputs.output === captured.device_outputs.output
        @test outputs.output == inputs.input
        @test_throws FilterGraphs.CapturedGraphBusy process_device!(identity, captured, inputs)
    end
    @test result == (output=true,)
    @test published[] == 1
    @test all(==(71.0f0), captured.host_outputs.output)
    @test FilterGraphs.output_metadata(captured) == (output=nothing,)

    inputs.input[1] = -1.0f0
    result = process_device!(captured, inputs, (input=SampleMetadata(5),)) do outputs
        published[] += 1
    end
    @test result === TestGateRejected
    @test published[] == 1
    @test FilterGraphs.output_metadata(captured) == (output=nothing,)

    captured.context.fail_copy_from_at = 0
    inputs.input[1] = 4.0f0
    output = zeros(Float32, 3)
    @test process!((; output), captured, inputs) == (output=true,)
    @test output == inputs.input

    document = FilterGraphs._parse_spa_json(CAPTURED_TWO_PASS_GRAPH)
    routed = prepare_captured_graph(prepare_graph(document; algorithms=(FastPassF32,)),
        TestCapturedDevice())
    @test process_device!(identity, routed, inputs, metadata) == (output=true,)
    @test FilterGraphs.output_metadata(routed) == (output=metadata.input,)
    @test_throws FilterGraphs.CapturedGraphFault process_device!(routed, inputs) do outputs
        process_device!(identity, routed, inputs)
    end
    @test captured_graph_state(routed) === FilterGraphs.CapturedGraphFaulted
    @test captured_graph_fault(routed).phase === :publication
    @test reset!(routed) === routed
    @test process_device!(identity, routed, inputs, metadata) == (output=true,)
end

@testset "captured host readback retains device diagnostics" begin
    document = (;
        nodes=((; name=:first, label="fast-pass-f32", config=(; extent=3)),
            (; name=:gate, label="captured-gate-f32", config=(; extent=3))),
        links=("first:output"=>"gate:input",),
        inputs=(; input="first:input"),
        outputs=(; diagnostic="first:output", output="gate:output"),
    )
    graph = prepare_graph(document; algorithms=(FastPassF32, CapturedGateF32))
    @test_throws GraphConfigurationError prepare_captured_graph(graph, TestCapturedDevice();
        outputs=(:output,), readback=(:diagnostic,))
    @test_throws GraphConfigurationError prepare_captured_graph(graph, TestCapturedDevice();
        readback=(:output,:output))

    captured = prepare_captured_graph(graph, TestCapturedDevice(); readback=(:output,))
    @test keys(FilterGraphs.output_formats(captured)) == (:diagnostic,:output)
    @test keys(captured.host_outputs) == (:output,)
    # The queue contains status and the selected output before synchronization.
    captured.context.fail_copy_from_at = 3
    inputs = (; input=Float32[1,2,3])
    output = zeros(Float32, 3)
    metadata = (; input=SampleMetadata(4))
    @test process!((; output), captured, inputs, metadata) == (; diagnostic=true, output=true)
    @test output == inputs.input
    @test captured.device_outputs.diagnostic == inputs.input
    @test FilterGraphs.output_metadata(captured).diagnostic == metadata.input

    inputs.input[1] = -1
    @test process!((; output), captured, inputs, (; input=SampleMetadata(5))) === TestGateRejected
    @test output == Float32[1,2,3]
    @test FilterGraphs.output_metadata(captured).diagnostic == metadata.input
end
