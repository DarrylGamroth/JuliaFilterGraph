@enum CapturedPipeWireFailure::UInt8 begin
    CapturedPipeWireRejected = 1
end

@algorithm CapturedPipeWireSplitF32 begin
    label = "captured-pipewire-split-f32"
    failure = CapturedPipeWireFailure

    @configuration CapturedPipeWireSplitF32Configuration begin
        extent::Int => graph_rebuild
        scale::Float32 => graph_rebuild
        (require_duration::Bool=false) => graph_rebuild
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

    prepare(configuration) = (
        extent=configuration.extent,
        scale=configuration.scale,
        require_duration=configuration.require_duration,
    )
    process!(plan) = begin
        input[1] <= 0.0f0 && return CapturedPipeWireRejected
        for index in eachindex(first, second, input)
            first[index] = plan.scale * input[index]
            second[index] = -plan.scale * input[index]
        end
        nothing
    end
end

FilterGraphAlgorithms.required_sample_metadata(prepared::FilterGraphAlgorithms.PreparedAlgorithm{CapturedPipeWireSplitF32},input_name::Symbol)=
    input_name===:input && prepared.plan.require_duration ? (:exposure_duration_nanoseconds,) : ()

struct PipeWireTestCapturedDevice end

mutable struct PipeWireTestCapturedContext
    active::Bool
    fail_launch::Bool
    fail_synchronize::Bool
    launches::Int
    pending_destinations::Vector{Any}
    pending_sources::Vector{Any}
    pending_count::Int
end

struct PipeWireTestCapturedExecutable{C,F,N}
    capture_status::C
    frame_status::F
    nodes::N
end

mutable struct PipeWireTestCapturedSplit{F,S,I,T}
    first::F
    second::S
    input::I
    frame_status::T
    failure_code::Int32
    scale::Float32
end

struct PipeWireTestClearFrameStatus end

function (::PipeWireTestClearFrameStatus)(status; ndrange)
    ndrange == 1 || error("test frame status requires one work item")
    status[1] = Int32(0)
    return nothing
end

FilterGraphAlgorithms.capture_support(
    ::Type{CapturedPipeWireSplitF32},
    ::PipeWireTestCapturedDevice,
) = SupportsCapture()

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::PipeWireTestCapturedDevice,
    prepared::FilterGraphAlgorithms.PreparedAlgorithm{CapturedPipeWireSplitF32},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    failure_code = (node_index << 8) | Int32(CapturedPipeWireRejected)
    return PipeWireTestCapturedSplit(
        outputs[1],
        outputs[2],
        only(inputs),
        frame_status,
        failure_code,
        prepared.plan.scale,
    )
end

FilterGraphAlgorithms.preflight(
    ::PipeWireTestCapturedSplit,
) = nothing

function FilterGraphAlgorithms.enqueue!(
    captured::PipeWireTestCapturedSplit,
)
    iszero(captured.frame_status[1]) || return nothing
    if captured.input[1] <= 0.0f0
        captured.frame_status[1] = captured.failure_code
        return nothing
    end
    for index in eachindex(captured.first, captured.second, captured.input)
        captured.first[index] = captured.scale * captured.input[index]
        captured.second[index] = -captured.scale * captured.input[index]
    end
    return nothing
end

function FilterGraphAlgorithms.complete!(
    captured::PipeWireTestCapturedSplit,
    status::Int32,
)
    status == captured.failure_code && return CapturedPipeWireRejected
    return nothing
end

function FilterGraphAlgorithms.reset!(
    captured::PipeWireTestCapturedSplit,
)
    fill!(captured.first, 0.0f0)
    fill!(captured.second, 0.0f0)
    return captured
end

FilterGraphs._prepare_captured_context(::PipeWireTestCapturedDevice) =
    PipeWireTestCapturedContext(
        false,
        false,
        false,
        0,
        Vector{Any}(undef, 8),
        Vector{Any}(undef, 8),
        0,
    )

function FilterGraphs._activate_captured_context(
    context::PipeWireTestCapturedContext,
)
    context.active && error("test captured context is already active")
    context.active = true
    return nothing
end

function FilterGraphs._restore_captured_context!(
    context::PipeWireTestCapturedContext,
    ::Nothing,
)
    context.active || error("test captured context is not active")
    context.active = false
    return nothing
end

function FilterGraphs._allocate_captured_buffer(
    ::PipeWireTestCapturedContext,
    format,
)
    return zeros(format.element_type, format.shape)
end

FilterGraphs._prepare_captured_frame_status(
    ::PipeWireTestCapturedContext,
) = FilterGraphs.CapturedFrameStatus(
    zeros(Int32, 1),
    zeros(Int32, 1),
    PipeWireTestClearFrameStatus(),
)

function FilterGraphs._copy_to_captured!(
    ::PipeWireTestCapturedContext,
    destination,
    source,
)
    copyto!(destination, source)
    return nothing
end

function FilterGraphs._copy_from_captured!(
    context::PipeWireTestCapturedContext,
    destination,
    source,
)
    index = context.pending_count + 1
    index <= length(context.pending_destinations) || error("test captured copy queue is full")
    context.pending_destinations[index] = destination
    context.pending_sources[index] = source
    context.pending_count = index
    return nothing
end

FilterGraphs._record_captured_graph(
    ::PipeWireTestCapturedContext,
    frame_status,
    nodes::Tuple,
) = PipeWireTestCapturedExecutable(
    frame_status,
    FilterGraphs.CapturedFrameOutcome(frame_status.value, frame_status.host),
    nodes,
)

function FilterGraphs._launch_captured_graph!(
    context::PipeWireTestCapturedContext,
    executable::PipeWireTestCapturedExecutable,
)
    context.fail_launch && error("injected captured launch failure")
    context.launches += 1
    FilterGraphs._enqueue_captured_frame!(
        executable.capture_status,
        executable.nodes,
    )
    return nothing
end

function FilterGraphs._synchronize_captured_graph!(
    context::PipeWireTestCapturedContext,
)
    if context.fail_synchronize
        context.pending_count = 0
        error("injected captured completion failure")
    end
    for index in 1:context.pending_count
        copyto!(context.pending_destinations[index], context.pending_sources[index])
    end
    context.pending_count = 0
    return nothing
end

const CAPTURED_PIPEWIRE_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = split
            label = captured-pipewire-split-f32
            config = { extent = 3 scale = 1.0 }
        }
    ]
    links = []
    inputs = [ "split:input" ]
    outputs = [ "split:first" "split:second" ]
}
"""

const CAPTURED_OUTPUT_UNAVAILABLE =
    PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

function prepared_captured_pipewire_graph(; extent=3, scale=1.0f0,require_duration=false, readback=nothing)
    graph = replace(
        CAPTURED_PIPEWIRE_GRAPH,
        "config = { extent = 3 scale = 1.0 }" =>
            "config = { extent = $extent scale = $scale require_duration = $require_duration }",
    )
    checked = prepare_graph_text(
        graph;
        algorithms=(CapturedPipeWireSplitF32,),
    )
    return prepare_captured_graph(checked, PipeWireTestCapturedDevice(); readback)
end

function captured_pipewire_callback(
    node,
    input,
    first,
    second;
    input_header=nothing,
    first_header_available=true,
    second_header_available=true,
    input_present=true,
    extra_property=nothing,
)
    native_input = native_callback_buffer(
        input;
        header=input_header,
        present=input_present,
        extra_property,
    )
    native_outputs = vcat(
        native_callback_buffer(first; header_available=first_header_available),
        native_callback_buffer(second; header_available=second_header_available),
    )
    GC.@preserve input first second native_input native_outputs begin
        invoke_captured_pipewire_callback(node, native_input, native_outputs)
    end
    return native_outputs
end

function invoke_captured_pipewire_callback(node, native_input, native_outputs)
    callback = node.filter.callbacks.on_process
    inputs = NdArrayFilterBuffers{false}(
        Ptr{Cvoid}(pointer(native_input)),
        UInt32(1),
    )
    outputs = NdArrayFilterBuffers{true}(
        Ptr{Cvoid}(pointer(native_outputs)),
        UInt32(2),
    )
    callback(node.filter, inputs, outputs)
    return nothing
end

@testset "captured Graph PipeWire publication is reserved before launch" begin
    subset = prepared_captured_pipewire_graph(; readback=(:second,))
    @test_throws FilterGraphs.GraphConfigurationError PipeWireNode(subset; connect=false)
    captured = prepared_captured_pipewire_graph(; readback=(:second, :first))
    @test_throws MethodError PipeWireNode(captured; connect=false,
        property_observer=(phase, props) -> nothing)
    @test_throws MethodError PipeWireNode(captured; connect=false,
        property_info=SPA.PropInfo[], property_getter=nothing)
    captured.device_inputs.input .= Float32[4, 5, 6]
    node = PipeWireNode(
        captured;
        name="test.julia.captured-filter-graph",
        rate=(1_000, 1),
        connect=false,
    )
    @test node.owner isa FilterGraphPipeWire.PreparedCapturedGraphSlot
    active_owner = FilterGraphPipeWire._active_captured_owner(node.owner)
    @test length(active_owner.input_bindings) == 1
    @test length(active_owner.output_bindings) == 2
    @test active_owner.staged_inputs.input == Float32[4, 5, 6]
    @test isopen(node)
    @test_throws InvalidStateException replace_parameters!(node, :gain => 1.0f0)

    FilterGraphPipeWire._prepare_owner(node.owner)
    @test FilterGraphPipeWire._active_captured_owner(node.owner).warmed
    @test captured_graph_state(captured) === CapturedGraphReady
    staged_storage = (
        values(active_owner.staged_inputs)...,
        values(active_owner.staged_outputs)...,
    )
    launches_after_warmup = captured.context.launches
    input = Float32[1, 2, 3]
    first = fill(71.0f0, 3)
    second = fill(83.0f0, 3)
    header = BufferHeader(
        UInt32(0),
        UInt32(17),
        Int64(9_001),
        Int64(0),
        UInt64(81),
    )
    native_outputs = captured_pipewire_callback(
        node,
        input,
        first,
        second;
        input_header=header, extra_property=Int64(123),
    )
    @test captured.context.launches == launches_after_warmup + 1
    @test first == input
    @test second == .-input
    @test all(
        iszero(
            native_callback_flags(native_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )
    @test native_callback_header(native_outputs, 1) == header
    @test native_callback_header(native_outputs, 2) == header
    GC.@preserve native_outputs begin
        buffers = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_outputs)),UInt32(2))
        @test all(native_callback_custom_value(native_outputs,index) == 123 for index in 1:2)
    end

    native_input = native_callback_buffer(input; header)
    measured_outputs = vcat(
        native_callback_buffer(first; header_available=true),
        native_callback_buffer(second; header_available=true),
    )
    GC.@preserve input first second native_input measured_outputs begin
        invoke_captured_pipewire_callback(node, native_input, measured_outputs)
        @test @inferred(invoke_captured_pipewire_callback(
            node,
            native_input,
            measured_outputs,
        )) === nothing
        @test @allocated(
            invoke_captured_pipewire_callback(node, native_input, measured_outputs)
        ) == 0
    end

    fill!(first, 71.0f0)
    fill!(second, 83.0f0)
    launches_before_incomplete = captured.context.launches
    unavailable_outputs = captured_pipewire_callback(
        node,
        input,
        first,
        second;
        input_header=header,
        input_present=false,
    )
    @test captured.context.launches == launches_before_incomplete
    @test first == fill(71.0f0, 3)
    @test second == fill(83.0f0, 3)
    @test all(
        !iszero(
            native_callback_flags(unavailable_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )
    @test all(
        map(
            ===,
            staged_storage,
            (
                values(active_owner.staged_inputs)...,
                values(active_owner.staged_outputs)...,
            ),
        ),
    )
    close(node)
end

@testset "captured Graph replacement is prepared before an atomic owner swap" begin
    initial = prepared_captured_pipewire_graph()
    node = PipeWireNode(initial; connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)
    input = Float32[1, 2, 3]
    first = fill(71.0f0, 3)
    second = fill(83.0f0, 3)
    header = BufferHeader(UInt32(0), UInt32(0), Int64(1), Int64(0), UInt64(1))

    replacement_graph = prepared_captured_pipewire_graph(; scale=2.0f0)
    replacement = prepare_replacement(node, replacement_graph)
    @test replacement isa PreparedCapturedGraphReplacement
    @test FilterGraphPipeWire._active_captured_owner(node.owner).graph === initial
    captured_pipewire_callback(
        node,
        input,
        first,
        second;
        input_header=header,
    )
    @test first == input
    @test second == .-input

    @test replace_graph!(node, replacement) === node
    @test FilterGraphPipeWire._active_captured_owner(node.owner).graph ===
        replacement_graph
    captured_pipewire_callback(
        node,
        input,
        first,
        second;
        input_header=header,
    )
    @test first == 2.0f0 .* input
    @test second == -2.0f0 .* input
    @test_throws InvalidStateException replace_graph!(node, replacement)

    @test_throws DimensionMismatch prepare_replacement(
        node,
        prepared_captured_pipewire_graph(; extent=4, scale=3.0f0),
    )
    @test FilterGraphPipeWire._active_captured_owner(node.owner).graph ===
        replacement_graph

    failed_graph = prepared_captured_pipewire_graph(; scale=3.0f0)
    failed_graph.context.fail_synchronize = true
    @test_throws CapturedGraphFault prepare_replacement(node, failed_graph)
    @test FilterGraphPipeWire._active_captured_owner(node.owner).graph ===
        replacement_graph

    final_graph = prepared_captured_pipewire_graph(; scale=3.0f0)
    final_replacement = prepare_replacement(node, final_graph)
    node.owner.callback_state[] = FilterGraphPipeWire._CAPTURED_CALLBACK_ACTIVE
    try
        busy = try
            replace_graph!(node, final_replacement)
            nothing
        catch error
            error
        end
        @test busy isa InvalidStateException
        @test busy.state === :busy
    finally
        node.owner.callback_state[] = FilterGraphPipeWire._CAPTURED_CALLBACK_IDLE
    end
    @test FilterGraphPipeWire._active_captured_owner(node.owner).graph ===
        replacement_graph
    @test replace_graph!(node, final_replacement) === node

    native_input = native_callback_buffer(input; header)
    native_outputs = vcat(
        native_callback_buffer(first; header_available=true),
        native_callback_buffer(second; header_available=true),
    )
    GC.@preserve input first second native_input native_outputs begin
        invoke_captured_pipewire_callback(node, native_input, native_outputs)
        @test first == 3.0f0 .* input
        @test second == -3.0f0 .* input
        @test @allocated(
            invoke_captured_pipewire_callback(node, native_input, native_outputs)
        ) == 0
    end
    close(node)
end

@testset "captured Graph PipeWire capacity misses suppress launch" begin
    captured = prepared_captured_pipewire_graph()
    node = PipeWireNode(captured; connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)
    input = Float32[1, 2, 3]
    first = fill(71.0f0, 3)
    second = fill(83.0f0, 3)
    header = BufferHeader(UInt32(0), UInt32(0), Int64(1), Int64(0), UInt64(1))
    launches = captured.context.launches

    native_input = native_callback_buffer(input; header)
    missing_header_outputs = vcat(
        native_callback_buffer(first; header_available=true),
        native_callback_buffer(second; header_available=false),
    )
    GC.@preserve input first second native_input missing_header_outputs begin
        @test_throws InvalidStateException invoke_captured_pipewire_callback(
            node,
            native_input,
            missing_header_outputs,
        )
    end
    @test captured.context.launches == launches
    @test captured_graph_state(captured) === CapturedGraphReady
    @test first == fill(71.0f0, 3)
    @test second == fill(83.0f0, 3)
    @test all(
        !iszero(
            native_callback_flags(missing_header_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )

    native_outputs = vcat(
        native_callback_buffer(first; header_available=true),
        native_callback_buffer(second; header_available=true),
    )
    unsafe_store!(callback_field(native_outputs,Val(:capacity),2), UInt32(sizeof(Float32)))
    callback = node.filter.callbacks.on_process
    GC.@preserve input first second native_input native_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_input)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_outputs)),
            UInt32(2),
        )
        @test_throws DimensionMismatch callback(node.filter, inputs, outputs)
    end
    @test captured.context.launches == launches
    @test all(
        !iszero(
            native_callback_flags(native_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )
    close(node)
end

@testset "captured Graph PipeWire failures publish nothing and remain fail-stop" begin
    captured = prepared_captured_pipewire_graph()
    node = PipeWireNode(captured; connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)
    first = fill(71.0f0, 3)
    second = fill(83.0f0, 3)
    header = BufferHeader(UInt32(0), UInt32(0), Int64(1), Int64(0), UInt64(1))

    failure_outputs = captured_pipewire_callback(
        node,
        Float32[-1, 2, 3],
        first,
        second;
        input_header=header,
    )
    @test first == fill(71.0f0, 3)
    @test second == fill(83.0f0, 3)
    @test all(
        !iszero(
            native_callback_flags(failure_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )
    @test captured_graph_state(captured) === CapturedGraphReady

    captured.context.fail_synchronize = true
    launches_before_fault = captured.context.launches
    @test_throws CapturedGraphFault captured_pipewire_callback(
        node,
        Float32[1, 2, 3],
        first,
        second;
        input_header=header,
    )
    @test captured.context.launches == launches_before_fault + 1
    @test captured_graph_state(captured) === CapturedGraphFaulted
    @test captured.host_outputs.first == zeros(Float32, 3)
    @test captured.host_outputs.second == zeros(Float32, 3)
    @test captured.context.pending_count == 0
    retained_fault = captured_graph_fault(captured)
    captured.context.fail_synchronize = false
    @test_throws CapturedGraphFault captured_pipewire_callback(
        node,
        Float32[1, 2, 3],
        first,
        second;
        input_header=header,
    )
    @test captured.context.launches == launches_before_fault + 1
    @test captured_graph_fault(captured) === retained_fault

    FilterGraphs.reset!(captured)
    recovery_outputs = captured_pipewire_callback(
        node,
        Float32[1, 2, 3],
        first,
        second;
        input_header=header,
    )
    @test captured_graph_state(captured) === CapturedGraphReady
    @test all(
        iszero(
            native_callback_flags(recovery_outputs, index) &
            CAPTURED_OUTPUT_UNAVAILABLE,
        ) for
        index in 1:2
    )
    close(node)
end

@testset "captured owners retain and preflight source sample facts without heap allocation" begin
    graph=prepared_captured_pipewire_graph(;require_duration=true)
    required=required_input_metadata(graph)
    @test only(required.input).type==SPA.META_EXPOSURE_TIMING
    @test only(required.input).required
    profile=(metadata=required.input,)
    node=PipeWireNode(graph;connect=false,input_metadata=(input=profile,),output_metadata=(first=profile,second=profile))
    FilterGraphPipeWire._prepare_owner(node.owner)
    input=Float32[1,2,3]; first=fill(42f0,3);second=fill(43f0,3)
    source=native_callback_buffer(input;header=BufferHeader(0,0,1,0,1))
    targets=vcat(native_callback_buffer(first),native_callback_buffer(second))
    GC.@preserve input first second source targets begin
        launches=graph.context.launches
        @test_throws InvalidStateException invoke_captured_pipewire_callback(node,source,targets)
        @test graph.context.launches==launches
        @test first==fill(42f0,3) && second==fill(43f0,3)
        set_exposure_timing!(writable_metadata_records(source.metadata[1]),ExposureTiming(;duration_nanoseconds=10))
        invoke_captured_pipewire_callback(node,source,targets)
        @test first==input && second==-input
        @test @allocated(invoke_captured_pipewire_callback(node,source,targets))==0
    end
    close(node)
end
