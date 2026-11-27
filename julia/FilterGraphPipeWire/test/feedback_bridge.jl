const CALLBACK_FEEDBACK_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = correction
            label = closed-loop-correction-f32
            config = { state_length = 1 hidden_mode_count = 1 }
            props = {
                gain = 1.0 pole = 0.99
                anti-windup-gain = 0.99 hidden-mode-gain = 0.0
            }
        }
        {
            type = ndarray
            name = projection
            label = controller-to-vdm-f32
            config = { controller_command_count = 1 active_vdm_count = 1 }
        }
        {
            type = ndarray
            name = physical
            label = vdm-to-pdm-f32
            config = { active_vdm_count = 1 full_vdm_count = 1 physical_actuator_count = 1 }
        }
        {
            type = ndarray
            name = command
            label = pdm-command-f32
            config = {
                actuator_count = 1
                command_limits = [-0.8, 0.8]
                slew_limit = null
                initial_current = null
                quantization_origin = null
                quantization_step = null
                dither_seed = null
            }
        }
        {
            type = ndarray
            name = feedback-vdm
            label = pdm-feedback-to-vdm-f32
            config = { active_vdm_count = 1 full_vdm_count = 1 physical_actuator_count = 1 }
        }
        {
            type = ndarray
            name = feedback-to-controller
            label = vdm-feedback-to-controller-f32
            config = { controller_command_count = 1 active_vdm_count = 1 }
        }
    ]
    links = [
        { output = "correction:correction" input = "projection:controller-command" }
        { output = "projection:vdm-command" input = "physical:vdm-command" }
        { output = "physical:requested-pdm-command" input = "command:requested" }
        { output = "command:constraint-feedback" input = "feedback-vdm:pdm-constraint-feedback" }
        { output = "feedback-vdm:vdm-constraint-feedback" input = "feedback-to-controller:vdm-constraint-feedback" }
    ]
    inputs = [
        "correction:residual-error"
        "correction:constraint-feedback"
        "correction:shape-to-hidden"
        "correction:hidden-to-shape"
        "projection:controller-to-vdm"
        "physical:active-to-full"
        "physical:vdm-to-pdm"
        "feedback-vdm:full-to-active"
        "feedback-vdm:pdm-to-vdm"
        "feedback-to-controller:vdm-to-controller"
    ]
    outputs = [
        "command:demanded"
        "feedback-to-controller:controller-constraint-feedback"
    ]
}
"""

const CALLBACK_FEEDBACK_PAIR =
    Symbol("constraint-feedback") => Symbol("controller-constraint-feedback")

function prepared_callback_feedback_graph(; extra_output=nothing)
    description = extra_output === nothing ? CALLBACK_FEEDBACK_GRAPH :
        replace(CALLBACK_FEEDBACK_GRAPH, "outputs = [" => "outputs = [\n        \"$extra_output\"")
    graph = prepare_graph_text(description; algorithms=algorithms())
    identity = ones(Float32, 1, 1)
    zero_projection = zeros(Float32, 1, 1)
    replace_parameters!(
        graph,
        Symbol("shape-to-hidden") => zero_projection,
        Symbol("hidden-to-shape") => zero_projection,
        Symbol("controller-to-vdm") => identity,
        Symbol("active-to-full") => identity,
        Symbol("vdm-to-pdm") => identity,
        Symbol("full-to-active") => identity,
        Symbol("pdm-to-vdm") => identity,
        Symbol("vdm-to-controller") => identity,
    )
    return graph
end

@algorithm AdapterProgressiveResidualF32 begin
    label = "adapter-progressive-residual-f32"

    @configuration AdapterProgressiveResidualF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :row,
            Float32,
            (1, configuration.extent),
            "test.progressive-controller-row/1";
            sample=:row_sample,
            rate=(2, 1),
        ),
        output(
            :residual_error,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            name="residual-error",
            metadata_from=:row,
            publication=:conditional,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(_plan) = begin
        (isnothing(row_sample) || !row_sample.terminal) &&
            return AlgorithmProcessDeferred()
        copyto!(residual_error, @view row[1, :])
        return nothing
    end
end

function prepared_progressive_feedback_graph()
    source = """
        {
            type = ndarray
            name = source
            label = adapter-progressive-residual-f32
            config = { extent = 253 }
        }
    """
    configuration = replace(
        CALLBACK_FEEDBACK_GRAPH,
        "nodes = [" => "nodes = [\n$source";
        count=1,
    )
    configuration = replace(
        configuration,
        "links = [" => "links = [\n" *
            "        { output = \"source:residual-error\" " *
            "input = \"correction:residual-error\" }";
        count=1,
    )
    configuration = replace(
        configuration,
        "\"correction:residual-error\"\n        \"correction:constraint-feedback\"" =>
            "\"source:row\"\n        \"correction:constraint-feedback\"",
    )
    for (old, new) in (
        "state_length = 1" => "state_length = 253",
        "controller_command_count = 1" => "controller_command_count = 253",
        "active_vdm_count = 1" => "active_vdm_count = 253",
        "full_vdm_count = 1" => "full_vdm_count = 253",
        "physical_actuator_count = 1" => "physical_actuator_count = 277",
        "active_vdm_count = 1" => "active_vdm_count = 253",
        "actuator_count = 1" => "actuator_count = 277",
    )
        configuration = replace(configuration, old => new)
    end
    graph = prepare_graph_text(
        configuration;
        algorithms=(AdapterProgressiveResidualF32, algorithms()...),
    )
    identity = zeros(Float32, 253, 253)
    to_physical = zeros(Float32, 277, 253)
    from_physical = zeros(Float32, 253, 277)
    for index in 1:253
        identity[index, index] = 1.0f0
        to_physical[index, index] = 1.0f0
        from_physical[index, index] = 1.0f0
    end
    replace_parameters!(
        graph,
        Symbol("shape-to-hidden") => zeros(Float32, 1, 253),
        Symbol("hidden-to-shape") => zeros(Float32, 253, 1),
        Symbol("controller-to-vdm") => identity,
        Symbol("active-to-full") => identity,
        Symbol("vdm-to-pdm") => to_physical,
        Symbol("full-to-active") => identity,
        Symbol("pdm-to-vdm") => from_physical,
        Symbol("vdm-to-controller") => identity,
    )
    return graph
end

function progressive_feedback_buffer(row, sequence::Integer, terminal::Bool)
    header = BufferHeader(
        terminal ? SPA.META_HEADER_FLAG_MARKER : UInt32(0),
        UInt32(terminal ? 1 : 0),
        Int64(sequence),
        Int64(0),
        UInt64(sequence),
    )
    return native_callback_buffer(row; header, acquisition_sequence=UInt64(sequence))
end

function process_progressive_feedback!(node, row, demanded, input_storage, output_storage)
    callback = node.filter.callbacks.on_process
    GC.@preserve row demanded input_storage output_storage begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(input_storage)), UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(output_storage)), UInt32(1),
        )
        callback(node.filter, inputs, outputs)
    end
    return nothing
end

@testset "253-mode progressive feedback completes row blocks without allocation" begin
    graph = prepared_progressive_feedback_graph()
    node = PipeWireNode(
        graph; rate=1000//1, boundary_layout=RowMajorLayout(),
        connect=false, feedback=CALLBACK_FEEDBACK_PAIR,
    )
    owner = node.owner
    bridge = owner.feedback
    @test keys(bridge.public_input_bindings) == (:row,)
    @test keys(bridge.public_output_bindings) == (:demanded,)
    FilterGraphPipeWire._prepare_owner(owner)

    row = zeros(Float32, 253)
    demanded = zeros(Float32, 277)
    output_storage = native_callback_buffer(
        demanded; header_available=true, properties_available=true,
    )
    first(sequence) = progressive_feedback_buffer(row, sequence, false)
    terminal(sequence) = progressive_feedback_buffer(row, sequence, true)
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

    row[1] = 2.0f0
    first_zero = first(0)
    terminal_zero = terminal(0)
    GC.@preserve first_zero terminal_zero begin
        set_exposure_timing!(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(first_zero))),ExposureTiming(;duration_nanoseconds=100))
        set_exposure_timing!(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(terminal_zero))),ExposureTiming(;duration_nanoseconds=100))
    end
    @test native_callback_acquisition(first_zero) == native_callback_acquisition(terminal_zero)
    process_progressive_feedback!(node, row, demanded, first_zero, output_storage)
    @test !iszero(native_callback_flags(output_storage) & unavailable)
    @test iszero(native_callback_metadata_valid(output_storage))
    process_progressive_feedback!(node, row, demanded, terminal_zero, output_storage)
    @test demanded[1] ≈ 0.8f0 atol=2.0f-7
    @test native_callback_acquisition(output_storage) == native_callback_acquisition(terminal_zero)
    @test bridge.slot[1] ≈ 1.2f0 atol=2.0f-6

    row[1] = 0.0f0
    process_progressive_feedback!(node, row, demanded, first(2), output_storage)
    terminal_two = terminal(2)
    process_progressive_feedback!(node, row, demanded, terminal_two, output_storage)
    @test demanded[1] ≈ 0.8f0 atol=2.0f-6
    @test native_callback_acquisition(output_storage) == native_callback_acquisition(terminal_two)
    @test bridge.slot[1] ≈ 0.00388f0 atol=2.0f-5

    process_progressive_feedback!(node, row, demanded, first(3), output_storage)
    process_progressive_feedback!(node, row, demanded, terminal(3), output_storage)
    @test demanded[1] ≈ 0.792038f0 atol=2.0f-5
    first_four = first(4)
    terminal_four = terminal(4)
    @test @allocated(process_progressive_feedback!(
        node, row, demanded, first_four, output_storage,
    )) == 0
    @test @allocated(process_progressive_feedback!(
        node, row, demanded, terminal_four, output_storage,
    )) == 0
    close(node)
end

@testset "PipeWire owner delays clipped feedback by one successful sample" begin
    graph = prepared_callback_feedback_graph()
    @test_throws ArgumentError FilterGraphPipeWire.PreparedFeedbackBridge(
        graph, Symbol("shape-to-hidden"), Symbol("controller-constraint-feedback"), nothing,
    )
    @test_throws ArgumentError FilterGraphPipeWire.PreparedFeedbackBridge(
        graph, Symbol("constraint-feedback"), :missing, nothing,
    )
    @test_throws ArgumentError FilterGraphPipeWire.PreparedFeedbackBridge(
        graph, Symbol("constraint-feedback"), :demanded, nothing,
    )
    node = PipeWireNode(
        graph; rate=1000//1, connect=false, feedback=CALLBACK_FEEDBACK_PAIR,
    )
    owner = node.owner
    bridge = owner.feedback
    @test keys(bridge.public_input_bindings) == (Symbol("residual-error"),)
    @test keys(bridge.public_output_bindings) == (:demanded,)
    @test bridge.output_metadata_source_inputs == (1,)
    reordered = prepare_graph_text(
        replace(
            CALLBACK_FEEDBACK_GRAPH,
            "\"correction:residual-error\"\n        \"correction:constraint-feedback\"" =>
                "\"correction:constraint-feedback\"\n        \"correction:residual-error\"",
        );
        algorithms=algorithms(),
    )
    reordered_bridge = FilterGraphPipeWire.PreparedFeedbackBridge(
        reordered, Symbol("constraint-feedback"), Symbol("controller-constraint-feedback"), nothing,
    )
    @test FilterGraphs.output_metadata_source_inputs(reordered)[1] == 2
    @test reordered_bridge.output_metadata_source_inputs == (1,)
    @test only(bridge.slot) == 0.0f0
    FilterGraphPipeWire._prepare_owner(owner)
    @test only(bridge.slot) == 0.0f0

    residual = Float32[2]
    demanded = zeros(Float32, 1)
    native_inputs = native_callback_buffer(residual;
        header=BufferHeader(UInt32(0), UInt32(0), Int64(0), Int64(0), UInt64(37)),
        acquisition_sequence=UInt64(37))
    native_outputs = native_callback_buffer(
        demanded; header_available=true, properties_available=true,
    )
    callback = node.filter.callbacks.on_process
    function process_one!()
        GC.@preserve residual demanded native_inputs native_outputs begin
            inputs = NdArrayFilterBuffers{false}(
                Ptr{Cvoid}(pointer(native_inputs)), UInt32(1),
            )
            outputs = NdArrayFilterBuffers{true}(
                Ptr{Cvoid}(pointer(native_outputs)), UInt32(1),
            )
            callback(node.filter, inputs, outputs)
        end
        return demanded[1]
    end
    @test process_one!() ≈ 0.8f0 atol=2.0f-7
    @test native_callback_acquisition(native_outputs) == native_callback_acquisition(native_inputs)
    @test metadata_record(first(native_outputs.metadata),UInt32(SPA.META_EXPOSURE_TIMING)) !== nothing
    @test bridge.slot[1] ≈ 1.2f0 atol=2.0f-7
    residual[1] = 0.0f0
    @test process_one!() ≈ 0.8f0 atol=2.0f-7
    @test bridge.slot[1] ≈ 0.00388f0 atol=2.0f-6
    @test process_one!() ≈ 0.792038f0 atol=2.0f-5
    @test bridge.slot[1] ≈ 0.0f0 atol=2.0f-7
    @test @allocated(process_one!()) == 0

    residual[1] = NaN32
    @test_throws InvalidStateException process_one!()
    @test bridge.failed
    old_slot = only(bridge.slot)
    residual[1] = 0.0f0
    @test_throws InvalidStateException process_one!()
    @test only(bridge.slot) == old_slot
    @test node.filter.callbacks.on_reset(node.filter) === graph
    @test !bridge.failed
    @test only(bridge.slot) == 0.0f0
    residual[1] = 2.0f0
    @test process_one!() ≈ 0.8f0 atol=2.0f-7
    close(node)
end
