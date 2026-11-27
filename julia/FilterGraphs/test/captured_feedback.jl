include(joinpath(@__DIR__, "data", "support", "captured_feedback.jl"))

struct FeedbackTestDevice end
FilterGraphs._prepare_captured_context(::FeedbackTestDevice) =
    FilterGraphs._prepare_captured_context(TestCapturedDevice())
FilterGraphs._require_captured_feedback_support(::FeedbackTestDevice, feedback) = nothing

mutable struct FeedbackTestAdapter{D,O,I,S}
    outputs::O
    inputs::I
    status::S
    node_index::Int32
    adoptions::Int
end

for algorithm in (CapturedFeedbackSumF32, CapturedFeedbackGateF32)
    @eval FilterGraphAlgorithms.capture_support(::Type{$algorithm}, ::FeedbackTestDevice) = SupportsCapture()
end

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::FeedbackTestDevice, ::FilterGraphAlgorithms.PreparedAlgorithm{D}, outputs::Tuple, inputs::Tuple, status, node_index::Int32,
) where {D<:Union{CapturedFeedbackSumF32,CapturedFeedbackGateF32}}
    return FeedbackTestAdapter{D,typeof(outputs),typeof(inputs),typeof(status)}(
        outputs, inputs, status, node_index, 0,
    )
end

FilterGraphAlgorithms.preflight(::FeedbackTestAdapter) = nothing
function FilterGraphAlgorithms.enqueue!(adapter::FeedbackTestAdapter{CapturedFeedbackSumF32})
    iszero(adapter.status[1]) || return nothing
    sum, feedback = adapter.outputs
    prior, drive = adapter.inputs
    for index in eachindex(sum, feedback, prior, drive)
        sum[index] = drive[index] + prior[index]
        feedback[index] = 0.5f0 * sum[index]
    end
    adapter.adoptions += 1
    return nothing
end
function FilterGraphAlgorithms.enqueue!(adapter::FeedbackTestAdapter{CapturedFeedbackGateF32})
    iszero(adapter.status[1]) || return nothing
    input = only(adapter.inputs)
    if all(value -> isfinite(value) && value > 0.0f0, input)
        copyto!(only(adapter.outputs), input)
    else
        adapter.status[1] = adapter.node_index
    end
    iszero(adapter.status[1]) && (adapter.adoptions += 1)
    return nothing
end
FilterGraphAlgorithms.complete!(::FeedbackTestAdapter{CapturedFeedbackSumF32}, ::Int32) = nothing
FilterGraphAlgorithms.complete!(adapter::FeedbackTestAdapter{CapturedFeedbackGateF32}, status::Int32) =
    status == adapter.node_index ? CapturedFeedbackRejected : nothing
function FilterGraphAlgorithms.reset!(adapter::FeedbackTestAdapter)
    adapter.adoptions = 0
    return nothing
end

struct FeedbackTestBinding{D,S,F}
    destination::D
    source::S
    status::F
end
function FilterGraphs._prepare_captured_feedback(
    ::FeedbackTestDevice, feedback, inputs, outputs, status,
)
    return map(keys(feedback)) do input
        FeedbackTestBinding(inputs[input], outputs[feedback[input]], status.value)
    end
end
function FilterGraphs._enqueue_captured_feedback!(binding::FeedbackTestBinding)
    iszero(binding.status[1]) && copyto!(binding.destination, binding.source)
    return nothing
end
FilterGraphs._reset_captured_feedback!(binding::FeedbackTestBinding) =
    (fill!(binding.destination, 0.0f0); nothing)

struct FeedbackTestExecutable{S,F,N,B}
    capture_status::S
    frame_status::F
    nodes::N
    feedback::B
end
function FilterGraphs._record_captured_graph(
    ::TestCapturedContext, status, nodes::Tuple,
    feedback::Tuple{<:FeedbackTestBinding,Vararg{FeedbackTestBinding}},
)
    return FeedbackTestExecutable(status,
        FilterGraphs.CapturedFrameOutcome(status.value, status.host), nodes, feedback)
end
function FilterGraphs._launch_captured_graph!(context::TestCapturedContext, executable::FeedbackTestExecutable)
    context.fail_launch && error("injected feedback launch failure")
    FilterGraphs._enqueue_captured_frame!(executable.capture_status, executable.nodes)
    FilterGraphs._enqueue_captured_feedback!(executable.feedback)
    return nothing
end
FilterGraphs._reset_captured_feedback!(executable::FeedbackTestExecutable) =
    FilterGraphs._reset_captured_feedback!(executable.feedback)

@algorithm CapturedFeedbackSourceF32 begin
    label = "captured-feedback-source-f32"
    @configuration CapturedFeedbackSourceF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        output(:generated, Float32, (configuration.extent,), "test.feedback.f32/1"),
    )
    prepare(configuration) = nothing
    process!(plan) = begin
        fill!(generated, 42.0f0)
        nothing
    end
end

struct FeedbackTestSource{O}
    output::O
end
FilterGraphAlgorithms.capture_support(::Type{CapturedFeedbackSourceF32}, ::TestCapturedDevice) = SupportsCapture()
function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition,
    ::TestCapturedDevice, ::FilterGraphAlgorithms.PreparedAlgorithm{CapturedFeedbackSourceF32},
    outputs::Tuple, inputs::Tuple, status, node_index::Int32,
)
    return FeedbackTestSource(only(outputs))
end
FilterGraphAlgorithms.preflight(::FeedbackTestSource) = nothing
FilterGraphAlgorithms.enqueue!(adapter::FeedbackTestSource) = (fill!(adapter.output, 42.0f0); nothing)
FilterGraphAlgorithms.complete!(::FeedbackTestSource, ::Int32) = nothing
FilterGraphAlgorithms.reset!(adapter::FeedbackTestSource) = (fill!(adapter.output, 0.0f0); nothing)

@testset "default captured source Graph preserves empty public interfaces" begin
    source = prepare_graph_text("""
        filter.graph = {
            nodes = [ { type = ndarray name = source label = captured-feedback-source-f32 config = { extent = 3 } } ]
            inputs = []
            outputs = [ "source:generated" ]
        }
        """; algorithms=(CapturedFeedbackSourceF32,))
    captured = prepare_captured_graph(source, TestCapturedDevice())
    @test isempty(captured.input_formats)
    @test isempty(captured.device_inputs)
    @test captured.output_metadata_sources == (0,)
    @test captured.executable isa TestCapturedExecutable
    output = fill(91.0f0, 3)
    @test process!((generated=output,), captured, NamedTuple()) == (generated=true,)
    @test output == fill(42.0f0, 3)
    @test FilterGraphs.output_metadata(captured) == (generated=nothing,)
    @test reset!(captured) === captured
    fill!(output, 92.0f0)
    @test process!((generated=output,), captured, NamedTuple()) == (generated=true,)
    @test output == fill(42.0f0, 3)

    # Ordinary prepare_graph requires an external output, but custom prebuilt
    # PreparedGraphs historically permit an empty output interface.
    fields = NamedTuple{fieldnames(typeof(source))}(
        ntuple(index -> getfield(source, index), fieldcount(typeof(source))))
    empty_outputs = merge(fields, (
        output_routes=(), output_formats=NamedTuple(), output_conditional=NamedTuple(),
        published_metadata=NamedTuple(),
        result_types=FilterGraphs._graph_result_types(NamedTuple(), source.node_failure_types),
    ))
    empty_source = FilterGraphs.PreparedGraph(values(empty_outputs)...)
    empty_captured = prepare_captured_graph(empty_source, TestCapturedDevice())
    @test isempty(empty_captured.output_formats)
    @test process!(NamedTuple(), empty_captured, NamedTuple()) == NamedTuple()
    @test reset!(empty_captured) === empty_captured
end

@testset "captured feedback validates its public Port projection before preparation" begin
    source = captured_feedback_graph()
    for kwargs in (
        (; outputs=()), (; outputs=(:sum, :sum)), (; outputs=(:unknown,)),
        (; feedback=(unknown=:sum,)), (; feedback=(feedback=:unknown,)),
        (; feedback=(feedback="sum",)),
        (; feedback=(feedback=:sum, drive=:sum), outputs=(:sum,)),
        (; feedback=CAPTURED_FEEDBACK_BINDING),
    )
        @test_throws GraphConfigurationError prepare_captured_graph(source, FeedbackTestDevice(); kwargs...)
    end
    @test_throws GraphConfigurationError prepare_captured_graph(source, TestCapturedDevice();
        feedback=CAPTURED_FEEDBACK_BINDING, outputs=(:sum,))

    formats = source.input_formats
    format = source.output_formats.sum
    original = (; format.element_type, format.shape, format.layout, format.schema, format.rate)
    for (field, replacement) in (
        (:element_type, Float64), (:shape, (4,)), (:layout, nothing),
        (:schema, "test.other/1"), (:rate, (2, 1)),
    )
        changed = merge(original, NamedTuple{(field,)}((replacement,)))
        @test_throws GraphConfigurationError FilterGraphs._captured_feedback_ports(
            formats, (sum=changed,), (2,), (feedback=:sum,), (:sum,),
        )
    end
    ports = FilterGraphs._captured_feedback_ports(
        formats, source.output_formats, (2, 1, 2), CAPTURED_FEEDBACK_BINDING, (:checked, :sum),
    )
    @test ports.inputs == (:drive,)
    @test ports.outputs == (:checked, :sum)
    @test ports.metadata_sources == (1, 1)
end

@testset "captured feedback keeps metadata, local adoption and failure recovery" begin
    captured = prepare_captured_graph(captured_feedback_graph(), FeedbackTestDevice();
        feedback=CAPTURED_FEEDBACK_BINDING, outputs=(:checked, :sum))
    @test keys(captured.input_formats) == (:drive,)
    @test keys(captured.device_inputs) == (:drive,)
    @test keys(captured.output_formats) == (:checked, :sum)
    @test keys(captured.host_outputs) == (:checked, :sum)
    @test captured.output_metadata_sources == (1, 1)
    binding = only(captured.executable.feedback)
    destination = binding.destination
    @test all(iszero, destination)
    output = (checked=fill(99.0f0, 3), sum=fill(98.0f0, 3))
    drive = zeros(Float32, 3)
    inputs = (drive=drive,)
    for (sequence, value, expected) in ((1, 2, 2), (2, 4, 5), (3, -10, -7.5), (4, 1, 3.5))
        fill!(drive, Float32(value))
        metadata = SampleMetadata(sequence;reference_timestamp=SampleReferenceTimestamp(1_000_000_000+sequence))
        if sequence == 3
            prior = copy(destination)
            published = FilterGraphs.output_metadata(captured)
            @test process!(output, captured, inputs, (drive=metadata,)) === CapturedFeedbackRejected
            @test output.sum == fill(5.0f0, 3)
            @test destination == prior
            @test FilterGraphs.output_metadata(captured) == published
        else
            @test process!(output, captured, inputs, (drive=metadata,)) == (checked=true, sum=true)
            @test output.sum == fill(Float32(expected), 3)
            @test output.checked == output.sum
            @test FilterGraphs.output_metadata(captured) == (checked=metadata, sum=metadata)
        end
        @test captured.nodes[1].adapter.adoptions == sequence
    end
    @test reset!(captured) === captured
    @test binding.destination === destination
    @test all(iszero, destination)
    fill!(drive, 1.0f0)
    @test process!(output, captured, inputs) == (checked=true, sum=true)
    @test output.sum == ones(Float32, 3)
    @test (@allocated process!(output, captured, inputs)) == 0

    # Device-success commit precedes publication. A later transport fault must
    # stop replay until reset, rather than pretend that feedback rolled back.
    expected_feedback = 0.5f0 .* (drive .+ destination)
    held_output = copy(output.sum)
    captured.context.fail_copy_from_at = 1
    @test_throws CapturedGraphFault process!(output, captured, inputs)
    @test captured_graph_state(captured) === CapturedGraphFaulted
    @test destination == expected_feedback
    @test output.sum == held_output
    @test_throws CapturedGraphFault process!(output, captured, inputs)
    captured.context.fail_copy_from_at = 0
    @test reset!(captured) === captured
    @test all(iszero, destination)

    selected = prepare_captured_graph(prepare_graph_text(CAPTURED_MULTI_OUTPUT_GRAPH;
        algorithms=(CapturedSplitF32,)), TestCapturedDevice(); outputs=(:second,))
    @test selected.executable isa TestCapturedExecutable
    selected_output = (second=zeros(Float32, 3),)
    @test process!(selected_output, selected, (input=ones(Float32, 3),)) == (second=true,)
    @test selected_output.second == -ones(Float32, 3)
end
