# Included after the native buffer helpers and delayed-feedback graph builder.
mutable struct FrameProbe{N}
    frames::Int
    invalidations::Int
    resets::Int
    names::NTuple{N,Symbol}
    metadata::@NamedTuple{sequence::UInt64, offset::UInt32, timestamp_nanoseconds::Int64,
        discontinuity::Bool, corrupted::Bool, terminal::Bool, identity_sequence::UInt64}
    metadata_present::NTuple{N,Bool}
    has_metadata::Bool
    value::Float32
    fail_phase::Symbol
end
FrameProbe() = FrameProbe(0, 0, 0, (:unobserved,), (sequence=UInt64(0), offset=UInt32(0),
    timestamp_nanoseconds=Int64(0), discontinuity=false, corrupted=false, terminal=false,
    identity_sequence=UInt64(0)), (false,), false, 0f0, :none)
@inline function (probe::FrameProbe)(::Val{:frame}, outputs::NamedTuple, metadata::NamedTuple)
    probe.frames += 1
    probe.names = keys(outputs)
    record = first(values(metadata))
    probe.metadata_present = map(value -> value.present, values(metadata))
    probe.has_metadata = record.present
    if record.present
        sample = record.value
        identity = sample_acquisition_identity(sample)
        probe.metadata = (sequence=sample.sequence, offset=sample.offset,
            timestamp_nanoseconds=sample.timestamp_nanoseconds,
            discontinuity=sample.discontinuity, corrupted=sample.corrupted, terminal=sample.terminal,
            identity_sequence=identity === nothing ? UInt64(0) : identity.sequence)
    end
    probe.value = first(first(values(outputs)))
    probe.fail_phase === :frame && error("frame observation failed")
    return nothing
end
function (probe::FrameProbe)(::Val{:invalidated})
    probe.invalidations += 1
    probe.fail_phase === :invalidated && error("invalidation observation failed")
    return nothing
end
function (probe::FrameProbe)(::Val{:reset})
    probe.resets += 1
    probe.fail_phase === :reset && error("reset observation failed")
    return nothing
end

function observer_frame!(node, input, output, native_input, native_output)
    GC.@preserve input output native_input native_output begin
        inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_input)), UInt32(length(native_input)))
        outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(length(native_output)))
        return node.filter.callbacks.on_process(node.filter, inputs, outputs)
    end
end
function observer_properties(node)
    return Dict(node.filter.callbacks.on_get_properties(node.filter).values)
end
observer_state(node) = PipeWireAO.pod_value(String,
    observer_properties(node)["pipewireao.frame-observer.state"])

@testset "Local Graph output leaves native feedback publication independent" begin
    receipt = Ref{Union{Nothing,Tuple{Float32,SampleMetadata}}}(nothing)
    failed = Ref(false)
    function observe(::Val{:frame}, array, metadata)
        failed[] && error("display handoff failed")
        receipt[] = (only(array), metadata.value)
        return nothing
    end
    observe(::Val{:reset}) = (receipt[] = nothing)
    for callback in (observe, nothing)
        graph = prepared_callback_feedback_graph(; extra_output="physical:requested-pdm-command")
        probe = FrameProbe()
        port = Symbol("requested-pdm-command")
        node = PipeWireNode(graph; rate=1000//1, connect=false, feedback=CALLBACK_FEEDBACK_PAIR,
            local_output=port => callback, frame_observer=probe)
        input, output = Float32[2], zeros(Float32, 1)
        header = BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(123), Int64(0), UInt64(19))
        native_input = native_callback_buffer(input; header, acquisition_sequence=UInt64(83))
        native_output = native_callback_buffer(output; header_available=true, properties_available=true)
        try
            @test haskey(FilterGraphs.output_formats(graph), port)
            @test keys(node.owner.feedback.public_output_bindings) === (:demanded,)
            @test FilterGraphPipeWire.prepare!(node; required_outputs=(:demanded,)) === node
            @test receipt[] === nothing
            FilterGraphPipeWire._prepare_owner(node.owner)
            observer_frame!(node, input, output, native_input, native_output)
            @test only(output) == 0.8f0
            @test native_callback_flags(native_output) == UInt32(0)
            @test only(node.owner.feedback.slot) ≈ 1.2f0
            @test probe.names === (:demanded,)
            if callback === nothing
                @test receipt[] === nothing
            else
                value, sample = receipt[]
                @test value === 2f0
                @test sample.acquisition_identity.sequence == 83
                @test sample.timestamp_nanoseconds == 123
            end
            @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
            accepted = receipt[]
            node.filter.callbacks.on_deactivate(node.filter)
            @test !node.owner.active
            @test receipt[] === accepted
            node.filter.callbacks.on_reset(node.filter)
            @test receipt[] === nothing
            @test all(iszero, node.owner.feedback.slot)
            FilterGraphPipeWire._prepare_owner(node.owner)
            failed[] = true
            observer_frame!(node, input, output, native_input, native_output)
            @test native_callback_flags(native_output) == UInt32(0)
            @test only(output) == 0.8f0
            @test node.owner.local_output.failed == (callback !== nothing)
        finally
            close(node)
        end
    end
end

@testset "Frame observation receives public values and actual metadata" begin
    probe = FrameProbe()
    node = PipeWireNode(prepared_conditional_graph(); connect=false, frame_observer=probe)
    input, output = Float32[3, 5], zeros(Float32, 2)
    flags = SPA.META_HEADER_FLAG_MARKER | SPA.META_HEADER_FLAG_DISCONT |
        SPA.META_HEADER_FLAG_CORRUPTED
    header = BufferHeader(flags, UInt32(6), Int64(987654), Int64(0), UInt64(81))
    native_input = native_callback_buffer(input; header, acquisition_sequence=UInt64(81))
    native_output = native_callback_buffer(output; header_available=true, properties_available=true)
    try
        @test observer_state(node) == "active"
        @test FilterGraphPipeWire.prepare!(node) === node
        @test probe.frames == 0 # Synthetic preparation is not a source observation.
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; execution=:row_block)
        FilterGraphPipeWire._prepare_owner(node.owner)
        @test observer_frame!(node, input, output, native_input, native_output) === nothing
        @test output == input
        @test probe.frames == 1 && probe.invalidations == 0
        @test probe.names === (:output,)
        @test probe.value == input[1]
        @test probe.has_metadata
        @test probe.metadata.sequence == 81 && probe.metadata.offset == 6
        @test probe.metadata.timestamp_nanoseconds == 987654
        @test probe.metadata.discontinuity && probe.metadata.corrupted && probe.metadata.terminal
        @test probe.metadata.identity_sequence == 81
        @test probe.metadata.sequence == only(values(FilterGraphs.output_metadata(node.owner.graph))).sequence
        @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
        # Sequence jumps and absent metadata are preserved, never reconstructed.
        later = native_callback_buffer(input; header=BufferHeader(SPA.META_HEADER_FLAG_MARKER,
            UInt32(0), Int64(999999), Int64(0), UInt64(99)))
        observer_frame!(node, input, output, later, native_output)
        @test probe.metadata.sequence == 99
        partial = native_callback_buffer(input; header=BufferHeader(UInt32(0), UInt32(0),
            Int64(1000000), Int64(0), UInt64(100)))
        before = probe.frames
        observer_frame!(node, input, output, partial, native_output)
        @test probe.frames == before && probe.invalidations == 1
        @test native_callback_flags(native_output) ==
            PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
        @test @allocated(observer_frame!(node, input, output, partial, native_output)) == 0
        @test probe.invalidations == 2
        resets = probe.resets
        FilterGraphPipeWire._reset_owner!(node.owner)
        @test probe.resets == resets + 1
        @test observer_state(node) == "active"
    finally
        close(node)
    end
end

@testset "Observation never repairs absent metadata or rejected science" begin
    probe = FrameProbe()
    node = PipeWireNode(prepared_failure_graph(); connect=false, frame_observer=probe)
    input, output = Float32[7, 11], zeros(Float32, 2)
    native_input, native_output = native_callback_buffer(input), native_callback_buffer(output)
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        observer_frame!(node, input, output, native_input, native_output)
        @test probe.frames == 1 && !probe.has_metadata
        @test output == input
        @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
        input[1] = -1f0
        before = copy(output)
        observer_frame!(node, input, output, native_input, native_output)
        @test output == before
        @test probe.frames == 2 && probe.invalidations == 1
        @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
        # Processing exceptions invalidate evidence as well as ordinary rejection.
        GC.@preserve input output native_input native_output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(0))
            @test_throws DimensionMismatch node.filter.callbacks.on_process(node.filter, inputs, outputs)
        end
        @test probe.invalidations == 3
        @test observer_state(node) == "active"
    finally
        close(node)
    end
end

@testset "Delayed-feedback observer excludes private ports" begin
    probe = FrameProbe()
    node = PipeWireNode(prepared_callback_feedback_graph(); rate=1000//1,
        connect=false, feedback=CALLBACK_FEEDBACK_PAIR, frame_observer=probe)
    input, output = Float32[2], zeros(Float32, 1)
    header = BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(123), Int64(0), UInt64(19))
    native_input = native_callback_buffer(input; header)
    native_output = native_callback_buffer(output; header_available=true)
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        observer_frame!(node, input, output, native_input, native_output)
        @test probe.names === (:demanded,)
        @test probe.frames == 1 && probe.invalidations == 0
        @test probe.value == only(output) == 0.8f0
        @test probe.metadata.sequence == 19
        @test probe.metadata.timestamp_nanoseconds == 123
        @test !iszero(only(node.owner.feedback.slot))
        @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
        FilterGraphPipeWire._reset_owner!(node.owner)
        @test probe.resets == 1
        @test all(iszero, node.owner.feedback.slot)
        input[1] = NaN32
        @test_throws InvalidStateException observer_frame!(node, input, output, native_input, native_output)
        @test probe.invalidations == 1
    finally
        close(node)
    end
end

@testset "Observer failures latch without changing accepted outputs or retrying" begin
    for phase in (:frame, :invalidated, :reset)
        probe = FrameProbe()
        probe.fail_phase = phase
        node = PipeWireNode(prepared_failure_graph(); connect=false, frame_observer=probe)
        input, output = Float32[7, 11], zeros(Float32, 2)
        native_input, native_output = native_callback_buffer(input), native_callback_buffer(output)
        try
            FilterGraphPipeWire._prepare_owner(node.owner)
            if phase === :invalidated
                input[1] = -1f0
                observer_frame!(node, input, output, native_input, native_output)
                input[1] = 7f0
            elseif phase === :reset
                FilterGraphPipeWire._reset_owner!(node.owner)
            else
                @test_logs observer_frame!(node, input, output, native_input, native_output)
                @test output == input
                @test native_callback_flags(native_output) == UInt32(0)
            end
            @test observer_state(node) == "failed"
            counts = (probe.frames, probe.invalidations, probe.resets)
            observer_frame!(node, input, output, native_input, native_output)
            @test output == input
            @test native_callback_flags(native_output) == UInt32(0)
            FilterGraphPipeWire._reset_owner!(node.owner)
            @test (probe.frames, probe.invalidations, probe.resets) == counts
            @test observer_state(node) == "failed"
            @test last_error(node) === nothing
            @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
        finally
            close(node)
        end
    end
    # Ordinary writes through either borrowed layout are refused before mutation.
    mutation(phase, outputs, metadata) = (outputs.output[1] = -100f0)
    for layout in (ColumnMajorLayout(), RowMajorLayout())
        node = PipeWireNode(prepared_failure_graph(); connect=false,
            boundary_layout=layout, frame_observer=mutation)
        input, output = Float32[7, 11], zeros(Float32, 2)
        native_input, native_output = native_callback_buffer(input), native_callback_buffer(output)
        try
            FilterGraphPipeWire._prepare_owner(node.owner)
            observer_frame!(node, input, output, native_input, native_output)
            @test output == input && observer_state(node) == "failed"
        finally
            close(node)
        end
    end
end

@testset "Mixed present and missing metadata stays exact" begin
    graph = prepare_graph_text("""
        filter.graph = {
            nodes = [
                { type=ndarray name=left label=adapter-metadata-source-f32 config={ extent=2 } }
                { type=ndarray name=right label=adapter-metadata-source-f32 config={ extent=2 } }
            ]
            links = []
            inputs = [ { name=a endpoint="left:left" } { name=b endpoint="left:right" }
                { name=c endpoint="right:left" } { name=d endpoint="right:right" } ]
            outputs = [ { name=left endpoint="left:output" } { name=right endpoint="right:output" } ]
        }
        """; algorithms=(AdapterMetadataSourceF32,))
    initial = FrameProbe()
    probe = FrameProbe(0, 0, 0, (:left, :right), initial.metadata, (false, false), false, 0f0, :none)
    node = PipeWireNode(graph; connect=false, frame_observer=probe)
    input = (Float32[7, 11], Float32[13, 17], Float32[19, 23], Float32[29, 31])
    output = (zeros(Float32, 2), zeros(Float32, 2))
    header = BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(123), Int64(0), UInt64(19))
    native_input = vcat(native_callback_buffer(input[1]), native_callback_buffer(input[2]; header),
        native_callback_buffer(input[3]), native_callback_buffer(input[4];header_available=false))
    native_output = vcat(native_callback_buffer(output[1]; header_available=true),
        native_callback_buffer(output[2]; header_available=true))
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        observer_frame!(node, input, output, native_input, native_output)
        @test probe.frames == 1 && probe.names === (:left, :right)
        @test probe.has_metadata && probe.metadata.sequence == 19
        @test probe.metadata_present === (true, false)
        @test output == (input[1] + input[2], input[3] + input[4])
        @test FilterGraphs.output_metadata(graph).right === nothing
        @test @allocated(observer_frame!(node, input, output, native_input, native_output)) == 0
    finally
        close(node)
    end
end

@testset "Default and unsupported owner boundaries" begin
    @test Base.ispublic(FilterGraphAlgorithms, :AdmittedSampleMetadata)
    @test !Base.isexported(FilterGraphAlgorithms, :AdmittedSampleMetadata)
    @test Base.Docs.hasdoc(FilterGraphAlgorithms, :AdmittedSampleMetadata)
    @test isbitstype(FilterGraphAlgorithms.AdmittedSampleMetadata)
    node = PipeWireNode(prepared_failure_graph(); connect=false)
    try
        @test node.owner.frame_observer === nothing
        @test !haskey(observer_properties(node), "pipewireao.frame-observer.state")
    finally
        close(node)
    end
    @test_throws ArgumentError PipeWireNode(prepared_failure_graph(); connect=false,
        defer_first_input=true, frame_observer=FrameProbe())
    join = FilterGraphs.prepare_latest_hold(prepared_latest_hold_graph(), :primary, :held)
    @test_throws ArgumentError FilterGraphPipeWire.PreparedGraphOwner(join.graph, join;
        frame_observer=FrameProbe())
end

mutable struct FrameNotificationCounter
    notifications::Int
end
PipeWireAO.filter_state(::FrameNotificationCounter) = PipeWireAO.NDARRAY_FILTER_STATE_PAUSED
function PipeWireAO.notify_properties!(counter::FrameNotificationCounter)
    counter.notifications += 1
    return nothing
end

@testset "Observer publication is bounded and preserves sample evidence" begin
    probe = FrameProbe()
    node = PipeWireNode(prepared_failure_graph(); connect=false, frame_observer=probe)
    counter = FrameNotificationCounter(0)
    owner = node.owner
    notify!(changed, now) = FilterGraphPipeWire._notify_graph_properties!(owner, counter, changed, now)
    input, output = Float32[7, 11], zeros(Float32, 2)
    native_input, native_output = native_callback_buffer(input), native_callback_buffer(output)
    try
        FilterGraphPipeWire._prepare_owner(owner)
        observer_frame!(node, input, output, native_input, native_output)
        baseline = UInt64(1_000_000_000)
        owner.recovery_properties_notified_timestamp_ns = baseline
        notify!(false, baseline + UInt64(99_999_999))
        @test counter.notifications == 0
        notify!(false, baseline + UInt64(100_000_000))
        @test counter.notifications == 1
        notify!(false, baseline + UInt64(200_000_000))
        @test counter.notifications == 1 # An unchanged snapshot does not republish.
        @test @allocated(notify!(false, baseline + UInt64(200_000_000))) == 0
        observer_frame!(node, input, output, native_input, native_output)
        notify!(false, baseline + UInt64(199_999_999))
        @test counter.notifications == 1
        notify!(false, baseline + UInt64(200_000_000))
        @test counter.notifications == 2
        @test !probe.has_metadata # Notification never manufactures source facts.
        @test FilterGraphPipeWire._recovery_properties_interval(owner) == 0
        FilterGraphPipeWire._frame_observer_event!(owner.frame_observer, Val(:invalidated))
        @test @allocated(notify!(false, baseline + UInt64(300_000_000))) == 0
        @test counter.notifications == 3
        # A newly failed observer requests immediate publication through the full callback.
        probe.fail_phase = :frame
        GC.@preserve input output native_input native_output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(1))
            owner.recovery_properties_notified_timestamp_ns = time_ns()
            FilterGraphPipeWire._process_graph_callback!(owner, counter, inputs, outputs)
        end
        @test counter.notifications == 4
        @test observer_state(node) == "failed" && output == input
        @test !FilterGraphPipeWire._frame_observer_pending(owner.frame_observer)
    finally
        close(node)
    end
end
