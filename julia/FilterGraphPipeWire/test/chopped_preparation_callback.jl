# The native callback owns publication; retained graph metadata alone does not.
@testset "labelled SCC preparation never publishes or supplies live pair history" begin
    node = CpuPreparationTests.chopped_preparation_node()
    samples = (
        CpuPreparationTests.chopped_preparation_sample(1, ChoppedImageFringed, 4f0, SccPixelResidualIncompletePair),
        CpuPreparationTests.chopped_preparation_sample(2, ChoppedImageUnfringed, 1f0, nothing),
        CpuPreparationTests.chopped_preparation_sample(3, ChoppedImageFringed, 6f0, nothing),
    )
    output = fill(73f0, 6)
    native_output = native_callback_buffer(output; header_available=true, properties_available=true)
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
    callback = node.filter.callbacks.on_process
    function live_frame(sequence, fringed, value)
        image = fill(value, 6)
        validity = fill(true, 6)
        header = BufferHeader(UInt32(0), UInt32(0), Int64(sequence-1)*10_000_000, Int64(0), UInt64(sequence))
        native_inputs = vcat(native_callback_buffer(image; header, acquisition_sequence=UInt64(sequence), exposure_duration_nanoseconds=UInt64(10_000_000),
                trigger_state=fringed ? ACQUISITION_TRIGGER_HIGH : ACQUISITION_TRIGGER_LOW,
                optical_classification=fringed ? OPTICAL_FRINGED : OPTICAL_UNFRINGED),
            native_callback_buffer(validity))
        GC.@preserve image validity native_inputs output native_output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_inputs)), UInt32(2))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(1))
            @test callback(node.filter, inputs, outputs) === nothing
            return iszero(native_callback_flags(native_output) & unavailable) ?
                FilterGraphPipeWire._buffer_metadata(outputs[1]) : nothing
        end
    end
    try
        @test FilterGraphPipeWire.prepare!(node; samples, required_outputs=(:measurements,)) === node
        @test node_state(node) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        @test node.owner.process_count == 0
        @test output == fill(73f0, 6)
        FilterGraphPipeWire._prepare_owner(node.owner)
        live_frame(1, true, 9f0)
        @test native_callback_flags(native_output) == unavailable
        @test output == fill(73f0, 6)
        completed = live_frame(2, false, 2f0)
        @test native_callback_flags(native_output) == UInt32(0)
        @test output == fill(7f0, 6)
        @test completed.value.acquisition_identity.sequence == 2
        @test completed.value.acquisition_identity.generation == 7
        @test native_callback_header(native_output).pts == 10_000_000
        # Missing window 3 must suppress a new pair despite alternating roles.
        live_frame(4, false, 3f0)
        @test native_callback_flags(native_output) == unavailable
        @test output == fill(7f0, 6)
        completed = live_frame(5, true, 10f0)
        @test native_callback_flags(native_output) == UInt32(0)
        @test completed.value.acquisition_identity.sequence == 5
        @test output == fill(7f0, 6)
        FilterGraphs.reset!(node.owner.graph)
        live_frame(6, false, 1f0)
        @test native_callback_flags(native_output) == unavailable
        @test last_error(node) === nothing
        @test node.owner.active
    finally
        close(node)
    end
end
