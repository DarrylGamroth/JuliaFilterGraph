prepared_cred2_ingress_graph() = prepare_graph(
    joinpath(@__DIR__, "data", "cred2-ingress.conf");
    algorithms=FilterGraphAlgorithms.algorithms(),
)

struct TestCred2Ingress{P}
    polarity::P
end
function (ingress::TestCred2Ingress)(metadata::SampleMetadata, raw)
    decoded = cred2_trigger_metadata(metadata, raw)
    decoded isa SampleMetadata || return decoded
    isnothing(ingress.polarity) && return decoded
    return chopped_image_metadata(decoded; trigger_high_is_fringed=ingress.polarity)
end

@testset "C-RED 2 native ingress before calibration" begin
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
    for polarity in (true, false)
        graph = prepared_cred2_ingress_graph()
        node = PipeWireNode(graph; connect=false, boundary_layout=RowMajorLayout(),
            ingress=(raw=TestCred2Ingress(polarity),))
        try
            FilterGraphPipeWire._prepare_owner(node.owner)
            callback = node.filter.callbacks.on_process
            raw = fill(UInt16(100), 8)
            # Physical word 4 differs from column-major linear index 4 of the
            # corresponding logical matrix, detecting accidental index order.
            raw[4] = polarity ? 0x3fff : 0x3ffe
            raw[2] = 104
            calibrated = fill(-99.0f0, 8)
            command = fill(-99.0f0, 1)
            header = BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(1000), Int64(0), UInt64(1))
            duration = UInt64(graph.nodes[2].prepared.plan.calibration.exposure_duration_nanoseconds)
            input_storage = native_callback_buffer(raw; header, acquisition_sequence=UInt64(1),
                exposure_duration_nanoseconds=duration)
            calibrated_storage = native_callback_buffer(calibrated; header_available=true, properties_available=true)
            command_storage = native_callback_buffer(command; header_available=true, properties_available=true)
            output_storage = vcat(calibrated_storage, command_storage)
            GC.@preserve raw calibrated command input_storage output_storage begin
                inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input_storage)), UInt32(1))
                outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output_storage)), UInt32(2))
                @test callback(node.filter, inputs, outputs) === nothing
                @test all(index -> native_callback_flags(output_storage, index) == 0, 1:2)
                @test calibrated[4] ≈ (Float32(raw[4]) - 100.0f0) * 0.01f0
                @test calibrated[2] ≈ 0.04f0
                published = FilterGraphs.output_metadata(graph).command
                @test sample_trigger_high(published) === polarity
                @test sample_chopped_image_kind(published) == ChoppedImageFringed
                @test sample_acquisition_identity(published) == SampleAcquisitionIdentity(ntuple(i -> UInt8(i), 16), 7, 1)
                @test sample_exposure_duration_nanoseconds(published) === duration
                @test command == [0.0f0]
                state = graph.nodes[2].prepared.workspace.state
                identity = state.committed_identity
                committed = copy(state.committed_command)
                before_metadata = FilterGraphs.output_metadata(graph)
                fill!(calibrated, -99.0f0)
                fill!(command, -99.0f0)
                set_buffer_header!(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage))),BufferHeader(header.flags,header.offset,header.pts,header.dts_offset,UInt64(2)))
                PipeWireAO.set_acquisition_identity!(
                    PipeWireAO.metadata_records(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage)))),
                    AcquisitionIdentity(AcquisitionDomain(ntuple(i -> UInt8(i), 16)), UInt64(7), UInt64(2)))
                PipeWireAO.set_exposure_timing!(
                    metadata_records(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage)))),ExposureTiming(;duration_nanoseconds=duration))
                raw[4] = 0x0001
                callback(node.filter, inputs, outputs)
                @test @allocated(callback(node.filter, inputs, outputs)) == 0
                @test all(index -> native_callback_flags(output_storage, index) == unavailable, 1:2)
                @test calibrated == fill(-99.0f0, 8)
                @test command == [-99.0f0]
                @test state.committed_identity == identity
                @test state.committed_command == committed
                @test FilterGraphs.output_metadata(graph) == before_metadata
                # Retry the next identity after rejection; a rejection must not
                # consume its graph state or its acquisition identity.
                raw[4] = polarity ? 0x3ffe : 0x3fff
                raw[2] = 100
                @test callback(node.filter, inputs, outputs) === nothing
                @test all(index -> native_callback_flags(output_storage, index) == 0, 1:2)
                @test command ≈ [-0.01f0]
                @test state.committed_identity.sequence == 2
                # Advance native identity before measuring a valid correction.
                raw[4] = polarity ? 0x3fff : 0x3ffe
                raw[2] = 104
                set_buffer_header!(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage))),BufferHeader(header.flags,header.offset,header.pts,header.dts_offset,UInt64(3)))
                PipeWireAO.set_acquisition_identity!(
                    PipeWireAO.metadata_records(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage)))),
                    AcquisitionIdentity(AcquisitionDomain(ntuple(i -> UInt8(i), 16)), UInt64(7), UInt64(3)))
                PipeWireAO.set_exposure_timing!(
                    metadata_records(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(input_storage)))),ExposureTiming(;duration_nanoseconds=duration))
                @test @allocated(callback(node.filter, inputs, outputs)) == 0
                @test state.committed_identity.sequence == 3
                @test command ≈ [-0.015f0]
                @test all(index -> native_callback_flags(output_storage, index) == 0, 1:2)
            end
        finally
            close(node)
        end
    end
end

@testset "C-RED 2 ingress contract restrictions and electrical annotation" begin
    graph = prepared_cred2_ingress_graph()
    callback = TestCred2Ingress(nothing)
    @test_throws ArgumentError PipeWireNode(graph; connect=false, ingress=(unknown=callback,))
    @test_throws ArgumentError PipeWireNode(graph; connect=false, ingress=(raw=callback,), defer_first_input=true)
    # Use a graph with a compatible feedback Port pair to reach ingress validation.
    leaky = prepared_leaky_graph()
    @test_throws ArgumentError PipeWireNode(leaky; connect=false,
        feedback=:input => :output, ingress=(input=callback,))
    graph = prepare_graph(joinpath(@__DIR__, "data", "cred2-calibration.conf");
        algorithms=FilterGraphAlgorithms.algorithms())
    node = PipeWireNode(graph; connect=false, ingress=(raw=callback,))
    try
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; execution=:row_block)
        FilterGraphPipeWire._prepare_owner(node.owner)
        process_callback = node.filter.callbacks.on_process
        raw = fill(UInt16(100), 8)
        raw[4] = 0x3fff
        calibrated = zeros(Float32, 8)
        header = BufferHeader(SPA.META_HEADER_FLAG_MARKER, UInt32(0), Int64(0), Int64(0), UInt64(1))
        input_storage = native_callback_buffer(raw; header, acquisition_sequence=UInt64(1))
        output_storage = native_callback_buffer(calibrated; header_available=true, properties_available=true)
        GC.@preserve raw calibrated input_storage output_storage begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input_storage)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output_storage)), UInt32(1))
            process_callback(node.filter, inputs, outputs)
            metadata = FilterGraphs.output_metadata(graph).calibrated
            @test sample_trigger_high(metadata) === true
            @test sample_chopped_image_kind(metadata) === nothing
            @test native_callback_flags(output_storage, 1) == 0
            # Every frame requires Header. A Header-less input is unavailable,
            # so even a bad tag must bypass ingress and leave prior state intact.
            previous_calibrated=copy(calibrated)
            raw[4] = 0x0001
            metadata_free_storage = native_callback_buffer(raw)
            GC.@preserve metadata_free_storage begin
                metadata_free_inputs = NdArrayFilterBuffers{false}(
                    Ptr{Cvoid}(pointer(metadata_free_storage)), UInt32(1))
                process_callback(node.filter, metadata_free_inputs, outputs)
                @test native_callback_flags(output_storage,1)==PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
                @test FilterGraphs.output_metadata(graph).calibrated==metadata
                @test calibrated==previous_calibrated
            end
        end
    finally
        close(node)
    end
end

@testset "ingress callback return and untouched input contracts" begin
    metadata = FilterGraphAlgorithms._admitted_sample_metadata(SampleMetadata(1))
    raw = UInt16[0, 0, 0, 0x3fff]
    @test_throws ArgumentError FilterGraphPipeWire._ingress_metadata((metadata, raw) -> nothing, metadata, raw)
    projected = FilterGraphPipeWire._ingress_metadata(
        (camera=TestCred2Ingress(nothing), other=nothing),
        (camera=metadata, other=metadata), (camera=raw, other=raw))
    @test sample_trigger_high(projected.camera.value) === true
    @test projected.other === metadata
    absent = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    @test FilterGraphPipeWire._ingress_metadata((metadata, raw) -> error("must bypass"), absent, raw) === absent
end

@testset "C-RED 2 transmitted tag position differs from logical array indexing" begin
    raw = fill(UInt16(100), 8)
    metadata = SampleMetadata(1)
    GC.@preserve raw begin
        borrowed = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(pointer(raw), (4, 2), 8)
        for high in (false, true)
            raw[4] = high ? 0x3fff : 0x3ffe
            @test borrowed[4] == 100
            @test invoke(cred2_trigger_metadata, Tuple{SampleMetadata,AbstractArray{UInt16}}, metadata, borrowed) === Cred2UnexpectedImageTag
            @test sample_trigger_high(cred2_trigger_metadata(metadata, borrowed)) === high
        end
    end
end
