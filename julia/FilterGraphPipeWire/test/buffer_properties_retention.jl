function property_callback!(owner,mode,input_storage,output_storage)
    GC.@preserve input_storage output_storage begin
        inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input_storage)),UInt32(length(input_storage)))
        outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output_storage)),UInt32(length(output_storage)))
        FilterGraphPipeWire._process_graph_buffers!(owner,mode,inputs,outputs)
    end
    return nothing
end
function retained_property(storage,index=1)
    record=metadata_record(storage.metadata[index],UInt32(0x900))
    record===nothing && return nothing
    raw=metadata_bytes(record)
    value=Int64(sum(UInt64(raw[i]) << (8*(i-1)) for i in eachindex(raw)))
    return iszero(value) ? nothing : value
end
function overwrite_source_property!(storage,value)
    records=writable_metadata_records(first(storage.metadata))
    set_metadata_bytes!(records,UInt32(0x900),ntuple(i -> UInt8((UInt64(value) >> (8*(i-1))) & 0xff),8))
    return nothing
end

@testset "latest hold owns opaque independent metadata after the input lease" begin
    join = FilterGraphs.prepare_latest_hold(prepared_latest_hold_graph(),:primary,:held)
    owner = FilterGraphPipeWire.PreparedGraphOwner(join)
    FilterGraphPipeWire._prepare_owner(owner)
    primary, held, destination = Float32[10,20],Float32[1,2],zeros(Float32,2)
    header(sequence) = BufferHeader(0,0,sequence,0,sequence)
    first_held = native_callback_buffer(held;header=header(100),acquisition_sequence=UInt64(7),
        exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(123))
    absent = native_callback_buffer(held;present=false)
    output = native_callback_buffer(destination;header_available=true)
    property_callback!(owner,join,vcat(absent,first_held),output)
    overwrite_source_property!(first_held,999)
    fast = native_callback_buffer(primary;header=header(200))
    property_callback!(owner,join,vcat(fast,absent),output)
    @test destination == Float32[11,22]
    @test retained_property(output) == 123
    @test native_callback_acquisition(output).identity.sequence == 7
    duplicate = native_callback_buffer(held;header=header(900),acquisition_sequence=UInt64(7),
        exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(456))
    property_callback!(owner,join,vcat(fast,duplicate),output)
    @test retained_property(output) == 123
    fresh = native_callback_buffer(held;header=header(901),acquisition_sequence=UInt64(8),
        exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(456))
    property_callback!(owner,join,vcat(fast,fresh),output)
    @test retained_property(output) == 456
    no_extra = native_callback_buffer(held;header=header(902),acquisition_sequence=UInt64(9),
        exposure_duration_nanoseconds=UInt64(10))
    property_callback!(owner,join,vcat(fast,no_extra),output)
    @test retained_property(output) === nothing
    FilterGraphPipeWire._reset_owner!(owner)
    @test all(snapshot -> !snapshot.present,values(owner.input_records))
    @test all(snapshot -> !snapshot.present,values(owner.output_records))
end

@testset "synchronized pending output keeps its own independent metadata before tail admission" begin
    text = replace(ADAPTER_SYNCHRONIZED_GRAPH,"\"add:a\" \"add:b\" \"add:c\"" => "\"add:c\" \"add:b\" \"add:a\"")
    graph = prepare_graph_text(text;algorithms=(AdapterSynchronizedSumF32,))
    join = FilterGraphs.prepare_synchronized_inputs(graph)
    owner = FilterGraphPipeWire.PreparedGraphOwner(join)
    FilterGraphPipeWire._prepare_owner(owner)
    a,b,c,destination = Float32[1,2],Float32[10,20],Float32[100,200],zeros(Float32,2)
    header(sequence) = BufferHeader(0,0,sequence,0,sequence)
    absent = native_callback_buffer(a;present=false)
    old_a = native_callback_buffer(a;header=header(30),extra_property=Int64(123))
    old_b = native_callback_buffer(b;header=header(30))
    old_c = native_callback_buffer(c;header=header(30))
    next_a = native_callback_buffer(a;header=header(31),extra_property=Int64(456))
    next_b = native_callback_buffer(b;header=header(31))
    next_c = native_callback_buffer(c;header=header(31))
    output = native_callback_buffer(destination;header_available=true)
    property_callback!(owner,join,vcat(absent,absent,old_a),output)
    property_callback!(owner,join,vcat(absent,old_b,absent),output)
    overwrite_source_property!(old_a,999)
    property_callback!(owner,join,vcat(old_c,absent,next_a),output)
    @test destination == Float32[111,222]
    @test native_callback_header(output).sequence == 30
    @test retained_property(output) == 123
    @test owner.input_records.a.metadata.value.sequence == 31
    property_callback!(owner,join,vcat(next_c,next_b,absent),output)
    @test native_callback_header(output).sequence == 31
    @test retained_property(output) == 456
end

@testset "changed source facts rebuild independent metadata without stale extensions" begin
    graph = prepared_metadata_source_graph()
    qualify(metadata,raw) = SampleMetadata(metadata;exposure_duration_nanoseconds=99)
    node = PipeWireNode(graph;connect=false,ingress=(right=qualify,))
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        left,right,destination = Float32[1,2],Float32[10,20],zeros(Float32,2)
        source = native_callback_buffer(right;header=BufferHeader(0,0,0,0,1),
            acquisition_sequence=UInt64(7),exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(123))
        output = native_callback_buffer(destination;header_available=true)
        property_callback!(node.owner,nothing,vcat(native_callback_buffer(left),source),output)
        @test destination == Float32[11,22]
        @test native_callback_acquisition(output).duration == 99
        @test retained_property(output) === nothing
    finally
        close(node)
    end
end

@testset "opaque optional metadata survive native projection and publication" begin
    graph = prepare_graph_text(ADAPTER_METADATA_SOURCE_GRAPH;algorithms=(AdapterMetadataSourceF32,))
    requirement = (MetadataRequirement(0x900,8),)
    settings = (metadata=requirement,propagation=:preserve)
    node = PipeWireNode(graph;connect=false,input_metadata=(right=settings,),
        output_metadata=(output=settings,))
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        left,right,destination = Float32[1,2],Float32[10,20],zeros(Float32,2)
        source = native_callback_buffer(right;header=BufferHeader(0,0,0,0,1),
            extra_property=Int64(7))
        output = native_callback_buffer(destination;header_available=true)
        property_callback!(node.owner,nothing,vcat(native_callback_buffer(left),source),output)
        @test destination == Float32[11,22]
        @test retained_property(output) == 7
        GC.@preserve source begin
            manual=metadata_records(first(source.metadata))
            @test metadata_records_valid(manual,requirement)
            @test !metadata_records_valid(manual,(MetadataRequirement(0x900,16;required=true),))
        end
    finally
        close(node)
    end
end

@algorithm HistoricalMetadataF32 begin
    label = "test-historical-properties-f32"
    @configuration HistoricalMetadataConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:input,Float32,(configuration.extent,),"test.signal.f32/1"),
        output(:output,Float32,(configuration.extent,),"test.signal.f32/1";metadata_from=:input),
    )
    prepare(configuration) = nothing
    process!(plan) = begin
        copyto!(output,input)
        AlgorithmProcessComplete(SampleMetadata(70;exposure_duration_nanoseconds=10,
            acquisition_identity=SampleAcquisitionIdentity(ntuple(UInt8,16),7,7)))
    end
end

@testset "historical scientific metadata does not copy another acquisition's extensions" begin
    graph = prepare_graph_text("""
        filter.graph = {
            nodes=[{type=ndarray name=copy label=test-historical-properties-f32 config={extent=2}}]
            links=[] inputs=["copy:input"] outputs=["copy:output"]
        }
        """;algorithms=(HistoricalMetadataF32,))
    node = PipeWireNode(graph;connect=false)
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        data,destination = Float32[1,2],zeros(Float32,2)
        output = native_callback_buffer(destination;header_available=true)
        first = native_callback_buffer(data;header=BufferHeader(0,0,0,0,70),
            acquisition_sequence=UInt64(7),exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(123))
        property_callback!(node.owner,nothing,first,output)
        @test retained_property(output) == 123
        next = native_callback_buffer(data;header=BufferHeader(0,0,0,0,71),
            acquisition_sequence=UInt64(8),exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(456))
        property_callback!(node.owner,nothing,next,output)
        @test destination == data
        @test native_callback_acquisition(output).identity.sequence == 7
        @test native_callback_header(output).sequence == 70
        @test retained_property(output) === nothing
    finally
        close(node)
    end
end

@testset "added classification preserves opaque source metadata" begin
    graph=prepared_metadata_source_graph()
    classify(metadata,raw)=SampleMetadata(metadata;chopped_image_kind=ChoppedImageFringed)
    node=PipeWireNode(graph;connect=false,ingress=(right=classify,))
    try
        FilterGraphPipeWire._prepare_owner(node.owner)
        left,right,destination=Float32[1,2],Float32[10,20],zeros(Float32,2)
        source=native_callback_buffer(right;header=BufferHeader(0,0,0,0,1),
            acquisition_sequence=UInt64(7),exposure_duration_nanoseconds=UInt64(10),extra_property=Int64(123))
        output=native_callback_buffer(destination)
        property_callback!(node.owner,nothing,vcat(native_callback_buffer(left),source),output)
        @test retained_property(output)==123
        @test optical_classification(output.metadata[1])==OPTICAL_FRINGED
        inputs=vcat(native_callback_buffer(left),source)
        property_callback!(node.owner,nothing,inputs,output)
        @test @allocated(property_callback!(node.owner,nothing,inputs,output)) == 0
    finally
        close(node)
    end
end
