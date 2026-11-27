module MetadataRecordsTests

using FilterGraphAlgorithms, FilterGraphPipeWire, FilterGraphs, PipeWireAO, Test

@algorithm PropertiesPassThroughF32 begin
    label = "properties-pass-through-f32"
    @configuration PropertiesPassThroughF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.properties.f32/1"),
        output(:output, Float32, (configuration.extent,), "test.properties.f32/1"; metadata_from=:input),
    )
    prepare(configuration) = nothing
    process!(plan) = begin
        copyto!(output, input)
        nothing
    end
end

function properties_graph()
    mktemp() do path, io
        write(io, """
        filter.graph = {
            nodes = [ { type=ndarray name=copy label=properties-pass-through-f32 config={ extent=4 } } ]
            links = [] inputs = [ "copy:input" ] outputs = [ "copy:output" ]
        }
        """)
        flush(io)
        return prepare_graph(path; algorithms=(PropertiesPassThroughF32,))
    end
end

const OPTICAL_REQUIREMENTS=(MetadataRequirement(SPA.META_OPTICAL_CHOPPING;required=true),)
const PORT_PROPERTIES=(metadata=(MetadataRequirement(SPA.META_ACQUISITION_IDENTITY),
    MetadataRequirement(SPA.META_EXPOSURE_TIMING),MetadataRequirement(SPA.META_REFERENCE_TIMESTAMP),
    OPTICAL_REQUIREMENTS...,MetadataRequirement(SPA.META_ELECTRICAL_TRIGGER;required=true),MetadataRequirement(0x900,8)),propagation=:preserve)
properties_metadata_fixture()=(buffer=MetadataSnapshot((MetadataRequirement(SPA.META_HEADER),PORT_PROPERTIES.metadata...)),)
function optional_value(records)
    record=metadata_record(records,UInt32(0x900))
    record===nothing && return nothing
    raw=metadata_bytes(record)
    value=Int64(sum(UInt64(raw[i]) << (8*(i-1)) for i in eachindex(raw)))
    return iszero(value) ? nothing : value
end

function sample_metadata_bytes(header,properties)
    FilterGraphPipeWire._buffer_metadata(header,properties)
    @allocated FilterGraphPipeWire._buffer_metadata(header,properties)
end

@testset "typed physical and classification facts project independently" begin
    fixture = properties_metadata_fixture()
    identity = AcquisitionIdentity(AcquisitionDomain(ntuple(UInt8,16)),typemax(UInt64),typemax(UInt64))
    header = BufferHeader(SPA.META_HEADER_FLAG_MARKER,0,77,0,7)
    GC.@preserve fixture begin
        properties = writable_metadata_records(fixture.buffer)
        for (optical,kind) in ((OPTICAL_FRINGED,ChoppedImageFringed),
                (OPTICAL_UNFRINGED,ChoppedImageUnfringed),(OPTICAL_MIXED,ChoppedImageMixed),(OPTICAL_ABSENT,ChoppedImageAbsent)),
                trigger in (ACQUISITION_TRIGGER_ABSENT,ACQUISITION_TRIGGER_LOW,ACQUISITION_TRIGGER_HIGH,ACQUISITION_TRIGGER_MIXED)
            initialize_metadata!(properties)
            set_optical_classification!(properties,optical)
            set_trigger_classification!(properties,trigger)
            set_acquisition_identity!(properties,identity)
            set_exposure_timing!(properties,ExposureTiming(5678;clock=METADATA_CLOCK_MONOTONIC,duration_nanoseconds=1234))
            admitted = FilterGraphPipeWire._buffer_metadata(header,properties)
            @test admitted.present
            @test sample_trigger_state(admitted.value) === SampleTriggerState(UInt8(trigger))
            @test sample_chopped_image_kind(admitted.value) === kind
            @test sample_acquisition_identity(admitted.value) ==
                SampleAcquisitionIdentity(identity.domain.bytes,identity.generation,identity.sequence)
            @test admitted.value.sequence == 7
            @test sample_exposure_duration_nanoseconds(admitted.value) == 1234
            @test sample_exposure_start(admitted.value).nanoseconds == 5678
            @test sample_metadata_bytes(header,properties) == 0
        end
        initialize_metadata!(properties)
        empty = FilterGraphPipeWire._buffer_metadata(header,properties)
        @test sample_trigger_state(empty.value) === SAMPLE_TRIGGER_ABSENT
        @test sample_chopped_image_kind(empty.value) === nothing
        @test sample_acquisition_identity(empty.value) === nothing
    end
end

@testset "owned reference timestamps preserve nanosecond precision without allocation" begin
    fixture=properties_metadata_fixture()
    records=writable_metadata_records(fixture.buffer)
    header=BufferHeader(0,0,7,0,8)
    for clock in (METADATA_CLOCK_MONOTONIC,METADATA_CLOCK_REALTIME,METADATA_CLOCK_TAI,METADATA_CLOCK_MODEL),
            duration in (nothing,UInt64(0),typemax(UInt64)-1)
        initialize_metadata!(records)
        set_exposure_timing!(records,ExposureTiming(7;clock=METADATA_CLOCK_MODEL,duration_nanoseconds=10))
        stamp=ReferenceTimestamp(typemax(UInt64)-1;clock,duration_nanoseconds=duration)
        set_reference_timestamp!(records,stamp)
        metadata=FilterGraphPipeWire._buffer_metadata(header,records).value
        reference=sample_reference_timestamp(metadata)
        @test reference.timestamp_nanoseconds==stamp.timestamp_nanoseconds
        @test reference.clock==SampleClock(UInt32(clock))
        @test reference.has_duration==(duration!==nothing)
        @test reference.duration_nanoseconds==something(duration,typemax(UInt64))
        @test sample_exposure_start(metadata).clock==SAMPLE_CLOCK_MODEL
        @test sample_metadata_bytes(header,records)==0
        target=MetadataSnapshot((MetadataRequirement(SPA.META_HEADER),PORT_PROPERTIES.metadata...))
        FilterGraphPipeWire._set_output_records!(writable_metadata_records(target),metadata,nothing)
        @test reference_timestamp(target)==stamp
        @test exposure_timing(target).start_nanoseconds==7
    end
    initialize_metadata!(records)
    @test sample_reference_timestamp(FilterGraphPipeWire._buffer_metadata(header,records).value)===nothing
end

@testset "explicit per-Port buffer property contracts" begin
    defaults = PipeWireNode(properties_graph(); connect=false)
    selected = PipeWireNode(properties_graph(); connect=false,
        input_metadata=(input=PORT_PROPERTIES,), output_metadata=(output=PORT_PROPERTIES,))
    try
        @test !defaults.filter.metadata_enabled
        @test selected.filter.metadata_enabled
        @test node_state(selected) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        @test node_id(selected) === nothing
        @test last_error(selected) === nothing
        @test FilterGraphPipeWire.prepare!(selected; required_outputs=(:output,)) === selected
    finally
        close(selected)
        close(defaults)
    end
    for settings in (
        (input_metadata=(missing=PORT_PROPERTIES,),),
        (output_metadata=(missing=PORT_PROPERTIES,),),
        (input_metadata=(input=128,),),
        (input_metadata=(input=nothing,),),
        (input_metadata=(input=(capacity=128,),),),
        (output_metadata=(output=(metadata=(MetadataRequirement(0x901,4096),),),),),
        (input_metadata=(input=(metadata=(OPTICAL_REQUIREMENTS...,OPTICAL_REQUIREMENTS...),),),),
    )
        @test_throws ArgumentError PipeWireNode(properties_graph(); connect=false, settings...)
    end
end

# A real retained row-block executor must reject these borrowed properties
# before connection, rather than silently dropping them between callbacks.
@testset "retained progressive boundaries declare independent metadata" begin
    graph = mktemp() do path, io
        write(io,"""
        filter.graph = {
            nodes = [ { type=ndarray name=rec label=incremental-dense-reconstructor-f32
                config={ measurement_count=4 measurement_block_capacity=2 blocks_per_sample=2
                    reconstructed_count=8 initial_reconstructor=null
                    measurements_schema=test.measurements/1 column_indices_schema=test.columns/1
                    measurement_count_schema=test.count/1 reconstructed_schema=test.output/1
                    reconstructor_schema=test.matrix/1 } } ]
            links=[] inputs=[ "rec:measurements" "rec:columns" "rec:count" "rec:reconstructor" ]
            outputs=[ "rec:reconstructed" ]
        }
        """)
        flush(io)
        prepare_cpu_progressive_graph(path;algorithms=(IncrementalDenseReconstructorF32,),workers=1)
    end
    try
        node=PipeWireNode(graph;connect=false,rate=(1000,1),input_metadata=(measurements=PORT_PROPERTIES,),
            output_metadata=(reconstructed=PORT_PROPERTIES,))
        @test node.filter.metadata_enabled
        close(node)
        @test_throws ArgumentError PipeWireNode(graph;connect=false,input_metadata=(measurements=PORT_PROPERTIES,))
        @test_throws ArgumentError PipeWireNode(graph.graph;connect=false,
            input_metadata=(reconstructor=PORT_PROPERTIES,))
    finally
        close(graph)
    end
end

# The integration selector points to a native installation, never to a desktop
# socket. Client library preferences must select the same installation.
function with_properties_private_core(f, prefix)
    libdir = isdir(joinpath(prefix, "lib", "x86_64-linux-gnu")) ?
        joinpath(prefix, "lib", "x86_64-linux-gnu") : joinpath(prefix, "lib")
    mktempdir(prefix="fgpw-properties-") do directory
        runtime, configuration = joinpath(directory, "runtime"), joinpath(directory, "configuration")
        mkpath(runtime)
        mkpath(configuration)
        remote = "fgpw-properties-$(getpid())-$(time_ns())"
        write(joinpath(configuration, "core.conf"), """
        context.properties = { core.daemon=true core.name=$remote support.dbus=false library.use-fallback=false }
        context.spa-libs = { support.*=support/libspa-support }
        context.modules = [
            { name=libpipewire-module-scheduler-v1 }
            { name=libpipewire-module-protocol-native }
            { name=libpipewire-module-client-node }
            { name=libpipewire-module-link-factory }
            { name=libpipewire-module-metadata }
            { name=libpipewire-module-access }
        ]
        """)
        write(joinpath(configuration, "client.conf"), """
        context.properties = { support.dbus=false }
        context.spa-libs = { support.*=support/libspa-support }
        context.modules = [
            { name=libpipewire-module-protocol-native }
            { name=libpipewire-module-client-node }
            { name=libpipewire-module-metadata }
        ]
        """)
        environment = Dict("XDG_RUNTIME_DIR"=>runtime, "PIPEWIREAO_RUNTIME_DIR"=>runtime,
            "PIPEWIREAO_CONFIG_DIR"=>configuration, "PIPEWIREAO_MODULE_DIR"=>joinpath(libdir,"pipewire-ao-0.3"),
            "PIPEWIREAO_SPA_PLUGIN_DIR"=>joinpath(libdir,"spa-ao-0.2"), "PIPEWIREAO_DEBUG"=>"0",
            "LD_LIBRARY_PATH"=>libdir*":"*get(ENV,"LD_LIBRARY_PATH",""), "JULIA_RTC_TRACE_GC"=>"1")
        open(joinpath(directory,"core.log"), "w+") do log
            daemon = run(pipeline(addenv(`$(joinpath(prefix,"bin","pipewire-ao")) -c core.conf`, environment);
                stdout=log, stderr=log); wait=false)
            try
                timedwait(() -> ispath(joinpath(runtime,remote)), 10; pollint=0.01) == :ok ||
                    error("private properties core socket deadline expired")
                withenv(environment...) do
                    f(remote)
                end
                @test !Base.process_exited(daemon)
            catch
                flush(log)
                seekstart(log)
                print(stderr, read(log,String))
                rethrow()
            finally
                Base.process_exited(daemon) || kill(daemon)
                wait(daemon)
            end
        end
    end
end

function properties_link(core, registry, source, sink)
    ports(direction,node) = find_globals(registry; interface="PipeWire:Interface:Port",
        properties=("node.id"=>string(node_id(node)), "port.direction"=>direction))
    timedwait(10; pollint=0.01) do
        roundtrip(registry)
        !isempty(ports("out",source)) && !isempty(ports("in",sink))
    end == :ok || error("properties ports did not appear")
    output, input = only(ports("out",source)), only(ports("in",sink))
    with_thread_loop_lock(main_loop(core)) do _
        create_object(core,"link-factory",Link; properties=Dict(
            "link.output.node"=>string(node_id(source)), "link.output.port"=>string(output.id),
            "link.input.node"=>string(node_id(sink)), "link.input.port"=>string(input.id), "object.linger"=>"false"))
    end
end

function properties_roundtrip(remote; mandatory=false)
    loop = ThreadLoop("test.fgpw.properties")
    context,core,registry = with_thread_loop_lock(loop) do _
        context = Context(loop)
        core = CoreConnection(context; properties=Dict("remote.name"=>remote))
        context,core,Registry(core)
    end
    start!(loop)
    source = sink = input_link = output_link = node = worker = nothing
    pending = Threads.Atomic{Bool}(false)
    sequence = Ref(UInt64(0))
    classification = Ref(OPTICAL_FRINGED)
    received = Tuple[]
    source_buffer,sink_buffer = StreamBuffer(),StreamBuffer()
    format = NdArrayFormat(NdArray.F32_LE,(4,); layout=NdArray.ROW_MAJOR)
    domain = AcquisitionDomain(ntuple(UInt8,16))
    requirements=map(r -> r.type==SPA.META_ELECTRICAL_TRIGGER ? MetadataRequirement(r.type,r.size;required=mandatory) : r,PORT_PROPERTIES.metadata)
    port_properties=(metadata=requirements,propagation=:preserve)
    parameters=(Pod(buffers_param(buffers=2,blocks=1,size=16,metadata_types=Int32(1 << SPA.META_HEADER))),
        Pod(header_metadata_param()),(Pod(metadata_param(r)) for r in requirements)...)
    function parameter_changed(stream,id,parameter)
        id == SPA.PARAM_FORMAT && parameter !== nothing && update_params!(stream,parameters)
        nothing
    end
    function produce(stream)
        pending[] || return nothing
        dequeue_buffer!(source_buffer,stream) || return nothing
        queued = false
        try
            data = buffer_data(source_buffer)
            fill!(reinterpret(Float32,buffer_memory(data,16)), Float32(sequence[]) + 0.5f0)
            set_chunk!(data;size=16,stride=4)
            set_buffer_size!(source_buffer,4)
            set_buffer_header!(source_buffer,BufferHeader(SPA.META_HEADER_FLAG_MARKER,0,Int64(sequence[]),0,sequence[]))
            initialize_metadata!(source_buffer)
            set_buffer_header!(source_buffer,BufferHeader(SPA.META_HEADER_FLAG_MARKER,0,Int64(sequence[]),0,sequence[]))
            properties = writable_metadata_records(source_buffer)
            set_optical_classification!(properties,classification[])
            set_trigger_classification!(properties,ACQUISITION_TRIGGER_LOW)
            set_acquisition_identity!(properties,AcquisitionIdentity(domain,UInt64(1),sequence[]))
            set_exposure_timing!(properties,ExposureTiming(;duration_nanoseconds=2000))
            if mandatory || isodd(sequence[])
                set_metadata_bytes!(properties,UInt32(0x900),ntuple(i -> UInt8(((sequence[]+100) >> (8*(i-1))) & 0xff),8))
            end
            queue_buffer!(source_buffer,stream)
            queued = true
            pending[] = false
        finally
            queued || return_buffer!(source_buffer,stream)
        end
        nothing
    end
    function consume(stream)
        dequeue_buffer!(sink_buffer,stream) || return nothing
        try
            properties = metadata_records(sink_buffer)
            properties === nothing && error("properties allocation is missing")
            metadata_records_valid(properties,requirements) || error("required properties are invalid")
            header = buffer_header(sink_buffer)
            values = copy(reinterpret(Float32,bytes(buffer_data(sink_buffer))))
            push!(received,(header.sequence,optical_classification(properties),
                acquisition_identity(properties),exposure_timing(properties).duration_nanoseconds,
                trigger_classification(properties),optional_value(properties),values))
        finally
            queue_buffer!(sink_buffer,stream)
        end
        nothing
    end
    try
        source,sink = with_thread_loop_lock(loop) do _
            source = Stream(core,"test.fgpw.properties.source";
                properties=Dict("node.name"=>"test.fgpw.properties.source", "media.type"=>"Application",
                    "media.category"=>"Filter", "node.reliable"=>"true", "node.want-driver"=>"true"),
                on_process=produce,on_param_changed=parameter_changed)
            sink = Stream(core,"test.fgpw.properties.sink";
                properties=Dict("node.name"=>"test.fgpw.properties.sink", "media.type"=>"Application",
                    "media.category"=>"Filter", "node.reliable"=>"true"),
                on_process=consume,on_param_changed=parameter_changed)
            flags = STREAM_INACTIVE | STREAM_MAP_BUFFERS | STREAM_DONT_RECONNECT | STREAM_NO_CONVERT
            formats = (ndarray_format(format;schema="test.properties.f32/1"),parameters...)
            connect!(source,:output;flags=flags|STREAM_DRIVER,params=formats)
            connect!(sink,:input;flags,params=formats)
            source,sink
        end
        ready = Channel{Any}(1)
        worker = Threads.@spawn begin
            local_node = nothing
            try
                local_node = PipeWireNode(properties_graph();name="test.fgpw.properties.graph",remote,
                    connect=false,trace_callbacks=16,boundary_layout=RowMajorLayout(),
                    input_metadata=(input=port_properties,),
                    output_metadata=(output=port_properties,))
                FilterGraphPipeWire.prepare!(local_node;required_outputs=(:output,))
                connect!(local_node)
                put!(ready,local_node)
                run!(local_node)
            catch error
                isready(ready) || put!(ready,error)
                rethrow()
            finally
                local_node === nothing || close(local_node)
            end
        end
        node = take!(ready)
        node isa Exception && throw(node)
        input_link = properties_link(core,registry,source,node)
        output_link = properties_link(core,registry,node,sink)
        roundtrip(core)
        with_thread_loop_lock(loop) do _
            set_active!(sink,true)
            set_active!(source,true)
        end
        classes = (OPTICAL_FRINGED,OPTICAL_UNFRINGED,OPTICAL_MIXED,OPTICAL_ABSENT,
            OPTICAL_UNFRINGED,OPTICAL_FRINGED,OPTICAL_ABSENT,OPTICAL_MIXED)
        for (index,value) in enumerate(classes)
            with_thread_loop_lock(loop) do _
                sequence[] = UInt64(index)
                classification[] = value
                pending[] = true
            end
            completed = timedwait(10;pollint=0.01) do
                with_thread_loop_lock(loop) do _
                    if stream_state(source) == PipeWireAO.LibPipeWire.PW_STREAM_STATE_STREAMING && is_driving(source)
                        trigger_process!(source)
                    end
                    PipeWireAO._check_callback_error(source)
                    PipeWireAO._check_callback_error(sink)
                    length(received) >= index
                end
            end
            completed == :ok || error("properties deadline: node=$(node_state(node)), failure=$(last_error(node)), callback=$(node.filter.callback_error[])")
            with_thread_loop_lock(loop) do _
                optional = mandatory || isodd(index) ? Int64(index + 100) : nothing
                @test received[index] == (UInt64(index),value,
                    AcquisitionIdentity(domain,UInt64(1),UInt64(index)),UInt64(2000),ACQUISITION_TRIGGER_LOW,
                    optional,fill(Float32(index)+0.5f0,4))
            end
        end
        trace = node.owner.callback_trace
        @test trace.count == length(classes)
        # Source and sink callbacks are idle while each Graph callback runs.
        # Ignore first-use compilation; later callbacks cover native property
        # lookup and complete-object propagation inside the actual adapter.
        @test all(iszero, trace.gc_allocated_bytes_end[3:trace.count] .- trace.gc_allocated_bytes_start[3:trace.count])
    finally
        # A successful source trigger can still have a queued loop event after
        # the sink receives its frame. Stop and drain that driver while its
        # stream remains alive before destroying the graph or transport.
        with_thread_loop_lock(loop) do _
            source === nothing || set_active!(source,false)
            sink === nothing || set_active!(sink,false)
        end
        roundtrip(core)
        with_thread_loop_lock(loop) do _
            input_link === nothing || close(input_link)
            output_link === nothing || close(output_link)
        end
        node isa PipeWireNode && isopen(node) && quit!(node)
        worker === nothing || fetch(worker)
        with_thread_loop_lock(loop) do _
            sink === nothing || close(sink)
            source === nothing || close(source)
            close(registry)
            close(core)
            close(context)
        end
        close(loop)
    end
end

if haskey(ENV,"FGPW_METADATA_PRIVATE_CORE_PREFIX")
    @testset "actual Graph preserves optical, opaque optional and required trigger records" begin
        Threads.nthreads() >= 2 && Threads.nthreads(:interactive) >= 1 ||
            error("private-core property integration requires --threads=2,1 or more")
        let
            with_properties_private_core(ENV["FGPW_METADATA_PRIVATE_CORE_PREFIX"]) do remote
                properties_roundtrip(remote)
                properties_roundtrip(remote;mandatory=true)
            end
        end
    end
end

end # module
