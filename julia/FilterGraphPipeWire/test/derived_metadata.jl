module DerivedMetadataTests
using FilterGraphAlgorithms, FilterGraphPipeWire, FilterGraphs, PipeWireAO, Test

@algorithm DerivedIntegrator begin
    label="test-derived-integrator"
    @configuration DerivedIntegratorConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(configuration)=(
        input(:input,Float32,(configuration.extent,),"test/1"),
        output(:output,Float32,(configuration.extent,),"test/1";metadata_from=:input),
    )
    prepare(configuration)=(plan=LeakyIntegratorPlan(1.0f0,0.5f0),
        workspace=LeakyIntegratorWorkspace(zeros(Float32,configuration.extent)))
    process!(plan,workspace)=FilterGraphAlgorithms._process_leaky_integrator!(output,workspace,plan,input)
end
@algorithm MetadataSource begin
    label="test-metadata-source"
    @configuration MetadataSourceConfiguration begin
        extent::Int => graph_rebuild
    end
    ports(configuration)=(output(:output,Float32,(configuration.extent,),"test/1"),)
    prepare(configuration)=nothing
    process!(plan)=begin
        fill!(output,1f0)
        nothing
    end
end
function fixture(;reference=true,required=true,source_only=false,source_reference=reference)
    graph=mktemp() do path,io
        label=source_only ? "test-metadata-source" : "test-derived-integrator"
        inputs=source_only ? "[]" : "[\"integrate:input\"]"
        write(io,"""filter.graph={nodes=[{type=ndarray name=integrate label=$label config={extent=2}}]
            links=[] inputs=$inputs outputs=[\"integrate:output\"]}""")
        flush(io)
        prepare_graph(path;algorithms=(DerivedIntegrator,MetadataSource))
    end
    requirements=(MetadataRequirement(SPA.META_REFERENCE_TIMESTAMP;required),)
    inputs=source_only ? NamedTuple() : (input=(metadata=requirements,),)
    node=PipeWireNode(graph;connect=false,input_metadata=inputs,output_metadata=(output=(metadata=requirements,),))
    FilterGraphPipeWire._prepare_owner(node.owner)
    input=Float32[2,3]; output=zeros(Float32,2)
    source=MetadataSnapshot((MetadataRequirement(SPA.META_HEADER),(source_reference ? requirements : ())...))
    target=MetadataSnapshot((MetadataRequirement(SPA.META_HEADER),(reference ? requirements : ())...))
    set_buffer_header!(writable_metadata_records(source),BufferHeader(0,0,7,0,99))
    source_reference && set_reference_timestamp!(writable_metadata_records(source),ReferenceTimestamp(1_700_000_000_000_000_123;
        clock=METADATA_CLOCK_TAI,duration_nanoseconds=17))
    n=PipeWireAO.LibPipeWire
    descriptor(data,records)=[n.pw_ndarray_filter_buffer(UInt32(sizeof(n.pw_ndarray_filter_buffer)),0,
        pointer(data),UInt32(sizeof(data)),UInt32(sizeof(data)),records.count,0,pointer(records.descriptors))]
    incoming=descriptor(input,source); outgoing=descriptor(output,target)
    inputs=NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(incoming)),UInt32(!source_only))
    outputs=NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(outgoing)),UInt32(1))
    return (;graph,node,input,output,source,target,incoming,outgoing,inputs,outputs)
end
function invoke(f)
    GC.@preserve f begin
        f.node.filter.callbacks.on_process(f.node.filter,f.inputs,f.outputs)
    end
end

if "--metadata-clock-fault" in ARGS
    @testset "late source clock failure faults publication" begin
        f=fixture(;source_only=true)
        @eval PipeWireAO begin
            const _test_clock_reads=Ref(0)
            function reference_timestamp_now(clock::MetadataClock=METADATA_CLOCK_REALTIME)
                _test_clock_reads[]+=1
                _test_clock_reads[]==1 || throw(InvalidStateException("injected source clock failure",:clock))
                return ReferenceTimestamp(100_000_000_000;clock)
            end
        end
        @test_throws FilterGraphPipeWire.MetadataPublicationError Base.invokelatest(invoke,f)
        @test !f.node.owner.active
        @test f.node.owner.metadata_fault!==nothing
        @test_throws FilterGraphPipeWire.MetadataPublicationError FilterGraphPipeWire._prepare_owner(f.node.owner)
        @test !iszero(f.outgoing[1].flags & PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE)
        @test_throws InvalidStateException Base.invokelatest(invoke,f)
        close(f.node)
    end
else
    @testset "derived publication preserves the triggering reference timestamp" begin
        f=fixture()
        invoke(f)
        @test f.output==f.input
        @test acquisition_identity(f.target)===nothing
        @test exposure_timing(f.target)===nothing
        @test buffer_header(f.target).sequence==99
        @test reference_timestamp(f.target)==reference_timestamp(f.source)
        invoke(f)
        @test reference_timestamp(f.target)==reference_timestamp(f.source)
        @test @allocated(invoke(f))==0
        close(f.node)
        missing=fixture(;reference=false)
        @test_throws InvalidStateException invoke(missing)
        @test all(iszero,leaky_integrator_state(missing.graph.nodes[1].prepared.workspace))
        @test missing.node.owner.active
        close(missing.node)
        optional=fixture(;reference=false,required=false,source_reference=false)
        invoke(optional)
        @test optional.output==optional.input
        @test reference_timestamp(optional.target)===nothing
        @test optional.node.owner.active
        close(optional.node)
    end
    @testset "source-only publication supplies its own Header and reference timestamp" begin
        f=fixture(;source_only=true)
        invoke(f)
        @test f.output==ones(Float32,2)
        @test buffer_header(f.target).sequence==1
        @test buffer_header(f.target).pts==typemin(Int64)
        @test reference_timestamp(f.target).clock==METADATA_CLOCK_REALTIME
        invoke(f)
        @test buffer_header(f.target).sequence==2
        @test @allocated(invoke(f))==0
        close(f.node)
    end
    @testset "explicit source timestamp is retained" begin
        records=MetadataSnapshot((MetadataRequirement(SPA.META_REFERENCE_TIMESTAMP),))
        stamp=ReferenceTimestamp(77;clock=METADATA_CLOCK_MODEL)
        writable=writable_metadata_records(records)
        set_reference_timestamp!(writable,stamp)
        snapshot=FilterGraphPipeWire.RetainedMetadata()
        snapshot.produce_time=true
        FilterGraphPipeWire._produce_output_time!(writable,snapshot)
        @test reference_timestamp(records)==stamp
    end
    @testset "source clock failure is isolated" begin
        command=`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $(@__FILE__) --metadata-clock-fault`
        @test success(pipeline(command;stdout=stdout,stderr=stderr))
    end
end
end
