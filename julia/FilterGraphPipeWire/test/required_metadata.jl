module RequiredMetadataTests
using FilterGraphAlgorithms, FilterGraphPipeWire, FilterGraphs, PipeWireAO, Test
include(joinpath(@__DIR__, "data", "examples", "cred2-fits-replay", "run.jl"))
const Replay=Cred2FitsReplayExample
@testset "strict SCC facts resolve through calibration provenance" begin
    graph=Replay.prepare_replay_graph()
    requirements=required_input_metadata(graph)
    @test keys(requirements)==(:raw,)
    @test Tuple(r.type for r in requirements.raw)==(SPA.META_ACQUISITION_IDENTITY,SPA.META_EXPOSURE_TIMING,SPA.META_OPTICAL_CHOPPING)
    @test all(r -> r.required,requirements.raw)
    produced=required_input_metadata(graph;produced=(raw=(:chopped_image_kind,),))
    @test Tuple(r.type for r in produced.raw)==(SPA.META_ACQUISITION_IDENTITY,SPA.META_EXPOSURE_TIMING)
    @test_throws ArgumentError required_input_metadata(graph;produced=(missing=(:chopped_image_kind,),))
    @test_throws ArgumentError required_input_metadata(graph;produced=(raw=(:invented_fact,),))
    profile=(metadata=requirements.raw,)
    node=PipeWireNode(graph;connect=false,input_metadata=(raw=profile,),output_metadata=(calibrated=profile,command=profile))
    FilterGraphPipeWire._prepare_owner(node.owner)
    raw=vec(Replay.synthetic_cube(;trigger_high_is_fringed=true)[:,:,1])
    calibrated=fill(42f0,8);command=fill(43f0,1)
    header=BufferHeader(SPA.META_HEADER_FLAG_MARKER,0,7,0,99)
    start=ExposureTiming(7;clock=METADATA_CLOCK_MODEL)
    source=Main.native_callback_buffer(raw;header,acquisition_sequence=UInt64(1),exposure_start=start,optical_classification=OPTICAL_FRINGED)
    targets=vcat(Main.native_callback_buffer(calibrated),Main.native_callback_buffer(command))
    GC.@preserve raw calibrated command source targets begin
        inputs=NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(source)),UInt32(1))
        outputs=NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(targets)),UInt32(2))
        @test metadata_records_valid(source.metadata[1],requirements.raw) # START alone is a valid record.
        @test_throws InvalidStateException node.filter.callbacks.on_process(node.filter,inputs,outputs)
        @test calibrated==fill(42f0,8) && command==fill(43f0,1)
        @test last(graph.nodes).prepared.workspace.state.committed_identity===nothing
        @test node.owner.active
        set_exposure_timing!(writable_metadata_records(source.metadata[1]),ExposureTiming(;duration_nanoseconds=Replay.SYNTHETIC_EXPOSURE_DURATION_NS))
        node.filter.callbacks.on_process(node.filter,inputs,outputs)
        @test calibrated≈(Float32.(raw).-100f0).*0.01f0
        @test only(command)==0f0
    end
    close(node)
end
@testset "independent arrivals check only the arriving branch's required facts" begin
    strict=FilterGraphPipeWire.RetainedMetadata()
    strict.required_facts=[:acquisition_identity,:exposure_duration_nanoseconds,:chopped_image_kind]
    other=FilterGraphPipeWire.RetainedMetadata()
    snapshots=(strict,other)
    values=Float32[1,2]
    absent=Main.native_callback_buffer(values;present=false)
    present=Main.native_callback_buffer(values;header=BufferHeader(0,0,7,0,1))
    independent=vcat(absent,present)
    missing=vcat(present,absent)
    GC.@preserve values absent present independent missing begin
        inputs=NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(independent)),UInt32(2))
        @test FilterGraphPipeWire._preflight_input_facts!(snapshots,inputs,1)===nothing
        @test_throws InvalidStateException FilterGraphPipeWire._preflight_input_facts!(snapshots,
            NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(missing)),UInt32(2)),1)
    end
end

end
