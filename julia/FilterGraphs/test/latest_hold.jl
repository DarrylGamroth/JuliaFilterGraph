@algorithm LatestHoldAddF32 begin
    label = "test-latest-hold-add-f32"

    @configuration LatestHoldAddF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :primary,
            Float32,
            (configuration.extent,),
            "test.primary.f32/1";
            rate=(2, 1),
        ),
        input(
            :held,
            Float32,
            (configuration.extent,),
            "test.held.f32/1";
            rate=(1, 1),
        ),
        output(
            :joined,
            Float32,
            (configuration.extent,),
            "test.joined.f32/1";
            metadata_from=:held,
            rate=(2, 1),
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @inbounds @simd for index in eachindex(joined, primary, held)
            joined[index] = primary[index] + held[index]
        end
        nothing
    end
end

const LATEST_HOLD_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = add
            label = test-latest-hold-add-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "add:primary" "add:held" ]
    outputs = [ "add:joined" ]
}
"""

prepare_latest_hold_test_graph() = prepare_graph_text(
    LATEST_HOLD_GRAPH;
    algorithms=(LatestHoldAddF32,),
)

@testset "bounded latest/hold input join" begin
    graph = prepare_latest_hold_test_graph()
    join = FilterGraphs.prepare_latest_hold(graph, :primary, :held)
    @test join isa FilterGraphs.PreparedLatestHold
    @test join.max_primary_cycles == 2

    primary = Float32[10, 20]
    held = Float32[1, 2]
    output = fill(-1.0f0, 2)
    outputs = (joined=output,)

    @test process!(outputs, join, primary, SampleMetadata(100)) == (joined=false,)
    @test output == fill(-1.0f0, 2)

    @test FilterGraphs.accept_held!(join, held, SampleMetadata(10))
    fill!(held, 99.0f0)
    @test process!(outputs, join, primary, SampleMetadata(101)) == (joined=true,)
    @test output == Float32[11, 22]
    @test FilterGraphs.output_metadata(graph) == (joined=SampleMetadata(10),)

    primary .= Float32[20, 30]
    @test process!(outputs, join, primary, SampleMetadata(102)) == (joined=true,)
    @test output == Float32[21, 32]

    fill!(output, -2.0f0)
    @test process!(outputs, join, primary, SampleMetadata(103)) == (joined=false,)
    @test output == fill(-2.0f0, 2)

    held .= Float32[3, 4]
    @test FilterGraphs.accept_held!(join, held, SampleMetadata(11))
    @test !FilterGraphs.accept_held!(join, held, SampleMetadata(10))

    held .= Float32[5, 6]
    @test FilterGraphs.accept_held!(join, held, SampleMetadata(12))
    held .= Float32[7, 8]
    @test FilterGraphs.accept_held!(join, held,
        SampleMetadata(14; trigger_high=false, chopped_image_kind=ChoppedImageFringed))
    primary .= Float32[1, 2]
    @test process!(outputs, join, primary, SampleMetadata(104)) == (joined=true,)
    @test output == Float32[8, 10]
    @test FilterGraphs.output_metadata(graph).joined.discontinuity
    @test FilterGraphs.output_metadata(graph).joined.sequence == 14
    @test sample_chopped_image_kind(FilterGraphs.output_metadata(graph).joined) == ChoppedImageFringed
    @test sample_trigger_high(FilterGraphs.output_metadata(graph).joined) === false

    held .= Float32[50, 60]
    @test !FilterGraphs.accept_held!(
        join,
        held,
        SampleMetadata(15; corrupted=true),
    )
    fill!(output, -3.0f0)
    @test process!(outputs, join, primary, SampleMetadata(105)) == (joined=false,)
    @test output == fill(-3.0f0, 2)

    held .= Float32[9, 10]
    @test FilterGraphs.accept_held!(join, held, SampleMetadata(16))
    @test process!(outputs, join, primary, SampleMetadata(106)) == (joined=true,)
    @test output == Float32[10, 12]

    fill!(output, -4.0f0)
    @test process!(
        outputs,
        join,
        primary,
        SampleMetadata(107; corrupted=true),
    ) == (joined=false,)
    @test output == fill(-4.0f0, 2)

    @test FilterGraphs.latest_hold_counters(join) == (
        primary_arrivals=8,
        held_arrivals=7,
        new_uses=3,
        retained_uses=1,
        skipped_primary=4,
        invalid_primary=1,
        invalid_held=1,
        stale_held=1,
        discontinuities=1,
        timeouts=1,
        overruns=2,
    )

    @test reset!(join) === join
    @test process!(outputs, join, primary) == (joined=false,)
end

@testset "latest/hold preparation rejects unsupported schedules" begin
    graph = prepare_latest_hold_test_graph()
    @test_throws ArgumentError FilterGraphs.prepare_latest_hold(graph, :held, :primary)
    @test_throws ArgumentError FilterGraphs.prepare_latest_hold(graph, :missing, :held)
    @test_throws ArgumentError FilterGraphs.prepare_latest_hold(graph, :primary, :missing)
end

@testset "latest/hold warmed allocation contract" begin
    join = FilterGraphs.prepare_latest_hold(
        prepare_latest_hold_test_graph(),
        :primary,
        :held,
    )
    primary = Float32[10, 20]
    held = Float32[1, 2]
    outputs = (joined=zeros(Float32, 2),)
    primary_metadata = SampleMetadata(100)
    held_metadata = SampleMetadata(10)

    FilterGraphs.accept_held!(join, held, held_metadata)
    process!(outputs, join, primary, primary_metadata)
    FilterGraphs.accept_held!(join, held, SampleMetadata(11))

    @test @allocated(
        FilterGraphs.accept_held!(join, held, SampleMetadata(12))
    ) == 0
    @test @allocated(process!(outputs, join, primary, primary_metadata)) == 0
    @test @allocated(process!(outputs, join, primary, primary_metadata)) == 0
    @test @allocated(process!(outputs, join, primary, primary_metadata)) == 0
    @test @allocated(
        FilterGraphs.accept_held!(
            join,
            held,
            SampleMetadata(13; corrupted=true),
        )
    ) == 0
    @test @allocated(process!(outputs, join, primary, primary_metadata)) == 0
    @test @allocated(FilterGraphs.latest_hold_counters(join)) == 0
    @test @allocated(reset!(join)) == 0
end

@testset "latest hold freshness retains physical acquisition across publications" begin
    graph = prepare_latest_hold_test_graph()
    join = FilterGraphs.prepare_latest_hold(graph, :primary, :held)
    domain = ntuple(i -> UInt8(i),16)
    metadata(publication, acquisition; generation=1) = SampleMetadata(publication;
        acquisition_identity=SampleAcquisitionIdentity(domain,generation,acquisition),
        reference_timestamp=SampleReferenceTimestamp(1_000_000_000+acquisition))
    @test FilterGraphs.accept_held!(join, Float32[1,2], metadata(100,7))
    @test !FilterGraphs.accept_held!(join, Float32[3,4], metadata(101,7))
    @test !FilterGraphs.accept_held!(join, Float32[3,4], metadata(102,6))
    @test FilterGraphs.accept_held!(join, Float32[3,4], metadata(200,8))
    output = (joined=zeros(Float32,2),)
    @test process!(output,join,zeros(Float32,2),SampleMetadata(300)) == (joined=true,)
    @test output.joined == Float32[3,4]
    @test sample_acquisition_identity(FilterGraphs.output_metadata(graph).joined).sequence == 8
    @test FilterGraphs.output_metadata(graph).joined.sequence == 200
    @test sample_reference_timestamp(FilterGraphs.output_metadata(graph).joined) ==
        SampleReferenceTimestamp(1_000_000_008)
    @test FilterGraphs.accept_held!(join, Float32[5,6], metadata(1,1;generation=2))
    @test !FilterGraphs.accept_held!(join, Float32[9,9], metadata(400,9))
    @test FilterGraphs.latest_hold_counters(join).stale_held == 3
end

struct ThrowingHeldSample <: AbstractVector{Float32} end
Base.size(::ThrowingHeldSample) = (2,)
Base.IndexStyle(::Type{ThrowingHeldSample}) = IndexLinear()
Base.getindex(::ThrowingHeldSample, i::Int) =
    i == 1 ? 99f0 : error("copy failed")

@testset "latest/hold keeps the previous sample on a partial copy" begin
    graph = prepare_latest_hold_test_graph()
    join = FilterGraphs.prepare_latest_hold(graph, :primary, :held)
    @test FilterGraphs.accept_held!(join, Float32[1, 2], SampleMetadata(10))
    @test_throws ErrorException FilterGraphs.accept_held!(
        join, ThrowingHeldSample(), SampleMetadata(11),
    )
    outputs = (joined=zeros(Float32, 2),)
    @test process!(outputs, join, Float32[10, 20], SampleMetadata(100)) == (joined=true,)
    @test outputs.joined == Float32[11, 22]
    @test FilterGraphs.output_metadata(graph) == (joined=SampleMetadata(10),)
    @test FilterGraphs.accept_held!(join, Float32[3, 4], SampleMetadata(11))
    @test process!(outputs, join, Float32[10, 20], SampleMetadata(101)) == (joined=true,)
    @test outputs.joined == Float32[13, 24]
end
