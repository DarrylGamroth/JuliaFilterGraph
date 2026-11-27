module Cred2FitsReplayTests

using FITSIO
using FilterGraphAlgorithms
using FilterGraphs
using Test

include(joinpath(@__DIR__, "data", "examples", "cred2-fits-replay/run.jl"))

const Replay = Cred2FitsReplayExample
const DOMAIN = ntuple(index -> UInt8(index), 16)
const OPTICAL_KINDS = (
    ChoppedImageFringed,
    ChoppedImageUnfringed,
    ChoppedImageFringed,
    ChoppedImageUnfringed,
)

function replay_metadata(sequence)
    return SampleMetadata(sequence;exposure_duration_nanoseconds=Replay.SYNTHETIC_EXPOSURE_DURATION_NS,
        acquisition_identity=SampleAcquisitionIdentity(DOMAIN, 7, sequence),
        terminal=true)
end

replay_outputs() = (calibrated=fill(42.0f0, 4, 2), command=fill(42.0f0, 1))

function expected_commands()
    # The selected science pixel has (104 - 100) * 0.01 = 0.04 signal.
    # Each complete rolling pair contributes -gain * signal to the history.
    expected = zeros(Float32, 4)
    for index in 2:4
        expected[index] = 0.5f0 * expected[index - 1] - 0.25f0 * 0.04f0
    end
    return expected
end

function scc_history(graph)
    state = last(graph.nodes).prepared.workspace.state
    return (
        identity=state.committed_identity,
        command=copy(state.committed_command),
        exposure_command=copy(state.committed_exposure_command),
        history=copy(FilterGraphAlgorithms.leaky_integrator_state(state.integrator)),
        primed=FilterGraphAlgorithms.chopped_image_is_primed(state.difference),
        kind=FilterGraphAlgorithms.chopped_image_retained_kind(state.difference),
        image=copy(FilterGraphAlgorithms.chopped_image_retained_image(state.difference)),
        validity=copy(FilterGraphAlgorithms.chopped_image_retained_validity(state.difference)),
    )
end

@testset "C-RED 2 FITS cube preserves raw words and asymmetric axes" begin
    @test Replay.FRAME_SHAPE == (4, 2)
    for polarity in (false, true)
        cube = Replay.synthetic_cube(; trigger_high_is_fringed=polarity)
        @test cube isa Array{UInt16,3}
        @test size(cube) == (4, 2, 4)
        for sequence in axes(cube, 3)
            raw = @view cube[:, :, sequence]
            high = isodd(sequence) == polarity
            @test raw[1:3] == UInt16[sequence, 0, 0]
            @test raw[4] == (high ? 0x3fff : 0x3ffe)
            @test raw[1, 2] == (isodd(sequence) ? 104 : 100)
            @test all(==(UInt16(100)), raw[6:8])
        end
        # Exercise the unsigned representation beyond the camera's 14-bit tags.
        cube[1:3, 1, 1] .= UInt16[0xffff, 0x8000, 0x0000]
        mktempdir() do directory
            path = joinpath(directory, "tagged.fits")
            Replay.write_cube(path, cube)
            loaded = Replay.read_cube(path)
            @test eltype(loaded) === UInt16
            @test axes(loaded) == axes(cube)
            @test loaded == cube
        end
    end
end

@testset "C-RED 2 FITS replay requires an explicit optical mapping" begin
    @test_throws UndefKeywordError Replay.synthetic_cube()
    @test_throws UndefKeywordError Replay.run_example()
    @test_throws UndefKeywordError Replay.replay_cube("unused.fits")
    graph = Replay.prepare_replay_graph()
    cube = Replay.synthetic_cube(; trigger_high_is_fringed=true)
    @test_throws UndefKeywordError Replay.process_frame!(
        replay_outputs(), graph, @view(cube[:, :, 1]), replay_metadata(1))
end

@testset "C-RED 2 FITS replay separates trigger tags from optical phases" begin
    commands = Vector{Float32}[]
    for polarity in (false, true)
        records = Replay.run_example(; trigger_high_is_fringed=polarity)
        @test length(records) == 4
        @test [record.sequence for record in records] == 1:4
        @test [record.trigger_high for record in records] ==
              [isodd(sequence) == polarity for sequence in 1:4]
        @test Tuple(record.kind for record in records) == OPTICAL_KINDS
        @test all(record -> record.command isa Vector{Float32} && length(record.command) == 1, records)
        actual = Float32[only(record.command) for record in records]
        @test actual ≈ expected_commands() atol = 2.0f-8 rtol = 0
        push!(commands, actual)

        identities = [sample_acquisition_identity(record.metadata) for record in records]
        @test all(!isnothing, identities)
        @test !all(iszero, first(identities).domain)
        @test first(identities).generation != 0
        for (sequence, record) in enumerate(records)
            metadata = record.metadata
            identity = identities[sequence]
            @test metadata.sequence == record.sequence == sequence
            @test identity.sequence == sequence
            @test identity.domain == Replay.SOURCE_DOMAIN
            @test identity.generation == Replay.SOURCE_GENERATION
            @test sample_trigger_high(metadata) === record.trigger_high
            @test sample_chopped_image_kind(metadata) === record.kind
            @test sample_exposure_duration_nanoseconds(metadata)==Replay.SYNTHETIC_EXPOSURE_DURATION_NS
            @test sample_exposure_start(metadata) === nothing
        end
    end
    @test commands[1] == commands[2]
end

@testset "C-RED 2 raw tags are decoded before calibration" begin
    for polarity in (false, true)
        graph = Replay.prepare_replay_graph()
        @test keys(graph.input_formats) == (:raw,)
        @test keys(graph.output_formats) == (:calibrated, :command)
        @test graph.input_formats.raw.shape == (4, 2)
        @test graph.output_formats.calibrated.shape == (4, 2)
        @test graph.output_formats.command.shape == (1,)
        @test last(graph.nodes).prepared.plan.calibration.residual.selected_indices == [5]
        cube = Replay.synthetic_cube(; trigger_high_is_fringed=polarity)
        original = copy(cube)
        outputs = replay_outputs()
        for sequence in 1:4
            raw = @view cube[:, :, sequence]
            metadata = replay_metadata(sequence)
            @test Replay.process_frame!(outputs, graph, raw, metadata;
                trigger_high_is_fringed=polarity) == (calibrated=true, command=true)
            optical = chopped_image_metadata(cred2_trigger_metadata(metadata, raw);
                trigger_high_is_fringed=polarity)
            @test FilterGraphs.output_metadata(graph) ==
                  (calibrated=optical, command=optical)
            @test outputs.calibrated ≈ (Float32.(raw) .- 100.0f0) .* 0.01f0
            @test all(index -> outputs.calibrated[index] != raw[index], 1:4)
            @test outputs.calibrated[1, 2] ≈ (isodd(sequence) ? 0.04f0 : 0.0f0)
            @test only(outputs.command) ≈ expected_commands()[sequence] atol = 2.0f-8 rtol = 0
            @test sample_trigger_high(metadata) === nothing
            @test sample_chopped_image_kind(metadata) === nothing
        end
        @test cube == original
    end
end

@testset "C-RED 2 replay follows valid tag words instead of frame parity" begin
    cube = Replay.synthetic_cube(; trigger_high_is_fringed=true)
    cube[4, 1, 2] = 0x3fff
    mktempdir() do directory
        path = joinpath(directory, "repeated-phase.fits")
        Replay.write_cube(path, cube)
        records = Replay.replay_cube(path; trigger_high_is_fringed=true,exposure_duration_nanoseconds=Replay.SYNTHETIC_EXPOSURE_DURATION_NS)
        @test [record.trigger_high for record in records] == [true, true, true, false]
        @test Tuple(record.kind for record in records) == (
            ChoppedImageFringed,
            ChoppedImageFringed,
            ChoppedImageFringed,
            ChoppedImageUnfringed,
        )
        @test Float32[only(record.command) for record in records] ≈
              Float32[0, 0, 0, -0.01] atol = 2.0f-8 rtol = 0
    end
end

@testset "unexpected C-RED 2 tags do not advance the Graph" begin
    for polarity in (false, true)
        cube = Replay.synthetic_cube(; trigger_high_is_fringed=polarity)
        graph = Replay.prepare_replay_graph()
        oracle = Replay.prepare_replay_graph()
        outputs, oracle_outputs = replay_outputs(), replay_outputs()
        for sequence in 1:2
            raw = @view cube[:, :, sequence]
            metadata = replay_metadata(sequence)
            @test Replay.process_frame!(outputs, graph, raw, metadata;
                trigger_high_is_fringed=polarity) == (calibrated=true, command=true)
            @test Replay.process_frame!(oracle_outputs, oracle, raw, metadata;
                trigger_high_is_fringed=polarity) == (calibrated=true, command=true)
        end
        retained_outputs = deepcopy(outputs)
        retained_metadata = FilterGraphs.output_metadata(graph)
        retained_history = scc_history(graph)
        for (sequence, tag) in ((3, 0x0001), (4, 0x0000), (3, 0xffff))
            invalid = copy(@view cube[:, :, sequence])
            invalid[4] = tag
            @test Replay.process_frame!(outputs, graph, invalid, replay_metadata(sequence);
                trigger_high_is_fringed=polarity) === Cred2UnexpectedImageTag
            @test outputs == retained_outputs
            @test FilterGraphs.output_metadata(graph) == retained_metadata
            @test scc_history(graph) == retained_history
        end
        # Retry the same ordinal with its valid tag; compare with a Graph that
        # never received the rejected frames, including the following update.
        for sequence in 3:4
            raw = @view cube[:, :, sequence]
            metadata = replay_metadata(sequence)
            @test Replay.process_frame!(outputs, graph, raw, metadata;
                trigger_high_is_fringed=polarity) == (calibrated=true, command=true)
            @test Replay.process_frame!(oracle_outputs, oracle, raw, metadata;
                trigger_high_is_fringed=polarity) == (calibrated=true, command=true)
            @test outputs == oracle_outputs
            @test FilterGraphs.output_metadata(graph) == FilterGraphs.output_metadata(oracle)
            @test scc_history(graph) == scc_history(oracle)
        end
    end
end

@testset "C-RED 2 FITS replay rejects incompatible image cubes" begin
    malformed = (
        (zeros(Int16, 4, 2, 4), ArgumentError),
        (zeros(Float32, 4, 2, 4), ArgumentError),
        (zeros(UInt16, 4, 2), DimensionMismatch),
        (zeros(UInt16, 2, 4, 4), DimensionMismatch),
        (zeros(UInt16, 4, 3, 4), DimensionMismatch),
    )
    mktempdir() do directory
        for (index, (cube, exception)) in enumerate(malformed)
            path = joinpath(directory, "malformed-$index.fits")
            FITS(path, "w") do fits
                write(fits, cube)
            end
            @test_throws exception Replay.read_cube(path)
        end
    end
end

end
