module ModalCommandAllocationTests
using Test
import FilterGraphs as FG
include(joinpath(@__DIR__, "data", "examples", "modal-command-allocation/run.jl"))
const E = ModalCommandAllocationExample

allocation_bytes(p, x) = @allocated E.process_frame!(p, x)
clip_allocation_bytes(p) = @allocated E.allocate_commands!(p,clip_tt!)
function clip_tt!(commanded, requested)
    for i in eachindex(commanded, requested)
        commanded[i] = clamp(requested[i], -0.025f0, 0.025f0)
    end
    return nothing
end
reject_tt!(commanded, requested) = :tt_rejected
nonfinite_tt!(commanded, requested) = (fill!(commanded,NaN32); nothing)

@testset "HEART layouts and one per-mode temporal bank" begin
    samples = (Float32[1,-.5,.25,.75,-.2], Float32[.2,.1,-.4,.3,.5], zeros(Float32,5))
    reference = nothing
    for low_modes_last in (false,true)
        p = E.prepare_example(; low_modes_last)
        s = p.selection
        history = zeros(Float32,5)
        command = zeros(Float32,5)
        results = []
        for sample in samples
            # Independent scalar difference equation; no package filter kernel.
            for mode in 1:5
                alpha = .15f0 + .1f0 * mode
                history[mode] = alpha * sample[mode] + (1f0-alpha) * history[mode]
                command[mode] = E.POLE * command[mode] + E.GAIN * history[mode]
            end
            @test E.process_frame!(p, sample[s.permutation]) === nothing
            b = p.buffers
            @test b.controller ≈ command[s.permutation]
            q = E.BASIS * command
            tt = E.TT * q
            @test b.vdm ≈ q
            @test b.commanded_tt ≈ tt
            @test b.woofer ≈ E.WOOFER*q - E.TT_TO_WOOFER*tt
            @test b.tweeter ≈ E.TWEETER*q
            push!(results, (copy(b.commanded_tt),copy(b.woofer),copy(b.tweeter)))
        end
        if reference === nothing
            reference = results
        else
            for (actual, expected) in zip(results,reference), i in 1:3
                @test actual[i] ≈ expected[i]
            end
        end
        E.reset!(p)
        @test E.process_frame!(p,samples[1][s.permutation]) === nothing
        fresh = E.prepare_example(; low_modes_last)
        @test E.process_frame!(fresh,samples[1][s.permutation]) === nothing
        @test p.buffers.controller == fresh.buffers.controller
        @test p.buffers.woofer == fresh.buffers.woofer
        x = samples[1][s.permutation]
        E.process_frame!(p,x)
        allocation_bytes(p,x)
        @test allocation_bytes(p,x) == 0
    end

end

@testset "a rejected allocation requires reset before another frame" begin
    p = E.prepare_example()
    sample = ones(Float32, 5)
    @test E.process_frame!(p, sample, reject_tt!) === :tt_rejected
    # The controller committed before the callback rejected the allocation.
    committed = copy(p.buffers.controller)
    @test any(!iszero, committed)
    @test_throws ErrorException E.process_frame!(p, sample)
    @test p.buffers.controller == committed

    @test E.reset!(p) === p
    @test all(buffer -> all(iszero, buffer), values(p.buffers))
    @test E.process_frame!(p, sample) === nothing
    fresh = E.prepare_example()
    @test E.process_frame!(fresh, sample) === nothing
    @test p.buffers == fresh.buffers
end

@testset "allocation uses conditioned TT and explicit VDM basis" begin
    p = E.prepare_example()
    p.buffers.controller .= Float32[.7,-.4,.3,.2,-.6]
    @test E.allocate_commands!(p,clip_tt!) === nothing
    b = p.buffers
    q = E.BASIS*b.controller
    @test q != b.controller
    @test b.requested_tt ≈ E.TT*q
    @test b.commanded_tt == clamp.(b.requested_tt,-.025f0,.025f0)
    @test b.woofer ≈ E.WOOFER*q - E.TT_TO_WOOFER*b.commanded_tt
    @test !isapprox(b.woofer,E.WOOFER*q - E.TT_TO_WOOFER*b.requested_tt)
    @test b.tweeter ≈ E.TWEETER*q
    E.allocate_commands!(p,clip_tt!)
    clip_allocation_bytes(p)
    @test clip_allocation_bytes(p) == 0
    fill!(b.woofer,99f0)
    fill!(b.tweeter,98f0)
    @test E.allocate_commands!(p,reject_tt!) === :tt_rejected
    @test b.woofer == fill(99f0,3)
    @test b.tweeter == fill(98f0,4)

    # Finite projections can still overflow during native woofer subtraction.
    projection = zeros(Float32,3,5)
    projection[1,1] = floatmax(Float32)
    E.FGA.replace_parameter!(p.plans.woofer, "vdm-to-pdm", projection)
    b.controller .= Float32[.75,0,0,0,0]
    extreme_tt!(commanded, requested) = (fill!(commanded,-floatmax(Float32)); nothing)
    @test E.allocate_commands!(p,extreme_tt!) === :nonfinite_woofer
    @test all(isfinite,b.requested_woofer)
    @test all(isfinite,b.tt_woofer)
    @test b.woofer == fill(99f0,3)
    @test b.tweeter == fill(98f0,4)
end

@testset "failure propagation and graph metadata" begin
    p = E.prepare_example()
    metadata = E.FGA.SampleMetadata(7; timestamp_nanoseconds=70)
    @test FG.process!((output=p.buffers.controller,), p.graph,
        (input=ones(Float32,5),),
        (input=metadata,)) == (output=true,)
    @test FG.output_metadata(p.graph) == (output=metadata,)
    previous = copy(p.buffers.controller)
    @test E.process_frame!(p,fill(NaN32,5)) === E.FGA.BiquadCascadeNonFiniteInput
    @test p.buffers.controller == previous
    @test FG.output_metadata(p.graph) == (output=metadata,)
    @test_throws ErrorException E.process_frame!(p, ones(Float32, 5))
    E.reset!(p)
    fill!(p.buffers.woofer,99f0)
    fill!(p.buffers.tweeter,98f0)
    @test E.allocate_commands!(p,nonfinite_tt!) === E.FGA.DenseReconstructionNonFiniteInput
    @test p.buffers.woofer == fill(99f0,3)
    @test p.buffers.tweeter == fill(98f0,4)
end
end
