module StatefulExampleTests

using FilterGraphAlgorithms: process!, reset!, commit!, discard!
using Test

include(joinpath(@__DIR__, "data", "examples", "stateful-algorithm", "run.jl"))
using .StatefulAlgorithmExample: prepare_accumulator, NonfiniteInput, NonfiniteOutput

@testset "scientist-owned stateful Algorithm example" begin
    result = StatefulAlgorithmExample.run_example()
    @test result.first_state == Float32[1, 2]
    @test result.rejection === NonfiniteInput
    @test result.final_state == Float32[3, 3]

    prepared = prepare_accumulator(2)
    output = fill(-1.0f0, 2)
    @test process!(output, prepared, Float32[2, 4]) === nothing
    @test output == Float32[1, 2]
    snapshot = copy(prepared.workspace.committed)
    @test process!(output, prepared, Float32[NaN, 1]) === NonfiniteInput
    @test output == snapshot
    @test prepared.workspace.committed == prepared.workspace.candidate == snapshot
    @test process!(output, prepared, Float32[2, 2]) === nothing
    @test process!(output, prepared, Float32[2, 2]) === nothing
    @test snapshot == Float32[1, 2]
    @test prepared.workspace.committed == Float32[3, 4]
    reset!(prepared)
    @test prepared.workspace.committed == prepared.workspace.candidate == zeros(Float32, 2)

    overflowing = prepare_accumulator(2; gain=floatmax(Float32))
    fill!(output, -1.0f0)
    @test process!(output, overflowing, Float32[2, 1]) === NonfiniteOutput
    @test output == fill(-1.0f0, 2)
    @test overflowing.workspace.committed == overflowing.workspace.candidate == zeros(Float32, 2)

    # The same numerical method can be used directly with explicit lifecycle.
    @test process!(output, prepared.workspace, prepared.plan, Float32[2, 4]) === nothing
    @test prepared.workspace.committed == zeros(Float32, 2)
    discard!(prepared.workspace)
    @test prepared.workspace.candidate == zeros(Float32, 2)
    @test process!(output, prepared.workspace, prepared.plan, Float32[2, 4]) === nothing
    commit!(prepared.workspace)
    @test prepared.workspace.committed == Float32[1, 2]
    @test_throws ArgumentError prepare_accumulator(0)
    @test_throws ArgumentError prepare_accumulator(2; gain=Inf32)
end

end
