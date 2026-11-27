module MultiNodeExampleTests

using Test

include(joinpath(@__DIR__, "data", "examples", "pixel-pyramid-row/run.jl"))

include(joinpath(@__DIR__, "data", "examples", "docrime-excitation/run.jl"))

include(joinpath(@__DIR__, "data", "examples", "leaky-controller/run.jl"))

include(joinpath(@__DIR__, "data", "examples", "optical-gain-workflow/run.jl"))

include(joinpath(@__DIR__, "data", "examples", "modal-biquad-controller/run.jl"))

@testset "documented modal biquad controller runs as written" begin
    commands = ModalBiquadControllerExample.run_example()
    @test length(commands) == 3
    @test all(command -> length(command) == 3 && all(isfinite, command), commands)
    @test commands[1] ≈ Float32[-0.28, 0.07, -0.07 / 3]
end

@testset "documented multi-Node row Graph runs as written" begin
    result = PixelPyramidRowExample.run_example()
    @test all(values(result.first_availability))
    @test all(values(result.second_availability))
    @test size(result.reconstruction_pixels) == (4, 3)
    @test any(!iszero, result.reconstruction_pixels)
    @test isfinite(result.mean_pupil_intensity)
    @test all(!isnothing, values(result.metadata))
end

@testset "documented source-only Graph runs as written" begin
    samples = DocrimeExcitationExample.run_example()
    @test length(samples) == 3
    @test all(sample -> length(sample) == 3, samples)
    @test all(sample -> all(isfinite, sample), samples)
    @test length(unique(samples)) == length(samples)
end

@testset "documented controller projection Graph runs as written" begin
    result = LeakyControllerExample.run_example()
    @test length(result.demanded_commands) == 2
    @test result.demanded_commands[1] ≈ Float32[-0.085, 0.5]
    @test result.demanded_commands[2] ≈ Float32[-0.4525, 0.8]
    @test result.controller_constraint_feedback[1] ≈
          Float32[-0.0503125, 0.503125, -0.1509375]
    @test result.controller_constraint_feedback[2] ≈
          Float32[-0.02703125, 0.2703125, -0.08109375]
end

@testset "documented optical-gain workflow runs as written" begin
    result = OpticalGainWorkflowExample.run_example()
    @test result.published_gain == 0.75f0
    @test result.corrected_before_adoption == fill(0.75f0, 4)
    @test result.corrected_after_adoption == 1.0f0
end

end
