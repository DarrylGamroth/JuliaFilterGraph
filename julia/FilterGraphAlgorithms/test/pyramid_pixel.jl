module PyramidPixelTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

@testset "pyramid pupil mask preparation" begin
    @test_throws ArgumentError PyramidPixelPlan(Bool[])
    @test_throws ArgumentError PyramidPixelPlan([false, false])

    source_mask = [true, false, true]
    plan = PyramidPixelPlan(source_mask)
    source_mask[1] = false
    @test pyramid_pupil_mask(plan) == [true, false, true]
    @test pyramid_pupil_pixel_count(plan) == 3
    @test pyramid_selected_pixel_count(plan) == 2
    @test pyramid_output_pixel_count(plan) == 8
    @test PyramidPixelPlan{Float16}([true]) isa PyramidPixelPlan{Float16}
    @test PyramidPixelPlan{Float64}([true]) isa PyramidPixelPlan{Float64}
end

@testset "previous-frame gain and full-pupil mean" begin
    plan = PyramidPixelPlan{Float32}([true, false, true])
    workspace = PyramidPixelWorkspace(plan)
    pixels = fill(Float32(NaN), 8)
    mean_pupil_intensity = fill(Float32(NaN), 1)
    output = PyramidPixelOutput(pixels, mean_pupil_intensity)
    first_frame = Float32[
        1 4 7 10
        2 5 8 11
        3 6 9 12
    ]

    @test process!(output, workspace, plan, first_frame) === output
    @test pixels == zeros(Float32, 8)
    @test mean_pupil_intensity[1] ≈ 6.5f0
    @test iszero(pyramid_normalization_gain(workspace))
    @test commit!(workspace) === workspace
    @test pyramid_normalization_gain(workspace) ≈ 2.0f0 / 13.0f0

    second_frame = Float32[
        2 6 10 14
        20 40 60 80
        4 8 12 16
    ]
    process!(output, workspace, plan, second_frame)
    expected_selected = Float32[2, 4, 6, 8, 10, 12, 14, 16]
    @test pixels ≈ expected_selected .* (2.0f0 / 13.0f0)
    @test mean_pupil_intensity[1] ≈ 272.0f0 / 12.0f0
    commit!(workspace)
    @test pyramid_normalization_gain(workspace) ≈ 12.0f0 / 272.0f0
end

@testset "padded and noncontiguous CPU views retain logical order" begin
    plan = PyramidPixelPlan{Float64}([false, true, true])
    workspace = PyramidPixelWorkspace(plan)
    input_parent = fill(99.0, 5, 4)
    input_parent[1:2:5, :] .= [
        1 4 7 10
        2 5 8 11
        3 6 9 12
    ]
    input = @view input_parent[1:2:5, :]
    pixel_parent = fill(-1.0, 16)
    pixels = @view pixel_parent[1:2:15]
    mean_parent = fill(-1.0, 2)
    mean_pupil_intensity = @view mean_parent[2:2]
    output = PyramidPixelOutput(pixels, mean_pupil_intensity)

    process!(output, workspace, plan, fill(1.0, 3, 4))
    commit!(workspace)
    process!(output, workspace, plan, input)
    @test collect(pixels) == [2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]
    @test mean_pupil_intensity[1] == 6.5
    @test all(==(-1.0), @view pixel_parent[2:2:16])
end

@testset "nonpositive mean commits zero following gain" begin
    plan = PyramidPixelPlan{Float32}([true])
    workspace = PyramidPixelWorkspace(plan)
    pixels = zeros(Float32, 4)
    mean_pupil_intensity = zeros(Float32, 1)
    output = PyramidPixelOutput(pixels, mean_pupil_intensity)

    process!(output, workspace, plan, fill(2.0f0, 1, 4))
    commit!(workspace)
    process!(output, workspace, plan, fill(-1.0f0, 1, 4))
    @test pixels == fill(-0.5f0, 4)
    @test mean_pupil_intensity == [-1.0f0]
    @test pyramid_normalization_gain(workspace) == 0.5f0
    commit!(workspace)
    @test iszero(pyramid_normalization_gain(workspace))

    process!(output, workspace, plan, fill(3.0f0, 1, 4))
    @test pixels == zeros(Float32, 4)
end

@testset "rejected frames preserve output and committed gain" begin
    plan = PyramidPixelPlan{Float32}([true, false])
    workspace = PyramidPixelWorkspace(plan)
    pixels = fill(7.0f0, 4)
    mean_pupil_intensity = fill(11.0f0, 1)
    output = PyramidPixelOutput(pixels, mean_pupil_intensity)
    process!(output, workspace, plan, fill(2.0f0, 2, 4))
    commit!(workspace)
    committed = pyramid_normalization_gain(workspace)
    pixels .= 7.0f0
    mean_pupil_intensity .= 11.0f0

    @test_throws DimensionMismatch process!(
        output,
        workspace,
        plan,
        ones(Float32, 1, 4),
    )
    @test pixels == fill(7.0f0, 4)
    @test mean_pupil_intensity == [11.0f0]
    @test pyramid_normalization_gain(workspace) == committed

    nonfinite = fill(1.0f0, 2, 4)
    nonfinite[2, 2] = Float32(NaN)
    error = try
        process!(output, workspace, plan, nonfinite)
        nothing
    catch caught
        caught
    end
    @test error isa PyramidPixelNumericalError
    @test error.status == PyramidPixelNonFiniteInput
    @test pixels == fill(7.0f0, 4)
    @test mean_pupil_intensity == [11.0f0]
    @test pyramid_normalization_gain(workspace) == committed

    process!(
        output,
        workspace,
        plan,
        fill(floatmin(Float32), 2, 4),
    )
    commit!(workspace)
    committed = pyramid_normalization_gain(workspace)
    pixels .= 7.0f0
    mean_pupil_intensity .= 11.0f0
    overflow = Float32[
        floatmax(Float32) -floatmax(Float32) floatmax(Float32) -floatmax(Float32)
        -floatmax(Float32) floatmax(Float32) -floatmax(Float32) floatmax(Float32)
    ]
    error = try
        process!(output, workspace, plan, overflow)
        nothing
    catch caught
        caught
    end
    @test error isa PyramidPixelNumericalError
    @test error.status == PyramidPixelNonFiniteOutput
    @test pixels == fill(7.0f0, 4)
    @test mean_pupil_intensity == [11.0f0]
    @test pyramid_normalization_gain(workspace) == committed
end

@testset "CPU pyramid processing allocates no heap after warmup" begin
    plan = PyramidPixelPlan{Float32}([true, false, true, true])
    workspace = PyramidPixelWorkspace(plan)
    input = reshape(Float32.(1:16), 4, 4)
    output = PyramidPixelOutput(zeros(Float32, 12), zeros(Float32, 1))
    process!(output, workspace, plan, input)
    commit!(workspace)
    @test @allocated(process!(output, workspace, plan, input)) == 0
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0
end

end
