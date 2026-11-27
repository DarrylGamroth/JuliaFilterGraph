# Exercise the identical portable eager kernels without advertising CPU capture.
include(joinpath(GPU_SOURCE, "algorithms", "docrime", "excitation.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_pixel.jl"))
_with_eager_device(f, ::TestDevice) = f()
_require_current_device(::TestDevice) = nothing
_require_array_device(name, values::Array, ::TestDevice) = nothing

@testset "portable eager DO-CRIME state and precision" begin
    for T in (Float32, Float64), Plan in (DocrimeExcitationPlan, DocrimeBinaryExcitationPlan)
        cpu = Plan(T[1, 2, 0.5], UInt64(0x123456789abcdef0))
        kernel = Plan === DocrimeExcitationPlan ? _docrime_continuous_kernel! : _docrime_binary_kernel!
        amplitudes = copy(cpu.amplitudes)
        backend = _docrime_backend(TEST_DEVICE, amplitudes, kernel)
        plan = FilterGraphAlgorithms._prepared_docrime_excitation_plan(Plan, amplitudes, cpu.seed, backend)
        workspace = DocrimeExcitationWorkspace(plan)
        reference = DocrimeExcitationWorkspace(cpu)
        expected, actual = zeros(T, 3), zeros(T, 3)
        for _ in 1:4
            process!(expected, reference, cpu)
            _process_docrime_excitation!(actual, workspace, plan)
            @test actual == expected
            @test docrime_excitation_state(workspace) == docrime_excitation_state(reference)
            commit!(workspace); commit!(reference)
        end
        reset!(workspace); reset!(reference)
        _process_docrime_excitation!(actual, workspace, plan)
        process!(expected, reference, cpu)
        @test actual == expected
        @test capture_support(DocrimeExcitationF32, TEST_DEVICE) isa NoCapture
    end
end

@testset "portable eager PyramidPixel publication and previous gain" begin
    for T in (Float32, Float64)
        cpu = PyramidPixelPlan{T}([true, false])
        backend = PyramidPixelBackend(TEST_DEVICE,
            _pyramid_compute_kernel!(BACKEND, PYRAMID_WORKGROUP_SIZE),
            _pyramid_publish_kernel!(BACKEND, PYRAMID_WORKGROUP_SIZE))
        plan = FilterGraphAlgorithms._prepared_pyramid_pixel_plan(T, copy(cpu.mask),
            copy(cpu.selected_pixel_offsets), cpu.output_pixel_count, cpu.mean_scale, backend)
        workspace = PyramidPixelWorkspace(plan)
        output = PyramidPixelOutput(fill(T(7), 4), T[11])
        input = fill(T(2), 2, 4)
        _process_pyramid_pixel!(output, workspace, plan, input)
        @test output.pixels == zeros(T, 4)
        @test output.mean_pupil_intensity == T[2]
        commit!(workspace)
        @test only(workspace.reciprocal_previous_frame_mean) == T(0.5)
        input .= T(4)
        _process_pyramid_pixel!(output, workspace, plan, input)
        @test output.pixels == fill(T(2), 4)
        discard!(workspace)
        prior = copy(output.pixels)
        input[2, 1] = T(NaN)
        @test_throws PyramidPixelNumericalError _process_pyramid_pixel!(output, workspace, plan, input)
        @test output.pixels == prior
        @test only(workspace.reciprocal_previous_frame_mean) == T(0.5)
        reset!(workspace)
        @test iszero(only(workspace.reciprocal_previous_frame_mean))
    end
end
