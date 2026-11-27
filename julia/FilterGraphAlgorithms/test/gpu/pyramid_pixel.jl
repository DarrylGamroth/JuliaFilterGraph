import FilterGraphAlgorithms: commit!, discard!, process!, reset!

function run_pyramid_pixel_tests(device, device_array, device_zeros, device_fill, deviceof, used_memory)
    @testset "GPU pyramid pupil-pixel processing" begin
        cpu_plan = PyramidPixelPlan{Float32}([true, false, true])
        plan = prepare_pyramid_pixel_plan(device, cpu_plan)
        workspace = PyramidPixelWorkspace(plan)
        pixels = device_fill(Float32(NaN), 8)
        mean_pupil_intensity = device_fill(Float32(NaN), 1)
        output = PyramidPixelOutput(pixels, mean_pupil_intensity)
        first_frame = device_array(
            Float32[
                1 4 7 10
                2 5 8 11
                3 6 9 12
            ],
        )

        @test process!(output, workspace, plan, first_frame) === output
        @test Array(pixels) == zeros(Float32, 8)
        @test only(Array(mean_pupil_intensity)) ≈ 6.5f0
        @test iszero(pyramid_normalization_gain(workspace))
        commit!(workspace)
        @test pyramid_normalization_gain(workspace) ≈ 2.0f0 / 13.0f0

        second_frame = device_array(
            Float32[
                2 6 10 14
                20 40 60 80
                4 8 12 16
            ],
        )
        process!(output, workspace, plan, second_frame)
        expected_selected = Float32[2, 4, 6, 8, 10, 12, 14, 16]
        @test Array(pixels) ≈ expected_selected .* (2.0f0 / 13.0f0)
        @test only(Array(mean_pupil_intensity)) ≈ 272.0f0 / 12.0f0
        commit!(workspace)
        @test pyramid_normalization_gain(workspace) ≈ 12.0f0 / 272.0f0

        float64_plan = prepare_pyramid_pixel_plan(
            device,
            PyramidPixelPlan{Float64}([true]),
        )
        float64_workspace = PyramidPixelWorkspace(float64_plan)
        float64_output = PyramidPixelOutput(
            device_zeros(Float64, 4),
            device_zeros(Float64, 1),
        )
        process!(
            float64_output,
            float64_workspace,
            float64_plan,
            device_fill(1.0, 1, 4),
        )
        commit!(float64_workspace)
        process!(
            float64_output,
            float64_workspace,
            float64_plan,
            device_fill(2.0, 1, 4),
        )
        @test Array(float64_output.pixels) == fill(2.0, 4)
        @test Array(float64_output.mean_pupil_intensity) == [2.0]
    end

    @testset "GPU failure preserves output and history" begin
        cpu_plan = PyramidPixelPlan{Float32}([true, false])
        plan = prepare_pyramid_pixel_plan(device, cpu_plan)
        workspace = PyramidPixelWorkspace(plan)
        pixels = device_fill(7.0f0, 4)
        mean_pupil_intensity = device_fill(11.0f0, 1)
        output = PyramidPixelOutput(pixels, mean_pupil_intensity)
        process!(output, workspace, plan, device_fill(2.0f0, 2, 4))
        commit!(workspace)
        committed = pyramid_normalization_gain(workspace)
        fill!(pixels, 7.0f0)
        fill!(mean_pupil_intensity, 11.0f0)

        nonfinite = fill(1.0f0, 2, 4)
        nonfinite[2, 2] = Float32(NaN)
        error = try
            process!(output, workspace, plan, device_array(nonfinite))
            nothing
        catch caught
            caught
        end
        @test error isa PyramidPixelNumericalError
        @test error.status == PyramidPixelNonFiniteInput
        @test Array(pixels) == fill(7.0f0, 4)
        @test Array(mean_pupil_intensity) == [11.0f0]
        @test pyramid_normalization_gain(workspace) == committed

        process!(
            output,
            workspace,
            plan,
            device_fill(floatmin(Float32), 2, 4),
        )
        commit!(workspace)
        committed = pyramid_normalization_gain(workspace)
        fill!(pixels, 7.0f0)
        fill!(mean_pupil_intensity, 11.0f0)
        maximum_finite = floatmax(Float32)
        overflow = Float32[
            maximum_finite maximum_finite -maximum_finite -maximum_finite
            -maximum_finite -maximum_finite maximum_finite maximum_finite
        ]
        error = try
            process!(output, workspace, plan, device_array(overflow))
            nothing
        catch caught
            caught
        end
        @test error isa PyramidPixelNumericalError
        @test error.status == PyramidPixelNonFiniteOutput
        @test Array(pixels) == fill(7.0f0, 4)
        @test Array(mean_pupil_intensity) == [11.0f0]
        @test pyramid_normalization_gain(workspace) == committed
    end

    @testset "GPU storage and dispatch remain device-resident" begin
        cpu_plan = PyramidPixelPlan{Float32}([true, true])
        plan = prepare_pyramid_pixel_plan(device, cpu_plan)
        workspace = PyramidPixelWorkspace(plan)
        output = PyramidPixelOutput(device_zeros(Float32, 8), device_zeros(Float32, 1))
        input = device_fill(1.0f0, 2, 4)

        @test deviceof(plan.mask) == device
        @test deviceof(plan.selected_pixel_offsets) == device
        @test deviceof(workspace.reciprocal_previous_frame_mean) == device
        @test deviceof(workspace.staged_pixels) == device
        @test deviceof(workspace.status) == device
        @test_throws MethodError process!(
            PyramidPixelOutput(zeros(Float32, 8), zeros(Float32, 1)),
            workspace,
            plan,
            input,
        )

        process!(output, workspace, plan, input)
        commit!(workspace)
        device_bytes = used_memory()
        for _ in 1:16
            process!(output, workspace, plan, input)
            commit!(workspace)
        end
        # Direct HIP launches may allocate host bookkeeping. The zero-host-allocation
        # contract applies to captured HIP Graph replay, tested separately.
        @test used_memory() == device_bytes
    end


    @testset "pyramid eager failure priority, discard and reset for both precisions" begin
        for T in (Float32, Float64)
            plan = prepare_pyramid_pixel_plan(device, PyramidPixelPlan{T}([true, false]))
            workspace = PyramidPixelWorkspace(plan)
            output = PyramidPixelOutput(device_fill(T(7), 4), device_fill(T(11), 1))
            input = device_fill(T(2), 2, 4)
            for array in (plan.mask, plan.selected_pixel_offsets,
                workspace.reciprocal_previous_frame_mean, workspace.staged_pixels,
                workspace.staged_mean_pupil_intensity,
                workspace.staged_reciprocal_current_frame_mean, workspace.status,
                output.pixels, output.mean_pupil_intensity, input)
                @test deviceof(array) == device
            end
            process!(output, workspace, plan, input)
            discard!(workspace)
            @test iszero(pyramid_normalization_gain(workspace))
            process!(output, workspace, plan, input)
            commit!(workspace)
            prior = pyramid_normalization_gain(workspace)
            @test prior == T(0.5)
            fill!(output.pixels, T(7)); fill!(output.mean_pupil_intensity, T(11))
            bad = fill(floatmax(T), 2, 4)
            bad[2, 1] = T(NaN) # unselected input beats simultaneous mean overflow
            caught = try
                process!(output, workspace, plan, device_array(bad))
                nothing
            catch error
                error
            end
            @test caught isa PyramidPixelNumericalError
            @test caught.status == PyramidPixelNonFiniteInput
            commit!(workspace) # a rejected candidate cannot adopt its gain
            @test pyramid_normalization_gain(workspace) == prior
            @test Array(output.pixels) == fill(T(7), 4)
            @test Array(output.mean_pupil_intensity) == T[11]
            reset!(workspace)
            @test iszero(pyramid_normalization_gain(workspace))
            process!(output, workspace, plan, input)
            @test Array(output.pixels) == zeros(T, 4)
        end
    end

end
