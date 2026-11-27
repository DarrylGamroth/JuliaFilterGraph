using LinearAlgebra

function _process_pyramid_reconstruction!(
    dm_error,
    pyramid_output,
    pyramid_workspace,
    pyramid_plan,
    reconstructor,
    pupil_images,
)
    process!(
        pyramid_output,
        pyramid_workspace,
        pyramid_plan,
        pupil_images,
    )
    mul!(dm_error, reconstructor, pyramid_output.pixels)
    AMDGPU.synchronize()
    commit!(pyramid_workspace)
    return dm_error
end

function _pyramid_reconstruction_fixture(::Type{T}) where {T<:AbstractFloat}
    pupil_frames = Matrix{T}[
        T[
            1 4 7 10
            2 5 8 11
            3 6 9 12
        ],
        T[
            2 6 10 14
            3 9 15 21
            4 12 20 28
        ],
        T[
            8 7 6 5
            4 3 2 1
            9 10 11 12
        ],
        T[
            1 2 3 4
            5 6 7 8
            9 10 11 12
        ],
    ]
    reconstructor = reshape(T.(1:24), 3, 8) ./ T(32)
    return pupil_frames, reconstructor
end

function _cpu_pyramid_reconstruction(
    ::Type{T},
    pupil_mask,
    pupil_frames,
    reconstructor,
) where {T<:AbstractFloat}
    plan = PyramidPixelPlan{T}(pupil_mask)
    workspace = PyramidPixelWorkspace(plan)
    pyramid_pixels = zeros(T, pyramid_output_pixel_count(plan))
    pyramid_mean = zeros(T, 1)
    pyramid_output = PyramidPixelOutput(pyramid_pixels, pyramid_mean)
    dm_error = zeros(T, size(reconstructor, 1))
    results = Vector{Vector{T}}(undef, length(pupil_frames))

    for (frame_index, pupil_images) in enumerate(pupil_frames)
        process!(pyramid_output, workspace, plan, pupil_images)
        mul!(dm_error, reconstructor, pyramid_pixels)
        commit!(workspace)
        results[frame_index] = copy(dm_error)
    end
    return results
end

@testset "AMDGPU resident pyramid and reconstruction chain" begin
    for T in (Float32, Float64)
        pupil_mask = Bool[true, false, true]
        pupil_frames, host_reconstructor =
            _pyramid_reconstruction_fixture(T)
        expected_dm_errors = _cpu_pyramid_reconstruction(
            T,
            pupil_mask,
            pupil_frames,
            host_reconstructor,
        )

        cpu_plan = PyramidPixelPlan{T}(pupil_mask)
        pyramid_plan = prepare_pyramid_pixel_plan(TEST_DEVICE, cpu_plan)
        pyramid_workspace = PyramidPixelWorkspace(pyramid_plan)
        reconstructor = AMDGPU.ROCArray(host_reconstructor)

        pupil_image_slots = [
            AMDGPU.zeros(T, size(pupil_frames[1])...),
            AMDGPU.zeros(T, size(pupil_frames[1])...),
        ]
        pyramid_output_slots = [
            PyramidPixelOutput(
                AMDGPU.zeros(T, pyramid_output_pixel_count(pyramid_plan)),
                AMDGPU.zeros(T, 1),
            ),
            PyramidPixelOutput(
                AMDGPU.zeros(T, pyramid_output_pixel_count(pyramid_plan)),
                AMDGPU.zeros(T, 1),
            ),
        ]
        dm_error_slots = [
            AMDGPU.zeros(T, size(host_reconstructor, 1)),
            AMDGPU.zeros(T, size(host_reconstructor, 1)),
        ]
        original_pupil_image_slots = copy(pupil_image_slots)
        original_pyramid_pixel_slots =
            [output.pixels for output in pyramid_output_slots]
        original_pyramid_mean_slots =
            [output.mean_pupil_intensity for output in pyramid_output_slots]
        original_dm_error_slots = copy(dm_error_slots)

        for (frame_index, pupil_images) in enumerate(pupil_frames)
            slot = mod1(frame_index, 2)
            copyto!(pupil_image_slots[slot], pupil_images)
            _process_pyramid_reconstruction!(
                dm_error_slots[slot],
                pyramid_output_slots[slot],
                pyramid_workspace,
                pyramid_plan,
                reconstructor,
                pupil_image_slots[slot],
            )
            # Eight-term reductions may differ through GPU FMA contraction.
            @test isapprox(
                Array(dm_error_slots[slot]),
                expected_dm_errors[frame_index];
                rtol=32 * eps(T),
                atol=32 * eps(T),
            )
        end

        @test iszero(norm(expected_dm_errors[1]))
        @test !iszero(norm(expected_dm_errors[2]))
        @test all(
            pupil_image_slots[index] === original_pupil_image_slots[index] for
            index in eachindex(pupil_image_slots)
        )
        @test all(
            pyramid_output_slots[index].pixels ===
            original_pyramid_pixel_slots[index] for
            index in eachindex(pyramid_output_slots)
        )
        @test all(
            pyramid_output_slots[index].mean_pupil_intensity ===
            original_pyramid_mean_slots[index] for
            index in eachindex(pyramid_output_slots)
        )
        @test all(
            dm_error_slots[index] === original_dm_error_slots[index] for
            index in eachindex(dm_error_slots)
        )

        @test AMDGPU.device(reconstructor) == TEST_DEVICE
        @test AMDGPU.device(pyramid_plan.mask) == TEST_DEVICE
        @test AMDGPU.device(pyramid_plan.selected_pixel_offsets) ==
              TEST_DEVICE
        @test AMDGPU.device(
            pyramid_workspace.reciprocal_previous_frame_mean,
        ) == TEST_DEVICE
        @test AMDGPU.device(pyramid_workspace.staged_pixels) == TEST_DEVICE
        @test AMDGPU.device(
            pyramid_workspace.staged_mean_pupil_intensity,
        ) == TEST_DEVICE
        @test AMDGPU.device(
            pyramid_workspace.staged_reciprocal_current_frame_mean,
        ) == TEST_DEVICE
        @test AMDGPU.device(pyramid_workspace.status) == TEST_DEVICE
        @test pyramid_workspace.completion_status isa Vector{Int32}
        for slot in eachindex(pupil_image_slots)
            @test AMDGPU.device(pupil_image_slots[slot]) == TEST_DEVICE
            @test AMDGPU.device(pyramid_output_slots[slot].pixels) ==
                  TEST_DEVICE
            @test AMDGPU.device(
                pyramid_output_slots[slot].mean_pupil_intensity,
            ) == TEST_DEVICE
            @test AMDGPU.device(dm_error_slots[slot]) == TEST_DEVICE
        end

        fill!(dm_error_slots[1], T(17))
        committed_gain = pyramid_normalization_gain(pyramid_workspace)
        nonfinite_frame = copy(pupil_frames[1])
        nonfinite_frame[2, 3] = T(NaN)
        copyto!(pupil_image_slots[1], nonfinite_frame)
        @test_throws PyramidPixelNumericalError _process_pyramid_reconstruction!(
            dm_error_slots[1],
            pyramid_output_slots[1],
            pyramid_workspace,
            pyramid_plan,
            reconstructor,
            pupil_image_slots[1],
        )
        @test Array(dm_error_slots[1]) == fill(T(17), length(dm_error_slots[1]))
        @test pyramid_normalization_gain(pyramid_workspace) == committed_gain

        for slot in eachindex(pupil_image_slots)
            copyto!(pupil_image_slots[slot], pupil_frames[slot])
            _process_pyramid_reconstruction!(
                dm_error_slots[slot],
                pyramid_output_slots[slot],
                pyramid_workspace,
                pyramid_plan,
                reconstructor,
                pupil_image_slots[slot],
            )
        end
        device_bytes = AMDGPU.used_memory()
        for sample in 1:16
            slot = mod1(sample, 2)
            _process_pyramid_reconstruction!(
                dm_error_slots[slot],
                pyramid_output_slots[slot],
                pyramid_workspace,
                pyramid_plan,
                reconstructor,
                pupil_image_slots[slot],
            )
        end
        @test AMDGPU.used_memory() == device_bytes
    end
end
