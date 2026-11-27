# Backend-neutral checks used by CUDA and HIP hardware runners.
using FilterGraphAlgorithms, LinearAlgebra, Test
import FilterGraphAlgorithms: process!, prepare_capture,
    enqueue!, complete!

function run_numerical_range_capture_tests(device, device_array, device_zeros, synchronize)
    @testset "captured SCC recurrence is invariant under measurement scaling" begin
        H = Float32[1 0; 0 2; 1 1]
        prepared = prepare_algorithm(SccWeightedRegularizedReconstructorF32,
            SccWeightedRegularizedReconstructorF32Configuration(
                measurement_count=3, controller_coordinate_count=2, regularization_row_count=2,
                measurement_schema=SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
                initial_interaction_matrix=collect(vec(transpose(H))),
                initial_regularization_operator=Float32[1, 0, 0, 1],
                regularization=0.125f0, iterations=8, relative_tolerance=1f-5, minimum_diagonal=1f-6))
        base = H * Float32[0.5, -0.25]
        host_weights = Float32[1, 0.5, 2]
        measurements, weights = device_array(base), device_array(host_weights)
        output, status = device_zeros(Float32, 2), device_zeros(Int32, 1)
        captured = prepare_capture(CompleteTransition(), device, prepared, (output,), (measurements, weights), status, Int32(1))
        for exponent in (-120, -90, -30, 0, 30, 90, 120)
            host_measurements = ldexp.(base, exponent)
            cpu_output = zeros(Float32, 2)
            @test process!(cpu_output, prepared, host_measurements, host_weights) === nothing
            copyto!(measurements, host_measurements)
            copyto!(weights, host_weights)
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test Array(output) ≈ cpu_output rtol=2f-5 atol=0
        end
        fill!(measurements, 0f0)
        fill!(status, Int32(0))
        enqueue!(captured)
        synchronize()
        @test only(Array(status)) == 0
        @test Array(output) == zeros(Float32, 2)
        fill!(output, 71f0)
        fill!(weights, -1f0)
        fill!(status, Int32(0))
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) === SccWeightedRegularizedReconstructionInvalidWeight
        @test Array(output) == fill(71f0, 2)
    end

    @testset "captured curvature image and channel preserve extreme ratios" begin
        signature_value = UInt64(7)
        image = prepare_algorithm(CurvaturePairedImageF32,
            CurvaturePairedImageF32Configuration(image_size=(2, 1),
                signal_size=(1, 1), branch_size=(1, 1),
                positive_branch_origin=[0, 0], negative_branch_origin=[1, 0],
                curvature_support=Bool[true], initial_reference_signal=Float32[0.125],
                initial_branch_scales=Float32[1, 1], initial_calibration_signature=signature_value))
        channel = prepare_algorithm(CurvaturePairedChannelF32,
            CurvaturePairedChannelF32Configuration(input_branch_rows=2, input_channels=1,
                signal_size=(1, 1), positive_branch_row=0, negative_branch_row=1,
                channel_offsets=[0], curvature_support=Bool[true], initial_reference_signal=Float32[0.125],
                initial_branch_scales=Float32[1, 1], initial_calibration_signature=signature_value))
        maximum = floatmax(Float32)
        for prepared in (image, channel)
            observation = device_zeros(Float32, 2, 1)
            signature = device_array(UInt64[signature_value])
            output, status = device_zeros(Float32, 1), device_zeros(Int32, 1)
            captured = prepare_capture(CompleteTransition(), device, prepared, (output,), (observation, signature), status, Int32(1))
            for pair in ((maximum, maximum / 2), (maximum, maximum),
                (maximum, -maximum / 2), (-maximum, -maximum), (eps(Float32), eps(Float32)))
                host_observation = reshape(Float32[pair...], 2, 1)
                cpu_output = zeros(Float32, 1)
                @test process!(cpu_output, prepared, host_observation, UInt64[signature_value]) === nothing
                copyto!(observation, host_observation)
                fill!(status, Int32(0))
                enqueue!(captured)
                synchronize()
                @test only(Array(status)) == 0
                @test Array(output) ≈ cpu_output rtol=8eps(Float32) atol=0
            end
            fill!(output, 71f0)
            fill!(observation, NaN32)
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test complete!(captured, only(Array(status))) === CurvaturePairedSignalNonFiniteInput
            @test Array(output) == Float32[71]
        end
    end
end

function run_vdm_correction_role_capture_tests(device, device_array, device_zeros, synchronize)
    @testset "captured VDM correction roles publish every code and preserve failed roles" begin
        prepared = prepare_algorithm(VdmToPdmF32,
            VdmToPdmF32Configuration(active_vdm_count=2, full_vdm_count=4, physical_actuator_count=4))
        replace_parameter!(prepared, "active-to-full", Float32[1 0; 0 1; 0.5 0.5; 0 0])
        replace_parameter!(prepared, "vdm-to-pdm", Float32[0 0 0 1; 1 0 0 0; 0 0 1 0; 1 1 0 0])
        @test prepared.plan.correction_roles == UInt8[0, 1, 2, 3]
        input = device_array(Float32[2, 4])
        output, roles, status = device_zeros(Float32, 4), device_zeros(UInt8, 4), device_zeros(Int32, 1)
        captured = prepare_capture(CompleteTransition(), device, prepared, (output, roles), (input,), status, Int32(1))
        for _ in 1:2
            copyto!(input, Float32[2, 4])
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test Array(output) == Float32[0, 2, 3, 6]
            @test Array(roles) == UInt8[0, 1, 2, 3]
            fill!(output, 71f0)
            fill!(roles, UInt8(71))
            fill!(input, NaN32)
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test complete!(captured, only(Array(status))) === VdmToPdmNonFiniteInput
            # Projection scratch is speculative; only role publication is gated here.
            @test Array(roles) == fill(UInt8(71), 4)
            fill!(input, floatmax(Float32))
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test complete!(captured, only(Array(status))) === VdmToPdmNonFiniteOutput
            @test Array(roles) == fill(UInt8(71), 4)
            @test FilterGraphAlgorithms.reset!(captured) === captured
            synchronize()
            @test Array(captured.full_vdm) == zeros(Float32, 4)
        end
    end
end
