using TOML

import FilterGraphAlgorithms:
    CurvaturePairedChannelF32,
    CurvaturePairedChannelF32Configuration,
    CurvaturePairedImageF32,
    CurvaturePairedImageF32Configuration,
    CurvaturePairedSignalCalibrationMismatch,
    CurvaturePairedSignalNonFiniteInput,
    CurvaturePairedSignalNonFiniteOutput,
    complete!,
    enqueue!,
    prepare_capture

const CURVATURE_PAIRED_SIGNAL_FIXTURE = TOML.parsefile(joinpath(@__DIR__, "..", "data", "algorithms", "curvature_paired_signal_synthetic.toml"))

function curvature_paired_image_configuration(;
    reference=Float32.(CURVATURE_PAIRED_SIGNAL_FIXTURE["calibration"]["reference_signal"]),
    scales=Float32.(CURVATURE_PAIRED_SIGNAL_FIXTURE["calibration"]["branch_scales"]),
)
    fixture = CURVATURE_PAIRED_SIGNAL_FIXTURE
    geometry, calibration = fixture["geometry"], fixture["calibration"]
    return CurvaturePairedImageF32Configuration(
        image_size=(geometry["image_size"][1], geometry["image_size"][2]),
        signal_size=(geometry["signal_size"][1], geometry["signal_size"][2]),
        branch_size=(geometry["branch_size"][1], geometry["branch_size"][2]),
        positive_branch_origin=geometry["positive_branch_origin"],
        negative_branch_origin=geometry["negative_branch_origin"],
        curvature_support=Bool.(geometry["support"]),
        initial_reference_signal=reference,
        initial_branch_scales=scales,
        initial_calibration_signature=UInt64(calibration["signature"]),
    )
end

function curvature_paired_channel_configuration(;
    reference=Float32.(CURVATURE_PAIRED_SIGNAL_FIXTURE["calibration"]["reference_signal"]),
    scales=Float32.(CURVATURE_PAIRED_SIGNAL_FIXTURE["calibration"]["branch_scales"]),
)
    fixture = CURVATURE_PAIRED_SIGNAL_FIXTURE
    geometry, calibration = fixture["geometry"], fixture["calibration"]
    return CurvaturePairedChannelF32Configuration(
        input_branch_rows=3,
        input_channels=length(geometry["channel_offsets"]),
        signal_size=(geometry["signal_size"][1], geometry["signal_size"][2]),
        positive_branch_row=geometry["positive_branch_row"],
        negative_branch_row=geometry["negative_branch_row"],
        channel_offsets=geometry["channel_offsets"],
        curvature_support=Bool.(geometry["support"]),
        initial_reference_signal=reference,
        initial_branch_scales=scales,
        initial_calibration_signature=UInt64(calibration["signature"]),
    )
end

function run_curvature_paired_signal_capture_tests(
    device,
    to_device,
    device_fill,
    device_zeros,
    synchronize,
    array_device,
    used_memory,
)
    function enqueue_curvature_paired_signal_frame!(captured, frame_status)
        fill!(frame_status, Int32(0))
        enqueue!(captured)
        synchronize()
        return only(Array(frame_status))
    end

    @testset "captured paired Curvature image matches the synthetic fixture" begin
        fixture = CURVATURE_PAIRED_SIGNAL_FIXTURE
        geometry, frame = fixture["geometry"], fixture["frame"]
        prepared = prepare_algorithm(
            CurvaturePairedImageF32,
            curvature_paired_image_configuration(),
        )
        host_image = reshape(Float32.(frame["image"]), Tuple(geometry["image_size"])...)
        expected = Float32.(fixture["expected"]["signal"])
        signature = UInt64[prepared.plan.calibration_signature]
        image = to_device(host_image)
        frame_calibration_signature = to_device(signature)
        signal = device_fill(-71.0f0, length(expected))
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (signal,), (image, frame_calibration_signature),
            frame_status, Int32(1),
        )

        @test iszero(enqueue_curvature_paired_signal_frame!(captured, frame_status))
        @test Array(signal) ≈ expected rtol=2eps(Float32)
        @test array_device(captured.staged_signal) == device
        @test array_device(captured.support) == device
        @test array_device(captured.reference_signal) == device
        device_bytes = used_memory()
        for _ in 1:8
            @test iszero(enqueue_curvature_paired_signal_frame!(captured, frame_status))
        end
        @test used_memory() <= device_bytes
    end

    @testset "captured paired Curvature channels match the synthetic fixture" begin
        fixture = CURVATURE_PAIRED_SIGNAL_FIXTURE
        geometry, frame = fixture["geometry"], fixture["frame"]
        prepared = prepare_algorithm(
            CurvaturePairedChannelF32,
            curvature_paired_channel_configuration(),
        )
        host_channels = reshape(
            Float32.(frame["channels"]), 3, length(geometry["channel_offsets"]),
        )
        expected = Float32.(fixture["expected"]["signal"])
        signature = UInt64[prepared.plan.calibration_signature]
        channels = to_device(host_channels)
        frame_calibration_signature = to_device(signature)
        signal = device_fill(-71.0f0, length(expected))
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (signal,), (channels, frame_calibration_signature),
            frame_status, Int32(1),
        )

        @test iszero(enqueue_curvature_paired_signal_frame!(captured, frame_status))
        @test Array(signal) ≈ expected rtol=2eps(Float32)
        @test array_device(captured.staged_signal) == device
        @test array_device(captured.channel_offsets) == device
        @test array_device(captured.support) == device
        @test array_device(captured.reference_signal) == device
        device_bytes = used_memory()
        for _ in 1:8
            @test iszero(enqueue_curvature_paired_signal_frame!(captured, frame_status))
        end
        @test used_memory() <= device_bytes
    end

    @testset "captured paired Curvature failures preserve caller output" begin
        fixture = CURVATURE_PAIRED_SIGNAL_FIXTURE
        geometry, frame, calibration = fixture["geometry"], fixture["frame"], fixture["calibration"]
        image_host = reshape(Float32.(frame["image"]), Tuple(geometry["image_size"])...)
        channels_host = reshape(
            Float32.(frame["channels"]), 3, length(geometry["channel_offsets"]),
        )
        declarations = (
            (CurvaturePairedImageF32, curvature_paired_image_configuration(), image_host),
            (CurvaturePairedChannelF32, curvature_paired_channel_configuration(), channels_host),
        )

        for (declaration, configuration, host_observation) in declarations
            prepared = prepare_algorithm(declaration, configuration)
            signature = UInt64[prepared.plan.calibration_signature]
            observation = to_device(host_observation)
            frame_calibration_signature = to_device(signature)
            sentinel = fill(71.0f0, length(fixture["expected"]["signal"]))
            signal = to_device(sentinel)
            frame_status = device_zeros(Int32, 1)
            captured = prepare_capture(
                CompleteTransition(),
                device, prepared, (signal,), (observation, frame_calibration_signature),
                frame_status, Int32(1),
            )

            copyto!(frame_calibration_signature, UInt64[signature[1] + 1])
            status = enqueue_curvature_paired_signal_frame!(captured, frame_status)
            @test complete!(captured, status) ===
                  CurvaturePairedSignalCalibrationMismatch
            @test Array(signal) == sentinel

            copyto!(frame_calibration_signature, signature)
            nonfinite = copy(host_observation)
            nonfinite[end] = NaN32
            copyto!(observation, nonfinite)
            status = enqueue_curvature_paired_signal_frame!(captured, frame_status)
            @test complete!(captured, status) ===
                  CurvaturePairedSignalNonFiniteInput
            @test Array(signal) == sentinel

            copyto!(frame_calibration_signature, UInt64[signature[1] + 1])
            status = enqueue_curvature_paired_signal_frame!(captured, frame_status)
            @test complete!(captured, status) ===
                  CurvaturePairedSignalCalibrationMismatch
            @test Array(signal) == sentinel
        end

        overflow_reference = Float32.(calibration["reference_signal"])
        overflow_reference[1] = -floatmax(Float32)
        overflow_scale = Float32[floatmax(Float32), 1.0f0]
        overflow_prepared = prepare_algorithm(
            CurvaturePairedImageF32,
            curvature_paired_image_configuration(
                reference=overflow_reference,
                scales=overflow_scale,
            ),
        )
        overflow_signal = device_fill(71.0f0, length(fixture["expected"]["signal"]))
        overflow_status = device_zeros(Int32, 1)
        overflow_captured = prepare_capture(
            CompleteTransition(),
            device,
            overflow_prepared,
            (overflow_signal,),
            (
                to_device(image_host),
                to_device(UInt64[overflow_prepared.plan.calibration_signature]),
            ),
            overflow_status,
            Int32(1),
        )
        status = enqueue_curvature_paired_signal_frame!(overflow_captured, overflow_status)
        @test complete!(overflow_captured, status) ===
              CurvaturePairedSignalNonFiniteOutput
        @test Array(overflow_signal) == fill(71.0f0, length(fixture["expected"]["signal"]))

        signed_overflow_image = zeros(Float32, Tuple(geometry["image_size"])...)
        signed_overflow_image[1:2, 2:3] .= 2.0f0
        signed_overflow_image[3:4, 2:3] .= -2.0f0
        signed_overflow_channels = zeros(Float32, 3, length(geometry["channel_offsets"]))
        signed_overflow_channels[1, geometry["channel_offsets"][1] + 1] = 2.0f0
        signed_overflow_channels[2, geometry["channel_offsets"][1] + 1] = -2.0f0
        signed_overflow_cases = (
            (
                CurvaturePairedImageF32,
                curvature_paired_image_configuration(scales=fill(floatmax(Float32), 2)),
                signed_overflow_image,
            ),
            (
                CurvaturePairedChannelF32,
                curvature_paired_channel_configuration(scales=fill(floatmax(Float32), 2)),
                signed_overflow_channels,
            ),
        )
        for (declaration, configuration, host_observation) in signed_overflow_cases
            prepared = prepare_algorithm(declaration, configuration)
            sentinel = fill(71.0f0, length(fixture["expected"]["signal"]))
            signal = to_device(sentinel)
            frame_status = device_zeros(Int32, 1)
            captured = prepare_capture(
                CompleteTransition(),
                device,
                prepared,
                (signal,),
                (
                    to_device(host_observation),
                    to_device(UInt64[prepared.plan.calibration_signature]),
                ),
                frame_status,
                Int32(1),
            )
            status = enqueue_curvature_paired_signal_frame!(captured, frame_status)
            @test complete!(captured, status) ===
                  CurvaturePairedSignalNonFiniteOutput
            @test Array(signal) == sentinel
        end
    end
end
