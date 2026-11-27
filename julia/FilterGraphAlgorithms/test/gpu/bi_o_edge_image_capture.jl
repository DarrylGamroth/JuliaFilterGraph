import FilterGraphAlgorithms:
    complete!,
    enqueue!,
    prepare_capture

function bi_o_edge_image_prepared(
    ;
    optical_gain=ones(Float32, 6),
    reference_signal=zeros(Float32, 6),
    normalization_policy="mean-valid-flux",
    incidence_flux=0.0f0,
)
    return prepare_algorithm(
        BiOEdgeImageF32,
        BiOEdgeImageF32Configuration(
            image_size=(4, 4),
            pupil_size=(2, 2),
            pupil_origins=[[0, 0], [2, 0], [2, 2], [0, 2]],
            bi_o_edge_support=Bool[true, false, true, true],
            initial_reference_signal=reference_signal,
            initial_optical_gain=optical_gain,
            normalization_policy=normalization_policy,
            incidence_flux=incidence_flux,
            initial_calibration_signature=UInt64(0x52a7_1b9c),
        ),
    )
end

function bi_o_edge_image_direct(image, plan)
    q1, q2, q3, q4 = plan.origins
    pupil_rows = plan.pupil_size[1]
    divisor = 0.0f0
    for offset in plan.selected_offsets
        row = rem(offset, pupil_rows) + 1
        column = div(offset, pupil_rows) + 1
        divisor +=
            image[q1[1] + row, q1[2] + column] +
            image[q2[1] + row, q2[2] + column] +
            image[q3[1] + row, q3[2] + column] +
            image[q4[1] + row, q4[2] + column]
    end
    divisor /= length(plan.selected_offsets)
    signal = Matrix{Float32}(undef, length(plan.selected_offsets), 2)
    for (selected, offset) in enumerate(plan.selected_offsets)
        row = rem(offset, pupil_rows) + 1
        column = div(offset, pupil_rows) + 1
        value1 = image[q1[1] + row, q1[2] + column]
        value2 = image[q2[1] + row, q2[2] + column]
        value3 = image[q3[1] + row, q3[2] + column]
        value4 = image[q4[1] + row, q4[2] + column]
        signal[selected, 1] = plan.optical_gain[selected, 1] * (
            (value1 - value2 + value4 - value3) / divisor -
            plan.reference_signal[selected, 1]
        )
        signal[selected, 2] = plan.optical_gain[selected, 2] * (
            (value1 - value4 + value2 - value3) / divisor -
            plan.reference_signal[selected, 2]
        )
    end
    return signal, Float32[divisor]
end

function run_bi_o_edge_image_capture_tests(
    device,
    to_device,
    device_fill,
    device_zeros,
    synchronize,
    array_device,
    used_memory,
)
    function enqueue_bi_o_edge_image_frame!(captured, frame_status)
        fill!(frame_status, Int32(0))
        enqueue!(captured)
        synchronize()
        return only(Array(frame_status))
    end

    @testset "captured Bi-O-edge normalization threshold matches CPU" begin
        prepared = bi_o_edge_image_prepared()
        host_image = fill(eps(Float32) / 4.0f0, 4, 4)
        host_signature = UInt64[prepared.plan.calibration_signature]
        cpu_signal = fill(71.0f0, 3, 2)
        cpu_divisor = Float32[19.0f0]
        @test process!(cpu_signal, cpu_divisor, prepared, host_image, host_signature) ===
              BiOEdgeImageInvalidNormalization
        @test cpu_signal == fill(71.0f0, 3, 2)
        @test cpu_divisor == Float32[19.0f0]

        image = to_device(host_image)
        frame_calibration_signature = to_device(host_signature)
        signal = device_fill(71.0f0, 3, 2)
        divisor = device_fill(19.0f0, 1)
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device,
            prepared,
            (signal, divisor),
            (image, frame_calibration_signature),
            frame_status,
            Int32(1),
        )
        status = enqueue_bi_o_edge_image_frame!(captured, frame_status)
        @test complete!(captured, status) === BiOEdgeImageInvalidNormalization
        @test Array(signal) == fill(71.0f0, 3, 2)
        @test Array(divisor) == Float32[19.0f0]
    end

    @testset "captured complete-frame Bi-O-edge image is current-frame and output-atomic" begin
        prepared = bi_o_edge_image_prepared()
        host_image = Float32[
            4 3 8 7
            4 3 8 7
            1 2 2 3
            1 2 2 3
        ]
        host_signature = UInt64[prepared.plan.calibration_signature]
        expected_signal, expected_divisor = bi_o_edge_image_direct(host_image, prepared.plan)

        image = to_device(host_image)
        frame_calibration_signature = to_device(host_signature)
        signal = device_fill(71.0f0, 3, 2)
        divisor = device_fill(19.0f0, 1)
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device,
            prepared,
            (signal, divisor),
            (image, frame_calibration_signature),
            frame_status,
            Int32(1),
        )

        @test iszero(enqueue_bi_o_edge_image_frame!(captured, frame_status))
        @test Array(signal) ≈ expected_signal rtol=2eps(Float32)
        @test Array(divisor) ≈ expected_divisor rtol=2eps(Float32)
        @test array_device(captured.staged_signal) == device
        @test array_device(captured.staged_divisor) == device
        @test array_device(captured.frame_calibration_signature) == device
        @test array_device(captured.validation_rows) == device
        @test array_device(captured.validation_columns) == device
        @test array_device(captured.selected_rows) == device
        @test array_device(captured.selected_columns) == device
        @test array_device(captured.origin_rows) == device
        @test array_device(captured.origin_columns) == device
        @test array_device(captured.reference_signal) == device
        @test array_device(captured.optical_gain) == device
        device_bytes = used_memory()
        for _ in 1:8
            @test iszero(enqueue_bi_o_edge_image_frame!(captured, frame_status))
        end
        @test used_memory() <= device_bytes

        sentinel_signal = fill(71.0f0, 3, 2)
        copyto!(signal, sentinel_signal)
        fill!(divisor, 19.0f0)
        invalid_image = copy(host_image)
        invalid_image[4, 4] = NaN32
        copyto!(image, invalid_image)
        copyto!(frame_calibration_signature, UInt64[0x52a7_1b9d])
        status = enqueue_bi_o_edge_image_frame!(captured, frame_status)
        @test complete!(captured, status) === BiOEdgeImageCalibrationMismatch
        @test Array(signal) == sentinel_signal
        @test Array(divisor) == Float32[19.0f0]

        copyto!(frame_calibration_signature, host_signature)
        status = enqueue_bi_o_edge_image_frame!(captured, frame_status)
        @test complete!(captured, status) === BiOEdgeImageNonFiniteInput
        @test Array(signal) == sentinel_signal
        @test Array(divisor) == Float32[19.0f0]
    end

    @testset "captured complete-frame Bi-O-edge output rejection is atomic" begin
        prepared = bi_o_edge_image_prepared(
            optical_gain=fill(floatmax(Float32), 6),
            reference_signal=-ones(Float32, 6),
        )
        host_image = Float32[
            4 3 8 7
            4 3 8 7
            1 2 2 3
            1 2 2 3
        ]
        host_signature = UInt64[prepared.plan.calibration_signature]
        image = to_device(host_image)
        frame_calibration_signature = to_device(host_signature)
        signal = device_fill(71.0f0, 3, 2)
        divisor = device_fill(19.0f0, 1)
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device,
            prepared,
            (signal, divisor),
            (image, frame_calibration_signature),
            frame_status,
            Int32(1),
        )

        status = enqueue_bi_o_edge_image_frame!(captured, frame_status)
        @test complete!(captured, status) === BiOEdgeImageNonFiniteOutput
        @test Array(signal) == fill(71.0f0, 3, 2)
        @test Array(divisor) == Float32[19.0f0]
    end
end
