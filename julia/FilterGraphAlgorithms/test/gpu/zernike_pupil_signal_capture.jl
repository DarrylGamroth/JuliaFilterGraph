using TOML

import FilterGraphAlgorithms:
    ZernikePupilSignalCalibrationMismatch,
    ZernikePupilSignalF32,
    ZernikePupilSignalF32Configuration,
    ZernikePupilSignalIncidenceFlux,
    ZernikePupilSignalInvalidNormalization,
    ZernikePupilSignalMeanValidFlux,
    ZernikePupilSignalNonFiniteInput,
    ZernikePupilSignalNonFiniteOutput,
    complete!,
    enqueue!,
    prepare_capture

const ZERNIKE_PUPIL_SIGNAL_FIXTURE = TOML.parsefile(joinpath(@__DIR__, "..", "data", "algorithms", "zernike_pupil_signal_synthetic.toml"))

function zernike_fixture_image()
    fixture = ZERNIKE_PUPIL_SIGNAL_FIXTURE
    return reshape(
        Float32.(fixture["frame"]["image"]),
        Tuple(fixture["geometry"]["image_size"])...,
    )
end

function zernike_fixture_support()
    fixture = ZERNIKE_PUPIL_SIGNAL_FIXTURE
    return Bool.(fixture["geometry"]["support"])
end

function zernike_fixture_reference()
    fixture = ZERNIKE_PUPIL_SIGNAL_FIXTURE
    return Float32.(fixture["calibration"]["reference_pupil_image"])
end

function zernike_pupil_signal_prepared(
    ;
    normalization_policy="mean-valid-flux",
    incidence_flux=0.0f0,
    reference=zernike_fixture_reference(),
)
    fixture = ZERNIKE_PUPIL_SIGNAL_FIXTURE
    geometry = fixture["geometry"]
    return prepare_algorithm(
        ZernikePupilSignalF32,
        ZernikePupilSignalF32Configuration(
            image_size=(geometry["image_size"][1], geometry["image_size"][2]),
            zernike_pupil_support=zernike_fixture_support(),
            initial_reference_pupil_image=reference,
            normalization_policy=normalization_policy,
            incidence_flux=incidence_flux,
            initial_calibration_signature=UInt64(fixture["calibration"]["signature"]),
        ),
    )
end

function one_pixel_zernike_pupil_signal_prepared()
    return prepare_algorithm(
        ZernikePupilSignalF32,
        ZernikePupilSignalF32Configuration(
            image_size=(1, 1),
            zernike_pupil_support=Bool[true],
            initial_reference_pupil_image=zeros(Float32, 1),
            normalization_policy="mean-valid-flux",
            incidence_flux=0.0f0,
            initial_calibration_signature=UInt64(1),
        ),
    )
end

function zernike_pupil_signal_direct(image, plan)
    divisor = if plan.normalization === ZernikePupilSignalMeanValidFlux
        total = 0.0f0
        for offset in plan.selected_offsets
            total += image[offset + 1]
        end
        max(total / length(plan.selected_offsets), eps(Float32))
    else
        plan.incidence_flux
    end
    signal = zeros(Float32, length(plan.selected_offsets))
    if divisor > eps(Float32)
        for (selected, offset) in enumerate(plan.selected_offsets)
            signal[selected] = image[offset + 1] / divisor - plan.reference_pupil_image[offset + 1]
        end
    end
    return signal, Float32[divisor]
end

function zernike_with_incidence_flux(captured, incidence_flux::Float32)
    return typeof(captured)(
        captured.stage!, captured.adopt!, captured.signal,
        captured.normalization_divisor, captured.staged_signal, captured.staged_divisor,
        captured.frame_status, captured.image, captured.frame_calibration_signature,
        captured.selected_rows, captured.selected_columns, captured.reference_pupil_image,
        captured.calibration_signature, captured.mean_valid_flux, incidence_flux,
        captured.mismatch_failure, captured.input_failure, captured.normalization_failure,
        captured.output_failure,
    )
end

function run_zernike_pupil_signal_capture_tests(
    device,
    to_device,
    device_fill,
    device_zeros,
    synchronize,
    array_device,
    used_memory,
)
    function enqueue_zernike_pupil_signal_frame!(captured, frame_status)
        fill!(frame_status, Int32(0))
        enqueue!(captured)
        synchronize()
        return only(Array(frame_status))
    end

    @testset "captured complete-frame Zernike pupil signal matches the synthetic fixture" begin
        host_image = zernike_fixture_image()
        prepared = zernike_pupil_signal_prepared()
        expected_signal, expected_divisor = zernike_pupil_signal_direct(host_image, prepared.plan)
        signature = UInt64[prepared.plan.calibration_signature]
        image = to_device(host_image)
        frame_calibration_signature = to_device(signature)
        signal = device_fill(-71.0f0, length(expected_signal))
        divisor = device_fill(-19.0f0, 1)
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

        @test iszero(enqueue_zernike_pupil_signal_frame!(captured, frame_status))
        @test Array(signal) ≈ expected_signal rtol=2eps(Float32)
        @test Array(divisor) ≈ expected_divisor rtol=2eps(Float32)
        @test array_device(captured.staged_signal) == device
        @test array_device(captured.staged_divisor) == device
        @test array_device(captured.frame_calibration_signature) == device
        @test array_device(captured.selected_rows) == device
        @test array_device(captured.selected_columns) == device
        @test array_device(captured.reference_pupil_image) == device

        device_bytes = used_memory()
        for _ in 1:8
            @test iszero(enqueue_zernike_pupil_signal_frame!(captured, frame_status))
        end
        @test used_memory() <= device_bytes
    end

    @testset "captured Zernike pupil signal accepts incidence and low-flux frames" begin
        host_image = zernike_fixture_image()
        fixture = ZERNIKE_PUPIL_SIGNAL_FIXTURE
        incidence_flux = Float32(fixture["incidence_flux"]["normalization_divisor"])
        prepared = zernike_pupil_signal_prepared(
            normalization_policy="incidence-flux",
            incidence_flux=incidence_flux,
        )
        expected_signal, expected_divisor = zernike_pupil_signal_direct(host_image, prepared.plan)
        signature = UInt64[prepared.plan.calibration_signature]
        image = to_device(host_image)
        frame_calibration_signature = to_device(signature)
        signal = device_fill(-71.0f0, length(expected_signal))
        divisor = device_fill(-19.0f0, 1)
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (signal, divisor),
            (image, frame_calibration_signature), frame_status, Int32(1),
        )
        @test iszero(enqueue_zernike_pupil_signal_frame!(captured, frame_status))
        @test Array(signal) ≈ expected_signal rtol=2eps(Float32)
        @test Array(divisor) == expected_divisor

        zero_prepared = zernike_pupil_signal_prepared(
            normalization_policy="incidence-flux",
            incidence_flux=0.0f0,
        )
        zero_signal = device_fill(71.0f0, length(expected_signal))
        zero_divisor = device_fill(19.0f0, 1)
        zero_status = device_zeros(Int32, 1)
        zero_captured = prepare_capture(
            CompleteTransition(),
            device, zero_prepared, (zero_signal, zero_divisor),
            (image, frame_calibration_signature), zero_status, Int32(1),
        )
        @test iszero(enqueue_zernike_pupil_signal_frame!(zero_captured, zero_status))
        @test Array(zero_signal) == zeros(Float32, length(expected_signal))
        @test Array(zero_divisor) == Float32[0]

        mean_prepared = zernike_pupil_signal_prepared()
        mean_signal = device_fill(71.0f0, length(expected_signal))
        mean_divisor = device_fill(19.0f0, 1)
        mean_status = device_zeros(Int32, 1)
        mean_image = device_zeros(Float32, size(host_image)...)
        mean_captured = prepare_capture(
            CompleteTransition(),
            device, mean_prepared, (mean_signal, mean_divisor),
            (mean_image, frame_calibration_signature), mean_status, Int32(1),
        )
        @test iszero(enqueue_zernike_pupil_signal_frame!(mean_captured, mean_status))
        @test Array(mean_signal) == zeros(Float32, length(expected_signal))
        @test Array(mean_divisor) == Float32[eps(Float32)]
    end

    @testset "captured Zernike pupil signal failures preserve caller outputs" begin
        host_image = zernike_fixture_image()
        prepared = zernike_pupil_signal_prepared()
        signature = UInt64[prepared.plan.calibration_signature]
        image = to_device(host_image)
        frame_calibration_signature = to_device(signature)
        sentinel_signal = fill(71.0f0, length(prepared.plan.selected_offsets))
        signal = to_device(sentinel_signal)
        divisor = to_device(Float32[19])
        frame_status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (signal, divisor),
            (image, frame_calibration_signature), frame_status, Int32(1),
        )

        copyto!(frame_calibration_signature, UInt64[signature[1] + 1])
        status = enqueue_zernike_pupil_signal_frame!(captured, frame_status)
        @test complete!(captured, status) === ZernikePupilSignalCalibrationMismatch
        @test Array(signal) == sentinel_signal
        @test Array(divisor) == Float32[19]

        copyto!(frame_calibration_signature, signature)
        invalid_image = copy(host_image)
        invalid_image[3, 1] = NaN32 # Outside the retained pupil support.
        copyto!(image, invalid_image)
        status = enqueue_zernike_pupil_signal_frame!(captured, frame_status)
        @test complete!(captured, status) === ZernikePupilSignalNonFiniteInput
        @test Array(signal) == sentinel_signal
        @test Array(divisor) == Float32[19]

        copyto!(image, host_image)
        invalid_normalization_prepared = zernike_pupil_signal_prepared(
            normalization_policy="incidence-flux",
            incidence_flux=1.0f0,
        )
        invalid_normalization_captured = prepare_capture(
            CompleteTransition(),
            device, invalid_normalization_prepared, (signal, divisor),
            (image, frame_calibration_signature), frame_status, Int32(1),
        )
        nonfinite_normalization = zernike_with_incidence_flux(
            invalid_normalization_captured,
            NaN32,
        )
        status = enqueue_zernike_pupil_signal_frame!(nonfinite_normalization, frame_status)
        @test complete!(nonfinite_normalization, status) ===
              ZernikePupilSignalInvalidNormalization
        @test Array(signal) == sentinel_signal
        @test Array(divisor) == Float32[19]

        overflow_reference = zernike_fixture_reference()
        overflow_reference[1] = -floatmax(Float32)
        overflow_prepared = zernike_pupil_signal_prepared(
            normalization_policy="incidence-flux",
            incidence_flux=1.0f0,
            reference=overflow_reference,
        )
        overflow_image = ones(Float32, 3, 3)
        overflow_image[1] = floatmax(Float32)
        overflow_signal = to_device(sentinel_signal)
        overflow_divisor = to_device(Float32[19])
        overflow_status = device_zeros(Int32, 1)
        overflow_captured = prepare_capture(
            CompleteTransition(),
            device, overflow_prepared, (overflow_signal, overflow_divisor),
            (to_device(overflow_image), to_device(signature)), overflow_status, Int32(1),
        )
        status = enqueue_zernike_pupil_signal_frame!(overflow_captured, overflow_status)
        @test complete!(overflow_captured, status) ===
              ZernikePupilSignalNonFiniteOutput
        @test Array(overflow_signal) == sentinel_signal
        @test Array(overflow_divisor) == Float32[19]
    end

    @testset "captured Zernike pupil signal rejects overlapping caller arrays" begin
        prepared = one_pixel_zernike_pupil_signal_prepared()
        signature = to_device(UInt64[prepared.plan.calibration_signature])
        frame_status = device_zeros(Int32, 1)
        signal = device_zeros(Float32, 1)
        divisor = device_zeros(Float32, 1)
        image = device_fill(1.0f0, 1, 1)

        @test Base.mightalias(signal, signal)
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(),
            device, prepared, (signal, signal), (image, signature), frame_status, Int32(1),
        )

        image_storage = device_fill(1.0f0, 1)
        image_alias = reshape(image_storage, 1, 1)
        @test Base.mightalias(image_storage, image_alias)
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(),
            device, prepared, (image_storage, divisor), (image_alias, signature),
            frame_status, Int32(1),
        )
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(),
            device, prepared, (signal, image_storage), (image_alias, signature),
            frame_status, Int32(1),
        )
    end
end
