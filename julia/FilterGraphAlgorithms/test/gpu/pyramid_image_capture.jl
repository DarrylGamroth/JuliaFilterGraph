function run_pyramid_image_capture_tests(
    device,
    to_device,
    device_fill,
    device_zeros,
    synchronize,
)
    @testset "captured complete-frame pyramid image is current-frame and output-atomic" begin
        prepared = prepare_algorithm(
            PyramidImageF32,
            PyramidImageF32Configuration(
                image_size=(4, 4),
                pupil_size=(2, 2),
                pupil_origins=[[0, 0], [2, 0], [2, 2], [0, 2]],
                i4q_support=Bool[true, false, true, true],
                initial_reference_signal=zeros(Float32, 6),
                initial_optical_gain=ones(Float32, 6),
                normalization_policy="mean-valid-flux",
                incidence_flux=0.0f0,
            ),
        )
        host_image = Float32[
            4 3 3 3
            4 3 3 3
            1 2 2 3
            1 2 2 3
        ]
        expected_signal = zeros(Float32, 3, 2)
        expected_divisor = zeros(Float32, 1)
        process!(expected_signal, expected_divisor, prepared, host_image)

        image = to_device(host_image)
        signal = device_fill(71.0f0, 3, 2)
        divisor = device_fill(19.0f0, 1)
        frame_status = device_zeros(Int32, 1)
        captured = FilterGraphAlgorithms.prepare_capture(
            CompleteTransition(),
            device,
            prepared,
            (signal, divisor),
            (image,),
            frame_status,
            Int32(1),
        )
        FilterGraphAlgorithms.enqueue!(captured)
        synchronize()
        @test Array(signal) ≈ expected_signal rtol=2eps(Float32)
        @test Array(divisor) ≈ expected_divisor rtol=2eps(Float32)
        @test iszero(only(Array(frame_status)))

        host_image[4, 4] = NaN32
        copyto!(image, host_image)
        FilterGraphAlgorithms.enqueue!(captured)
        synchronize()
        status = only(Array(frame_status))
        @test FilterGraphAlgorithms.complete!(captured, status) ===
              PyramidImageNonFiniteInput
        @test Array(signal) ≈ expected_signal rtol=2eps(Float32)
        @test Array(divisor) ≈ expected_divisor rtol=2eps(Float32)
    end
    @testset "captured pyramid normalizes an overflowing raw difference" begin
        flux = floatmax(Float32)
        prepared = prepare_algorithm(PyramidImageF32, PyramidImageF32Configuration(
            image_size=(2, 2), pupil_size=(1, 1),
            pupil_origins=[[0, 0], [1, 0], [1, 1], [0, 1]],
            i4q_support=[true], initial_reference_signal=zeros(Float32, 2),
            initial_optical_gain=ones(Float32, 2), normalization_policy="incidence-flux",
            incidence_flux=flux,
        ))
        image = to_device(Float32[0.75f0 * flux 0; -0.75f0 * flux 0])
        signal, divisor = device_zeros(Float32, 1, 2), device_zeros(Float32, 1)
        frame_status = device_zeros(Int32, 1)
        captured = FilterGraphAlgorithms.prepare_capture(
            CompleteTransition(), device, prepared, (signal, divisor),
            (image,), frame_status, Int32(1),
        )
        FilterGraphAlgorithms.enqueue!(captured)
        synchronize()
        @test iszero(only(Array(frame_status)))
        @test Array(signal) ≈ Float32[1.5 0]
        @test Array(divisor) == [flux]
    end

end
