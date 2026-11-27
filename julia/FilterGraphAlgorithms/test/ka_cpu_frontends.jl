include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_image.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "bi_o_edge_image.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "zernike_pupil_signal.jl"))

@testset "portable KA image frontends against CPU operations" begin
    rows, columns = 17, 19
    common = (image_size=(2rows, 2columns),
        pupil_size=(rows, columns),
        pupil_origins=[[0, 0], [rows, 0], [rows, columns], [0, columns]],
        initial_reference_signal=fill(0.25f0, 2rows * columns),
        initial_optical_gain=fill(0.5f0, 2rows * columns),
        normalization_policy="mean-valid-flux", incidence_flux=0.0f0)
    pyramid = prepare_algorithm(PyramidImageF32,
        PyramidImageF32Configuration(; common..., i4q_support=fill(true, rows * columns)))
    bio = prepare_algorithm(BiOEdgeImageF32,
        BiOEdgeImageF32Configuration(; common..., bi_o_edge_support=fill(true, rows * columns),
            initial_calibration_signature=UInt64(31)))
    zernike = prepare_algorithm(ZernikePupilSignalF32,
        ZernikePupilSignalF32Configuration(image_size=(rows, columns),
            zernike_pupil_support=fill(true, rows * columns),
            initial_reference_pupil_image=fill(0.25f0, rows * columns),
            normalization_policy="mean-valid-flux", incidence_flux=0.0f0,
            initial_calibration_signature=UInt64(37)))
    for (declaration, prepared, shape, image_shape, signature) in (
        (PyramidImageF32, pyramid, (rows * columns, 2), (2rows, 2columns), nothing),
        (BiOEdgeImageF32, bio, (rows * columns, 2), (2rows, 2columns), UInt64[31]),
        (ZernikePupilSignalF32, zernike, (rows * columns,), (rows, columns), UInt64[37]),
    )
        @testset "$declaration" begin
            @test capture_support(declaration, BACKEND) isa NoCapture
            image = reshape(Float32[1 + mod(index, 17) / 4 for index in 1:prod(image_shape)], image_shape)
            inputs = signature === nothing ? (image,) : (image, signature)
            actual, divisor = fill(71.0f0, shape), Float32[73]
            expected, expected_divisor = copy(actual), copy(divisor)
            status = zeros(Int32, 1)
            adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
                (actual, divisor), inputs, status, Int32(1))
            for invalid in (false, true)
                invalid && (image[end] = Float32(NaN))
                result = process!(expected, expected_divisor, prepared, inputs...)
                fill!(status, 0)
                enqueue!(adapter)
                KernelAbstractions.synchronize(BACKEND)
                @test complete!(adapter, only(status)) === result
                @test actual ≈ expected rtol=32eps(Float32) atol=32eps(Float32)
                @test divisor ≈ expected_divisor rtol=32eps(Float32)
            end
        end
    end
end
