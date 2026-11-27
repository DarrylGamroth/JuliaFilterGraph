"""Public eager-plan entry points are distinct from Algorithm capture support."""
function run_eager_capability_tests(device; additional=())
    @testset "explicit CUDA/AMDGPU capture capability surface" begin
        # Each of these declarations has an eager prepared path and device
        # replay tests that exercise its packet metadata or publication state.
        eager = (PixelCalibrationRowU16F32, PixelCalibrationReadoutU16F32,
            DetectorFrameAssemblerF32, ChoppedDetectorDifferenceF32, SccCorrectionF32,
            ShackHartmannMeasurementBlockF32, ShackHartmannReadoutMeasurementBlockF32,
            ShackHartmannWeightedMeasurementBlockF32, ShackHartmannRowReconstructorF32,
            ShackHartmannReadoutReconstructorF32, PyramidPixelRowF32, PyramidMeasurementBlockF32,
            PyramidRowReconstructorF32, PyramidReadoutReconstructorF32,
            PyramidReadoutMeasurementBlockF32, IncrementalDenseReconstructorF32)
        for declaration in FilterGraphAlgorithms.algorithms()
            expected = declaration in eager ? NoCapture : SupportsCapture
            @test capture_support(declaration, device) isa expected
        end
        @test all(D -> capture_support(D, device) isa SupportsCapture, additional)
    end
    @testset "explicit eager plan capabilities" begin
        for T in (Float32, Float64)
            for Plan in (DocrimeExcitationPlan, DocrimeBinaryExcitationPlan)
                @test applicable(prepare_docrime_excitation_plan, device, Plan(T[1], UInt64(1)))
            end
            @test applicable(prepare_pyramid_pixel_plan, device, PyramidPixelPlan{T}([true]))
        end
    end
end
