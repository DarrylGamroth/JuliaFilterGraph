using Test, FilterGraphAlgorithms

function resident_replay(adapter, record, status, synchronize)
    # Compile the launches without publishing a sample or advancing state.
    fill!(status, Int32(1))
    FilterGraphAlgorithms.enqueue!(adapter)
    synchronize()
    replay = record(() -> FilterGraphAlgorithms.enqueue!(adapter))
    fill!(status, Int32(0))
    return replay
end

function test_resident_detector_measurements(device, upload, synchronize, record)
    F = FilterGraphAlgorithms
    detector = reshape(Float32.(1:20), 4, 5)
    origins = [[0, 1], [2, 3]]
    weighted = WeightedCenterOfGravityF32Configuration(pixel_count=3,
        initial_coordinates=[Float32[0,0], Float32[1,0], Float32[2,1]], initial_weights=Float32[0,1,2])
    signed = PixelCalibrationF32Configuration(image_size=(4,5), input_schema="raw-f32/1",
        initial_flat=fill(2f0,20), initial_background=ones(Float32,20))
    cases = (
        (RegionExtractionF32, RegionExtractionF32Configuration(image_size=(4,5), region_size=(2,2),
            origins=origins), (detector,), (zeros(Float32,2,2,2),)),
        (PyramidPupilFluxU16F64, PyramidPupilFluxU16F64Configuration(image_size=(4,5), pupil_size=(2,2),
            pupil_origins=[[0,0],[2,0],[0,3],[2,3]]), (UInt16.(detector),), (zeros(Float64,4),)),
        (WeightedCenterOfGravityF32, weighted, (Float32[1,2,4],),
            (zeros(Float32,2), Float32[0], Bool[false])),
        (ShackHartmannSummaryF32, ShackHartmannSummaryF32Configuration(subaperture_count=3),
            (Float32[3 4; 0 0; -3 4], Float32[2,NaN,4], Bool[true,false,true]), (zeros(Float32,8),)),
        (PixelCalibrationF32, signed, (detector,), (zeros(Float32,4,5),)),
    )
    @testset "Resident detector measurements on $device" begin
        for (decl, config, inputs, outputs) in cases
            p = prepare_algorithm(decl, config)
            raw = map(upload, inputs)
            result = map(upload, outputs)
            status = upload(Int32[0])
            captured = F.prepare_capture(CompleteTransition(), device, p, result, raw, status, Int32(1))
            replay = resident_replay(captured, record, status, synchronize)
            @test F.process!(outputs..., p, inputs...) === nothing
            replay()
            synchronize()
            @test only(Array(status)) == 0
            @test all(map((a,b) -> isequal(a, Array(b)), outputs, result))
            for output in result
                fill!(output, one(eltype(output)))
            end
            fill!(status, Int32(747))
            replay()
            synchronize()
            @test only(Array(status)) == 747
            @test all(output -> all(isone, Array(output)), result)
            if decl === PixelCalibrationF32
                bad = copy(only(inputs))
                bad[1:2] = [floatmax(Float32), NaN32]
                for failure in (F.PixelCalibrationNonFiniteOutput, F.PixelCalibrationNonFiniteInput)
                    fill!(status, Int32(0))
                    copyto!(only(raw),bad)
                    replay()
                    synchronize()
                    @test F.complete!(captured, only(Array(status))) === failure
                    @test all(isone, Array(only(result)))
                    bad[1] = 1f0
                end
            end
        end
    end
end
