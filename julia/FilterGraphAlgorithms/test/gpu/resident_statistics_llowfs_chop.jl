using FilterGraphAlgorithms, Test

function test_resident_statistics(device, upload, synchronize, record)
    F = FilterGraphAlgorithms
    @testset "Resident statistics on $device" begin
        for mode in ("finite", "exponential", "sliding"), shape in ((2,3), (1,1))
            cfg = ArrayStatisticsF32Configuration(size=collect(shape), mode=mode, window_samples=3,
                max_state_bytes=prod(shape) * (mode == "sliding" ? 80 : 52),
                input_schema="sample/1", mean_schema="mean/1", variance_schema="variance/1", stddev_schema="stddev/1")
            cpu = prepare_algorithm(ArrayStatisticsF32,cfg)
            outputs = (zeros(Float32,shape), zeros(Float32,shape), zeros(Float32,shape),
                UInt64[0], Bool[false], Bool[false])
            out, sample, status = map(upload, outputs), upload(zeros(Float32,shape)), upload(Int32[0])
            adapter = F.prepare_capture(CompleteTransition(),device,cpu,out,(sample,),status,Int32(2))
            replay = resident_replay(adapter, record, status, synchronize)
            for t in (1,2,0,4,5,8,12,20)
                # Introduce a rejection in the partially filled window and after wrap/freeze.
                t == 12 && (F.reset!(adapter); F.reset!(cpu))
                values = reshape(Float32.(1:prod(shape)) .* t .- t^2,shape)
                t in (0,8) && (values[end] = NaN32)
                copyto!(sample, values)
                fill!(status, Int32(0))
                expected = F.process!(outputs...,cpu,values)
                replay()
                synchronize()
                @test F.complete!(adapter,only(Array(status))) === expected
                @test all(map((a,b)->isapprox(a,Array(b);rtol=2eps(Float32)),outputs,out))
            end
        end
    end
end

function test_resident_llowfs(device, upload, synchronize, record)
    F = FilterGraphAlgorithms
    SC = F.SimpleChains
    @testset "Resident LLOWFS with a supplied model on $device" begin
        model = MlLlowfsModel(SC.SimpleChain((4, 6, 1), SC.Conv(tanh, (2, 1), 2),
            SC.MaxPool((1, 2)), SC.Flatten(3), SC.TurboDense(tanh, 3), SC.TurboDense(identity, 2)))
        cpu = prepare_algorithm(MlLlowfsF32, MlLlowfsF32Configuration(;
            model, modes_schema="test/llowfs-modes/1", parameters_schema="test/llowfs-parameters/1"))
        status, output = upload(Int32[0]), upload(fill(75f0, 2))
        cpu_modes = fill(75f0, 2)
        values = Float32[mod(19*(r-1) + 23*(c-1),101)+1 for _ in 1:1, r in 1:4, c in 1:6]
        pixels = upload(values)
        cancelling = zeros(Float32, size(values))
        cancelling[[1, 4, 5]] .= Float32[1f8, -1f8, 1]
        for test_model in (:missing, :synthetic, :bias, :overflow)
            count = model.parameter_count
            if test_model === :synthetic
                params = Float32[(mod((i - 1) * 37, 257) - 128) / 4096f0 for i in 1:count]
            elseif test_model === :bias
                params = zeros(Float32, count)
                params[end-1] = 1f0
            elseif test_model === :overflow
                params = zeros(Float32, count)
                layers = SC.params(model.chain, params)
                layers[end-1][2] .= 1f0
                layers[end][1][1, :] .= floatmax(Float32)
                layers[end][2][1] = floatmax(Float32)
            end
            if test_model !== :missing
                F.replace_parameter!(cpu, Symbol("model-parameters"), params)
            end
            adapter = F.prepare_capture(CompleteTransition(), device, cpu, (output,), (pixels,), status, Int32(1))
            replay = resident_replay(adapter, record, status, synchronize)
            for data in (values, cancelling, fill(0f0, size(values)), fill(1f0, size(values)),
                         fill(NaN32, size(values)), values)
                fill!(status, Int32(0))
                copyto!(pixels, data)
                expected = F.process!(cpu_modes, cpu, data)
                valid_frame = data === values || data === cancelling
                if test_model === :bias && valid_frame
                    @test expected === nothing
                    @test cpu_modes[1] === 1f0
                elseif test_model === :overflow && valid_frame
                    @test expected === MlLlowfsNonFiniteOutput
                end
                replay()
                synchronize()
                @test F.complete!(adapter, only(Array(status))) === expected
                @test Array(output) ≈ cpu_modes atol=5f-7 rtol=0
            end
        end
    end
end

function test_resident_detector_chop(device, upload, synchronize)
    F = FilterGraphAlgorithms
    @testset "Resident chopped detector on $device" begin
        cfg = ChoppedDetectorDifferenceF32Configuration(image_size=(2,3),input_schema="raw-f32/1")
        cpu = prepare_algorithm(ChoppedDetectorDifferenceF32,cfg)
        gpu = prepare_algorithm(ChoppedDetectorDifferenceF32,cfg,device)
        outputs = (zeros(Float32,2,3),UInt8[0])
        out = map(upload,outputs)
        raw = upload(zeros(Float32,2,3))
        domain = ntuple(UInt8,16)
        for (sequence, kind) in ((1,ChoppedImageFringed),(2,ChoppedImageUnfringed),
                (2,ChoppedImageFringed),(4,ChoppedImageFringed),(5,ChoppedImageMixed),
                (6,ChoppedImageFringed),(7,ChoppedImageUnfringed),(8,ChoppedImageFringed))
            sequence == 8 && (F.reset!(gpu); F.reset!(cpu))
            data = Float32[3 6 9; 4 0 7] .* sequence
            data[sequence % 6 + 1] = NaN32
            copyto!(raw,data)
            sample = SampleMetadata(sequence; terminal=true,chopped_image_kind=kind,
                exposure_duration_nanoseconds=10,acquisition_identity=SampleAcquisitionIdentity(domain,1,sequence))
            if sequence == 1
                @test_throws ArgumentError F.process!(out...,gpu,gpu.workspace.candidate,(image=sample,))
            end
            expected = F.process!(outputs...,cpu,data,(image=sample,))
            actual = F.process!(out...,gpu,raw,(image=sample,))
            synchronize()
            @test isequal(actual,expected)
            @test all(map((a,b) -> isequal(Array(a),b),out,outputs))
        end
    end
end
