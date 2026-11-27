# Backend-neutral adapter checks: preparation, both launch geometries, exact
# aliasing, ranked failure, reset and fixed retained storage.
import FilterGraphAlgorithms: process!, reset!

function run_pdm_offset_capture_tests(device, to_device, device_zeros, synchronize)
    @testset "physical offset capture preparation and publication" begin
        for n in (1, 256, 257, 512, 513), engineering in (false, true), in_place in (false, true)
            declaration, configuration, parameter = engineering ?
                (PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration(actuator_count=n), "engineering-offset") :
                (PdmSystemFlatF32, PdmSystemFlatF32Configuration(actuator_count=n), "system-flat")
            prepared = prepare_algorithm(declaration, configuration)
            offset = fill(0.125f0, n)
            offset[1] = -0.0f0
            offset[end] = floatmax(Float32)
            n > 2 && (offset[2] = floatmax(Float32))
            replace_parameter!(prepared, parameter, offset)
            input = to_device(zeros(Float32, n))
            output = in_place ? input : device_zeros(Float32, n)
            outputs = engineering ? (output, device_zeros(Float32, n)) : (output,)
            status = device_zeros(Int32, 1)
            captured = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), device,
                prepared, outputs, (input,), status, Int32(1))
            @test capture_support(declaration, device) isa SupportsCapture
            reuse = FilterGraphAlgorithms._supports_validation_reuse(
                KernelAbstractions.get_backend(status))
            @test FilterGraphAlgorithms.preserves_inputs(captured) == reuse
            @test FilterGraphAlgorithms.finite_output(captured, Val(1)) == reuse
            @test FilterGraphAlgorithms.preflight(captured) === nothing
            addresses = map(pointer, (outputs..., input, captured.offset, status))
            for failure in (:none, :input_first, :output_first)
                sample = fill(0.25f0, n)
                sample[1] = -0.0f0
                if failure === :input_first
                    sample[1] = NaN32
                    sample[end] = floatmax(Float32)
                elseif failure === :output_first
                    sample[end] = floatmax(Float32)
                    if n > 2
                        sample[2] = floatmax(Float32)
                        sample[end] = NaN32
                    end
                end
                expected = map(_ -> zeros(Float32, n), outputs)
                result = process!(expected..., prepared, sample)
                foreach(array -> fill!(array, 71.0f0), outputs)
                copyto!(input, sample)
                supplied = map(Array, outputs)
                fill!(status, Int32(0))
                FilterGraphAlgorithms.enqueue!(captured)
                synchronize()
                @test FilterGraphAlgorithms.complete!(captured, only(Array(status))) === result
                actual = map(Array, outputs)
                reference = isnothing(result) ? expected : supplied
                @test map(array -> reinterpret(UInt32, array), actual) ==
                      map(array -> reinterpret(UInt32, array), reference)
            end
            fill!(status, Int32(42))
            supplied = map(Array, outputs)
            FilterGraphAlgorithms.enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 42
            @test isequal(map(Array, outputs), supplied)
            @test reset!(captured) === captured
            @test map(pointer, (outputs..., input, captured.offset, status)) == addresses
        end
    end
end
