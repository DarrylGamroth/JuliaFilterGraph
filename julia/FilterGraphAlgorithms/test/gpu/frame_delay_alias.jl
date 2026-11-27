import FilterGraphAlgorithms: complete!, enqueue!, prepare_capture

function run_frame_delay_alias_tests(device, to_device, device_zeros, synchronize)
    @testset "captured zero-delay buffers can coincide but cannot overlap in part" begin
        count = 513
        prepared = prepare_algorithm(FrameDelayF32, FrameDelayF32Configuration(
            extent=count, delay_frames=0, initial_value=0.0f0,
            input_schema=CONTROLLER_COMMAND_V1, output_schema=CONTROLLER_COMMAND_V1))
        values = Float32.(1:2count)
        storage = to_device(values)
        input = view(storage, 1:count)
        shifted = view(storage, 2:count+1)
        adjacent = view(storage, count+1:2count)
        status = device_zeros(Int32, 1)
        @test_throws ArgumentError prepare_capture(CompleteTransition(), device,
            prepared, (shifted,), (input,), status, Int32(1))
        for output in (input, adjacent)
            captured = prepare_capture(CompleteTransition(), device,
                prepared, (output,), (input,), status, Int32(1))
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test Array(output) == values[1:count]
        end
    end
end
