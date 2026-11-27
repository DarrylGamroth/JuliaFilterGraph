module FrameDelayCaptureTests

using CUDA, FilterGraphAlgorithms, Test
import FilterGraphAlgorithms: complete!, enqueue!, prepare_capture, process!, reset!

function prepare_delay(delay_frames)
    return prepare_algorithm(FrameDelayF32, FrameDelayF32Configuration(
        extent=2, delay_frames=delay_frames, initial_value=-1.0f0,
        input_schema=CONTROLLER_COMMAND_V1, output_schema=CONTROLLER_COMMAND_V1))
end

@testset "CUDA complete frame-delay transition advances exactly once" begin
    for delay_frames in (2, 3), recorded in (false, true)
        prepared = prepare_delay(delay_frames)
        input, output = CUDA.zeros(Float32, 2), CUDA.fill(71.0f0, 2)
        status = CUDA.zeros(Int32, 1)
        # Both retired direct arities must reject before returning a changed adapter.
        @test_throws FilterGraphAlgorithms.CaptureProtocolError prepare_capture(
            CUDA.device(), prepared, (output,), (input,), status, Int32(1))
        @test_throws FilterGraphAlgorithms.CaptureProtocolError prepare_capture(
            CUDA.device(), prepared, (output,), (input,), status, Int32(1), nothing)
        @test Array(output) == fill(71.0f0, 2)
        @test frame_delay_head(prepared.workspace) == 1
        adapter = prepare_capture(
            CompleteTransition(), CUDA.device(), prepared, (output,), (input,), status, Int32(1))
        storage = (adapter.history, adapter.initial_history, adapter.head)
        addresses = map(pointer, storage)
        enqueue!(adapter)
        CUDA.synchronize()
        reset!(adapter)
        CUDA.synchronize()
        executable = recorded ? CUDA.instantiate(CUDA.capture() do
            enqueue!(adapter)
        end) : nothing
        step!() = recorded ? CUDA.launch(executable) : enqueue!(adapter)
        expected = zeros(Float32, 2)
        for frame in 1:(2 * delay_frames + 2)
            sample = Float32[frame, -frame]
            copyto!(input, sample)
            fill!(status, Int32(0))
            @test process!(expected, prepared, sample) === nothing
            step!()
            CUDA.synchronize()
            @test complete!(adapter, only(Array(status))) === nothing
            @test Array(output) == expected
            @test only(Array(adapter.head)) == frame_delay_head(prepared.workspace)
            @test Array(adapter.history) == frame_delay_history(prepared.workspace)
        end
        saved_output, saved_history, saved_head = Array(output), Array(adapter.history), Array(adapter.head)
        for (sample, prior_status, expected_failure) in (
            (Float32[NaN, 0], Int32(0), FrameDelayNonFiniteInput),
            (Float32[9, -9], Int32(42), nothing),
        )
            copyto!(input, sample)
            fill!(status, prior_status)
            step!()
            CUDA.synchronize()
            @test complete!(adapter, only(Array(status))) === expected_failure
            @test Array(output) == saved_output
            @test Array(adapter.history) == saved_history
            @test Array(adapter.head) == saved_head
        end
        sample = Float32[11, -11]
        copyto!(input, sample)
        fill!(status, Int32(0))
        @test process!(expected, prepared, sample) === nothing
        step!()
        CUDA.synchronize()
        @test Array(output) == expected
        @test Array(adapter.history) == frame_delay_history(prepared.workspace)
        @test reset!(adapter) === adapter
        CUDA.synchronize()
        @test (adapter.history, adapter.initial_history, adapter.head) === storage
        @test map(pointer, storage) == addresses
        @test only(Array(adapter.head)) == 1
        @test Array(adapter.history) == fill(-1.0f0, 2, delay_frames)
    end
end

end # module
