module LeakyIntegratorCaptureTests

using CUDA, FilterGraphAlgorithms, KernelAbstractions, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, reset!

const CUDAX = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
bits(values) = reinterpret(UInt32, Array(values))
replay_bytes(executable) = @allocated CUDA.launch(executable)

function buffers(count, in_place)
    input = CUDA.zeros(Float32, count)
    candidate = in_place ? input : CUDA.fill(-17f0, count)
    return (; input, candidate, committed=CUDA.fill(-2f0, count),
        status=CUDA.zeros(Int32, 1))
end

function compare_transition!(baseline, candidate, values, prior, closed, gain, pole)
    for case in (baseline, candidate)
        copyto!(case.input, values)
        fill!(case.status, prior)
    end
    backend = KernelAbstractions.get_backend(baseline.input)
    arguments(case) = (case.candidate, case.status, case.committed, case.input,
        gain, pole, closed, Int32(513), Int32(514))
    CUDAX._leaky_step!(backend, 256)(arguments(baseline)...; ndrange=256)
    CUDAX._adopt_leaky_state!(backend, 256)(baseline.committed, baseline.candidate,
        baseline.status; ndrange=length(values))
    CUDAX._leaky_transition!(backend, 256)(arguments(candidate)...; ndrange=256)
    CUDA.synchronize()
    @test Array(candidate.status) == Array(baseline.status)
    @test bits(candidate.input) == bits(baseline.input)
    @test bits(candidate.candidate) == bits(baseline.candidate)
    @test bits(candidate.committed) == bits(baseline.committed)
end

@testset "CUDA leaky fused transition preserves staged observations" begin
    for count in (1, 31, 32, 33, 255, 256, 257, 511, 513, 1810),
            in_place in (false, true), closed in (false, true)
        baseline, candidate = buffers(count, in_place), buffers(count, in_place)
        finite = Float32[(-1)^i * (i % 7) / 8 for i in 1:count]
        finite[1] = -0.0f0
        count > 1 && (finite[2] = nextfloat(0f0))
        # Success, an earlier Graph rejection, input rejection, overflow and
        # recovery share evolving state. Exact aliasing is included above.
        compare_transition!(baseline, candidate, finite, Int32(0), closed, 0.5f0, 0.75f0)
        compare_transition!(baseline, candidate, finite, Int32(257), closed, 0.5f0, 0.75f0)
        bad = copy(finite); bad[end] = NaN32
        compare_transition!(baseline, candidate, bad, Int32(0), closed, 0.5f0, 0.75f0)
        bad[end] = Inf32
        compare_transition!(baseline, candidate, bad, Int32(0), closed, 0.5f0, 0.75f0)
        overflow = fill(floatmax(Float32), count)
        compare_transition!(baseline, candidate, overflow, Int32(0), closed, 2f0, 1f0)
        count > 1 && (overflow[1] = NaN32)
        compare_transition!(baseline, candidate, overflow, Int32(0), closed, 2f0, 1f0)
        compare_transition!(baseline, candidate, finite, Int32(0), closed, 0.5f0, 0.75f0)
    end
    # The existing CUDA build contracts this update to FMA. Preserve its finite
    # cancellation result as well as exact agreement with the retained kernel.
    baseline, candidate = buffers(257, false), buffers(257, false)
    fill!(baseline.committed, -floatmax(Float32))
    fill!(candidate.committed, -floatmax(Float32))
    compare_transition!(baseline, candidate, fill(floatmax(Float32), 257),
        Int32(0), true, 2f0, 1f0)
    @test only(Array(candidate.status)) == Int32(0)
    @test all(==(floatmax(Float32)), Array(candidate.committed))
end

@testset "CUDA leaky preparation selects local adoption and reset" begin
    prepared = prepare_algorithm(LeakyIntegratorF32, LeakyIntegratorF32Configuration(
        extent=257, initial_state=2f0, input_schema="test.signal/1", output_schema="test.signal/1"))
    input = CUDA.fill(1f0, 257)
    output = CUDA.zeros(Float32, 257)
    status = CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (input,), status, Int32(2))
    @test captured.adopt! isa CUDAX._LeakyAdopted
    enqueue!(captured)
    CUDA.synchronize()
    @test iszero(only(Array(status)))
    @test bits(output) == bits(captured.committed_state)
    graph = CUDA.capture(() -> enqueue!(captured))
    nodes = Ref{Csize_t}(0)
    CUDA.CUDACore.cuGraphGetNodes(graph, C_NULL, nodes)
    @test nodes[] == 1
    executable = CUDA.instantiate(graph)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test bits(output) == bits(captured.committed_state)
    replay_bytes(executable)
    CUDA.synchronize()
    @test replay_bytes(executable) == 0
    CUDA.synchronize()
    reset!(captured)
    CUDA.synchronize()
    @test Array(captured.committed_state) == fill(2f0, 257)
    @test Array(output) == fill(2f0, 257)
end

end # module
