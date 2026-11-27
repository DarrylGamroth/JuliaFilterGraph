module PdmCommandCaptureTests

using CUDA, FilterGraphAlgorithms, Test
import FilterGraphAlgorithms: process!, reset!, prepare_capture,
                              enqueue!, complete!

const FGA = FilterGraphAlgorithms
const CUDAX = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)
const EXTENT = 277
const LOWER = fill(-0.8f0, EXTENT)
const UPPER = fill(0.8f0, EXTENT)
const SLEW_LIMITS = Float32[0.025f0 + 0.005f0 * (i % 7) for i in 1:EXTENT]
const INITIAL_CURRENT = Float32[0.03f0 * sin(Float32(i) / 13) for i in 1:EXTENT]
const RTOL = 4.0f-5
const ATOL = 4.0f-6

function prepared_command(slew::Symbol; extent = EXTENT)
    lower = extent == EXTENT ? LOWER : fill(-0.8f0, extent)
    upper = extent == EXTENT ? UPPER : fill(0.8f0, extent)
    slew_limits = extent == EXTENT ? SLEW_LIMITS :
                  Float32[0.025f0 + 0.005f0 * (i % 7) for i in 1:extent]
    initial = extent == EXTENT ? INITIAL_CURRENT :
              Float32[0.03f0 * sin(Float32(i) / 13) for i in 1:extent]
    slew_limit = slew === :none ? nothing : slew_limits
    configuration = PdmCommandF32Configuration(
        actuator_count = extent,
        command_limits=(lower, upper),
        slew_limit = slew_limit,
        initial_current = initial,
    )
    return prepare_algorithm(PdmCommandF32, configuration)
end

function captured_command(prepared)
    extent = length(prepared.plan.lower_limits)
    requested = CUDA.zeros(Float32, extent)
    demanded = CUDA.zeros(Float32, extent)
    feedback = CUDA.zeros(Float32, extent)
    status = CUDA.zeros(Int32, 1)
    adapter = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (demanded, feedback),
        (requested,), status, Int32(1))
    return (; adapter, requested, demanded, feedback, status)
end

function enqueue_step!(case)
    enqueue!(case.adapter)
    return nothing
end

function record_step(case, execution = case.adapter.execution)
    fill!(case.status, 0)
    CUDAX._enqueue_pdm!(case.adapter, execution)
    CUDA.synchronize()
    reset!(case.adapter)
    CUDA.synchronize()
    graph = CUDA.capture() do
        CUDAX._enqueue_pdm!(case.adapter, execution)
    end
    # Retain the source graph for inspection independently of executable internals.
    return graph, CUDA.instantiate(graph)
end

function replay!(case, executable, requested)
    copyto!(case.requested, requested)
    fill!(case.status, 0)
    CUDA.launch(executable)
    CUDA.synchronize()
    return complete!(case.adapter, only(Array(case.status)))
end

replay_bytes(executable) = @allocated CUDA.launch(executable)

function native_graph_node_count(graph)
    count = Ref{Csize_t}(0)
    CUDA.CUDACore.cuGraphGetNodes(graph, C_NULL, count)
    return count[]
end

function requested_frame(frame, extent = EXTENT)
    return Float32[0.94f0 * sin(0.071f0 * Float32(i) + 0.37f0 * Float32(frame)) +
                   0.16f0 * cos(0.013f0 * Float32(i - frame))
                   for i in 1:extent]
end

function assert_workspace(case, prepared)
    adapter = case.adapter
    workspace = prepared.workspace
    @test Array(adapter.committed_current)≈workspace.committed_current rtol=RTOL atol=ATOL
end

function assert_rejected_without_adoption!(case, executable, prepared, bad_input)
    adapter = case.adapter
    saved_current = Array(adapter.committed_current)
    saved_demanded, saved_feedback = Array(case.demanded), Array(case.feedback)

    @test process!(zeros(Float32, EXTENT), zeros(Float32, EXTENT), prepared,
        bad_input) === PdmCommandNonFiniteInput
    @test replay!(case, executable, bad_input) === PdmCommandNonFiniteInput
    @test Array(adapter.committed_current) == saved_current
    @test Array(case.demanded) == saved_demanded
    @test Array(case.feedback) == saved_feedback

    # A failure from an earlier Node is authoritative and prevents adoption.
    copyto!(case.requested, requested_frame(19))
    fill!(case.status, Int32(0x103))
    CUDA.launch(executable)
    CUDA.synchronize()
    @test only(Array(case.status)) == Int32(0x103)
    @test complete!(adapter, only(Array(case.status))) === nothing
    @test Array(adapter.committed_current) == saved_current
    @test Array(case.demanded) == saved_demanded
    @test Array(case.feedback) == saved_feedback
end

@testset "CUDA PDM captured policies match the CPU oracle" begin
    @test capture_support(PdmCommandF32, CUDA.device()) isa SupportsCapture
    # Both accelerator extensions use the scalar core helper without installing
    # competing methods on that shared function.
    @test CUDAX._pdm_current === FGA._pdm_current
    @test all(method -> method.module === FGA, methods(FGA._pdm_current))
    for slew in (:none, :independent)
        cpu = prepared_command(slew)
        case = captured_command(prepared_command(slew))
        adapter = case.adapter

        expected_slew = slew === :none ? CUDAX.NoSlew : CUDAX.IndependentSlew
        @test isa(adapter.slew, expected_slew)

        native_graph, executable = record_step(case)
        @test native_graph_node_count(native_graph) ==
              (adapter.execution isa CUDAX._PdmBlock ? 1 : 4)
        demanded, feedback = zeros(Float32, EXTENT), zeros(Float32, EXTENT)
        outputs_before_reset = Tuple{Vector{Float32}, Vector{Float32}}[]
        for frame in 1:12
            requested = requested_frame(frame)
            @test process!(demanded, feedback, cpu, requested) === nothing
            @test replay!(case, executable, requested) === nothing
            @test Array(case.demanded)≈demanded rtol=RTOL atol=ATOL
            @test Array(case.feedback)≈feedback rtol=RTOL atol=ATOL
            assert_workspace(case, cpu)
            push!(outputs_before_reset,
                (Array(case.demanded), Array(case.feedback)))
        end

        assert_rejected_without_adoption!(case, executable, cpu,
            let bad = requested_frame(13)
                bad[137] = NaN32
                bad
            end)
        assert_rejected_without_adoption!(case, executable, cpu,
            let bad = requested_frame(14)
                bad[211] = Inf32
                bad
            end)

        # Reset restores all command state; replaying the same input sequence
        # must reproduce both outputs and the final committed state.
        first_final = Array(adapter.committed_current)
        reset!(cpu)
        reset!(adapter)
        CUDA.synchronize()
        assert_workspace(case, cpu)
        for frame in 1:12
            requested = requested_frame(frame)
            @test process!(demanded, feedback, cpu, requested) === nothing
            @test replay!(case, executable, requested) === nothing
            @test Array(case.demanded)≈demanded rtol=RTOL atol=ATOL
            @test Array(case.feedback)≈feedback rtol=RTOL atol=ATOL
            @test Array(case.demanded)≈outputs_before_reset[frame][1] rtol=RTOL atol=ATOL
            @test Array(case.feedback)≈outputs_before_reset[frame][2] rtol=RTOL atol=ATOL
            assert_workspace(case, cpu)
        end
        @test Array(adapter.committed_current)≈first_final rtol=RTOL atol=ATOL

        replay_bytes(executable)
        CUDA.synchronize()
        @test replay_bytes(executable) == 0
        CUDA.synchronize()
    end
end

@testset "CUDA PDM zero slew limit leaves other actuators independent" begin
    prepared = prepare_algorithm(PdmCommandF32,
        PdmCommandF32Configuration(
            actuator_count = 3,
            command_limits = (fill(-1.0f0, 3), fill(1.0f0, 3)),
            slew_limit = Float32[0.0, 0.1, 0.2],
            initial_current = zeros(Float32, 3),
        ))
    case = captured_command(prepared)
    @test isa(case.adapter.slew, CUDAX.IndependentSlew)
    _, executable = record_step(case)
    requested = Float32[0.5, 0.5, -0.5]
    demanded, feedback = zeros(Float32, 3), zeros(Float32, 3)
    @test process!(demanded, feedback, prepared, requested) === nothing
    @test replay!(case, executable, requested) === nothing
    @test Array(case.adapter.committed_current) == Float32[0, 0.1, -0.2]
    @test Array(case.demanded) == demanded == Float32[0, 0.1, -0.2]
    @test Array(case.feedback) == feedback
end

@testset "CUDA PDM output overflow does not adopt candidate state" begin
    extreme = floatmax(Float32)
    prepared = prepare_algorithm(PdmCommandF32,
        PdmCommandF32Configuration(
            actuator_count = 1,
            command_limits=(-extreme, extreme),
            slew_limit = 0.0f0,
            initial_current = Float32[-extreme],
        ))
    case = let
        requested = CUDA.zeros(Float32, 1)
        demanded = CUDA.zeros(Float32, 1)
        feedback = CUDA.zeros(Float32, 1)
        status = CUDA.zeros(Int32, 1)
        adapter = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (demanded, feedback),
            (requested,), status, Int32(1))
        (; adapter, requested, demanded, feedback, status)
    end
    _, executable = record_step(case)

    cpu_demanded, cpu_feedback = zeros(Float32, 1), zeros(Float32, 1)
    @test process!(cpu_demanded, cpu_feedback, prepared, Float32[-extreme]) === nothing
    @test replay!(case, executable, Float32[-extreme]) === nothing
    @test Array(case.demanded) == cpu_demanded
    @test Array(case.feedback) == cpu_feedback
    saved_current = Array(case.adapter.committed_current)
    @test saved_current == Float32[-extreme]
    cpu_current = copy(prepared.workspace.committed_current)
    @test process!(cpu_demanded, cpu_feedback, prepared, Float32[extreme]) ===
          PdmCommandNonFiniteOutput
    @test replay!(case, executable, Float32[extreme]) === PdmCommandNonFiniteOutput
    @test prepared.workspace.committed_current == cpu_current
    @test Array(case.adapter.committed_current) == saved_current
end

const PDM_CAPTURE_ARRAYS = (
    :demanded, :constraint_feedback, :committed_current, :candidate_current,
)

function assert_staged_state(case, staged)
    @test Array(case.status) == Array(staged.status)
    for field in PDM_CAPTURE_ARRAYS
        @test isequal(Array(getfield(case.adapter, field)), Array(getfield(staged.adapter, field)))
    end
end

@testset "CUDA bounded PDM transition matches stages at block boundaries" begin
    for extent in (1, 255, 256, 257, 277, 511, 512, 513, 1025),
        slew in (:none, :independent)
        @testset "N=$extent slew=$slew" begin
            cpu = prepared_command(slew; extent)
            case, staged = captured_command(cpu), captured_command(cpu)
            @test (case.adapter.execution isa CUDAX._PdmBlock) == (extent <= 512)
            native_graph, executable = record_step(case)
            reference_graph, reference = record_step(staged, CUDAX._PdmStages())
            @test native_graph_node_count(native_graph) ==
                  (extent <= 512 ? 1 : 4)
            @test native_graph_node_count(reference_graph) == 4
            storage = map(field -> getfield(case.adapter, field), PDM_CAPTURE_ARRAYS)
            addresses = map(pointer, storage)
            demanded, feedback = zeros(Float32, extent), zeros(Float32, extent)
            for (kind, sample) in ((:changing, requested_frame(1, extent)),
                (:signed_zero, fill(-0.0f0, extent)),
                (:subnormal, fill(reinterpret(Float32, UInt32(1)), extent)))
                @testset "sample=$kind" begin
                    @test process!(demanded, feedback, cpu, sample) === nothing
                    @test replay!(case, executable, sample) === nothing
                    @test replay!(staged, reference, sample) === nothing
                    assert_staged_state(case, staged)
                    @test Array(case.demanded) ≈ demanded rtol=RTOL atol=ATOL
                    @test Array(case.feedback) ≈ feedback rtol=RTOL atol=ATOL
                    assert_workspace(case, cpu)
                end
            end
            bad = requested_frame(4, extent)
            bad[end] = NaN32
            @test replay!(case, executable, bad) === PdmCommandNonFiniteInput
            @test replay!(staged, reference, bad) === PdmCommandNonFiniteInput
            assert_staged_state(case, staged)
            for (owner, graph) in ((case, executable), (staged, reference))
                copyto!(owner.requested, requested_frame(5, extent))
                fill!(owner.status, Int32(0x103))
                CUDA.launch(graph)
            end
            CUDA.synchronize()
            assert_staged_state(case, staged)
            @test only(Array(case.status)) == Int32(0x103)
            @testset "sample=recovery" begin
                sample = requested_frame(6, extent)
                @test process!(demanded, feedback, cpu, sample) === nothing
                @test replay!(case, executable, sample) === nothing
                @test replay!(staged, reference, sample) === nothing
                assert_staged_state(case, staged)
                @test Array(case.demanded) ≈ demanded rtol=RTOL atol=ATOL
                @test Array(case.feedback) ≈ feedback rtol=RTOL atol=ATOL
                assert_workspace(case, cpu)
            end
            # The Graph proof changes checking policy without replacing execution/storage.
            if extent == 277
                refined = FilterGraphAlgorithms.prepare_validation(case.adapter, (true,))
                @test refined.execution === case.adapter.execution
                @test CUDAX._pdm_check_input(refined.validate_input!) === Val(false)
                proven = merge(case, (; adapter=refined))
                _, proven_graph = record_step(proven)
                reset!(cpu)
                sample = requested_frame(6, extent)
                @test process!(demanded, feedback, cpu, sample) === nothing
                @test replay!(proven, proven_graph, sample) === nothing
                @test Array(proven.demanded) ≈ demanded rtol=RTOL atol=ATOL
                @test Array(proven.feedback) ≈ feedback rtol=RTOL atol=ATOL
                assert_workspace(proven, cpu)
            end
            @test reset!(case.adapter) === case.adapter
            CUDA.synchronize()
            @test map(field -> getfield(case.adapter, field), PDM_CAPTURE_ARRAYS) === storage
            @test map(pointer, storage) == addresses
        end
    end
end

@testset "CUDA bounded PDM rejects before any state commitment" begin
    extent = 257
    bound = floatmax(Float32)
    lower, upper, initial = fill(-0.8f0, extent), fill(0.8f0, extent), zeros(Float32, extent)
    lower[1] = upper[1] = initial[1] = -bound
    lower[end] = upper[end] = initial[end] = -bound
    for slew in (:none, :independent)
        cpu = prepare_algorithm(PdmCommandF32, PdmCommandF32Configuration(
            actuator_count=extent, command_limits=(lower, upper),
            slew_limit=slew === :none ? nothing : 0.05f0,
            initial_current=initial))
        case, staged = captured_command(cpu), captured_command(cpu)
        _, executable = record_step(case)
        _, reference = record_step(staged, CUDAX._PdmStages())
        demanded, feedback = fill(71.0f0, extent), fill(72.0f0, extent)
        # Earlier actuators have real candidate motion when the final lane rejects.
        bad_input = fill(0.5f0, extent)
        bad_input[1], bad_input[end] = bound, NaN32
        bad_output = fill(0.5f0, extent)
        bad_output[end] = bound
        valid = fill(0.25f0, extent)
        valid[end] = -bound
        for (sample, failure) in ((bad_input, PdmCommandNonFiniteInput),
            (bad_output, PdmCommandNonFiniteOutput), (valid, nothing))
            @test process!(demanded, feedback, cpu, sample) === failure
            @test replay!(case, executable, sample) === failure
            @test replay!(staged, reference, sample) === failure
            assert_staged_state(case, staged)
            assert_workspace(case, cpu)
        end
    end
end

@testset "CUDA aliased PDM preparations reject cold" begin
    prepared = prepared_command(:none; extent=2)
    input, demanded, feedback = CUDA.CuArray(Float32[1, -2]), CUDA.fill(71.0f0, 2), CUDA.fill(72.0f0, 2)
    status = CUDA.zeros(Int32, 1)
    initial_current = copy(prepared.workspace.committed_current)
    for outputs in ((input, feedback), (demanded, input), (demanded, demanded))
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(), CUDA.device(), prepared, outputs, (input,), status, Int32(1))
        @test Array(input) == Float32[1, -2]
        @test Array(demanded) == fill(71.0f0, 2)
        @test Array(feedback) == fill(72.0f0, 2)
        @test Array(status) == Int32[0]
        @test prepared.workspace.committed_current == initial_current
    end
end

end # module
