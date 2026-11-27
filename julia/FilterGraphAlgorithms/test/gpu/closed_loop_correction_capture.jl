module ClosedLoopCorrectionCaptureTests

using CUDA
using KernelAbstractions
using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!, prepare_capture,
                              enqueue!,
                              complete!

function prepared_correction(n = 5; gain = 0.5f0, pole = 0.75f0,
        anti_windup_gain = 0.6f0, hidden_mode_gain = 0.3f0, hidden_modes = 2,
        shape_to_hidden = Float32[(-1)^(i + j) * (i + j) / (4n) for i in 1:hidden_modes, j in 1:n],
        hidden_to_shape = Float32[(2i - j) / (3n) for i in 1:n, j in 1:hidden_modes])
    prepared = prepare_algorithm(ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(state_length = n,
            hidden_mode_count = size(shape_to_hidden, 1)))
    replace_parameter!(prepared, "shape-to-hidden", shape_to_hidden)
    replace_parameter!(prepared, "hidden-to-shape", hidden_to_shape)
    prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
        (
            PropertyAssignment(0, gain), PropertyAssignment(1, pole),
            PropertyAssignment(2, anti_windup_gain), PropertyAssignment(3, hidden_mode_gain)
        ))
    return prepared
end

function captured_case(prepared; in_place = false)
    n = prepared.plan.state_length
    input, feedback = CUDA.zeros(Float32, n), CUDA.zeros(Float32, n)
    correction, state = in_place ? (input, feedback) :
                        (CUDA.zeros(Float32, n), CUDA.zeros(Float32, n))
    status = CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (correction, state), (input, feedback), status, Int32(2))
    return (; captured, input, feedback, correction, state, status)
end

function enqueue_step!(case)
    fill!(case.status, Int32(0))
    enqueue!(case.captured)
    return nothing
end

function record_step(case)
    enqueue_step!(case)
    CUDA.synchronize()
    reset!(case.captured)
    CUDA.synchronize()
    graph = CUDA.capture() do
        enqueue_step!(case)
    end
    return CUDA.instantiate(graph)
end

replay_bytes(executable) = @allocated CUDA.launch(executable)

function checked_replay!(case, executable, residual, feedback)
    copyto!(case.input, residual)
    copyto!(case.feedback, feedback)
    CUDA.launch(executable)
    CUDA.synchronize()
    return complete!(case.captured, only(Array(case.status)))
end

function assert_history(case, cpu)
    @test all(isapprox(actual, expected; rtol=3.0f-5, atol=3.0f-6) for
        (actual, expected) in zip(Array(case.captured.vectors.committed_state),
            closed_loop_correction_state(cpu.workspace)))
    expected_output = closed_loop_correction_last_output(cpu.workspace)
    @test only(Array(case.captured.has_output)) == Int32(!isnothing(expected_output))
    if !isnothing(expected_output)
        @test all(isapprox(actual, expected; rtol=3.0f-5, atol=3.0f-6) for
            (actual, expected) in zip(Array(case.captured.vectors.committed_output), expected_output))
    end
end

@testset "open CUDA replay holds seeded correction and state" begin
    cpu = prepared_correction()
    n = cpu.plan.state_length
    expected = (zeros(Float32, n), zeros(Float32, n))
    residual = fill(0.25f0, n); feedback = fill(0.125f0, n)
    process!(expected..., cpu, residual, feedback)
    process!(expected..., cpu, residual, feedback)
    cpu.plan = FilterGraphAlgorithms.prepare_properties(cpu.plan, (PropertyAssignment(4, false),))
    held = map(copy, expected)
    case = captured_case(cpu)
    copyto!(case.input, fill(floatmax(Float32), n))
    copyto!(case.feedback, fill(floatmax(Float32), n))
    enqueue_step!(case); CUDA.synchronize()
    graph = CUDA.capture() do
        enqueue_step!(case)
    end
    executable = CUDA.instantiate(graph)
    for _ in 1:3
        CUDA.launch(executable); CUDA.synchronize()
        @test complete!(case.captured, only(Array(case.status))) === nothing
        @test Array(case.correction) == held[1]
        @test Array(case.state) == held[2]
        assert_history(case, cpu)
    end
    copyto!(case.feedback, fill(NaN32, n))
    CUDA.launch(executable); CUDA.synchronize()
    @test complete!(case.captured, only(Array(case.status))) === ClosedLoopCorrectionNonFiniteFeedback
    @test Array(case.correction) == held[1]
    reset!(case.captured); reset!(cpu); CUDA.synchronize()
    @test checked_replay!(case, executable, residual, fill(NaN32, n)) === nothing
    @test all(iszero, Array(case.correction))
    @test all(iszero, Array(case.state))
    process!(expected..., cpu, residual, fill(NaN32, n))
    assert_history(case, cpu)
end

@testset "open CUDA leaky replay holds mathematical state" begin
    prepared = prepare_algorithm(LeakyIntegratorF32, LeakyIntegratorF32Configuration(
        extent=5, initial_state=2f0, input_schema="test.signal/1", output_schema="test.signal/1"))
    prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
        (PropertyAssignment(0, floatmax(Float32)), PropertyAssignment(1, 0.5f0),
            PropertyAssignment(2, false)))
    input = CUDA.fill(floatmax(Float32), 5); output = CUDA.zeros(Float32, 5)
    status = CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (input,), status, Int32(2))
    step!() = (fill!(status, Int32(0)); enqueue!(captured); nothing)
    step!(); CUDA.synchronize()
    graph = CUDA.capture() do
        step!()
    end
    executable = CUDA.instantiate(graph)
    for _ in 1:3
        CUDA.launch(executable); CUDA.synchronize()
        @test complete!(captured, only(Array(status))) === nothing
        @test Array(output) == fill(2f0, 5)
        @test Array(captured.committed_state) == fill(2f0, 5)
    end
    fill!(input, NaN32)
    CUDA.launch(executable); CUDA.synchronize()
    @test complete!(captured, only(Array(status))) === LeakyIntegratorNonFiniteInput
    @test Array(captured.committed_state) == fill(2f0, 5)
    reset!(captured); CUDA.synchronize()
    @test Array(captured.committed_state) == fill(2f0, 5)
end

const GPU = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)

@kernel function expand_mode!(output, coefficients, modes)
    i = @index(Global, Linear)
    if i <= length(output)
        GPU._expand_mode(coefficients, output, modes, i)
    end
end

@testset "single-mode expansion preserves cuBLAS products" begin
    values = Float32[0, -0.0, nextfloat(0.0f0), -nextfloat(0.0f0),
        floatmin(Float32), -floatmin(Float32), prevfloat(floatmin(Float32)),
        0.5, -0.5, 1, -1, 2, -2, floatmax(Float32), -floatmax(Float32)]
    @test_throws DimensionMismatch GPU._prepare_expansion(CUDA.device(),
        zeros(Float32, 2, 1), CUDA.zeros(Float32, 3), CUDA.zeros(Float32, 1))
    @test_throws DimensionMismatch GPU._prepare_expansion(CUDA.device(),
        zeros(Float32, 2, 1), CUDA.zeros(Float32, 2), CUDA.zeros(Float32, 2))
    @test_throws ArgumentError GPU._prepare_expansion(CUDA.device(),
        zeros(Float64, 2, 1), CUDA.zeros(Float32, 2), CUDA.zeros(Float32, 1))
    # Dense products, repeated selection and 1×1 identity have distinct policies.
    for matrix in (reshape(values, :, 1), ones(Float32, length(values), 1), ones(Float32, 1, 1))
        input, reference, output = CUDA.zeros(Float32, 1), CUDA.zeros(Float32, size(matrix, 1)),
            CUDA.zeros(Float32, size(matrix, 1))
        status = CUDA.zeros(Int32, 1)
        projection = GPU._prepare_projection(CUDA.device(), matrix, reference, input, 128)
        expansion = GPU._prepare_expansion(CUDA.device(), matrix, output, input)
        @test expansion isa GPU.ModeExpansion
        coefficients = GPU._expansion_values(expansion)
        kernel! = expand_mode!(get_backend(output), 32)
        for h in vcat(values, Float32[Inf, -Inf, NaN])
            copyto!(input, [h])
            fill!(reference, NaN32)
            fill!(output, NaN32)
            GPU._enqueue_projection!(projection, reference, status, input)
            kernel!(output, coefficients, input; ndrange=length(output))
            CUDA.synchronize()
            @test isequal(Array(output), Array(reference))
        end
    end
end

@testset "larger hidden expansions keep ordinary projections" begin
    n = GPU.VALIDATION_WORKGROUP_SIZE + 1
    output = CUDA.zeros(Float32, n)
    modes = CUDA.zeros(Float32, 1)
    @test !(GPU._prepare_expansion(CUDA.device(), ones(Float32, n, 1), output, modes) isa GPU.ModeExpansion)
end

@testset "CUDA workgroup requirements reject unsupported limits before launch" begin
    limits = (max_threads=256, max_x=128, shared_memory=2048, warp_width=32)
    @test GPU._check_capture_workgroup(128, limits; shared_memory=2048, warp_width=32) === nothing
    @test_throws ArgumentError GPU._check_capture_workgroup(129, limits)
    @test_throws ArgumentError GPU._check_capture_workgroup(0, limits)
    @test_throws ArgumentError GPU._check_capture_workgroup(128, limits; shared_memory=2049)
    @test_throws ArgumentError GPU._check_capture_workgroup(128, limits; warp_width=64)
end

@testset "CUDA closed-loop correction captures delayed feedback and hidden modes" begin
    @test capture_support(ClosedLoopCorrectionF32, CUDA.device()) isa SupportsCapture
    for (n, modes) in ((5, 2), (257, 2), (1, 1), (256, 1), (257, 1))
        cpu = prepared_correction(n; hidden_modes=modes)
        source = prepared_correction(n; hidden_modes=modes)
        case = captured_case(source)
        @test (case.captured.projections.hidden_to_shape isa GPU.ModeExpansion) ==
              (modes == 1 && n <= GPU.VALIDATION_WORKGROUP_SIZE)
        executable = record_step(case)
        # Capture owns fixed copies of both maps and all four gains.
        replace_parameter!(source, "shape-to-hidden", zeros(Float32, modes, n))
        replace_parameter!(source, "hidden-to-shape", zeros(Float32, n, modes))
        source.plan = FilterGraphAlgorithms.prepare_properties(source.plan,
            (
                PropertyAssignment(0, 0.0f0), PropertyAssignment(1, 0.0f0),
                PropertyAssignment(2, 0.0f0), PropertyAssignment(3, 0.0f0)
            ))
        expected_output, expected_state = zeros(Float32, n), zeros(Float32, n)
        previous = zeros(Float32, n)
        for frame in 1:12
            residual = Float32[0.4sin(Float32(frame + i) / 3) + 0.03frame for i in 1:n]
            # The feedback reports the difference from the preceding requested
            # correction after downstream clipping, hence is delayed one frame.
            feedback = frame == 1 ? fill(NaN32, n) :
                       previous .- clamp.(previous, -0.08f0, 0.08f0)
            @test process!(expected_output, expected_state, cpu, residual, feedback) ===
                  nothing
            @test checked_replay!(case, executable, residual, feedback) === nothing
            @test Array(case.correction)≈expected_output rtol=3.0f-5 atol=3.0f-6
            @test Array(case.state)≈expected_state rtol=3.0f-5 atol=3.0f-6
            assert_history(case, cpu)
            copyto!(previous, expected_output)
        end

        # A rejected later feedback must retain both histories and caller outputs.
        saved_output, saved_state = Array(case.correction), Array(case.state)
        residual, bad_feedback = fill(0.2f0, n), fill(NaN32, n)
        @test process!(expected_output, expected_state, cpu, residual, bad_feedback) ===
              ClosedLoopCorrectionNonFiniteFeedback
        @test checked_replay!(case, executable, residual, bad_feedback) ===
              ClosedLoopCorrectionNonFiniteFeedback
        @test Array(case.correction) == saved_output
        @test Array(case.state) == saved_state
        assert_history(case, cpu)

        # Input rejection outranks simultaneous bad feedback and output arithmetic.
        residual[end] = NaN32
        @test process!(expected_output, expected_state, cpu, residual, bad_feedback) ===
              ClosedLoopCorrectionNonFiniteInput
        @test checked_replay!(case, executable, residual, bad_feedback) ===
              ClosedLoopCorrectionNonFiniteInput
        assert_history(case, cpu)

        addresses = map(pointer,
            (case.captured.vectors.committed_state,
                case.captured.vectors.committed_output, case.captured.has_output))
        @test reset!(case.captured) === case.captured
        reset!(cpu)
        CUDA.synchronize()
        @test map(pointer,
            (case.captured.vectors.committed_state,
                case.captured.vectors.committed_output, case.captured.has_output)) ==
              addresses
        @test Array(case.captured.vectors.hidden_modes) == zeros(Float32, modes)
        @test Array(case.captured.vectors.hidden_in_state) == zeros(Float32, n)
        assert_history(case, cpu)
        residual .= 0.25f0
        @test process!(expected_output, expected_state, cpu, residual, bad_feedback) ===
              nothing
        @test checked_replay!(case, executable, residual, bad_feedback) === nothing
        assert_history(case, cpu)

        # Only native replay is the steady-state host submission path.
        fill!(case.feedback, 0.0f0)
        replay_bytes(executable)
        CUDA.synchronize()
        @test replay_bytes(executable) == 0
        CUDA.synchronize()
    end
end

@testset "CUDA closed-loop numerical failures preserve priorities and history" begin
    n = 5
    cpu = prepared_correction(n; gain = floatmax(Float32))
    case = captured_case(prepared_correction(n; gain = floatmax(Float32)))
    executable = record_step(case)
    output, state = zeros(Float32, n), zeros(Float32, n)
    input, feedback = fill(2.0f0, n), fill(NaN32, n)
    @test process!(output, state, cpu, input, feedback) ===
          ClosedLoopCorrectionNonFiniteOutput
    @test checked_replay!(case, executable, input, feedback) ===
          ClosedLoopCorrectionNonFiniteOutput
    assert_history(case, cpu)
    # An output-rejected initial frame still ignores feedback on the next attempt.
    input .= 0.125f0
    @test process!(output, state, cpu, input, feedback) === nothing
    @test checked_replay!(case, executable, input, feedback) === nothing
    assert_history(case, cpu)
    input .= 2.0f0
    @test process!(output, state, cpu, input, feedback) ===
          ClosedLoopCorrectionNonFiniteFeedback
    @test checked_replay!(case, executable, input, feedback) ===
          ClosedLoopCorrectionNonFiniteFeedback
    input[end] = Inf32
    @test process!(output, state, cpu, input, feedback) ===
          ClosedLoopCorrectionNonFiniteInput
    @test checked_replay!(case, executable, input, feedback) ===
          ClosedLoopCorrectionNonFiniteInput
    assert_history(case, cpu)

    # Finite external data can overflow only inside hidden-mode projection.
    forward = fill(floatmax(Float32), 2, n)
    backward = ones(Float32, n, 2)
    cpu = prepared_correction(n; shape_to_hidden = forward, hidden_to_shape = backward)
    case = captured_case(prepared_correction(n; shape_to_hidden = forward, hidden_to_shape = backward))
    executable = record_step(case)
    input, feedback = fill(1.0f0, n), zeros(Float32, n)
    @test process!(output, state, cpu, input, feedback) === nothing
    @test checked_replay!(case, executable, input, feedback) === nothing
    @test process!(output, state, cpu, input, feedback) ===
          ClosedLoopCorrectionNonFiniteOutput
    @test checked_replay!(case, executable, input, feedback) ===
          ClosedLoopCorrectionNonFiniteOutput
    assert_history(case, cpu)
end

@testset "single-mode hidden overflow is checked even with zero hidden gain" begin
    n = 5
    forward = reshape(Float32[1, 0, 0, 0, 0], 1, :)
    backward = reshape(Float32[2, 0, 0, 0, 0], :, 1)
    for (shape_to_hidden, hidden_to_shape, feedback_value) in (
        (forward, backward, -floatmax(Float32)),
        (fill(floatmax(Float32), 1, n), zeros(Float32, n, 1), 0.0f0),
        (fill(floatmax(Float32), 1, n), ones(Float32, n, 1), 0.0f0),
    )
        prepare() = prepared_correction(n; gain=1.0f0, pole=0.0f0,
            anti_windup_gain=0.0f0, hidden_mode_gain=0.0f0,
            shape_to_hidden, hidden_to_shape)
        cpu, case = prepare(), captured_case(prepare())
        executable = record_step(case)
        output, state = zeros(Float32, n), zeros(Float32, n)
        residual, feedback = ones(Float32, n), zeros(Float32, n)
        @test process!(output, state, cpu, residual, feedback) === nothing
        @test checked_replay!(case, executable, residual, feedback) === nothing
        feedback[1] = feedback_value
        saved_output, saved_state = Array(case.correction), Array(case.state)
        @test process!(output, state, cpu, residual, feedback) === ClosedLoopCorrectionNonFiniteOutput
        @test checked_replay!(case, executable, residual, feedback) === ClosedLoopCorrectionNonFiniteOutput
        @test isequal(Array(case.correction), saved_output)
        @test isequal(Array(case.state), saved_state)
        assert_history(case, cpu)
    end
end

@testset "CUDA closed-loop adopts before a later Node rejects" begin
    cpu = prepared_correction(; hidden_modes=1)
    case = captured_case(prepared_correction(; hidden_modes=1))
    leaky = prepare_algorithm(LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(extent = 5, initial_state = 0.0f0,
            input_schema = CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema = CONTROLLER_COMMAND_V1))
    bad_input, later_output = CUDA.fill(NaN32, 5), CUDA.zeros(Float32, 5)
    later = prepare_capture(CompleteTransition(), CUDA.device(), leaky, (later_output,),
        (bad_input,), case.status, Int32(3))
    enqueue_step!(case)
    enqueue!(later)
    CUDA.synchronize()
    reset!(case.captured)
    CUDA.synchronize()
    graph = CUDA.capture() do
        enqueue_step!(case)
        enqueue!(later)
    end
    executable = CUDA.instantiate(graph)
    output, state = zeros(Float32, 5), zeros(Float32, 5)
    for frame in 1:3
        input = fill(Float32(frame) / 4, 5)
        feedback = fill(0.1f0 * (frame - 1), 5)
        @test process!(output, state, cpu, input, feedback) === nothing
        checked_replay!(case, executable, input, feedback)
        @test complete!(later, only(Array(case.status))) ===
              LeakyIntegratorNonFiniteInput
        assert_history(case, cpu)
        @test Array(case.correction)≈output rtol=3.0f-5 atol=3.0f-6
    end
    # An earlier disposition prevents this Node from adopting or publishing.
    saved_state = Array(case.captured.vectors.committed_state)
    saved_output = Array(case.correction)
    # Leave a rejected, visibly different local candidate before the earlier
    # failure. An unguarded adoption would otherwise copy the preceding success.
    fill!(case.feedback, NaN32)
    enqueue_step!(case)
    CUDA.synchronize()
    @test complete!(case.captured, only(Array(case.status))) ===
          ClosedLoopCorrectionNonFiniteFeedback
    fill!(case.status, Int32(0x103))
    enqueue!(case.captured)
    CUDA.synchronize()
    @test only(Array(case.status)) == Int32(0x103)
    @test Array(case.captured.vectors.committed_state) == saved_state
    @test Array(case.correction) == saved_output
end

@testset "CUDA controller finalization joins all candidate readers before commitment" begin
    for n in (1, 257, 513), modes in (1, 2), in_place in (false, true)
        prepare() = prepared_correction(n; gain = floatmax(Float32), pole = 0.25f0,
            anti_windup_gain = 0.0f0, hidden_mode_gain = 0.0f0,
            shape_to_hidden = zeros(Float32, modes, n),
            hidden_to_shape = zeros(Float32, n, modes))
        cpu, case = prepare(), captured_case(prepare(); in_place)
        output, state = zeros(Float32, n), zeros(Float32, n)
        addresses = map(pointer, (values(case.captured.vectors)..., case.captured.has_output))
        for (step, reject) in enumerate((true, false, true, false))
            residual = fill(step <= 2 ? 0.125f0 : 0.0625f0, n)
            reject && (residual[end] = 2.0f0)
            feedback = fill(step <= 2 ? NaN32 : 0.0f0, n)
            copyto!(case.input, residual)
            copyto!(case.feedback, feedback)
            supplied = (Array(case.correction), Array(case.state))
            expected = process!(output, state, cpu, residual, feedback)
            enqueue_step!(case)
            CUDA.synchronize()
            @test complete!(case.captured, only(Array(case.status))) === expected
            if reject
                @test expected === ClosedLoopCorrectionNonFiniteOutput
                @test isequal((Array(case.correction), Array(case.state)), supplied)
            else
                @test expected === nothing
                @test all(isapprox(actual, expected; rtol=3.0f-5, atol=3.0f-6) for
                    (actual, expected) in zip(Array(case.correction), output))
                @test all(isapprox(actual, expected; rtol=3.0f-5, atol=3.0f-6) for
                    (actual, expected) in zip(Array(case.state), state))
                @test Array(case.captured.vectors.committed_state) == Array(case.state)
                @test Array(case.captured.vectors.committed_output) == Array(case.correction)
            end
            assert_history(case, cpu)
        end
        supplied = (Array(case.correction), Array(case.state))
        fill!(case.status, Int32(0x103))
        enqueue!(case.captured)
        CUDA.synchronize()
        @test only(Array(case.status)) == Int32(0x103)
        @test isequal((Array(case.correction), Array(case.state)), supplied)
        assert_history(case, cpu)
        reset!(case.captured)
        reset!(cpu)
        CUDA.synchronize()
        @test map(pointer, (values(case.captured.vectors)..., case.captured.has_output)) == addresses
        assert_history(case, cpu)
    end
end

end # module
