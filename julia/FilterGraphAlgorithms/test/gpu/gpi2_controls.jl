using FilterGraphAlgorithms, KernelAbstractions, Test

gpi2_control_replay_bytes(launch) = @allocated launch()
import FilterGraphAlgorithms: process!, reset!, prepare_capture, enqueue!, complete!

function run_gpi2_control_tests(device, to_device, device_zeros, synchronize;
    record=nothing, replay=nothing)
    cascade(n, coefficients) = prepare_algorithm(BiquadCascadeF32,
        BiquadCascadeF32Configuration(extent=n, section_count=size(coefficients, 2),
            per_mode=size(coefficients, 3) != 1,
            initial_coefficients=vec(copy(coefficients)),
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_RESIDUAL_ERROR_V1))
    function integrator(n; gain=0.3f0, closed=true)
        prepared = prepare_algorithm(ClosedLoopIntegratorF32,
            ClosedLoopIntegratorF32Configuration(state_length=n))
        prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
            (PropertyAssignment(0, gain), PropertyAssignment(1, 0.8f0),
                PropertyAssignment(2, 0.6f0), PropertyAssignment(4, closed)))
        return prepared
    end
    function capture(source, n, controller; in_place=false)
        inputs = controller ? (device_zeros(Float32, n), device_zeros(Float32, n)) :
            (device_zeros(Float32, n),)
        outputs = in_place ? inputs : map(_ -> to_device(fill(71f0, n)), inputs)
        status = device_zeros(Int32, 1)
        adapter = prepare_capture(CompleteTransition(), device, source, outputs,
            inputs, status, Int32(2))
        return (; adapter, inputs, outputs, status)
    end
    function execution(case, recorded)
        step!() = (fill!(case.status, Int32(0)); enqueue!(case.adapter); nothing)
        step!()
        synchronize()
        reset!(case.adapter)
        synchronize()
        if recorded
            executable = record(step!)
            reset!(case.adapter)
            synchronize()
            return () -> replay(executable)
        end
        return step!
    end
    function check_step(case, cpu, expected, samples, launch)
        foreach(copyto!, case.inputs, samples)
        saved = map(Array, case.outputs)
        result = process!(expected..., cpu, samples...)
        launch()
        synchronize()
        @test complete!(case.adapter, only(Array(case.status))) === result
        if isnothing(result)
            for (actual, reference) in zip(case.outputs, expected)
                @test Array(actual) ≈ reference rtol=4f-6 atol=3f-7
            end
        else
            @test isequal(map(Array, case.outputs), saved)
        end
        return result
    end
    function check_history(case, cpu, controller)
        if controller
            @test Array(case.adapter.vectors.committed_state) ≈ cpu.workspace.committed_state rtol=4f-6 atol=3f-7
            @test Array(case.adapter.vectors.committed_output) ≈ cpu.workspace.committed_output rtol=4f-6 atol=3f-7
            @test only(Array(case.adapter.has_output)) == Int32(cpu.workspace.committed_has_output)
        else
            @test Array(case.adapter.committed_state) ≈ cpu.workspace.committed_state rtol=4f-6 atol=3f-7
            @test Array(case.adapter.candidate_state) ≈ cpu.workspace.candidate_state rtol=4f-6 atol=3f-7
            @test Array(case.adapter.candidate_output) ≈ cpu.workspace.candidate_output rtol=4f-6 atol=3f-7
        end
    end
    function storage(case, controller)
        if controller
            return (values(case.adapter.vectors)..., case.adapter.has_output,
                case.outputs..., case.status)
        end
        return (case.adapter.coefficients, case.adapter.committed_state,
            case.adapter.candidate_state, case.adapter.candidate_output,
            case.inputs..., case.outputs..., case.status)
    end

    @testset "GPI2 temporal controls: trajectories, rejection, reset and capture" begin
        for n in (1, 257, 513), controller in (false, true), in_place in (false, true),
            recorded in (isnothing(record) ? (false,) : (false, true))
            coefficients = zeros(Float32, 5, 3, n)
            for mode in 1:n, section in 1:3
                coefficients[:, section, mode] .=
                    Float32[0.4 + mod(mode, 3)/32, 0.2, 0.1, -0.3, 0.05]
            end
            make() = controller ? integrator(n) : cascade(n, coefficients)
            cpu, source = make(), make()
            case = capture(source, n, controller; in_place)
            @test capture_support(controller ? ClosedLoopIntegratorF32 : BiquadCascadeF32,
                device) isa SupportsCapture
            @test FilterGraphAlgorithms.preflight(case.adapter) === nothing
            arrays = storage(case, controller)
            addresses = map(pointer, arrays)
            launch = execution(case, recorded)
            expected = map(_ -> zeros(Float32, n), case.outputs)
            # Capture owns its plan and history independently of the CPU source.
            if !controller
                fill!(source.plan.coefficients, 0f0)
                fill!(source.workspace.committed_state, 19f0)
            end
            for frame in 1:8
                residual = Float32[(-1)^mode * mod(mode + frame, 13)/32 for mode in 1:n]
                samples = controller ? (residual,
                    frame == 1 ? fill(NaN32, n) : fill(0.025f0, n)) : (residual,)
                @test check_step(case, cpu, expected, samples, launch) === nothing
                check_history(case, cpu, controller)
            end
            residual = fill(0.125f0, n)
            residual[end] = NaN32
            samples = controller ? (residual, fill(NaN32, n)) : (residual,)
            expected_failure = controller ? ClosedLoopCorrectionNonFiniteInput : BiquadCascadeNonFiniteInput
            @test check_step(case, cpu, expected, samples, launch) === expected_failure
            check_history(case, cpu, controller)
            if controller
                @test check_step(case, cpu, expected,
                    (fill(0.125f0, n), fill(NaN32, n)), launch) === ClosedLoopCorrectionNonFiniteFeedback
                check_history(case, cpu, controller)
            end
            saved = map(Array, case.outputs)
            fill!(case.status, Int32(0x103))
            enqueue!(case.adapter)
            synchronize()
            @test only(Array(case.status)) == Int32(0x103)
            @test isequal(map(Array, case.outputs), saved)
            check_history(case, cpu, controller)
            @test reset!(case.adapter) === case.adapter
            reset!(cpu)
            synchronize()
            check_history(case, cpu, controller)
            @test map(pointer, arrays) == addresses
            samples = controller ? (fill(0.125f0, n), fill(NaN32, n)) : (fill(0.125f0, n),)
            @test check_step(case, cpu, expected, samples, launch) === nothing
            check_history(case, cpu, controller)
            if recorded
                launch(); synchronize()
                gpi2_control_replay_bytes(launch)
                synchronize()
                @test gpi2_control_replay_bytes(launch) == 0
                synchronize()
            end
        end
    end

    @testset "GPI2 late overflow rejects complete sample and preserves local adoption" begin
        n = 257
        for controller in (false, true), in_place in (false, true)
            coefficients = zeros(Float32, 5, 2, 1)
            coefficients[1, :, :] .= 1f0
            coefficients[2, 2, 1] = floatmax(Float32)
            make() = controller ? integrator(n; gain=floatmax(Float32)) : cascade(n, coefficients)
            cpu, case = make(), capture(make(), n, controller; in_place)
            expected = map(_ -> zeros(Float32, n), case.outputs)
            launch = execution(case, false)
            residual = fill(0.125f0, n)
            residual[end] = 2f0
            samples = controller ? (residual, fill(NaN32, n)) : (residual,)
            failure = controller ? ClosedLoopCorrectionNonFiniteOutput : BiquadCascadeNonFiniteOutput
            @test check_step(case, cpu, expected, samples, launch) === failure
            check_history(case, cpu, controller)
            residual[end] = 0.125f0
            @test check_step(case, cpu, expected, samples, launch) === nothing
            check_history(case, cpu, controller)
            # A later Node's numerical rejection does not undo this adoption.
            later = prepare_algorithm(LeakyIntegratorF32, LeakyIntegratorF32Configuration(
                extent=n, initial_state=0f0, input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
                output_schema=CONTROLLER_RESIDUAL_ERROR_V1))
            bad = to_device(fill(NaN32, n))
            adapter = prepare_capture(CompleteTransition(), device, later,
                (device_zeros(Float32, n),), (bad,), case.status, Int32(3))
            reset!(case.adapter); reset!(cpu); synchronize()
            foreach(copyto!, case.inputs, samples)
            @test process!(expected..., cpu, samples...) === nothing
            launch(); enqueue!(adapter); synchronize()
            @test complete!(adapter, only(Array(case.status))) === LeakyIntegratorNonFiniteInput
            check_history(case, cpu, controller)
        end
    end

    @testset "Biquad validation reuse preserves numerical rejection and alias facts" begin
        n = 257
        coefficients = zeros(Float32, 5, 2, 1)
        coefficients[1, :, :] .= 1f0
        coefficients[2, 2, 1] = floatmax(Float32)
        for in_place in (false, true), recorded in (isnothing(record) ? (false,) : (false, true))
            cpu = cascade(n, coefficients)
            original = capture(cascade(n, coefficients), n, false; in_place)
            adapter = original.adapter
            qualified = FilterGraphAlgorithms._supports_validation_reuse(
                KernelAbstractions.get_backend(original.status))
            finite_input_type = getfield(parentmodule(typeof(adapter)), :_FiniteInput)
            @test !(adapter.validate_input! isa finite_input_type)
            @test FilterGraphAlgorithms.prepare_validation(adapter, (false,)) === adapter
            @test FilterGraphAlgorithms.preserves_inputs(adapter) == (qualified && !in_place)
            @test FilterGraphAlgorithms.finite_output(adapter, Val(1)) == qualified
            refined = FilterGraphAlgorithms.prepare_validation(adapter, (true,))
            @test (refined.validate_input! isa finite_input_type) == qualified
            qualified || @test refined === adapter
            @test refined.kernels === adapter.kernels
            @test refined.kernels.validate! === adapter.kernels.validate!
            @test FilterGraphAlgorithms.preserves_inputs(refined) == (qualified && !in_place)
            @test FilterGraphAlgorithms.finite_output(refined, Val(1)) == qualified
            case = merge(original, (; adapter=refined))
            arrays = storage(case, false)
            addresses = map(pointer, arrays)
            launch = execution(case, recorded)
            expected = (zeros(Float32, n),)
            # A finite upstream proof cannot remove intermediate/state checks.
            residual = fill(0.125f0, n)
            residual[end] = 2f0
            @test check_step(case, cpu, expected, (residual,), launch) === BiquadCascadeNonFiniteOutput
            check_history(case, cpu, false)
            residual[end] = 0.125f0
            @test check_step(case, cpu, expected, (residual,), launch) === nothing
            check_history(case, cpu, false)
            # An earlier rejection makes the proof conditional on a status that
            # is no longer successful; this Node must neither stage nor adopt.
            fill!(only(case.inputs), NaN32)
            held_output = Array(only(case.outputs))
            held_state = Array(refined.committed_state)
            held_candidate = Array(refined.candidate_state)
            fill!(case.status, Int32(0x103))
            enqueue!(refined)
            synchronize()
            @test only(Array(case.status)) == Int32(0x103)
            @test isequal(Array(only(case.outputs)), held_output)
            @test Array(refined.committed_state) == held_state
            @test Array(refined.candidate_state) == held_candidate
            reset!(refined); reset!(cpu); synchronize()
            fill!(only(case.inputs), 0.125f0)
            @test check_step(case, cpu, expected, (residual,), launch) === nothing
            check_history(case, cpu, false)
            @test map(pointer, arrays) == addresses
            if recorded
                launch(); synchronize()
                gpi2_control_replay_bytes(launch); synchronize()
                @test gpi2_control_replay_bytes(launch) == 0
                synchronize()
            end
            # The unrefined adapter retains input-before-output failure priority.
            unrefined = capture(cascade(n, coefficients), n, false; in_place)
            rejected = fill(0.125f0, n)
            rejected[1] = 2f0
            rejected[end] = NaN32
            unrefined_cpu = cascade(n, coefficients)
            step!() = (fill!(unrefined.status, Int32(0)); enqueue!(unrefined.adapter); nothing)
            @test check_step(unrefined, unrefined_cpu, expected, (rejected,), step!) === BiquadCascadeNonFiniteInput
            check_history(unrefined, unrefined_cpu, false)
        end
        # Shifted input/output views invalidate preserved-input proofs too.
        backing = device_zeros(Float32, n+1)
        status = device_zeros(Int32, 1)
        shifted = prepare_capture(CompleteTransition(), device, cascade(n, coefficients),
            (view(backing, 2:n+1),), (view(backing, 1:n),), status, Int32(2))
        @test !FilterGraphAlgorithms.preserves_inputs(shifted)
    end

    @testset "GPI2 shifted input/output views stage before publication" begin
        n = 513
        coefficients = reshape(Float32[0.5,0.25,0.125,-0.3,0.05], 5, 1, 1)
        for controller in (false, true)
            cpu = controller ? integrator(n) : cascade(n, coefficients)
            source = controller ? integrator(n) : cascade(n, coefficients)
            backing = device_zeros(Float32, n+1)
            input = view(backing, 1:n)
            output = view(backing, 2:n+1)
            inputs = controller ? (input, device_zeros(Float32, n)) : (input,)
            outputs = controller ? (output, device_zeros(Float32, n)) : (output,)
            status = device_zeros(Int32, 1)
            adapter = prepare_capture(CompleteTransition(), device, source,
                outputs, inputs, status, Int32(2))
            case = (; adapter, inputs, outputs, status)
            expected = map(_ -> zeros(Float32, n), outputs)
            launch() = (fill!(status, Int32(0)); enqueue!(adapter); nothing)
            for frame in 1:3
                residual = Float32[mod(mode+frame, 11)/32 for mode in 1:n]
                samples = controller ? (residual, fill(frame == 1 ? NaN32 : 0.025f0, n)) : (residual,)
                @test check_step(case, cpu, expected, samples, launch) === nothing
                check_history(case, cpu, controller)
            end
        end
    end

    @testset "GPI2 biquad fused output cancellation preserves finite result" begin
        coefficients = reshape(Float32[floatmax(Float32),0,0,0,0], 5, 1, 1)
        cpu, source = cascade(1, coefficients), cascade(1, coefficients)
        cpu.workspace.committed_state[1] = -floatmax(Float32)
        source.workspace.committed_state[1] = -floatmax(Float32)
        case = capture(source, 1, false)
        expected = (zeros(Float32, 1),)
        launch() = (fill!(case.status, Int32(0)); enqueue!(case.adapter); nothing)
        @test check_step(case, cpu, expected, (Float32[2],), launch) === nothing
        @test only(Array(case.outputs[1])) == floatmax(Float32)
        check_history(case, cpu, false)
    end

    @testset "GPI2 integrator fused cancellation preserves CPU grouping" begin
        for feedback_case in (false, true)
            function seeded()
                prepared = integrator(1; gain=feedback_case ? 0f0 : floatmax(Float32))
                prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
                    (PropertyAssignment(1, 1f0), PropertyAssignment(2, floatmax(Float32))))
                if feedback_case
                    prepared.workspace.committed_output[1] = floatmax(Float32)
                    prepared.workspace.committed_has_output = true
                else
                    prepared.workspace.committed_state[1] = -floatmax(Float32)
                end
                return prepared
            end
            cpu, case = seeded(), capture(seeded(), 1, true)
            expected = (zeros(Float32, 1), zeros(Float32, 1))
            launch() = (fill!(case.status, Int32(0)); enqueue!(case.adapter); nothing)
            samples = (Float32[feedback_case ? 0 : 2], Float32[feedback_case ? 2 : NaN])
            @test check_step(case, cpu, expected, samples, launch) === nothing
            @test only(Array(case.outputs[1])) == (feedback_case ? -floatmax(Float32) : floatmax(Float32))
            check_history(case, cpu, true)
        end
    end

    @testset "GPI2 integrator open hold and cold buffer checks" begin
        n = 5
        cpu = integrator(n)
        expected = (zeros(Float32, n), zeros(Float32, n))
        @test process!(expected..., cpu, fill(0.5f0, n), fill(NaN32, n)) === nothing
        cpu.plan = FilterGraphAlgorithms.prepare_properties(cpu.plan, (PropertyAssignment(4, false),))
        case = capture(cpu, n, true)
        launch() = (fill!(case.status, Int32(0)); enqueue!(case.adapter); nothing)
        for _ in 1:3
            @test check_step(case, cpu, expected,
                (fill(floatmax(Float32), n), fill(floatmax(Float32), n)), launch) === nothing
            check_history(case, cpu, true)
        end
        @test_throws ArgumentError prepare_capture(CompleteTransition(), device, cpu,
            (case.outputs[1], case.outputs[1]), case.inputs, case.status, Int32(2))
        overlapping = device_zeros(Float32, n+1)
        @test_throws ArgumentError prepare_capture(CompleteTransition(), device, cpu,
            (view(overlapping, 1:n), view(overlapping, 2:n+1)),
            case.inputs, case.status, Int32(2))
        @test_throws DimensionMismatch prepare_capture(CompleteTransition(), device, cpu,
            case.outputs, (device_zeros(Float32, n+1), case.inputs[2]), case.status, Int32(2))
        coefficients = reshape(Float32[1,0,0,0,0], 5, 1, 1)
        source = cascade(n, coefficients)
        @test_throws ArgumentError prepare_capture(CompleteTransition(), device, source,
            (), (case.inputs[1],), case.status, Int32(2))
        @test_throws DimensionMismatch prepare_capture(CompleteTransition(), device, source,
            (device_zeros(Float32, n+1),), (case.inputs[1],), case.status, Int32(2))
        source.workspace.candidate_state = zeros(Float32, 2, n+1, 1)
        @test_throws DimensionMismatch prepare_capture(CompleteTransition(), device, source,
            (case.outputs[1],), (case.inputs[1],), case.status, Int32(2))
    end
end
