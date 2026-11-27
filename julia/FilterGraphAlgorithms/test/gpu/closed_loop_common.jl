using FilterGraphs, KernelAbstractions
import FilterGraphAlgorithms: process!, reset!

function run_closed_loop_capture_tests(device, to_device, device_zeros, synchronize)
    function controller(n, modes; gain=0.5f0, hidden_gain=0.25f0,
            forward=Float32[(-1)^(i+j) / 32 for i in 1:modes, j in 1:n],
            backward=Float32[(2i-j) / 64 for i in 1:n, j in 1:modes])
        prepared = prepare_algorithm(ClosedLoopCorrectionF32,
            ClosedLoopCorrectionF32Configuration(state_length=n, hidden_mode_count=modes))
        replace_parameter!(prepared, "shape-to-hidden", forward)
        replace_parameter!(prepared, "hidden-to-shape", backward)
        prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
            (PropertyAssignment(0, gain), PropertyAssignment(1, 0.75f0),
                PropertyAssignment(2, 0.5f0), PropertyAssignment(3, hidden_gain)))
        return prepared
    end
    function capture(source; in_place=false)
        n = source.plan.state_length
        residual, feedback = to_device(zeros(Float32, n)), device_zeros(Float32, n)
        correction, state = in_place ? (residual, feedback) :
            (device_zeros(Float32, n), device_zeros(Float32, n))
        status = device_zeros(Int32, 1)
        adapter = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), device,
            source, (correction, state), (residual, feedback), status, Int32(2))
        return (; adapter, residual, feedback, correction, state, status)
    end
    function history(case, cpu)
        @test Array(case.adapter.vectors.committed_state) ≈
              cpu.workspace.committed_state rtol=3f-5 atol=3f-6
        @test Array(case.adapter.vectors.committed_output) ≈
              cpu.workspace.committed_output rtol=3f-5 atol=3f-6
        @test only(Array(case.adapter.has_output)) == Int32(cpu.workspace.committed_has_output)
    end
    function step(case, cpu, expected, residual, feedback)
        copyto!(case.residual, residual)
        copyto!(case.feedback, feedback)
        supplied = (Array(case.correction), Array(case.state))
        result = process!(expected..., cpu, residual, feedback)
        fill!(case.status, Int32(0))
        FilterGraphAlgorithms.enqueue!(case.adapter)
        synchronize()
        @test FilterGraphAlgorithms.complete!(case.adapter, only(Array(case.status))) === result
        if isnothing(result)
            @test Array(case.correction) ≈ expected[1] rtol=3f-5 atol=3f-6
            @test Array(case.state) ≈ expected[2] rtol=3f-5 atol=3f-6
        else
            @test isequal((Array(case.correction), Array(case.state)), supplied)
        end
        history(case, cpu)
        return result
    end

    @testset "closed-loop capture preserves delayed state, reset and storage" begin
        for n in (5, 256, 257, 513), modes in (1, 2), in_place in (false, true)
            cpu, source = controller(n, modes), controller(n, modes)
            case = capture(source; in_place)
            @test capture_support(ClosedLoopCorrectionF32, device) isa SupportsCapture
            @test FilterGraphAlgorithms.preflight(case.adapter) === nothing
            reuse = FilterGraphAlgorithms._supports_validation_reuse(
                KernelAbstractions.get_backend(case.status))
            @test FilterGraphAlgorithms.preserves_inputs(case.adapter) == reuse
            @test FilterGraphAlgorithms.finite_output(case.adapter, Val(1)) == reuse
            reuse || @test FilterGraphAlgorithms.prepare_validation(case.adapter, (true, true)) === case.adapter
            arrays = (values(case.adapter.vectors)..., case.adapter.has_output,
                case.correction, case.state, case.status)
            addresses = map(pointer, arrays)
            expected = (zeros(Float32, n), zeros(Float32, n))
            # Captured coefficients remain independent of their source.
            replace_parameter!(source, "shape-to-hidden", zeros(Float32, modes, n))
            replace_parameter!(source, "hidden-to-shape", zeros(Float32, n, modes))
            for frame in 1:4
                residual = Float32[mod(i+frame, 11) / 64 for i in 1:n]
                feedback = frame == 1 ? fill(NaN32, n) :
                    expected[1] .- clamp.(expected[1], -0.0625f0, 0.0625f0)
                @test step(case, cpu, expected, residual, feedback) === nothing
            end
            for lane in unique((min(n, 129), min(n, 256)))
                residual, feedback = fill(0.125f0, n), zeros(Float32, n)
                feedback[lane] = NaN32
                @test step(case, cpu, expected, residual, feedback) === ClosedLoopCorrectionNonFiniteFeedback
                residual[lane] = Inf32
                @test step(case, cpu, expected, residual, feedback) === ClosedLoopCorrectionNonFiniteInput
            end
            @test step(case, cpu, expected, fill(0.0625f0, n), zeros(Float32, n)) === nothing
            supplied = (Array(case.correction), Array(case.state))
            fill!(case.status, Int32(0x103))
            FilterGraphAlgorithms.enqueue!(case.adapter)
            synchronize()
            @test only(Array(case.status)) == Int32(0x103)
            @test isequal((Array(case.correction), Array(case.state)), supplied)
            history(case, cpu)
            @test reset!(case.adapter) === case.adapter
            reset!(cpu)
            synchronize()
            history(case, cpu)
            @test all(iszero, Array(case.adapter.vectors.hidden_modes))
            @test all(iszero, Array(case.adapter.vectors.hidden_in_state))
            @test map(pointer, arrays) == addresses
            @test step(case, cpu, expected, fill(0.125f0, n), fill(NaN32, n)) === nothing
        end
    end

    @testset "cross-wave controller failure joins precede publication" begin
        n = 257
        for lane in (129, 256), modes in (1, 2), in_place in (false, true)
            prepare() = controller(n, modes; gain=floatmax(Float32), hidden_gain=0f0,
                forward=zeros(Float32, modes, n), backward=zeros(Float32, n, modes))
            cpu, case = prepare(), capture(prepare(); in_place)
            expected = (zeros(Float32, n), zeros(Float32, n))
            residual, feedback = fill(0.125f0, n), fill(NaN32, n)
            residual[lane] = 2f0
            @test step(case, cpu, expected, residual, feedback) === ClosedLoopCorrectionNonFiniteOutput
            residual[lane] = 0.125f0
            @test step(case, cpu, expected, residual, feedback) === nothing
            residual[lane] = 2f0
            feedback .= 0f0
            feedback[129] = NaN32
            @test step(case, cpu, expected, residual, feedback) === ClosedLoopCorrectionNonFiniteFeedback
            residual[256] = NaN32
            @test step(case, cpu, expected, residual, feedback) === ClosedLoopCorrectionNonFiniteInput
        end
    end

    @testset "hidden arithmetic overflow is rejected with zero hidden gain" begin
        n = 5
        for modes in (1, 2)
            prepare() = controller(n, modes; gain=1f0, hidden_gain=0f0,
                forward=fill(floatmax(Float32), modes, n), backward=zeros(Float32, n, modes))
            cpu, case = prepare(), capture(prepare())
            expected = (zeros(Float32, n), zeros(Float32, n))
            @test step(case, cpu, expected, ones(Float32, n), zeros(Float32, n)) === nothing
            @test step(case, cpu, expected, ones(Float32, n), zeros(Float32, n)) === ClosedLoopCorrectionNonFiniteOutput
        end
    end
end

function run_closed_loop_graph_tests(device, synchronize)
    function graph()
        mktemp() do path, io
            write(io, """
            filter.graph = {
              nodes = [
                { type = ndarray name = controller label = closed-loop-correction-f32
                  config = { state_length = 257 hidden_mode_count = 1 }
                  props = { gain = 0.5 pole = 0.75 anti-windup-gain = 0.5 hidden-mode-gain = 0.25 } }
                { type = ndarray name = later label = leaky-integrator-f32
                  config = { extent = 257 initial_state = 0.0
                    input_schema = org.example.controller/1 output_schema = org.example.command/1 } }
              ]
              inputs = [ "controller:residual-error" "controller:constraint-feedback"
                "controller:shape-to-hidden" "controller:hidden-to-shape" "later:input" ]
              outputs = [ "controller:correction" "controller:controller-state" "later:output" ]
            }
            """)
            flush(io)
            return prepare_graph(path; algorithms=(ClosedLoopCorrectionF32, LeakyIntegratorF32))
        end
    end
    @testset "native controller graph adopts before downstream rejection" begin
        cpu = graph()
        gpu = prepare_captured_graph(graph(), device)
        expected = (correction=zeros(Float32, 257), var"controller-state"=zeros(Float32, 257),
            output=zeros(Float32, 257))
        actual = deepcopy(expected)
        controller = gpu.nodes[1].adapter
        for frame in 1:4
            input = (var"residual-error"=fill(Float32(frame)/16, 257),
                var"constraint-feedback"=zeros(Float32, 257),
                input=fill(frame == 2 ? NaN32 : 0.25f0, 257))
            held = deepcopy(actual)
            result = process!(expected, cpu, input)
            @test process!(actual, gpu, input) == result
            @test actual == expected
            frame == 2 && @test actual == held
            workspace = cpu.nodes[1].prepared.workspace
            @test Array(controller.vectors.committed_state) ≈ workspace.committed_state
            @test Array(controller.vectors.committed_output) ≈ workspace.committed_output
            @test only(Array(controller.has_output)) == Int32(workspace.committed_has_output)
            @test captured_graph_state(gpu) == CapturedGraphReady
        end
        reset!(gpu)
        synchronize()
        @test all(iszero, Array(controller.vectors.committed_state))
        @test all(iszero, Array(controller.vectors.committed_output))
        @test iszero(only(Array(controller.has_output)))
    end
end
