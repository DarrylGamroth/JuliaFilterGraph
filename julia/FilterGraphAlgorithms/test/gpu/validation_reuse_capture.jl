module ValidationReuseCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, LinearAlgebra, Test
import FilterGraphAlgorithms: process!, reset!, prepare_capture,
                              enqueue!,
                              complete!

const CUDAX = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)

# Closed-loop correction stores its validation callable in a NamedTuple.
skipped_checks(value) = value isa CUDAX._FiniteInput ? 1 : 0
skipped_checks(values::Union{Tuple, NamedTuple}) = sum(skipped_checks, values; init = 0)
function skipped_adapter_checks(adapter)
    return sum(name -> skipped_checks(getfield(adapter, name)),
        fieldnames(typeof(adapter)); init = 0)
end

function projection_graph(controller, physical; integrate = false)
    extent = size(controller, 2)
    active = size(controller, 1)
    prefix = integrate ? """
        { type = ndarray name = integrate label = leaky-integrator-f32
          config = { extent = $extent initial_state = 0.0
            input_schema = $(CONTROLLER_RESIDUAL_ERROR_V1)
            output_schema = $(CONTROLLER_COMMAND_V1) }
          props = { gain = 0.25 pole = 0.5 } }
        """ : ""
    prefix_link = integrate ?
                  "{ output = \"integrate:output\" input = \"project:controller-command\" }" :
                  ""
    input = integrate ? "integrate:input" : "project:controller-command"
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ $prefix
            { type = ndarray name = project label = controller-to-vdm-f32
              config = { controller_command_count = $extent active_vdm_count = $active } }
            { type = ndarray name = physical label = vdm-to-pdm-f32
              config = { active_vdm_count = $active full_vdm_count = $active
                physical_actuator_count = $(size(physical, 1)) } } ]
          links = [ $prefix_link
            { output = "project:vdm-command" input = "physical:vdm-command" } ]
          inputs = [ "$input" "project:controller-to-vdm"
            "physical:active-to-full" "physical:vdm-to-pdm" ]
          outputs = [ "physical:requested-pdm-command" "physical:correction-role" ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms = (ControllerToVdmF32, VdmToPdmF32,
            LeakyIntegratorF32))
        replace_parameters!(graph, Symbol("controller-to-vdm") => controller,
            Symbol("active-to-full") => Matrix{Float32}(I, active, active),
            Symbol("vdm-to-pdm") => physical)
        return graph
    end
end

function outputs(count)
    return (var"requested-pdm-command" = fill(71.0f0, count),
        var"correction-role" = fill(UInt8(71), count))
end
function sentinel!(values)
    fill!(values.var"requested-pdm-command", 71.0f0)
    fill!(values.var"correction-role", UInt8(71))
end
function check_sentinel(values)
    @test all(==(71.0f0), values.var"requested-pdm-command")
    @test all(==(UInt8(71)), values.var"correction-role")
end
frame_bytes(output, graph, input) = @allocated process!(output, graph, input)

function check_reuse(graph)
    project, physical = graph.nodes[(end - 1):end]
    @test !(project.adapter.validate_input! isa CUDAX._FiniteInput)
    @test physical.adapter.validate_input! isa CUDAX._FiniteInput
    @test physical.adapter.input === project.adapter.output
    @test !(project.adapter.validate_output! isa CUDAX._FiniteInput)
    @test !(physical.adapter.validate_output! isa CUDAX._FiniteInput)
    @test sum(node -> skipped_adapter_checks(node.adapter), graph.nodes) == 1
    @test all(node -> isconcretetype(typeof(node.adapter)), graph.nodes)
end

@testset "CUDA linked validation reuse preserves values, attribution and recovery" begin
    identity = Matrix{Float32}(I, 4, 4)
    cpu = projection_graph(identity, identity)
    gpu = prepare_captured_graph(projection_graph(identity, identity), CUDA.device())
    check_reuse(gpu)
    expected, actual = outputs(4), outputs(4)
    bits = UInt32[0x80000000, 0x00000000, 0x00000001, 0x80000001]
    values = collect(reinterpret(Float32, bits))
    input = (var"controller-command" = values,)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"requested-pdm-command") == bits
    @test actual.var"correction-role" == expected.var"correction-role"
    frame_bytes(actual, gpu, input)
    @test frame_bytes(actual, gpu, input) == 0
    for nonfinite in (NaN32, Inf32, -Inf32)
        bad = copy(values)
        bad[end] = nonfinite
        rejected = (var"controller-command" = bad,)
        sentinel!(actual)
        sentinel!(expected)
        @test process!(actual, gpu, rejected) === ControllerToVdmNonFiniteInput
        @test process!(expected, cpu, rejected) === ControllerToVdmNonFiniteInput
        check_sentinel(actual)
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test reinterpret(UInt32, actual.var"requested-pdm-command") == bits
    end
    reset!(cpu)
    reset!(gpu)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"requested-pdm-command") == bits
    @test captured_graph_state(gpu) == CapturedGraphReady
end

@testset "CUDA direct adapter retains external input validation" begin
    prepared = prepare_algorithm(VdmToPdmF32,
        VdmToPdmF32Configuration(active_vdm_count = 3, full_vdm_count = 3,
            physical_actuator_count = 3))
    identity = Matrix{Float32}(I, 3, 3)
    replace_parameter!(prepared, "active-to-full", identity)
    replace_parameter!(prepared, "vdm-to-pdm", identity)
    input, output = CUDA.zeros(Float32, 3), CUDA.zeros(Float32, 3)
    roles, status = CUDA.zeros(UInt8, 3), CUDA.zeros(Int32, 1)
    for workgroups in ((), (nothing,))
        adapter = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (output, roles),
            (input,), status, Int32(2), workgroups...)
        @test skipped_adapter_checks(adapter) == 0
        for nonfinite in (NaN32, Inf32, -Inf32)
            fill!(input, nonfinite)
            fill!(status, 0)
            fill!(output, 71)
            fill!(roles, UInt8(71))
            enqueue!(adapter)
            CUDA.synchronize()
            @test complete!(adapter, only(Array(status))) ===
                  VdmToPdmNonFiniteInput
            @test all(==(71), Array(output))
            @test all(==(UInt8(71)), Array(roles))
        end
    end
end

@testset "CUDA partial residual proof retains closed-loop feedback validation" begin
    prepared = prepare_algorithm(ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(state_length = 3, hidden_mode_count = 1))
    residual, feedback = CUDA.fill(0.25f0, 3), CUDA.zeros(Float32, 3)
    correction, state = CUDA.zeros(Float32, 3), CUDA.zeros(Float32, 3)
    status = CUDA.zeros(Int32, 1)
    original = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (correction, state),
        (residual, feedback), status, Int32(2))
    refined = FilterGraphAlgorithms.prepare_validation(original, (
        true, false))
    @test skipped_adapter_checks(original) == 0
    @test skipped_adapter_checks(refined) == 1
    @test refined.kernels.validate! isa CUDAX._FiniteInput
    for name in keys(original.kernels)
        name === :validate! && continue
        @test getproperty(refined.kernels, name) === getproperty(original.kernels, name)
    end
    for name in fieldnames(typeof(original))
        name === :kernels && continue
        @test getfield(refined, name) === getfield(original, name)
    end
    function step!()
        fill!(status, 0)
        enqueue!(refined)
    end
    # The first successful frame establishes the delayed feedback history.
    step!()
    CUDA.synchronize()
    executable = CUDA.instantiate(CUDA.capture(step!))
    for nonfinite in (NaN32, Inf32, -Inf32)
        fill!(feedback, nonfinite)
        fill!(correction, 71)
        fill!(state, 71)
        committed = Array(refined.vectors.committed_output)
        CUDA.launch(executable)
        CUDA.synchronize()
        @test complete!(refined, only(Array(status))) ===
              ClosedLoopCorrectionNonFiniteFeedback
        @test all(==(71), Array(correction))
        @test all(==(71), Array(state))
        @test Array(refined.vectors.committed_output) == committed
        fill!(feedback, 0)
        CUDA.launch(executable)
        CUDA.synchronize()
        @test only(Array(status)) == 0
        @test Array(correction) == fill(0.25f0, 3)
    end
end

@testset "CUDA validation reuse retains producer overflow failure" begin
    identity = Matrix{Float32}(I, 3, 3)
    overflowing = copy(identity)
    overflowing[1, 1] = floatmax(Float32)
    cpu = projection_graph(overflowing, identity)
    gpu = prepare_captured_graph(projection_graph(overflowing, identity), CUDA.device())
    check_reuse(gpu)
    @test gpu.nodes[1].adapter.project isa CUDAX.CUBLASProjection
    expected, actual = outputs(3), outputs(3)
    rejected = (var"controller-command" = Float32[2, 1, 1],)
    @test process!(actual, gpu, rejected) === ControllerToVdmNonFiniteOutput
    @test process!(expected, cpu, rejected) === ControllerToVdmNonFiniteOutput
    check_sentinel(actual)
    @test any(!isfinite, Array(gpu.nodes[1].adapter.output))
    valid = (var"controller-command" = Float32[0.125, 1, 1],)
    @test process!(actual, gpu, valid) == process!(expected, cpu, valid)
    @test actual.var"requested-pdm-command" ≈ expected.var"requested-pdm-command"
    reset!(gpu)
    reset!(cpu)
    @test process!(actual, gpu, valid) == process!(expected, cpu, valid)
end

@testset "CUDA downstream overflow preserves immediate earlier adoption" begin
    identity = Matrix{Float32}(I, 3, 3)
    overflowing = copy(identity)
    overflowing[1, 1] = floatmax(Float32)
    cpu = projection_graph(identity, overflowing; integrate = true)
    gpu = prepare_captured_graph(projection_graph(identity, overflowing; integrate = true),
        CUDA.device())
    check_reuse(gpu)
    @test !FilterGraphAlgorithms.preserves_inputs(gpu.nodes[1].adapter)
    @test gpu.nodes[3].adapter.project_pdm isa CUDAX.CUBLASProjection
    expected, actual = outputs(3), outputs(3)
    for state in (2.0f0, 3.0f0)
        sentinel!(actual)
        rejected = (input = Float32[8, 0, 0],)
        @test process!(actual, gpu, rejected) === VdmToPdmNonFiniteOutput
        @test process!(expected, cpu, rejected) === VdmToPdmNonFiniteOutput
        check_sentinel(actual)
        @test Array(gpu.nodes[1].adapter.committed_state) == Float32[state, 0, 0]
        @test Array(gpu.nodes[1].adapter.committed_state) ==
              cpu.nodes[1].prepared.workspace.committed_state
    end
    valid = (input = Float32[-6, 0, 0],)
    @test process!(actual, gpu, valid) == process!(expected, cpu, valid)
    @test actual.var"requested-pdm-command" == expected.var"requested-pdm-command"
    reset!(cpu)
    reset!(gpu)
    @test all(iszero, Array(gpu.nodes[1].adapter.committed_state))
    valid = (input = Float32[1, 0, 0],)
    @test process!(actual, gpu, valid) == process!(expected, cpu, valid)
    @test actual.var"requested-pdm-command" ≈ expected.var"requested-pdm-command"
    @test captured_graph_state(gpu) == CapturedGraphReady
end

end # module
