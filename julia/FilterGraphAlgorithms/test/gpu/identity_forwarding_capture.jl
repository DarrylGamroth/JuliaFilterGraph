module IdentityForwardingCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, KernelAbstractions, LinearAlgebra, Test
import FilterGraphAlgorithms: process!, reset!, prepare_capture, enqueue!, complete!

const FGA = FilterGraphAlgorithms
const CUDAX = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)

@algorithm ForwardingTapF32 begin
    label = "forwarding-tap-f32"
    @configuration ForwardingTapF32Configuration begin
        extent::Int => graph_rebuild
        opaque::Bool => construction
    end
    ports(configuration) = (
        input(:signal, Float32, (configuration.extent,), VDM_COMMAND_V1),
        output(:tap, Float32, (configuration.extent,), VDM_COMMAND_V1; metadata_from=:signal),
    )
    prepare(configuration) = (opaque=configuration.opaque,)
    process!(plan) = begin
        copyto!(tap, signal)
        nothing
    end
end

@kernel function tap_copy!(output, input, status)
    index = @index(Global, Linear)
    if index <= length(output) && @inbounds(iszero(status[1]))
        @inbounds output[index] = input[index]
    end
end

struct CapturedTap{Opaque,O,I,S,K}
    output::O
    input::I
    status::S
    copy!::K
end
FGA.capture_support(::Type{ForwardingTapF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, ::CUDA.CuDevice, prepared::FGA.PreparedAlgorithm{ForwardingTapF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    output, input = only(outputs), only(inputs)
    copy! = tap_copy!(KernelAbstractions.get_backend(output), 32)
    return CapturedTap{prepared.plan.opaque,typeof(output),typeof(input),typeof(status),typeof(copy!)}(
        output, input, status, copy!)
end
# Both variants copy only into their output; only this variant is qualified.
FGA.preserves_inputs(::CapturedTap{false}) = true
FGA.preflight(::CapturedTap) = nothing
enqueue!(adapter::CapturedTap) =
    (adapter.copy!(adapter.output, adapter.input, adapter.status; ndrange=length(adapter.output)); nothing)
complete!(::CapturedTap, ::Int32) = nothing
reset!(adapter::CapturedTap) = adapter

"""Prepare an identity projection chain, optionally exporting its producer and a branch."""
function projection_graph(matrix; producer=false, branch=false, opaque=false)
    active, extent = size(matrix)
    prefix = producer ? """
        { type = ndarray name = controller label = closed-loop-correction-f32
          config = { state_length = $extent hidden_mode_count = 1 }
          props = { gain = 0.5 pole = 0.25 anti-windup-gain = 0.5 hidden-mode-gain = 0.0 } }
        """ : ""
    tap = branch ? """
        { type = ndarray name = tap label = forwarding-tap-f32
          config = { extent = $active opaque = $opaque } }
        """ : ""
    upstream = producer ?
        "{ output = \"controller:correction\" input = \"project:controller-command\" }" : ""
    tap_link = branch ? "{ output = \"project:vdm-command\" input = \"tap:signal\" }" : ""
    inputs = producer ? """
        "controller:residual-error" "controller:constraint-feedback"
        "controller:shape-to-hidden" "controller:hidden-to-shape"
        """ : "\"project:controller-command\""
    exported = producer ? "\"controller:correction\"" : ""
    branch_output = branch ? "\"tap:tap\"" : ""
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ $prefix
            { type = ndarray name = project label = controller-to-vdm-f32
              config = { controller_command_count = $extent active_vdm_count = $active } }
            { type = ndarray name = physical label = vdm-to-pdm-f32
              config = { active_vdm_count = $active full_vdm_count = $active physical_actuator_count = $active } }
            $tap ]
          links = [ $upstream $tap_link
            { output = "project:vdm-command" input = "physical:vdm-command" } ]
          inputs = [ $inputs "project:controller-to-vdm" "physical:active-to-full" "physical:vdm-to-pdm" ]
          outputs = [ $exported "project:vdm-command" "physical:requested-pdm-command"
            "physical:correction-role" $branch_output ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms=(ClosedLoopCorrectionF32, ControllerToVdmF32,
            VdmToPdmF32, ForwardingTapF32))
        replace_parameters!(graph, Symbol("controller-to-vdm") => matrix,
            Symbol("active-to-full") => Matrix{Float32}(I, active, active),
            Symbol("vdm-to-pdm") => Matrix{Float32}(I, active, active))
        producer && replace_parameters!(graph,
            Symbol("shape-to-hidden") => zeros(Float32, 1, extent),
            Symbol("hidden-to-shape") => zeros(Float32, extent, 1))
        return graph
    end
end

"""Allocate independent caller outputs, all initialized to publication sentinels."""
buffers(graph) = map(format -> fill(convert(format.element_type, 71), format.shape), graph.output_formats)
project(graph) = only(node.adapter for node in graph.nodes if node.adapter isa CUDAX.ControllerToVdm)
physical(graph) = only(node.adapter for node in graph.nodes if node.adapter isa CUDAX.VdmToPdm)
frame_bytes(outputs, graph, inputs) = @allocated process!(outputs, graph, inputs)

@testset "CUDA forwarded external identity retains bits, validation and borrowed storage" begin
    matrix = Matrix{Float32}(I, 4, 4)
    owner = prepare_captured_graph(projection_graph(matrix), CUDA.device())
    adapter = project(owner)
    @test FGA.alias_input(adapter, Val(1)) == 1
    @test FGA.alias_input(adapter, Val(2)) === nothing
    @test adapter.output === adapter.input === owner.device_inputs.var"controller-command"
    @test physical(owner).input === adapter.output
    @test owner.device_outputs.var"vdm-command" === adapter.input
    @test adapter.project.copy! === nothing
    @test !(adapter.validate_input! isa CUDAX._FiniteInput)
    @test !(adapter.validate_output! isa CUDAX._FiniteInput)
    bits = UInt32[0x80000000, 0x00000000, 0x00000001, 0x80000001]
    samples = collect(reinterpret(Float32, bits))
    device = CUDA.CuArray(samples)
    output = buffers(owner)
    inputs = (var"controller-command"=device,)
    @test process!(output, owner, inputs) isa NamedTuple
    @test reinterpret(UInt32, output.var"vdm-command") == bits
    @test reinterpret(UInt32, output.var"requested-pdm-command") == bits
    frame_bytes(output, owner, inputs)
    @test frame_bytes(output, owner, inputs) == 0
    borrowed = adapter.input
    @test reset!(owner) === owner
    @test reinterpret(UInt32, Array(borrowed)) == bits
    @test adapter.input === borrowed
    for value in (NaN32, Inf32, -Inf32)
        bad = copy(samples)
        bad[end] = value
        for buffer in values(output)
            fill!(buffer, eltype(buffer)(71))
        end
        @test process!(output, owner, (var"controller-command"=bad,)) === ControllerToVdmNonFiniteInput
        @test all(buffer -> all(==(eltype(buffer)(71)), buffer), values(output))
        @test process!(output, owner, inputs) isa NamedTuple
        @test reinterpret(UInt32, output.var"vdm-command") == bits
    end
end

@testset "CUDA finalized aliases feed exports and branching consumers" begin
    matrix = Matrix{Float32}(I, 3, 3)
    for opaque in (false, true)
        cpu = projection_graph(matrix; producer=true, branch=true, opaque)
        owner = prepare_captured_graph(projection_graph(matrix; producer=true, branch=true, opaque), CUDA.device())
        adapter = project(owner)
        producer = only(node.adapter for node in owner.nodes if node.adapter isa CUDAX.ClosedLoopCorrection)
        tap = only(node.adapter for node in owner.nodes if node.adapter isa CapturedTap)
        @test (adapter.output === adapter.input) == !opaque
        @test physical(owner).input === adapter.output
        @test tap.input === adapter.output
        @test owner.device_outputs.correction === producer.outputs[1]
        @test owner.device_outputs.var"vdm-command" === adapter.output
        if !opaque
            @test producer.outputs[1] === adapter.output
            @test adapter.validate_input! isa CUDAX._FiniteInput
            @test adapter.validate_output! isa CUDAX._FiniteInput
            @test adapter.project.copy! === nothing
        end
        expected, actual = buffers(cpu), buffers(owner)
        for sequence in 1:3
            inputs = (var"residual-error"=Float32[sequence, -sequence, 0.25],
                var"constraint-feedback"=zeros(Float32, 3))
            @test process!(actual, owner, inputs) == process!(expected, cpu, inputs)
            for name in keys(actual)
                @test actual[name] ≈ expected[name]
            end
            @test actual.correction == actual.var"vdm-command" == actual.tap
        end
    end
end

@testset "CUDA identity forwarding snapshots parameters and rejects near identities" begin
    identity = Matrix{Float32}(I, 3, 3)
    near = copy(identity)
    near[1, 2] = eps(Float32)
    source = projection_graph(identity)
    original = prepare_captured_graph(source, CUDA.device())
    replace_parameters!(source, Symbol("controller-to-vdm") => near)
    replacement = prepare_captured_graph(source, CUDA.device())
    @test project(original).output === project(original).input
    @test project(replacement).output !== project(replacement).input
    @test FGA.alias_input(project(replacement), Val(1)) === nothing
    @test project(replacement).project isa CUDAX.CUBLASProjection
    expected, actual = buffers(source), buffers(replacement)
    inputs = (var"controller-command"=Float32[1, 2, 3],)
    @test process!(actual, replacement, inputs) == process!(expected, source, inputs)
    @test actual.var"vdm-command" ≈ expected.var"vdm-command"
    old_output = buffers(original)
    @test process!(old_output, original, inputs) isa NamedTuple
    @test old_output.var"vdm-command" == inputs.var"controller-command"
end

"""Prepare a controller-coordinate feedback identity from a VDM-coordinate input."""
function reverse_graph(matrix)
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ { type = ndarray name = reverse label = vdm-feedback-to-controller-f32
            config = { controller_command_count = $(size(matrix, 1)) active_vdm_count = $(size(matrix, 2)) } } ]
          inputs = [ "reverse:vdm-constraint-feedback" "reverse:vdm-to-controller" ]
          outputs = [ "reverse:controller-constraint-feedback" ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms=(VdmFeedbackToControllerF32,))
        replace_parameters!(graph, Symbol("vdm-to-controller") => matrix)
        return graph
    end
end

@testset "CUDA reverse identities preserve borrowed inputs and protected feedback snapshots" begin
    source = reverse_graph(Matrix{Float32}(I, 3, 3))
    owner = prepare_captured_graph(source, CUDA.device())
    adapter = only(owner.nodes).adapter
    @test FGA.alias_input(adapter, Val(1)) == 1
    @test adapter.output === adapter.input
    @test adapter.project.copy! === nothing
    values = Float32[-0.0, reinterpret(Float32, UInt32(1)), -0.25]
    output = buffers(owner)
    @test process!(output, owner, (var"vdm-constraint-feedback"=CUDA.CuArray(values),)) isa NamedTuple
    @test reinterpret(UInt32, only(output)) == reinterpret(UInt32, values)
    @test reset!(owner) === owner
    @test reinterpret(UInt32, Array(adapter.input)) == reinterpret(UInt32, values)
    # Exercise the existing cold planner directly: public metadata rules do not
    # permit exporting this identity when its source is a bound feedback input.
    distinct = ((CUDA.zeros(Float32, 3),),)
    planned = FilterGraphs._captured_alias_outputs(owner.nodes, source.input_routes,
        owner.device_inputs, distinct, (adapter.input,))
    @test only(only(planned)) === only(only(distinct))
    unprotected = FilterGraphs._captured_alias_outputs(owner.nodes, source.input_routes,
        owner.device_inputs, distinct)
    @test only(only(unprotected)) === adapter.input
end

end # module
