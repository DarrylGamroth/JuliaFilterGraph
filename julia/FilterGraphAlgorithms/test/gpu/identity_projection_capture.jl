module IdentityProjectionCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, LinearAlgebra, Test
import FilterGraphAlgorithms: process!, reset!, prepare_capture,
                              enqueue!,
                              complete!

const FGA = FilterGraphAlgorithms
const CUDAX = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)

function controller_graph(matrix)
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ { type = ndarray name = projection label = controller-to-vdm-f32
            config = { controller_command_count = $(size(matrix, 2))
                       active_vdm_count = $(size(matrix, 1)) } } ]
          inputs = [ "projection:controller-command" "projection:controller-to-vdm" ]
          outputs = [ "projection:vdm-command" ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms = (ControllerToVdmF32,))
        replace_parameters!(graph, Symbol("controller-to-vdm") => matrix)
        return graph
    end
end

function controller_adapter(matrix, status = CUDA.zeros(Int32, 1))
    prepared = prepare_algorithm(ControllerToVdmF32,
        ControllerToVdmF32Configuration(controller_command_count = size(matrix, 2),
            active_vdm_count = size(matrix, 1)))
    replace_parameter!(prepared, "controller-to-vdm", matrix)
    input = CUDA.zeros(Float32, size(matrix, 2))
    output = CUDA.zeros(Float32, size(matrix, 1))
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (input,), status, Int32(2))
    return (; prepared, captured, input, output, status)
end

function record!(step!)
    step!()
    CUDA.synchronize()
    return CUDA.instantiate(CUDA.capture(step!))
end

frame_bytes(output, graph, input) = @allocated process!(output, graph, input)
replay_bytes(executable) = @allocated CUDA.launch(executable)

@testset "cooperative reduction preparation rejects incomplete trees" begin
    input, output, status = CUDA.zeros(Float32, 2), CUDA.zeros(Float32, 1), CUDA.zeros(Int32, 1)
    for lanes in (31, 33)
        @test_throws ArgumentError CUDAX._prepare_projection(Val(:ka), CUDA.device(),
            ones(Float32, 1, 2), output, input, lanes)
    end
    for lanes in (32, 64, 128, 256)
        @test CUDAX._prepare_projection(Val(:ka), CUDA.device(),
            ones(Float32, 1, 2), output, input, lanes) isa CUDAX.KAProjection
    end
    sh = prepare_algorithm(ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(reconstructed_count=1, subaperture_count=1,
            initial_reconstructor=nothing, reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1))
    pyramid = prepare_algorithm(PyramidReconstructorF32,
        PyramidReconstructorF32Configuration(reconstructed_count=1, selected_coordinate_count=1,
            initial_reconstructor=nothing, reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1))
    for lanes in (31, 33)
        @test_throws ArgumentError prepare_capture(CompleteTransition(), CUDA.device(), sh,
            (output,), (CUDA.zeros(Float32, 1, 2),), status, Int32(1),
            (; shack_hartmann_reconstruction_lanes=lanes))
        @test_throws ArgumentError prepare_capture(CompleteTransition(), CUDA.device(), pyramid,
            (output,), (CUDA.zeros(Float32, 1, 2),), status, Int32(1),
            (; pyramid_reconstruction_lanes=lanes))
    end
end

@testset "CUDA qualifies shared deformable-mirror validation policy" begin
    case = controller_adapter(Matrix{Float32}(I, 2, 2))
    adapter = case.captured
    backend = CUDAX.KernelAbstractions.get_backend(adapter.frame_status)
    @test FGA._supports_validation_reuse(backend)
    @test FGA.preserves_inputs(adapter)
    @test FGA.finite_output(adapter, Val(1))
    @test FGA.alias_input(adapter, Val(1)) == 1
    @test FGA.prepare_validation(adapter, (false,)) === adapter
    refined = FGA.prepare_validation(adapter, (true,))
    @test refined.validate_input! isa CUDAX._FiniteInput
    @test refined.validate_output! isa CUDAX._FiniteInput
    @test refined.project === adapter.project
    @test refined.input === adapter.input
    @test refined.output === adapter.output
    @test refined.frame_status === adapter.frame_status
end

@testset "CUDA projection preparation dispatches exact structure" begin
    identity = Matrix{Float32}(I, 3, 3)
    almost_identity = copy(identity)
    almost_identity[1, 2] = eps(Float32)
    for (matrix, execution_type) in (
        (identity, CUDAX.IdentityProjection),
        (2identity, CUDAX.CUBLASProjection),
        (almost_identity, CUDAX.CUBLASProjection),
        (Float32[0 1 0; 1 0 0; 0 0 1], CUDAX.SelectionProjection),
        (Float32[0 1 0; 1 0 0], CUDAX.SelectionProjection),
        (Float32[0 1 0; 0 1 0; 1 0 0; 0 0 1], CUDAX.SelectionProjection),
        (Float32[0 1 0; eps(Float32) 0 1], CUDAX.CUBLASProjection),
        (Float32[0 prevfloat(1.0f0) 0; 1 0 0], CUDAX.CUBLASProjection),
        (Float32[1 0.25 0; 0 1 -0.5; 0.75 0 1], CUDAX.CUBLASProjection),
        (zeros(Float32, 3, 3), CUDAX.CUBLASProjection))
        cpu = controller_graph(matrix)
        gpu = prepare_captured_graph(controller_graph(matrix), CUDA.device())
        expected = (var"vdm-command" = zeros(Float32, size(matrix, 1)),)
        actual = deepcopy(expected)
        @test only(gpu.nodes).adapter.project isa execution_type
        @test isconcretetype(typeof(only(gpu.nodes).adapter.project))
        if execution_type === CUDAX.CUBLASProjection
            @test only(gpu.nodes).adapter.project.maxabs === Float64(maximum(abs, matrix))
        end
        for sequence in 1:3
            input = (var"controller-command" =
            Float32[(i - sequence) / 4 for i in 1:size(matrix, 2)],)
            @test process!(actual, gpu, input) == process!(expected, cpu, input)
            @test actual.var"vdm-command" ≈ expected.var"vdm-command"
        end
        input = (var"controller-command" = ones(Float32, size(matrix, 2)),)
        frame_bytes(actual, gpu, input)
        @test frame_bytes(actual, gpu, input) == 0
        @test captured_graph_state(gpu) == CapturedGraphReady
    end
end

function selection_matrix(columns, input_count)
    matrix = zeros(Float32, length(columns), input_count)
    for row in eachindex(columns)
        matrix[row, columns[row]] = 1
    end
    return matrix
end

@testset "CUDA selections preserve bits across tiles, rejection, and reset" begin
    # Neither input nor output is a multiple of the workgroup size. Include
    # repeated columns and columns beyond the first cooperative input tile.
    input_count = 523
    columns = [mod(17row, input_count) + 1 for row in 1:277]
    columns[2] = columns[1]
    columns[end] = input_count
    matrix = selection_matrix(columns, input_count)
    matrix[1, 1] = -0.0f0
    cpu = controller_graph(matrix)
    gpu = prepare_captured_graph(controller_graph(matrix), CUDA.device())
    @test only(gpu.nodes).adapter.project isa CUDAX.SelectionProjection
    @test only(gpu.nodes).adapter.projection === nothing
    values = [reinterpret(Float32,
                  UInt32(mod(i, 4) == 0 ? 0x80000001 :
                         mod(i, 4) == 1 ? 0x80000000 :
                         mod(i, 4) == 2 ? 0x00000001 : 0x00000000))
              for i in 1:input_count]
    expected = (var"vdm-command" = zeros(Float32, length(columns)),)
    actual = deepcopy(expected)
    input = (var"controller-command" = values,)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"vdm-command") ==
          reinterpret(UInt32, values[columns])
    @test reinterpret(UInt32, actual.var"vdm-command") ==
          reinterpret(UInt32, expected.var"vdm-command")
    unselected = findfirst(index -> index ∉ columns, eachindex(values))
    @test !isnothing(unselected)
    for nonfinite in (NaN32, Inf32, -Inf32)
        bad = copy(values)
        bad[unselected] = nonfinite
        rejected = (var"controller-command" = bad,)
        fill!(actual.var"vdm-command", 71)
        fill!(expected.var"vdm-command", 71)
        result = process!(actual, gpu, rejected)
        @test result == process!(expected, cpu, rejected)
        @test !(result isa NamedTuple)
        @test all(==(71), actual.var"vdm-command")
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test reinterpret(UInt32, actual.var"vdm-command") ==
              reinterpret(UInt32, values[columns])
    end
    reset!(cpu)
    reset!(gpu)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"vdm-command") ==
          reinterpret(UInt32, values[columns])
    frame_bytes(actual, gpu, input)
    @test frame_bytes(actual, gpu, input) == 0
end

@testset "CUDA selection snapshots preparation and respects upstream disposition" begin
    matrix = selection_matrix([3, 2, 3], 4)
    case = controller_adapter(matrix)
    fill!(case.prepared.plan.controller_to_vdm, 7)
    step!() = enqueue!(case.captured)
    executable = record!(step!)
    copyto!(case.input, Float32[-0.0, 0.125, -0.25, 0.5])
    fill!(case.status, 0)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test Array(case.output) == Float32[-0.25, 0.125, -0.25]
    @test only(Array(case.status)) == 0
    fill!(case.output, 71)
    fill!(case.status, Int32(0x103))
    fill!(case.input, NaN32)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test only(Array(case.status)) == Int32(0x103)
    @test all(==(71), Array(case.output))
    @test reset!(case.captured) === case.captured
    fill!(case.status, 0)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test complete!(case.captured, only(Array(case.status))) ===
          ControllerToVdmNonFiniteInput
    @test all(==(71), Array(case.output))
    replay_bytes(executable)
    CUDA.synchronize()
    @test replay_bytes(executable) == 0
    CUDA.synchronize()
end

@testset "CUDA identity preserves bit patterns, rejection, and reset" begin
    count = 277
    matrix = Matrix{Float32}(I, count, count)
    # Either zero sign is accepted by the same exact CPU identity classifier.
    matrix[1, 2] = -0.0f0
    cpu = controller_graph(matrix)
    gpu = prepare_captured_graph(controller_graph(matrix), CUDA.device())
    @test only(gpu.nodes).adapter.project isa CUDAX.IdentityProjection
    @test only(gpu.nodes).adapter.projection === nothing
    values = [reinterpret(Float32,
                  UInt32(mod(i, 4) == 0 ? 0x80000001 :
                         mod(i, 4) == 1 ? 0x80000000 :
                         mod(i, 4) == 2 ? 0x00000001 : 0x00000000))
              for i in 1:count]
    expected = (var"vdm-command" = zeros(Float32, count),)
    actual = deepcopy(expected)
    input = (var"controller-command" = values,)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"vdm-command") == reinterpret(UInt32, values)
    @test reinterpret(UInt32, actual.var"vdm-command") ==
          reinterpret(UInt32, expected.var"vdm-command")
    for nonfinite in (NaN32, Inf32, -Inf32)
        bad = copy(values)
        bad[end] = nonfinite
        rejected = (var"controller-command" = bad,)
        fill!(actual.var"vdm-command", 71)
        fill!(expected.var"vdm-command", 71)
        result = process!(actual, gpu, rejected)
        @test result == process!(expected, cpu, rejected)
        @test !(result isa NamedTuple)
        @test all(==(71), actual.var"vdm-command")
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test reinterpret(UInt32, actual.var"vdm-command") == reinterpret(UInt32, values)
    end
    reset!(cpu)
    reset!(gpu)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test reinterpret(UInt32, actual.var"vdm-command") == reinterpret(UInt32, values)
end

@testset "CUDA identity snapshots preparation and respects upstream disposition" begin
    case = controller_adapter(Matrix{Float32}(I, 3, 3))
    # Matrix metadata is consumed during cold preparation, never during replay.
    fill!(case.prepared.plan.controller_to_vdm, 7)
    step!() = enqueue!(case.captured)
    executable = record!(step!)
    copyto!(case.input, Float32[-0.0, 0.125, -0.25])
    fill!(case.status, 0)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, Array(case.input))
    @test only(Array(case.status)) == 0
    fill!(case.output, 71)
    fill!(case.status, Int32(0x103))
    fill!(case.input, NaN32)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test only(Array(case.status)) == Int32(0x103)
    @test all(==(71), Array(case.output))
    @test reset!(case.captured) === case.captured
    fill!(case.status, 0)
    CUDA.launch(executable)
    CUDA.synchronize()
    @test complete!(case.captured, only(Array(case.status))) ===
          ControllerToVdmNonFiniteInput
    @test all(==(71), Array(case.output))
    replay_bytes(executable)
    CUDA.synchronize()
    @test replay_bytes(executable) == 0
    CUDA.synchronize()

    # An intermediate identity must copy nonfinite values without dropping
    # them, so an existing downstream validation sees the arithmetic failure.
    project = case.captured.project
    copyto!(case.input, Float32[Inf, -Inf, reinterpret(Float32, UInt32(0x7fc01234))])
    fill!(case.status, 0)
    copy_step!() = CUDAX._enqueue_projection!(project,
        case.output, case.status, case.input)
    copy_executable = record!(copy_step!)
    CUDA.launch(copy_executable)
    CUDA.synchronize()
    @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, Array(case.input))
end

function feedback_graph(first, second)
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ { type = ndarray name = feedback label = pdm-feedback-to-vdm-f32
            config = { active_vdm_count = $(size(second, 1))
                       full_vdm_count = $(size(first, 1))
                       physical_actuator_count = $(size(first, 2)) } } ]
          inputs = [ "feedback:pdm-constraint-feedback" "feedback:full-to-active"
                     "feedback:pdm-to-vdm" ]
          outputs = [ "feedback:vdm-constraint-feedback" ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms = (PdmFeedbackToVdmF32,))
        replace_parameters!(graph, Symbol("pdm-to-vdm") => first,
            Symbol("full-to-active") => second)
        return graph
    end
end

function feedback_adapter(first, second; overlapping=false)
    prepared = prepare_algorithm(PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(active_vdm_count=size(second, 1),
            full_vdm_count=size(first, 1), physical_actuator_count=size(first, 2)))
    replace_parameter!(prepared, "pdm-to-vdm", first)
    replace_parameter!(prepared, "full-to-active", second)
    if overlapping
        storage = CUDA.zeros(Float32, size(first, 2) + size(second, 1) - 1)
        input = view(storage, 1:size(first, 2))
        output = view(storage, size(first, 2):length(storage))
    else
        input = CUDA.zeros(Float32, size(first, 2))
        output = CUDA.zeros(Float32, size(second, 1))
    end
    status = CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (input,), status, Int32(2))
    return (; prepared, captured, input, output, status)
end

@testset "CUDA structured feedback preserves supplied storage on reset and rejection" begin
    identity = Matrix{Float32}(I, 3, 3)
    selection = selection_matrix([3, 1], 3)
    values = collect(reinterpret(Float32, UInt32[0x80000000, 0x00000001, 0x80000001]))
    for (first, second) in ((identity, identity),
            (identity, selection), (selection, Matrix{Float32}(I, 2, 2)))
        for overlapping in (false, true)
            case = feedback_adapter(first, second; overlapping)
            @test Base.mightalias(case.input, case.output) == overlapping
            @test (case.captured.full_vdm_feedback === case.input ||
                   case.captured.full_vdm_feedback === case.output) == !overlapping
            step!() = enqueue!(case.captured)
            executable = record!(step!)
            expected = zeros(Float32, size(second, 1))
            @test process!(expected, case.prepared, values) === nothing
            copyto!(case.input, values)
            CUDA.launch(executable)
            CUDA.synchronize()
            @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, expected)

            input_before, output_before = Array(case.input), Array(case.output)
            reset!(case.captured)
            CUDA.synchronize()
            @test reinterpret(UInt32, Array(case.input)) == reinterpret(UInt32, input_before)
            @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, output_before)
            if overlapping
                @test all(iszero, Array(case.captured.full_vdm_feedback))
            end

            copyto!(case.input, values)
            output_before = Array(case.output)
            fill!(case.status, Int32(0x103))
            CUDA.launch(executable)
            CUDA.synchronize()
            @test only(Array(case.status)) == Int32(0x103)
            @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, output_before)

            bad = copy(values)
            bad[2] = NaN32 # Unselected by the rectangular map, but still invalid input.
            copyto!(case.input, bad)
            output_before = Array(case.output)
            fill!(case.status, 0)
            CUDA.launch(executable)
            CUDA.synchronize()
            @test complete!(case.captured, only(Array(case.status))) === PdmFeedbackToVdmNonFiniteInput
            @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, output_before)

            copyto!(case.input, values)
            fill!(case.status, 0)
            CUDA.launch(executable)
            CUDA.synchronize()
            @test reinterpret(UInt32, Array(case.output)) == reinterpret(UInt32, expected)
            replay_bytes(executable)
            CUDA.synchronize()
            @test replay_bytes(executable) == 0
            CUDA.synchronize()
        end
    end
end

@testset "CUDA dense feedback retains staging beside either identity" begin
    identity = Matrix{Float32}(I, 3, 3)
    dense = copy(identity)
    dense[1, 2] = eps(Float32)
    for (first, second) in ((dense, identity), (identity, dense))
        case = feedback_adapter(first, second)
        @test case.captured.full_vdm_feedback !== case.input
        @test case.captured.full_vdm_feedback !== case.output
        step!() = enqueue!(case.captured)
        executable = record!(step!)
        # The failed frame must retain the old status-gated copy boundary:
        # a dense tail still reads the preceding private intermediate, while
        # an identity tail leaves supplied output untouched.
        copyto!(case.input, Float32[1, 2, 3])
        fill!(case.output, 71)
        fill!(case.status, Int32(0x103))
        CUDA.launch(executable)
        CUDA.synchronize()
        @test only(Array(case.status)) == Int32(0x103)
        @test Array(case.output) == (first === dense ? fill(71.0f0, 3) : zeros(Float32, 3))
    end
end

@testset "CUDA selection cannot hide unselected overflow in two-stage feedback" begin
    first = Matrix{Float32}(I, 523, 523)
    # Overflow outside the first cooperative input tile and outside every
    # selected column. All output workgroups must still reject the frame.
    first[end, end] = floatmax(Float32)
    columns = [mod(row, 3) + 1 for row in 1:277]
    second = selection_matrix(columns, size(first, 1))
    cpu = feedback_graph(first, second)
    gpu = prepare_captured_graph(feedback_graph(first, second), CUDA.device())
    adapter = only(gpu.nodes).adapter
    @test adapter.project_full isa CUDAX.CUBLASProjection
    @test adapter.project_active isa CUDAX.SelectionProjection
    expected = (var"vdm-constraint-feedback" = fill(71.0f0, length(columns)),)
    actual = deepcopy(expected)
    values = ones(Float32, size(first, 2))
    values[end] = 2
    input = (var"pdm-constraint-feedback" = values,)
    result = process!(actual, gpu, input)
    @test result == process!(expected, cpu, input)
    @test !(result isa NamedTuple)
    @test all(==(71), actual.var"vdm-constraint-feedback")
    @test any(!isfinite, Array(adapter.full_vdm_feedback))
    @test all(!isfinite, Array(adapter.output))
    reset!(cpu)
    reset!(gpu)
    @test all(iszero, Array(adapter.full_vdm_feedback))
    values[end] = 0.125f0
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test actual.var"vdm-constraint-feedback" ≈ expected.var"vdm-constraint-feedback"
    @test all(isfinite, actual.var"vdm-constraint-feedback")
end

@testset "CUDA identity cannot hide overflow in two-stage feedback" begin
    first = Float32[floatmax(Float32) 0 0; 0 1 0; 0 0 1]
    second = Matrix{Float32}(I, 3, 3)
    cpu = feedback_graph(first, second)
    gpu = prepare_captured_graph(feedback_graph(first, second), CUDA.device())
    adapter = only(gpu.nodes).adapter
    @test adapter.project_full isa CUDAX.CUBLASProjection
    @test adapter.project_active isa CUDAX.IdentityProjection
    expected = (var"vdm-constraint-feedback" = fill(71.0f0, 3),)
    actual = deepcopy(expected)
    input = (var"pdm-constraint-feedback" = Float32[2, 1, 1],)
    result = process!(actual, gpu, input)
    @test result == process!(expected, cpu, input)
    @test !(result isa NamedTuple)
    @test all(==(71), actual.var"vdm-constraint-feedback")
    @test any(!isfinite, Array(adapter.full_vdm_feedback))
    reset!(cpu)
    reset!(gpu)
    @test all(iszero, Array(adapter.full_vdm_feedback))
    valid = (var"pdm-constraint-feedback" = Float32[0.125, 1, 1],)
    @test process!(actual, gpu, valid) == process!(expected, cpu, valid)
    @test actual.var"vdm-constraint-feedback" ≈ expected.var"vdm-constraint-feedback"
end

@testset "CUDA identity retains preceding state adoption on later rejection" begin
    prepared = prepare_algorithm(LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(extent = 3, initial_state = 0.0f0,
            input_schema = CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema = CONTROLLER_COMMAND_V1))
    case = controller_adapter(Matrix{Float32}(I, 3, 3))
    residual = CUDA.fill(0.25f0, 3)
    leaky = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (case.input,),
        (residual,), case.status, Int32(1))
    bad = controller_adapter(Matrix{Float32}(I, 3, 3), case.status)
    fill!(bad.input, NaN32)
    function step!()
        fill!(case.status, 0)
        enqueue!(leaky)
        enqueue!(case.captured)
        enqueue!(bad.captured)
        return nothing
    end
    executable = record!(step!)
    reset!(leaky)
    CUDA.synchronize()
    expected = zeros(Float32, 3)
    for _ in 1:3
        @test process!(expected, prepared, fill(0.25f0, 3)) === nothing
        CUDA.launch(executable)
        CUDA.synchronize()
        @test complete!(bad.captured, only(Array(case.status))) ===
              ControllerToVdmNonFiniteInput
        @test Array(case.output) ≈ expected
        @test Array(leaky.committed_state) ≈ expected
    end
    reset!(leaky)
    CUDA.synchronize()
    @test all(iszero, Array(leaky.committed_state))
end


function physical_adapter(active_to_full, physical; aliased=false)
    extent = size(active_to_full, 2)
    prepared = prepare_algorithm(VdmToPdmF32,
        VdmToPdmF32Configuration(active_vdm_count=extent,
            full_vdm_count=size(active_to_full, 1), physical_actuator_count=size(physical, 1)))
    replace_parameter!(prepared, "active-to-full", active_to_full)
    replace_parameter!(prepared, "vdm-to-pdm", physical)
    input = CUDA.zeros(Float32, extent)
    output = aliased ? input : CUDA.zeros(Float32, size(physical, 1))
    roles = CUDA.zeros(UInt8, size(physical, 1))
    status = CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared, (output, roles), (input,), status, Int32(2))
    return (; prepared, captured, input, output, roles, status)
end

@testset "CUDA identity expansion borrows only disjoint projection inputs" begin
    identity = Matrix{Float32}(I, 2, 2)
    physical = Float32[1 2; 3 4]
    for aliased in (false, true)
        case = physical_adapter(identity, physical; aliased)
        @test (case.captured.full_vdm === case.input) == !aliased
        @test (case.captured.complete_vdm.copy! === nothing) == !aliased
        copyto!(case.input, Float32[1, 2])
        reset!(case.captured)
        CUDA.synchronize()
        @test Array(case.input) == Float32[1, 2]
        step!() = (fill!(case.status, 0); enqueue!(case.captured); nothing)
        executable = record!(step!)
        copyto!(case.input, Float32[1, 2])
        CUDA.launch(executable)
        CUDA.synchronize()
        @test Array(case.output) ≈ Float32[5, 11]
        @test only(Array(case.status)) == 0
        @test complete!(case.captured, Int32(0)) === nothing
        CUDA.launch(executable)
        CUDA.synchronize()
        @test replay_bytes(executable) == 0
        CUDA.synchronize()
    end

    case = physical_adapter(identity, physical)
    storage = CUDA.CuArray(Float32[1, 2, 0])
    input, output = view(storage, 1:2), view(storage, 2:3)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), case.prepared, (output, case.roles),
        (input,), case.status, Int32(2))
    @test captured.full_vdm !== input
    enqueue!(captured)
    CUDA.synchronize()
    @test Array(output) ≈ Float32[5, 11]

    almost = copy(identity)
    almost[1, 2] = eps(Float32)
    case = physical_adapter(almost, physical)
    @test case.captured.full_vdm !== case.input
    @test case.captured.complete_vdm isa CUDAX.CUBLASProjection
end

@testset "CUDA borrowed identity retains rejection roles and input on reset" begin
    case = physical_adapter(Matrix{Float32}(I, 2, 2), Float32[1 2; 3 4])
    step!() = (fill!(case.status, 0); enqueue!(case.captured); nothing)
    executable = record!(step!)
    copyto!(case.input, Float32[NaN, 2])
    fill!(case.roles, UInt8(71))
    CUDA.launch(executable)
    CUDA.synchronize()
    @test complete!(case.captured, only(Array(case.status))) === VdmToPdmNonFiniteInput
    @test all(==(UInt8(71)), Array(case.roles))
    copyto!(case.input, Float32[1, 2])
    reset!(case.captured)
    CUDA.synchronize()
    @test Array(case.input) == Float32[1, 2]
    CUDA.launch(executable)
    CUDA.synchronize()
    @test Array(case.output) ≈ Float32[5, 11]
    @test only(Array(case.status)) == 0
end

end # module
