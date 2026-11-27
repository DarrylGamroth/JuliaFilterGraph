module SparseFeedbackCaptureTests

using CUDA, FilterGraphAlgorithms, KernelAbstractions, LinearAlgebra, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, complete!, reset!
const FGA = FilterGraphAlgorithms
const Ext = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)

function adapter(matrix, sparse_feedback)
    rows, columns = size(matrix)
    prepared = prepare_algorithm(PdmFeedbackToVdmF32, PdmFeedbackToVdmF32Configuration(
        active_vdm_count=rows, full_vdm_count=columns, physical_actuator_count=columns,
        sparse_feedback=sparse_feedback))
    replace_parameter!(prepared, "pdm-to-vdm", Matrix{Float32}(I, columns, columns))
    replace_parameter!(prepared, "full-to-active", matrix)
    input, output, status = CUDA.zeros(Float32, columns), CUDA.zeros(Float32, rows), CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (input,), status, Int32(2))
    return (; prepared, captured, input, output, status)
end

function record(case; leased=true)
    run = () -> enqueue!(case.captured)
    run()
    CUDA.synchronize()
    workspace = CUDA.zeros(UInt8, 4 << 20)
    graph = if leased
        Ext._with_cuda_conditional_workspace(workspace) do
            CUDA.capture(run)
        end
    else
        CUDA.capture(run)
    end
    return (; case..., workspace, graph, executable=CUDA.instantiate(graph))
end

function replay!(case, x, initial_status)
    copyto!(case.input, x)
    fill!(case.output, 71f0)
    fill!(case.status, initial_status)
    CUDA.launch(case.executable)
    CUDA.synchronize()
    status = only(Array(case.status))
    return (; status, failure=complete!(case.captured, status), output=Array(case.output))
end

function run_tests()
    rows, columns = 513, 512
    matrix = fill(1.25f0, rows, columns)
    matrix[end, 2] = -1.25f0  # Cancellation in the final, partial warp.
    matrix[end-1, columns] = 0f0
    matrix[1, columns] = -1.25f0
    dense, bounded = record(adapter(matrix, false)), record(adapter(matrix, true))
    @test dense.captured.project_active isa Ext.CUBLASProjection
    @test bounded.captured.project_active isa Ext.BoundedFeedbackProjection
    execution = bounded.prepared.plan.sparse_execution
    weight = execution.max_input / 4
    function feedback(indices, value=weight)
        x = zeros(Float32, columns)
        x[indices] .= value
        return x
    end
    # The contract includes the supplied output on previous and current failure.
    # Alternating accepted and rejected frames exercises retained scratch.
    for (label, x, status, fallback, failure) in (
            ("empty", feedback(1:0), 0, false, nothing),
            ("one positive, zero coefficient", feedback([columns]), 0, false, nothing),
            ("one negative, zero coefficient", feedback([columns], -weight), 0, true, nothing),
            ("64 distributed", feedback(1:8:columns), 0, false, nothing),
            ("65 distributed", feedback(1:7:449), 0, true, nothing),
            ("65 contiguous", feedback(1:65), 0, true, nothing),
            ("row cancellation", feedback(1:2), 0, true, nothing),
            ("signed zero", feedback([columns], -0f0), 0, true, nothing),
            ("subnormal", feedback([2], nextfloat(0f0)), 0, true, nothing),
            ("below lower bound", feedback([1], prevfloat(execution.min_input)), 0, true, nothing),
            ("above upper bound", feedback([1], nextfloat(execution.max_input)), 0, true, nothing),
            ("overflow", feedback([1], floatmax(Float32)), 0, true, PdmFeedbackToVdmNonFiniteOutput),
            ("recovery", feedback([3]), 0, false, nothing),
            ("NaN", feedback([2], NaN32), 0, true, PdmFeedbackToVdmNonFiniteInput),
            ("Inf", feedback([2], Inf32), 0, true, PdmFeedbackToVdmNonFiniteInput),
            ("upstream failure", feedback([2], -Inf32), 257, true, nothing),
            ("empty after rejection", feedback(1:0), 0, false, nothing))
        @testset "$label" begin
            lhs, rhs = replay!(dense, x, Int32(status)), replay!(bounded, x, Int32(status))
            @test only(Array(bounded.captured.project_active.sparse.fallback)) == fallback
            @test rhs.status == lhs.status
            @test rhs.failure === lhs.failure === failure
            if fallback || count(!iszero, x) <= 1
                @test reinterpret(UInt32, rhs.output) == reinterpret(UInt32, lhs.output)
            else
                @test all(abs.(Float64.(rhs.output) .- lhs.output) .<= 1e-11 .+ 2e-4 .* abs.(lhs.output))
            end
        end
    end
    bare = record(adapter(matrix, true); leased=false)
    flag = bare.captured.project_active.sparse.fallback
    fill!(flag, Int32(-1))
    @test replay!(bare, feedback([columns]), Int32(0)).output ==
          replay!(dense, feedback([columns]), Int32(0)).output
    @test only(Array(flag)) == -1  # Unleased capture uses only dense GEMV.
    foreach(c -> reset!(c.captured), (dense, bounded))
    lhs, rhs = replay!(dense, fill(NaN32, columns), Int32(0)),
               replay!(bounded, fill(NaN32, columns), Int32(0))
    @test lhs.status == rhs.status && !iszero(rhs.status)
    @test isequal(lhs.output, rhs.output) && all(iszero, rhs.output)
    return nothing
end

@testset "CUDA bounded DM feedback and conditional dense GEMV" begin
    if Ext._supports_cuda_conditional()
        old = CUDA.CUDACore.math_mode()
        try
            CUDA.math_mode!(CUDA.PEDANTIC_MATH)
            run_tests()
        finally
            CUDA.math_mode!(old)
        end
    end
end
end
