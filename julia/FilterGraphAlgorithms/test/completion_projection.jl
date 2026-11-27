module CompletionProjectionTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test

const FGA = FilterGraphAlgorithms
const complete! = FGA._process_vdm_to_pdm!
unsigned(::Type{Float32}) = UInt32
unsigned(::Type{Float64}) = UInt64

function completion(::Type{T}; controlled=[8, 2, 10, 4, 6]) where {T}
    matrix = zeros(T, 10, 5)
    for (column, row) in enumerate(controlled)
        matrix[row, column] = one(T)
    end
    for row in setdiff(1:10, controlled), column in 1:5
        matrix[row, column] = T(row - 2column) / T(17)
    end
    return matrix
end

function allocation_bytes(args...)
    complete!(args...)
    return @allocated complete!(args...)
end

@testset "Compact completion preserves coefficients and workspace ownership" begin
    for T in (Float32, Float64)
        matrix = completion(T)
        plan = VdmToPdmPlan(matrix, Matrix{T}(I, 10, 10))
        execution = plan.active_to_full_execution
        @test execution isa FGA.CompletionProjection{T}
        @test execution.controlled_rows == [8, 2, 10, 4, 6]
        @test execution.extrapolated_rows == [1, 3, 5, 7, 9]
        @test execution.coefficients == matrix[execution.extrapolated_rows, :]
        @test plan.active_to_full == matrix
        @test plan.active_to_full !== matrix
        matrix[1, 1] = T(99)
        @test execution.coefficients[1, 1] != T(99)

        input = T[1, -2, 3, -4, 5] / T(7)
        for physical in (Matrix{T}(I, 10, 10),
                         Matrix{T}(I, 10, 10)[[8, 1, 4], :],
                         fill(T(0.2), 3, 10))
            selected_plan = VdmToPdmPlan(plan.active_to_full, physical)
            workspace = VdmToPdmWorkspace(selected_plan)
            output = zeros(T, size(physical, 1))
            @test complete!(output, workspace, selected_plan, input) === nothing
            @test output ≈ physical * (plan.active_to_full * input)
            @test allocation_bytes(output, workspace, selected_plan, input) == 0
            input_storage, output_storage = zeros(T, 10), fill(T(71), 2length(output))
            input_view = view(input_storage, 1:2:9)
            output_view = view(output_storage, 2:2:2length(output))
            input_view .= input
            @test complete!(output_view, workspace, selected_plan, input_view) === nothing
            @test output_view ≈ output
            @test output_storage[1:2:end] == fill(T(71), length(output))
            @test allocation_bytes(output_view, workspace, selected_plan, input_view) == 0
        end

        first, second = VdmToPdmWorkspace(plan), VdmToPdmWorkspace(plan)
        a, b = zeros(T, 10), zeros(T, 10)
        coefficients = copy(execution.coefficients)
        @test first.full_vdm !== second.full_vdm
        @test first.extrapolated_vdm !== second.extrapolated_vdm
        @test complete!(a, first, plan, input) === nothing
        @test complete!(b, second, plan, -input) === nothing
        @test b ≈ -a
        @test execution.coefficients == coefficients
        @test FGA.reset!(first) === first
        @test all(iszero, first.full_vdm)
        @test all(iszero, first.extrapolated_vdm)

        # The existing input/output alias allowance is safe through full scratch.
        storage = zeros(T, 10)
        view(storage, 1:5) .= input
        @test complete!(storage, first, plan, view(storage, 1:5)) === nothing
        @test storage ≈ a
        special = T[-0.0, nextfloat(zero(T)), -nextfloat(zero(T)), 1, -1]
        @test complete!(a, first, plan, special) === nothing
        @test reinterpret(unsigned(T), a[execution.controlled_rows]) ==
              reinterpret(unsigned(T), special)

        for value in (T(NaN), T(Inf), -T(Inf))
            invalid = copy(input)
            invalid[end] = value
            fill!(a, T(71)); fill!(first.full_vdm, T(72)); fill!(first.extrapolated_vdm, T(73))
            @test complete!(a, first, plan, invalid) === VdmToPdmNonFiniteInput
            @test a == fill(T(71), 10)
            @test first.full_vdm == fill(T(72), 10)
            @test first.extrapolated_vdm == fill(T(73), 5)
            @test allocation_bytes(a, first, plan, invalid) == 0
        end
        @test_throws DimensionMismatch complete!(a, VdmToPdmWorkspace(zeros(T, 10), zeros(T, 4)), plan, input)
        @test_throws ArgumentError complete!(a, first, plan, first.extrapolated_vdm)
        @test_throws ArgumentError complete!(first.full_vdm, first, plan, input)
        alias_workspace = VdmToPdmWorkspace(view(storage, 1:10), view(storage, 1:5))
        @test_throws ArgumentError complete!(a, alias_workspace, plan, input)
        if T === Float32 && sizeof(Int) == 8
            @test_throws ArgumentError complete!(reinterpret(T, execution.controlled_rows), first, plan, input)
            @test_throws ArgumentError complete!(reinterpret(T, execution.extrapolated_rows), first, plan, input)
        end
        short_plan = VdmToPdmPlan(plan.active_to_full, Matrix{T}(I, 10, 10)[1:3, :])
        @test_throws ArgumentError complete!(view(short_plan.active_to_full_execution.coefficients, 1:3, 1),
            VdmToPdmWorkspace(short_plan), short_plan, input)
    end
end

@testset "Exact classification and safe fallbacks" begin
    matrix = completion(Float32)
    missing = copy(matrix); missing[8, 1] = prevfloat(1f0)
    @test VdmToPdmPlan(missing, Matrix{Float32}(I, 10, 10)).active_to_full_execution isa FGA.DenseProjection
    @test VdmToPdmPlan(Float32[1 0; 0 1; 0.2 0.3], Matrix{Float32}(I, 3, 3)).active_to_full_execution isa FGA.DenseProjection
    zero_row = copy(matrix); zero_row[1, :] .= 0
    duplicate = copy(matrix); duplicate[1, :] .= 0; duplicate[1, 1] = 1
    for candidate in (zero_row, duplicate)
        plan = VdmToPdmPlan(candidate, Matrix{Float32}(I, 10, 10))
        @test plan.active_to_full_execution isa FGA.CompletionProjection
        input, output = ones(Float32, 5), zeros(Float32, 10)
        @test complete!(output, VdmToPdmWorkspace(plan), plan, input) === nothing
        @test output ≈ candidate * input
    end
    # Overflow in an extrapolated row must not be hidden by physical selection.
    overflow = copy(matrix); overflow[1, 1] = floatmax(Float32)
    for physical in (Matrix{Float32}(I, 10, 10), Matrix{Float32}(I, 10, 10)[[2, 4], :])
        plan = VdmToPdmPlan(overflow, physical)
        output, input = zeros(Float32, size(physical, 1)), fill(2f0, 5)
        workspace = VdmToPdmWorkspace(plan)
        @test complete!(output, workspace, plan, input) === VdmToPdmNonFiniteOutput
        @test !all(isfinite, output)
        @test allocation_bytes(output, workspace, plan, input) == 0
    end
end


@testset "Compact completion preserves legacy rejection near overflow" begin
    cancellation = vcat(Matrix{Float32}(I, 5, 5), Float32[
        -1 -.5 1 -1 1; 0 -1 1 -.5 1; -.5 1 -2 1 .5;
        -.5 -2 2 -2 1; -.5 -1 -2 -1 .5])
    large_input = (floatmax(Float32) / 4) .* Float32[1, 1, -1, -1, 1]
    amplified = vcat(Matrix{Float32}(I, 5, 5), Float32[
        -0.25236535 -1.7607181 -0.3198552 -1.0050037 0.9295957;
        0.3186941 0.59811974 0.6917372 1.9190879 1.2753632;
        -1.5871141 -0.71483374 0.94019103 -1.9419198 -1.4180245;
        0.28808117 -1.8572052 -0.2346592 -1.518641 1.9075863;
        -1.6377821 -0.0697453 -0.60522294 0.7130325 -1.3770995])
    ordinary_input = Float32[0.6352534, -0.78257155, -0.4822327, 0.5566608, 0.37254834]
    physical = zeros(Float32, 1, 10); physical[1, 9] = 2.1071927f38
    for (matrix, projection, input) in (
        (cancellation, Matrix{Float32}(I, 10, 10), large_input),
        (cancellation, Matrix{Float32}(I, 10, 10)[1:5, :], large_input),
        (amplified, physical, ordinary_input),
    )
        plan = VdmToPdmPlan(matrix, projection)
        @test plan.active_to_full_execution isa FGA.CompletionProjection
        @test maximum(abs, input) > plan.active_to_full_execution.safe_input_limit
        full_reference = matrix * input
        expected = zeros(Float32, size(projection, 1))
        FGA._cpu_project_intermediate!(expected, projection, full_reference, plan.vdm_to_pdm_execution)
        output = similar(expected)
        status = complete!(output, VdmToPdmWorkspace(plan), plan, input)
        @test status === (all(isfinite, expected) ? nothing : VdmToPdmNonFiniteOutput)
        @test isequal(output, expected)
        @test allocation_bytes(output, VdmToPdmWorkspace(plan), plan, input) == 0
    end
    # Exercise the guard's exact boundary without relying on an overflow.
    boundary_plan = VdmToPdmPlan(completion(Float32), Matrix{Float32}(I, 10, 10))
    boundary_workspace = VdmToPdmWorkspace(boundary_plan)
    limit = boundary_plan.active_to_full_execution.safe_input_limit
    for magnitude in (prevfloat(limit), limit, nextfloat(limit))
        fill!(boundary_workspace.extrapolated_vdm, NaN32)
        input, output = fill(magnitude, 5), zeros(Float32, 10)
        @test complete!(output, boundary_workspace, boundary_plan, input) === nothing
        @test output ≈ boundary_plan.active_to_full * input
        @test all(isnan, boundary_workspace.extrapolated_vdm) == (magnitude > limit)
    end
    # A nonfinite Float64 preparation bound means a conservative zero limit.
    enormous = completion(Float64); enormous[1, :] .= floatmax(Float64)
    plan = VdmToPdmPlan(enormous, Matrix{Float64}(I, 10, 10))
    @test iszero(plan.active_to_full_execution.safe_input_limit)
    output = zeros(Float64, 10)
    @test complete!(output, VdmToPdmWorkspace(plan), plan, zeros(Float64, 5)) === nothing
    @test all(iszero, output)
end

@testset "Parameter adoption keeps completion scratch and concrete types" begin
    prepared = prepare_algorithm(VdmToPdmF32,
        VdmToPdmF32Configuration(active_vdm_count=5, full_vdm_count=10, physical_actuator_count=10))
    workspace, original_type = prepared.workspace, typeof(prepared.plan)
    full, extra = workspace.full_vdm, workspace.extrapolated_vdm
    replace_parameter!(prepared, "vdm-to-pdm", Matrix{Float32}(I, 10, 10))
    compact = completion(Float32)
    selection = vcat(Matrix{Float32}(I, 5, 5), Matrix{Float32}(I, 5, 5))
    general = copy(compact); general[8, 1] = 0.9f0
    for (matrix, policy) in ((compact, FGA.CompletionProjection), (selection, FGA.SelectionProjection),
                              (general, FGA.DenseProjection), (completion(Float32; controlled=[1,3,5,7,9]), FGA.CompletionProjection))
        replace_parameter!(prepared, "active-to-full", matrix)
        @test typeof(prepared.plan) === original_type
        @test prepared.plan.active_to_full_execution isa policy
        @test prepared.workspace === workspace
        @test workspace.full_vdm === full
        @test workspace.extrapolated_vdm === extra
        input, output = ones(Float32, 5), zeros(Float32, 10)
        @test FGA.process!(output, prepared, input) === nothing
        @test output ≈ matrix * input
        @test allocation_bytes(output, workspace, prepared.plan, input) == 0
    end
    previous_limit = prepared.plan.active_to_full_execution.safe_input_limit
    replace_parameter!(prepared, "vdm-to-pdm", 1f20 * Matrix{Float32}(I, 10, 10))
    @test prepared.plan.active_to_full_execution.safe_input_limit < previous_limit
    @test typeof(prepared.plan) === original_type
    @test prepared.workspace === workspace
    @test workspace.full_vdm === full
    @test workspace.extrapolated_vdm === extra
end
end
