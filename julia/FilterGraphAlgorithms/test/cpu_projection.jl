module CpuProjectionTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test

import FilterGraphAlgorithms: process!

const FGA = FilterGraphAlgorithms

function projection_allocation_bytes(f, args...)
    f(args...)
    return @allocated f(args...)
end

function check_single_projection(::Type{T}, matrix, execution_type) where {T}
    input = T.(1:size(matrix, 2)) ./ T(3)
    expected = matrix * input
    for (constructor, process, field, input_failure) in (
        (
            ControllerToVdmPlan,
            FGA._process_controller_to_vdm!,
            :controller_to_vdm,
            ControllerToVdmNonFiniteInput,
        ),
        (
            VdmFeedbackToControllerPlan,
            FGA._process_vdm_feedback_to_controller!,
            :vdm_to_controller,
            VdmFeedbackToControllerNonFiniteInput,
        ),
    )
        plan = constructor(matrix)
        output = zeros(T, size(matrix, 1))
        @test getproperty(plan, Symbol(field, :_execution)) isa execution_type
        @test getproperty(plan, field) == matrix
        @test getproperty(plan, field) !== matrix
        @test process(output, plan, input) === nothing
        @test output ≈ expected
        @test projection_allocation_bytes(process, output, plan, input) == 0
        @test_throws DimensionMismatch process(output, plan, input[1:end-1])
        @test_throws DimensionMismatch process(output[1:end-1], plan, input)
        @test_throws ArgumentError process(
            view(getproperty(plan, field), :, 1), plan, input,
        )
        if size(matrix, 1) == size(matrix, 2)
            @test_throws ArgumentError process(input, plan, input)
        end

        input_storage = fill(T(99), 2length(input))
        output_storage = fill(T(71), 2length(output))
        input_view = view(input_storage, 1:2:2length(input))
        output_view = view(output_storage, 2:2:2length(output))
        input_view .= input
        @test process(output_view, plan, input_view) === nothing
        @test output_view ≈ expected
        @test output_storage[1:2:end] == fill(T(71), length(output))
        @test projection_allocation_bytes(process, output_view, plan, input_view) == 0

        for nonfinite in (T(NaN), T(Inf), -T(Inf))
            invalid = copy(input)
            invalid[end] = nonfinite
            fill!(output, T(71))
            @test process(output, plan, invalid) === input_failure
            @test output == fill(T(71), length(output))
            @test projection_allocation_bytes(process, output, plan, invalid) == 0
        end
    end
end

function check_two_stage_projection(::Type{T}, first, second) where {T}
    input = T.(1:size(first, 2)) ./ T(3)
    expected = second * (first * input)
    for (plan, workspace, process, scratch_field, first_field, second_field, input_failure) in (
        let plan = VdmToPdmPlan(first, second)
            (plan, VdmToPdmWorkspace(plan), FGA._process_vdm_to_pdm!,
             :full_vdm, :active_to_full, :vdm_to_pdm, VdmToPdmNonFiniteInput)
        end,
        let plan = PdmFeedbackToVdmPlan(second, first)
            (plan, PdmFeedbackToVdmWorkspace(plan), FGA._process_pdm_feedback_to_vdm!,
             :full_vdm_feedback, :pdm_to_vdm, :full_to_active,
             PdmFeedbackToVdmNonFiniteInput)
        end,
    )
        output = zeros(T, size(second, 1))
        @test process(output, workspace, plan, input) === nothing
        @test output ≈ expected
        @test projection_allocation_bytes(process, output, workspace, plan, input) == 0
        @test_throws DimensionMismatch process(output, workspace, plan, input[1:end-1])
        @test_throws DimensionMismatch process(output[1:end-1], workspace, plan, input)
        for field in (first_field, second_field)
            @test getproperty(plan, field) !== (field === first_field ? first : second)
            @test_throws ArgumentError process(
                view(getproperty(plan, field), 1:size(second, 1), 1),
                workspace, plan, input,
            )
        end
        scratch = getproperty(workspace, scratch_field)
        @test_throws ArgumentError process(
            output, workspace, plan, view(scratch, 1:length(input)),
        )
        @test_throws ArgumentError process(
            view(scratch, 1:length(output)), workspace, plan, input,
        )
        if length(output) == length(input)
            in_place = copy(input)
            @test process(in_place, workspace, plan, in_place) === nothing
            @test in_place ≈ expected
            @test projection_allocation_bytes(
                process, in_place, workspace, plan, in_place,
            ) == 0
        end

        input_storage = fill(T(99), 2length(input))
        output_storage = fill(T(71), 2length(output))
        input_view = view(input_storage, 1:2:2length(input))
        output_view = view(output_storage, 2:2:2length(output))
        input_view .= input
        @test process(output_view, workspace, plan, input_view) === nothing
        @test output_view ≈ expected
        @test output_storage[1:2:end] == fill(T(71), length(output))
        @test projection_allocation_bytes(
            process, output_view, workspace, plan, input_view,
        ) == 0

        for nonfinite in (T(NaN), T(Inf), -T(Inf))
            invalid = copy(input)
            invalid[end] = nonfinite
            fill!(output, T(71))
            @test process(output, workspace, plan, invalid) === input_failure
            @test output == fill(T(71), length(output))
            @test projection_allocation_bytes(
                process, output, workspace, plan, invalid,
            ) == 0
        end
    end
end

@testset "CPU projection execution preserves dense numerical contracts" begin
    for T in (Float32, Float64)
        identity = Matrix{T}(I, 3, 3)
        selection = T[0 1 0; 0 1 0; 1 0 0]
        rectangle = T[0 1 0; 1 0 0]
        expansion = T[0 1; 0 1; 1 0]
        dense = T[1 0.25 0; 0 1 -0.5; 0.75 0 1]
        check_single_projection(T, identity, FGA.IdentityProjection)
        check_single_projection(T, selection, FGA.SelectionProjection)
        check_single_projection(T, rectangle, FGA.SelectionProjection)
        check_single_projection(T, expansion, FGA.SelectionProjection)
        check_single_projection(T, dense, FGA.DenseProjection)
        for first in (identity, selection, dense), second in (identity, selection, dense)
            check_two_stage_projection(T, first, second)
        end
        check_two_stage_projection(T, expansion, rectangle)

        # Detection is exact: even the smallest representable extra coefficient
        # requires the original dense multiplication.
        for matrix in (
            T[0 0; 0 1], T[1 1; 0 1], T[-1 0; 0 1],
            T[1 nextfloat(zero(T)); 0 1], T[prevfloat(one(T)) 0; 0 1],
        )
            @test FGA._cpu_projection_execution(matrix) isa FGA.DenseProjection
        end
        owned_source = copy(selection)
        owned_plan = ControllerToVdmPlan(owned_source)
        fill!(owned_source, zero(T))
        @test owned_plan.controller_to_vdm == selection
        @test owned_plan.controller_to_vdm_execution.columns == [2, 2, 1]
    end
end

@testset "CPU identity and selection preserve selected zero signs and subnormals" begin
    for T in (Float32, Float64)
        identity = Matrix{T}(I, 4, 4)
        selection = T[0 0 0 1; 1 0 0 0; 1 0 0 0; 0 0 1 0]
        tiny = nextfloat(zero(T))
        input = T[-0.0, 0.0, tiny, -tiny]
        @test issubnormal(input[3])
        @test issubnormal(input[4])
        for (matrix, columns) in ((identity, [1, 2, 3, 4]), (selection, [4, 1, 1, 3]))
            selected = input[columns]
            dense = matrix * input
            for (plan, process) in (
                (ControllerToVdmPlan(matrix), FGA._process_controller_to_vdm!),
                (VdmFeedbackToControllerPlan(matrix), FGA._process_vdm_feedback_to_controller!),
            )
                output = zeros(T, 4)
                @test process(output, plan, input) === nothing
                # Dense implementations need not produce identical zero signs.
                @test output == dense
                @test isequal(output, selected)
                @test signbit.(output) == signbit.(selected)
                @test projection_allocation_bytes(process, output, plan, input) == 0
            end
            for (first, second) in ((identity, matrix), (matrix, identity))
                dense_two_stage = second * (first * input)
                for (plan, workspace, process) in (
                    let plan = VdmToPdmPlan(first, second)
                        (plan, VdmToPdmWorkspace(plan), FGA._process_vdm_to_pdm!)
                    end,
                    let plan = PdmFeedbackToVdmPlan(second, first)
                        (plan, PdmFeedbackToVdmWorkspace(plan), FGA._process_pdm_feedback_to_vdm!)
                    end,
                )
                    output = zeros(T, 4)
                    @test process(output, workspace, plan, input) === nothing
                    @test output == dense_two_stage
                    @test isequal(output, selected)
                    @test signbit.(output) == signbit.(selected)
                    @test projection_allocation_bytes(process, output, workspace, plan, input) == 0
                end
            end
        end
    end
end

@testset "second-stage CPU selection preserves intermediate overflow rejection" begin
    for T in (Float32, Float64)
        first = T[1 0; 0 floatmax(T)]
        input = T[1, 2]
        scratch_oracle = first * input
        @test isfinite(scratch_oracle[1])
        @test isinf(scratch_oracle[2])
        for second in (T[1 0], T[1 0; 1 0], Matrix{T}(I, 2, 2), T[1 0; 0.5 0])
            expected = second * scratch_oracle
            for (plan, workspace, process, failure) in (
                let plan = VdmToPdmPlan(first, second)
                    (plan, VdmToPdmWorkspace(plan), FGA._process_vdm_to_pdm!,
                     VdmToPdmNonFiniteOutput)
                end,
                let plan = PdmFeedbackToVdmPlan(second, first)
                    (plan, PdmFeedbackToVdmWorkspace(plan), FGA._process_pdm_feedback_to_vdm!,
                     PdmFeedbackToVdmNonFiniteOutput)
                end,
            )
                output = zeros(T, size(second, 1))
                @test process(output, workspace, plan, input) === failure
                @test isequal(output, expected)
                @test projection_allocation_bytes(
                    process, output, workspace, plan, input,
                ) == 0
            end
        end
    end
end

@testset "projection parameter replacement rebuilds execution with a stable plan type" begin
    identity = Matrix{Float32}(I, 3, 3)
    selection = Float32[0 1 0; 0 1 0; 1 0 0]
    dense = copy(identity)
    dense[1, 2] = nextfloat(0.0f0)
    input = Float32[1, 2, 3]
    for (prepared, fields) in (
        (prepare_algorithm(ControllerToVdmF32, ControllerToVdmF32Configuration(
            controller_command_count=3, active_vdm_count=3,
        )), (("controller-to-vdm", :controller_to_vdm),)),
        (prepare_algorithm(VdmFeedbackToControllerF32, VdmFeedbackToControllerF32Configuration(
            controller_command_count=3, active_vdm_count=3,
        )), (("vdm-to-controller", :vdm_to_controller),)),
        (prepare_algorithm(VdmToPdmF32, VdmToPdmF32Configuration(
            active_vdm_count=3, full_vdm_count=3, physical_actuator_count=3,
        )), (("active-to-full", :active_to_full), ("vdm-to-pdm", :vdm_to_pdm))),
        (prepare_algorithm(PdmFeedbackToVdmF32, PdmFeedbackToVdmF32Configuration(
            active_vdm_count=3, full_vdm_count=3, physical_actuator_count=3,
        )), (("full-to-active", :full_to_active), ("pdm-to-vdm", :pdm_to_vdm))),
    )
        plan_type = typeof(prepared.plan)
        for (name, _) in fields
            replace_parameter!(prepared, name, identity)
        end
        for (name, field) in fields
            for (matrix, execution_type) in (
                (identity, FGA.IdentityProjection), (selection, FGA.SelectionProjection),
                (dense, FGA.DenseProjection), (identity, FGA.IdentityProjection),
            )
                @test replace_parameter!(prepared, name, matrix) === prepared
                @test typeof(prepared.plan) === plan_type
                @test getproperty(prepared.plan, field) == matrix
                @test getproperty(prepared.plan, Symbol(field, :_execution)) isa execution_type
                output = zeros(Float32, 3)
                @test process!(output, prepared, input) === nothing
                @test output ≈ matrix * input
                @test projection_allocation_bytes(process!, output, prepared, input) == 0
            end
        end
    end
end

end
