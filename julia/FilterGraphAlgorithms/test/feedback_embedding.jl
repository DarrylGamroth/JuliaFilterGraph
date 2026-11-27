module FeedbackEmbeddingTests

using FilterGraphAlgorithms, LinearAlgebra, Test
import FilterGraphAlgorithms: process!, reset!, commit!
const FGA = FilterGraphAlgorithms

allocated(output, plan, input) = @allocated FGA._process_vdm_feedback_to_controller!(output, plan, input)

function embedding(::Type{T}, rows, m) where {T}
    matrix = zeros(T, m, length(rows))
    for column in eachindex(rows)
        matrix[rows[column], column] = one(T)
    end
    return matrix
end

@testset "exact feedback embedding preserves dense results" begin
    for T in (Float32, Float64)
        matrix = embedding(T, [4, 2], 5)
        plan = VdmFeedbackToControllerPlan(matrix)
        @test nameof(typeof(plan.vdm_to_controller_execution)) == :EmbeddingProjection
        # The generic projection classifier remains independent of feedback embeddings.
        @test FGA._cpu_projection_execution(matrix) isa FGA.DenseProjection
        fill!(matrix, T(7))
        @test plan.vdm_to_controller == embedding(T, [4, 2], 5)
        output, expected = zeros(T, 5), zeros(T, 5)
        old_mode = Base.get_zero_subnormals()
        try
            for flush in (false, true)
                Base.set_zero_subnormals(flush)
                for input in (T[0.2, -0.3], zeros(T, 2), fill(-zero(T), 2),
                    T[nextfloat(zero(T)), -nextfloat(zero(T))],
                    T[floatmax(T), -floatmax(T)])
                    mul!(expected, plan.vdm_to_controller, input)
                    fill!(output, T(NaN))
                    @test FGA._process_vdm_feedback_to_controller!(output, plan, input) === nothing
                    @test reinterpret(UInt8, output) == reinterpret(UInt8, expected)
                    @test allocated(output, plan, input) == 0
                end
            end
        finally
            Base.set_zero_subnormals(old_mode)
        end
        for stride in (2, -2)
            input_storage, output_storage = fill(T(9), 4), fill(T(7), 10)
            input = view(input_storage, stride > 0 ? (1:2:3) : (3:-2:1))
            output = view(output_storage, stride > 0 ? (2:2:10) : (10:-2:2))
            input .= T[0.2, -0.3]
            @test FGA._process_vdm_feedback_to_controller!(output, plan, input) === nothing
            @test output == plan.vdm_to_controller * input
            @test output_storage[1:2:end] == fill(T(7), 5)
            @test allocated(output, plan, input) == 0
        end
        for bad in (T(NaN), T(Inf), -T(Inf))
            fill!(output, T(7))
            @test FGA._process_vdm_feedback_to_controller!(output, plan, T[bad, 1]) === VdmFeedbackToControllerNonFiniteInput
            @test output == fill(T(7), 5)
        end
        input = T[1, 2]
        @test_throws DimensionMismatch FGA._process_vdm_feedback_to_controller!(output, plan, T[1])
        @test_throws DimensionMismatch FGA._process_vdm_feedback_to_controller!(zeros(T, 4), plan, input)
        @test_throws ArgumentError FGA._process_vdm_feedback_to_controller!(output, plan, view(output, 1:2))
        @test_throws ArgumentError FGA._process_vdm_feedback_to_controller!(view(plan.vdm_to_controller, :, 1), plan, input)
    end
    plan = VdmFeedbackToControllerPlan(embedding(Float32, [3, 1], 3))
    rows = plan.vdm_to_controller_execution.rows
    original = copy(rows)
    aliased_output = view(reinterpret(Float32, rows), 1:3)
    @test_throws ArgumentError FGA._process_vdm_feedback_to_controller!(aliased_output, plan, Float32[1, 2])
    @test rows == original
end

@testset "embedding classification is exact in either subnormal mode" begin
    for T in (Float32, Float64)
        matrix = embedding(T, [4, 2], 5)
        variants = (copy(matrix), copy(matrix), copy(matrix), copy(matrix))
        variants[1][1, 1] = -zero(T)
        variants[2][1, 1] = nextfloat(zero(T))
        variants[3][3, 1] = one(T) # A repeated input needs dense summation.
        variants[4][4, 1] = prevfloat(one(T))
        old_mode = Base.get_zero_subnormals()
        try
            for flush in (false, true)
                Base.set_zero_subnormals(flush)
                for candidate in variants
                    @test VdmFeedbackToControllerPlan(candidate).vdm_to_controller_execution isa FGA.DenseProjection
                end
            end
        finally
            Base.set_zero_subnormals(old_mode)
        end
    end
end

@testset "feedback parameter replacement rebuilds execution" begin
    prepared = prepare_algorithm(VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(controller_command_count=5, active_vdm_count=2))
    output, input = zeros(Float32, 5), Float32[-0.5, 0.25]
    for matrix in (embedding(Float32, [4, 2], 5), fill(0.25f0, 5, 2),
        embedding(Float32, [1, 5], 5))
        @test replace_parameter!(prepared, "vdm-to-controller", matrix) === prepared
        expected = matrix * input
        fill!(matrix, 7)
        @test process!(output, prepared, input) === nothing
        @test output == expected
    end
    plan = prepared.plan
    @test_throws ArgumentError replace_parameter!(prepared, "vdm-to-controller", fill(NaN32, 5, 2))
    @test prepared.plan === plan
    @test reset!(prepared) === prepared
    @test process!(output, prepared, input) === nothing
    @test output == Float32[-0.5, 0, 0, 0, 0.25]
end

@testset "coupled physical feedback and clipping preserve controller history" begin
    # Synthetic coupled physical map, with feedback returned into a larger
    # controller space. Both trajectories use the same scientific operators;
    # only the final feedback embedding differs from the dense oracle.
    returned = VdmFeedbackToControllerPlan(embedding(Float32, [1, 3], 4))
    selection = ControllerToVdmPlan(Float32[1 0 0 0; 0 0 1 0])
    physical = Float32[1 0.25; -0.5 1; 0.25 -0.5]
    forward = VdmToPdmPlan(Matrix{Float32}(I, 2, 2), physical)
    backward = PdmFeedbackToVdmPlan(Matrix{Float32}(I, 2, 2), pinv(physical))
    controller = ClosedLoopCorrectionPlan(0.5f0, 0.875f0, 0.5f0, 0.25f0, 4,
        Float32[0.5 0 -0.25 0], reshape(Float32[0.25, 0, 0.5, 0], 4, 1))
    cases = map(1:2) do _
        command = prepare_algorithm(PdmCommandF32, PdmCommandF32Configuration(
            actuator_count=3, command_limits=(-0.125f0, 0.125f0),
            slew_limit=ones(Float32, 3), initial_current=zeros(Float32, 3),
            quantization_origin=nothing, quantization_step=nothing, dither_seed=nothing))
        (; controller=ClosedLoopCorrectionWorkspace(controller), command,
            forward=VdmToPdmWorkspace(forward), backward=PdmFeedbackToVdmWorkspace(backward),
            correction=zeros(Float32, 4), state=zeros(Float32, 4),
            active=zeros(Float32, 2), requested=zeros(Float32, 3), demanded=zeros(Float32, 3),
            pdm_feedback=zeros(Float32, 3), vdm_feedback=zeros(Float32, 2), feedback=zeros(Float32, 4))
    end
    first_history = Vector{Float32}[]
    clipped = false
    for replay in 1:2
        for frame in 1:24
            residual = Float32[1, -0.5, 0.75, -1] .* (frame % 6 < 3 ? 1f0 : -1f0)
            for (index, case) in enumerate(cases)
                @test process!(case.correction, case.state, case.controller, controller,
                    residual, frame == 1 ? nothing : case.feedback) === nothing
                @test commit!(case.controller) === case.controller
                @test FGA._process_controller_to_vdm!(case.active, selection, case.correction) === nothing
                @test FGA._process_vdm_to_pdm!(case.requested, case.forward, forward, case.active) === nothing
                @test process!(case.demanded, case.pdm_feedback, case.command, case.requested) === nothing
                @test FGA._process_pdm_feedback_to_vdm!(case.vdm_feedback, case.backward, backward, case.pdm_feedback) === nothing
                if index == 1
                    @test FGA._process_vdm_feedback_to_controller!(case.feedback, returned, case.vdm_feedback) === nothing
                else
                    mul!(case.feedback, returned.vdm_to_controller, case.vdm_feedback)
                end
                clipped |= any(!iszero, case.pdm_feedback)
            end
            for field in (:correction, :state, :requested, :demanded, :pdm_feedback, :feedback)
                @test reinterpret(UInt32, getproperty(cases[1], field)) == reinterpret(UInt32, getproperty(cases[2], field))
            end
            if replay == 1
                push!(first_history, copy(cases[1].state))
            else
                @test cases[1].state == first_history[frame]
            end
        end
        for case in cases
            reset!(case.controller)
            reset!(case.command)
            fill!(case.feedback, 0)
        end
    end
    @test clipped
end

end
