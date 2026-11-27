module CorrectionLoopStateTests
using Test, FilterGraphAlgorithms
import FilterGraphAlgorithms: commit!, discard!, process!, reset!, prepare_properties,
    prepare_properties!, commit_properties!, discard_properties!, adopt_properties!

bytes(args...) = @allocated process!(args...)

@testset "correction loop open holds correction and mathematical state" begin
    plan = ClosedLoopCorrectionPlan(0.3f0, 0.8f0, 0.6f0, 0.4f0, 2,
        Float32[1 -1], reshape(Float32[0.5, -0.5], 2, 1))
    @test plan.closed
    open_constructor = ClosedLoopCorrectionPlan(0.3f0, 0.8f0, 0.6f0, 0.4f0, 2,
        Float32[1 -1], reshape(Float32[0.5, -0.5], 2, 1); closed=false)
    @test !open_constructor.closed
    @test !FilterGraphAlgorithms._replace_shape_to_hidden(open_constructor, Float32[0 1]).closed
    @test !FilterGraphAlgorithms._replace_hidden_to_shape(open_constructor, zeros(Float32, 2, 1)).closed
    binding = PropertyBinding(plan)
    workspace = ClosedLoopCorrectionWorkspace(plan)
    correction = zeros(Float32, 2); state = similar(correction)
    residual = Float32[1, -2]; feedback = Float32[0.2, -0.4]
    @test process!(correction, state, workspace, active_plan(binding), residual, nothing) === nothing
    commit!(workspace)
    @test process!(correction, state, workspace, active_plan(binding), residual, feedback) === nothing
    commit!(workspace)
    held_correction = copy(correction); held_state = copy(state)
    prepare_properties!(binding, (PropertyAssignment(4, false),))
    @test active_plan(binding).closed
    commit_properties!(binding)
    @test requested_generation(binding) == 1
    @test active_generation(binding) == 0
    @test adopt_properties!(binding)
    @test !active_plan(binding).closed
    @test active_generation(binding) == 1
    residual .= floatmax(Float32); feedback .= floatmax(Float32)
    for _ in 1:3
        @test process!(correction, state, workspace, active_plan(binding), residual, feedback) === nothing
        @test correction == held_correction
        @test state == held_state
        commit!(workspace)
        @test closed_loop_correction_state(workspace) == held_state
        @test closed_loop_correction_last_output(workspace) == held_correction
    end
    bytes(correction, state, workspace, active_plan(binding), residual, feedback)
    @test bytes(correction, state, workspace, active_plan(binding), residual, feedback) == 0
    @test process!(correction, state, workspace, active_plan(binding), Float32[NaN, 0], feedback) === ClosedLoopCorrectionNonFiniteInput
    @test process!(correction, state, workspace, active_plan(binding), residual, Float32[NaN, 0]) === ClosedLoopCorrectionNonFiniteFeedback
    discard!(workspace)
    @test correction == held_correction
    @test closed_loop_correction_state(workspace) == held_state
    @test_throws DimensionMismatch process!(correction, state, workspace, active_plan(binding), Float32[1], feedback)
    @test_throws PropertyTypeError prepare_properties!(binding, (PropertyAssignment(4, 1),))
    prepare_properties!(binding, (PropertyAssignment(4, true),))
    discard_properties!(binding)
    @test !adopt_properties!(binding)
    @test !active_plan(binding).closed
    prepare_properties!(binding, (PropertyAssignment(4, true),))
    commit_properties!(binding); @test adopt_properties!(binding)
    residual .= Float32[0.7, -0.1]; feedback .= Float32[0.1, -0.2]
    realized = held_correction - feedback
    expected_state = held_correction - plan.anti_windup_gain * feedback -
        plan.hidden_mode_gain * plan.hidden_to_shape * (plan.shape_to_hidden * realized)
    @test process!(correction, state, workspace, active_plan(binding), residual, feedback) === nothing
    @test state ≈ expected_state
    @test correction ≈ plan.pole * expected_state + plan.gain * residual
    discard!(workspace)
    @test closed_loop_correction_last_output(workspace) == held_correction
    open_plan = prepare_properties(active_plan(binding), (PropertyAssignment(4, false),))
    reset!(workspace)
    @test process!(correction, state, workspace, open_plan, residual, nothing) === nothing
    @test correction == state == zeros(Float32, 2)
    commit!(workspace)
    @test closed_loop_correction_last_output(workspace) == correction
    @test process!(correction, state, workspace, open_plan, residual, nothing) === ClosedLoopCorrectionMissingFeedback
end

@testset "leaky integrator open retains initial and accumulated state" begin
    plan = LeakyIntegratorPlan(0.3f0, 0.8f0; closed=false)
    workspace = LeakyIntegratorWorkspace(Float32[2, -3])
    output = zeros(Float32, 2); input = fill(floatmax(Float32), 2)
    @test process!(output, workspace, plan, input) === output
    @test output == Float32[2, -3]
    commit!(workspace)
    @test leaky_integrator_state(workspace) == output
    bytes(output, workspace, plan, input)
    @test bytes(output, workspace, plan, input) == 0
    closed = prepare_properties(plan, (PropertyAssignment(2, true),))
    input .= Float32[1, -1]
    @test process!(output, workspace, closed, input) === output
    @test output ≈ Float32[1.9, -2.7]
    commit!(workspace)
    held = copy(output)
    @test process!(output, workspace, plan, input) === output
    @test output == held
    input[1] = NaN32
    @test process!(output, workspace, plan, input) === LeakyIntegratorNonFiniteInput
    discard!(workspace)
    @test leaky_integrator_state(workspace) == held
    reset!(workspace); input .= 1
    @test process!(output, workspace, plan, input) === output
    @test output == Float32[2, -3]
    invalid_workspace = LeakyIntegratorWorkspace(Float32[NaN, 1])
    @test process!(output, invalid_workspace, plan, input) === LeakyIntegratorNonFiniteOutput
    @test output == Float32[2, -3]
end
end
