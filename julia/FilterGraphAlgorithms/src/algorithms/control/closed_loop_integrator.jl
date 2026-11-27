"""
Feedback-aware leaky integration without hidden-mode projection ports.

Uses [`ClosedLoopCorrectionPlan`](@ref) with zero hidden modes and the same
numerical update, state, failure, reset and property contracts. For correction
`c`, residual `e`, and preceding constraint feedback `b`, the update is
`cₙ = pole * (cₙ₋₁ - anti_windup_gain * bₙ₋₁) + gain * eₙ`.
No preceding feedback is used before the first successful output. The inherited
hidden-mode-gain property has no effect because no hidden modes are present.
The runtime `closed` property defaults to `true`; `false` holds both correction
and mathematical state while publishing each successful frame. Closing resumes
this recurrence from the held correction, without resetting state.
"""
@algorithm ClosedLoopIntegratorF32 begin
    label = "closed-loop-integrator-f32"
    failure = ClosedLoopCorrectionFailure

    """Declare the positive controller-coordinate state length."""
    @configuration ClosedLoopIntegratorF32Configuration begin
        state_length::Int => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(:residual_error, Float32, (configuration.state_length,), CONTROLLER_RESIDUAL_ERROR_V1;
            name="residual-error"),
        input(:constraint_feedback, Float32, (configuration.state_length,), CONTROLLER_CONSTRAINT_FEEDBACK_V1;
            name="constraint-feedback"),
        output(:correction, Float32, (configuration.state_length,), CONTROLLER_COMMAND_V1;
            metadata_from=:residual_error),
        output(:controller_state, Float32, (configuration.state_length,), CONTROLLER_STATE_V1;
            name="controller-state", metadata_from=:residual_error),
    )

    prepare(configuration) = begin
        0 < configuration.state_length <= typemax(UInt32) ÷ sizeof(Float32) ||
            throw(ArgumentError("controller state byte extent must be positive and fit UInt32"))
        n = configuration.state_length
        plan = ClosedLoopCorrectionPlan(1f0, 0f0, 0f0, 0f0, n,
            zeros(Float32, 0, n), zeros(Float32, n, 0))
        (plan=plan, workspace=ClosedLoopCorrectionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        feedback = isnothing(closed_loop_correction_last_output(workspace)) ? nothing : constraint_feedback
        process!(correction, controller_state, workspace, plan, residual_error, feedback)
    end
end

"""
Feedback-aware integration gated by a Boolean measurement-validity vector.
All validity elements must be true to update. Otherwise publish the held
correction and state, without reading residuals or applying leakage or delayed
feedback. A held frame still produces a command: its newly computed physical
constraint feedback replaces the preceding feedback at the owner's ordinary
delay boundary. Only that latest preceding feedback is consumed on the next
valid update; skipped-frame feedback is neither accumulated nor replayed.
"""
@algorithm ValidityGatedClosedLoopIntegratorF32 begin
    label = "validity-gated-closed-loop-integrator-f32"
    failure = ClosedLoopCorrectionFailure

    @configuration ValidityGatedClosedLoopIntegratorF32Configuration begin
        state_length::Int => graph_rebuild
        validity_length::Int => graph_rebuild
        validity_schema::String => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(:residual_error, Float32, (configuration.state_length,), CONTROLLER_RESIDUAL_ERROR_V1;
            name="residual-error"),
        input(:constraint_feedback, Float32, (configuration.state_length,), CONTROLLER_CONSTRAINT_FEEDBACK_V1;
            name="constraint-feedback"),
        input(:measurement_validity, Bool, (configuration.validity_length,), configuration.validity_schema;
            name="measurement-validity"),
        output(:correction, Float32, (configuration.state_length,), CONTROLLER_COMMAND_V1;
            metadata_from=:residual_error),
        output(:controller_state, Float32, (configuration.state_length,), CONTROLLER_STATE_V1;
            name="controller-state", metadata_from=:residual_error),
    )

    prepare(configuration) = begin
        0 < configuration.state_length <= typemax(UInt32) ÷ sizeof(Float32) ||
            throw(ArgumentError("controller state byte extent must be positive and fit UInt32"))
        0 < configuration.validity_length <= typemax(UInt32) ||
            throw(ArgumentError("measurement validity byte extent must be positive and fit UInt32"))
        isempty(configuration.validity_schema) && throw(ArgumentError("measurement validity schema must be nonempty"))
        n = configuration.state_length
        plan = ClosedLoopCorrectionPlan(1f0, 0f0, 0f0, 0f0, n,
            zeros(Float32, 0, n), zeros(Float32, n, 0))
        (plan=plan, workspace=ClosedLoopCorrectionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        if all(measurement_validity)
            feedback = isnothing(closed_loop_correction_last_output(workspace)) ? nothing : constraint_feedback
            process!(correction, controller_state, workspace, plan, residual_error, feedback)
        else
            (_closed_loop_correction_output_aliases_owned(correction, workspace, plan) ||
             _closed_loop_correction_output_aliases_owned(controller_state, workspace, plan) ||
             Base.mightalias(correction, controller_state)) && throw(ArgumentError(
                "closed-loop outputs must not overlap each other or prepared storage"))
            discard!(workspace)
            copyto!(correction, workspace.candidate_output)
            copyto!(controller_state, workspace.candidate_state)
            workspace.candidate_has_output = true
            nothing
        end
    end
end
