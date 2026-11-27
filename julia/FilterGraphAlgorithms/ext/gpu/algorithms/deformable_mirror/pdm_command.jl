import FilterGraphAlgorithms:
                              PdmCommandF32, TipTiltCommandF32,
                              PdmCommandNonFiniteInput,
                              PdmCommandNonFiniteOutput,
                              PdmCommandPlan,
                              PdmCommandWorkspace,
                              PreparedAlgorithm,
                              NoSlew,
                              IndependentSlew,
                              _pdm_current,
                              complete!,
                              enqueue!,
                              preflight,
                              reset!

@kernel function _pdm_command_candidate!(
        demanded,
        constraint_feedback,
        candidate_current,
        frame_status,
        committed_current,
        requested,
        lower_limits,
        upper_limits,
        slew_limits,
        slew
)
    index = @index(Global, Linear)
    if index <= length(demanded) && @inbounds(iszero(frame_status[1]))
        @inbounds begin
            safe_target = clamp(
                requested[index],
                lower_limits[index],
                upper_limits[index]
            )
            current = _pdm_current(slew, committed_current[index], safe_target,
                slew_limits[index])
            candidate_current[index] = current
            value = clamp(current, lower_limits[index], upper_limits[index])
            demanded[index] = value
            constraint_feedback[index] = requested[index] - value
        end
    end
end

@kernel function _validate_pdm_command_output!(
        frame_status,
        demanded,
        constraint_feedback,
        candidate_current,
        failure
)
    local_index = @index(Local, Linear)
    local_failures = @localmem Int32 (VALIDATION_WORKGROUP_SIZE,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    local_failure = Int32(0)
    actuator = Int(local_index)
    while frame_valid && actuator <= length(demanded)
        @inbounds if !isfinite(demanded[actuator]) ||
                     !isfinite(constraint_feedback[actuator]) ||
                     !isfinite(candidate_current[actuator])
            local_failure = Int32(1)
            break
        end
        actuator += VALIDATION_WORKGROUP_SIZE
    end
    @inbounds local_failures[local_index] = local_failure
    @synchronize

    _reduce_validation!(local_failures, Int(local_index), 128)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 64)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 32)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 16)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 8)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 4)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 2)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 1)
    @synchronize

    if frame_valid && local_index == 1 && @inbounds(!iszero(local_failures[1]))
        @inbounds frame_status[1] = failure
    end
end

@kernel function _adopt_pdm_command_state!(
        committed_current,
        candidate_current,
        frame_status
)
    index = @index(Global, Linear)
    if index <= length(committed_current) && @inbounds(iszero(frame_status[1]))
        @inbounds committed_current[index] = candidate_current[index]
    end
end

@kernel function _reset_pdm_command_state!(
        committed_current,
        candidate_current,
        initial_current
)
    index = @index(Global, Linear)
    if index <= length(committed_current)
        @inbounds begin
            initial = initial_current[index]
            committed_current[index] = initial
            candidate_current[index] = initial
        end
    end
end

struct _PdmStages end

_prepare_pdm_execution(device, backend, demanded, feedback, requested) = _PdmStages()

struct PdmCommand{
    ValidateInput,
    Candidate,
    ValidateOutput,
    Adopt,
    Reset,
    D,
    S,
    V,
    Slew,
    Execution
}
    validate_input!::ValidateInput
    candidate!::Candidate
    validate_output!::ValidateOutput
    adopt!::Adopt
    reset!::Reset
    demanded::D
    constraint_feedback::D
    frame_status::S
    committed_current::V
    candidate_current::V
    initial_current::V
    requested::D
    lower_limits::V
    upper_limits::V
    slew_limits::V
    slew::Slew
    input_failure::Int32
    output_failure::Int32
    execution::Execution
end

function _prepare_capture(
        ::CompleteTransition,
        device,
        prepared::PreparedAlgorithm{
            <:Union{PdmCommandF32,TipTiltCommandF32},
            <:PdmCommandPlan{Float32},
            <:PdmCommandWorkspace{Float32}
        },
        outputs::Tuple,
        inputs::Tuple,
        frame_status,
        node_index::Int32
)
    length(outputs) == 2 || throw(
        ArgumentError("command limiter capture requires two outputs"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("command limiter capture requires one input"),
    )
    demanded, constraint_feedback = outputs
    requested = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :requested),
        requested,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :demanded),
        demanded,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("constraint-feedback")),
        constraint_feedback
    )
    _require_capture_array("requested PDM command input", requested, device)
    _require_capture_array("demanded PDM command output", demanded, device)
    _require_capture_array(
        "PDM constraint-feedback output",
        constraint_feedback,
        device
    )
    _require_capture_array("captured frame status", frame_status, device)
    # Captured kernels require disjoint buffers; checked CPU execution stages
    # outputs and separately supports safe input/output aliasing.
    (Base.mightalias(demanded, constraint_feedback) ||
     Base.mightalias(demanded, requested) ||
     Base.mightalias(constraint_feedback, requested)) && throw(
        ArgumentError("captured PDM requested, demanded, and constraint-feedback buffers must not overlap"),
    )

    plan = prepared.plan
    workspace = prepared.workspace
    extent = length(demanded)
    initial_current = _capture_array(device, workspace.initial_current)
    committed_current = _capture_array(device, workspace.committed_current)
    candidate_current = _capture_array(device, workspace.candidate_current)
    lower_limits = _capture_array(device, plan.lower_limits)
    upper_limits = _capture_array(device, plan.upper_limits)
    has_slew = !isnothing(plan.slew_limits)
    slew_limits = _capture_array(
        device,
        has_slew ? plan.slew_limits : zeros(Float32, extent)
    )

    backend = KernelAbstractions.get_backend(demanded)
    return PdmCommand(
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE
        ),
        _pdm_command_candidate!(
            backend,
            PROJECTION_WORKGROUP_SIZE
        ),
        _validate_pdm_command_output!(
            backend,
            VALIDATION_WORKGROUP_SIZE
        ),
        _adopt_pdm_command_state!(
            backend,
            PROJECTION_WORKGROUP_SIZE
        ),
        _reset_pdm_command_state!(
            backend,
            PROJECTION_WORKGROUP_SIZE
        ),
        demanded,
        constraint_feedback,
        frame_status,
        committed_current,
        candidate_current,
        initial_current,
        requested,
        lower_limits,
        upper_limits,
        slew_limits,
        plan.slew,
        _failure_code(node_index, PdmCommandNonFiniteInput),
        _failure_code(node_index, PdmCommandNonFiniteOutput),
        _prepare_pdm_execution(device, backend, demanded, constraint_feedback, requested)
    )
end

preflight(::PdmCommand) = nothing

enqueue!(captured::PdmCommand) = _enqueue_pdm!(captured, captured.execution)

function _enqueue_pdm!(captured::PdmCommand, ::_PdmStages)
    captured.validate_input!(
        captured.frame_status,
        captured.requested,
        captured.input_failure;
        ndrange = VALIDATION_WORKGROUP_SIZE
    )
    captured.candidate!(
        captured.demanded,
        captured.constraint_feedback,
        captured.candidate_current,
        captured.frame_status,
        captured.committed_current,
        captured.requested,
        captured.lower_limits,
        captured.upper_limits,
        captured.slew_limits,
        captured.slew;
        ndrange = length(captured.demanded)
    )
    captured.validate_output!(
        captured.frame_status,
        captured.demanded,
        captured.constraint_feedback,
        captured.candidate_current,
        captured.output_failure;
        ndrange = VALIDATION_WORKGROUP_SIZE
    )
    captured.adopt!(
        captured.committed_current,
        captured.candidate_current,
        captured.frame_status;
        ndrange = length(captured.demanded)
    )
    return nothing
end

function complete!(captured::PdmCommand, status::Int32)
    status == captured.input_failure && return PdmCommandNonFiniteInput
    status == captured.output_failure && return PdmCommandNonFiniteOutput
    return nothing
end

function reset!(captured::PdmCommand)
    captured.reset!(
        captured.committed_current,
        captured.candidate_current,
        captured.initial_current;
        ndrange = length(captured.demanded)
    )
    return captured
end

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs, finite_output, prepare_validation

# Sharing this adapter does not qualify another backend for validation reuse.
# Qualified backends preserve supplied inputs in their ordered stream.

preserves_inputs(c::PdmCommand) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::PdmCommand, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

finite_output(c::PdmCommand, ::Val{2}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::PdmCommand, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return PdmCommand(
        _FiniteInput(), c.candidate!, c.validate_output!,
        c.adopt!, c.reset!, c.demanded, c.constraint_feedback, c.frame_status,
        c.committed_current, c.candidate_current, c.initial_current,
        c.requested, c.lower_limits, c.upper_limits, c.slew_limits,
        c.slew, c.input_failure, c.output_failure,
        c.execution
    )
end
