import FilterGraphAlgorithms:
    _supports_validation_reuse,
    preserves_inputs,
    finite_output,
    prepare_validation,
    alias_input,
    ControllerToVdmF32,
    ControllerToVdmNonFiniteInput,
    ControllerToVdmNonFiniteOutput,
    ControllerToVdmPlan,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!

struct ControllerToVdm{ValidateInput,Project,ValidateOutput,O,S,P,I}
    validate_input!::ValidateInput
    project::Project
    validate_output!::ValidateOutput
    output::O
    frame_status::S
    projection::P
    input::I
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{ControllerToVdmF32,<:ControllerToVdmPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    length(outputs) == 1 || throw(
        ArgumentError("ControllerToVdmF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("ControllerToVdmF32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("controller-command")),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("vdm-command")),
        output,
    )
    _require_capture_array("controller command input", input, device)
    _require_capture_array("VDM command output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    lanes = isnothing(workgroups) ?
            COOPERATIVE_PROJECTION_DEFAULT_LANES :
            workgroups.projection_lanes
    backend = KernelAbstractions.get_backend(output)
    validate_input! = _validate_projection_values!(
        backend,
        VALIDATION_WORKGROUP_SIZE,
    )
    project = _prepare_projection(
        device,
        prepared.plan.controller_to_vdm,
        output,
        input,
        lanes,
    )
    validate_output! = _validate_projection_values!(
        backend,
        VALIDATION_WORKGROUP_SIZE,
    )
    return ControllerToVdm(
        validate_input!,
        project,
        validate_output!,
        output,
        frame_status,
        _projection_matrix(project),
        input,
        _failure_code(node_index, ControllerToVdmNonFiniteInput),
        _failure_code(node_index, ControllerToVdmNonFiniteOutput),
    )
end

preflight(::ControllerToVdm) = nothing

function enqueue!(captured::ControllerToVdm)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    _enqueue_projection!(
        captured.project,
        captured.output,
        captured.frame_status,
        captured.input,
    )
    captured.validate_output!(
        captured.frame_status,
        captured.output,
        captured.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    return nothing
end

function complete!(
    captured::ControllerToVdm,
    status::Int32,
)
    status == captured.input_failure &&
        return ControllerToVdmNonFiniteInput
    status == captured.output_failure &&
        return ControllerToVdmNonFiniteOutput
    return nothing
end

reset!(captured::ControllerToVdm) = captured


# Qualified backends preserve supplied inputs in their ordered stream.
# Sharing this adapter does not qualify a backend for validation reuse.

preserves_inputs(c::ControllerToVdm) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::ControllerToVdm, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

alias_input(c::ControllerToVdm, index::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) ?
    alias_input(c.project, index) : nothing

function prepare_validation(c::ControllerToVdm, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return ControllerToVdm(
        _FiniteInput(), c.project, _identity_validation(c), c.output,
        c.frame_status, c.projection, c.input, c.input_failure, c.output_failure
    )
end

# An exact identity of an already finite buffer cannot overflow. Dense and
# selection projections retain their output scans, including unselected inputs.

_identity_validation(c::ControllerToVdm) =
    _identity_validation(c.project, c.validate_output!)
