import FilterGraphAlgorithms:
    _supports_validation_reuse,
    preserves_inputs,
    finite_output,
    prepare_validation,
    PreparedAlgorithm,
    VdmToPdmF32,
    VdmToPdmNonFiniteInput,
    VdmToPdmNonFiniteOutput,
    VdmToPdmPlan,
    complete!,
    enqueue!,
    preflight,
    reset!

struct VdmToPdm{
    ClearScratch,
    ValidateInput,
    CompleteVdm,
    ProjectPdm,
    ValidateOutput,
    O,
    R,
    C,
    S,
    A,
    P,
    F,
    I,
}
    clear_scratch!::ClearScratch
    validate_input!::ValidateInput
    complete_vdm::CompleteVdm
    project_pdm::ProjectPdm
    validate_output!::ValidateOutput
    output::O
    role_output::R
    correction_roles::C
    frame_status::S
    active_to_full::A
    vdm_to_pdm::P
    full_vdm::F
    input::I
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{VdmToPdmF32,<:VdmToPdmPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    length(outputs) == 2 || throw(
        ArgumentError("VdmToPdmF32 capture requires command and correction-role outputs"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("VdmToPdmF32 capture requires one input"),
    )
    output, role_output = outputs
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("vdm-command")),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("requested-pdm-command")),
        output,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("correction-role")),
        role_output,
    )
    _require_capture_array("correction role output", role_output, device)
    _require_capture_array("VDM command input", input, device)
    _require_capture_array("requested PDM command output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    full_vdm = _capture_projection_input(device, prepared.plan.active_to_full, input, output)
    lanes = isnothing(workgroups) ?
            COOPERATIVE_PROJECTION_DEFAULT_LANES :
            workgroups.projection_lanes
    backend = KernelAbstractions.get_backend(output)
    clear_scratch! = _clear_projection_scratch!(
        backend,
        PROJECTION_WORKGROUP_SIZE,
    )
    validate_input! = _validate_projection_values!(
        backend,
        VALIDATION_WORKGROUP_SIZE,
    )
    complete_vdm = _prepare_projection(
        device,
        prepared.plan.active_to_full,
        full_vdm,
        input,
        lanes,
    )
    project_pdm = _prepare_projection(
        device,
        prepared.plan.vdm_to_pdm,
        output,
        full_vdm,
        lanes,
    )
    validate_output! = _validate_projection_values!(
        backend,
        VALIDATION_WORKGROUP_SIZE,
    )
    return VdmToPdm(
        clear_scratch!,
        validate_input!,
        complete_vdm,
        project_pdm,
        validate_output!,
        output,
        role_output,
        _capture_array(device, prepared.plan.correction_roles),
        frame_status,
        _projection_matrix(complete_vdm),
        _projection_matrix(project_pdm),
        full_vdm,
        input,
        _failure_code(node_index, VdmToPdmNonFiniteInput),
        _failure_code(node_index, VdmToPdmNonFiniteOutput),
    )
end

preflight(::VdmToPdm) = nothing

function enqueue!(captured::VdmToPdm)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    _enqueue_projection!(
        captured.complete_vdm,
        captured.full_vdm,
        captured.frame_status,
        captured.input,
    )
    _enqueue_projection!(
        captured.project_pdm,
        captured.output,
        captured.frame_status,
        captured.full_vdm,
    )
    captured.validate_output!(
        captured.frame_status,
        captured.output,
        captured.output_failure,
        captured.role_output,
        captured.correction_roles;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    return nothing
end

function complete!(captured::VdmToPdm, status::Int32)
    status == captured.input_failure && return VdmToPdmNonFiniteInput
    status == captured.output_failure && return VdmToPdmNonFiniteOutput
    return nothing
end

function reset!(captured::VdmToPdm)
    # An exact identity can borrow upstream storage; reset owns only scratch.
    if captured.full_vdm !== captured.input
        captured.clear_scratch!(captured.full_vdm; ndrange=length(captured.full_vdm))
    end
    return captured
end


# Qualified backends preserve supplied inputs in their ordered stream.
# Sharing this adapter does not qualify a backend for validation reuse.

preserves_inputs(c::VdmToPdm) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::VdmToPdm, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::VdmToPdm, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return VdmToPdm(
        c.clear_scratch!, _FiniteInput(), c.complete_vdm, c.project_pdm,
        c.validate_output!, c.output, c.role_output, c.correction_roles, c.frame_status,
        c.active_to_full, c.vdm_to_pdm, c.full_vdm,
        c.input, c.input_failure, c.output_failure
    )
end
