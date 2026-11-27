import FilterGraphAlgorithms:
    _supports_validation_reuse,
    preserves_inputs,
    finite_output,
    prepare_validation,
    PdmFeedbackToVdmF32,
    PdmFeedbackToVdmNonFiniteInput,
    PdmFeedbackToVdmNonFiniteOutput,
    PdmFeedbackToVdmPlan,
    PdmFeedbackToVdmWorkspace,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!

struct PdmFeedbackToVdm{
    ClearScratch,
    ValidateInput,
    ProjectFull,
    ProjectActive,
    ValidateOutput,
    O,
    S,
    F,
    P,
    W,
    I,
}
    clear_scratch!::ClearScratch
    validate_input!::ValidateInput
    project_full::ProjectFull
    project_active::ProjectActive
    validate_output!::ValidateOutput
    output::O
    frame_status::S
    full_to_active::F
    pdm_to_vdm::P
    full_vdm_feedback::W
    input::I
    input_failure::Int32
    output_failure::Int32
end

_capture_feedback_intermediate(device, plan, input, output) =
    _capture_feedback_intermediate(_projection_borrowing(device), device, plan, input, output)
_capture_feedback_intermediate(::Val{false}, device, plan, input, output) =
    _capture_zeros(device, eltype(input), size(plan.pdm_to_vdm, 1))

function _capture_feedback_intermediate(
        ::Val{true}, device, plan::FilterGraphAlgorithms.PdmFeedbackToVdmPlan,
        input, output
)
    if _capture_arrays_overlap(input, output)
        return _capture_zeros(device, eltype(input), size(plan.pdm_to_vdm, 1))
    end
    return _capture_feedback_intermediate(
        plan.pdm_to_vdm_execution, plan.full_to_active_execution,
        device, plan, input, output,
    )
end

function _capture_feedback_intermediate(first, last, device, plan, input, output)
    # Dense cuBLAS work is unpredicated. Keep its original intermediate even
    # beside an identity: forwarding could change supplied output on rejection.
    return _capture_zeros(device, eltype(input), size(plan.pdm_to_vdm, 1))
end

_capture_feedback_intermediate(
    ::FilterGraphAlgorithms.IdentityProjection,
    ::Union{FilterGraphAlgorithms.IdentityProjection,FilterGraphAlgorithms.SelectionProjection},
    device, plan, input, output,
) = input

_capture_feedback_intermediate(
    ::FilterGraphAlgorithms.SelectionProjection, ::FilterGraphAlgorithms.IdentityProjection,
    device, plan, input, output,
) = output

_prepare_active_feedback(device, plan, output, input, lanes) =
    _prepare_projection(device, plan.full_to_active, output, input, lanes)

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        PdmFeedbackToVdmF32,
        <:PdmFeedbackToVdmPlan,
        <:PdmFeedbackToVdmWorkspace,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    length(outputs) == 1 || throw(
        ArgumentError("PdmFeedbackToVdmF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("PdmFeedbackToVdmF32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("pdm-constraint-feedback")),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("vdm-constraint-feedback")),
        output,
    )
    _require_capture_array("PDM constraint-feedback input", input, device)
    _require_capture_array("VDM constraint-feedback output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    full_vdm_feedback = _capture_feedback_intermediate(device, prepared.plan, input, output)
    lanes = isnothing(workgroups) ?
            COOPERATIVE_PROJECTION_DEFAULT_LANES :
            workgroups.projection_lanes
    backend = KernelAbstractions.get_backend(output)
    project_full = _prepare_projection(
        device,
        prepared.plan.pdm_to_vdm,
        full_vdm_feedback,
        input,
        lanes,
    )
    project_active = _prepare_active_feedback(
        device,
        prepared.plan,
        output,
        full_vdm_feedback,
        lanes,
    )
    return PdmFeedbackToVdm(
        _clear_projection_scratch!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        project_full,
        project_active,
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        output,
        frame_status,
        _projection_matrix(project_active),
        _projection_matrix(project_full),
        full_vdm_feedback,
        input,
        _failure_code(node_index, PdmFeedbackToVdmNonFiniteInput),
        _failure_code(node_index, PdmFeedbackToVdmNonFiniteOutput),
    )
end

preflight(::PdmFeedbackToVdm) = nothing

function enqueue!(captured::PdmFeedbackToVdm)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    _enqueue_projection!(
        captured.project_full,
        captured.full_vdm_feedback,
        captured.frame_status,
        captured.input,
    )
    _enqueue_projection!(
        captured.project_active,
        captured.output,
        captured.frame_status,
        captured.full_vdm_feedback,
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
    captured::PdmFeedbackToVdm,
    status::Int32,
)
    status == captured.input_failure && return PdmFeedbackToVdmNonFiniteInput
    status == captured.output_failure && return PdmFeedbackToVdmNonFiniteOutput
    return nothing
end

function reset!(captured::PdmFeedbackToVdm)
    if captured.full_vdm_feedback !== captured.input &&
            captured.full_vdm_feedback !== captured.output
        captured.clear_scratch!(
            captured.full_vdm_feedback;
            ndrange=length(captured.full_vdm_feedback),
        )
    end
    return captured
end


# Qualified backends preserve supplied inputs in their ordered stream.
# Sharing this adapter does not qualify a backend for validation reuse.

preserves_inputs(c::PdmFeedbackToVdm) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::PdmFeedbackToVdm, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::PdmFeedbackToVdm, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return PdmFeedbackToVdm(
        c.clear_scratch!, _FiniteInput(), c.project_full, c.project_active,
        c.validate_output!, c.output, c.frame_status, c.full_to_active,
        c.pdm_to_vdm, c.full_vdm_feedback, c.input, c.input_failure, c.output_failure
    )
end
