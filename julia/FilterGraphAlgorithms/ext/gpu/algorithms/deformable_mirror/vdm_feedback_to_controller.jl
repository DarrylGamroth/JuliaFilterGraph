import FilterGraphAlgorithms:
    _supports_validation_reuse,
    preserves_inputs,
    finite_output,
    prepare_validation,
    alias_input,
    PreparedAlgorithm,
    VdmFeedbackToControllerF32,
    VdmFeedbackToControllerNonFiniteInput,
    VdmFeedbackToControllerNonFiniteOutput,
    VdmFeedbackToControllerPlan,
    complete!,
    enqueue!,
    preflight,
    reset!

struct EmbeddingProjection{Launch,Columns}
    gather!::Launch
    columns::Columns
end

@kernel function _embedding_projection!(output, frame_status, columns, input)
    row = @index(Global, Linear)
    if row <= length(output) && @inbounds(iszero(frame_status[1]))
        @inbounds column = columns[row]
        value = iszero(column) ? 0f0 : @inbounds(input[column])
        # Dense feedback sums start at +0. Normalize both zero signs while
        # preserving the bit patterns of nonzero subnormals.
        @inbounds output[row] = iszero(reinterpret(UInt32, value) & 0x7fffffff) ? 0f0 : value
    end
end

function _prepare_feedback_projection(device, plan, output, input, lanes)
    return _prepare_feedback_projection(
        plan.vdm_to_controller_execution, device, plan.vdm_to_controller, output, input, lanes,
    )
end

_prepare_feedback_projection(execution, device, matrix, output, input, lanes) =
    _prepare_projection(device, matrix, output, input, lanes)

function _prepare_feedback_projection(
    embedding::FilterGraphAlgorithms.EmbeddingProjection, device,
    matrix::AbstractMatrix{Float32}, output::AbstractArray{Float32}, input::AbstractArray{Float32}, lanes,
)
    if size(matrix) != (length(output), length(input)) || _capture_arrays_overlap(output, input)
        return _prepare_projection(device, matrix, output, input, lanes)
    end
    _require_capture_workgroup(device, PROJECTION_WORKGROUP_SIZE)
    columns = zeros(Int32, length(output))
    for (column, row) in pairs(embedding.rows)
        columns[row] = Int32(column)
    end
    backend = KernelAbstractions.get_backend(output)
    return EmbeddingProjection(
        _embedding_projection!(backend, PROJECTION_WORKGROUP_SIZE),
        _capture_array(device, columns),
    )
end

function _enqueue_projection!(projection::EmbeddingProjection, output, frame_status, input)
    projection.gather!(output, frame_status, projection.columns, input; ndrange=length(output))
    return nothing
end

_projection_matrix(::EmbeddingProjection) = nothing

# The gather copies values from a proven finite input and inserts +0.
_feedback_output_validation(project, validate!) = validate!
_feedback_output_validation(::EmbeddingProjection, validate!) = _FiniteInput()

struct VdmFeedbackToController{
    ValidateInput,
    Project,
    ValidateOutput,
    O,
    S,
    P,
    I,
}
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
    prepared::PreparedAlgorithm{
        VdmFeedbackToControllerF32,
        <:VdmFeedbackToControllerPlan,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    length(outputs) == 1 || throw(
        ArgumentError("VdmFeedbackToControllerF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("VdmFeedbackToControllerF32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("vdm-constraint-feedback")),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("controller-constraint-feedback")),
        output,
    )
    _require_capture_array("VDM constraint-feedback input", input, device)
    _require_capture_array(
        "controller constraint-feedback output",
        output,
        device,
    )
    _require_capture_array("captured frame status", frame_status, device)

    lanes = isnothing(workgroups) ?
            COOPERATIVE_PROJECTION_DEFAULT_LANES :
            workgroups.projection_lanes
    backend = KernelAbstractions.get_backend(output)
    project = _prepare_feedback_projection(device, prepared.plan, output, input, lanes)
    validate! = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    return VdmFeedbackToController(
        validate!,
        project,
        _feedback_output_validation(project, validate!),
        output,
        frame_status,
        _projection_matrix(project),
        input,
        _failure_code(
            node_index,
            VdmFeedbackToControllerNonFiniteInput,
        ),
        _failure_code(
            node_index,
            VdmFeedbackToControllerNonFiniteOutput,
        ),
    )
end

preflight(::VdmFeedbackToController) = nothing

function enqueue!(captured::VdmFeedbackToController)
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
    captured::VdmFeedbackToController,
    status::Int32,
)
    status == captured.input_failure &&
        return VdmFeedbackToControllerNonFiniteInput
    status == captured.output_failure &&
        return VdmFeedbackToControllerNonFiniteOutput
    return nothing
end

reset!(captured::VdmFeedbackToController) = captured


# Qualified backends preserve supplied inputs in their ordered stream.
# Sharing this adapter does not qualify a backend for validation reuse.

preserves_inputs(c::VdmFeedbackToController) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness follows from the embedding or the retained runtime validation.

finite_output(c::VdmFeedbackToController, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

alias_input(c::VdmFeedbackToController, index::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) ?
    alias_input(c.project, index) : nothing

function prepare_validation(c::VdmFeedbackToController, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return VdmFeedbackToController(
        _FiniteInput(), c.project, _identity_validation(c), c.output,
        c.frame_status, c.projection, c.input, c.input_failure, c.output_failure
    )
end

# An exact identity of an already finite buffer cannot overflow. Dense and
# selection projections retain their output scans, including unselected inputs.

_identity_validation(c::VdmFeedbackToController) =
    _identity_validation(c.project, c.validate_output!)
