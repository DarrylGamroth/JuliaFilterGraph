import FilterGraphAlgorithms:
    DenseReconstructionNonFiniteInput,
    DenseReconstructionNonFiniteOutput,
    DenseReconstructorPlan,
    DenseVectorReconstructorF32,
    PreparedAlgorithm,
    SccReconstructorF32,
    _supports_validation_reuse,
    complete!,
    enqueue!,
    preflight,
    reset!

struct DenseVectorReconstruction{
    ValidateInput,
    Project,
    ValidateOutput,
    O,
    S,
    R,
    I,
}
    validate_input!::ValidateInput
    reconstruct::Project
    validate_output!::ValidateOutput
    output::O
    frame_status::S
    reconstructor::R
    input::I
    input_failure::Int32
    output_failure::Int32
end


_prepare_dense_vector_projection(device, matrix, output, input, _status, lanes) =
    _prepare_projection(device, matrix, output, input, lanes)

function _prepare_dense_vector_reconstruction(
    device,
    prepared::PreparedAlgorithm{D,<:DenseReconstructorPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups,
    algorithm_name::String,
    input_name::String,
    output_name::String,
    input_port::Symbol,
    output_port::Symbol,
) where {D}
    length(outputs) == 1 || throw(
        ArgumentError("$algorithm_name capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("$algorithm_name capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, input_port), input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, output_port), output,
    )
    _require_capture_array(input_name, input, device)
    _require_capture_array(output_name, output, device)
    _require_capture_array("captured frame status", frame_status, device)

    lanes = isnothing(workgroups) ?
            COOPERATIVE_PROJECTION_DEFAULT_LANES :
            workgroups.projection_lanes
    backend = KernelAbstractions.get_backend(output)
    reconstruct = _prepare_dense_vector_projection(
        device,
        prepared.plan.matrix,
        output,
        input,
        frame_status,
        lanes,
    )
    return DenseVectorReconstruction(
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        reconstruct,
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        output,
        frame_status,
        _projection_matrix(reconstruct),
        input,
        _failure_code(
            node_index,
            DenseReconstructionNonFiniteInput,
        ),
        _failure_code(
            node_index,
            DenseReconstructionNonFiniteOutput,
        ),
    )
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{SccReconstructorF32,<:DenseReconstructorPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    return _prepare_dense_vector_reconstruction(
        device,
        prepared,
        outputs,
        inputs,
        frame_status,
        node_index,
        workgroups,
        "SccReconstructorF32",
        "SCC selected-residual input",
        "SCC controller-residual output",
        Symbol("selected-residuals"),
        Symbol("controller-residual-error"),
    )
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{DenseVectorReconstructorF32,<:DenseReconstructorPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    return _prepare_dense_vector_reconstruction(
        device,
        prepared,
        outputs,
        inputs,
        frame_status,
        node_index,
        workgroups,
        "DenseVectorReconstructorF32",
        "dense-vector input",
        "dense-vector output",
        Symbol("ordered-input"),
        :output,
    )
end

preflight(::DenseVectorReconstruction) = nothing

enqueue!(captured::DenseVectorReconstruction) =
    _enqueue_dense_vector_reconstruction!(captured.reconstruct, captured)

function _enqueue_dense_vector_reconstruction!(projection, captured::DenseVectorReconstruction)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    _enqueue_projection!(
        projection,
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
    captured::DenseVectorReconstruction,
    status::Int32,
)
    status == captured.input_failure &&
        return DenseReconstructionNonFiniteInput
    status == captured.output_failure &&
        return DenseReconstructionNonFiniteOutput
    return nothing
end

reset!(captured::DenseVectorReconstruction) = captured

function _without_dense_input_scan(c::DenseVectorReconstruction)
    return DenseVectorReconstruction(
        _FiniteInput(), c.reconstruct, c.validate_output!, c.output, c.frame_status,
        c.reconstructor, c.input, c.input_failure, c.output_failure,
    )
end

# The retained output validator gates publication even when GEMV can overflow.
_dense_output_is_finite(c::DenseVectorReconstruction) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))
