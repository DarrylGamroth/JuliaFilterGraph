@kernel function _order_pyramid_pupil_pixels!(ordered, frame_status, input)
    column = @index(Global, Linear)
    if column <= length(ordered) && @inbounds(iszero(frame_status[1]))
        selected_count = size(input, 2)
        pupil = div(column - 1, selected_count) + 1
        selected = rem(column - 1, selected_count) + 1
        @inbounds ordered[column] = input[pupil, selected]
    end
end


struct PyramidPupilReconstruction{ValidateInput, Order, Projection, ValidateOutput, O, S, I, V}
    validate_input!::ValidateInput
    order!::Order
    projection::Projection
    validate_output!::ValidateOutput
    output::O
    frame_status::S
    input::I
    ordered_input::V
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
        ::CompleteTransition,
        device,
        prepared::PreparedAlgorithm{
            PyramidPupilReconstructorF32, <:DenseReconstructorPlan, <:DenseReconstructorWorkspace
        },
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32,
        workgroups = nothing
)
    length(outputs) == 1 ||
        throw(ArgumentError("PyramidPupilReconstructorF32 capture requires one output"))
    length(inputs) == 1 ||
        throw(ArgumentError("PyramidPupilReconstructorF32 capture requires one input"))
    output, input = only(outputs), only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("reconstruction-pixels")),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :reconstructed),
        output,
    )
    _require_pupil_array("pyramid reconstruction pixels", input, device)
    _require_capture_array("pyramid reconstructed output", output, device)
    _require_capture_array("captured frame status", frame_status, device)
    _pupil_mightalias(input, output) &&
        throw(ArgumentError("captured pyramid reconstructor input and output must not overlap"))
    backend = KernelAbstractions.get_backend(output)
    ordered_input = _pupil_order_buffer(device, input)
    return PyramidPupilReconstruction(
        _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE),
        _pupil_order_kernel(backend, input),
        _prepare_pupil_projection(
            device, prepared.plan.matrix, output, ordered_input,
            COOPERATIVE_PROJECTION_DEFAULT_LANES
        ),
        _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE),
        output, frame_status, input, ordered_input,
        _failure_code(node_index, DenseReconstructionNonFiniteInput),
        _failure_code(node_index, DenseReconstructionNonFiniteOutput)
    )
end

preflight(::PyramidPupilReconstruction) = nothing

function enqueue!(captured::PyramidPupilReconstruction)
    captured.validate_input!(
        captured.frame_status, captured.input, captured.input_failure;
        ndrange = VALIDATION_WORKGROUP_SIZE
    )
    captured.order!(
        captured.ordered_input, captured.frame_status, captured.input;
        ndrange = length(captured.ordered_input)
    )
    _enqueue_projection!(
        captured.projection, captured.output, captured.frame_status, captured.ordered_input
    )
    captured.validate_output!(
        captured.frame_status, captured.output, captured.output_failure;
        ndrange = VALIDATION_WORKGROUP_SIZE
    )
    return nothing
end

function complete!(captured::PyramidPupilReconstruction, status::Int32)
    status == captured.input_failure && return DenseReconstructionNonFiniteInput
    status == captured.output_failure && return DenseReconstructionNonFiniteOutput
    return nothing
end

reset!(captured::PyramidPupilReconstruction) = captured

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs, finite_output

# Sharing this adapter does not qualify another backend for validation reuse.
# Qualified backends preserve supplied inputs in their ordered stream.

preserves_inputs(c::PyramidPupilReconstruction) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::PyramidPupilReconstruction, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Backend specializations may retain qualified storage optimizations. The
# ordinary pupil ordering above presents the same dense vector to this fallback.
_prepare_pupil_projection(device, matrix, output, input, lanes::Integer) =
    _prepare_projection(device, matrix, output, input, lanes)

# Backend qualification is separate from these shared certificate operations.
import FilterGraphAlgorithms: prepare_validation

function prepare_validation(c::PyramidPupilReconstruction, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return PyramidPupilReconstruction(
        _FiniteInput(), c.order!, c.projection, c.validate_output!,
        c.output, c.frame_status, c.input, c.ordered_input, c.input_failure, c.output_failure
    )
end

struct _BoundedOutput
    bound::_FiniteBound
end

(::_BoundedOutput)(status, values, failure; ndrange) = nothing

_prepare_bounded_pupil_projection(projection, output) = projection

function _prepare_bounded_validation(c::PyramidPupilReconstruction,
        inputs::Tuple{_FiniteBound})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) || return c
    bound = _projection_bound(c.projection, only(inputs), length(c.ordered_input))
    isnothing(bound) && return c
    projection = _prepare_bounded_pupil_projection(c.projection, c.output)
    return PyramidPupilReconstruction(
        c.validate_input!, c.order!, projection, _BoundedOutput(bound),
        c.output, c.frame_status, c.input, c.ordered_input,
        c.input_failure, c.output_failure,
    )
end
