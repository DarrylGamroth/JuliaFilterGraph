import FilterGraphAlgorithms: VectorLinearCombinationF32, VectorLinearCombinationPlan,
    VectorLinearCombinationNonFiniteInput, VectorLinearCombinationNonFiniteOutput,
    prepare_validation

# Match the CPU contract: round the right product before the fused left product/sum.
@inline _vector_combination_value(left_scale, left, right_scale, right) =
    fma(left_scale, left, right_scale * right)

@kernel function _vector_combination_candidate!(candidate, left, right, left_scale, right_scale, status)
    index = @index(Global, Linear)
    if index <= length(candidate) && @inbounds(iszero(status[1]))
        @inbounds candidate[index] = _vector_combination_value(left_scale, left[index], right_scale, right[index])
    end
end

@kernel function _publish_vector_combination!(output, candidate, current, offset, status)
    index = @index(Global, Linear)
    if @inbounds(iszero(status[1]))
        if index <= length(output)
            @inbounds output[index] = candidate[index]
        end
        if current !== nothing && index <= length(current)
            @inbounds current[index] = offset[index]
        end
    end
end

struct _VectorCombinationStages{V,K,P,B}
    validate!::V
    candidate!::K
    publish!::P
    candidate_output::B
end

function _prepare_vector_combination_stages(device, backend, extent, validate!)
    return _VectorCombinationStages(validate!,
        _vector_combination_candidate!(backend, PROJECTION_WORKGROUP_SIZE),
        _publish_vector_combination!(backend, PROJECTION_WORKGROUP_SIZE),
        _capture_array(device, zeros(Float32, extent)))
end

_prepare_vector_combination_output(device, backend, extent, validate!) =
    _prepare_vector_combination_stages(device, backend, extent, validate!)

struct VectorCombination{C,LV,RV,E,O,L,R,S}
    validate_left!::LV
    validate_right!::RV
    execution::E
    output::O
    current_offset::C
    left::L
    right::R
    frame_status::S
    left_scale::Float32
    right_scale::Float32
    input_failure::Int32
    output_failure::Int32
end

function _prepare_vector_combination(device, outputs, left, right, plan, frame_status, node_index;
        right_finite=false)
    _require_capture_array("vector left input", left, device)
    _require_capture_array("vector right input", right, device)
    _require_capture_array("captured frame status", frame_status, device)
    for array in outputs
        _require_capture_array("vector combination output", array, device)
        # Exact input/output identity is supported. Partial device views are
        # excluded at preparation because retained graph pointers must be stable.
        for input in (left, right)
            array === input || !_capture_arrays_overlap(array, input) ||
                throw(ArgumentError("vector combination output must not partially overlap input"))
        end
    end
    current = length(outputs) == 2 ? last(outputs) : nothing
    if current !== nothing
        (_capture_arrays_overlap(outputs...) || _capture_arrays_overlap(current, left) || _capture_arrays_overlap(current, right)) &&
            throw(ArgumentError("current-offset output must not alias command or parameter buffers"))
        _capture_arrays_overlap(first(outputs), right) &&
            throw(ArgumentError("vector offset output must not alias the adopted offset"))
    end
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE;
        shared_memory=VALIDATION_WORKGROUP_SIZE * sizeof(Int32))
    backend = KernelAbstractions.get_backend(left)
    validate! = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    right_validation = right_finite && _supports_validation_reuse(backend) ? _FiniteInput() : validate!
    execution = _prepare_vector_combination_output(device, backend, plan.extent, validate!)
    return VectorCombination(validate!, right_validation, execution,
        first(outputs), current, left, right,
        frame_status, plan.left_scale, plan.right_scale,
        _failure_code(node_index, VectorLinearCombinationNonFiniteInput),
        _failure_code(node_index, VectorLinearCombinationNonFiniteOutput))
end

function _prepare_capture(::CompleteTransition, device,
        prepared::PreparedAlgorithm{VectorLinearCombinationF32, <:VectorLinearCombinationPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == 1 && length(inputs) == 2 ||
        throw(ArgumentError("vector combination capture requires two inputs and one output"))
    for (name, array) in zip((:output, :left, :right), (outputs..., inputs...))
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, name), array)
    end
    return _prepare_vector_combination(device, outputs, inputs..., prepared.plan, frame_status, node_index)
end

preflight(::VectorCombination) = nothing
function enqueue!(c::VectorCombination)
    c.validate_left!(c.frame_status, c.left, c.input_failure; ndrange=VALIDATION_WORKGROUP_SIZE)
    c.validate_right!(c.frame_status, c.right, c.input_failure; ndrange=VALIDATION_WORKGROUP_SIZE)
    return _enqueue_vector_combination_output!(c, c.execution)
end
function _enqueue_vector_combination_output!(c::VectorCombination, execution::_VectorCombinationStages)
    execution.candidate!(execution.candidate_output, c.left, c.right, c.left_scale, c.right_scale,
        c.frame_status; ndrange=length(c.output))
    execution.validate!(c.frame_status, execution.candidate_output, c.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE)
    execution.publish!(c.output, execution.candidate_output, c.current_offset, c.right, c.frame_status;
        ndrange=length(c.output))
    return nothing
end
function complete!(c::VectorCombination, status::Int32)
    status == c.input_failure && return VectorLinearCombinationNonFiniteInput
    status == c.output_failure && return VectorLinearCombinationNonFiniteOutput
    return nothing
end
reset!(c::VectorCombination) = c
function reset!(c::VectorCombination{<:Any,<:Any,<:Any,<:_VectorCombinationStages})
    fill!(c.execution.candidate_output, 0.0f0)
    return c
end
preserves_inputs(c::VectorCombination) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
    !_capture_arrays_overlap(c.output, c.left) && !_capture_arrays_overlap(c.output, c.right)
finite_output(c::VectorCombination, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))
finite_output(c::VectorCombination, ::Val{2}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::VectorCombination, proofs::Tuple{Bool,Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        any(proofs) || return c
    return VectorCombination(
        proofs[1] ? _FiniteInput() : c.validate_left!,
        proofs[2] ? _FiniteInput() : c.validate_right!,
        c.execution, c.output, c.current_offset, c.left, c.right,
        c.frame_status, c.left_scale, c.right_scale,
        c.input_failure, c.output_failure)
end
