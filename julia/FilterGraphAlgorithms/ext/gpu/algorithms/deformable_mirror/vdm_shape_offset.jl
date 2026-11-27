import FilterGraphAlgorithms: VdmShapeOffsetF32, VdmShapeOffsetPlan

# Shape offset rejects the earliest coordinate, checking its input before its
# addition. This differs from VectorOffset's complete input scan before sums.
@kernel function _vdm_shape_offset_candidate!(candidate, requested, active_offset,
        status, input_failure, output_failure)
    lane = @index(Local, Linear)
    failures = @localmem Int32 (VALIDATION_WORKGROUP_SIZE,)
    @uniform valid = @inbounds iszero(status[1])
    failure = typemax(Int32)
    index = Int(lane)
    while valid && index <= length(requested)
        @inbounds value = requested[index]
        if !isfinite(value)
            failure = Int32(2index - 1)
            break
        end
        @inbounds addition = FilterGraphAlgorithms._pdm_offset_sum(value, active_offset[index])
        if !isfinite(addition)
            failure = Int32(2index)
            break
        end
        @inbounds candidate[index] = addition
        index += VALIDATION_WORKGROUP_SIZE
    end
    @inbounds failures[lane] = failure
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds failures[lane] = min(failures[lane], failures[lane + stride])
        end
        @synchronize
    end
    if valid && lane == 1 && @inbounds(failures[1] != typemax(Int32))
        @inbounds status[1] = isodd(failures[1]) ? input_failure : output_failure
    end
end

struct VdmShapeOffset{K,P,O,C,I,A,S,B}
    candidate!::K
    publish!::P
    output::O
    current_offset::C
    requested::I
    active_offset::A
    shape_offset::A
    candidate_output::B
    frame_status::S
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(::CompleteTransition, device,
        prepared::PreparedAlgorithm{VdmShapeOffsetF32, <:VdmShapeOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == 2 && length(inputs) == 1 ||
        throw(ArgumentError("shape offset capture requires one input and two outputs"))
    output, current = outputs
    requested = only(inputs)
    length(requested) <= (typemax(Int32) >> 1) ||
        throw(ArgumentError("shape offset extent exceeds status indexing capacity"))
    for (name, array) in zip((:output, Symbol("current-offset"), :requested), (outputs..., inputs...))
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, name), array)
        _require_capture_array("shape offset input/output", array, device)
    end
    _require_capture_array("captured frame status", frame_status, device)
    (_capture_arrays_overlap(output, current) || _capture_arrays_overlap(current, requested)) &&
        throw(ArgumentError("shape offset readback must not alias commands"))
    output === requested || !_capture_arrays_overlap(output, requested) ||
        throw(ArgumentError("shape offset command must not partially overlap input"))
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE;
        shared_memory=VALIDATION_WORKGROUP_SIZE * sizeof(Int32))
    backend = KernelAbstractions.get_backend(requested)
    return VdmShapeOffset(_vdm_shape_offset_candidate!(backend, VALIDATION_WORKGROUP_SIZE),
        _publish_vector_combination!(backend, PROJECTION_WORKGROUP_SIZE), output, current, requested,
        _capture_array(device, prepared.plan.active_offset), _capture_array(device, prepared.plan.shape_offset),
        _capture_array(device, zeros(Float32, length(output))), frame_status,
        _failure_code(node_index, VectorLinearCombinationNonFiniteInput),
        _failure_code(node_index, VectorLinearCombinationNonFiniteOutput))
end

preflight(::VdmShapeOffset) = nothing
function enqueue!(c::VdmShapeOffset)
    c.candidate!(c.candidate_output, c.requested, c.active_offset, c.frame_status,
        c.input_failure, c.output_failure; ndrange=VALIDATION_WORKGROUP_SIZE)
    c.publish!(c.output, c.candidate_output, c.current_offset, c.shape_offset, c.frame_status;
        ndrange=max(length(c.output), length(c.current_offset)))
    return nothing
end
function complete!(c::VdmShapeOffset, status::Int32)
    status == c.input_failure && return VectorLinearCombinationNonFiniteInput
    status == c.output_failure && return VectorLinearCombinationNonFiniteOutput
    return nothing
end
function reset!(c::VdmShapeOffset)
    fill!(c.candidate_output, 0.0f0)
    return c
end
preserves_inputs(c::VdmShapeOffset) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
    !_capture_arrays_overlap(c.output, c.requested)
finite_output(c::VdmShapeOffset, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))
finite_output(c::VdmShapeOffset, ::Val{2}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))
