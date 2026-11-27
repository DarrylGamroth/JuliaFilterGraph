struct _VectorCombinationBlock{K}
    transition!::K
end

function _prepare_vector_combination_output(device::CUDA.CuDevice, backend, extent, validate!)
    if extent <= PROJECTION_WORKGROUP_SIZE
        _require_capture_workgroup(device, PROJECTION_WORKGROUP_SIZE)
        return _VectorCombinationBlock(_vector_combination_block!(backend, PROJECTION_WORKGROUP_SIZE))
    end
    return _prepare_vector_combination_stages(device, backend, extent, validate!)
end

@kernel unsafe_indices=true function _vector_combination_block!(
        output, current_offset, left, right, left_scale, right_scale, frame_status, output_failure)
    lane = Int(@index(Local, Linear))
    active = @inbounds iszero(frame_status[1]) && lane <= length(output)
    right_value = 0.0f0
    value = 0.0f0
    if active
        @inbounds begin
            right_value = right[lane]
            value = _vector_combination_value(left_scale, left[lane], right_scale, right_value)
        end
    end
    # One block covers the vector. Every lane votes before any output store,
    # including when output aliases an input. Inactive lanes vote finite.
    failed = CUDA.sync_threads_or(!isfinite(value))
    if failed
        if lane == 1
            @inbounds frame_status[1] = output_failure
        end
    elseif active
        @inbounds output[lane] = value
        if current_offset !== nothing
            @inbounds current_offset[lane] = right_value
        end
    end
end

function _enqueue_vector_combination_output!(c::VectorCombination, execution::_VectorCombinationBlock)
    execution.transition!(c.output, c.current_offset, c.left, c.right, c.left_scale, c.right_scale,
        c.frame_status, c.output_failure; ndrange=PROJECTION_WORKGROUP_SIZE)
    return nothing
end
