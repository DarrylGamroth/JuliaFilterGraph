struct _CUDAFiniteValidation{Single,Multiple}
    single!::Single
    multiple!::Multiple
end

function _validate_projection_values!(backend::CUDA.CUDACore.CUDABackend,
        lanes::Integer)
    lanes == VALIDATION_WORKGROUP_SIZE ||
        return _validate_projection_values_portable!(backend, lanes)
    return _CUDAFiniteValidation(
        _validate_projection_values_cuda!(backend, lanes),
        _validate_projection_values_blocks!(backend, lanes),
    )
end

function (validate::_CUDAFiniteValidation)(frame_status, values, failure, role_args...;
        ndrange, workgroupsize=nothing)
    return validate.single!(frame_status, values, failure, role_args...; ndrange, workgroupsize)
end

function (validate::_CUDAFiniteValidation)(
        frame_status::CUDA.CuArray{Int32,1,CUDA.DeviceMemory}, values::CUDA.CuArray, failure::Int32;
        ndrange, workgroupsize=nothing)
    if ndrange == VALIDATION_WORKGROUP_SIZE && workgroupsize === nothing &&
            length(values) > VALIDATION_WORKGROUP_SIZE &&
            !_capture_arrays_overlap(frame_status, values)
        return validate.multiple!(frame_status, values, failure; ndrange=length(values))
    end
    return validate.single!(frame_status, values, failure; ndrange, workgroupsize)
end

@kernel unsafe_indices=true function _validate_projection_values_blocks!(
        frame_status, values, failure)
    lane = Int(@index(Local, Linear))
    group = Int(@index(Group, Linear))
    index = (group - 1) * VALIDATION_WORKGROUP_SIZE + lane
    failed = index <= length(values) && !isfinite(@inbounds values[index])
    failed = CUDA.sync_threads_or(failed)
    # Captured values are disjoint from frame status. Between the ordered
    # kernels that produce and consume status, only failing blocks access it.
    if failed && lane == 1
        @inbounds KernelAbstractions.@atomicreplace(:monotonic, frame_status[1], Int32(0) => failure)
    end
end

# Role publication stays in the block that scanned all values.
@kernel unsafe_indices=true function _validate_projection_values_cuda!(
        frame_status, values, failure, role_output=nothing, roles=nothing)
    lane = Int(@index(Local, Linear))
    valid = @inbounds iszero(frame_status[1])
    failed = false
    index = lane
    while valid && index <= length(values)
        if !isfinite(@inbounds values[index])
            failed = true
            break
        end
        index += VALIDATION_WORKGROUP_SIZE
    end
    # Inactive lanes and prior-rejected frames participate too. Use this uniform
    # result for publication, rather than reading lane one's concurrent store.
    failed = CUDA.sync_threads_or(failed)
    if valid && failed && lane == 1
        @inbounds frame_status[1] = failure
    end
    if role_output !== nothing && valid && !failed
        index = lane
        while index <= length(role_output)
            @inbounds role_output[index] = roles[index]
            index += VALIDATION_WORKGROUP_SIZE
        end
    end
end
