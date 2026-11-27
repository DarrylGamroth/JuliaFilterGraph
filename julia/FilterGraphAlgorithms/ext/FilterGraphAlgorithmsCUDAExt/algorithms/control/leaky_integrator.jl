# The existing step already assigns all elements to one strided block. Join its
# readers and checks with native votes, then adopt before leaving that block.
struct _LeakyAdopted end
(::_LeakyAdopted)(committed, candidate, status; ndrange) = nothing

function _prepare_leaky_transition(device::CUDA.CuDevice, backend)
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE)
    return (_leaky_transition!(backend, VALIDATION_WORKGROUP_SIZE), _LeakyAdopted())
end

@kernel unsafe_indices=true function _leaky_transition!(
        candidate, frame_status, committed, input, gain, pole, closed,
        input_failure, output_failure)
    lane = Int(@index(Local, Linear))
    valid = @inbounds iszero(frame_status[1])
    bad_input = false
    bad_output = false
    index = lane
    while valid && index <= length(candidate)
        @inbounds input_value = input[index]
        if !isfinite(input_value)
            bad_input = true
        else
            @inbounds state_value = committed[index]
            value = state_value
            if closed
                state_term = pole * state_value
                input_term = gain * input_value
                value = state_term + input_term
            end
            if !isfinite(value)
                bad_output = true
            else
                @inbounds candidate[index] = value
            end
        end
        index += VALIDATION_WORKGROUP_SIZE
    end
    # Two unconditional collectives retain input-before-output priority. Every
    # old-state/input reader completes before any committed state is replaced.
    bad_input = CUDA.sync_threads_or(bad_input)
    bad_output = CUDA.sync_threads_or(bad_output)
    if valid && lane == 1
        if bad_input
            @inbounds frame_status[1] = input_failure
        elseif bad_output
            @inbounds frame_status[1] = output_failure
        end
    end
    if valid && !bad_input && !bad_output
        index = lane
        while index <= length(committed)
            @inbounds committed[index] = candidate[index]
            index += VALIDATION_WORKGROUP_SIZE
        end
    end
end
