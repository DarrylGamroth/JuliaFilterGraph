# CUDA native workgroup votes join input validation and state adoption in one
# block. The portable staged transition remains in the shared GPU source.
# The bounded transition owns at most two actuators per lane. This geometry
# controls the input scan, candidate map, collective and state adoption together.
const PDM_COMMAND_LANES = 256
const PDM_COMMAND_TILE = 2 * PDM_COMMAND_LANES

# One block owns every actuator only for the bounded, disjoint-buffer case.
# Larger extents retain staged execution; the disjoint selector is defensive.
# Public captured preparation rejects overlapping supplied buffers.
struct _PdmBlock{K}
    transition!::K
end

_pdm_check_input(validator) = Val(true)

function _prepare_pdm_execution(device::CUDA.CuDevice, backend, demanded, feedback, requested)
    if 1 <= length(requested) <= PDM_COMMAND_TILE &&
       !Base.mightalias(demanded, feedback) &&
       !Base.mightalias(demanded, requested) &&
       !Base.mightalias(feedback, requested)
        _require_capture_workgroup(device, PDM_COMMAND_LANES)
        return _PdmBlock(_pdm_command_block!(backend, PDM_COMMAND_LANES))
    end
    return _PdmStages()
end

@kernel unsafe_indices=true function _pdm_command_block!(
        demanded, constraint_feedback, candidate_current,
        frame_status, committed_current,
        requested, lower_limits, upper_limits, slew_limits,
        slew, input_failure, output_failure,
        ::Val{CheckInput},
) where {CheckInput}
    lane = Int(@index(Local, Linear))
    frame_valid = @inbounds iszero(frame_status[1])
    input_valid = frame_valid
    if CheckInput
        failed = Int32(0)
        actuator = lane
        while frame_valid && actuator <= length(requested)
            if !isfinite(@inbounds requested[actuator])
                failed = Int32(1)
                break
            end
            actuator += PDM_COMMAND_LANES
        end
        # Every lane participates, including lanes without an actuator. The
        # result stays in registers, so no shared-scratch reader join is needed.
        input_failed = CUDA.sync_threads_or(!iszero(failed))
        input_valid = frame_valid && !input_failed
        if frame_valid && !input_valid && lane == 1
            @inbounds frame_status[1] = input_failure
        end
    end

    failed = Int32(0)
    actuator = lane
    while input_valid && actuator <= length(demanded)
        @inbounds begin
            safe_target = clamp(requested[actuator], lower_limits[actuator], upper_limits[actuator])
            current = _pdm_current(slew, committed_current[actuator],
                safe_target, slew_limits[actuator])
            candidate_current[actuator] = current
            value = clamp(current, lower_limits[actuator], upper_limits[actuator])
            feedback = requested[actuator] - value
            demanded[actuator] = value
            constraint_feedback[actuator] = feedback
            if !isfinite(value) || !isfinite(feedback) || !isfinite(current)
                failed = Int32(1)
            end
        end
        actuator += PDM_COMMAND_LANES
    end
    # This block collective joins every candidate producer and every old-state
    # reader before any committed state can be overwritten.
    output_failed = CUDA.sync_threads_or(!iszero(failed))
    if input_valid && output_failed
        if lane == 1
            @inbounds frame_status[1] = output_failure
        end
    elseif input_valid
        actuator = lane
        while actuator <= length(committed_current)
            @inbounds committed_current[actuator] = candidate_current[actuator]
            actuator += PDM_COMMAND_LANES
        end
    end
end

function _enqueue_pdm!(captured::PdmCommand, execution::_PdmBlock)
    execution.transition!(
        captured.demanded, captured.constraint_feedback, captured.candidate_current,
        captured.frame_status, captured.committed_current,
        captured.requested, captured.lower_limits, captured.upper_limits,
        captured.slew_limits, captured.slew, captured.input_failure, captured.output_failure,
        _pdm_check_input(captured.validate_input!); ndrange=PDM_COMMAND_LANES,
    )
    return nothing
end

# CUDA input proofs may omit the checked input scan.
_pdm_check_input(::_FiniteInput) = Val(false)
