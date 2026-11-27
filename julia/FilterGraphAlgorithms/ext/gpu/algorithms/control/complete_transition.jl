import FilterGraphAlgorithms: InputPort, OutputPort, FrameData, algorithm_ports, algorithm_formats

# The array identities and launch geometry are fixed before recording. The
# final guarded launches own publication and any local state adoption.
struct ControlCapture{K,F,R}
    kernels::K
    failures::F
    reset_state::R
end

ControlCapture(kernels, failures) = ControlCapture(kernels, failures, ())

preflight(::ControlCapture) = nothing

function enqueue!(capture::ControlCapture)
    for kernel in capture.kernels
        enqueue!(kernel)
    end
    return nothing
end

function complete!(capture::ControlCapture, status::Int32)
    for (code, failure) in capture.failures
        status == code && return failure
    end
    return nothing
end

function reset!(capture::ControlCapture)
    for (state, initial) in capture.reset_state
        if initial === nothing
            fill!(state, zero(eltype(state)))
        else
            copyto!(state, initial)
        end
    end
    return capture
end

_control_failures(node_index, failures) =
    map(failure -> (_failure_code(node_index, failure), failure), failures)

function _check_control_capture(device, prepared, outputs, inputs, status; input_overlap=:none)
    _check_capture_ports(device, prepared, outputs, inputs, status; input_overlap)
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE;
        shared_memory=VALIDATION_WORKGROUP_SIZE * sizeof(Int32))
    return nothing
end

_control_kernel(kernel, args, extent::Int; lanes=PROJECTION_WORKGROUP_SIZE) =
    CapturedKernel(kernel(KernelAbstractions.get_backend(first(args)), lanes), args, extent)

_control_copy(output, status, input) =
    _control_kernel(_identity_projection!, (output, status, input), length(output))

_control_validate(status, values, code) = _control_kernel(_validate_projection_values!,
    (status, values, code), VALIDATION_WORKGROUP_SIZE; lanes=VALIDATION_WORKGROUP_SIZE)

# Ordered row accumulation matches the existing portable projection kernel.
_control_projection(device, output, status, matrix, input) =
    _control_kernel(_projection!, (output, status, _capture_array(device, matrix), input),
        length(output))

@kernel function _control_combine!(output, status, left, right, op)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds output[i] = op(left[i], right[i])
    end
end

_control_combine(output, status, left, right, op) =
    _control_kernel(_control_combine!, (output, status, left, right, op), length(output))
