import FilterGraphAlgorithms: FrameData, InputPort, OutputPort, PreparedAlgorithm,
    complete!, enqueue!, preflight, reset!

# All callbacks close over fixed device storage and only enqueue ordered work.
# The captured Graph owns the shared frame status and the completion readback.
struct CalibrationTransition{Enqueue,Reset,Failures}
    enqueue::Enqueue
    reset::Reset
    failures::Failures
end

preflight(::CalibrationTransition) = nothing
enqueue!(transition::CalibrationTransition) = transition.enqueue()

function complete!(transition::CalibrationTransition, status::Int32)
    for (code, failure) in transition.failures
        status == code && return failure
    end
    return nothing
end

function reset!(transition::CalibrationTransition)
    transition.reset()
    return transition
end

function _calibration_bind(device, prepared, outputs, inputs, status)
    backend = _check_capture_ports(device, prepared, outputs, inputs, status; input_overlap=:all)
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE;
        shared_memory=VALIDATION_WORKGROUP_SIZE * sizeof(Int32))
    return backend
end

_calibration_failures(node, failures...) =
    map(failure -> (_failure_code(node, failure), failure), failures)

function _calibration_validate!(validate, status, values, code)
    validate(status, values, code; ndrange=VALIDATION_WORKGROUP_SIZE)
    return nothing
end

function _calibration_publish!(publish, status, output, staging)
    publish(output, status, staging; ndrange=length(output))
    return nothing
end

include("probe.jl")
include("matched_differencing.jl")
include("response.jl")
include("structured_interaction_matrix.jl")
include("stability.jl")
include("covariance.jl")
include("docrime_interaction_matrix.jl")
include("excitation.jl")
