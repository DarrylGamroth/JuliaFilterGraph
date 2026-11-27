@kernel function _commit_captured_feedback!(destination, source, status)
    index = @index(Global, Linear)
    if index <= length(destination) && @inbounds(iszero(status[1]))
        @inbounds destination[index] = source[index]
    end
end

struct CUDACapturedFeedback{D,S,F,K}
    destination::D
    source::S
    frame_status::F
    commit!::K
end

FilterGraphs._require_captured_feedback_support(::CUDA.CuDevice, feedback) = nothing

function FilterGraphs._prepare_captured_feedback(
    ::CUDA.CuDevice, feedback, device_inputs, device_outputs, frame_status,
)
    return map(keys(feedback)) do input
        destination = device_inputs[input]
        source = device_outputs[feedback[input]]
        commit! = _commit_captured_feedback!(KernelAbstractions.get_backend(destination), 256)
        CUDACapturedFeedback(destination, source, frame_status.value, commit!)
    end
end

function FilterGraphs._enqueue_captured_feedback!(feedback::CUDACapturedFeedback)
    feedback.commit!(feedback.destination, feedback.source, feedback.frame_status;
        ndrange=length(feedback.destination))
    return nothing
end

function FilterGraphs._reset_captured_feedback!(feedback::CUDACapturedFeedback)
    fill!(feedback.destination, zero(eltype(feedback.destination)))
    return nothing
end

FilterGraphs._reset_captured_feedback!(executable::CUDACapturedExecutable) =
    FilterGraphs._reset_captured_feedback!(executable.feedback)
