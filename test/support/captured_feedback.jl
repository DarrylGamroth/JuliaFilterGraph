@enum CapturedFeedbackFailure::UInt8 begin
    CapturedFeedbackRejected = 1
end

@algorithm CapturedFeedbackSumF32 begin
    label = "captured-feedback-sum-f32"
    @configuration CapturedFeedbackSumF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:feedback, Float32, (configuration.extent,), "test.feedback.f32/1"),
        input(:drive, Float32, (configuration.extent,), "test.feedback.f32/1"),
        output(:sum, Float32, (configuration.extent,), "test.feedback.f32/1"; metadata_from=:drive),
        output(:next_feedback, Float32, (configuration.extent,), "test.feedback.f32/1";
            name="next-feedback", metadata_from=:feedback),
    )
    prepare(configuration) = nothing
    process!(plan) = begin
        for index in eachindex(sum, next_feedback, drive, feedback)
            sum[index] = drive[index] + feedback[index]
            next_feedback[index] = 0.5f0 * sum[index]
        end
        nothing
    end
end

@algorithm CapturedFeedbackGateF32 begin
    label = "captured-feedback-gate-f32"
    failure = CapturedFeedbackFailure
    @configuration CapturedFeedbackGateF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:sum, Float32, (configuration.extent,), "test.feedback.f32/1"),
        output(:checked, Float32, (configuration.extent,), "test.feedback.f32/1"; metadata_from=:sum),
    )
    prepare(configuration) = nothing
    process!(plan) = begin
        all(value -> isfinite(value) && value > 0.0f0, sum) || return CapturedFeedbackRejected
        copyto!(checked, sum)
        nothing
    end
end

const CAPTURED_FEEDBACK_GRAPH = """
filter.graph = {
    nodes = [
        { type = ndarray name = sum label = captured-feedback-sum-f32 config = { extent = 3 } }
        { type = ndarray name = gate label = captured-feedback-gate-f32 config = { extent = 3 } }
    ]
    links = [ { output = "sum:sum" input = "gate:sum" } ]
    inputs = [ "sum:feedback" "sum:drive" ]
    outputs = [ "sum:sum" "sum:next-feedback" "gate:checked" ]
}
"""

function captured_feedback_graph()
    return mktemp() do path, io
        write(io, CAPTURED_FEEDBACK_GRAPH)
        flush(io)
        prepare_graph(path; algorithms=(CapturedFeedbackSumF32, CapturedFeedbackGateF32))
    end
end

const CAPTURED_FEEDBACK_BINDING = (feedback=Symbol("next-feedback"),)
