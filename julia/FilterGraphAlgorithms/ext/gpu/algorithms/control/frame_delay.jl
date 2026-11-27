import FilterGraphAlgorithms:
    FrameDelayF32,
    FrameDelayNonFiniteInput,
    FrameDelayPlan,
    FrameDelayWorkspace,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!

@kernel function _frame_delay_output!(
    output,
    frame_status,
    history,
    head,
    input,
    delay_frames,
)
    index = @index(Global, Linear)
    if index <= length(output) && @inbounds(iszero(frame_status[1]))
        if iszero(delay_frames)
            @inbounds output[index] = input[index]
        else
            @inbounds output[index] = history[index, head[1]]
        end
    end
end

# This copy intentionally does not advance `head`.  A separately enqueued
# kernel advances it only after every work-item has written the ring column.
@kernel function _adopt_frame_delay_history!(
    history,
    head,
    input,
    frame_status,
)
    index = @index(Global, Linear)
    if index <= length(input) && @inbounds(iszero(frame_status[1]))
        @inbounds history[index, head[1]] = input[index]
    end
end

@kernel function _advance_frame_delay_head!(
    head,
    frame_status,
    delay_frames,
)
    if @index(Global, Linear) == 1 && @inbounds(iszero(frame_status[1]))
        @inbounds begin
            current_head = head[1]
            head[1] = current_head == delay_frames ?
                      one(current_head) : current_head + one(current_head)
        end
    end
end

@kernel function _reset_frame_delay_history!(history, initial_history)
    index = @index(Global, Linear)
    if index <= length(history)
        @inbounds history[index] = initial_history[index]
    end
end

@kernel function _reset_frame_delay_head!(head)
    if @index(Global, Linear) == 1
        @inbounds head[1] = one(eltype(head))
    end
end

struct FrameDelay{
    Validate,
    Output,
    Adopt,
    Advance,
    ResetHistory,
    ResetHead,
    O,
    S,
    H,
    I,
    Head,
    Input,
}
    validate_input!::Validate
    output!::Output
    adopt_history!::Adopt
    advance_head!::Advance
    reset_history!::ResetHistory
    reset_head!::ResetHead
    output::O
    frame_status::S
    history::H
    initial_history::I
    head::Head
    input::Input
    delay_frames::Int32
    input_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        FrameDelayF32,
        <:FrameDelayPlan,
        <:FrameDelayWorkspace,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 1 || throw(
        ArgumentError("FrameDelayF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("FrameDelayF32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :input),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :output),
        output,
    )
    _require_capture_array("frame-delay input", input, device)
    _require_capture_array("frame-delay output", output, device)
    _require_capture_array("captured frame status", frame_status, device)
    (output !== input || !iszero(prepared.plan.delay_frames)) &&
        _capture_arrays_overlap(output, input) && throw(
        ArgumentError("frame-delay buffers must be disjoint or identical at zero delay"),
    )

    backend = KernelAbstractions.get_backend(output)
    history = _capture_array(device, prepared.workspace.history)
    initial_history_host = similar(prepared.workspace.history)
    for frame in axes(initial_history_host, 2),
        index in eachindex(prepared.workspace.initial_state)
        initial_history_host[index, frame] = prepared.workspace.initial_state[index]
    end
    initial_history = _capture_array(device, initial_history_host)
    head = _capture_array(device, Int32[prepared.workspace.head])
    return FrameDelay(
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        _frame_delay_output!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _adopt_frame_delay_history!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _advance_frame_delay_head!(backend, 1),
        _reset_frame_delay_history!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _reset_frame_delay_head!(backend, 1),
        output,
        frame_status,
        history,
        initial_history,
        head,
        input,
        Int32(prepared.plan.delay_frames),
        _failure_code(node_index, FrameDelayNonFiniteInput),
    )
end

preflight(::FrameDelay) = nothing

function enqueue!(captured::FrameDelay)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    captured.output!(
        captured.output,
        captured.frame_status,
        captured.history,
        captured.head,
        captured.input,
        captured.delay_frames;
        ndrange=length(captured.output),
    )
    iszero(captured.delay_frames) && return nothing
    captured.adopt_history!(
        captured.history,
        captured.head,
        captured.input,
        captured.frame_status;
        ndrange=length(captured.input),
    )
    captured.advance_head!(
        captured.head,
        captured.frame_status,
        captured.delay_frames;
        ndrange=1,
    )
    return nothing
end

function complete!(captured::FrameDelay, status::Int32)
    status == captured.input_failure && return FrameDelayNonFiniteInput
    return nothing
end

function reset!(captured::FrameDelay)
    if !iszero(captured.delay_frames)
        captured.reset_history!(
            captured.history,
            captured.initial_history;
            ndrange=length(captured.history),
        )
    end
    captured.reset_head!(captured.head; ndrange=1)
    return captured
end
