import FilterGraphAlgorithms:
    LeakyIntegratorF32,
    LeakyIntegratorNonFiniteInput,
    LeakyIntegratorNonFiniteOutput,
    LeakyIntegratorPlan,
    LeakyIntegratorWorkspace,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!

const LEAKY_VALID = Int32(0)
const LEAKY_OUTPUT_FAILURE = Int32(1)
const LEAKY_INPUT_FAILURE = Int32(2)

@kernel function _leaky_step!(
    candidate,
    frame_status,
    committed,
    input,
    gain,
    pole,
    closed,
    input_failure,
    output_failure,
)
    local_index = @index(Local, Linear)
    local_failures = @localmem Int32 (VALIDATION_WORKGROUP_SIZE,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    local_failure = LEAKY_VALID
    if frame_valid
        for index in Int(local_index):VALIDATION_WORKGROUP_SIZE:length(candidate)
            @inbounds input_value = input[index]
            if !isfinite(input_value)
                local_failure = LEAKY_INPUT_FAILURE
            else
                @inbounds state_value = committed[index]
                value = state_value
                if closed
                    state_term = pole * state_value
                    input_term = gain * input_value
                    value = state_term + input_term
                end
                if !isfinite(value)
                    local_failure = max(
                        local_failure,
                        LEAKY_OUTPUT_FAILURE,
                    )
                else
                    @inbounds candidate[index] = value
                end
            end
        end
    end
    @inbounds local_failures[local_index] = local_failure
    @synchronize

    _reduce_validation!(local_failures, Int(local_index), 128)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 64)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 32)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 16)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 8)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 4)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 2)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 1)
    # Lane one is the only subsequent reader of its own final result.

    if frame_valid && local_index == 1
        @inbounds reduced_failure = local_failures[1]
        if reduced_failure == LEAKY_INPUT_FAILURE
            @inbounds frame_status[1] = input_failure
        elseif reduced_failure == LEAKY_OUTPUT_FAILURE
            @inbounds frame_status[1] = output_failure
        end
    end
end

@kernel function _adopt_leaky_state!(committed, candidate, frame_status)
    index = @index(Global, Linear)
    if index <= length(committed) && @inbounds(iszero(frame_status[1]))
        @inbounds committed[index] = candidate[index]
    end
end

@kernel function _reset_leaky_state!(committed, candidate, initial)
    index = @index(Global, Linear)
    if index <= length(committed)
        @inbounds begin
            committed[index] = initial[index]
            candidate[index] = initial[index]
        end
    end
end

_prepare_leaky_transition(device, backend) = (
    _leaky_step!(backend, VALIDATION_WORKGROUP_SIZE),
    _adopt_leaky_state!(backend, PROJECTION_WORKGROUP_SIZE),
)

struct LeakyIntegrator{
    Step,
    Adopt,
    Reset,
    O,
    S,
    C,
    I,
    Input,
    T,
}
    step!::Step
    adopt!::Adopt
    reset!::Reset
    output::O
    frame_status::S
    committed_state::C
    initial_state::I
    input::Input
    gain::T
    pole::T
    closed::Bool
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        LeakyIntegratorF32,
        <:LeakyIntegratorPlan,
        <:LeakyIntegratorWorkspace,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 1 || throw(
        ArgumentError("LeakyIntegratorF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("LeakyIntegratorF32 capture requires one input"),
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
    _require_capture_array("leaky integrator input", input, device)
    _require_capture_array("leaky integrator candidate output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    initial_state = _capture_array(device, prepared.workspace.initial_state)
    committed_state = _capture_array(device, prepared.workspace.committed_state)
    backend = KernelAbstractions.get_backend(output)
    step!, adopt! = _prepare_leaky_transition(device, backend)
    return LeakyIntegrator(
        step!,
        adopt!,
        _reset_leaky_state!(backend, PROJECTION_WORKGROUP_SIZE),
        output,
        frame_status,
        committed_state,
        initial_state,
        input,
        prepared.plan.gain,
        prepared.plan.pole,
        prepared.plan.closed,
        _failure_code(node_index, LeakyIntegratorNonFiniteInput),
        _failure_code(node_index, LeakyIntegratorNonFiniteOutput),
    )
end

preflight(::LeakyIntegrator) = nothing

function enqueue!(captured::LeakyIntegrator)
    captured.step!(
        captured.output,
        captured.frame_status,
        captured.committed_state,
        captured.input,
        captured.gain,
        captured.pole,
        captured.closed,
        captured.input_failure,
        captured.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    captured.adopt!(
        captured.committed_state,
        captured.output,
        captured.frame_status;
        ndrange=length(captured.output),
    )
    return nothing
end

function complete!(
    captured::LeakyIntegrator,
    status::Int32,
)
    status == captured.input_failure &&
        return LeakyIntegratorNonFiniteInput
    status == captured.output_failure &&
        return LeakyIntegratorNonFiniteOutput
    return nothing
end

function reset!(captured::LeakyIntegrator)
    captured.reset!(
        captured.committed_state,
        captured.output,
        captured.initial_state;
        ndrange=length(captured.output),
    )
    return captured
end
