# A one-column map is bᵢ = coefficientᵢ × amplitude, not a reduction. For at
# most one coordinate per lane, evaluate it inside the correction kernel. Keep
# the forward dot and larger/multiple-mode maps on the ordinary projection path.
# The amplitude lives in a one-element device array (modes[1]).
struct ModeExpansion{M}
    coefficients::M
end

@inline function _expand_mode(coefficients::AbstractArray, scratch, modes, index)
    # The qualified one-term cuBLAS path uses a product, preserving signed zeros
    # and subnormals. Materialize it before hidden-gain arithmetic; keep 0*Inf.
    @inbounds value = coefficients[index] * modes[1]
    @inbounds scratch[index] = value
    return value
end

@inline function _expand_mode(::Val{:identity}, scratch, modes, index)
    # Identity/selection projections copy; preserve zero signs and subnormals.
    @inbounds value = modes[1]
    @inbounds scratch[index] = value
    return value
end

@inline function _expand_mode(::Val{:selection}, scratch, modes, index)
    @inbounds value = modes[1]
    value = isfinite(value) ? value : oftype(value, NaN)
    @inbounds scratch[index] = value
    return value
end

@inline _expand_mode(::Nothing, scratch, modes, index) = @inbounds scratch[index]

_expansion_values(projection) = nothing
_expansion_values(expansion::ModeExpansion) = expansion.coefficients

_enqueue_projection!(::ModeExpansion, output, status, input) = nothing

function _require_expansion(matrix, output, input)
    size(matrix) == (length(output), length(input)) || throw(
        DimensionMismatch("hidden expansion size $(size(matrix)) does not match " *
            "$(length(output)) outputs and $(length(input)) inputs"),
    )
    eltype(matrix) === eltype(output) === eltype(input) || throw(
        ArgumentError("hidden expansion element types must match"),
    )
    return nothing
end

function _prepare_expansion(device, matrix, output, input)
    _require_expansion(matrix, output, input)
    if size(matrix, 2) == 1 && length(output) <= VALIDATION_WORKGROUP_SIZE
        return _prepare_mode_expansion(FilterGraphAlgorithms._cpu_projection_execution(matrix),
            device, matrix)
    end
    return _prepare_projection(device, matrix, output, input,
        COOPERATIVE_PROJECTION_DEFAULT_LANES)
end

_prepare_mode_expansion(::FilterGraphAlgorithms.DenseProjection, device, matrix) =
    ModeExpansion(_capture_array(device, vec(matrix)))
_prepare_mode_expansion(::FilterGraphAlgorithms.IdentityProjection, device, matrix) =
    ModeExpansion(Val(:identity))
_prepare_mode_expansion(::FilterGraphAlgorithms.SelectionProjection, device, matrix) =
    ModeExpansion(Val(:selection))

# Ordered validation workgroups preserve input-before-feedback-before-output
# rejection. Projection storage is speculative and never becomes committed state
# until this Node's output workgroup has validated every candidate.
@kernel unsafe_indices=true function _closed_loop_feedback!(
        candidate_state, realized, frame_status, committed_state, committed_output,
        has_output, feedback, anti_windup_gain, closed, no_hidden_modes, feedback_failure, vote
)
    lane = Int(@index(Local, Linear))
    valid = @inbounds iszero(frame_status[1])
    previous = @inbounds !iszero(has_output[1])
    failure = false
    index = lane
    while valid && index <= length(candidate_state)
        if previous
            @inbounds begin
                value = feedback[index]
                output = committed_output[index]
                realized[index] = closed ? output - value : zero(eltype(realized))
                candidate_state[index] = closed ?
                    (no_hidden_modes ? fma(-anti_windup_gain, value, output) :
                        output - anti_windup_gain * value) : committed_state[index]
            end
            isfinite(value) || (failure = true)
        else
            # The declared CPU Algorithm ignores supplied feedback on frame one.
            @inbounds begin
                realized[index] = zero(eltype(realized))
                candidate_state[index] = committed_state[index]
            end
        end
        index += VALIDATION_WORKGROUP_SIZE
    end
    # Preparation launches exactly one full 256-lane block. Every lane must
    # reach the collective, including those outside the vector extent.
    failed = vote(failure)
    if valid && lane == 1 && failed
        @inbounds frame_status[1] = feedback_failure
    end
end

@kernel unsafe_indices=true function _closed_loop_output!(
        correction, controller_state, candidate_output, candidate_state, frame_status,
        hidden_in_state, expansion, hidden_modes, committed_state, committed_output, has_output,
        residual_error, gain, pole, hidden_gain, closed,
        output_failure, vote
)
    lane = Int(@index(Local, Linear))
    valid = @inbounds iszero(frame_status[1])
    previous = @inbounds !iszero(has_output[1])
    failure = false
    index = lane
    while valid && index <= length(candidate_state)
        @inbounds state = candidate_state[index]
        @inbounds value = committed_output[index]
        if closed
            if isempty(hidden_modes)
                # The zero-mode integrator follows CPU rounding explicitly:
                # round the pole product before fusing the residual product.
                @inbounds value = fma(gain, residual_error[index], pole * state)
            else
                hidden = _expand_mode(expansion, hidden_in_state, hidden_modes, index)
                if previous
                    state -= hidden_gain * hidden
                end
                @inbounds value = pole * state + gain * residual_error[index]
            end
        end
        if !(isfinite(state) && isfinite(value))
            failure = true
        end
        @inbounds begin
            candidate_state[index] = state
            candidate_output[index] = value
        end
        index += VALIDATION_WORKGROUP_SIZE
    end
    failed = vote(failure)
    if valid && lane == 1 && failed
        @inbounds frame_status[1] = output_failure
    end
    # The collective joins all residual/previous-frame readers before
    # publication or history writes, including overlapping supplied inputs.
    # Use its uniform result rather than racing lane one's status write.
    if valid && !failed
        index = lane
        while index <= length(candidate_state)
            @inbounds begin
                correction[index] = candidate_output[index]
                controller_state[index] = candidate_state[index]
                committed_state[index] = candidate_state[index]
                committed_output[index] = candidate_output[index]
            end
            index += VALIDATION_WORKGROUP_SIZE
        end
        lane == 1 && @inbounds(has_output[1] = Int32(1))
    end
end

@kernel function _reset_closed_loop!(
        committed_state, committed_output, candidate_state, candidate_output,
        realized, hidden_modes, hidden_in_state, has_output
)
    index = @index(Global, Linear)
    if index <= length(committed_state)
        @inbounds begin
            committed_state[index] = zero(eltype(committed_state))
            committed_output[index] = zero(eltype(committed_state))
            candidate_state[index] = zero(eltype(committed_state))
            candidate_output[index] = zero(eltype(committed_state))
            realized[index] = zero(eltype(committed_state))
            hidden_in_state[index] = zero(eltype(committed_state))
        end
    end
    index <= length(hidden_modes) && @inbounds(hidden_modes[index] = zero(eltype(committed_state)))
    index == 1 && @inbounds(has_output[1] = Int32(0))
end

import FilterGraphAlgorithms:
                              ClosedLoopCorrectionF32,
                              ClosedLoopIntegratorF32,
                              ClosedLoopCorrectionPlan,
                              ClosedLoopCorrectionWorkspace,
                              ClosedLoopCorrectionNonFiniteInput,
                              ClosedLoopCorrectionNonFiniteFeedback,
                              ClosedLoopCorrectionNonFiniteOutput,
                              PreparedAlgorithm,
                              complete!,
                              enqueue!,
                              preflight,
                              reset!

struct ClosedLoopCorrection{K, O, V, S, H, P, T, F}
    kernels::K
    outputs::O
    vectors::V
    frame_status::S
    has_output::H
    projections::P
    gain::T
    pole::T
    anti_windup_gain::T
    hidden_mode_gain::T
    closed::Bool
    failures::F
end

function _prepare_capture(
        ::CompleteTransition,
        device,
        prepared::PreparedAlgorithm{
            <:Union{ClosedLoopCorrectionF32,ClosedLoopIntegratorF32},
            <:ClosedLoopCorrectionPlan, <:ClosedLoopCorrectionWorkspace
        },
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, vote
)
    length(outputs) == 2 ||
        throw(ArgumentError("ClosedLoopCorrectionF32 capture requires two outputs"))
    length(inputs) == 2 ||
        throw(ArgumentError("ClosedLoopCorrectionF32 capture requires two inputs"))
    residual_error, constraint_feedback = inputs
    correction, controller_state = outputs
    residual_format = FilterGraphAlgorithms.algorithm_format(prepared, Symbol("residual-error"))
    feedback_format = FilterGraphAlgorithms.algorithm_format(prepared, Symbol("constraint-feedback"))
    correction_format = FilterGraphAlgorithms.algorithm_format(prepared, :correction)
    state_format = FilterGraphAlgorithms.algorithm_format(prepared, Symbol("controller-state"))
    for (format, input) in zip((residual_format, feedback_format), inputs)
        FilterGraphAlgorithms._validate_declared_array(format, input)
        _require_capture_array("closed-loop input", input, device)
    end
    for (format, output) in zip((correction_format, state_format), outputs)
        FilterGraphAlgorithms._validate_declared_array(format, output)
        _require_capture_array("closed-loop output", output, device)
    end
    Base.mightalias(outputs...) &&
        throw(ArgumentError("closed-loop outputs must not overlap"))
    _require_capture_array("captured frame status", frame_status, device)
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE)
    plan, workspace = prepared.plan, prepared.workspace
    backend = KernelAbstractions.get_backend(correction)
    vectors = (
        committed_state = _capture_array(device, workspace.committed_state),
        committed_output = _capture_array(device, workspace.committed_output),
        candidate_state = _capture_zeros(device, Float32, plan.state_length),
        candidate_output = _capture_zeros(device, Float32, plan.state_length),
        realized = _capture_zeros(device, Float32, plan.state_length),
        hidden_modes = _capture_zeros(device, Float32, plan.hidden_mode_count),
        hidden_in_state = _capture_zeros(device, Float32, plan.state_length),
        residual_error = residual_error, constraint_feedback = constraint_feedback
    )
    projections = iszero(plan.hidden_mode_count) ?
        (shape_to_hidden=NoHiddenModes(), hidden_to_shape=NoHiddenModes()) : (
        shape_to_hidden = _prepare_projection(device, plan.shape_to_hidden,
            vectors.hidden_modes, vectors.realized,
            COOPERATIVE_PROJECTION_DEFAULT_LANES),
        hidden_to_shape = _prepare_expansion(device, plan.hidden_to_shape,
            vectors.hidden_in_state, vectors.hidden_modes)
    )
    kernels = (
        vote = vote,
        validate! = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE),
        feedback! = _closed_loop_feedback!(backend, VALIDATION_WORKGROUP_SIZE),
        output! = _closed_loop_output!(backend, VALIDATION_WORKGROUP_SIZE),
        reset! = _reset_closed_loop!(backend, PROJECTION_WORKGROUP_SIZE)
    )
    return ClosedLoopCorrection(
        kernels, (; correction, controller_state), vectors, frame_status,
        _capture_array(device, Int32[workspace.committed_has_output]), projections,
        plan.gain, plan.pole, plan.anti_windup_gain, plan.hidden_mode_gain, plan.closed,
        (input = _failure_code(node_index, ClosedLoopCorrectionNonFiniteInput),
            feedback = _failure_code(node_index, ClosedLoopCorrectionNonFiniteFeedback),
            output = _failure_code(node_index, ClosedLoopCorrectionNonFiniteOutput))
    )
end

preflight(::ClosedLoopCorrection) = nothing

function enqueue!(captured::ClosedLoopCorrection)
    v, k = captured.vectors, captured.kernels
    k.validate!(captured.frame_status, v.residual_error, captured.failures.input;
        ndrange = VALIDATION_WORKGROUP_SIZE)
    k.feedback!(v.candidate_state, v.realized, captured.frame_status,
        v.committed_state, v.committed_output, captured.has_output,
        v.constraint_feedback, captured.anti_windup_gain, captured.closed, isempty(v.hidden_modes),
        captured.failures.feedback, k.vote;
        ndrange = VALIDATION_WORKGROUP_SIZE)
    if captured.closed
        _enqueue_projection!(captured.projections.shape_to_hidden, v.hidden_modes,
            captured.frame_status, v.realized)
        _enqueue_projection!(captured.projections.hidden_to_shape, v.hidden_in_state,
            captured.frame_status, v.hidden_modes)
    end
    k.output!(captured.outputs.correction, captured.outputs.controller_state, v.candidate_output,
        v.candidate_state, captured.frame_status, v.hidden_in_state,
        _expansion_values(captured.projections.hidden_to_shape), v.hidden_modes, v.committed_state, v.committed_output, captured.has_output,
        v.residual_error, captured.gain, captured.pole,
        captured.hidden_mode_gain, captured.closed, captured.failures.output, k.vote;
        ndrange = VALIDATION_WORKGROUP_SIZE)
    return nothing
end

function complete!(captured::ClosedLoopCorrection, status::Int32)
    status == captured.failures.input && return ClosedLoopCorrectionNonFiniteInput
    status == captured.failures.feedback && return ClosedLoopCorrectionNonFiniteFeedback
    status == captured.failures.output && return ClosedLoopCorrectionNonFiniteOutput
    return nothing
end

function reset!(captured::ClosedLoopCorrection)
    v = captured.vectors
    captured.kernels.reset!(v.committed_state, v.committed_output,
        v.candidate_state, v.candidate_output, v.realized, v.hidden_modes,
        v.hidden_in_state, captured.has_output;
        ndrange = max(length(v.committed_state), length(v.hidden_modes)))
    return captured
end

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs, finite_output, prepare_validation

# Sharing this adapter does not qualify another backend for validation reuse.
# Qualified backends preserve supplied inputs in their ordered stream.

preserves_inputs(c::ClosedLoopCorrection) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::ClosedLoopCorrection, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::ClosedLoopCorrection, proofs::Tuple{
        Bool, Bool})
    residual_finite, _ = proofs
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        residual_finite || return c
    kernels = merge(c.kernels, (; validate! = _FiniteInput()))
    return ClosedLoopCorrection(
        kernels, c.outputs, c.vectors, c.frame_status, c.has_output,
        c.projections, c.gain, c.pole, c.anti_windup_gain, c.hidden_mode_gain, c.closed, c.failures
    )
end
