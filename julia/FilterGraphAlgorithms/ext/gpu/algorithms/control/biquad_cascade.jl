import FilterGraphAlgorithms:
    BiquadCascadeF32, BiquadCascadePlan, BiquadCascadeWorkspace,
    BiquadCascadeNonFiniteInput, BiquadCascadeNonFiniteOutput,
    PreparedAlgorithm, complete!, enqueue!, preflight, reset!,
    _supports_validation_reuse, preserves_inputs, finite_output, prepare_validation

# Each lane owns one mode through the complete cascade. All input reads and
# speculative state writes finish before validation and local publication.
@kernel function _stage_biquad!(candidate_state, candidate_output, committed_state,
    coefficients, input, frame_status)
    mode = @index(Global, Linear)
    if mode <= length(input) && @inbounds(iszero(frame_status[1]))
        @inbounds value = input[mode]
        set = size(coefficients, 3) == 1 ? 1 : mode
        for section in axes(coefficients, 2)
            @inbounds begin
                b0 = coefficients[1, section, set]
                b1 = coefficients[2, section, set]
                b2 = coefficients[3, section, set]
                a1 = coefficients[4, section, set]
                a2 = coefficients[5, section, set]
                s1 = committed_state[1, mode, section]
                s2 = committed_state[2, mode, section]
                section_output = fma(b0, value, s1)
                next_s1 = fma(-a1, section_output, b1 * value) + s2
                next_s2 = fma(-a2, section_output, b2 * value)
                candidate_state[1, mode, section] = next_s1
                candidate_state[2, mode, section] = next_s2
            end
            # Preserve evidence of any nonfinite intermediate even if later
            # sections could turn it into a finite output.
            if !(isfinite(section_output) && isfinite(next_s1) && isfinite(next_s2))
                value = oftype(value, NaN)
                break
            end
            value = section_output
        end
        @inbounds candidate_output[mode] = value
    end
end

@kernel function _finish_biquad!(output, committed_state, candidate_state,
    candidate_output, frame_status, input_failure, output_failure)
    index = @index(Global, Linear)
    @inbounds status = frame_status[1]
    if iszero(status)
        if index <= length(committed_state)
            @inbounds begin
                previous = committed_state[index]
                committed_state[index] = candidate_state[index]
                candidate_state[index] = previous
            end
        end
        if index <= length(output)
            @inbounds output[index] = candidate_output[index]
        end
    elseif status == input_failure || status == output_failure
        if index <= length(committed_state)
            @inbounds candidate_state[index] = committed_state[index]
        end
        if index <= length(candidate_output)
            @inbounds candidate_output[index] = zero(eltype(candidate_output))
        end
    end
end

@kernel function _reset_biquad!(committed_state, candidate_state, candidate_output)
    index = @index(Global, Linear)
    if index <= length(committed_state)
        @inbounds begin
            committed_state[index] = zero(eltype(committed_state))
            candidate_state[index] = zero(eltype(candidate_state))
        end
    end
    if index <= length(candidate_output)
        @inbounds candidate_output[index] = zero(eltype(candidate_output))
    end
end

struct BiquadCascade{Validate,K,O,I,S,C,H,V}
    validate_input!::Validate
    kernels::K
    output::O
    input::I
    frame_status::S
    coefficients::C
    committed_state::H
    candidate_state::H
    candidate_output::V
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(::CompleteTransition, device,
    prepared::PreparedAlgorithm{BiquadCascadeF32,<:BiquadCascadePlan{Float32},
        <:BiquadCascadeWorkspace{Float32}},
    outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == length(inputs) == 1 ||
        throw(ArgumentError("BiquadCascadeF32 capture requires one input and one output"))
    output, input = only(outputs), only(inputs)
    for (name, array) in ((:input, input), (:output, output))
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, name), array)
        _require_capture_array("biquad $name", array, device)
    end
    _require_capture_array("captured frame status", frame_status, device)
    _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE)
    plan, workspace = prepared.plan, prepared.workspace
    # Validate mutable source workspace dimensions and ownership before copying.
    FilterGraphAlgorithms._validate_biquad_cascade_buffers(
        zeros(Float32, plan.extent), workspace, plan, zeros(Float32, plan.extent))
    backend = KernelAbstractions.get_backend(output)
    kernels = (
        validate! = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE),
        stage! = _stage_biquad!(backend, PROJECTION_WORKGROUP_SIZE),
        finish! = _finish_biquad!(backend, PROJECTION_WORKGROUP_SIZE),
        reset! = _reset_biquad!(backend, PROJECTION_WORKGROUP_SIZE),
    )
    return BiquadCascade(kernels.validate!, kernels, output, input, frame_status,
        _capture_array(device, plan.coefficients),
        _capture_array(device, workspace.committed_state),
        _capture_array(device, workspace.candidate_state),
        _capture_array(device, workspace.candidate_output),
        _failure_code(node_index, BiquadCascadeNonFiniteInput),
        _failure_code(node_index, BiquadCascadeNonFiniteOutput))
end

preflight(::BiquadCascade) = nothing

function enqueue!(captured::BiquadCascade)
    k = captured.kernels
    captured.validate_input!(captured.frame_status, captured.input, captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE)
    k.stage!(captured.candidate_state, captured.candidate_output,
        captured.committed_state, captured.coefficients, captured.input,
        captured.frame_status; ndrange=length(captured.input))
    k.validate!(captured.frame_status, captured.candidate_output, captured.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE)
    k.finish!(captured.output, captured.committed_state, captured.candidate_state,
        captured.candidate_output, captured.frame_status,
        captured.input_failure, captured.output_failure;
        ndrange=length(captured.committed_state))
    return nothing
end

function complete!(captured::BiquadCascade, status::Int32)
    status == captured.input_failure && return BiquadCascadeNonFiniteInput
    status == captured.output_failure && return BiquadCascadeNonFiniteOutput
    return nothing
end

function reset!(captured::BiquadCascade)
    captured.kernels.reset!(captured.committed_state, captured.candidate_state,
        captured.candidate_output; ndrange=length(captured.committed_state))
    return captured
end

# Reuse only certified finite inputs on qualified backends. Intermediate/state
# rejection and the output scan remain active even when the input scan is skipped.
preserves_inputs(c::BiquadCascade) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
    !_capture_arrays_overlap(c.output, c.input)

finite_output(c::BiquadCascade, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

function prepare_validation(c::BiquadCascade, proofs::Tuple{Bool})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) &&
        only(proofs) || return c
    return BiquadCascade(_FiniteInput(), c.kernels, c.output, c.input, c.frame_status,
        c.coefficients, c.committed_state, c.candidate_state, c.candidate_output,
        c.input_failure, c.output_failure)
end
