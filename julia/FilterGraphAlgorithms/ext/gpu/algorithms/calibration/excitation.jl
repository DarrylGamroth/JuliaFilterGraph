const DocrimeExcitation = Union{DocrimeExcitationF32,DocrimeBinaryExcitationF32}

@kernel function _calibration_excitation!(output, status, amplitudes, state, excite)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        draw_index = UInt64(index - 1)
        @inbounds output[index] = excite(eltype(output), amplitudes[index], state[1], draw_index)
    end
end

@kernel function _calibration_excitation_advance!(status, state, count)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds state[1] = splitmix64_advance(state[1], count)
    end
end

capture_support(::Type{<:DocrimeExcitation}, ::DeviceType) = SupportsCapture()

function _prepare_calibration_excitation(
    ::DocrimeBinaryExcitationPlan, backend, compute!, output, status, amplitudes, state, node,
)
    excitation = () -> compute!(
        output, status, amplitudes, state, _docrime_binary_value; ndrange=length(output),
    )
    return excitation, ()
end

function _prepare_calibration_excitation(
    ::DocrimeExcitationPlan, backend, compute!, output, status, amplitudes, state, node,
)
    staged = similar(output)
    failures = _calibration_failures(node, DocrimeExcitationNonFiniteCalculation)
    validate = _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE)
    publish = _identity_projection!(backend, PROJECTION_WORKGROUP_SIZE)
    excitation = function ()
        compute!(staged, status, amplitudes, state, _docrime_continuous_value; ndrange=length(output))
        _calibration_validate!(validate, status, staged, first(only(failures)))
        _calibration_publish!(publish, status, output, staged)
    end
    return excitation, failures
end

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{<:DocrimeExcitation},
    outputs::Tuple, inputs::Tuple, status, node::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    output = only(outputs)
    plan, workspace = prepared.plan, prepared.workspace
    FilterGraphAlgorithms._validate_docrime_execution(output, workspace, plan)
    amplitudes = _capture_array(device, plan.amplitudes)
    state = _capture_array(device, UInt64[workspace.committed_random_state])
    initial_state = workspace.initial_random_state
    compute! = _calibration_excitation!(backend, PROJECTION_WORKGROUP_SIZE)
    excitation, failures = _prepare_calibration_excitation(
        plan, backend, compute!, output, status, amplitudes, state, node,
    )
    advance! = _calibration_excitation_advance!(backend, 1)
    enqueue = function ()
        excitation()
        advance!(status, state, UInt64(length(output)); ndrange=1)
        return nothing
    end
    return CalibrationTransition(enqueue, () -> fill!(state, initial_state), failures)
end
