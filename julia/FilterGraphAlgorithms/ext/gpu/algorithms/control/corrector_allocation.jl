import FilterGraphAlgorithms: CorrectorAllocationF32, CorrectorAllocationNonFiniteInput,
    CorrectorAllocationNonFiniteOutput, _pdm_current

@kernel function _corrector_projection!(output, status, matrix, input, subtract)
    row = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            value = subtract ? output[row] : 0f0
            for column in axes(matrix, 2)
                coefficient = matrix[row, column]
                value = fma(subtract ? -coefficient : coefficient, input[column], value)
            end
            output[row] = value
        end
    end
end

@kernel function _stage_corrector_command!(commanded, candidate_current, status, filtered,
        held, current, counter, lower, upper, slew, slew_limits)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            if iszero(counter[1])
                requested = filtered[i]
                target = clamp(requested, lower[i], upper[i])
                limit = slew_limits === nothing ? 0f0 : slew_limits[i]
                next = _pdm_current(slew, current[i], target, limit)
                candidate_current[i] = next
                value = clamp(next, lower[i], upper[i])
                commanded[i] = isfinite(next) && isfinite(requested - value) ? value : NaN32
            else
                commanded[i] = held[i]
                candidate_current[i] = current[i]
            end
        end
    end
end

@kernel function _finish_corrector_stage!(commands, held, status, commanded, offset,
        filter_state, candidate_filter, current, candidate_current, counter, period)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        if i <= length(filter_state)
            @inbounds filter_state[i] = candidate_filter[i]
        end
        if i <= length(held)
            @inbounds begin
                value = commanded[i]
                held[i] = value
                current[i] = candidate_current[i]
                commands[offset + i] = value
            end
        end
        if i == 1
            @inbounds counter[1] = counter[1] == period - 1 ? 0 : counter[1] + 1
        end
    end
end

# Reuse the committed slew value on held ticks. Keep the command in candidate
# storage until the final residual of every stage has passed validation.
function _prepare_corrector_stage(device, stage, workspace, commands, residual, status,
        offset, output_failure)
    requested = _capture_array(device, workspace.requested)
    filtered = similar(requested)
    commanded = similar(requested)
    held = _capture_array(device, workspace.held)
    filter_state = _capture_array(device, workspace.filter.committed_state)
    candidate_filter = similar(filter_state)
    current = _capture_array(device, workspace.conditioner.committed_current)
    candidate_current = similar(current)
    counter = _capture_array(device, [workspace.counter])
    initial_current = _capture_array(device, workspace.conditioner.initial_current)
    limits = stage.conditioner.slew_limits
    slew_limits = limits === nothing ? nothing : _capture_array(device, limits)
    lower = _capture_array(device, stage.conditioner.lower_limits)
    upper = _capture_array(device, stage.conditioner.upper_limits)
    to_command = _capture_array(device, stage.to_command)
    to_controller = _capture_array(device, stage.to_controller)
    coefficients = _capture_array(device, stage.filter.coefficients)
    n = length(requested)
    stage_kernels = (
        _control_kernel(_corrector_projection!,
            (requested, status, to_command, residual, false), n),
        _control_kernel(_stage_biquad!,
            (candidate_filter, filtered, filter_state, coefficients, requested, status), n),
        _control_validate(status, filtered, output_failure),
        _control_kernel(_stage_corrector_command!,
            (commanded, candidate_current, status, filtered, held, current, counter,
                lower, upper, stage.conditioner.slew, slew_limits), n),
        _control_validate(status, commanded, output_failure),
        _control_kernel(_corrector_projection!,
            (residual, status, to_controller, commanded, true), length(residual)),
        _control_validate(status, residual, output_failure),
    )
    finish_kernels = (
        _control_kernel(_finish_corrector_stage!,
            (commands, held, status, commanded, offset, filter_state, candidate_filter,
                current, candidate_current, counter, stage.update_period), length(filter_state)),
    )
    state = ((held, nothing), (filter_state, nothing), (current, initial_current), (counter, nothing))
    return stage_kernels, finish_kernels, state
end

capture_support(::Type{CorrectorAllocationF32}, ::DeviceType) = SupportsCapture()

function prepare_capture(::CompleteTransition, device::DeviceType,
        prepared::PreparedAlgorithm{CorrectorAllocationF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32)
    _check_control_capture(device, prepared, outputs, inputs, status)
    commands, feedback = outputs
    demand = only(inputs)
    FilterGraphAlgorithms._validate_corrector_buffers(commands, feedback, prepared.workspace, prepared.plan, demand)
    residual = similar(feedback)
    failures = _control_failures(node_index, (CorrectorAllocationNonFiniteInput,
        CorrectorAllocationNonFiniteOutput))
    input_failure, output_failure = map(first, failures)
    kernels = (_control_validate(status, demand, input_failure),
        _control_copy(residual, status, demand))
    finish = ()
    state = ()
    offset = 0
    for (stage, workspace) in zip(prepared.plan.stages, prepared.workspace.stages)
        stage_kernels, finish_kernels, stage_state = _prepare_corrector_stage(device,
            stage, workspace, commands, residual, status, offset, output_failure)
        kernels = (kernels..., stage_kernels...)
        finish = (finish..., finish_kernels...)
        state = (state..., stage_state...)
        offset += length(workspace.held)
    end
    return ControlCapture((kernels..., finish..., _control_copy(feedback, status, residual)),
        failures, state)
end
