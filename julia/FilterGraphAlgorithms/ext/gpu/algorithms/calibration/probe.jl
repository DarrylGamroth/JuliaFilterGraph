const CalibrationProbe = Union{ModalPushPullF32,ZonalPushPullF32,HadamardPushPullF32}

@kernel function _calibration_probe_command!(excitation, status, commands, row)
    index = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds excitation[index] = commands[index, row[1]]
    end
end

@kernel function _calibration_probe_advance!(pattern, polarity, completes, status, row, rows)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(status[1]))
        @inbounds begin
            step = row[1]
            pattern[1] = UInt64((step - 1) ÷ 2)
            polarity[1] = UInt8((step - 1) % 2)
            completes[1] = step == rows
            row[1] = step == rows ? 1 : step + 1
        end
    end
end

capture_support(::Type{<:CalibrationProbe}, ::DeviceType) = SupportsCapture()

function prepare_capture(
    ::CompleteTransition, device::DeviceType,
    prepared::PreparedAlgorithm{<:CalibrationProbe},
    outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing,
)
    backend = _calibration_bind(device, prepared, outputs, inputs, status)
    excitation, pattern, polarity, completes = outputs
    plan, workspace = prepared.plan, prepared.workspace
    FilterGraphAlgorithms._validate_calibration_probe_storage(
        outputs..., workspace, plan.commands, calibration_probe_pattern_count(plan),
        calibration_probe_corrector_coordinate_count(plan),
    )
    commands = _capture_array(device, plan.commands)
    initial_row = 2 * workspace.committed_pattern_index +
                  (workspace.committed_polarity === CalibrationProbePositive ? 1 : 2)
    row = _capture_array(device, Int[initial_row])
    command! = _calibration_probe_command!(backend, PROJECTION_WORKGROUP_SIZE)
    advance! = _calibration_probe_advance!(backend, 1)
    enqueue = function ()
        command!(excitation, status, commands, row; ndrange=length(excitation))
        advance!(pattern, polarity, completes, status, row, size(commands, 2); ndrange=1)
        return nothing
    end
    return CalibrationTransition(enqueue, () -> fill!(row, 1), ())
end
