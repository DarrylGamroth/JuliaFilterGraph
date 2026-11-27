import FilterGraphAlgorithms:
    CurvaturePairedChannelF32,
    CurvaturePairedChannelPlan,
    CurvaturePairedImageF32,
    CurvaturePairedImagePlan,
    CurvaturePairedSignalCalibrationMismatch,
    CurvaturePairedSignalFailure,
    CurvaturePairedSignalNonFiniteInput,
    CurvaturePairedSignalNonFiniteOutput,
    CurvaturePairedSignalWorkspace,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!

const CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE = 256

@kernel function _validate_curvature_paired_signal_signature!(
    frame_status,
    frame_calibration_signature,
    calibration_signature,
    mismatch_failure,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(frame_calibration_signature[1] != calibration_signature)
        @inbounds KernelAbstractions.@atomicreplace(
            frame_status[1], Int32(0) => mismatch_failure,
        )
    end
end

@kernel function _validate_curvature_paired_signal_input!(
    frame_status,
    observation,
    input_failure,
)
    index = @index(Global, Linear)
    if index <= length(observation)
        @inbounds if !isfinite(observation[index])
            KernelAbstractions.@atomicreplace(
                frame_status[1], Int32(0) => input_failure,
            )
        end
    end
end

@kernel function _stage_curvature_paired_image!(
    staged_signal,
    frame_status,
    observation,
    support,
    reference_signal,
    positive_scale,
    negative_scale,
    positive_origin_row,
    positive_origin_column,
    negative_origin_row,
    negative_origin_column,
    row_reduction,
    column_reduction,
    signal_rows,
    signal_columns,
    output_failure,
)
    output_index = @index(Global, Linear)
    if output_index <= length(staged_signal)
        value = zero(eltype(staged_signal))
        @inbounds if support[output_index]
            row = signal_rows[output_index]
            column = signal_columns[output_index]
            positive = zero(eltype(staged_signal))
            negative = zero(eltype(staged_signal))
            for jj in 1:column_reduction, ii in 1:row_reduction
                positive += observation[
                    positive_origin_row + (row - 1) * row_reduction + ii,
                    positive_origin_column + (column - 1) * column_reduction + jj,
                ]
                negative += observation[
                    negative_origin_row + (row - 1) * row_reduction + ii,
                    negative_origin_column + (column - 1) * column_reduction + jj,
                ]
            end
            positive *= positive_scale
            negative *= negative_scale
            value = FilterGraphAlgorithms._curvature_paired_signal_value(
                positive, negative, reference_signal[output_index],
            )
        end
        @inbounds staged_signal[output_index] = value
        if !isfinite(value)
            @inbounds KernelAbstractions.@atomicreplace(
                frame_status[1], Int32(0) => output_failure,
            )
        end
    end
end

@kernel function _stage_curvature_paired_channel!(
    staged_signal,
    frame_status,
    observation,
    support,
    reference_signal,
    positive_scale,
    negative_scale,
    positive_branch_row,
    negative_branch_row,
    channel_offsets,
    output_failure,
)
    output_index = @index(Global, Linear)
    if output_index <= length(staged_signal)
        value = zero(eltype(staged_signal))
        @inbounds if support[output_index]
            channel = channel_offsets[output_index] + 1
            positive = observation[positive_branch_row + 1, channel] * positive_scale
            negative = observation[negative_branch_row + 1, channel] * negative_scale
            value = FilterGraphAlgorithms._curvature_paired_signal_value(
                positive, negative, reference_signal[output_index],
            )
        end
        @inbounds staged_signal[output_index] = value
        if !isfinite(value)
            @inbounds KernelAbstractions.@atomicreplace(
                frame_status[1], Int32(0) => output_failure,
            )
        end
    end
end

@kernel function _adopt_curvature_paired_signal!(
    signal,
    staged_signal,
    frame_status,
)
    index = @index(Global, Linear)
    if index <= length(signal) && @inbounds(iszero(frame_status[1]))
        @inbounds signal[index] = staged_signal[index]
    end
end

struct CurvaturePairedImageSignal{
    ValidateSignature,
    ValidateInput,
    Stage,
    Adopt,
    Signal,
    Status,
    Observation,
    Signature,
    Support,
    Reference,
    Coordinates,
}
    validate_signature!::ValidateSignature
    validate_input!::ValidateInput
    stage!::Stage
    adopt!::Adopt
    signal::Signal
    staged_signal::Signal
    frame_status::Status
    observation::Observation
    frame_calibration_signature::Signature
    support::Support
    reference_signal::Reference
    signal_rows::Coordinates
    signal_columns::Coordinates
    branch_scales::NTuple{2,Float32}
    branch_origins::NTuple{2,NTuple{2,Int}}
    row_reduction::Int
    column_reduction::Int
    calibration_signature::UInt64
    mismatch_failure::Int32
    input_failure::Int32
    output_failure::Int32
end

struct CurvaturePairedChannelSignal{
    ValidateSignature,
    ValidateInput,
    Stage,
    Adopt,
    Signal,
    Status,
    Observation,
    Signature,
    Support,
    Reference,
    Offsets,
}
    validate_signature!::ValidateSignature
    validate_input!::ValidateInput
    stage!::Stage
    adopt!::Adopt
    signal::Signal
    staged_signal::Signal
    frame_status::Status
    observation::Observation
    frame_calibration_signature::Signature
    support::Support
    reference_signal::Reference
    channel_offsets::Offsets
    branch_scales::NTuple{2,Float32}
    branch_rows::NTuple{2,Int}
    calibration_signature::UInt64
    mismatch_failure::Int32
    input_failure::Int32
    output_failure::Int32
end


function _require_curvature_paired_capture(
    device,
    prepared,
    observation_format_name::Symbol,
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 1 || throw(
        ArgumentError("captured Curvature paired signal requires one output"),
    )
    length(inputs) == 2 || throw(
        ArgumentError("captured Curvature paired signal requires two inputs"),
    )
    signal = only(outputs)
    observation, frame_calibration_signature = inputs
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, observation_format_name), observation,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(
            prepared, Symbol("frame-calibration-signature"),
        ),
        frame_calibration_signature,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :signal),
        signal,
    )
    Base.mightalias(signal, observation) && throw(
        ArgumentError("Curvature paired-signal input and output must not overlap"),
    )
    _require_capture_array("complete Curvature observation input", observation, device)
    _require_capture_array(
        "Curvature frame calibration signature input",
        frame_calibration_signature,
        device,
    )
    _require_capture_array("Curvature paired signal output", signal, device)
    _require_capture_array("captured frame status", frame_status, device)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))
    return signal, observation, frame_calibration_signature
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        CurvaturePairedImageF32,
        <:CurvaturePairedImagePlan{Float32},
        <:CurvaturePairedSignalWorkspace{Float32},
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    signal, observation, frame_calibration_signature =
        _require_curvature_paired_capture(
            device, prepared, :image, outputs, inputs, frame_status, node_index,
        )
    plan = prepared.plan
    row_reduction = div(plan.branch_size[1], plan.signal_size[1])
    column_reduction = div(plan.branch_size[2], plan.signal_size[2])
    signal_rows = Int32[
        rem(index - 1, plan.signal_size[1]) + 1 for
        index in 1:Base.checked_mul(plan.signal_size...)
    ]
    signal_columns = Int32[
        div(index - 1, plan.signal_size[1]) + 1 for
        index in 1:Base.checked_mul(plan.signal_size...)
    ]
    backend = KernelAbstractions.get_backend(signal)
    return CurvaturePairedImageSignal(
        _validate_curvature_paired_signal_signature!(backend, 1),
        _validate_curvature_paired_signal_input!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        _stage_curvature_paired_image!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        _adopt_curvature_paired_signal!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        signal,
        _capture_zeros(device, Float32, size(signal)...),
        frame_status,
        observation,
        frame_calibration_signature,
        _capture_array(device, plan.support),
        _capture_array(device, plan.reference_signal),
        _capture_array(device, signal_rows),
        _capture_array(device, signal_columns),
        plan.branch_scales,
        plan.branch_origins,
        row_reduction,
        column_reduction,
        plan.calibration_signature,
        _failure_code(node_index, CurvaturePairedSignalCalibrationMismatch),
        _failure_code(node_index, CurvaturePairedSignalNonFiniteInput),
        _failure_code(node_index, CurvaturePairedSignalNonFiniteOutput),
    )
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        CurvaturePairedChannelF32,
        <:CurvaturePairedChannelPlan{Float32},
        <:CurvaturePairedSignalWorkspace{Float32},
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    signal, observation, frame_calibration_signature =
        _require_curvature_paired_capture(
            device, prepared, :channels, outputs, inputs, frame_status, node_index,
        )
    plan = prepared.plan
    backend = KernelAbstractions.get_backend(signal)
    return CurvaturePairedChannelSignal(
        _validate_curvature_paired_signal_signature!(backend, 1),
        _validate_curvature_paired_signal_input!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        _stage_curvature_paired_channel!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        _adopt_curvature_paired_signal!(
            backend, CURVATURE_PAIRED_SIGNAL_WORKGROUP_SIZE,
        ),
        signal,
        _capture_zeros(device, Float32, size(signal)...),
        frame_status,
        observation,
        frame_calibration_signature,
        _capture_array(device, plan.support),
        _capture_array(device, plan.reference_signal),
        _capture_array(device, Int32.(plan.channel_offsets)),
        plan.branch_scales,
        plan.branch_rows,
        plan.calibration_signature,
        _failure_code(node_index, CurvaturePairedSignalCalibrationMismatch),
        _failure_code(node_index, CurvaturePairedSignalNonFiniteInput),
        _failure_code(node_index, CurvaturePairedSignalNonFiniteOutput),
    )
end

preflight(::CurvaturePairedImageSignal) = nothing
preflight(::CurvaturePairedChannelSignal) = nothing

function enqueue!(captured::CurvaturePairedImageSignal)
    captured.validate_signature!(
        captured.frame_status,
        captured.frame_calibration_signature,
        captured.calibration_signature,
        captured.mismatch_failure;
        ndrange=1,
    )
    captured.validate_input!(
        captured.frame_status,
        captured.observation,
        captured.input_failure;
        ndrange=length(captured.observation),
    )
    positive_origin, negative_origin = captured.branch_origins
    captured.stage!(
        captured.staged_signal,
        captured.frame_status,
        captured.observation,
        captured.support,
        captured.reference_signal,
        captured.branch_scales[1],
        captured.branch_scales[2],
        positive_origin[1],
        positive_origin[2],
        negative_origin[1],
        negative_origin[2],
        captured.row_reduction,
        captured.column_reduction,
        captured.signal_rows,
        captured.signal_columns,
        captured.output_failure;
        ndrange=length(captured.staged_signal),
    )
    _adopt_curvature_paired_signal!(captured)
    return nothing
end

function enqueue!(captured::CurvaturePairedChannelSignal)
    captured.validate_signature!(
        captured.frame_status,
        captured.frame_calibration_signature,
        captured.calibration_signature,
        captured.mismatch_failure;
        ndrange=1,
    )
    captured.validate_input!(
        captured.frame_status,
        captured.observation,
        captured.input_failure;
        ndrange=length(captured.observation),
    )
    captured.stage!(
        captured.staged_signal,
        captured.frame_status,
        captured.observation,
        captured.support,
        captured.reference_signal,
        captured.branch_scales[1],
        captured.branch_scales[2],
        captured.branch_rows[1],
        captured.branch_rows[2],
        captured.channel_offsets,
        captured.output_failure;
        ndrange=length(captured.staged_signal),
    )
    _adopt_curvature_paired_signal!(captured)
    return nothing
end

function _adopt_curvature_paired_signal!(captured::Union{
    CurvaturePairedImageSignal,
    CurvaturePairedChannelSignal,
})
    captured.adopt!(
        captured.signal,
        captured.staged_signal,
        captured.frame_status;
        ndrange=length(captured.signal),
    )
    return nothing
end

function complete!(
    captured::Union{
        CurvaturePairedImageSignal,
        CurvaturePairedChannelSignal,
    },
    status::Int32,
)
    status == captured.mismatch_failure && return CurvaturePairedSignalCalibrationMismatch
    status == captured.input_failure && return CurvaturePairedSignalNonFiniteInput
    status == captured.output_failure && return CurvaturePairedSignalNonFiniteOutput
    return nothing
end

reset!(captured::CurvaturePairedImageSignal) = captured
reset!(captured::CurvaturePairedChannelSignal) = captured
