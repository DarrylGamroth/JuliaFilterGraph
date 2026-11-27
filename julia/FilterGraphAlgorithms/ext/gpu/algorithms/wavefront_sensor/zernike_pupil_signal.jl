import FilterGraphAlgorithms:
    PreparedAlgorithm,
    ZernikePupilSignalCalibrationMismatch,
    ZernikePupilSignalF32,
    ZernikePupilSignalFailure,
    ZernikePupilSignalIncidenceFlux,
    ZernikePupilSignalInvalidNormalization,
    ZernikePupilSignalMeanValidFlux,
    ZernikePupilSignalNonFiniteInput,
    ZernikePupilSignalNonFiniteOutput,
    ZernikePupilSignalPlan,
    ZernikePupilSignalWorkspace,
    complete!,
    enqueue!,
    preflight,
    reset!

const ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES = 256

"""
Fuse validation, normalization, staging, and staged-output validation into one
workgroup.  The graph still clears status before this node and adopts staged
products in a following node, so no rejected frame can update caller outputs.
"""
@kernel function _stage_zernike_pupil_signal!(
    staged_divisor,
    staged_signal,
    frame_status,
    image,
    frame_calibration_signature,
    calibration_signature,
    selected_rows,
    selected_columns,
    reference_pupil_image,
    mean_valid_flux,
    incidence_flux,
    mismatch_failure,
    input_failure,
    normalization_failure,
    output_failure,
)
    local_index = @index(Local, Linear)
    local_failures = @localmem Int32 (ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES,)
    local_sums = @localmem eltype(image) (ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES,)

    if local_index == 1 && @inbounds(iszero(frame_status[1]))
        @inbounds frame_calibration_signature[1] == calibration_signature ||
            (frame_status[1] = mismatch_failure)
    end
    @synchronize

    input_failure_found = Int32(0)
    input_index = Int(local_index)
    while @inbounds(iszero(frame_status[1])) && input_index <= length(image)
        @inbounds if !isfinite(image[input_index])
            input_failure_found = Int32(1)
            break
        end
        input_index += ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES
    end
    @inbounds local_failures[local_index] = input_failure_found
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
    @synchronize
    if local_index == 1 && @inbounds(iszero(frame_status[1])) &&
       @inbounds(!iszero(local_failures[1]))
        @inbounds frame_status[1] = input_failure
    end
    @synchronize

    sum_flux = zero(eltype(image))
    if @inbounds(iszero(frame_status[1])) && mean_valid_flux
        selected = Int(local_index)
        while selected <= length(selected_rows)
            @inbounds sum_flux += image[selected_rows[selected], selected_columns[selected]]
            selected += ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES
        end
    end
    @inbounds local_sums[local_index] = sum_flux
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 128)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 64)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 32)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 16)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 8)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 4)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 2)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 1)
    @synchronize

    if local_index == 1 && @inbounds(iszero(frame_status[1]))
        divisor = incidence_flux
        if mean_valid_flux
            divisor = max(
                @inbounds(local_sums[1]) / eltype(image)(length(selected_rows)),
                eps(eltype(image)),
            )
        end
        if isfinite(divisor)
            @inbounds staged_divisor[1] = divisor
        else
            @inbounds frame_status[1] = normalization_failure
        end
    end
    @synchronize

    output_failure_found = Int32(0)
    selected = Int(local_index)
    if @inbounds(iszero(frame_status[1]))
        divisor = @inbounds staged_divisor[1]
        if divisor <= eps(eltype(staged_signal))
            while selected <= length(staged_signal)
                @inbounds staged_signal[selected] = zero(eltype(staged_signal))
                selected += ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES
            end
        else
            while selected <= length(staged_signal)
                @inbounds value =
                    image[selected_rows[selected], selected_columns[selected]] / divisor -
                    reference_pupil_image[selected_rows[selected], selected_columns[selected]]
                @inbounds staged_signal[selected] = value
                isfinite(value) || (output_failure_found = Int32(1))
                selected += ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES
            end
        end
    end
    @inbounds local_failures[local_index] = output_failure_found
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
    @synchronize
    if local_index == 1 && @inbounds(iszero(frame_status[1])) &&
       @inbounds(!iszero(local_failures[1]))
        @inbounds frame_status[1] = output_failure
    end
end

@kernel function _adopt_zernike_pupil_signal!(
    signal,
    normalization_divisor,
    staged_signal,
    staged_divisor,
    frame_status,
)
    index = @index(Global, Linear)
    if @inbounds(iszero(frame_status[1]))
        if index <= length(signal)
            @inbounds signal[index] = staged_signal[index]
        end
        if index == 1
            @inbounds normalization_divisor[1] = staged_divisor[1]
        end
    end
end

struct ZernikePupilSignal{
    Stage,
    Adopt,
    Signal,
    Divisor,
    StagedSignal,
    StagedDivisor,
    Status,
    Image,
    Signature,
    Coordinates,
    Reference,
}
    stage!::Stage
    adopt!::Adopt
    signal::Signal
    normalization_divisor::Divisor
    staged_signal::StagedSignal
    staged_divisor::StagedDivisor
    frame_status::Status
    image::Image
    frame_calibration_signature::Signature
    selected_rows::Coordinates
    selected_columns::Coordinates
    reference_pupil_image::Reference
    calibration_signature::UInt64
    mean_valid_flux::Bool
    incidence_flux::Float32
    mismatch_failure::Int32
    input_failure::Int32
    normalization_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        ZernikePupilSignalF32,
        <:ZernikePupilSignalPlan{Float32},
        <:ZernikePupilSignalWorkspace{Float32},
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 2 || throw(
        ArgumentError("ZernikePupilSignalF32 capture requires two outputs"),
    )
    length(inputs) == 2 || throw(
        ArgumentError("ZernikePupilSignalF32 capture requires two inputs"),
    )
    signal, normalization_divisor = outputs
    image, frame_calibration_signature = inputs
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :image),
        image,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("frame-calibration-signature")),
        frame_calibration_signature,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :signal),
        signal,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("normalization-divisor")),
        normalization_divisor,
    )
    (Base.mightalias(signal, normalization_divisor) ||
     Base.mightalias(signal, image) ||
     Base.mightalias(normalization_divisor, image)) && throw(
        ArgumentError("Zernike pupil-signal caller arrays must not overlap"),
    )
    _require_capture_array("complete Zernike pupil image input", image, device)
    _require_capture_array("Zernike frame calibration signature input", frame_calibration_signature, device)
    _require_capture_array("Zernike pupil signal output", signal, device)
    _require_capture_array("Zernike normalization-divisor output", normalization_divisor, device)
    _require_capture_array("captured frame status", frame_status, device)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))

    plan = prepared.plan
    image_rows, _ = plan.image_size
    selected_rows = Int32[rem(offset, image_rows) + 1 for offset in plan.selected_offsets]
    selected_columns = Int32[div(offset, image_rows) + 1 for offset in plan.selected_offsets]
    backend = KernelAbstractions.get_backend(signal)
    staged_signal = _capture_zeros(device, Float32, size(signal)...)
    staged_divisor = _capture_zeros(device, Float32, 1)
    return ZernikePupilSignal(
        _stage_zernike_pupil_signal!(
            backend,
            ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES,
        ),
        _adopt_zernike_pupil_signal!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        signal,
        normalization_divisor,
        staged_signal,
        staged_divisor,
        frame_status,
        image,
        frame_calibration_signature,
        _capture_array(device, selected_rows),
        _capture_array(device, selected_columns),
        _capture_array(device, plan.reference_pupil_image),
        plan.calibration_signature,
        plan.normalization === ZernikePupilSignalMeanValidFlux,
        plan.incidence_flux,
        _failure_code(node_index, ZernikePupilSignalCalibrationMismatch),
        _failure_code(node_index, ZernikePupilSignalNonFiniteInput),
        _failure_code(node_index, ZernikePupilSignalInvalidNormalization),
        _failure_code(node_index, ZernikePupilSignalNonFiniteOutput),
    )
end

preflight(::ZernikePupilSignal) = nothing

function enqueue!(captured::ZernikePupilSignal)
    captured.stage!(
        captured.staged_divisor,
        captured.staged_signal,
        captured.frame_status,
        captured.image,
        captured.frame_calibration_signature,
        captured.calibration_signature,
        captured.selected_rows,
        captured.selected_columns,
        captured.reference_pupil_image,
        captured.mean_valid_flux,
        captured.incidence_flux,
        captured.mismatch_failure,
        captured.input_failure,
        captured.normalization_failure,
        captured.output_failure;
        ndrange=ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_LANES,
    )
    captured.adopt!(
        captured.signal,
        captured.normalization_divisor,
        captured.staged_signal,
        captured.staged_divisor,
        captured.frame_status;
        ndrange=length(captured.signal),
    )
    return nothing
end

function complete!(captured::ZernikePupilSignal, status::Int32)
    status == captured.mismatch_failure && return ZernikePupilSignalCalibrationMismatch
    status == captured.input_failure && return ZernikePupilSignalNonFiniteInput
    status == captured.normalization_failure && return ZernikePupilSignalInvalidNormalization
    status == captured.output_failure && return ZernikePupilSignalNonFiniteOutput
    return nothing
end

reset!(captured::ZernikePupilSignal) = captured
