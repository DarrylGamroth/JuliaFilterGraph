import FilterGraphAlgorithms:
    PyramidImageF32,
    PyramidImageIncidenceFlux,
    PyramidImageInvalidNormalization,
    PYRAMID_IMAGE_NORMALIZATION_LANES,
    PyramidImageNonFiniteInput,
    PyramidImageNonFiniteOutput,
    PyramidImagePlan,
    PyramidImageWorkspace,
    PreparedAlgorithm,
    complete!,
    enqueue!,
    preflight,
    reset!,
    _normalized_pupil_difference

@kernel function _validate_pyramid_image_input!(
    frame_status, image, rows, columns, input_failure,
)
    local_index = @index(Local, Linear)
    local_failures = @localmem Int32 (PYRAMID_IMAGE_NORMALIZATION_LANES,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    failure = Int32(0)
    coordinate = Int(local_index)
    while frame_valid && coordinate <= length(rows)
        @inbounds isfinite(image[rows[coordinate], columns[coordinate]]) ||
            (failure = Int32(1))
        coordinate += PYRAMID_IMAGE_NORMALIZATION_LANES
    end
    @inbounds local_failures[local_index] = failure
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
    if frame_valid && local_index == 1 && @inbounds(!iszero(local_failures[1]))
        @inbounds frame_status[1] = input_failure
    end
end

@kernel function _normalize_pyramid_image!(
    staged_divisor, frame_status, image, selected_rows, selected_columns,
    origin_rows, origin_columns, mean_valid_flux, incidence_flux,
    normalization_failure,
)
    local_index = @index(Local, Linear)
    local_sums = @localmem eltype(image) (PYRAMID_IMAGE_NORMALIZATION_LANES,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    sum_flux = zero(eltype(image))
    if frame_valid && mean_valid_flux
        selected = Int(local_index)
        while selected <= length(selected_rows)
            row = @inbounds selected_rows[selected]
            column = @inbounds selected_columns[selected]
            @inbounds coordinate_flux =
                image[origin_rows[1] + row, origin_columns[1] + column] +
                image[origin_rows[2] + row, origin_columns[2] + column] +
                image[origin_rows[3] + row, origin_columns[3] + column] +
                image[origin_rows[4] + row, origin_columns[4] + column]
            sum_flux += coordinate_flux
            selected += PYRAMID_IMAGE_NORMALIZATION_LANES
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

    if frame_valid && local_index == 1
        divisor = incidence_flux
        if mean_valid_flux
            sum_flux = @inbounds local_sums[1]
            divisor = sum_flux / eltype(image)(length(selected_rows))
        end
        if isfinite(divisor) && divisor > zero(eltype(image))
            @inbounds staged_divisor[1] = divisor
        else
            @inbounds frame_status[1] = normalization_failure
        end
    end
end

@kernel function _stage_pyramid_image_signal!(
    staged_signal, frame_status, staged_divisor, image, selected_rows,
    selected_columns, origin_rows, origin_columns, reference_signal, optical_gain,
)
    selected = @index(Global, Linear)
    if selected <= size(staged_signal, 1) && @inbounds(iszero(frame_status[1]))
        row = @inbounds selected_rows[selected]
        column = @inbounds selected_columns[selected]
        @inbounds begin
            q1 = image[origin_rows[1] + row, origin_columns[1] + column]
            q2 = image[origin_rows[2] + row, origin_columns[2] + column]
            q3 = image[origin_rows[3] + row, origin_columns[3] + column]
            q4 = image[origin_rows[4] + row, origin_columns[4] + column]
            divisor = staged_divisor[1]
            staged_signal[selected, 1] = optical_gain[selected, 1] * (
                _normalized_pupil_difference(q1, q2, q4, q3, divisor) - reference_signal[selected, 1]
            )
            staged_signal[selected, 2] = optical_gain[selected, 2] * (
                _normalized_pupil_difference(q1, q4, q2, q3, divisor) - reference_signal[selected, 2]
            )
        end
    end
end

@kernel function _adopt_pyramid_image!(
    signal, normalization_divisor, staged_signal, staged_divisor, frame_status,
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

struct PyramidImage{
    Validate,
    Normalize,
    Stage,
    ValidateOutput,
    Adopt,
    Signal,
    Divisor,
    StagedSignal,
    StagedDivisor,
    Status,
    Image,
    Coordinates,
    Origins,
    Reference,
    Gain,
}
    validate!::Validate
    normalize!::Normalize
    stage!::Stage
    validate_output!::ValidateOutput
    adopt!::Adopt
    signal::Signal
    normalization_divisor::Divisor
    staged_signal::StagedSignal
    staged_divisor::StagedDivisor
    frame_status::Status
    image::Image
    validation_rows::Coordinates
    validation_columns::Coordinates
    selected_rows::Coordinates
    selected_columns::Coordinates
    origin_rows::Origins
    origin_columns::Origins
    reference_signal::Reference
    optical_gain::Gain
    mean_valid_flux::Bool
    incidence_flux::Float32
    input_failure::Int32
    normalization_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        PyramidImageF32,
        <:PyramidImagePlan{Float32},
        <:PyramidImageWorkspace{Float32},
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 2 || throw(ArgumentError("PyramidImageF32 capture requires two outputs"))
    length(inputs) == 1 || throw(ArgumentError("PyramidImageF32 capture requires one input"))
    signal, normalization_divisor = outputs
    image = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :image),
        image,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :signal),
        signal,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("normalization-divisor")),
        normalization_divisor,
    )
    _require_capture_array("calibrated pyramid image input", image, device)
    _require_capture_array("pyramid I4Q signal output", signal, device)
    _require_capture_array("pyramid normalization-divisor output", normalization_divisor, device)
    _require_capture_array("captured frame status", frame_status, device)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))

    plan = prepared.plan
    selected_rows = Int32[
        rem(offset, plan.pupil_size[1]) + 1 for
        offset in plan.selected_offsets
    ]
    selected_columns = Int32[
        div(offset, plan.pupil_size[1]) + 1 for
        offset in plan.selected_offsets
    ]
    validation_rows = Int32[]
    validation_columns = Int32[]
    sizehint!(validation_rows, 4 * prod(plan.pupil_size))
    sizehint!(validation_columns, 4 * prod(plan.pupil_size))
    for (origin_row, origin_column) in plan.origins
        for column in 1:plan.pupil_size[2], row in 1:plan.pupil_size[1]
            push!(validation_rows, origin_row + row)
            push!(validation_columns, origin_column + column)
        end
    end
    backend = KernelAbstractions.get_backend(signal)
    staged_signal = _capture_zeros(device, Float32, size(signal)...)
    staged_divisor = _capture_zeros(device, Float32, 1)
    return PyramidImage(
        _validate_pyramid_image_input!(
            backend,
            PYRAMID_IMAGE_NORMALIZATION_LANES,
        ),
        _normalize_pyramid_image!(backend, PYRAMID_IMAGE_NORMALIZATION_LANES),
        _stage_pyramid_image_signal!(backend, PROJECTION_WORKGROUP_SIZE),
        _validate_projection_values!(backend, VALIDATION_WORKGROUP_SIZE),
        _adopt_pyramid_image!(backend, PROJECTION_WORKGROUP_SIZE),
        signal, normalization_divisor, staged_signal, staged_divisor, frame_status, image,
        _capture_array(device, validation_rows), _capture_array(device, validation_columns),
        _capture_array(device, selected_rows), _capture_array(device, selected_columns),
        _capture_array(device, Int32[first(origin) for origin in plan.origins]),
        _capture_array(device, Int32[last(origin) for origin in plan.origins]),
        _capture_array(device, plan.reference_signal), _capture_array(device, plan.optical_gain),
        plan.normalization !== PyramidImageIncidenceFlux, plan.incidence_flux,
        _failure_code(node_index, PyramidImageNonFiniteInput),
        _failure_code(node_index, PyramidImageInvalidNormalization),
        _failure_code(node_index, PyramidImageNonFiniteOutput),
    )
end

preflight(::PyramidImage) = nothing

function enqueue!(captured::PyramidImage)
    captured.validate!(
        captured.frame_status,
        captured.image,
        captured.validation_rows,
        captured.validation_columns,
        captured.input_failure;
        ndrange=PYRAMID_IMAGE_NORMALIZATION_LANES,
    )
    captured.normalize!(
        captured.staged_divisor,
        captured.frame_status,
        captured.image,
        captured.selected_rows,
        captured.selected_columns,
        captured.origin_rows,
        captured.origin_columns,
        captured.mean_valid_flux,
        captured.incidence_flux,
        captured.normalization_failure;
        ndrange=PYRAMID_IMAGE_NORMALIZATION_LANES,
    )
    captured.stage!(
        captured.staged_signal,
        captured.frame_status,
        captured.staged_divisor,
        captured.image,
        captured.selected_rows,
        captured.selected_columns,
        captured.origin_rows,
        captured.origin_columns,
        captured.reference_signal,
        captured.optical_gain;
        ndrange=size(captured.staged_signal, 1),
    )
    captured.validate_output!(
        captured.frame_status,
        captured.staged_signal,
        captured.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
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

function complete!(captured::PyramidImage, status::Int32)
    status == captured.input_failure && return PyramidImageNonFiniteInput
    status == captured.normalization_failure && return PyramidImageInvalidNormalization
    status == captured.output_failure && return PyramidImageNonFiniteOutput
    return nothing
end

reset!(captured::PyramidImage) = captured
