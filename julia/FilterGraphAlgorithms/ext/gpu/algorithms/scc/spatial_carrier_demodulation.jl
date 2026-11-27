import FilterGraphAlgorithms:
    ChoppedImageComplete,
    ChoppedImagePriming,
    PreparedAlgorithm,
    SccSpatialCarrierDemodulationF32,
    SccSpatialCarrierDemodulationIncompletePair,
    SccSpatialCarrierDemodulationInsufficientSupport,
    SccSpatialCarrierDemodulationInvalidPairStatus,
    SccSpatialCarrierDemodulationPlan,
    complete!,
    enqueue!,
    preflight,
    reset!

@kernel function _prepare_scc_spatial_carrier!(
    masked_carrier,
    mask,
    frame_status,
    difference,
    difference_validity,
    reference_difference,
    static_validity,
    carrier_conjugate,
)
    pixel = @index(Global, Linear)
    if pixel <= length(difference) && @inbounds(iszero(frame_status[1]))
        @inbounds valid = static_validity[pixel] && difference_validity[pixel]
        value = zero(eltype(masked_carrier))
        if valid
            @inbounds residual = difference[pixel] - reference_difference[pixel]
            valid = isfinite(residual)
            valid && @inbounds(value = residual * carrier_conjugate[pixel])
        end
        @inbounds begin
            mask[pixel] = valid ? one(eltype(mask)) : zero(eltype(mask))
            masked_carrier[pixel] = valid ? value : zero(eltype(masked_carrier))
        end
    end
end

@kernel function _scc_spatial_horizontal!(
    horizontal_carrier,
    horizontal_support,
    frame_status,
    masked_carrier,
    mask,
    fir,
    pixel_rows,
    pixel_columns,
    rows,
    columns,
    radius,
)
    target = @index(Global, Linear)
    if target <= length(horizontal_carrier) && @inbounds(iszero(frame_status[1]))
        @inbounds begin
            row0 = pixel_rows[target]
            column0 = pixel_columns[target]
        end
        carrier = zero(eltype(horizontal_carrier))
        support = zero(eltype(horizontal_support))
        for tap in -radius:radius
            source_column0 = clamp(column0 + tap, 0, columns - 1)
            source = row0 + source_column0 * rows + 1
            @inbounds coefficient = fir[tap + radius + 1]
            @inbounds begin
                carrier += coefficient * masked_carrier[source]
                support += coefficient * mask[source]
            end
        end
        @inbounds begin
            horizontal_carrier[target] = carrier
            horizontal_support[target] = support
        end
    end
end

@kernel function _scc_spatial_selected!(
    measurements,
    weights,
    measurement_validity,
    valid_count,
    frame_status,
    horizontal_carrier,
    horizontal_support,
    fir,
    selected_rows,
    selected_columns,
    rows,
    radius,
    minimum_support,
)
    selected = @index(Global, Linear)
    if selected <= length(selected_rows) && @inbounds(iszero(frame_status[1]))
        @inbounds begin
            row0 = selected_rows[selected]
            column0 = selected_columns[selected]
        end
        carrier = zero(eltype(horizontal_carrier))
        support = zero(eltype(horizontal_support))
        for tap in -radius:radius
            source_row0 = clamp(row0 + tap, 0, rows - 1)
            source = source_row0 + column0 * rows + 1
            @inbounds coefficient = fir[tap + radius + 1]
            @inbounds begin
                carrier += coefficient * horizontal_carrier[source]
                support += coefficient * horizontal_support[source]
            end
        end
        valid = isfinite(support) && support >= minimum_support
        value = zero(eltype(horizontal_carrier))
        if valid
            value = carrier / support
            valid = isfinite(real(value)) && isfinite(imag(value))
        end
        imaginary = selected + length(selected_rows)
        weight = valid ? max(support, zero(eltype(weights))) : zero(eltype(weights))
        @inbounds begin
            measurements[selected] = valid ? real(value) : zero(eltype(measurements))
            measurements[imaginary] = valid ? imag(value) : zero(eltype(measurements))
            weights[selected] = weight
            weights[imaginary] = weight
            measurement_validity[selected] = valid
            measurement_validity[imaginary] = valid
        end
        valid && @inbounds(KernelAbstractions.@atomic valid_count[1] += Int32(1))
    end
end

@kernel function _finish_scc_spatial_carrier!(
    valid_fraction,
    valid_count,
    frame_status,
    selected_count,
    minimum_valid_count,
    insufficient_failure,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(frame_status[1]))
        @inbounds count = valid_count[1]
        @inbounds valid_fraction[1] =
            eltype(valid_fraction)(count) / eltype(valid_fraction)(selected_count)
        count < minimum_valid_count && @inbounds(frame_status[1] = insufficient_failure)
    end
end

struct SccSpatialCarrier{
    Clear,
    Pair,
    Prepare,
    Horizontal,
    Selected,
    Finish,
    M,
    W,
    MV,
    VF,
    FS,
    D,
    DV,
    PS,
    R,
    SV,
    C,
    F,
    PI,
    SI,
    CC,
    CR,
    VC,
}
    clear!::Clear
    validate_pair!::Pair
    prepare!::Prepare
    horizontal!::Horizontal
    selected!::Selected
    finish!::Finish
    measurements::M
    weights::W
    measurement_validity::MV
    valid_fraction::VF
    frame_status::FS
    difference::D
    difference_validity::DV
    pair_status::PS
    reference_difference::R
    static_validity::SV
    carrier_conjugate::C
    fir::F
    pixel_rows::PI
    pixel_columns::PI
    selected_rows::SI
    selected_columns::SI
    horizontal_carrier::CC
    horizontal_support::CR
    masked_carrier::CC
    mask::CR
    valid_count::VC
    rows::Int
    columns::Int
    radius::Int
    minimum_support::Float32
    minimum_valid_count::Int32
    incomplete_failure::Int32
    invalid_failure::Int32
    insufficient_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        SccSpatialCarrierDemodulationF32,
        <:SccSpatialCarrierDemodulationPlan,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 4 || throw(ArgumentError(
        "SccSpatialCarrierDemodulationF32 capture requires four outputs",
    ))
    length(inputs) == 3 || throw(ArgumentError(
        "SccSpatialCarrierDemodulationF32 capture requires three inputs",
    ))
    measurements, weights, measurement_validity, valid_fraction = outputs
    difference, difference_validity, pair_status = inputs
    for (port, array) in (
        (:difference, difference),
        (Symbol("difference-validity"), difference_validity),
        (Symbol("pair-status"), pair_status),
        (:measurements, measurements),
        (:weights, weights),
        (Symbol("measurement-validity"), measurement_validity),
        (Symbol("valid-fraction"), valid_fraction),
    )
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, port), array,
        )
    end
    for (name, array) in (
        ("SCC difference input", difference),
        ("SCC difference-validity input", difference_validity),
        ("SCC pair-status input", pair_status),
        ("SCC spatial measurements output", measurements),
        ("SCC spatial weights output", weights),
        ("SCC spatial validity output", measurement_validity),
        ("SCC spatial valid-fraction output", valid_fraction),
        ("captured frame status", frame_status),
    )
        _require_capture_array(name, array, device)
    end

    for (index, output) in pairs(outputs)
        any(previous -> _capture_arrays_overlap(output, previous), outputs[1:index-1]) &&
            throw(ArgumentError("captured SCC spatial outputs must not overlap"))
    end

    plan = prepared.plan
    pixel_count = prod(plan.image_size)
    selected_count = length(plan.selected_indices)
    pixel_count <= typemax(Int32) ||
        throw(ArgumentError("SCC detector extent must fit Int32 for capture"))
    selected_count <= typemax(Int32) ||
        throw(ArgumentError("SCC selected count must fit Int32 for capture"))
    plan.minimum_valid_count <= typemax(Int32) || throw(
        ArgumentError("SCC minimum valid count must fit Int32 for capture"),
    )
    reference_difference = _capture_array(device, plan.reference_difference)
    static_validity = _capture_array(device, plan.static_validity)
    carrier_conjugate = _capture_array(device, plan.carrier_conjugate)
    fir = _capture_array(device, plan.fir)
    rows = plan.image_size[1]
    pixel_rows = _capture_array(
        device,
        [mod(pixel - 1, rows) for pixel in 1:pixel_count],
    )
    pixel_columns = _capture_array(
        device,
        [div(pixel - 1, rows) for pixel in 1:pixel_count],
    )
    selected_rows = _capture_array(
        device,
        [mod(pixel - 1, rows) for pixel in plan.selected_indices],
    )
    selected_columns = _capture_array(
        device,
        [div(pixel - 1, rows) for pixel in plan.selected_indices],
    )
    masked_carrier = _capture_zeros(device, ComplexF32, pixel_count)
    horizontal_carrier = _capture_zeros(device, ComplexF32, pixel_count)
    mask = _capture_zeros(device, Float32, pixel_count)
    horizontal_support = _capture_zeros(device, Float32, pixel_count)
    valid_count = _capture_zeros(device, Int32, 1)
    backend = KernelAbstractions.get_backend(measurements)
    return SccSpatialCarrier(
        _clear_scc_valid_count!(backend, 1),
        _validate_scc_pair_status!(backend, 1),
        _prepare_scc_spatial_carrier!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_spatial_horizontal!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _scc_spatial_selected!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _finish_scc_spatial_carrier!(backend, 1),
        measurements,
        weights,
        measurement_validity,
        valid_fraction,
        frame_status,
        difference,
        difference_validity,
        pair_status,
        reference_difference,
        static_validity,
        carrier_conjugate,
        fir,
        pixel_rows,
        pixel_columns,
        selected_rows,
        selected_columns,
        horizontal_carrier,
        horizontal_support,
        masked_carrier,
        mask,
        valid_count,
        plan.image_size[1],
        plan.image_size[2],
        length(plan.fir) >> 1,
        plan.minimum_support,
        Int32(plan.minimum_valid_count),
        _failure_code(
            node_index,
            SccSpatialCarrierDemodulationIncompletePair,
        ),
        _failure_code(
            node_index,
            SccSpatialCarrierDemodulationInvalidPairStatus,
        ),
        _failure_code(
            node_index,
            SccSpatialCarrierDemodulationInsufficientSupport,
        ),
    )
end

preflight(::SccSpatialCarrier) = nothing

function enqueue!(captured::SccSpatialCarrier)
    captured.clear!(captured.valid_count; ndrange=1)
    captured.validate_pair!(
        captured.frame_status,
        captured.pair_status,
        captured.incomplete_failure,
        captured.invalid_failure;
        ndrange=1,
    )
    captured.prepare!(
        captured.masked_carrier,
        captured.mask,
        captured.frame_status,
        captured.difference,
        captured.difference_validity,
        captured.reference_difference,
        captured.static_validity,
        captured.carrier_conjugate;
        ndrange=length(captured.difference),
    )
    captured.horizontal!(
        captured.horizontal_carrier,
        captured.horizontal_support,
        captured.frame_status,
        captured.masked_carrier,
        captured.mask,
        captured.fir,
        captured.pixel_rows,
        captured.pixel_columns,
        captured.rows,
        captured.columns,
        captured.radius;
        ndrange=length(captured.difference),
    )
    captured.selected!(
        captured.measurements,
        captured.weights,
        captured.measurement_validity,
        captured.valid_count,
        captured.frame_status,
        captured.horizontal_carrier,
        captured.horizontal_support,
        captured.fir,
        captured.selected_rows,
        captured.selected_columns,
        captured.rows,
        captured.radius,
        captured.minimum_support;
        ndrange=length(captured.selected_rows),
    )
    captured.finish!(
        captured.valid_fraction,
        captured.valid_count,
        captured.frame_status,
        Int32(length(captured.selected_rows)),
        captured.minimum_valid_count,
        captured.insufficient_failure;
        ndrange=1,
    )
    return nothing
end

function complete!(
    captured::SccSpatialCarrier,
    status::Int32,
)
    status == captured.incomplete_failure &&
        return SccSpatialCarrierDemodulationIncompletePair
    status == captured.invalid_failure &&
        return SccSpatialCarrierDemodulationInvalidPairStatus
    status == captured.insufficient_failure &&
        return SccSpatialCarrierDemodulationInsufficientSupport
    return nothing
end

reset!(captured::SccSpatialCarrier) = captured
