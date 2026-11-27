import FilterGraphAlgorithms:
    ChoppedImageComplete,
    ChoppedImagePriming,
    PreparedAlgorithm,
    SccPixelResidualF32,
    SccPixelResidualIncompletePair,
    SccPixelResidualInsufficientValidPixels,
    SccPixelResidualInvalidPairStatus,
    SccPixelResidualPlan,
    complete!,
    enqueue!,
    preflight,
    reset!

@kernel function _clear_scc_valid_count!(valid_count)
    index = @index(Global, Linear)
    index == 1 && @inbounds(valid_count[1] = Int32(0))
end

@kernel function _validate_scc_pair_status!(
    frame_status,
    pair_status,
    incomplete_failure,
    invalid_failure,
)
    index = @index(Global, Linear)
    if index == 1 && @inbounds(iszero(frame_status[1]))
        @inbounds status = pair_status[1]
        if status == UInt8(ChoppedImagePriming)
            @inbounds frame_status[1] = incomplete_failure
        elseif status != UInt8(ChoppedImageComplete)
            @inbounds frame_status[1] = invalid_failure
        end
    end
end

@kernel function _scc_pixel_residual!(
    selected_residuals,
    selected_validity,
    valid_count,
    frame_status,
    difference,
    difference_validity,
    reference_image,
    selected_indices,
)
    selected = @index(Global, Linear)
    if selected <= length(selected_indices) &&
       @inbounds(iszero(frame_status[1]))
        @inbounds pixel = selected_indices[selected]
        @inbounds valid = difference_validity[pixel]
        value = zero(eltype(selected_residuals))
        if valid
            @inbounds sample = difference[pixel]
            valid = isfinite(sample)
            if valid
                @inbounds value = sample - reference_image[pixel]
                valid = isfinite(value)
            end
        end
        @inbounds begin
            selected_residuals[selected] =
                valid ? value : zero(eltype(selected_residuals))
            selected_validity[selected] = valid
        end
        if valid
            @inbounds KernelAbstractions.@atomic valid_count[1] += Int32(1)
        end
    end
end

@kernel function _finish_scc_pixel_residual!(
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
        if count < minimum_valid_count
            @inbounds frame_status[1] = insufficient_failure
        end
    end
end

struct SccPixelResidual{
    Clear,
    Pair,
    Gather,
    Finish,
    R,
    V,
    F,
    FS,
    D,
    DV,
    PS,
    Ref,
    I,
    C,
}
    clear!::Clear
    validate_pair!::Pair
    gather!::Gather
    finish!::Finish
    selected_residuals::R
    selected_validity::V
    valid_fraction::F
    frame_status::FS
    difference::D
    difference_validity::DV
    pair_status::PS
    reference_image::Ref
    selected_indices::I
    valid_count::C
    minimum_valid_count::Int32
    incomplete_failure::Int32
    invalid_failure::Int32
    insufficient_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{SccPixelResidualF32,<:SccPixelResidualPlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 3 || throw(
        ArgumentError("SccPixelResidualF32 capture requires three outputs"),
    )
    length(inputs) == 3 || throw(
        ArgumentError("SccPixelResidualF32 capture requires three inputs"),
    )
    selected_residuals, selected_validity, valid_fraction = outputs
    difference, difference_validity, pair_status = inputs
    for (port, array) in (
        (:difference, difference),
        (Symbol("difference-validity"), difference_validity),
        (Symbol("pair-status"), pair_status),
        (Symbol("selected-residuals"), selected_residuals),
        (Symbol("selected-validity"), selected_validity),
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
        ("SCC selected-residual output", selected_residuals),
        ("SCC selected-validity output", selected_validity),
        ("SCC valid-fraction output", valid_fraction),
        ("captured frame status", frame_status),
    )
        _require_capture_array(name, array, device)
    end

    plan = prepared.plan
    length(plan.selected_indices) <= typemax(Int32) || throw(
        ArgumentError("SCC selected-pixel count must fit Int32 for capture"),
    )
    plan.minimum_valid_count <= typemax(Int32) || throw(
        ArgumentError("SCC minimum valid-pixel count must fit Int32 for capture"),
    )
    reference_image = _capture_array(device, plan.reference_image)
    selected_indices = _capture_array(device, Int32.(plan.selected_indices))
    valid_count = _capture_zeros(device, Int32, 1)
    backend = KernelAbstractions.get_backend(selected_residuals)
    return SccPixelResidual(
        _clear_scc_valid_count!(backend, 1),
        _validate_scc_pair_status!(backend, 1),
        _scc_pixel_residual!(
            backend,
            PROJECTION_WORKGROUP_SIZE,
        ),
        _finish_scc_pixel_residual!(backend, 1),
        selected_residuals,
        selected_validity,
        valid_fraction,
        frame_status,
        difference,
        difference_validity,
        pair_status,
        reference_image,
        selected_indices,
        valid_count,
        Int32(plan.minimum_valid_count),
        _failure_code(node_index, SccPixelResidualIncompletePair),
        _failure_code(node_index, SccPixelResidualInvalidPairStatus),
        _failure_code(
            node_index,
            SccPixelResidualInsufficientValidPixels,
        ),
    )
end

preflight(::SccPixelResidual) = nothing

function enqueue!(captured::SccPixelResidual)
    captured.clear!(captured.valid_count; ndrange=1)
    captured.validate_pair!(
        captured.frame_status,
        captured.pair_status,
        captured.incomplete_failure,
        captured.invalid_failure;
        ndrange=1,
    )
    captured.gather!(
        captured.selected_residuals,
        captured.selected_validity,
        captured.valid_count,
        captured.frame_status,
        captured.difference,
        captured.difference_validity,
        captured.reference_image,
        captured.selected_indices;
        ndrange=length(captured.selected_indices),
    )
    captured.finish!(
        captured.valid_fraction,
        captured.valid_count,
        captured.frame_status,
        Int32(length(captured.selected_indices)),
        captured.minimum_valid_count,
        captured.insufficient_failure;
        ndrange=1,
    )
    return nothing
end

function complete!(
    captured::SccPixelResidual,
    status::Int32,
)
    status == captured.incomplete_failure &&
        return SccPixelResidualIncompletePair
    status == captured.invalid_failure &&
        return SccPixelResidualInvalidPairStatus
    status == captured.insufficient_failure &&
        return SccPixelResidualInsufficientValidPixels
    return nothing
end

reset!(captured::SccPixelResidual) = captured
