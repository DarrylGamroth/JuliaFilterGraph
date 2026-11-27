@kernel function _scatter_readout!(detector, block, indices)
    index = @index(Global, Linear)
    @inbounds detector[indices[index]] = block[index]
end

@kernel function _scatter_row_block!(detector, block, first_row)
    row, column = @index(Global, NTuple)
    @inbounds detector[first_row + row, column] = block[row, column]
end

function _enqueue_scatter_readout!(detector, block, indices)
    backend = KernelAbstractions.get_backend(detector)
    _scatter_readout!(backend, 256)(detector, block, indices; ndrange=length(block))
    return nothing
end

function _enqueue_scatter_row_block!(detector, block, first_row)
    backend = KernelAbstractions.get_backend(detector)
    _scatter_row_block!(backend, (16, 16))(detector, block, first_row; ndrange=size(block))
    return nothing
end

@kernel function _calibrate_row_block!(output, raw, flat, background, first_row)
    row, column = @index(Global, NTuple)
    @inbounds output[row, column] = _calibrated_pixel(
        raw[row, column], flat[first_row + row, column], background[first_row + row, column],
    )
end

@kernel function _calibrate_readout_block!(output, raw, indices, flat, background)
    index = @index(Global, Linear)
    @inbounds begin
        pixel = indices[index]
        output[index] = _calibrated_pixel(raw[index], flat[pixel], background[pixel])
    end
end

function _progressive_plan_storage(::Type{PixelCalibrationRowU16F32}, device, plan)
    return (; flat=_capture_array(device, plan.flat),
             background=_capture_array(device, plan.background))
end

function _progressive_plan_storage(::Type{PixelCalibrationReadoutU16F32}, device, plan)
    # The CPU calibration plan stores coefficients in acquisition order. Use
    # the same mapping to place coefficients back in detector coordinates.
    flat = Matrix{Float32}(undef, plan.readout.detector_size)
    background = similar(flat)
    @inbounds for index in eachindex(plan.flat)
        row = plan.readout.detector_rows[index]
        column = plan.readout.detector_columns[index]
        flat[row, column] = plan.flat[index]
        background[row, column] = plan.background[index]
    end
    return (; flat=_capture_array(device, flat),
             background=_capture_array(device, background),
             indices=map(indices -> _capture_array(device, indices), FGA._readout_detector_indices(plan.readout)))
end

function _progressive_plan_storage(::Type{DetectorFrameAssemblerF32}, device, plan)
    return map(indices -> _capture_array(device, indices), FGA._readout_detector_indices(plan.readout))
end

_progressive_workspace(::Type{PixelCalibrationRowU16F32}, plan) = FGA.PixelCalibrationRowWorkspace()
_progressive_workspace(::Type{PixelCalibrationReadoutU16F32}, plan) = FGA.DetectorReadoutCalibrationWorkspace()

function _progressive_workspace(::Type{DetectorFrameAssemblerF32}, plan)
    detector = _capture_zeros(plan.device, Float32, plan.source.readout.detector_size...)
    return FGA.DetectorFrameAssemblerWorkspace(detector, FGA.DetectorReadoutFrame())
end

function _replace_prepared_parameter(replacement, plan::ProgressiveDevicePlan{FGA.PixelCalibrationPlan}, values)
    return _replace_progressive_parameter(PixelCalibrationRowU16F32, replacement, plan, values)
end

function _replace_prepared_parameter(replacement, plan::ProgressiveDevicePlan{FGA.DetectorReadoutCalibrationPlan}, values)
    return _replace_progressive_parameter(PixelCalibrationReadoutU16F32, replacement, plan, values)
end

function _process_admitted!(output::AbstractArray,
    prepared::PreparedAlgorithm{PixelCalibrationRowU16F32,<:ProgressiveDevicePlan},
    raw::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
    plan, workspace = _progressive_admitted(prepared, (output,), (raw,))
    _require_progressive_output_aliases((output,), (plan.storage.flat, plan.storage.background))
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return FGA.PixelCalibrationRowMetadataMissing
    first_row = Int(sample.offset)
    block_rows = size(raw, 1)
    image_rows = size(plan.storage.flat, 1)
    rem(first_row, block_rows) == 0 || return FGA.PixelCalibrationRowOffsetMisaligned
    0 <= first_row <= image_rows - block_rows || return FGA.PixelCalibrationRowOutOfBounds
    sample.terminal == (first_row + block_rows == image_rows) || return FGA.PixelCalibrationRowTerminalMismatch

    backend = KernelAbstractions.get_backend(output)
    _calibrate_row_block!(backend, (16, 16))(
        output, raw, plan.storage.flat, plan.storage.background, first_row; ndrange=size(raw),
    )
    KernelAbstractions.synchronize(backend)
    discontinuity = sample.discontinuity || workspace.candidate_pending_discontinuity
    workspace.candidate_pending_discontinuity = false
    return AlgorithmProcessComplete(SampleMetadata(sample; discontinuity))
end

function _process_admitted!(output::AbstractArray,
    prepared::PreparedAlgorithm{PixelCalibrationReadoutU16F32,<:ProgressiveDevicePlan},
    raw::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
    plan, workspace = _progressive_admitted(prepared, (output,), (raw,))
    _require_progressive_output_aliases((output,), (plan.storage.flat, plan.storage.background))
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return FGA.PixelCalibrationReadoutMetadataMissing
    ordinal = Int(sample.offset)
    blocks = plan.source.readout.block_count
    0 <= ordinal < blocks || return FGA.PixelCalibrationReadoutOrdinalRange
    sample.terminal == (ordinal + 1 == blocks) || return FGA.PixelCalibrationReadoutTerminalMismatch

    backend = KernelAbstractions.get_backend(output)
    _calibrate_readout_block!(backend, 256)(
        output, raw, plan.storage.indices[ordinal + 1], plan.storage.flat, plan.storage.background;
        ndrange=length(raw),
    )
    KernelAbstractions.synchronize(backend)
    discontinuity = sample.discontinuity || workspace.candidate_pending_discontinuity
    workspace.candidate_pending_discontinuity = false
    return AlgorithmProcessComplete(SampleMetadata(sample; discontinuity))
end

function _process_admitted!(output::AbstractArray,
    prepared::PreparedAlgorithm{DetectorFrameAssemblerF32,<:ProgressiveDevicePlan},
    block::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
    plan, workspace = _progressive_admitted(prepared, (output,), (block,))
    _require_progressive_output_aliases((output, block), (workspace.detector,))
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return FGA.DetectorFrameAssemblerMetadataMissing
    FGA._accept_detector_readout!(workspace.frame, sample, plan.source.readout.block_count) ||
        return AlgorithmProcessDeferred()
    _enqueue_scatter_readout!(workspace.detector, block, plan.storage[Int(sample.offset) + 1])
    if sample.terminal
        discontinuity = workspace.frame.current_discontinuity
        copyto!(output, workspace.detector)
        FGA._complete_detector_readout!(workspace.frame)
    end
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(output))
    return sample.terminal ? _progressive_complete(sample, discontinuity) : AlgorithmProcessDeferred()
end
