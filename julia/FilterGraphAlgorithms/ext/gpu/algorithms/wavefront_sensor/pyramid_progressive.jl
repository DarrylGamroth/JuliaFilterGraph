const PyramidMeasurementDeclaration = Union{PyramidMeasurementBlockF32,PyramidReadoutMeasurementBlockF32}
const PyramidProgressiveReconstructor = Union{PyramidRowReconstructorF32,PyramidReadoutReconstructorF32}
const PyramidProgressiveDeclaration =
    Union{PyramidPixelRowF32,PyramidMeasurementDeclaration,PyramidProgressiveReconstructor}

_pyramid_progressive_pixels(source) = source
_pyramid_progressive_pixels(source::FGA.PyramidRowReconstructionPlan) = source.pixels
_pyramid_progressive_pixels(source::FGA.PyramidReadoutMeasurementBlockPlan) = source.source

function _pyramid_row_groups(device, plan)
    linear = LinearIndices((plan.image_rows, plan.image_columns))
    block_count = div(plan.image_rows, plan.block_rows)
    indices = [Int[] for _ in 1:block_count]
    columns = [Int[] for _ in 1:block_count]
    slots = [Int[] for _ in 1:block_count]
    selected_counts = zeros(Int, block_count)
    full_pupil = Int[]
    arrival = Int[]
    for (pupil, (first_row, first_column)) in pairs(plan.pupil_origins)
        canonical = (pupil - 1) * plan.selected_pixel_count
        for row in 1:plan.pupil_rows, column in 1:plan.pupil_columns
            detector_row = first_row + row
            detector_index = linear[detector_row, first_column + column]
            ordinal = cld(detector_row, plan.block_rows)
            push!(full_pupil, detector_index)
            push!(arrival, ordinal)
            push!(indices[ordinal], detector_index)
            if plan.pupil_mask[row, column]
                canonical += 1
                selected_counts[ordinal] += 1
                push!(columns[ordinal], canonical)
                push!(slots[ordinal], selected_counts[ordinal])
            else
                push!(columns[ordinal], 0)
                push!(slots[ordinal], 0)
            end
        end
    end
    # Keep the longest available pupil-major prefix of the mean. Rows that
    # arrive early in a later pupil are normalized immediately and summed
    # only once earlier pupil rows have arrived.
    summed = 0
    groups = map(1:block_count) do ordinal
        first = summed + 1
        while summed < length(arrival) && arrival[summed + 1] <= ordinal
            summed += 1
        end
        (; indices=_capture_array(device, indices[ordinal]),
           columns=_capture_array(device, columns[ordinal]),
           slots=_capture_array(device, slots[ordinal]),
           count=selected_counts[ordinal], sum_range=first:summed)
    end
    return groups, _capture_array(device, full_pupil)
end

function _pyramid_readout_groups(device, plan)
    linear = LinearIndices(plan.readout.detector_size)
    groups = map(plan.completion_groups) do ready
        indices = Int[linear[plan.selected_detector_rows[pixel], plan.selected_detector_columns[pixel]]
                      for pixel in ready]
        (; indices=_capture_array(device, indices), columns=_capture_array(device, ready), count=length(ready))
    end
    rows, columns = plan.regions.region_size
    full_pupil = Int[linear[first_row + row, first_column + column]
        for (first_row, first_column) in plan.regions.origins for row in 1:rows for column in 1:columns]
    return groups, _capture_array(device, full_pupil)
end

function _pyramid_progressive_geometry(device, plan::PyramidPixelRowPlan)
    groups, full_pupil = _pyramid_row_groups(device, plan)
    return (;
        mapping=plan.block_rows, groups, full_pupil, plan.mean_scale,
        detector_size=(plan.image_rows, plan.image_columns), selected=plan.selected_pixel_count,
    )
end

function _pyramid_progressive_geometry(device, plan::FGA.PyramidReadoutReconstructionPlan)
    groups, full_pupil = _pyramid_readout_groups(device, plan)
    mapping = map(i -> _capture_array(device, i), FGA._readout_detector_indices(plan.readout))
    return (;
        mapping, groups, full_pupil, mean_scale=plan.normalization.mean_scale,
        detector_size=plan.readout.detector_size, selected=length(plan.normalization.selected_pixel_offsets),
    )
end

function _progressive_plan_storage(::Type{D}, device, plan) where {D<:PyramidProgressiveDeclaration}
    geometry = _pyramid_progressive_geometry(device, _pyramid_progressive_pixels(plan))
    FGA.PYRAMID_PUPIL_COUNT * UInt64(geometry.selected) <= typemax(UInt32) ||
        throw(ArgumentError("pyramid measurement count exceeds UInt32 capacity"))
    reconstruction = _progressive_reconstructor_storage(device, plan)
    return (; geometry..., reconstruction)
end

mutable struct PyramidProgressiveWorkspace{Owner,F,D,P,I,V,M,R}
    device::Owner
    frame::F
    detector::D
    packet::P
    flags::I
    status::I
    completion::Vector{Int32}
    committed_gain::V
    candidate_gain::V
    sum::V
    mean::V
    pixels::M
    reconstructed::R
    terminal_candidate::Bool
end

function _progressive_workspace(::Type{D}, plan) where {D<:PyramidProgressiveDeclaration}
    device, storage = plan.device, plan.storage
    frame = _progressive_frame(storage.mapping)
    detector = _capture_zeros(device, Float32, storage.detector_size...)
    packet = _capture_array(device, fill(MeasurementEntry(0f0, 0), FGA.PYRAMID_PUPIL_COUNT * storage.selected + 1))
    # Replacing pupil origins can increase the number of arriving pixels in
    # one block without changing its public format.
    flags = _capture_zeros(device, Int32, length(storage.full_pupil))
    status = _capture_zeros(device, Int32, 1)
    committed_gain, candidate_gain, sum, mean = ntuple(_ -> _capture_zeros(device, Float32, 1), 4)
    pixels = _progressive_pyramid_pixels(D, device, storage.selected)
    reconstructed = _progressive_reconstructed(device, plan.source)
    return PyramidProgressiveWorkspace(
        device, frame, detector, packet, flags, status, zeros(Int32, 1), committed_gain,
        candidate_gain, sum, mean, pixels, reconstructed, false,
    )
end

_progressive_pyramid_pixels(::Type{<:PyramidMeasurementDeclaration}, device, selected) = nothing
_progressive_pyramid_pixels(::Type{<:Union{PyramidPixelRowF32,PyramidProgressiveReconstructor}}, device, selected) =
    _capture_zeros(device, Float32, FGA.PYRAMID_PUPIL_COUNT, selected)

function FGA.commit!(workspace::PyramidProgressiveWorkspace)
    if workspace.terminal_candidate
        copyto!(workspace.committed_gain, workspace.candidate_gain)
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(workspace.committed_gain))
        _complete_progressive_frame!(workspace.frame)
        workspace.terminal_candidate = false
    end
    return workspace
end

function FGA.discard!(workspace::PyramidProgressiveWorkspace)
    _abandon_progressive_frame!(workspace.frame)
    workspace.terminal_candidate = false
    return workspace
end

function FGA.reset!(workspace::PyramidProgressiveWorkspace)
    _with_eager_device(workspace.device) do
        fill!(workspace.committed_gain, 0f0)
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(workspace.committed_gain))
    end
    workspace.frame = typeof(workspace.frame)()
    workspace.terminal_candidate = false
    return workspace
end

FGA._plan_adopted!(workspace::PyramidProgressiveWorkspace) = FGA.discard!(workspace)

# PyramidPixelRowF32 and PyramidMeasurementBlockF32 share coefficients and
# readiness geometry. Only their workspaces and publication paths differ.
for (Plan, Declaration) in (
    (PyramidPixelRowPlan, PyramidPixelRowF32),
    (FGA.PyramidRowReconstructionPlan, PyramidRowReconstructorF32),
    (FGA.PyramidReadoutReconstructionPlan, PyramidReadoutReconstructorF32),
)
    @eval function _replace_prepared_parameter(replacement, plan::ProgressiveDevicePlan{<:$Plan}, values)
        return _replace_progressive_parameter($Declaration, replacement, plan, values)
    end
end

@kernel function _pyramid_row_packet!(packet, flags, detector, indices, columns, slots, gain)
    index = @index(Global, Linear)
    @inbounds begin
        input = detector[indices[index]]
        slot = slots[index]
        if iszero(slot)
            flags[index] = isfinite(input) ? Int32(0) : PYRAMID_ROW_INPUT_FAILURE
        else
            normalized, failure = _pyramid_normalize_readout(input, gain[1])
            flags[index] = failure
            packet[slot + 1] = eltype(packet)(normalized, UInt32(columns[index] - 1))
        end
    end
end

@kernel function _sum_progressive_pyramid!(status, sum_buffer, candidate_gain, mean_buffer,
    detector, full_pupil, first_index, last_index, mean_scale, start, terminal)
    # Checked processing preserves the CPU's pupil-major sum and checks for
    # overflow at each addition, including on nonterminal row blocks.
    @inbounds if iszero(status[1])
        value = start ? 0f0 : sum_buffer[1]
        for index in first_index:last_index
            input = detector[full_pupil[index]]
            if !isfinite(input)
                status[1] = PYRAMID_ROW_INPUT_FAILURE
                break
            end
            value += input
            if !isfinite(value)
                status[1] = PYRAMID_ROW_MEAN_FAILURE
                break
            end
        end
        sum_buffer[1] = value
        if terminal && iszero(status[1])
            mean = value * mean_scale
            gain = mean > 0f0 ? inv(mean) : 0f0
            if !isfinite(mean)
                status[1] = PYRAMID_ROW_MEAN_FAILURE
            elseif !isfinite(gain)
                status[1] = PYRAMID_ROW_GAIN_FAILURE
            else
                mean_buffer[1] = mean
                candidate_gain[1] = gain
            end
        end
    end
end

@kernel function _stage_pyramid_packet!(pixels, packet, status, selected)
    index = @index(Global, Linear)
    @inbounds if iszero(status[1])
        entry = packet[index + 1]
        pupil, pixel = divrem(Int(entry.column), selected)
        pixels[pupil + 1, pixel + 1] = entry.value
    end
end

function _enqueue_pyramid_sum!(backend, workspace, storage, range, start, terminal)
    _sum_progressive_pyramid!(backend, 1)(
        workspace.status, workspace.sum, workspace.candidate_gain, workspace.mean, workspace.detector,
        storage.full_pupil, first(range), last(range), storage.mean_scale, start, terminal; ndrange=1,
    )
    return nothing
end

function _enqueue_pyramid_ready!(backend, workspace, storage, ordinal, terminal, frame::FGA.RowBlockFrame)
    group = storage.groups[ordinal]
    entries = length(group.indices)
    if !iszero(entries)
        _pyramid_row_packet!(backend, PYRAMID_READOUT_LANES)(
            workspace.packet, workspace.flags, workspace.detector, group.indices,
            group.columns, group.slots, workspace.committed_gain; ndrange=entries,
        )
    end
    _enqueue_pyramid_packet_status!(backend, workspace, storage, entries, group.count)
    _enqueue_pyramid_sum!(backend, workspace, storage, group.sum_range, ordinal == 1, terminal)
    return group.count
end

function _enqueue_pyramid_ready!(backend, workspace, storage, ordinal, terminal, frame::FGA.DetectorReadoutFrame)
    group = storage.groups[ordinal]
    if !iszero(group.count)
        _pyramid_readout_pixels!(backend, PYRAMID_READOUT_LANES)(
            workspace.packet, workspace.flags, workspace.detector, group.indices,
            group.columns, workspace.committed_gain; ndrange=group.count,
        )
    end
    _enqueue_pyramid_packet_status!(backend, workspace, storage, group.count, group.count)
    return group.count
end

function _enqueue_pyramid_packet_status!(backend, workspace, storage, flag_count, count)
    _finish_pyramid_readout!(backend, PYRAMID_READOUT_LANES)(
        workspace.packet, workspace.status, workspace.flags, workspace.detector,
        storage.full_pupil, storage.mean_scale, workspace.candidate_gain, workspace.mean,
        false, flag_count, count; ndrange=PYRAMID_READOUT_LANES,
    )
    return nothing
end

# Readout accumulation is checked before the terminal mean. Row admission
# already enqueued its incremental sum before reconstruction.
_enqueue_pyramid_terminal_sum!(backend, workspace, storage, terminal, ::FGA.RowBlockFrame) = nothing
function _enqueue_pyramid_terminal_sum!(backend, workspace, storage, terminal, ::FGA.DetectorReadoutFrame)
    terminal && _enqueue_pyramid_sum!(backend, workspace, storage, eachindex(storage.full_pupil), true, true)
    return nothing
end

_pyramid_progressive_failures(::Type{<:Union{PyramidPixelRowF32,PyramidMeasurementBlockF32}}) =
    (FGA.PyramidPixelRowMetadataMissing, FGA.PyramidPixelRowOffsetOverflow, nothing)
_pyramid_progressive_failures(::Type{PyramidRowReconstructorF32}) =
    (FGA.PyramidRowMetadataMissing, FGA.PyramidRowOffsetOverflow, FGA.PyramidRowNonFiniteReconstruction)
_pyramid_progressive_failures(::Type{PyramidReadoutReconstructorF32}) =
    (FGA.PyramidReadoutMetadataMissing, FGA.PyramidReadoutOffsetOverflow, FGA.PyramidReadoutNonFiniteReconstruction)
_pyramid_progressive_failures(::Type{PyramidReadoutMeasurementBlockF32}) =
    (FGA.PyramidReadoutMetadataMissing, FGA.PyramidReadoutOffsetOverflow, nothing)

function _process_pyramid_progressive!(outputs, prepared::PreparedAlgorithm{D}, input, metadata) where {D}
    plan, workspace = _progressive_admitted(prepared, outputs, (input,))
    _require_progressive_workspace_disjoint(
        (input, outputs...),
        (workspace.detector, workspace.packet, workspace.pixels, workspace.reconstructed,
         workspace.mean, workspace.sum, workspace.committed_gain, workspace.candidate_gain,
         workspace.flags, workspace.status),
    )
    storage = plan.storage
    missing, overflow, nonfinite = _pyramid_progressive_failures(D)
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return missing
    UInt64(sample.offset) <= UInt64(typemax(Int)) || return overflow
    ordinal = _progressive_block_ordinal!(workspace.frame, sample, input, workspace.detector, storage.mapping)
    isnothing(ordinal) && return AlgorithmProcessDeferred()
    backend = KernelAbstractions.get_backend(workspace.detector)
    entries = _enqueue_pyramid_ready!(backend, workspace, storage, ordinal, sample.terminal, workspace.frame)
    if !isnothing(workspace.pixels) && !iszero(entries)
        _stage_pyramid_packet!(backend, PYRAMID_READOUT_LANES)(
            workspace.pixels, workspace.packet, workspace.status, storage.selected; ndrange=entries,
        )
    end
    if !isnothing(nonfinite)
        ordinal == 1 && fill!(workspace.reconstructed, 0f0)
        _enqueue_measurement_accumulation!(
            backend, workspace.reconstructed, storage.reconstruction, workspace.packet, workspace.status, nonfinite,
        )
    end
    _enqueue_pyramid_terminal_sum!(backend, workspace, storage, sample.terminal, workspace.frame)
    failure = _progressive_status!(workspace.completion, workspace.status, typeof(missing))
    isnothing(failure) || return failure

    workspace.terminal_candidate = sample.terminal
    return _publish_pyramid_progressive!(D, outputs, workspace, sample, backend, entries)
end

function _publish_pyramid_progressive!(
    ::Type{<:PyramidMeasurementDeclaration}, outputs, workspace, sample, backend, entries,
)
    _enqueue_measurement_publication!(outputs..., workspace.packet, entries)
    return AlgorithmProcessComplete(_progressive_measurement_metadata(sample, workspace.frame))
end

_pyramid_terminal_outputs(::Type{PyramidPixelRowF32}, workspace) = (workspace.pixels, workspace.mean)
_pyramid_terminal_outputs(::Type{<:PyramidProgressiveReconstructor}, workspace) =
    (workspace.pixels, workspace.mean, workspace.reconstructed)

function _publish_pyramid_progressive!(
    ::Type{D}, outputs, workspace, sample, backend, entries,
) where {D<:Union{PyramidPixelRowF32,PyramidProgressiveReconstructor}}
    sample.terminal || return AlgorithmProcessDeferred()

    foreach(copyto!, outputs, _pyramid_terminal_outputs(D, workspace))
    KernelAbstractions.synchronize(backend)
    return _progressive_complete(sample, workspace.frame.current_discontinuity)
end

for Declaration in (PyramidMeasurementBlockF32, PyramidReadoutMeasurementBlockF32,
                    PyramidRowReconstructorF32, PyramidReadoutReconstructorF32)
    @eval function _process_admitted!(first::AbstractArray, second::AbstractArray, third::AbstractArray,
        prepared::PreparedAlgorithm{$Declaration,<:ProgressiveDevicePlan},
        input::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
        return _process_pyramid_progressive!((first, second, third), prepared, input, metadata)
    end
end

function _process_admitted!(pixels::AbstractArray, mean::AbstractArray,
    prepared::PreparedAlgorithm{PyramidPixelRowF32,<:ProgressiveDevicePlan},
    input::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
    return _process_pyramid_progressive!((pixels, mean), prepared, input, metadata)
end
