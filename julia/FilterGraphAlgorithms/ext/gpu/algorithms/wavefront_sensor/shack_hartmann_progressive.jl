const ShackHartmannMeasurementDeclaration = Union{
    ShackHartmannMeasurementBlockF32, ShackHartmannWeightedMeasurementBlockF32,
    ShackHartmannReadoutMeasurementBlockF32,
}
const ShackHartmannProgressiveReconstructor = Union{
    ShackHartmannRowReconstructorF32, ShackHartmannReadoutReconstructorF32,
}
const ShackHartmannRowPlan = Union{
    FGA.ShackHartmannMeasurementBlockPlan, FGA.ShackHartmannWeightedMeasurementBlockPlan,
    FGA.ShackHartmannRowReconstructionPlan,
}
const ShackHartmannReadoutPlan = Union{
    FGA.ShackHartmannReadoutMeasurementBlockPlan, FGA.ShackHartmannReadoutReconstructionPlan,
}

function prepare_algorithm(
    declaration::Type{ShackHartmannWeightedMeasurementBlockF32},
    configuration::ShackHartmannWeightedMeasurementBlockF32Configuration,
    device::DeviceType,
)
    prepared = FGA._prepare_declared_algorithm(declaration, configuration; prepare_host_plan=false)
    return prepare_algorithm(prepared, device)
end

function _progressive_source(plan::FGA.ShackHartmannWeightedMeasurementBlockPlan)
    isnothing(plan.sensing.packed_weights) && return plan
    return FGA.ShackHartmannWeightedMeasurementBlockPlan(plan; pack_weights=false)
end

# Under consecutive row admission the scientific prefix is determined by
# geometry alone. Prepare each new prefix once, including empty blocks.
function _shack_hartmann_row_groups(sensing, block_rows)
    image_rows = sensing.regions.image_size[1]
    window_rows = sensing.regions.region_size[1]
    groups = Vector{Vector{Int}}()
    last_ready = 0
    origins = sensing.regions.origins
    for completed_rows in block_rows:block_rows:image_rows
        first_ready = last_ready + 1
        while last_ready < length(origins) &&
              origins[last_ready + 1][1] + window_rows <= completed_rows
            last_ready += 1
        end
        push!(groups, collect(first_ready:last_ready))
    end
    return groups
end

_shack_hartmann_progressive_sensing(plan) = plan.sensing
_shack_hartmann_progressive_sensing(plan::FGA.ShackHartmannWeightedMeasurementBlockPlan) = plan.sensing.sensing
_shack_hartmann_progressive_weights(plan) = nothing
_shack_hartmann_progressive_weights(plan::FGA.ShackHartmannWeightedMeasurementBlockPlan) = plan.sensing.weights

function _shack_hartmann_progressive_groups(device, source::ShackHartmannRowPlan, sensing)
    return source.block_rows, _shack_hartmann_row_groups(sensing, source.block_rows)
end

function _shack_hartmann_progressive_groups(device, source::ShackHartmannReadoutPlan, sensing)
    indices = map(i -> _capture_array(device, i), FGA._readout_detector_indices(source.readout))
    return indices, source.completion_groups
end

function _progressive_plan_storage(::Type{D}, device, source) where {
    D<:Union{ShackHartmannMeasurementDeclaration,ShackHartmannProgressiveReconstructor}
}
    sensing = _shack_hartmann_progressive_sensing(source)
    mapping, groups = _shack_hartmann_progressive_groups(device, source, sensing)
    weighted = _shack_hartmann_progressive_weights(source)
    weights = isnothing(weighted) ? nothing : _capture_array(device, weighted)
    reconstruction = _progressive_reconstructor_storage(device, source)
    return (;
        mapping,
        ready=map(i -> _capture_array(device, i), groups),
        origins=_capture_array(device, hcat((collect(origin) for origin in sensing.regions.origins)...)),
        x=_capture_array(device, sensing.x_coordinates),
        y=_capture_array(device, sensing.y_coordinates),
        reference=_capture_array(device, sensing.reference_slopes),
        pixel_thresholds=_capture_array(device, sensing.pixel_thresholds),
        flux_thresholds=_capture_array(device, sensing.flux_thresholds),
        active=_capture_array(device, sensing.active),
        weights, reconstruction,
    )
end

mutable struct ShackHartmannProgressiveWorkspace{F,D,P,S,R}
    frame::F
    detector::D
    packet::P
    status::S
    completion::Vector{Int32}
    reconstructed::R
    terminal_candidate::Bool
end

function _progressive_workspace(::Type{D}, plan) where {
    D<:Union{ShackHartmannMeasurementDeclaration,ShackHartmannProgressiveReconstructor}
}
    source = plan.source
    sensing = _shack_hartmann_progressive_sensing(source)
    frame = _progressive_frame(plan.storage.mapping)
    detector = _capture_zeros(plan.device, Float32, sensing.regions.image_size...)
    packet = _capture_array(plan.device, fill(MeasurementEntry(0f0, 0), 2 * length(sensing.regions.origins) + 1))
    status = _capture_zeros(plan.device, Int32, 1)
    reconstructed = _progressive_reconstructed(plan.device, source)
    return ShackHartmannProgressiveWorkspace(frame, detector, packet, status, zeros(Int32, 1), reconstructed, false)
end

function FGA.commit!(workspace::ShackHartmannProgressiveWorkspace)
    if workspace.terminal_candidate
        _complete_progressive_frame!(workspace.frame)
        workspace.terminal_candidate = false
    end
    return workspace
end

function FGA.discard!(workspace::ShackHartmannProgressiveWorkspace)
    _abandon_progressive_frame!(workspace.frame)
    workspace.terminal_candidate = false
    return workspace
end

FGA._plan_adopted!(workspace::ShackHartmannProgressiveWorkspace) = FGA.discard!(workspace)

FGA.reset!(workspace::ShackHartmannProgressiveWorkspace) = FGA.discard!(workspace)

function FGA.reset!(workspace::ShackHartmannProgressiveWorkspace{FGA.RowBlockFrame})
    FGA._reset_row_block_frame!(workspace.frame)
    workspace.terminal_candidate = false
    return workspace
end

for (Plan, Declaration) in (
    (FGA.ShackHartmannMeasurementBlockPlan, ShackHartmannMeasurementBlockF32),
    (FGA.ShackHartmannWeightedMeasurementBlockPlan, ShackHartmannWeightedMeasurementBlockF32),
    (FGA.ShackHartmannReadoutMeasurementBlockPlan, ShackHartmannReadoutMeasurementBlockF32),
    (FGA.ShackHartmannRowReconstructionPlan, ShackHartmannRowReconstructorF32),
)
    @eval function _replace_prepared_parameter(replacement, plan::ProgressiveDevicePlan{<:$Plan}, values)
        return _replace_progressive_parameter($Declaration, replacement, plan, values)
    end
end

_shack_hartmann_progressive_failures(::Type{<:Union{ShackHartmannMeasurementBlockF32,ShackHartmannWeightedMeasurementBlockF32}}) =
    (FGA.ShackHartmannMeasurementBlockMetadataMissing, FGA.ShackHartmannMeasurementBlockOffsetOverflow)
_shack_hartmann_progressive_failures(::Type{ShackHartmannReadoutMeasurementBlockF32}) =
    (FGA.ShackHartmannReadoutMeasurementBlockMetadataMissing, FGA.ShackHartmannReadoutMeasurementBlockOffsetOverflow)
_shack_hartmann_progressive_failures(::Type{ShackHartmannRowReconstructorF32}) =
    (FGA.ShackHartmannRowMetadataMissing, FGA.ShackHartmannRowOffsetOverflow, FGA.ShackHartmannRowNonFiniteReconstruction)
_shack_hartmann_progressive_failures(::Type{ShackHartmannReadoutReconstructorF32}) =
    (FGA.ShackHartmannReadoutMetadataMissing, FGA.ShackHartmannReadoutOffsetOverflow, FGA.ShackHartmannReadoutNonFiniteReconstruction)

function _enqueue_shack_hartmann_progressive!(plan, workspace, ready)
    backend = KernelAbstractions.get_backend(workspace.detector)
    sensing = _shack_hartmann_progressive_sensing(plan.source)
    rows, columns = sensing.regions.region_size
    storage = plan.storage
    _shack_hartmann_readout!(backend, SHACK_HARTMANN_READOUT_LANES)(
        workspace.packet, workspace.status, workspace.detector, ready,
        storage.origins, storage.x, storage.y, storage.reference, storage.pixel_thresholds,
        storage.flux_thresholds, storage.active, rows, columns, storage.weights;
        ndrange=max(length(ready), 1),
    )
    return backend
end

function _process_shack_hartmann_progressive!(
    measurements, columns, count, prepared::PreparedAlgorithm{D}, input, metadata,
) where {D}
    plan, workspace = _progressive_admitted(prepared, (measurements, columns, count), (input,))
    _require_progressive_workspace_disjoint(
        (input, measurements, columns, count), (workspace.detector, workspace.packet, workspace.status),
    )
    missing, overflow = _shack_hartmann_progressive_failures(D)
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return missing
    UInt64(sample.offset) <= UInt64(typemax(Int)) || return overflow
    ordinal = _progressive_block_ordinal!(workspace.frame, sample, input, workspace.detector, plan.storage.mapping)
    isnothing(ordinal) && return AlgorithmProcessDeferred()

    ready = plan.storage.ready[ordinal]
    _enqueue_shack_hartmann_progressive!(plan, workspace, ready)
    _enqueue_measurement_publication!(measurements, columns, count, workspace.packet, 2 * length(ready))
    workspace.terminal_candidate = sample.terminal
    return AlgorithmProcessComplete(_progressive_measurement_metadata(sample, workspace.frame))
end

function _process_shack_hartmann_progressive!(
    reconstructed, prepared::PreparedAlgorithm{D}, input, metadata,
) where {D}
    plan, workspace = _progressive_admitted(prepared, (reconstructed,), (input,))
    _require_progressive_workspace_disjoint(
        (input, reconstructed), (workspace.detector, workspace.packet, workspace.status, workspace.reconstructed),
    )
    missing, overflow, nonfinite = _shack_hartmann_progressive_failures(D)
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return missing
    UInt64(sample.offset) <= UInt64(typemax(Int)) || return overflow
    ordinal = _progressive_block_ordinal!(workspace.frame, sample, input, workspace.detector, plan.storage.mapping)
    isnothing(ordinal) && return AlgorithmProcessDeferred()

    ordinal == 1 && fill!(workspace.reconstructed, 0f0)
    backend = _enqueue_shack_hartmann_progressive!(plan, workspace, plan.storage.ready[ordinal])
    _enqueue_measurement_accumulation!(
        backend, workspace.reconstructed, plan.storage.reconstruction, workspace.packet, workspace.status, nonfinite,
    )
    failure = _progressive_status!(workspace.completion, workspace.status, typeof(nonfinite))
    isnothing(failure) || return failure
    sample.terminal || return AlgorithmProcessDeferred()

    copyto!(reconstructed, workspace.reconstructed)
    KernelAbstractions.synchronize(backend)
    workspace.terminal_candidate = true
    return _progressive_complete(sample, workspace.frame.current_discontinuity)
end

for Declaration in (ShackHartmannMeasurementBlockF32, ShackHartmannWeightedMeasurementBlockF32,
                    ShackHartmannReadoutMeasurementBlockF32)
    @eval function _process_admitted!(measurements::AbstractArray, columns::AbstractArray, count::AbstractArray,
        prepared::PreparedAlgorithm{$Declaration,<:ProgressiveDevicePlan},
        input::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
        return _process_shack_hartmann_progressive!(measurements, columns, count, prepared, input, metadata)
    end
end

for Declaration in (ShackHartmannRowReconstructorF32, ShackHartmannReadoutReconstructorF32)
    @eval function _process_admitted!(reconstructed::AbstractArray,
        prepared::PreparedAlgorithm{$Declaration,<:ProgressiveDevicePlan},
        input::AbstractArray, metadata::FGA.AdmittedSampleMetadata)
        return _process_shack_hartmann_progressive!(reconstructed, prepared, input, metadata)
    end
end
