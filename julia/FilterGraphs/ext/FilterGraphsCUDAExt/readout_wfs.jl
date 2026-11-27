# These buffers belong to one frame. The WFS stream orders detector writes;
# the compute stream reads immutable per-ordinal packets after an event.
struct CUDAReadoutStorage{D, B, I, P, S, C}
    detector::D
    device_blocks::B
    detector_indices::I
    packets::P
    statuses::S
    counts::Vector{Int}
    calibration::C
end

struct CUDAPyramidReadout{P, S, I, V, F, K}
    plan::P
    storage::S
    selected_indices::I
    columns::I
    full_pupil_indices::V
    flags::I
    committed_gain::F
    candidate_gain::F
    mean::F
    kernels::K
end

struct CUDAShackHartmannReadout{P, S, I, M, F, B, K}
    plan::P
    storage::S
    ready::I
    origins::M
    x_coordinates::F
    y_coordinates::F
    reference_slopes::F
    pixel_thresholds::F
    flux_thresholds::F
    active::B
    kernels::K
end

function _prepare_cuda_readout_storage(plan, device_blocks, counts, capacity, calibration)
    length(device_blocks) == plan.block_count || throw(
        DimensionMismatch("CUDA readout requires one device block per ordinal"),
    )
    shape = FilterGraphAlgorithms.detector_readout_block_shape(plan)
    input_shape = isnothing(calibration) ? shape : (shape[2], shape[3])
    all(block -> size(block) == input_shape, device_blocks) ||
        throw(DimensionMismatch("CUDA readout blocks must match calibrated F32 input"))
    prod(plan.detector_size) <= typemax(Int32) || throw(
        ArgumentError("CUDA readout detector exceeds Int32 indexing capacity"),
    )
    capacity <= typemax(UInt32) || throw(
        ArgumentError("CUDA readout measurement count exceeds UInt32 capacity"),
    )
    zero_entry = CUDADenseMeasurement(0.0f0, UInt32(0))
    packets = [CUDA.fill(zero_entry, capacity + 1) for _ in 1:plan.block_count]
    return CUDAReadoutStorage(
        CUDA.zeros(Float32, plan.detector_size),
        device_blocks,
        CUDA.CuArray.(FilterGraphAlgorithms._readout_detector_indices(plan, Int32)),
        packets,
        [CUDA.zeros(Int32, 1) for _ in 1:plan.block_count],
        counts, calibration
    )
end

function _cuda_readout_kernels(detector)
    extension = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
    isnothing(extension) &&
        throw(ArgumentError("CUDA readout measurement kernels are unavailable"))
    return extension._readout_kernels(KernelAbstractions.get_backend(detector))
end

function _prepare_cuda_readout_wfs(
        prepared::FilterGraphAlgorithms.PreparedAlgorithm{FilterGraphAlgorithms.PyramidReadoutMeasurementBlockF32},
        device_blocks,
        calibration
)
    plan = prepared.plan.source
    capacity = plan.normalization.output_pixel_count
    storage = _prepare_cuda_readout_storage(
        plan.readout, device_blocks, length.(plan.completion_groups), capacity, calibration
    )
    linear = LinearIndices(plan.readout.detector_size)
    selected = [Int32[linear[plan.selected_detector_rows[pixel], plan.selected_detector_columns[pixel]]
                      for pixel in group] for group in plan.completion_groups]
    pupil_rows, pupil_columns = plan.regions.region_size
    full_pupil = Int32[]
    for (first_row, first_column) in plan.regions.origins
        for row in 1:pupil_rows, column in 1:pupil_columns

            push!(full_pupil, Int32(linear[first_row + row, first_column + column]))
        end
    end
    return CUDAPyramidReadout(
        prepared.plan,
        storage,
        CUDA.CuArray.(selected),
        CUDA.CuArray.([Int32.(group) for group in plan.completion_groups]),
        CUDA.CuArray(full_pupil),
        [CUDA.zeros(Int32, length(group)) for group in plan.completion_groups],
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Float32, 1),
        CUDA.zeros(Float32, 1),
        _cuda_readout_kernels(storage.detector)
    )
end

function _prepare_cuda_readout_wfs(
        prepared::FilterGraphAlgorithms.PreparedAlgorithm{FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32},
        device_blocks,
        calibration
)
    plan = prepared.plan
    sensing = plan.sensing
    storage = _prepare_cuda_readout_storage(
        plan.readout, device_blocks, 2 .* length.(plan.completion_groups),
        2 * length(sensing.regions.origins), calibration
    )
    return CUDAShackHartmannReadout(
        plan,
        storage,
        CUDA.CuArray.([Int32.(group) for group in plan.completion_groups]),
        CUDA.CuArray(hcat((Int32[origin...] for origin in sensing.regions.origins)...)),
        CUDA.CuArray(sensing.x_coordinates),
        CUDA.CuArray(sensing.y_coordinates),
        CUDA.CuArray(sensing.reference_slopes),
        CUDA.CuArray(sensing.pixel_thresholds),
        CUDA.CuArray(sensing.flux_thresholds),
        CUDA.CuArray(sensing.active),
        _cuda_readout_kernels(storage.detector)
    )
end

@kernel function _cuda_calibrate_readout!(detector, block, indices, flat, background)
    index = @index(Global, Linear)
    @inbounds begin
        pixel = indices[index]
        detector[pixel] = flat[pixel] * (Float32(block[index]) - background[pixel])
    end
end

function _enqueue_cuda_readout_scatter!(storage, scatter!, ordinal)
    detector = storage.detector
    backend = KernelAbstractions.get_backend(detector)
    block = storage.device_blocks[ordinal]
    indices = storage.detector_indices[ordinal]
    if isnothing(storage.calibration)
        scatter!(detector, block, indices; ndrange=length(block))
    else
        _cuda_calibrate_readout!(backend, 256)(
            detector, block, indices, storage.calibration.flat, storage.calibration.background;
            ndrange=length(block))
    end
    return nothing
end

_empty_raw_pyramid_block(wfs, ordinal) = false
_empty_raw_pyramid_block(wfs::CUDAPyramidReadout, ordinal) =
    !isnothing(wfs.storage.calibration) && iszero(wfs.storage.counts[ordinal])

function _enqueue_cuda_readout_wfs!(wfs::CUDAPyramidReadout, ordinal::Int)
    storage = wfs.storage
    _enqueue_cuda_readout_scatter!(storage, wfs.kernels.scatter!, ordinal)
    terminal = ordinal == length(storage.packets)
    !terminal && _empty_raw_pyramid_block(wfs, ordinal) && return nothing
    if !iszero(storage.counts[ordinal])
        wfs.kernels.pixels!(
            storage.packets[ordinal], wfs.flags[ordinal], storage.detector,
            wfs.selected_indices[ordinal],
            wfs.columns[ordinal],
            wfs.committed_gain;
            ndrange = storage.counts[ordinal]
        )
    end
    wfs.kernels.finish!(
        storage.packets[ordinal], storage.statuses[ordinal], wfs.flags[ordinal],
        storage.detector, wfs.full_pupil_indices, wfs.plan.source.normalization.mean_scale,
        wfs.candidate_gain, wfs.mean, terminal; ndrange = 256
    )
    return nothing
end

function _enqueue_cuda_readout_wfs!(wfs::CUDAShackHartmannReadout, ordinal::Int)
    storage = wfs.storage
    _enqueue_cuda_readout_scatter!(storage, wfs.kernels.scatter!, ordinal)
    region_rows, region_columns = wfs.plan.sensing.regions.region_size
    count = max(1, length(wfs.plan.completion_groups[ordinal]))
    # This selection runs during capture, not on each callback. Tiny windows
    # avoid a cooperative reduction; larger windows share their pixel work.
    cooperative = region_rows * region_columns > 32
    measure! = cooperative ? wfs.kernels.shack_hartmann_cooperative! :
               wfs.kernels.shack_hartmann!
    measure!(
        storage.packets[ordinal], storage.statuses[ordinal], storage.detector,
        wfs.ready[ordinal], wfs.origins, wfs.x_coordinates, wfs.y_coordinates,
        wfs.reference_slopes, wfs.pixel_thresholds, wfs.flux_thresholds, wfs.active,
        region_rows, region_columns;
        ndrange = cooperative ? (32, count) : count
    )
    return nothing
end

function _enqueue_cuda_readout_terminal!(wfs::CUDAPyramidReadout, global_status)
    wfs.kernels.adopt!(
        wfs.committed_gain, wfs.candidate_gain, global_status; ndrange = 1
    )
    return nothing
end

_enqueue_cuda_readout_terminal!(::CUDAShackHartmannReadout, global_status) = nothing

function _reset_cuda_readout_wfs!(wfs::CUDAPyramidReadout)
    fill!(wfs.storage.detector, 0.0f0)
    fill!(wfs.committed_gain, 0.0f0)
    fill!(wfs.candidate_gain, 0.0f0)
    fill!(wfs.mean, 0.0f0)
    return nothing
end

function _reset_cuda_readout_wfs!(wfs::CUDAShackHartmannReadout)
    fill!(wfs.storage.detector, 0.0f0)
    return nothing
end

function _cuda_readout_failure(::CUDAPyramidReadout, status::Int32)
    FilterGraphAlgorithms.PyramidReadoutReconstructionFailure(status)
end
_cuda_readout_failure(::CUDAShackHartmannReadout, status::Int32) = nothing
