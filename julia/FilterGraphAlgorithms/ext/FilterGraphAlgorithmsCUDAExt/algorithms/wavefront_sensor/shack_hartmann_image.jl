# Every centroid workgroup has exactly one full CUDA warp. All 32 lanes,
# including those without pixels or a ready subaperture, participate. The
# descending shuffle tree preserves the shared tree's lane-one addition order.
@inline function _warp_moments(x, y, flux)
    for delta in (16, 8, 4, 2, 1)
        x += CUDA.shfl_down_sync(CUDA.FULL_MASK, x, delta)
        y += CUDA.shfl_down_sync(CUDA.FULL_MASK, y, delta)
        flux += CUDA.shfl_down_sync(CUDA.FULL_MASK, flux, delta)
    end
    return x, y, flux
end

function _shack_hartmann_kernel(device::CUDA.CuDevice, backend)
    _require_capture_workgroup(device, SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE; warp_width=32)
    return _ShackHartmannKernel(
        _shack_hartmann_image!(backend, SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE), _warp_moments)
end

function _shack_hartmann_readout_kernel(backend::CUDA.CUDACore.CUDABackend)
    _require_capture_workgroup(CUDA.device(), SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE; warp_width=32)
    return _ShackHartmannKernel(
        _shack_hartmann_readout_cooperative!(backend, SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE),
        _warp_moments)
end
