# Kernel bundle consumed by the progressive readout executor.
function _readout_kernels(backend)
    return (;
        scatter! = _scatter_readout!(backend, 256),
        pixels! = _pyramid_readout_pixels!(backend, PYRAMID_READOUT_LANES),
        finish! = _finish_pyramid_readout!(backend, PYRAMID_READOUT_LANES),
        shack_hartmann! = _shack_hartmann_readout!(backend, SHACK_HARTMANN_READOUT_LANES),
        shack_hartmann_cooperative! = _shack_hartmann_readout_kernel(backend),
        adopt! = _adopt_pyramid_readout_gain!(backend, 1)
    )
end
