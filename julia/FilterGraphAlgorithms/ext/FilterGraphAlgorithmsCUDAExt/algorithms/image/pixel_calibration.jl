# Certificate construction is enabled only for the qualified CUDA backend.
_capture_pixel_bounds(::CUDA.CuDevice, plan::PixelCalibrationPlan) =
    _pixel_bounds(plan.flat, plan.background)
