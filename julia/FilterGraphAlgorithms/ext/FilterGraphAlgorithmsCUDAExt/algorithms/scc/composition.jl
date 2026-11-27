capture_support(::Type{ChoppedImageDifferenceF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{ChoppedImageDifferenceF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_chopped_capture(device, prepared, outputs, inputs, status)
end

capture_support(::Type{PixelSelectionF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{PixelSelectionF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_pixel_capture(device, prepared, outputs, inputs, status)
end

capture_support(::Type{PixelValidityF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{PixelValidityF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_pixel_capture(device, prepared, outputs, inputs, status)
end

capture_support(::Type{ValidityGatedClosedLoopIntegratorF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{ValidityGatedClosedLoopIntegratorF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32, workgroups=nothing)
    return _prepare_composition_integrator(device, prepared, outputs, inputs, status, node_index)
end
