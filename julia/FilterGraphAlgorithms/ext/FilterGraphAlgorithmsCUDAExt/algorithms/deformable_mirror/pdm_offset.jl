# Public CUDA qualification delegates to shared GPU preparation.

capture_support(::Type{PdmSystemFlatF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{PdmSystemFlatF32, <:PdmSystemFlatPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PdmEngineeringOffsetF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{PdmEngineeringOffsetF32, <:PdmEngineeringOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end
