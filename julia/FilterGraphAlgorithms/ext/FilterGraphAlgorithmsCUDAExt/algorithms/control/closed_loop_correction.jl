# CUDA supplies the native workgroup vote; the adapter lives in the GPU layer.
struct _BlockAny end
@inline (::_BlockAny)(failure) = CUDA.sync_threads_or(failure)

import FilterGraphAlgorithms: capture_support, prepare_capture

capture_support(::Type{ClosedLoopCorrectionF32}, ::CUDA.CuDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::CUDA.CuDevice,
        prepared::PreparedAlgorithm{
            ClosedLoopCorrectionF32, <:ClosedLoopCorrectionPlan, <:ClosedLoopCorrectionWorkspace
        },
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, _BlockAny())
end
