# Public capture qualification is explicit for this backend. Preparation and
# adapter execution are shared through extension-local GPU helpers.

capture_support(::Type{FrameDelayF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{FrameDelayF32, <:FrameDelayPlan, <:FrameDelayWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{LeakyIntegratorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{LeakyIntegratorF32, <:LeakyIntegratorPlan, <:LeakyIntegratorWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{ControllerToVdmF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ControllerToVdmF32, <:ControllerToVdmPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{<:Union{PdmCommandF32,TipTiltCommandF32}}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{<:Union{PdmCommandF32,TipTiltCommandF32}, <:PdmCommandPlan{Float32}, <:PdmCommandWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PdmFeedbackToVdmF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PdmFeedbackToVdmF32, <:PdmFeedbackToVdmPlan, <:PdmFeedbackToVdmWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{VdmFeedbackToControllerF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{VdmFeedbackToControllerF32, <:VdmFeedbackToControllerPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{VdmToPdmF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{VdmToPdmF32, <:VdmToPdmPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{PixelCalibrationU16F32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PixelCalibrationU16F32, <:PixelCalibrationPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{SccPixelResidualF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{SccPixelResidualF32, <:SccPixelResidualPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{SccReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{SccReconstructorF32, <:DenseReconstructorPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{DenseVectorReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{DenseVectorReconstructorF32, <:DenseReconstructorPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{SccSpatialCarrierDemodulationF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{SccSpatialCarrierDemodulationF32, <:SccSpatialCarrierDemodulationPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{SccWeightedRegularizedReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{SccWeightedRegularizedReconstructorF32, <:SccWeightedRegularizedReconstructionPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{BiOEdgeImageF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{BiOEdgeImageF32, <:BiOEdgeImagePlan{Float32}, <:BiOEdgeImageWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{CurvaturePairedImageF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{CurvaturePairedImageF32, <:CurvaturePairedImagePlan{Float32}, <:CurvaturePairedSignalWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{CurvaturePairedChannelF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{CurvaturePairedChannelF32, <:CurvaturePairedChannelPlan{Float32}, <:CurvaturePairedSignalWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PyramidImageF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PyramidImageF32, <:PyramidImagePlan{Float32}, <:PyramidImageWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PyramidReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PyramidReconstructorF32, <:DenseReconstructorPlan, <:DenseReconstructorWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{ShackHartmannImageF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ShackHartmannImageF32, <:ShackHartmannImagePlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{ShackHartmannReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ShackHartmannReconstructorF32, <:DenseReconstructorPlan, <:DenseReconstructorWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end

capture_support(::Type{ShackHartmannWeightedCentroidF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ShackHartmannWeightedCentroidF32, <:ShackHartmannWeightedCentroidPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{ZernikePupilSignalF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ZernikePupilSignalF32, <:ZernikePupilSignalPlan{Float32}, <:ZernikePupilSignalWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end


capture_support(::Type{PdmSystemFlatF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PdmSystemFlatF32, <:PdmSystemFlatPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PdmEngineeringOffsetF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PdmEngineeringOffsetF32, <:PdmEngineeringOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PyramidPupilImageF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PyramidPupilImageF32, <:PyramidPixelImagePlan, <:PyramidPixelWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{PyramidPupilReconstructorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PyramidPupilReconstructorF32, <:DenseReconstructorPlan, <:DenseReconstructorWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32, workgroups = nothing)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, workgroups)
end


capture_support(::Type{ClosedLoopCorrectionF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ClosedLoopCorrectionF32, <:ClosedLoopCorrectionPlan, <:ClosedLoopCorrectionWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, _BlockAny())
end

# Complete captured transitions for control and vector algorithms.

capture_support(::Type{BiquadCascadeF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{BiquadCascadeF32, <:BiquadCascadePlan{Float32}, <:BiquadCascadeWorkspace{Float32}},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{ClosedLoopIntegratorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ClosedLoopIntegratorF32, <:ClosedLoopCorrectionPlan, <:ClosedLoopCorrectionWorkspace},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index, _BlockAny())
end

capture_support(::Type{VectorLinearCombinationF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{VectorLinearCombinationF32, <:VectorLinearCombinationPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{VectorOffsetF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{VectorOffsetF32, <:VectorOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end

capture_support(::Type{VdmShapeOffsetF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(
        protocol::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{VdmShapeOffsetF32, <:VdmShapeOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    return _prepare_capture(protocol, device, prepared, outputs, inputs,
        frame_status, node_index)
end
