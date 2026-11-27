module FilterGraphAlgorithmsAMDGPUExt

using AMDGPU
using FilterGraphAlgorithms
import FilterGraphAlgorithms: capture_support, prepare_capture
using KernelAbstractions

const DeviceType = AMDGPU.HIPDevice

include("FilterGraphAlgorithmsAMDGPUExt/device_arrays.jl")
include("gpu/algorithms/docrime/excitation.jl")
include("FilterGraphAlgorithmsAMDGPUExt/algorithms/docrime/excitation.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_pixel.jl")
include("FilterGraphAlgorithmsAMDGPUExt/algorithms/wavefront_sensor/pyramid_pixel.jl")


include("gpu/linear_algebra/arithmetic_bounds.jl")
include("gpu/input_proofs.jl")
include("gpu/linear_algebra/finite_values.jl")
include("gpu/linear_algebra/projection.jl")
include("gpu/captured_ports.jl")
include("FilterGraphAlgorithmsAMDGPUExt/linear_algebra/projection.jl")
include("gpu/algorithms/image/pixel_calibration.jl")
include("gpu/algorithms/wavefront_sensor/bi_o_edge_image.jl")
include("gpu/algorithms/wavefront_sensor/zernike_pupil_signal.jl")
include("gpu/algorithms/wavefront_sensor/curvature_paired_signal.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_image.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_reconstruction.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_pupil_image.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_pupil_reconstruction.jl")
include("gpu/algorithms/wavefront_sensor/shack_hartmann_image.jl")
include("gpu/algorithms/wavefront_sensor/shack_hartmann_weighted_centroid.jl")
include("gpu/algorithms/wavefront_sensor/shack_hartmann_reconstruction.jl")
include("gpu/algorithms/scc/pixel_residual.jl")
include("gpu/algorithms/scc/spatial_carrier_demodulation.jl")
include("gpu/algorithms/scc/reconstruction.jl")
include("gpu/algorithms/scc/weighted_regularized_reconstruction.jl")
include("gpu/algorithms/scc/weighted_regularized_block.jl")
include("FilterGraphAlgorithmsAMDGPUExt/algorithms/scc/weighted_regularized_reconstruction.jl")
include("gpu/algorithms/control/leaky_integrator.jl")
include("gpu/algorithms/control/closed_loop_correction.jl")
include("FilterGraphAlgorithmsAMDGPUExt/algorithms/control/closed_loop_correction.jl")
include("gpu/algorithms/control/frame_delay.jl")
include("gpu/algorithms/deformable_mirror/controller_to_vdm.jl")
include("gpu/algorithms/deformable_mirror/vdm_to_pdm.jl")
include("gpu/algorithms/deformable_mirror/pdm_command.jl")
include("gpu/algorithms/deformable_mirror/pdm_offset.jl")
include("gpu/algorithms/deformable_mirror/pdm_feedback_to_vdm.jl")
include("gpu/algorithms/deformable_mirror/vdm_feedback_to_controller.jl")

include("gpu/algorithms/control/closed_loop_integrator.jl")
include("gpu/algorithms/control/biquad_cascade.jl")
include("gpu/algorithms/vector/vector_linear_combination.jl")
include("gpu/algorithms/vector/vector_offset.jl")
include("gpu/algorithms/deformable_mirror/vdm_shape_offset.jl")

include("FilterGraphAlgorithmsAMDGPUExt/preparation.jl")

include("gpu/algorithms/scc/composition.jl")

capture_support(::Type{ChoppedImageDifferenceF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ChoppedImageDifferenceF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_chopped_capture(device, prepared, outputs, inputs, status)
end

capture_support(::Type{PixelSelectionF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PixelSelectionF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_pixel_capture(device, prepared, outputs, inputs, status)
end

capture_support(::Type{PixelValidityF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{PixelValidityF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_pixel_capture(device, prepared, outputs, inputs, status)
end

_require_cog_workgroup(device::AMDGPU.HIPDevice) =
    _require_capture_workgroup(device, COG_TILE; shared_memory=COG_TILE * (3 * sizeof(Float32) + sizeof(Bool)))

capture_support(::Type{CenterOfGravityF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{CenterOfGravityF32}, outputs::Tuple, inputs::Tuple, status, ::Int32, workgroups=nothing)
    return _prepare_center_of_gravity(device, prepared, outputs, inputs, status; ordered=true)
end

capture_support(::Type{ValidityGatedClosedLoopIntegratorF32}, ::AMDGPU.HIPDevice) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::AMDGPU.HIPDevice,
        prepared::PreparedAlgorithm{ValidityGatedClosedLoopIntegratorF32},
        outputs::Tuple, inputs::Tuple, status, node_index::Int32, workgroups=nothing)
    return _prepare_composition_integrator(device, prepared, outputs, inputs, status, node_index)
end

include("gpu/algorithms/calibration/preparation.jl")
include("gpu/algorithms/image/detector_measurements.jl")
include("gpu/algorithms/image/pixel_calibration_f32.jl")

include("gpu/algorithms/statistics/array_statistics.jl")
include("gpu/algorithms/control/complete_transition.jl")
include("gpu/algorithms/control/pseudo_open_loop_reconstruction.jl")
include("gpu/algorithms/control/corrector_allocation.jl")
include("gpu/algorithms/deformable_mirror/pdm_command_power_limit.jl")
include("gpu/algorithms/deformable_mirror/inter_actuator_stroke_limit.jl")
include("gpu/algorithms/deformable_mirror/pdm_stroke_optimization.jl")
include("gpu/algorithms/vector/physical_offset_reference.jl")
include("gpu/algorithms/optical_gain/correction.jl")
include("gpu/algorithms/optical_gain/dither.jl")
include("gpu/algorithms/optical_gain/estimator.jl")
include("gpu/algorithms/optical_gain/update.jl")
include("gpu/algorithms/scc/correction.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_readout_measurement_block.jl")
include("gpu/algorithms/wavefront_sensor/shack_hartmann_readout_measurement_block.jl")
include("gpu/progressive.jl")
include("gpu/algorithms/image/progressive_readout.jl")
include("gpu/algorithms/wavefront_sensor/incremental_dense_reconstruction.jl")
include("gpu/algorithms/wavefront_sensor/shack_hartmann_progressive.jl")
include("gpu/algorithms/wavefront_sensor/pyramid_progressive.jl")
include("gpu/algorithms/scc/chopped_detector.jl")
include("gpu/algorithms/wavefront_sensor/ml_llowfs.jl")

end
