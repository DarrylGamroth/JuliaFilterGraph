using AMDGPU
using FilterGraphAlgorithms
using KernelAbstractions
using FilterGraphs
using Test

import FilterGraphAlgorithms: commit!, process!

AMDGPU.functional() || error("AMDGPU hardware is required for GPU tests")
AMDGPU.allowscalar(false)

const TEST_DEVICE = AMDGPU.device()

include("splitmix64.jl")
include("docrime_excitation.jl")
run_docrime_excitation_tests(TEST_DEVICE, AMDGPU.zeros, AMDGPU.device, AMDGPU.used_memory)
include("eager_capabilities.jl")
run_eager_capability_tests(TEST_DEVICE; additional=(
    ChoppedImageDifferenceF32, PixelSelectionF32, PixelValidityF32,
    CenterOfGravityF32, ValidityGatedClosedLoopIntegratorF32,
))
include("scc_capture.jl")
run_scc_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.fill, AMDGPU.synchronize)
include("structured_projection_common.jl")
include("amdgpu_alias_capture.jl")
include("frame_delay_alias.jl")
run_frame_delay_alias_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
run_structured_projection_tests(TEST_DEVICE,
    Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsAMDGPUExt),
    AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
run_structured_projection_graph_tests(TEST_DEVICE, AMDGPU.synchronize, AMDGPU.used_memory)
include("weighted_scc_amdgpu.jl")
include("numerical_range_capture.jl")
run_numerical_range_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
run_vdm_correction_role_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
include("closed_loop_common.jl")
run_closed_loop_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
run_closed_loop_graph_tests(TEST_DEVICE, AMDGPU.synchronize)
include("pdm_offset_common.jl")
run_pdm_offset_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize)
include("pyramid_pupil_capture.jl")
run_pyramid_pupil_capture_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros,
    AMDGPU.synchronize, AMDGPU.used_memory)
include("dense_vector_capture.jl")
include("dense_vector_captured_graph.jl")
include("shack_hartmann_weighted_centroid_capture.jl")
run_shack_hartmann_weighted_capture_tests(
    TEST_DEVICE,
    AMDGPU.ROCArray,
    AMDGPU.zeros,
    AMDGPU.synchronize,
    AMDGPU.used_memory,
)

include("pyramid_pixel.jl")
run_pyramid_pixel_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.fill,
    AMDGPU.device, AMDGPU.used_memory)

include("progressive_device.jl")
ProgressiveDeviceTests.run_progressive_device_tests(TEST_DEVICE, AMDGPU.ROCArray)

include("pyramid_reconstruction_chain.jl")
include("bi_o_edge_image_capture.jl")
run_bi_o_edge_image_capture_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.fill, AMDGPU.zeros, AMDGPU.synchronize,
    AMDGPU.device, AMDGPU.used_memory,
)
include("zernike_pupil_signal_capture.jl")
run_zernike_pupil_signal_capture_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.fill, AMDGPU.zeros, AMDGPU.synchronize,
    AMDGPU.device, AMDGPU.used_memory,
)
include("curvature_paired_signal_capture.jl")
run_curvature_paired_signal_capture_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.fill, AMDGPU.zeros, AMDGPU.synchronize,
    AMDGPU.device, AMDGPU.used_memory,
)
include("pyramid_image_capture.jl")
run_pyramid_image_capture_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.fill, AMDGPU.zeros, AMDGPU.synchronize,
)

# Complete GPI2 temporal/vector adapters share the portable device contract.
include("gpi2_controls.jl")
run_gpi2_control_tests(TEST_DEVICE,AMDGPU.ROCArray,AMDGPU.zeros,AMDGPU.synchronize)
include("gpi2_vectors.jl")
run_gpi2_vector_capture_tests(TEST_DEVICE,AMDGPU.ROCArray,AMDGPU.zeros,AMDGPU.synchronize)
include("control_algorithms.jl")
run_control_algorithm_tests(TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.zeros, AMDGPU.synchronize;
    record=f -> AMDGPU.HIP.instantiate(AMDGPU.HIP.capture(f)), replay=AMDGPU.HIP.launch)

include(joinpath(@__DIR__, "device_detector_measurements.jl"))
include(joinpath(@__DIR__, "resident_statistics_llowfs_chop.jl"))
function record_resident_test(f)
    executable = AMDGPU.HIP.instantiate(AMDGPU.HIP.capture(f))
    return () -> AMDGPU.HIP.launch(executable, AMDGPU.stream())
end
test_resident_detector_measurements(AMDGPU.device(), AMDGPU.ROCArray, AMDGPU.synchronize, record_resident_test)
test_resident_statistics(AMDGPU.device(), AMDGPU.ROCArray, AMDGPU.synchronize, record_resident_test)
test_resident_llowfs(AMDGPU.device(), AMDGPU.ROCArray, AMDGPU.synchronize, record_resident_test)
test_resident_detector_chop(AMDGPU.device(), AMDGPU.ROCArray, AMDGPU.synchronize)

include("calibration_capture.jl")
CalibrationCaptureTests.run_calibration_capture_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.synchronize, record_resident_test,
)
CalibrationCaptureTests.run_docrime_captured_graph_tests(TEST_DEVICE)

include("device_catalog_regression.jl")
DeviceCatalogRegression.run_catalog_regression_tests(
    TEST_DEVICE, AMDGPU.ROCArray, AMDGPU.synchronize, record_resident_test,
)
