module AMDWeightedSccCaptureTests

using AMDGPU, FilterGraphAlgorithms, KernelAbstractions, LinearAlgebra, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, complete!, process!, reset!

include("weighted_scc_common.jl")

function node_count(graph)
    count = Ref{Csize_t}(0)
    AMDGPU.HIP.hipGraphGetNodes(graph, C_NULL, count)
    return Int(count[])
end
run_weighted_scc_capture_tests(AMDGPU.device(),
    Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsAMDGPUExt),
    AMDGPU.zeros, AMDGPU.ones, AMDGPU.synchronize, AMDGPU.HIP.capture,
    AMDGPU.HIP.instantiate, AMDGPU.HIP.launch, node_count, AMDGPU.used_memory)

end
