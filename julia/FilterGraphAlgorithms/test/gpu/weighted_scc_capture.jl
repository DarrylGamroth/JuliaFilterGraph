module WeightedSccCaptureTests

using CUDA, FilterGraphAlgorithms, KernelAbstractions, LinearAlgebra, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, complete!, process!, reset!

include("weighted_scc_common.jl")

function node_count(graph)
    count = Ref{Csize_t}(0)
    CUDA.CUDACore.cuGraphGetNodes(graph, C_NULL, count)
    return Int(count[])
end
used_memory() = begin
    bytes = CUDA.used_memory()
    ismissing(bytes) ? CUDA.total_memory() - CUDA.free_memory() : bytes
end
# Newer SDKs defer finalizer retirement beyond a normal stream synchronize.
function prepare_memory_measurement()
    if isdefined(CUDA, :RECLAIM_GC)
        CUDA.reclaim(CUDA.RECLAIM_GC)
    end
    return nothing
end
CUDA.allowscalar(false)
run_weighted_scc_capture_tests(CUDA.device(),
    Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt),
    CUDA.zeros, CUDA.ones, CUDA.synchronize, CUDA.capture, CUDA.instantiate,
    CUDA.launch, node_count, used_memory; prepare_memory_measurement)

end
