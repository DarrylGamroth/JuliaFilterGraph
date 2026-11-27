const CapturedContext = CUDACapturedContext
const algorithm_extension = :FilterGraphAlgorithmsCUDAExt
include("../captured_composition.jl")

_prepare_composition_node(args...) = FG._prepare_captured_node(args...)

function _prepare_composition_node(node::FG.PreparedNode{FGA.CenterOfGravityF32},
        outputs, routes, device_inputs, node_outputs, device, status, index, workgroups)
    inputs = FG._route_inputs(routes, device_inputs, node_outputs)
    extension = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)
    count = length(only(inputs))
    ordered = !(4096 <= count <= 65_536)
    adapter = extension._prepare_center_of_gravity(device, node.prepared,
        outputs, inputs, status; ordered)
    return FG.CapturedNode{FGA.CenterOfGravityF32,typeof(adapter)}(adapter)
end

"""Capture the stages of a fresh prepared composition on CUDA.

Bind calibration before capture. Runtime properties are read at each stage's
execution boundary. The host retains arrival admission and publication, while each accepted stage runs as a native
CUDA executable. One shared DM transition consumes the retained corrections
after the admitted camera stages finish. Set `parallel_cog=true` to enable a
parallel reduction for large CoG windows. Validity remains guarded by ordered
fallback. Centroids can round differently and require controller qualification.
"""
function FG.prepare_captured_graph(composition::FG.PreparedComposition, device::CUDA.CuDevice;
        parallel_cog::Bool=false)
    prepare_node = parallel_cog ? _prepare_composition_node : FG._prepare_captured_node
    extension = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)
    return _prepare_captured_composition(composition, device, extension, prepare_node)
end
