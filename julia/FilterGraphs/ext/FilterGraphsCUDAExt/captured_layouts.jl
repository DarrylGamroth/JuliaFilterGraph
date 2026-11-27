using FilterGraphAlgorithms: PyramidPupilImageF32, PyramidPupilReconstructorF32
import FilterGraphAlgorithms

# Only this qualified internal layout may bypass dense CUDA storage admission.
FilterGraphs._captured_storage(
    array::PermutedDimsArray{Float32,2,(2,1),(2,1),<:CUDA.CuArray},
) = parent(array)

function FilterGraphs._prepare_captured_layouts(
        ::CUDA.CuDevice, nodes, routes, output_routes, outputs, roots=nothing)
    all(node -> FilterGraphAlgorithms.preserves_inputs(node.adapter), nodes) ||
        return outputs
    return ntuple(length(nodes)) do index
        node = nodes[index]
        node isa FilterGraphs.CapturedNode{PyramidPupilImageF32} || return outputs[index]
        source = FilterGraphs.NodeOutput{index, 1}()
        source in values(output_routes) && return outputs[index]
        consumers = findall(routes) do inputs
            source in inputs
        end
        length(consumers) == 1 || return outputs[index]
        consumer = only(consumers)
        nodes[consumer] isa FilterGraphs.CapturedNode{PyramidPupilReconstructorF32} ||
            return outputs[index]
        routes[consumer] == (source,) || return outputs[index]
        pixels, mean = outputs[index]
        # Preserve the logical (pupil, selected) matrix while storing pixels in
        # the reconstructor's existing pupil-major order. No matrix permutation
        # or change to floating-point accumulation is needed.
        storage = CUDA.zeros(eltype(pixels), size(pixels, 2), size(pixels, 1))
        isnothing(roots) || push!(roots, storage)
        return (PermutedDimsArray(storage, (2, 1)), mean)
    end
end
