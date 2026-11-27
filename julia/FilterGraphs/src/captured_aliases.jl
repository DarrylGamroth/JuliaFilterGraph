# CUDA qualifies this cold pass. Other backends retain distinct node outputs.
_prepare_captured_aliases(device, nodes, routes, inputs, outputs, protected) = outputs

_prepare_captured_layouts(device, nodes, routes, output_routes, outputs, roots=nothing) = outputs

_captured_storage(array) = array
_captured_mightalias(left, right) =
    Base.mightalias(_captured_storage(left), _captured_storage(right))

function _prepare_captured_bindings(
        context, graph, device, status, inputs, outputs, workgroups, feedback)
    # Lifetime roots only: retain both preparation passes until queued work drains.
    roots = Any[context, status, inputs, outputs]
    try
        nodes = _prepare_captured_nodes(
            graph.nodes, outputs, graph.input_routes, inputs, outputs, device,
            status.value, Int32(1), workgroups; roots,
        )
        protected = map(name -> getproperty(inputs, name), keys(feedback))
        finalized = _prepare_captured_aliases(
            device, nodes, graph.input_routes, inputs, outputs, protected,
        )
        finalized = _prepare_captured_layouts(
            device, nodes, graph.input_routes, graph.output_routes, finalized, roots,
        )
        push!(roots, finalized)
        if !_captured_aliases_unchanged(outputs, finalized)
            nodes = _prepare_captured_nodes(
                graph.nodes, finalized, graph.input_routes, inputs, finalized,
                device, status.value, Int32(1), workgroups; roots,
            )
            # The discarded pass can contain queued asynchronous initialization.
            # Drain it while both old and new adapter storage remain rooted.
            GC.@preserve roots _synchronize_captured_graph!(context)
        end
        return nodes, finalized
    catch
        _drain_captured_storage!(context, roots)
        rethrow()
    end
end

function _captured_alias_outputs(nodes, routes, inputs, outputs, protected=())
    # An opaque adapter may write through hidden aliases or retain asynchronous
    # writers. Do not share any routed output in an unqualified graph.
    all(node -> FilterGraphAlgorithms.preserves_inputs(node.adapter), nodes) ||
        return outputs
    finalized = ()
    for (node, node_routes, node_outputs) in zip(nodes, routes, outputs)
        node_inputs = _route_inputs(node_routes, inputs, finalized)
        aliases = ntuple(length(node_outputs)) do index
            output = node_outputs[index]
            input_index = FilterGraphAlgorithms.alias_input(node.adapter, Val(index))
            isnothing(input_index) && return output
            input_index isa Int && 1 <= input_index <= length(node_inputs) ||
                throw(GraphConfigurationError("captured output alias has an invalid input index"))
            input = node_inputs[input_index]
            # Feedback commits overwrite delayed inputs before host publication.
            # Keep identity snapshots rooted in those inputs in distinct storage.
            any(buffer -> _captured_mightalias(buffer, input), protected) && return output
            axes(input) == axes(output) && eltype(input) === eltype(output) ||
                throw(GraphConfigurationError("captured output alias requires identical array axes and element type"))
            return input
        end
        finalized = (finalized..., aliases)
    end
    return finalized
end

function _captured_aliases_unchanged(original, finalized)
    return all(zip(original, finalized)) do (before, after)
        all(pair -> pair[1] === pair[2], zip(before, after))
    end
end

function _captured_safe_overlap(adapter, outputs, inputs)
    return all(enumerate(outputs)) do (index, output)
        alias = FilterGraphAlgorithms.alias_input(adapter, Val(index))
        all(enumerate(inputs)) do (input_index, input)
            !_captured_mightalias(output, input) ||
                (alias == input_index && output === input)
        end
    end
end
