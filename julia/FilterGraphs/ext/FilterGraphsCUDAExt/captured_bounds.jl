# Cold CUDA-only range propagation uses exact bound-buffer identities and the
# same preservation/alias barriers as finite-value reuse. No fact survives into
# the executable and no new public adapter hook is introduced.
function _reuse_captured_bounds(nodes, routes, outputs, device_inputs, node_outputs)
    extension = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
    isnothing(extension) && return nodes
    facts = Pair{AbstractArray,extension._CaptureBound}[]
    return _reuse_captured_bounds(
        extension, nodes, routes, outputs, device_inputs, node_outputs, facts,
    )
end

_captured_bound(input, facts) = begin
    index = findfirst(fact -> first(fact) === input, facts)
    isnothing(index) ? nothing : last(facts[index])
end

function _reuse_captured_bounds(extension, nodes::Tuple{}, routes::Tuple{}, outputs::Tuple{},
        device_inputs, node_outputs, facts)
    return ()
end

function _reuse_captured_bounds(extension, nodes::Tuple, routes::Tuple, outputs::Tuple,
        device_inputs, node_outputs, facts)
    node, node_routes, node_outputs_here = first(nodes), first(routes), first(outputs)
    adapter = node.adapter
    inputs = FilterGraphs._route_inputs(node_routes, device_inputs, node_outputs)
    readonly = FilterGraphAlgorithms.preserves_inputs(adapter)
    safe = FilterGraphs._captured_safe_overlap(adapter, node_outputs_here, inputs)
    if readonly && safe
        bounds = map(input -> _captured_bound(input, facts), inputs)
        adapter = extension._prepare_bounded_validation(adapter, bounds)
    end
    if !readonly || !safe
        empty!(facts)
    else
        filter!(facts) do fact
            all(enumerate(node_outputs_here)) do (index, output)
                alias = FilterGraphAlgorithms.alias_input(adapter, Val(index))
                !FilterGraphs._captured_mightalias(first(fact), output) ||
                    (alias !== nothing && output === first(fact) && inputs[alias] === first(fact))
            end
        end
    end
    for (index, output) in enumerate(node_outputs_here)
        bound = extension._capture_output_bound(adapter, Val(index))
        isnothing(bound) || push!(facts, output => bound)
    end
    prepared = FilterGraphs._captured_node_with_adapter(node, adapter)
    return (prepared, _reuse_captured_bounds(
        extension, Base.tail(nodes), Base.tail(routes), Base.tail(outputs),
        device_inputs, node_outputs, facts,
    )...)
end
