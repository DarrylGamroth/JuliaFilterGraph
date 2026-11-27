# Backends specialize this preparation step for the validation proofs they support.
function _prepare_captured_validation(
        device, nodes, routes, outputs, device_inputs, node_outputs)
    return nodes
end

_captured_input_finite(::ExternalInput, inputs, node_outputs, finite) = false

function _captured_input_finite(route::NodeOutput, inputs, node_outputs, finite)
    value = _route_value(route, inputs, node_outputs)
    return any(buffer -> buffer === value, finite)
end

function _captured_finite_outputs!(finite, adapter, outputs, finite_output)
    for (index, output) in enumerate(outputs)
        if finite_output(adapter, Val(index))
            push!(finite, output)
        end
    end
    return nothing
end

function _reuse_captured_validation(
        ::Tuple{}, ::Tuple{}, ::Tuple{}, device_inputs, node_outputs, finite, validation)
    return ()
end

function _reuse_captured_validation(
        nodes::Tuple, routes::Tuple, outputs::Tuple, device_inputs, node_outputs, finite, validation
)
    node, node_routes, node_outputs_here = first(nodes), first(routes), first(outputs)
    adapter = node.adapter
    inputs = _route_inputs(node_routes, device_inputs, node_outputs)
    readonly = validation.preserves_inputs(adapter)
    safe = _captured_safe_overlap(adapter, node_outputs_here, inputs)
    if readonly && safe
        proofs = map(node_routes) do route
            _captured_input_finite(route, device_inputs, node_outputs, finite)
        end
        adapter = validation.prepare_validation(adapter, proofs)
    end

    # Treat an unknown writer as an opaque barrier, even when it is not known
    # to alias a proven buffer. A qualified reader may still overwrite its
    # supplied outputs; any prior proof of overlapping storage expires now.
    if !readonly || !safe
        empty!(finite)
    else
        filter!(finite) do buffer
            all(enumerate(node_outputs_here)) do (index, output)
                alias = FilterGraphAlgorithms.alias_input(adapter, Val(index))
                !_captured_mightalias(buffer, output) ||
                    (alias !== nothing && output === buffer && inputs[alias] === buffer)
            end
        end
    end
    _captured_finite_outputs!(finite, adapter, node_outputs_here, validation.finite_output)
    captured = _captured_node_with_adapter(node, adapter)
    return (captured,
        _reuse_captured_validation(
            Base.tail(nodes), Base.tail(routes), Base.tail(outputs),
            device_inputs, node_outputs, finite, validation
        )...)
end

function _captured_node_with_adapter(::CapturedNode{D}, adapter) where {D}
    return CapturedNode{D, typeof(adapter)}(adapter)
end

function _reuse_captured_validation(nodes, routes, outputs, device_inputs, node_outputs;
        validation=(preserves_inputs=FilterGraphAlgorithms.preserves_inputs,
            finite_output=FilterGraphAlgorithms.finite_output,
            prepare_validation=FilterGraphAlgorithms.prepare_validation))
    # Temporary control-plane facts. No proof set survives into replay, and no
    # fields or device allocations are added to a captured Graph owner.
    finite = AbstractArray[]
    return _reuse_captured_validation(
        nodes, routes, outputs, device_inputs, node_outputs, finite, validation)
end
