"""A serial CPU graph with independently selectable input and output loan slots."""
struct PreparedCpuLoans{G,I,N,O}
    graph::G
    input_slots::I
    nodes_by_input::N
    node_outputs_by_slot::O
end

"""
    prepare_cpu_loan_node(prepared, outputs, inputs, ::Val{Fixed})

Optional Algorithm hook for the CPU loan executor. Return the original
prepared Algorithm or a prepared binding to the supplied arrays. The
executor calls this once for a node with fixed internal inputs and once
per input slot otherwise. Alternate bindings must share mutable scientific
state and scratch because only one slot executes at a time. Resetting the
first binding must restore the state for every slot.

Exported terminal outputs can change with the output loan. An Algorithm
must use the output arrays passed to its process method for those Ports.
"""
prepare_cpu_loan_node(prepared, outputs, inputs, fixed) = prepared

function _prepare_cpu_loan_binding(node::PreparedNode{D}, outputs, inputs, fixed) where {D}
    prepared = prepare_cpu_loan_node(node.prepared, outputs, inputs, fixed)
    prepared === node.prepared && return node
    return PreparedNode(D, node.name, prepared, node.input_names, node.output_names,
        node.parameter_names, node.input_formats, node.output_formats, node.parameter_formats,
        node.output_conditional, node.output_metadata_sources)
end

function _cpu_loan_bindings(graph, input_slots, node_outputs)
    columns = map(graph.nodes, node_outputs, graph.input_routes) do node, outputs, routes
        if any(route -> route isa ExternalInput, routes)
            return map(input_slots) do slot
                inputs = _route_inputs(routes, slot, node_outputs)
                _prepare_cpu_loan_binding(node, outputs, inputs, Val(false))
            end
        end
        inputs = _route_inputs(routes, first(input_slots), node_outputs)
        bound = _prepare_cpu_loan_binding(node, outputs, inputs, Val(true))
        return ntuple(_ -> bound, length(input_slots))
    end
    return ntuple(index -> map(column -> column[index], columns), length(input_slots))
end

function _bind_cpu_loan_output(node_outputs, ::NodeOutput{Node,Port}, buffer) where {Node,Port}
    outputs = Base.setindex(node_outputs[Node], buffer, Port)
    return Base.setindex(node_outputs, outputs, Node)
end

function _cpu_loan_outputs(graph, slots, node_outputs)
    return map(slots) do slot
        foldl(zip(graph.output_routes, values(slot)); init=node_outputs) do outputs, (route, array)
            _bind_cpu_loan_output(outputs, route, array)
        end
    end
end

function _validate_cpu_loan_slots(kind, formats, slots)
    isempty(slots) && throw(ArgumentError("CPU loan pools must have at least one slot"))
    return map(slots) do slot
        arrays = _validate_external_arrays(kind, formats, slot)
        all(array -> array isa Array, values(arrays)) ||
            throw(ArgumentError("CPU loans require dense host Arrays"))
        arrays
    end
end

function _validate_cpu_loan_aliases(inputs, outputs, node_outputs)
    input_arrays = Tuple(Iterators.flatten(map(values, inputs)))
    output_arrays = Tuple(Iterators.flatten(map(values, outputs)))
    fixed_arrays = Tuple(Iterators.flatten(node_outputs))
    for (index, array) in pairs(output_arrays)
        (any(input -> _external_output_overlap(input, array), input_arrays) ||
            any(previous -> _external_output_overlap(previous, array), output_arrays[1:index-1]) ||
            any(fixed -> _external_output_overlap(fixed, array), fixed_arrays)) &&
            throw(ArgumentError("CPU output loans must not alias input, fixed, or other output buffers"))
    end
    for input in input_arrays
        any(fixed -> _external_output_overlap(fixed, input), fixed_arrays) &&
            throw(ArgumentError("CPU input loans must not alias fixed buffers"))
    end
    return nothing
end

"""
    prepare_cpu_loans(description; algorithms, input_slots, output_slots)

Prepare a serial CPU graph over two caller-owned pools of NamedTuples of
dense host Arrays. The graph binds internal routes once and permits each
frame to select its input and output slots independently. The arrays must
retain their storage and shapes for the life of this prepared owner. An
output loan may not alias any input or another output loan.

The profile accepts unconditional whole-frame Algorithms and distinct,
terminal exported outputs. Ordinary `process!` remains available on graphs
prepared without the loan pools.
"""
function prepare_cpu_loans(description; algorithms::Tuple, input_slots::Tuple, output_slots::Tuple)
    graph = prepare_graph(description; algorithms)
    return _prepare_cpu_loans(graph, input_slots, output_slots)
end

# An enclosing executor may supply its already-admitted fixed node output arrays.
function _prepare_cpu_loans(graph::PreparedGraph, input_slots::Tuple, output_slots::Tuple,
        node_outputs=graph.node_outputs)
    any(node -> any(node.output_conditional), graph.nodes) &&
        throw(GraphConfigurationError("CPU loans require unconditional whole-frame Nodes"))
    all(route -> all(routes -> !(route in routes), graph.input_routes), graph.output_routes) ||
        throw(GraphConfigurationError("CPU loan exports must be terminal Node outputs"))
    length(unique(graph.output_routes)) == length(graph.output_routes) ||
        throw(GraphConfigurationError("CPU loan exports must have distinct Node outputs"))
    inputs = _validate_cpu_loan_slots(:input, graph.input_formats, input_slots)
    outputs = _validate_cpu_loan_slots(:output, graph.output_formats, output_slots)
    _validate_cpu_loan_aliases(inputs, outputs, node_outputs)
    return PreparedCpuLoans(graph, inputs, _cpu_loan_bindings(graph, inputs, node_outputs),
        _cpu_loan_outputs(graph, outputs, node_outputs))
end

@inline _publish_cpu_outputs!(::Nothing, graph, node_outputs) =
    map(route -> _route_available(route, graph.node_output_available), graph.output_routes)

"""
    process_cpu_loan!(owner, input_slot, output_slot[, input_metadata])

Process one frame into the selected output loan. The caller keeps its input
read-only for the call and supplies an output slot with no outstanding readers.
Publish that slot only when the returned availability is true. On rejection or
error, the selected output is unpublished scratch and may contain partial
results. Previously published slots and output metadata retain their usual
ownership and publication rules. Serialize processing and reset.
"""
function process_cpu_loan!(owner::PreparedCpuLoans, input_slot::Int, output_slot::Int,
        input_metadata::NamedTuple=owner.graph.default_input_metadata)
    return _process_cpu_loan!(owner, input_slot, owner.node_outputs_by_slot[output_slot], input_metadata)
end

# The enclosing executor admits this short-lived loan. The terminal output
# is passed into the schedule without storing the borrowed array in the graph.
function _process_cpu_output_loan!(owner::PreparedCpuLoans, output, input_metadata)
    outputs = _bind_cpu_loan_output(first(owner.node_outputs_by_slot),
        only(owner.graph.output_routes), output)
    return _process_cpu_loan!(owner, 1, outputs, input_metadata)
end

function _process_cpu_loan!(owner::PreparedCpuLoans, input_slot::Int, outputs::Tuple, input_metadata)
    graph = owner.graph
    inputs = owner.input_slots[input_slot]
    nodes = owner.nodes_by_input[input_slot]
    metadata = _validate_external_metadata(graph.input_formats, input_metadata)
    result = _process_candidate_admitted!(nothing, graph, inputs, metadata, graph.result_types,
        nodes, outputs)
    if result isa NamedTuple
        result.terminal || throw(GraphConfigurationError("CPU loans cannot publish a deferred frame"))
        return result.availability
    end
    return result
end

input_formats(owner::PreparedCpuLoans) = input_formats(owner.graph)
output_formats(owner::PreparedCpuLoans) = output_formats(owner.graph)
output_metadata(owner::PreparedCpuLoans) = output_metadata(owner.graph)
rejection_context(owner::PreparedCpuLoans) = rejection_context(owner.graph)

function reset!(owner::PreparedCpuLoans)
    graph = owner.graph
    _reset_nodes!(first(owner.nodes_by_input))
    _clear_inflight!(graph.node_inflight)
    _clear_available!(graph.node_output_available)
    _clear_rejection!(graph.rejection)
    return owner
end
