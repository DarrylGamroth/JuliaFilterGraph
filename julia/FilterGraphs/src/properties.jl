function _qualified_property_name(name::Union{Symbol,AbstractString})
    text = String(name)
    separator = findfirst(':', text)
    separator === nothing && throw(
        ArgumentError("Graph Property '$text' must use the qualified name 'node:property'"),
    )
    separator == firstindex(text) && throw(
        ArgumentError("Graph Property '$text' has no Node name"),
    )
    separator == lastindex(text) && throw(
        ArgumentError("Graph Property '$text' has no algorithm-local Property name"),
    )
    return Symbol(SubString(text, firstindex(text), prevind(text, separator))),
    Symbol(SubString(text, nextind(text, separator), lastindex(text)))
end

_qualified_property_name(name) = throw(
    ArgumentError("Graph Property name must be a Symbol or string, received $(typeof(name))"),
)

function _property_node_index(graph::PreparedGraph, name::Symbol)
    for index in eachindex(graph.nodes)
        graph.nodes[index].name === name && return index
    end
    throw(ArgumentError("Graph has no Node '$name'"))
end

function _named_property_assignment(plan, node::Symbol, name::Symbol, value)
    plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan || throw(
        ArgumentError("Node '$node' has no scalar Property contract"),
    )
    for info in FilterGraphAlgorithms.property_infos(plan)
        info.name === name || continue
        value isa FilterGraphAlgorithms.PropertyScalar || throw(
            FilterGraphAlgorithms.PropertyTypeError(
                info.id, typeof(info.default_value), typeof(value),
            ),
        )
        return FilterGraphAlgorithms.PropertyAssignment(info.id, value)
    end
    throw(ArgumentError("Node '$node' has no Property '$name'"))
end

function _prepare_property_transaction(graph::PreparedGraph, updates::Pair...)
    isempty(updates) && throw(ArgumentError("Property transaction must not be empty"))
    assignments = [FilterGraphAlgorithms.PropertyAssignment[] for _ in graph.nodes]
    names = Set{Tuple{Symbol,Symbol}}()
    for update in updates
        name = _qualified_property_name(first(update))
        name in names && throw(
            ArgumentError("Graph Property '$(name[1]):$(name[2])' is assigned more than once"),
        )
        push!(names, name)
        node_index = _property_node_index(graph, name[1])
        plan = graph.nodes[node_index].prepared.plan
        push!(assignments[node_index], _named_property_assignment(plan, name..., last(update)))
    end

    active_plans = map(node -> node.prepared.plan, graph.nodes)
    # This checked control-plane operation prepares all replacements before
    # touching active Plans or any in-flight publication unit.
    plans = map(active_plans, Tuple(assignments)) do plan, updates
        isempty(updates) ? plan : FilterGraphAlgorithms.prepare_properties(plan, updates)
    end
    typeof(plans) === typeof(active_plans) || error(
        "prepared Property transaction changed a Plan type",
    )
    return (; plans, touched=map(updates -> !isempty(updates), Tuple(assignments)))
end

"""
    replace_properties!(graph::PreparedGraph, updates::Pair...) -> graph

Replace scalar runtime Properties as one transaction. Keys are qualified Node
and Property names, such as `"controller:gain"` or `Symbol("controller:pole")`.
Values must have the exact scalar types declared by each Algorithm's Property
schema; for a Float32 controller, use `0.2f0` rather than `0.2`.

All assignments, including each Algorithm's cross-property constraints, are
validated together before active Plans or in-flight work change. On acceptance,
the Graph discards in-flight publication units and reconciles affected Workspaces
using the same non-failing lifecycle hook as Parameter replacement. This hook
is Algorithm-specific: a leaky integrator retains committed state, while other
Algorithms may reset their state after a Plan change.

This checked control-plane call may allocate and throw. Serialize it with
`process!`, `reset!`, and other control-plane calls. It applies to ordinary
prepared Graphs; specialized execution owners require preparation of a new owner
from updated configuration rather than mutation of their retained Graph.

# Example

```julia
replace_properties!(graph, "controller:gain" => 0.2f0, "controller:pole" => 0.95f0)
```
"""
function replace_properties!(graph::PreparedGraph, updates::Pair...)
    transaction = _prepare_property_transaction(graph, updates...)
    _discard_inflight!(graph)
    _adopt_parameter_plans!(graph.nodes, transaction.plans, transaction.touched)
    return graph
end

function replace_properties!(
    owner::Union{
        PreparedCapturedGraph,
        PreparedCapturedPipeline,
        PreparedCpuProgressiveGraph,
        PreparedGpuProgressiveGraph,
        PreparedGpuReadoutGraph,
    },
    updates::Pair...,
)
    throw(ArgumentError(
        "replace_properties! supports ordinary PreparedGraph owners only; " *
        "stop processing, drain and close the old owner where applicable, " *
        "update the Node's props in the configuration, and prepare a new owner " *
        "with the same preparation function. Do not mutate the owner's retained Graph.",
    ))
end
