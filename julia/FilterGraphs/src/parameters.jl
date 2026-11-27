function _parameter_name(name::Symbol)
    isempty(String(name)) && throw(
        ArgumentError("Parameter Port name must not be empty"),
    )
    return name
end

function _parameter_name(name::AbstractString)
    isempty(name) && throw(
        ArgumentError("Parameter Port name must not be empty"),
    )
    return Symbol(name)
end

function _parameter_name(name)
    throw(
        ArgumentError(
            "Parameter Port name must be a Symbol or string, received " *
            "$(typeof(name))",
        ),
    )
end

function _parameter_route(graph::PreparedGraph, name::Symbol)
    hasproperty(graph.parameter_routes, name) || throw(
        ArgumentError(
            "Graph has no external Parameter Port '$name'; configured Ports are " *
            "$(keys(graph.parameter_formats))",
        ),
    )
    return getproperty(graph.parameter_routes, name)
end

function _prepare_parameter_update!(
    candidate_plans::Vector{Any},
    touched::BitVector,
    nodes::Tuple,
    ::ParameterInput{NodeIndex,Name},
    values::AbstractArray,
) where {NodeIndex,Name}
    node = getfield(nodes, NodeIndex)
    candidate_plans[NodeIndex] =
        FilterGraphAlgorithms._prepare_parameter_replacement(
            node.prepared,
            candidate_plans[NodeIndex],
            Val(Name),
            values,
        )
    touched[NodeIndex] = true
    return nothing
end

struct PreparedParameterTransaction{Plans,Touched,Derived}
    plans::Plans
    touched::Touched
    derived::Derived
end

_parameter_candidate(node, plan, touched::Bool) = nothing
_adopt_parameter_candidate!(node, ::Nothing) = nothing

function _parameter_candidate(
    node::PreparedNode{D,P}, plan, touched::Bool,
) where {D,P<:PreparedCpuDenseReconstructor}
    executor = _cpu_dense_executor(node)
    touched || return (; bounds=executor.shadow_column_bounds, layout=executor.matrix_layout)
    return _cpu_bound_candidate(node, plan, touched)
end

function _adopt_parameter_candidate!(
    node::PreparedNode{D,P}, candidate::NamedTuple,
) where {D,P<:PreparedCpuDenseReconstructor}
    executor = _cpu_dense_executor(node)
    # Untouched candidates retain the active layout and must leave its bounds alone.
    executor.matrix === node.prepared.plan.matrix && return nothing
    copyto!(executor.column_bounds, candidate.bounds)
    executor.matrix_layout = candidate.layout
    executor.matrix = node.prepared.plan.matrix
    return nothing
end

function _parameter_transaction(graph::PreparedGraph, plans::Tuple, touched::Tuple)
    # Preparation owns only candidate storage; active workers retain their old
    # matrix, bounds and layout until adoption discards and joins their frame.
    derived = map(_parameter_candidate, graph.nodes, plans, touched)
    return PreparedParameterTransaction(plans, touched, derived)
end

function _prepare_parameter_transaction(graph::PreparedGraph, updates::Pair...)
    isempty(updates) && throw(
        ArgumentError("Parameter Port transaction must not be empty"),
    )

    names = Symbol[]
    routes = Any[]
    for update in updates
        name = _parameter_name(first(update))
        name in names && throw(
            ArgumentError("Parameter Port '$name' is assigned more than once"),
        )
        values = last(update)
        values isa AbstractArray || throw(
            ArgumentError(
                "Parameter Port '$name' requires an AbstractArray, received " *
                "$(typeof(values))",
            ),
        )
        push!(names, name)
        push!(routes, _parameter_route(graph, name))
    end

    candidate_plans = Any[node.prepared.plan for node in graph.nodes]
    touched = falses(length(graph.nodes))
    for index in eachindex(updates)
        _prepare_parameter_update!(
            candidate_plans,
            touched,
            graph.nodes,
            routes[index],
            last(updates[index]),
        )
    end
    plans = Tuple(candidate_plans)
    active_plan_type = typeof(map(node -> node.prepared.plan, graph.nodes))
    typeof(plans) === active_plan_type || error(
        "prepared Parameter Port transaction changed a Plan type",
    )
    return _parameter_transaction(graph, plans, Tuple(touched))
end

@inline _adopt_parameter_plans!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

@inline function _adopt_parameter_plans!(nodes::Tuple, plans::Tuple, touched::Tuple)
    if first(touched)
        node = first(nodes)
        FilterGraphAlgorithms._plan_adopted!(node.prepared.workspace)
        node.prepared.plan = first(plans)
    end
    return _adopt_parameter_plans!(
        Base.tail(nodes),
        Base.tail(plans),
        Base.tail(touched),
    )
end

@inline function _adopt_parameter_transaction!(
    graph::PreparedGraph,
    transaction::PreparedParameterTransaction,
)
    _discard_inflight!(graph)
    _adopt_parameter_plans!(graph.nodes, transaction.plans, transaction.touched)
    map(_adopt_parameter_candidate!, graph.nodes, transaction.derived)
    return graph
end

"""
    replace_parameters!(graph::PreparedGraph, updates::Pair...)

Replace a nonempty set of external Parameter Ports as one transaction. Pair
keys are the external Port names reported by `parameter_formats(graph)`;
values are ordinary one-based Julia arrays. Every candidate Plan is prepared
before the Graph abandons any in-flight publication unit or changes an active
Plan. An accepted transaction reconciles each affected Workspace exactly once;
as with `commit!` and `discard!`, that lifecycle hook is an admitted,
non-failing operation.

This checked control-plane call may allocate and throw. It must be serialized
with `process!` and other control operations. Fixed-calibration captured, pipeline,
and GPU readout owners reject replacement; prepare a fresh checked Graph, install
its Parameters, then prepare a new specialized owner. Do not mutate the owner's
retained Graph.
"""
function replace_parameters!(graph::PreparedGraph, updates::Pair...)
    transaction = _prepare_parameter_transaction(graph, updates...)
    return _adopt_parameter_transaction!(graph, transaction)
end

function replace_parameters!(
    owner::Union{PreparedCapturedGraph,PreparedCapturedPipeline,PreparedGpuReadoutGraph},
    updates::Pair...,
)
    throw(ArgumentError(
        "this owner has fixed calibration; prepare a fresh checked Graph, " *
        "install its external Parameters with replace_parameters!, and prepare a new owner " *
        "with the same preparation function. Switch at a serialized boundary and close " *
        "the old owner where applicable. Do not mutate the owner's retained Graph.",
    ))
end
