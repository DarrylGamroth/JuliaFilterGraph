# Cold scope admission. No arbitrary internal request is promoted to a static
# reference: supported command paths must be built from the recovered controller,
# zero-input temporal filters and known fixed-parameter projections/offsets.
_recovery_route_source(::FilterGraphs.ExternalInput) = 0
_recovery_route_source(::FilterGraphs.NodeOutput{Node,Port}) where {Node,Port} = Node

_recovery_command_owner(workspace) = false
_recovery_command_owner(::Union{FilterGraphAlgorithms.PdmCommandWorkspace,
    FilterGraphAlgorithms.CorrectorAllocationWorkspace}) = true
_recovery_temporal_owner(workspace) = false
_recovery_temporal_owner(::FilterGraphAlgorithms.BiquadCascadeWorkspace) = true

Base.@nospecializeinfer function _recovery_ancestors!(ancestors, @nospecialize(graph), index)
    ancestors[index] && return nothing
    ancestors[index] = true
    for route in graph.input_routes[index]
        source = _recovery_route_source(route)
        iszero(source) || _recovery_ancestors!(ancestors, graph, source)
    end
    return nothing
end

# Recovery admission walks runtime node indices. Sharing the traversal keeps
# compilation independent of the concrete graph and callback types.
@noinline Base.@nospecializeinfer function _validate_recovery_scope(
    @nospecialize(owner), controller_index, @nospecialize(selected), reference_supported,
)
    graph = owner.graph
    feedback_required = falses(length(graph.nodes))
    workspace = graph.nodes[controller_index].prepared.workspace
    count(node -> _recovery_controller_owner(node.prepared.workspace, workspace), graph.nodes) == 1 ||
        throw(ArgumentError("single-controller recovery requires one correction controller per Graph"))
    _validate_single_recovery_feedback(workspace, owner.feedback, graph, controller_index,
        reference_supported)
    ancestors = falses(length(graph.nodes))
    _recovery_ancestors!(ancestors, graph, controller_index)
    reachable = falses(length(graph.nodes))
    reachable[controller_index] = true
    # _prepare_graph stores Nodes in topological execution order and translates
    # authored links to those prepared indices (graph.jl _topological_order).
    for index in eachindex(graph.nodes)
        index == controller_index && continue
        reachable[index] = any(graph.input_routes[index]) do route
            source = _recovery_route_source(route)
            !iszero(source) && reachable[source]
        end
    end
    command_paths = falses(length(graph.nodes))
    for index in eachindex(graph.nodes)
        workspace = graph.nodes[index].prepared.workspace
        if reachable[index] && _recovery_command_owner(workspace)
            _recovery_ancestors!(command_paths, graph, index)
        end
    end
    for index in eachindex(graph.nodes)
        workspace = graph.nodes[index].prepared.workspace
        required = (command_paths[index] && reachable[index] && _recovery_workspace_supported(workspace)) ||
            (ancestors[index] && _recovery_temporal_owner(workspace))
        required && !selected[index] && throw(ArgumentError(
            "recovery must include causal history Node '$(graph.nodes[index].name)'"))
        if reference_supported && selected[index] && _recovery_command_owner(workspace)
            # Qualify every causal input of each selected conditioner, including
            # side inputs. An external bypass or an unknown transform is refused.
            for route in graph.input_routes[index]
                _recovery_reference_route(graph, controller_index, selected, feedback_required, route)
            end
        end
    end
    return Tuple(feedback_required)
end

_recovery_reference_route(graph, controller, selected, feedback_required, ::FilterGraphs.ExternalInput) =
    throw(ArgumentError("recovery command references cannot depend on external bypass inputs"))
Base.@nospecializeinfer function _recovery_reference_route(
    @nospecialize(graph), controller, @nospecialize(selected), feedback_required,
    @nospecialize(route::FilterGraphs.NodeOutput),
)
    node_index, port_index = _recovery_route_indices(route)
    _recovery_root(node_index, controller) && return UInt8(0) # recovered controller outputs are zero
    node = graph.nodes[node_index]
    return _recovery_reference_node(node.prepared.plan, node.output_names[port_index], graph,
        controller, selected, feedback_required, node_index)
end

_recovery_route_indices(::FilterGraphs.NodeOutput{Node,Port}) where {Node,Port} = (Node, Port)

Base.@nospecializeinfer function _recovery_reference_node(
    @nospecialize(plan), port_name, @nospecialize(graph), controller,
    @nospecialize(selected), feedback_required, index,
)
    origin = UInt8(0)
    for route in graph.input_routes[index]
        origin |= _recovery_reference_route(graph, controller, selected, feedback_required, route)
    end
    return _recovery_reference_output(plan, port_name,
        origin, selected[index], feedback_required, index)
end

function _recovery_reference_node(::FilterGraphAlgorithms.PhysicalOffsetReferencePlan,
    port_name, graph, controller, selected, feedback_required, index)
    # This output copies only the adopted physical parameter. Its paired
    # compensated residual still depends on external measurements and cannot
    # establish a static command reference. A successful owner frame witnesses
    # both outputs from the same Plan; parameter adoption invalidates the witness.
    port_name === Symbol("physical-offset") || throw(ArgumentError(
        "only physical-offset can establish a physical offset reference command bias"))
    return UInt8(1)
end

_recovery_reference_output(plan, port_name, origin, selected, feedback_required, index) =
    _recovery_reference_transform(plan, origin, selected)

function _recovery_reference_output(::FilterGraphAlgorithms.PdmCommandPlan, port_name,
    origin, selected, feedback_required, index)
    selected && port_name === Symbol("constraint-feedback") || throw(ArgumentError(
        "only selected conditioner constraint-feedback may enter another recovery command path"))
    feedback_required[index] = true
    # This branch is zero only after its committed current reaches its request.
    # A temporal filter downstream could retain the earlier nonzero feedback.
    return UInt8(2)
end

_recovery_reference_transform(plan, origin, selected) = throw(ArgumentError(
    "recovery reference topology does not support $(typeof(plan))"))
_recovery_reference_transform(::Union{
    FilterGraphAlgorithms.ControllerToVdmPlan,
    FilterGraphAlgorithms.VdmToPdmPlan,
    FilterGraphAlgorithms.VectorLinearCombinationPlan,
    FilterGraphAlgorithms.PdmCommandPowerLimitPlan,
    FilterGraphAlgorithms.DenseReconstructorPlan{T,Matrix{T}} where T,
}, origin, selected) = origin
_recovery_reference_transform(::Union{
    FilterGraphAlgorithms.VectorOffsetPlan,
    FilterGraphAlgorithms.VdmShapeOffsetPlan,
    FilterGraphAlgorithms.PdmSystemFlatPlan,
    FilterGraphAlgorithms.PdmEngineeringOffsetPlan,
}, origin, selected) = origin | UInt8(1)
function _recovery_reference_transform(::FilterGraphAlgorithms.BiquadCascadePlan, origin, selected)
    selected || throw(ArgumentError("recovery must include every command-path temporal filter"))
    iszero(origin) || throw(ArgumentError(
        "recovery cannot qualify a temporal filter downstream of static bias or conditioner feedback"))
    return UInt8(0)
end

_recovery_controller_owner(workspace) = false
_recovery_controller_owner(::FilterGraphAlgorithms.ClosedLoopCorrectionWorkspace) = true
_recovery_controller_owner(workspace, selected) = _recovery_controller_owner(workspace)
_recovery_controller_owner(::FilterGraphAlgorithms.LeakyIntegratorWorkspace,
    ::FilterGraphAlgorithms.LeakyIntegratorWorkspace) = true

_validate_single_recovery_feedback(workspace, feedback, graph, controller, reference_supported) =
    _validate_recovery_feedback(feedback, graph, controller, reference_supported)
function _validate_single_recovery_feedback(::FilterGraphAlgorithms.LeakyIntegratorWorkspace,
    feedback, graph, controller, reference_supported)
    feedback === nothing || throw(ArgumentError("leaky integrator recovery requires no feedback bridge"))
    return nothing
end
function _validate_recovery_feedback(::Nothing, graph, controller, reference_supported)
    reference_supported && throw(ArgumentError(
        "automatic recovery reference proof requires an owned delayed-feedback bridge"))
    return nothing
end
function _validate_recovery_feedback(::PreparedFeedbackBridge{Input}, graph, controller,
    reference_supported) where {Input}
    external_index = findfirst(isequal(Input), keys(graph.input_formats))
    input_index = findfirst(format -> format.schema ==
        FilterGraphAlgorithms.CONTROLLER_CONSTRAINT_FEEDBACK_V1,
        graph.nodes[controller].input_formats)
    input_index !== nothing && _recovery_feedback_route_matches(
        graph.input_routes[controller][input_index], external_index) || throw(ArgumentError(
        "recovery controller must own the prepared delayed-feedback input"))
    return nothing
end
_recovery_feedback_route_matches(route, expected) = false
_recovery_feedback_route_matches(::FilterGraphs.ExternalInput{Index}, expected) where {Index} =
    Index == expected

# Recovery may remove a stale physical correction while preserving protection.
# Admit exactly one offset-only zeroing update; calibration, gain and every
# other parameter remain inhibited. Normal adoption invalidates the reference
# witness, and the conditioner must settle before a fresh witness permits close.
function _recovery_offset_clear(graph, updates::Tuple)
    length(updates) == 1 || return false
    update = only(updates)
    name = FilterGraphs._parameter_name(first(update))
    route = FilterGraphs._parameter_route(graph, name)
    return _recovery_offset_clear(graph, route, last(update))
end
_recovery_offset_clear(graph, route, values) = false
function _recovery_offset_clear(graph, ::FilterGraphs.ParameterInput{Node,:offset}, values) where {Node}
    plan = graph.nodes[Node].prepared.plan
    return plan isa FilterGraphAlgorithms.PhysicalOffsetReferencePlan &&
        values isa AbstractVector{Float32} && size(values) == size(plan.offset) && all(iszero, values)
end

# Resolve only the external offset Port; names or vector extents alone do not
# grant permission to replace another Algorithm's pending parameter data.
function _physical_offset_update_index(graph, updates::Tuple)
    length(updates) == 1 || return nothing
    update = only(updates)
    name = FilterGraphs._parameter_name(first(update))
    return _physical_offset_update_index(graph, FilterGraphs._parameter_route(graph, name))
end
_physical_offset_update_index(graph, route) = nothing
function _physical_offset_update_index(graph, ::FilterGraphs.ParameterInput{Node,:offset}) where {Node}
    return graph.nodes[Node].prepared.plan isa FilterGraphAlgorithms.PhysicalOffsetReferencePlan ? Node : nothing
end

function _physical_offset_transaction(graph, transaction, index::Int)
    count(identity, transaction.touched) == 1 && transaction.touched[index] || return false
    previous = graph.nodes[index].prepared.plan
    candidate = transaction.plans[index]
    return previous isa FilterGraphAlgorithms.PhysicalOffsetReferencePlan &&
        candidate isa FilterGraphAlgorithms.PhysicalOffsetReferencePlan &&
        candidate.combination === previous.combination &&
        candidate.physical_to_modes === previous.physical_to_modes &&
        candidate.modes_to_reference === previous.modes_to_reference &&
        candidate.reference_sign === previous.reference_sign
end

function _recovery_offset_clear_candidate(graph, transaction)
    index = findfirst(identity, transaction.touched)
    index === nothing && return false
    _physical_offset_transaction(graph, transaction, index) || return false
    candidate = transaction.plans[index]
    return all(iszero, candidate.offset) && all(iszero, candidate.correction)
end
