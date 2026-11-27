# Coupled CPU recovery uses the existing owner control slot and graph executor.
_recovery_admission(join) = nothing
_recovery_admission(composition::FilterGraphs.PreparedComposition) =
    composition.failed ? throw(InvalidStateException("failed composition requires reset or replacement", :failed)) : nothing
_reconcile_recovery!(owner, join) = nothing
_recovery_failed!(join) = nothing
_recovery_failed!(composition::FilterGraphs.PreparedComposition) = (composition.failed = true; nothing)
_reset_feedback!(::CompositionBindings) = nothing
_recovery_workspace_supported(::FilterGraphAlgorithms.ChoppedImageDifferenceWorkspace) = true
_recovery_temporal_owner(::FilterGraphAlgorithms.ChoppedImageDifferenceWorkspace) = true
_recovery_root(index, controller::Int) = index == controller
_recovery_root(index, controllers::Tuple) = index in controllers
_recovery_history_free(workspace) = false
_recovery_history_free(::Union{FilterGraphAlgorithms.EmptyWorkspace,FilterGraphAlgorithms.VectorLinearCombinationWorkspace}) = true

_composition_feedback_origin(graph, selected, required, ::FilterGraphs.ExternalInput) =
    throw(ArgumentError("private feedback cannot depend on external bypass inputs"))
function _composition_feedback_origin(graph, selected, required, ::FilterGraphs.NodeOutput{Node,Port}) where {Node,Port}
    node = graph.nodes[Node]
    return _composition_feedback_node(node.prepared.plan, node.output_names[Port], graph, selected, required, Node)
end
function _composition_feedback_node(plan::FilterGraphAlgorithms.PdmCommandPlan, port, graph, selected, required, index)
    return _recovery_reference_output(plan, port, UInt8(0), selected[index], required, index)
end
function _composition_feedback_node(plan, port, graph, selected, required, index)
    origin = foldl(|, map(route -> _composition_feedback_origin(graph, selected, required, route),
        graph.input_routes[index]); init=UInt8(0))
    return _recovery_reference_output(plan, port, origin, selected[index], required, index)
end

function _recovery_open_plans(current, indices)
    plans, requested, active = current.plans, current.requested_property_generations, current.active_property_generations
    for index in indices
        plan = plans[index]
        info = _property_info(plan, "closed")
        opened = FilterGraphAlgorithms.prepare_properties(plan, (FilterGraphAlgorithms.PropertyAssignment(info.id, false),))
        requested[index] < UInt64(typemax(Int64)) || throw(FilterGraphAlgorithms.PropertyGenerationExhausted())
        plans = Base.setindex(plans, opened, index)
        requested = Base.setindex(requested, requested[index] + UInt64(1), index)
        active = Base.setindex(active, requested[index], index)
    end
    return plans, requested, active
end

function _scope_controllers(status, scope)
    scope.plural || return status
    indices = ntuple(i -> i <= length(scope.controller_indices) ? scope.controller_indices[i] : 0, 16)
    guards = status.status in (:pending, :applied, :release_pending, :release_refused, :released) ?
        (UInt16(1) << length(scope.controller_indices)) - UInt16(1) : UInt16(0)
    return GraphRecoveryStatus(status.scope_id, status.controller_index, status.requested_generation,
        status.active_generation, status.status, status.closure_inhibited, status.reference_supported,
        status.native, status.close_guarded, indices, length(scope.controller_indices), guards)
end
_controller_guarded(status, slot) = !iszero(status.close_guard_mask & (UInt16(1) << (slot - 1)))

function _prepare_composition_recovery(owner, controllers, participants, timeout)
    composition = owner.join
    composition isa FilterGraphs.PreparedComposition && !owner.defer_first_input &&
        !FilterGraphs._captured_composition(composition) ||
        throw(ArgumentError("plural recovery requires a CPU complete-frame composition"))
    1 <= timeout <= 60000 || throw(ArgumentError("reference_timeout_ms must lie in 1:60000"))
    2 <= length(controllers) <= 16 || throw(ArgumentError("plural recovery requires 2:16 controllers"))
    length(controllers) == length(composition.branches) || throw(ArgumentError("recover every composition branch together"))
    graph, nodes = owner.graph, owner.graph.nodes
    names = (controllers..., participants...)
    allunique(names) || throw(ArgumentError("recovery Node names must be unique"))
    indices = map(name -> _property_node_index(nodes, name), names)
    roots = indices[1:length(controllers)]
    count(node -> _recovery_controller_owner(node.prepared.workspace), nodes) == length(roots) ||
        throw(ArgumentError("declare every coupled controller"))
    all(node -> !any(node.output_conditional) && FilterGraphs._cpu_dense_executor(node) === nothing, nodes) ||
        throw(ArgumentError("composition recovery requires unconditional complete-frame CPU Nodes"))
    all(outputs -> all(_recovery_cpu_storage, values(outputs)), graph.node_outputs) ||
        throw(ArgumentError("composition recovery requires CPU storage"))
    for index in roots
        _require_recovery_controller(nodes[index].prepared.workspace)
        _recovery_closed_property_supported(_property_info(nodes[index].prepared.plan, "closed")) ||
            throw(ArgumentError("each recovery controller needs runtime Boolean closed"))
    end
    for index in indices
        workspace = nodes[index].prepared.workspace
        _recovery_workspace_supported(workspace) && hasmethod(FilterGraphAlgorithms.recover!, Tuple{typeof(workspace)}) ||
            throw(ArgumentError("Node '$(nodes[index].name)' has no supported recovery contract"))
        workspace isa FilterGraphAlgorithms.CorrectorAllocationWorkspace &&
            throw(ArgumentError("composition recovery requires internal conditioner witnesses"))
    end
    selected = Tuple(index in indices for index in eachindex(nodes))
    feedback_required = falses(length(nodes))
    conditioners = Int[]
    for (branch_index, branch) in pairs(composition.branches)
        branch_roots = filter(index -> index in branch.nodes, roots)
        length(branch_roots) == 1 || throw(ArgumentError("each branch requires exactly one declared controller"))
        controller = only(branch_roots)
        feedback_input = findfirst(format -> format.schema == FilterGraphAlgorithms.CONTROLLER_CONSTRAINT_FEEDBACK_V1,
            nodes[controller].input_formats)
        external = findfirst(==(branch.feedback), keys(graph.input_formats))
        feedback_input !== nothing && _recovery_feedback_route_matches(graph.input_routes[controller][feedback_input], external) ||
            throw(ArgumentError("branch controller must own its private delayed feedback"))
        # Retained correction cuts are zeroed without running a camera/front end.
        # Admit only tails proven zero-preserving for the recovered roots.
        iszero(_recovery_reference_route(graph, roots, selected, feedback_required, branch.route)) ||
            throw(ArgumentError("retained branch correction tails must be zero-preserving"))
        ancestors = falses(length(nodes))
        _recovery_ancestors!(ancestors, graph, controller)
        for index in eachindex(nodes)
            ancestors[index] && _recovery_temporal_owner(nodes[index].prepared.workspace) && !selected[index] &&
                throw(ArgumentError("recovery must include causal history Node '$(nodes[index].name)'"))
            if ancestors[index]
                workspace = nodes[index].prepared.workspace
                (_recovery_history_free(workspace) ||
                    (selected[index] && _recovery_workspace_supported(workspace))) ||
                    throw(ArgumentError("unknown causal history Node '$(nodes[index].name)'"))
            end
        end
        origin = _composition_feedback_origin(graph, selected, feedback_required, branch.returned_route)
        origin == UInt8(2) || throw(ArgumentError("private feedback must derive only from selected conditioner constraint-feedback"))
    end
    for index in composition.shared_nodes
        workspace = nodes[index].prepared.workspace
        if _recovery_command_owner(workspace)
            selected[index] || throw(ArgumentError("recover every shared conditioner"))
            push!(conditioners, index)
            for route in graph.input_routes[index]
                _recovery_reference_route(graph, roots, selected, feedback_required, route)
            end
        end
    end
    isempty(conditioners) && throw(ArgumentError("composition recovery requires a shared conditioner"))
    # Every supported command-path history is explicit, including branch tails.
    causal = falses(length(nodes))
    for index in conditioners; _recovery_ancestors!(causal, graph, index); end
    for index in eachindex(nodes)
        causal[index] && _recovery_temporal_owner(nodes[index].prepared.workspace) && !selected[index] &&
            throw(ArgumentError("recovery must include causal history Node '$(nodes[index].name)'"))
    end
    id = Threads.atomic_add!(_RECOVERY_SCOPE_IDS, UInt64(1)) + UInt64(1)
    iszero(id) && throw(OverflowError("recovery scope identity exhausted"))
    return PreparedGraphRecovery(owner, id, first(controllers), first(roots), selected, Tuple(feedback_required), true,
        UInt64(timeout) * UInt64(1_000_000), controllers, roots, true)
end

@inline _zero_recovery_outputs!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline _zero_recovery_arrays!(::Tuple{}) = nothing
@inline function _zero_recovery_arrays!(arrays::Tuple)
    fill!(first(arrays), zero(eltype(first(arrays))))
    _zero_recovery_arrays!(Base.tail(arrays))
    return nothing
end
@inline function _zero_recovery_outputs!(nodes::Tuple, outputs::Tuple, selected::Tuple)
    workspace = first(nodes).prepared.workspace
    if first(selected) && (_recovery_controller_owner(workspace) || _recovery_temporal_owner(workspace))
        _zero_recovery_arrays!(first(outputs))
    end
    _zero_recovery_outputs!(Base.tail(nodes), Base.tail(outputs), Base.tail(selected))
    return nothing
end
@inline _zero_recovery_branches!(::Tuple{}, outputs) = nothing
@inline function _zero_recovery_branches!(branches::Tuple, outputs)
    branch = first(branches)
    fill!(FilterGraphs._route_value(branch.route, (), outputs), 0)
    fill!(branch.feedback_buffer, 0)
    _zero_recovery_branches!(Base.tail(branches), outputs)
    return nothing
end
@inline _clear_recovery_metadata!(::Tuple{}) = nothing
@inline function _clear_recovery_metadata!(metadata::Tuple)
    for slot in first(metadata); FilterGraphs._store_metadata!(slot, nothing); end
    _clear_recovery_metadata!(Base.tail(metadata))
    return nothing
end
@inline function _reconcile_recovery!(owner, composition::FilterGraphs.PreparedComposition)
    graph = composition.graph
    _zero_recovery_outputs!(graph.nodes, graph.node_outputs, owner.active_recovery_selection)
    _zero_recovery_branches!(composition.branches, graph.node_outputs)
    FilterGraphs._clear_available!(graph.node_output_available)
    _clear_recovery_metadata!(graph.node_output_metadata)
    for slot in values(graph.published_metadata); FilterGraphs._store_metadata!(slot, nothing); end
    for slot in values(composition.metadata); FilterGraphs._store_metadata!(slot, nothing); end
    fill!(composition.contributors, nothing)
    fill!(composition.accepted, false)
    fill!(composition.trigger, false)
    return nothing
end

function _branch_for_controller(composition, index)
    return something(findfirst(branch -> index in branch.nodes, composition.branches))
end
@inline _branch_measurement_ready(graph, index) =
    _branch_measurement_ready(graph.nodes, graph.input_routes, graph.node_outputs, index, Val(1))
@inline _branch_measurement_ready(::Tuple{}, ::Tuple{}, outputs, index, ::Val) = false
@inline function _branch_measurement_ready(nodes::Tuple, routes::Tuple, outputs, index, ::Val{Index}) where {Index}
    index == Index && return _branch_validity_ready(first(nodes).input_formats, first(routes), outputs)
    return _branch_measurement_ready(Base.tail(nodes), Base.tail(routes), outputs, index, Val(Index + 1))
end
@inline _branch_validity_ready(::Tuple{}, ::Tuple{}, outputs) = true
@inline function _branch_validity_ready(formats::Tuple, routes::Tuple, outputs)
    ready = first(formats).element_type !== Bool || all(FilterGraphs._route_value(first(routes), (), outputs))
    return ready && _branch_validity_ready(Base.tail(formats), Base.tail(routes), outputs)
end
@inline function _composition_command_coherent(owner, status)
    return _controller_recovered(owner.graph.nodes, owner.active_recovery_guard_selection) &&
        _reference_witnesses(owner.graph.nodes, owner.graph.input_routes, owner.active_recovery_selection,
            owner.active_recovery_feedback, owner.graph.node_outputs)
end

@inline _invalidate_arrived_branches!(::Tuple{}, owner, presence, status) = nothing
@inline function _invalidate_arrived_branches!(branches::Tuple, owner, presence, status)
    branch = first(branches)
    if getproperty(presence, branch.camera)
        for slot in 1:status.controller_count
            status.controllers[slot] in branch.nodes && (owner.recovery_branch_ready[slot][] = false)
        end
    end
    _invalidate_arrived_branches!(Base.tail(branches), owner, presence, status)
    return nothing
end
function _begin_composition_recovery_frame!(owner, presence)
    status = (@atomic :acquire owner.control_snapshot).recovery
    status.controller_count > 1 && _recovery_reference_observed(status) || return nothing
    owner.recovery_observation_epoch[] += UInt64(1)
    _invalidate_arrived_branches!(owner.join.branches, owner, presence, status)
    owner.recovery_observation_epoch[] += UInt64(1)
    _invalidate_recovery_observation!(owner)
    return nothing
end
_invalidate_recovery_branch_readiness!(owner) = nothing
@inline function _invalidate_recovery_branch_readiness!(owner::PreparedGraphOwner)
    (@atomic :acquire owner.control_snapshot).recovery.controller_count > 1 || return nothing
    owner.recovery_observation_epoch[] += UInt64(1)
    for ready in owner.recovery_branch_ready; ready[] = false; end
    owner.recovery_observation_epoch[] += UInt64(1)
    return nothing
end
function _observe_composition_recovery!(owner)
    status = (@atomic :acquire owner.control_snapshot).recovery
    _recovery_reference_observed(status) || return nothing
    composition = owner.join
    coherent = _composition_command_coherent(owner, status)
    flat = coherent && _controller_recovered(owner.graph.nodes, owner.active_recovery_selection)
    # Route/validity checks precede the seqlock's non-failing publication block.
    for slot in 1:status.controller_count
        index = status.controllers[slot]
        branch = _branch_for_controller(composition, index)
        owner.recovery_branch_candidate_ready[slot] = composition.accepted[branch] &&
            _branch_measurement_ready(owner.graph, index)
    end
    now = time_ns()
    sequence = owner.process_count + UInt64(1)
    owner.recovery_observation_epoch[] += UInt64(1)
    owner.recovery_observation_generation[] = status.active_generation
    owner.recovery_observation_sequence[] = sequence
    owner.recovery_observation_timestamp_ns[] = now
    owner.recovery_reference_reached[] = flat
    owner.recovery_command_coherent[] = coherent
    for slot in 1:status.controller_count
        index = status.controllers[slot]
        branch = _branch_for_controller(composition, index)
        if composition.accepted[branch]
            owner.recovery_branch_sequences[slot][] = sequence
            owner.recovery_branch_timestamps[slot][] = now
            owner.recovery_branch_ready[slot][] = owner.recovery_branch_candidate_ready[slot]
        end
    end
    owner.recovery_observation_epoch[] += UInt64(1)
    return nothing
end

function _branch_observation(owner, slot)
    while true
        before = owner.recovery_observation_epoch[]
        isodd(before) && continue
        generation = owner.recovery_observation_generation[]
        sequence = owner.recovery_branch_sequences[slot][]
        timestamp_ns = owner.recovery_branch_timestamps[slot][]
        ready = owner.recovery_branch_ready[slot][]
        coherent = owner.recovery_command_coherent[]
        command_timestamp = owner.recovery_observation_timestamp_ns[]
        before == owner.recovery_observation_epoch[] && return (; generation, sequence, timestamp_ns, ready, coherent, command_timestamp)
    end
end
function _composition_control_observation(owner)
    while true
        snapshot = @atomic :acquire owner.control_snapshot
        before = owner.recovery_observation_epoch[]
        isodd(before) && continue
        generation = owner.recovery_observation_generation[]
        sequence = owner.recovery_observation_sequence[]
        timestamp_ns = owner.recovery_observation_timestamp_ns[]
        reached = owner.recovery_reference_reached[]
        coherent = owner.recovery_command_coherent[]
        branches = ntuple(16) do slot
            (; generation, sequence=owner.recovery_branch_sequences[slot][],
                timestamp_ns=owner.recovery_branch_timestamps[slot][], ready=owner.recovery_branch_ready[slot][],
                coherent, command_timestamp=timestamp_ns)
        end
        before == owner.recovery_observation_epoch[] && snapshot === (@atomic :acquire owner.control_snapshot) &&
            return snapshot, (; generation, sequence, timestamp_ns, reached, coherent, branches)
    end
end
function _branch_reference_ready(status, observation, timeout, slot, now)
    _recovery_reference_observed(status) && status.requested_generation == status.active_generation &&
        (status.closure_inhibited || _controller_guarded(status, slot)) || return false
    return observation.generation == status.active_generation && observation.ready && observation.coherent &&
        observation.sequence > 0 && now - observation.timestamp_ns <= timeout &&
        now - observation.command_timestamp <= timeout
end
_branch_ready(owner, status, slot, now=time_ns()) = _branch_reference_ready(status,
    _branch_observation(owner, slot), owner.recovery_reference_timeout_ns[], slot, now)
function _current_composition_close_fresh(owner, status, transaction)
    _composition_command_coherent(owner, status) || return false
    for slot in 1:status.controller_count
        index = status.controllers[slot]
        _controller_guarded(status, slot) && transaction.touched[index] && transaction.closed[index] || continue
        _branch_ready(owner, status, slot) || return false
    end
    return true
end
function _validate_composition_close(owner, snapshot, transaction, properties)
    _guarded_recovery_close(snapshot, transaction) || return nothing
    all(properties.values) do pair
        name, local_name = _qualified_property(first(pair))
        local_name == "closed" && any(slot -> owner.graph.nodes[snapshot.recovery.controllers[slot]].name == name,
            1:snapshot.recovery.controller_count)
    end || throw(ArgumentError("guarded first close cannot mix unrelated property writes"))
    # Control preparation can overlap an already running camera callback. Read
    # only its published atomic proof here; adoption owns and rechecks live state.
    current, observation = _composition_control_observation(owner)
    current === snapshot || throw(InvalidStateException("recovery scope changed during close preparation", :changed))
    status, now = snapshot.recovery, time_ns()
    for slot in 1:status.controller_count
        index = status.controllers[slot]
        _controller_guarded(status, slot) && transaction.touched[index] && transaction.closed[index] || continue
        _branch_reference_ready(status, observation.branches[slot], owner.recovery_reference_timeout_ns[], slot, now) ||
            throw(InvalidStateException("selected close requires fresh branch and shared command evidence", :unreached))
    end
    return nothing
end
function _composition_property_recovery_state(status, transaction)
    mask = status.close_guard_mask
    for slot in 1:status.controller_count
        index = status.controllers[slot]
        transaction.touched[index] && transaction.closed[index] && (mask &= ~(UInt16(1) << (slot - 1)))
    end
    return GraphRecoveryStatus(status.scope_id, status.controller_index, status.requested_generation, status.active_generation,
        status.status, status.closure_inhibited, status.reference_supported, status.native, !iszero(mask),
        status.controllers, status.controller_count, mask)
end

function _recovery_descriptor_values(control, values; status=nothing, observation=nothing)
    control.scope.plural || return values
    filter!(pair -> first(pair) != "controller", values)
    values[findfirst(pair -> first(pair) == "version", values)] = "version" => Int32(2)
    push!(values, "controller-count" => Int32(length(control.scope.controllers)))
    for slot in eachindex(control.scope.controllers)
        matching = status !== nothing && status.scope_id == control.scope.id &&
            status.requested_generation == status.active_generation
        branch = matching ? observation.branches[slot] : nothing
        matching = matching && branch.generation == status.active_generation
        append!(values, [
            "controller-$slot" => String(control.scope.controllers[slot]),
            "controller-$slot-close-guarded" => (status !== nothing && status.scope_id == control.scope.id && _controller_guarded(status, slot)),
            "controller-$slot-ready" => (matching && _branch_reference_ready(status, branch,
                control.scope.reference_timeout_ns, slot, time_ns())),
            "controller-$slot-observation-sequence" => Int64(matching ? branch.sequence : 0),
            "controller-$slot-observation-monotonic-nanoseconds" => Int64(matching ? branch.timestamp_ns : 0)])
    end
    return values
end
