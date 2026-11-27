const _RECOVERY_SCOPE_IDS = Threads.Atomic{UInt64}(0)

"""An explicitly named CPU recovery scope bound to one exact Graph owner."""
struct PreparedGraphRecovery{Owner,Selection,Controllers,Indices}
    owner::Owner
    id::UInt64
    controller::Symbol
    controller_index::Int
    selected::Selection
    feedback_required::Selection
    reference_supported::Bool
    reference_timeout_ns::UInt64
    controllers::Controllers
    controller_indices::Indices
    plural::Bool
end

_recovery_workspace_supported(workspace) = false
_recovery_workspace_supported(::Union{
    FilterGraphAlgorithms.ClosedLoopCorrectionWorkspace,
    FilterGraphAlgorithms.BiquadCascadeWorkspace,
    FilterGraphAlgorithms.CorrectorAllocationWorkspace,
    FilterGraphAlgorithms.PdmCommandWorkspace{T,Vector{T}} where T,
}) = true

"""
    prepare_recovery(node_or_owner; controller::Symbol, participants=(), reference_timeout_ms=5000)
    prepare_recovery(node_or_owner; controllers::Tuple, participants=(), reference_timeout_ms=5000)

Prepare a named recovery scope without changing scientific state. The controller
must own a CPU closed-loop correction workspace and a runtime Boolean `closed`
property. The singular form also accepts a zero-initialized CPU leaky integrator
without delayed feedback. Explicit participants may own CPU biquad, PDM conditioner or corrector
allocation workspaces. Include every history participating in the declared loop.
The plural form admits 2:16 complete composition branches and their shared
conditioners. Include every causal biquad and SCC differencer; branch correction
tails must be known zero-preserving maps. Recovery clears SCC F/U history without
resetting its acquisition watermark. Shared static flats/offsets remain intact.
Joined, conditional/progressive and captured execution are rejected before effects.
Configuration, calibration, offsets, flats and acquisition remain outside scope.
Reference observations expire after `reference_timeout_ms` (1–60000 ms).

The scope belongs to this exact owner incarnation. Do not retain it across owner
replacement. Calls that stage controls use the owner's existing single control
slot; the ordinary serialized processing callback owns all scientific mutation.
"""
function prepare_recovery(owner::PreparedGraphOwner;
    controller::Union{Nothing,Symbol}=nothing, controllers::Union{Nothing,Tuple{Vararg{Symbol}}}=nothing,
    participants::Tuple{Vararg{Symbol}}=(),
    reference_timeout_ms::Integer=5000)
    if controllers !== nothing
        controller === nothing || throw(ArgumentError("declare controller or controllers, never both"))
        return _prepare_composition_recovery(owner, controllers, participants, reference_timeout_ms)
    end
    controller === nothing && throw(ArgumentError("recovery requires a controller declaration"))
    1 <= reference_timeout_ms <= 60000 || throw(ArgumentError(
        "reference_timeout_ms must lie in 1:60000"))
    owner.join === nothing && !owner.defer_first_input || throw(
        ArgumentError("recovery requires an ordinary complete-frame Graph owner"))
    nodes = owner.graph.nodes
    all(node -> !any(node.output_conditional) &&
        FilterGraphs._cpu_dense_executor(node) === nothing, nodes) || throw(
        ArgumentError("recovery does not support conditional or progressive Graphs"))
    all(outputs -> all(_recovery_cpu_storage, values(outputs)),
        owner.graph.node_outputs) || throw(
        ArgumentError("recovery requires CPU Graph storage"))
    names = (controller, participants...)
    allunique(names) || throw(ArgumentError("recovery Node names must be unique"))
    indices = map(name -> _property_node_index(nodes, name), names)
    controller_index = first(indices)
    prepared = nodes[controller_index].prepared
    _require_single_recovery_controller(prepared.workspace)
    info = _property_info(prepared.plan, "closed")
    _recovery_closed_property_supported(info) ||
        throw(ArgumentError("recovery controller needs a runtime Boolean closed property"))
    for index in indices
        workspace = nodes[index].prepared.workspace
        (index == controller_index || _recovery_workspace_supported(workspace)) || throw(
            ArgumentError("Node '$(nodes[index].name)' has no supported CPU recovery contract"))
        hasmethod(FilterGraphAlgorithms.recover!, Tuple{typeof(workspace)}) || throw(
            ArgumentError("Node '$(nodes[index].name)' has no scientific recovery method"))
    end
    selected = map(index -> index in indices, eachindex(nodes)) |> Tuple
    kinds = map(indices) do index
        _recovery_reference_kind(nodes[index].prepared.workspace, owner.graph.input_routes[index])
    end
    has_conditioner = any(first, kinds)
    has_allocator = any(last, kinds)
    reference_supported = has_conditioner && !has_allocator
    feedback_required = _validate_recovery_scope(owner, controller_index, selected, reference_supported)
    id = Threads.atomic_add!(_RECOVERY_SCOPE_IDS, UInt64(1)) + UInt64(1)
    iszero(id) && throw(OverflowError("recovery scope identity exhausted"))
    return PreparedGraphRecovery(owner, id, controller, controller_index, selected, feedback_required, reference_supported,
        UInt64(reference_timeout_ms) * UInt64(1_000_000), (controller,), (controller_index,), false)
end

prepare_recovery(node::PipeWireNode; kwargs...) = prepare_recovery(node.owner; kwargs...)
prepare_recovery(owner; kwargs...) = throw(ArgumentError(
    "recovery requires a CPU complete-frame Graph owner"))

"""
    recovery_status(scope)

Read one atomically published owner snapshot. `:pending` becomes `:applied` once
controller opening, selected history recovery and delayed-feedback clearing have
completed. This does not establish command arrival, reference reached or physical
settling. `:unknown` records an unexpected application failure and cannot release
closure. `scope_matches` must hold when interpreting an applied generation.
"""
recovery_status(scope::PreparedGraphRecovery) = _recovery_status(scope, time_ns())
function _recovery_status(scope::PreparedGraphRecovery, now_ns::UInt64)
    if scope.plural
        snapshot, observation = _composition_control_observation(scope.owner)
    else
        snapshot = @atomic :acquire scope.owner.control_snapshot
        observation = _recovery_observation(scope.owner)
    end
    status = snapshot.recovery
    valid = status.status in (:applied, :release_pending, :release_refused, :released) &&
        observation.generation == status.requested_generation && status.scope_id == scope.id
    values = (; requested_generation=status.requested_generation,
        active_generation=status.active_generation, status=status.status,
        closure_inhibited=status.closure_inhibited, close_guarded=status.close_guarded,
        scope_matches=status.scope_id == scope.id,
        controller=scope.controller, reference_supported=status.reference_supported,
        reference_reached=valid && _reference_fresh(observation, scope.reference_timeout_ns, now_ns),
        observation_sequence=valid ? observation.sequence : UInt64(0),
        observation_monotonic_nanoseconds=valid ? observation.timestamp_ns : UInt64(0),
        reference_timeout_ms=scope.reference_timeout_ns ÷ UInt64(1_000_000))
    scope.plural || return values
    return merge(values, (; controllers=scope.controllers,
        controller_requested_generations=map(index -> snapshot.requested_property_generations[index], scope.controller_indices),
        controller_active_generations=map(index -> snapshot.active_property_generations[index], scope.controller_indices),
        controller_close_guarded=ntuple(slot -> _controller_guarded(status, slot), length(scope.controllers)),
        controller_ready=ntuple(slot -> valid && _branch_reference_ready(status, observation.branches[slot],
            scope.reference_timeout_ns, slot, now_ns), length(scope.controllers))))
end

function _recovery_snapshot(snapshot, plans, requested_properties, active_properties, recovery)
    return GraphControlSnapshot(plans, requested_properties, active_properties,
        snapshot.requested_parameter_sequences, snapshot.active_parameter_sequences,
        recovery)
end

function _reserve_recovery!(owner; throw_if_busy::Bool=true)
    previous = Threads.atomic_cas!(owner.control_state, _CONTROL_SLOT_AVAILABLE,
        _CONTROL_SLOT_RECOVERY_PREPARING)
    reserved = previous == _CONTROL_SLOT_AVAILABLE
    !reserved && throw_if_busy && throw(FilterGraphAlgorithms.PropertyUpdateBusy())
    return reserved
end

"""
    stage_recovery!(scope) -> UInt64

Stage one recovery and return its requested generation. At the next callback
adoption boundary, open the controller, recover precisely the named scientific
workspaces and clear delayed feedback. Conditioner current and allocator
held commands/cadence are retained by their scientific
contracts. No whole-Graph reset or direct actuator command is issued.

Closure remains inhibited until an explicit matching release. Pending requests
share the property/parameter control slot and are never automatically replayed.
"""
stage_recovery!(scope::PreparedGraphRecovery) = _stage_recovery!(scope, nothing, nothing)

function _stage_recovery!(scope::PreparedGraphRecovery, native_token, observer)
    owner = scope.owner
    _reserve_recovery!(owner)
    try
        current = @atomic :acquire owner.control_snapshot
        current.recovery.closure_inhibited && throw(InvalidStateException(
            "an earlier recovery requires explicit reconciliation and release", :inhibited))
        generation = current.recovery.requested_generation
        generation < UInt64(typemax(Int64)) || throw(OverflowError("recovery generation exhausted"))
        generation += UInt64(1)
        _recovery_admission(owner.join)
        index = scope.controller_index
        plans, properties, active_properties = _recovery_open_plans(current, scope.controller_indices)
        native_pending, native_applied = _prepare_native_recovery_status(current.recovery.native,
            native_token, nothing)
        pending = GraphRecoveryStatus(scope.id, index, generation,
            current.recovery.active_generation, :pending, true, scope.reference_supported, native_pending)
        applied = GraphRecoveryStatus(scope.id, index, generation, generation, :applied, true, scope.reference_supported, native_applied)
        unknown = GraphRecoveryStatus(scope.id, index, generation,
            current.recovery.active_generation, :unknown, true, scope.reference_supported, native_pending)
        pending = _scope_controllers(pending, scope)
        applied = _scope_controllers(applied, scope)
        unknown = _scope_controllers(unknown, scope)
        owner.recovery_transaction = PreparedPropertyTransaction(plans, scope.selected)
        owner.pending_recovery_feedback = scope.feedback_required
        owner.pending_recovery_reference_timeout_ns = scope.reference_timeout_ns
        owner.pending_control_snapshot = _recovery_snapshot(current, plans,
            properties, active_properties, applied)
        owner.recovery_failure_snapshot = _recovery_snapshot(current, current.plans,
            properties, current.active_property_generations, unknown)
        requested = _recovery_snapshot(current, current.plans,
            properties, current.active_property_generations, pending)
        observed_properties, requested_properties = _recovery_observer_properties(owner,
            current, plans, properties, observer)
        @atomic :release owner.control_snapshot = requested
        owner.control_state[] = _CONTROL_SLOT_RECOVERY_PENDING
        _notify_property_observer!(observer, Val(:observed), observed_properties)
        _notify_property_observer!(observer, Val(:requested), requested_properties)
        return generation
    catch
        owner.control_state[] = _CONTROL_SLOT_AVAILABLE
        rethrow()
    end
end

"""
    stage_recovery_release!(scope, generation)

Stage release of closure inhibition for this scope's applied generation. The
owner must have published a successful reference observation for all selected
conditioners. This establishes software command-reference coherence; the caller
remains responsible for any required device acknowledgment or physical settling.
Release rechecks freshness at the callback boundary and leaves `closed=false`;
a subsequent ordinary `closed=true` transaction explicitly resumes the loop.
Stale, foreign, pending, unknown and already released generations are rejected.
"""
stage_recovery_release!(scope::PreparedGraphRecovery, generation::UInt64) =
    _stage_recovery_release!(scope, generation, false)

function _stage_recovery_release!(scope::PreparedGraphRecovery, generation::UInt64, native::Bool;
    throw_on_rejection::Bool=true)
    owner = scope.owner
    _reserve_recovery!(owner; throw_if_busy=throw_on_rejection) || return nothing
    handed_off = false
    try
        current = @atomic :acquire owner.control_snapshot
        status = current.recovery
        if !(status.scope_id == scope.id && status.status in (:applied, :release_refused) &&
                status.active_generation == generation && status.closure_inhibited)
            throw_on_rejection || return nothing
            throw(InvalidStateException("release needs this scope's applied recovery generation", :stale))
        end
        observation = _recovery_observation(owner)
        fresh = status.reference_supported && observation.generation == generation &&
            _reference_fresh(observation, scope.reference_timeout_ns, time_ns())
        !fresh && throw_on_rejection &&
            throw(InvalidStateException("release requires a successful owner reference observation", :unreached))
        native_pending, native_applied = _prepare_native_recovery_status(current.recovery.native,
            nothing, native ? Int64(generation) : nothing)
        refused = GraphRecoveryStatus(scope.id, scope.controller_index, generation,
            generation, :release_refused, true, scope.reference_supported, native_applied)
        refused = _scope_controllers(refused, scope)
        refused_snapshot = _recovery_snapshot(current, current.plans, current.requested_property_generations,
            current.active_property_generations, refused)
        if !fresh
            # Reference evidence can expire between native admission and this callback.
            @atomic :release owner.control_snapshot = refused_snapshot
            return generation
        end
        pending = GraphRecoveryStatus(scope.id, scope.controller_index, generation,
            generation, :release_pending, true, scope.reference_supported, native_pending)
        released = GraphRecoveryStatus(scope.id, scope.controller_index, generation,
            generation, :released, false, scope.reference_supported, native_applied)
        pending = _scope_controllers(pending, scope)
        released = _scope_controllers(released, scope)
        owner.pending_control_snapshot = _recovery_snapshot(current, current.plans,
            current.requested_property_generations, current.active_property_generations, released)
        owner.recovery_failure_snapshot = refused_snapshot
        requested = _recovery_snapshot(current, current.plans,
            current.requested_property_generations, current.active_property_generations, pending)
        @atomic :release owner.control_snapshot = requested
        owner.control_state[] = _CONTROL_SLOT_RECOVERY_RELEASE_PENDING
        handed_off = true
    finally
        handed_off || (owner.control_state[] = _CONTROL_SLOT_AVAILABLE)
    end
    return generation
end

function _validate_recovery_closure(owner, transaction, properties)
    snapshot = @atomic :acquire owner.control_snapshot
    status = snapshot.recovery
    status.closure_inhibited && throw(InvalidStateException(
        "property mutation is inhibited during recovery", :inhibited))
    status.controller_count > 1 && return _validate_composition_close(owner, snapshot, transaction, properties)
    if _guarded_recovery_close(snapshot, transaction)
        length(properties.values) == 1 && only(properties.values).first ==
            "$(owner.graph.nodes[status.controller_index].name):closed" || throw(ArgumentError(
                "guarded first close cannot be mixed with other property writes"))
        _recovery_observation_fresh(owner, status) || throw(InvalidStateException(
            "first close requires fresh current recovery reference evidence", :unreached))
    end
    return nothing
end

@inline _install_recovery_plans!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function _install_recovery_plans!(nodes::Tuple, plans::Tuple, selected::Tuple)
    first(selected) && (first(nodes).prepared.plan = first(plans))
    _install_recovery_plans!(Base.tail(nodes), Base.tail(plans), Base.tail(selected))
    return nothing
end

@inline _recover_selected!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function _recover_selected!(nodes::Tuple, plans::Tuple, selected::Tuple)
    if first(selected)
        node = first(nodes)
        # Methods are admitted during scope preparation and are non-failing
        # scientific lifecycle operations on CPU storage.
        FilterGraphAlgorithms.recover!(node.prepared.workspace)
    end
    _recover_selected!(Base.tail(nodes), Base.tail(plans), Base.tail(selected))
    return nothing
end

@inline function _adopt_pending_recovery!(owner::PreparedGraphOwner)
    state = owner.control_state[]
    if state == _CONTROL_SLOT_RECOVERY_RELEASE_PENDING
        snapshot = @atomic :acquire owner.control_snapshot
        fresh = _current_recovery_reference_fresh(owner, snapshot.recovery)
        adopted = fresh ? owner.pending_control_snapshot : owner.recovery_failure_snapshot
        @atomic :release owner.control_snapshot = adopted
        owner.control_state[] = _CONTROL_SLOT_AVAILABLE
        return true
    end
    state == _CONTROL_SLOT_RECOVERY_PENDING || return false
    # Publish an uncertain outcome before any effects. An unexpected exception
    # consumes the request, preserves inhibition and never repeats the operation.
    @atomic :release owner.control_snapshot = owner.recovery_failure_snapshot
    try
        transaction = owner.recovery_transaction
        owner.active_recovery_selection = transaction.touched
        owner.active_recovery_guard_selection = transaction.touched
        owner.active_recovery_feedback = owner.pending_recovery_feedback
        owner.recovery_reference_timeout_ns[] = owner.pending_recovery_reference_timeout_ns
        _install_recovery_plans!(owner.graph.nodes, transaction.plans, transaction.touched)
        FilterGraphs._discard_inflight!(owner.graph)
        _recover_selected!(owner.graph.nodes, transaction.plans, transaction.touched)
        _reset_feedback!(owner.feedback)
        _reconcile_recovery!(owner, owner.join)
        _invalidate_recovery_observation!(owner)
        owner.recovery_observation_epoch[] += UInt64(1)
        for value in owner.recovery_branch_sequences; value[] = 0; end
        for value in owner.recovery_branch_timestamps; value[] = 0; end
        for value in owner.recovery_branch_ready; value[] = false; end
        owner.recovery_observation_epoch[] += UInt64(1)
        @atomic :release owner.control_snapshot = owner.pending_control_snapshot
    catch
        _recovery_failed!(owner.join)
        rethrow()
    finally
        owner.control_state[] = _CONTROL_SLOT_AVAILABLE
    end
    return true
end

@inline _adopt_pending_recovery!(::PreparedCapturedGraphOwner) = false
@inline _adopt_pending_recovery!(::PreparedCapturedGraphSlot) = false

# A serial callback writes the observation under a seqlock. Readers obtain
# one complete observation without allocating a snapshot on every frame.
function _recovery_observation(owner::PreparedGraphOwner)
    while true
        before = owner.recovery_observation_epoch[]
        isodd(before) && continue
        generation = owner.recovery_observation_generation[]
        sequence = owner.recovery_observation_sequence[]
        reached = owner.recovery_reference_reached[]
        coherent = owner.recovery_command_coherent[]
        timestamp_ns = owner.recovery_observation_timestamp_ns[]
        before == owner.recovery_observation_epoch[] && return (; generation, sequence, reached, coherent, timestamp_ns)
    end
end

@inline function _publish_recovery_observation!(owner, generation, sequence, reached)
    owner.recovery_observation_epoch[] += UInt64(1)
    owner.recovery_observation_generation[] = generation
    owner.recovery_observation_sequence[] = sequence
    owner.recovery_reference_reached[] = reached
    owner.recovery_command_coherent[] = reached
    owner.recovery_observation_timestamp_ns[] = time_ns()
    owner.recovery_observation_epoch[] += UInt64(1)
    return nothing
end

@inline function _invalidate_recovery_observation!(owner::PreparedGraphOwner)
    owner.recovery_reference_reached[] = false
    owner.recovery_command_coherent[] = false
    return nothing
end
@inline _invalidate_recovery_observation!(owner) = nothing

@inline _reference_witness(workspace, routes, outputs) = true
@inline function _reference_witness(workspace::FilterGraphAlgorithms.PdmCommandWorkspace,
    routes, outputs)
    requested = FilterGraphs._route_value(only(routes), (), outputs)
    return workspace.committed_current == requested
end
@inline _reference_witness(workspace::FilterGraphAlgorithms.CorrectorAllocationWorkspace,
    routes, outputs) = false

@inline _reference_witnesses(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, outputs) = true
@inline function _reference_witnesses(nodes::Tuple, routes::Tuple, selected::Tuple,
    feedback_required::Tuple, outputs)
    workspace = first(nodes).prepared.workspace
    reached = !first(selected) || _reference_witness(workspace, first(routes), outputs)
    feedback_zero = !first(feedback_required) || _recovery_feedback_zero(workspace)
    return reached && feedback_zero && _reference_witnesses(Base.tail(nodes), Base.tail(routes),
        Base.tail(selected), Base.tail(feedback_required), outputs)
end

_recovery_feedback_zero(workspace::FilterGraphAlgorithms.PdmCommandWorkspace) =
    all(iszero, workspace.candidate_constraint_feedback)
_recovery_feedback_zero(workspace) = false

@inline _controller_recovered(::Tuple{}, ::Tuple{}) = true
@inline function _controller_recovered(nodes::Tuple, selected::Tuple)
    prepared = first(nodes).prepared
    coherent = !first(selected) || _controller_recovered(prepared.workspace, prepared.plan)
    return coherent && _controller_recovered(Base.tail(nodes), Base.tail(selected))
end
@inline _controller_recovered(workspace, plan) = true
@inline _controller_recovered(workspace::FilterGraphAlgorithms.ClosedLoopCorrectionWorkspace,
    plan) = !plan.closed && all(iszero, workspace.committed_state) &&
        all(iszero, workspace.committed_output)
@inline _controller_recovered(workspace::FilterGraphAlgorithms.LeakyIntegratorWorkspace,
    plan) = !plan.closed && all(iszero, workspace.committed_state)

@inline _recovery_reference_observed(status) = status.reference_supported &&
    (status.status in (:applied, :release_pending, :release_refused) ||
        (status.status === :released && status.close_guarded))

@inline function _observe_recovery_reference!(owner::PreparedGraphOwner)
    snapshot = @atomic :acquire owner.control_snapshot
    status = snapshot.recovery
    _recovery_reference_observed(status) || return nothing
    selected = owner.active_recovery_selection
    reached = _controller_recovered(owner.graph.nodes, selected) &&
        _reference_witnesses(owner.graph.nodes, owner.graph.input_routes,
            selected, owner.active_recovery_feedback, owner.graph.node_outputs)
    _publish_recovery_observation!(owner, status.active_generation,
        owner.process_count + UInt64(1), reached)
    return nothing
end

function _recovery_observer_properties(owner, current, plans, properties, ::Nothing)
    return nothing, nothing
end
function _recovery_observer_properties(owner, current, plans, properties, observer)
    requested = _recovery_snapshot(current, plans, properties,
        current.active_property_generations, current.recovery)
    return _graph_properties(owner, current), _graph_properties(owner, requested)
end

function _prepare_native_recovery_status(current::NativeRecoveryStatus, request, release)
    request === nothing && release === nothing && return current, current
    if request !== nothing
        request > current.request_token && request > 0 || throw(ArgumentError(
            "recovery request-token must be positive and strictly increasing"))
    end
    generation = current.requested_generation
    generation < UInt64(typemax(Int64)) || throw(OverflowError("native recovery generation exhausted"))
    generation += UInt64(1)
    request_token = request === nothing ? current.request_token : request
    release_token = release === nothing ? current.release_token : release
    return NativeRecoveryStatus(generation, current.active_generation, request_token, release_token),
        NativeRecoveryStatus(generation, generation, request_token, release_token)
end

const _RECOVERY_NODE = "pipewireao-recovery"
const _RECOVERY_PREFIX = _RECOVERY_NODE * ":"

struct NativeRecoveryControl{Scope}
    scope::Scope
end

_native_recovery(owner, ::Nothing) = nothing
function _native_recovery(owner::PreparedGraphOwner, configuration::NamedTuple)
    xor(haskey(configuration, :controller), haskey(configuration, :controllers)) && haskey(configuration, :participants) &&
        all(name -> name in (:controller, :controllers, :participants, :reference_timeout_ms), keys(configuration)) ||
        throw(ArgumentError("recovery must declare controller, participants and optional reference_timeout_ms"))
    any(node -> String(node.name) == _RECOVERY_NODE, owner.graph.nodes) && throw(
        ArgumentError("Graph Node name pipewireao-recovery is reserved by recovery controls"))
    scope = prepare_recovery(owner; configuration...)
    scope.reference_supported || throw(ArgumentError(
        "native recovery requires internal CPU conditioner reference witnesses"))
    return NativeRecoveryControl(scope)
end
_native_recovery(owner, configuration) = throw(ArgumentError(
    "recovery requires a named CPU complete-frame scope"))

_native_recovery_property_info(::Nothing) = PipeWireAO.SPA.PropInfo[]
function _native_recovery_property_info(control::NativeRecoveryControl)
    defaults = (
        "version" => Int32(1), "request-token" => Int64(0), "release-token" => Int64(0),
        "requested-generation" => Int64(0), "active-generation" => Int64(0),
        "recovery-generation" => Int64(0), "state" => "idle", "closure-inhibited" => false,
        "reference-reached" => false, "close-guarded" => false, "observation-sequence" => Int64(0),
        "controller" => String(control.scope.controller),
        "observation-monotonic-nanoseconds" => Int64(0),
        "reference-timeout-ms" => Int32(control.scope.reference_timeout_ns ÷ UInt64(1_000_000)))
    return map(_recovery_descriptor_values(control, collect(defaults))) do (name, value)
        PipeWireAO.SPA.PropInfo(_RECOVERY_PREFIX * name, PipeWireAO.Pod(value);
            description="Scoped CPU recovery " * name, display_name=name,
            group=_RECOVERY_NODE * "/Recovery", params=true,
            flags=name in ("request-token", "release-token") ? 0 : PipeWireAO.SPA.PROPERTY_READONLY)
    end
end

_graph_properties_for_control(owner, ::Nothing) = _graph_properties(owner)
function _graph_properties_for_control(owner, control::NativeRecoveryControl)
    if control.scope.plural
        snapshot, observation = _composition_control_observation(owner)
    else
        snapshot = @atomic :acquire owner.control_snapshot
        observation = _recovery_observation(owner)
    end
    properties = _graph_properties(owner, snapshot)
    status = snapshot.recovery
    native = status.native
    valid = status.status in (:applied, :release_pending, :release_refused, :released) &&
        observation.generation == status.requested_generation && status.scope_id == control.scope.id
    values = (
        "version" => Int32(1), "request-token" => native.request_token,
        "release-token" => native.release_token,
        "requested-generation" => Int64(native.requested_generation),
        "active-generation" => Int64(native.active_generation),
        "recovery-generation" => Int64(status.active_generation),
        "state" => String(status.status), "closure-inhibited" => status.closure_inhibited,
        "close-guarded" => status.close_guarded,
        "reference-reached" => valid && _reference_fresh(observation, control.scope.reference_timeout_ns, time_ns()),
        "observation-sequence" => Int64(valid ? observation.sequence : UInt64(0)),
        "controller" => String(control.scope.controller),
        "observation-monotonic-nanoseconds" => Int64(valid ? observation.timestamp_ns : UInt64(0)),
        "reference-timeout-ms" => Int32(control.scope.reference_timeout_ns ÷ UInt64(1_000_000)))
    append!(properties.values, PipeWireAO.SPA.Props(
        [(_RECOVERY_PREFIX * name) => value for (name, value) in
            _recovery_descriptor_values(control, collect(values); status, observation)]).values)
    return properties
end

_stage_native_recovery!(::Nothing, properties, observer; throw_on_release_rejection::Bool) = false
function _stage_native_recovery!(control::NativeRecoveryControl, properties, observer;
    throw_on_release_rejection::Bool)
    any(pair -> startswith(pair.first, _RECOVERY_PREFIX), properties.values) || return false
    length(properties.values) == 1 || throw(ArgumentError(
        "recovery controls cannot be mixed with other property writes"))
    name, pod = only(properties.values)
    name in (_RECOVERY_PREFIX * "request-token", _RECOVERY_PREFIX * "release-token") ||
        throw(ArgumentError("recovery property is read-only or unknown"))
    PipeWireAO.pod_type(pod) == PipeWireAO.LibPipeWire.SPA_TYPE_Long || throw(
        ArgumentError("recovery tokens require SPA Long values"))
    token = PipeWireAO.pod_value(Int64, pod)
    token > 0 || throw(ArgumentError("recovery tokens must be positive"))
    if name == _RECOVERY_PREFIX * "request-token"
        _stage_recovery!(control.scope, token, observer)
    else
        _stage_recovery_release!(control.scope, UInt64(token), true;
            throw_on_rejection=throw_on_release_rejection)
    end
    return true
end

"""
    stage_properties!(node, properties::SPA.Props)

Stage the Node's declared scalar controls through the same admission, observer
and serialized handoff used by its native property callback. This does not wait
for processing adoption. Read the Node's ordinary requested/active generations
or scoped recovery status to establish completion.
This synchronous API throws on release admission rejection. The PipeWire callback reports
loss of the current reference for a matching release through the scoped recovery status.
"""
function stage_properties!(node::PipeWireNode, properties::PipeWireAO.SPA.Props)
    callback = node.filter.callbacks.on_properties
    callback === nothing && throw(ArgumentError("Node has no writable property controls"))
    if callback isa GraphSetPropertiesCallback
        callback(node.filter, properties; throw_on_release_rejection=true)
    else
        callback(node.filter, properties)
    end
    return nothing
end

@inline _recovery_reference_reached(owner::PreparedGraphOwner) = owner.recovery_reference_reached[]
@inline _recovery_reference_reached(owner) = false

_recovery_cpu_storage(array) = false
_recovery_cpu_storage(::Array) = true
_require_recovery_controller(workspace) = throw(ArgumentError(
    "recovery controller must own a CPU closed-loop correction workspace"))
_require_recovery_controller(::FilterGraphAlgorithms.ClosedLoopCorrectionWorkspace) = nothing
_require_single_recovery_controller(workspace) = _require_recovery_controller(workspace)
function _require_single_recovery_controller(
    workspace::FilterGraphAlgorithms.LeakyIntegratorWorkspace{T,Vector{T}},
) where {T}
    all(iszero, workspace.initial_state) || throw(ArgumentError(
        "leaky integrator recovery requires a zero initial state"))
    return nothing
end
_recovery_closed_property_supported(info) = false
_recovery_closed_property_supported(::FilterGraphAlgorithms.PropertyInfo{
    Bool,FilterGraphAlgorithms.RuntimeProperty}) = true

_recovery_reference_kind(workspace, routes) = (false, false)
_recovery_reference_kind(::FilterGraphAlgorithms.CorrectorAllocationWorkspace, routes) = (false, true)
function _recovery_reference_kind(::FilterGraphAlgorithms.PdmCommandWorkspace, routes)
    _require_internal_reference(only(routes))
    return true, false
end
_require_internal_reference(::FilterGraphs.NodeOutput) = nothing
_require_internal_reference(route) = throw(ArgumentError(
    "recovery conditioner requests must come from an internal Graph Node"))

@inline _reference_fresh(observation, timeout_ns::UInt64, now_ns::UInt64) =
    observation.reached && now_ns - observation.timestamp_ns <= timeout_ns

@inline function _recovery_observation_fresh(owner, status)
    observation = _recovery_observation(owner)
    return status.reference_supported && observation.generation == status.active_generation &&
        _reference_fresh(observation, owner.recovery_reference_timeout_ns[], time_ns())
end

# Only a serialized processing, control or reset boundary may inspect scientific storage.
@inline function _current_recovery_reference_fresh(owner, status)
    selected = owner.active_recovery_selection
    return _recovery_observation_fresh(owner, status) &&
        _controller_recovered(owner.graph.nodes, selected) &&
        _reference_witnesses(owner.graph.nodes, owner.graph.input_routes, selected,
            owner.active_recovery_feedback, owner.graph.node_outputs)
end
@inline _current_recovery_reference_fresh(owner, status, transaction) =
    status.controller_count > 1 ? _current_composition_close_fresh(owner, status, transaction) :
    _current_recovery_reference_fresh(owner, status)

@inline function _guarded_recovery_close(snapshot, transaction)
    status = snapshot.recovery
    status.close_guarded || return false
    if status.controller_count > 1
        for slot in 1:status.controller_count
            index = status.controllers[slot]
            _controller_guarded(status, slot) && transaction.touched[index] && transaction.closed[index] && return true
        end
        return false
    end
    index = status.controller_index
    return transaction.touched[index] && transaction.closed[index]
end

function _recovery_close_guard(status, guarded)
    return GraphRecoveryStatus(status.scope_id, status.controller_index,
        status.requested_generation, status.active_generation, status.status,
        status.closure_inhibited, status.reference_supported, status.native, guarded)
end

function _property_recovery_state(snapshot, transaction)
    snapshot.recovery.controller_count > 1 && return _composition_property_recovery_state(snapshot.recovery, transaction)
    return _guarded_recovery_close(snapshot, transaction) ?
        _recovery_close_guard(snapshot.recovery, false) : snapshot.recovery
end

function _prepare_guarded_close_failure!(owner, snapshot, transaction, requested_properties)
    owner.pending_recovery_guard_selection = owner.active_recovery_guard_selection
    if snapshot.recovery.controller_count > 1
        for slot in 1:snapshot.recovery.controller_count
            index = snapshot.recovery.controllers[slot]
            if transaction.touched[index] && transaction.closed[index]
                owner.pending_recovery_guard_selection = Base.setindex(owner.pending_recovery_guard_selection, false, index)
            end
        end
    end
    _guarded_recovery_close(snapshot, transaction) || return nothing
    status = snapshot.recovery
    if status.controller_count > 1
        owner.recovery_failure_snapshot = _recovery_snapshot(snapshot, snapshot.plans,
            requested_properties, snapshot.active_property_generations, status)
        return nothing
    end
    refused = GraphRecoveryStatus(status.scope_id, status.controller_index,
        status.requested_generation, status.active_generation, :release_refused, true,
        status.reference_supported, status.native, true)
    owner.recovery_failure_snapshot = _recovery_snapshot(snapshot, snapshot.plans,
        requested_properties, snapshot.active_property_generations, refused)
    return nothing
end

# Props are cached by native consumers. Publish advancing evidence periodically,
# even when its Boolean result is unchanged; never restamp the owner observation.
@inline _recovery_properties_interval(owner::PreparedGraphOwner) =
    _recovery_reference_observed((@atomic :acquire owner.control_snapshot).recovery) ?
        min(owner.recovery_reference_timeout_ns[] ÷ UInt64(4), UInt64(100_000_000)) : UInt64(0)

@inline function _notify_graph_properties!(owner::PreparedGraphOwner, filter, changed)
    (changed || !iszero(_recovery_properties_interval(owner)) ||
        _frame_observer_pending(owner.frame_observer)) || return nothing
    return _notify_graph_properties!(owner, filter, changed, time_ns())
end

@inline function _notify_graph_properties!(owner::PreparedGraphOwner, filter, changed, now_ns::UInt64)
    interval = _recovery_properties_interval(owner)
    sequence = owner.recovery_observation_sequence[]
    elapsed = now_ns - owner.recovery_properties_notified_timestamp_ns
    refresh = (!iszero(interval) && sequence != owner.recovery_properties_notified_sequence &&
        elapsed >= interval) ||
        (_frame_observer_pending(owner.frame_observer) && elapsed >= UInt64(100_000_000))
    if (changed || refresh) &&
        PipeWireAO.filter_state(filter) != PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        PipeWireAO.notify_properties!(filter)
        owner.recovery_properties_notified_sequence = sequence
        owner.recovery_properties_notified_timestamp_ns = now_ns
        _frame_observer_notified!(owner.frame_observer)
    end
    return nothing
end

@inline function _notify_graph_properties!(owner, filter, changed)
    if changed && PipeWireAO.filter_state(filter) != PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        PipeWireAO.notify_properties!(filter)
    end
    return nothing
end
