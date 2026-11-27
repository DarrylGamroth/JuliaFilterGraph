"""Lifecycle state of a capacity-one prepared captured Graph owner."""
@enum CapturedGraphState::UInt8 begin
    CapturedGraphReady = 1
    CapturedGraphRunning = 2
    CapturedGraphFaulted = 3
end

"""A captured Graph is prepared and can admit a new execution."""
CapturedGraphReady

"""A captured Graph has an execution in flight and cannot admit another."""
CapturedGraphRunning

"""A captured Graph retained an unexpected failure and rejects further execution."""
CapturedGraphFaulted

"""An unexpected post-admission failure retained by a captured Graph owner."""
struct CapturedGraphFault{C} <: Exception
    phase::Symbol
    cause::C
end

function Base.showerror(io::IO, fault::CapturedGraphFault)
    print(io, "captured Graph fault during ", fault.phase, ": ")
    showerror(io, fault.cause)
end

"""Raised when a capacity-one captured Graph is entered while already running."""
struct CapturedGraphBusy <: Exception end

Base.showerror(io::IO, ::CapturedGraphBusy) =
    print(io, "captured Graph already has one frame in flight")

const _CAPTURED_WORKGROUP_LANE_CANDIDATES = (32, 64, 128, 256)

"""
Cold, qualified workgroup choices used while preparing a captured Graph.

These values select static kernel specializations before native graph
recording. They are control-plane deployment data, not Algorithm properties,
and cannot change while the prepared Graph is armed.
"""
struct CapturedWorkgroupProfile
    projection_lanes::Int
    pyramid_reconstruction_lanes::Int
    shack_hartmann_reconstruction_lanes::Int

    function CapturedWorkgroupProfile(
        projection_lanes::Integer,
        pyramid_reconstruction_lanes::Integer,
        shack_hartmann_reconstruction_lanes::Integer,
    )
        lanes = (
            Int(projection_lanes),
            Int(pyramid_reconstruction_lanes),
            Int(shack_hartmann_reconstruction_lanes),
        )
        all(in(_CAPTURED_WORKGROUP_LANE_CANDIDATES), lanes) || throw(
            ArgumentError(
                "captured workgroup lanes must be one of " *
                "$(collect(_CAPTURED_WORKGROUP_LANE_CANDIDATES))",
            ),
        )
        return new(lanes...)
    end
end

CapturedWorkgroupProfile(;
    projection_lanes::Integer=128,
    pyramid_reconstruction_lanes::Integer=256,
    shack_hartmann_reconstruction_lanes::Integer=128,
) = CapturedWorkgroupProfile(
    projection_lanes,
    pyramid_reconstruction_lanes,
    shack_hartmann_reconstruction_lanes,
)

# Backends can combine launches while retaining the graph's logical nodes.
_prepare_captured_fusion(device, nodes, routes) = nodes

struct CapturedNode{D,A}
    adapter::A
end

"""
One fixed-storage, capacity-one native captured Graph execution owner.

Construct this value with [`prepare_captured_graph`](@ref). Its device, stream,
buffers, Algorithm adapters, native graph, and executable remain retained for
the owner's lifetime.
"""
mutable struct PreparedCapturedGraph{C,E,N,I,O,H,IF,OF,A,PM,DM,MS,RT,W,PF,NN,RS,MF,CM}
    context::C
    executable::E
    nodes::N
    device_inputs::I
    device_outputs::O
    host_outputs::H
    input_formats::IF
    output_formats::OF
    parameter_formats::PF
    availability::A
    published_metadata::PM
    default_input_metadata::DM
    output_metadata_sources::MS
    result_types::RT
    workgroups::W
    state::CapturedGraphState
    fault::Union{Nothing,CapturedGraphFault}
    node_names::NN
    rejection::RS
    required_sample_metadata::MF
    camera_metadata::CM
end

rejection_context(graph::PreparedCapturedGraph) =
    _rejection_context(graph.rejection, graph.node_names)

function _prepare_captured_context end
function _activate_captured_context end
function _restore_captured_context! end
function _allocate_captured_buffer end
"""Allocate retained host staging for an asynchronously copied captured buffer."""
function _allocate_captured_host_buffer(context, format)
    return zeros(format.element_type, format.shape)
end
function _prepare_captured_frame_status end
function _copy_to_captured! end
"""Enqueue a captured device-to-host copy on the active retained stream."""
function _copy_from_captured! end
function _record_captured_graph end
# Cold preparation only; native backends may retain library scratch storage.
_with_captured_workspace(f, context) = f()
# Cold preparation for backends that need retained input copy buffers.
_prepare_captured_inputs(context, executable, device_inputs) = executable
function _launch_captured_graph! end
function _synchronize_captured_graph! end

# Failed asynchronous work can outlive a throwing API call. Only an explicitly
# completed stream establishes that its retained buffers may be collected.
# Keys are lifetime roots, including preparation storage when no owner could
# be returned; values retain the secondary drain failure and its traceback.
struct CapturedStorageQuarantine{Roots}
    roots::Roots
    failure::CapturedException
end

const _captured_storage_quarantine = IdDict{Any,CapturedStorageQuarantine}()
const _captured_storage_quarantine_lock = ReentrantLock()

function _release_captured_storage!(owner)
    lock(_captured_storage_quarantine_lock) do
        delete!(_captured_storage_quarantine, owner)
    end
    return nothing
end

function _captured_drain_failure(owner)
    return lock(_captured_storage_quarantine_lock) do
        quarantine = get(_captured_storage_quarantine, owner, nothing)
        isnothing(quarantine) ? nothing : quarantine.failure
    end
end

function _drain_captured_storage!(context, owner, roots=owner)
    try
        _synchronize_captured_graph!(context)
    catch failure
        retained = CapturedStorageQuarantine(
            roots,
            CapturedException(failure, catch_backtrace()),
        )
        lock(_captured_storage_quarantine_lock) do
            # A failed recovery must retain the original borrowed inputs too.
            # Its new exception is available through the caller's fault path;
            # the first unsuccessful drain remains the quarantine diagnostic.
            get!(_captured_storage_quarantine, owner, retained)
        end
        return false
    end
    _release_captured_storage!(owner)
    return true
end

struct CapturedFrameStatus{S,H,C}
    value::S
    host::H
    clear!::C
end


struct CapturedFrameOutcome{S,H}
    value::S
    host::H
end

@inline function _with_captured_context(f::F, context) where {F}
    previous = _activate_captured_context(context)
    try
        return f()
    finally
        _restore_captured_context!(context, previous)
    end
end

function _require_minimal_captured_graph(graph::PreparedGraph)
    all(!, values(graph.output_conditional)) || throw(
        GraphConfigurationError(
            "the initial captured profile does not support conditional outputs",
        ),
    )
    return graph
end

function _require_captured_support(node::PreparedNode{D}, device) where {D}
    support = FilterGraphAlgorithms.capture_support(D, device)
    support isa FilterGraphAlgorithms.SupportsCapture || throw(
        GraphConfigurationError(
            "Node '$(node.name)' Algorithm '$(FilterGraphAlgorithms.algorithm_label(D))' " *
            "has no qualified captured adapter for $(typeof(device)); " *
            "implement prepare_capture(CompleteTransition(), ...) and the complete enqueue! transition",
        ),
    )
    return nothing
end

_require_captured_support(::Tuple{}, device) = nothing

function _require_captured_support(nodes::Tuple, device)
    _require_captured_support(first(nodes), device)
    return _require_captured_support(Base.tail(nodes), device)
end

_prepare_captured_camera_metadata(context, graph, nodes, ports) = nothing
_admit_captured_camera_metadata!(::Nothing, metadata, rejection) = nothing
_copy_captured_camera_metadata!(context, ::Nothing) = nothing

function _captured_external_buffers(context, formats::NamedTuple)
    return NamedTuple{keys(formats)}(
        map(format -> _allocate_captured_buffer(context, format), values(formats)),
    )
end

function _captured_node_output_buffers(context, nodes::Tuple)
    return map(nodes) do node
        map(format -> _allocate_captured_buffer(context, format), node.output_formats)
    end
end

function _captured_protocol_error(node::PreparedNode{D}, device, reason) where {D}
    return GraphConfigurationError(
        "Node '$(node.name)' Algorithm '$(FilterGraphAlgorithms.algorithm_label(D))' " *
        "cannot prepare captured execution for $(typeof(device)): $reason; " *
        "implement prepare_capture(CompleteTransition(), device, prepared, outputs::Tuple, " *
        "inputs::Tuple, status, index::Integer[, workgroups]) and a complete " *
        "guarded enqueue! transition with preflight, complete!, and reset! methods",
    )
end

function _require_captured_methods(node, device, adapter)
    for (hook, arguments) in (
        (FilterGraphAlgorithms.enqueue!, (adapter,)),
        (FilterGraphAlgorithms.preflight, (adapter,)),
        (FilterGraphAlgorithms.complete!, (adapter, Int32(0))),
        (FilterGraphAlgorithms.reset!, (adapter,)),
    )
        applicable(hook, arguments...) || throw(
            _captured_protocol_error(node, device, "missing $(nameof(hook)) method"),
        )
    end
    return nothing
end

function _require_captured_methods(nodes::Tuple, device, captured::Tuple)
    for (node, prepared) in zip(nodes, captured)
        _require_captured_methods(node, device, prepared.adapter)
    end
    return nothing
end

function _prepare_captured_node(
    node::PreparedNode{D},
    outputs::Tuple,
    routes::Tuple,
    device_inputs::NamedTuple,
    node_outputs::Tuple,
    device,
    frame_status,
    node_index::Int32,
    workgroups,
) where {D}
    inputs = _route_inputs(routes, device_inputs, node_outputs)
    adapter = try
        FilterGraphAlgorithms.prepare_capture(
            FilterGraphAlgorithms.CompleteTransition(),
            device,
            node.prepared,
            outputs,
            inputs,
            frame_status,
            node_index,
            workgroups,
        )
    catch error
        error isa FilterGraphAlgorithms.CaptureProtocolError || rethrow()
        throw(_captured_protocol_error(node, device, sprint(showerror, error)))
    end
    return CapturedNode{D,typeof(adapter)}(adapter)
end

_prepare_captured_nodes(
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    device_inputs,
    node_outputs,
    device,
    frame_status,
    node_index::Int32,
    workgroups;
    roots=nothing,
) = ()

function _prepare_captured_nodes(
    nodes::Tuple,
    outputs::Tuple,
    routes::Tuple,
    device_inputs,
    node_outputs,
    device,
    frame_status,
    node_index::Int32,
    workgroups;
    roots=nothing,
)
    captured = _prepare_captured_node(
        first(nodes),
        first(outputs),
        first(routes),
        device_inputs,
        node_outputs,
        device,
        frame_status,
        node_index,
        workgroups,
    )
    isnothing(roots) || push!(roots, captured)
    _require_captured_methods(first(nodes), device, captured.adapter)
    return (
        captured,
        _prepare_captured_nodes(
            Base.tail(nodes),
            Base.tail(outputs),
            Base.tail(routes),
            device_inputs,
            node_outputs,
            device,
            frame_status,
            node_index + Int32(1),
            workgroups;
            roots,
        )...,
    )
end

@inline function _enqueue_captured_frame!(frame_status::CapturedFrameStatus, nodes)
    frame_status.clear!(frame_status.value; ndrange=1)
    _enqueue_captured_nodes!(nodes)
    return nothing
end

_enqueue_captured_nodes!(::Tuple{}) = nothing

@inline function _enqueue_captured_nodes!(nodes::Tuple)
    FilterGraphAlgorithms.enqueue!(first(nodes).adapter)
    return _enqueue_captured_nodes!(Base.tail(nodes))
end

_reset_captured_nodes!(::Tuple{}) = nothing

function _reset_captured_nodes!(nodes::Tuple)
    FilterGraphAlgorithms.reset!(first(nodes).adapter)
    return _reset_captured_nodes!(Base.tail(nodes))
end

_preflight_captured_nodes(::Tuple{}, rejection=nothing, index::Int=1) = nothing

function _preflight_captured_nodes(nodes::Tuple, rejection=nothing, index::Int=1)
    node = first(nodes)
    result = FilterGraphAlgorithms.preflight(node.adapter)
    result = _validate_captured_node_result(node, result)
    if result !== nothing
        isnothing(rejection) || _record_rejection!(rejection, index, result)
        return result
    end
    return _preflight_captured_nodes(Base.tail(nodes), rejection, index + 1)
end

function _complete_captured_nodes!(::Tuple{}, status::Int32, rejection=nothing, index::Int=1)
    iszero(status) || error("captured Graph returned unknown frame status $status")
    return nothing
end

function _complete_captured_nodes!(nodes::Tuple, status::Int32, rejection=nothing, index::Int=1)
    node = first(nodes)
    result = FilterGraphAlgorithms.complete!(node.adapter, status)
    result = _validate_captured_node_result(node, result)
    if result !== nothing
        isnothing(rejection) || _record_rejection!(rejection, index, result)
        return result
    end
    return _complete_captured_nodes!(Base.tail(nodes), status, rejection, index + 1)
end

@inline function _enqueue_captured_frame_status_copy!(context, frame_status)
    _copy_from_captured!(context, frame_status.host, frame_status.value)
    return nothing
end

@inline _read_captured_frame_status(frame_status) = frame_status.host[1]

function _validate_captured_node_result(::CapturedNode{D}, result) where {D}
    validated = FilterGraphAlgorithms._validate_process_result(D, result)
    validated isa FilterGraphAlgorithms.AlgorithmProcessResult && throw(
        GraphConfigurationError(
            "captured Algorithm '$(FilterGraphAlgorithms.algorithm_label(D))' " *
            "returned conditional or replacement metadata, which the initial " *
            "captured profile does not support",
        ),
    )
    return validated
end

@inline _copy_to_captured_inputs!(context, ::Tuple{}, ::Tuple{}) = nothing

@inline function _copy_to_captured_inputs!(context, device::Tuple, host::Tuple)
    _copy_to_captured!(context, first(device), first(host))
    return _copy_to_captured_inputs!(context, Base.tail(device), Base.tail(host))
end

@inline _copy_from_captured_outputs!(context, ::Tuple{}, ::Tuple{}) = nothing

@inline function _copy_from_captured_outputs!(context, host::Tuple, device::Tuple)
    _copy_from_captured!(context, first(host), first(device))
    return _copy_from_captured_outputs!(context, Base.tail(host), Base.tail(device))
end

@inline _copy_captured_outputs!(::Tuple{}, ::Tuple{}) = nothing

@inline function _copy_captured_outputs!(destination::Tuple, source::Tuple)
    copyto!(first(destination), first(source))
    return _copy_captured_outputs!(Base.tail(destination), Base.tail(source))
end

function _captured_external_outputs(graph::PreparedGraph, node_outputs::Tuple)
    arrays = map(graph.output_routes) do route
        _route_value(route, (), node_outputs)
    end
    return NamedTuple{keys(graph.output_formats)}(arrays)
end

@inline _publish_captured_metadata!(::Tuple{}, ::Tuple{}, input_metadata) = nothing

@inline function _publish_captured_metadata!(
    destinations::Tuple,
    sources::Tuple,
    input_metadata,
)
    source = first(sources)
    metadata = iszero(source) ? nothing : getfield(input_metadata, source)
    _store_metadata!(first(destinations), metadata)
    return _publish_captured_metadata!(
        Base.tail(destinations),
        Base.tail(sources),
        input_metadata,
    )
end

@inline _captured_admitted_metadata(::Tuple{}, input_metadata) = ()

@inline function _captured_admitted_metadata(sources::Tuple, input_metadata)
    source = first(sources)
    metadata = iszero(source) ?
        FilterGraphAlgorithms._NO_SAMPLE_METADATA :
        getfield(input_metadata, source)
    return (
        metadata,
        _captured_admitted_metadata(Base.tail(sources), input_metadata)...,
    )
end

"""Derive captured output metadata before launch for transport reservation."""
@inline function _captured_output_metadata_admitted(
    graph::PreparedCapturedGraph,
    input_metadata::NamedTuple,
)
    return NamedTuple{keys(graph.output_formats)}(
        _captured_admitted_metadata(
            graph.output_metadata_sources,
            input_metadata,
        ),
    )
end

function _warm_captured!(context, frame_status, nodes, feedback)
    _enqueue_captured_frame!(frame_status, nodes)
    _enqueue_captured_feedback!(feedback)
    _enqueue_captured_frame_status_copy!(context, frame_status)
    _synchronize_captured_graph!(context)
    # Zero-initialized buffers can validly reject. Complete the warm frame
    # before resetting the adapters and feedback in their retained storage.
    _complete_captured_nodes!(nodes, _read_captured_frame_status(frame_status))
    _reset_captured_nodes!(nodes)
    _reset_captured_feedback!(feedback)
    frame_status.clear!(frame_status.value; ndrange=1)
    _synchronize_captured_graph!(context)
    return nothing
end

function _capture_warm_and_record(context, frame_status, nodes::Tuple, feedback=())
    try
        return _with_captured_workspace(context) do
            _warm_captured!(context, frame_status, nodes, feedback)
            return _record_captured_graph(context, frame_status, nodes, feedback)
        end
    catch
        _drain_captured_storage!(context, (context, frame_status, nodes, feedback))
        rethrow()
    end
end

_prepare_captured_executable(context, status, nodes, feedback, routes, inputs, outputs) =
    _capture_warm_and_record(context, status, nodes, feedback)

"""
    prepare_captured_graph(
        graph::PreparedGraph,
        device;
        feedback=NamedTuple(),
        outputs=keys(graph.output_formats),
        readback=nothing,
        workgroups=CapturedWorkgroupProfile(),
    )

Prepare the initial capacity-one native captured profile for an explicitly
qualified Node chain. Preparation allocates fixed device buffers, warms the
exact enqueue path, resets the adapters without replacing their storage, and
records a native executable on a retained backend stream. Unsupported Nodes
reject the whole operation; checked execution is never used as a replay
fallback.

The current vertical slice accepts one or more unconditional, explicitly
qualified Nodes. A declared failure suppresses publication of the whole captured
graph. A successful Node adopts recoverable candidate state in prepared Node
order with respect to failures. Qualified CUDA dependency branches may defer
staged PDM adoption to the join before native completion. A later Node failure
suppresses external publication while preserving earlier successful adoption,
matching the FGN processing contract.

CUDA packs host `SubArray` inputs backed by an `Array`, including nested views,
into retained pinned input buffers. Ordinary host arrays and device arrays copy
directly into the fixed device inputs.

On CUDA and AMDGPU, `feedback` may bind an original external input name to an
original external output name with identical Port format. Bound inputs start at
zero and are removed from the public input interface. After all Nodes succeed, each
binding copies the current device output into its next-frame input. A declared
failure preserves prior feedback while earlier successful Node adoptions remain.
The commit precedes host transfer/publication; unexpected later failures retain
the existing fail-stop behavior. `reset!` clears feedback without replacing storage.

`outputs` selects public output names in the requested order and cannot remove
every output from a Graph with public outputs. Nonempty feedback requires at
least one remaining public input and output. Without feedback, source Graphs
with no public inputs and prebuilt Graphs with empty output interfaces retain
their existing behavior. Selected output metadata cannot originate at a bound
input. Resident feedback is capacity-one and unavailable to captured pipelines;
other backends reject nonempty feedback.

By default every selected output is staged to the host by `process!`.
Supply `readback` as a tuple of selected output names to stage only those
outputs. Staging follows the public output order. Use `process_device!` to consume
all selected outputs on the device.
`output_formats`, availability, and metadata describe all selected outputs.
"""
function prepare_captured_graph(
    graph::PreparedGraph,
    device;
    feedback::NamedTuple=NamedTuple(),
    outputs=keys(graph.output_formats),
    readback::Union{Nothing,Tuple}=nothing,
    workgroups::CapturedWorkgroupProfile=CapturedWorkgroupProfile(),
)
    _require_minimal_captured_graph(graph)
    ports = _captured_feedback_ports(
        graph.input_formats, graph.output_formats, output_metadata_source_inputs(graph),
        feedback, outputs,
    )
    host_names = readback === nothing ? ports.outputs : readback
    all(name -> name isa Symbol && name in ports.outputs, host_names) ||
        throw(GraphConfigurationError("captured readback contains an unselected output Port"))
    allunique(host_names) ||
        throw(GraphConfigurationError("captured readback contains duplicate output Ports"))
    # Host staging follows the public Port order for every captured owner.
    staged_names = filter(name -> name in host_names, ports.outputs)
    _require_captured_feedback_support(device, feedback)
    _require_captured_support(graph.nodes, device)
    context = _prepare_captured_context(device)
    return _with_captured_context(context) do
        device_inputs = _captured_external_buffers(context, graph.input_formats)
        node_outputs = _captured_node_output_buffers(context, graph.nodes)
        frame_status = _prepare_captured_frame_status(context)
        nodes, node_outputs = _prepare_captured_bindings(
            context, graph, device, frame_status, device_inputs, node_outputs,
            workgroups, feedback,
        )
        original_nodes = nodes
        try
            nodes = _prepare_captured_validation(
                device, nodes, graph.input_routes, node_outputs, device_inputs, node_outputs,
            )
            nodes = _prepare_captured_fusion(device, nodes, graph.input_routes)
            _require_captured_methods(graph.nodes, device, nodes)
        catch
            _drain_captured_storage!(context,
                (context, frame_status, original_nodes, nodes, device_inputs, node_outputs))
            rethrow()
        end
        device_outputs = _captured_external_outputs(graph, node_outputs)
        bindings = try
            _prepare_captured_feedback(device, feedback, device_inputs, device_outputs, frame_status)
        catch
            _drain_captured_storage!(context, (context, frame_status, nodes, device_inputs, node_outputs))
            rethrow()
        end
        camera_metadata = _prepare_captured_camera_metadata(context, graph, nodes, ports)
        executable = _prepare_captured_executable(context, frame_status, nodes, bindings,
            graph.input_routes, device_inputs, node_outputs)
        device_inputs = _select_captured_ports(device_inputs, ports.inputs)
        device_outputs = _select_captured_ports(device_outputs, ports.outputs)
        input_formats = _select_captured_ports(graph.input_formats, ports.inputs)
        output_formats = _select_captured_ports(graph.output_formats, ports.outputs)
        executable = _prepare_captured_inputs(context, executable, device_inputs)
        availability = NamedTuple{keys(output_formats)}(
            map(_ -> true, values(output_formats)),
        )
        return PreparedCapturedGraph(
            context,
            executable,
            nodes,
            device_inputs,
            device_outputs,
            map(format -> _allocate_captured_host_buffer(context, format),
                _select_captured_ports(output_formats, staged_names)),
            input_formats,
            output_formats,
            graph.parameter_formats,
            availability,
            _external_output_metadata(output_formats),
            _select_captured_ports(graph.default_input_metadata, ports.inputs),
            ports.metadata_sources,
            _captured_result_types(graph.result_types, output_formats),
            workgroups,
            CapturedGraphReady,
            nothing,
            map(node -> node.name, graph.nodes),
            _rejection_state(_captured_result_types(graph.result_types, output_formats), input_formats),
            _select_captured_ports(required_input_sample_metadata(graph),ports.inputs),
            camera_metadata,
        )
    end
end

"""
    captured_graph_state(graph::PreparedCapturedGraph)

Return the lifecycle state of a prepared captured Graph owner.
"""
captured_graph_state(graph::PreparedCapturedGraph) = graph.state

"""
    captured_graph_fault(graph::PreparedCapturedGraph)

Return the first retained unexpected fault, or `nothing` when unfaulted.
"""
captured_graph_fault(graph::PreparedCapturedGraph) = graph.fault

"""
    captured_workgroup_profile(graph::PreparedCapturedGraph)

Return the fixed workgroup profile recorded into a captured Graph.
"""
captured_workgroup_profile(graph::PreparedCapturedGraph) = graph.workgroups

function _throw_unavailable_captured_graph(graph::PreparedCapturedGraph)
    graph.state === CapturedGraphRunning && throw(CapturedGraphBusy())
    if graph.state === CapturedGraphFaulted
        fault = graph.fault
        if isnothing(fault)
            fault = CapturedGraphFault(
                :invariant,
                ErrorException("faulted captured Graph has no retained failure"),
            )
            graph.fault = fault
        end
        throw(fault)
    end
    return nothing
end

function _fault_captured_graph!(graph::PreparedCapturedGraph, phase::Symbol, cause)
    fault = something(graph.fault, CapturedGraphFault(phase, cause))
    graph.fault = fault
    graph.state = CapturedGraphFaulted
    return fault
end

@noinline function _transfer_captured_inputs!(context, device_inputs, inputs)
    _copy_to_captured_inputs!(context, values(device_inputs), values(inputs))
    return nothing
end

@noinline function _launch_and_complete_captured_frame!(
    context,
    executable,
    host_outputs,
    device_outputs,
)
    _launch_captured_graph!(context, executable)
    _enqueue_captured_frame_status_copy!(context, executable.frame_status)
    _transfer_captured_outputs!(context, host_outputs, device_outputs)
    _synchronize_captured_graph!(context)
    return _read_captured_frame_status(executable.frame_status)
end

@noinline function _transfer_captured_outputs!(context, host_outputs, device_outputs)
    _copy_from_captured_outputs!(
        context,
        values(host_outputs),
        values(_select_captured_ports(device_outputs, keys(host_outputs))),
    )
    return nothing
end

_transfer_captured_outputs!(context, ::Nothing, device_outputs) = nothing

@inline function _execute_captured_frame!(graph, inputs, host_outputs)
    _transfer_captured_inputs!(
        graph.context,
        graph.device_inputs,
        inputs,
    )
    status = _launch_and_complete_captured_frame!(
        graph.context,
        graph.executable,
        host_outputs,
        graph.device_outputs,
    )
    result = _complete_captured_nodes!(graph.nodes, status, graph.rejection)
    result === nothing || return result
    return nothing
end

struct _CapturedDeviceConsumer{F}
    consume::F
end

function _require_full_captured_readback(graph::PreparedCapturedGraph)
    keys(graph.host_outputs) == keys(graph.output_formats) || throw(
        GraphConfigurationError("this captured Graph owner requires host readback for all public output Ports"),
    )
    return nothing
end

_captured_host_outputs(graph, ::NamedTuple) = graph.host_outputs
_captured_host_outputs(graph, ::_CapturedDeviceConsumer) = nothing

_publish_captured_outputs!(outputs::NamedTuple, graph) =
    _copy_captured_outputs!(values(outputs), values(graph.host_outputs))
function _publish_captured_outputs!(consumer::_CapturedDeviceConsumer, graph)
    try
        consumer.consume(graph.device_outputs)
    catch error
        throw(_fault_captured_graph!(graph, :publication, error))
    end
    return nothing
end

function _process_captured_admitted!(
    outputs,
    graph::PreparedCapturedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
    ::GraphResultTypes{Availability,Failure},
)::Union{Availability,Failure} where {Availability,Failure}
    _throw_unavailable_captured_graph(graph)
    _clear_rejection!(graph.rejection)
    preflight = _preflight_captured_nodes(graph.nodes, graph.rejection)
    if preflight !== nothing
        _rejection_inputs!(graph.rejection, input_metadata)
        return preflight
    end

    graph.state = CapturedGraphRunning
    try
        previous_context = _activate_captured_context(graph.context)
        result = try
            admission = _admit_captured_camera_metadata!(graph.camera_metadata, input_metadata, graph.rejection)
            if admission === nothing
                _copy_captured_camera_metadata!(graph.context, graph.camera_metadata)
                _execute_captured_frame!(graph, inputs, _captured_host_outputs(graph, outputs))
            else
                admission
            end
        catch
            # The context is still active here. Preserve the execution error
            # even if draining fails, retaining all DMA storage in that case.
            _drain_captured_storage!(graph.context, graph, inputs)
            rethrow()
        finally
            _restore_captured_context!(graph.context, previous_context)
        end
        if result !== nothing
            _rejection_inputs!(graph.rejection, input_metadata)
            graph.state = CapturedGraphReady
            return result
        end
        _publish_captured_outputs!(outputs, graph)
        _publish_captured_metadata!(
            values(graph.published_metadata),
            graph.output_metadata_sources,
            input_metadata,
        )
        graph.state = CapturedGraphReady
        return graph.availability
    catch error
        error isa CapturedGraphBusy && rethrow()
        throw(_fault_captured_graph!(graph, :execution, error))
    end
end

"""
    process!(outputs, graph::PreparedCapturedGraph, inputs[, input_metadata])

Validate and admit one complete frame, copy it into fixed device input storage,
launch the retained native executable exactly once, enqueue status and output
copies into fixed host staging, wait once for completion, and copy successful
outputs before publishing their metadata. Declared expected Algorithm failures
leave caller outputs and retained metadata unchanged. Inputs and input
metadata require the public input Port names. Output arrays require the
prepared host readback Port names. Both accept any field order.
Unexpected transfer, launch, completion, or finalization failures make the
owner fail-stop until [`reset!`](@ref) succeeds or the owner is replaced.
An execution failure attempts to drain the retained stream before returning.
If draining also fails, the owner, admitted inputs, and secondary failure
traceback remain retained until a successful reset establishes completion.
After an unexpected failure, callers must preserve borrowed input storage and
must not modify, resize, unpin, or release it until `reset!` succeeds. Retaining
a Julia reference prevents collection; it does not prevent caller mutation or
release of externally owned storage. Abandoning an undrained owner retains its
storage until process exit rather than releasing buffers with pending DMA.
"""
function process!(
    outputs::NamedTuple,
    graph::PreparedCapturedGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
)
    _throw_unavailable_captured_graph(graph)
    inputs = _validate_external_arrays(:input, graph.input_formats, inputs)
    formats = _select_captured_ports(graph.output_formats, keys(graph.host_outputs))
    outputs = _validate_external_arrays(:output, formats, outputs)
    _validate_external_output_aliases!(keys(formats), values(outputs))
    input_metadata = _validate_external_metadata(graph.input_formats, input_metadata)
    return _process_captured_admitted!(
        outputs,
        graph,
        inputs,
        input_metadata,
        graph.result_types,
    )
end

function process!(
    outputs::NamedTuple,
    graph::PreparedCapturedGraph,
    inputs::NamedTuple,
)
    return process!(outputs, graph, inputs, graph.default_input_metadata)
end

"""
    process_device!(consume, graph::PreparedCapturedGraph, inputs[, input_metadata])

Run a complete captured frame and wait for its device and status completion.
On success call `consume(outputs)` with the selected graph-owned device arrays.
The arrays are read-only borrows and the consumer must finish all use, including
asynchronous work, before returning or throwing. The graph remains busy during
the callback. Successful processing returns the usual output availability.
Expected rejection returns the declared failure without invoking the consumer.
Image outputs stay on the device. A callback error faults the owner until reset.
Frame status and metadata follow the ordinary captured processing contract.
"""
function process_device!(consume::F, graph::PreparedCapturedGraph, inputs::NamedTuple,
        input_metadata::NamedTuple=graph.default_input_metadata) where {F}
    _throw_unavailable_captured_graph(graph)
    inputs = _validate_external_arrays(:input, graph.input_formats, inputs)
    metadata = _validate_external_metadata(graph.input_formats, input_metadata)
    return _process_captured_admitted!(_CapturedDeviceConsumer(consume), graph, inputs,
        metadata, graph.result_types)
end

function output_metadata(graph::PreparedCapturedGraph)
    return NamedTuple{keys(graph.output_formats)}(
        _metadata_values(values(graph.published_metadata)),
    )
end

"""
    reset!(graph::PreparedCapturedGraph)

At a serialized lifecycle boundary, reset every captured Algorithm adapter and
clear a retained fail-stop fault without replacing captured addresses. A reset
failure leaves the owner faulted and preserves the first recorded fault.
"""
function reset!(graph::PreparedCapturedGraph)
    graph.state === CapturedGraphRunning && throw(CapturedGraphBusy())
    try
        _with_captured_context(graph.context) do
            try
                _reset_captured_nodes!(graph.nodes)
                _reset_captured_feedback!(graph.executable)
                _synchronize_captured_graph!(graph.context)
            catch
                _drain_captured_storage!(graph.context, graph)
                rethrow()
            end
        end
    catch error
        throw(_fault_captured_graph!(graph, :reset, error))
    end
    _release_captured_storage!(graph)
    graph.fault = nothing
    graph.state = CapturedGraphReady
    _clear_rejection!(graph.rejection)
    return graph
end
