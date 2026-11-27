"""A controller execution or publisher failure requiring an explicit reset.

`cause` is the original exception or declared feedback rejection. A published
command cannot be recalled; reset clears scientific state without sending a
replacement hardware command.
"""
struct ControllerFault{C} <: Exception
    cause::C
end
function Base.showerror(io::IO, fault::ControllerFault)
    print(io, "controller requires reset: ")
    fault.cause isa Exception ? showerror(io, fault.cause) : show(io, fault.cause)
end

"""
One private, single-writer controller Graph and its command publisher.

Construct with [`prepare_controller_graph`](@ref). Serialize processing and reset;
the underlying Graph must not be used independently. Scientific Algorithms and
their per-node commit boundaries are unchanged.
"""
mutable struct PreparedControllerGraph{G,E,P,K}
    graph::G
    execution::E
    publisher::P
    command::K
    running::Bool
    faulted::Bool
    command_published::Bool
end

struct CompleteControllerPublication end
struct CpuControllerPublication{P,S,R,O,I}
    prefix::P
    suffix::S
    command_routes::R
    other_routes::O
    suffix_index::I
end

_controller_algorithm(::PreparedNode{D}) where D = D
_controller_algorithm(::CapturedNode{D}) where D = D
_controller_buffer(x, ::Val{K}) where K = getproperty(x, K)
_controller_without_command(x, ::Val{K}) where K =
    Base.structdiff(x, NamedTuple{(K,)}((getproperty(x, K),)))

# Qualified synchronous controller profile, rather than an interpretation of
# arbitrary user Algorithms' deferred completion or externally visible effects.
const _CONTROLLER_PREFIX = (
    FilterGraphAlgorithms.PixelCalibrationU16F32,
    FilterGraphAlgorithms.PyramidPupilImageF32,
    FilterGraphAlgorithms.PyramidPupilReconstructorF32,
    FilterGraphAlgorithms.ClosedLoopCorrectionF32,
    FilterGraphAlgorithms.ControllerToVdmF32,
    FilterGraphAlgorithms.VdmToPdmF32,
    FilterGraphAlgorithms.PdmSystemFlatF32,
    FilterGraphAlgorithms.PdmEngineeringOffsetF32,
    FilterGraphAlgorithms.PdmCommandF32,
)
const _CONTROLLER_SUFFIX = (
    FilterGraphAlgorithms.PdmFeedbackToVdmF32,
    FilterGraphAlgorithms.VdmFeedbackToControllerF32,
)

function _controller_cut(g, command)
    command in keys(g.output_formats) || throw(ArgumentError("command must be a public output"))
    indices = findall(n -> _controller_algorithm(n) === FilterGraphAlgorithms.PdmCommandF32, g.nodes)
    length(indices) == 1 || throw(GraphConfigurationError("controller requires one PDM command node"))
    cut = only(indices)
    all(n -> _controller_algorithm(n) in _CONTROLLER_PREFIX, g.nodes[1:cut]) ||
        throw(GraphConfigurationError("unsupported controller prefix"))
    map(_controller_algorithm, g.nodes[cut+1:end]) == _CONTROLLER_SUFFIX ||
        throw(GraphConfigurationError("controller requires the two feedback projection nodes after command"))
    _validate_controller_command(g, Val(command), cut)
    return cut
end

function _controller_reject_alias(command, value)
    if value isa AbstractArray
        Base.mightalias(command, value) && throw(GraphConfigurationError("command aliases feedback storage"))
    elseif value isa NamedTuple || value isa Tuple
        foreach(v -> _controller_reject_alias(command, v), value)
    elseif isstructtype(typeof(value)) &&
            startswith(string(parentmodule(typeof(value))), "FilterGraphAlgorithms")
        for name in fieldnames(typeof(value))
            _controller_reject_alias(command, getfield(value, name))
        end
    end
    return nothing
end

function _validate_controller_command(g::PreparedGraph, key, cut)
    all(n -> !any(n.output_conditional), g.nodes) ||
        throw(GraphConfigurationError("conditional controller outputs are unsupported"))
    routes = NamedTuple{keys(g.output_formats)}(g.output_routes)
    _controller_buffer(routes, key) isa NodeOutput{cut,1} ||
        throw(GraphConfigurationError("command must refer to the PDM demanded output"))
    for n in g.nodes[cut+1:end]
        _controller_reject_alias(g.node_outputs[cut][1], n.prepared)
    end
    return nothing
end
function _validate_controller_command(g::PreparedCapturedGraph, key, cut)
    _throw_unavailable_captured_graph(g)
    command = _controller_buffer(g.device_outputs, key)
    command === g.nodes[cut].adapter.demanded ||
        throw(GraphConfigurationError("command must refer to the PDM demanded output"))
    for n in g.nodes[cut+1:end]
        _controller_reject_alias(command, n.adapter)
    end
    return nothing
end

"""
    prepare_controller_graph(graph, publisher; command=:demanded, publication=:early)

Own a prepared CPU Graph or CUDA captured Graph for complete-frame controller
execution. Early command publication is the default. `publisher(command, metadata)`
receives the validated limited command and its `SampleMetadata` (or `nothing`)
at command readiness, without waiting for full-frame output handling. CPU calls
the publisher before feedback; CUDA overlaps command readback with feedback where
possible, without guaranteeing that the host callback precedes GPU completion.
`publication=:complete` publishes
after complete-frame processing for comparison or consumers requiring that order.
The supported profile ends in PDM command and the two existing feedback projection
Algorithms; unsupported Graphs reject preparation. Generic `process!` is unchanged.

Preparation resets the supplied Graph and takes exclusive ownership. Bind
calibrations first; prepare a new owner to change them. The publisher runs
synchronously and must not reenter/reset the owner, mutate inputs or Graph storage,
or retain borrowed command storage across calls without copying it. A blocking
publisher delays frame completion. No next frame is admitted before feedback
completes; the controller's feedback sample and numerical order are unchanged.

A declared rejection before publication returns its existing enum and withholds
all outputs. A rejection after publication retains the command and its metadata,
holds remaining outputs and faults until `reset!`. Unexpected processing/publisher
exceptions also fault. Earlier node state is not rolled back. CPU faults throw
`ControllerFault`; CUDA faults use `CapturedGraphFault` and drain native work.
After an unexpected CUDA fault, keep borrowed input/output storage unmodified
and unreleased until reset succeeds: a failed drain retains that storage.
No compensating or zero hardware command is sent. Serialize calls from one writer.
Captured Graphs must stage every public output Port.
AMDGPU retains ordinary capture support and supports `publication=:complete` here.
"""
function prepare_controller_graph(
    graph::Union{PreparedGraph,PreparedCapturedGraph}, publisher::P;
    command::Symbol=:demanded, publication::Symbol=:early,
) where P
    publication in (:early, :complete) || throw(ArgumentError("publication must be :early or :complete"))
    graph isa PreparedCapturedGraph && _require_full_captured_readback(graph)
    cut = _controller_cut(graph, command)
    reset!(graph)
    key = Val(command)
    execution = publication === :early ? _prepare_controller_execution(graph, key, cut) :
        CompleteControllerPublication()
    return PreparedControllerGraph(graph, execution, publisher, key, false, false, false)
end

_controller_segment(g, indices) = (
    nodes=g.nodes[indices], outputs=g.node_outputs[indices], routes=g.input_routes[indices],
    available=g.node_output_available[indices], metadata=g.node_output_metadata[indices],
    inflight=g.node_inflight[indices],
)
function _prepare_controller_execution(g::PreparedGraph, key, cut)
    routes = NamedTuple{keys(g.output_formats)}(g.output_routes)
    return CpuControllerPublication(
        _controller_segment(g, 1:cut), _controller_segment(g, cut+1:length(g.nodes)),
        (_controller_buffer(routes, key),), values(_controller_without_command(routes, key)), Val(cut+1),
    )
end
_prepare_controller_execution(g::PreparedCapturedGraph, key, cut) = throw(
    GraphConfigurationError("early controller publication is not qualified for $(typeof(g.context))"),
)

@inline _controller_check_aliases!(command, ::Tuple{}) = nothing
@inline function _controller_check_aliases!(command, inputs::Tuple)
    _external_output_overlap(command, first(inputs)) && throw(ArgumentError("command aliases an input"))
    return _controller_check_aliases!(command, Base.tail(inputs))
end
@inline _publish_controller_command!(owner, command) =
    owner.publisher(command, _controller_buffer(output_metadata(owner.graph), owner.command))

"""
    process!(outputs, owner::PreparedControllerGraph, inputs[, metadata])

Process a complete input frame, invoking the owner's publisher at the configured
command boundary. Return the ordinary output availability or a declared prefix
rejection. Feedback completes before return. Inputs remain borrowed until return.
After an unexpected CUDA failure, retain borrowed input/output storage unchanged
until `reset!` succeeds if native completion could not be established.
"""
function process!(outputs::NamedTuple, owner::PreparedControllerGraph,
        inputs::NamedTuple, metadata::NamedTuple)
    return _process_controller!(outputs, owner, inputs, metadata, owner.graph.result_types)
end
process!(outputs::NamedTuple, owner::PreparedControllerGraph, inputs::NamedTuple) =
    process!(outputs, owner, inputs, owner.graph.default_input_metadata)

function _process_controller!(outputs, owner, inputs, metadata,
        ::GraphResultTypes{A,F})::Union{A,F} where {A,F}
    owner.running && throw(InvalidStateException("controller frame is running", :running))
    owner.faulted && throw(_controller_fault(owner, ControllerFault(ErrorException("reset required"))))
    g = owner.graph
    inputs = _validate_external_arrays(:input, g.input_formats, inputs)
    outputs = _validate_external_arrays(:output, g.output_formats, outputs)
    _validate_external_output_aliases!(keys(g.output_formats), values(outputs))
    metadata = _validate_external_metadata(g.input_formats, metadata)
    _controller_check_aliases!(_controller_buffer(outputs, owner.command), values(inputs))
    owner.running = true
    owner.command_published = false
    try
        return _execute_controller!(outputs, owner, inputs, metadata, owner.execution)
    catch error
        owner.faulted = true
        throw(_controller_fault(owner, error))
    finally
        owner.running = false
    end
end

_controller_fault(owner, error) = error isa ControllerFault ? error : ControllerFault(error)
_controller_fault(owner::PreparedControllerGraph{<:PreparedCapturedGraph}, error) =
    _fault_captured_graph!(owner.graph, :controller, error)

function _execute_controller!(outputs, owner, inputs, metadata, ::CompleteControllerPublication)
    result = process!(outputs, owner.graph, inputs, metadata)
    if result isa NamedTuple && all(values(result))
        owner.command_published = true
        _publish_controller_command!(owner, _controller_buffer(outputs, owner.command))
    end
    return result
end

@inline function _controller_nodes!(g, segment, inputs, metadata, ::Val{I}) where I
    return _process_nodes!(segment.nodes, segment.outputs, segment.routes, segment.available,
        segment.metadata, segment.inflight, inputs, metadata, g.node_outputs,
        g.node_output_available, g.node_output_metadata, false, g.rejection, I)
end
function _execute_controller!(outputs, owner, inputs, metadata, execution::CpuControllerPublication)
    return _execute_cpu_controller!(outputs, owner, inputs, metadata, execution, owner.graph.result_types)
end
function _execute_cpu_controller!(outputs, owner, inputs, metadata, execution,
        ::GraphResultTypes{A,F})::Union{A,F} where {A,F}
    g = owner.graph
    _clear_available!(g.node_output_available)
    _clear_rejection!(g.rejection)
    try
        result, deferred = _controller_nodes!(g, execution.prefix, inputs, metadata, Val(1))
        result = result::Union{Nothing,F}
        if result !== nothing
            _rejection_inputs!(g.rejection, metadata)
            _clear_available!(g.node_output_available)
            return result
        end
        deferred && error("qualified controller prefix unexpectedly deferred")
        command = _controller_buffer(outputs, owner.command)
        owner.command_published = true
        _copy_external_outputs!(execution.command_routes, (command,), g.node_outputs, g.node_output_available)
        _copy_external_metadata!(execution.command_routes,
            (_controller_buffer(g.published_metadata, owner.command),), g.node_output_metadata, g.node_output_available)
        _publish_controller_command!(owner, command)
        result, deferred = _controller_nodes!(g, execution.suffix, inputs, metadata, execution.suffix_index)
        if result !== nothing
            _rejection_inputs!(g.rejection, metadata)
            throw(ControllerFault(result))
        end
        deferred && error("qualified controller feedback unexpectedly deferred")
        _copy_external_outputs!(execution.other_routes, values(_controller_without_command(outputs, owner.command)),
            g.node_outputs, g.node_output_available)
        _copy_external_metadata!(execution.other_routes,
            values(_controller_without_command(g.published_metadata, owner.command)),
            g.node_output_metadata, g.node_output_available)
        return map(_ -> true, g.output_formats)
    catch
        _clear_available!(g.node_output_available)
        rethrow()
    end
end

_reset_controller_execution!(owner, execution) = reset!(owner.graph)

"""Reset controller scientific state and clear its fault after work completes.

Retained publication metadata is unchanged. This sends no actuator command.
"""
function reset!(owner::PreparedControllerGraph)
    owner.running && throw(InvalidStateException("controller frame is running", :running))
    owner.faulted = true
    _reset_controller_execution!(owner, owner.execution)
    owner.command_published = false
    owner.faulted = false
    return owner
end

input_formats(owner::PreparedControllerGraph) = input_formats(owner.graph)
output_formats(owner::PreparedControllerGraph) = output_formats(owner.graph)
parameter_formats(owner::PreparedControllerGraph) = parameter_formats(owner.graph)
output_metadata(owner::PreparedControllerGraph) = output_metadata(owner.graph)
output_metadata_source_inputs(owner::PreparedControllerGraph) = output_metadata_source_inputs(owner.graph)
required_input_sample_metadata(owner::PreparedControllerGraph) = required_input_sample_metadata(owner.graph)
rejection_context(owner::PreparedControllerGraph) = rejection_context(owner.graph)
