function _prepare_gpu_readout_backend end
function _submit_gpu_readout! end
function _finish_gpu_readout! end
function _drain_gpu_readout! end
function _reset_gpu_readout! end
function _gpu_readout_failure end
# Snapshot retained host bookkeeping only, even after a fault or failed drain.
# Implementations must not launch, wait for, or dereference device storage.
function _gpu_readout_counters end

function _prepare_gpu_readout_backend(graph, device, lanes, ports, calibration, feedback)
    throw(GraphConfigurationError(
        "GPU readout execution is not supported on $(typeof(device))"))
end

function _gpu_readout_plan(plan::FilterGraphAlgorithms.PyramidReadoutMeasurementBlockPlan)
    plan.source.readout
end
function _gpu_readout_plan(plan::FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockPlan)
    plan.readout
end

function _gpu_readout_frame(workspace::FilterGraphAlgorithms.PyramidReadoutMeasurementBlockWorkspace)
    workspace.source.frame
end
function _gpu_readout_frame(workspace::FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockWorkspace)
    workspace.frame
end

const _GpuReadoutWfs = Union{
    FilterGraphAlgorithms.PyramidReadoutMeasurementBlockF32,
    FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32
}

function _require_gpu_readout_graph(graph, device, ports, feedback)
    length(graph.nodes) >= 2 ||
        throw(GraphConfigurationError("readout requires WFS and reconstruction Nodes"))
    first(graph.nodes) isa PreparedNode{<:_GpuReadoutWfs} || throw(
        GraphConfigurationError("the first Node must produce pyramid or Shack–Hartmann readout measurements"),
    )
    graph.nodes[2] isa
    PreparedNode{FilterGraphAlgorithms.IncrementalDenseReconstructorF32} ||
        throw(GraphConfigurationError("the second Node must be incremental dense reconstruction"))
    length(ports.inputs) == 1 &&
        graph.input_routes[1] == (ExternalInput{findfirst(==(only(ports.inputs)), keys(graph.input_formats))}(),) ||
        throw(GraphConfigurationError("readout requires one calibrated detector-block input"))
    graph.input_routes[2] == (NodeOutput{1, 1}(), NodeOutput{1, 2}(), NodeOutput{1, 3}()) ||
        throw(GraphConfigurationError("reconstruction must consume the WFS measurements, columns and count"))
    wfs_count = graph.nodes[1].output_formats[1].shape[1]
    size(graph.nodes[2].prepared.plan.matrix, 2) == wfs_count ||
        throw(GraphConfigurationError("WFS and reconstruction measurement counts must match"))
    for routes in graph.input_routes[3:end], route in routes

        (_gpu_readout_suffix_route(route) || _gpu_readout_feedback_route(route, graph, feedback)) || throw(
            GraphConfigurationError("captured suffix inputs must come from reconstruction, earlier suffix Nodes, or retained feedback"),
        )
    end
    all(_gpu_readout_suffix_route, graph.output_routes) || throw(
        GraphConfigurationError("readout outputs must come from reconstruction or its captured suffix"),
    )
    any(n -> n.prepared isa FilterGraphAlgorithms.PreparedAlgorithm{FilterGraphAlgorithms.ChoppedImageDifferenceF32},
        graph.nodes[3:end]) && throw(GraphConfigurationError(
        "GPU readout capture does not support host-admitted chopped-image metadata"))
    _require_captured_support(graph.nodes[3:end], device)
    all(node -> all(!, node.output_conditional), graph.nodes[3:end]) ||
        throw(GraphConfigurationError("captured suffix Nodes must have unconditional outputs"))
    return nothing
end

_gpu_readout_suffix_route(::Any) = false
_gpu_readout_suffix_route(::NodeOutput{N, P}) where {N, P} = N >= 2

_gpu_readout_feedback_route(route, graph, feedback) = false
_gpu_readout_feedback_route(::ExternalInput{I}, graph, feedback) where {I} =
    haskey(feedback, keys(graph.input_formats)[I])

function _readout_input_formats(graph, ports, calibration)
    formats = _select_captured_ports(graph.input_formats, ports.inputs)
    isnothing(calibration) && return formats
    readout = _gpu_readout_plan(graph.nodes[1].prepared.plan)
    shape = only(values(formats)).shape
    shape[1] == 1 && readout.detector_size == size(calibration.flat) &&
        readout.detector_size == (shape[2] * readout.block_count, shape[3]) ||
        throw(GraphConfigurationError("raw calibration requires a contiguous detector row stream"))
    index = 0
    for ordinal in 1:readout.block_count, row in 1:shape[2], col in 1:shape[3]
        index += 1
        readout.detector_rows[index] == (ordinal-1)*shape[2]+row &&
            readout.detector_columns[index] == col ||
            throw(GraphConfigurationError("raw readout mapping must follow detector row order"))
    end
    format = only(values(formats))
    raw = FilterGraphAlgorithms.AlgorithmFormat(UInt16, (shape[2], shape[3]),
        FilterGraphAlgorithms.RAW_DETECTOR_PIXELS_V1, format.layout; rate=format.rate)
    return NamedTuple{ports.inputs}((raw,))
end

"""
    PreparedGpuReadoutGraph

An owned CUDA progressive WFS/reconstruction Graph with a captured terminal
suffix. Use [`prepare_gpu_readout_graph`](@ref) and close it in `finally`.
Calls must be serialized by the caller; blocks of the same frame execute
asynchronously on retained GPU streams.
"""
mutable struct PreparedGpuReadoutGraph{G, B, M, A, R, I, O, T, S}
    graph::G
    backend::B
    input_formats::I
    output_formats::O
    result_types::T
    output_metadata_sources::S
    row_lines::Int
    frame::FilterGraphAlgorithms.DetectorReadoutFrame
    published_metadata::M
    availability::A
    deferred::A
    rejection::R
    closed::Bool
end

"""
    isopen(owner::PreparedGpuReadoutGraph) -> Bool

Return whether the GPU readout owner's resources remain open. A faulted owner
remains open until a successful `close`, but cannot process or reset.
"""
Base.isopen(owner::PreparedGpuReadoutGraph) = !owner.closed

"""
    prepare_gpu_readout_graph(graph::PreparedGraph, device; lanes=128)
    prepare_gpu_readout_graph(path, device; algorithms, lanes=128)

Prepare fixed storage for calibrated Float32 detector blocks → pyramid or
Shack–Hartmann readout measurements → incremental dense reconstruction → a
capture-qualified terminal suffix. Configure calibration Parameters on the
checked Graph before preparation. The device owner has independent state;
replace it to change Parameters. `lanes` must be 32, 64, 128, or 256.

Borrowed blocks are copied before return. Input DMA and WFS processing can
overlap ordered reconstruction of earlier blocks from the same frame. Only
terminal processing copies final outputs and status to the host and waits.
Intermediate measurements, reconstruction and controller state stay on-device.
Expected numerical failures are reported at terminal completion; invalid
metadata abandons partial work. An unexpected host staging or submission error
faults the owner. Close it and prepare a fresh owner to resume processing.
Floating-point reductions require application-specific CPU/GPU tolerances.
The first device failure gates later state adoption for that logical frame.
This differs from continuing CPU callbacks after an earlier Node rejection.

The initial profile admits exactly one WFS producer followed by one incremental
reconstructor; all remaining Nodes must support capture and depend only on
these Nodes. Pass a PixelCalibrationPlan as `calibration` for contiguous raw UInt16
rows with row-offset metadata. Map private retained inputs to outputs with
`feedback` and select public outputs with `outputs`. Retained feedback
advances only after successful terminal processing. Always call `close` before
releasing the owner.
"""
function prepare_gpu_readout_graph(graph::PreparedGraph, device; lanes::Integer = 128,
        calibration::Union{Nothing,FilterGraphAlgorithms.PixelCalibrationPlan}=nothing,
        feedback::NamedTuple=NamedTuple(), outputs=keys(graph.output_formats))
    lanes in (32, 64, 128, 256) || throw(ArgumentError("lanes must be 32, 64, 128, or 256"))
    ports = _captured_feedback_ports(graph.input_formats, graph.output_formats,
        output_metadata_source_inputs(graph), feedback, outputs)
    _require_gpu_readout_graph(graph, device, ports, feedback)
    _require_captured_feedback_support(device, feedback)
    inputs = _readout_input_formats(graph, ports, calibration)
    selected = _select_captured_ports(graph.output_formats, ports.outputs)
    backend = _prepare_gpu_readout_backend(graph, device, Int(lanes), ports, calibration, feedback)
    availability = map(_ -> true, selected)
    deferred = map(_ -> false, availability)
    frame = FilterGraphAlgorithms.DetectorReadoutFrame()
    frame.pending_discontinuity = _gpu_readout_frame(graph.nodes[1].prepared.workspace).pending_discontinuity ||
                                  graph.nodes[2].prepared.workspace.pending_discontinuity
    owner = PreparedGpuReadoutGraph(graph, backend, inputs, selected,
        _captured_result_types(graph.result_types, selected), ports.metadata_sources,
        isnothing(calibration) ? 1 : only(values(inputs)).shape[1],
        frame, _external_output_metadata(selected), availability, deferred,
        _rejection_state(graph.result_types, inputs), false)
    lock(_gpu_progressive_roots_lock) do
        _gpu_progressive_roots[owner] = nothing
    end
    return owner
end

function prepare_gpu_readout_graph(path::AbstractString, device; algorithms::Tuple, kwargs...)
    return prepare_gpu_readout_graph(prepare_graph(path; algorithms), device; kwargs...)
end

function _assert_gpu_readout_open(owner)
    owner.closed && throw(ArgumentError("GPU readout Graph is closed"))
    owner.backend.faulted &&
        error("GPU readout execution faulted; close and prepare a new owner")
    return nothing
end

function _gpu_readout_metadata_failure(::PreparedNode{FilterGraphAlgorithms.PyramidReadoutMeasurementBlockF32})
    FilterGraphAlgorithms.PyramidReadoutMetadataMissing
end
function _gpu_readout_metadata_failure(::PreparedNode{FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32})
    FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockMetadataMissing
end
function _gpu_readout_offset_failure(::PreparedNode{FilterGraphAlgorithms.PyramidReadoutMeasurementBlockF32})
    FilterGraphAlgorithms.PyramidReadoutOffsetOverflow
end
function _gpu_readout_offset_failure(::PreparedNode{FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32})
    FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockOffsetOverflow
end

function _abandon_gpu_readout!(owner)
    _drain_gpu_readout!(owner.backend)
    FilterGraphAlgorithms._abandon_detector_readout!(owner.frame)
    return nothing
end

function _process_gpu_readout!(outputs, owner, inputs, metadata,
        ::GraphResultTypes{A, F})::Union{A, F} where {A, F}
    sample = _rejection_input_metadata(first(metadata))
    if isnothing(sample)
        _abandon_gpu_readout!(owner)
        failure = _gpu_readout_metadata_failure(first(owner.graph.nodes))
        _record_rejection!(owner.rejection, 1, failure)
        _rejection_inputs!(owner.rejection, metadata)
        return failure
    end
    if UInt64(sample.offset) > UInt64(typemax(Int)) || !iszero(rem(sample.offset, owner.row_lines))
        _abandon_gpu_readout!(owner)
        failure = _gpu_readout_offset_failure(first(owner.graph.nodes))
        _record_rejection!(owner.rejection, 1, failure)
        _rejection_inputs!(owner.rejection, metadata)
        return failure
    end
    block_sample = FilterGraphAlgorithms.SampleMetadata(sample; offset = div(sample.offset, owner.row_lines))
    iszero(sample.offset) && _drain_gpu_readout!(owner.backend)
    accepted = FilterGraphAlgorithms._accept_detector_readout!(
        owner.frame, block_sample, owner.backend.block_count
    )
    if !accepted
        _drain_gpu_readout!(owner.backend)
        return owner.deferred
    end
    _submit_gpu_readout!(owner.backend, first(inputs), Int(block_sample.offset) + 1)
    sample.terminal || return owner.deferred
    status = _finish_gpu_readout!(owner.backend)
    discontinuity = owner.frame.current_discontinuity
    if !iszero(status) && (status >> 8) <= 2
        FilterGraphAlgorithms._abandon_detector_readout!(owner.frame)
    else
        FilterGraphAlgorithms._complete_detector_readout!(owner.frame)
    end
    failure = _gpu_readout_failure(owner.backend, status)
    if !isnothing(failure)
        _record_rejection!(owner.rejection, Int(status >> 8), failure)
        _rejection_inputs!(owner.rejection, metadata)
        return failure
    end
    _copy_captured_outputs!(values(outputs), values(owner.backend.host_outputs))
    terminal = FilterGraphAlgorithms.SampleMetadata(sample; offset = 0, terminal = true, discontinuity)
    _publish_captured_metadata!(values(owner.published_metadata), owner.backend.metadata_sources, (terminal,))
    return owner.availability
end

"""
    process!(outputs, owner::PreparedGpuReadoutGraph, inputs[, metadata])

Admit one ordered detector block and enqueue GPU work without waiting for
nonterminal completion. Return unavailable outputs until terminal completion,
then publish final command outputs and metadata together on success. Numerical
failures retain caller outputs and published metadata. Metadata must identify
one frame with the correct terminal bit. Supply contiguous zero-based block
ordinals for calibrated input and first-row offsets for raw input.
"""
function process!(outputs::NamedTuple, owner::PreparedGpuReadoutGraph, inputs::NamedTuple, metadata::NamedTuple)
    _assert_gpu_readout_open(owner)
    inputs = _validate_external_arrays(:input, owner.input_formats, inputs)
    outputs = _validate_external_arrays(:output, owner.output_formats, outputs)
    _validate_external_output_aliases!(keys(outputs), values(outputs))
    metadata = _validate_external_metadata(owner.input_formats, metadata)
    _clear_rejection!(owner.rejection)
    return _process_gpu_readout!(outputs, owner, inputs, metadata, owner.result_types)
end

function process!(outputs::NamedTuple, owner::PreparedGpuReadoutGraph, inputs::NamedTuple)
    process!(outputs, owner, inputs, _select_captured_ports(owner.graph.default_input_metadata, keys(owner.input_formats)))
end

"""
    reset!(owner::PreparedGpuReadoutGraph)

Drain queued blocks and reset WFS, reconstruction and captured suffix state in
place. Previously published output metadata is retained until new publication.
"""
function reset!(owner::PreparedGpuReadoutGraph)
    _assert_gpu_readout_open(owner)
    _reset_gpu_readout!(owner.backend)
    owner.frame = FilterGraphAlgorithms.DetectorReadoutFrame()
    if first(owner.graph.nodes) isa
       PreparedNode{FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32}
        FilterGraphAlgorithms._abandon_detector_readout!(owner.frame)
    end
    _clear_rejection!(owner.rejection)
    return owner
end

"""
    output_metadata(owner::PreparedGpuReadoutGraph)

Return metadata retained at the last successful GPU readout publication.
"""
function output_metadata(owner::PreparedGpuReadoutGraph)
    _assert_gpu_readout_open(owner)
    return NamedTuple{keys(owner.published_metadata)}(
        _metadata_values(values(owner.published_metadata)),
    )
end

"""
    gpu_progressive_counters(owner::PreparedGpuReadoutGraph)

Return a cold diagnostic snapshot of block submissions, terminal waits and
prepared capacities. This call allocates; use it outside processing callbacks.
It reads retained host bookkeeping without GPU operations, including after a
backend fault, failed drain or successful `close`; submission counts do not
establish GPU completion. Serialize this query with processing, reset and close
on the owner. It does not make a faulted owner admissible for processing.
"""
function gpu_progressive_counters(owner::PreparedGpuReadoutGraph)
    return Dict(owner.graph.nodes[2].name => _gpu_readout_counters(owner.backend))
end

"""
    close(owner::PreparedGpuReadoutGraph)

Drain every owned CUDA stream before releasing lifetime roots. On a failed
drain the owner stays rooted and may be closed again; its buffers are not reused.
"""
function Base.close(owner::PreparedGpuReadoutGraph)
    owner.closed && return nothing
    _drain_gpu_readout!(owner.backend)
    owner.closed = true
    lock(_gpu_progressive_roots_lock) do
        delete!(_gpu_progressive_roots, owner)
    end
    return nothing
end
