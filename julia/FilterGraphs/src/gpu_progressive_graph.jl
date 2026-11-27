# A focused execution adapter: only incremental dense reconstruction moves to
# the device. Checked Graph traversal and all other Algorithms remain unchanged.
function _prepare_gpu_dense_executor end
function _submit_gpu_dense_block! end
function _finish_gpu_dense! end
function _clear_gpu_dense! end
function _replay_gpu_dense! end
function _drain_gpu_dense! end
# Snapshot retained host bookkeeping only, even after a fault or failed drain.
# Implementations must not launch, wait for, or dereference device storage.
function _gpu_dense_counters end

_prepare_gpu_dense_executor(prepared, device, lanes) = throw(
    GraphConfigurationError(
        "progressive dense reconstruction is not supported on $(typeof(device))",
    ),
)

mutable struct GpuDenseWorkspace{S,E}
    source::S
    executor::E
    synchronous::Bool
    fallback_frames::UInt64
end

mutable struct PreparedGpuDenseReconstructor{P,W,F}
    plan::P
    workspace::W
    formats::F
end

function commit!(workspace::GpuDenseWorkspace)
    _clear_gpu_dense!(workspace.executor)
    commit!(workspace.source)
    workspace.synchronous = false
    return workspace
end

function discard!(workspace::GpuDenseWorkspace)
    _clear_gpu_dense!(workspace.executor)
    discard!(workspace.source)
    workspace.synchronous = false
    return workspace
end

function reset!(workspace::GpuDenseWorkspace)
    _clear_gpu_dense!(workspace.executor)
    reset!(workspace.source)
    workspace.synchronous = false
    return workspace
end

function FilterGraphAlgorithms._plan_adopted!(workspace::GpuDenseWorkspace)
    # The owner drains before entering parameter adoption. This hook performs
    # no backend operation once that precondition has been established.
    _clear_gpu_dense!(workspace.executor)
    FilterGraphAlgorithms._plan_adopted!(workspace.source)
    workspace.synchronous = false
    return workspace
end

function reset!(prepared::PreparedGpuDenseReconstructor)
    reset!(prepared.workspace)
    return prepared
end

function FilterGraphAlgorithms._prepare_parameter_replacement(
    prepared::PreparedGpuDenseReconstructor,
    plan,
    name::Val{:reconstructor},
    values::AbstractArray,
)
    source = FilterGraphAlgorithms.PreparedAlgorithm(
        FilterGraphAlgorithms.IncrementalDenseReconstructorF32,
        plan,
        prepared.workspace.source,
        prepared.formats,
    )
    return FilterGraphAlgorithms._prepare_parameter_replacement(source, plan, name, values)
end

# Preparation certifies a single symmetric input range for every partial sum.
# Sum column maxima bounds every row, every column order, and every partition.
function _gpu_dense_measurement_limit(matrix::Matrix{Float32})
    norm_bound = 0.0
    for column in axes(matrix, 2)
        maximum_coefficient = 0.0
        for row in axes(matrix, 1)
            maximum_coefficient =
                max(maximum_coefficient, abs(Float64(matrix[row, column])))
        end
        norm_bound = nextfloat(norm_bound + maximum_coefficient)
    end
    # Bound multiplication, lane accumulation, reduction, and block adoption
    # rounding. Unreasonably long dot products use the checked path instead.
    rounding = (size(matrix, 2) + 64.0) * eps(Float32)
    rounding < 0.5 || return 0.0f0
    limit = Float64(floatmax(Float32)) * (1 - rounding) / (4 * norm_bound)
    limit >= floatmax(Float32) && return floatmax(Float32)
    return max(0.0f0, prevfloat(Float32(limit)))
end

function _gpu_dense_serial_block!(candidate, matrix, measurements, columns, count)
    FGA = FilterGraphAlgorithms
    for index = 1:count
        FGA._accumulate_incremental_dense_reconstruction_rows!(
            candidate,
            matrix,
            Int(columns[index]) + 1,
            measurements[index],
            eachindex(candidate),
        )
    end
    return nothing
end

function FilterGraphAlgorithms._process_admitted!(
    reconstructed::AbstractArray,
    prepared::PreparedGpuDenseReconstructor,
    measurements::AbstractArray,
    columns::AbstractArray,
    count::AbstractArray,
    measurement_metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
    ::FilterGraphAlgorithms.AdmittedSampleMetadata,
    ::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    FGA = FilterGraphAlgorithms
    state = prepared.workspace
    source, executor = state.source, state.executor
    sample = FGA._sample_metadata(measurement_metadata)
    isnothing(sample) && return FGA.IncrementalDenseReconstructionMetadataMissing
    FGA._validate_incremental_dense_reconstruction_storage(
        reconstructed,
        measurements,
        columns,
        count,
        source,
        prepared.plan,
    )
    UInt64(count[1]) <= UInt64(length(measurements)) ||
        return FGA.IncrementalDenseReconstructionCountOutOfRange
    # A new frame or an invalid continuation can cause the source to clear its
    # candidate. Join the old DMA/compute work before any staging is reused.
    if sample.corrupted ||
       iszero(sample.offset) ||
       !source.active ||
       source.sequence != sample.sequence ||
       sample.discontinuity ||
       sample.offset <= source.last_offset
        _clear_gpu_dense!(executor)
        state.synchronous = false
    end
    FGA._accept_incremental_dense_reconstruction_block!(source, sample) ||
        return FGA.AlgorithmProcessDeferred()

    block_count = Int(count[1])
    safe = true
    for index = 1:block_count
        column_offset = columns[index]
        UInt64(column_offset) < UInt64(length(source.seen)) ||
            return FGA.IncrementalDenseReconstructionColumnOutOfRange
        column = Int(column_offset) + 1
        source.seen[column] && return FGA.IncrementalDenseReconstructionDuplicateColumn
        measurement = measurements[index]
        isfinite(measurement) || return FGA.IncrementalDenseReconstructionNonFiniteInput
        safe &= abs(measurement) <= executor.measurement_limit
        source.seen[column] = true
        source.seen_count += 1
    end
    # More nonempty callbacks than the declared rate is legal in the checked
    # Algorithm. Fall back instead of growing storage or changing its result.
    if !state.synchronous &&
       (!safe || !_submit_gpu_dense_block!(executor, measurements, columns, block_count))
        _replay_gpu_dense!(source.candidate_reconstructed, executor, prepared.plan.matrix)
        state.synchronous = true
        state.fallback_frames += UInt64(1)
    end
    if state.synchronous
        _gpu_dense_serial_block!(
            source.candidate_reconstructed,
            prepared.plan.matrix,
            measurements,
            columns,
            block_count,
        )
        all(isfinite, source.candidate_reconstructed) ||
            return FGA.IncrementalDenseReconstructionNonFiniteOutput
    end
    sample.terminal || return FGA.AlgorithmProcessDeferred()
    source.seen_count == length(source.seen) ||
        return FGA.IncrementalDenseReconstructionIncompleteSample
    if !state.synchronous
        _finish_gpu_dense!(source.candidate_reconstructed, executor)
    end
    copyto!(reconstructed, source.candidate_reconstructed)
    source.terminal_candidate = true
    return FGA.AlgorithmProcessComplete(
        FGA.SampleMetadata(
            sample;
            offset = 0,
            terminal = true,
            discontinuity = source.current_discontinuity,
        ),
    )
end

@inline _gpu_dense_workspace(::Any) = nothing
@inline _gpu_dense_workspace(
    node::PreparedNode{D,<:PreparedGpuDenseReconstructor},
) where {D} = node.prepared.workspace

_gpu_progressive_node(node, device, lanes) = node
function _gpu_progressive_node(
    node::PreparedNode{FilterGraphAlgorithms.IncrementalDenseReconstructorF32},
    device,
    lanes,
)
    source = node.prepared
    workspace = GpuDenseWorkspace(
        source.workspace,
        _prepare_gpu_dense_executor(source, device, lanes),
        false,
        UInt64(0),
    )
    prepared = PreparedGpuDenseReconstructor(source.plan, workspace, source.formats)
    return PreparedNode(
        FilterGraphAlgorithms.IncrementalDenseReconstructorF32,
        node.name,
        prepared,
        node.input_names,
        node.output_names,
        node.parameter_names,
        node.input_formats,
        node.output_formats,
        node.parameter_formats,
        node.output_conditional,
        node.output_metadata_sources,
    )
end

"""
An ordered checked Graph with one asynchronous GPU incremental reconstructor.

Upstream row/region measurement and downstream controller Nodes execute on the
CPU. Only one logical frame is admitted. Serialize all owner operations and
call `close` after use; an abandoned owner retains its buffers for DMA safety.
Construct with [`prepare_gpu_progressive_graph`](@ref).
"""
mutable struct PreparedGpuProgressiveGraph{G}
    graph::G
    closed::Bool
end

"""
    isopen(owner::PreparedGpuProgressiveGraph) -> Bool

Return whether the GPU progressive owner's resources remain open. A faulted
owner remains open until a successful `close`, but cannot process or reset.
Serialize this query with other owner operations.
"""
Base.isopen(owner::PreparedGpuProgressiveGraph) = !owner.closed

const _gpu_progressive_roots = IdDict{Any,Nothing}()
const _gpu_progressive_roots_lock = ReentrantLock()

"""
    prepare_gpu_progressive_graph(path, device; algorithms, lanes=128)

Prepare a checked row/region Graph with exactly one
`IncrementalDenseReconstructorF32` offloaded to CUDA. The bounded staging
capacity follows the declared block capacity and block count. Each nonempty
block is copied before return and enqueued without a host completion wait;
terminal processing waits once, then the ordinary Graph runs its CPU suffix.
Empty blocks consume no device slot. Additional nonempty blocks or inputs
outside a preparation-certified finite range use checked serial reconstruction.

GPU reduction changes floating-point summation order. Qualify the complete
controller with an application-specific numerical tolerance; bitwise CPU
agreement and identical decisions at arbitrary numerical thresholds are not
promised. Other Algorithm checks and node-local adoption semantics are retained.
The current implementation supports CUDA only. `lanes` is fixed before capture
and must be 32, 64, 128, or 256. Call `close` in `finally` after use.
"""
function prepare_gpu_progressive_graph(
    path::AbstractString,
    device;
    algorithms::Tuple,
    lanes::Integer = 128,
)
    lanes in (32, 64, 128, 256) || throw(ArgumentError("lanes must be 32, 64, 128, or 256"))
    graph = prepare_graph(path; algorithms)
    count(
        node ->
            node isa PreparedNode{FilterGraphAlgorithms.IncrementalDenseReconstructorF32},
        graph.nodes,
    ) == 1 || throw(
        GraphConfigurationError(
            "GPU progressive execution requires exactly one incremental dense reconstructor",
        ),
    )
    nodes = map(node -> _gpu_progressive_node(node, device, Int(lanes)), graph.nodes)
    prepared = PreparedGraph(
        nodes,
        graph.node_failure_types,
        graph.node_outputs,
        graph.node_output_available,
        graph.node_output_metadata,
        graph.node_inflight,
        graph.input_routes,
        graph.parameter_routes,
        graph.output_routes,
        graph.input_formats,
        graph.parameter_formats,
        graph.output_formats,
        graph.output_conditional,
        graph.published_metadata,
        graph.default_input_metadata,
        graph.result_types,
    )
    owner = PreparedGpuProgressiveGraph(prepared, false)
    lock(_gpu_progressive_roots_lock) do
        _gpu_progressive_roots[owner] = nothing
    end
    return owner
end

function _assert_gpu_progressive_open(owner::PreparedGpuProgressiveGraph)
    owner.closed && throw(ArgumentError("GPU progressive Graph is closed"))
    _assert_gpu_nodes_open(owner.graph.nodes)
    return nothing
end

@inline _assert_gpu_node_open(::Any) = nothing
@inline function _assert_gpu_node_open(
    node::PreparedNode{D,<:PreparedGpuDenseReconstructor},
) where {D}
    node.prepared.workspace.executor.faulted &&
        error("GPU progressive execution faulted; close and prepare a new owner")
    return nothing
end
@inline _assert_gpu_nodes_open(::Tuple{}) = nothing
@inline function _assert_gpu_nodes_open(nodes::Tuple)
    _assert_gpu_node_open(first(nodes))
    return _assert_gpu_nodes_open(Base.tail(nodes))
end

function process!(
    outputs::NamedTuple,
    owner::PreparedGpuProgressiveGraph,
    inputs::NamedTuple,
    metadata::NamedTuple,
)
    _assert_gpu_progressive_open(owner)
    return process!(outputs, owner.graph, inputs, metadata)
end

function process!(
    outputs::NamedTuple,
    owner::PreparedGpuProgressiveGraph,
    inputs::NamedTuple,
)
    _assert_gpu_progressive_open(owner)
    return process!(outputs, owner.graph, inputs)
end

function reset!(owner::PreparedGpuProgressiveGraph)
    _assert_gpu_progressive_open(owner)
    reset!(owner.graph)
    return owner
end

function output_metadata(owner::PreparedGpuProgressiveGraph)
    _assert_gpu_progressive_open(owner)
    return output_metadata(owner.graph)
end

function _parameter_candidate(
    node::PreparedNode{D,<:PreparedGpuDenseReconstructor},
    plan,
    touched::Bool,
) where {D}
    touched || return nothing
    source = FilterGraphAlgorithms.PreparedAlgorithm(
        FilterGraphAlgorithms.IncrementalDenseReconstructorF32,
        plan,
        node.prepared.workspace.source,
        node.prepared.formats,
    )
    executor = node.prepared.workspace.executor
    return _prepare_gpu_dense_executor(source, executor.context.device, executor.lanes)
end

_adopt_parameter_candidate!(
    ::PreparedNode{D,<:PreparedGpuDenseReconstructor},
    ::Nothing,
) where {D} = nothing

function _adopt_parameter_candidate!(
    node::PreparedNode{D,<:PreparedGpuDenseReconstructor},
    executor,
) where {D}
    node.prepared.workspace.executor = executor
    return nothing
end

function replace_parameters!(owner::PreparedGpuProgressiveGraph, updates::Pair...)
    _assert_gpu_progressive_open(owner)
    transaction = _prepare_parameter_transaction(owner.graph, updates...)
    # All fallible CUDA work precedes plan adoption. Adoption only clears
    # already-drained bookkeeping and swaps prepared objects.
    for node in owner.graph.nodes
        workspace = _gpu_dense_workspace(node)
        isnothing(workspace) || _drain_gpu_dense!(workspace.executor)
    end
    _adopt_parameter_transaction!(owner.graph, transaction)
    return owner
end

"""
    gpu_progressive_counters(owner)

Return cold diagnostic snapshots by reconstructor Node, including the
preparation-certified measurement limit and serial fallback count. This call
allocates and belongs outside timed callbacks. It reads retained host
bookkeeping without GPU operations, including after a backend fault, failed
drain or successful `close`; submission counts do not establish GPU completion.
Serialize this query with processing, reset, parameter replacement and close
on the owner. It does not make a faulted owner admissible for processing.
"""
function gpu_progressive_counters(owner::PreparedGpuProgressiveGraph)
    result = Dict{Symbol,NamedTuple}()
    for node in owner.graph.nodes
        workspace = _gpu_dense_workspace(node)
        isnothing(workspace) && continue
        result[node.name] = merge(
            _gpu_dense_counters(workspace.executor),
            (; fallback_frames = workspace.fallback_frames),
        )
    end
    return result
end

"""
    close(owner::PreparedGpuProgressiveGraph)

Drain every owned GPU stream before releasing retained storage. A failed drain
retains the owner for a later close attempt; it does not permit buffer reuse.
"""
function Base.close(owner::PreparedGpuProgressiveGraph)
    owner.closed && return nothing
    for node in owner.graph.nodes
        workspace = _gpu_dense_workspace(node)
        isnothing(workspace) || _drain_gpu_dense!(workspace.executor)
    end
    owner.closed = true
    lock(_gpu_progressive_roots_lock) do
        delete!(_gpu_progressive_roots, owner)
    end
    return nothing
end
