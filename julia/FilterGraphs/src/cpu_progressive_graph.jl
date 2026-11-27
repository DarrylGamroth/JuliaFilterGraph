# The first CPU overlap adapter is deliberately specific to the progressive
# dense reconstructor. The Graph retains Node ordering and publication rules;
# workers own disjoint row views and join before publishing the candidate.

struct CpuSharedMatrix end

struct CpuShardedMatrix{T}
    shards::Vector{Matrix{T}}
end

# Julia permits throwing arbitrary values. Keep the exact payload in a typed,
# cold wrapper, and rethrow its cause at the coordinator boundary.
struct CpuWorkerFailure{C,B}
    cause::C
    backtrace::B
end

struct CpuChannelHandoff
    requests::Vector{Channel{Bool}}
    completions::Vector{Channel{Union{Nothing,CpuWorkerFailure}}}
end

mutable struct CpuAtomicHandoff
    # Request and completion cells have separate, padded cache lines. The
    # coordinator publishes one frame epoch at a time; each worker publishes
    # only its own completion epoch.
    requests::AtomicMemory{UInt64}
    completions::AtomicMemory{UInt64}
    errors::Vector{Union{Nothing,CpuWorkerFailure}}
    shutdown::Threads.Atomic{Bool}
    epoch::UInt64
end

"""Fixed storage for one worker timestamp at one MVM column per frame."""
struct CpuProgressiveMvmTrace
    item::Int
    capacity::Int
    workers::Int
    base_handoff_epoch::UInt64
    handoff_epochs::Vector{UInt64}
    timestamps_ns::Vector{UInt64}
    worker_terminal_timestamps_ns::Vector{UInt64}
    coordinator_wait_start_ns::Vector{UInt64}
    coordinator_wait_end_ns::Vector{UInt64}
end

mutable struct CpuDenseRowExecutor{
    T<:Union{Float32,Float64},
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    S<:AbstractVector{T},
    L,
    H,
    TaskType,
}
    candidate::V
    scratch::Vector{S}
    row_ranges::Vector{UnitRange{Int}}
    measurements::Vector{T}
    columns::Vector{UInt32}
    frame_capacity::Int
    matrix::M
    matrix_layout::L
    column_bounds::Vector{Float64}
    shadow_column_bounds::Vector{Float64}
    safe_bound::Float64
    synchronous::Bool
    published::Threads.Atomic{Int}
    terminal::Threads.Atomic{Int}
    # Each worker writes one progress cell, spaced by two cache lines.
    progress::AtomicMemory{Int}
    handoff::H
    handoff_epoch::UInt64
    trace::Union{Nothing,CpuProgressiveMvmTrace}
    tasks::Vector{TaskType}
    worker_thread_ids::Union{Nothing,Vector{Int}}
    spin_between_blocks::Bool
    active::Bool
    closed::Bool
    fault::Union{Nothing,CpuWorkerFailure}
    delayed_nonfinite_outputs::UInt64
    last_failed_sequence::UInt64
    preterminal_frames::UInt64
    preterminal_columns::UInt64
    numerical_fallbacks::UInt64
    direct_columns::UInt64
    copied_columns::UInt64
end

function _cpu_pack_dense_shard(matrix::Matrix{T}, rows::UnitRange{Int}) where {T}
    shard = Matrix{T}(undef, length(rows), size(matrix, 2))
    first_row = first(rows)
    @inbounds for column in axes(matrix, 2), local_row in axes(shard, 1)
        shard[local_row, column] = matrix[first_row + local_row - 1, column]
    end
    return shard
end

function _cpu_dense_matrix_layout(
    ::Val{:sharded}, matrix::Matrix{T}, ranges::Vector{UnitRange{Int}},
    worker_thread_ids::Union{Nothing,Vector{Int}},
) where {T}
    tasks = map(eachindex(ranges)) do index
        rows = ranges[index]
        if isnothing(worker_thread_ids)
            Threads.@spawn _cpu_pack_dense_shard(matrix, rows)
        else
            ThreadPinning.@spawnat worker_thread_ids[index] _cpu_pack_dense_shard(matrix, rows)
        end
    end
    return CpuShardedMatrix{T}(Matrix{T}[fetch(task) for task in tasks])
end

_cpu_dense_matrix_layout(
    ::Val{:shared}, matrix::Matrix, ranges::Vector{UnitRange{Int}},
    worker_thread_ids::Union{Nothing,Vector{Int}},
) = CpuSharedMatrix()

_cpu_dense_replacement_layout(
    ::CpuSharedMatrix, matrix::Matrix, ranges::Vector{UnitRange{Int}},
    worker_thread_ids::Union{Nothing,Vector{Int}},
) = CpuSharedMatrix()

_cpu_dense_replacement_layout(
    ::CpuShardedMatrix, matrix::Matrix, ranges::Vector{UnitRange{Int}},
    worker_thread_ids::Union{Nothing,Vector{Int}},
) = _cpu_dense_matrix_layout(Val(:sharded), matrix, ranges, worker_thread_ids)

@inline _cpu_dense_worker_matrix(
    executor::CpuDenseRowExecutor{T,V,M,S,CpuSharedMatrix},
    index::Int, rows::UnitRange{Int},
) where {T,V,M,S} = (executor.matrix, first(rows))

@inline _cpu_dense_worker_matrix(
    executor::CpuDenseRowExecutor{T,V,M,S,<:CpuShardedMatrix},
    index::Int, rows::UnitRange{Int},
) where {T,V,M,S} = (executor.matrix_layout.shards[index], 1)

function _cpu_row_ranges(candidate::AbstractVector, workers::Int)
    rows = length(candidate)
    return [
        (fld((index - 1) * rows, workers) + 1):fld(index * rows, workers)
        for index in 1:workers
    ]
end

const _CPU_PROGRESS_STRIDE = 16

@inline _cpu_progress_index(index::Int) = (index - 1) * _CPU_PROGRESS_STRIDE + 1
@inline _cpu_progress(executor::CpuDenseRowExecutor, index::Int) =
    @atomic :acquire executor.progress[_cpu_progress_index(index)]

@inline function _cpu_trace_index(
    trace::CpuProgressiveMvmTrace, ordinal::Int, worker::Int,
)
    return ((ordinal - 1) * trace.workers + worker - 1) * _CPU_PROGRESS_STRIDE + 1
end

function CpuProgressiveMvmTrace(
    item::Int, capacity::Int, workers::Int, base_handoff_epoch::UInt64,
)
    item > 0 || throw(ArgumentError("CPU MVM trace item must be positive"))
    capacity > 0 || throw(ArgumentError("CPU MVM trace capacity must be positive"))
    slots = capacity * workers * _CPU_PROGRESS_STRIDE
    return CpuProgressiveMvmTrace(
        item,
        capacity,
        workers,
        base_handoff_epoch,
        zeros(UInt64, slots),
        zeros(UInt64, slots),
        zeros(UInt64, slots),
        zeros(UInt64, capacity),
        zeros(UInt64, capacity),
    )
end

@inline function _cpu_record_mvm_trace!(
    ::Nothing, worker::Int, epoch::UInt64, item::Int,
)
    return nothing
end

@inline _cpu_trace_ordinal(::Nothing, epoch::UInt64) = 0
@inline function _cpu_trace_ordinal(trace::CpuProgressiveMvmTrace, epoch::UInt64)
    epoch > trace.base_handoff_epoch || return 0
    ordinal = Int(epoch - trace.base_handoff_epoch)
    return ordinal <= trace.capacity ? ordinal : 0
end

@inline function _cpu_record_worker_terminal_trace!(
    ::Nothing, worker::Int, epoch::UInt64,
)
    return nothing
end

@inline function _cpu_record_worker_terminal_trace!(
    trace::CpuProgressiveMvmTrace, worker::Int, epoch::UInt64,
)
    ordinal = _cpu_trace_ordinal(trace, epoch)
    iszero(ordinal) && return nothing
    slot = _cpu_trace_index(trace, ordinal, worker)
    @inbounds trace.handoff_epochs[slot] = epoch
    @inbounds trace.worker_terminal_timestamps_ns[slot] = time_ns()
    return nothing
end

@inline function _cpu_record_coordinator_wait_start!(
    ::Nothing, epoch::UInt64,
)
    return nothing
end

@inline function _cpu_record_coordinator_wait_start!(
    trace::CpuProgressiveMvmTrace, epoch::UInt64,
)
    ordinal = _cpu_trace_ordinal(trace, epoch)
    iszero(ordinal) && return nothing
    @inbounds trace.coordinator_wait_start_ns[ordinal] = time_ns()
    return nothing
end

@inline function _cpu_record_coordinator_wait_end!(
    ::Nothing, epoch::UInt64,
)
    return nothing
end

@inline function _cpu_record_coordinator_wait_end!(
    trace::CpuProgressiveMvmTrace, epoch::UInt64,
)
    ordinal = _cpu_trace_ordinal(trace, epoch)
    iszero(ordinal) && return nothing
    @inbounds trace.coordinator_wait_end_ns[ordinal] = time_ns()
    return nothing
end

@inline function _cpu_record_mvm_trace!(
    trace::CpuProgressiveMvmTrace, worker::Int, epoch::UInt64, item::Int,
)
    item == trace.item || return nothing
    epoch > trace.base_handoff_epoch || return nothing
    ordinal = Int(epoch - trace.base_handoff_epoch)
    ordinal <= trace.capacity || return nothing
    slot = _cpu_trace_index(trace, ordinal, worker)
    @inbounds trace.handoff_epochs[slot] = epoch
    @inbounds trace.timestamps_ns[slot] = time_ns()
    return nothing
end

mutable struct CpuFrameOutputView{T,E<:CpuDenseRowExecutor} <: AbstractVector{T}
    executor::E
    storage::Vector{T}
    capacity::Int
    first_item::Int
end

@inline _cpu_direct_window(
    executor::CpuDenseRowExecutor,
    measurements::AbstractVector,
    columns::AbstractVector,
    first_item::Int,
) = false

@inline function _cpu_direct_window(
    executor::CpuDenseRowExecutor,
    measurements::CpuFrameOutputView,
    columns::CpuFrameOutputView,
    first_item::Int,
)
    return measurements.executor === executor && columns.executor === executor &&
        measurements.first_item == first_item && columns.first_item == first_item
end

@inline function _copy_cpu_frame_values!(
    destination::Vector{T}, first_item::Int,
    source::AbstractVector{T}, count::Int, executor::CpuDenseRowExecutor,
) where {T}
    copyto!(destination, first_item, source, 1, count)
    return nothing
end

@inline function _copy_cpu_frame_values!(
    destination::Vector{T}, first_item::Int,
    source::CpuFrameOutputView{T}, count::Int, executor::CpuDenseRowExecutor,
) where {T}
    if source.executor === executor
        copyto!(destination, first_item, source.storage, source.first_item, count)
    else
        copyto!(destination, first_item, source, 1, count)
    end
    return nothing
end

Base.IndexStyle(::Type{<:CpuFrameOutputView}) = IndexLinear()
Base.size(view::CpuFrameOutputView) = (view.capacity,)
Base.dataids(view::CpuFrameOutputView) = Base.dataids(view.storage)

@inline function Base.getindex(view::CpuFrameOutputView, index::Int)
    @boundscheck checkbounds(view, index)
    return @inbounds view.storage[view.first_item + index - 1]
end

@inline function Base.setindex!(view::CpuFrameOutputView{T}, value, index::Int) where {T}
    @boundscheck checkbounds(view, index)
    @inbounds view.storage[view.first_item + index - 1] = convert(T, value)
    return value
end

@inline function _prepare_node_output!(view::CpuFrameOutputView, routed_metadata)
    executor = view.executor
    sample = FilterGraphAlgorithms._sample_metadata(first(routed_metadata))
    # A new frame can arrive before the preceding one has finished. The old
    # frame still owns the log prefix until the reconstructor joins it. Stage
    # the first new block after the entire log; its view remains unchanged if
    # the reconstructor has to copy that block back to offset one.
    restarting = executor.active && !isnothing(sample) && iszero(sample.offset)
    view.first_item = restarting ? executor.frame_capacity + 1 :
        (executor.active ? executor.published[] + 1 : 1)
    return nothing
end

@inline _cpu_mailbox_index(index::Int) = index * _CPU_PROGRESS_STRIDE + 1

function _cpu_handoff(::Nothing, workers::Int)
    return CpuChannelHandoff(
        [Channel{Bool}(1) for _ in 1:workers],
        [Channel{Union{Nothing,CpuWorkerFailure}}(1) for _ in 1:workers],
    )
end

function _cpu_handoff(::Vector{Int}, workers::Int)
    capacity = (workers + 2) * _CPU_PROGRESS_STRIDE
    requests = AtomicMemory{UInt64}(undef, capacity)
    completions = AtomicMemory{UInt64}(undef, capacity)
    for index in 1:workers
        @atomic :release requests[_cpu_mailbox_index(index)] = UInt64(0)
        @atomic :release completions[_cpu_mailbox_index(index)] = UInt64(0)
    end
    errors = Vector{Union{Nothing,CpuWorkerFailure}}(undef, workers)
    fill!(errors, nothing)
    return CpuAtomicHandoff(
        requests, completions, errors, Threads.Atomic{Bool}(false), UInt64(0),
    )
end

@inline function _cpu_cooperate!(polls::Int)
    ccall(:jl_cpu_pause, Cvoid, ())
    polls += 1
    if polls == 256
        GC.safepoint()
        yield()
        return 0
    end
    return polls
end

@inline function _cpu_wait_request!(
    handoff::CpuChannelHandoff, index::Int, previous::UInt64,
    ::CpuDenseRowExecutor,
)
    return take!(handoff.requests[index]) ? previous + UInt64(1) : UInt64(0)
end

function _cpu_wait_request!(
    handoff::CpuAtomicHandoff, index::Int, previous::UInt64,
    ::CpuDenseRowExecutor,
)
    next = previous + UInt64(1)
    polls = 0
    while true
        handoff.shutdown[] && return UInt64(0)
        request = @atomic :acquire handoff.requests[_cpu_mailbox_index(index)]
        request == next && return next
        request > next && error("CPU worker request epoch advanced without completion")
        polls = _cpu_cooperate!(polls)
    end
end

function _cpu_start_handoff!(handoff::CpuChannelHandoff)
    for request in handoff.requests
        put!(request, true)
    end
    return nothing
end

function _cpu_start_handoff!(handoff::CpuAtomicHandoff)
    handoff.epoch < typemax(UInt64) || error("CPU frame epoch exhausted")
    handoff.epoch += UInt64(1)
    fill!(handoff.errors, nothing)
    for index in eachindex(handoff.errors)
        @atomic :release handoff.requests[_cpu_mailbox_index(index)] = handoff.epoch
    end
    return nothing
end

function _cpu_start_frame_handoff!(executor::CpuDenseRowExecutor)
    executor.handoff_epoch < typemax(UInt64) || error("CPU frame epoch exhausted")
    executor.handoff_epoch += UInt64(1)
    _cpu_start_handoff!(executor.handoff)
    return executor.handoff_epoch
end

@inline function _cpu_publish_completion!(
    handoff::CpuChannelHandoff, index::Int, ::UInt64, result,
)
    put!(handoff.completions[index], result)
    return nothing
end

@inline function _cpu_publish_completion!(
    handoff::CpuAtomicHandoff, index::Int, epoch::UInt64, result,
)
    if !isnothing(result)
        @inbounds handoff.errors[index] = result
    end
    @atomic :release handoff.completions[_cpu_mailbox_index(index)] = epoch
    return nothing
end

@inline _cpu_wait_completion!(
    handoff::CpuChannelHandoff, index::Int, ::CpuDenseRowExecutor,
) = take!(handoff.completions[index])

function _cpu_wait_completion!(
    handoff::CpuAtomicHandoff, index::Int, executor::CpuDenseRowExecutor,
)
    polls = 0
    while true
        completed = @atomic :acquire handoff.completions[_cpu_mailbox_index(index)]
        completed == handoff.epoch && return @inbounds handoff.errors[index]
        completed > handoff.epoch && error("CPU worker completion epoch is ahead of frame")
        polls = _cpu_cooperate!(polls)
        if polls == 0 && istaskdone(executor.tasks[index])
            fetch(executor.tasks[index])
            error("CPU worker stopped before completing its frame")
        end
    end
end

function _cpu_close_handoff!(handoff::CpuChannelHandoff)
    for request in handoff.requests
        put!(request, false)
    end
    return nothing
end

function _cpu_close_handoff!(handoff::CpuAtomicHandoff)
    handoff.shutdown[] = true
    return nothing
end

function _cpu_dense_row_worker!(executor::CpuDenseRowExecutor, index::Int, rows::UnitRange{Int})
    FGA = FilterGraphAlgorithms
    handoff = executor.handoff
    scratch = executor.scratch[index]
    epoch = UInt64(0)
    while true
        epoch = _cpu_wait_request!(handoff, index, epoch, executor)
        iszero(epoch) && break
        result = try
            worker_matrix, first_row = _cpu_dense_worker_matrix(executor, index, rows)
            consumed = 0
            empty_polls = 0
            while true
                state = executor.terminal[]
                state == 2 && break
                # A terminal store follows the final published-prefix store.
                # Read it first so a worker cannot see terminal with an older
                # prefix and retire before consuming the last block.
                available = executor.published[]
                if consumed < available
                    first_item = consumed + 1
                    trace = executor.trace
                    if !isnothing(trace) && first_item <= trace.item <= available
                        # Finish the traced column before accumulating any later item.
                        FGA._accumulate_incremental_dense_reconstruction_columns!(
                            scratch, worker_matrix, executor.measurements, executor.columns,
                            first_item:trace.item, first_row - 1,
                        )
                        _cpu_record_mvm_trace!(trace, index, epoch, trace.item)
                        first_item = trace.item + 1
                    end
                    FGA._accumulate_incremental_dense_reconstruction_columns!(
                        scratch, worker_matrix, executor.measurements, executor.columns,
                        first_item:available, first_row - 1,
                    )
                    consumed = available
                    @atomic :release executor.progress[_cpu_progress_index(index)] = consumed
                    empty_polls = 0
                elseif state == 1
                    break
                else
                    # A sticky task has its own configured Julia thread. A
                    # movable task must yield so one-thread operation can run.
                    if executor.spin_between_blocks
                        # A selected Julia thread may host other Graphs or
                        # the caller. Cooperate at the same bounded interval
                        # as the request and completion mailbox waits.
                        empty_polls = _cpu_cooperate!(empty_polls)
                    else
                        yield()
                    end
                end
            end
            nothing
        catch error
            CpuWorkerFailure(error, catch_backtrace())
        end
        _cpu_record_worker_terminal_trace!(executor.trace, index, epoch)
        _cpu_publish_completion!(handoff, index, epoch, result)
    end
    return nothing
end

_cpu_worker_tasks(::Nothing) = Task[]

function _cpu_worker_tasks(thread_ids::Vector{Int})
    # ThreadPinning returns its own typed task handle. Establish that handle
    # type during preparation without depending on StableTasks internals.
    task = ThreadPinning.@spawnat first(thread_ids) nothing
    fetch(task)
    return typeof(task)[]
end

function CpuDenseRowExecutor(
    prepared,
    workers::Int,
    worker_thread_ids::Union{Nothing,Vector{Int}},
    matrix_layout::Symbol,
)
    candidate = prepared.workspace.candidate_reconstructed
    matrix = prepared.plan.matrix
    rows = length(candidate)
    1 <= workers <= rows || throw(
        ArgumentError("CPU row worker count must be between 1 and $rows"),
    )
    capacity = length(prepared.workspace.seen)
    row_ranges = _cpu_row_ranges(candidate, workers)
    storage = _cpu_dense_matrix_layout(Val(matrix_layout), matrix, row_ranges, worker_thread_ids)
    # Each worker writes only inside its own view. A full cache line of unused
    # storage surrounds every active view, independent of allocation address.
    padding = cld(64, sizeof(eltype(candidate)))
    scratch = [
        view(
            zeros(eltype(candidate), length(rows) + 2padding),
            (padding + 1):(padding + length(rows)),
        ) for rows in row_ranges
    ]
    handoff = _cpu_handoff(worker_thread_ids, workers)
    progress = AtomicMemory{Int}(undef, workers * _CPU_PROGRESS_STRIDE)
    for index in 1:workers
        @atomic :release progress[_cpu_progress_index(index)] = 0
    end
    executor = CpuDenseRowExecutor(
        candidate,
        scratch,
        row_ranges,
        Vector{eltype(candidate)}(undef, capacity),
        Vector{UInt32}(undef, capacity),
        capacity,
        matrix,
        storage,
        zeros(Float64, size(matrix, 2)),
        zeros(Float64, size(matrix, 2)),
        0.0,
        false,
        Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0),
        progress,
        handoff,
        UInt64(0),
        nothing,
        _cpu_worker_tasks(worker_thread_ids),
        worker_thread_ids,
        !isnothing(worker_thread_ids),
        false,
        false,
        nothing,
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
    )
    _cpu_dense_column_bounds!(executor.column_bounds, matrix)
    try
        for index in 1:workers
            selected_rows = row_ranges[index]
            task = if isnothing(worker_thread_ids)
                Threads.@spawn _cpu_dense_row_worker!(
                    executor,
                    index,
                    selected_rows,
                )
            else
                ThreadPinning.@spawnat worker_thread_ids[index] _cpu_dense_row_worker!(
                    executor,
                    index,
                    selected_rows,
                )
            end
            push!(executor.tasks, task)
        end
    catch
        close(executor)
        rethrow()
    end
    return executor
end

function _cpu_dense_column_bounds!(bounds::Vector{Float64}, matrix)
    @inbounds for column in axes(matrix, 2)
        maximum_coefficient = 0.0
        for row in axes(matrix, 1)
            maximum_coefficient = max(
                maximum_coefficient,
                abs(Float64(matrix[row, column])),
            )
        end
        bounds[column] = maximum_coefficient
    end
    return nothing
end

function _cpu_dense_safe_block!(
    executor::CpuDenseRowExecutor{T},
    matrix,
    measurements,
    columns,
    count::Int,
) where {T}
    if executor.matrix !== matrix
        executor.active && error("CPU reconstructor matrix changed during a frame")
        _cpu_dense_column_bounds!(executor.column_bounds, matrix)
        executor.matrix_layout = _cpu_dense_replacement_layout(
            executor.matrix_layout, matrix, executor.row_ranges,
            executor.worker_thread_ids,
        )
        executor.matrix = matrix
    end
    # This bounds every output row's absolute sum. Round both operations upward
    # so Float64 rounding cannot understate the bound; the quarter-limit leaves
    # ample margin for rounding in the Float32 or Float64 accumulation itself.
    bound = executor.safe_bound
    limit = Float64(floatmax(T)) / 4
    @inbounds for index in 1:count
        column = Int(columns[index]) + 1
        term = nextfloat(
            abs(Float64(measurements[index])) * executor.column_bounds[column],
        )
        bound = nextfloat(bound + term)
        bound < limit || return false
    end
    executor.safe_bound = bound
    return true
end

function _join_cpu_dense_rows!(executor::CpuDenseRowExecutor; cancel::Bool=false)
    executor.active || return executor.fault
    executor.terminal[] = cancel ? 2 : 1
    _cpu_record_coordinator_wait_start!(executor.trace, executor.handoff_epoch)
    for index in eachindex(executor.row_ranges)
        result = _cpu_wait_completion!(executor.handoff, index, executor)
        if !isnothing(result) && isnothing(executor.fault)
            executor.fault = result
        end
    end
    _cpu_record_coordinator_wait_end!(executor.trace, executor.handoff_epoch)
    if !cancel && isnothing(executor.fault)
        for index in eachindex(executor.scratch)
            rows = executor.row_ranges[index]
            copyto!(
                executor.candidate, first(rows),
                executor.scratch[index], 1, length(rows),
            )
        end
    end
    executor.active = false
    return executor.fault
end

function _append_cpu_dense_rows!(
    executor::CpuDenseRowExecutor{T,V,M},
    matrix::M,
    measurements::AbstractVector{T},
    columns::AbstractVector{UInt32},
    count::Int,
) where {T,V,M}
    executor.closed && throw(ArgumentError("CPU progressive Graph is closed"))
    isnothing(executor.fault) || throw(executor.fault.cause)
    if !executor.active
        executor.matrix = matrix
        executor.published[] = 0
        executor.terminal[] = 0
        for part in executor.scratch
            fill!(part, zero(T))
        end
        for index in eachindex(executor.row_ranges)
            @atomic :release executor.progress[_cpu_progress_index(index)] = 0
        end
        _cpu_start_frame_handoff!(executor)
        executor.active = true
    end
    first_item = executor.published[] + 1
    last_item = first_item + count - 1
    last_item <= executor.frame_capacity ||
        error("CPU progressive frame storage capacity exceeded")
    direct = _cpu_direct_window(executor, measurements, columns, first_item)
    if !direct
        _copy_cpu_frame_values!(
            executor.measurements, first_item, measurements, count, executor,
        )
        _copy_cpu_frame_values!(
            executor.columns, first_item, columns, count, executor,
        )
        executor.copied_columns += UInt64(count)
    else
        executor.direct_columns += UInt64(count)
    end
    executor.published[] = last_item
    return nothing
end

@inline _cpu_node_output_port(::NodeOutput{NodeIndex,PortIndex}) where {
    NodeIndex,PortIndex,
} = (NodeIndex, PortIndex)
@inline _cpu_node_output_port(::Any) = nothing

@inline _cpu_direct_source(::PreparedNode{FilterGraphAlgorithms.PyramidReadoutMeasurementBlockF32}) = true
@inline _cpu_direct_source(::PreparedNode{FilterGraphAlgorithms.ShackHartmannReadoutMeasurementBlockF32}) = true
@inline _cpu_direct_source(::PreparedNode{FilterGraphAlgorithms.PyramidMeasurementBlockF32}) = true
@inline _cpu_direct_source(::PreparedNode{FilterGraphAlgorithms.ShackHartmannMeasurementBlockF32}) = true
@inline _cpu_direct_source(::Any) = false

@inline _cpu_dense_executor(::Any) = nothing

@inline _cpu_frame_views(measurements, columns, executor) = nothing

function _cpu_frame_views(
    measurements::Vector{Float32},
    columns::Vector{UInt32},
    executor::CpuDenseRowExecutor,
)
    capacity = length(measurements)
    resize!(executor.measurements, executor.frame_capacity + capacity)
    resize!(executor.columns, executor.frame_capacity + capacity)
    return (
        CpuFrameOutputView(executor, executor.measurements, capacity, 1),
        CpuFrameOutputView(executor, executor.columns, capacity, 1),
    )
end

function _bind_cpu_frame_outputs!(node_outputs::Vector{Any}, graph::PreparedGraph, nodes::Tuple)
    for (node_index, node) in pairs(nodes)
        executor = _cpu_dense_executor(node)
        isnothing(executor) && continue
        node.input_names == (:measurements, :columns, :count) || continue
        routes = graph.input_routes[node_index]
        measurement_port = _cpu_node_output_port(routes[1])
        column_port = _cpu_node_output_port(routes[2])
        isnothing(measurement_port) && continue
        isnothing(column_port) && continue
        source_index, measurement_index = measurement_port
        column_port[1] == source_index || continue
        _cpu_direct_source(nodes[source_index]) || continue
        source = nodes[source_index]
        source.output_names[measurement_index] === :measurements || continue
        source.output_names[column_port[2]] === :columns || continue
        outputs = node_outputs[source_index]
        measurements = outputs[measurement_index]
        columns = outputs[column_port[2]]
        # One producer has one frame owner. Shared producers retain the copy
        # path for later consumers, rather than giving two executors a slab.
        views = _cpu_frame_views(measurements, columns, executor)
        isnothing(views) && continue
        measurement_view, column_view = views
        outputs = Base.setindex(outputs, measurement_view, measurement_index)
        outputs = Base.setindex(outputs, column_view, column_port[2])
        node_outputs[source_index] = outputs
    end
    return Tuple(node_outputs)
end

function Base.close(executor::CpuDenseRowExecutor)
    executor.closed && return nothing
    failure = try
        _join_cpu_dense_rows!(executor; cancel=true)
        nothing
    catch error
        error
    end
    executor.closed = true
    _cpu_close_handoff!(executor.handoff)
    for task in executor.tasks
        try
            wait(task)
        catch error
            isnothing(failure) && (failure = error)
        end
    end
    executor.active = false
    isnothing(failure) || throw(failure)
    return nothing
end

struct CpuDenseWorkspace{W,E<:CpuDenseRowExecutor}
    source::W
    executor::E
end

function commit!(workspace::CpuDenseWorkspace)
    _join_cpu_dense_rows!(workspace.executor)
    commit!(workspace.source)
    return workspace
end

function discard!(workspace::CpuDenseWorkspace)
    _join_cpu_dense_rows!(workspace.executor; cancel=true)
    workspace.executor.synchronous = false
    workspace.executor.safe_bound = 0.0
    discard!(workspace.source)
    return workspace
end

function reset!(workspace::CpuDenseWorkspace)
    _join_cpu_dense_rows!(workspace.executor; cancel=true)
    workspace.executor.synchronous = false
    workspace.executor.safe_bound = 0.0
    reset!(workspace.source)
    workspace.executor.fault = nothing
    return workspace
end

function FilterGraphAlgorithms._plan_adopted!(workspace::CpuDenseWorkspace)
    _join_cpu_dense_rows!(workspace.executor; cancel=true)
    workspace.executor.synchronous = false
    workspace.executor.safe_bound = 0.0
    FilterGraphAlgorithms._plan_adopted!(workspace.source)
    return workspace
end

mutable struct PreparedCpuDenseReconstructor{P,S<:CpuDenseWorkspace,F}
    plan::P
    workspace::S
    formats::F
end

@inline function _cpu_dense_executor(node::PreparedNode{D,P}) where {
    D,P<:PreparedCpuDenseReconstructor,
}
    return node.prepared.workspace.executor
end

function FilterGraphAlgorithms._prepare_parameter_replacement(
    prepared::PreparedCpuDenseReconstructor,
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
    return FilterGraphAlgorithms._prepare_parameter_replacement(
        source,
        plan,
        name,
        values,
    )
end

function reset!(prepared::PreparedCpuDenseReconstructor)
    reset!(prepared.workspace)
    return prepared
end

function FilterGraphAlgorithms._process_admitted!(
    reconstructed::AbstractArray,
    prepared::PreparedCpuDenseReconstructor,
    measurements::AbstractArray,
    columns::AbstractArray,
    count::AbstractArray,
    measurement_metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
    ::FilterGraphAlgorithms.AdmittedSampleMetadata,
    ::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    FGA = FilterGraphAlgorithms
    workspace = prepared.workspace.source
    executor = prepared.workspace.executor
    sample = FGA._sample_metadata(measurement_metadata)
    isnothing(sample) && return FGA.IncrementalDenseReconstructionMetadataMissing
    FGA._validate_incremental_dense_reconstruction_storage(
        reconstructed,
        measurements,
        columns,
        count,
        workspace,
        prepared.plan,
    )
    UInt64(count[1]) <= UInt64(length(measurements)) ||
        return FGA.IncrementalDenseReconstructionCountOutOfRange

    # _accept_incremental_dense_reconstruction_block! may clear the candidate.
    # A superseding sample drains the prior frame first; invalid continuations
    # cancel outstanding work before the source Workspace can abandon it.
    if (executor.active || executor.synchronous) && (
        sample.corrupted || iszero(sample.offset) || !workspace.active ||
        !FGA._same_row_block_metadata(workspace.metadata,sample) || sample.discontinuity ||
        sample.offset <= workspace.last_offset
    )
        fault = _join_cpu_dense_rows!(executor; cancel=sample.corrupted || !iszero(sample.offset))
        isnothing(fault) || throw(fault.cause)
        executor.synchronous = false
        executor.safe_bound = 0.0
    end
    accepted = FGA._accept_incremental_dense_reconstruction_block!(workspace, sample)
    accepted || return FGA.AlgorithmProcessDeferred()
    if iszero(sample.offset)
        executor.safe_bound = 0.0
        executor.synchronous = false
    end

    block_count = Int(count[1])
    column_count = size(prepared.plan.matrix, 2)
    @inbounds for index in 1:block_count
        column_offset = columns[index]
        if UInt64(column_offset) >= UInt64(column_count)
            _join_cpu_dense_rows!(executor; cancel=true)
            FGA._abandon_incremental_dense_reconstruction!(workspace)
            return FGA.IncrementalDenseReconstructionColumnOutOfRange
        end
        column = Int(column_offset) + 1
        if workspace.seen[column]
            _join_cpu_dense_rows!(executor; cancel=true)
            FGA._abandon_incremental_dense_reconstruction!(workspace)
            return FGA.IncrementalDenseReconstructionDuplicateColumn
        end
        if !isfinite(measurements[index])
            _join_cpu_dense_rows!(executor; cancel=true)
            FGA._abandon_incremental_dense_reconstruction!(workspace)
            return FGA.IncrementalDenseReconstructionNonFiniteInput
        end
        workspace.seen[column] = true
        workspace.seen_count += 1
    end
    if sample.terminal && executor.active
        completed_before_terminal = minimum(
            _cpu_progress(executor, index) for index in eachindex(executor.row_ranges)
        )
        if completed_before_terminal > 0
            executor.preterminal_frames += UInt64(1)
            executor.preterminal_columns += UInt64(completed_before_terminal)
        end
    end
    safe = executor.synchronous || _cpu_dense_safe_block!(
        executor,
        prepared.plan.matrix,
        measurements,
        columns,
        block_count,
    )
    if executor.synchronous || !safe
        fault = _join_cpu_dense_rows!(executor)
        isnothing(fault) || throw(fault.cause)
        !executor.synchronous && (executor.numerical_fallbacks += UInt64(1))
        executor.synchronous = true
        candidate = workspace.candidate_reconstructed
        if Base.mightalias(columns, candidate)
            @inbounds for index in 1:block_count
                FGA._accumulate_incremental_dense_reconstruction_rows!(
                    candidate, prepared.plan.matrix, Int(columns[index]) + 1,
                    measurements[index], eachindex(candidate),
                )
            end
        else
            FGA._accumulate_incremental_dense_reconstruction_columns!(
                candidate, prepared.plan.matrix, measurements, columns, 1:block_count,
            )
        end
        for value in candidate
            if !isfinite(value)
                FGA._abandon_incremental_dense_reconstruction!(workspace)
                return FGA.IncrementalDenseReconstructionNonFiniteOutput
            end
        end
    else
        _append_cpu_dense_rows!(
            executor,
            prepared.plan.matrix,
            measurements,
            columns,
            block_count,
        )
    end
    sample.terminal || return FGA.AlgorithmProcessDeferred()

    fault = _join_cpu_dense_rows!(executor)
    isnothing(fault) || throw(fault.cause)
    for value in workspace.candidate_reconstructed
        if !isfinite(value)
            FGA._abandon_incremental_dense_reconstruction!(workspace)
            return FGA.IncrementalDenseReconstructionNonFiniteOutput
        end
    end
    if workspace.seen_count != length(workspace.seen)
        FGA._abandon_incremental_dense_reconstruction!(workspace)
        return FGA.IncrementalDenseReconstructionIncompleteSample
    end

    copyto!(reconstructed, workspace.candidate_reconstructed)
    workspace.terminal_candidate = true
    metadata = FGA.SampleMetadata(
        sample;
        offset=0,
        terminal=true,
        discontinuity=workspace.current_discontinuity,
    )
    return FGA.AlgorithmProcessComplete(metadata)
end

function _cpu_progressive_node(
    node::PreparedNode,
    workers::Int,
    worker_thread_ids::Union{Nothing,Vector{Int}},
    matrix_layout::Symbol,
)
    return node
end

function _cpu_progressive_node(
    node::PreparedNode{FilterGraphAlgorithms.IncrementalDenseReconstructorF32},
    workers::Int,
    worker_thread_ids::Union{Nothing,Vector{Int}},
    matrix_layout::Symbol,
)
    source = node.prepared
    executor = CpuDenseRowExecutor(source, workers, worker_thread_ids, matrix_layout)
    try
        prepared = PreparedCpuDenseReconstructor(
            source.plan,
            CpuDenseWorkspace(source.workspace, executor),
            source.formats,
        )
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
    catch
        close(executor)
        rethrow()
    end
end

@inline _cpu_dense_rows(::Any) = nothing
@inline _cpu_dense_rows(
    node::PreparedNode{FilterGraphAlgorithms.IncrementalDenseReconstructorF32},
) = length(node.prepared.workspace.candidate_reconstructed)

function _cpu_progressive_graph(
    graph::PreparedGraph,
    workers::Int,
    worker_thread_ids::Union{Nothing,Vector{Int}},
    matrix_layout::Symbol,
)
    workers > 0 || throw(ArgumentError("CPU row worker count must be positive"))
    any(node -> !isnothing(_cpu_dense_rows(node)), graph.nodes) || throw(
        GraphConfigurationError(
            "CPU row workers require an incremental-dense-reconstructor-f32 Node",
        ),
    )
    for node in graph.nodes
        rows = _cpu_dense_rows(node)
        isnothing(rows) && continue
        workers <= rows || throw(
            ArgumentError("CPU row worker count must be between 1 and $rows"),
        )
    end
    prepared_nodes = Any[]
    nodes = try
        for node in graph.nodes
            push!(
                prepared_nodes,
                _cpu_progressive_node(node, workers, worker_thread_ids, matrix_layout),
            )
        end
        Tuple(prepared_nodes)
    catch
        _close_cpu_progressive_nodes!(Tuple(prepared_nodes))
        rethrow()
    end
    try
        node_outputs = _bind_cpu_frame_outputs!(Any[graph.node_outputs...], graph, nodes)
        return PreparedGraph(
            nodes,
            graph.node_failure_types,
            node_outputs,
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
    catch
        _close_cpu_progressive_nodes!(nodes)
        rethrow()
    end
end

"""
An ordered CPU Graph owner with persistent workers for incremental reconstruction.

Construct with [`prepare_cpu_progressive_graph`](@ref), serialize operations on
the owner, and call `close` after use to stop its workers. With zero workers,
the owner executes the ordinary serial progressive Graph. Thread placement and
matrix layout are chosen during preparation; the deployment pins Julia threads.
"""
mutable struct PreparedCpuProgressiveGraph{G}
    graph::G
    closed::Bool
end

"""
    isopen(owner::PreparedCpuProgressiveGraph) -> Bool

Return whether the CPU progressive owner's execution resources remain open.
Serialize this query with other owner operations.
"""
Base.isopen(owner::PreparedCpuProgressiveGraph) = !owner.closed

function _close_cpu_progressive_nodes!(nodes::Tuple)
    for node in nodes
        executor = _cpu_dense_executor(node)
        isnothing(executor) || close(executor)
    end
    return nothing
end

function Base.close(owner::PreparedCpuProgressiveGraph)
    owner.closed && return nothing
    _close_cpu_progressive_nodes!(owner.graph.nodes)
    owner.closed = true
    return nothing
end

"""
    cpu_progressive_counters(owner)

Return progress and numerical fallback counts by progressive reconstructor
Node. A conservative finite-output bound switches a frame to serial
accumulation before admitting a block that could overflow. The legacy delayed
failure fields remain for benchmark-record compatibility and stay zero.
The progress counters are sampled inside the terminal reconstructor call,
after the terminal input and upstream Nodes have been admitted.
This allocating host snapshot remains available after `close`. Serialize this
query with processing, reset, parameter replacement and close on the owner.
"""
function cpu_progressive_counters(owner::PreparedCpuProgressiveGraph)
    counters = Dict{Symbol,NamedTuple}()
    for node in owner.graph.nodes
        executor = _cpu_dense_executor(node)
        if !isnothing(executor)
            counters[node.name] = (
                delayed_nonfinite_outputs=executor.delayed_nonfinite_outputs,
                last_failed_sequence=executor.last_failed_sequence,
                frames_with_work_before_terminal_mvm=executor.preterminal_frames,
                columns_completed_before_terminal_mvm=executor.preterminal_columns,
                numerical_fallbacks=executor.numerical_fallbacks,
                direct_columns=executor.direct_columns,
                copied_columns=executor.copied_columns,
            )
        end
    end
    return counters
end

"""
    install_cpu_progressive_trace!(owner; capacity, item=1800)

Install fixed CPU-worker trace storage while no progressive frame is active.
Each worker records `time_ns()` immediately after accumulating `item` in a
frame. Trace ordinal one is the first handoff epoch after installation, so
offline warmup remains outside the returned ordinal range. Installing a new
trace replaces the previous trace. Serial zero-worker owners have no trace.
"""
function install_cpu_progressive_trace!(
    owner::PreparedCpuProgressiveGraph; capacity::Int, item::Int=1800,
)
    _assert_cpu_progressive_open(owner)
    for node in owner.graph.nodes
        executor = _cpu_dense_executor(node)
        isnothing(executor) && continue
        executor.active && throw(ArgumentError(
            "CPU MVM trace installation requires no active progressive frame",
        ))
        trace = CpuProgressiveMvmTrace(
            item,
            capacity,
            length(executor.row_ranges),
            executor.handoff_epoch,
        )
        # Compile the trace write outside the worker's first traced frame.
        _cpu_record_mvm_trace!(trace, 1, executor.handoff_epoch, 0)
        _cpu_record_worker_terminal_trace!(trace, 1, executor.handoff_epoch)
        _cpu_record_coordinator_wait_start!(trace, executor.handoff_epoch)
        _cpu_record_coordinator_wait_end!(trace, executor.handoff_epoch)
        executor.trace = trace
    end
    return owner
end

"""
    cpu_progressive_trace(owner)

Export trace records after the progressive run. Each record carries the
installation-relative `trace_ordinal`, its absolute `handoff_epoch`, one-based
`worker`, completed `item`, monotonic MVM and worker-terminal timestamps, and
the coordinator's terminal-wait interval. The export allocates; worker
processing does not. Calling this while a frame is active is invalid.
"""
function cpu_progressive_trace(owner::PreparedCpuProgressiveGraph)
    _assert_cpu_progressive_open(owner)
    traces = Dict{Symbol,Any}()
    for node in owner.graph.nodes
        executor = _cpu_dense_executor(node)
        isnothing(executor) && continue
        executor.active && throw(ArgumentError(
            "CPU MVM trace export requires no active progressive frame",
        ))
        trace = executor.trace
        isnothing(trace) && continue
        records = NamedTuple{
            (
                :trace_ordinal, :handoff_epoch, :worker, :item, :timestamp_ns,
                :worker_terminal_timestamp_ns, :coordinator_wait_start_ns,
                :coordinator_wait_end_ns,
            ),
            Tuple{Int,UInt64,Int,Int,UInt64,UInt64,UInt64,UInt64},
        }[]
        for ordinal in 1:trace.capacity, worker in 1:trace.workers
            slot = _cpu_trace_index(trace, ordinal, worker)
            timestamp = @inbounds trace.timestamps_ns[slot]
            terminal_timestamp = @inbounds trace.worker_terminal_timestamps_ns[slot]
            wait_start = @inbounds trace.coordinator_wait_start_ns[ordinal]
            wait_end = @inbounds trace.coordinator_wait_end_ns[ordinal]
            iszero(timestamp) && iszero(terminal_timestamp) &&
                iszero(wait_start) && iszero(wait_end) && continue
            push!(records, (
                trace_ordinal=ordinal,
                handoff_epoch=@inbounds(trace.handoff_epochs[slot]),
                worker=worker,
                item=trace.item,
                timestamp_ns=timestamp,
                worker_terminal_timestamp_ns=terminal_timestamp,
                coordinator_wait_start_ns=wait_start,
                coordinator_wait_end_ns=wait_end,
            ))
        end
        traces[node.name] = (
            item=trace.item,
            capacity=trace.capacity,
            workers=trace.workers,
            base_handoff_epoch=trace.base_handoff_epoch,
            records=records,
        )
    end
    return traces
end

@inline _cpu_bound_candidate(node, plan, touched::Bool) = nothing
@inline function _cpu_bound_candidate(
    node::PreparedNode{D,P}, plan, touched::Bool,
) where {D,P<:PreparedCpuDenseReconstructor}
    touched || return nothing
    executor = _cpu_dense_executor(node)
    result = executor.shadow_column_bounds
    _cpu_dense_column_bounds!(result, plan.matrix)
    layout = _cpu_dense_replacement_layout(
        executor.matrix_layout, plan.matrix, executor.row_ranges,
        executor.worker_thread_ids,
    )
    return (; bounds=result, layout)
end

"""
    prepare_cpu_progressive_graph(path; algorithms, workers, worker_thread_ids, matrix_layout)

Prepare an ordered CPU Graph whose incremental dense reconstructors partition
each admitted measurement block over persistent output-row workers. Supported
internal WFS Nodes write into frame-owned output windows; borrowed external
measurement inputs are copied before callback return. Workers
consume published columns in serial order per output row, and terminal
processing joins them once before publication. Call `close` after use. Caller
operations on this owner remain serialized. A conservative output bound keeps
ordinary blocks asynchronous and switches numerically risky frames to serial
accumulation before an expected nonfinite-output failure can be delayed.
The optional `filter.graph.execution.cpu` section supplies `row_workers`,
`worker_thread_ids`, and `matrix_layout` (`shared` or `sharded`). Explicit
keywords override that section. Zero workers
uses the ordinary serial progressive Graph and works with one Julia thread.
When both the section and keywords are omitted, the legacy default is
`max(1, Threads.nthreads() - 1)` workers.
When `worker_thread_ids` is supplied, each worker is a sticky task on its
specified Julia thread. Pin those Julia threads to CPU cores in the deployment
with ThreadPinning.jl before preparation. Sticky workers poll the published
prefix and yield after a bounded spin; unplaced workers yield on an empty poll.
"""
function prepare_cpu_progressive_graph(
    path::AbstractString;
    algorithms::Tuple,
    worker_thread_ids::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    workers::Union{Nothing,Int}=nothing,
    matrix_layout::Union{Nothing,Symbol}=nothing,
)
    spec = _load_graph(path)
    configured = spec.cpu_execution
    selected_workers = isnothing(workers) ?
        (isnothing(worker_thread_ids) ?
            (configured.configured ? configured.row_workers : max(1, Threads.nthreads() - 1)) :
            length(worker_thread_ids)) : workers
    selected_ids = if !isnothing(worker_thread_ids)
        worker_thread_ids
    elseif isnothing(workers)
        configured.worker_thread_ids
    else
        nothing
    end
    selected_layout = isnothing(matrix_layout) ? configured.matrix_layout : matrix_layout
    selected_layout in (:shared, :sharded) || throw(
        ArgumentError("matrix_layout must be :shared or :sharded"),
    )
    selected_workers >= 0 || throw(
        ArgumentError("CPU row worker count must be nonnegative"),
    )
    ids = if isnothing(selected_ids)
        nothing
    else
        selected = [Int(id) for id in selected_ids]
        length(selected) == selected_workers || throw(
            ArgumentError("worker_thread_ids must contain one thread ID per worker"),
        )
        length(unique(selected)) == selected_workers || throw(
            ArgumentError("worker_thread_ids must be distinct"),
        )
        available = ThreadPinning.threadids(; threadpool=:default)
        all(in(available), selected) || throw(
            ArgumentError("worker_thread_ids must belong to the default Julia thread pool"),
        )
        selected
    end
    graph = _prepare_graph(spec; algorithms)
    return PreparedCpuProgressiveGraph(
        iszero(selected_workers) ? graph :
            _cpu_progressive_graph(graph, selected_workers, ids, selected_layout),
        false,
    )
end

function _assert_cpu_progressive_open(owner::PreparedCpuProgressiveGraph)
    owner.closed && throw(ArgumentError("CPU progressive Graph is closed"))
    return nothing
end

function process!(
    outputs::NamedTuple,
    owner::PreparedCpuProgressiveGraph,
    inputs::NamedTuple,
)
    _assert_cpu_progressive_open(owner)
    return process!(outputs, owner.graph, inputs)
end

function process!(
    outputs::NamedTuple,
    owner::PreparedCpuProgressiveGraph,
    inputs::NamedTuple,
    input_metadata::NamedTuple,
)
    _assert_cpu_progressive_open(owner)
    return process!(outputs, owner.graph, inputs, input_metadata)
end

function output_metadata(owner::PreparedCpuProgressiveGraph)
    _assert_cpu_progressive_open(owner)
    return output_metadata(owner.graph)
end

function reset!(owner::PreparedCpuProgressiveGraph)
    _assert_cpu_progressive_open(owner)
    reset!(owner.graph)
    return owner
end

function replace_parameters!(owner::PreparedCpuProgressiveGraph, updates::Pair...)
    _assert_cpu_progressive_open(owner)
    transaction = _prepare_parameter_transaction(owner.graph, updates...)
    _adopt_parameter_transaction!(owner.graph, transaction)
    return owner
end
