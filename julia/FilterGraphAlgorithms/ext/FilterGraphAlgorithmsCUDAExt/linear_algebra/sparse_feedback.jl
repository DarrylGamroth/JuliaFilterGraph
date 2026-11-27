const _SPARSE_FEEDBACK_LIMIT = FilterGraphAlgorithms._PDM_SPARSE_FEEDBACK_COLUMN_LIMIT
const _SPARSE_FEEDBACK_WARP = 32
const _SPARSE_FEEDBACK_WARPS = 8
const _SPARSE_FEEDBACK_THREADS = _SPARSE_FEEDBACK_WARP * _SPARSE_FEEDBACK_WARPS

struct CUDASparseFeedback{Matrix, Vector}
    matrix::Matrix
    columns::Vector
    count::Vector
    fallback::Vector
    execution::FilterGraphAlgorithms.SparseFeedbackProjection
end

function _prepare_sparse_feedback(
        device::CUDA.CuDevice, matrix::Matrix{Float32},
        execution::FilterGraphAlgorithms.SparseFeedbackProjection,
)
    _require_capture_workgroup(device, _SPARSE_FEEDBACK_THREADS;
        warp_width=_SPARSE_FEEDBACK_WARP,
        shared_memory=sizeof(Int32) * (_SPARSE_FEEDBACK_LIMIT + 1) * _SPARSE_FEEDBACK_WARPS,
    )
    return CUDASparseFeedback(
        _capture_array(device, matrix),
        _capture_zeros(device, Int32, _SPARSE_FEEDBACK_LIMIT),
        _capture_zeros(device, Int32, 1),
        _capture_zeros(device, Int32, 1),
        execution,
    )
end

function _enqueue_sparse_feedback!(sparse::CUDASparseFeedback, output, frame_status, input)
    CUDA.@cuda fastmath=false threads=_SPARSE_FEEDBACK_THREADS _compact_sparse_feedback!(
        sparse.columns, sparse.count, sparse.fallback, frame_status, input, sparse.execution,
    )
    blocks = cld(length(output), _SPARSE_FEEDBACK_WARP)
    CUDA.@cuda fastmath=false threads=_SPARSE_FEEDBACK_WARP blocks=blocks _project_sparse_feedback!(
        output, sparse.fallback, sparse.matrix, sparse.columns, sparse.count, input,
    )
    return nothing
end

function _compact_sparse_feedback!(columns, count, fallback, frame_status, input, execution)
    thread = CUDA.threadIdx().x
    if @inbounds !iszero(frame_status[1])
        if thread == 1
            @inbounds begin
                count[1] = Int32(-1)
                fallback[1] = Int32(1)
            end
        end
        return nothing
    end

    warp_columns = CUDA.CuStaticSharedArray(Int32, (_SPARSE_FEEDBACK_LIMIT, _SPARSE_FEEDBACK_WARPS))
    warp_counts = CUDA.CuStaticSharedArray(Int32, _SPARSE_FEEDBACK_WARPS)
    warp, lane = divrem(thread - 1, _SPARSE_FEEDBACK_WARP)
    preceding_lanes = (UInt32(1) << lane) - UInt32(1)
    # Contiguous ranges per warp and ballots in index order preserve the
    # original column order without an atomic append or a sort.
    columns_per_warp = cld(length(input), _SPARSE_FEEDBACK_THREADS) * _SPARSE_FEEDBACK_WARP
    first_column = warp * columns_per_warp + 1
    last_column = min(length(input), first_column + columns_per_warp - 1)
    warp_count = Int32(0)
    ineligible = false
    min_input = reinterpret(UInt32, execution.min_input)
    max_input = reinterpret(UInt32, execution.max_input)
    for base_column in first_column:_SPARSE_FEEDBACK_WARP:last_column
        column = base_column + lane
        weight = column <= last_column ? (@inbounds input[column]) : 0f0
        bits = reinterpret(UInt32, weight)
        active = !iszero(bits)
        # Positive IEEE bit patterns have magnitude order. This classifies
        # negatives, signed zero and exceptional inputs by their stored bits.
        ineligible |= active && !(min_input <= (bits & 0x7fffffff) <= max_input)
        mask = CUDA.vote_ballot_sync(typemax(UInt32), active)
        next_count = warp_count + count_ones(mask)
        if next_count > _SPARSE_FEEDBACK_LIMIT
            ineligible = true
            break
        end
        if active
            slot = warp_count + count_ones(mask & preceding_lanes) + 1
            @inbounds warp_columns[slot, warp + 1] = Int32(column)
        end
        warp_count = Int32(next_count)
    end
    if lane == 0
        @inbounds warp_counts[warp + 1] = warp_count
    end
    ineligible = CUDA.sync_threads_or(ineligible)
    offset = Int32(0)
    total = Int32(0)
    if !ineligible
        @inbounds for preceding_warp in 0:(_SPARSE_FEEDBACK_WARPS - 1)
            n = warp_counts[preceding_warp + 1]
            total += n
            offset += ifelse(preceding_warp < warp, n, Int32(0))
        end
        ineligible = total > _SPARSE_FEEDBACK_LIMIT
    end
    if thread == 1
        @inbounds begin
            # The projection reads count, leaving fallback available for its
            # independent writes from any block that found a suspect row.
            count[1] = ineligible ? Int32(-1) : total
            fallback[1] = Int32(ineligible)
        end
    end
    if !ineligible
        @inbounds for slot in (lane + 1):_SPARSE_FEEDBACK_WARP:warp_count
            columns[offset + slot] = warp_columns[slot, warp + 1]
        end
    end
    return nothing
end

function _project_sparse_feedback!(output, fallback, matrix, columns, count, input)
    active = @inbounds count[1]
    active < 0 && return nothing
    row = (Int(CUDA.blockIdx().x) - 1) * _SPARSE_FEEDBACK_WARP + CUDA.threadIdx().x
    if iszero(active)
        row <= length(output) && (@inbounds output[row] = 0f0)
        return nothing
    end

    if active == 1
        column = @inbounds columns[1]
        weight = @inbounds input[column]
        if weight > 0f0
            # The only nonzero product rounds once. A zero coefficient supplies
            # +0 even when the omitted products have mixed zero signs.
            row <= length(output) && (@inbounds output[row] = matrix[row, column] * weight)
            return nothing
        end
    end

    failed = false
    if row <= length(output)
        value = 0f0
        sum_abs = 0f0
        @inbounds for slot in Int32(1):active
            column = columns[slot]
            weight = input[column]
            coefficient = matrix[row, column]
            value = fma(coefficient, weight, value)
            sum_abs = fma(abs(coefficient), abs(weight), sum_abs)
        end
        # Use the CPU policy's error bound for the two reduction orders. Each
        # nonzero product is normal. The additive term covers rounding of
        # tiny partial sums, including a dense kernel that may flush to zero.
        operations = Float32(2 * active + 8)
        rounding_bound = operations * (eps(Float32) / 2)
        error_scale = 4f0 * rounding_bound / (1f0 - rounding_bound)
        error = fma(error_scale, sum_abs, operations * floatmin(Float32))
        magnitude = abs(value)
        failed = !(isfinite(magnitude) && magnitude > error + floatmin(Float32) &&
                   error <= 1f-11 + 2f-4 * (magnitude - error))
        @inbounds output[row] = value
    end
    failed = CUDA.vote_any_sync(typemax(UInt32), failed)
    if failed && CUDA.threadIdx().x == 1
        CUDA.atomic_or!(pointer(fallback), Int32(1))
    end
    return nothing
end
