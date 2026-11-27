struct ShortCUBLASProjection{Matrix, Dense, Kernel, Columns}
    matrix::Matrix
    dense::Dense
    kernel::Kernel
    columns::Columns
    input_limit::Float32
end

const SHORT_PROJECTION_THREADS = 128

function _short_projection(dense::CUBLASProjection, device, matrix, output, input, status)
    columns = size(matrix, 2)
    # Even the sum of the absolute products stays below one quarter of the
    # Float32 range. This leaves room for rounding with either reduction order.
    contribution_bound = 4 * columns * max(1.0, dense.maxabs)
    limit = prevfloat(Float32(Float64(floatmax(Float32)) / contribution_bound))
    column_major = _capture_array(device, matrix)
    width = Val(columns)
    # Load the kernel before recording the graph. Older drivers cannot load
    # new modules during capture.
    kernel = CUDA.@cuda launch=false fastmath=false _short_projection_kernel!(
        output, column_major, input, status, limit, UInt64(0), width,
    )
    return ShortCUBLASProjection(column_major, dense, kernel, width, limit)
end

function _prepare_dense_vector_projection(
    device::CUDA.CuDevice, matrix::Matrix{Float32}, output, input, status, lanes,
)
    dense = _prepare_projection(device, matrix, output, input, lanes)
    return _prepare_short_projection(dense, device, matrix, output, input, status)
end

_prepare_short_projection(dense, device, matrix, output, input, status) = dense
function _prepare_short_projection(dense::CUBLASProjection, device, matrix, output, input, status)
    rows, columns = size(matrix)
    # One thread computes one row. Keep enough blocks to benefit from the
    # checked kernel's lower launch count, and bound the per-thread FMA chain
    # and the input scan. Captured timing across synthetic shapes supports
    # these conservative limits.
    15 * SHORT_PROJECTION_THREADS <= rows <= 32 * SHORT_PROJECTION_THREADS &&
        2 <= columns <= 32 || return dense
    isfinite(dense.maxabs) &&
        !_capture_arrays_overlap(input, output) && !_capture_arrays_overlap(status, input) &&
        !_capture_arrays_overlap(status, output) &&
        CUDA.CUDACore.math_mode() != CUDA.CUDACore.FAST_MATH && _supports_cuda_conditional() ||
        return dense
    _require_capture_workgroup(device, SHORT_PROJECTION_THREADS)
    return _short_projection(dense, device, matrix, output, input, status)
end

_projection_matrix(projection::ShortCUBLASProjection) = _projection_matrix(projection.dense)

function _short_projection_kernel!(output, matrix, input, status, limit, handle, ::Val{Columns}) where {Columns}
    row = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    if row == 1
        safe_input = @inbounds iszero(status[1])
        @inbounds for column in Int32(1):Int32(Columns)
            safe_input &= abs(input[column]) <= limit
        end
        set_cuda_conditional!(handle, Cuint(!safe_input))
    end
    # Adjacent lanes read adjacent coefficients from each column.
    if row <= length(output)
        value = 0f0
        @inbounds for column in Int32(1):Int32(Columns)
            value = fma(matrix[row, column], input[column], value)
        end
        @inbounds output[row] = value
    end
    return nothing
end

function _enqueue_short_projection!(projection::ShortCUBLASProjection, output, status, input, handle)
    threads = SHORT_PROJECTION_THREADS
    blocks = cld(length(output), threads)
    projection.kernel(
        output, projection.matrix, input, status, projection.input_limit, handle, projection.columns;
        threads, blocks,
    )
    return nothing
end

function _enqueue_dense_vector_reconstruction!(
        projection::ShortCUBLASProjection, captured::DenseVectorReconstruction,
)
    dense = projection.dense
    if _cuda_conditional_workspace[] === nothing ||
            CUDA.CUDACore.math_mode() == CUDA.CUDACore.FAST_MATH
        return _enqueue_dense_vector_reconstruction!(dense, captured)
    end
    handle = cuda_conditional_handle()
    handle === nothing && return _enqueue_dense_vector_reconstruction!(dense, captured)

    # The short kernel checks all inputs and bounds all output magnitudes.
    # An exceptional frame runs the original validators and GEMV in order.
    _enqueue_short_projection!(
        projection, captured.output, captured.frame_status, captured.input, handle,
    )
    capture_cuda_if!(handle) do
        _enqueue_dense_vector_reconstruction!(dense, captured)
    end
    return nothing
end
