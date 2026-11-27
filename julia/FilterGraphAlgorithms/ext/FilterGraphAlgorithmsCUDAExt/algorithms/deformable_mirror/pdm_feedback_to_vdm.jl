struct BoundedFeedbackProjection{Sparse,Dense}
    sparse::Sparse
    dense::Dense
end

function _prepare_active_feedback(device::CUDA.CuDevice, plan, output, input, lanes)
    dense = _prepare_projection(device, plan.full_to_active, output, input, lanes)
    return _prepare_active_feedback(dense, device, plan.full_to_active, plan.sparse_execution)
end

_prepare_active_feedback(dense, device, matrix, sparse_execution) = dense
function _prepare_active_feedback(
        dense::CUBLASProjection, device, matrix,
        sparse_execution::FilterGraphAlgorithms.SparseFeedbackProjection,
)
    CUDA.CUDACore.math_mode() == CUDA.CUDACore.PEDANTIC_MATH &&
        _supports_cuda_conditional() || return dense
    sparse = _prepare_sparse_feedback(device, matrix, sparse_execution)
    return BoundedFeedbackProjection(sparse, dense)
end

_projection_matrix(projection::BoundedFeedbackProjection) = _projection_matrix(projection.dense)

function _enqueue_projection!(projection::BoundedFeedbackProjection, output, status, input)
    sparse, dense = projection.sparse, projection.dense
    # Bare stream capture does not guarantee an allocation-free cuBLAS body.
    if _cuda_conditional_workspace[] === nothing ||
            CUDA.CUDACore.math_mode() != CUDA.CUDACore.PEDANTIC_MATH
        return _enqueue_projection!(dense, output, status, input)
    end
    handle = cuda_conditional_handle()
    handle === nothing && return _enqueue_projection!(dense, output, status, input)
    _enqueue_sparse_feedback!(sparse, output, status, input)
    enqueue_cuda_conditional!(handle, sparse.fallback)
    capture_cuda_if!(handle) do
        _enqueue_projection!(dense, output, status, input)
    end
    return nothing
end
