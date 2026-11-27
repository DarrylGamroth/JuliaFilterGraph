import FilterGraphAlgorithms: MlLlowfsF32, MlLlowfsParametersMissing, MlLlowfsNonFiniteInput,
    MlLlowfsInvalidFlux, MlLlowfsInvalidStandardDeviation, MlLlowfsNonFiniteOutput
import FilterGraphAlgorithms.SimpleChains as SC

@kernel function _normalize_llowfs_region!(input, pixels, ready, status, failures)
    lane = @index(Global, Linear)
    @inbounds if lane == 1 && iszero(status[1])
        failure = ready ? FilterGraphAlgorithms._normalize_llowfs!(input, pixels) : MlLlowfsParametersMissing
        if !isnothing(failure)
            status[1] = failures[Int(failure)]
        end
    end
end

@kernel function _llowfs_convolution!(out, input, weights, bias, activation, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        r,c,channel = Tuple(CartesianIndices(out)[i])
        x = 0f0
        for k in axes(weights, 3), b in axes(weights, 2), a in axes(weights, 1)
            x = muladd(input[r+a-1,c+b-1,k], weights[a,b,k,channel], x)
        end
        out[i] = activation(x + bias[channel])
    end
end

@kernel function _llowfs_max_pool!(out, input, window, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        r,c,k = Tuple(CartesianIndices(out)[i])
        x = -Inf32
        for b in 1:window[2], a in 1:window[1]
            x = max(x, input[(r-1)*window[1]+a, (c-1)*window[2]+b, k])
        end
        out[i] = x
    end
end

@kernel function _llowfs_dense!(out, input, weights, bias, activation, status)
    row = @index(Global, Linear)
    @inbounds if iszero(status[1])
        x = 0f0
        for j in axes(weights,2)
            x = muladd(weights[row,j], input[j], x)
        end
        x += bias[row]
        out[row] = activation(x)
    end
end

function _llowfs_layer(device, backend, layer::SC.Conv, parameters, input, status)
    weights,bias = map(x -> _capture_array(device, Array(x)), parameters)
    out = _capture_zeros(device, Float32, size(input,1)-size(weights,1)+1,
        size(input,2)-size(weights,2)+1, length(bias))
    kernel = CapturedKernel(_llowfs_convolution!(backend, 128),
        (out,input,weights,bias,layer.f,status), length(out))
    return out, (kernel,)
end
function _llowfs_layer(device, backend, ::SC.MaxPool{W}, ::Nothing, input, status) where {W}
    out = _capture_zeros(device, Float32, size(input,1)÷W[1], size(input,2)÷W[2], size(input,3))
    return out, (CapturedKernel(_llowfs_max_pool!(backend, 128), (out,input,W,status), length(out)),)
end
_llowfs_layer(device, backend, ::SC.Flatten, ::Nothing, input, status) = (vec(input), ())
function _llowfs_layer(device, backend, layer::SC.TurboDense, parameters, input, status)
    weights,bias = map(x -> _capture_array(device, Array(x)), parameters)
    out = _capture_zeros(device, Float32, length(bias))
    launch = CapturedKernel(_llowfs_dense!(backend, 128),
        (out,input,weights,bias,layer.f,status), length(out))
    return out, (launch,)
end

struct CapturedMlLlowfs{K,F}
    kernels::K
    failures::F
end

capture_support(::Type{MlLlowfsF32}, ::DeviceType) = SupportsCapture()
function prepare_capture(::CompleteTransition, device::DeviceType, p::PreparedAlgorithm{MlLlowfsF32},
        outputs::Tuple, inputs::Tuple, status, index::Int32)
    _check_capture_ports(device, p, outputs, inputs, status)
    backend = KernelAbstractions.get_backend(only(inputs))
    errors = (MlLlowfsParametersMissing, MlLlowfsNonFiniteInput, MlLlowfsInvalidFlux,
        MlLlowfsInvalidStandardDeviation, MlLlowfsNonFiniteOutput)
    failures = map(f -> _failure_code(index, f), errors)
    length(p.plan.parameters) == p.plan.model.parameter_count ||
        throw(DimensionMismatch("ML LLOWFS plan has the wrong parameter extent"))
    chain = p.plan.model.chain
    parameters = SC.params(chain, p.plan.parameters)
    _, rows, columns = size(only(inputs))
    input = _capture_zeros(device, Float32, rows, columns, 1)
    normalize = CapturedKernel(_normalize_llowfs_region!(backend, 1),
        (input, only(inputs), p.plan.ready, status, failures), 1)
    kernels = (normalize,)
    for (layer, params) in zip(chain.layers, parameters)
        input, launches = _llowfs_layer(device, backend, layer, params, input, status)
        kernels = (kernels..., launches...)
    end
    validate = CapturedKernel(_validate_projection_values_portable!(backend, VALIDATION_WORKGROUP_SIZE),
        (status, input, last(failures)), VALIDATION_WORKGROUP_SIZE)
    publish = CapturedKernel(_publish_llowfs!(backend,32), (only(outputs),input,status), length(input))
    return CapturedMlLlowfs((kernels..., validate, publish), map(Pair, failures, errors))
end

@kernel function _publish_llowfs!(out, input, status)
    i = @index(Global, Linear)
    @inbounds iszero(status[1]) && (out[i] = input[i])
end

preflight(::CapturedMlLlowfs) = nothing
enqueue!(a::CapturedMlLlowfs) = foreach(enqueue!, a.kernels)
function complete!(a::CapturedMlLlowfs, status::Int32)
    for (code, failure) in a.failures
        status == code && return failure
    end
    return nothing
end
reset!(a::CapturedMlLlowfs) = a
