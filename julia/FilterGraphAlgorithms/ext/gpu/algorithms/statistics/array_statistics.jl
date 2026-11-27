import FilterGraphAlgorithms: ArrayStatisticsF32, ArrayStatisticsNonFiniteInput,
    ArrayStatisticsNonFiniteOutput, ArrayStatisticsSampleCountOverflow, _compensated_add

@inline function _statistics_update(::Val{:finite}, sum, correction, m2, x, count, alpha,
        history, row, window)
    return FilterGraphAlgorithms._statistics_finite_moment(sum, correction, m2, x,
        Float64(count + UInt64(1)), Float64(count))
end

@inline function _statistics_update(::Val{:exponential}, sum, correction, m2, x, count, alpha,
        history, row, window)
    return FilterGraphAlgorithms._statistics_exponential_moment(sum, correction, m2, x,
        alpha, 1 - alpha)
end

@inline function _statistics_update(::Val{:sliding}, sum, correction, m2, x, count, alpha,
        history, row, window)
    retained = Int(min(count, UInt64(window)))
    evict = count >= window ? Int(rem(count, UInt64(window))) + 1 : 0
    n = Float64(min(count + UInt64(1), UInt64(window)))
    s = c = 0.0
    @inbounds for col in 1:retained
        col == evict && continue
        s, c = _compensated_add(s, c, Float64(history[row, col]))
    end
    s, c = _compensated_add(s, c, x)
    μ = s / n + c / n
    v = vc = 0.0
    @inbounds for col in 1:retained
        col == evict && continue
        δ = Float64(history[row, col]) - μ
        v, vc = _compensated_add(v, vc, δ * δ)
    end
    v, vc = _compensated_add(v, vc, (x - μ) * (x - μ))
    return s, c, v + vc
end

@inline function _statistics_values(mode, s, c, m2, count, window)
    n = _statistics_sample_count(mode, count, window)
    valid = n >= 2
    divisor = Float64(n)
    v = valid ? m2 / Float64(n - 1) : 0.0
    return s / divisor + c / divisor, v, valid
end

@inline _statistics_values(::Val{:exponential}, s, c, m2, count, window) = (s + c, m2, true)
@inline _statistics_sample_count(mode, count, window) = count
@inline _statistics_sample_count(::Val{:sliding}, count, window) = min(count, UInt64(window))
@inline _statistics_frozen(mode, count, window) = false
@inline _statistics_frozen(::Val{:finite}, count, window) = count == window

@inline _record_statistics_sample!(mode, history, i, count, window, sample) = nothing
@inline function _record_statistics_sample!(::Val{:sliding}, history, i, count, window, sample)
    @inbounds history[i, Int(rem(count, UInt64(window))) + 1] = sample[i]
    return nothing
end

@kernel function _statistics_begin!(errors)
    i = @index(Global, Linear)
    @inbounds errors[i] = Int32(0)
end

@inline function _next_statistics(mode, committed, history, i, x, count, alpha, window)
    iszero(count) && return (x, 0.0, 0.0)
    @inbounds return _statistics_update(mode, committed[1][i], committed[2][i], committed[3][i],
        x, count, alpha, history, i, window)
end

@kernel function _statistics_stage!(candidate, committed, history, counter, errors, sample,
        mode, window, alpha, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        count = counter[1]
        frozen = _statistics_frozen(mode, count, window)
        x = Float64(sample[i])
        if !frozen && !isfinite(x)
            KernelAbstractions.@atomic max(errors[1], Int32(1))
        elseif frozen || count != typemax(UInt64)
            if frozen
                moments = (committed[1][i], committed[2][i], committed[3][i])
            else
                moments = _next_statistics(mode, committed, history, i, x, count, alpha, window)
                if !isempty(candidate)
                    candidate[1][i], candidate[2][i], candidate[3][i] = moments
                end
                count += UInt64(1)
            end
            μ, v, _ = _statistics_values(mode, moments..., count, window)
            if !isfinite(Float32(μ)) || !isfinite(Float32(v)) || !(v >= 0)
                KernelAbstractions.@atomic max(errors[2], Int32(1))
            end
        end
    end
end

@kernel function _statistics_status!(status, errors, counter, failures, mode, window)
    i = @index(Global, Linear)
    @inbounds if i == 1 && iszero(status[1])
        if !iszero(errors[1])
            status[1] = failures[1]
        elseif counter[1] == typemax(UInt64) && !_statistics_frozen(mode, counter[1], window)
            status[1] = failures[2]
        elseif !iszero(errors[2])
            status[1] = failures[3]
        end
    end
end

@kernel function _statistics_publish!(mean, variance, stddev, sample_count, complete, variance_valid,
        candidate, committed, history, counter, sample, mode, window, alpha, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        old = counter[1]
        frozen = _statistics_frozen(mode, old, window)
        count = frozen ? old : old + UInt64(1)
        if !frozen
            moments = if isempty(candidate)
                _next_statistics(mode, committed, history, i, Float64(sample[i]), old, alpha, window)
            else
                (candidate[1][i], candidate[2][i], candidate[3][i])
            end
            committed[1][i], committed[2][i], committed[3][i] = moments
            _record_statistics_sample!(mode, history, i, old, window, sample)
        end
        μ,v,valid = _statistics_values(mode, committed[1][i], committed[2][i], committed[3][i], count, window)
        mean[i], variance[i], stddev[i] = Float32(μ), Float32(v), Float32(sqrt(v))
        if i == 1
            sample_count[1] = count
            complete[1] = _statistics_frozen(mode, count, window)
            variance_valid[1] = valid
        end
    end
end

@kernel function _statistics_commit_count!(counter, sample_count, status)
    i = @index(Global, Linear)
    @inbounds if i == 1 && iszero(status[1])
        counter[1] = sample_count[1]
    end
end

struct CapturedArrayStatistics{K,C,F}
    kernels::K
    counter::C
    failures::F
end

capture_support(::Type{ArrayStatisticsF32}, ::DeviceType) = SupportsCapture()

_statistics_history_samples(mode, window) = 0
_statistics_history_samples(::Val{:sliding}, window) = window

function prepare_capture(::CompleteTransition, device::DeviceType, p::PreparedAlgorithm{ArrayStatisticsF32},
        outputs::Tuple, inputs::Tuple, status, index::Int32)
    _check_capture_ports(device, p, outputs, inputs, status)
    sample = only(inputs)
    backend = KernelAbstractions.get_backend(sample)
    mode, window = Val(p.plan.mode), p.plan.window_samples
    state = p.workspace.committed
    all(x -> size(x) == size(sample), (state.sum, state.correction, state.m2)) ||
        throw(DimensionMismatch("committed statistics arrays must match the sample"))
    history_samples = _statistics_history_samples(mode, window)
    size(p.workspace.history) == (length(sample), history_samples) ||
        throw(DimensionMismatch("statistics history does not match the configured window"))
    committed = map(x -> _capture_array(device, x), (state.sum, state.correction, state.m2))
    # The CPU state budget reserves four sample bytes per pixel in addition to
    # six moments. Recompute up to three pixels when those bytes cannot hold
    # the two Int32 error flags and UInt64 counter. Sliding has extra scratch.
    candidate = history_samples > 0 || sizeof(sample) >= 16 ? map(similar, committed) : ()
    history = _capture_array(device, p.workspace.history)
    counter = _capture_array(device, UInt64[state.sample_count])
    errors = _capture_zeros(device, Int32, 2)
    failures = map(f -> _failure_code(index, f), (ArrayStatisticsNonFiniteInput,
        ArrayStatisticsSampleCountOverflow, ArrayStatisticsNonFiniteOutput))
    kernels = (
        CapturedKernel(_statistics_begin!(backend, 2), (errors,), 2),
        CapturedKernel(_statistics_stage!(backend, 128), (candidate, committed, history, counter,
            errors, sample, mode, window, p.plan.alpha, status), length(sample)),
        CapturedKernel(_statistics_status!(backend, 1), (status, errors, counter, failures, mode, window), 1),
        CapturedKernel(_statistics_publish!(backend, 128), (outputs..., candidate, committed, history,
            counter, sample, mode, window, p.plan.alpha, status), length(sample)),
        CapturedKernel(_statistics_commit_count!(backend, 1), (counter, outputs[4], status), 1),
    )
    return CapturedArrayStatistics(kernels, counter, failures)
end

preflight(::CapturedArrayStatistics) = nothing
enqueue!(a::CapturedArrayStatistics) = foreach(enqueue!, a.kernels)
function complete!(a::CapturedArrayStatistics, status::Int32)
    status == a.failures[1] && return ArrayStatisticsNonFiniteInput
    status == a.failures[2] && return ArrayStatisticsSampleCountOverflow
    status == a.failures[3] && return ArrayStatisticsNonFiniteOutput
    return nothing
end
function reset!(a::CapturedArrayStatistics)
    fill!(a.counter, UInt64(0))
    return a
end
