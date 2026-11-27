# Finite/sliding state stores the compensated sum and centered sum of squares M₂.
# Exponential state stores the compensated weighted mean and population variance.
struct ArrayStatisticsPlan{N} <: AbstractAlgorithmPlan
    size::NTuple{N,Int}
    mode::Symbol
    window_samples::Int
    alpha::Float64
end

function _array_statistics_plan(configuration)
    dimensions = Tuple(configuration.size)
    isempty(dimensions) && throw(ArgumentError("statistics size must be nonempty"))
    all(>(0), dimensions) || throw(ArgumentError("statistics dimensions must be positive"))
    elements = UInt128(1)
    for dimension in dimensions
        elements *= UInt128(dimension)
        elements <= div(UInt128(typemax(UInt32)), sizeof(Float32)) ||
            throw(ArgumentError("statistics Port byte extent must fit UInt32"))
    end
    mode = Symbol(configuration.mode)
    mode in (:finite, :sliding, :exponential) ||
        throw(ArgumentError("statistics mode must be finite, sliding, or exponential"))
    window = configuration.window_samples
    window > 0 || throw(ArgumentError("window_samples must be positive"))
    if mode === :sliding
        elements * UInt128(window) <= div(UInt128(typemax(UInt32)), sizeof(Float32)) ||
            throw(ArgumentError("statistics history byte extent must fit UInt32"))
    end
    max_state_bytes = configuration.max_state_bytes
    max_state_bytes > 0 || throw(ArgumentError("max_state_bytes must be positive"))
    history_samples = mode === :sliding ? window : 0
    scratch_arrays = mode === :sliding ? 2 : 0
    state_bytes = elements * ((6 + scratch_arrays) * sizeof(Float64) +
        sizeof(Float32) + UInt128(history_samples) * sizeof(Float32))
    state_bytes <= UInt128(max_state_bytes) ||
        throw(ArgumentError("statistics owned state exceeds max_state_bytes"))
    period = configuration.sample_period_s
    tau = configuration.time_constant_s
    isfinite(period) && period > 0 ||
        throw(ArgumentError("sample_period_s must be finite and positive"))
    isfinite(tau) && tau > 0 ||
        throw(ArgumentError("time_constant_s must be finite and positive"))
    alpha = -expm1(-period / tau)
    0 < alpha <= 1 || throw(ArgumentError("exponential weight is not representable"))
    return ArrayStatisticsPlan(dimensions, mode, window, alpha)
end

mutable struct ArrayStatisticsState{N}
    sample_count::UInt64
    sum::Array{Float64,N}
    correction::Array{Float64,N}
    m2::Array{Float64,N}
end

# History belongs to the persistent state owner. Only commit! changes the ring.
mutable struct ArrayStatisticsWorkspace{N}
    committed::ArrayStatisticsState{N}
    candidate::ArrayStatisticsState{N}
    history::Matrix{Float32}
    sample::Array{Float32,N}
    mean::Array{Float64,N}
    variance_correction::Array{Float64,N}
    head::Int
    pending::Bool
end

function ArrayStatisticsWorkspace(plan::ArrayStatisticsPlan)
    state() = ArrayStatisticsState(
        UInt64(0), zeros(Float64, plan.size), zeros(Float64, plan.size),
        zeros(Float64, plan.size),
    )
    history_length = plan.mode === :sliding ? plan.window_samples : 0
    scratch_size = plan.mode === :sliding ? plan.size : map(_ -> 0, plan.size)
    return ArrayStatisticsWorkspace(
        state(), state(),
        zeros(Float32, prod(plan.size), history_length),
        zeros(Float32, plan.size), zeros(Float64, scratch_size),
        zeros(Float64, scratch_size), 1, false,
    )
end

function commit!(workspace::ArrayStatisticsWorkspace)
    workspace.pending || return workspace
    if size(workspace.history, 2) > 0
        # Preparation and the wrapped head establish matching dense extents.
        @inbounds @simd for i in eachindex(workspace.sample)
            workspace.history[i, workspace.head] = workspace.sample[i]
        end
        workspace.head = workspace.head == size(workspace.history, 2) ? 1 : workspace.head + 1
    end
    workspace.committed, workspace.candidate = workspace.candidate, workspace.committed
    workspace.pending = false
    return workspace
end

function discard!(workspace::ArrayStatisticsWorkspace)
    workspace.pending = false
    return workspace
end

function reset!(workspace::ArrayStatisticsWorkspace)
    for state in (workspace.committed, workspace.candidate)
        state.sample_count = 0
        fill!(state.sum, 0)
        fill!(state.correction, 0)
        fill!(state.m2, 0)
    end
    fill!(workspace.history, 0)
    fill!(workspace.sample, 0)
    fill!(workspace.mean, 0)
    fill!(workspace.variance_correction, 0)
    workspace.head = 1
    workspace.pending = false
    return workspace
end

"""Expected input or arithmetic rejection from array statistics."""
@enum ArrayStatisticsFailure::UInt8 begin
    ArrayStatisticsNonFiniteInput = 1
    ArrayStatisticsNonFiniteOutput = 2
    ArrayStatisticsSampleCountOverflow = 3
end

@doc "The accepted array contains a non-finite value." ArrayStatisticsNonFiniteInput
@doc "A candidate statistic cannot be published as finite Float32." ArrayStatisticsNonFiniteOutput
@doc "The committed UInt64 accepted-sample count cannot be incremented." ArrayStatisticsSampleCountOverflow

_array_statistics_noalias(array, ::Tuple{}) = nothing
function _array_statistics_noalias(array, others::Tuple)
    Base.mightalias(array, first(others)) &&
        throw(ArgumentError("statistics caller arrays must not alias each other or owned state"))
    _array_statistics_noalias(array, Base.tail(others))
    return nothing
end

_array_statistics_disjoint(::Tuple{}, others::Tuple) = nothing
function _array_statistics_disjoint(arrays::Tuple, others::Tuple)
    _array_statistics_noalias(first(arrays), others)
    _array_statistics_disjoint(Base.tail(arrays), others)
    return nothing
end

_array_statistics_pairwise_disjoint(::Tuple{}) = nothing
function _array_statistics_pairwise_disjoint(arrays::Tuple)
    _array_statistics_noalias(first(arrays), Base.tail(arrays))
    _array_statistics_pairwise_disjoint(Base.tail(arrays))
    return nothing
end

function _validate_array_statistics_arrays(outputs, workspace, plan, sample)
    expected_axes = map(Base.OneTo, plan.size)
    for array in (sample, outputs[1], outputs[2], outputs[3])
        axes(array) == expected_axes ||
            throw(DimensionMismatch("statistics sample and array output axes must match configured size"))
    end
    scalar_axes = (Base.OneTo(1),)
    axes(outputs[4]) == scalar_axes && axes(outputs[5]) == scalar_axes &&
        axes(outputs[6]) == scalar_axes ||
        throw(DimensionMismatch("statistics scalar outputs must have extent one"))
    owned = (workspace.committed.sum, workspace.committed.correction, workspace.committed.m2,
        workspace.candidate.sum, workspace.candidate.correction, workspace.candidate.m2,
        workspace.history, workspace.sample, workspace.mean, workspace.variance_correction)
    _array_statistics_disjoint((sample, outputs...), owned)
    _array_statistics_pairwise_disjoint((sample, outputs...))
    return nothing
end

# These kernels use exclusively owned dense arrays prepared with the same shape.
# SIMD reorders independent elements only: each Neumaier addition and each
# pixel's history order remain unchanged, without fast-math reassociation.
function _array_statistics_initial!(workspace::ArrayStatisticsWorkspace)
    candidate, sample = workspace.candidate, workspace.sample
    @inbounds @simd for i in eachindex(sample)
        candidate.sum[i] = Float64(sample[i])
        candidate.correction[i] = 0.0
        candidate.m2[i] = 0.0
    end
    return nothing
end

@inline function _statistics_finite_moment(sum, correction, m2, x, divisor, old_divisor)
    total, next_correction = _compensated_add(sum, correction, x)
    mean = total / divisor + next_correction / divisor
    old_mean = sum / old_divisor + correction / old_divisor
    return total, next_correction, fma(x - old_mean, x - mean, m2)
end

@inline function _statistics_exponential_moment(sum, correction, m2, x, alpha, beta)
    scaled = beta * sum
    remainder = fma(beta, correction, fma(beta, sum, -scaled))
    term = alpha * x
    remainder += fma(alpha, x, -term)
    total, next_correction = _compensated_add(scaled, remainder, term)
    delta = x - (sum + correction)
    return total, next_correction, beta * fma(alpha * delta, delta, m2)
end

function _array_statistics_finite!(workspace::ArrayStatisticsWorkspace, plan, count::UInt64)
    committed, candidate, sample = workspace.committed, workspace.candidate, workspace.sample
    divisor, old_divisor = Float64(count), Float64(count - 1)
    @inbounds @simd for i in eachindex(sample)
        candidate.sum[i], candidate.correction[i], candidate.m2[i] = _statistics_finite_moment(
            committed.sum[i], committed.correction[i], committed.m2[i], Float64(sample[i]), divisor, old_divisor)
    end
    return nothing
end

function _array_statistics_exponential!(workspace::ArrayStatisticsWorkspace, plan, count::UInt64)
    committed, candidate, sample = workspace.committed, workspace.candidate, workspace.sample
    alpha, beta = plan.alpha, 1 - plan.alpha
    @inbounds @simd for i in eachindex(sample)
        candidate.sum[i], candidate.correction[i], candidate.m2[i] = _statistics_exponential_moment(
            committed.sum[i], committed.correction[i], committed.m2[i], Float64(sample[i]), alpha, beta)
    end
    return nothing
end

function _array_statistics_sliding!(workspace::ArrayStatisticsWorkspace, plan, count::UInt64)
    candidate, sample = workspace.candidate, workspace.sample
    retained = Int(min(workspace.committed.sample_count, UInt64(plan.window_samples)))
    n = Float64(min(count, UInt64(plan.window_samples)))
    full = workspace.committed.sample_count >= UInt64(plan.window_samples)
    fill!(candidate.sum, 0.0)
    fill!(candidate.correction, 0.0)
    # Keep ring-storage summation order per element while visiting
    # each retained frame contiguously across independent pixels.
    for column in 1:retained
        full && column == workspace.head && continue
        @inbounds @simd for i in eachindex(sample)
            candidate.sum[i], candidate.correction[i] = _compensated_add(
                candidate.sum[i], candidate.correction[i], Float64(workspace.history[i, column]),
            )
        end
    end
    @inbounds @simd for i in eachindex(sample)
        total, correction = _compensated_add(candidate.sum[i], candidate.correction[i], Float64(sample[i]))
        candidate.sum[i], candidate.correction[i] = total, correction
        workspace.mean[i] = total / n + correction / n
        candidate.m2[i] = 0.0
        workspace.variance_correction[i] = 0.0
    end
    for column in 1:retained
        full && column == workspace.head && continue
        @inbounds @simd for i in eachindex(sample)
            delta = Float64(workspace.history[i, column]) - workspace.mean[i]
            candidate.m2[i], workspace.variance_correction[i] = _compensated_add(
                candidate.m2[i], workspace.variance_correction[i], delta * delta,
            )
        end
    end
    @inbounds @simd for i in eachindex(sample)
        delta = Float64(sample[i]) - workspace.mean[i]
        m2, correction = _compensated_add(candidate.m2[i], workspace.variance_correction[i], delta * delta)
        candidate.m2[i] = m2 + correction
    end
    return nothing
end

function _process_array_statistics!(
    mean::AbstractArray{Float32}, variance::AbstractArray{Float32},
    stddev::AbstractArray{Float32}, sample_count::AbstractVector{UInt64},
    complete::AbstractVector{Bool}, variance_valid::AbstractVector{Bool},
    workspace::ArrayStatisticsWorkspace, plan::ArrayStatisticsPlan,
    sample::AbstractArray{Float32},
)::Union{Nothing,ArrayStatisticsFailure}
    outputs = (mean, variance, stddev, sample_count, complete, variance_valid)
    _validate_array_statistics_arrays(outputs, workspace, plan, sample)
    workspace.pending = false
    committed, candidate = workspace.committed, workspace.candidate
    frozen = plan.mode === :finite && committed.sample_count == UInt64(plan.window_samples)
    count = committed.sample_count
    if !frozen
        finite_input = true
        @inbounds @simd for i in eachindex(sample)
            finite_input &= isfinite(sample[i])
        end
        finite_input || return ArrayStatisticsNonFiniteInput
        count == typemax(UInt64) && return ArrayStatisticsSampleCountOverflow
        count += 1
        # Stage the caller array into its already allocated dense candidate owner.
        # Every numerical loop then traverses independent contiguous elements.
        copyto!(workspace.sample, sample)
        if count == 1
            _array_statistics_initial!(workspace)
        elseif plan.mode === :sliding
            _array_statistics_sliding!(workspace, plan, count)
        elseif plan.mode === :finite
            _array_statistics_finite!(workspace, plan, count)
        else
            _array_statistics_exponential!(workspace, plan, count)
        end
    end
    selected = frozen ? committed : candidate
    n = plan.mode === :sliding ? min(count, UInt64(plan.window_samples)) : count
    valid = plan.mode === :exponential ? count >= 1 : n >= 2
    mean_divisor = plan.mode === :exponential ? 1.0 : Float64(n)
    divisor = plan.mode === :exponential ? 1.0 : Float64(max(n, UInt64(2)) - 1)
    finite_output = true
    # A nonnegative variance representable as finite Float32 also has a finite
    # Float32 square root. Only Boolean AND is reduced; sums retain their order.
    @inbounds @simd for i in eachindex(mean, variance, stddev, selected.sum, selected.m2)
        v = valid ? selected.m2[i] / divisor : 0.0
        mu = selected.sum[i] / mean_divisor + selected.correction[i] / mean_divisor
        finite_output &= isfinite(Float32(mu)) & isfinite(Float32(v)) & (v >= 0)
    end
    finite_output || return ArrayStatisticsNonFiniteOutput
    # Publish only after validating every element and all candidate calculations.
    @inbounds @simd for i in eachindex(mean, variance, stddev, selected.sum, selected.m2)
        v = valid ? selected.m2[i] / divisor : 0.0
        mean[i] = Float32(selected.sum[i] / mean_divisor + selected.correction[i] / mean_divisor)
        variance[i] = Float32(v)
        stddev[i] = Float32(sqrt(v))
    end
    sample_count[1] = count
    complete[1] = plan.mode === :finite && count == UInt64(plan.window_samples)
    variance_valid[1] = valid
    if !frozen
        candidate.sample_count = count
        workspace.pending = true
    end
    return nothing
end

"""
Online per-element mean, variance and standard deviation of complete Float32 arrays.
Finite mode freezes after `window_samples` accepted samples. Sliding mode retains
exactly the latest `window_samples` arrays (including warmup). Both publish
unbiased sample variance M₂/(N−1), or zero with `variance-valid=false` for N<2.
`sample-count` counts all accepted samples since reset, including sliding eviction.
`complete` is true only for a full finite window.

Exponential mode initializes mean from the first sample and variance to zero,
then uses α = −expm1(−sample_period_s/time_constant_s), μ′ = μ + α(x−μ),
v′ = (1−α)(v + α(x−μ)²). This is central weighted population variance, without
an unbiased correction; it is valid from the first accepted sample. The nominal
period applies to accepted analysis frames, not guaranteed contiguous exposures.
Means use compensated Float64 sums. Sliding variance is reconstructed from the
retained window with a centered second pass, so eviction cannot erase a small
variance after a large level shift. Finite variance uses a centered recurrence;
exponential means use the equivalent convex weighted sum. Products retain the
configured shape as Float32. Sliding processing is O(PW) per frame for P elements
and W retained samples; finite/exponential processing is O(P). Owned storage,
including the sliding ring, must fit `max_state_bytes`
(default 256 MiB); all memory is allocated during preparation.
Mean and standard deviation use input units; variance uses their square. Explicit
schema fields let each application declare these different quantity contracts.

Processing stages a candidate; local `commit!` adopts it and `discard!` preserves
committed history. A later Graph node failure does not undo a successful local
adoption. Non-finite inputs and unrepresentable Float32 outputs reject atomically.
A completed finite window republishes frozen results and ignores further inputs
until reset. The Algorithm performs no I/O and has no live-control side effects.
"""
@algorithm ArrayStatisticsF32 begin
    label = "array-statistics-f32"
    failure = ArrayStatisticsFailure

    """
    Set positive array `size`, mode (`finite`, `sliding`, or `exponential`), and
    quantity-specific schemas. `window_samples` is positive and counts accepted
    analysis frames. Exponential mode uses finite positive time constants in seconds.
    Configuration changes require reconstruction; reset starts a new collection.
    """
    @configuration ArrayStatisticsF32Configuration begin
        size::Vector{Int} => graph_rebuild
        mode::String => graph_rebuild
        (window_samples::Int = 1) => graph_rebuild
        (time_constant_s::Float64 = 1.0) => graph_rebuild
        (sample_period_s::Float64 = 1.0) => graph_rebuild
        (max_state_bytes::Int = 268435456) => graph_rebuild
        input_schema::String => graph_rebuild
        mean_schema::String => graph_rebuild
        variance_schema::String => graph_rebuild
        stddev_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:sample, Float32, Tuple(configuration.size), configuration.input_schema),
        output(:mean, Float32, Tuple(configuration.size), configuration.mean_schema; metadata_from=:sample),
        output(:variance, Float32, Tuple(configuration.size), configuration.variance_schema; metadata_from=:sample),
        output(:stddev, Float32, Tuple(configuration.size), configuration.stddev_schema; metadata_from=:sample),
        output(:sample_count, UInt64, (1,), "pipewireao/array-statistics-sample-count-u64/1"; name="sample-count", metadata_from=:sample),
        output(:complete, Bool, (1,), "pipewireao/array-statistics-complete-bool/1"; metadata_from=:sample),
        output(:variance_valid, Bool, (1,), "pipewireao/array-statistics-variance-valid-bool/1"; name="variance-valid", metadata_from=:sample),
    )

    prepare(configuration) = begin
        plan = _array_statistics_plan(configuration)
        (plan=plan, workspace=ArrayStatisticsWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_array_statistics!(mean, variance, stddev, sample_count, complete,
            variance_valid, workspace, plan, sample)
    end
end
