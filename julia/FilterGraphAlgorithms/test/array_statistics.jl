module ArrayStatisticsTests

using FilterGraphAlgorithms
using Statistics
using Test
import FilterGraphAlgorithms: process!, reset!, commit!, discard!, _process_admitted!

function configuration(; size=[2, 3], mode="finite", window_samples=4,
    time_constant_s=2.0, sample_period_s=0.25, max_state_bytes=268435456, input_schema="example/sample-f32/1")
    ArrayStatisticsF32Configuration(; size, mode, window_samples,
        time_constant_s, sample_period_s, max_state_bytes,
        input_schema=input_schema, mean_schema="example/mean-f32/1",
        variance_schema="example/variance-f32/1", stddev_schema="example/stddev-f32/1")
end
prepared(; kwargs...) = prepare_algorithm(ArrayStatisticsF32, configuration(; kwargs...))
outputs(size=(2, 3)) = (fill(71f0, size), fill(72f0, size), fill(73f0, size),
    UInt64[74], Bool[false], Bool[false])
allocations(out, p, sample) = @allocated process!(out..., p, sample)
commit_bytes(workspace) = @allocated commit!(workspace)
discard_bytes(workspace) = @allocated discard!(workspace)
reset_bytes(workspace) = @allocated reset!(workspace)
admitted(out, p, sample) = _process_admitted!(out..., p, sample, FilterGraphAlgorithms._NO_SAMPLE_METADATA)

@testset "array statistics declaration and configuration" begin
    @test ArrayStatisticsF32 in algorithms()
    @test algorithm_label(ArrayStatisticsF32) == "array-statistics-f32"
    @test map(p -> p.name, algorithm_ports(ArrayStatisticsF32)) ==
        (:sample, :mean, :variance, :stddev, Symbol("sample-count"), :complete, Symbol("variance-valid"))
    @test all(p -> p.metadata_source == :sample, algorithm_ports(ArrayStatisticsF32)[2:end])
    for kwargs in ((; size=Int[]), (; size=[0, 3]), (; size=[-1]),
        (; size=[typemax(Int), 3]), (; window_samples=0), (; window_samples=-1),
        (; mode="other"), (; max_state_bytes=0), (; max_state_bytes=215), (; mode="sliding", window_samples=typemax(Int)),
        (; time_constant_s=NaN), (; time_constant_s=0.0),
        (; sample_period_s=Inf), (; sample_period_s=-1.0),
        (; time_constant_s=floatmax(Float64), sample_period_s=nextfloat(0.0)))
        @test_throws ArgumentError prepared(; kwargs...)
    end
    @test_throws ArgumentError prepared(; input_schema="unversioned")
    @test prepared(; size=[1], max_state_bytes=52).workspace.committed.sample_count == 0
    @test_throws ArgumentError prepared(; size=[1], max_state_bytes=51)
    @test prepared(; size=[1], mode="sliding", window_samples=2,
        max_state_bytes=76).workspace.committed.sample_count == 0
    @test_throws ArgumentError prepared(; size=[1], mode="sliding", window_samples=2,
        max_state_bytes=75)
end

@testset "finite and sliding match independent batch statistics" begin
    # The offset and representable small perturbations expose raw sum-of-squares cancellation.
    frames = [reshape(Float32[1e6 + 0.125 * ((i*j)%13 - 6) for i in 1:6], 2, 3) for j in 1:37]
    for mode in ("finite", "sliding"), window in (1, 2, 4, 7)
        p = prepared(; mode, window_samples=window)
        out = outputs()
        for (k, frame) in enumerate(frames)
            @test process!(out..., p, frame) === nothing
            stop = mode == "finite" ? min(k, window) : k
            start = mode == "finite" ? 1 : max(1, k-window+1)
            batch = Float64.(cat(frames[start:stop]...; dims=3))
            expected_mean = dropdims(mean(batch; dims=3); dims=3)
            expected_var = stop-start+1 > 1 ? dropdims(var(batch; dims=3); dims=3) : zeros(2, 3)
            @test out[1] ≈ Float32.(expected_mean)
            @test out[2] ≈ Float32.(expected_var) rtol=2e-6 atol=1e-8
            @test out[3] ≈ Float32.(sqrt.(expected_var)) rtol=2e-6 atol=1e-8
            @test out[4][1] == (mode == "finite" ? min(k, window) : k)
            @test out[5][1] == (mode == "finite" && k >= window)
            @test out[6][1] == (stop-start+1 >= 2)
        end
        reset!(p)
        @test p.workspace.committed.sample_count == 0
        @test iszero(p.workspace.committed.sum)
        @test iszero(p.workspace.committed.m2)
        @test process!(out..., p, frames[1]) === nothing
        @test out[1] == frames[1]
        @test out[2] == zeros(Float32, 2, 3)
    end
end

@testset "long sliding history rejects large-DC cancellation" begin
    p = prepared(; size=[1], mode="sliding", window_samples=17)
    out = outputs((1,))
    sample = zeros(Float32, 1)
    history = Float64[]
    for k in 1:100_000
        sample[1] = k > 99_950 ? 1f8 : Float32(1e8 + 8 * (mod(k*19, 31)-15))
        push!(history, sample[1])
        length(history) > 17 && popfirst!(history)
        process!(out..., p, sample)
        if k <= 17 || k % 997 == 0 || k > 99_950
            expected = length(history) > 1 ? var(history) : 0.0
            @test out[1][1] == Float32(mean(history))
            @test out[2][1] ≈ Float32(expected) rtol=5e-7 atol=1e-8
        end
    end
    @test out[2] == Float32[0]
end

@testset "exponential mean and population variance match independent weighted batch oracle" begin
    for period in (0.25, 100.0)
        p = prepared(; mode="exponential", sample_period_s=period)
        out = outputs()
        frames = [reshape(Float32[10 + sin(i*k) for i in 1:6], 2, 3) for k in 1:20]
        alpha = -expm1(-period/2)
        for k in eachindex(frames)
            @test process!(out..., p, frames[k]) === nothing
            weights = [(1-alpha)^(k-1); [alpha*(1-alpha)^(k-j) for j in 2:k]]
            expected_mean = sum(weights[j] .* Float64.(frames[j]) for j in 1:k)
            expected_var = sum(weights[j] .* (Float64.(frames[j]) .- expected_mean).^2 for j in 1:k)
            @test out[1] ≈ Float32.(expected_mean) rtol=2e-6
            @test out[2] ≈ Float32.(expected_var) rtol=2e-6 atol=1e-7
            @test out[3] ≈ Float32.(sqrt.(expected_var)) rtol=2e-6 atol=1e-7
            @test out[4] == UInt64[k]
            @test out[5] == [false]
            @test out[6] == [true]
        end
        reset!(p)
        process!(out..., p, frames[end])
        @test out[1] == frames[end]
        @test iszero(out[2])
    end
end

@testset "candidate discard, atomic rejection, completion and reset" begin
    for mode in ("finite", "sliding", "exponential")
        p = prepared(; mode, window_samples=2)
        out = outputs()
        sample = fill(1f0, 2, 3)
        @test admitted(out, p, sample) === nothing
        @test p.workspace.committed.sample_count == 0
        @test p.workspace.pending
        discard!(p.workspace)
        @test p.workspace.committed.sample_count == 0
        @test iszero(p.workspace.history)
        process!(out..., p, sample)
        before = deepcopy(out)
        history = copy(p.workspace.history)
        invalid = copy(sample)
        invalid[end] = NaN32
        @test process!(out..., p, invalid) === ArrayStatisticsNonFiniteInput
        @test out == before
        @test p.workspace.committed.sample_count == 1
        @test p.workspace.history == history
        # The second sample's unbiased variance exceeds Float32 without overflowing Float64.
        if mode != "exponential"
            @test process!(out..., p, fill(floatmax(Float32), 2, 3)) === ArrayStatisticsNonFiniteOutput
            @test out == before
            @test p.workspace.committed.sample_count == 1
            @test p.workspace.history == history
        end
        sample .= 3
        @test admitted(out, p, sample) === nothing
        discard!(p.workspace)
        @test p.workspace.committed.sample_count == 1
        @test p.workspace.history == history
        process!(out..., p, sample)
        @test out[4] == UInt64[2]
        if mode == "finite"
            before = deepcopy(out)
            @test process!(out..., p, invalid) === nothing
            @test out == before
            @test p.workspace.committed.sample_count == 2
        end
        reset!(p)
        @test !p.workspace.pending
        @test iszero(p.workspace.history)
        @test p.workspace.head == 1
        process!(out..., p, fill(9f0, 2, 3))
        @test out[4] == UInt64[1]
        @test out[1] == fill(9f0, 2, 3)
        @test iszero(out[2])
    end
end

@testset "array shapes, views, aliases and allocation contract" begin
    for shape in ((3,), (2, 3), (2, 1, 3))
        p = prepared(; size=collect(shape), mode="sliding")
        sample = fill(2f0, shape)
        out = outputs(shape)
        @test @inferred Union{Nothing,ArrayStatisticsFailure} process!(out..., p, sample) === nothing
        for _ in 1:6
            process!(out..., p, sample)
        end
        allocations(out, p, sample)
        @test allocations(out, p, sample) == 0
        invalid = copy(sample)
        invalid[end] = Inf32
        allocations(out, p, invalid)
        @test allocations(out, p, invalid) == 0
        @test_throws ArgumentError process!(sample, out[2:end]..., p, sample)
        @test_throws ArgumentError process!(out[1], out[1], out[3:end]..., p, sample)
        @test_throws ArgumentError process!(out..., p, p.workspace.sample)
        @test_throws DimensionMismatch process!(out..., p, zeros(Float32, prod(shape)+1))
    end
    p = prepared(; mode="sliding")
    sample = @view fill(2f0, 4, 6)[1:2:4, 1:2:6]
    out = outputs()
    for _ in 1:7
        @test process!(out..., p, sample) === nothing
    end
    @test out[1] == fill(2f0, 2, 3)
    view_out = ((@view zeros(Float32, 4, 6)[1:2:4, 1:2:6]), out[2:end]...)
    @test process!(view_out..., p, sample) === nothing
    @test view_out[1] == sample
    for mode in ("finite", "sliding", "exponential")
        p = prepared(; mode)
        out = outputs()
        sample = fill(2f0, 2, 3)
        for _ in 1:6
            process!(out..., p, sample)
        end
        allocations(out, p, sample)
        @test allocations(out, p, sample) == 0
        @test @inferred(commit!(p.workspace)) === p.workspace
        @test @inferred(discard!(p.workspace)) === p.workspace
        @test @inferred(reset!(p.workspace)) === p.workspace
        commit_bytes(p.workspace)
        discard_bytes(p.workspace)
        reset_bytes(p.workspace)
        @test commit_bytes(p.workspace) == 0
        @test discard_bytes(p.workspace) == 0
        @test reset_bytes(p.workspace) == 0
    end
    p = prepared(; mode="exponential")
    p.workspace.committed.sample_count = typemax(UInt64)
    out = outputs()
    before = deepcopy(out)
    @test process!(out..., p, fill(1f0, 2, 3)) === ArrayStatisticsSampleCountOverflow
    @test out == before
end

@testset "statistics retain small signals after large level shifts" begin
    for frames in (Float32[1e18, 1, 1, 2, 2], Float32[1e8, 1, 2, 3])
        p = prepared(; size=[1], mode="sliding", window_samples=2)
        out = outputs((1,))
        for k in eachindex(frames)
            @test process!(out..., p, frames[k:k]) === nothing
            batch = Float64.(frames[max(1, k-1):k])
            expected_variance = k == 1 ? 0.0 : var(batch)
            @test out[1][1] ≈ Float32(mean(batch)) rtol=2e-6
            @test out[2][1] ≈ Float32(expected_variance) rtol=2e-6
            @test out[3][1] ≈ Float32(sqrt(expected_variance)) rtol=2e-6
        end
    end

    # A candidate eviction must leave the committed large sample available after discard.
    p = prepared(; size=[1], mode="sliding", window_samples=2)
    out = outputs((1,))
    process!(out..., p, Float32[1e18])
    process!(out..., p, Float32[1])
    history = copy(p.workspace.history)
    @test admitted(out, p, Float32[2]) === nothing
    @test out[1] == Float32[1.5]
    @test out[2] == Float32[0.5]
    discard!(p.workspace)
    @test p.workspace.history == history
    @test p.workspace.committed.sample_count == 2
    @test process!(out..., p, Float32[3]) === nothing
    @test out[1] == Float32[2]
    @test out[2] == Float32[2]

    # Finite means must retain a small term between cancelling large terms.
    p = prepared(; size=[1], mode="finite", window_samples=3)
    out = outputs((1,))
    for frame in Float32[1e18, 1, -1e18]
        @test process!(out..., p, Float32[frame]) === nothing
    end
    @test out[1] == Float32[1/3]
    @test out[2][1] ≈ Float32(Float64(1f18)^2) rtol=2e-6
    @test out[5] == [true]

    # A rounded alpha of one is exactly the current sample, including after reset.
    p = prepared(; size=[1], mode="exponential", sample_period_s=100.0, time_constant_s=1.0)
    out = outputs((1,))
    @test p.plan.alpha == 1.0
    for (k, frame) in enumerate(Float32[1e18, 1, 2, 3])
        @test process!(out..., p, Float32[frame]) === nothing
        @test out[1] == Float32[frame]
        @test out[2] == Float32[0]
        @test out[4] == UInt64[k]
    end
    reset!(p)
    @test process!(out..., p, Float32[7]) === nothing
    @test out[1] == Float32[7]
    @test out[4] == UInt64[1]
    @test out[6] == [true]

    # Weighted cancellation exposes the low part of the compensated mean.
    p = prepared(; size=[1], mode="exponential", sample_period_s=log(2.0), time_constant_s=1.0)
    out = outputs((1,))
    @test p.plan.alpha == 0.5
    for frame in Float32[1e18, 1, -5e17]
        @test process!(out..., p, Float32[frame]) === nothing
    end
    @test out[1] == Float32[0.25]
end

@testset "SIMD elements and tails preserve compensated statistics" begin
    shape = (5, 7)
    large = reshape(Float32[isodd(i) ? 1e18 : -1e18 for i in 1:prod(shape)], shape)
    small = reshape(Float32[i/8 for i in 1:prod(shape)], shape)
    for (mode, alpha, frames) in (
        ("finite", 0.5, (large, small, -large)),
        ("sliding", 0.5, (large, small, 2small, 3small)),
        ("exponential", 0.5, (large, small, -large/2)),
        ("exponential", 1.0, (large, small, 2small)),
    )
        period = alpha == 1 ? 100.0 : log(2.0)
        p = prepared(; size=collect(shape), mode, window_samples=mode == "sliding" ? 2 : 4,
            sample_period_s=period, time_constant_s=1.0)
        out = outputs(shape)
        for k in eachindex(frames)
            @test process!(out..., p, frames[k]) === nothing
            first = mode == "sliding" ? max(1, k-1) : 1
            n = k-first+1
            expected_mean = zeros(Float32, shape)
            expected_variance = zeros(Float32, shape)
            expected_stddev = zeros(Float32, shape)
            for i in eachindex(expected_mean)
                batch = BigFloat[frames[j][i] for j in first:k]
                if mode == "exponential"
                    weights = BigFloat[(1-alpha)^(k-1);
                        [alpha*(1-alpha)^(k-j) for j in 2:k]]
                    mu = sum(weights .* batch)
                    v = sum(weights .* (batch .- mu).^2)
                else
                    mu = sum(batch)/n
                    v = n < 2 ? BigFloat(0) : sum((batch .- mu).^2)/(n-1)
                end
                expected_mean[i] = Float32(mu)
                expected_variance[i] = Float32(v)
                expected_stddev[i] = Float32(sqrt(v))
            end
            @test out[1] ≈ expected_mean rtol=2e-6
            @test out[2] ≈ expected_variance rtol=2e-6
            @test out[3] ≈ expected_stddev rtol=2e-6
        end
        before = deepcopy(out)
        count = p.workspace.committed.sample_count
        for invalid_index in (5, length(small))
            invalid = copy(small)
            invalid[invalid_index] = NaN32
            @test process!(out..., p, invalid) === ArrayStatisticsNonFiniteInput
            @test out == before
            @test p.workspace.committed.sample_count == count
            if mode != "exponential" || alpha != 1
                invalid[invalid_index] = floatmax(Float32)
                @test process!(out..., p, invalid) === ArrayStatisticsNonFiniteOutput
                @test out == before
                @test p.workspace.committed.sample_count == count
            end
        end
        reset!(p)
        @test iszero(p.workspace.mean)
        @test iszero(p.workspace.variance_correction)
    end
end

end
