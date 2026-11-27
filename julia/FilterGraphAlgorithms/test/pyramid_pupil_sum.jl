module PyramidPupilSumTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: discard!, process!, reset!

const FGA = FilterGraphAlgorithms

function prepared_pupils(rows, columns; image_size=(2rows, 2columns),
    origins=[[0, 0], [0, columns], [rows, 0], [rows, columns]],
    mask=trues(rows * columns))
    return prepare_algorithm(PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(image_size=(image_size[1], image_size[2]), pupil_size=(rows, columns),
            pupil_origins=origins, pupil_mask=mask))
end

outputs(prepared) = (zeros(Float32, 4, count(prepared.plan.normalization.mask)),
    zeros(Float32, 1))
sum_bytes(plan, image) = @allocated FGA._pyramid_image_sum(plan, image)
process_bytes(out, prepared, image) = @allocated process!(out..., prepared, image)

function serial_sum(plan, image)
    total = 0f0
    rows, columns = plan.pupil_regions.region_size
    for (first_row, first_column) in plan.pupil_regions.origins
        for row in 1:rows, column in 1:columns
            value = image[first_row + row, first_column + column]
            isfinite(value) || return total, PyramidPixelImageNonFiniteInput
            total += value
            isfinite(total) || return total, PyramidPixelImageNonFiniteMean
        end
    end
    return total, nothing
end

function reference_mean_bound(plan, image)
    return setprecision(BigFloat, 384) do
        total = BigFloat(0)
        sumabs = BigFloat(0)
        rows, columns = plan.pupil_regions.region_size
        for (first_row, first_column) in plan.pupil_regions.origins
            for row in 1:rows, column in 1:columns
                value = BigFloat(image[first_row + row, first_column + column])
                total += value
                sumabs += abs(value)
            end
        end
        # Maximum addition depth, followed by Float32 sum conversion and
        # Float32 mean multiplication. Reference uses the stored scale exactly.
        depth = max(rows + 4columns, columns + 4rows)
        u64 = BigFloat(2)^-53
        u32 = BigFloat(2)^-24
        eta32 = BigFloat(2)^-149
        gamma64 = depth * u64 / (1 - depth * u64)
        scale = BigFloat(plan.normalization.mean_scale)
        coefficient = 2u32 + u32^2 + (1 + u32)^2 * gamma64
        bound = scale * coefficient * sumabs + ((1 + u32) * scale + 1) * eta32 / 2
        return total * scale, bound
    end
end

struct ZeroBasedImage{A<:AbstractMatrix{Float32}} <: AbstractMatrix{Float32}
    parent::A
end
Base.size(image::ZeroBasedImage) = size(image.parent)
Base.axes(image::ZeroBasedImage) = map(n -> 0:(n - 1), size(image.parent))
Base.IndexStyle(::Type{<:ZeroBasedImage}) = IndexCartesian()
Base.getindex(image::ZeroBasedImage, row::Int, column::Int) =
    image.parent[row + 1, column + 1]

@testset "CPU pupil mean has a bounded error through both Float32 roundings" begin
    prepared = prepared_pupils(17, 19)
    n = prod(prepared.plan.pupil_regions.image_size)
    ordinary = reshape(Float32[0.25 + sin(i / 7) for i in 1:n], 34, 38)
    disparate = reshape(Float32[ldexp(isodd(i) ? 1.25 : -1.0, mod(i, 61) - 30)
        for i in 1:n], 34, 38)
    tiny = fill(nextfloat(0f0), 34, 38)
    for image in (ordinary, disparate, tiny)
        sum, failure = FGA._pyramid_image_sum(prepared.plan, image)
        @test failure === nothing
        mean = sum * prepared.plan.normalization.mean_scale
        reference, bound = reference_mean_bound(prepared.plan, image)
        @test abs(BigFloat(mean) - reference) <= bound
    end

    # Casting the completed sum is observable: multiplying in Float64 first
    # would publish a different mean for this exactly representable input sum.
    cast_case = prepared_pupils(2, 3)
    image = zeros(Float32, 4, 6)
    image[1, 1:2] .= Float32[2^24, 1]
    out = outputs(cast_case)
    @test process!(out..., cast_case, image) === nothing
    scale = cast_case.plan.normalization.mean_scale
    @test out[2][1] === Float32(2^24) * scale
    @test out[2][1] !== Float32(Float64(2^24 + 1) * Float64(scale))
    @test pyramid_normalization_gain(cast_case.workspace) === inv(out[2][1])
end

@testset "bounded reduction and serial fallback have distinct qualified domains" begin
    sparse_mask = falses(25 * 41)
    sparse_mask[1] = true
    @test FGA.pyramid_geometry_requires_ordered_sum(prepared_pupils(25, 41; mask=sparse_mask).plan)

    for (rows, columns, ordered_geometry) in ((32, 32, false), (25, 41, true), (1, 3, true))
        prepared = prepared_pupils(rows, columns)
        @test FGA.pyramid_geometry_requires_ordered_sum(prepared.plan) == ordered_geometry
        image = zeros(Float32, 2rows, 2columns)
        image[1, 1:3] .= Float32[2^25, 1, -2^25]
        @test FGA._pyramid_image_sum(prepared.plan, image) === (0f0, nothing)
        @test serial_sum(prepared.plan, image) === (0f0, nothing)
        FGA._pyramid_image_sum(prepared.plan, image)
        @test sum_bytes(prepared.plan, image) == 0
    end

    prepared = prepared_pupils(2, 3)
    image = zeros(Float32, 4, 6)
    image[1, 1:3] .= Float32[2^25, 1, -2^25]
    storage = zeros(Float32, 8, 12)
    strided = @view storage[8:-2:2, 12:-2:2]
    strided .= image
    generic = PermutedDimsArray(permutedims(image), (2, 1))
    @test strided isa StridedMatrix{Float32}
    @test !(generic isa StridedMatrix{Float32})
    @test FGA._pyramid_image_sum(prepared.plan, strided) === (0f0, nothing)
    @test FGA._pyramid_image_sum(prepared.plan, generic) === (0f0, nothing)
    @test FGA._pyramid_image_sum(prepared.plan, generic) === serial_sum(prepared.plan, generic)
    for input in (strided, generic)
        out = outputs(prepared)
        reset!(prepared.workspace)
        @test process!(out..., prepared, input) === nothing
        process!(out..., prepared, input)
        @test process_bytes(out, prepared, input) == 0
    end
    @test_throws ArgumentError process!(outputs(prepared)..., prepared, ZeroBasedImage(image))
end

@testset "signed small-image cancellation retains the ordered next-frame gain" begin
    prepared = prepared_pupils(2, 4)
    large = Float32(2^25)
    next_image = fill(4f0, 4, 8)
    for values in ((large, 1f0, -large, 0f0),
                   (large, 2f0, -large, -1f0),
                   (-large, -2f0, large, 1f0))
        reset!(prepared.workspace)
        out = outputs(prepared)
        seed = fill(2f0, 4, 8)
        @test process!(out..., prepared, seed) === nothing
        image = zeros(Float32, 4, 8)
        image[1, 1:4] .= values
        expected = first(serial_sum(prepared.plan, image))
        @test FGA._pyramid_image_sum(prepared.plan, image) === (expected, nothing)
        @test process!(out..., prepared, image) === nothing
        @test out[1][1, 1] === large * sign(values[1]) * 0.5f0
        gain = expected > 0f0 ? inv(expected * prepared.plan.normalization.mean_scale) : 0f0
        @test pyramid_normalization_gain(prepared.workspace) === gain
        @test process!(out..., prepared, next_image) === nothing
        @test all(==(4f0 * gain), out[1])
        @test process_bytes(out, prepared, image) == 0
    end
end

@testset "retained-transpose 14400-sample sum is bounded and rescans sensitive frames" begin
    prepared = prepared_pupils(60, 60)
    plan = prepared.plan
    dense = reshape(Float32[50f0 + mod(7919i + 31, 4096) / 16f0
        for i in 1:120*120], 120, 120)
    image = transpose(permutedims(dense))
    sum, failure = FGA._pyramid_image_sum(plan, image)
    @test failure === nothing
    reference, bound = reference_mean_bound(plan, image)
    mean = sum * plan.normalization.mean_scale
    @test abs(BigFloat(mean) - reference) <= bound
    @test sum_bytes(plan, image) == 0

    for (ordinal, index) in enumerate(eachindex(image))
        image[index] = mod(ordinal, 4) == 0 ? -99.875f0 :
            100f0 + Float32(mod(ordinal, 17)) / 16f0
    end
    signed_sum, signed_failure = FGA._pyramid_image_sum(plan, image)
    @test signed_failure === nothing
    signed_reference, signed_bound = reference_mean_bound(plan, image)
    @test abs(BigFloat(signed_sum * plan.normalization.mean_scale) -
              signed_reference) <= signed_bound

    for (values, expected) in (
        ((Float32(2^25), 1f0, -Float32(2^25)), nothing),
        ((NaN32, 0f0, 0f0), PyramidPixelImageNonFiniteInput),
        ((floatmax(Float32), floatmax(Float32), 0f0), PyramidPixelImageNonFiniteMean),
        ((Inf32, 0f0, 0f0), PyramidPixelImageNonFiniteInput),
        ((nextfloat(0f0), 0f0, 0f0), nothing),
    )
        fill!(image, 0f0)
        image[1, 1], image[1, 2], image[1, 3] = values
        @test FGA._pyramid_image_sum(plan, image) === serial_sum(plan, image)
        @test FGA._pyramid_image_sum(plan, image)[2] === expected
    end
end

@testset "transpose reduction admits rectangular pupils up to the sample bound" begin
    for (rows, columns, expected) in ((31, 125, Float32(2^25 + 4)),
                                     (32, 128, Float32(2^25 + 4)),
                                     (32, 129, Float32(2^25)))
        plan = prepared_pupils(rows, columns).plan
        image = transpose(zeros(Float32, 2columns, 2rows))
        # Three unit samples vanish in the ordered Float32 sum. Their
        # Float64 sum changes the final narrowing by one Float32 step.
        image[1, 1:3] .= Float32[2^25, 1, 1]
        image[end, end] = 1f0
        @test serial_sum(plan, image) === (Float32(2^25), nothing)
        @test FGA._pyramid_image_sum(plan, image) === (expected, nothing)
        @test sum_bytes(plan, image) == 0

        image[end, end] = 0f0
        image[1, 1:4] .= Float32[2^25, 1, -2^25, 0]
        @test FGA._pyramid_image_sum(plan, image) === serial_sum(plan, image)
    end
end

@testset "full-pupil flux counts masked and overlapping detector samples" begin
    prepared = prepared_pupils(2, 3; image_size=(4, 5),
        origins=[[0, 0], [0, 1], [1, 0], [2, 2]],
        mask=Bool[true, false, false, false, false, false])
    image = reshape(Float32.(1:20), 4, 5)
    reference, bound = reference_mean_bound(prepared.plan, image)
    out = outputs(prepared)
    @test process!(out..., prepared, image) === nothing
    @test abs(BigFloat(out[2][1]) - reference) <= bound
    gain = pyramid_normalization_gain(prepared.workspace)
    @test process!(out..., prepared, image) === nothing
    @test vec(out[1]) == gain .* Float32[image[1, 1], image[1, 2], image[2, 1], image[3, 3]]
    retained = map(copy, out)
    image[1, 3] = NaN32 # Not selected by any pupil; still contributes to full flux.
    @test process!(out..., prepared, image) === PyramidPixelImageNonFiniteInput
    @test out == retained
    @test pyramid_normalization_gain(prepared.workspace) === gain
end

@testset "exceptional rescan retains first failure and its successful sum" begin
    prepared = prepared_pupils(2, 2)
    out = outputs(prepared)
    @test process!(out..., prepared, fill(4f0, 4, 4)) === nothing
    retained = map(copy, out)
    gain = pyramid_normalization_gain(prepared.workspace)
    maximum = floatmax(Float32)
    for (values, failure) in (
        (Float32[NaN, maximum, maximum], PyramidPixelImageNonFiniteInput),
        (Float32[maximum, maximum, NaN], PyramidPixelImageNonFiniteMean),
        (Float32[Inf, 0, 0], PyramidPixelImageNonFiniteInput),
    )
        image = zeros(Float32, 4, 4)
        image[1, 1], image[1, 2], image[2, 1] = values
        @test process!(out..., prepared, image) === failure
        @test out == retained
        @test pyramid_normalization_gain(prepared.workspace) === gain
        @test !prepared.workspace.candidate_ready
        @test only(prepared.workspace.status) == Int32(failure)
        @test only(prepared.workspace.completion_status) == Int32(failure)
        process!(out..., prepared, image)
        @test process_bytes(out, prepared, image) == 0
    end
    image = zeros(Float32, 4, 4)
    image[1:2, 1:2] .= Float32[maximum -maximum; 1 0]
    # Column Float64 arithmetic loses the unit; logical serial arithmetic
    # cancels the extreme pair first. A successful rescan must return its one.
    @test FGA._pyramid_image_sum(prepared.plan, image) === (1f0, nothing)
    FGA._pyramid_image_sum(prepared.plan, image)
    @test sum_bytes(prepared.plan, image) == 0
    reset!(prepared.workspace)
    @test process!(out..., prepared, image) === nothing
    @test out[2] == Float32[1 / 16]
    @test pyramid_normalization_gain(prepared.workspace) === 16f0

    for (value, failure) in ((nextfloat(0f0), PyramidPixelImageNonFiniteNormalizationGain),
        (maximum, PyramidPixelImageNonFiniteMean))
        reset!(prepared.workspace)
        image = fill(value, 4, 4)
        retained = map(copy, out)
        @test process!(out..., prepared, image) === failure
        @test out == retained
        @test iszero(pyramid_normalization_gain(prepared.workspace))
        process!(out..., prepared, image)
        @test process_bytes(out, prepared, image) == 0
    end
end

@testset "reassociated mean retains staged publication and prior-frame ownership" begin
    prepared = prepared_pupils(2, 2)
    workspace = prepared.workspace
    out = outputs(prepared)
    @test process!(out..., prepared, fill(4f0, 4, 4)) === nothing
    retained = map(copy, out)
    @test FGA._stage_pyramid_pixel_image!(workspace, prepared.plan, fill(8f0, 4, 4)) === nothing
    @test out == retained
    @test !workspace.candidate_ready
    @test pyramid_normalization_gain(workspace) === 0.25f0
    @test only(workspace.staged_mean_pupil_intensity) === 8f0
    @test only(workspace.staged_reciprocal_current_frame_mean) === 0.125f0
    @test only(workspace.status) == only(workspace.completion_status) == 0
    FGA._publish_pyramid_pixel_image!(out..., workspace, prepared.plan)
    @test all(==(2f0), out[1])
    @test out[2] == Float32[8]
    @test workspace.candidate_ready
    @test pyramid_normalization_gain(workspace) === 0.25f0
    discard!(workspace)
    @test !workspace.candidate_ready
    @test pyramid_normalization_gain(workspace) === 0.25f0
    @test process!(out..., prepared, fill(8f0, 4, 4)) === nothing
    @test pyramid_normalization_gain(workspace) === 0.125f0
    @test process!(out..., prepared, fill(2f0, 4, 4)) === nothing
    @test all(==(0.25f0), out[1])
    @test pyramid_normalization_gain(workspace) === 0.5f0
    for value in (-1f0, -0f0, 0f0)
        @test process!(out..., prepared, fill(value, 4, 4)) === nothing
        @test pyramid_normalization_gain(workspace) === 0f0
    end
end

@testset "host arithmetic changes select the original serial scan" begin
    prepared = prepared_pupils(2, 3)
    image = zeros(Float32, 4, 6)
    image[1, 1:3] .= Float32[2^25, 1, -2^25]
    image[2, 1] = nextfloat(0f0)
    original_subnormals = get_zero_subnormals()
    original_rounding = Base.Rounding.rounding_raw(Float32)
    try
        for subnormals in (false, true), mode in (RoundNearest, RoundDown, RoundUp, RoundToZero)
            set_zero_subnormals(subnormals)
            Base.Rounding.setrounding_raw(Float32, Base.Rounding.to_fenv(mode))
            if subnormals || mode !== RoundNearest
                actual = FGA._pyramid_image_sum(prepared.plan, image)
                expected = serial_sum(prepared.plan, image)
                @test actual[2] === expected[2]
                @test reinterpret(UInt32, actual[1]) == reinterpret(UInt32, expected[1])
                FGA._pyramid_image_sum(prepared.plan, image)
                @test sum_bytes(prepared.plan, image) == 0
            end
        end
    finally
        Base.Rounding.setrounding_raw(Float32, original_rounding)
        set_zero_subnormals(original_subnormals)
    end
end

end
