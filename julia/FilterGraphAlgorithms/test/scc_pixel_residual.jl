module SccPixelResidualTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

function prepared_scc_pixel_residual(; minimum_valid_fraction=0.5f0)
    return prepare_algorithm(
        SccPixelResidualF32,
        SccPixelResidualF32Configuration(
            pixel_count=6,
            mask=Bool[true, false, true, true, false, true],
            initial_reference_image=Float32[1, 10, -1, 2, 0, 4],
            minimum_valid_fraction=minimum_valid_fraction,
        ),
    )
end

function scc_pixel_residual_allocation_bytes(outputs, prepared, inputs)
    return @allocated process!(outputs..., prepared, inputs...)
end

struct GuardedDifference{T} <: AbstractVector{T}
    values::Vector{T}
    invalid_pixel::Int
end

Base.size(difference::GuardedDifference) = size(difference.values)
function Base.getindex(difference::GuardedDifference, pixel::Int)
    pixel == difference.invalid_pixel && error("read of an upstream-invalid SCC pixel")
    return difference.values[pixel]
end

@testset "SCC pixel-residual declaration preserves explicit SCC spaces" begin
    @test algorithm_label(SccPixelResidualF32) == "scc-pixel-residual-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(SccPixelResidualF32) ===
          SccPixelResidualFailure
    @test map(field -> field.name, algorithm_configuration(SccPixelResidualF32)) == (
        :pixel_count,
        :mask,
        :initial_reference_image,
        :minimum_valid_fraction,
    )

    ports = algorithm_ports(SccPixelResidualF32)
    @test map(port -> port.name, ports) == (
        :difference,
        Symbol("difference-validity"),
        Symbol("pair-status"),
        Symbol("selected-residuals"),
        Symbol("selected-validity"),
        Symbol("valid-fraction"),
        Symbol("reference-image"),
        :mask,
    )
    @test all(port -> isnothing(port.metadata_source), ports[1:3])
    @test all(port -> port.metadata_source === :difference, ports[4:6])
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[7:8],
    )

    formats = algorithm_formats(prepared_scc_pixel_residual())
    @test getproperty.(formats, :shape) == (
        (6,),
        (6,),
        (1,),
        (4,),
        (4,),
        (1,),
        (6,),
        (6,),
    )
    @test getproperty.(formats, :schema) == (
        SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
        SCC_CHOPPED_DIFFERENCE_STATUS_V1,
        SCC_SELECTED_PIXEL_RESIDUALS_V1,
        SCC_SELECTED_PIXEL_VALIDITY_V1,
        SCC_SELECTED_PIXEL_VALID_FRACTION_V1,
        SCC_REFERENCE_IMAGE_V1,
        PIXEL_SELECTION_MASK_V1,
    )
end

@testset "SCC residuals match the SPIDERS selected-pixel convention" begin
    prepared = prepared_scc_pixel_residual()
    residuals = fill(71.0f0, 4)
    selected_validity = fill(true, 4)
    valid_fraction = fill(-1.0f0, 1)
    difference = Float32[3, 100, 4, NaN, 0, 1]
    validity = Bool[true, true, true, true, true, false]

    @test process!(
        residuals,
        selected_validity,
        valid_fraction,
        prepared,
        difference,
        validity,
        UInt8[UInt8(ChoppedImageComplete)],
    ) === nothing
    @test residuals == Float32[2, 5, 0, 0]
    @test selected_validity == Bool[true, true, false, false]
    @test valid_fraction == Float32[0.5]

    reference = Float32[0, 10, 0, 2, 0, 0]
    @test replace_parameter!(prepared, "reference-image", reference) === prepared
    @test process!(
        residuals,
        selected_validity,
        valid_fraction,
        prepared,
        Float32[3, 100, 4, 2, 0, 1],
        trues(6),
        UInt8[UInt8(ChoppedImageComplete)],
    ) === nothing
    @test residuals == Float32[3, 4, 0, 1]
    @test valid_fraction == Float32[1]

    replacement_mask = Bool[false, true, true, false, true, true]
    @test replace_parameter!(prepared, :mask, replacement_mask) === prepared
    @test process!(
        residuals,
        selected_validity,
        valid_fraction,
        prepared,
        Float32[3, 100, 4, 2, 7, 1],
        trues(6),
        UInt8[UInt8(ChoppedImageComplete)],
    ) === nothing
    @test residuals == Float32[90, 4, 7, 1]
end

@testset "SCC pair and valid-fraction dispositions precede control" begin
    prepared = prepared_scc_pixel_residual(; minimum_valid_fraction=0.75f0)
    outputs = (fill(71.0f0, 4), fill(true, 4), fill(-1.0f0, 1))
    inputs = (
        Float32[3, 100, 4, NaN, 0, 1],
        Bool[true, true, true, true, true, false],
    )

    @test process!(
        outputs...,
        prepared,
        inputs...,
        UInt8[UInt8(ChoppedImagePriming)],
    ) === SccPixelResidualIncompletePair
    @test outputs == (fill(71.0f0, 4), fill(true, 4), fill(-1.0f0, 1))
    @test process!(outputs..., prepared, inputs..., UInt8[0xff]) ===
          SccPixelResidualInvalidPairStatus
    @test process!(
        outputs...,
        prepared,
        inputs...,
        UInt8[UInt8(ChoppedImageComplete)],
    ) === SccPixelResidualInsufficientValidPixels
    @test outputs[1] == Float32[2, 5, 0, 0]
    @test outputs[2] == Bool[true, true, false, false]
    @test outputs[3] == Float32[0.5]
end

@testset "All-selected SCC residuals preserve arithmetic and disposition" begin
    for T in (Float32, Float64)
        n = 129
        reference = fill(zero(T), n)
        difference = fill(T(3), n)
        validity = fill(true, n)
        difference[61] = -zero(T)
        difference[77] = T(0x1p-103)
        reference[77] = T(nextfloat(Float32(0x1p-103)))
        difference[107] = T(prevfloat(Float32(0x1p127)))
        reference[107] = -difference[107]
        outputs = (fill(T(71), n), fill(true, n), fill(T(-1), 1))
        status = UInt8[UInt8(ChoppedImageComplete)]
        inputs = (difference, validity, status)
        plan = SccPixelResidualPlan(reference, fill(true, n), 1)
        @test process!(outputs..., plan, inputs...) === nothing
        @test isequal(outputs[1], difference .- reference)
        @test all(outputs[2]) && only(outputs[3]) === one(T)
        @test scc_pixel_residual_allocation_bytes(outputs, plan, inputs) == 0

        plan.selected_indices[end] = 61
        @test process!(outputs..., plan, inputs...) === nothing
        @test outputs[1][end] === -zero(T)

        difference[17] = T(Inf)
        difference[33] = T(NaN)
        difference[79] = floatmax(T)
        difference[97] = floatmax(T)
        reference[79] = -floatmax(T)
        reference[97] = -floatmax(T)
        validity[79] = false
        validity[113] = false
        difference[end] = nextfloat(zero(T))
        foreach(fill!, outputs, (T(71), true, T(-1)))
        status[1] = UInt8(ChoppedImagePriming)
        plan = SccPixelResidualPlan(reference, fill(true, n), 1)

        @test process!(outputs..., plan, inputs...) === SccPixelResidualIncompletePair
        @test outputs == (fill(T(71), n), fill(true, n), fill(T(-1), 1))
        status[1] = 0xff
        @test process!(outputs..., plan, inputs...) === SccPixelResidualInvalidPairStatus
        @test outputs == (fill(T(71), n), fill(true, n), fill(T(-1), 1))
        status[1] = UInt8(ChoppedImageComplete)
        @test process!(outputs..., plan, inputs...) === SccPixelResidualInsufficientValidPixels

        expected_validity = fill(true, n)
        expected_validity[[17, 33, 79, 97, 113]] .= false
        expected_residuals = difference .- reference
        expected_residuals[.!expected_validity] .= zero(T)
        @test isequal(outputs[1], expected_residuals)
        @test outputs[2] == expected_validity
        @test only(outputs[3]) === T(n - 5) / T(n)
        @test scc_pixel_residual_allocation_bytes(outputs, plan, inputs) == 0

        plan = SccPixelResidualPlan(reference, trues(n), 0.9)
        strided = view(fill(T(-2), 2n), 1:2:2n)
        bits = falses(n)
        fraction = zeros(T, 1)
        @test process!(strided, bits, fraction, plan, view(difference, :),
            BitVector(validity), status) === nothing
        @test isequal(strided, expected_residuals)
        @test bits == expected_validity && fraction == outputs[3]
        @test process!(outputs..., plan, GuardedDifference(difference, 79),
            validity, status) === nothing
        @test isequal(outputs[1], expected_residuals)
    end
end

@testset "SCC residual selection follows the current plan indices" begin
    outputs = (zeros(Float32, 4), fill(false, 4), zeros(Float32, 1))
    inputs = (Float32[1, 2, 3, 4], fill(true, 4), UInt8[UInt8(ChoppedImageComplete)])
    mask = fill(true, 4)
    reference = zeros(Float32, 4)
    plan = SccPixelResidualPlan(reference, mask, 1)
    @test process!(outputs..., plan, inputs...) === nothing
    @test outputs[1] == inputs[1]
    plan.selected_indices .= [4, 2, 2, 1]
    @test process!(outputs..., plan, inputs...) === nothing
    @test outputs[1] == Float32[4, 2, 2, 1]

    direct = SccPixelResidualPlan{Float32}(reference, copy(mask), [3, 1, 4, 2], 1f0, 4)
    @test process!(outputs..., direct, inputs...) === nothing
    @test outputs[1] == Float32[3, 1, 4, 2]
    direct.mask .= false
    direct.selected_indices .= 1:4
    @test process!(outputs..., direct, inputs...) === nothing
    @test outputs[1] == inputs[1]

    prepared = prepare_algorithm(SccPixelResidualF32, SccPixelResidualF32Configuration(
        pixel_count=4, mask=mask, initial_reference_image=reference, minimum_valid_fraction=1f0,
    ))
    @test replace_parameter!(prepared, "reference-image", ones(Float32, 4)) === prepared
    @test replace_parameter!(prepared, :mask, fill(true, 4)) === prepared
    @test process!(outputs..., prepared, inputs...) === nothing
    @test outputs[1] == Float32[0, 1, 2, 3]
end

@testset "SCC residuals preserve the update order of overlapping validity vectors" begin
    n = 129
    shared = fill(true, n + 1)
    shared[1] = false
    residuals = fill(71.0f0, n)
    fraction = fill(-1.0f0, 1)
    plan = SccPixelResidualPlan(zeros(Float32, n), trues(n), 1)
    difference = ones(Float32, n)
    status = UInt8[UInt8(ChoppedImageComplete)]
    GC.@preserve shared begin
        validity = unsafe_wrap(Vector{Bool}, pointer(shared), n)
        selected_validity = unsafe_wrap(Vector{Bool}, pointer(shared, 2), n)
        @test process!(residuals, selected_validity, fraction, plan, difference,
            validity, status) === SccPixelResidualInsufficientValidPixels
    end
    @test all(iszero, residuals) && !any(shared) && only(fraction) === 0f0
end

@testset "SCC pixel-residual preparation and storage are bounded" begin
    @test_throws ArgumentError SccPixelResidualPlan(
        Float32[],
        Bool[],
        1.0f0,
    )
    @test_throws DimensionMismatch SccPixelResidualPlan(
        zeros(Float32, 2),
        Bool[true],
        1.0f0,
    )
    @test_throws ArgumentError SccPixelResidualPlan(
        Float32[0, NaN],
        trues(2),
        1.0f0,
    )
    @test_throws ArgumentError SccPixelResidualPlan(
        zeros(Float32, 2),
        falses(2),
        1.0f0,
    )
    for fraction in (0.0f0, 1.1f0, NaN32)
        @test_throws ArgumentError SccPixelResidualPlan(
            zeros(Float32, 2),
            trues(2),
            fraction,
        )
    end

    prepared = prepared_scc_pixel_residual()
    original_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :mask,
        Bool[true, false, false, false, false, false],
    )
    @test prepared.plan === original_plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "reference-image",
        Float32[0, 0, NaN, 0, 0, 0],
    )
    @test prepared.plan === original_plan

    outputs = (zeros(Float32, 4), falses(4), zeros(Float32, 1))
    inputs = (
        Float32[3, 100, 4, 2, 0, 1],
        trues(6),
        UInt8[UInt8(ChoppedImageComplete)],
    )
    process!(outputs..., prepared, inputs...)
    @test scc_pixel_residual_allocation_bytes(outputs, prepared, inputs) == 0
    @test_throws DimensionMismatch process!(
        outputs...,
        prepared,
        zeros(Float32, 5),
        trues(6),
        inputs[3],
    )
    @test_throws ArgumentError process!(
        @view(inputs[1][1:4]),
        outputs[2],
        outputs[3],
        prepared,
        inputs...,
    )
end

@testset "SCC pixel-residual public API exports" begin
    for name in (
        :SCC_RECONSTRUCTOR_V1,
        :SCC_REFERENCE_IMAGE_V1,
        :SCC_SELECTED_PIXEL_RESIDUALS_V1,
        :SCC_SELECTED_PIXEL_VALIDITY_V1,
        :SCC_SELECTED_PIXEL_VALID_FRACTION_V1,
        :SccPixelResidualF32,
        :SccPixelResidualF32Configuration,
        :SccPixelResidualFailure,
        :SccPixelResidualIncompletePair,
        :SccPixelResidualInsufficientValidPixels,
        :SccPixelResidualInvalidPairStatus,
        :SccPixelResidualPlan,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
