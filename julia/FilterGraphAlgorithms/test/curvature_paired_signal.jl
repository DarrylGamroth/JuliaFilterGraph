module CurvaturePairedSignalTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

struct OffsetAxisVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(values::OffsetAxisVector) = size(values.parent)
Base.axes(values::OffsetAxisVector) = (0:(length(values.parent) - 1),)
Base.IndexStyle(::Type{<:OffsetAxisVector}) = IndexLinear()
Base.getindex(values::OffsetAxisVector, index::Int) = values.parent[index + 1]

const FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "algorithms", "curvature_paired_signal_synthetic.toml"))

function fixture_image()
    geometry, frame = FIXTURE["geometry"], FIXTURE["frame"]
    return reshape(Float32.(frame["image"]), Tuple(geometry["image_size"])...)
end

function fixture_channels()
    geometry, frame = FIXTURE["geometry"], FIXTURE["frame"]
    return reshape(Float32.(frame["channels"]), 3, length(geometry["channel_offsets"]))
end

function fixture_calibration()
    geometry, calibration = FIXTURE["geometry"], FIXTURE["calibration"]
    size = Tuple(geometry["signal_size"])
    return (
        reshape(Bool.(geometry["support"]), size...),
        reshape(Float32.(calibration["reference_signal"]), size...),
        Tuple(Float32.(calibration["branch_scales"])),
        UInt64(calibration["signature"]),
    )
end

function image_plan(; reference=nothing, scales=nothing, signature=nothing)
    geometry = FIXTURE["geometry"]
    support, fixture_reference, fixture_scales, fixture_signature = fixture_calibration()
    return CurvaturePairedImagePlan(
        Tuple(geometry["image_size"]), Tuple(geometry["signal_size"]), Tuple(geometry["branch_size"]),
        (Tuple(geometry["positive_branch_origin"]), Tuple(geometry["negative_branch_origin"])),
        support, isnothing(reference) ? fixture_reference : reference,
        isnothing(scales) ? fixture_scales : scales,
        isnothing(signature) ? fixture_signature : signature,
    )
end

function channel_plan(; reference=nothing, scales=nothing, signature=nothing)
    geometry = FIXTURE["geometry"]
    support, fixture_reference, fixture_scales, fixture_signature = fixture_calibration()
    return CurvaturePairedChannelPlan(
        (3, length(geometry["channel_offsets"])), Tuple(geometry["signal_size"]),
        (geometry["positive_branch_row"], geometry["negative_branch_row"]), geometry["channel_offsets"],
        support, isnothing(reference) ? fixture_reference : reference,
        isnothing(scales) ? fixture_scales : scales,
        isnothing(signature) ? fixture_signature : signature,
    )
end

function image_configuration()
    geometry, calibration = FIXTURE["geometry"], FIXTURE["calibration"]
    return CurvaturePairedImageF32Configuration(
        image_size=(geometry["image_size"][1], geometry["image_size"][2]),
        signal_size=(geometry["signal_size"][1], geometry["signal_size"][2]),
        branch_size=(geometry["branch_size"][1], geometry["branch_size"][2]),
        positive_branch_origin=geometry["positive_branch_origin"],
        negative_branch_origin=geometry["negative_branch_origin"],
        curvature_support=Bool.(geometry["support"]),
        initial_reference_signal=Float32.(calibration["reference_signal"]),
        initial_branch_scales=Float32.(calibration["branch_scales"]),
        initial_calibration_signature=UInt64(calibration["signature"]),
    )
end

function channel_configuration()
    geometry, calibration = FIXTURE["geometry"], FIXTURE["calibration"]
    return CurvaturePairedChannelF32Configuration(
        input_branch_rows=3, input_channels=length(geometry["channel_offsets"]),
        signal_size=(geometry["signal_size"][1], geometry["signal_size"][2]),
        positive_branch_row=geometry["positive_branch_row"],
        negative_branch_row=geometry["negative_branch_row"], channel_offsets=geometry["channel_offsets"],
        curvature_support=Bool.(geometry["support"]),
        initial_reference_signal=Float32.(calibration["reference_signal"]),
        initial_branch_scales=Float32.(calibration["branch_scales"]),
        initial_calibration_signature=UInt64(calibration["signature"]),
    )
end

function oracle(plus, minus, support, reference, scales)
    result = zeros(Float32, length(plus))
    for index in eachindex(result)
        support[index] || continue
        positive, negative = scales[1] * plus[index], scales[2] * minus[index]
        total = positive + negative
        total <= eps(Float32) || (result[index] = (positive - negative) / (total + eps(Float32)) - reference[index])
    end
    return result
end

function allocation_bytes(signal, prepared, input, signature)
    process!(signal, prepared, input, signature)
    return @allocated process!(signal, prepared, input, signature)
end

@testset "synthetic paired curvature image and channels agree" begin
    support, reference, scales, signature_value = fixture_calibration()
    expected = Float32.(FIXTURE["expected"]["signal"])
    plus = Float32.(FIXTURE["frame"]["positive_reduced"])
    minus = Float32.(FIXTURE["frame"]["negative_reduced"])
    @test oracle(plus, minus, support, reference, scales) ≈ expected rtol=2f-7

    image = fixture_image()
    image_algorithm = image_plan()
    image_workspace = CurvaturePairedSignalWorkspace(image_algorithm)
    image_signal = fill(-8f0, 6)
    signature = UInt64[signature_value]
    @test @inferred(Union{Nothing,CurvaturePairedSignalFailure}, process!(image_signal, image_workspace, image_algorithm, image, signature)) === nothing
    @test image_signal ≈ expected rtol=2f-7

    channels = fixture_channels()
    channel_algorithm = channel_plan()
    channel_workspace = CurvaturePairedSignalWorkspace(channel_algorithm)
    channel_signal = fill(-8f0, 6)
    @test @inferred(Union{Nothing,CurvaturePairedSignalFailure}, process!(channel_signal, channel_workspace, channel_algorithm, channels, signature)) === nothing
    @test channel_signal ≈ expected rtol=2f-7
    @test channel_signal == image_signal
    @test channel_algorithm.channel_offsets == [4, 1, 5, 3, 0, 2]
    @test channel_signal[1] != channel_signal[2]
    @test channel_signal[3] == 0f0 && channel_signal[6] == 0f0
end

@testset "paired curvature declarations and calibration replacement" begin
    image_prepared = prepare_algorithm(CurvaturePairedImageF32, image_configuration())
    channel_prepared = prepare_algorithm(CurvaturePairedChannelF32, channel_configuration())
    @test getproperty.(algorithm_formats(image_prepared), :shape) == ((6, 8), (1,), (6,), (2, 3), (2, 3), (2,), (1,))
    @test getproperty.(algorithm_formats(channel_prepared), :shape) == ((3, 6), (1,), (6,), (2, 3), (2, 3), (2,), (1,))
    @test getproperty.(algorithm_formats(image_prepared), :schema) == (
        CURVATURE_PAIRED_IMAGE_V1, CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1,
        CURVATURE_PAIRED_SIGNAL_V1, CURVATURE_PAIRED_SUPPORT_V1,
        CURVATURE_PAIRED_REFERENCE_SIGNAL_V1, CURVATURE_PAIRED_BRANCH_SCALES_V1,
        CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1,
    )
    @test getproperty.(algorithm_formats(channel_prepared), :schema)[1] == CURVATURE_PAIRED_CHANNELS_V1
    for prepared in (image_prepared, channel_prepared)
        @test isconcretetype(typeof(prepared.plan))
        @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
        @test isconcretetype(typeof(prepared.workspace))
        @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
        old_plan = prepared.plan
        support = copy(old_plan.support); support[1] = !support[1]
        reference = copy(old_plan.reference_signal); reference[2] += 0.25f0
        @test replace_curvature_signal_calibration!(
            prepared, support, reference, Float32[0.5, 1.5], UInt64[9],
        ) === prepared
        @test prepared.plan !== old_plan
        @test prepared.plan.branch_scales == (0.5f0, 1.5f0)
        @test prepared.plan.calibration_signature == UInt64(9)
        retained = prepared.plan
        @test_throws ArgumentError replace_curvature_signal_calibration!(
            prepared, support, reference, Float32[0, 1], UInt64[10],
        )
        @test prepared.plan === retained
    end
end

@testset "paired curvature generic floating-point and wrapper execution" begin
    plan = CurvaturePairedImagePlan(
        (2, 4), (1, 2), (1, 2), ((0, 1), (1, 1)), trues(1, 2),
        zeros(Float64, 1, 2), (1.0, 1.0), UInt64(17),
    )
    workspace = CurvaturePairedSignalWorkspace(plan)
    image_storage = zeros(Float64, 4, 8)
    image = @view image_storage[1:2:3, 1:2:7]
    image[1, 2:3] .= (3.0, 5.0)
    image[2, 2:3] .= (1.0, 1.0)
    signal_storage = fill(-1.0, 4)
    signal = @view signal_storage[1:2:3]
    signature_storage = UInt64[17, 99]
    signature = @view signature_storage[1:1]

    @test @inferred(
        Union{Nothing,CurvaturePairedSignalFailure},
        process!(signal, workspace, plan, image, signature),
    ) === nothing
    @test signal ≈ [0.5, 2 / 3]
    @test @allocated(process!(signal, workspace, plan, image, signature)) == 0
end

@testset "paired curvature validation, failures, and atomic publication" begin
    image_prepared = prepare_algorithm(CurvaturePairedImageF32, image_configuration())
    channel_prepared = prepare_algorithm(CurvaturePairedChannelF32, channel_configuration())
    image, channels = fixture_image(), fixture_channels()
    signature = UInt64[image_prepared.plan.calibration_signature]
    for (prepared, input) in ((image_prepared, image), (channel_prepared, channels))
        signal = fill(17f0, 6)
        @test process!(signal, prepared, input, UInt64[signature[1] + 1]) === CurvaturePairedSignalCalibrationMismatch
        @test signal == fill(17f0, 6)
        nonfinite = copy(input); nonfinite[end] = NaN32
        @test process!(signal, prepared, nonfinite, signature) === CurvaturePairedSignalNonFiniteInput
        @test signal == fill(17f0, 6)
        @test allocation_bytes(signal, prepared, input, signature) == 0
        @test allocation_bytes(signal, prepared, input, UInt64[signature[1] + 1]) == 0
        @test allocation_bytes(signal, prepared, nonfinite, signature) == 0
    end
    support, reference, _, signature_value = fixture_calibration()
    output_failure_reference = copy(reference); output_failure_reference[1] = -floatmax(Float32)
    output_failure = image_plan(reference=output_failure_reference, scales=(floatmax(Float32), 1f0))
    signal = fill(17f0, 6)
    @test process!(signal, CurvaturePairedSignalWorkspace(output_failure), output_failure, image, UInt64[signature_value]) === CurvaturePairedSignalNonFiniteOutput
    @test signal == fill(17f0, 6)
    low_image = zeros(Float32, 6, 8)
    low_image[1, 2] = -1f0
    low_channels = zeros(Float32, 3, 6); low_channels[1, 5] = -1f0
    @test process!(signal, image_prepared, low_image, signature) === nothing
    @test signal == zeros(Float32, 6)
    fill!(signal, 17f0)
    @test process!(signal, channel_prepared, low_channels, signature) === nothing
    @test signal == zeros(Float32, 6)
    @test_throws DimensionMismatch CurvaturePairedImagePlan((6, 8), (3, 3), (4, 6), ((0, 1), (2, 1)), support, reference, (1f0, 1f0), signature_value)
    @test_throws DimensionMismatch CurvaturePairedChannelPlan((3, 6), (2, 3), (0, 1), [0, 1], support, reference, (1f0, 1f0), signature_value)
    offset_channels = OffsetAxisVector([4, 1, 5, 3, 0, 2])
    normalized_plan = CurvaturePairedChannelPlan(
        (3, 6), (2, 3), (0, 1), offset_channels, support, reference,
        (1f0, 1f0), signature_value,
    )
    @test axes(normalized_plan.channel_offsets) == (Base.OneTo(6),)
    @test normalized_plan.channel_offsets == [4, 1, 5, 3, 0, 2]
    @test_throws DimensionMismatch prepare_algorithm(
        CurvaturePairedImageF32,
        CurvaturePairedImageF32Configuration(
            image_size=(6, 8), signal_size=(3, 3),
            branch_size=(4, 6), positive_branch_origin=[0, 1],
            negative_branch_origin=[2, 1], curvature_support=nothing,
            initial_reference_signal=nothing, initial_branch_scales=nothing,
            initial_calibration_signature=signature_value,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        CurvaturePairedChannelF32,
        CurvaturePairedChannelF32Configuration(
            input_branch_rows=3, input_channels=6, signal_size=(2, 3),
            positive_branch_row=0, negative_branch_row=1, channel_offsets=[0],
            curvature_support=nothing, initial_reference_signal=nothing,
            initial_branch_scales=nothing, initial_calibration_signature=signature_value,
        ),
    )
    @test_throws DimensionMismatch process!(signal, image_prepared, image, UInt64[])
    for (plan, input) in ((image_prepared.plan, image), (channel_prepared.plan, channels))
        for extent in (5, 7)
            direct_signal = fill(17f0, extent)
            direct_workspace = CurvaturePairedSignalWorkspace(zeros(Float32, extent))
            @test_throws DimensionMismatch process!(
                direct_signal, direct_workspace, plan, input, signature,
            )
            @test direct_signal == fill(17f0, extent)
        end
    end
    @test_throws ArgumentError process!(image_prepared.workspace.staged_signal, image_prepared, image, signature)
    @test_throws ArgumentError process!(
        @view(vec(image)[1:6]), image_prepared, image, signature,
    )
end

@testset "paired curvature public interface exports" begin
    for name in (
        :CurvaturePairedImageF32, :CurvaturePairedImageF32Configuration,
        :CurvaturePairedImagePlan, :CurvaturePairedChannelF32,
        :CurvaturePairedChannelF32Configuration, :CurvaturePairedChannelPlan,
        :CurvaturePairedSignalCalibrationMismatch, :CurvaturePairedSignalFailure,
        :CurvaturePairedSignalNonFiniteInput, :CurvaturePairedSignalNonFiniteOutput,
        :CurvaturePairedSignalWorkspace, :replace_curvature_signal_calibration!,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "curvature normalization preserves finite extreme branch ratios" begin
    for T in (Float32, Float64)
        signature = UInt64(7)
        image_plan = CurvaturePairedImagePlan((2, 1), (1, 1), (1, 1), ((0, 0), (1, 0)),
            trues(1, 1), fill(T(0.125), 1, 1), (one(T), one(T)), signature)
        channel_plan = CurvaturePairedChannelPlan((2, 1), (1, 1), (0, 1), [0],
            trues(1, 1), fill(T(0.125), 1, 1), (one(T), one(T)), signature)
        maximum = floatmax(T)
        branch_pairs = ((maximum, maximum / T(2)), (maximum, maximum),
            (maximum, -maximum / T(2)), (-maximum, -maximum),
            (eps(T) / T(2), eps(T) / T(2)), (eps(T), eps(T)), (zero(T), zero(T)))
        for (positive, negative) in branch_pairs, plan in (image_plan, channel_plan)
            observation = reshape(T[positive, negative], 2, 1)
            output = T[71]
            total = BigFloat(positive) + BigFloat(negative)
            expected = total <= BigFloat(eps(T)) ? zero(BigFloat) :
                (BigFloat(positive) - BigFloat(negative)) / (total + BigFloat(eps(T))) - BigFloat(T(0.125))
            @test process!(output, CurvaturePairedSignalWorkspace(plan), plan, observation, UInt64[signature]) === nothing
            @test isapprox(only(output), T(expected); rtol=T(8) * eps(T), atol=zero(T))
        end
    end
end


end
