module ZernikePupilSignalTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "algorithms", "zernike_pupil_signal_synthetic.toml"))

function fixture_image(::Type{T}=Float32) where {T<:AbstractFloat}
    return reshape(T.(FIXTURE["frame"]["image"]), Tuple(FIXTURE["geometry"]["image_size"])...)
end

function fixture_support()
    return reshape(Bool.(FIXTURE["geometry"]["support"]), Tuple(FIXTURE["geometry"]["image_size"])...)
end

function fixture_reference(::Type{T}=Float32) where {T<:AbstractFloat}
    return reshape(
        T.(FIXTURE["calibration"]["reference_pupil_image"]),
        Tuple(FIXTURE["geometry"]["image_size"])...,
    )
end

function fixture_plan(
    ::Type{T}=Float32;
    normalization=ZernikePupilSignalMeanValidFlux,
    incidence_flux=zero(T),
    signature=UInt64(FIXTURE["calibration"]["signature"]),
) where {T<:AbstractFloat}
    return ZernikePupilSignalPlan(
        Tuple(FIXTURE["geometry"]["image_size"]), fixture_support(), fixture_reference(T),
        normalization, signature, incidence_flux,
    )
end

function fixture_configuration(
    ;
    normalization_policy="mean-valid-flux",
    incidence_flux=0.0f0,
    signature=UInt64(FIXTURE["calibration"]["signature"]),
)
    geometry = FIXTURE["geometry"]
    return ZernikePupilSignalF32Configuration(
        image_size=(geometry["image_size"][1], geometry["image_size"][2]),
        zernike_pupil_support=Bool.(geometry["support"]),
        initial_reference_pupil_image=Float32.(FIXTURE["calibration"]["reference_pupil_image"]),
        normalization_policy=normalization_policy, incidence_flux=incidence_flux,
        initial_calibration_signature=signature,
    )
end

function direct_signal(image, support, reference, divisor)
    signal = eltype(reference)[]
    for index in eachindex(support)
        support[index] && push!(signal, image[index] / divisor - reference[index])
    end
    return signal
end

function allocation_bytes(signal, divisor, workspace, plan, image, signature)
    process!(signal, divisor, workspace, plan, image, signature)
    return @allocated process!(signal, divisor, workspace, plan, image, signature)
end

@testset "synthetic complete-frame Zernike pupil calibration" begin
    plan = fixture_plan()
    workspace = ZernikePupilSignalWorkspace(plan)
    image = fixture_image()
    signal = fill(-9.0f0, 6)
    divisor = fill(-9.0f0, 1)
    signature = UInt64[plan.calibration_signature]
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal, divisor, workspace, plan, image, signature)) === nothing
    mean_case = FIXTURE["mean_valid_flux"]
    @test divisor ≈ Float32[mean_case["normalization_divisor"]] rtol=2f-7
    @test signal ≈ Float32.(mean_case["signal"]) rtol=2f-7
    @test plan.selected_offsets == [0, 1, 4, 5, 6, 8]
    @test signal ≈ direct_signal(image, plan.support, plan.reference_pupil_image, divisor[1]) rtol=2f-7

    incidence_case = FIXTURE["incidence_flux"]
    incidence_plan = fixture_plan(
        normalization=ZernikePupilSignalIncidenceFlux,
        incidence_flux=Float32(incidence_case["normalization_divisor"]),
    )
    incidence_workspace = ZernikePupilSignalWorkspace(incidence_plan)
    @test process!(signal, divisor, incidence_workspace, incidence_plan, image, signature) === nothing
    @test divisor == Float32[incidence_case["normalization_divisor"]]
    @test signal ≈ Float32.(incidence_case["signal"]) rtol=2f-7

    plan64 = fixture_plan(Float64)
    workspace64 = ZernikePupilSignalWorkspace(plan64)
    signal64 = zeros(Float64, 6)
    divisor64 = zeros(Float64, 1)
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal64, divisor64, workspace64, plan64, fixture_image(Float64), signature)) === nothing
    @test signal64 ≈ Float64.(mean_case["signal"]) rtol=2e-14

    integer_signal = zeros(Float32, 6)
    integer_divisor = zeros(Float32, 1)
    integer_image = Int.(image)
    @test process!(integer_signal, integer_divisor, workspace, plan, integer_image, signature) === nothing
    @test integer_signal ≈ Float32.(mean_case["signal"]) rtol=2f-7
    @test allocation_bytes(signal, divisor, workspace, plan, image, signature) == 0
end

@testset "complete-frame Zernike pupil signal declaration and replacement" begin
    prepared = prepare_algorithm(ZernikePupilSignalF32, fixture_configuration())
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3, 3), (1,), (6,), (1,), (3, 3), (3, 3), (1,), (1,))
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1, ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1,
        ZERNIKE_PUPIL_SIGNAL_V1, ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1,
        ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1, ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1,
        ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1,
        ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1,
    )
    @test prepared.plan isa ZernikePupilSignalPlan{Float32}
    @test prepared.workspace isa ZernikePupilSignalWorkspace{Float32}
    @test isconcretetype(typeof(prepared.plan))
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))

    support_storage = falses(6, 3)
    support_view = @view support_storage[1:2:5, :]
    support_view .= fixture_support()
    view_plan = ZernikePupilSignalPlan(
        (3, 3), support_view, fixture_reference(), ZernikePupilSignalMeanValidFlux,
        UInt64(1),
    )
    @test view_plan.selected_offsets == [0, 1, 4, 5, 6, 8]

    storage = fill(-7.0f0, 12)
    signal = @view storage[1:2:end]
    divisor_storage = fill(-7.0f0, 2)
    divisor = @view divisor_storage[2:2]
    frame_storage = zeros(Float32, 6, 3)
    image = @view frame_storage[1:2:5, :]
    image .= fixture_image()
    signature = UInt64[prepared.plan.calibration_signature]
    @test process!(signal, divisor, prepared, image, signature) === nothing
    @test signal ≈ Float32.(FIXTURE["mean_valid_flux"]["signal"]) rtol=2f-7
    @test all(==(-7.0f0), @view(storage[2:2:end]))

    old_plan = prepared.plan
    reference = copy(old_plan.reference_pupil_image); reference[1] = 0.1f0
    @test replace_parameter!(prepared, :"reference-pupil-image", reference) === prepared
    @test prepared.plan !== old_plan
    old_plan = prepared.plan
    support = Bool[false true false; true true true; true false true]
    @test replace_parameter!(prepared, :"zernike-pupil-support", support) === prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.selected_offsets == [1, 2, 3, 4, 7, 8]
    old_plan = prepared.plan
    @test replace_parameter!(prepared, :"incidence-flux", Float32[13.5]) === prepared
    @test prepared.plan !== old_plan
    old_plan = prepared.plan
    @test replace_parameter!(prepared, :"calibration-signature", UInt64[9]) === prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.calibration_signature == UInt64(9)

    old_plan = prepared.plan
    replacement_support = fixture_support()
    replacement_reference = fixture_reference(); replacement_reference[2] = 0.2f0
    @test replace_zernike_pupil_signal_calibration!(
        prepared, replacement_support, replacement_reference, Float32[13.5], UInt64[17],
    ) === prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.calibration_signature == UInt64(17)
    retained_plan = prepared.plan
    @test_throws DimensionMismatch replace_zernike_pupil_signal_calibration!(
        prepared, falses(3, 3), replacement_reference, Float32[13.5], UInt64[18],
    )
    @test prepared.plan === retained_plan
    @test_throws ArgumentError prepare_algorithm(
        ZernikePupilSignalF32, fixture_configuration(normalization_policy="unexpected"),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        ZernikePupilSignalF32,
        ZernikePupilSignalF32Configuration(
            image_size=(3, 3), zernike_pupil_support=nothing,
            initial_reference_pupil_image=zeros(Float32, 8),
            normalization_policy="mean-valid-flux", incidence_flux=0.0f0,
            initial_calibration_signature=UInt64(1),
        ),
    )
end

@testset "complete-frame Zernike pupil signal failures are output-atomic and allocation-free" begin
    plan = fixture_plan()
    workspace = ZernikePupilSignalWorkspace(plan)
    image = fixture_image()
    signature = UInt64[plan.calibration_signature]
    signal = fill(17.0f0, 6)
    divisor = fill(19.0f0, 1)
    @test process!(signal, divisor, workspace, plan, image, UInt64[signature[1] + 1]) === ZernikePupilSignalCalibrationMismatch
    @test signal == fill(17.0f0, 6)
    @test divisor == Float32[19]
    bad_input = copy(image); bad_input[3, 1] = NaN32
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal, divisor, workspace, plan, bad_input, signature)) === ZernikePupilSignalNonFiniteInput
    @test signal == fill(17.0f0, 6)
    @test divisor == Float32[19]
    @test_throws DimensionMismatch process!(signal, divisor, workspace, plan, zeros(Float32, 2, 3), signature)
    @test_throws ArgumentError process!(signal, divisor, workspace, plan, plan.reference_pupil_image, signature)
    alias_storage = zeros(Float32, 6)
    @test_throws ArgumentError process!(
        alias_storage, @view(alias_storage[1:1]), workspace, plan, image, signature,
    )
    @test_throws ArgumentError ZernikePupilSignalPlan((3, 3), falses(3, 3), zeros(Float32, 3, 3), ZernikePupilSignalMeanValidFlux, UInt64(1))
    @test_throws ArgumentError ZernikePupilSignalPlan((3, 3), trues(3, 3), fill(NaN32, 3, 3), ZernikePupilSignalMeanValidFlux, UInt64(1))
    @test_throws ArgumentError ZernikePupilSignalPlan((3, 3), trues(3, 3), zeros(Float32, 3, 3), ZernikePupilSignalIncidenceFlux, UInt64(1), -1.0f0)

    low_plan = fixture_plan(normalization=ZernikePupilSignalIncidenceFlux, incidence_flux=0.0f0)
    low_workspace = ZernikePupilSignalWorkspace(low_plan)
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal, divisor, low_workspace, low_plan, zeros(Float32, 3, 3), signature)) === nothing
    @test signal == zeros(Float32, 6)
    @test divisor == Float32[0]
    zero_plan = fixture_plan()
    zero_workspace = ZernikePupilSignalWorkspace(zero_plan)
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal, divisor, zero_workspace, zero_plan, zeros(Float32, 3, 3), signature)) === nothing
    @test signal == zeros(Float32, 6)
    @test divisor == Float32[eps(Float32)]

    overflow_plan = fixture_plan(
        normalization=ZernikePupilSignalIncidenceFlux, incidence_flux=1.0f0,
    )
    overflow_reference = copy(overflow_plan.reference_pupil_image)
    overflow_reference[1] = -floatmax(Float32)
    overflow_plan = ZernikePupilSignalPlan(
        overflow_plan.image_size, overflow_plan.support, overflow_reference,
        overflow_plan.normalization, overflow_plan.calibration_signature, overflow_plan.incidence_flux,
    )
    overflow_workspace = ZernikePupilSignalWorkspace(overflow_plan)
    fill!(signal, 17.0f0); fill!(divisor, 19.0f0)
    overflow_image = ones(Float32, 3, 3); overflow_image[1] = floatmax(Float32)
    @test @inferred(Union{Nothing,ZernikePupilSignalFailure}, process!(signal, divisor, overflow_workspace, overflow_plan, overflow_image, signature)) === ZernikePupilSignalNonFiniteOutput
    @test signal == fill(17.0f0, 6)
    @test divisor == Float32[19]
    @test allocation_bytes(signal, divisor, workspace, plan, image, signature) == 0
    @test allocation_bytes(signal, divisor, workspace, plan, image, UInt64[signature[1] + 1]) == 0
    @test allocation_bytes(signal, divisor, workspace, plan, bad_input, signature) == 0
    @test allocation_bytes(signal, divisor, low_workspace, low_plan, zeros(Float32, 3, 3), signature) == 0
    @test allocation_bytes(signal, divisor, zero_workspace, zero_plan, zeros(Float32, 3, 3), signature) == 0
    @test allocation_bytes(signal, divisor, overflow_workspace, overflow_plan, overflow_image, signature) == 0
end

@testset "complete-frame Zernike pupil signal public interface exports" begin
    for name in (
        :ZernikePupilSignalF32, :ZernikePupilSignalF32Configuration,
        :ZernikePupilSignalCalibrationMismatch, :ZernikePupilSignalFailure,
        :ZernikePupilSignalIncidenceFlux, :ZernikePupilSignalInvalidNormalization,
        :ZernikePupilSignalMeanValidFlux, :ZernikePupilSignalNonFiniteInput,
        :ZernikePupilSignalNonFiniteOutput, :ZernikePupilSignalNormalization,
        :ZernikePupilSignalPlan, :ZernikePupilSignalWorkspace,
        :replace_zernike_pupil_signal_calibration!,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
