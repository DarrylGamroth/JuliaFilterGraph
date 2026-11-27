module BiOEdgeImageTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "algorithms", "bi_o_edge_image_synthetic.toml"))

function fixture_image(::Type{T}=Float32) where {T<:AbstractFloat}
    geometry = FIXTURE["geometry"]
    return reshape(T.(FIXTURE["frame"]["image"]), geometry["image_size"]...)
end

function fixture_plan(
    ::Type{T}=Float32;
    signature=UInt64(FIXTURE["calibration"]["signature"]),
    normalization=BiOEdgeImageMeanValidFlux,
    incidence_flux=zero(T),
) where {T<:AbstractFloat}
    geometry = FIXTURE["geometry"]
    calibration = FIXTURE["calibration"]
    support = reshape(Bool.(geometry["support"]), geometry["pupil_size"]...)
    selected = count(support)
    reference = reshape(T.(calibration["reference_signal"]), selected, 2)
    gain = reshape(T.(calibration["optical_gain"]), selected, 2)
    origins = ntuple(index -> Tuple(geometry["pupil_origins"][index]), 4)
    return BiOEdgeImagePlan(
        Tuple(geometry["image_size"]), Tuple(geometry["pupil_size"]),
        origins, support, reference, gain, normalization, signature, incidence_flux,
    )
end

function fixture_configuration(
    ;
    signature=UInt64(FIXTURE["calibration"]["signature"]),
    normalization_policy=FIXTURE["mean_valid_flux"]["normalization_policy"],
    incidence_flux=0.0f0,
)
    geometry = FIXTURE["geometry"]
    calibration = FIXTURE["calibration"]
    return BiOEdgeImageF32Configuration(
        image_size=(geometry["image_size"][1], geometry["image_size"][2]),
        pupil_size=(geometry["pupil_size"][1], geometry["pupil_size"][2]),
        pupil_origins=geometry["pupil_origins"], bi_o_edge_support=Bool.(geometry["support"]),
        initial_reference_signal=Float32.(calibration["reference_signal"]),
        initial_optical_gain=Float32.(calibration["optical_gain"]),
        normalization_policy=normalization_policy, incidence_flux=incidence_flux,
        initial_calibration_signature=signature,
    )
end

function allocation_bytes(signal, divisor, prepared, image, signature)
    process!(signal, divisor, prepared, image, signature)
    return @allocated process!(signal, divisor, prepared, image, signature)
end

@testset "synthetic complete-frame Bi-O-edge image calibration" begin
    plan = fixture_plan()
    workspace = BiOEdgeImageWorkspace(plan)
    image = fixture_image()
    signature = UInt64[plan.calibration_signature]
    signal = fill(-9.0f0, 3, 2)
    divisor = fill(-9.0f0, 1)
    @test @inferred(Union{Nothing,BiOEdgeImageFailure}, process!(signal, divisor, workspace, plan, image, signature)) === nothing
    mean_case = FIXTURE["mean_valid_flux"]
    @test divisor ≈ Float32[mean_case["normalization_divisor"]] rtol=2f-7
    @test vec(signal) ≈ Float32.(mean_case["signal"]) rtol=2f-7

    pupil_rows = plan.pupil_size[1]
    direct_raw_x = Float32[]
    direct_raw_y = Float32[]
    q1, q2, q3, q4 = plan.origins
    for offset in plan.selected_offsets
        row = rem(offset, pupil_rows) + 1
        column = div(offset, pupil_rows) + 1
        value1 = image[q1[1] + row, q1[2] + column]
        value2 = image[q2[1] + row, q2[2] + column]
        value3 = image[q3[1] + row, q3[2] + column]
        value4 = image[q4[1] + row, q4[2] + column]
        push!(direct_raw_x, value1 - value2 + value4 - value3)
        push!(direct_raw_y, value1 - value4 + value2 - value3)
    end
    @test direct_raw_x == Float32.(mean_case["raw_x"])
    @test direct_raw_y == Float32.(mean_case["raw_y"])

    incidence_case = FIXTURE["incidence_flux"]
    incidence_plan = fixture_plan(
        normalization=BiOEdgeImageIncidenceFlux,
        incidence_flux=Float32(incidence_case["normalization_divisor"]),
    )
    incidence_workspace = BiOEdgeImageWorkspace(incidence_plan)
    @test process!(
        signal,
        divisor,
        incidence_workspace,
        incidence_plan,
        image,
        signature,
    ) === nothing
    @test divisor == Float32[incidence_case["normalization_divisor"]]
    @test vec(signal) ≈ Float32.(incidence_case["signal"]) rtol=2f-7

    plan64 = fixture_plan(Float64)
    workspace64 = BiOEdgeImageWorkspace(plan64)
    signal64 = zeros(Float64, 3, 2)
    divisor64 = zeros(Float64, 1)
    @test @inferred(Union{Nothing,BiOEdgeImageFailure}, process!(signal64, divisor64, workspace64, plan64, fixture_image(Float64), signature)) === nothing
    @test vec(signal64) ≈ Float64.(mean_case["signal"]) rtol=2e-7
end

@testset "complete-frame Bi-O-edge declaration and replacement" begin
    prepared = prepare_algorithm(BiOEdgeImageF32, fixture_configuration())
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((4, 4), (1,), (3, 2), (1,), (4, 2), (2, 2), (3, 2), (3, 2), (1,))
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1, BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1,
        BI_O_EDGE_SIGNAL_V1, BI_O_EDGE_NORMALIZATION_DIVISOR_V1, REGION_ORIGINS_V1,
        BI_O_EDGE_SUPPORT_V1, BI_O_EDGE_REFERENCE_SIGNAL_V1, BI_O_EDGE_OPTICAL_GAIN_V1,
        BI_O_EDGE_CALIBRATION_SIGNATURE_V1,
    )
    @test prepared.plan isa BiOEdgeImagePlan{Float32}
    @test prepared.workspace isa BiOEdgeImageWorkspace{Float32}
    @test isconcretetype(typeof(prepared.plan))
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    @test prepared.plan.selected_offsets == [0, 2, 3]

    support_storage = falses(4, 4)
    support_view = @view support_storage[1:2:3, 1:2:3]
    support_view .= Bool[true true; false true]
    view_plan = BiOEdgeImagePlan(
        prepared.plan.image_size, prepared.plan.pupil_size, prepared.plan.origins,
        support_view, prepared.plan.reference_signal, prepared.plan.optical_gain,
        prepared.plan.normalization, prepared.plan.calibration_signature,
    )
    @test view_plan.selected_offsets == [0, 2, 3]

    image = fixture_image()
    storage = fill(-7.0f0, 6, 2)
    signal = @view storage[1:2:end, :]
    divisor_storage = fill(-7.0f0, 2)
    divisor = @view divisor_storage[2:2]
    signature = UInt64[prepared.plan.calibration_signature]
    @test process!(signal, divisor, prepared, image, signature) === nothing
    @test vec(signal) ≈ Float32.(FIXTURE["mean_valid_flux"]["signal"]) rtol=2f-7
    @test all(==(-7.0f0), @view(storage[2:2:end, :]))

    old_plan = prepared.plan
    reference = copy(old_plan.reference_signal); reference[1, 1] = 0.1f0
    @test replace_parameter!(prepared, :"reference-signal", reference) === prepared
    @test prepared.plan !== old_plan
    old_plan = prepared.plan
    gain = copy(old_plan.optical_gain); gain[1, 1] = 0.5f0
    @test replace_parameter!(prepared, :"optical-gain", gain) === prepared
    @test prepared.plan !== old_plan
    old_plan = prepared.plan
    support = Bool[false true; true true]
    @test replace_parameter!(prepared, :"bi-o-edge-support", support) === prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.selected_offsets == [1, 2, 3]
    old_plan = prepared.plan
    @test replace_parameter!(prepared, :"calibration-signature", UInt64[9]) === prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.calibration_signature == UInt64(9)
    retained_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(prepared, :"bi-o-edge-support", Bool[true false; false true])
    @test prepared.plan === retained_plan
end

@testset "complete-frame Bi-O-edge failures are output-atomic and allocation-free" begin
    prepared = prepare_algorithm(BiOEdgeImageF32, fixture_configuration())
    image = fixture_image()
    signature = UInt64[prepared.plan.calibration_signature]
    signal = fill(17.0f0, 3, 2)
    divisor = fill(19.0f0, 1)
    wrong_signature = UInt64[signature[1] + 1]
    @test process!(signal, divisor, prepared, image, wrong_signature) === BiOEdgeImageCalibrationMismatch
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]
    bad_input = copy(image); bad_input[2, 1] = NaN32
    @test process!(signal, divisor, prepared, bad_input, signature) === BiOEdgeImageNonFiniteInput
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]
    zero_image = zeros(Float32, 4, 4)
    @test process!(signal, divisor, prepared, zero_image, signature) === BiOEdgeImageInvalidNormalization
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]
    @test_throws ArgumentError prepare_algorithm(
        BiOEdgeImageF32,
        fixture_configuration(normalization_policy="incidence-flux", incidence_flux=0.0f0),
    )
    @test_throws ArgumentError prepare_algorithm(
        BiOEdgeImageF32,
        fixture_configuration(normalization_policy="incidence-flux", incidence_flux=eps(Float32)),
    )
    low_flux = prepare_algorithm(
        BiOEdgeImageF32,
        fixture_configuration(
            normalization_policy="incidence-flux", incidence_flux=nextfloat(eps(Float32)),
        ),
    )
    @test process!(signal, divisor, low_flux, image, signature) === nothing
    overflow_reference = zeros(Float32, 3, 2)
    overflow_reference[:, 1] .= -4.0f0
    overflow_plan = BiOEdgeImagePlan(
        prepared.plan.image_size, prepared.plan.pupil_size, prepared.plan.origins,
        prepared.plan.support, overflow_reference,
        fill(floatmax(Float32), 3, 2), BiOEdgeImageMeanValidFlux,
        prepared.plan.calibration_signature,
    )
    overflow_workspace = BiOEdgeImageWorkspace(overflow_plan)
    fill!(signal, 17.0f0)
    fill!(divisor, 19.0f0)
    @test process!(signal, divisor, overflow_workspace, overflow_plan, image, signature) === BiOEdgeImageNonFiniteOutput
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]
    @test_throws DimensionMismatch process!(signal, divisor, prepared, image, UInt64[])
    @test_throws ArgumentError process!(prepared.plan.reference_signal, divisor, prepared, image, signature)
    @test_throws ArgumentError process!(signal, @view(signal[1:1]), prepared, image, signature)
    @test allocation_bytes(signal, divisor, prepared, image, signature) == 0
    @test allocation_bytes(signal, divisor, prepared, image, wrong_signature) == 0
    @test allocation_bytes(signal, divisor, prepared, bad_input, signature) == 0
    @test allocation_bytes(signal, divisor, prepared, zero_image, signature) == 0
end

@testset "complete-frame Bi-O-edge public interface exports" begin
    for name in (
        :BiOEdgeImageF32, :BiOEdgeImageF32Configuration, :BiOEdgeImageCalibrationMismatch,
        :BiOEdgeImageFailure, :BiOEdgeImageIncidenceFlux, :BiOEdgeImageInvalidNormalization,
        :BiOEdgeImageMeanValidFlux, :BiOEdgeImageNonFiniteInput,
        :BiOEdgeImageNonFiniteOutput, :BiOEdgeImageNormalization, :BiOEdgeImagePlan,
        :BiOEdgeImageWorkspace,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "overflowing raw Bi-O-edge difference has finite normalization" begin
    flux = floatmax(Float32)
    plan = BiOEdgeImagePlan((2, 2), (1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)),
        trues(1, 1), zeros(Float32, 1, 2), ones(Float32, 1, 2),
        BiOEdgeImageIncidenceFlux, UInt64(1), flux)
    image = Float32[0.75f0 * flux 0; -0.75f0 * flux 0]
    signal, divisor = zeros(Float32, 1, 2), zeros(Float32, 1)
    @test process!(signal, divisor, BiOEdgeImageWorkspace(plan), plan, image, UInt64[1]) === nothing
    @test signal[1, 1] ≈ 1.5f0
    @test only(divisor) == flux
end

end
