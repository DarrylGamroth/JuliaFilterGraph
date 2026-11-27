module PyramidImageTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "algorithms", "pyramid_image_synthetic.toml"))

row_major_matrix(::Type{T}, values, rows, columns) where {T} =
    permutedims(reshape(T.(values), columns, rows))

function fixture_image()
    geometry = FIXTURE["geometry"]
    return reshape(
        Float32.(FIXTURE["frame"]["image"]),
        geometry["image_size"]...,
    )
end

function fixture_configuration(policy; incidence_flux=0.0f0)
    geometry = FIXTURE["geometry"]
    calibration = FIXTURE["calibration"]
    return PyramidImageF32Configuration(
        image_size=(geometry["image_size"][1], geometry["image_size"][2]),
        pupil_size=(geometry["pupil_size"][1], geometry["pupil_size"][2]),
        pupil_origins=geometry["pupil_origins"],
        i4q_support=Bool.(geometry["i4q_support"]),
        initial_reference_signal=Float32.(calibration["reference_signal"]),
        initial_optical_gain=Float32.(calibration["optical_gain"]),
        normalization_policy=policy,
        incidence_flux=incidence_flux,
    )
end

function allocation_bytes(signal, divisor, prepared, image)
    process!(signal, divisor, prepared, image)
    return @allocated process!(signal, divisor, prepared, image)
end

# Frozen lane-major reference from 6c8ccd2. A complete-frame call can reject
# after reduction, so compare the intermediate partials as well as the divisor.
function reference_mean_valid_flux!(partials, plan, image)
    fill!(partials, zero(eltype(partials)))
    rows = plan.pupil_size[1]
    selected_count = length(plan.selected_offsets)
    q1, q2, q3, q4 = plan.origins
    @inbounds for lane in 1:256
        selected = lane
        lane_sum = zero(eltype(partials))
        while selected <= selected_count
            offset = plan.selected_offsets[selected]
            row = rem(offset, rows) + 1
            column = div(offset, rows) + 1
            flux = image[q1[1] + row, q1[2] + column] +
                   image[q2[1] + row, q2[2] + column] +
                   image[q3[1] + row, q3[2] + column] +
                   image[q4[1] + row, q4[2] + column]
            lane_sum += flux
            selected += 256
        end
        partials[lane] = lane_sum
    end
    stride = 128
    @inbounds while stride > 0
        for lane in 1:stride
            partials[lane] += partials[lane + stride]
        end
        stride = div(stride, 2)
    end
    return partials[1] / eltype(partials)(selected_count)
end

@testset "chunk traversal preserves every normalization lane and tree bit" begin
    rows, columns = 60, 60
    origins = ((0, 0), (rows, 0), (rows, columns), (0, columns))
    image = reshape(Float32[1f0 + mod(19i, 127) / 64f0 for i in 1:4rows*columns],
        2rows, 2columns)
    for selected_count in (1, 3, 255, 256, 257, 511, 512, 513, 2776, 3600)
        support = falses(rows * columns)
        support[1:selected_count] .= true
        support = reshape(support, rows, columns)
        plan = PyramidImagePlan((2rows, 2columns), (rows, columns), origins,
            support, zeros(Float32, selected_count, 2),
            ones(Float32, selected_count, 2), PyramidImageMeanValidFlux)
        for input in (image, transpose(permutedims(image)))
            baseline_partials = zeros(Float32, 256)
            candidate_partials = similar(baseline_partials)
            baseline = reference_mean_valid_flux!(baseline_partials, plan, input)
            candidate = FilterGraphAlgorithms._pyramid_image_mean_valid_flux!(
                candidate_partials, plan, input)
            @test reinterpret(UInt32, baseline) == reinterpret(UInt32, candidate)
            @test reinterpret(UInt32, baseline_partials) ==
                  reinterpret(UInt32, candidate_partials)
        end
    end
end

@testset "synthetic complete-frame pyramid image calibration" begin
    geometry = FIXTURE["geometry"]
    calibration = FIXTURE["calibration"]
    image = fixture_image()
    support = reshape(Bool.(geometry["i4q_support"]), 2, 2)
    reference = reshape(Float32.(calibration["reference_signal"]), 3, 2)
    gain = reshape(Float32.(calibration["optical_gain"]), 3, 2)
    plan = PyramidImagePlan(
        (4, 4),
        (2, 2),
        ((0, 0), (2, 0), (2, 2), (0, 2)),
        support,
        reference,
        gain,
        PyramidImageMeanValidFlux,
    )
    workspace = PyramidImageWorkspace(plan)
    signal = fill(-9.0f0, 3, 2)
    divisor = fill(-9.0f0, 1)

    @test @inferred(Union{Nothing,PyramidImageFailure}, process!(
        signal,
        divisor,
        workspace,
        plan,
        image,
    )) === nothing
    mean_case = FIXTURE["mean_valid_flux"]
    @test divisor ≈ Float32[mean_case["divisor"]] rtol=2f-7
    @test vec(signal) ≈ Float32.(mean_case["signal"]) rtol=2f-7

    @test_throws DimensionMismatch PyramidImagePlan(
        (4, 4), (2, 2), ((-1, 0), (2, 0), (2, 2), (0, 2)), support,
        reference, gain, PyramidImageMeanValidFlux,
    )
    @test_throws DimensionMismatch PyramidImagePlan(
        (4, 4), (2, 2), ((typemax(Int), 0), (2, 0), (2, 2), (0, 2)),
        support, reference, gain, PyramidImageMeanValidFlux,
    )

    FilterGraphAlgorithms.reset!(workspace)
    second = copy(signal)
    process!(second, divisor, workspace, plan, image)
    @test second == signal # no previous-frame behavior, including reset

    incidence_case = FIXTURE["incidence_flux"]
    incidence_plan = PyramidImagePlan(
        (4, 4), (2, 2), plan.origins, support, reference, gain,
        PyramidImageIncidenceFlux, Float32(incidence_case["divisor"]),
    )
    process!(signal, divisor, workspace, incidence_plan, image)
    @test divisor == Float32[incidence_case["divisor"]]
    @test vec(signal) ≈ Float32.(incidence_case["signal"]) rtol=2f-7

    plan64 = PyramidImagePlan(
        plan.image_size,
        plan.pupil_size,
        plan.origins,
        support,
        Float64.(reference),
        Float64.(gain),
        PyramidImageMeanValidFlux,
    )
    workspace64 = PyramidImageWorkspace(plan64)
    signal64 = zeros(Float64, size(signal))
    divisor64 = zeros(Float64, 1)
    @test @inferred(Union{Nothing,PyramidImageFailure}, process!(
        signal64,
        divisor64,
        workspace64,
        plan64,
        Float64.(image),
    )) === nothing
    @test vec(signal64) ≈ Float64.(mean_case["signal"]) rtol=2e-7
end

@testset "complete-frame pyramid declaration, replacement, and reconstruction" begin
    mean_case = FIXTURE["mean_valid_flux"]
    prepared = prepare_algorithm(
        PyramidImageF32,
        fixture_configuration(mean_case["normalization_policy"]),
    )
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == (
        (4, 4),
        (3, 2),
        (1,),
        (4, 2),
        (2, 2),
        (3, 2),
        (3, 2),
    )
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1,
        PYRAMID_I4Q_SIGNAL_V1,
        PYRAMID_NORMALIZATION_DIVISOR_V1,
        REGION_ORIGINS_V1,
        PYRAMID_I4Q_SUPPORT_V1,
        PYRAMID_REFERENCE_SIGNAL_V1,
        PYRAMID_OPTICAL_GAIN_V1,
    )
    @test prepared.plan isa PyramidImagePlan{Float32}
    @test prepared.workspace isa PyramidImageWorkspace{Float32}
    @test isconcretetype(typeof(prepared.plan))
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    @test prepared.plan.selected_offsets == [0, 2, 3]

    support_storage = falses(4, 4)
    support_view = @view support_storage[1:2:3, 1:2:3]
    support_view .= Bool[true true; false true]
    view_plan = PyramidImagePlan(
        prepared.plan.image_size,
        prepared.plan.pupil_size,
        prepared.plan.origins,
        support_view,
        prepared.plan.reference_signal,
        prepared.plan.optical_gain,
        prepared.plan.normalization,
    )
    @test view_plan.selected_offsets == [0, 2, 3]

    storage = fill(-7.0f0, 6, 2)
    signal = @view storage[1:2:end, :]
    divisor_storage = fill(-7.0f0, 2)
    divisor = @view divisor_storage[2:2]
    image = fixture_image()
    @test process!(signal, divisor, prepared, image) === nothing
    @test vec(signal) ≈ Float32.(mean_case["signal"]) rtol=2f-7
    @test all(==(-7.0f0), @view(storage[2:2:end, :]))

    old_plan = prepared.plan
    replacement = copy(prepared.plan.reference_signal)
    replacement[1, 1] = 0.0f0
    @test replace_parameter!(prepared, :"reference-signal", replacement) === prepared
    @test prepared.plan !== old_plan

    old_plan = prepared.plan
    replacement_gain = copy(old_plan.optical_gain)
    replacement_gain[2, 2] = 1.25f0
    @test replace_parameter!(prepared, :"optical-gain", replacement_gain) ===
          prepared
    @test prepared.plan !== old_plan

    old_plan = prepared.plan
    replacement_support_storage = falses(4, 4)
    replacement_support = @view replacement_support_storage[1:2:3, 1:2:3]
    replacement_support .= Bool[false true; true true]
    @test count(replacement_support) == 3
    @test replace_parameter!(prepared, :"i4q-support", replacement_support) ===
          prepared
    @test prepared.plan !== old_plan
    @test prepared.plan.selected_offsets == [1, 2, 3]
    fill!(replacement_support, false)
    @test prepared.plan.support == Bool[false true; true true]

    old_plan = prepared.plan
    replacement_origins = UInt32[0 2; 0 0; 2 0; 2 2]
    @test replace_parameter!(prepared, :"pupil-origins", replacement_origins) ===
          prepared
    @test prepared.plan !== old_plan

    retained_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared, :"i4q-support", Bool[true false; false true],
    )
    @test prepared.plan === retained_plan
    invalid_reference = copy(retained_plan.reference_signal)
    invalid_reference[1, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :"reference-signal",
        invalid_reference,
    )
    @test prepared.plan === retained_plan
    invalid_gain = copy(retained_plan.optical_gain)
    invalid_gain[1, 1] = Inf32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :"optical-gain",
        invalid_gain,
    )
    @test prepared.plan === retained_plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :"pupil-origins",
        UInt32[3 0; 0 0; 2 2; 0 2],
    )
    @test prepared.plan === retained_plan

    reconstructor = row_major_matrix(Float32, mean_case["reconstructor"], 2, 6)
    reconstruction = prepare_algorithm(
        PyramidReconstructorF32,
        PyramidReconstructorF32Configuration(
            reconstructed_count=2,
            selected_coordinate_count=3,
            initial_reconstructor=vec(permutedims(reconstructor)),
            reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
    reconstructed = zeros(Float32, 2)
    process!(reconstructed, reconstruction, signal)
    @test reconstructed ≈ Float32.(mean_case["reconstructed"]) rtol=2f-7
end

@testset "complete-frame pyramid failures are output-atomic and allocation-free" begin
    prepared = prepare_algorithm(
        PyramidImageF32,
        fixture_configuration("mean-valid-flux"),
    )
    image = fixture_image()
    signal = fill(17.0f0, 3, 2)
    divisor = fill(19.0f0, 1)
    bad_input = copy(image)
    # This pixel is outside the sparse support but inside a configured pupil.
    bad_input[2, 1] = NaN32
    @test process!(signal, divisor, prepared, bad_input) === PyramidImageNonFiniteInput
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]

    @test process!(signal, divisor, prepared, zeros(Float32, size(image))) ===
          PyramidImageInvalidNormalization
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]

    signed_cancellation = zeros(Float32, size(image))
    magnitude = 0.75f0 * floatmax(Float32)
    signed_cancellation[1, 1] = magnitude
    signed_cancellation[1, 2] = magnitude
    signed_cancellation[2, 2] = -magnitude
    @test process!(signal, divisor, prepared, signed_cancellation) === nothing
    @test all(isfinite, signal)
    @test isfinite(only(divisor)) && only(divisor) > 0.0f0
    fill!(signal, 17.0f0)
    fill!(divisor, 19.0f0)

    overflow = prepare_algorithm(
        PyramidImageF32,
        fixture_configuration(
            "incidence-flux";
            incidence_flux=floatmin(Float32),
        ),
    )
    @test process!(signal, divisor, overflow, image) ===
          PyramidImageNonFiniteOutput
    @test signal == fill(17.0f0, 3, 2)
    @test divisor == Float32[19]

    @test_throws ArgumentError process!(
        prepared.plan.reference_signal,
        divisor,
        prepared,
        image,
    )
    @test_throws ArgumentError process!(
        signal,
        @view(image[1:1]),
        prepared,
        image,
    )
    @test_throws ArgumentError process!(
        signal,
        @view(prepared.plan.optical_gain[1:1]),
        prepared,
        image,
    )
    @test allocation_bytes(signal, divisor, prepared, image) == 0
    @test allocation_bytes(signal, divisor, prepared, bad_input) == 0
    zero_image = zeros(Float32, size(image))
    @test allocation_bytes(signal, divisor, prepared, zero_image) == 0
    @test allocation_bytes(signal, divisor, overflow, image) == 0
    @test @allocated(FilterGraphAlgorithms.commit!(prepared.workspace)) == 0
    @test @allocated(FilterGraphAlgorithms.discard!(prepared.workspace)) == 0
    @test @allocated(FilterGraphAlgorithms.reset!(prepared.workspace)) == 0
end

@testset "invalid complete-frame pyramid preparation is rejected" begin
    @test_throws ArgumentError prepare_algorithm(
        PyramidImageF32,
        fixture_configuration("unknown"),
    )
    @test_throws ArgumentError prepare_algorithm(
        PyramidImageF32,
        fixture_configuration("incidence-flux"; incidence_flux=0.0f0),
    )
    geometry = FIXTURE["geometry"]
    calibration = FIXTURE["calibration"]
    invalid_reference = Float32.(calibration["reference_signal"])
    invalid_reference[1] = NaN32
    @test_throws ArgumentError prepare_algorithm(
        PyramidImageF32,
        PyramidImageF32Configuration(
            image_size=(4, 4),
            pupil_size=(2, 2),
            pupil_origins=geometry["pupil_origins"],
            i4q_support=Bool.(geometry["i4q_support"]),
            initial_reference_signal=invalid_reference,
            initial_optical_gain=Float32.(calibration["optical_gain"]),
            normalization_policy="mean-valid-flux",
            incidence_flux=0.0f0,
        ),
    )
    invalid_gain = Float32.(calibration["optical_gain"])
    invalid_gain[end] = Inf32
    @test_throws ArgumentError prepare_algorithm(
        PyramidImageF32,
        PyramidImageF32Configuration(
            image_size=(4, 4),
            pupil_size=(2, 2),
            pupil_origins=geometry["pupil_origins"],
            i4q_support=Bool.(geometry["i4q_support"]),
            initial_reference_signal=Float32.(
                calibration["reference_signal"],
            ),
            initial_optical_gain=invalid_gain,
            normalization_policy="mean-valid-flux",
            incidence_flux=0.0f0,
        ),
    )
    empty_support = fill(false, 4)
    @test_throws ArgumentError prepare_algorithm(
        PyramidImageF32,
        PyramidImageF32Configuration(
            image_size=(4, 4),
            pupil_size=(2, 2),
            pupil_origins=geometry["pupil_origins"],
            i4q_support=empty_support,
            initial_reference_signal=Float32[],
            initial_optical_gain=Float32[],
            normalization_policy="mean-valid-flux",
            incidence_flux=0.0f0,
        ),
    )
end

@testset "complete-frame pyramid public interface exports" begin
    for name in (
        :PyramidImageF32,
        :PyramidImageF32Configuration,
        :PyramidImageFailure,
        :PyramidImageIncidenceFlux,
        :PyramidImageInvalidNormalization,
        :PyramidImageMeanValidFlux,
        :PyramidImageNonFiniteInput,
        :PyramidImageNonFiniteOutput,
        :PyramidImageNormalization,
        :PyramidImagePlan,
        :PyramidImageWorkspace,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "overflowing raw pupil difference has a finite normalized result" begin
    M = floatmax(Float32)
    plan = PyramidImagePlan((2,2), (1,1), ((0,0),(1,0),(1,1),(0,1)),
        trues(1,1), zeros(Float32,1,2), ones(Float32,1,2), PyramidImageIncidenceFlux, M)
    image = Float32[0.75f0*M 0; -0.75f0*M 0]
    signal = fill(-9f0,1,2)
    divisor = fill(-9f0,1)
    @test process!(signal, divisor, PyramidImageWorkspace(plan), plan, image) === nothing
    @test signal[1, 1] ≈ 1.5f0
    @test only(divisor) == M
end

end
