module SccSpatialCarrierDemodulationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

const ROI = Bool[
    true,
    false,
    false,
    false,
    true,
    false,
    false,
    false,
    false,
    false,
    false,
    true,
]

function prepared_spatial_carrier_demodulation(;
    fir=Float32[0, 1, 0],
    minimum_support=0.8f0,
    minimum_valid_fraction=2.0f0 / 3.0f0,
)
    phase = zeros(Float32, 12)
    phase[5] = Float32(pi / 2)
    return prepare_algorithm(
        SccSpatialCarrierDemodulationF32,
        SccSpatialCarrierDemodulationF32Configuration(
            image_size=(3, 4),
            roi=ROI,
            fir_tap_count=length(fir),
            initial_reference_difference=ones(Float32, 12),
            initial_static_validity=nothing,
            initial_carrier_phase=phase,
            initial_fir=fir,
            minimum_support=minimum_support,
            minimum_valid_fraction=minimum_valid_fraction,
        ),
    )
end

function spatial_outputs()
    return (zeros(Float32, 6), zeros(Float32, 6), falses(6), zeros(Float32, 1))
end

function spatial_allocation_bytes(outputs, prepared, difference, validity, status)
    return @allocated process!(outputs..., prepared, difference, validity, status)
end

@testset "spatial carrier declaration owns one scientific operation" begin
    @test algorithm_label(SccSpatialCarrierDemodulationF32) ==
          "scc-spatial-carrier-demodulation-f32"
    @test algorithm_operation(SccSpatialCarrierDemodulationF32) isa
          SccSpatialCarrierDemodulationF32Operation
    @test FilterGraphAlgorithms.algorithm_failure_type(
        SccSpatialCarrierDemodulationF32,
    ) === SccSpatialCarrierDemodulationFailure
    ports = algorithm_ports(SccSpatialCarrierDemodulationF32)
    @test map(port -> port.name, ports) == (
        :difference,
        Symbol("difference-validity"),
        Symbol("pair-status"),
        :measurements,
        :weights,
        Symbol("measurement-validity"),
        Symbol("valid-fraction"),
        Symbol("reference-difference"),
        Symbol("static-validity"),
        Symbol("carrier-phase"),
        :fir,
        :roi,
    )
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[8:12],
    )
    formats = algorithm_formats(prepared_spatial_carrier_demodulation())
    @test getproperty.(formats[1:7], :shape) ==
          ((12,), (12,), (1,), (6,), (6,), (6,), (1,))
    @test getproperty.(formats[1:7], :schema) == (
        SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
        SCC_CHOPPED_DIFFERENCE_STATUS_V1,
        SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
        SCC_MEASUREMENT_WEIGHTS_V1,
        SCC_MEASUREMENT_VALIDITY_V1,
        SCC_SPATIAL_CARRIER_VALID_FRACTION_V1,
    )
end

@testset "delta FIR demodulates selected real and imaginary components" begin
    prepared = prepared_spatial_carrier_demodulation()
    outputs = spatial_outputs()
    difference = Float32.(1:12)
    validity = trues(12)
    status = UInt8[UInt8(ChoppedImageComplete)]
    @test process!(outputs..., prepared, difference, validity, status) === nothing
    measurements, weights, selected_validity, valid_fraction = outputs
    @test measurements[[1, 3, 4, 6]] == Float32[0, 11, 0, 0]
    @test measurements[2] ≈ 0.0f0 atol = 2.0f-6
    @test measurements[5] ≈ -4.0f0 atol = 2.0f-6
    @test weights == ones(Float32, 6)
    @test all(selected_validity)
    @test valid_fraction == Float32[1]

    validity[5] = false
    @test process!(outputs..., prepared, difference, validity, status) === nothing
    @test measurements[[2, 5]] == zeros(Float32, 2)
    @test weights[[2, 5]] == zeros(Float32, 2)
    @test selected_validity == Bool[true, false, true, true, false, true]
    @test valid_fraction[1] ≈ 2.0f0 / 3.0f0
end

@testset "operation partitions reproduce complete spatial work" begin
    prepared = prepared_spatial_carrier_demodulation(
        fir=Float32[0.25, 0.5, 0.25],
        minimum_support=0.25f0,
    )
    workspace = prepared.workspace
    plan = prepared.plan
    difference = Float32.(1:12)
    validity = trues(12)
    FilterGraphAlgorithms._prepare_scc_spatial_carrier_input!(
        workspace,
        plan,
        difference,
        validity,
    )
    FilterGraphAlgorithms._scc_spatial_carrier_horizontal!(workspace, plan, 1:5)
    FilterGraphAlgorithms._scc_spatial_carrier_horizontal!(workspace, plan, 6:12)
    FilterGraphAlgorithms._scc_spatial_carrier_selected!(workspace, plan, 1:1)
    FilterGraphAlgorithms._scc_spatial_carrier_selected!(workspace, plan, 2:3)
    partitioned = copy(workspace.candidate_measurements)

    outputs = spatial_outputs()
    @test process!(
        outputs...,
        prepared,
        difference,
        validity,
        UInt8[UInt8(ChoppedImageComplete)],
    ) === nothing
    @test outputs[1] == partitioned
end

@testset "spatial dispositions preserve caller outputs and replacement is bounded" begin
    prepared = prepared_spatial_carrier_demodulation()
    outputs = (
        fill(71.0f0, 6),
        fill(72.0f0, 6),
        fill(true, 6),
        fill(73.0f0, 1),
    )
    difference = Float32.(1:12)
    validity = trues(12)
    @test process!(
        outputs...,
        prepared,
        difference,
        validity,
        UInt8[UInt8(ChoppedImagePriming)],
    ) === SccSpatialCarrierDemodulationIncompletePair
    @test outputs == (
        fill(71.0f0, 6),
        fill(72.0f0, 6),
        fill(true, 6),
        fill(73.0f0, 1),
    )
    validity[[1, 5]] .= false
    @test process!(
        outputs...,
        prepared,
        difference,
        validity,
        UInt8[UInt8(ChoppedImageComplete)],
    ) === SccSpatialCarrierDemodulationInsufficientSupport
    @test outputs[1] == fill(71.0f0, 6)

    @test replace_parameter!(prepared, :fir, Float32[0, 2, 0]) === prepared
    @test_throws ArgumentError replace_parameter!(prepared, :roi, trues(12))
    @test length(prepared.plan.selected_indices) == 3
end

@testset "spatial carrier processing is allocation-free after warmup" begin
    prepared = prepared_spatial_carrier_demodulation()
    outputs = spatial_outputs()
    difference = Float32.(1:12)
    validity = trues(12)
    status = UInt8[UInt8(ChoppedImageComplete)]
    process!(outputs..., prepared, difference, validity, status)
    @test spatial_allocation_bytes(outputs, prepared, difference, validity, status) == 0
end

@testset "spatial carrier public API exports" begin
    for name in (
        :SccSpatialCarrierDemodulationF32,
        :SccSpatialCarrierDemodulationF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "SCC positions and weights require distinct outputs" begin
    plan = SccSpatialCarrierDemodulationPlan(Float32, (1,1), trues(1), 1)
    ws = SccSpatialCarrierDemodulationWorkspace(plan)
    measurements = zeros(Float32,2)
    weights = zeros(Float32,2)
    validity = trues(2)
    fraction = zeros(Float32,1)
    @test_throws ArgumentError process!(measurements, measurements, validity, fraction, ws, plan,
        Float32[4], trues(1), UInt8[1])
    @test iszero(measurements)
end

end
