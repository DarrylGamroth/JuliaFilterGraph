module WeightedCenterOfGravityTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: prepare_properties, process!

const COORDINATES = [Float32[0, 0], Float32[1, 0], Float32[2, 1]]

function weighted_center_of_gravity_prepared(
    coordinates=COORDINATES,
    weights=Float32[0, 1, 2],
)
    return prepare_algorithm(
        WeightedCenterOfGravityF32,
        WeightedCenterOfGravityF32Configuration(
            pixel_count=length(coordinates),
            initial_coordinates=coordinates,
            initial_weights=weights,
        ),
    )
end

function weighted_cog_allocation_bytes(offset, flux, validity, prepared, pixels)
    return @allocated process!(offset, flux, validity, prepared, pixels)
end

function weighted_cog_admitted_allocation_bytes(
    offset,
    flux,
    validity,
    prepared,
    pixels,
)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        offset,
        flux,
        validity,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "weighted center-of-gravity declaration has an explicit calibration map" begin
    prepared = weighted_center_of_gravity_prepared()

    @test algorithm_label(WeightedCenterOfGravityF32) ==
          "weighted-center-of-gravity-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(WeightedCenterOfGravityF32) ===
          Union{}
    fields = algorithm_configuration(WeightedCenterOfGravityF32)
    @test map(field -> field.name, fields) == (
        :pixel_count,
        :initial_coordinates,
        :initial_weights,
    )
    @test fields[1].update isa GraphRebuildConfiguration
    @test all(field -> field.update isa ConstructionConfiguration, fields[2:3])

    ports = algorithm_ports(WeightedCenterOfGravityF32)
    @test map(port -> port.name, ports) == (
        :pixels,
        Symbol("centroid-offset"),
        :flux,
        :validity,
        :coordinates,
        :weights,
    )
    @test all(port -> port.metadata_source === :pixels, ports[2:4])
    @test all(port -> port.role isa FilterGraphAlgorithms.SparseParameter, ports[5:6])
    @test getproperty.(algorithm_formats(prepared), :shape) ==
          ((3,), (2,), (1,), (1,), (3, 2), (3,))
    @test getproperty.(algorithm_formats(prepared), :schema) == (
        CENTER_OF_GRAVITY_PIXELS_V1,
        CENTER_OF_GRAVITY_OFFSET_V1,
        WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1,
        CENTER_OF_GRAVITY_VALIDITY_V1,
        CENTER_OF_GRAVITY_COORDINATES_V1,
        CENTER_OF_GRAVITY_WEIGHTS_V1,
    )

    infos = @inferred property_infos(prepared.plan)
    @test getproperty.(infos, :id) == ntuple(index -> UInt32(index - 1), 5)
    @test property_values(prepared.plan) == NamedTuple{getproperty.(infos, :name)}(
        (0.0f0, 0.0f0, false, 0.0f0, 0.0f0),
    )
end

@testset "weighted center-of-gravity applies prepared weights without normalization" begin
    prepared = weighted_center_of_gravity_prepared()
    prepared.plan = prepare_properties(
        prepared.plan,
        (PropertyAssignment(0, 0.8f0), PropertyAssignment(1, 0.3f0)),
    )
    offset = fill(NaN32, 2)
    flux = fill(NaN32, 1)
    validity = fill(false, 1)

    @test process!(offset, flux, validity, prepared, Float32[100, 2, 4]) === nothing
    @test offset ≈ Float32[1.0, 0.5]
    @test flux == Float32[10]
    @test validity == Bool[true]

    @test replace_parameter!(prepared, :weights, Float32[0, 2, 4]) === prepared
    @test process!(offset, flux, validity, prepared, Float32[100, 2, 4]) === nothing
    @test offset ≈ Float32[1.0, 0.5]
    @test flux == Float32[20]
    @test validity == Bool[true]
end

@testset "weighted center-of-gravity retains calibrated-pixel thresholding and invalidity" begin
    prepared = weighted_center_of_gravity_prepared(
        COORDINATES,
        Float32[5, 1, 2],
    )
    prepared.plan = prepare_properties(
        prepared.plan,
        (PropertyAssignment(2, true), PropertyAssignment(3, 2.0f0)),
    )
    offset = fill(NaN32, 2)
    flux = fill(NaN32, 1)
    validity = fill(false, 1)

    @test process!(offset, flux, validity, prepared, Float32[1, 2, 4]) === nothing
    @test offset == Float32[1.8, 0.8]
    @test flux == Float32[10]
    @test validity == Bool[true]

    zero_weight = weighted_center_of_gravity_prepared(
        COORDINATES,
        Float32[0, 0, 0],
    )
    @test process!(offset, flux, validity, zero_weight, Float32[1, 2, 4]) === nothing
    @test offset == zeros(Float32, 2)
    @test flux == Float32[0]
    @test validity == Bool[false]

    @test process!(offset, flux, validity, zero_weight, Float32[NaN, 2, 4]) === nothing
    @test offset == zeros(Float32, 2)
    @test isnan(only(flux))
    @test validity == Bool[false]
end

@testset "weighted center-of-gravity preparation and replacement are failure-atomic" begin
    @test_throws DimensionMismatch WeightedCenterOfGravityPlan(
        Float32[0],
        Float32[0],
        Float32[1, 2],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError WeightedCenterOfGravityPlan(
        Float32[0],
        Float32[0],
        Float32[NaN],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError WeightedCenterOfGravityPlan(
        Float32[0],
        Float32[0],
        Float32[-1],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws DimensionMismatch prepare_algorithm(
        WeightedCenterOfGravityF32,
        WeightedCenterOfGravityF32Configuration(
            pixel_count=2,
            initial_coordinates=COORDINATES[1:2],
            initial_weights=Float32[1],
        ),
    )

    initial_weights = Float32[0, 1, 2]
    prepared = weighted_center_of_gravity_prepared(COORDINATES, initial_weights)
    initial_weights .= 0.0f0
    offset = zeros(Float32, 2)
    flux = zeros(Float32, 1)
    validity = fill(false, 1)
    process!(offset, flux, validity, prepared, Float32[100, 2, 4])
    @test flux == Float32[10]

    original = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(prepared, :weights, Float32[1, 2])
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(prepared, :weights, Float32[0, NaN, 1])
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(prepared, :weights, Float32[0, -1, 1])
    @test prepared.plan === original
end

@testset "weighted center-of-gravity supports strided views and has no warmed allocations" begin
    prepared = weighted_center_of_gravity_prepared()
    pixel_storage = fill(71.0f0, 6)
    pixels = @view pixel_storage[1:2:5]
    pixels .= Float32[100, 2, 4]
    offset_storage = fill(83.0f0, 4)
    offset = @view offset_storage[1:2:3]
    flux_storage = fill(89.0f0, 2)
    flux = @view flux_storage[2:2]
    validity_storage = fill(false, 3)
    validity = @view validity_storage[2:2]

    @test process!(offset, flux, validity, prepared, pixels) === nothing
    @test offset == Float32[1.8, 0.8]
    @test flux == Float32[10]
    @test validity == Bool[true]
    @test offset_storage[2:2:4] == fill(83.0f0, 2)
    @test flux_storage[1] == 89.0f0
    @test !validity_storage[1] && !validity_storage[3]

    two_pixel_prepared = weighted_center_of_gravity_prepared(
        COORDINATES[1:2],
        Float32[1, 1],
    )
    @test_throws ArgumentError process!(
        two_pixel_prepared.plan.weights,
        zeros(Float32, 1),
        fill(false, 1),
        two_pixel_prepared,
        Float32[1, 2],
    )
    @test_throws ArgumentError process!(
        two_pixel_prepared.plan.x_coordinates,
        zeros(Float32, 1),
        fill(false, 1),
        two_pixel_prepared,
        Float32[1, 2],
    )
    one_pixel_prepared = weighted_center_of_gravity_prepared(
        COORDINATES[1:1],
        Float32[1],
    )
    @test_throws ArgumentError process!(
        zeros(Float32, 2),
        one_pixel_prepared.plan.weights,
        fill(false, 1),
        one_pixel_prepared,
        Float32[1],
    )

    process!(offset, flux, validity, prepared, pixels)
    @test weighted_cog_allocation_bytes(offset, flux, validity, prepared, pixels) == 0
    @test @inferred(FilterGraphAlgorithms._process_admitted!(
        offset,
        flux,
        validity,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )) === nothing
    @test weighted_cog_admitted_allocation_bytes(
        offset,
        flux,
        validity,
        prepared,
        pixels,
    ) == 0
end

@testset "weighted moment FMA retains cancellation and finite overflow cancellation" begin
    for (values, coordinates) in (
        (Float32[1, 1 + 2.0^-12], Float32[-(1 + 2.0^-11), 1 + 2.0^-12]),
        (Float32[1, 2], Float32[-floatmax(Float32), floatmax(Float32)]),
    )
        prepared = weighted_center_of_gravity_prepared(
            [Float32[x, x] for x in coordinates], ones(Float32, 2),
        )
        offset, flux, valid = zeros(Float32, 2), zeros(Float32, 1), falses(1)
        # Independent exact-product reference, not another Float32 accumulation.
        expected = setprecision(BigFloat, 1024) do
            Float32(sum(BigFloat.(values) .* BigFloat.(coordinates)) / sum(BigFloat.(values)))
        end
        @test process!(offset, flux, valid, prepared, values) === nothing
        @test offset == fill(expected, 2)
        @test valid == [true]
        @test flux == [sum(values)]
        @test weighted_cog_allocation_bytes(offset, flux, valid, prepared, values) == 0
    end
end

@testset "weighted pixels round before fused coordinate moments" begin
    a = 1f0 + 1f0 / 4096
    rounded_product = 1f0 + 1f0 / 2048
    prepared = weighted_center_of_gravity_prepared(
        [Float32[-rounded_product, -rounded_product], Float32[1, 1]], Float32[1, a])
    offset, flux, valid = zeros(Float32, 2), zeros(Float32, 1), falses(1)
    @test process!(offset, flux, valid, prepared, Float32[1, a]) === nothing
    @test offset == zeros(Float32, 2)
    @test flux == [1f0 + rounded_product]
    @test valid == [true]
end

end
