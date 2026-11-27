module CenterOfGravityTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: prepare_properties, process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "center_of_gravity.toml"),
)

coordinate_vectors(values) = [Float32.(coordinate) for coordinate in values]

function coordinate_matrix(values)
    matrix = Matrix{Float32}(undef, length(values), 2)
    for pixel in eachindex(values)
        matrix[pixel, 1] = Float32(values[pixel][1])
        matrix[pixel, 2] = Float32(values[pixel][2])
    end
    return matrix
end

function center_of_gravity_prepared(
    coordinates=FIXTURE["configuration"]["initial_coordinates"],
)
    return prepare_algorithm(
        CenterOfGravityF32,
        CenterOfGravityF32Configuration(
            pixel_count=length(coordinates),
            initial_coordinates=coordinate_vectors(coordinates),
        ),
    )
end

function set_measurement_properties!(prepared)
    measurement = FIXTURE["measurement"]
    prepared.plan = prepare_properties(
        prepared.plan,
        (
            PropertyAssignment(0, Float32(measurement["reference_centroid"][1])),
            PropertyAssignment(1, Float32(measurement["reference_centroid"][2])),
            PropertyAssignment(4, Float32(measurement["flux_threshold"])),
        ),
    )
    return prepared
end

function cog_allocation_bytes(offset, flux, validity, prepared, pixels)
    return @allocated process!(offset, flux, validity, prepared, pixels)
end

@testset "center-of-gravity startup coordinates may come from parameters" begin
    prepared = prepare_algorithm(CenterOfGravityF32,
        CenterOfGravityF32Configuration(pixel_count=2, initial_coordinates=nothing))
    offset = zeros(Float32,2); flux = zeros(Float32,1); validity = [false]
    pixels = Float32[1,3]
    @test process!(offset,flux,validity,prepared,pixels) === nothing
    @test offset == zeros(Float32,2)
    replace_parameter!(prepared, :coordinates, Float32[-1 2;3 6])
    @test process!(offset,flux,validity,prepared,pixels) === nothing
    @test validity == [true]
    @test offset == Float32[2,5]
end

function cog_admitted_allocation_bytes(offset, flux, validity, prepared, pixels)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        offset,
        flux,
        validity,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "center-of-gravity declaration preserves the FGN contract" begin
    prepared = center_of_gravity_prepared()
    configuration = FIXTURE["configuration"]

    @test algorithm_label(CenterOfGravityF32) == "center-of-gravity-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(CenterOfGravityF32) === Union{}
    @test algorithm_property_policy(CenterOfGravityF32) isa RuntimeDeclaredProperties
    fields = algorithm_configuration(CenterOfGravityF32)
    @test map(field -> field.name, fields) == (:pixel_count, :initial_coordinates)
    @test fields[1].update isa GraphRebuildConfiguration
    @test fields[2].update isa ConstructionConfiguration

    ports = algorithm_ports(CenterOfGravityF32)
    @test map(port -> port.name, ports) == (
        :pixels,
        Symbol("centroid-offset"),
        :flux,
        :validity,
        :coordinates,
    )
    @test all(port -> port.metadata_source === :pixels, ports[2:4])
    @test ports[5].role isa FilterGraphAlgorithms.SparseParameter

    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == (
        (configuration["pixel_count"],),
        (2,),
        (1,),
        (1,),
        (configuration["pixel_count"], 2),
    )
    @test getproperty.(formats, :schema) == (
        CENTER_OF_GRAVITY_PIXELS_V1,
        CENTER_OF_GRAVITY_OFFSET_V1,
        CENTER_OF_GRAVITY_FLUX_V1,
        CENTER_OF_GRAVITY_VALIDITY_V1,
        CENTER_OF_GRAVITY_COORDINATES_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, formats)

    infos = @inferred property_infos(prepared.plan)
    @test getproperty.(infos, :id) == ntuple(index -> UInt32(index - 1), 5)
    @test getproperty.(infos, :name) == (
        Symbol("reference-centroid-x"),
        Symbol("reference-centroid-y"),
        Symbol("pixel-threshold-enabled"),
        Symbol("pixel-threshold"),
        Symbol("flux-threshold"),
    )
    @test getproperty.(infos, :unit) == ("px", "px", "1", "PDE", "PDE")
    @test all(info -> info.update isa RuntimeProperty, infos)
    @test property_values(prepared.plan) == NamedTuple{getproperty.(infos, :name)}(
        (0.0f0, 0.0f0, false, 0.0f0, 0.0f0),
    )
end

@testset "center-of-gravity matches the frozen numerical oracle" begin
    prepared = set_measurement_properties!(center_of_gravity_prepared())
    measurement = FIXTURE["measurement"]
    pixels = Float32.(measurement["pixels"])
    offset = fill(NaN32, 2)
    flux = fill(NaN32, 1)
    validity = fill(false, 1)

    @test process!(offset, flux, validity, prepared, pixels) === nothing
    @test offset == Float32.(measurement["centroid_offset"])
    @test flux == Float32[measurement["flux"]]
    @test validity == Bool[measurement["validity"]]

    replacement = FIXTURE["replacement"]
    @test replace_parameter!(
        prepared,
        :coordinates,
        coordinate_matrix(replacement["coordinates"]),
    ) === prepared
    @test process!(offset, flux, validity, prepared, pixels) === nothing
    @test offset == Float32.(replacement["centroid_offset"])
end

@testset "pixel threshold remains optional and inclusive" begin
    fixture = FIXTURE["pixel_threshold"]
    coordinates = coordinate_matrix(fixture["coordinates"])
    pixels = Float32.(fixture["pixels"])
    offset = zeros(Float32, 2)
    flux = zeros(Float32, 1)
    validity = fill(false, 1)

    absent = CenterOfGravityPlan(
        view(coordinates, :, 1),
        view(coordinates, :, 2),
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    prepared_absent = center_of_gravity_prepared(fixture["coordinates"])
    prepared_absent.plan = absent
    @test process!(offset, flux, validity, prepared_absent, pixels) === nothing
    @test offset == Float32.(fixture["absent_centroid_offset"])
    @test flux == Float32[fixture["absent_flux"]]
    @test validity == Bool[true]

    enabled = prepare_properties(
        absent,
        (
            PropertyAssignment(2, true),
            PropertyAssignment(3, Float32(fixture["enabled_threshold"])),
        ),
    )
    prepared_absent.plan = enabled
    @test process!(offset, flux, validity, prepared_absent, pixels) === nothing
    @test offset == Float32.(fixture["enabled_centroid_offset"])
    @test flux == Float32[fixture["enabled_flux"]]
    @test validity == Bool[true]
end

@testset "invalid measurements zero offsets and preserve measured flux" begin
    fixture = FIXTURE["invalid"]
    plan = CenterOfGravityPlan(
        Float32[0],
        Float32[0],
        (0.0f0, 0.0f0),
        nothing,
        Float32(fixture["flux_threshold"]),
    )
    prepared = center_of_gravity_prepared([[0.0, 0.0]])
    prepared.plan = plan
    offset = fill(NaN32, 2)
    flux = fill(NaN32, 1)
    validity = fill(true, 1)

    @test process!(offset, flux, validity, prepared, Float32.(fixture["pixels"])) ===
          nothing
    @test offset == Float32.(fixture["centroid_offset"])
    @test flux == Float32[fixture["flux"]]
    @test validity == Bool[fixture["validity"]]

    @test process!(offset, flux, validity, prepared, Float32[NaN]) === nothing
    @test offset == zeros(Float32, 2)
    @test isnan(only(flux))
    @test validity == Bool[false]
end

@testset "preparation and replacement are failure-atomic" begin
    @test_throws ArgumentError CenterOfGravityPlan(
        Float32[],
        Float32[],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws DimensionMismatch CenterOfGravityPlan(
        Float32[0],
        Float32[0, 1],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        Float32[NaN],
        Float32[0],
        (0.0f0, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        [1.0e100],
        [0.0],
        (0.0, 0.0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        [0.0],
        [0.0],
        (1.0e100, 0.0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        Float32[0],
        Float32[0],
        (NaN32, 0.0f0),
        nothing,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        Float32[0],
        Float32[0],
        (0.0f0, 0.0f0),
        NaN32,
        0.0f0,
    )
    @test_throws ArgumentError CenterOfGravityPlan(
        Float32[0],
        Float32[0],
        (0.0f0, 0.0f0),
        nothing,
        -1.0f0,
    )
    @test_throws DimensionMismatch prepare_algorithm(
        CenterOfGravityF32,
        CenterOfGravityF32Configuration(
            pixel_count=2,
            initial_coordinates=[Float32[0, 0]],
        ),
    )

    prepared = center_of_gravity_prepared()
    original = prepared.plan
    @test_throws PropertyConstraintError prepare_properties(
        prepared.plan,
        (PropertyAssignment(4, -1.0f0),),
    )
    @test_throws PropertyTypeError prepare_properties(
        prepared.plan,
        (PropertyAssignment(0, 1.0),),
    )
    @test prepared.plan === original

    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :coordinates,
        zeros(Float32, 2, 3),
    )
    @test prepared.plan === original
    invalid_coordinates = coordinate_matrix(FIXTURE["configuration"]["initial_coordinates"])
    invalid_coordinates[2, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :coordinates,
        invalid_coordinates,
    )
    @test prepared.plan === original
end

@testset "center-of-gravity supports strided views and rejects aliases" begin
    prepared = center_of_gravity_prepared()
    pixel_storage = fill(71.0f0, 6)
    pixels = @view pixel_storage[1:2:5]
    pixels .= Float32[0, 0, 4]
    offset_storage = fill(83.0f0, 4)
    offset = @view offset_storage[1:2:3]
    flux_storage = fill(89.0f0, 2)
    flux = @view flux_storage[2:2]
    validity_storage = fill(false, 3)
    validity = @view validity_storage[2:2]

    @test process!(offset, flux, validity, prepared, pixels) === nothing
    @test offset == Float32[1, 2]
    @test flux == Float32[4]
    @test validity == Bool[true]
    @test offset_storage[2:2:4] == fill(83.0f0, 2)
    @test flux_storage[1] == 89.0f0
    @test !validity_storage[1] && !validity_storage[3]

    aliased_pixels = Float32[0, 0, 4]
    @test_throws ArgumentError process!(
        @view(aliased_pixels[1:2]),
        zeros(Float32, 1),
        fill(false, 1),
        prepared,
        aliased_pixels,
    )
    aliased_outputs = zeros(Float32, 2)
    @test_throws ArgumentError process!(
        aliased_outputs,
        @view(aliased_outputs[1:1]),
        fill(false, 1),
        prepared,
        Float32[0, 0, 4],
    )
    validity_alias = @view reinterpret(Bool, aliased_pixels)[1:1]
    @test_throws ArgumentError process!(
        zeros(Float32, 2),
        zeros(Float32, 1),
        validity_alias,
        prepared,
        aliased_pixels,
    )
end

@testset "checked and admitted centroid calls allocate no Julia heap" begin
    prepared = set_measurement_properties!(center_of_gravity_prepared())
    pixels = Float32.(FIXTURE["measurement"]["pixels"])
    offset = zeros(Float32, 2)
    flux = zeros(Float32, 1)
    validity = fill(false, 1)

    process!(offset, flux, validity, prepared, pixels)
    @test cog_allocation_bytes(offset, flux, validity, prepared, pixels) == 0
    @test @inferred(FilterGraphAlgorithms._process_admitted!(
        offset,
        flux,
        validity,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )) === nothing
    @test cog_admitted_allocation_bytes(offset, flux, validity, prepared, pixels) == 0
end


@testset "center-of-gravity output cannot overwrite prepared coordinates" begin
    prepared = prepare_algorithm(CenterOfGravityF32,
        CenterOfGravityF32Configuration(pixel_count=2, initial_coordinates=[Float32[0, 0], Float32[1, 0]]))
    x, y = copy(prepared.plan.x_coordinates), copy(prepared.plan.y_coordinates)
    @test_throws ArgumentError process!(prepared.plan.x_coordinates, zeros(Float32, 1), fill(false, 1), prepared, ones(Float32, 2))
    @test_throws ArgumentError process!(zeros(Float32, 2), view(prepared.plan.y_coordinates, 1:1), fill(false, 1), prepared, ones(Float32, 2))
    @test prepared.plan.x_coordinates == x
    @test prepared.plan.y_coordinates == y
end


end
