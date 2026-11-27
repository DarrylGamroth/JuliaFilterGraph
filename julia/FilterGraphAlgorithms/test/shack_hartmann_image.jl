module ShackHartmannImageTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: prepare_properties, process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_image.toml"),
)

const SYNTHETIC_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_synthetic.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function pair_matrix(::Type{T}, values) where {T}
    matrix = Matrix{T}(undef, length(values), 2)
    for index in eachindex(values)
        matrix[index, 1] = T(values[index][1])
        matrix[index, 2] = T(values[index][2])
    end
    return matrix
end

function shack_hartmann_prepared()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            subaperture_count=configuration["subaperture_count"],
            initial_subaperture_origins=[
                Int.(origin) for
                origin in configuration["initial_subaperture_origins"]
            ],
            coordinate_scale=Float32(configuration["coordinate_scale"]),
            pixel_threshold=Float32(configuration["pixel_threshold"]),
            flux_threshold=Float32(configuration["flux_threshold"]),
            reference_slopes=Float32.(configuration["reference_slopes"]),
            active=Bool.(configuration["active"]),
        ),
    )
end

function fixture_image()
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        FIXTURE["sample"]["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
end

function measure!(prepared, image=fixture_image())
    count = length(prepared.plan.regions.origins)
    slopes = zeros(Float32, count, 2)
    flux = zeros(Float32, count)
    validity = fill(false, count)
    process!(slopes, flux, validity, prepared, image)
    return slopes, flux, validity
end

function shack_allocation_bytes(slopes, flux, validity, prepared, image)
    return @allocated process!(slopes, flux, validity, prepared, image)
end

function shack_admitted_allocation_bytes(slopes, flux, validity, prepared, image)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        slopes,
        flux,
        validity,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

function synthetic_prepared()
    observation = SYNTHETIC_FIXTURE["observation"]
    fga = SYNTHETIC_FIXTURE["fga"]
    return prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(observation["image_size"][2], observation["image_size"][1]),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(observation["subaperture_size"][2], observation["subaperture_size"][1]),
            subaperture_count=observation["subaperture_count"],
            initial_subaperture_origins=[
                Int.(origin) for origin in fga["initial_subaperture_origins"]
            ],
            coordinate_scale=Float32(fga["coordinate_scale"]),
            pixel_threshold=Float32(fga["absolute_pixel_threshold"]),
            flux_threshold=Float32(fga["flux_threshold"]),
            reference_slopes=Float32.(fga["reference_slopes_interleaved"]),
            active=Bool.(observation["active"]),
        ),
    )
end

function synthetic_image()
    observation = SYNTHETIC_FIXTURE["observation"]
    image_size = observation["image_size"]
    frame_xy = reshape(Float32.(observation["frame_xy"]), image_size...)
    return PermutedDimsArray(frame_xy, (2, 1))
end

@testset "Shack-Hartmann image declaration preserves the FGN contract" begin
    prepared = shack_hartmann_prepared()
    configuration = FIXTURE["configuration"]

    @test algorithm_label(ShackHartmannImageF32) == "shack-hartmann-image-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(ShackHartmannImageF32) ===
          Union{}
    @test algorithm_property_policy(ShackHartmannImageF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    fields = algorithm_configuration(ShackHartmannImageF32)
    @test map(field -> field.name, fields) == (
        :image_size,
        :image_schema,
        :subaperture_size,
        :subaperture_count,
        :initial_subaperture_origins,
        :coordinate_scale,
        :pixel_threshold,
        :flux_threshold,
        :reference_slopes,
        :active,
    )
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:4])
    @test all(field -> field.update isa ConstructionConfiguration, fields[5:10])

    ports = algorithm_ports(ShackHartmannImageF32)
    @test map(port -> port.name, ports) == (
        :image,
        :slopes,
        :flux,
        :validity,
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
    )
    @test all(port -> port.metadata_source === :image, ports[2:4])
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[5:9],
    )

    formats = algorithm_formats(prepared)
    pixel_count = configuration["subaperture_rows"] *
        configuration["subaperture_columns"]
    count = configuration["subaperture_count"]
    @test getproperty.(formats, :shape) == (
        (configuration["image_rows"], configuration["image_columns"]),
        (count, 2),
        (count,),
        (count,),
        (count, 2),
        (pixel_count, 2),
        (count, 2),
        (count, 2),
        (count,),
    )
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1,
        SHACK_HARTMANN_SLOPES_V1,
        SHACK_HARTMANN_FLUX_V1,
        SHACK_HARTMANN_VALIDITY_V1,
        REGION_ORIGINS_V1,
        SHACK_HARTMANN_COORDINATES_V1,
        SHACK_HARTMANN_REFERENCE_SLOPES_V1,
        SHACK_HARTMANN_THRESHOLDS_V1,
        SHACK_HARTMANN_ACTIVE_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, formats)
    @test prepared.workspace isa FilterGraphAlgorithms.EmptyWorkspace
end

@testset "image-backed centroids match the frozen FGN oracle" begin
    slopes, flux, validity = measure!(shack_hartmann_prepared())
    @test slopes == pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    @test flux == Float32.(FIXTURE["sample"]["flux"])
    @test validity == Bool.(FIXTURE["sample"]["validity"])
end

@testset "synthetic complete-frame centroid calibration" begin
    observation = SYNTHETIC_FIXTURE["observation"]
    fga = SYNTHETIC_FIXTURE["fga"]
    prepared = synthetic_prepared()
    image = synthetic_image()
    count = observation["subaperture_count"]
    slopes = zeros(Float32, count, 2)
    flux = zeros(Float32, count)
    validity = falses(count)

    response = Float32(observation["centroid_response"])
    subaperture_size = observation["subaperture_size"]
    center = Float32.((subaperture_size .- 1) ./ 2)
    references = pair_matrix(Float32, observation["reference_centroids"])
    transformed = Float32.(fga["reference_slopes_interleaved"])
    for subaperture in axes(references, 1), axis in axes(references, 2)
        reference_index = 2 * (subaperture - 1) + axis
        expected_reference =
            references[subaperture, axis] / response - center[axis] / response
        @test isapprox(
            transformed[reference_index],
            expected_reference;
            rtol=0.0f0,
            atol=eps(Float32),
        )
    end
    @test Float32(fga["coordinate_scale"]) == inv(response)
    @test Float32(fga["absolute_pixel_threshold"]) ==
          Float32(observation["relative_pixel_threshold"]) *
          Float32(observation["frame_peak"])

    @test @inferred(process!(slopes, flux, validity, prepared, image)) === nothing
    expected_slopes = pair_matrix(Float32, fga["slopes"])
    @test isapprox(slopes, expected_slopes; rtol=2eps(Float32), atol=0.0f0)
    @test flux == Float32.(fga["flux"])
    @test validity == Bool.(fga["validity"])

    # The threshold-equal pixel in the first subaperture contributes to flux.
    @test image[1, 2] == fga["absolute_pixel_threshold"]
    @test flux[1] == image[1, 1] + image[1, 2]

    # Inactive and empty subapertures have zero slopes. The inactive tile still carries flux.
    @test vec(slopes)[[2, 3, 6, 7]] == zeros(Float32, 4)
    @test flux[2] > 0f0 && validity[2]
    @test flux[3] == 0f0 && !validity[3]
    @test all(value -> !signbit(value), vec(slopes)[[2, 3, 6, 7]])

    process!(slopes, flux, validity, prepared, image)
    @test shack_allocation_bytes(slopes, flux, validity, prepared, image) == 0

    # Negating both coordinate and reference calibration reverses valid slopes.
    negative_prepared = synthetic_prepared()
    coordinates = hcat(
        negative_prepared.plan.x_coordinates,
        negative_prepared.plan.y_coordinates,
    )
    reference_pairs = Matrix{Float32}(undef, count, 2)
    for subaperture in axes(reference_pairs, 1), axis in axes(reference_pairs, 2)
        reference_pairs[subaperture, axis] =
            negative_prepared.plan.reference_slopes[2 * (subaperture - 1) + axis]
    end
    replace_parameter!(negative_prepared, :coordinates, -coordinates)
    replace_parameter!(negative_prepared, Symbol("reference-slopes"), -reference_pairs)
    process!(slopes, flux, validity, negative_prepared, image)
    @test isapprox(slopes, -expected_slopes; rtol=2eps(Float32), atol=0.0f0)
    @test flux == Float32.(fga["flux"])
    @test validity == Bool.(fga["validity"])

    nonfinite_image = copy(image)
    nonfinite_image[1, 1] = NaN32
    process!(slopes, flux, validity, prepared, nonfinite_image)
    @test slopes[1, :] == zeros(Float32, 2)
    @test isnan(flux[1])
    @test !validity[1]
end

@testset "image-backed measurement matches materialized Julia composition" begin
    configuration = FIXTURE["configuration"]
    image = fixture_image()
    extraction = prepare_algorithm(
        RegionExtractionF32,
        RegionExtractionF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            region_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            origins=configuration["initial_subaperture_origins"],
        ),
    )
    regions = zeros(
        Float32,
        configuration["subaperture_count"],
        configuration["subaperture_rows"],
        configuration["subaperture_columns"],
    )
    process!(regions, extraction, image)
    image_slopes, image_flux, image_validity = measure!(shack_hartmann_prepared(), image)

    coordinates = [
        Float32[column - 2, row - 1.5] for
        row in 1:configuration["subaperture_rows"] for
        column in 1:configuration["subaperture_columns"]
    ]
    for subaperture in 1:configuration["subaperture_count"]
        centroid = prepare_algorithm(
            CenterOfGravityF32,
            CenterOfGravityF32Configuration(
                pixel_count=length(coordinates),
                initial_coordinates=coordinates,
            ),
        )
        reference = 2 * subaperture - 1
        centroid.plan = prepare_properties(
            centroid.plan,
            (
                PropertyAssignment(
                    0,
                    Float32(configuration["reference_slopes"][reference]),
                ),
                PropertyAssignment(
                    1,
                    Float32(configuration["reference_slopes"][reference + 1]),
                ),
                PropertyAssignment(2, true),
                PropertyAssignment(
                    3,
                    Float32(configuration["pixel_threshold"]),
                ),
                PropertyAssignment(
                    4,
                    Float32(configuration["flux_threshold"]),
                ),
            ),
        )
        pixels = Float32[
            regions[subaperture, row, column] for
            row in 1:configuration["subaperture_rows"] for
            column in 1:configuration["subaperture_columns"]
        ]
        offset = zeros(Float32, 2)
        measured_flux = zeros(Float32, 1)
        measured_validity = fill(false, 1)
        process!(offset, measured_flux, measured_validity, centroid, pixels)
        @test offset == vec(image_slopes[subaperture, :])
        @test only(measured_flux) == image_flux[subaperture]
        @test only(measured_validity) == image_validity[subaperture]
    end
end

@testset "all sparse calibration replacements match frozen outcomes" begin
    cases = (
        (
            :coordinates,
            pair_matrix(Float32, FIXTURE["coordinate_replacement"]["values"]),
            FIXTURE["coordinate_replacement"],
        ),
        (
            Symbol("reference-slopes"),
            pair_matrix(Float32, FIXTURE["reference_replacement"]["values"]),
            FIXTURE["reference_replacement"],
        ),
        (
            :thresholds,
            pair_matrix(Float32, FIXTURE["threshold_replacement"]["values"]),
            FIXTURE["threshold_replacement"],
        ),
        (
            :active,
            Bool.(FIXTURE["active_replacement"]["values"]),
            FIXTURE["active_replacement"],
        ),
        (
            Symbol("subaperture-origins"),
            pair_matrix(UInt32, FIXTURE["origin_replacement"]["values"]),
            FIXTURE["origin_replacement"],
        ),
    )

    for (name, values, expected) in cases
        prepared = shack_hartmann_prepared()
        original = prepared.plan
        @test replace_parameter!(prepared, name, values) === prepared
        @test prepared.plan !== original
        slopes, flux, validity = measure!(prepared)
        @test slopes == pair_matrix(Float32, expected["slopes"])
        @test flux == Float32.(get(expected, "flux", FIXTURE["sample"]["flux"]))
        @test validity == Bool[true, true]
    end
end

@testset "threshold, validity, and active state remain distinct" begin
    function threshold_prepared(flux_threshold, active)
        return prepare_algorithm(
            ShackHartmannImageF32,
            ShackHartmannImageF32Configuration(
                image_size=(1, 3),
                image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(1, 3),
                subaperture_count=1,
                initial_subaperture_origins=nothing,
                coordinate_scale=1.0f0,
                pixel_threshold=2.0f0,
                flux_threshold=flux_threshold,
                reference_slopes=nothing,
                active=Bool[active],
            ),
        )
    end

    prepared = threshold_prepared(5.0f0, true)
    slopes, flux, validity = measure!(prepared, reshape(Float32[1, 2, 3], 1, 3))
    @test slopes == Float32[0.6 0.0]
    @test flux == Float32[5]
    @test validity == Bool[true]

    inactive = threshold_prepared(5.0f0, false)
    slopes, flux, validity = measure!(inactive, reshape(Float32[1, 2, 3], 1, 3))
    @test slopes == zeros(Float32, 1, 2)
    @test flux == Float32[5]
    @test validity == Bool[true]

    low_flux = threshold_prepared(6.0f0, true)
    slopes, flux, validity = measure!(low_flux, reshape(Float32[1, 2, 3], 1, 3))
    @test slopes == zeros(Float32, 1, 2)
    @test flux == Float32[5]
    @test validity == Bool[false]

    nonfinite = reshape(Float32[NaN, 2, 3], 1, 3)
    slopes, flux, validity = measure!(prepared, nonfinite)
    @test slopes == zeros(Float32, 1, 2)
    @test isnan(only(flux))
    @test validity == Bool[false]
end

@testset "preparation and parameter replacement are failure-atomic" begin
    configuration = FIXTURE["configuration"]
    @test_throws ArgumentError ShackHartmannImagePlan(
        (3, 7),
        (2, 3),
        [(0, 1), (1, 3)],
        0.0f0,
        0.0f0,
        1.0f0,
    )
    @test_throws ArgumentError prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(2, 3),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(2, 3),
            subaperture_count=2,
            initial_subaperture_origins=nothing,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=1.0f0,
            reference_slopes=nothing,
            active=nothing,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            subaperture_count=configuration["subaperture_count"],
            initial_subaperture_origins=[[0, 1]],
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=1.0f0,
            reference_slopes=nothing,
            active=nothing,
        ),
    )

    prepared = shack_hartmann_prepared()
    original = prepared.plan
    invalid_coordinates = zeros(Float32, 6, 2)
    invalid_coordinates[1, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :coordinates,
        invalid_coordinates,
    )
    @test prepared.plan === original
    invalid_references = zeros(Float32, 2, 2)
    invalid_references[2, 2] = Inf32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "reference-slopes",
        invalid_references,
    )
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :thresholds,
        Float32[-1 0; 0 0],
    )
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "subaperture-origins",
        UInt32[0 0; 99 99],
    )
    @test prepared.plan === original
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :active,
        Bool[true],
    )
    @test prepared.plan === original
end

@testset "logical CPU views work and overlapping storage is rejected" begin
    prepared = shack_hartmann_prepared()
    image_storage = fill(71.0f0, 6, 14)
    image = @view image_storage[1:2:5, 2:2:14]
    image .= fixture_image()
    slope_storage = fill(83.0f0, 4, 4)
    slopes = @view slope_storage[1:2:3, 2:2:4]
    flux_storage = fill(89.0f0, 4)
    flux = @view flux_storage[1:2:3]
    validity_storage = fill(false, 4)
    validity = @view validity_storage[2:2:4]

    @test process!(slopes, flux, validity, prepared, image) === nothing
    @test slopes == pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    @test flux == Float32.(FIXTURE["sample"]["flux"])
    @test validity == Bool[true, true]

    aliased_image = fixture_image()
    aliased_slopes = reshape(@view(vec(aliased_image)[1:4]), 2, 2)
    @test_throws ArgumentError process!(
        aliased_slopes,
        zeros(Float32, 2),
        fill(false, 2),
        prepared,
        aliased_image,
    )
    aliased_outputs = zeros(Float32, 2, 2)
    @test_throws ArgumentError process!(
        aliased_outputs,
        @view(aliased_outputs[1:2]),
        fill(false, 2),
        prepared,
        fixture_image(),
    )
    @test_throws ArgumentError process!(
        zeros(Float32, 2, 2),
        prepared.plan.pixel_thresholds,
        fill(false, 2),
        prepared,
        fixture_image(),
    )
    validity_alias = @view reinterpret(Bool, aliased_image)[1:2]
    @test_throws ArgumentError process!(
        zeros(Float32, 2, 2),
        zeros(Float32, 2),
        validity_alias,
        prepared,
        aliased_image,
    )
end

@testset "REVOLT-shaped checked and admitted calls allocate no Julia heap" begin
    detector_rows = 352
    detector_columns = 352
    subaperture_rows = 22
    subaperture_columns = 22
    grid_columns = 16
    subaperture_count = 256
    prepared = prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(detector_rows, detector_columns),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(subaperture_rows, subaperture_columns),
            subaperture_count=subaperture_count,
            initial_subaperture_origins=nothing,
            coordinate_scale=0.5f0,
            pixel_threshold=0.0f0,
            flux_threshold=1.0f0,
            reference_slopes=nothing,
            active=nothing,
        ),
    )
    image = Matrix{Float32}(undef, detector_rows, detector_columns)
    for row in 0:(detector_rows - 1), column in 0:(detector_columns - 1)
        subaperture = div(row, subaperture_rows) * grid_columns +
            div(column, subaperture_columns)
        image[row + 1, column + 1] = 1.0f0 +
            0.01f0 * Float32(rem(row, subaperture_rows)) +
            0.02f0 * Float32(rem(column, subaperture_columns)) +
            0.001f0 * Float32(subaperture)
    end
    slopes = zeros(Float32, subaperture_count, 2)
    flux = zeros(Float32, subaperture_count)
    validity = fill(false, subaperture_count)

    process!(slopes, flux, validity, prepared, image)
    @test all(validity)
    @test shack_allocation_bytes(slopes, flux, validity, prepared, image) == 0
    @test @inferred(FilterGraphAlgorithms._process_admitted!(
        slopes,
        flux,
        validity,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )) === nothing
    @test shack_admitted_allocation_bytes(
        slopes,
        flux,
        validity,
        prepared,
        image,
    ) == 0
end

@testset "contiguous detector rows preserve calibrated pixel order and CPU views" begin
    rows, columns = 22, 22
    regions = FilterGraphAlgorithms.RegionExtractionPlan((rows, columns), [(0, 0)], (rows, columns))
    x = Float32[(3 * row - 7 * column) / 13 for row in 1:rows for column in 1:columns]
    y = Float32[(row^2 + row * column) / 29 for row in 1:rows for column in 1:columns]
    plan = ShackHartmannImagePlan(regions, x, y, Float32[0.25, -0.5], Float32[2], Float32[1], Bool[true])
    image = Float32[mod(3 * row + 7 * column, 19) / 3 for row in 1:rows, column in 1:columns]
    selected = [(row, column) for row in 1:rows for column in 1:columns if image[row, column] >= 2.0f0]
    expected_flux = sum(Float64(image[row, column]) for (row, column) in selected)
    expected_slopes = [
        sum(Float64(image[row, column]) * coordinate[(row - 1) * columns + column]
            for (row, column) in selected) / expected_flux - reference
        for (coordinate, reference) in ((x, 0.25), (y, -0.5))
    ]
    reversed = reverse(image; dims=(1, 2))
    padded = zeros(Float32, 2 * rows, 2 * columns)
    nonunit = @view padded[2:2:end, 2:2:end]
    nonunit .= image
    column_reversed = reverse(image; dims=2)
    generic = PermutedDimsArray(permutedims(image), (2, 1))
    for source in (image, @view(reversed[end:-1:1, end:-1:1]), nonunit,
                   @view(column_reversed[:, end:-1:1]), generic)
        slopes, flux, validity = zeros(Float32, 1, 2), zeros(Float32, 1), fill(false, 1)
        FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, source, 1:1)
        @test vec(slopes) ≈ expected_slopes rtol=2e-5 atol=4e-6
        @test only(flux) ≈ expected_flux rtol=2e-5
        @test only(validity)
    end
end

@testset "packed rectangular coordinates follow parameter replacement and in-place edits" begin
    rows, columns = 19, 5
    prepared = prepare_algorithm(
        ShackHartmannImageF32,
        ShackHartmannImageF32Configuration(
            image_size=(rows, 2columns), image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(rows, columns), subaperture_count=2,
            initial_subaperture_origins=nothing, coordinate_scale=1.0f0,
            pixel_threshold=0.0f0, flux_threshold=0.0f0, reference_slopes=nothing, active=nothing,
        ),
    )
    image = zeros(Float32, rows, 2columns)
    image[4, 3] = 2
    image[7, columns + 1] = 4
    pixels = [(4 - 1) * columns + 3, (7 - 1) * columns + 1]
    calibration = hcat(Float32.(1:rows*columns), -Float32.(1:rows*columns))
    replace_parameter!(prepared, :coordinates, calibration)
    slopes, flux, validity = zeros(Float32, 2, 2), zeros(Float32, 2), fill(false, 2)
    @test process!(slopes, flux, validity, prepared, image) === nothing
    @test slopes == calibration[pixels, :]
    @test flux == Float32[2, 4] && all(validity)

    prepared.plan.x_coordinates[pixels[1]] = -17
    prepared.plan.y_coordinates[pixels[2]] = 31
    process!(slopes, flux, validity, prepared, image)
    @test slopes == [-17 calibration[pixels[1], 2]; calibration[pixels[2], 1] 31]
    @test shack_allocation_bytes(slopes, flux, validity, prepared, image) == 0

    replace_parameter!(prepared, :coordinates, -calibration)
    process!(slopes, flux, validity, prepared, image)
    @test slopes == -calibration[pixels, :]
    # An output view of the SIMD coordinates would corrupt future frames.
    aliased_slopes = reshape(@view(prepared.plan._column_coordinates[1][1:4]), 2, 2)
    @test_throws ArgumentError process!(aliased_slopes, flux, validity, prepared, image)

    replace_parameter!(prepared, :thresholds, Float32[0 nextfloat(2.0f0); 0 4])
    process!(slopes, flux, validity, prepared, image)
    @test flux == Float32[2, 4] && validity == [false, true]
    @test all(iszero, slopes[1, :]) && !any(signbit, slopes[1, :])
    @test slopes[2, :] == -calibration[pixels[2], :]
    image[1, 1] = Inf32
    process!(slopes, flux, validity, prepared, image)
    @test isinf(flux[1]) && flux[2] == 4 && validity == [false, true]
    @test all(iszero, slopes[1, :]) && !any(signbit, slopes[1, :])
end

@testset "SIMD centroids retain serial exceptional and threshold classifications" begin
    rows, columns = 22, 22
    regions = FilterGraphAlgorithms.RegionExtractionPlan((rows, columns), [(0, 0)], (rows, columns))
    plan = ShackHartmannImagePlan(regions, zeros(Float32, rows * columns),
        zeros(Float32, rows * columns), zeros(Float32, 2), Float32[2], Float32[1], Bool[true])
    slopes, flux, validity = zeros(Float32, 1, 2), zeros(Float32, 1), fill(false, 1)
    image = fill(3.0f0, rows, columns)
    for bad in (NaN32, Inf32, -Inf32)
        image[1, 1] = bad
        FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, image, 1:1)
        @test !only(validity)
        @test !isfinite(only(flux))
        @test all(iszero, slopes) && !any(signbit, slopes)
    end
    # Row-major partial moments overflow; column-major partial moments cancel.
    # The serial fallback must retain the original invalid classification.
    image .= 0.0f0
    image[1:2, 1:8] .= 3.0f0
    plan.x_coordinates[1:8] .= 2.0f37
    plan.x_coordinates[(columns + 1):(columns + 8)] .= -2.0f37
    FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, image, 1:1)
    @test only(flux) == 48.0f0
    @test !only(validity)
    @test all(iszero, slopes) && !any(signbit, slopes)

    plan.x_coordinates .= 0.0f0
    image .= Float32[2 + mod(row + 7 * column, 17) / 13 for row in 1:rows, column in 1:columns]
    serial_flux = 0.0f0
    for row in 1:rows, column in 1:columns
        serial_flux += image[row, column]
    end
    for threshold in (prevfloat(serial_flux), serial_flux, nextfloat(serial_flux))
        plan.flux_thresholds[1] = threshold
        FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, image, 1:1)
        @test only(flux) === serial_flux
        @test only(validity) == (serial_flux >= threshold)
    end
    plan.active[1] = false
    plan.flux_thresholds[1] = serial_flux
    FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, image, 1:1)
    @test only(validity)
    @test only(flux) === serial_flux
    @test all(iszero, slopes) && !any(signbit, slopes)
end

@testset "nondefault floating-point modes retain serial centroid arithmetic" begin
    plan = ShackHartmannImagePlan((22, 22), (22, 22), [(0, 0)], 0.5f0, 0.0f0, 0.0f0)
    image = Float32[mod(3 * row + 7 * column, 19) / 13 for row in 1:22, column in 1:22]
    image[1, 1] = nextfloat(0.0f0)
    slopes, flux, validity = zeros(Float32, 1, 2), zeros(Float32, 1), fill(false, 1)
    original_rounding = Base.Rounding.rounding_raw(Float32)
    original_subnormals = get_zero_subnormals()
    try
        for subnormals in (false, true), mode in (RoundNearest, RoundDown, RoundUp, RoundToZero)
            subnormals || mode !== RoundNearest || continue
            set_zero_subnormals(subnormals)
            Base.Rounding.setrounding_raw(Float32, Base.Rounding.to_fenv(mode))
            x, y, expected_flux = FilterGraphAlgorithms._shack_hartmann_moments_serial(plan, image, 1)
            expected_x, expected_y, _, expected_validity = FilterGraphAlgorithms._finish_center_of_gravity(
                x, y, expected_flux, 0.0f0, 0.0f0, 0.0f0,
            )
            FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan, image, 1:1)
            @test isequal(vec(slopes), [expected_x, expected_y])
            @test only(flux) === expected_flux
            @test only(validity) == expected_validity
        end
    finally
        Base.Rounding.setrounding_raw(Float32, original_rounding)
        set_zero_subnormals(original_subnormals)
    end
    @test Base.Rounding.rounding_raw(Float32) == original_rounding
    @test get_zero_subnormals() == original_subnormals
end

end
