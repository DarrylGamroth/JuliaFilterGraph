module ShackHartmannWeightedCentroidTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: prepare_properties, process!

const WEIGHTS = [
    Float32[0, 1, 2, 3, 0.5, 1],
    Float32[1, 0, 2, 0, 3, 4],
]

struct ZeroBasedWeightedImage{A<:AbstractMatrix{Float32}} <: AbstractMatrix{Float32}
    parent::A
end

Base.size(image::ZeroBasedWeightedImage) = size(image.parent)
Base.axes(image::ZeroBasedWeightedImage) =
    (0:(size(image, 1) - 1), 0:(size(image, 2) - 1))
Base.IndexStyle(::Type{<:ZeroBasedWeightedImage}) = IndexCartesian()
Base.getindex(image::ZeroBasedWeightedImage, row::Int, column::Int) =
    image.parent[row + 1, column + 1]

function weighted_shack_hartmann_prepared(weights=WEIGHTS)
    return prepare_algorithm(
        ShackHartmannWeightedCentroidF32,
        ShackHartmannWeightedCentroidF32Configuration(
            image_size=(4, 6),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(2, 3),
            subaperture_count=2,
            initial_subaperture_origins=[[0, 0], [2, 3]],
            coordinate_scale=1.0f0,
            pixel_threshold=2.0f0,
            flux_threshold=4.0f0,
            reference_slopes=Float32[0.25, -0.5, -0.75, 0.25],
            active=Bool[true, true],
            initial_weights=weights,
        ),
    )
end

function weighted_shack_hartmann_image()
    return Float32[
        1 2 3 4 5 6
        7 8 9 10 11 12
        13 14 15 16 17 18
        19 20 21 22 23 24
    ]
end

function weighted_shack_hartmann_outputs(prepared)
    count = length(prepared.plan.sensing.regions.origins)
    return zeros(Float32, count, 2), zeros(Float32, count), falses(count)
end

function weighted_shack_hartmann_allocation_bytes(
    slopes,
    flux,
    validity,
    prepared,
    image,
)
    return @allocated process!(slopes, flux, validity, prepared, image)
end

function weighted_shack_hartmann_admitted_allocation_bytes(
    slopes,
    flux,
    validity,
    prepared,
    image,
)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        slopes,
        flux,
        validity,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

function selected_pixels(image, sensing, subaperture)
    rows, columns = sensing.regions.region_size
    first_row, first_column = sensing.regions.origins[subaperture]
    pixels = Vector{Float32}(undef, rows * columns)
    pixel = 1
    for row in 1:rows, column in 1:columns
        pixels[pixel] = image[first_row + row, first_column + column]
        pixel += 1
    end
    return pixels
end

function independent_weighted_oracle(plan, image, subaperture)
    sensing = plan.sensing
    rows, columns = sensing.regions.region_size
    first_row, first_column = sensing.regions.origins[subaperture]
    x_moment = 0.0f0
    y_moment = 0.0f0
    measured_flux = 0.0f0
    pixel = 1
    for row in 1:rows, column in 1:columns
        value = image[first_row + row, first_column + column]
        if !isfinite(value) || value >= sensing.pixel_thresholds[subaperture]
            weighted_value = isnothing(plan.weights) ? value : plan.weights[subaperture, pixel] * value
            if isnothing(plan.weights)
                x_moment += weighted_value * sensing.x_coordinates[pixel]
                y_moment += weighted_value * sensing.y_coordinates[pixel]
            else
                x_moment = fma(weighted_value, sensing.x_coordinates[pixel], x_moment)
                y_moment = fma(weighted_value, sensing.y_coordinates[pixel], y_moment)
            end
            measured_flux += weighted_value
        end
        pixel += 1
    end
    valid = isfinite(measured_flux) && measured_flux > 0.0f0 &&
        measured_flux >= sensing.flux_thresholds[subaperture]
    if valid
        reference = 2 * subaperture - 1
        x_offset = x_moment / measured_flux - sensing.reference_slopes[reference]
        y_offset = y_moment / measured_flux - sensing.reference_slopes[reference + 1]
        valid = isfinite(x_offset) && isfinite(y_offset)
        if valid && sensing.active[subaperture]
            return Float32[x_offset, y_offset], measured_flux, true
        end
    end
    return zeros(Float32, 2), measured_flux, valid
end

@testset "weighted Shack-Hartmann declaration has distinct weighted artifacts" begin
    prepared = weighted_shack_hartmann_prepared()
    fields = algorithm_configuration(ShackHartmannWeightedCentroidF32)
    @test algorithm_label(ShackHartmannWeightedCentroidF32) ==
          "shack-hartmann-weighted-centroid-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannWeightedCentroidF32,
    ) === Union{}
    @test algorithm_operation(ShackHartmannWeightedCentroidF32) isa
          ShackHartmannWeightedCentroidF32Operation
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
        :initial_weights,
        :simd_width,
    )
    @test all(field -> field.update isa GraphRebuildConfiguration, fields[1:4])
    @test all(field -> field.update isa ConstructionConfiguration, fields[5:12])

    ports = algorithm_ports(ShackHartmannWeightedCentroidF32)
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
        :weights,
    )
    @test all(port -> port.metadata_source === :image, ports[2:4])
    @test all(port -> port.role isa FilterGraphAlgorithms.SparseParameter, ports[5:10])
    @test getproperty.(algorithm_formats(prepared), :shape) == (
        (4, 6),
        (2, 2),
        (2,),
        (2,),
        (2, 2),
        (6, 2),
        (2, 2),
        (2, 2),
        (2,),
        (2, 6),
    )
    @test getproperty.(algorithm_formats(prepared), :schema) == (
        CALIBRATED_PIXELS_V1,
        SHACK_HARTMANN_SLOPES_V1,
        WEIGHTED_SHACK_HARTMANN_FLUX_V1,
        SHACK_HARTMANN_VALIDITY_V1,
        REGION_ORIGINS_V1,
        SHACK_HARTMANN_COORDINATES_V1,
        SHACK_HARTMANN_REFERENCE_SLOPES_V1,
        SHACK_HARTMANN_THRESHOLDS_V1,
        SHACK_HARTMANN_ACTIVE_V1,
        SHACK_HARTMANN_WEIGHTS_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, algorithm_formats(prepared))
    @test prepared.plan.sensing isa ShackHartmannImagePlan
    @test prepared.plan.weights isa Matrix{Float32}
    @test size(prepared.plan.weights) == (2, 6)
end

@testset "weighted Shack-Hartmann matches standalone and independent centroids" begin
    prepared = weighted_shack_hartmann_prepared()
    image = weighted_shack_hartmann_image()
    slopes, flux, validity = weighted_shack_hartmann_outputs(prepared)
    @test @inferred(process!(slopes, flux, validity, prepared, image)) === nothing

    sensing = prepared.plan.sensing
    for subaperture in eachindex(sensing.regions.origins)
        oracle_slopes, oracle_flux, oracle_validity = independent_weighted_oracle(
            prepared.plan,
            image,
            subaperture,
        )
        @test vec(slopes[subaperture, :]) == oracle_slopes
        @test flux[subaperture] == oracle_flux
        @test validity[subaperture] == oracle_validity

        coordinates = [
            Float32[sensing.x_coordinates[pixel], sensing.y_coordinates[pixel]] for
            pixel in eachindex(sensing.x_coordinates)
        ]
        centroid = prepare_algorithm(
            WeightedCenterOfGravityF32,
            WeightedCenterOfGravityF32Configuration(
                pixel_count=length(coordinates),
                initial_coordinates=coordinates,
                initial_weights=vec(copy(prepared.plan.weights[subaperture, :])),
            ),
        )
        reference = 2 * subaperture - 1
        centroid.plan = prepare_properties(
            centroid.plan,
            (
                PropertyAssignment(0, sensing.reference_slopes[reference]),
                PropertyAssignment(1, sensing.reference_slopes[reference + 1]),
                PropertyAssignment(2, true),
                PropertyAssignment(3, sensing.pixel_thresholds[subaperture]),
                PropertyAssignment(4, sensing.flux_thresholds[subaperture]),
            ),
        )
        offset = zeros(Float32, 2)
        standalone_flux = zeros(Float32, 1)
        standalone_validity = falses(1)
        process!(
            offset,
            standalone_flux,
            standalone_validity,
            centroid,
            selected_pixels(image, sensing, subaperture),
        )
        @test vec(slopes[subaperture, :]) == offset
        @test flux[subaperture] == only(standalone_flux)
        @test validity[subaperture] == only(standalone_validity)
    end
end

@testset "weighted Shack-Hartmann weights are owned and replaced atomically" begin
    initial_weights = deepcopy(WEIGHTS)
    prepared = weighted_shack_hartmann_prepared(initial_weights)
    initial_weights .= [fill(0.0f0, 6), fill(0.0f0, 6)]
    slopes, flux, validity = weighted_shack_hartmann_outputs(prepared)
    process!(slopes, flux, validity, prepared, weighted_shack_hartmann_image())
    @test all(validity)

    original = prepared.plan
    replacement = fill(1.0f0, 2, 6)
    @test replace_parameter!(prepared, :weights, replacement) === prepared
    @test prepared.plan !== original
    replacement .= 0.0f0
    @test all(==(1.0f0), prepared.plan.weights)
    process!(slopes, flux, validity, prepared, weighted_shack_hartmann_image())
    @test flux == Float32[29, 120]

    original = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(prepared, :weights, ones(Float32, 2, 5))
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :weights,
        Float32[1 1 1 1 1 1; 1 1 NaN 1 1 1],
    )
    @test prepared.plan === original
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :weights,
        Float32[1 1 1 1 1 1; 1 1 -1 1 1 1],
    )
    @test prepared.plan === original

    @test_throws DimensionMismatch weighted_shack_hartmann_prepared([Float32[1, 2]])
    @test_throws DimensionMismatch weighted_shack_hartmann_prepared(
        [ones(Float32, 6), ones(Float32, 5)],
    )
    @test_throws ArgumentError weighted_shack_hartmann_prepared(
        [ones(Float32, 6), Float32[1, 1, NaN, 1, 1, 1]],
    )
    @test_throws ArgumentError weighted_shack_hartmann_prepared(
        [ones(Float32, 6), Float32[1, 1, -1, 1, 1, 1]],
    )
end

@testset "weighted Shack-Hartmann preserves thresholding and rejects aliases" begin
    prepared = weighted_shack_hartmann_prepared()
    image = weighted_shack_hartmann_image()
    slopes, flux, validity = weighted_shack_hartmann_outputs(prepared)

    zero_weights = zeros(Float32, 2, 6)
    replace_parameter!(prepared, :weights, zero_weights)
    process!(slopes, flux, validity, prepared, image)
    @test slopes == zeros(Float32, 2, 2)
    @test flux == zeros(Float32, 2)
    @test validity == falses(2)

    image[1, 1] = NaN32
    process!(slopes, flux, validity, prepared, image)
    @test slopes[1, :] == zeros(Float32, 2)
    @test isnan(flux[1])
    @test !validity[1]

    prepared = weighted_shack_hartmann_prepared()
    image_storage = fill(71.0f0, 7, 12)
    image_view = @view image_storage[1:2:7, 2:2:12]
    image_view .= weighted_shack_hartmann_image()
    slope_storage = fill(83.0f0, 4, 4)
    slopes_view = @view slope_storage[1:2:3, 2:2:4]
    flux_storage = fill(89.0f0, 4)
    flux_view = @view flux_storage[1:2:3]
    validity_storage = fill(false, 4)
    validity_view = @view validity_storage[2:2:4]
    @test process!(
        slopes_view,
        flux_view,
        validity_view,
        prepared,
        image_view,
    ) === nothing
    @test all(validity_view)

    @test_throws ArgumentError process!(
        @view(prepared.plan.weights[:, 1:2]),
        zeros(Float32, 2),
        falses(2),
        prepared,
        weighted_shack_hartmann_image(),
    )
    @test_throws ArgumentError process!(
        zeros(Float32, 2, 2),
        @view(vec(prepared.plan.weights)[1:2]),
        falses(2),
        prepared,
        weighted_shack_hartmann_image(),
    )
    @test_throws ArgumentError process!(
        zeros(Float32, 2, 2),
        zeros(Float32, 2),
        falses(2),
        prepared,
        ZeroBasedWeightedImage(weighted_shack_hartmann_image()),
    )
end

@testset "weighted Shack-Hartmann checked and admitted calls allocate no heap" begin
    prepared = weighted_shack_hartmann_prepared()
    image = weighted_shack_hartmann_image()
    slopes, flux, validity = weighted_shack_hartmann_outputs(prepared)
    process!(slopes, flux, validity, prepared, image)
    @test weighted_shack_hartmann_allocation_bytes(
        slopes,
        flux,
        validity,
        prepared,
        image,
    ) == 0
    @test @inferred(FilterGraphAlgorithms._process_admitted!(
        slopes,
        flux,
        validity,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )) === nothing
    @test weighted_shack_hartmann_admitted_allocation_bytes(
        slopes,
        flux,
        validity,
        prepared,
        image,
    ) == 0
end


function packed_fixture(count=64; simd_width=0, subaperture_size=(8, 8))
    rows, columns = subaperture_size
    side = max(8, isqrt(count - 1) + 1)
    image_size = (side * rows, side * columns)
    origins = reverse([[i * rows, j * columns] for i in 0:(side - 1) for j in 0:(side - 1)])[1:count]
    weights = [[Float32(mod(i + 3j, 13)) / 7f0 for j in 1:(rows * columns)] for i in 1:count]
    prepared=prepare_algorithm(ShackHartmannWeightedCentroidF32,
        ShackHartmannWeightedCentroidF32Configuration(
            image_size=image_size,image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=subaperture_size,subaperture_count=count,
            initial_subaperture_origins=origins,coordinate_scale=.75f0,
            pixel_threshold=.5f0,flux_threshold=1f0,
            reference_slopes=fill(.125f0,2count),active=[i%3!=0 for i in 1:count],
            initial_weights=weights, simd_width=simd_width))
    image = [Float32(mod(7i + 11j, 31) - 3) / 8f0 for i in 1:image_size[1], j in 1:image_size[2]]
    return prepared,image
end

function check_packed_oracle(prepared,image)
    slopes,flux,validity=weighted_shack_hartmann_outputs(prepared)
    @test process!(slopes,flux,validity,prepared,image) === nothing
    for subaperture in eachindex(flux)
        expected,expected_flux,expected_valid=independent_weighted_oracle(prepared.plan,image,subaperture)
        @test reinterpret(UInt32,vec(slopes[subaperture,:])) == reinterpret(UInt32,expected)
        @test reinterpret(UInt32,flux[subaperture]) == reinterpret(UInt32,expected_flux)
        @test validity[subaperture] == expected_valid
    end
    return slopes,flux,validity
end

@testset "packed weights own calibration at the packing boundary" begin
    small,image=packed_fixture(63; subaperture_size=(1, 64))
    @test isnothing(small.plan.packed_weights)
    check_packed_oracle(small,image)
    prepared,image=packed_fixture(; subaperture_size=(1, 64))
    cache=prepared.plan.packed_weights
    @test cache isa Matrix{Float32}
    @test cache == permutedims(prepared.plan.weights)
    @test !Base.mightalias(cache,prepared.plan.weights)
    check_packed_oracle(prepared,image)
    old=prepared.plan
    weights=fill(.5f0,64,64)
    replace_parameter!(prepared,:weights,weights)
    weights .= 0
    @test prepared.plan.packed_weights == fill(.5f0,64,64)
    @test old.packed_weights === cache
    @test !Base.mightalias(old.packed_weights, prepared.plan.packed_weights)
    check_packed_oracle(prepared,image)
    replacements=(
        :coordinates=>hcat(old.sensing.x_coordinates,old.sensing.y_coordinates),
        Symbol("reference-slopes")=>fill(.25f0,64,2),
        :thresholds=>hcat(fill(.75f0,64),fill(2f0,64)),
        :active=>fill(true,64),
        Symbol("subaperture-origins")=>UInt32.(hcat(first.(old.sensing.regions.origins),last.(old.sensing.regions.origins))),
    )
    for (name,values) in replacements
        prior=prepared.plan
        replace_parameter!(prepared,name,values)
        @test prepared.plan !== prior
        @test prepared.plan.weights === prior.weights
        @test prepared.plan.packed_weights === prior.packed_weights
        check_packed_oracle(prepared,image)
    end
end

@testset "SIMD centroid preparation keeps canonical weights" begin
    for (count, width) in ((1024, 0), (65, 4), (69, 8), (69, 16))
        prepared, image = packed_fixture(count; simd_width=width)
        @test isnothing(prepared.plan.packed_weights)
        outputs = check_packed_oracle(prepared, image)
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
        @test weighted_shack_hartmann_admitted_allocation_bytes(outputs..., prepared, image) == 0

        if count == 65
            image_view = @view image[:, :]
            check_packed_oracle(prepared, image_view)
            @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image_view) == 0
            views = (view(fill(83f0, count, 2), :, :), view(fill(89f0, count), :), falses(count))
            @test process!(views..., prepared, image) === nothing
            @test all(isequal.(views, outputs))
            @test weighted_shack_hartmann_allocation_bytes(views..., prepared, image) == 0
        end

        weights = fill(0.5f0, size(prepared.plan.weights))
        replace_parameter!(prepared, :weights, weights)
        fill!(weights, 0f0)
        @test all(==(0.5f0), prepared.plan.weights)
        @test isnothing(prepared.plan.packed_weights)
        check_packed_oracle(prepared, image)
    end
end

@testset "large scalar windows prepare packed weights" begin
    for (count, expected) in ((15, false), (16, true))
        prepared, image = packed_fixture(count; subaperture_size=(16, 16))
        @test (prepared.plan.packed_weights isa Matrix{Float32}) == expected
        outputs = check_packed_oracle(prepared, image)
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
        @test weighted_shack_hartmann_admitted_allocation_bytes(outputs..., prepared, image) == 0
    end
end

@testset "packed weights preserve nonfinite, zero and overflow behavior" begin
    prepared,image=packed_fixture(; subaperture_size=(1, 64))
    first_row,first_column=first(prepared.plan.sensing.regions.origins)
    weights=copy(prepared.plan.weights);weights[1,1]=0f0
    replace_parameter!(prepared,:weights,weights)
    for value in (NaN32,Inf32,-Inf32,floatmax(Float32),-0f0,0f0,.5f0,prevfloat(.5f0))
        image[first_row+1,first_column+1]=value
        check_packed_oracle(prepared,image)
    end
    weights[1,1]=floatmax(Float32)
    replace_parameter!(prepared,:weights,weights)
    image[first_row+1,first_column+1]=4f0
    check_packed_oracle(prepared,image)
    image .= -0f0
    replace_parameter!(prepared,:thresholds,zeros(Float32,64,2))
    check_packed_oracle(prepared,image)
end

@testset "packed weights preserve views, ranges, aliases and zero allocation" begin
    prepared,image=packed_fixture(; subaperture_size=(1, 64))
    expected=check_packed_oracle(prepared,image)
    rows, columns = size(image)
    storage = fill(71f0, 2rows, 2columns)
    view_image = @view storage[1:2:(2rows), 2:2:(2columns)]
    view_image .= image
    slope_storage=fill(83f0,128,4);slopes=@view slope_storage[1:2:128,2:2:4]
    flux_storage=fill(89f0,128);flux=@view flux_storage[1:2:128]
    valid_storage=fill(false,128);valid=@view valid_storage[2:2:128]
    process!(slopes,flux,valid,prepared,view_image)
    @test reinterpret(UInt32,vec(collect(slopes))) == reinterpret(UInt32,vec(expected[1]))
    @test reinterpret(UInt32,collect(flux)) == reinterpret(UInt32,expected[2])
    @test valid == expected[3]
    @test all(==(83f0),slope_storage[2:2:128,:])
    @test all(==(89f0),flux_storage[2:2:128])
    @test weighted_shack_hartmann_allocation_bytes(slopes, flux, valid, prepared, view_image) == 0
    cache=prepared.plan.packed_weights
    @test_throws ArgumentError process!(@view(cache[:,1:2]),zeros(Float32,64),falses(64),prepared,image)
    @test_throws ArgumentError process!(zeros(Float32,64,2),@view(vec(cache)[1:64]),falses(64),prepared,image)
    for selection in (5:27,[64,1,31,4],1:0)
        out,fl,va=fill(83f0,64,2),fill(89f0,64),fill(false,64)
        FilterGraphAlgorithms._measure_shack_hartmann_weighted_centroid_range!(out,fl,va,prepared.plan,image,selection)
        for i in 1:64
            @test out[i,:] == (i in selection ? expected[1][i,:] : fill(83f0,2))
            @test fl[i] == (i in selection ? expected[2][i] : 89f0)
            @test va[i] == (i in selection ? expected[3][i] : false)
        end
    end
    for count in (63,64)
        p,img=packed_fixture(count; subaperture_size=(1, 64));out,fl,va=weighted_shack_hartmann_outputs(p)
        process!(out,fl,va,p,img)
        @test weighted_shack_hartmann_allocation_bytes(out,fl,va,p,img)==0
        @test weighted_shack_hartmann_admitted_allocation_bytes(out,fl,va,p,img)==0
    end
end

@testset "centroid lenslet blocks preserve tails and rectangular-window boundaries" begin
    cases = [(weighted, shape) for weighted in (false, true)
             for shape in ((2, 3), (4, 16), (8, 8), (8, 9), (1, 64), (1, 65), (15, 1), (16, 1))]
    append!(cases, [(true, (16, 16)), (true, (16, 17)),
                    (true, (1, 256)), (true, (1, 257))])
    for (weighted, shape) in cases, width in (weighted ? (0, 8, 16) : (0,)),
        count in (shape == (2, 3) ? (3, 4, 5, 6, 7, 8, 9) : (9,))
        rows, columns = shape
        origins = reverse([[r * rows, c * columns] for r in 0:2 for c in 0:2])[1:count]
        common = (;image_size=(3rows, 3columns), image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=shape, subaperture_count=count,
            initial_subaperture_origins=origins, coordinate_scale=.75f0,
            pixel_threshold=2f0, flux_threshold=1f0,
            reference_slopes=Float32[(-1)^i * .125f0 for i in 1:2count],
            active=Bool[i % 3 != 0 for i in 1:count])
        weights = [Float32[mod(i + 3j, 13) / 7f0 for j in 1:rows*columns] for i in 1:count]
        prepared = weighted ? prepare_algorithm(ShackHartmannWeightedCentroidF32,
            ShackHartmannWeightedCentroidF32Configuration(;common..., initial_weights=weights, simd_width=width)) :
            prepare_algorithm(ShackHartmannImageF32, ShackHartmannImageF32Configuration(;common...))
        if count == 9 && shape == (2, 3)
            # Per-lenslet thresholds and signed coordinates; repeated origins are read-only.
            replace_parameter!(prepared, :thresholds,
                hcat(Float32[isodd(i) ? 2 : 0 for i in 1:count], Float32[mod(i, 3) for i in 1:count]))
            replace_parameter!(prepared, :coordinates,
                hcat(Float32[(-1)^i * i / 8 for i in 1:rows*columns],
                     Float32[(-1)^(i+1) * i / 4 for i in 1:rows*columns]))
            origins[end] = origins[1]
            replace_parameter!(prepared, Symbol("subaperture-origins"),
                UInt32.(reduce(vcat, permutedims.(origins))))
        end
        sensing = weighted ? prepared.plan.sensing : prepared.plan
        oracle = (;sensing, weights=weighted ? prepared.plan.weights : nothing)
        image = [Float32(mod(7r + 11c, 41) - 3) / 8f0 for r in 1:3rows, c in 1:3columns]
        outputs = (zeros(Float32, count, 2), zeros(Float32, count), falses(count))
        function check(values, buffers=outputs)
            @test process!(buffers..., prepared, values) === nothing
            for subaperture in 1:count
                slopes, flux, valid = independent_weighted_oracle(oracle, values, subaperture)
                @test reinterpret(UInt32, vec(buffers[1][subaperture, :])) == reinterpret(UInt32, slopes)
                @test reinterpret(UInt32, buffers[2][subaperture]) == reinterpret(UInt32, flux)
                @test buffers[3][subaperture] == valid
            end
        end
        check(image)
        # Same first pixel in the last block/tail, with exact threshold and nonfinite cases.
        r, c = origins[end]
        retained = image[r+1, c+1]
        values = shape == (2, 3) && count in (4, 5, 8, 9) ?
            (2f0, prevfloat(2f0), -0.0f0, nextfloat(0f0),
             reinterpret(Float32, 0x7fc01234), Inf32, -Inf32, floatmax(Float32)) : ()
        for value in values
            image[r+1, c+1] = value
            check(image)
        end
        image[r+1, c+1] = retained
        count == 9 && shape in ((2, 3), (16, 17)) && check(@view image[:, :])
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
        @test weighted_shack_hartmann_admitted_allocation_bytes(outputs..., prepared, image) == 0
    end
end

@testset "weighted lenslet blocks and scalar tails use the same fused moments" begin
    for width in (4, 8, 16), (values, coordinates, weights) in (
        (Float32[1, 1 + 2.0^-12], Float32[-(1 + 2.0^-11), 1 + 2.0^-12], ones(Float32, 2)),
        (Float32[1, 2], Float32[-floatmax(Float32), floatmax(Float32)], ones(Float32, 2)),
        (Float32[1, 1 + 2.0^-12], Float32[-(1 + 2.0^-11), 1], Float32[1, 1 + 2.0^-12]),
    )
        count = width + 5
        prepared = prepare_algorithm(ShackHartmannWeightedCentroidF32,
            ShackHartmannWeightedCentroidF32Configuration(
                image_size=(2count, 2), image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(2, 2), subaperture_count=count,
                initial_subaperture_origins=[[2(i-1), 0] for i in 1:count],
                coordinate_scale=1f0, pixel_threshold=0f0, flux_threshold=0f0,
                reference_slopes=zeros(Float32, 2count), active=fill(true, count),
                initial_weights=[vcat(weights, ones(Float32, 2)) for _ in 1:count], simd_width=width))
        padded_coordinates = vcat(coordinates, zeros(Float32, 2))
        replace_parameter!(prepared, :coordinates, hcat(padded_coordinates, padded_coordinates))
        image = zeros(Float32, 2count, 2)
        image[1:2:2count-1, :] .= permutedims(values)
        weighted_pixels = weights .* values
        expected_flux = sum(weighted_pixels)
        outputs = weighted_shack_hartmann_outputs(prepared)
        expected = setprecision(BigFloat, 1024) do
            Float32(sum(BigFloat.(weighted_pixels) .* BigFloat.(coordinates)) / BigFloat(expected_flux))
        end
        for input in (image, @view(image[:, :]))
            @test process!(outputs..., prepared, input) === nothing
            @test outputs[1] == fill(expected, count, 2)
            @test all(outputs[3])
            @test outputs[2] == fill(expected_flux, count)
        end
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
    end
end

@testset "weighted lenslet masks preserve every lane and scalar tail" begin
    for width in (4, 8, 16)
        count = width + 5
        prepared = prepare_algorithm(ShackHartmannWeightedCentroidF32,
            ShackHartmannWeightedCentroidF32Configuration(
                image_size=(2count, 2), image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(2, 2), subaperture_count=count,
                initial_subaperture_origins=[[2(i-1), 0] for i in 1:count],
                coordinate_scale=1f0, pixel_threshold=1f0, flux_threshold=0f0,
                reference_slopes=Float32[i / 8 for i in 1:2count],
                active=Bool[i != 3 for i in 1:count],
                initial_weights=[Float32[i / 4, 1, 1, 1] for i in 1:count], simd_width=width))
        image = zeros(Float32, 2count, 2)
        outputs = weighted_shack_hartmann_outputs(prepared)
        patterns = width <= 8 ? (0:(1 << width)-1) :
            unique(vcat([0, 65535, 21845, 43690], [1 << i for i in 0:15],
                [65535 ⊻ (1 << i) for i in 0:15], [mod(7919i,65536) for i in 1:64]))
        for bits in patterns, selected_value in (3f0, NaN32, Inf32, -Inf32)
            for lane in 1:count
                selected = lane <= width ? !iszero(bits & (1 << (lane-1))) : isodd(bits)
                image[2lane-1, 1] = selected ? selected_value : .5f0
                image[2lane-1, 2] = 1f0 # Inclusive threshold; keeps finite cases valid.
            end
            @test process!(outputs..., prepared, image) === nothing
            for lane in 1:count
                slopes, flux, valid = independent_weighted_oracle(prepared.plan, image, lane)
                @test reinterpret(UInt32, outputs[1][lane, :]) == reinterpret(UInt32, slopes)
                @test reinterpret(UInt32, outputs[2][lane]) == reinterpret(UInt32, flux)
                @test outputs[3][lane] == valid
            end
        end
        # Selected subnormals must survive weighting and packed flux addition.
        # A large coordinate makes their moment visible instead of rounding into zero.
        replace_parameter!(prepared, :weights, ones(Float32, count, 4))
        replace_parameter!(prepared, :thresholds, zeros(Float32, count, 2))
        large_coordinate = Float32(2.0^126)
        coordinates = Float32[large_coordinate, 0, 0, 0]
        replace_parameter!(prepared, :coordinates, hcat(coordinates, -coordinates))
        for value in (nextfloat(0f0), Float32(2.0^-126))
            fill!(image, 0f0)
            image[1:2:end, 1] .= value
            @test process!(outputs..., prepared, image) === nothing
            @test outputs[2] == fill(value, count)
            @test all(outputs[3])
            @test outputs[1] == [i == 3 ? 0f0 : sign * large_coordinate
                                for i in 1:count, sign in (1f0, -1f0)]
        end
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
        @test weighted_shack_hartmann_admitted_allocation_bytes(outputs..., prepared, image) == 0
    end
end

@testset "SIMD width defaults, validation, tails and replacement" begin
    default = weighted_shack_hartmann_prepared()
    @test default.plan.simd_width == 4
    fields = algorithm_configuration(ShackHartmannWeightedCentroidF32)
    @test fields[end].default == Some(0)
    @test all(field -> field.default === nothing, fields[1:end-1])
    @test ShackHartmannWeightedCentroidF32Configuration(
        (4, 6), CALIBRATED_PIXELS_V1, (2, 3), 2, [[0,0], [2,3]],
        1f0, 2f0, 4f0, zeros(Float32, 4), fill(true, 2), WEIGHTS).simd_width == 0
    for invalid in (-1, 1, 2, 3, 5, 32)
        @test_throws ArgumentError ShackHartmannWeightedCentroidPlan(
            default.plan.sensing, default.plan.weights; simd_width=invalid)
    end
    for width in (0, 4, 8, 16), count in 1:37
        prepared, image = packed_fixture(count; simd_width=width)
        @test prepared.plan.simd_width == (width == 0 ? 4 : width)
        outputs = check_packed_oracle(prepared, image)
        @test weighted_shack_hartmann_allocation_bytes(outputs..., prepared, image) == 0
        @test weighted_shack_hartmann_admitted_allocation_bytes(outputs..., prepared, image) == 0
    end
    for width in (8, 16)
        prepared, image = packed_fixture(37; simd_width=width)
        check_packed_oracle(prepared, @view image[:, :])
        old = prepared.plan
        replacements = (
            :weights => copy(old.weights),
            :coordinates => hcat(old.sensing.x_coordinates, old.sensing.y_coordinates),
            Symbol("reference-slopes") => zeros(Float32, 37, 2),
            :thresholds => zeros(Float32, 37, 2),
            :active => fill(true, 37),
            Symbol("subaperture-origins") => UInt32.(hcat(first.(old.sensing.regions.origins), last.(old.sensing.regions.origins))),
        )
        for (name, values) in replacements
            replace_parameter!(prepared, name, values)
            @test prepared.plan.simd_width == width
            check_packed_oracle(prepared, image)
        end
    end
end

end
