module PixelSelectionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pixel_selection.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function selection_prepared()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PixelSelectionF32,
        PixelSelectionF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            mask=configuration["mask"],
            input_schema=configuration["input_schema"],
            output_schema=configuration["output_schema"],
        ),
    )
end

selection_allocation_bytes(output, prepared, image) =
    @allocated process!(output, prepared, image)

selection_admitted_allocation_bytes(output, prepared, image) = @allocated(
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
)

@testset "pixel selection preserves the FGN row-major oracle" begin
    prepared = selection_prepared()
    fixture = FIXTURE
    configuration = fixture["configuration"]
    image = row_major_matrix(
        Float32,
        fixture["sample"]["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
    output = fill(-1.0f0, count(configuration["mask"]))

    @test algorithm_label(PixelSelectionF32) == "pixel-selection-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PixelSelectionF32) === Union{}
    @test map(field -> field.name, algorithm_configuration(PixelSelectionF32)) ==
          (:image_size, :mask, :input_schema, :output_schema)
    ports = algorithm_ports(PixelSelectionF32)
    @test map(port -> port.name, ports) == (:image, Symbol("selected-pixels"), :mask)
    @test ports[2].metadata_source === :image
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3, 4), (6,), (3, 4))
    @test getproperty.(formats, :schema) == (
        configuration["input_schema"],
        configuration["output_schema"],
        PIXEL_SELECTION_MASK_V1,
    )

    @test process!(output, prepared, image) === nothing
    @test output == Float32.(fixture["sample"]["selected"])

    replacement = row_major_matrix(Bool, fixture["replacement"]["mask"], 3, 4)
    @test replace_parameter!(prepared, :mask, replacement) === prepared
    @test process!(output, prepared, image) === nothing
    @test output == Float32.(fixture["replacement"]["selected"])
end

@testset "pixel selection supports logical CPU views and raw values" begin
    prepared = selection_prepared()
    image_storage = fill(71.0f0, 6, 8)
    image = @view image_storage[1:2:5, 2:2:8]
    image .= row_major_matrix(Float32, FIXTURE["sample"]["image"], 3, 4)
    output_storage = fill(83.0f0, 12)
    output = @view output_storage[2:2:12]

    @test process!(output, prepared, image) === nothing
    @test output == Float32.(FIXTURE["sample"]["selected"])
    @test output_storage[1:2:11] == fill(83.0f0, 6)

    unusual = copy(image)
    unusual[1, 2] = NaN32
    unusual[3, 4] = Inf32
    @test process!(output, prepared, unusual) === nothing
    @test isnan(output[1])
    @test output[end] == Inf32

    aliased_image = row_major_matrix(Float32, FIXTURE["sample"]["image"], 3, 4)
    aliased_output = @view vec(aliased_image)[1:6]
    @test_throws ArgumentError process!(aliased_output, prepared, aliased_image)
end

@testset "pixel-selection preparation and replacement are failure-atomic" begin
    @test_throws DimensionMismatch prepare_algorithm(
        PixelSelectionF32,
        PixelSelectionF32Configuration(
            image_size=(2, 3),
            mask=Bool[true, false],
            input_schema=CALIBRATED_PIXELS_V1,
            output_schema="org.calculon.test.selected/1",
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PixelSelectionF32,
        PixelSelectionF32Configuration(
            image_size=(2, 3),
            mask=fill(false, 6),
            input_schema=CALIBRATED_PIXELS_V1,
            output_schema="org.calculon.test.selected/1",
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PixelSelectionF32,
        PixelSelectionF32Configuration(
            image_size=(2, 3),
            mask=Bool[true, false, true, false, true, false],
            input_schema="invalid-schema",
            output_schema="org.calculon.test.selected/1",
        ),
    )

    prepared = selection_prepared()
    original = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :mask,
        reshape([true; fill(false, 11)], 3, 4),
    )
    @test prepared.plan === original
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :mask,
        fill(true, 2, 6),
    )
    @test prepared.plan === original
end

@testset "pixel-selection checked and admitted calls allocate no heap" begin
    prepared = selection_prepared()
    image = row_major_matrix(Float32, FIXTURE["sample"]["image"], 3, 4)
    output = zeros(Float32, 6)
    process!(output, prepared, image)
    @test selection_allocation_bytes(output, prepared, image) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test selection_admitted_allocation_bytes(output, prepared, image) == 0
end

@testset "packed selection retains rectangular coordinates and replacement" begin
    rows, columns = 19, 43
    bits = UInt32.(1:(rows * columns))
    bits[1:6] .= (0x7fc00121, 0x7f800121, 0x80000000, 0x7f800000, 0xff800000, 0x00000000)
    image = row_major_matrix(Float32, reinterpret(Float32, bits), rows, columns)
    whole = trues(rows * columns)
    runs = falses(rows * columns)
    runs[1:32] .= true
    runs[39:86] .= true  # One run crosses a logical row boundary.
    fragmented = falses(rows * columns)
    fragmented[1:2:159] .= true

    for mask in (whole, runs)
        prepared = prepare_algorithm(
            PixelSelectionF32,
            PixelSelectionF32Configuration(
                image_size=(rows, columns), mask=mask,
                input_schema="test.detector/1", output_schema="test.selected-pixels/1",
            ),
        )
        output = zeros(Float32, count(mask))
        @test process!(output, prepared, image) === nothing
        @test reinterpret(UInt32, output) == bits[mask]
        @test selection_allocation_bytes(output, prepared, image) == 0

        rows_field = getfield(prepared.plan, :selected_rows)
        columns_field = getfield(prepared.plan, :selected_columns)
        rows_field[1] = rows
        columns_field[1] = columns
        @test process!(output, prepared, image) === nothing
        expected = bits[mask]
        expected[1] = bits[end]
        @test reinterpret(UInt32, output) == expected

        if mask === runs
            @test replace_parameter!(prepared, :mask,
                row_major_matrix(Bool, fragmented, rows, columns)) === prepared
            @test process!(output, prepared, image) === nothing
            @test reinterpret(UInt32, output) == bits[fragmented]
            @test replace_parameter!(prepared, :mask,
                row_major_matrix(Bool, runs, rows, columns)) === prepared
            @test process!(output, prepared, image) === nothing
            @test reinterpret(UInt32, output) == bits[runs]
        end
    end
end

@testset "externally supplied coordinate vectors remain live" begin
    prepared = selection_prepared()
    plan = prepared.plan
    rows = copy(plan.selected_rows)
    columns = copy(plan.selected_columns)
    prepared.plan = PixelSelectionPlan(
        plan.image_size, plan.mask, rows, columns, plan.input_schema, plan.output_schema,
    )
    @test prepared.plan.selected_rows === rows
    @test prepared.plan.selected_columns === columns
    rows[1] = 3
    columns[1] = 4
    image = row_major_matrix(Float32, FIXTURE["sample"]["image"], 3, 4)
    output = zeros(Float32, length(rows))
    @test process!(output, prepared, image) === nothing
    @test output == [image[r, c] for (r, c) in zip(rows, columns)]
end

end
