module PixelCalibrationTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pixel_calibration.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_prepared()
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            initial_flat=Float32.(configuration["initial_flat"]),
            initial_background=Float32.(configuration["initial_background"]),
        ),
    )
end

function pixel_calibration_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function pixel_calibration_admitted_allocation_bytes(output, prepared, input)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "complete-frame pixel calibration preserves the FGN contract" begin
    @test algorithm_label(PixelCalibrationU16F32) == "pixel-calibration-u16-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PixelCalibrationU16F32) ===
          Union{}
    @test map(
        field -> field.name,
        algorithm_configuration(PixelCalibrationU16F32),
    ) == (:image_size, :initial_flat, :initial_background)

    ports = algorithm_ports(PixelCalibrationU16F32)
    @test map(port -> port.name, ports) == (:raw, :calibrated, :flat, :background)
    @test ports[2].metadata_source === :raw
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[3:4],
    )

    prepared = fixture_prepared()
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2, 3), (2, 3), (2, 3), (2, 3))
    @test getproperty.(formats, :schema) == (
        RAW_DETECTOR_PIXELS_V1,
        CALIBRATED_PIXELS_V1,
        FLAT_CALIBRATION_V1,
        BACKGROUND_CALIBRATION_V1,
    )

    output = zeros(Float32, 2, 3)
    for sample in FIXTURE["samples"]
        input = row_major_matrix(UInt16, sample["raw"], 2, 3)
        @test process!(output, prepared, input) === nothing
        @test vec(permutedims(output)) == Float32.(sample["calibrated"])
    end
end

@testset "complete-frame calibration supports CPU views and replacement" begin
    prepared = fixture_prepared()
    raw_storage = fill(UInt16(99), 4, 5)
    raw = @view raw_storage[1:2:3, 1:2:5]
    raw .= row_major_matrix(UInt16, FIXTURE["samples"][1]["raw"], 2, 3)
    output_storage = fill(71.0f0, 4, 5)
    output = @view output_storage[2:2:4, 1:2:5]

    @test process!(output, prepared, raw) === nothing
    @test vec(permutedims(output)) ==
          Float32.(FIXTURE["samples"][1]["calibrated"])
    @test output_storage[1:2:3, :] == fill(71.0f0, 2, 5)

    flat_storage = fill(NaN32, 4, 5)
    flat = @view flat_storage[1:2:3, 1:2:5]
    fill!(flat, 2.0f0)
    @test replace_parameter!(prepared, :flat, flat) === prepared
    @test process!(output, prepared, raw) === nothing
    @test vec(permutedims(output)) == Float32[2, 4, 6, 8, 10, 12]

    background = zeros(Float32, 2, 3)
    @test replace_parameter!(prepared, :background, background) === prepared
    @test process!(output, prepared, raw) === nothing
    @test vec(permutedims(output)) == Float32[4, 6, 8, 10, 12, 14]
end

@testset "complete-frame calibration rejects invalid preparation atomically" begin
    @test_throws ArgumentError prepare_algorithm(
        PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(
            image_size=(0, 3),
            initial_flat=nothing,
            initial_background=nothing,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(
            image_size=(typemax(Int), 2),
            initial_flat=nothing,
            initial_background=nothing,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(
            image_size=(2, 3),
            initial_flat=ones(Float32, 5),
            initial_background=nothing,
        ),
    )

    prepared = fixture_prepared()
    original_plan = prepared.plan
    invalid = ones(Float32, 2, 3)
    invalid[1] = NaN32
    @test_throws ArgumentError replace_parameter!(prepared, :flat, invalid)
    @test prepared.plan === original_plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :background,
        zeros(Float32, 1, 3),
    )
    @test prepared.plan === original_plan
end

@testset "complete-frame calibration allocates no heap after warmup" begin
    prepared = fixture_prepared()
    sample = FIXTURE["samples"][1]
    input = row_major_matrix(UInt16, sample["raw"], 2, 3)
    output = zeros(Float32, 2, 3)

    process!(output, prepared, input)
    @test pixel_calibration_allocation_bytes(output, prepared, input) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test pixel_calibration_admitted_allocation_bytes(
        output,
        prepared,
        input,
    ) == 0
end

end
