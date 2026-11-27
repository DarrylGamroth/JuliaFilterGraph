module PixelCalibrationRowTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pixel_calibration_row.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function row_process_allocation_bytes(output, prepared, input, metadata)
    return @allocated process!(output, prepared, input, metadata)
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "row-block pixel calibration matches the repository fixture" begin
    fixture = FIXTURE
    configuration = fixture["configuration"]
    rows = configuration["image_size"][1]
    columns = configuration["image_size"][2]
    block_rows = configuration["row_block_rows"]
    prepared = prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(rows, columns),
            row_block_rows=block_rows,
            initial_flat=Float32.(configuration["initial_flat"]),
            initial_background=Float32.(configuration["initial_background"]),
        ),
    )

    @test algorithm_label(PixelCalibrationRowU16F32) ==
          "pixel-calibration-row-u16-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PixelCalibrationRowU16F32) ===
          PixelCalibrationRowFailure
    ports = algorithm_ports(PixelCalibrationRowU16F32)
    @test map(port -> port.name, ports) == (:raw, :calibrated, :flat, :background)
    @test ports[1].role isa FilterGraphAlgorithms.FrameData
    @test ports[2].metadata_source === :raw
    @test !ports[2].conditional
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[3:4],
    )

    formats = algorithm_formats(prepared)
    @test formats[1].shape == (block_rows, columns)
    @test formats[1].schema == RAW_PIXEL_ROW_BLOCK_V1
    @test formats[1].rate == (0x00000002, 0x00000001)
    @test formats[2].schema == CALIBRATED_PIXEL_ROW_BLOCK_V1
    @test formats[2].rate == formats[1].rate
    @test formats[3].shape == (rows, columns)
    @test formats[3].schema == FLAT_CALIBRATION_V1
    @test formats[4].schema == BACKGROUND_CALIBRATION_V1

    output = fill(-1.0f0, block_rows, columns)
    first = fixture["samples"][1]
    first_input = row_major_matrix(UInt16, first["raw"], block_rows, columns)
    first_metadata = SampleMetadata(
        first["sequence"],
        first["offset"];
        timestamp_nanoseconds=first["timestamp_nanoseconds"],
        terminal=first["terminal"],
    )
    result = process!(output, prepared, first_input, (raw=first_metadata,))
    @test result == AlgorithmProcessComplete(first_metadata)
    @test vec(permutedims(output)) == Float32.(first["calibrated"])

    replacement = row_major_matrix(
        Float32,
        fixture["replacement"]["flat"],
        rows,
        columns,
    )
    @test replace_parameter!(prepared, :flat, replacement) === prepared

    last = fixture["samples"][2]
    last_input = row_major_matrix(UInt16, last["raw"], block_rows, columns)
    last_metadata = SampleMetadata(
        last["sequence"],
        last["offset"];
        timestamp_nanoseconds=last["timestamp_nanoseconds"],
        terminal=last["terminal"],
    )
    result = process!(output, prepared, last_input, (raw=last_metadata,))
    expected_metadata = SampleMetadata(
        last_metadata;
        discontinuity=last["expected_discontinuity"],
    )
    @test result == AlgorithmProcessComplete(expected_metadata)
    @test vec(permutedims(output)) == Float32.(last["calibrated"])

    process!(output, prepared, last_input, (raw=last_metadata,))
    @test row_process_allocation_bytes(
        output,
        prepared,
        last_input,
        (raw=last_metadata,),
    ) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0
end

@testset "row metadata rejection preserves output and candidate state" begin
    prepared = prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(4, 3),
            row_block_rows=2,
            initial_flat=nothing,
            initial_background=nothing,
        ),
    )
    input = reshape(UInt16.(1:6), 2, 3)
    output = fill(71.0f0, 2, 3)

    @test process!(output, prepared, input) === PixelCalibrationRowMetadataMissing
    @test output == fill(71.0f0, 2, 3)
    @test process!(
        output,
        prepared,
        input,
        (raw=SampleMetadata(1, 1),),
    ) === PixelCalibrationRowOffsetMisaligned
    @test output == fill(71.0f0, 2, 3)
    @test process!(
        output,
        prepared,
        input,
        (raw=SampleMetadata(1, 4; terminal=true),),
    ) === PixelCalibrationRowOutOfBounds
    @test process!(
        output,
        prepared,
        input,
        (raw=SampleMetadata(1, 0; terminal=true),),
    ) === PixelCalibrationRowTerminalMismatch

    missing_metadata = (raw=nothing,)
    process!(output, prepared, input, missing_metadata)
    @test row_process_allocation_bytes(
        output,
        prepared,
        input,
        missing_metadata,
    ) == 0

    reset!(prepared)
    valid_metadata = SampleMetadata(2, 0)
    result = process!(output, prepared, input, (raw=valid_metadata,))
    @test result.has_metadata
    @test result.metadata.discontinuity
end

@testset "invalid row calibration is rejected during preparation" begin
    @test_throws ArgumentError prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(4, 3),
            row_block_rows=3,
            initial_flat=nothing,
            initial_background=nothing,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(4, 3),
            row_block_rows=2,
            initial_flat=ones(Float32, 11),
            initial_background=nothing,
        ),
    )
    invalid_flat = ones(Float32, 12)
    invalid_flat[2] = NaN32
    @test_throws ArgumentError prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(4, 3),
            row_block_rows=2,
            initial_flat=invalid_flat,
            initial_background=nothing,
        ),
    )
end

end
