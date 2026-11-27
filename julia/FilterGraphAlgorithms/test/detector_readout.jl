module DetectorReadoutTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "detector_readout.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function readout_block(::Type{T}, values, regions, lines, samples) where {T}
    block = Array{T}(undef, regions, lines, samples)
    source = 1
    for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        block[region, line, sample] = values[source]
        source += 1
    end
    return block
end

function readout_configuration(::Type{PixelCalibrationReadoutU16F32})
    configuration = FIXTURE["configuration"]
    return PixelCalibrationReadoutU16F32Configuration(
        detector_size=(configuration["detector_rows"], configuration["detector_columns"]),
        region_lines=configuration["region_lines"],
        block_lines=configuration["block_lines"],
        region_samples=configuration["region_samples"],
        detector_origins=configuration["detector_origins"],
        detector_line_steps=configuration["detector_line_steps"],
        detector_sample_steps=configuration["detector_sample_steps"],
        initial_flat=Float32.(configuration["initial_flat"]),
        initial_background=Float32.(configuration["initial_background"]),
    )
end

function readout_configuration(::Type{DetectorFrameAssemblerF32})
    configuration = FIXTURE["configuration"]
    return DetectorFrameAssemblerF32Configuration(
        detector_size=(configuration["detector_rows"], configuration["detector_columns"]),
        region_lines=configuration["region_lines"],
        block_lines=configuration["block_lines"],
        region_samples=configuration["region_samples"],
        detector_origins=configuration["detector_origins"],
        detector_line_steps=configuration["detector_line_steps"],
        detector_sample_steps=configuration["detector_sample_steps"],
    )
end

function calibration_allocation_bytes(output, prepared, input, metadata)
    return @allocated process!(output, prepared, input, metadata)
end

function assembler_allocation_bytes(output, prepared, input, metadata)
    return @allocated process!(output, prepared, input, metadata)
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "detector-readout mapping preserves FGN coordinate semantics" begin
    row_mapping = DetectorReadoutPlan(
        (4, 4),
        4,
        2,
        4,
        [DetectorReadoutRegion((0, 0), (1, 0), (0, 1))],
    )
    @test FilterGraphAlgorithms.detector_readout_detector_size(row_mapping) == (4, 4)
    @test FilterGraphAlgorithms.detector_readout_block_shape(row_mapping) == (1, 2, 4)
    @test FilterGraphAlgorithms.detector_readout_block_count(row_mapping) == 2
    @test FilterGraphAlgorithms.detector_readout_arrival_ordinal(row_mapping, 0, 3) == 0
    @test FilterGraphAlgorithms.detector_readout_arrival_ordinal(row_mapping, 3, 0) == 1
    @test isnothing(
        FilterGraphAlgorithms.detector_readout_arrival_ordinal(row_mapping, 4, 0),
    )

    opposed = DetectorReadoutPlan(
        (4, 4),
        2,
        1,
        4,
        [
            DetectorReadoutRegion((0, 0), (1, 0), (0, 1)),
            DetectorReadoutRegion((3, 0), (-1, 0), (0, 1)),
        ],
    )
    detector = row_major_matrix(UInt16, 0:15, 4, 4)
    @test FilterGraphAlgorithms._pack_detector_readout(opposed, detector) ==
          UInt16[0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]

    transposed = DetectorReadoutPlan(
        (2, 3),
        3,
        1,
        2,
        [DetectorReadoutRegion((0, 0), (0, 1), (1, 0))],
    )
    detector = row_major_matrix(UInt16, 0:5, 2, 3)
    @test FilterGraphAlgorithms._pack_detector_readout(transposed, detector) ==
          UInt16[0, 3, 1, 4, 2, 5]
end

@testset "invalid detector-readout mappings fail off path" begin
    detector_plane = DetectorReadoutPlan(
        (1, 2),
        1,
        1,
        2,
        [DetectorReadoutRegion((0, 0), (1, 0), (0, 1))],
    )
    @test_throws DimensionMismatch FilterGraphAlgorithms._detector_plane(
        detector_plane,
        Float32[1],
    )
    @test_throws DimensionMismatch FilterGraphAlgorithms._detector_plane(
        detector_plane,
        Float32[1, 2, 3],
    )

    @test_throws ArgumentError DetectorReadoutPlan(
        (2, 2),
        2,
        1,
        2,
        [DetectorReadoutRegion((0, 0), (-1, 0), (0, 1))],
    )
    @test_throws ArgumentError DetectorReadoutRegion((0, 0), (0, 0), (0, 1))
    @test_throws ArgumentError DetectorReadoutRegion((0, 0), (1, 0), (0, 0))
    @test_throws ArgumentError DetectorReadoutPlan(
        (2, 2),
        2,
        3,
        2,
        [DetectorReadoutRegion((0, 0), (1, 0), (0, 1))],
    )
    @test_throws ArgumentError FilterGraphAlgorithms._prepare_detector_readout(
        2,
        2,
        1,
        1,
        2,
        [[0, 0], [1, 0]],
        [[1, 0]],
        [[0, 1], [0, 1]],
    )
    @test_throws ArgumentError DetectorReadoutPlan(
        (2, 2),
        1,
        1,
        2,
        [
            DetectorReadoutRegion((0, 0), (1, 0), (0, 1)),
            DetectorReadoutRegion((0, 0), (1, 0), (0, 1)),
        ],
    )
end

@testset "readout calibration and frame assembly match the FGN oracle" begin
    calibration = prepare_algorithm(
        PixelCalibrationReadoutU16F32,
        readout_configuration(PixelCalibrationReadoutU16F32),
    )
    assembler = prepare_algorithm(
        DetectorFrameAssemblerF32,
        readout_configuration(DetectorFrameAssemblerF32),
    )
    configuration = FIXTURE["configuration"]
    regions = length(configuration["detector_origins"])
    calibrated = fill(
        -1.0f0,
        regions,
        configuration["block_lines"],
        configuration["region_samples"],
    )
    frame = fill(-1.0f0, configuration["detector_rows"], configuration["detector_columns"])

    @test algorithm_label(PixelCalibrationReadoutU16F32) ==
          "pixel-calibration-readout-u16-f32"
    @test algorithm_label(DetectorFrameAssemblerF32) ==
          "detector-frame-assembler-f32"
    calibration_ports = algorithm_ports(PixelCalibrationReadoutU16F32)
    @test map(port -> port.name, calibration_ports) ==
          (:raw, :calibrated, :flat, :background)
    assembler_ports = algorithm_ports(DetectorFrameAssemblerF32)
    @test map(port -> port.name, assembler_ports) ==
          (Symbol("readout-block"), Symbol("detector-frame"))
    @test assembler_ports[2].conditional

    calibration_formats = algorithm_formats(calibration)
    @test calibration_formats[1].shape == (2, 1, 4)
    @test calibration_formats[1].rate == (0x00000002, 0x00000001)
    @test calibration_formats[1].schema == RAW_PIXEL_READOUT_BLOCK_V1
    @test calibration_formats[2].schema == CALIBRATED_PIXEL_READOUT_BLOCK_V1
    @test algorithm_formats(assembler)[1].rate == calibration_formats[2].rate
    @test algorithm_formats(assembler)[2].shape == (4, 4)

    for block_fixture in FIXTURE["blocks"]
        raw = readout_block(
            UInt16,
            block_fixture["raw"],
            regions,
            configuration["block_lines"],
            configuration["region_samples"],
        )
        metadata = SampleMetadata(
            block_fixture["sequence"],
            block_fixture["ordinal"];
            timestamp_nanoseconds=block_fixture["timestamp_nanoseconds"],
            terminal=block_fixture["terminal"],
        )
        @test process!(calibrated, calibration, raw, (raw=metadata,)) ==
              AlgorithmProcessComplete(metadata)
        @test vec(permutedims(calibrated, (3, 2, 1))) ==
              Float32.(block_fixture["calibrated"])
        result = process!(
            frame,
            assembler,
            calibrated,
            (readout_block=metadata,),
        )
        if block_fixture["terminal"]
            @test result == AlgorithmProcessComplete(
                SampleMetadata(metadata; offset=0, terminal=true),
            )
        else
            @test result == AlgorithmProcessDeferred()
            @test frame == fill(-1.0f0, 4, 4)
        end
    end
    @test vec(permutedims(frame)) == Float32.(FIXTURE["expected"]["frame"])
end

@testset "pixel calibration is invariant across detector delivery forms" begin
    configuration = FIXTURE["configuration"]
    rows = configuration["detector_rows"]
    columns = configuration["detector_columns"]
    block_rows = 2
    flat = Float32.(configuration["initial_flat"])
    background = Float32.(configuration["initial_background"])

    complete = prepare_algorithm(
        PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(
            image_size=(rows, columns),
            initial_flat=flat,
            initial_background=background,
        ),
    )
    row = prepare_algorithm(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration(
            image_size=(rows, columns),
            row_block_rows=block_rows,
            initial_flat=flat,
            initial_background=background,
        ),
    )
    readout = prepare_algorithm(
        PixelCalibrationReadoutU16F32,
        readout_configuration(PixelCalibrationReadoutU16F32),
    )

    raw = reshape(UInt16.(1:Base.checked_mul(rows, columns)), rows, columns)
    complete_frame = zeros(Float32, rows, columns)
    @test process!(complete_frame, complete, raw) === nothing

    row_frame = zeros(Float32, rows, columns)
    row_output = zeros(Float32, block_rows, columns)
    for first_row in 0:block_rows:(rows - block_rows)
        terminal = first_row + block_rows == rows
        input = @view raw[(first_row + 1):(first_row + block_rows), :]
        metadata = SampleMetadata(71, first_row; terminal)
        @test process!(row_output, row, input, (raw=metadata,)) ==
              AlgorithmProcessComplete(metadata)
        copyto!(
            @view(row_frame[(first_row + 1):(first_row + block_rows), :]),
            row_output,
        )
    end

    mapping = readout.plan.readout
    packed = FilterGraphAlgorithms._pack_detector_readout(mapping, raw)
    readout_frame = zeros(Float32, rows, columns)
    readout_output = zeros(
        Float32,
        FilterGraphAlgorithms.detector_readout_block_shape(mapping),
    )
    for ordinal in 0:(mapping.block_count - 1)
        first = ordinal * mapping.block_elements + 1
        last = first + mapping.block_elements - 1
        input = readout_block(
            UInt16,
            @view(packed[first:last]),
            length(mapping.regions),
            mapping.block_lines,
            mapping.region_samples,
        )
        terminal = ordinal + 1 == mapping.block_count
        metadata = SampleMetadata(71, ordinal; terminal)
        @test process!(readout_output, readout, input, (raw=metadata,)) ==
              AlgorithmProcessComplete(metadata)
        FilterGraphAlgorithms._scatter_detector_readout_block!(
            readout_frame,
            mapping,
            readout_output,
            ordinal,
        )
    end

    @test row_frame == complete_frame
    @test readout_frame == complete_frame
end

@testset "readout arrays support one-based CPU views and parameter replacement" begin
    calibration = prepare_algorithm(
        PixelCalibrationReadoutU16F32,
        readout_configuration(PixelCalibrationReadoutU16F32),
    )
    block_fixture = FIXTURE["blocks"][1]
    raw_storage = fill(UInt16(99), 4, 2, 8)
    raw = @view raw_storage[1:2:3, 1:1, 1:2:7]
    raw .= readout_block(UInt16, block_fixture["raw"], 2, 1, 4)
    output_storage = fill(71.0f0, 4, 2, 8)
    output = @view output_storage[2:2:4, 2:2, 2:2:8]
    metadata = SampleMetadata(15, 0)

    @test process!(output, calibration, raw, (raw=metadata,)) ==
          AlgorithmProcessComplete(metadata)
    @test vec(permutedims(output, (3, 2, 1))) ==
          Float32.(block_fixture["calibrated"])

    flat_storage = fill(NaN32, 6, 6)
    flat = @view flat_storage[1:4, 2:5]
    fill!(flat, 2.0f0)
    @test replace_parameter!(calibration, :flat, flat) === calibration
    @test process!(output, calibration, raw, (raw=metadata,)) ==
          AlgorithmProcessComplete(SampleMetadata(metadata; discontinuity=true))
    @test all(==(4.0f0), output)

    background = zeros(Float32, 4, 4)
    @test replace_parameter!(calibration, :background, background) === calibration
    @test process!(output, calibration, raw, (raw=metadata,)) ==
          AlgorithmProcessComplete(SampleMetadata(metadata; discontinuity=true))
    @test all(==(6.0f0), output)

    aliased_output = reshape(@view(calibration.plan.flat[1:8]), 2, 1, 4)
    @test_throws ArgumentError process!(
        aliased_output,
        calibration,
        raw,
        (raw=metadata,),
    )

    assembler = prepare_algorithm(
        DetectorFrameAssemblerF32,
        readout_configuration(DetectorFrameAssemblerF32),
    )
    frame_storage = fill(83.0f0, 6, 6)
    frame = @view frame_storage[2:5, 2:5]
    first = readout_block(Float32, FIXTURE["blocks"][1]["calibrated"], 2, 1, 4)
    last = readout_block(Float32, FIXTURE["blocks"][2]["calibrated"], 2, 1, 4)
    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(22, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(22, 1; terminal=true),),
    ).complete
    @test frame_storage[1, :] == fill(83.0f0, 6)
    @test frame_storage[6, :] == fill(83.0f0, 6)
end

@testset "frame assembly drops malformed sequences and recovers discontinuously" begin
    assembler = prepare_algorithm(
        DetectorFrameAssemblerF32,
        readout_configuration(DetectorFrameAssemblerF32),
    )
    first = readout_block(Float32, FIXTURE["blocks"][1]["calibrated"], 2, 1, 4)
    last = readout_block(Float32, FIXTURE["blocks"][2]["calibrated"], 2, 1, 4)
    frame = fill(99.0f0, 4, 4)

    @test process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(1, 1; terminal=true),),
    ) == AlgorithmProcessDeferred()
    @test frame == fill(99.0f0, 4, 4)
    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(2, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(3, 1; terminal=true),),
    ) == AlgorithmProcessDeferred()
    @test frame == fill(99.0f0, 4, 4)

    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(4, 0),),
    ) == AlgorithmProcessDeferred()
    result = process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(4, 1; terminal=true),),
    )
    @test result == AlgorithmProcessComplete(
        SampleMetadata(4, 0; terminal=true, discontinuity=true),
    )

    fill!(frame, 101.0f0)
    @test process!(frame, assembler, first) === DetectorFrameAssemblerMetadataMissing
    @test frame == fill(101.0f0, 4, 4)
end

@testset "frame assembly covers every recoverable admission fault" begin
    assembler = prepare_algorithm(
        DetectorFrameAssemblerF32,
        readout_configuration(DetectorFrameAssemblerF32),
    )
    first = readout_block(Float32, FIXTURE["blocks"][1]["calibrated"], 2, 1, 4)
    last = readout_block(Float32, FIXTURE["blocks"][2]["calibrated"], 2, 1, 4)
    frame = fill(107.0f0, 4, 4)

    for metadata in (
        SampleMetadata(5, 2),
        SampleMetadata(6, 0; terminal=true),
        SampleMetadata(7, 0; corrupted=true),
    )
        @test process!(
            frame,
            assembler,
            first,
            (readout_block=metadata,),
        ) == AlgorithmProcessDeferred()
        @test frame == fill(107.0f0, 4, 4)
    end

    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(8, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(8, 1; terminal=true, discontinuity=true),),
    ) == AlgorithmProcessDeferred()

    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(9, 0),),
    ) == AlgorithmProcessDeferred()
    @test process!(
        frame,
        assembler,
        first,
        (readout_block=SampleMetadata(10, 0),),
    ) == AlgorithmProcessDeferred()
    result = process!(
        frame,
        assembler,
        last,
        (readout_block=SampleMetadata(10, 1; terminal=true),),
    )
    @test result.metadata.discontinuity
end

@testset "readout calibration rejects block identity before output mutation" begin
    calibration = prepare_algorithm(
        PixelCalibrationReadoutU16F32,
        readout_configuration(PixelCalibrationReadoutU16F32),
    )
    raw = readout_block(UInt16, FIXTURE["blocks"][1]["raw"], 2, 1, 4)
    output = fill(109.0f0, 2, 1, 4)

    @test process!(
        output,
        calibration,
        raw,
        (raw=SampleMetadata(1, 2),),
    ) === PixelCalibrationReadoutOrdinalRange
    @test output == fill(109.0f0, 2, 1, 4)
    @test process!(
        output,
        calibration,
        raw,
        (raw=SampleMetadata(1, 0; terminal=true),),
    ) === PixelCalibrationReadoutTerminalMismatch
    @test output == fill(109.0f0, 2, 1, 4)

    corrupted = SampleMetadata(2, 0; corrupted=true)
    result = process!(output, calibration, raw, (raw=corrupted,))
    @test result == AlgorithmProcessComplete(corrupted)
    reset!(calibration)
    result = process!(output, calibration, raw, (raw=SampleMetadata(3, 0),))
    @test result.metadata.discontinuity
end

@testset "readout warmed paths and Workspace lifecycle allocate no heap" begin
    calibration = prepare_algorithm(
        PixelCalibrationReadoutU16F32,
        readout_configuration(PixelCalibrationReadoutU16F32),
    )
    assembler = prepare_algorithm(
        DetectorFrameAssemblerF32,
        readout_configuration(DetectorFrameAssemblerF32),
    )
    raw = readout_block(UInt16, FIXTURE["blocks"][1]["raw"], 2, 1, 4)
    calibrated = zeros(Float32, 2, 1, 4)
    frame = zeros(Float32, 4, 4)
    first_metadata = (raw=SampleMetadata(31, 0),)
    missing_metadata = (raw=nothing,)

    process!(calibrated, calibration, raw, first_metadata)
    @test calibration_allocation_bytes(
        calibrated,
        calibration,
        raw,
        first_metadata,
    ) == 0
    process!(calibrated, calibration, raw, missing_metadata)
    @test calibration_allocation_bytes(
        calibrated,
        calibration,
        raw,
        missing_metadata,
    ) == 0

    assembler_metadata = (readout_block=SampleMetadata(31, 0),)
    process!(frame, assembler, calibrated, assembler_metadata)
    reset!(assembler)
    @test assembler_allocation_bytes(
        frame,
        assembler,
        calibrated,
        assembler_metadata,
    ) == 0
    missing_assembler_metadata = (readout_block=nothing,)
    process!(frame, assembler, calibrated, missing_assembler_metadata)
    @test assembler_allocation_bytes(
        frame,
        assembler,
        calibrated,
        missing_assembler_metadata,
    ) == 0

    last_metadata = (readout_block=SampleMetadata(31, 1; terminal=true),)
    process!(frame, assembler, calibrated, assembler_metadata)
    process!(frame, assembler, calibrated, last_metadata)
    process!(frame, assembler, calibrated, assembler_metadata)
    @test assembler_allocation_bytes(
        frame,
        assembler,
        calibrated,
        last_metadata,
    ) == 0

    for workspace in (calibration.workspace, assembler.workspace)
        @test workspace_commit_allocation_bytes(workspace) == 0
        @test workspace_discard_allocation_bytes(workspace) == 0
        @test workspace_reset_allocation_bytes(workspace) == 0
    end
end

end
