module ProgressiveDeviceTests

using FilterGraphAlgorithms
using Test
import FilterGraphAlgorithms as FGA
import FilterGraphAlgorithms: process!, reset!

const ROW = (; image_size=(4, 4), row_block_rows=1)
const READOUT = (; detector_size=(4, 4), region_lines=4, block_lines=1, region_samples=2,
    detector_origins=[[0, 0], [3, 3]], detector_line_steps=[[1, 0], [-1, 0]],
    detector_sample_steps=[[0, 1], [0, -1]])
const PYRAMID = (; pupil_size=(2, 2), pupil_origins=[[2, 2], [0, 0], [2, 0], [0, 2]],
    pupil_mask=Bool[true, false, true, true])
const SHACK_HARTMANN = (; subaperture_size=(2, 2), subaperture_count=4,
    initial_subaperture_origins=[[1, 1], [0, 2], [2, 0], [0, 0]],
    coordinate_scale=1f0, pixel_threshold=3f0, flux_threshold=1f0,
    reference_slopes=Float32[0.1, 0.2, 0.3, -0.3, 0.4, 0.5, -0.1, 0.7],
    active=Bool[true, false, true, true])
const PACKET = (; measurements_schema="test.filtered-measurements/1",
    column_indices_schema="test.reconstructor-columns/1",
    measurement_count_schema="test.measurement-count/1")
const RECONSTRUCTED = (; reconstructed_count=2, reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1)

function detector_blocks(image, layout)
    if layout === :row
        return [image[row:row, :] for row in axes(image, 1)]
    end
    readout = FGA._prepare_detector_readout(
        READOUT.detector_size..., READOUT.region_lines, READOUT.block_lines, READOUT.region_samples,
        READOUT.detector_origins, READOUT.detector_line_steps, READOUT.detector_sample_steps,
    )
    indices = FGA._readout_detector_indices(readout)
    return [reshape(image[group], FGA.detector_readout_block_shape(readout)) for group in indices]
end

function outputs_for(prepared::FGA.PreparedAlgorithm{D}) where {D}
    return Tuple(fill(format.element_type(91), format.shape)
        for (port, format) in zip(FGA.algorithm_ports(D), algorithm_formats(prepared))
        if port.direction isa FGA.OutputPort)
end

metadata(key, sample) = NamedTuple{(key,)}((sample,))

function compare_block!(cpu, gpu, host, device, block, to_device, metadata...)
    expected = process!(host..., cpu, block, metadata...)
    actual = process!(device..., gpu, to_device(block), metadata...)
    @test actual == expected
    for (expected_array, actual_array) in zip(host, device)
        @test Array(actual_array) ≈ expected_array rtol=8f-6 atol=1f-6
    end
    return actual
end

function compare_frames!(cpu, gpu, host, device, key, blocks, to_device)
    # Replaying changing samples exercises previous-frame pyramid gain.
    for sequence in 1:2, (index, block) in pairs(blocks)
        sample = SampleMetadata(sequence, index - 1; terminal=index == length(blocks),
            timestamp_nanoseconds=1000sequence)
        compare_block!(cpu, gpu, host, device, block .* eltype(block)(sequence), to_device, metadata(key, sample))
    end
    # Interrupt a prefix and require the same discontinuity on the next frame.
    compare_block!(cpu, gpu, host, device, blocks[1], to_device, metadata(key, SampleMetadata(3, 0)))
    compare_block!(cpu, gpu, host, device, blocks[2], to_device,
        metadata(key, SampleMetadata(3, 1; corrupted=true)))
    for (index, block) in pairs(blocks)
        sample = SampleMetadata(4, index - 1; terminal=index == length(blocks))
        compare_block!(cpu, gpu, host, device, block, to_device, metadata(key, sample))
    end
end

function run_weighted_centroid_layout_tests(selected_device, to_device)
    @testset "Resident weighted centroid uses canonical weights" begin
        declaration = ShackHartmannWeightedMeasurementBlockF32
        config = ShackHartmannWeightedMeasurementBlockF32Configuration(;
            image_size=(16, 256), row_block_rows=8, subaperture_size=(8, 8), subaperture_count=64,
            initial_subaperture_origins=nothing, coordinate_scale=0.75f0,
            pixel_threshold=0.5f0, flux_threshold=1f0, reference_slopes=nothing, active=nothing,
            initial_weights=[[Float32(mod(i + 3j, 13)) / 7f0 for j in 1:64] for i in 1:64],
            PACKET...)
        cpu = prepare_algorithm(declaration, config)
        cache = cpu.plan.sensing.packed_weights
        @test cache isa Matrix{Float32}
        direct = prepare_algorithm(declaration, config, selected_device)
        adapted = prepare_algorithm(cpu, selected_device)
        @test cpu.plan.sensing.packed_weights === cache
        @test isnothing(direct.plan.source.sensing.packed_weights)
        @test isnothing(adapted.plan.source.sensing.packed_weights)

        host = outputs_for(cpu)
        device = map(to_device, host)
        image = [Float32(mod(7i + 11j, 31)) / 8f0 for i in 1:16, j in 1:256]
        for gpu in (direct, adapted)
            reset!(cpu)
            for (first_row, sample) in ((1, SampleMetadata(1)), (9, SampleMetadata(1, 8; terminal=true)))
                block = image[first_row:(first_row + 7), :]
                compare_block!(cpu, gpu, host, device, block, to_device, (; row_block=sample))
            end
        end

        weights = reverse(cpu.plan.sensing.weights; dims=2)
        replace_parameter!(cpu, :weights, weights)
        replace_parameter!(direct, :weights, weights)
        @test isnothing(direct.plan.source.sensing.packed_weights)
        @test cpu.plan.sensing.packed_weights == permutedims(weights)
        @test adapted.plan.source.sensing.weights != weights
        for (first_row, sample) in ((1, SampleMetadata(2)), (9, SampleMetadata(2, 8; terminal=true)))
            block = image[first_row:(first_row + 7), :]
            compare_block!(cpu, direct, host, device, block, to_device, (; row_block=sample))
        end
    end
end

function run_progressive_device_tests(selected_device, to_device)
    run_weighted_centroid_layout_tests(selected_device, to_device)
    pyramid_matrix = Float32[0.0625 * (mod(index, 9) - 4) for index in 1:24]
    shack_matrix = Float32[0.125 * (mod(index, 7) - 3) for index in 1:16]
    weights = [Float32[0, 1, 3, 2], Float32[1, 1, 1, 1],
               Float32[1, 2, 0.5, 0], Float32[0.5, 0, 1, 1]]
    calibration = (; initial_flat=Float32.(1:16), initial_background=Float32.(16:-1:1))
    pyramid_reconstruction = (; RECONSTRUCTED..., initial_reconstructor=pyramid_matrix)
    sh_reconstruction = (; RECONSTRUCTED..., initial_reconstructor=shack_matrix)
    cases = (
        (PixelCalibrationRowU16F32, PixelCalibrationRowU16F32Configuration(; ROW..., calibration...),
            :row, :raw, UInt16, :flat => fill(2f0, 4, 4)),
        (PixelCalibrationReadoutU16F32, PixelCalibrationReadoutU16F32Configuration(; READOUT..., calibration...),
            :readout, :raw, UInt16, :background => fill(2f0, 4, 4)),
        (DetectorFrameAssemblerF32, DetectorFrameAssemblerF32Configuration(; READOUT...),
            :readout, :readout_block, Float32, nothing),
        (PyramidPixelRowF32, PyramidPixelRowF32Configuration(; ROW..., PYRAMID...),
            :row, :row_block, Float32, "pupil-mask" => Bool[false true; true true]),
        (PyramidMeasurementBlockF32, PyramidMeasurementBlockF32Configuration(; ROW..., PYRAMID..., PACKET...),
            :row, :row_block, Float32, "pupil-mask" => Bool[false true; true true]),
        (PyramidReadoutMeasurementBlockF32, PyramidReadoutMeasurementBlockF32Configuration(; READOUT..., PYRAMID..., PACKET...),
            :readout, :readout_block, Float32, nothing),
        (PyramidRowReconstructorF32, PyramidRowReconstructorF32Configuration(; ROW..., PYRAMID..., pyramid_reconstruction...),
            :row, :row_block, Float32, :reconstructor => reshape(reverse(pyramid_matrix), 2, 12)),
        (PyramidReadoutReconstructorF32, PyramidReadoutReconstructorF32Configuration(; READOUT..., PYRAMID..., pyramid_reconstruction...),
            :readout, :readout_block, Float32, nothing),
        (ShackHartmannMeasurementBlockF32, ShackHartmannMeasurementBlockF32Configuration(; ROW..., SHACK_HARTMANN..., PACKET...),
            :row, :row_block, Float32, :active => fill(true, 4)),
        (ShackHartmannWeightedMeasurementBlockF32, ShackHartmannWeightedMeasurementBlockF32Configuration(;
            ROW..., SHACK_HARTMANN..., PACKET..., initial_weights=weights),
            :row, :row_block, Float32, :weights => reshape(Float32.(1:16), 4, 4)),
        (ShackHartmannReadoutMeasurementBlockF32, ShackHartmannReadoutMeasurementBlockF32Configuration(; READOUT..., SHACK_HARTMANN..., PACKET...),
            :readout, :readout_block, Float32, "reference-slopes" => reshape(fill(0.125f0, 8), 4, 2)),
        (ShackHartmannRowReconstructorF32, ShackHartmannRowReconstructorF32Configuration(; ROW..., SHACK_HARTMANN..., sh_reconstruction...),
            :row, :row_block, Float32, :reconstructor => reshape(reverse(shack_matrix), 2, 8)),
        (ShackHartmannReadoutReconstructorF32, ShackHartmannReadoutReconstructorF32Configuration(; READOUT..., SHACK_HARTMANN..., sh_reconstruction...),
            :readout, :readout_block, Float32, nothing),
    )
    @testset "Eager resident progressive $declaration" for (declaration, config, layout, key, T, parameter) in cases
        cpu = prepare_algorithm(declaration, config)
        gpu = prepare_algorithm(declaration, config, selected_device)
        @test capture_support(declaration, selected_device) isa NoCapture
        host = outputs_for(cpu)
        device = map(to_device, host)
        blocks = detector_blocks(reshape(T.(1:16), 4, 4), layout)
        compare_block!(cpu, gpu, host, device, blocks[1], to_device)
        @test_throws ArgumentError process!(device..., gpu, blocks[1], metadata(key, SampleMetadata(1)))
        if declaration ∉ (PixelCalibrationRowU16F32, PixelCalibrationReadoutU16F32)
            detector = gpu.workspace.detector
            prior = reshape(Float32.(1:length(detector)), size(detector))
            copyto!(detector, prior)
            block = blocks[1]
            shifted = reshape(view(vec(detector), 2:(length(block) + 1)), size(block))
            @test typeof(shifted) === typeof(to_device(block))
            @test_throws ArgumentError process!(device..., gpu, shifted, metadata(key, SampleMetadata(1)))
            @test Array(detector) == prior
        end
        compare_frames!(cpu, gpu, host, device, key, blocks, to_device)
        if declaration in (PyramidPixelRowF32, PyramidMeasurementBlockF32, PyramidReadoutMeasurementBlockF32,
                           PyramidRowReconstructorF32, PyramidReadoutReconstructorF32)
            bad_blocks = copy.(blocks)
            for bad in bad_blocks
                fill!(bad, Float32(NaN))
            end
            failed = false
            for (index, block) in pairs(bad_blocks)
                sample = SampleMetadata(5, index - 1; terminal=index == length(bad_blocks))
                result = compare_block!(cpu, gpu, host, device, block, to_device, metadata(key, sample))
                failed |= result isa FGA.algorithm_failure_type(declaration)
            end
            @test failed
        end
        if !isnothing(parameter)
            name, values = parameter
            replace_parameter!(cpu, name, values)
            replace_parameter!(gpu, name, values)
            compare_frames!(cpu, gpu, host, device, key, blocks, to_device)
        end
        if declaration in (PyramidPixelRowF32, PyramidMeasurementBlockF32, PyramidRowReconstructorF32)
            # All pupils now complete together. A new plan can have larger
            # completion groups while preserving the admitted buffer formats.
            origins = zeros(UInt32, 4, 2)
            replace_parameter!(cpu, "pupil-origins", origins)
            replace_parameter!(gpu, "pupil-origins", origins)
            compare_frames!(cpu, gpu, host, device, key, blocks, to_device)
        end
        reset!(cpu)
        reset!(gpu)
        compare_frames!(cpu, gpu, host, device, key, blocks, to_device)
    end

    @testset "Eager resident sparse incremental reconstruction ($storage)" for storage in ("fp32", "fp16")
        config = IncrementalDenseReconstructorF32Configuration(;
            PACKET..., RECONSTRUCTED..., measurement_count=4, measurement_block_capacity=2,
            blocks_per_sample=2, initial_reconstructor=shack_matrix[1:8],
            reconstructor_schema="test.reconstructor/1", coefficient_storage=storage)
        cpu = prepare_algorithm(IncrementalDenseReconstructorF32, config)
        gpu = prepare_algorithm(cpu, selected_device)
        @test capture_support(IncrementalDenseReconstructorF32, selected_device) isa NoCapture
        host = only(outputs_for(cpu))
        device = to_device(host)
        function compare_incremental(values, columns, count, sample)
            count_array = UInt32[count]
            metadata = (; measurements=sample, columns=nothing, count=nothing)
            expected = process!(host, cpu, values, columns, count_array, metadata)
            actual = process!(device, gpu, to_device(values), to_device(columns),
                              to_device(count_array), metadata)
            @test actual == expected
            @test Array(device) ≈ host
            return actual
        end
        for sequence in 1:2
            compare_incremental(Float32[1, -2], UInt32[3, 1], 2, SampleMetadata(sequence))
            compare_incremental(Float32[4, 6], UInt32[0, 2], 2, SampleMetadata(sequence, 1; terminal=true))
        end
        for (values, columns, count, sample, failure) in (
            (Float32[1, 2], UInt32[1, 0], 3, SampleMetadata(3), FGA.IncrementalDenseReconstructionCountOutOfRange),
            (Float32[1, 2], UInt32[4, 0], 2, SampleMetadata(4), FGA.IncrementalDenseReconstructionColumnOutOfRange),
            (Float32[1, 2], UInt32[1, 1], 2, SampleMetadata(5), FGA.IncrementalDenseReconstructionDuplicateColumn),
            (Float32[1, NaN], UInt32[0, 1], 2, SampleMetadata(6), FGA.IncrementalDenseReconstructionNonFiniteInput),
            (Float32[1, 2], UInt32[0, 1], 2, SampleMetadata(7; terminal=true), FGA.IncrementalDenseReconstructionIncompleteSample),
        )
            @test compare_incremental(values, columns, count, sample) === failure
        end
        coefficients = reshape(fill(0.75f0, 8), 2, 4)
        replace_parameter!(cpu, :reconstructor, coefficients)
        replace_parameter!(gpu, :reconstructor, coefficients)
        compare_incremental(Float32[2, 4], UInt32[0, 1], 2, SampleMetadata(8))
        compare_incremental(Float32[3, 5], UInt32[3, 2], 2, SampleMetadata(8, 1; terminal=true))
        reset!(cpu)
        reset!(gpu)
        compare_incremental(Float32[1, -2], UInt32[3, 1], 2, SampleMetadata(9))
        compare_incremental(Float32[4, 6], UInt32[0, 2], 2, SampleMetadata(9, 1; terminal=true))

        large = fill(ldexp(1f0, 126), 2, 4)
        replace_parameter!(cpu, :reconstructor, large)
        replace_parameter!(gpu, :reconstructor, large)
        @test compare_incremental(Float32[4, 0], UInt32[0, 1], 1, SampleMetadata(10)) ===
            FGA.IncrementalDenseReconstructionNonFiniteOutput
    end

    @testset "Resident pyramid measurements feed incremental reconstruction" begin
        sensing = prepare_algorithm(PyramidMeasurementBlockF32,
            PyramidMeasurementBlockF32Configuration(; ROW..., PYRAMID..., PACKET...), selected_device)
        reconstruction = prepare_algorithm(IncrementalDenseReconstructorF32,
            IncrementalDenseReconstructorF32Configuration(;
                PACKET..., RECONSTRUCTED..., measurement_count=12, measurement_block_capacity=12,
                blocks_per_sample=4, initial_reconstructor=pyramid_matrix,
                reconstructor_schema="test.reconstructor/1"), selected_device)
        cpu = prepare_algorithm(PyramidRowReconstructorF32,
            PyramidRowReconstructorF32Configuration(; ROW..., PYRAMID..., pyramid_reconstruction...))
        packet = map(to_device, outputs_for(sensing))
        actual = to_device(only(outputs_for(reconstruction)))
        expected = outputs_for(cpu)
        blocks = detector_blocks(reshape(Float32.(1:16), 4, 4), :row)
        for sequence in 1:2, (index, block) in pairs(blocks)
            sample = SampleMetadata(sequence, index - 1; terminal=index == length(blocks))
            measured = process!(packet..., sensing, to_device(block), (; row_block=sample))
            @test measured.complete
            result = process!(actual, reconstruction, packet...,
                (; measurements=measured.metadata, columns=nothing, count=nothing))
            @test result == process!(expected..., cpu, block, (; row_block=sample))
        end
        @test Array(actual) ≈ last(expected)
    end
end

end # module
