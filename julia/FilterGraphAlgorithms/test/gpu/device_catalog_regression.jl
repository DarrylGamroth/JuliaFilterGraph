module DeviceCatalogRegression

using FilterGraphAlgorithms, Test
import FilterGraphs
import FilterGraphAlgorithms: FrameData, OutputPort, complete!, enqueue!, prepare_capture,
    prepare_properties, process!, reset!

const SCHEMA = "example/device-catalog/1"

function check_catalog_adapter(device, upload, synchronize, record, declaration, prepared, samples)
    formats = algorithm_formats(prepared)
    ports = algorithm_ports(declaration)
    expected = Tuple(zeros(format.element_type, format.shape)
        for (port, format) in zip(ports, formats)
        if port.direction isa OutputPort && port.role isa FrameData)
    outputs = map(upload, expected)
    inputs = map(upload, first(samples))
    status = upload(zeros(Int32, 1))
    adapter = prepare_capture(
        CompleteTransition(), device, prepared, outputs, inputs, status, Int32(1),
    )
    enqueue!(adapter)
    synchronize()
    reset!(adapter)
    foreach(copyto!, outputs, expected)
    replay! = record(() -> enqueue!(adapter))
    for sample in samples
        foreach(copyto!, inputs, sample)
        fill!(status, Int32(0))
        failure = process!(expected..., prepared, sample...)
        replay!()
        synchronize()
        code = only(Array(status))
        @test complete!(adapter, code) === failure
        if failure === nothing
            @test iszero(code)
            for (actual, target) in zip(map(Array, outputs), expected)
                if eltype(target) === Float32
                    @test isapprox(actual, target; nans=true)
                else
                    @test actual == target
                end
            end
        else
            # Vendor GEMV can write disposable candidates on rejection. The
            # captured Graph publishes only after a successful status.
            @test !iszero(code)
        end
    end
end

const CHOPPED_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = chop
            label = chopped-image-difference-f32
            config = { pixel_count = 2 trigger_high_is_fringed = true }
        }
    ]
    links = []
    inputs = [ "chop:image" "chop:input-validity" ]
    outputs = [ "chop:difference" "chop:output-validity" "chop:pair-status" ]
}
"""

function check_primed_chopped_adapter(device, upload, synchronize)
    cpu = prepare_algorithm(ChoppedImageDifferenceF32, ChoppedImageDifferenceF32Configuration(
        pixel_count=2, trigger_high_is_fringed=true))
    expected = (zeros(Float32, 2), fill(false, 2), UInt8[0])
    first = (Float32[11, 9], Bool[true, false])
    @test isnothing(process!(expected..., cpu, first...,
        (image=SampleMetadata(1; trigger_high=true), input_validity=nothing)))

    second = (Float32[7, 3], Bool[true, true])
    inputs, outputs = map(upload, second), map(upload, expected)
    status = upload(Int32[0])
    malformed = deepcopy(cpu)
    empty!(malformed.workspace.committed_validity)
    @test_throws DimensionMismatch prepare_capture(
        CompleteTransition(), device, malformed, outputs, inputs, status, Int32(1))

    adapter = prepare_capture(CompleteTransition(), device, cpu, outputs, inputs, status, Int32(1))
    sample = SampleMetadata(2; trigger_high=false)
    # Replay the metadata preparation used by the captured graph owner.
    extension = parentmodule(typeof(adapter))
    copyto!(adapter.flags, collect(extension._admit_chopped_capture!(adapter, sample)))
    enqueue!(adapter)
    synchronize()
    @test isnothing(complete!(adapter, only(Array(status))))
    @test isnothing(process!(expected..., cpu, second..., (image=sample, input_validity=nothing)))
    @test map(Array, outputs) == expected
    @test expected[1] == Float32[4, 0]
end

function check_catalog_chopped(device, upload)
    checked = mktemp() do path, io
        write(io, CHOPPED_GRAPH)
        flush(io)
        FilterGraphs.prepare_graph(path; algorithms=(ChoppedImageDifferenceF32,))
    end
    captured = FilterGraphs.prepare_captured_graph(checked, device)
    outputs = (difference=zeros(Float32, 2), var"output-validity"=fill(false, 2),
        var"pair-status"=zeros(UInt8, 1))
    expected = map(copy, outputs)
    inputs = (image=upload(zeros(Float32, 2)), var"input-validity"=upload(fill(false, 2)))
    frames = (
        (1, true, Float32[10, 20], Bool[true, true]),
        (2, false, Float32[3, 8], Bool[true, false]),
        (3, true, Float32[12, 18], Bool[true, true]),
        (4, true, Float32[4, 7], Bool[true, true]),
        (6, false, Float32[2, 3], Bool[true, true]),
        (7, true, Float32[7, 11], Bool[true, true]),
    )
    for (sequence, trigger_high, image, validity) in frames
        cpu_inputs = (image=image, var"input-validity"=validity)
        sample = SampleMetadata(sequence; trigger_high)
        metadata = (image=sample, var"input-validity"=nothing)
        foreach(copyto!, inputs, cpu_inputs)
        @test FilterGraphs.process!(outputs, captured, inputs, metadata) ==
              FilterGraphs.process!(expected, checked, cpu_inputs, metadata)
        @test outputs == expected
        @test FilterGraphs.output_metadata(captured).difference === sample
    end
    @test outputs.difference == Float32[5, 8]
    @test outputs.var"pair-status" == UInt8[UInt8(ChoppedImageComplete)]
end

function run_catalog_regression_tests(device, upload, synchronize, record)
    runner = (declaration, prepared, samples) ->
        check_catalog_adapter(device, upload, synchronize, record, declaration, prepared, samples)
    @testset "Resident Algorithm catalog regression" begin
        @testset "Pixel calibration, selection, validity and centroid" begin
            u16 = PixelCalibrationU16F32Configuration(image_size=(2, 3),
                initial_flat=Float32[1, 2, 0.5, 1, 1.5, 2],
                initial_background=Float32[0, 1, 2, 3, 4, 5])
            selected = PixelSelectionF32Configuration(image_size=(2, 3),
                mask=Bool[true, false, true, false, false, true], input_schema=SCHEMA, output_schema=SCHEMA)
            validity = PixelValidityF32Configuration(extent=4, input_schema=SCHEMA)
            cog = CenterOfGravityF32Configuration(pixel_count=4,
                initial_coordinates=[Float32[-1, -1], Float32[1, -1], Float32[-1, 1], Float32[1, 1]])
            data = (
                (PixelCalibrationU16F32, u16, ((UInt16[2 3 4; 5 6 7],), (UInt16[0 5 9; 8 2 4],))),
                (PixelSelectionF32, selected, ((Float32[2 3 4; 5 6 7],), (Float32[0 5 9; 8 2 4],))),
                (PixelValidityF32, validity, ((Float32[1, NaN, -Inf, 0],), (Float32[NaN, 1, 2, Inf],))),
                (CenterOfGravityF32, cog, ((Float32[1, 2, 3, 4],), (Float32[2, 3, 0, 1],),
                    (zeros(Float32, 4),), (Float32[NaN, 1, 2, 3],))),
            )
            for (declaration, configuration, samples) in data
                @testset "$(nameof(declaration))" begin
                    runner(declaration, prepare_algorithm(declaration, configuration), samples)
                end
            end
        end
        @testset "Dense reconstruction and reverse VDM projection" begin
            coefficients = Float32[1, 0.5, -2, 3, -1, 2, 0.25, 0]
            input_shapes = ((Float32[3 1; 5 2],), (Float32[NaN 1; 0 2],), (Float32[0 4; -2 1],))
            pyramid = PyramidReconstructorF32Configuration(reconstructed_count=2, selected_coordinate_count=2,
                initial_reconstructor=coefficients, reconstructed_schema=SCHEMA)
            shack = ShackHartmannReconstructorF32Configuration(reconstructed_count=2, subaperture_count=2,
                initial_reconstructor=coefficients, reconstructed_schema=SCHEMA)
            scc = SccReconstructorF32Configuration(controller_coordinate_count=2, selected_pixel_count=4,
                initial_reconstructor=coefficients)
            for (declaration, configuration, samples) in (
                (PyramidReconstructorF32, pyramid, input_shapes),
                (ShackHartmannReconstructorF32, shack, input_shapes),
                (SccReconstructorF32, scc, map(sample -> (vec(only(sample)),), input_shapes)),
            )
                @testset "$(nameof(declaration))" begin
                    runner(declaration, prepare_algorithm(declaration, configuration), samples)
                end
            end
            vdm = prepare_algorithm(VdmFeedbackToControllerF32,
                VdmFeedbackToControllerF32Configuration(controller_command_count=2, active_vdm_count=3))
            replace_parameter!(vdm, Symbol("vdm-to-controller"), Float32[1 2 0; -1 0.5 3])
            runner(VdmFeedbackToControllerF32, vdm,
                ((Float32[1, 2, 3],), (Float32[NaN, 1, 2],), (Float32[-2, 1, 4],)))
        end
        @testset "Chopped camera metadata" begin
            check_catalog_chopped(device, upload)
            check_primed_chopped_adapter(device, upload, synchronize)
        end
        @testset "Validity-gated store and hold" begin
            gated = prepare_algorithm(ValidityGatedClosedLoopIntegratorF32,
                ValidityGatedClosedLoopIntegratorF32Configuration(
                    state_length=2, validity_length=3, validity_schema=CALIBRATED_PIXEL_VALIDITY_V1))
            gated.plan = prepare_properties(gated.plan,
                (PropertyAssignment(0, 0.5f0), PropertyAssignment(1, 0.75f0), PropertyAssignment(2, 0.25f0)))
            runner(ValidityGatedClosedLoopIntegratorF32, gated, (
                (Float32[1, 2], Float32[0, 0], Bool[true, true, true]),
                (Float32[NaN, 20], Float32[Inf, 30], Bool[true, false, true]),
                (Float32[2, 3], Float32[1, 2], Bool[true, true, true]),
                (Float32[2, 3], Float32[NaN, 1], Bool[true, true, true]),
                (Float32[-1, 0], Float32[0, 4], Bool[true, true, true]),
            ))
        end
    end
end

end
