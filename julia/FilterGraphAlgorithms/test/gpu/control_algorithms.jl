using FilterGraphAlgorithms, Test, TOML
import FilterGraphAlgorithms: PreparedAlgorithm, process!, reset!, prepare_capture, enqueue!, complete!, replace_parameter!

function run_control_algorithm_tests(device, to_device, device_zeros, synchronize;
        record=nothing, replay=nothing)
    function capture(cpu::PreparedAlgorithm{D}) where {D}
        formats = zip(algorithm_ports(D), algorithm_formats(cpu))
        input_format = [format for (port, format) in formats if
            port.direction isa FilterGraphAlgorithms.InputPort && port.role isa FilterGraphAlgorithms.FrameData]
        output_format = [format for (port, format) in formats if
            port.direction isa FilterGraphAlgorithms.OutputPort]
        inputs = Tuple(device_zeros(f.element_type, f.shape...) for f in input_format)
        expected = Tuple(fill(convert(f.element_type, one(Int)), f.shape) for f in output_format)
        outputs = map(to_device, expected)
        status = device_zeros(Int32, 1)
        adapter = prepare_capture(CompleteTransition(), device, cpu, outputs, inputs, status, Int32(2))
        step!() = (fill!(status, Int32(0)); enqueue!(adapter); nothing)
        fill!(status, Int32(1))
        enqueue!(adapter)
        synchronize()
        @test map(Array, outputs) == expected
        @test Array(status) == Int32[1]
        step!()
        synchronize()
        executable = record === nothing ? nothing : record(step!)
        reset!(adapter)
        foreach(copyto!, outputs, expected)
        synchronize()
        execute! = executable === nothing ? step! : () -> replay(executable)
        return (; cpu, inputs, outputs, expected, status, adapter, execute!)
    end
    function check(case, values...)
        saved = map(Array, case.outputs)
        foreach(copyto!, case.inputs, values)
        expected_result = process!(case.expected..., case.cpu, values...)
        case.execute!()
        synchronize()
        @test complete!(case.adapter, only(Array(case.status))) === expected_result
        for (actual, expected) in zip(case.outputs, isnothing(expected_result) ? case.expected : saved)
            values = Array(actual)
            if eltype(values) === Float32 && isnothing(expected_result)
                @test values ≈ expected rtol=4f-6 atol=3f-8
            else
                @test isequal(values, expected)
            end
        end
        return expected_result
    end

    @testset verbose=true "Resident control Algorithms on $device" begin
        @testset "Optical gain device transitions" begin
            correction = capture(prepare_algorithm(OpticalGainCorrectionF32,
                OpticalGainCorrectionF32Configuration(optical_gains=Float32[floatmin(Float32), 0.5])))
            @test check(correction, Float32[0.25, 2]) === nothing
            @test check(correction, Float32[10, NaN32]) === OpticalGainCorrectionNonFiniteOutput
            @test check(correction, Float32[0, Inf32]) === OpticalGainCorrectionNonFiniteInput

            modes = [OpticalGainDitherMode(0.5f0, 4, 2), OpticalGainDitherMode(0.25f0, 8, 1)]
            dither = capture(prepare_algorithm(OpticalGainDitherF32,
                OpticalGainDitherF32Configuration(; modes)))
            estimator = capture(prepare_algorithm(OpticalGainEstimatorF32,
                OpticalGainEstimatorF32Configuration(; modes, initial_calibration_responses=Float32[1, 0.5])))
            criteria = fill(OpticalGainUpdateCriteria(0.1f0, 0.05f0, 0.1f0, 1f0, 0.5f0), 2)
            assignments = [OpticalGainAssignment(0, 0.5f0), OpticalGainAssignment{Float32}(),
                OpticalGainAssignment(1, 1f0)]
            update = capture(prepare_algorithm(OpticalGainUpdateF32, OpticalGainUpdateF32Configuration(;
                criteria, assignments, initial_optical_gains=Float32[1, 0.4, 1])))
            for tick in 1:17
                @test check(dither) === nothing
                command, frame, phases, complete = dither.expected
                response = command .* Float32[0.5, 0.25]
                if tick == 3
                    @test check(estimator, command, response, UInt64[typemax(UInt64)], phases, complete) ===
                        OpticalGainEstimatorWindowFrameOutOfRange
                    @test check(estimator, command, response, frame .+ UInt64(1), phases, complete) ===
                        OpticalGainEstimatorWindowFrameMismatch
                    @test check(estimator, command, response, frame, phases, Bool[!only(complete)]) ===
                        OpticalGainEstimatorWindowCompletionMismatch
                    @test check(estimator, command, response, frame, phases .+ UInt64(1), complete) ===
                        OpticalGainEstimatorPhaseIndexMismatch
                    @test check(estimator, command, Float32[Inf32, 1], frame, phases, complete) ===
                        OpticalGainEstimatorNonFiniteResponse
                end
                @test check(estimator, command, response, frame, phases, complete) === nothing
                @test check(update, estimator.expected...) === nothing
            end
            @test update.expected[1] ≈ Float32[0.625, 0.4, 0.5]
            @test check(update, Float32[0.5, NaN], Float32[0.25, 0.25], Float32[0.5, 0.5],
                zeros(Float32, 2), Bool[true, true], Bool[true]) === OpticalGainUpdateNonFiniteInput
            for case in (dither, estimator, update)
                reset!(case.cpu)
                reset!(case.adapter)
            end
            @test check(dither) === nothing
            @test dither.expected[2] == UInt64[0]
            @test check(update, fill(NaN32, 2), fill(NaN32, 2), fill(NaN32, 2),
                fill(NaN32, 2), Bool[true,true], Bool[false]) === nothing
            @test update.expected[1] == Float32[1, 0.4, 1]
        end

        @testset "Corrector allocation and pseudo-open-loop reconstruction" begin
            allocation = capture(prepare_algorithm(CorrectorAllocationF32,
                CorrectorAllocationF32Configuration(
                    controller_command_count=1, stage_command_counts=[1, 1], update_periods=[2, 1],
                    command_projection=Float32[1, 4], realization_projection=Float32[1, 0.5],
                    section_count=1, initial_coefficients=Float32[0.25, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    command_limits=(Float32[-2, -4], Float32[2, 4]), slew_limits=Float32[0.125, 1],
                    command_schema="test.allocated-command/1")))
            for x in (0.2f0, 0.5f0, 1f0, floatmax(Float32), -0.3f0, -2f0)
                result = check(allocation, Float32[x])
                @test result === (x == floatmax(Float32) ? CorrectorAllocationNonFiniteOutput : nothing)
            end
            reset!(allocation.cpu)
            reset!(allocation.adapter)
            @test check(allocation, Float32[0.2]) === nothing

            fixture = TOML.parsefile(joinpath(@__DIR__, "..", "data", "algorithms",
                "pseudo_open_loop_reconstruction.toml"))["cases"][1]
            pseudo = capture(prepare_algorithm(PseudoOpenLoopReconstructionF32,
                PseudoOpenLoopReconstructionF32Configuration(
                    measurement_count=fixture["measurement_count"],
                    corrector_shape_count=fixture["corrector_shape_count"],
                    residual_error_count=fixture["residual_error_count"],
                    initial_interaction_matrix=Float32.(fixture["interaction_matrix"]),
                    initial_reconstructor=Float32.(fixture["reconstructor"]),
                    initial_corrector_to_error=Float32.(fixture["corrector_to_error"]),
                    residual_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
                    corrector_shape_schema=REQUESTED_PDM_COMMAND_V1,
                    pseudo_open_loop_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
                    residual_error_schema=CONTROLLER_RESIDUAL_ERROR_V1)))
            residual = fill(0.125f0, fixture["measurement_count"])
            shape = fill(0.25f0, fixture["corrector_shape_count"])
            @test check(pseudo, residual, shape) === nothing
            @test check(pseudo, fill(Inf32, length(residual)), fill(Inf32, length(shape))) ===
                PseudoOpenLoopNonFiniteResidualMeasurements
            @test check(pseudo, residual, fill(Inf32, length(shape))) === PseudoOpenLoopNonFiniteCorrectorShape
        end

        @testset "PDM limits, stroke optimization and physical offset reference" begin
            smallest = nextfloat(0f0)
            largest = floatmax(Float32)
            stroke = capture(prepare_algorithm(InterActuatorStrokeLimitF32,
                InterActuatorStrokeLimitF32Configuration(actuator_count=3, edges=[[0, 1], [1, 2]],
                    max_differences=Float32[3smallest, smallest])))
            @test check(stroke, Float32[largest, -largest, -largest]) === nothing
            stroke_result = Array(only(stroke.outputs))
            @test any(!iszero, stroke_result)
            @test abs(Float64(stroke_result[1]) - Float64(stroke_result[2])) <= Float64(3smallest)
            @test abs(Float64(stroke_result[2]) - Float64(stroke_result[3])) <= Float64(smallest)
            @test check(stroke, Float32[Inf32, 0, 0]) === InterActuatorStrokeLimitNonFiniteInput

            power = capture(prepare_algorithm(PdmCommandPowerLimitF32,
                PdmCommandPowerLimitF32Configuration(actuator_count=3, max_command_power=1f0)))
            @test check(power, Float32[1, smallest, -0.0]) === nothing
            @test check(power, Float32[largest, -largest, largest]) === nothing
            @test sum(x -> abs2(Rational{BigInt}(x)), Array(only(power.outputs))) <= 1
            @test check(power, Float32[NaN32, 0, 0]) === PdmCommandPowerLimitNonFiniteInput

            optimization = capture(prepare_algorithm(PdmStrokeOptimizationF32,
                PdmStrokeOptimizationF32Configuration(actuator_count=2, null_mode_count=1,
                    preferred_command=Float32[1, 0.5], command_to_null_modes=Float32[1, 0],
                    null_modes_to_command=Float32[1, 1])))
            @test check(optimization, Float32[3, 2]) === nothing
            @test check(optimization, Float32[3, NaN32]) === PdmStrokeOptimizationNonFiniteInput

            offset = prepare_algorithm(PhysicalOffsetReferenceF32, PhysicalOffsetReferenceF32Configuration(
                residual_count=2, physical_count=3, mode_count=1, reference_sign=-1f0,
                residual_schema="test.residual/1", physical_schema="test.offset/1",
                physical_to_modes_schema="test.offset-to-modes/1", modes_to_reference_schema="test.modes-to-residual/1"))
            replace_parameter!(offset, Symbol("physical-to-modes"), reshape(Float32[1, 2, 3], 1, 3))
            replace_parameter!(offset, Symbol("modes-to-reference"), reshape(Float32[1, -2], 2, 1))
            replace_parameter!(offset, :offset, Float32[1, 3, -2])
            captured = capture(offset)
            @test check(captured, Float32[3, 2]) === nothing
            @test check(captured, Float32[3, NaN32]) === VectorLinearCombinationNonFiniteInput
            replace_parameter!(offset, :offset, Float32[-1, 1, 0])
            @test check(capture(offset), Float32[3, 2]) === nothing
        end

        @testset "SCC device exposure admission and correction" begin
            cal = SccCalibration(; frame_shape=(2, 2), reference_difference=zeros(Float32, 2, 2),
                correction_mask=Bool[true false; false true], reconstructor=Float32[1 0; 0 2],
                coordinate_order=(1, 2), command_unit="normalized",
                fringed_response=Float32[0.5 0; 0 0.3], unfringed_response=Float32[0.2 0; 0 0.6],
                reconstructor_sign=:correction, exposure_duration_nanoseconds=1000,
                image_schema="test.scc-frame/1", bias_command=Float32[0.01, -0.01],
                lower_limits=fill(-0.5f0, 2), upper_limits=fill(0.5f0, 2))
            cpu = prepare_algorithm(SccCorrectionF32, SccCorrectionF32Configuration(cal; gain=0.5f0, pole=0.8f0))
            gpu = prepare_algorithm(cpu, device)
            frame = device_zeros(Float32, 2, 2)
            output = to_device(Float32[4, 4])
            reference = Float32[4, 4]
            domain = ntuple(UInt8, 16)
            function step(sequence, kind; pixels=fill(0.1f0, 2, 2), duration=1000, discontinuity=false)
                sample = SampleMetadata(sequence; discontinuity, chopped_image_kind=kind,
                    exposure_duration_nanoseconds=duration,
                    acquisition_identity=SampleAcquisitionIdentity(domain, 1, sequence))
                copyto!(frame, pixels)
                result = process!(reference, cpu, pixels, (frame=sample,))
                @test process!(output, gpu, frame, (frame=sample,)) === result
                @test Array(output) ≈ reference rtol=4f-6 atol=3f-8
                return result
            end
            @test process!(output, gpu, frame) === SccMetadataMissing
            @test Array(output) == reference
            @test step(1, ChoppedImageFringed) === nothing
            @test step(2, ChoppedImageUnfringed; duration=900) === SccExposureMismatch
            @test step(2, ChoppedImageUnfringed; pixels=fill(NaN32, 2, 2)) === SccNonFiniteInput
            @test step(2, ChoppedImageUnfringed; pixels=fill(-5f0, 2, 2)) === SccCommandOutOfBounds
            @test step(2, ChoppedImageUnfringed; pixels=fill(0f0, 2, 2)) === nothing
            for prepared in (cpu, gpu)
                prepared.plan = FilterGraphAlgorithms.prepare_properties(prepared.plan,
                    (PropertyAssignment(0, 0.2f0), PropertyAssignment(1, 0.5f0)))
            end
            @test step(3, ChoppedImageFringed; pixels=fill(0.15f0, 2, 2)) === nothing
            @test step(3, ChoppedImageUnfringed) === SccSequenceInvalid
            @test step(4, ChoppedImageMixed) === nothing
            @test step(5, ChoppedImageUnfringed) === nothing
            @test step(7, ChoppedImageFringed) === nothing
            @test step(8, ChoppedImageUnfringed; discontinuity=true) === nothing
            reset!(cpu)
            reset!(gpu)
            @test step(1, ChoppedImageFringed) === nothing
            @test Array(output) == cal.bias_command
        end
    end
end
