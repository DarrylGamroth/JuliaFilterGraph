include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "controller_to_vdm.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "vdm_to_pdm.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "pdm_feedback_to_vdm.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "vdm_feedback_to_controller.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "pdm_command.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "deformable_mirror", "pdm_offset.jl"))

@testset "portable KA PDM state and offset reductions" begin
    @testset "command state and feedback match CPU with independent slew" begin
        extent = 277
        prepared = prepare_algorithm(PdmCommandF32, PdmCommandF32Configuration(
            actuator_count=extent, command_limits=(fill(-1.0f0, extent), fill(1.0f0, extent)), slew_limit=fill(0.125f0, extent),
            initial_current=zeros(Float32, extent)))
        requested = zeros(Float32, extent)
        actual = (fill(71.0f0, extent), fill(73.0f0, extent))
        expected = map(copy, actual)
        status = zeros(Int32, 1)
        adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
            actual, (requested,), status, Int32(1))
        @test capture_support(PdmCommandF32, BACKEND) isa NoCapture
        @test FilterGraphAlgorithms.prepare_validation(adapter, (true,)) === adapter
        @test !Base.mightalias(adapter.committed_current, prepared.workspace.committed_current)
        for frame in 1:3
            requested .= Float32[mod(index + frame, 19) / 16 - 0.5 for index in 1:extent]
            frame == 3 && (requested[end] = Float32(NaN))
            result = process!(expected..., prepared, requested)
            fill!(status, 0)
            enqueue!(adapter)
            KernelAbstractions.synchronize(BACKEND)
            @test complete!(adapter, only(status)) === result
            @test actual == expected
            @test adapter.committed_current == prepared.workspace.committed_current
        end
        reset!(adapter)
        reset!(prepared)
        KernelAbstractions.synchronize(BACKEND)
        @test adapter.committed_current == prepared.workspace.committed_current
        @test adapter.candidate_current == prepared.workspace.candidate_current
    end

    for extent in (277, 513)
        @testset "offset tile $extent" begin
            kernel! = (extent <= PDM_OFFSET_TILE ? _pdm_offset_small! : _pdm_offset!)(BACKEND, PDM_OFFSET_LANES)
            for engineering in (false, true), sample in (:finite, :overflow_first, :input_first)
                requested = Float32[mod(index, 7) / 16 for index in 1:extent]
                offset = fill(0.125f0, extent)
                if sample === :overflow_first
                    requested[1] = offset[1] = floatmax(Float32)
                    requested[end] = Float32(NaN)
                elseif sample === :input_first
                    requested[1] = Float32(NaN)
                    requested[end] = offset[end] = floatmax(Float32)
                end
                actual, expected = fill(71.0f0, extent), fill(71.0f0, extent)
                current = engineering ? fill(73.0f0, extent) : nothing
                expected_current = engineering ? copy(current) : nothing
                result = if engineering
                    FilterGraphAlgorithms._process_pdm_engineering_offset!(expected, expected_current,
                        PdmEngineeringOffsetPlan(offset), requested)
                else
                    FilterGraphAlgorithms._process_pdm_system_flat!(expected, PdmSystemFlatPlan(offset), requested)
                end
                status = zeros(Int32, 1)
                kernel!(actual, current, requested, offset, status, Int32(1), Int32(2); ndrange=PDM_OFFSET_LANES)
                KernelAbstractions.synchronize(BACKEND)
                @test only(status) == (result === nothing ? 0 : Int32(result))
                @test actual == expected
                @test current == expected_current
            end
        end
    end
end

@testset "unqualified KA CPU retains deformable-mirror validation and storage" begin
    identity = Matrix{Float32}(I, 2, 2)
    controller = prepare_algorithm(ControllerToVdmF32,
        ControllerToVdmF32Configuration(controller_command_count=2, active_vdm_count=2))
    replace_parameter!(controller, "controller-to-vdm", identity)
    physical = prepare_algorithm(VdmToPdmF32,
        VdmToPdmF32Configuration(active_vdm_count=2, full_vdm_count=2,
            physical_actuator_count=2))
    replace_parameter!(physical, "active-to-full", identity)
    replace_parameter!(physical, "vdm-to-pdm", identity)
    feedback = prepare_algorithm(PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(active_vdm_count=2, full_vdm_count=2,
            physical_actuator_count=2))
    replace_parameter!(feedback, "pdm-to-vdm", identity)
    replace_parameter!(feedback, "full-to-active", identity)
    returned = prepare_algorithm(VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(controller_command_count=2, active_vdm_count=2))
    replace_parameter!(returned, "vdm-to-controller", identity)

    @test !FilterGraphAlgorithms._supports_validation_reuse(BACKEND)
    @test !FilterGraphAlgorithms._supports_validation_reuse(Val(:unqualified))
    for (prepared, input_failure) in (
        (controller, ControllerToVdmNonFiniteInput),
        (physical, VdmToPdmNonFiniteInput),
        (feedback, PdmFeedbackToVdmNonFiniteInput),
        (returned, VdmFeedbackToControllerNonFiniteInput),
    )
        input = Float32[1, 2]
        output = fill(71.0f0, 2)
        outputs = prepared === physical ? (output, fill(UInt8(73), 2)) : (output,)
        status = zeros(Int32, 1)
        adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
            outputs, (input,), status, Int32(1))
        @test !FilterGraphAlgorithms.preserves_inputs(adapter)
        @test !FilterGraphAlgorithms.finite_output(adapter, Val(1))
        @test FilterGraphAlgorithms.alias_input(adapter, Val(1)) === nothing
        @test FilterGraphAlgorithms.alias_input(adapter, Val(2)) === nothing
        refined = FilterGraphAlgorithms.prepare_validation(adapter, (true,))
        @test refined === adapter
        @test !(refined.validate_input! isa _FiniteInput)
        @test !(refined.validate_output! isa _FiniteInput)
        if adapter isa Union{ControllerToVdm,VdmFeedbackToController}
            @test _identity_validation(adapter) === adapter.validate_output!
        end
        @test !Base.mightalias(output, input)
        if adapter isa VdmToPdm
            @test !Base.mightalias(adapter.full_vdm, input)
            @test !Base.mightalias(adapter.full_vdm, output)
        elseif adapter isa PdmFeedbackToVdm
            @test !Base.mightalias(adapter.full_vdm_feedback, input)
            @test !Base.mightalias(adapter.full_vdm_feedback, output)
        end
        enqueue!(refined)
        KernelAbstractions.synchronize(BACKEND)
        @test complete!(refined, only(status)) === nothing
        @test output == input

        # Even a supplied true proof must not remove this backend's scan.
        input[end] = Float32(NaN)
        fill!(output, 71)
        length(outputs) == 2 && fill!(outputs[2], UInt8(73))
        fill!(status, 0)
        enqueue!(refined)
        KernelAbstractions.synchronize(BACKEND)
        @test complete!(refined, only(status)) === input_failure
        @test output == fill(71.0f0, 2)
        length(outputs) == 2 && @test outputs[2] == fill(UInt8(73), 2)
        reset!(refined)
        KernelAbstractions.synchronize(BACKEND)
        @test isequal(input, Float32[1, NaN])
    end
end
