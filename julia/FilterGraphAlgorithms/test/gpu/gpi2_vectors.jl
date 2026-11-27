# Shared qualification for the CUDA and AMDGPU vector adapters. A backend may
# supply record(c) -> launch! to exercise its native graph replay as well.
import FilterGraphAlgorithms: process!, reset!, prepare_capture, enqueue!, complete!

function run_gpi2_vector_capture_tests(device, to_device, device_zeros, synchronize;
        record=c -> (() -> enqueue!(c)))
    bits(arrays) = map(a -> reinterpret(UInt32, vec(Array(a))), arrays)
    function candidate_storage(owner)
        if hasproperty(owner, :candidate_output)
            return (owner.candidate_output,)
        end
        return ()
    end
    function fixture(prepared, output_extents, input_extents)
        outputs = map(n -> device_zeros(Float32, n), output_extents)
        inputs = map(n -> device_zeros(Float32, n), input_extents)
        status = device_zeros(Int32, 1)
        captured = prepare_capture(CompleteTransition(), device, prepared,
            outputs, inputs, status, Int32(1))
        launch! = record(captured)
        reset!(captured)
        synchronize()
        return (; prepared, outputs, inputs, status, captured, launch!)
    end
    function step!(case, samples; preserve_outputs=true)
        expected = map(a -> fill(71.0f0, length(a)), case.outputs)
        result = process!(expected..., case.prepared, samples...)
        foreach(a -> fill!(a, 71.0f0), case.outputs)
        foreach(pair -> copyto!(pair...), zip(case.inputs, samples))
        fill!(case.status, Int32(0))
        case.launch!()
        synchronize()
        @test complete!(case.captured, only(Array(case.status))) === result
        if result === nothing
            @test bits(case.outputs) == bits(expected)
        elseif preserve_outputs
            @test all(a -> all(==(71.0f0), Array(a)), case.outputs)
        end
        return result
    end
    function gated!(case, retained)
        fill!(case.status, Int32(42))
        storage = map(Array, retained)
        case.launch!()
        synchronize()
        @test only(Array(case.status)) == 42
        @test isequal(map(Array, retained), storage)
    end

    @testset "vector combination arithmetic and input-first rejection" begin
        for n in (3, 277, 513), scales in ((0.75f0, -0.125f0), (0.0f0, 1.0f0))
            prepared = prepare_algorithm(VectorLinearCombinationF32,
                VectorLinearCombinationF32Configuration(extent=n, left_scale=scales[1],
                    right_scale=scales[2], schema="test.vector/1"))
            case = fixture(prepared, (n,), (n, n))
            scratch = candidate_storage(case.captured.execution)
            retained = (case.outputs..., case.inputs..., scratch..., case.status)
            addresses = map(pointer, retained)
            for frame in 1:3
                left = Float32[(i-frame) * 0.003f0 for i in 1:n]
                right = Float32[(i+frame) * -0.005f0 for i in 1:n]
                @test step!(case, (left, right)) === nothing
            end
            bad = fill(1.0f0, n); bad[end] = NaN32
            @test step!(case, (bad, ones(Float32, n))) === VectorLinearCombinationNonFiniteInput
            gated!(case, retained)
            @test reset!(case.captured) === case.captured
            synchronize()
            @test all(a -> all(iszero, Array(a)), scratch)
            @test map(pointer, retained) == addresses
            input = case.inputs[1]
            aliased = prepare_capture(CompleteTransition(), device, prepared, (input,),
                case.inputs, case.status, Int32(1))
            @test !FilterGraphAlgorithms.preserves_inputs(aliased)
            copyto!(input, fill(0.5f0, n)); copyto!(case.inputs[2], fill(-0.25f0, n))
            fill!(case.status, Int32(0)); enqueue!(aliased); synchronize()
            @test Array(input) == fill(fma(scales[1], 0.5f0, scales[2] * -0.25f0), n)
        end
        n = 277
        prepared = prepare_algorithm(VectorLinearCombinationF32,
            VectorLinearCombinationF32Configuration(extent=n, left_scale=2f0, right_scale=1f0,
                schema="test.vector/1"))
        case = fixture(prepared, (n,), (n,n))
        left = fill(floatmax(Float32), n); right = zeros(Float32,n)
        @test step!(case, (left,right)) === VectorLinearCombinationNonFiniteOutput
        right[end] = NaN32
        @test step!(case, (left,right)) === VectorLinearCombinationNonFiniteInput
    end

    @testset "single-block vector publication and staged boundary" begin
        scale = nextfloat(1.0f0)
        for n in (1, 3, 256, 257)
            prepared = prepare_algorithm(VectorLinearCombinationF32,
                VectorLinearCombinationF32Configuration(extent=n, left_scale=scale,
                    right_scale=-scale, schema="test.vector/1"))
            case = fixture(prepared, (n,), (n, n))
            has_native_block = isdefined(parentmodule(typeof(case.captured)), :_VectorCombinationBlock)
            @test isempty(candidate_storage(case.captured.execution)) == (has_native_block && n <= 256)

            left = fill(scale, n)
            right = copy(left)
            @test step!(case, (left, right)) === nothing
            # Rounding the right product and fusing the left product yields ε².
            @test Array(only(case.outputs)) == fill(eps(Float32)^2, n)
            gated!(case, (case.outputs..., case.inputs..., case.status))

            # One overflowing lane must preserve every output, including
            # lanes with finite results. A bad input still takes priority.
            left[end] = floatmax(Float32)
            right[end] = -floatmax(Float32)
            @test step!(case, (left, right)) === VectorLinearCombinationNonFiniteOutput
            right[1] = NaN32
            @test step!(case, (left, right)) === VectorLinearCombinationNonFiniteInput

            alias_output = case.inputs[2]
            aliased = prepare_capture(CompleteTransition(), device, prepared, (alias_output,),
                case.inputs, case.status, Int32(1))
            launch_alias! = record(aliased)
            copyto!(case.inputs[1], left)
            right[1] = scale
            right[end] = -floatmax(Float32)
            copyto!(alias_output, right)
            fill!(case.status, Int32(0))
            launch_alias!()
            synchronize()
            @test complete!(aliased, only(Array(case.status))) === VectorLinearCombinationNonFiniteOutput
            @test bits((alias_output,)) == (reinterpret(UInt32, right),)

            fill!(case.inputs[1], scale)
            fill!(alias_output, scale)
            fill!(case.status, Int32(0))
            launch_alias!()
            synchronize()
            @test only(Array(case.status)) == 0
            @test Array(alias_output) == fill(eps(Float32)^2, n)
        end
    end

    @testset "vector and shape offsets retain adopted parameters" begin
        for shape in (false,true), n in (3,277,513)
            physical = shape ? n+2 : n
            declaration, configuration = shape ?
                (VdmShapeOffsetF32, VdmShapeOffsetF32Configuration(active_vdm_count=n,
                    full_vdm_count=physical, physical_actuator_count=physical)) :
                (VectorOffsetF32, VectorOffsetF32Configuration(extent=n, schema="test.vector/1"))
            prepared = prepare_algorithm(declaration, configuration)
            offset = fill(0.125f0, physical); offset[1] = -0.0f0
            offset[2] = floatmax(Float32)
            if shape
                # Include a dependent completion row; its physical shape entry
                # remains in readback while the direct coordinates get projected.
                completion = zeros(Float32,physical,n)
                for i in 1:n; completion[i,i] = 1; end
                completion[end,1] = 0.5f0
                inverse = zeros(Float32,physical,physical)
                for i in 1:physical; inverse[i,i] = 1; end
                replace_parameter!(prepared, "active-to-full", completion)
                replace_parameter!(prepared, "pdm-to-vdm", inverse)
            end
            replace_parameter!(prepared, shape ? "shape-offset" : "offset", offset)
            case = fixture(prepared, (n,physical), (n,))
            adopted = shape ? (case.captured.active_offset,case.captured.shape_offset) : (case.captured.right,)
            scratch = candidate_storage(shape ? case.captured : case.captured.execution)
            retained = (case.outputs...,case.inputs...,adopted...,scratch...,case.status)
            addresses = map(pointer,retained)
            for frame in 1:3
                sample = fill(Float32(frame)*0.003f0,n); sample[1] = -0.0f0
                @test step!(case,(sample,)) === nothing
                @test reinterpret(UInt32,Array(case.outputs[2])) == reinterpret(UInt32,offset)
            end
            bad = zeros(Float32,n); bad[end] = NaN32
            @test step!(case,(bad,)) === VectorLinearCombinationNonFiniteInput
            bad[2] = floatmax(Float32)
            # Shape preserves per-coordinate failure ordering; VectorOffset
            # validates every requested coordinate before any arithmetic.
            expected = shape ? VectorLinearCombinationNonFiniteOutput : VectorLinearCombinationNonFiniteInput
            @test step!(case,(bad,)) === expected
            bad[end] = 0.0f0
            @test step!(case,(bad,)) === VectorLinearCombinationNonFiniteOutput
            gated!(case,retained)
            @test reset!(case.captured) === case.captured
            synchronize()
            @test all(a -> all(iszero, Array(a)), scratch)
            @test step!(case,(zeros(Float32,n),)) === nothing
            @test map(pointer,retained) == addresses
            @test_throws (shape ? DimensionMismatch : ArgumentError) prepare_capture(CompleteTransition(),device,prepared,
                (case.outputs[1],case.outputs[1]),case.inputs,case.status,Int32(1))
            # Adopting a host replacement after capture cannot mutate the retained
            # device parameter. Captured graphs pin their prepared parameters.
            replace_parameter!(prepared,shape ? "shape-offset" : "offset",zeros(Float32,physical))
            fill!(case.status,Int32(0)); copyto!(only(case.inputs),zeros(Float32,n))
            case.launch!(); synchronize()
            @test Array(case.outputs[2]) == offset
        end
    end

    @testset "Physical DM and tip/tilt commands preserve capture state and feedback" begin
        for n in (2,3,277,513), slew in (nothing,0.125f0)
            settings = (; command_limits=(-1f0,1f0), slew_limit=slew, initial_current=zeros(Float32,n))
            prepared = if n == 2
                prepare_algorithm(TipTiltCommandF32, TipTiltCommandF32Configuration(; settings...))
            else
                prepare_algorithm(PdmCommandF32, PdmCommandF32Configuration(; actuator_count=n, settings...))
            end
            case = fixture(prepared,(n,n),(n,))
            c = case.captured
            retained = (case.outputs...,case.inputs...,c.committed_current,c.candidate_current,case.status)
            addresses = map(pointer,retained)
            for frame in 1:3
                sample = Float32[mod(i+frame,9)*0.25f0-1f0 for i in 1:n]
                @test step!(case,(sample,)) === nothing
                @test Array(c.committed_current) == case.prepared.workspace.committed_current
            end
            before = Array(c.committed_current)
            bad = zeros(Float32,n); bad[end] = NaN32
            @test step!(case,(bad,)) === PdmCommandNonFiniteInput
            @test Array(c.committed_current) == before
            gated!(case,retained)
            reset!(case.prepared); @test reset!(c) === c; synchronize()
            @test Array(c.committed_current) == Array(c.candidate_current) == zeros(Float32,n)
            @test step!(case,(fill(0.25f0,n),)) === nothing
            @test map(pointer,retained) == addresses
            @test_throws ArgumentError prepare_capture(CompleteTransition(),device,case.prepared,
                (only(case.inputs),case.outputs[2]),case.inputs,case.status,Int32(1))
        end

        n = 277
        bound = floatmax(Float32)
        lower, upper, initial = fill(-1f0,n), fill(1f0,n), zeros(Float32,n)
        lower[end] = upper[end] = initial[end] = -bound
        config = PdmCommandF32Configuration(actuator_count=n,
            command_limits=(lower,upper), slew_limit=nothing,
            initial_current=initial)
        case = fixture(prepare_algorithm(PdmCommandF32,config),(n,n),(n,))
        c = case.captured
        before = Array(c.committed_current)
        sample = fill(0.5f0,n); sample[end] = bound
        # The transition stages outputs in Graph-owned buffers. Rejection
        # preserves committed current.
        @test step!(case,(sample,); preserve_outputs=false) === PdmCommandNonFiniteOutput
        @test Array(c.committed_current) == before
        sample[end] = -bound
        @test step!(case,(sample,)) === nothing
        @test Array(c.committed_current) == sample
    end

    @testset "vector validation reuses only qualified finite input proofs (extent=$n)" for n in (256, 277)
        qualified = FilterGraphAlgorithms._supports_validation_reuse(
            KernelAbstractions.get_backend(device_zeros(Int32,1)))
        prepared = prepare_algorithm(VectorLinearCombinationF32,
            VectorLinearCombinationF32Configuration(extent=n, left_scale=2f0, right_scale=1f0,
                schema="test.vector/1"))
        case = fixture(prepared,(n,),(n,n))
        original = case.captured
        extension = parentmodule(typeof(original))
        finite_input = getfield(extension,:_FiniteInput)
        @test FilterGraphAlgorithms.prepare_validation(original,(false,false)) === original
        @test FilterGraphAlgorithms.prepare_validation(original,(true,)) === original
        for proofs in ((false,false),(true,false),(false,true),(true,true))
            refined = FilterGraphAlgorithms.prepare_validation(original,proofs)
            @test (refined.validate_left! isa finite_input) == (qualified && proofs[1])
            @test (refined.validate_right! isa finite_input) == (qualified && proofs[2])
            !qualified && @test refined === original
            for name in fieldnames(typeof(original))
                name in (:validate_left!,:validate_right!) && continue
                @test getfield(refined,name) === getfield(original,name)
            end
            refined_case = merge(case,(captured=refined,launch! = record(refined)))
            # A finite certificate cannot certify a scaled result. Keep the
            # output validator and complete publication decision in every case.
            @test step!(refined_case,(fill(floatmax(Float32),n),zeros(Float32,n))) ===
                VectorLinearCombinationNonFiniteOutput
            if !qualified || !proofs[2]
                right = zeros(Float32,n); right[end] = NaN32
                @test step!(refined_case,(fill(floatmax(Float32),n),right)) ===
                    VectorLinearCombinationNonFiniteInput
            end
            if !qualified || !proofs[1]
                left = fill(floatmax(Float32),n); left[end] = NaN32
                @test step!(refined_case,(left,zeros(Float32,n))) ===
                    VectorLinearCombinationNonFiniteInput
            end
        end
        for poisoned in (false,true)
            prepared = prepare_algorithm(VectorOffsetF32,
                VectorOffsetF32Configuration(extent=n,schema="test.vector/1"))
            offset = fill(0.125f0,n); offset[1] = floatmax(Float32)
            replace_parameter!(prepared,"offset",offset)
            # A mutable internal source is not itself a finite certificate.
            # Prove the actual private snapshot or retain the original scan.
            poisoned && (prepared.plan.offset[end] = NaN32)
            case = fixture(prepared,(n,n),(n,))
            original = case.captured
            @test (original.validate_right! isa finite_input) == (qualified && !poisoned)
            @test FilterGraphAlgorithms.prepare_validation(original,(false,)) === original
            refined = FilterGraphAlgorithms.prepare_validation(original,(true,))
            @test (refined.validate_left! isa finite_input) == qualified
            @test refined.validate_right! === original.validate_right!
            @test refined.execution === original.execution
            @test refined.right === original.right
            refined_case = merge(case,(captured=refined,launch! = record(refined)))
            failure = poisoned ? VectorLinearCombinationNonFiniteInput : VectorLinearCombinationNonFiniteOutput
            @test step!(refined_case,(fill(floatmax(Float32),n),)) === failure
            if !poisoned
                @test step!(refined_case,(zeros(Float32,n),)) === nothing
                prepared.plan.offset[end] = NaN32
                fill!(case.status,Int32(0)); copyto!(only(case.inputs),zeros(Float32,n))
                refined_case.launch!(); synchronize()
                @test only(Array(case.status)) == 0
                @test Array(case.outputs[2]) == offset
            end
        end
    end
end
