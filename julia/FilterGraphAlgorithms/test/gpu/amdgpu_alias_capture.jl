module AMDGPUAliasCaptureTests

using AMDGPU, FilterGraphAlgorithms, LinearAlgebra, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, reset!, complete!

const EXT = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsAMDGPUExt)

primitive type Padded24Bits 24 end

@testset "AMDGPU cold contiguous overlap includes derived storage" begin
    storage = AMDGPU.zeros(Float32, 32)
    first = view(storage, 1:16)
    shifted = view(storage, 16:32)
    adjacent = view(storage, 17:32)
    nested = view(view(storage, 4:24), 7:18) # Parent entries 10:21.
    shaped = reshape(view(storage, 8:23), 4, 4)
    empty = view(storage, 1:0)
    distinct = AMDGPU.zeros(Float32, 16)
    for (left, right, expected) in ((first, first, true),
            (first, shifted, true), (first, nested, true),
            (first, shaped, true), (first, adjacent, false),
            (first, distinct, false), (empty, first, false),
            (empty, empty, false))
        @test EXT._capture_arrays_overlap(left, right) == expected
        @test EXT._capture_arrays_overlap(right, left) == expected
    end
    # This method is private: do not change the library's global alias methods.
    @test EXT._capture_arrays_overlap(zeros(Float32, 2), zeros(Float32, 2)) == false

    # Array element spacing can exceed sizeof(T), including on ROCArray.
    padded = AMDGPU.ROCArray{Padded24Bits}(undef, 8)
    @test sizeof(Padded24Bits) == 3
    @test Base.elsize(typeof(padded)) == 4
    @test EXT._capture_arrays_overlap(view(padded, 1:4), view(padded, 4:4))
    @test !EXT._capture_arrays_overlap(view(padded, 1:4), view(padded, 5:5))
end

@testset "AMDGPU shifted feedback retains scratch across several blocks" begin
    count = 513
    identity = Matrix{Float32}(I, count, count)
    prepared = prepare_algorithm(PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(active_vdm_count=count,
            full_vdm_count=count, physical_actuator_count=count))
    replace_parameter!(prepared, "pdm-to-vdm", identity)
    replace_parameter!(prepared, "full-to-active", identity)
    storage = AMDGPU.zeros(Float32, 2count-1)
    input = view(storage, 1:count)
    output = view(storage, count:2count-1)
    status = AMDGPU.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), AMDGPU.device(), prepared,
        (output,), (input,), status, Int32(2))
    @test captured.full_vdm_feedback !== input
    @test captured.full_vdm_feedback !== output
    @test !EXT._capture_arrays_overlap(captured.full_vdm_feedback, input)
    @test !EXT._capture_arrays_overlap(captured.full_vdm_feedback, output)
    values = Float32[isodd(i) ? i / 1024 : -i / 1024 for i in 1:count]
    values[1] = -0.0f0
    copyto!(input, values)
    enqueue!(captured)
    AMDGPU.synchronize()
    @test iszero(only(Array(status)))
    @test reinterpret(UInt32, Array(output)) == reinterpret(UInt32, values)
    before = Array(storage)
    reset!(captured)
    AMDGPU.synchronize()
    @test reinterpret(UInt32, Array(storage)) == reinterpret(UInt32, before)
    fill!(status, Int32(257))
    enqueue!(captured)
    AMDGPU.synchronize()
    @test Array(status) == Int32[257]
    @test reinterpret(UInt32, Array(storage)) == reinterpret(UInt32, before)
    bad = copy(values); bad[end] = NaN32
    copyto!(input, bad)
    before_output = Array(output)
    fill!(status, Int32(0))
    enqueue!(captured)
    AMDGPU.synchronize()
    @test complete!(captured, only(Array(status))) === PdmFeedbackToVdmNonFiniteInput
    @test reinterpret(UInt32, Array(output)) == reinterpret(UInt32, before_output)
    copyto!(input, values)
    fill!(status, Int32(0))
    enqueue!(captured)
    AMDGPU.synchronize()
    @test iszero(only(Array(status)))
    @test reinterpret(UInt32, Array(output)) == reinterpret(UInt32, values)

    # Exercise the other borrowing decision used by VDM-to-PDM completion.
    intermediate = EXT._capture_projection_input(AMDGPU.device(), identity, input, output)
    @test intermediate !== input
    @test !EXT._capture_arrays_overlap(intermediate, input)
    @test !EXT._capture_arrays_overlap(intermediate, output)
    disjoint = AMDGPU.zeros(Float32, count)
    @test EXT._capture_projection_input(AMDGPU.device(), identity, input, disjoint) === input
end

end # module
