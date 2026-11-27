import FilterGraphAlgorithms:
    complete!,
    enqueue!,
    prepare_capture

function prepared_dense_vector_reconstructor(; matrix=Float32[1 10 100; -1 -10 -100])
    return prepare_algorithm(
        DenseVectorReconstructorF32,
        DenseVectorReconstructorF32Configuration(
            input_count=3,
            output_count=2,
            initial_reconstructor=collect(vec(transpose(matrix))),
            input_schema="org.example.ao.ordered-input/1",
            output_schema="org.example.ao.reconstructed-output/1",
            reconstructor_schema="org.example.ao.dense-reconstructor/1",
        ),
    )
end

@testset "AMDGPU dense-vector capture preserves the checked projection contract" begin
    prepared = prepared_dense_vector_reconstructor()
    input = AMDGPU.ROCArray(Float32[2, 3, 4])
    output = AMDGPU.fill(71.0f0, 2)
    frame_status = AMDGPU.zeros(Int32, 1)
    captured = prepare_capture(
        CompleteTransition(),
        TEST_DEVICE,
        prepared,
        (output,),
        (input,),
        frame_status,
        Int32(1),
    )
    @test capture_support(DenseVectorReconstructorF32, TEST_DEVICE) isa SupportsCapture
    @test AMDGPU.device(captured.reconstructor) == TEST_DEVICE
    enqueue!(captured)
    AMDGPU.synchronize()
    @test only(Array(frame_status)) == 0
    @test complete!(captured, Int32(0)) === nothing
    @test Array(output) == Float32[432, -432]

    copyto!(input, Float32[2, NaN, 4])
    fill!(output, 71.0f0)
    fill!(frame_status, Int32(0))
    enqueue!(captured)
    AMDGPU.synchronize()
    input_status = only(Array(frame_status))
    @test complete!(captured, input_status) ===
          DenseReconstructionNonFiniteInput
    @test Array(output) == fill(71.0f0, 2)

    overflow = prepared_dense_vector_reconstructor(
        matrix=fill(floatmax(Float32), 2, 3),
    )
    overflow_output = AMDGPU.fill(71.0f0, 2)
    overflow_status = AMDGPU.zeros(Int32, 1)
    overflow_captured = prepare_capture(
        CompleteTransition(),
        TEST_DEVICE,
        overflow,
        (overflow_output,),
        (AMDGPU.ROCArray(Float32[2, 2, 2]),),
        overflow_status,
        Int32(2),
    )
    enqueue!(overflow_captured)
    AMDGPU.synchronize()
    status = only(Array(overflow_status))
    @test complete!(overflow_captured, status) ===
          DenseReconstructionNonFiniteOutput
    @test all(isinf, Array(overflow_output))

    copyto!(input, Float32[2, 3, 4])
    fill!(frame_status, Int32(0))
    enqueue!(captured)
    AMDGPU.synchronize()
end
