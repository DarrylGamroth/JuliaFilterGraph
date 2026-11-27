using FilterGraphs

const DENSE_VECTOR_CAPTURED_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = dense-vector-reconstructor-f32
            config = {
                input_count = 3
                output_count = 2
                initial_reconstructor = [
                     1.0  10.0  100.0
                    -1.0 -10.0 -100.0
                ]
                input_schema = org.example.ao.ordered-input/1
                output_schema = org.example.ao.reconstructed-output/1
                reconstructor_schema = org.example.ao.dense-reconstructor/1
            }
        }
    ]
    links = []
    inputs = [ "reconstruct:ordered-input" "reconstruct:reconstructor" ]
    outputs = [ "reconstruct:output" ]
}
"""

const DENSE_VECTOR_CAPTURED_OVERFLOW_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = dense-vector-reconstructor-f32
            config = {
                input_count = 3
                output_count = 2
                initial_reconstructor = [
                    3.0e38 3.0e38 3.0e38
                    3.0e38 3.0e38 3.0e38
                ]
                input_schema = org.example.ao.ordered-input/1
                output_schema = org.example.ao.reconstructed-output/1
                reconstructor_schema = org.example.ao.dense-reconstructor/1
            }
        }
    ]
    links = []
    inputs = [ "reconstruct:ordered-input" "reconstruct:reconstructor" ]
    outputs = [ "reconstruct:output" ]
}
"""

function prepare_dense_vector_captured_graph(text)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        checked = FilterGraphs.prepare_graph(
            path;
            algorithms=(DenseVectorReconstructorF32,),
        )
        FilterGraphs.prepare_captured_graph(checked, TEST_DEVICE)
    end
end

const DENSE_VECTOR_CAPTURED_INPUT_NAME = Symbol("ordered-input")

dense_vector_captured_inputs(values) =
    NamedTuple{(DENSE_VECTOR_CAPTURED_INPUT_NAME,)}((collect(Float32.(values)),))

dense_vector_captured_metadata(metadata) =
    NamedTuple{(DENSE_VECTOR_CAPTURED_INPUT_NAME,)}((metadata,))

function dense_vector_captured_replay_allocation_bytes(
    outputs,
    captured,
    inputs,
    input_metadata,
)
    return @allocated FilterGraphs.process!(
        outputs,
        captured,
        inputs,
        input_metadata,
    )
end

@testset "captured dense-vector Graph preserves failure publication and replay allocation" begin
    captured = prepare_dense_vector_captured_graph(DENSE_VECTOR_CAPTURED_GRAPH_TEXT)
    output = fill(71.0f0, 2)
    outputs = (output=output,)
    metadata = SampleMetadata(51; timestamp_nanoseconds=5_100)
    inputs = dense_vector_captured_inputs((2, 3, 4))
    input_metadata = dense_vector_captured_metadata(metadata)

    @test FilterGraphs.process!(outputs, captured, inputs, input_metadata) ==
          (output=true,)
    @test output == Float32[432, -432]
    published_metadata = FilterGraphs.output_metadata(captured)
    @test published_metadata.output === metadata
    @test dense_vector_captured_replay_allocation_bytes(
        outputs,
        captured,
        inputs,
        input_metadata,
    ) == 0
    @test @inferred(
        Union{NamedTuple{(:output,),Tuple{Bool}},DenseReconstructionFailure},
        FilterGraphs.process!(outputs, captured, inputs, input_metadata),
    ) == (output=true,)

    published_output = copy(output)
    rejected_metadata = SampleMetadata(52; timestamp_nanoseconds=5_200)
    @test FilterGraphs.process!(
        outputs,
        captured,
        dense_vector_captured_inputs((2, NaN, 4)),
        dense_vector_captured_metadata(rejected_metadata),
    ) === DenseReconstructionNonFiniteInput
    @test output == published_output
    @test FilterGraphs.output_metadata(captured) == published_metadata

    recovered_metadata = SampleMetadata(53; timestamp_nanoseconds=5_300)
    @test FilterGraphs.process!(
        outputs,
        captured,
        dense_vector_captured_inputs((1, 2, 3)),
        dense_vector_captured_metadata(recovered_metadata),
    ) == (output=true,)
    @test output == Float32[321, -321]
    @test FilterGraphs.output_metadata(captured).output === recovered_metadata

    overflow = prepare_dense_vector_captured_graph(
        DENSE_VECTOR_CAPTURED_OVERFLOW_GRAPH_TEXT,
    )
    overflow_output = fill(83.0f0, 2)
    overflow_outputs = (output=overflow_output,)
    overflow_published_metadata = SampleMetadata(54; timestamp_nanoseconds=5_400)
    @test FilterGraphs.process!(
        overflow_outputs,
        overflow,
        dense_vector_captured_inputs((0, 0, 0)),
        dense_vector_captured_metadata(overflow_published_metadata),
    ) == (output=true,)
    @test overflow_output == zeros(Float32, 2)
    @test FilterGraphs.output_metadata(overflow).output ===
          overflow_published_metadata
    overflow_rejected_metadata = SampleMetadata(55; timestamp_nanoseconds=5_500)
    @test FilterGraphs.process!(
        overflow_outputs,
        overflow,
        dense_vector_captured_inputs((2, 2, 2)),
        dense_vector_captured_metadata(overflow_rejected_metadata),
    ) === DenseReconstructionNonFiniteOutput
    @test overflow_output == zeros(Float32, 2)
    @test FilterGraphs.output_metadata(overflow).output ===
          overflow_published_metadata
end

@testset "captured dense-vector Graph accepts scaled FP16 coefficient storage" begin
    text = replace(
        DENSE_VECTOR_CAPTURED_GRAPH_TEXT,
        "input_count = 3" => "input_count = 3\n                coefficient_storage = fp16",
    )
    captured = prepare_dense_vector_captured_graph(text)
    outputs = (output=zeros(Float32, 2),)
    inputs = dense_vector_captured_inputs((2, 3, 4))
    metadata = dense_vector_captured_metadata(SampleMetadata(1; timestamp_nanoseconds=100))
    @test FilterGraphs.process!(outputs, captured, inputs, metadata) == (output=true,)
    @test outputs.output == Float32[432, -432]
end
