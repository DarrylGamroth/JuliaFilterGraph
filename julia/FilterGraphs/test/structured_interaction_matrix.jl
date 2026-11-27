const STRUCTURED_ESTIMATOR_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "structured_interaction_matrix.toml"),
)

const ZONAL_INTERACTION_MATRIX_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = estimate
            label = zonal-interaction-matrix-f32
            config = {
                measurement_count = 2
                amplitudes = [ 0.5 2.0 1.5 ]
                response_schema = "example/test-zonal-response-cycle/1"
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
        }
    ]
    links = []
    inputs = [ "estimate:responses" "estimate:amplitudes" ]
    outputs = [ "estimate:interaction-matrix" ]
}
"""

const HADAMARD_INTERACTION_MATRIX_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = estimate
            label = hadamard-interaction-matrix-f32
            config = {
                measurement_count = 2
                amplitudes = [ 0.25 2.0 0.5 ]
                response_schema = "example/test-hadamard-response-cycle/1"
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
        }
    ]
    links = []
    inputs = [ "estimate:responses" "estimate:amplitudes" ]
    outputs = [ "estimate:interaction-matrix" ]
}
"""

function structured_graph_matrix(values, rows, columns)
    return permutedims(reshape(Float32.(values), columns, rows))
end

function structured_graph_fixture(name)
    fixture = STRUCTURED_ESTIMATOR_FIXTURE[name]
    response_rows = name == "zonal" ?
                    2 * fixture["corrector_coordinate_count"] :
                    2 * fixture["order"]
    responses = structured_graph_matrix(
        fixture["responses"],
        response_rows,
        fixture["measurement_count"],
    )
    expected = structured_graph_matrix(
        fixture["expected_interaction_matrix"],
        fixture["measurement_count"],
        fixture["corrector_coordinate_count"],
    )
    return responses, expected
end

structured_graph_outputs(output) = NamedTuple{
    (Symbol("interaction-matrix"),),
}((output,))

structured_graph_process_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function structured_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end


@testset "structured estimators prepare as Graph Nodes" begin
    cases = (
        (
            name="zonal",
            text=ZONAL_INTERACTION_MATRIX_GRAPH_TEXT,
            declaration=ZonalInteractionMatrixF32,
            response_shape=(6, 2),
        ),
        (
            name="hadamard",
            text=HADAMARD_INTERACTION_MATRIX_GRAPH_TEXT,
            declaration=HadamardInteractionMatrixF32,
            response_shape=(8, 2),
        ),
    )

    for case in cases
        graph = prepare_graph_text(case.text; algorithms=(case.declaration,))
        @test keys(graph.input_formats) == (:responses,)
        @test keys(graph.parameter_formats) == (:amplitudes,)
        @test keys(graph.output_formats) == (Symbol("interaction-matrix"),)
        @test graph.node_failure_types == (StructuredInteractionMatrixFailure,)
        @test graph.input_formats.responses.shape == case.response_shape
        @test graph.parameter_formats.amplitudes.shape == (3,)
        @test getproperty(
            graph.output_formats,
            Symbol("interaction-matrix"),
        ).shape == (2, 3)

        responses, expected = structured_graph_fixture(case.name)
        output = fill(NaN32, 2, 3)
        outputs = structured_graph_outputs(output)
        inputs = (responses=responses,)
        @test process!(outputs, graph, inputs) == NamedTuple{
            (Symbol("interaction-matrix"),),
        }((true,))
        @test output ≈ expected atol =
              STRUCTURED_ESTIMATOR_FIXTURE["absolute_tolerance_f32"]

        metadata = SampleMetadata(
            71,
            0;
            timestamp_nanoseconds=9013,
            terminal=true,
        )
        @test process!(outputs, graph, inputs, (responses=metadata,)) ==
              NamedTuple{(Symbol("interaction-matrix"),)}((true,))
        @test getproperty(
            FilterGraphs.output_metadata(graph),
            Symbol("interaction-matrix"),
        ) == metadata
    end
end

@testset "structured Graph amplitude replacement is failure-atomic" begin
    graph = prepare_graph_text(
        ZONAL_INTERACTION_MATRIX_GRAPH_TEXT;
        algorithms=(ZonalInteractionMatrixF32,),
    )
    responses, expected = structured_graph_fixture("zonal")
    output = zeros(Float32, 2, 3)
    outputs = structured_graph_outputs(output)
    inputs = (responses=responses,)

    process!(outputs, graph, inputs)
    @test output ≈ expected atol =
          STRUCTURED_ESTIMATOR_FIXTURE["absolute_tolerance_f32"]
    original = only(graph.nodes).prepared.plan
    doubled = Float32[1, 4, 3]
    @test replace_parameters!(graph, :amplitudes => doubled) === graph
    @test only(graph.nodes).prepared.plan !== original
    process!(outputs, graph, inputs)
    @test output ≈ expected ./ 2 atol =
          STRUCTURED_ESTIMATOR_FIXTURE["absolute_tolerance_f32"]

    active = only(graph.nodes).prepared.plan
    @test_throws ArgumentError replace_parameters!(
        graph,
        :amplitudes => Float32[1, 0, 3],
    )
    @test only(graph.nodes).prepared.plan === active
end

@testset "structured Graph rejection withholds output and metadata" begin
    graph = prepare_graph_text(
        ZONAL_INTERACTION_MATRIX_GRAPH_TEXT;
        algorithms=(ZonalInteractionMatrixF32,),
    )
    responses, _ = structured_graph_fixture("zonal")
    output = fill(71.0f0, 2, 3)
    outputs = structured_graph_outputs(output)
    retained = copy(output)
    retained_metadata = FilterGraphs.output_metadata(graph)

    nonfinite = copy(responses)
    nonfinite[1] = NaN32
    @test process!(outputs, graph, (responses=nonfinite,)) ===
          StructuredNonFiniteResponse
    @test output == retained
    @test FilterGraphs.output_metadata(graph) == retained_metadata

    overflow = copy(responses)
    overflow[1, 1] = floatmax(Float32)
    overflow[2, 1] = -floatmax(Float32)
    @test process!(outputs, graph, (responses=overflow,)) ===
          StructuredNonFiniteCalculation
    @test output == retained
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "structured estimator Graph outcomes allocate no heap after warmup" begin
    cases = (
        (
            name="zonal",
            text=ZONAL_INTERACTION_MATRIX_GRAPH_TEXT,
            declaration=ZonalInteractionMatrixF32,
        ),
        (
            name="hadamard",
            text=HADAMARD_INTERACTION_MATRIX_GRAPH_TEXT,
            declaration=HadamardInteractionMatrixF32,
        ),
    )

    for case in cases
        graph = prepare_graph_text(case.text; algorithms=(case.declaration,))
        responses, expected = structured_graph_fixture(case.name)
        output = similar(expected)
        outputs = structured_graph_outputs(output)
        success = (responses=responses,)
        nonfinite_responses = copy(responses)
        nonfinite_responses[1] = NaN32
        nonfinite = (responses=nonfinite_responses,)
        overflow_responses = copy(responses)
        overflow_responses[1, 1] = floatmax(Float32)
        overflow_responses[2, 1] = -floatmax(Float32)
        overflow = (responses=overflow_responses,)

        for (inputs, result) in (
            (success, (var"interaction-matrix"=true,)),
            (nonfinite, StructuredNonFiniteResponse),
            (overflow, StructuredNonFiniteCalculation),
        )
            @test process!(outputs, graph, inputs) === result
            @test structured_graph_process_allocation_bytes(
                outputs,
                graph,
                inputs,
            ) == 0
            @test structured_graph_admitted_allocation_bytes(
                outputs,
                graph,
                inputs,
            ) == 0
        end

    end
end
