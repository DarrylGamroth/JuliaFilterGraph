@enum DocrimeCovarianceGateFailure::UInt8 begin
    DocrimeCovarianceGateRejected = 1
end

struct DocrimeCovarianceGatePlan
    rows::Int
    columns::Int
end

@algorithm DocrimeCovarianceGateF32 begin
    label = "test-docrime-covariance-gate-f32"
    failure = DocrimeCovarianceGateFailure

    @configuration DocrimeCovarianceGateF32Configuration begin
        rows::Int => graph_rebuild
        columns::Int => graph_rebuild
        schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.rows, configuration.columns),
            configuration.schema,
        ),
        output(
            :accepted,
            Float32,
            (configuration.rows, configuration.columns),
            configuration.schema;
            metadata_from=:input,
        ),
    )

    prepare(configuration) =
        DocrimeCovarianceGatePlan(configuration.rows, configuration.columns)

    process!(plan) = begin
        input[1] < 0.0f0 && return DocrimeCovarianceGateRejected
        copyto!(accepted, input)
        nothing
    end
end

const DOCRIME_COVARIANCE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = covariance
            label = docrime-covariance-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                sample_limit = null
                measurements_schema = "example/test-measurements/1"
                corrector_sample_schema =
                    "example/test-corrector-sample/1"
                cross_covariance_schema =
                    "example/test-measurement-corrector-covariance/1"
                corrector_covariance_schema =
                    "example/test-corrector-covariance/1"
            }
        }
    ]
    links = []
    inputs = [
        "covariance:measurements"
        "covariance:corrector-sample"
    ]
    outputs = [
        "covariance:measurement-corrector-cross-covariance-sum"
        "covariance:corrector-covariance-sum"
        "covariance:sample-count"
        "covariance:complete"
    ]
}
"""

const DOCRIME_COVARIANCE_GATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = covariance
            label = docrime-covariance-f32
            config = {
                measurement_count = 1
                corrector_coordinate_count = 1
                sample_limit = 3
                measurements_schema = "example/test-measurements/1"
                corrector_sample_schema =
                    "example/test-corrector-sample/1"
                cross_covariance_schema =
                    "example/test-measurement-corrector-covariance/1"
                corrector_covariance_schema =
                    "example/test-corrector-covariance/1"
            }
        }
        {
            type = ndarray
            name = gate
            label = test-docrime-covariance-gate-f32
            config = {
                rows = 1
                columns = 1
                schema =
                    "example/test-measurement-corrector-covariance/1"
            }
        }
    ]
    links = [
        {
            output =
                "covariance:measurement-corrector-cross-covariance-sum"
            input = "gate:input"
        }
    ]
    inputs = [
        "covariance:measurements"
        "covariance:corrector-sample"
    ]
    outputs = [
        "gate:accepted"
        "covariance:corrector-covariance-sum"
        "covariance:sample-count"
        "covariance:complete"
    ]
}
"""

const DOCRIME_COVARIANCE_INPUT_NAMES = (
    :measurements,
    Symbol("corrector-sample"),
)
const DOCRIME_COVARIANCE_OUTPUT_NAMES = (
    Symbol("measurement-corrector-cross-covariance-sum"),
    Symbol("corrector-covariance-sum"),
    Symbol("sample-count"),
    :complete,
)
const DOCRIME_COVARIANCE_GATE_OUTPUT_NAMES = (
    :accepted,
    Symbol("corrector-covariance-sum"),
    Symbol("sample-count"),
    :complete,
)

docrime_covariance_graph_inputs(measurements, corrector_sample) = NamedTuple{
    DOCRIME_COVARIANCE_INPUT_NAMES,
}((measurements, corrector_sample))

docrime_covariance_graph_outputs(cross, corrector, count, complete) = NamedTuple{
    DOCRIME_COVARIANCE_OUTPUT_NAMES,
}((cross, corrector, count, complete))

docrime_covariance_gate_outputs(accepted, corrector, count, complete) = NamedTuple{
    DOCRIME_COVARIANCE_GATE_OUTPUT_NAMES,
}((accepted, corrector, count, complete))

docrime_covariance_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function docrime_covariance_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "DO-CRIME covariance prepares and processes as a Graph Node" begin
    graph = prepare_graph_text(
        DOCRIME_COVARIANCE_GRAPH_TEXT;
        algorithms=(DocrimeCovarianceF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:covariance,)
    @test graph.node_failure_types == (DocrimeCovarianceFailure,)
    @test keys(graph.input_formats) == DOCRIME_COVARIANCE_INPUT_NAMES
    @test keys(graph.output_formats) == DOCRIME_COVARIANCE_OUTPUT_NAMES
    @test getproperty.(values(graph.input_formats), :shape) == ((2,), (2,))
    @test getproperty.(values(graph.output_formats), :shape) ==
          ((2, 2), (2, 2), (1,), (1,))
    @test getproperty(
        graph.output_formats,
        Symbol("sample-count"),
    ).schema == DOCRIME_COVARIANCE_SAMPLE_COUNT_V1

    cross = fill(NaN32, 2, 2)
    corrector_covariance = fill(NaN32, 2, 2)
    count = fill(typemax(UInt64), 1)
    complete = fill(true, 1)
    outputs = docrime_covariance_graph_outputs(
        cross,
        corrector_covariance,
        count,
        complete,
    )
    first = docrime_covariance_graph_inputs(
        Float32[10, 4.5],
        Float32[1, 2],
    )
    availability = NamedTuple{DOCRIME_COVARIANCE_OUTPUT_NAMES}(
        (true, true, true, true),
    )
    @test process!(outputs, graph, first) == availability
    @test cross == zeros(Float32, 2, 2)
    @test corrector_covariance == zeros(Float32, 2, 2)
    @test count == UInt64[1]
    @test complete == [false]

    second = docrime_covariance_graph_inputs(
        Float32[16, -0.5],
        Float32[3, 0],
    )
    metadata = NamedTuple{DOCRIME_COVARIANCE_INPUT_NAMES}(
        (
            SampleMetadata(41, 0; timestamp_nanoseconds=9013, terminal=true),
            SampleMetadata(41, 0; timestamp_nanoseconds=9017, terminal=true),
        ),
    )
    @test process!(outputs, graph, second, metadata) == availability
    @test cross == Float32[6 -6; -5 5]
    @test corrector_covariance == Float32[2 -2; -2 2]
    @test count == UInt64[2]
    @test complete == [false]
    @test all(
        value -> value == metadata.measurements,
        values(FilterGraphs.output_metadata(graph)),
    )
end

@testset "downstream rejection preserves outputs and advances covariance history" begin
    graph = prepare_graph_text(
        DOCRIME_COVARIANCE_GATE_GRAPH_TEXT;
        algorithms=(DocrimeCovarianceF32, DocrimeCovarianceGateF32),
    )
    workspace = first(graph.nodes).prepared.workspace
    outputs = docrime_covariance_gate_outputs(
        fill(71.0f0, 1, 1),
        fill(72.0f0, 1, 1),
        fill(UInt64(73), 1),
        fill(true, 1),
    )
    availability = NamedTuple{DOCRIME_COVARIANCE_GATE_OUTPUT_NAMES}(
        (true, true, true, true),
    )

    first_inputs = docrime_covariance_graph_inputs(Float32[10], Float32[1])
    @test process!(outputs, graph, first_inputs) == availability
    @test docrime_covariance_sample_count(workspace) == 1
    retained_outputs = map(copy, outputs)
    retained_mean = docrime_measurement_mean(workspace)

    rejected = docrime_covariance_graph_inputs(Float32[5], Float32[2])
    @test process!(outputs, graph, rejected) === DocrimeCovarianceGateRejected
    @test outputs == retained_outputs
    @test docrime_covariance_sample_count(workspace) == 2
    @test docrime_measurement_mean(workspace) != retained_mean
    @test docrime_measurement_mean(workspace) == Float32[7.5]

    accepted = docrime_covariance_graph_inputs(Float32[15], Float32[2])
    @test process!(outputs, graph, accepted) == availability
    @test outputs.accepted == zeros(Float32, 1, 1)
    @test getproperty(outputs, Symbol("sample-count")) == UInt64[3]
    @test outputs.complete == [true]
    @test docrime_covariance_sample_count(workspace) == 3
end

@testset "DO-CRIME covariance Graph outcomes allocate no heap" begin
    graph = prepare_graph_text(
        DOCRIME_COVARIANCE_GRAPH_TEXT;
        algorithms=(DocrimeCovarianceF32,),
    )
    outputs = docrime_covariance_graph_outputs(
        zeros(Float32, 2, 2),
        zeros(Float32, 2, 2),
        zeros(UInt64, 1),
        falses(1),
    )
    valid = docrime_covariance_graph_inputs(Float32[10, 4.5], Float32[1, 2])
    failures = (
        (
            docrime_covariance_graph_inputs(Float32[NaN, 0], Float32[0, 0]),
            DocrimeCovarianceNonFiniteMeasurements,
        ),
        (
            docrime_covariance_graph_inputs(Float32[0, 0], Float32[0, Inf]),
            DocrimeCovarianceNonFiniteCorrectorSample,
        ),
    )
    process!(outputs, graph, valid)
    @test docrime_covariance_graph_allocation_bytes(outputs, graph, valid) == 0
    @test docrime_covariance_graph_admitted_allocation_bytes(
        outputs,
        graph,
        valid,
    ) == 0
    for (inputs, failure) in failures
        @test process!(outputs, graph, inputs) === failure
        @test docrime_covariance_graph_allocation_bytes(outputs, graph, inputs) == 0
        @test docrime_covariance_graph_admitted_allocation_bytes(
            outputs,
            graph,
            inputs,
        ) == 0
    end

end
