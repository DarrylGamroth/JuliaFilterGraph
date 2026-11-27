struct DocrimeLinearSensorPlan
    interaction_matrix::Matrix{Float32}
    offset::Vector{Float32}
end

@algorithm DocrimeLinearSensorF32 begin
    label = "test-docrime-linear-sensor-f32"

    @configuration DocrimeLinearSensorF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :excitation,
            Float32,
            (configuration.corrector_count,),
            DOCRIME_EXCITATION_V1,
        ),
        output(
            :measurements,
            Float32,
            (configuration.measurement_count,),
            "example/test-measurements/1";
            metadata_from=:excitation,
        ),
    )

    prepare(configuration) = begin
        configuration.measurement_count == 2 ||
            throw(ArgumentError("test sensor requires two measurements"))
        configuration.corrector_count == 2 ||
            throw(ArgumentError("test sensor requires two corrector coordinates"))
        DocrimeLinearSensorPlan(
            Float32[1.5 -0.25; 0.4 2.0],
            Float32[10, -3],
        )
    end

    process!(plan) = begin
        for row in axes(plan.interaction_matrix, 1)
            value = plan.offset[row]
            for column in axes(plan.interaction_matrix, 2)
                value += plan.interaction_matrix[row, column] * excitation[column]
            end
            measurements[row] = value
        end
        nothing
    end
end

const DOCRIME_SOLVE_INPUT_NAMES = (
    Symbol("measurement-corrector-cross-covariance-sum"),
    Symbol("corrector-covariance-sum"),
    Symbol("sample-count"),
)
const DOCRIME_SOLVE_OUTPUT_NAMES = (
    Symbol("interaction-matrix"),
    :status,
    Symbol("status-coordinates"),
    Symbol("diagnostic-sample-count"),
    Symbol("corrector-variance-range"),
    Symbol("maximum-absolute-corrector-correlation"),
    Symbol("minimum-factorization-pivot-ratio"),
)

const DOCRIME_OPEN_LOOP_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = solve
            label = docrime-open-loop-interaction-matrix-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                minimum_corrector_variance = 0.1
                cross_covariance_schema =
                    "example/test-cross-covariance/1"
                corrector_covariance_schema =
                    "example/test-corrector-covariance/1"
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
            props = {
                maximum-absolute-corrector-correlation = 0.8
                minimum-factorization-pivot-ratio = 0.01
            }
        }
    ]
    links = []
    inputs = [
        "solve:measurement-corrector-cross-covariance-sum"
        "solve:corrector-covariance-sum"
        "solve:sample-count"
    ]
    outputs = [
        "solve:interaction-matrix"
        "solve:status"
        "solve:status-coordinates"
        "solve:diagnostic-sample-count"
        "solve:corrector-variance-range"
        "solve:maximum-absolute-corrector-correlation"
        "solve:minimum-factorization-pivot-ratio"
    ]
}
"""

const DOCRIME_CLOSED_LOOP_GRAPH_TEXT = replace(
    DOCRIME_OPEN_LOOP_GRAPH_TEXT,
    "docrime-open-loop-interaction-matrix-f32" =>
        "docrime-closed-loop-interaction-matrix-f32",
    "maximum-absolute-corrector-correlation = 0.8" =>
        "loop-gain = 0.4\n                maximum-absolute-corrector-correlation = 0.8",
)

const DOCRIME_CHAIN_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = excitation
            label = docrime-excitation-f32
            config = {
                amplitudes = [ 0.5 1.0 ]
                seed = 81985529216486895
            }
        }
        {
            type = ndarray
            name = sensor
            label = test-docrime-linear-sensor-f32
            config = { measurement_count = 2 corrector_count = 2 }
        }
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
                    "pipewireao/docrime-excitation-f32/1"
                cross_covariance_schema =
                    "example/test-cross-covariance/1"
                corrector_covariance_schema =
                    "example/test-corrector-covariance/1"
            }
        }
        {
            type = ndarray
            name = solve
            label = docrime-open-loop-interaction-matrix-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                minimum_corrector_variance = 0.0001
                cross_covariance_schema =
                    "example/test-cross-covariance/1"
                corrector_covariance_schema =
                    "example/test-corrector-covariance/1"
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
            props = {
                maximum-absolute-corrector-correlation = 0.99
                minimum-factorization-pivot-ratio = 0.0001
            }
        }
    ]
    links = [
        { output = "excitation:excitation" input = "sensor:excitation" }
        {
            output = "excitation:excitation"
            input = "covariance:corrector-sample"
        }
        {
            output = "sensor:measurements"
            input = "covariance:measurements"
        }
        {
            output =
                "covariance:measurement-corrector-cross-covariance-sum"
            input = "solve:measurement-corrector-cross-covariance-sum"
        }
        {
            output = "covariance:corrector-covariance-sum"
            input = "solve:corrector-covariance-sum"
        }
        {
            output = "covariance:sample-count"
            input = "solve:sample-count"
        }
    ]
    inputs = []
    outputs = [
        "solve:interaction-matrix"
        "solve:status"
        "solve:diagnostic-sample-count"
        "covariance:complete"
    ]
}
"""

docrime_solve_inputs(cross, corrector, count) = NamedTuple{
    DOCRIME_SOLVE_INPUT_NAMES,
}((cross, corrector, count))

function docrime_solve_outputs()
    return NamedTuple{DOCRIME_SOLVE_OUTPUT_NAMES}(
        (
            fill(NaN32, 2, 2),
            fill(typemax(UInt8), 1),
            fill(typemax(UInt64), 2),
            fill(typemax(UInt64), 1),
            fill(NaN32, 2),
            fill(NaN32, 1),
            fill(NaN32, 1),
        ),
    )
end

docrime_solve_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function docrime_solve_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "DO-CRIME solve declarations prepare as Graph Nodes" begin
    for (text, declaration, scale) in (
        (
            DOCRIME_OPEN_LOOP_GRAPH_TEXT,
            DocrimeOpenLoopInteractionMatrixF32,
            1.0f0,
        ),
        (
            DOCRIME_CLOSED_LOOP_GRAPH_TEXT,
            DocrimeClosedLoopInteractionMatrixF32,
            0.4f0,
        ),
    )
        graph = prepare_graph_text(text; algorithms=(declaration,))
        @test graph.node_failure_types == (DocrimeInteractionMatrixFailure,)
        @test keys(graph.input_formats) == DOCRIME_SOLVE_INPUT_NAMES
        @test keys(graph.output_formats) == DOCRIME_SOLVE_OUTPUT_NAMES
        @test getproperty.(values(graph.input_formats), :shape) ==
              ((2, 2), (2, 2), (1,))
        @test getproperty.(values(graph.output_formats), :shape) == (
            (2, 2),
            (1,),
            (2,),
            (1,),
            (2,),
            (1,),
            (1,),
        )
        @test property_value(only(graph.nodes).prepared.plan, 0) ===
              (declaration === DocrimeClosedLoopInteractionMatrixF32 ? 0.4f0 : 0.8f0)

        inputs = docrime_solve_inputs(
            Float32[6 0; 8 13],
            Float32[4 2; 2 4],
            UInt64[3],
        )
        outputs = docrime_solve_outputs()
        availability = NamedTuple{DOCRIME_SOLVE_OUTPUT_NAMES}(
            ntuple(_ -> true, Val(7)),
        )
        metadata = NamedTuple{DOCRIME_SOLVE_INPUT_NAMES}(
            (
                SampleMetadata(71, 0; timestamp_nanoseconds=9013, terminal=true),
                SampleMetadata(71, 0; timestamp_nanoseconds=9017, terminal=true),
                SampleMetadata(71, 0; timestamp_nanoseconds=9021, terminal=true),
            ),
        )
        @test process!(outputs, graph, inputs, metadata) == availability
        @test getproperty(outputs, Symbol("interaction-matrix")) ≈
              Float32[2 -1; 0.5 3] .* scale atol = 2.0f-6
        @test outputs.status == UInt8[0]
        @test getproperty(outputs, Symbol("status-coordinates")) == UInt64[0, 0]
        @test all(
            value -> value == metadata[1],
            values(FilterGraphs.output_metadata(graph)),
        )
    end
end

@testset "DO-CRIME Graph rejection withholds every solve output" begin
    graph = prepare_graph_text(
        DOCRIME_OPEN_LOOP_GRAPH_TEXT;
        algorithms=(DocrimeOpenLoopInteractionMatrixF32,),
    )
    valid = docrime_solve_inputs(
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
        UInt64[3],
    )
    outputs = docrime_solve_outputs()
    process!(outputs, graph, valid)
    retained_outputs = map(copy, outputs)
    retained_metadata = FilterGraphs.output_metadata(graph)
    failures = (
        (
            docrime_solve_inputs(
                Float32[NaN 0; 0 0],
                valid[2],
                valid[3],
            ),
            DocrimeInteractionMatrixNonFiniteCrossCovariance,
        ),
        (
            docrime_solve_inputs(
                valid[1],
                Float32[4 1; 2 4],
                valid[3],
            ),
            DocrimeInteractionMatrixNonSymmetricCorrectorCovariance,
        ),
        (
            docrime_solve_inputs(
                valid[1],
                valid[2],
                UInt64[16_777_218],
            ),
            DocrimeInteractionMatrixSampleCountNotRepresentable,
        ),
    )
    for (inputs, failure) in failures
        @test process!(outputs, graph, inputs) === failure
        @test outputs == retained_outputs
        @test FilterGraphs.output_metadata(graph) == retained_metadata
        @test docrime_solve_graph_allocation_bytes(outputs, graph, inputs) == 0
        @test docrime_solve_graph_admitted_allocation_bytes(
            outputs,
            graph,
            inputs,
        ) == 0
    end
    @test docrime_solve_graph_allocation_bytes(outputs, graph, valid) == 0
    @test docrime_solve_graph_admitted_allocation_bytes(outputs, graph, valid) == 0
end

@testset "Julia Graph dogfoods the complete open-loop DO-CRIME chain" begin
    graph = prepare_graph_text(
        DOCRIME_CHAIN_GRAPH_TEXT;
        algorithms=(
            DocrimeExcitationF32,
            DocrimeLinearSensorF32,
            DocrimeCovarianceF32,
            DocrimeOpenLoopInteractionMatrixF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) ==
          (:excitation, :sensor, :covariance, :solve)
    @test isempty(keys(graph.input_formats))
    output_names = (
        Symbol("interaction-matrix"),
        :status,
        Symbol("diagnostic-sample-count"),
        :complete,
    )
    outputs = NamedTuple{output_names}(
        (
            fill(NaN32, 2, 2),
            fill(typemax(UInt8), 1),
            fill(typemax(UInt64), 1),
            fill(true, 1),
        ),
    )
    availability = NamedTuple{output_names}((true, true, true, true))
    for _ in 1:64
        @test process!(outputs, graph, NamedTuple()) == availability
    end
    @test outputs.status == UInt8[0]
    @test getproperty(outputs, Symbol("diagnostic-sample-count")) == UInt64[64]
    @test outputs.complete == [false]
    @test getproperty(outputs, Symbol("interaction-matrix")) ≈
          Float32[1.5 -0.25; 0.4 2.0] atol = 2.0f-5
    @test @allocated(process!(outputs, graph, NamedTuple())) == 0
end
