@enum MatchedCalibrationGateFailure::UInt8 begin
    MatchedCalibrationGateRejected = 1
end

struct MatchedCalibrationGatePlan
    extent::Int
end

@algorithm MatchedCalibrationGateF32 begin
    label = "test-matched-calibration-gate-f32"
    failure = MatchedCalibrationGateFailure

    @configuration MatchedCalibrationGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "example/test-filtered-measurements/1",
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            "example/test-filtered-measurements/1";
            metadata_from=:input,
        ),
    )

    prepare(configuration) = MatchedCalibrationGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] < 0.0f0 && return MatchedCalibrationGateRejected
        copyto!(accepted, input)
        nothing
    end
end

const MATCHED_CALIBRATION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = difference
            label = matched-calibration-differencing-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                measurements_schema = "example/test-measurements/1"
                corrector_sample_schema =
                    "example/test-corrector-sample/1"
                filtered_measurements_schema =
                    "example/test-filtered-measurements/1"
                filtered_corrector_sample_schema =
                    "example/test-filtered-corrector-sample/1"
            }
        }
    ]
    links = []
    inputs = [
        "difference:measurements"
        "difference:corrector-sample"
    ]
    outputs = [
        "difference:filtered-measurements"
        "difference:filtered-corrector-sample"
        "difference:filtered"
    ]
}
"""

const MATCHED_CALIBRATION_GATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = difference
            label = matched-calibration-differencing-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                measurements_schema = "example/test-measurements/1"
                corrector_sample_schema =
                    "example/test-corrector-sample/1"
                filtered_measurements_schema =
                    "example/test-filtered-measurements/1"
                filtered_corrector_sample_schema =
                    "example/test-filtered-corrector-sample/1"
            }
        }
        {
            type = ndarray
            name = gate
            label = test-matched-calibration-gate-f32
            config = { extent = 2 }
        }
    ]
    links = [
        {
            output = "difference:filtered-measurements"
            input = "gate:input"
        }
    ]
    inputs = [
        "difference:measurements"
        "difference:corrector-sample"
    ]
    outputs = [
        "gate:accepted"
        "difference:filtered-corrector-sample"
        "difference:filtered"
    ]
}
"""

const MATCHED_CALIBRATION_INPUT_NAMES = (
    :measurements,
    Symbol("corrector-sample"),
)
const MATCHED_CALIBRATION_OUTPUT_NAMES = (
    Symbol("filtered-measurements"),
    Symbol("filtered-corrector-sample"),
    :filtered,
)
const MATCHED_CALIBRATION_GATE_OUTPUT_NAMES = (
    :accepted,
    Symbol("filtered-corrector-sample"),
    :filtered,
)

matched_graph_inputs(measurements, corrector_sample) = NamedTuple{
    MATCHED_CALIBRATION_INPUT_NAMES,
}((measurements, corrector_sample))

matched_graph_outputs(filtered_measurements, filtered_corrector, filtered) =
    NamedTuple{MATCHED_CALIBRATION_OUTPUT_NAMES}(
        (filtered_measurements, filtered_corrector, filtered),
    )

matched_gate_outputs(accepted, filtered_corrector, filtered) = NamedTuple{
    MATCHED_CALIBRATION_GATE_OUTPUT_NAMES,
}((accepted, filtered_corrector, filtered))

matched_graph_allocation_bytes(outputs, graph, inputs) =
    @allocated process!(outputs, graph, inputs)

function matched_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

function matched_graph_candidate_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_candidate_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end


@testset "matched differencing prepares as one stateful Graph Node" begin
    graph = prepare_graph_text(
        MATCHED_CALIBRATION_GRAPH_TEXT;
        algorithms=(MatchedCalibrationDifferencingF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:difference,)
    @test graph.node_failure_types == (MatchedCalibrationDifferencingFailure,)
    @test keys(graph.input_formats) == MATCHED_CALIBRATION_INPUT_NAMES
    @test keys(graph.output_formats) == MATCHED_CALIBRATION_OUTPUT_NAMES
    @test isempty(graph.parameter_formats)
    @test getproperty.(values(graph.input_formats), :shape) == ((2,), (2,))
    @test getproperty.(values(graph.output_formats), :shape) ==
          ((2,), (2,), (1,))
    @test getproperty(
        graph.output_formats,
        Symbol("filtered-measurements"),
    ).schema == "example/test-filtered-measurements/1"
    @test graph.output_formats.filtered.schema == MATCHED_CALIBRATION_FILTERED_V1

    filtered_measurements = fill(71.0f0, 2)
    filtered_corrector = fill(72.0f0, 2)
    filtered = fill(true, 1)
    outputs = matched_graph_outputs(
        filtered_measurements,
        filtered_corrector,
        filtered,
    )
    availability = NamedTuple{MATCHED_CALIBRATION_OUTPUT_NAMES}(
        (true, true, true),
    )
    first = matched_graph_inputs(Float32[10, 20], Float32[1, -2])
    @test process!(outputs, graph, first) == availability
    @test filtered_measurements == zeros(Float32, 2)
    @test filtered_corrector == zeros(Float32, 2)
    @test filtered == [false]

    second = matched_graph_inputs(Float32[13, 14], Float32[0.5, 4])
    @test process!(outputs, graph, second) == availability
    @test filtered_measurements == Float32[3, -6]
    @test filtered_corrector == Float32[-0.5, 6]
    @test filtered == [true]

    metadata = NamedTuple{MATCHED_CALIBRATION_INPUT_NAMES}(
        (
            SampleMetadata(103, 0; timestamp_nanoseconds=9013, terminal=true),
            SampleMetadata(103, 1; timestamp_nanoseconds=9017, terminal=true),
        ),
    )
    @test process!(outputs, graph, second, metadata) == availability
    expected_metadata = metadata.measurements
    @test all(
        value -> value == expected_metadata,
        values(FilterGraphs.output_metadata(graph)),
    )

    @test reset!(graph) === graph
    @test !matched_calibration_is_primed(only(graph.nodes).prepared.workspace)
    @test process!(outputs, graph, second) == availability
    @test filtered_measurements == zeros(Float32, 2)
    @test filtered == [false]
end

@testset "matched history advances after Node success" begin
    graph = prepare_graph_text(
        MATCHED_CALIBRATION_GRAPH_TEXT;
        algorithms=(MatchedCalibrationDifferencingF32,),
    )
    workspace = only(graph.nodes).prepared.workspace
    outputs = matched_graph_outputs(
        zeros(Float32, 2),
        zeros(Float32, 2),
        falses(1),
    )
    first_inputs = matched_graph_inputs(Float32[10, 20], Float32[1, -2])

    @test process!(outputs, graph, first_inputs) ==
          (var"filtered-measurements"=true, var"filtered-corrector-sample"=true, filtered=true)
    @test matched_calibration_is_primed(workspace)
    @test matched_calibration_previous_measurements(workspace) == Float32[10, 20]

    second_inputs = matched_graph_inputs(Float32[13, 14], Float32[0.5, 4])
    process!(outputs, graph, second_inputs)
    @test getproperty(outputs, Symbol("filtered-measurements")) == Float32[3, -6]
    @test matched_calibration_previous_measurements(workspace) == Float32[13, 14]
end

@testset "downstream rejection preserves outputs and advances earlier history" begin
    graph = prepare_graph_text(
        MATCHED_CALIBRATION_GATE_GRAPH_TEXT;
        algorithms=(MatchedCalibrationDifferencingF32, MatchedCalibrationGateF32),
    )
    workspace = first(graph.nodes).prepared.workspace
    accepted = fill(71.0f0, 2)
    filtered_corrector = fill(72.0f0, 2)
    filtered = fill(true, 1)
    outputs = matched_gate_outputs(accepted, filtered_corrector, filtered)

    first_inputs = matched_graph_inputs(Float32[10, 20], Float32[1, -2])
    @test process!(outputs, graph, first_inputs) == NamedTuple{
        MATCHED_CALIBRATION_GATE_OUTPUT_NAMES,
    }((true, true, true))
    @test accepted == zeros(Float32, 2)
    @test matched_calibration_previous_measurements(workspace) == Float32[10, 20]
    retained_outputs = map(copy, outputs)

    rejected = matched_graph_inputs(Float32[7, 14], Float32[0.25, 3])
    @test process!(outputs, graph, rejected) === MatchedCalibrationGateRejected
    @test outputs == retained_outputs
    @test matched_calibration_previous_measurements(workspace) == Float32[7, 14]
    @test matched_calibration_previous_corrector_sample(workspace) == Float32[0.25, 3]

    next_inputs = matched_graph_inputs(Float32[13, 14], Float32[0.5, 4])
    @test process!(outputs, graph, next_inputs) == NamedTuple{
        MATCHED_CALIBRATION_GATE_OUTPUT_NAMES,
    }((true, true, true))
    @test accepted == Float32[6, 0]
    @test filtered_corrector == Float32[0.25, 1]
    @test filtered == [true]
    @test matched_calibration_previous_measurements(workspace) == Float32[13, 14]
end

@testset "matched Graph paths allocate nothing after warmup" begin
    graph = prepare_graph_text(
        MATCHED_CALIBRATION_GRAPH_TEXT;
        algorithms=(MatchedCalibrationDifferencingF32,),
    )
    outputs = matched_graph_outputs(
        zeros(Float32, 2),
        zeros(Float32, 2),
        falses(1),
    )
    first_inputs = matched_graph_inputs(Float32[10, 20], Float32[1, -2])
    valid = matched_graph_inputs(Float32[13, 14], Float32[0.5, 4])
    large = matched_graph_inputs(
        Float32[floatmax(Float32), 14],
        Float32[0.5, 4],
    )
    rejected_inputs = (
        (
            matched_graph_inputs(Float32[NaN, 0], Float32[0, 0]),
            MatchedCalibrationNonFiniteMeasurements,
        ),
        (
            matched_graph_inputs(Float32[0, 0], Float32[0, Inf]),
            MatchedCalibrationNonFiniteCorrectorSample,
        ),
        (
            matched_graph_inputs(
                Float32[-floatmax(Float32), 14],
                Float32[0.5, 4],
            ),
            MatchedCalibrationNonFiniteDifference,
        ),
    )
    process!(outputs, graph, first_inputs)
    process!(outputs, graph, valid)
    @test matched_graph_allocation_bytes(outputs, graph, valid) == 0
    @test matched_graph_admitted_allocation_bytes(outputs, graph, valid) == 0
    process!(outputs, graph, large)

    for (inputs, failure) in rejected_inputs
        @test process!(outputs, graph, inputs) === failure
        @test matched_graph_allocation_bytes(outputs, graph, inputs) == 0
        @test matched_graph_admitted_allocation_bytes(outputs, graph, inputs) == 0
    end

    @test matched_graph_candidate_allocation_bytes(outputs, graph, valid) == 0

    gate_graph = prepare_graph_text(
        MATCHED_CALIBRATION_GATE_GRAPH_TEXT;
        algorithms=(MatchedCalibrationDifferencingF32, MatchedCalibrationGateF32),
    )
    gate_outputs = matched_gate_outputs(
        zeros(Float32, 2),
        zeros(Float32, 2),
        falses(1),
    )
    process!(gate_outputs, gate_graph, first_inputs)
    downstream_rejection =
        matched_graph_inputs(Float32[7, 14], Float32[0.25, 3])
    process!(gate_outputs, gate_graph, downstream_rejection)
    @test matched_graph_allocation_bytes(
        gate_outputs,
        gate_graph,
        downstream_rejection,
    ) == 0
end
