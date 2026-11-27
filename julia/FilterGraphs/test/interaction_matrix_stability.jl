const INTERACTION_MATRIX_STABILITY_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = stability
            label = interaction-matrix-stability-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                minimum_column_norm = 0.1
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
        }
    ]
    links = []
    inputs = [
        "stability:first-window"
        "stability:second-window"
    ]
    outputs = [
        "stability:status"
        "stability:status-coordinate"
        "stability:minimum-column-norms"
        "stability:maximum-relative-column-difference"
        "stability:maximum-relative-column-difference-coordinate"
        "stability:relative-frobenius-difference"
    ]
}
"""

const INTERACTION_MATRIX_STABILITY_FROBENIUS_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = stability
            label = interaction-matrix-stability-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                minimum_column_norm = 0.1
                interaction_matrix_schema =
                    "example/test-interaction-matrix/1"
            }
            props = {
                maximum-relative-column-difference = 1.0
                maximum-relative-frobenius-difference = 0.5
            }
        }
    ]
    links = []
    inputs = [
        "stability:first-window"
        "stability:second-window"
    ]
    outputs = [
        "stability:status"
        "stability:status-coordinate"
        "stability:minimum-column-norms"
        "stability:maximum-relative-column-difference"
        "stability:maximum-relative-column-difference-coordinate"
        "stability:relative-frobenius-difference"
    ]
}
"""

const INTERACTION_MATRIX_STABILITY_INPUT_NAMES = (
    Symbol("first-window"),
    Symbol("second-window"),
)
const INTERACTION_MATRIX_STABILITY_OUTPUT_NAMES = (
    :status,
    Symbol("status-coordinate"),
    Symbol("minimum-column-norms"),
    Symbol("maximum-relative-column-difference"),
    Symbol("maximum-relative-column-difference-coordinate"),
    Symbol("relative-frobenius-difference"),
)

interaction_matrix_stability_inputs(first_window, second_window) = NamedTuple{
    INTERACTION_MATRIX_STABILITY_INPUT_NAMES,
}((first_window, second_window))

function interaction_matrix_stability_outputs(
    status,
    status_coordinate,
    minimum_column_norms,
    maximum_relative_column_difference,
    maximum_relative_column_difference_coordinate,
    relative_frobenius_difference,
)
    return NamedTuple{INTERACTION_MATRIX_STABILITY_OUTPUT_NAMES}(
        (
            status,
            status_coordinate,
            minimum_column_norms,
            maximum_relative_column_difference,
            maximum_relative_column_difference_coordinate,
            relative_frobenius_difference,
        ),
    )
end

function interaction_matrix_stability_graph_outputs()
    return interaction_matrix_stability_outputs(
        fill(typemax(UInt8), 1),
        fill(typemax(UInt64), 1),
        fill(71.0f0, 2),
        fill(72.0f0, 1),
        fill(typemax(UInt64), 1),
        fill(73.0f0, 1),
    )
end

function interaction_matrix_stability_graph_allocation_bytes(
    outputs,
    graph,
    inputs,
)
    return @allocated process!(outputs, graph, inputs)
end

function interaction_matrix_stability_graph_admitted_allocation_bytes(
    outputs,
    graph,
    inputs,
)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "interaction-matrix stability prepares as one Graph Node" begin
    graph = prepare_graph_text(
        INTERACTION_MATRIX_STABILITY_GRAPH_TEXT;
        algorithms=(InteractionMatrixStabilityF32,),
    )
    @test map(node -> node.name, graph.nodes) == (:stability,)
    @test graph.node_failure_types == (InteractionMatrixStabilityFailure,)
    @test keys(graph.input_formats) == INTERACTION_MATRIX_STABILITY_INPUT_NAMES
    @test keys(graph.output_formats) == INTERACTION_MATRIX_STABILITY_OUTPUT_NAMES
    @test isempty(graph.parameter_formats)
    @test getproperty.(values(graph.input_formats), :shape) == ((2, 2), (2, 2))
    @test all(
        format -> format.schema == "example/test-interaction-matrix/1",
        values(graph.input_formats),
    )
    @test getproperty.(values(graph.output_formats), :schema) == (
        INTERACTION_MATRIX_STABILITY_STATUS_V1,
        CORRECTOR_COORDINATE_V1,
        INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
        INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
        CORRECTOR_COORDINATE_V1,
        INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1,
    )
    @test property_values(only(graph.nodes).prepared.plan) == NamedTuple{
        (
            Symbol("maximum-relative-column-difference"),
            Symbol("maximum-relative-frobenius-difference"),
        ),
    }((0.5f0, 0.5f0))

    outputs = interaction_matrix_stability_graph_outputs()
    inputs = interaction_matrix_stability_inputs(
        Float32[1 0; 0 1],
        Float32[2 0; 0 2],
    )
    @test process!(outputs, graph, inputs) == NamedTuple{
        INTERACTION_MATRIX_STABILITY_OUTPUT_NAMES,
    }((true, true, true, true, true, true))
    @test outputs.status == UInt8[2]
    @test getproperty(outputs, Symbol("status-coordinate")) == UInt64[0]
    @test getproperty(outputs, Symbol("minimum-column-norms")) == Float32[1, 2]
    @test reinterpret(
        UInt32,
        getproperty(outputs, Symbol("maximum-relative-column-difference")),
    ) == UInt32[0x3f2aaaab]
    @test getproperty(
        outputs,
        Symbol("maximum-relative-column-difference-coordinate"),
    ) == UInt64[0]
    @test reinterpret(
        UInt32,
        getproperty(outputs, Symbol("relative-frobenius-difference")),
    ) == UInt32[0x3f2aaaab]

    first_metadata = SampleMetadata(97, 0; timestamp_nanoseconds=9007, terminal=true)
    second_metadata = SampleMetadata(97, 1; timestamp_nanoseconds=9011, terminal=true)
    metadata = NamedTuple{INTERACTION_MATRIX_STABILITY_INPUT_NAMES}(
        (first_metadata, second_metadata),
    )
    @test process!(outputs, graph, inputs, metadata) == NamedTuple{
        INTERACTION_MATRIX_STABILITY_OUTPUT_NAMES,
    }((true, true, true, true, true, true))
    @test all(
        value -> value == second_metadata,
        values(FilterGraphs.output_metadata(graph)),
    )
end

@testset "Graph preparation applies interaction-matrix stability Properties" begin
    graph = prepare_graph_text(
        INTERACTION_MATRIX_STABILITY_FROBENIUS_GRAPH_TEXT;
        algorithms=(InteractionMatrixStabilityF32,),
    )
    properties = property_values(only(graph.nodes).prepared.plan)
    @test getproperty(
        properties,
        Symbol("maximum-relative-column-difference"),
    ) === 1.0f0
    @test getproperty(
        properties,
        Symbol("maximum-relative-frobenius-difference"),
    ) === 0.5f0

    outputs = interaction_matrix_stability_graph_outputs()
    inputs = interaction_matrix_stability_inputs(
        Float32[1 0; 0 1],
        Float32[2 0; 0 2],
    )
    @test process!(outputs, graph, inputs) isa NamedTuple
    @test outputs.status == UInt8[3]
    @test getproperty(outputs, Symbol("status-coordinate")) == UInt64[0]
end

@testset "stability rejection withholds every Graph output and metadata" begin
    graph = prepare_graph_text(
        INTERACTION_MATRIX_STABILITY_GRAPH_TEXT;
        algorithms=(InteractionMatrixStabilityF32,),
    )
    outputs = interaction_matrix_stability_graph_outputs()
    valid = interaction_matrix_stability_inputs(
        Float32[1 0; 0 1],
        Float32[2 0; 0 2],
    )
    metadata = NamedTuple{INTERACTION_MATRIX_STABILITY_INPUT_NAMES}(
        (
            SampleMetadata(101, 0; terminal=true),
            SampleMetadata(101, 1; terminal=true),
        ),
    )
    process!(outputs, graph, valid, metadata)
    retained_outputs = map(copy, outputs)
    retained_metadata = FilterGraphs.output_metadata(graph)

    rejected_inputs = (
        (
            interaction_matrix_stability_inputs(
                Float32[NaN 0; 0 1],
                Float32[2 0; 0 2],
            ),
            InteractionMatrixStabilityNonFiniteFirstWindow,
        ),
        (
            interaction_matrix_stability_inputs(
                Float32[1 0; 0 1],
                Float32[2 0; 0 Inf],
            ),
            InteractionMatrixStabilityNonFiniteSecondWindow,
        ),
        (
            interaction_matrix_stability_inputs(
                Float32[floatmax(Float32) 0; 0 1],
                Float32[-floatmax(Float32) 0; 0 2],
            ),
            InteractionMatrixStabilityNonFiniteCalculation,
        ),
    )
    for (inputs, failure) in rejected_inputs
        @test process!(outputs, graph, inputs) === failure
        @test outputs == retained_outputs
        @test FilterGraphs.output_metadata(graph) == retained_metadata
    end
end

@testset "stability Graph paths allocate nothing after warmup" begin
    graph = prepare_graph_text(
        INTERACTION_MATRIX_STABILITY_GRAPH_TEXT;
        algorithms=(InteractionMatrixStabilityF32,),
    )
    outputs = interaction_matrix_stability_graph_outputs()
    valid = interaction_matrix_stability_inputs(
        Float32[1 0; 0 1],
        Float32[2 0; 0 2],
    )
    rejected_inputs = (
        interaction_matrix_stability_inputs(
            Float32[NaN 0; 0 1],
            Float32[2 0; 0 2],
        ),
        interaction_matrix_stability_inputs(
            Float32[1 0; 0 1],
            Float32[2 0; 0 Inf],
        ),
        interaction_matrix_stability_inputs(
            Float32[floatmax(Float32) 0; 0 1],
            Float32[-floatmax(Float32) 0; 0 2],
        ),
    )

    process!(outputs, graph, valid)
    @test interaction_matrix_stability_graph_allocation_bytes(
        outputs,
        graph,
        valid,
    ) == 0
    FilterGraphs._process_admitted!(
        outputs,
        graph,
        valid,
        graph.default_input_metadata,
    )
    @test interaction_matrix_stability_graph_admitted_allocation_bytes(
        outputs,
        graph,
        valid,
    ) == 0

    for inputs in rejected_inputs
        process!(outputs, graph, inputs)
        @test interaction_matrix_stability_graph_allocation_bytes(
            outputs,
            graph,
            inputs,
        ) == 0
        FilterGraphs._process_admitted!(
            outputs,
            graph,
            inputs,
            graph.default_input_metadata,
        )
        @test interaction_matrix_stability_graph_admitted_allocation_bytes(
            outputs,
            graph,
            inputs,
        ) == 0
    end
end
