const CLOSED_LOOP_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "closed-loop-correction.conf")
const CLOSED_LOOP_NUMERICAL_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "closed_loop_correction.toml"),
)
const CLOSED_LOOP_INPUT_NAMES = (
    Symbol("residual-error"),
    Symbol("constraint-feedback"),
)
const CLOSED_LOOP_OUTPUT_NAMES = (:correction, Symbol("controller-state"))

closed_loop_inputs(residual_error, constraint_feedback) =
    NamedTuple{CLOSED_LOOP_INPUT_NAMES}((residual_error, constraint_feedback))
closed_loop_outputs(correction, controller_state) =
    NamedTuple{CLOSED_LOOP_OUTPUT_NAMES}((correction, controller_state))

function closed_loop_row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepare_closed_loop_graph()
    graph = prepare_graph(
        CLOSED_LOOP_GRAPH_FIXTURE;
        algorithms=algorithms(),
    )
    configuration = CLOSED_LOOP_NUMERICAL_FIXTURE["configuration"]
    parameters = CLOSED_LOOP_NUMERICAL_FIXTURE["parameters"]
    replace_parameters!(
        graph,
        "shape-to-hidden" => closed_loop_row_major_matrix(
            Float32,
            parameters["shape_to_hidden"],
            configuration["hidden_mode_count"],
            configuration["state_length"],
        ),
        "hidden-to-shape" => closed_loop_row_major_matrix(
            Float32,
            parameters["hidden_to_shape"],
            configuration["state_length"],
            configuration["hidden_mode_count"],
        ),
    )
    return graph
end

function closed_loop_candidate_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_candidate_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end


@testset "closed-loop Graph exposes the FGN boundary" begin
    graph = prepare_closed_loop_graph()
    @test map(node -> node.name, graph.nodes) == (Symbol("closed-loop-correction"),)
    @test keys(graph.input_formats) == CLOSED_LOOP_INPUT_NAMES
    @test keys(graph.parameter_formats) == (
        Symbol("shape-to-hidden"),
        Symbol("hidden-to-shape"),
    )
    @test keys(graph.output_formats) == CLOSED_LOOP_OUTPUT_NAMES
    @test getproperty(
        graph.input_formats,
        Symbol("constraint-feedback"),
    ).schema == CONTROLLER_CONSTRAINT_FEEDBACK_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("shape-to-hidden"),
    ).shape == (1, 2)
    @test getproperty(
        graph.output_formats,
        Symbol("controller-state"),
    ).schema == CONTROLLER_STATE_V1

    properties = property_values(only(graph.nodes).prepared.plan)
    @test properties.gain === 0.5f0
    @test properties.pole === 0.75f0
    @test getproperty(properties, Symbol("anti-windup-gain")) === 0.75f0
    @test getproperty(properties, Symbol("hidden-mode-gain")) === 0.5f0

    correction = fill(-1.0f0, 2)
    controller_state = fill(-2.0f0, 2)
    outputs = closed_loop_outputs(correction, controller_state)
    for frame in CLOSED_LOOP_NUMERICAL_FIXTURE["frames"]
        availability = process!(
            outputs,
            graph,
            closed_loop_inputs(
                Float32.(frame["residual_error"]),
                Float32.(frame["constraint_feedback"]),
            ),
        )
        @test availability == (correction=true, var"controller-state"=true)
        @test correction == Float32.(frame["correction"])
        @test controller_state == Float32.(frame["controller_state"])
    end
end

@testset "host feedback follows publication and Node state follows local outcomes" begin
    graph = prepare_closed_loop_graph()
    workspace = only(graph.nodes).prepared.workspace
    frames = CLOSED_LOOP_NUMERICAL_FIXTURE["frames"]
    correction = fill(-1.0f0, 2)
    controller_state = fill(-2.0f0, 2)
    outputs = closed_loop_outputs(correction, controller_state)

    first = frames[1]
    host_feedback = Float32.(first["constraint_feedback"])
    candidate = process!(
        outputs,
        graph,
        closed_loop_inputs(Float32.(first["residual_error"]), host_feedback),
    )
    @test candidate == (correction=true, var"controller-state"=true)
    @test correction == Float32.(first["correction"])
    @test closed_loop_correction_state(workspace) ==
          Float32.(first["controller_state"])
    host_feedback .= Float32.(frames[2]["constraint_feedback"])

    second = frames[2]
    candidate = process!(
        outputs,
        graph,
        closed_loop_inputs(Float32.(second["residual_error"]), host_feedback),
    )
    @test candidate == (correction=true, var"controller-state"=true)
    @test correction == Float32.(second["correction"])
    @test closed_loop_correction_state(workspace) ==
          Float32.(second["controller_state"])
    @test host_feedback == Float32.(second["constraint_feedback"])

    host_feedback .= Float32.(frames[3]["constraint_feedback"])

    fill!(correction, 71.0f0)
    fill!(controller_state, 72.0f0)
    rejected = process!(
        outputs,
        graph,
        closed_loop_inputs(Float32.(frames[3]["residual_error"]), Float32[NaN, 0]),
    )
    @test rejected === ClosedLoopCorrectionNonFiniteFeedback
    @test correction == fill(71.0f0, 2)
    @test controller_state == fill(72.0f0, 2)
    @test closed_loop_correction_state(workspace) ==
          Float32.(second["controller_state"])
    @test host_feedback == Float32.(frames[3]["constraint_feedback"])

    third = frames[3]
    candidate = process!(
        outputs,
        graph,
        closed_loop_inputs(Float32.(third["residual_error"]), host_feedback),
    )
    @test candidate == (correction=true, var"controller-state"=true)
    @test correction == Float32.(third["correction"])
    @test closed_loop_correction_state(workspace) ==
          Float32.(third["controller_state"])
end

@testset "closed-loop Graph publication paths allocate no heap after warmup" begin
    graph = prepare_closed_loop_graph()
    frames = CLOSED_LOOP_NUMERICAL_FIXTURE["frames"]
    correction = zeros(Float32, 2)
    controller_state = zeros(Float32, 2)
    outputs = closed_loop_outputs(correction, controller_state)

    first_inputs = closed_loop_inputs(
        Float32.(frames[1]["residual_error"]),
        Float32.(frames[1]["constraint_feedback"]),
    )
    process!(outputs, graph, first_inputs)

    second_inputs = closed_loop_inputs(
        Float32.(frames[2]["residual_error"]),
        Float32.(frames[2]["constraint_feedback"]),
    )
    @test closed_loop_candidate_allocation_bytes(outputs, graph, second_inputs) == 0

    third_inputs = closed_loop_inputs(
        Float32.(frames[3]["residual_error"]),
        Float32.(frames[3]["constraint_feedback"]),
    )
    rejected_inputs = closed_loop_inputs(Float32[0, 0], Float32[NaN, 0])
    @test closed_loop_candidate_allocation_bytes(outputs, graph, rejected_inputs) == 0
end

@testset "open correction publishes held values with each frame metadata" begin
    graph = prepare_closed_loop_graph()
    workspace = only(graph.nodes).prepared.workspace
    outputs = closed_loop_outputs(zeros(Float32, 2), zeros(Float32, 2))
    inputs = closed_loop_inputs(Float32[1, -2], zeros(Float32, 2))
    @test process!(outputs, graph, inputs) == (correction=true, var"controller-state"=true)
    @test process!(outputs, graph, inputs) == (correction=true, var"controller-state"=true)
    held = map(copy, outputs)
    @test replace_properties!(graph, "closed-loop-correction:closed" => false) === graph
    @test !only(graph.nodes).prepared.plan.closed
    @test workspace.committed_state == held.var"controller-state"
    inputs.var"residual-error" .= floatmax(Float32)
    inputs.var"constraint-feedback" .= floatmax(Float32)
    for sequence in 1:3
        metadata = SampleMetadata(sequence; timestamp_nanoseconds=100sequence, terminal=true)
        @test process!(outputs, graph, inputs,
            closed_loop_inputs(metadata, metadata)) == (correction=true, var"controller-state"=true)
        @test outputs == held
        @test all(==(metadata), values(FilterGraphs.output_metadata(graph)))
    end
    @test_throws PropertyTypeError replace_properties!(graph, "closed-loop-correction:closed" => 1)
    @test !only(graph.nodes).prepared.plan.closed
    inputs.var"residual-error" .= Float32[0.1, -0.2]
    inputs.var"constraint-feedback" .= 0
    reference = prepare_closed_loop_graph()
    process!(closed_loop_outputs(zeros(Float32, 2), zeros(Float32, 2)), reference,
        closed_loop_inputs(Float32[1, -2], zeros(Float32, 2)))
    process!(closed_loop_outputs(zeros(Float32, 2), zeros(Float32, 2)), reference,
        closed_loop_inputs(Float32[1, -2], zeros(Float32, 2)))
    expected = map(similar, outputs)
    @test replace_properties!(graph, "closed-loop-correction:closed" => true) === graph
    @test process!(expected, reference, inputs) == process!(outputs, graph, inputs)
    @test outputs == expected
    replace_properties!(graph, "closed-loop-correction:closed" => false)
    reset!(graph)
    @test process!(outputs, graph, inputs) == (correction=true, var"controller-state"=true)
    @test all(iszero, outputs.correction)
    @test all(iszero, outputs.var"controller-state")
end
