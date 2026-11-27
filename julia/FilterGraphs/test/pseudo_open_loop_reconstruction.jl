@enum PseudoOpenLoopGateFailure::UInt8 begin
    PseudoOpenLoopGateRejected = 1
end

struct PseudoOpenLoopGatePlan
    extent::Int
end

@algorithm PseudoOpenLoopGateF32 begin
    label = "test-pseudo-open-loop-gate-f32"
    failure = PseudoOpenLoopGateFailure

    @configuration PseudoOpenLoopGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            CONTROLLER_COMMAND_V1,
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            CONTROLLER_COMMAND_V1;
            metadata_from=:input,
        ),
    )

    prepare(configuration) = PseudoOpenLoopGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] >= 0.0f0 && return PseudoOpenLoopGateRejected
        copyto!(accepted, input)
        nothing
    end
end

const PSEUDO_OPEN_LOOP_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = pseudo-open-loop
            label = pseudo-open-loop-reconstruction-f32
            config = {
                measurement_count = 2
                corrector_shape_count = 1
                residual_error_count = 1
                initial_interaction_matrix = [ 2.0 1.0 ]
                initial_reconstructor = [ 0.5 1.0 ]
                initial_corrector_to_error = [ 2.0 ]
                residual_measurements_schema = "pipewireao/shack-hartmann-slopes-f32/1"
                corrector_shape_schema = "pipewireao/requested-pdm-command-f32/1"
                pseudo_open_loop_measurements_schema =
                    "pipewireao/shack-hartmann-slopes-f32/1"
                residual_error_schema = "pipewireao/controller-residual-error-f32/1"
            }
        }
    ]
    links = []
    inputs = [
        "pseudo-open-loop:residual-measurements"
        "pseudo-open-loop:corrector-shape"
        "pseudo-open-loop:interaction-matrix"
        "pseudo-open-loop:reconstructor"
        "pseudo-open-loop:corrector-to-error"
    ]
    outputs = [
        "pseudo-open-loop:pseudo-open-loop-measurements"
        "pseudo-open-loop:residual-error"
    ]
}
"""

const PSEUDO_OPEN_LOOP_CONTROL_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = pseudo-open-loop
            label = pseudo-open-loop-reconstruction-f32
            config = {
                measurement_count = 1
                corrector_shape_count = 1
                residual_error_count = 1
                initial_interaction_matrix = [ 1.0 ]
                initial_reconstructor = [ 1.0 ]
                initial_corrector_to_error = [ 1.0 ]
                residual_measurements_schema = "pipewireao/shack-hartmann-slopes-f32/1"
                corrector_shape_schema = "pipewireao/requested-pdm-command-f32/1"
                pseudo_open_loop_measurements_schema =
                    "pipewireao/shack-hartmann-slopes-f32/1"
                residual_error_schema = "pipewireao/controller-residual-error-f32/1"
            }
        }
        {
            type = ndarray
            name = correction
            label = closed-loop-correction-f32
            config = {
                state_length = 1
                hidden_mode_count = 1
            }
            props = {
                gain = -0.5
                pole = 1.0
                anti-windup-gain = 0.0
                hidden-mode-gain = 0.0
            }
        }
        {
            type = ndarray
            name = gate
            label = test-pseudo-open-loop-gate-f32
            config = { extent = 1 }
        }
    ]
    links = [
        {
            output = "pseudo-open-loop:residual-error"
            input = "correction:residual-error"
        }
        { output = "correction:correction" input = "gate:input" }
    ]
    inputs = [
        "pseudo-open-loop:residual-measurements"
        "pseudo-open-loop:corrector-shape"
        "pseudo-open-loop:interaction-matrix"
        "pseudo-open-loop:reconstructor"
        "pseudo-open-loop:corrector-to-error"
        "correction:constraint-feedback"
        "correction:shape-to-hidden"
        "correction:hidden-to-shape"
    ]
    outputs = [
        "pseudo-open-loop:pseudo-open-loop-measurements"
        "correction:controller-state"
        "gate:accepted"
    ]
}
"""

const PSEUDO_OPEN_LOOP_INPUT_NAMES = (
    Symbol("residual-measurements"),
    Symbol("corrector-shape"),
)
const PSEUDO_OPEN_LOOP_OUTPUT_NAMES = (
    Symbol("pseudo-open-loop-measurements"),
    Symbol("residual-error"),
)
const PSEUDO_OPEN_LOOP_CONTROL_INPUT_NAMES = (
    Symbol("residual-measurements"),
    Symbol("corrector-shape"),
    Symbol("constraint-feedback"),
)
const PSEUDO_OPEN_LOOP_CONTROL_OUTPUT_NAMES = (
    Symbol("pseudo-open-loop-measurements"),
    Symbol("controller-state"),
    :accepted,
)

pseudo_open_loop_inputs(residual_measurements, corrector_shape) =
    NamedTuple{PSEUDO_OPEN_LOOP_INPUT_NAMES}(
        (residual_measurements, corrector_shape),
    )
pseudo_open_loop_outputs(measurements, residual_error) =
    NamedTuple{PSEUDO_OPEN_LOOP_OUTPUT_NAMES}((measurements, residual_error))
pseudo_open_loop_control_inputs(
    residual_measurements,
    corrector_shape,
    constraint_feedback,
) = NamedTuple{PSEUDO_OPEN_LOOP_CONTROL_INPUT_NAMES}(
    (residual_measurements, corrector_shape, constraint_feedback),
)
pseudo_open_loop_control_outputs(measurements, state, accepted) =
    NamedTuple{PSEUDO_OPEN_LOOP_CONTROL_OUTPUT_NAMES}(
        (measurements, state, accepted),
    )

function pseudo_open_loop_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function pseudo_open_loop_graph_admitted_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    )
end

@testset "pseudo-open-loop reconstruction prepares as one Graph Node" begin
    graph = prepare_graph_text(
        PSEUDO_OPEN_LOOP_GRAPH_TEXT;
        algorithms=(PseudoOpenLoopReconstructionF32,),
    )
    @test map(node -> node.name, graph.nodes) == (Symbol("pseudo-open-loop"),)
    @test graph.node_failure_types == (PseudoOpenLoopReconstructionFailure,)
    @test keys(graph.input_formats) == PSEUDO_OPEN_LOOP_INPUT_NAMES
    @test keys(graph.output_formats) == PSEUDO_OPEN_LOOP_OUTPUT_NAMES
    @test keys(graph.parameter_formats) == (
        Symbol("interaction-matrix"),
        :reconstructor,
        Symbol("corrector-to-error"),
    )
    @test getproperty(
        graph.input_formats,
        Symbol("residual-measurements"),
    ).schema == SHACK_HARTMANN_SLOPES_V1
    @test getproperty(
        graph.input_formats,
        Symbol("corrector-shape"),
    ).schema == REQUESTED_PDM_COMMAND_V1
    @test getproperty(
        graph.output_formats,
        Symbol("residual-error"),
    ).schema == CONTROLLER_RESIDUAL_ERROR_V1
    @test getproperty(
        graph.parameter_formats,
        Symbol("interaction-matrix"),
    ).shape == (2, 1)
    @test graph.parameter_formats.reconstructor.shape == (1, 2)

    measurements = fill(71.0f0, 2)
    residual_error = fill(83.0f0, 1)
    outputs = pseudo_open_loop_outputs(measurements, residual_error)
    inputs = pseudo_open_loop_inputs(Float32[4, 3], Float32[1])
    @test process!(outputs, graph, inputs) == NamedTuple{
        PSEUDO_OPEN_LOOP_OUTPUT_NAMES,
    }((true, true))
    @test measurements == Float32[6, 4]
    @test residual_error == Float32[5]

    metadata = SampleMetadata(93, 0; timestamp_nanoseconds=9003, terminal=true)
    metadata_inputs = NamedTuple{PSEUDO_OPEN_LOOP_INPUT_NAMES}(
        (metadata, SampleMetadata(93, 1; terminal=true)),
    )
    @test process!(outputs, graph, inputs, metadata_inputs) == NamedTuple{
        PSEUDO_OPEN_LOOP_OUTPUT_NAMES,
    }((true, true))
    published = FilterGraphs.output_metadata(graph)
    @test getproperty(
        published,
        Symbol("pseudo-open-loop-measurements"),
    ) == metadata
    @test getproperty(published, Symbol("residual-error")) == metadata

    fill!(measurements, 97.0f0)
    fill!(residual_error, 98.0f0)
    retained_metadata = FilterGraphs.output_metadata(graph)
    rejected = pseudo_open_loop_inputs(Float32[NaN, 0], Float32[1])
    @test process!(outputs, graph, rejected) ===
          PseudoOpenLoopNonFiniteResidualMeasurements
    @test measurements == fill(97.0f0, 2)
    @test residual_error == fill(98.0f0, 1)
    @test FilterGraphs.output_metadata(graph) == retained_metadata
end

@testset "pseudo-open-loop Graph parameters replace atomically" begin
    graph = prepare_graph_text(
        PSEUDO_OPEN_LOOP_GRAPH_TEXT;
        algorithms=(PseudoOpenLoopReconstructionF32,),
    )
    original = only(graph.nodes).prepared.plan
    @test replace_parameters!(
        graph,
        Symbol("interaction-matrix") => zeros(Float32, 2, 1),
    ) === graph
    @test only(graph.nodes).prepared.plan !== original
    @test only(graph.nodes).prepared.plan.reconstructor ===
          original.reconstructor
    @test only(graph.nodes).prepared.plan.corrector_to_error ===
          original.corrector_to_error

    measurements = zeros(Float32, 2)
    residual_error = zeros(Float32, 1)
    outputs = pseudo_open_loop_outputs(measurements, residual_error)
    inputs = pseudo_open_loop_inputs(Float32[4, 3], Float32[1])
    @test process!(outputs, graph, inputs) == NamedTuple{
        PSEUDO_OPEN_LOOP_OUTPUT_NAMES,
    }((true, true))
    @test measurements == Float32[4, 3]
    @test residual_error == Float32[3]

    active = only(graph.nodes).prepared.plan
    @test_throws ArgumentError replace_parameters!(
        graph,
        :reconstructor => Float32[NaN 0],
    )
    @test only(graph.nodes).prepared.plan === active
    @test process!(outputs, graph, inputs) == NamedTuple{
        PSEUDO_OPEN_LOOP_OUTPUT_NAMES,
    }((true, true))
    @test residual_error == Float32[3]
end

@testset "pseudo-open-loop and closed-loop correction compose as Nodes" begin
    graph = prepare_graph_text(
        PSEUDO_OPEN_LOOP_CONTROL_GRAPH_TEXT;
        algorithms=(
            PseudoOpenLoopReconstructionF32,
            ClosedLoopCorrectionF32,
            PseudoOpenLoopGateF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) == (
        Symbol("pseudo-open-loop"),
        :correction,
        :gate,
    )
    @test graph.node_failure_types == (
        PseudoOpenLoopReconstructionFailure,
        ClosedLoopCorrectionFailure,
        PseudoOpenLoopGateFailure,
    )

    measurements = Float32[71]
    controller_state = Float32[72]
    accepted = Float32[73]
    outputs = pseudo_open_loop_control_outputs(
        measurements,
        controller_state,
        accepted,
    )
    success = pseudo_open_loop_control_inputs(
        Float32[2],
        Float32[3],
        Float32[99],
    )
    @test process!(outputs, graph, success) == NamedTuple{
        PSEUDO_OPEN_LOOP_CONTROL_OUTPUT_NAMES,
    }((true, true, true))
    @test measurements == Float32[5]
    @test controller_state == Float32[0]
    @test accepted == Float32[-1]

    workspace = graph.nodes[2].prepared.workspace
    @test closed_loop_correction_state(workspace) == Float32[0]
    retained_outputs = (copy(measurements), copy(controller_state), copy(accepted))
    downstream_rejection = pseudo_open_loop_control_inputs(
        Float32[-2],
        Float32[-3],
        Float32[0],
    )
    @test process!(outputs, graph, downstream_rejection) ===
          PseudoOpenLoopGateRejected
    @test (measurements, controller_state, accepted) == retained_outputs
    @test closed_loop_correction_state(workspace) == Float32[-1]
end

@testset "pseudo-open-loop Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        PSEUDO_OPEN_LOOP_GRAPH_TEXT;
        algorithms=(PseudoOpenLoopReconstructionF32,),
    )
    outputs = pseudo_open_loop_outputs(zeros(Float32, 2), zeros(Float32, 1))
    success = pseudo_open_loop_inputs(Float32[4, 3], Float32[1])
    rejection = pseudo_open_loop_inputs(Float32[NaN, 0], Float32[1])
    process!(outputs, graph, success)
    @test pseudo_open_loop_graph_allocation_bytes(outputs, graph, success) == 0
    process!(outputs, graph, rejection)
    @test pseudo_open_loop_graph_allocation_bytes(outputs, graph, rejection) == 0
    FilterGraphs._process_admitted!(
        outputs,
        graph,
        success,
        graph.default_input_metadata,
    )
    @test pseudo_open_loop_graph_admitted_allocation_bytes(
        outputs,
        graph,
        success,
    ) == 0

    chain = prepare_graph_text(
        PSEUDO_OPEN_LOOP_CONTROL_GRAPH_TEXT;
        algorithms=(
            PseudoOpenLoopReconstructionF32,
            ClosedLoopCorrectionF32,
            PseudoOpenLoopGateF32,
        ),
    )
    chain_outputs = pseudo_open_loop_control_outputs(
        zeros(Float32, 1),
        zeros(Float32, 1),
        zeros(Float32, 1),
    )
    chain_success = pseudo_open_loop_control_inputs(
        Float32[2],
        Float32[3],
        Float32[99],
    )
    process!(chain_outputs, chain, chain_success)
    downstream_rejection = pseudo_open_loop_control_inputs(
        Float32[-2],
        Float32[-3],
        Float32[0],
    )
    @test pseudo_open_loop_graph_allocation_bytes(
        chain_outputs,
        chain,
        downstream_rejection,
    ) == 0
end
