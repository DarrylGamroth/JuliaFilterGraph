const SCC_CONTROLLER_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = residual
            label = scc-pixel-residual-f32
            config = {
                pixel_count = 6
                mask = [ true false true true false true ]
                initial_reference_image = null
                minimum_valid_fraction = 0.5
            }
        }
        {
            type = ndarray
            name = reconstruct
            label = scc-reconstructor-f32
            config = {
                controller_coordinate_count = 2
                selected_pixel_count = 4
                initial_reconstructor = null
            }
        }
        {
            type = ndarray
            name = law
            label = leaky-integrator-f32
            config = {
                extent = 2
                initial_state = 0.0
                input_schema = pipewireao/controller-residual-error-f32/1
                output_schema = pipewireao/controller-command-f32/1
            }
            props = { gain = -0.5 pole = 1.0 }
        }
        {
            type = ndarray
            name = project
            label = controller-to-vdm-f32
            config = {
                controller_command_count = 2
                active_vdm_count = 2
            }
        }
    ]
    links = [
        {
            output = "residual:selected-residuals"
            input = "reconstruct:selected-residuals"
        }
        {
            output = "reconstruct:controller-residual-error"
            input = "law:input"
        }
        {
            output = "law:output"
            input = "project:controller-command"
        }
    ]
    inputs = [
        "residual:difference"
        "residual:difference-validity"
        "residual:pair-status"
        "residual:reference-image"
        "residual:mask"
        "reconstruct:reconstructor"
        "project:controller-to-vdm"
    ]
    outputs = [
        "project:vdm-command"
        "residual:selected-validity"
        "residual:valid-fraction"
    ]
}
"""

const SCC_CONTROLLER_INPUT_NAMES = (
    :difference,
    Symbol("difference-validity"),
    Symbol("pair-status"),
)
const SCC_CONTROLLER_OUTPUT_NAMES = (
    Symbol("vdm-command"),
    Symbol("selected-validity"),
    Symbol("valid-fraction"),
)

function scc_controller_inputs(difference, validity, pair_status)
    return NamedTuple{SCC_CONTROLLER_INPUT_NAMES}(
        (difference, validity, pair_status),
    )
end

function scc_controller_outputs(command, validity, valid_fraction)
    return NamedTuple{SCC_CONTROLLER_OUTPUT_NAMES}(
        (command, validity, valid_fraction),
    )
end

function prepare_scc_controller_graph()
    graph = prepare_graph_text(
        SCC_CONTROLLER_GRAPH;
        algorithms=(
            SccPixelResidualF32,
            SccReconstructorF32,
            LeakyIntegratorF32,
            ControllerToVdmF32,
        ),
    )
    replace_parameters!(
        graph,
        "reference-image" => Float32[1, 10, -1, 2, 0, 4],
        :mask => Bool[true, false, true, true, false, true],
        :reconstructor => Float32[1 2 3 4; -1 0.5 0 2],
        "controller-to-vdm" => Float32[1 2; 0.5 -1],
    )
    return graph
end

function scc_controller_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "simplified SCC graph preserves controller coordinates" begin
    graph = prepare_scc_controller_graph()
    @test map(node -> node.name, graph.nodes) == (
        :residual,
        :reconstruct,
        :law,
        :project,
    )
    @test keys(graph.input_formats) == SCC_CONTROLLER_INPUT_NAMES
    @test keys(graph.output_formats) == SCC_CONTROLLER_OUTPUT_NAMES
    @test keys(graph.parameter_formats) == (
        Symbol("reference-image"),
        :mask,
        :reconstructor,
        Symbol("controller-to-vdm"),
    )
    @test graph.input_formats.difference.schema ==
          SCC_CHOPPED_IMAGE_DIFFERENCE_V1
    @test graph.output_formats.var"vdm-command".schema == VDM_COMMAND_V1
    @test graph.output_formats.var"selected-validity".schema ==
          SCC_SELECTED_PIXEL_VALIDITY_V1

    command = fill(71.0f0, 2)
    selected_validity = falses(4)
    valid_fraction = zeros(Float32, 1)
    outputs = scc_controller_outputs(
        command,
        selected_validity,
        valid_fraction,
    )
    inputs = scc_controller_inputs(
        Float32[3, 100, 4, 4, 0, 1],
        Bool[true, true, true, true, true, false],
        UInt8[UInt8(ChoppedImageComplete)],
    )
    metadata = NamedTuple{SCC_CONTROLLER_INPUT_NAMES}(
        (
            SampleMetadata(41, 0; timestamp_nanoseconds=501),
            nothing,
            nothing,
        ),
    )

    @test process!(outputs, graph, inputs, metadata) == NamedTuple{
        SCC_CONTROLLER_OUTPUT_NAMES,
    }((true, true, true))
    @test selected_validity == Bool[true, true, true, false]
    @test valid_fraction == Float32[0.75]
    @test graph.node_outputs[1][1] == Float32[2, 5, 2, 0]
    @test graph.node_outputs[2][1] == Float32[18, 0.5]
    @test leaky_integrator_state(graph.nodes[3].prepared.workspace) ==
          Float32[-9, -0.25]
    @test command == Float32[-9.5, -4.25]
    @test all(
        value -> value == metadata.difference,
        values(FilterGraphs.output_metadata(graph)),
    )

    selected = graph.node_outputs[1][1]
    reconstructor = graph.nodes[2].prepared.plan.matrix
    projection = graph.nodes[4].prepared.plan.controller_to_vdm
    @test command == projection * (-0.5f0 .* (reconstructor * selected))
end

@testset "unusable SCC intervals retain controller state and publication" begin
    graph = prepare_scc_controller_graph()
    command = zeros(Float32, 2)
    selected_validity = falses(4)
    valid_fraction = zeros(Float32, 1)
    outputs = scc_controller_outputs(
        command,
        selected_validity,
        valid_fraction,
    )
    valid = scc_controller_inputs(
        Float32[3, 100, 4, 4, 0, 1],
        Bool[true, true, true, true, true, false],
        UInt8[UInt8(ChoppedImageComplete)],
    )
    @test process!(outputs, graph, valid) == NamedTuple{
        SCC_CONTROLLER_OUTPUT_NAMES,
    }((true, true, true))
    retained_outputs = map(copy, outputs)
    retained_state = copy(
        leaky_integrator_state(graph.nodes[3].prepared.workspace),
    )

    priming = scc_controller_inputs(
        valid.difference,
        valid.var"difference-validity",
        UInt8[UInt8(ChoppedImagePriming)],
    )
    @test process!(outputs, graph, priming) ===
          SccPixelResidualIncompletePair
    @test outputs == retained_outputs
    @test leaky_integrator_state(graph.nodes[3].prepared.workspace) ==
          retained_state

    insufficient = scc_controller_inputs(
        valid.difference,
        Bool[true, true, false, false, true, false],
        UInt8[UInt8(ChoppedImageComplete)],
    )
    @test process!(outputs, graph, insufficient) ===
          SccPixelResidualInsufficientValidPixels
    @test outputs == retained_outputs
    @test leaky_integrator_state(graph.nodes[3].prepared.workspace) ==
          retained_state

    process!(outputs, graph, valid)
    @test scc_controller_allocation_bytes(outputs, graph, valid) == 0
    @test scc_controller_allocation_bytes(outputs, graph, priming) == 0
    @test scc_controller_allocation_bytes(outputs, graph, insufficient) == 0
end
