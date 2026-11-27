const SCC_SPATIAL_WEIGHTED_CONTROLLER_CONFIGURATION = normpath(
    joinpath(@__DIR__, "data", "examples", "scc-spatial-weighted-controller/graph.conf"),
)

const SCC_SPATIAL_WEIGHTED_INPUT_NAMES = (
    :difference,
    Symbol("difference-validity"),
    Symbol("pair-status"),
)
const SCC_SPATIAL_WEIGHTED_OUTPUT_NAMES = (
    Symbol("vdm-command"),
    Symbol("measurement-validity"),
    Symbol("valid-fraction"),
)

function prepare_scc_spatial_weighted_controller_graph()
    graph = prepare_graph(
        SCC_SPATIAL_WEIGHTED_CONTROLLER_CONFIGURATION;
        algorithms=(
            SccSpatialCarrierDemodulationF32,
            SccWeightedRegularizedReconstructorF32,
            LeakyIntegratorF32,
            ControllerToVdmF32,
        ),
    )
    interaction = Float32[
        1 0
        0 0
        1 0
        0 0
        0 1
        0 0
    ]
    replace_parameters!(
        graph,
        "reference-difference" => ones(Float32, 12),
        "static-validity" => trues(12),
        "carrier-phase" => Float32[0, 0, 0, 0, pi / 2, 0, 0, 0, 0, 0, 0, 0],
        :fir => Float32[0, 1, 0],
        :roi => Bool[
            true,
            false,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        ],
        "interaction-matrix" => interaction,
        "regularization-operator" => Float32[1 0; 0 1],
        "controller-to-vdm" => Float32[1 0; 0 1],
    )
    return graph
end

function scc_spatial_weighted_inputs(difference, validity, pair_status)
    return NamedTuple{SCC_SPATIAL_WEIGHTED_INPUT_NAMES}(
        (difference, validity, pair_status),
    )
end

function scc_spatial_weighted_outputs(command, validity, valid_fraction)
    return NamedTuple{SCC_SPATIAL_WEIGHTED_OUTPUT_NAMES}(
        (command, validity, valid_fraction),
    )
end

function scc_spatial_weighted_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "9.2 and 9.5 compose without delivery-specific science" begin
    graph = prepare_scc_spatial_weighted_controller_graph()
    @test map(node -> node.name, graph.nodes) == (
        :demodulate,
        :solve,
        :law,
        :project,
    )
    @test algorithm_operation(SccSpatialCarrierDemodulationF32) isa
          SccSpatialCarrierDemodulationF32Operation
    @test algorithm_operation(SccWeightedRegularizedReconstructorF32) isa
          SccWeightedRegularizedReconstructionF32Operation

    command = fill(71.0f0, 2)
    measurement_validity = falses(6)
    valid_fraction = zeros(Float32, 1)
    outputs = scc_spatial_weighted_outputs(
        command,
        measurement_validity,
        valid_fraction,
    )
    inputs = scc_spatial_weighted_inputs(
        Float32.(1:12),
        trues(12),
        UInt8[UInt8(ChoppedImageComplete)],
    )
    metadata = NamedTuple{SCC_SPATIAL_WEIGHTED_INPUT_NAMES}(
        (
            SampleMetadata(53, 0; timestamp_nanoseconds=601),
            nothing,
            nothing,
        ),
    )

    @test process!(outputs, graph, inputs, metadata) == NamedTuple{
        SCC_SPATIAL_WEIGHTED_OUTPUT_NAMES,
    }((true, true, true))
    @test all(measurement_validity)
    @test valid_fraction == Float32[1]
    expected_residual = Float32[110 / 21, -40 / 11]
    @test graph.node_outputs[2][1] ≈ expected_residual rtol = 2.0f-5
    @test command ≈ -0.5f0 .* expected_residual rtol = 2.0f-5
    @test all(
        value -> value == metadata.difference,
        values(FilterGraphs.output_metadata(graph)),
    )
end

@testset "9.2 failure preserves 9.5 controller state and publication" begin
    graph = prepare_scc_spatial_weighted_controller_graph()
    command = zeros(Float32, 2)
    measurement_validity = falses(6)
    valid_fraction = zeros(Float32, 1)
    outputs = scc_spatial_weighted_outputs(
        command,
        measurement_validity,
        valid_fraction,
    )
    valid = scc_spatial_weighted_inputs(
        Float32.(1:12),
        trues(12),
        UInt8[UInt8(ChoppedImageComplete)],
    )
    @test process!(outputs, graph, valid) == NamedTuple{
        SCC_SPATIAL_WEIGHTED_OUTPUT_NAMES,
    }((true, true, true))
    retained_outputs = map(copy, outputs)
    retained_state = copy(leaky_integrator_state(graph.nodes[3].prepared.workspace))

    priming = scc_spatial_weighted_inputs(
        valid.difference,
        valid.var"difference-validity",
        UInt8[UInt8(ChoppedImagePriming)],
    )
    @test process!(outputs, graph, priming) ===
          SccSpatialCarrierDemodulationIncompletePair
    @test outputs == retained_outputs
    @test leaky_integrator_state(graph.nodes[3].prepared.workspace) ==
          retained_state

    sparse_validity = trues(12)
    sparse_validity[[1, 5]] .= false
    insufficient = scc_spatial_weighted_inputs(
        valid.difference,
        sparse_validity,
        UInt8[UInt8(ChoppedImageComplete)],
    )
    @test process!(outputs, graph, insufficient) ===
          SccSpatialCarrierDemodulationInsufficientSupport
    @test outputs == retained_outputs
    @test leaky_integrator_state(graph.nodes[3].prepared.workspace) ==
          retained_state

    process!(outputs, graph, valid)
    @test scc_spatial_weighted_allocation_bytes(outputs, graph, valid) == 0
    @test scc_spatial_weighted_allocation_bytes(outputs, graph, priming) == 0
    @test scc_spatial_weighted_allocation_bytes(outputs, graph, insufficient) == 0
end
