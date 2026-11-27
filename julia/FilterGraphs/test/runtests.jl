module FilterGraphsTests

using FilterGraphAlgorithms
using FilterGraphs
using Test
using TOML

include("composition.jl")
include("external_aliases.jl")
include("cpu_loans.jl")
include("scc_llowfs.jl")
include("cred2_fits_replay.jl")

import FilterGraphAlgorithms: process!, reset!

@enum TestGateFailure::UInt8 begin
    TestGateRejected = 1
end

struct TestGatePlan
    extent::Int
end

@algorithm TestGateF32 begin
    label = "test-gate-f32"
    failure = TestGateFailure

    @configuration TestGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] > 100.0f0 && error("simulated downstream abort")
        input[1] < 0.0f0 && return TestGateRejected
        copyto!(output, input)
        return nothing
    end
end

@algorithm FastPassF32 begin
    label = "fast-pass-f32"

    @configuration FastPassF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            rate=(2, 1),
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
            rate=(2, 1),
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = begin
        copyto!(output, input)
        nothing
    end
end

graph_allocation_bytes(outputs, graph, inputs) = @allocated process!(outputs, graph, inputs)
graph_reset_allocation_bytes(graph) = @allocated reset!(graph)

function prepare_graph_text(text; algorithms)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        return prepare_graph(path; algorithms=algorithms)
    end
end

const GRAPH_FIXTURE = joinpath(@__DIR__, "data", "two-node-transaction.conf")

@testset "prepared Julia Filter Graph" begin
    graph = prepare_graph(
        GRAPH_FIXTURE;
        algorithms=(algorithms()..., TestGateF32),
    )

    @test graph isa PreparedGraph
    @test map(node -> node.name, graph.nodes) == (:integrate, :gate)
    @test keys(graph.input_formats) == (:input,)
    @test keys(graph.output_formats) == (:output,)
    @test graph.node_failure_types == (LeakyIntegratorFailure, TestGateFailure)
    @test graph.output_conditional == (output=false,)
    @test graph.input_formats.input.shape == (3,)
    @test graph.input_formats.input.rate == (0x00000001, 0x00000001)
    @test graph.output_formats.output.schema == "test.signal.f32/1"
    @test property_values(graph.nodes[1].prepared.plan) == (gain=1.0f0, pole=0.5f0, closed=true)

    input = Float32[1, 2, 3]
    output = zeros(Float32, 3)
    inputs = (input=input,)
    outputs = (output=output,)
    workspace = graph.nodes[1].prepared.workspace

    @test process!(outputs, graph, inputs) == (output=true,)
    @test output == Float32[1, 2, 3]
    @test FilterGraphs.output_metadata(graph) == (output=nothing,)
    @test FilterGraphs.output_metadata_source_inputs(graph) == (1,)
    @test leaky_integrator_state(workspace) == output

    fill!(output, 71.0f0)
    rejected_inputs = (input=Float32[-2, 1, 1],)
    @test process!(outputs, graph, rejected_inputs) === TestGateRejected
    @test output == fill(71.0f0, 3)
    @test leaky_integrator_state(workspace) == Float32[-1.5, 2, 2.5]

    next_inputs = (input=Float32[3, 2, 1],)
    @test process!(outputs, graph, next_inputs) == (output=true,)
    @test output == Float32[2.25, 3, 2.25]
    @test leaky_integrator_state(workspace) == output

    fill!(output, 83.0f0)
    invalid_inputs = (input=Float32[NaN, 0, 0],)
    committed_before_invalid = copy(leaky_integrator_state(workspace))
    @test process!(outputs, graph, invalid_inputs) ===
          LeakyIntegratorNonFiniteInput
    @test output == fill(83.0f0, 3)
    @test leaky_integrator_state(workspace) == committed_before_invalid

    fill!(output, 91.0f0)
    @test_throws ErrorException process!(outputs, graph, (input=Float32[200, 0, 0],))
    @test output == fill(91.0f0, 3)
    @test leaky_integrator_state(workspace) == Float32[201.125, 1.5, 1.125]

    @test reset!(graph) === graph
    @test leaky_integrator_state(workspace) == zeros(Float32, 3)

    @test_throws ArgumentError process!((wrong=output,), graph, inputs)
    @test_throws ArgumentError process!(outputs, graph, (input=Float64[1, 2, 3],))
    @test_throws DimensionMismatch process!(outputs, graph, (input=zeros(Float32, 2),))
    @test_throws GraphConfigurationError prepare_graph(
        GRAPH_FIXTURE;
        algorithms=(LeakyIntegratorF32,),
    )
    @test_throws GraphConfigurationError prepare_graph(
        GRAPH_FIXTURE;
        algorithms=(LeakyIntegratorF32, LeakyIntegratorF32),
    )

    reset!(graph)
    process!(outputs, graph, inputs)
    process!(outputs, graph, rejected_inputs)
    @test graph_allocation_bytes(outputs, graph, inputs) == 0
    @test graph_allocation_bytes(outputs, graph, rejected_inputs) == 0
    @test graph_allocation_bytes(outputs, graph, invalid_inputs) == 0
    reset!(graph)
    @test FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        graph.default_input_metadata,
    ) == (output=true,)
    @test @allocated(
        FilterGraphs._process_admitted!(
            outputs,
            graph,
            inputs,
            graph.default_input_metadata,
        )
    ) == 0

    @test graph_reset_allocation_bytes(graph) == 0
end

@testset "long vector rejection retains graph publication" begin
    extent = 4097
    graph = prepare_graph_text("""
        filter.graph = {
          nodes = [{name=sum type=ndarray label=vector-linear-combination-f32
            config={extent=$extent left_scale=2.0 right_scale=0.0 schema="test.signal.f32/1"}}]
          inputs = [{name=left endpoint=sum:left} {name=right endpoint=sum:right}]
          outputs = [{name=output endpoint=sum:output}]
        }
        """; algorithms=algorithms())
    inputs = (left=ones(Float32, extent), right=ones(Float32, extent))
    outputs = (output=zeros(Float32, extent),)
    initial_metadata = (left=SampleMetadata(10), right=SampleMetadata(20))
    @test process!(outputs, graph, inputs, initial_metadata) == (output=true,)
    @test all(==(2f0), outputs.output)
    @test FilterGraphs.output_metadata(graph) == (output=initial_metadata.left,)

    inputs.left[1] = floatmax(Float32)
    inputs.right[end] = NaN32
    failed_metadata = (left=SampleMetadata(11), right=SampleMetadata(21))
    @test process!(outputs, graph, inputs, failed_metadata) === VectorLinearCombinationNonFiniteInput
    @test all(==(2f0), outputs.output)
    @test FilterGraphs.output_metadata(graph) == (output=initial_metadata.left,)
    rejection = FilterGraphs.rejection_context(graph)
    @test rejection.node === :sum
    @test rejection.reason === VectorLinearCombinationNonFiniteInput
    @test rejection.inputs == failed_metadata

    fill!(inputs.left, 3f0)
    inputs.right[end] = 1f0
    recovered_metadata = (left=SampleMetadata(12), right=SampleMetadata(22))
    @test process!(outputs, graph, inputs, recovered_metadata) == (output=true,)
    @test all(==(6f0), outputs.output)
    @test FilterGraphs.output_metadata(graph) == (output=recovered_metadata.left,)
    @test FilterGraphs.rejection_context(graph) === nothing
end

include("conditional_availability.jl")
include("captured_graph.jl")
include("captured_protocol.jl")
include("captured_feedback.jl")
include("captured_validation.jl")
include("captured_aliases.jl")
include("latest_hold.jl")
include("frame_delay.jl")
include("array_statistics.jl")
include("biquad_cascade.jl")
include("synchronized_inputs.jl")
include("leaky_controller.jl")
include("cpu_projection.jl")
include("pdm_command.jl")
include("pdm_command_power_limit.jl")
include("inter_actuator_stroke_limit.jl")
include("pdm_stroke_optimization.jl")
include("pseudo_open_loop_reconstruction.jl")
include("interaction_matrix_stability.jl")
include("matched_calibration_differencing.jl")
include("calibration_probe.jl")
include("optical_gain_dither.jl")
include("optical_gain_estimator.jl")
include("optical_gain_update.jl")
include("optical_gain_workflow.jl")
include("structured_interaction_matrix.jl")
include("scc_push_pull_response.jl")
include("chopped_image_difference.jl")
include("chopped_pixel_controller.jl")
include("scc_controller.jl")
include("scc_spatial_weighted_controller.jl")
include("pixel_calibration_row.jl")
include("detector_readout.jl")
include("image_geometry.jl")
include("center_of_gravity.jl")
include("shack_hartmann_image.jl")
include("shack_hartmann_summary.jl")
include("shack_hartmann_weighted_centroid.jl")
include("wavefront_reconstruction.jl")
include("shack_hartmann_row_reconstruction.jl")
include("shack_hartmann_readout_reconstruction.jl")
include("pyramid_pixel_row.jl")
include("pyramid_image.jl")
include("bi_o_edge_image.jl")
include("pyramid_row_reconstruction.jl")
include("progressive_wavefront_reconstruction.jl")
include("pyramid_readout_reconstruction.jl")
include("pixel_pyramid_row.jl")
include("closed_loop_correction.jl")
include("docrime_excitation.jl")
include("docrime_covariance.jl")
include("docrime_interaction_matrix.jl")
include("scao_examples.jl")
include("examples.jl")
include("modal_command_allocation.jl")
include("corrector_stages.jl")
include("vector_offset.jl")
include("physical_offset_reference.jl")
include("engine_remediation.jl")

@testset "SPA-JSON preserves full-width unsigned integers" begin
    document = FilterGraphs._parse_spa_json(
        """
        small = 17
        negative = -1
        first_unsigned = $(UInt64(typemax(Int)) + UInt64(1))
        max_unsigned = $(typemax(UInt64))
        fractional = 1.5
        """,
    )

    @test document["small"] === 17
    @test document["negative"] === -1
    @test document["first_unsigned"] === UInt64(typemax(Int)) + UInt64(1)
    @test document["max_unsigned"] === typemax(UInt64)
    @test document["fractional"] === 1.5
end

@testset "SPA-JSON Graph configuration rejection" begin
    cyclic = """
    filter.graph = {
        nodes = [
            { type = ndarray name = first label = test-gate-f32 config = { extent = 3 } }
            { type = ndarray name = second label = test-gate-f32 config = { extent = 3 } }
        ]
        links = [
            { output = "first:output" input = "second:input" }
            { output = "second:output" input = "first:input" }
        ]
        inputs = []
        outputs = []
    }
    """
    @test_throws GraphConfigurationError prepare_graph_text(
        cyclic;
        algorithms=(TestGateF32,),
    )

    unknown_configuration = """
    filter.graph = {
        nodes = [
            {
                type = ndarray
                name = gate
                label = test-gate-f32
                config = { extent = 3 invented = true }
            }
        ]
        links = []
        inputs = [ "gate:input" ]
        outputs = [ "gate:output" ]
    }
    """
    @test_throws GraphConfigurationError prepare_graph_text(
        unknown_configuration;
        algorithms=(TestGateF32,),
    )

    duplicate_configuration = """
    filter.graph = {
        nodes = [
            {
                type = ndarray
                name = gate
                label = test-gate-f32
                config = { extent = 3 extent = 4 }
            }
        ]
        links = []
        inputs = [ "gate:input" ]
        outputs = [ "gate:output" ]
    }
    """
    @test_throws GraphConfigurationError prepare_graph_text(
        duplicate_configuration;
        algorithms=(TestGateF32,),
    )

    rate_mismatch = """
    filter.graph = {
        nodes = [
            { type = ndarray name = base label = test-gate-f32 config = { extent = 3 } }
            { type = ndarray name = fast label = fast-pass-f32 config = { extent = 3 } }
        ]
        links = [ { output = "base:output" input = "fast:input" } ]
        inputs = [ "base:input" ]
        outputs = [ "fast:output" ]
    }
    """
    @test_throws GraphConfigurationError prepare_graph_text(
        rate_mismatch;
        algorithms=(TestGateF32, FastPassF32),
    )
end

include("named_ports.jl")
include("lifecycle.jl")
include("properties.jl")
include("formats.jl")
include("rejection.jl")
include("stateful_example.jl")
include("construction.jl")
include("version.jl")
include("controller_graph.jl")
ControllerGraphTests.run_controller_tests()

@testset "package surface" begin
    @test isempty(Test.detect_ambiguities(FilterGraphs; recursive=true))
    for name in (
        :GraphRejection,
        :rejection_context,
        :input_formats,
        :output_formats,
        :parameter_formats,
        :replace_properties!,
        :GraphConfigurationError,
        :CapturedGraphBusy,
        :CapturedGraphFault,
        :CapturedGraphState,
        :PreparedCapturedGraph,
        :PreparedGraph,
        :PreparedComposition,
        :prepare_composition,
        :composition_observation,
        :PreparedSynchronizedInputs,
        :expire!,
        :captured_graph_fault,
        :captured_graph_state,
        :output_metadata,
        :output_metadata_source_inputs,
        :prepare_captured_graph,
        :prepare_graph,
        :prepare_synchronized_inputs,
        :process!,
        :replace_parameters!,
        :reset!,
        :synchronized_input_counters,
        :software_version,
    )
        @test Base.ispublic(FilterGraphs, name)
        @test Base.Docs.hasdoc(FilterGraphs, name)
    end
    @test !Base.isexported(FilterGraphs, :process!)
    @test !Base.isexported(FilterGraphs, :reset!)
end

include("aqua.jl")

end
