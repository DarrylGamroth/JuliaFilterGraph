module CpuProjectionGraphTests

using FilterGraphAlgorithms
using FilterGraphs
using LinearAlgebra
using Test

import FilterGraphAlgorithms: process!

const FGA = FilterGraphAlgorithms

projection_graph_bytes(outputs, graph, inputs) = @allocated process!(outputs, graph, inputs)

function prepare_projection_graph(text)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        prepare_graph(path; algorithms=algorithms())
    end
end

@testset "coordinate-count configuration rejects superseded names" begin
    for (label, configuration, input_name, output_name, renames) in (
        ("controller-to-vdm-f32",
         "controller_command_count = 2 active_vdm_count = 3",
         "controller-command", "vdm-command",
         ("controller_command_count" => "controller_command_extent",
          "active_vdm_count" => "vdm_size")),
        ("vdm-feedback-to-controller-f32",
         "controller_command_count = 2 active_vdm_count = 3",
         "vdm-constraint-feedback", "controller-constraint-feedback",
         ("controller_command_count" => "controller_command_extent",
          "active_vdm_count" => "vdm_size")),
        ("vdm-to-pdm-f32",
         "active_vdm_count = 2 full_vdm_count = 3 physical_actuator_count = 4",
         "vdm-command", "requested-pdm-command",
         ("active_vdm_count" => "active_vdm_size",
          "full_vdm_count" => "full_vdm_size",
          "physical_actuator_count" => "physical_actuator_extent")),
        ("pdm-feedback-to-vdm-f32",
         "active_vdm_count = 2 full_vdm_count = 3 physical_actuator_count = 4",
         "pdm-constraint-feedback", "vdm-constraint-feedback",
         ("active_vdm_count" => "active_vdm_size",
          "full_vdm_count" => "full_vdm_size",
          "physical_actuator_count" => "physical_actuator_extent")),
    )
        text = """
        filter.graph = {
            nodes = [ {
                type = ndarray name = projection label = $label
                config = { $configuration }
            } ]
            inputs = [ "projection:$input_name" ]
            outputs = [ "projection:$output_name" ]
        }
        """
        @test prepare_projection_graph(text) isa PreparedGraph
        for rename in renames
            error = try
                prepare_projection_graph(replace(text, rename))
                nothing
            catch exception
                exception
            end
            @test error isa GraphConfigurationError
            @test occursin("unknown configuration field '$(last(rename))'",
                           sprint(showerror, error))
        end
    end
end

@testset "Graph projection transactions rebuild both execution policies atomically" begin
    identity = Matrix{Float32}(I, 3, 3)
    selection = Float32[0 1 0; 0 1 0; 1 0 0]
    dense = Float32[1 0.25 0; 0 1 -0.5; 0.75 0 1]
    for (label, input_name, output_name, first_name, second_name, input_failure) in (
        ("vdm-to-pdm-f32", "vdm-command", "requested-pdm-command", "active-to-full", "vdm-to-pdm", VdmToPdmNonFiniteInput),
        ("pdm-feedback-to-vdm-f32", "pdm-constraint-feedback", "vdm-constraint-feedback", "pdm-to-vdm", "full-to-active", PdmFeedbackToVdmNonFiniteInput),
    )
        text = """
        filter.graph = {
            nodes = [ {
                type = ndarray
                name = projection
                label = $label
                config = { active_vdm_count = 3 full_vdm_count = 3 physical_actuator_count = 3 }
            } ]
            links = []
            inputs = [ "projection:$input_name" "projection:$first_name" "projection:$second_name" ]
            outputs = [ "projection:$output_name" ]
        }
        """
        graph = prepare_projection_graph(text)
        prepared = only(graph.nodes).prepared
        plan_type = typeof(prepared.plan)
        output = zeros(Float32, 3)
        input = Float32[1, 2, 3]
        outputs = NamedTuple{(Symbol(output_name),)}((output,))
        inputs = NamedTuple{(Symbol(input_name),)}((input,))
        available = NamedTuple{(Symbol(output_name),)}((true,))
        @test process!(outputs, graph, inputs) == available
        @test iszero(output)
        for (first, second) in ((identity, identity), (selection, selection),
                                (dense, selection), (selection, dense), (identity, identity))
            @test replace_parameters!(graph, first_name => first, second_name => second) === graph
            @test typeof(prepared.plan) === plan_type
            @test process!(outputs, graph, inputs) == available
            expected = second * (first * input)
            @test output ≈ expected
            @test projection_graph_bytes(outputs, graph, inputs) == 0

            active = prepared.plan
            @test_throws DimensionMismatch replace_parameters!(
                graph, first_name => dense, second_name => ones(Float32, 1, 1),
            )
            @test prepared.plan === active
            @test process!(outputs, graph, inputs) == available
            @test output ≈ expected
        end
        fill!(output, 71.0f0)
        input[end] = Inf32
        @test process!(outputs, graph, inputs) === input_failure
        @test output == fill(71.0f0, 3)
    end
end

@testset "Graph compact VDM completion stays live across parameter replacement" begin
    active_to_full = zeros(Float32, 10, 5)
    for column in 1:5
        active_to_full[column, column] = 1.0f0
    end
    for row in 6:10, column in 1:5
        active_to_full[row, column] = Float32(row + column) / 10.0f0
    end
    # A large coefficient makes completion of this extra row overflow for a
    # finite input while retaining the compact completion shape.
    active_to_full[6, 1] = 2.0f0

    general_dense = copy(active_to_full)
    for row in axes(general_dense, 1), column in axes(general_dense, 2)
        iszero(general_dense[row, column]) &&
            (general_dense[row, column] = Float32(row + column) / 100.0f0)
    end
    physical_projection = Matrix{Float32}(I, 10, 10)
    text = """
    filter.graph = {
        nodes = [ {
            type = ndarray
            name = projection
            label = vdm-to-pdm-f32
            config = { active_vdm_count = 5 full_vdm_count = 10 physical_actuator_count = 10 }
        } ]
        links = []
        inputs = [ "projection:vdm-command" "projection:active-to-full" "projection:vdm-to-pdm" ]
        outputs = [ "projection:requested-pdm-command" ]
    }
    """
    graph = prepare_projection_graph(text)
    prepared = only(graph.nodes).prepared
    workspace = prepared.workspace
    full_vdm_storage = workspace.full_vdm
    extrapolated_vdm_storage = workspace.extrapolated_vdm
    plan_type = typeof(prepared.plan)
    workspace_type = typeof(workspace)
    input = Float32[1, 2, 3, 4, 5]
    output = zeros(Float32, 10)
    outputs = NamedTuple{(Symbol("requested-pdm-command"),)}((output,))
    inputs = NamedTuple{(Symbol("vdm-command"),)}((input,))
    available = NamedTuple{(Symbol("requested-pdm-command"),)}((true,))

    @test length(workspace.full_vdm) == 10
    @test length(workspace.extrapolated_vdm) == 5
    @test replace_parameters!(
        graph,
        "active-to-full" => active_to_full,
        "vdm-to-pdm" => physical_projection,
    ) === graph
    @test typeof(prepared.plan) === plan_type
    @test typeof(prepared.workspace) === workspace_type
    @test prepared.workspace.full_vdm === full_vdm_storage
    @test prepared.workspace.extrapolated_vdm === extrapolated_vdm_storage
    @test prepared.plan.active_to_full_execution isa FGA.CompletionProjection
    @test process!(outputs, graph, inputs) == available
    @test output ≈ physical_projection * (active_to_full * input)
    @test projection_graph_bytes(outputs, graph, inputs) == 0

    @test replace_parameters!(graph, "active-to-full" => general_dense) === graph
    @test typeof(prepared.plan) === plan_type
    @test typeof(prepared.workspace) === workspace_type
    @test prepared.workspace.full_vdm === full_vdm_storage
    @test prepared.workspace.extrapolated_vdm === extrapolated_vdm_storage
    @test prepared.plan.active_to_full_execution isa FGA.DenseProjection
    @test process!(outputs, graph, inputs) == available
    @test output ≈ physical_projection * (general_dense * input)
    @test projection_graph_bytes(outputs, graph, inputs) == 0

    @test replace_parameters!(graph, "active-to-full" => active_to_full) === graph
    @test typeof(prepared.plan) === plan_type
    @test typeof(prepared.workspace) === workspace_type
    @test prepared.workspace.full_vdm === full_vdm_storage
    @test prepared.workspace.extrapolated_vdm === extrapolated_vdm_storage
    @test prepared.plan.active_to_full_execution isa FGA.CompletionProjection
    @test process!(outputs, graph, inputs) == available
    @test output ≈ physical_projection * (active_to_full * input)
    @test projection_graph_bytes(outputs, graph, inputs) == 0

    active_plan = prepared.plan
    expected_before_invalid_transaction = copy(output)
    @test_throws DimensionMismatch replace_parameters!(
        graph,
        "active-to-full" => general_dense,
        "vdm-to-pdm" => ones(Float32, 1, 1),
    )
    @test prepared.plan === active_plan
    @test output == expected_before_invalid_transaction
    @test process!(outputs, graph, inputs) == available
    @test output ≈ physical_projection * (active_to_full * input)

    fill!(output, 71.0f0)
    input .= Float32[floatmax(Float32), 0, 0, 0, 0]
    @test process!(outputs, graph, inputs) === VdmToPdmNonFiniteOutput
    @test output == fill(71.0f0, 10)
end

end
