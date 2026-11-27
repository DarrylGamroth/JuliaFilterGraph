const LEGACY_DAC_SETTINGS = """
    quantization_origin = null
    quantization_step = null
    dither_seed = null
    """

function pdm_command_graph_text(slew_limit::String; dac_settings=LEGACY_DAC_SETTINGS)
    return """
    filter.graph = {
        nodes = [
            {
                type = ndarray
                name = command
                label = pdm-command-f32
                config = {
                    actuator_count = 2
                    command_limits = [-1.0, 1.0]
                    slew_limit = $slew_limit
                    initial_current = [ 0.0 0.0 ]
                    $dac_settings
                }
            }
        ]
        links = []
        inputs = [ "command:requested" ]
        outputs = [ "command:demanded" "command:constraint-feedback" ]
    }
    """
end

@testset "PDM conditioning accepts disabled legacy DAC settings" begin
    prepare_conditioning(dac_settings) = prepare_graph_text(
        pdm_command_graph_text("[0.2 0.1]"; dac_settings); algorithms=(PdmCommandF32,))

    for settings in (LEGACY_DAC_SETTINGS, "", "dither_seed = 42", "dither_seed = 0")
        graph = prepare_conditioning(settings)
        demanded, feedback = zeros(Float32, 2), zeros(Float32, 2)
        outputs = (demanded=demanded, var"constraint-feedback"=feedback)
        requested = Float32[1.3, -0.08]
        @test process!(outputs, graph, (; requested)) ==
            (demanded=true, var"constraint-feedback"=true)
        @test demanded == Float32[0.2, -0.08]
        @test feedback == requested - demanded
    end

    for settings in (
        "quantization_origin = 0.0 quantization_step = 0.125",
        "quantization_origin = 0.0 quantization_step = 0.125 dither_seed = 42",
        "quantization_origin = 0.0",
        "quantization_step = 0.125",
    )
        failure = try
            prepare_conditioning(settings)
            nothing
        catch exception
            exception
        end
        @test failure isa GraphConfigurationError
        @test occursin("driver", lowercase(sprint(showerror, failure)))
    end
end

const PDM_COMMAND_OUTPUT_NAMES = (:demanded, Symbol("constraint-feedback"))

@testset "PDM command limit pairs retain independent actuator bounds" begin
    for (limits, expected) in (
        ("[-0.3 0.4]", Float32[-0.3, 0.4]),
        ("[[-0.2 -0.3] [0.4 0.5]]", Float32[-0.2, 0.5]),
        ("[-0.3 [0.4 0.5]]", Float32[-0.3, 0.5]),
        ("[[-0.2 -0.3] 0.4]", Float32[-0.2, 0.4]),
    )
        graph = prepare_graph_text(replace(pdm_command_graph_text("null"),
            "command_limits = [-1.0, 1.0]" => "command_limits = $limits");
            algorithms=(PdmCommandF32,))
        demanded, feedback = zeros(Float32, 2), zeros(Float32, 2)
        process!(NamedTuple{PDM_COMMAND_OUTPUT_NAMES}((demanded, feedback)), graph,
            (requested=Float32[-2, 2],))
        @test demanded == expected
        @test feedback == Float32[-2, 2] - expected
    end
    for limits in ("[]", "[-1]", "[-1 1 2]", "[[ -1 ] 1]",
        "[-1 [1]]", "[1 -1]", "[-1 [1e100 1]]")
        @test_throws GraphConfigurationError prepare_graph_text(
            replace(pdm_command_graph_text("null"),
                "command_limits = [-1.0, 1.0]" => "command_limits = $limits");
            algorithms=(PdmCommandF32,))
    end
end

function pdm_command_outputs(demanded, constraint_feedback)
    return NamedTuple{PDM_COMMAND_OUTPUT_NAMES}((demanded, constraint_feedback))
end

function pdm_command_graph_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "PDM command slew configuration matrix" begin
    cases = (
        (
            slew_limit="null",
            expected=Float32[0.5, -0.5],
        ),
        (
            slew_limit="[ 0.2 0.1 ]",
            expected=Float32[0.2, -0.1],
        ),
    )
    requested = Float32[0.5, -0.5]

    for case in cases
        graph = prepare_graph_text(
            pdm_command_graph_text(case.slew_limit);
            algorithms=(PdmCommandF32,),
        )
        demanded = zeros(Float32, 2)
        constraint_feedback = zeros(Float32, 2)
        outputs = pdm_command_outputs(demanded, constraint_feedback)
        inputs = (requested=requested,)

        @test process!(outputs, graph, inputs) == (
            demanded=true,
            var"constraint-feedback"=true,
        )
        @test demanded ≈ case.expected
        @test constraint_feedback ≈ requested - demanded

        process!(outputs, graph, inputs)
        @test pdm_command_graph_allocation_bytes(outputs, graph, inputs) == 0
    end

    for obsolete_mode in ("independent", "vector-preserving")
        obsolete_configuration = replace(
            pdm_command_graph_text("[ 0.2 0.1 ]"),
            "slew_limit =" => "slew_mode = $obsolete_mode\n                    slew_limit =",
        )
        @test_throws GraphConfigurationError prepare_graph_text(
            obsolete_configuration;
            algorithms=(PdmCommandF32,),
        )
    end
end

@testset "Graph rejects aliased external output buffers before processing" begin
    graph = prepare_graph_text(
        pdm_command_graph_text("null");
        algorithms=(PdmCommandF32,),
    )
    workspace = only(graph.nodes).prepared.workspace
    state_before = copy(pdm_command_state(workspace))
    storage = fill(73.0f0, 3)
    outputs = pdm_command_outputs(view(storage, 1:2), view(storage, 2:3))

    @test Base.mightalias(
        outputs.demanded,
        getproperty(outputs, Symbol("constraint-feedback")),
    )
    @test_throws ArgumentError process!(
        outputs,
        graph,
        (requested=Float32[0.5, -0.5],),
    )
    @test storage == fill(73.0f0, 3)
    @test pdm_command_state(workspace) == state_before
end
