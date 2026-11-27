module ParameterPortsTests
using Test, FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire, PipeWireAO
include(joinpath(@__DIR__, "data", "support", "composition_fixture.jl"))
using .CompositionFixture: CompositionGateF32, composition_fixture
const FG = FilterGraphs
const FGP = FilterGraphPipeWire

function check_ports(ports, names, parameters, inputs)
    @test getfield.(ports, :name) == names
    @test getfield.(ports, :parameter) == parameters
    @test getfield.(ports, :direction) ==
        [fill(PipeWireAO.DIRECTION_INPUT, inputs); fill(PipeWireAO.DIRECTION_OUTPUT, length(ports) - inputs)]
    native = PipeWireAO._native_ndarray_filter_ports(ports).native
    flag = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_PORT_FLAG_PARAMETER
    @test getfield.(native, :flags) == [parameter ? flag : UInt32(0) for parameter in parameters]
end

@testset "optional Graph Parameter Ports preserve bound calibration" begin
    graph = FG.prepare_graph(joinpath(@__DIR__, "data", "pixel-calibration-row.conf");
        algorithms=(PixelCalibrationRowU16F32,))
    FG.replace_parameters!(graph, :flat => fill(3f0, 4, 3), :background => fill(2f0, 4, 3))
    declarations = FG.parameter_formats(graph)
    plan = only(graph.nodes).prepared.plan
    flat, background = plan.flat, plan.background
    check_ports(FGP._graph_ports(graph, 1000//1),
        ["raw", "flat", "background", "calibrated"], [false, true, true, false], 3)
    check_ports(FGP._graph_ports(graph.input_formats, NamedTuple(), graph.output_formats, 1000//1),
        ["raw", "calibrated"], [false, false], 1)
    for enabled in (true, false)
        node = enabled ? PipeWireNode(graph; connect=false, rate=1000//1) :
            PipeWireNode(graph; connect=false, rate=1000//1, parameter_ports=false)
        try
            @test enabled ? node.filter.callbacks.on_parameter isa FGP.GraphParameterCallback :
                node.filter.callbacks.on_parameter === nothing
            @test node.filter.input_port_indices == UInt32[0]
            @test FG.parameter_formats(graph) === declarations
            @test keys(declarations) == (:flat, :background)
            @test only(graph.nodes).prepared.plan === plan
            @test plan.flat === flat
            @test plan.background === background
            raw = reshape(UInt16[3, 4, 5, 6, 7, 8], 2, 3)
            output = zeros(Float32, 2, 3)
            FilterGraphAlgorithms.process!((calibrated=output,), graph, (raw=raw,),
                (raw=SampleMetadata(1, 0; terminal=false),))
            @test output == 3f0 .* (Float32.(raw) .- 2f0)
        finally
            close(node)
        end
    end
end

@testset "optional composition Parameter Ports preserve bound reference" begin
    mktempdir() do directory
        composition = FG.prepare_composition(composition_fixture(directory);
            algorithms=(FilterGraphAlgorithms.algorithms()..., CompositionGateF32))
        FG.replace_parameters!(composition, Symbol("a-reference") => [.1f0])
        declarations = FG.parameter_formats(composition)
        # This is the fixture's actual VectorOffsetF32 owner, not a test graph model.
        plan = composition.graph.nodes[2].prepared.plan
        offset = plan.offset
        names = ["a-image", "b-image", "a-reference", "a-correction", "b-correction", "command", "requested"]
        check_ports(FGP._composition_ports(composition, 1000//1), names,
            [false, false, true, false, false, false, false], 3)
        check_ports(FGP._composition_ports(composition, 1000//1; parameter_ports=false),
            names[[1, 2, 4, 5, 6, 7]], fill(false, 6), 2)
        for enabled in (true, false)
            FG.reset!(composition)
            node = enabled ? PipeWireNode(composition; connect=false) :
                PipeWireNode(composition; connect=false, parameter_ports=false)
            try
                @test enabled ? node.filter.callbacks.on_parameter isa FGP.GraphParameterCallback :
                    node.filter.callbacks.on_parameter === nothing
                @test node.filter.input_port_indices == UInt32[0, 1]
                @test FG.parameter_formats(composition) === declarations
                @test keys(declarations) == (Symbol("a-reference"),)
                @test composition.graph.nodes[2].prepared.plan === plan
                @test plan.offset === offset
                outputs = map(f -> zeros(f.element_type, f.shape), FG.output_formats(composition))
                result = FG.process!(outputs, composition, (;var"a-image"=[0f0]),
                    (;var"a-image"=SampleMetadata(1; terminal=true)))
                @test result.command
                @test outputs.command ≈ [.1f0]
            finally
                close(node)
            end
        end
    end
end
end
