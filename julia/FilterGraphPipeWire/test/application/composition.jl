module CompositionServiceTests
using Test, FilterGraphPipeWire, FilterGraphs, FilterGraphAlgorithms
include(joinpath(pkgdir(FilterGraphs),"test","composition_fixture.jl"))
using .CompositionFixture: CompositionGateF32, composition_fixture
@testset "existing graph CLI admits composition descriptors" begin
    mktempdir() do directory
        descriptor = composition_fixture(directory)
        declarations = (FilterGraphAlgorithms.algorithms()...,CompositionGateF32)
        options = FilterGraphPipeWire.parse_arguments(["--graph",descriptor,"--execution","complete-frame","--check"])
        prepared = FilterGraphPipeWire._prepare_service_graph(options;declarations)
        @test prepared isa FilterGraphs.PreparedComposition
        @test keys(FilterGraphs.parameter_formats(prepared)) == (Symbol("a-reference"),)
        @test FilterGraphPipeWire.close_execution(prepared) === nothing
        feedback_options = FilterGraphPipeWire.parse_arguments(["--graph",descriptor,"--check","--feedback","a-feedback","a-returned"])
        @test_throws ArgumentError FilterGraphPipeWire._prepare_service_graph(feedback_options;declarations)
        write(descriptor,replace(read(descriptor,String),"shared = shared.conf"=>"shared = shared.conf unknown = true"))
        @test_throws FilterGraphs.GraphConfigurationError FilterGraphPipeWire._prepare_service_graph(options;declarations)
    end
end
end
