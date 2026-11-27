module SourceChannelArgumentsTests

using Test
import FilterGraphPipeWire
using FilterGraphPipeWire: parse_arguments, run_service

const LEAKY = joinpath(@__DIR__, "data", "leaky-graph.conf")

@testset "source channel unit arguments" begin
    args = ["--graph", LEAKY, "--check"]
    @test isempty(parse_arguments(args).channel_units)
    options = parse_arguments([args; "--channel-unit"; "summary"; "mean_flux"; "ADC"])
    @test options.channel_units == Dict("summary" => Dict("mean_flux" => "ADC"))
    malformed = String(UInt8[0xff])
    for extra in (
        ["--channel-unit", malformed, "mean_flux", "ADC"],
        ["--channel-unit", "summary", malformed, "ADC"],
        ["--channel-unit", "summary", "mean_flux", malformed],
        ["--channel-unit", "summary", "mean_flux"],
        ["--channel-unit", "summary", "mean_flux", ""],
        ["--channel-unit", "summary", "mean-flux", "ADC"],
        ["--channel-unit", "summary", "mean_flux", "bad\nunits"],
        ["--channel-unit", "summary", "mean_flux", "ADC",
            "--channel-unit", "summary", "mean_flux", "ADC"],
    )
        @test_throws ArgumentError parse_arguments([args; extra])
    end
    # Semantic Port/channel checking occurs against the actual prepared source.
    @test_throws ArgumentError run_service(options)
end

end
