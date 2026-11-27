module StatelessDeclarationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!

struct ScalePlan
    gain::Float32
end

"""Multiply each input component by the configured gain."""
@algorithm ScaleF32 begin
    label = "scale-f32"

    """Configuration for a scaling declaration using $(nameof(ScalePlan))."""
    @configuration ScaleF32Configuration begin
        extent::Int => graph_rebuild
        (gain::Float32 = 2f0) => construction
    end

    ports(config) = (
        input(:input, Float32, (config.extent,), "test.samples.f32/1"),
        output(
            :output,
            Float32,
            (config.extent,),
            "test.samples.f32/1";
            metadata_from=:input,
        ),
    )

    prepare(config) = ScalePlan(config.gain)

    process!(plan) = begin
        @. output = plan.gain * input
        nothing
    end
end

@testset "literal configuration defaults" begin
    @test ScaleF32Configuration(extent=3).gain === 2f0
    @test ScaleF32Configuration(extent=3, gain=5f0).gain === 5f0
    fields = algorithm_configuration(ScaleF32)
    @test fields[1].default === nothing
    @test fields[2].default == Some(2f0)
    @test AlgorithmConfiguration(:old, Int, ConstructionConfiguration()).default === nothing
    for default in (:extent, :(extent + 1), :(zeros(2)))
        declaration = :(@configuration BadConfiguration begin
            extent::Int => graph_rebuild
            (other::Int = $default) => construction
        end)
        @test_throws ArgumentError FilterGraphAlgorithms._parse_configuration(declaration)
    end
    optional = FilterGraphAlgorithms._parse_configuration(:(@configuration Optional begin
        (value::Union{Nothing,Int} = nothing) => construction
    end))
    @test optional[2][1][4] == Some(:nothing)
end

@testset "declaration and configuration docstrings" begin
    # Inspect the registered text without depending on REPL's Markdown renderer.
    declaration_doc = join(Base.Docs.docstr(Base.Docs.Binding(@__MODULE__, :ScaleF32)).text)
    configuration_doc = join(
        Base.Docs.docstr(Base.Docs.Binding(@__MODULE__, :ScaleF32Configuration)).text,
    )
    @test occursin("Multiply each input component", declaration_doc)
    @test occursin("Configuration for a scaling declaration using ScalePlan", configuration_doc)
    @test !occursin("Multiply each input component", configuration_doc)
    @test !occursin("Configuration for a scaling declaration", declaration_doc)
    configuration_metadata =
        Base.Docs.docstr(Base.Docs.Binding(@__MODULE__, :ScaleF32Configuration)).data
    @test configuration_metadata[:path] == @__FILE__
    @test occursin(
        "Configuration for a scaling declaration",
        readlines(@__FILE__)[configuration_metadata[:linenumber]],
    )
end

function admitted_allocation_bytes(output, prepared, input)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

function checked_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

@testset "stateless scientist declaration contract" begin
    @test isdefined(FilterGraphAlgorithms, Symbol("@algorithm"))
    @test algorithm_label(ScaleF32) == "scale-f32"
    @test FilterGraphAlgorithms.algorithm_configuration_type(ScaleF32) ===
          ScaleF32Configuration
    @test algorithm_property_policy(ScaleF32) isa FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(ScaleF32)
    @test map(field -> field.name, configuration) == (:extent, :gain)
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].update isa ConstructionConfiguration

    ports = algorithm_ports(ScaleF32)
    @test map(port -> port.name, ports) == (:input, :output)
    @test ports[1].direction isa FilterGraphAlgorithms.InputPort
    @test ports[2].direction isa FilterGraphAlgorithms.OutputPort
    @test ports[2].metadata_source === :input

    prepared = prepare_algorithm(
        ScaleF32,
        ScaleF32Configuration(extent=4, gain=2.0f0),
    )
    @test prepared.plan == ScalePlan(2.0f0)
    @test prepared.workspace !== nothing

    formats = algorithm_formats(prepared)
    @test length(formats) == 2
    @test all(format -> format.element_type === Float32, formats)
    @test all(format -> format.shape == (4,), formats)
    @test all(format -> format.layout isa ColumnMajorLayout, formats)
    @test all(format -> format.schema == "test.samples.f32/1", formats)

    input = Float32[1, 2, 3, 4]
    output = zeros(Float32, 4)
    @test process!(output, prepared, input) === nothing
    @test output == Float32[2, 4, 6, 8]
    @test reset!(prepared) === prepared

    @test_throws ArgumentError process!(output, prepared, Float64[1, 2, 3, 4])
    @test_throws DimensionMismatch process!(zeros(Float32, 3), prepared, input)

    @test FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    checked_allocation_bytes(output, prepared, input)
    @test checked_allocation_bytes(output, prepared, input) == 0
    admitted_allocation_bytes(output, prepared, input)
    @test admitted_allocation_bytes(output, prepared, input) == 0
end

end
