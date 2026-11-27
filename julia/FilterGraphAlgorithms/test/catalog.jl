module AlgorithmCatalogTests

using FilterGraphAlgorithms
using Test
import FilterGraphAlgorithms: process!

const FGA = FilterGraphAlgorithms
const PREPARATIONS = Ref(0)

# Scientist-owned declaration: preparation is observable so cold inspection
# cannot accidentally satisfy its contract by constructing an instance.
@algorithm ScaleF32 begin
    label = "catalog-scale-f32"
    @configuration ScaleConfiguration begin
        extent::Int => graph_rebuild
        (gain::Float32 = 2f0) => construction
        (optional::Union{Nothing,Int} = nothing) => construction
    end
    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(:output, Float32, (configuration.extent,), "test.signal.f32/1";
            metadata_from=:input, publication=:conditional),
    )
    prepare(configuration) = begin
        PREPARATIONS[] += 1
        configuration.gain
    end
    process!(plan) = begin
        input[1] < 0f0 && return AlgorithmProcessDeferred()
        @. output = plan * input
        nothing
    end
end

struct MutableDefaults <: AbstractAlgorithmDeclaration end
const DEFAULT = Float32[1, 2]
FGA.algorithm_label(::Type{MutableDefaults}) = "catalog-mutable-defaults"
FGA.algorithm_configuration_type(::Type{MutableDefaults}) = Vector{Float32}
FGA.algorithm_configuration(::Type{MutableDefaults}) = (
    AlgorithmConfiguration(:initial, Vector{Float32}, ConstructionConfiguration(), Some(DEFAULT)),
)
FGA.algorithm_ports(::Type{MutableDefaults}) = ()
FGA.algorithm_property_policy(::Type{MutableDefaults}) = FGA.NoDeclaredProperties()

# This projection deliberately excludes native Julia/Rust type spellings and
# unavailable defaults. It is a comparison probe, not a serialization API.
function common_projection(declaration)
    entry = algorithm_description(declaration)
    element_types = Dict(Bool => "bool", UInt16 => "uint16",
        UInt32 => "uint32", Float32 => "float32")
    return (
        version=1,
        label=entry.label,
        configuration=map(entry.configuration) do field
            (name=String(field.name), update=field.update isa ConstructionConfiguration ?
                "construction" : "graph-rebuild")
        end,
        ports=map(entry.ports) do port
            (
                name=String(port.name),
                direction=port.direction isa FGA.InputPort ? "input" : "output",
                role=port.role isa FGA.SparseParameter ? "parameter" : "data",
                element_type=element_types[port.element_type],
                metadata_source=isnothing(port.metadata_source) ? nothing : String(port.metadata_source),
                publication=port.conditional ? "conditional" : "always",
            )
        end,
    )
end

@testset "cold declaration catalog and scientist declarations" begin
    entries = algorithm_catalog((ScaleF32, LeakyIntegratorF32, MutableDefaults))
    @test entries isa Tuple
    @test map(entry -> entry.declaration, entries) == (ScaleF32, LeakyIntegratorF32, MutableDefaults)
    entry = first(entries)
    @test entry.stage === :declaration
    @test entry.configuration_type === ScaleConfiguration
    @test entry.configuration[1].default === nothing
    @test entry.configuration[2].default == Some(2f0)
    @test entry.configuration[3].default == Some(nothing)
    @test entry.formats === entry.properties === entry.required_metadata === nothing
    @test PREPARATIONS[] == 0
    @test algorithm_catalog(()) === ()
    @test_throws ArgumentError algorithm_catalog((ScaleF32, ScaleF32))
    @test_throws MethodError algorithm_description("unknown")
    @test length(algorithm_catalog()) == length(algorithms())
    @test_throws MethodError setindex!(entry, :other, :label)

    snapshot = last(entries).configuration[1].default.value
    snapshot[1] = 99f0
    @test DEFAULT == Float32[1, 2]
    @test algorithm_description(MutableDefaults).configuration[1].default.value == DEFAULT
end

@testset "prepared metadata does not change processing or state" begin
    prepared = prepare_algorithm(ScaleF32, ScaleConfiguration(extent=2))
    entry = algorithm_description(prepared)
    @test PREPARATIONS[] == 1
    @test entry.stage === :prepared
    @test entry.formats[1].shape == (2,)
    @test entry.properties === ()
    @test entry.required_metadata == (input=(),)
    @test !hasproperty(entry, :plan) && !hasproperty(entry, :workspace)
    output = zeros(Float32, 2)
    @test process!(output, prepared, Float32[1, 3]) === nothing
    @test output == Float32[2, 6]
    @test !process!(output, prepared, Float32[-1, 3]).complete
    @test entry.ports[2].conditional

    stateful = prepare_algorithm(LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(2, 0f0, "test.input.f32/1", "test.output.f32/1"))
    before = deepcopy(stateful.workspace)
    description = algorithm_description(stateful)
    @test description.property_policy isa RuntimeDeclaredProperties
    @test description.failure_type === LeakyIntegratorFailure
    @test map(info -> info.name, description.properties) == (:gain, :pole, :closed)
    @test description.properties == property_infos(stateful.plan)
    @test stateful.workspace == before || all(
        name -> getfield(stateful.workspace, name) == getfield(before, name),
        fieldnames(typeof(before)),
    )
    @test process!(output, stateful, Float32[2, 4]) === nothing
    @test output == Float32[2, 4]
end

@testset "Parameter and progressive declarations preserve availability" begin
    prepared = prepare_algorithm(PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration((2, 3), nothing, nothing))
    description = algorithm_description(prepared)
    @test description.formats[3].shape == (2, 3)
    @test description.formats[3].layout isa ColumnMajorLayout
    @test description.ports[3].role isa FGA.SparseParameter
    @test description.required_metadata == (raw=(), flat=(), background=())
    flat_before = copy(prepared.plan.flat)
    replace_parameter!(prepared, :flat, fill(3f0, 2, 3))
    @test prepared.plan.flat != flat_before
    @test description.formats == algorithm_description(prepared).formats
    raw = UInt16[1 2 3; 4 5 6]
    calibrated = zeros(Float32, 2, 3)
    @test process!(calibrated, prepared, raw) === nothing
    @test calibrated == Float32[3 6 9; 12 15 18]
    @test first(description.configuration).name === :image_size

    progressive = algorithm_description(PyramidRowReconstructorF32)
    @test any(port -> port.conditional, progressive.ports)
    @test progressive.formats === nothing
    @test progressive.required_metadata === nothing
    @test algorithm_description(PixelCalibrationRowU16F32).formats === nothing

    chopped = prepare_algorithm(ChoppedImageDifferenceF32,
        ChoppedImageDifferenceF32Configuration(pixel_count=2,
            trigger_high_is_fringed=true, require_acquisition_identity=true))
    @test algorithm_description(chopped).required_metadata.image ==
        (:acquisition_identity, :exposure_duration_nanoseconds, :chopped_image_kind)
end

@testset "common projection v1 matches Rust declaration facts" begin
    projected = common_projection(LeakyIntegratorF32)
    @test projected.version == 1
    @test projected.label == "leaky-integrator-f32"
    @test projected.configuration == (
        (name="extent", update="graph-rebuild"),
        (name="initial_state", update="construction"),
        (name="input_schema", update="graph-rebuild"),
        (name="output_schema", update="graph-rebuild"),
    )
    @test projected.ports == (
        (name="input", direction="input", role="data", element_type="float32",
            metadata_source=nothing, publication="always"),
        (name="output", direction="output", role="data", element_type="float32",
            metadata_source="input", publication="always"),
    )
    @test count(port -> port.role == "parameter", common_projection(PixelCalibrationU16F32).ports) == 2
    progressive = common_projection(PyramidRowReconstructorF32)
    @test any(port -> port.publication == "conditional", progressive.ports)
    @test progressive.ports[5].element_type == "uint32"
    @test progressive.ports[6].element_type == "bool"
end

end
