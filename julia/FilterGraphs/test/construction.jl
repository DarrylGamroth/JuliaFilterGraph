module NativeConstructionTests

using FilterGraphAlgorithms
using FilterGraphs
using Test
using ..FilterGraphsTests: TestGateF32, FastPassF32, GRAPH_FIXTURE,
    PairAccumulatorF32, ConditionalConsumerF32, PreviewF32, CONDITIONAL_GRAPH_FIXTURE
import FilterGraphAlgorithms: process!, reset!

const DECLARATIONS = (FilterGraphAlgorithms.algorithms()..., TestGateF32, FastPassF32)
const INTEGRATOR = (
    name=:integrate, label="leaky-integrator-f32",
    config=(extent=3, initial_state=0f0,
        input_schema="test.signal.f32/1", output_schema="test.signal.f32/1"),
    props=(gain=1f0, pole=0.5f0))
const GATE = (name=:gate, label="test-gate-f32", config=(extent=3,))
const DESCRIPTION = (nodes=(INTEGRATOR, GATE),
    links=("integrate:output" => "gate:input",),
    inputs=("integrate:input",), outputs=("gate:output",))

formats(graph) = (FilterGraphs.input_formats(graph),
    FilterGraphs.output_formats(graph), FilterGraphs.parameter_formats(graph))
allocations(output, graph, input, metadata) = @allocated process!(output, graph, input, metadata)

@testset "file and native Graph boundaries and state trajectories" begin
    native = prepare_graph(DESCRIPTION; algorithms=DECLARATIONS)
    file = prepare_graph(GRAPH_FIXTURE; algorithms=DECLARATIONS)
    @test formats(native) == formats(file)
    @test map(node -> node.name, native.nodes) == (:integrate, :gate)
    outputs = map(_ -> (output=fill(71f0, 3),), (native, file))
    for (sequence, values) in enumerate(([1f0, 2, 3], [-2f0, 1, 1], [3f0, 2, 1], [NaN32, 0, 0]))
        inputs = (input=values,)
        metadata = (input=SampleMetadata(sequence; terminal=true),)
        results = map((graph, output) -> process!(output, graph, inputs, metadata),
            (native, file), outputs)
        @test results[1] == results[2]
        @test outputs[1] == outputs[2]
        @test FilterGraphs.output_metadata(native) == FilterGraphs.output_metadata(file)
        @test leaky_integrator_state(native.nodes[1].prepared.workspace) ==
            leaky_integrator_state(file.nodes[1].prepared.workspace)
        if !(results[1] isa NamedTuple)
            @test FilterGraphs.rejection_context(native).node ==
                FilterGraphs.rejection_context(file).node
        end
    end
    for graph in (native, file)
        reset!(graph)
        inputs = (input=Float32[1, 2, 3],)
        metadata = (input=SampleMetadata(5; terminal=true),)
        output = (output=zeros(Float32, 3),)
        process!(output, graph, inputs, metadata)
        allocations(output, graph, inputs, metadata)
        @test allocations(output, graph, inputs, metadata) == 0
        @test reset!(graph) === graph
    end
end

struct SnapshotPlan
    weights::Vector{Float32}
end

@algorithm SnapshotF32 begin
    label = "test-native-snapshot-f32"
    @configuration SnapshotConfiguration begin
        weights::Vector{Float32} => construction
    end
    ports(configuration) = (
        input(:input, Float32, (length(configuration.weights),), "test.signal/1"),
        output(:output, Float32, (length(configuration.weights),), "test.signal/1";
            metadata_from=:input))
    prepare(configuration) = SnapshotPlan(configuration.weights)
    process!(plan) = begin
        @. output = input * plan.weights
        nothing
    end
end

@testset "native construction snapshots mutable values and accepts typed config" begin
    weights = Float32[2, 3]
    config = SnapshotConfiguration(weights)
    nodes = [(name=:scale, label="test-native-snapshot-f32", config)]
    description = (nodes, inputs=(input="scale:input",), outputs=(output="scale:output",))
    a = prepare_graph(description; algorithms=(SnapshotF32,))
    b = prepare_graph(description; algorithms=(SnapshotF32,))
    @test a.nodes[1].prepared.plan.weights !== weights
    @test a.nodes[1].prepared.plan.weights !== b.nodes[1].prepared.plan.weights
    weights[1] = 99f0
    empty!(nodes)
    output = (output=zeros(Float32, 2),)
    process!(output, a, (input=Float32[4, 5],))
    @test output.output == Float32[8, 15]
    a.nodes[1].prepared.plan.weights[1] = 4f0
    process!(output, b, (input=Float32[4, 5],))
    @test output.output == Float32[8, 15]
end

@testset "subgraphs flatten with qualified diagnostics and independent state" begin
    component = (nodes=(INTEGRATOR,),
        inputs=(signal="integrate:input",), outputs=(signal="integrate:output",))
    nested = (nodes=((name=:left, graph=component), (name=:right, graph=component), GATE),
        links=("left:signal" => "gate:input",),
        inputs=(left="left:signal", right="right:signal"),
        outputs=(left="gate:output", right="right:signal"))
    graph = prepare_graph(nested; algorithms=DECLARATIONS)
    @test Set(map(node -> node.name, graph.nodes)) == Set((Symbol("left.integrate"), Symbol("right.integrate"), :gate))
    @test graph.nodes[1].prepared.workspace !== graph.nodes[2].prepared.workspace
    output = (left=zeros(Float32, 3), right=zeros(Float32, 3))
    process!(output, graph, (left=ones(Float32, 3), right=fill(2f0, 3)))
    @test output.left == ones(Float32, 3)
    @test output.right == fill(2f0, 3)
    replace_properties!(graph, "left.integrate:gain" => 0f0)
    process!(output, graph, (left=ones(Float32, 3), right=fill(2f0, 3)))
    @test output.left == fill(0.5f0, 3)
    @test output.right == fill(3f0, 3)

    double = (nodes=((name=:outer, graph=(nodes=((name=:inner, graph=component),),
        inputs=(signal="inner:signal",), outputs=(out="inner:signal",))),),
        inputs=(input="outer:signal",), outputs=(output="outer:out",))
    deep = prepare_graph(double; algorithms=DECLARATIONS)
    @test only(deep.nodes).name === Symbol("outer.inner.integrate")
    inputs = (input=fill(NaN32, 3),)
    outputs = (output=fill(71f0, 3),)
    @test process!(outputs, deep, inputs) === LeakyIntegratorNonFiniteInput
    @test outputs.output == fill(71f0, 3)
    @test FilterGraphs.rejection_context(deep).node === Symbol("outer.inner.integrate")
end

@testset "Parameters cross subgraph boundaries without extra Nodes" begin
    component = (nodes=((name=:offset, label="vector-offset-f32",
        config=(extent=2, schema="test.signal/1")),),
        inputs=(input="offset:requested", offset="offset:offset"),
        outputs=(output="offset:output", current="offset:current-offset"))
    graph = prepare_graph((nodes=((name=:front, graph=component),),
        inputs=(input="front:input", calibration="front:offset"),
        outputs=(output="front:output", current="front:current")); algorithms=DECLARATIONS)
    @test keys(FilterGraphs.parameter_formats(graph)) == (:calibration,)
    replace_parameters!(graph, :calibration => Float32[2, 3])
    outputs = (output=zeros(Float32, 2), current=zeros(Float32, 2))
    process!(outputs, graph, (input=Float32[4, 5],))
    @test outputs == (output=Float32[6, 8], current=Float32[2, 3])
end

@testset "native structural errors and nested boundary checks" begin
    bad = (
        merge(DESCRIPTION, (nodes=(INTEGRATOR, INTEGRATOR),)),
        merge(DESCRIPTION, (links=("absent:output" => "gate:input",),)),
        merge(DESCRIPTION, (links=("gate:output" => "integrate:input", "integrate:output" => "gate:input"),)),
        merge(DESCRIPTION, (nodes=(merge(INTEGRATOR, (config=(extent=0,),)), GATE),)),
        merge(DESCRIPTION, (execution=(cpu=(row_workers=1,),),)),
        merge(DESCRIPTION, (nodes=(merge(INTEGRATOR, (typo=1,)), GATE),)),
        (nodes=((name=:child, graph=DESCRIPTION),), inputs=(input="child:input",), outputs=(output="child:output",)),
    )
    for description in bad
        @test_throws GraphConfigurationError prepare_graph(description; algorithms=DECLARATIONS)
    end
    @test_throws GraphConfigurationError prepare_graph(
        merge(DESCRIPTION, (nodes=(GATE, (name=:fast, label="fast-pass-f32", config=(extent=3,))),
            links=("gate:output" => "fast:input",), inputs=("gate:input",), outputs=("fast:output",)));
        algorithms=DECLARATIONS)
    recursive = Dict{String,Any}()
    recursive["nodes"] = [(name=:self, graph=recursive)]
    @test_throws GraphConfigurationError prepare_graph(recursive; algorithms=DECLARATIONS)
    collision = (nodes=(INTEGRATOR,), inputs=(input="integrate:input",), outputs=(output="integrate:output",))
    @test_throws GraphConfigurationError prepare_graph((
        nodes=((name=:front, graph=collision), merge(INTEGRATOR, (name=Symbol("front.integrate"),))),
        inputs=(input="front:input",), outputs=(output="front:output",)); algorithms=DECLARATIONS)
    for component in (
        merge(collision, (outputs=(output="integrate:output", unused="integrate:absent"),)),
        merge(collision, (inputs=(input="integrate:input", unused="integrate:absent"),)),
        merge(collision, (inputs=(input="integrate:input", unused="integrate:output"),)),
        merge(collision, (outputs=(output="integrate:output", unused="integrate:input"),)),
    )
        @test_throws GraphConfigurationError prepare_graph((
            nodes=((name=:front, graph=component),),
            inputs=(input="front:input",), outputs=(output="front:output",)); algorithms=DECLARATIONS)
    end
end

@testset "native progressive and source-only descriptions match file execution" begin
    declarations = (PairAccumulatorF32, ConditionalConsumerF32, PreviewF32)
    # The dictionary path also accepts already-parsed SPA-JSON without changing
    # the ordinary file API or inventing a second declaration representation.
    description = FilterGraphs._parse_spa_json(read(CONDITIONAL_GRAPH_FIXTURE, String))
    native = prepare_graph(description; algorithms=declarations)
    file = prepare_graph(CONDITIONAL_GRAPH_FIXTURE; algorithms=declarations)
    outputs = map(_ -> (consumed=fill(71f0, 3), preview=fill(72f0, 3)), (native, file))
    @test formats(native) == formats(file)
    for index in 1:4
        inputs = (piece=fill(Float32(index), 3), monitor=ones(Float32, 3))
        metadata = (piece=SampleMetadata(cld(index, 2), mod(index - 1, 2); terminal=iseven(index)),
            monitor=SampleMetadata(index; terminal=true))
        result = map((graph, output) -> process!(output, graph, inputs, metadata), (native, file), outputs)
        @test result[1] == result[2]
        @test outputs[1] == outputs[2]
        @test FilterGraphs.output_metadata(native) == FilterGraphs.output_metadata(file)
        if isodd(index)
            @test result[1] == (consumed=false, preview=false)
        end
    end
    path = joinpath(@__DIR__, "data", "examples", "docrime-excitation", "graph.conf")
    source = prepare_graph(FilterGraphs._parse_spa_json(read(path, String)); algorithms=DECLARATIONS)
    reference = prepare_graph(path; algorithms=DECLARATIONS)
    @test isempty(FilterGraphs.input_formats(source))
    output = map(_ -> (excitation=zeros(Float32, 3),), (source, reference))
    for _ in 1:3
        @test process!(output[1], source, (;)) == process!(output[2], reference, (;))
        @test output[1] == output[2]
    end
end

end
