module GraphPropertyTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

import FilterGraphAlgorithms:
    commit!, discard!, process!, property_schema, property_values,
    replace_properties, reset!, _plan_adopted!

function graph_from_text(text; algorithms=FilterGraphAlgorithms.algorithms())
    return mktemp() do path, io
        write(io, text)
        flush(io)
        prepare_graph(path; algorithms=algorithms)
    end
end

const INTEGRATORS = """
filter.graph = {
    nodes = [
        { type = ndarray name = first label = leaky-integrator-f32
          config = { extent = 2 initial_state = 0.0
                     input_schema = test.signal/1 output_schema = test.signal/1 }
          props = { gain = 1.0 pole = 0.5 } }
        { type = ndarray name = second label = leaky-integrator-f32
          config = { extent = 2 initial_state = 0.0
                     input_schema = test.signal/1 output_schema = test.signal/1 }
          props = { gain = 1.0 pole = 0.0 } }
    ]
    links = [ { output = "first:output" input = "second:input" } ]
    inputs = [ "first:input" ]
    outputs = [ "second:output" ]
}
"""

@testset "authored Graph Boolean Property starts integration open" begin
    graph = graph_from_text(replace(INTEGRATORS,
        "gain = 1.0 pole = 0.5" => "gain = 1.0 pole = 0.5 closed = false"))
    @test !graph.nodes[1].prepared.plan.closed
    outputs = (output=zeros(Float32, 2),)
    inputs = (input=Float32[2, 4],)
    @test process!(outputs, graph, inputs) == (output=true,)
    @test outputs.output == zeros(Float32, 2)
    replace_properties!(graph, "first:closed" => true)
    @test process!(outputs, graph, inputs) == (output=true,)
    @test outputs.output == Float32[2, 4]
end

@testset "named Graph scalar Property transactions preserve controller state" begin
    graph = graph_from_text(INTEGRATORS)
    outputs = (output=zeros(Float32, 2),)
    inputs = (input=Float32[2, 4],)
    @test process!(outputs, graph, inputs) == (output=true,)
    @test outputs.output == Float32[2, 4]
    @test replace_properties!(
        graph, "first:gain" => 0.25f0, Symbol("first:pole") => 0.75f0,
    ) === graph
    @test property_values(graph.nodes[1].prepared.plan) == (gain=0.25f0, pole=0.75f0, closed=true)
    @test leaky_integrator_state(graph.nodes[1].prepared.workspace) == Float32[2, 4]
    @test process!(outputs, graph, inputs) == (output=true,)
    @test outputs.output == Float32[2, 4]
    @test replace_properties!(graph, "first:gain" => 0.5f0) === graph
    @test process!(outputs, graph, inputs) == (output=true,)
    @test outputs.output == Float32[2.5, 5]

    plans = map(node -> node.prepared.plan, graph.nodes)
    states = map(node -> copy(leaky_integrator_state(node.prepared.workspace)), graph.nodes)
    candidate = map(node -> copy(node.prepared.workspace.candidate_state), graph.nodes)
    available = map(ports -> map(port -> port[], ports), graph.node_output_available)
    for (exception, updates) in (
        (ArgumentError, ()),
        (ArgumentError, ("first:gain" => 0.2f0, Symbol("first:gain") => 0.3f0)),
        (ArgumentError, ("unknown:gain" => 0.2f0,)),
        (ArgumentError, ("first:unknown" => 0.2f0,)),
        (ArgumentError, ("gain" => 0.2f0,)),
        (ArgumentError, (":gain" => 0.2f0,)),
        (ArgumentError, ("first:" => 0.2f0,)),
        (ArgumentError, (1 => 0.2f0,)),
        (PropertyTypeError, ("first:gain" => 0.2,)),
        (PropertyTypeError, ("first:gain" => Float32[0.2],)),
        (PropertyConstraintError, ("first:pole" => 1.1f0,)),
        (PropertyConstraintError, ("first:gain" => NaN32,)),
        (PropertyConstraintError, ("first:gain" => 0.1f0, "second:pole" => -0.1f0)),
    )
        @test_throws exception replace_properties!(graph, updates...)
        @test map(node -> node.prepared.plan, graph.nodes) === plans
        @test map(node -> leaky_integrator_state(node.prepared.workspace), graph.nodes) == states
        @test map(node -> node.prepared.workspace.candidate_state, graph.nodes) == candidate
        @test map(ports -> map(port -> port[], ports), graph.node_output_available) == available
    end
end

struct CoupledPlan <: AbstractAlgorithmPlan
    lower::Int32
    upper::Int32
    mode::UInt32
    function CoupledPlan(lower::Int32, upper::Int32, mode::UInt32)
        lower <= upper || throw(ArgumentError("lower must not exceed upper"))
        new(lower, upper, mode)
    end
end

property_schema(::Type{CoupledPlan}) = (
    runtime_property(10, :lower, Int32; default=Int32(0), description="Lower bound", unit="1"),
    runtime_property(11, :upper, Int32; default=Int32(1), description="Upper bound", unit="1"),
    read_only_property(12, :mode, UInt32; default=UInt32(7), description="Mode", unit="1"),
)
property_values(plan::CoupledPlan) = (lower=plan.lower, upper=plan.upper, mode=plan.mode)
replace_properties(::CoupledPlan, values) = CoupledPlan(values.lower, values.upper, values.mode)

mutable struct CoupledWorkspace
    committed::Float32
    candidate::Float32
    adopted::Int
end
commit!(workspace::CoupledWorkspace) = (workspace.committed = workspace.candidate; workspace)
discard!(workspace::CoupledWorkspace) = (workspace.candidate = workspace.committed; workspace)
reset!(workspace::CoupledWorkspace) = (workspace.committed = workspace.candidate = 0.0f0; workspace)
function _plan_adopted!(workspace::CoupledWorkspace)
    workspace.adopted += 1
    return discard!(workspace)
end

@algorithm CoupledBounds begin
    label = "test-coupled-bounds"
    @configuration CoupledConfiguration begin
        extent::Int => graph_rebuild
    end
    properties = :runtime
    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal/1"),
        output(:output, Float32, (configuration.extent,), "test.signal/1"),
    )
    prepare(configuration) = (
        plan=CoupledPlan(Int32(0), Int32(1), UInt32(7)),
        workspace=CoupledWorkspace(0.0f0, 0.0f0, 0),
    )
    process!(plan, workspace) = begin
        workspace.candidate = clamp(input[1], plan.lower, plan.upper)
        output[1] = workspace.candidate
        nothing
    end
end

const COUPLED = """
filter.graph = {
    nodes = [
        { type = ndarray name = first label = test-coupled-bounds config = { extent = 1 } }
        { type = ndarray name = second label = test-coupled-bounds config = { extent = 1 } }
    ]
    links = [ { output = "first:output" input = "second:input" } ]
    inputs = [ "first:input" ]
    outputs = [ "second:output" ]
}
"""

@testset "all Property candidates validate before adoption" begin
    graph = graph_from_text(COUPLED; algorithms=(CoupledBounds,))
    first = graph.nodes[1].prepared
    second = graph.nodes[2].prepared
    @test process!((output=zeros(Float32, 1),), graph, (input=Float32[0.5],)) == (output=true,)
    @test first.workspace.committed == 0.5f0
    # Individually setting lower=3 is invalid against upper=1. Grouping both
    # assignments must construct the valid final Plan only once.
    @test replace_properties!(
        graph, "first:lower" => Int32(3), "first:upper" => Int32(4),
    ) === graph
    @test property_values(first.plan) == (lower=Int32(3), upper=Int32(4), mode=UInt32(7))
    @test first.workspace.adopted == 1
    @test second.workspace.adopted == 0
    @test first.workspace.committed == first.workspace.candidate == 0.5f0

    old = (first.plan, second.plan)
    @test_throws ArgumentError replace_properties!(
        graph, "first:upper" => Int32(5), "second:lower" => Int32(10),
    )
    @test (first.plan, second.plan) === old
    @test first.workspace.adopted == 1
    @test second.workspace.adopted == 0
    @test_throws PropertyUpdateClassError replace_properties!(graph, "first:mode" => UInt32(7))
    @test first.workspace.adopted == 1
end

@testset "specialized owners reject scalar mutation without changing retained Graph" begin
    owner = mktemp() do path, io
        write(io, INTEGRATORS)
        flush(io)
        prepare_cpu_progressive_graph(path; algorithms=algorithms(), workers=0)
    end
    try
        graph = owner.graph
        old = map(node -> node.prepared.plan, graph.nodes)
        exception = try
            replace_properties!(owner, "first:gain" => 0.25f0)
        catch error
            error
        end
        @test exception isa ArgumentError
        @test occursin("prepare a new owner", sprint(showerror, exception))
        @test map(node -> node.prepared.plan, graph.nodes) === old
        @test !owner.closed
    finally
        close(owner)
    end
end

end
