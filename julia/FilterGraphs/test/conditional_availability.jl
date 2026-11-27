import FilterGraphAlgorithms: commit!, discard!

mutable struct PairAccumulatorWorkspace
    first_piece::Vector{Float32}
    pending::Bool
end

commit!(workspace::PairAccumulatorWorkspace) = workspace

function discard!(workspace::PairAccumulatorWorkspace)
    workspace.pending = false
    return workspace
end

function reset!(workspace::PairAccumulatorWorkspace)
    fill!(workspace.first_piece, 0.0f0)
    workspace.pending = false
    return workspace
end

@algorithm PairAccumulatorF32 begin
    label = "pair-accumulator-f32"

    @configuration PairAccumulatorF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :piece,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            sample=:piece_sample,
        ),
        output(
            :aggregate,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:piece,
            publication=:conditional,
        ),
    )

    prepare(configuration) = (
        plan=configuration.extent,
        workspace=PairAccumulatorWorkspace(zeros(Float32, configuration.extent), false),
    )

    process!(plan, workspace) = begin
        if !workspace.pending
            copyto!(workspace.first_piece, piece)
            workspace.pending = true
            return AlgorithmProcessDeferred()
        end
        @. aggregate = workspace.first_piece + piece
        workspace.pending = false
        isnothing(piece_sample) && return nothing
        return AlgorithmProcessComplete(
            SampleMetadata(piece_sample; offset=0, terminal=true),
        )
    end
end

@enum ConditionalConsumerFailure::UInt8 begin
    ConditionalConsumerRejected = 1
end

@algorithm ConditionalConsumerF32 begin
    label = "conditional-consumer-f32"
    failure = ConditionalConsumerFailure

    @configuration ConditionalConsumerF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:aggregate, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :consumed,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:aggregate,
        ),
    )

    prepare(configuration) = configuration.extent

    process!(plan) = begin
        aggregate[1] < 0.0f0 && return ConditionalConsumerRejected
        copyto!(consumed, aggregate)
        return nothing
    end
end

mutable struct PreviewWorkspace
    committed::Vector{Float32}
    candidate::Vector{Float32}
end

function commit!(workspace::PreviewWorkspace)
    copyto!(workspace.committed, workspace.candidate)
    return workspace
end

function discard!(workspace::PreviewWorkspace)
    copyto!(workspace.candidate, workspace.committed)
    return workspace
end

function reset!(workspace::PreviewWorkspace)
    fill!(workspace.committed, 0.0f0)
    fill!(workspace.candidate, 0.0f0)
    return workspace
end

@algorithm PreviewF32 begin
    label = "preview-f32"

    @configuration PreviewF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:monitor, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :preview,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:monitor,
        ),
    )

    prepare(configuration) = (
        plan=configuration.extent,
        workspace=PreviewWorkspace(
            zeros(Float32, configuration.extent),
            zeros(Float32, configuration.extent),
        ),
    )

    process!(plan, workspace) = begin
        @. workspace.candidate = 2.0f0 * monitor
        copyto!(preview, workspace.candidate)
        return nothing
    end
end

const CONDITIONAL_GRAPH_FIXTURE =
    joinpath(@__DIR__, "data", "conditional-branches.conf")

@testset "per-Port conditional availability" begin
    graph = prepare_graph(
        CONDITIONAL_GRAPH_FIXTURE;
        algorithms=(PairAccumulatorF32, ConditionalConsumerF32, PreviewF32),
    )
    @test graph.node_failure_types == (Union{}, ConditionalConsumerFailure, Union{})
    @test graph.output_conditional == (consumed=true, preview=false)
    @test @inferred(
        FilterGraphs._unavailable_external_outputs(values(graph.output_formats)),
    ) == (false, false)
    workspace = graph.nodes[1].prepared.workspace
    preview_workspace = graph.nodes[3].prepared.workspace
    consumed = fill(91.0f0, 3)
    preview = fill(92.0f0, 3)
    outputs = (consumed=consumed, preview=preview)

    first_inputs = (
        piece=Float32[1, 2, 3],
        monitor=Float32[10, 20, 30],
    )
    @test process!(outputs, graph, first_inputs) == (consumed=false, preview=false)
    @test consumed == fill(91.0f0, 3)
    @test preview == fill(92.0f0, 3)
    @test workspace.pending
    @test graph.node_output_available[3][1][]
    @test preview_workspace.committed == Float32[20, 40, 60]
    @test preview_workspace.candidate == Float32[20, 40, 60]

    second_inputs = (
        piece=Float32[4, 5, 6],
        monitor=Float32[7, 8, 9],
    )
    @test process!(outputs, graph, second_inputs) == (consumed=true, preview=true)
    @test consumed == Float32[5, 7, 9]
    @test preview == Float32[14, 16, 18]
    @test !workspace.pending
    @test preview_workspace.committed == preview

    reset!(graph)
    rejected_first = (
        piece=Float32[-5, 0, 0],
        monitor=Float32[1, 1, 1],
    )
    rejected_second = (
        piece=Float32[1, 0, 0],
        monitor=Float32[2, 2, 2],
    )
    @test process!(outputs, graph, rejected_first) == (consumed=false, preview=false)
    @test preview == Float32[14, 16, 18]
    fill!(consumed, 81.0f0)
    fill!(preview, 82.0f0)
    @test process!(outputs, graph, rejected_second) === ConditionalConsumerRejected
    @test consumed == fill(81.0f0, 3)
    @test preview == fill(82.0f0, 3)
    @test !workspace.pending
    @test preview_workspace.committed == Float32[2, 2, 2]
    @test preview_workspace.candidate == Float32[2, 2, 2]

    fresh_inputs = (
        piece=Float32[2, 2, 2],
        monitor=Float32[3, 3, 3],
    )
    @test process!(outputs, graph, fresh_inputs) == (consumed=false, preview=false)
    @test workspace.pending

    reset!(graph)
    process!(outputs, graph, first_inputs)
    @test graph_allocation_bytes(outputs, graph, second_inputs) == 0
    reset!(graph)
    @test graph_allocation_bytes(outputs, graph, first_inputs) == 0
    reset!(graph)
    process!(outputs, graph, rejected_first)
    @test graph_allocation_bytes(outputs, graph, rejected_second) == 0

    reset!(graph)
    piece_first = SampleMetadata(71, 0; timestamp_nanoseconds=100)
    monitor_first = SampleMetadata(81, 0; timestamp_nanoseconds=101, terminal=true)
    first_metadata = (piece=piece_first, monitor=monitor_first)
    retained_metadata = FilterGraphs.output_metadata(graph)
    @test process!(outputs, graph, first_inputs, first_metadata) ==
          (consumed=false, preview=false)
    @test FilterGraphs.output_metadata(graph) == retained_metadata

    piece_last = SampleMetadata(71, 1; timestamp_nanoseconds=200, terminal=true)
    monitor_last = SampleMetadata(82, 0; timestamp_nanoseconds=201, terminal=true)
    second_metadata = (piece=piece_last, monitor=monitor_last)
    @test process!(outputs, graph, second_inputs, second_metadata) ==
          (consumed=true, preview=true)
    expected_aggregate_metadata = SampleMetadata(piece_last; offset=0, terminal=true)
    @test FilterGraphs.output_metadata(graph) == (
        consumed=expected_aggregate_metadata,
        preview=monitor_last,
    )
    @test @allocated(FilterGraphs.output_metadata(graph)) == 0
    @test_throws ArgumentError process!(
        outputs,
        graph,
        second_inputs,
        (wrong=piece_last, monitor=monitor_last),
    )
    @test_throws ArgumentError process!(
        outputs,
        graph,
        second_inputs,
        (piece=41, monitor=monitor_last),
    )

    reset!(graph)
    process!(outputs, graph, first_inputs, first_metadata)
    @test @allocated(process!(outputs, graph, second_inputs, second_metadata)) == 0

    reset!(graph)
    @test @allocated(process!(outputs, graph, rejected_first, first_metadata)) == 0
    retained_metadata = FilterGraphs.output_metadata(graph)
    @test process!(outputs, graph, rejected_second, second_metadata) ===
          ConditionalConsumerRejected
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    reset!(graph)
    process!(outputs, graph, rejected_first, first_metadata)
    @test @allocated(process!(outputs, graph, rejected_second, second_metadata)) == 0
end

@testset "unexported deferred branch holds an unconditional export" begin
    nodes = (
        (name=:pair, label="pair-accumulator-f32", config=(extent=2,)),
        (name=:preview, label="preview-f32", config=(extent=2,)),
    )
    graph = prepare_graph((; nodes, inputs=(piece="pair:piece", monitor="preview:monitor"),
        outputs=(preview="preview:preview",)); algorithms=(PairAccumulatorF32, PreviewF32))
    @test graph.output_conditional == (preview=false,)
    inputs = (piece=ones(Float32, 2), monitor=Float32[3, 4])
    outputs = (preview=fill(71f0, 2),)
    @test process!(outputs, graph, inputs) == (preview=false,)
    @test outputs.preview == [71, 71]
    @test graph.nodes[2].prepared.workspace.committed == [6, 8]
    @test process!(outputs, graph, inputs) == (preview=true,)
    @test outputs.preview == [6, 8]
end
