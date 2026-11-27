module GraphCommandEndpointTests
using Test, FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire, PipeWireAO

@testset "Graph owner command endpoint dispatches outside frame processing" begin
    graph = prepare_graph(joinpath(@__DIR__, "data", "leaky-graph.conf");
        algorithms=(LeakyIntegratorF32,))
    node = PipeWireNode(graph; connect=false)
    received = Tuple{Int,Int}[]
    receiver = nothing
    try
        receiver = FilterGraphPipeWire.command_endpoint(node,
            message -> push!(received, (message, Threads.threadid())), Int; capacity=1)
        sender = PipeWireAO.commands(receiver)
        @test PipeWireAO.try_send!(sender, 7) == PipeWireAO.COMMAND_ACCEPTED
        @test PipeWireAO.try_send!(sender, 8) == PipeWireAO.COMMAND_FULL
        @test isempty(received)
        native = PipeWireAO.LibPipeWire
        loop = native.pw_ndarray_filter_get_loop(node.filter.handle)
        GC.@preserve node receiver begin
            native.pw_loop_enter(loop)
            try
                @test native.pw_loop_iterate(loop, 0) >= 0
            finally
                native.pw_loop_leave(loop)
            end
        end
        @test received == [(7, Threads.threadid())]
        @test PipeWireAO.check_command_error(receiver) === nothing
        close(receiver)
        @test PipeWireAO.try_send!(sender, 9) == PipeWireAO.COMMAND_CLOSED
    finally
        receiver === nothing || close(receiver)
        close(node)
    end
end
end
