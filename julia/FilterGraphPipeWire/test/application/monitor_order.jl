using Test
import FilterGraphPipeWire
using FilterGraphPipeWire: monitor_quit_request

struct QuitOrderNode
    requested::Threads.Atomic{Bool}
end

FilterGraphPipeWire.quit!(node::QuitOrderNode) = (node.requested[] = true; node)

@testset "quit request precedes monitor logging" begin
    mktempdir() do directory
        marker = joinpath(directory, "quit")
        touch(marker)
        node = QuitOrderNode(Threads.Atomic{Bool}(false))
        finished = Threads.Atomic{Bool}(false)
        monitor = nothing
        lock(stdout)
        try
            monitor = monitor_quit_request(node, marker, finished)
            deadline = time() + 2
            while !node.requested[] && time() < deadline
                sleep(0.001)
            end
            # The monitor cannot acquire stdout while this task owns its lock.
            # Its public quit request must still run before it tries to log.
            @test node.requested[]
        finally
            unlock(stdout)
            finished[] = true
            isnothing(monitor) || wait(monitor)
        end
    end
end
