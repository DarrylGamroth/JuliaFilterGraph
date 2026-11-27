using FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire, Test

function parameter_cpu_graph(layout, thread_ids)
    mktemp() do path, io
        write(io, """
        filter.graph = {
            nodes = [ { type = ndarray name = rec label = incremental-dense-reconstructor-f32
                config = { measurement_count = 4 measurement_block_capacity = 2 blocks_per_sample = 2
                    reconstructed_count = 8 initial_reconstructor = null
                    measurements_schema = test.measurements/1 column_indices_schema = test.columns/1
                    measurement_count_schema = test.count/1 reconstructed_schema = test.output/1
                    reconstructor_schema = test.matrix/1 }
            } ]
            links = [] inputs = [ "rec:measurements" "rec:columns" "rec:count" "rec:reconstructor" ]
            outputs = [ "rec:reconstructed" ]
        }
        """)
        flush(io)
        return prepare_cpu_progressive_graph(path;
            algorithms=(IncrementalDenseReconstructorF32,), workers=1,
            worker_thread_ids=thread_ids, matrix_layout=layout)
    end
end

parameter_process_bytes(y, graph, x, m) = @allocated FilterGraphAlgorithms.process!(y, graph, x, m)
parameter_adopt_bytes(owner) = @allocated FilterGraphPipeWire._adopt_pending_parameters!(owner)
parameter_metadata(sequence, offset; terminal=false) = let m=SampleMetadata(sequence, offset; terminal)
    (measurements=m, columns=m, count=m)
end

@testset "adapter adopts prepared CPU bounds and shards at a joined frame boundary" begin
    # Julia lazily creates a per-thread scheduler task on the first empty-queue
    # wait. A Channel warmup may complete without blocking, so initialize this
    # runtime state explicitly before measuring steady-state graph allocations.
    Base.get_sched_task()
    for layout in (:shared, :sharded), thread_ids in (nothing, [Threads.threadpoolsize(:interactive) + 1])
        graph = parameter_cpu_graph(layout, thread_ids)
        owner = FilterGraphPipeWire.PreparedGraphOwner(graph.graph, nothing)
        executor = FilterGraphs._cpu_dense_executor(only(graph.graph.nodes))
        tasks = copy(executor.tasks)
        @test isconcretetype(eltype(executor.tasks))
        @test eltype(executor.tasks) !== Any
        @test isnothing(thread_ids) ? eltype(executor.tasks) === Task : eltype(executor.tasks) !== Task
        first_block = (measurements=Float32[1, 2], columns=UInt32[0, 1], count=UInt32[2])
        last_block = (measurements=Float32[3, 4], columns=UInt32[2, 3], count=UInt32[2])
        outputs = (reconstructed=fill(71f0, 8),)
        try
            # Warm each exact admitted block path and the persistent workers before
            # measuring the first frame after a subsequent matrix replacement.
            FilterGraphAlgorithms.process!(outputs, graph, first_block, parameter_metadata(0, 0))
            FilterGraphAlgorithms.process!(outputs, graph, last_block, parameter_metadata(0, 1; terminal=true))
            for sequence in 1:4
                matrix = reshape(Float32.(1:32), 8, 4) .* Float32(sequence)
                old_matrix, old_layout = executor.matrix, executor.matrix_layout
                @test FilterGraphPipeWire._stage_parameter_transaction!(owner, :reconstructor => matrix)
                transaction = owner.parameter_transaction
                candidate = only(transaction.derived)
                @test executor.matrix === old_matrix
                @test executor.matrix_layout === old_layout
                @test candidate.bounds == vec(maximum(abs.(Float64.(matrix)); dims=1))
                if sequence == 1
                    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
                else
                    @test parameter_adopt_bytes(owner) == 0
                end
                @test executor.matrix === only(graph.graph.nodes).prepared.plan.matrix
                @test executor.matrix_layout === candidate.layout
                @test executor.column_bounds == candidate.bounds
                @test executor.tasks == tasks
                m1 = parameter_metadata(sequence, 0)
                m2 = parameter_metadata(sequence, 1; terminal=true)
                @test parameter_process_bytes(outputs, graph, first_block, m1) == 0
                @test parameter_process_bytes(outputs, graph, last_block, m2) == 0
                @test outputs.reconstructed == matrix * Float32[1, 2, 3, 4]
                @test executor.matrix_layout === candidate.layout
            end
            # Preparing replacement storage must leave running workers on the old matrix.
            before = copy(outputs.reconstructed)
            @test FilterGraphAlgorithms.process!(outputs, graph, first_block, parameter_metadata(5, 0)) == (reconstructed=false,)
            @test executor.active
            old_matrix, old_layout = executor.matrix, executor.matrix_layout
            matrix = fill(2f0, 8, 4)
            @test FilterGraphPipeWire._stage_parameter_transaction!(owner, :reconstructor => matrix)
            @test executor.active
            @test executor.matrix === old_matrix
            @test executor.matrix_layout === old_layout
            @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
            @test !executor.active
            @test outputs.reconstructed == before
            @test executor.tasks == tasks
            @test parameter_process_bytes(outputs, graph, first_block, parameter_metadata(6, 0)) == 0
            @test parameter_process_bytes(outputs, graph, last_block, parameter_metadata(6, 1; terminal=true)) == 0
            @test outputs.reconstructed == fill(20f0, 8)
            # Cold diagnostics preserve Julia's arbitrary thrown-value semantics
            # through both the atomic mailbox and the coordinator's throw site.
            for cause in (ErrorException("injected worker fault"), :worker_fault, nothing)
                failure = try
                    throw(cause)
                catch error
                    FilterGraphs.CpuWorkerFailure(error, catch_backtrace())
                end
                @test !isempty(failure.backtrace)
                handoff = FilterGraphs._cpu_handoff([Threads.threadpoolsize(:interactive) + 1], 1)
                handoff.epoch = UInt64(1)
                FilterGraphs._cpu_publish_completion!(handoff, 1, UInt64(1), failure)
                @test FilterGraphs._cpu_wait_completion!(handoff, 1, executor).cause === cause
                executor.fault = failure
                caught = try
                    FilterGraphs._append_cpu_dense_rows!(executor, executor.matrix,
                        first_block.measurements, first_block.columns, 2)
                    (thrown=false, value=nothing)
                catch error
                    (thrown=true, value=error)
                finally
                    executor.fault = nothing
                end
                @test caught.thrown
                @test caught.value === cause
            end
        finally
            close(graph)
        end
        @test all(istaskdone, tasks)
    end
end

@testset "CPU progressive Graph status publication" begin
    graph = parameter_cpu_graph(:shared, nothing)
    calls = Ref(0)
    try
        node = PipeWireNode(graph; connect=false, rate=(1_000, 1), property_info=[status_info()],
            property_getter=(props, available) -> begin
                calls[] += 1
                SPA.Props("application:ready" => available)
            end)
        try
            @test calls[] == 1
            @test pod_value(Bool, last(node.filter.callbacks.on_get_properties(node.filter).values).second)
            @test calls[] == 2
        finally
            close(node)
        end
    finally
        close(graph)
    end
end
