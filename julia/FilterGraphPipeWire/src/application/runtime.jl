function wait_for_connect_request(
    connect_request::AbstractString,
    quit_request::AbstractString,
)
    while !isfile(connect_request) && !isfile(quit_request)
        GC.safepoint()
        deployment_sleep(0.01)
    end
    if isfile(quit_request)
        println("JULIA_FILTER_GRAPH_QUIT_REQUESTED path=$quit_request")
        flush(stdout)
        return false
    end
    return true
end

function signal_event(path::Union{Nothing,AbstractString})
    isnothing(path) || touch(path)
    return nothing
end

function monitor_quit_request(node, quit_request::AbstractString, finished)
    return spawn_monitor() do
        while !finished[] && !isfile(quit_request)
            GC.safepoint()
            deployment_sleep(0.05)
        end
        if !finished[]
            FilterGraphPipeWire.quit!(node)
            println("JULIA_FILTER_GRAPH_QUIT_REQUESTED path=$quit_request")
            flush(stdout)
        end
        nothing
    end
end

function start_ready_monitor(node, options, finished)
    return spawn_monitor() do
        while !finished[]
            state = FilterGraphPipeWire.node_state(node)
            if PipeWireAO.isrunning(node) &&
               (state == PipeWireAO.NDARRAY_FILTER_STATE_PAUSED ||
                state == PipeWireAO.NDARRAY_FILTER_STATE_STREAMING)
                remote = something(options.remote, "default")
                println(
                    "JULIA_FILTER_GRAPH_READY name=$(options.name) " *
                    "remote=$remote graph=$(options.graph) state=$state",
                )
                flush(stdout)
                return nothing
            end
            state == PipeWireAO.NDARRAY_FILTER_STATE_ERROR && return nothing
            GC.safepoint()
            deployment_sleep(0.001)
        end
        return nothing
    end
end

function run_service(options)
    declarations = _algorithm_declarations(options)
    selected_worker_ids = configure_threads(options)
    graph = _prepare_service_graph(options; declarations, worker_thread_ids=selected_worker_ids)
    try
        Base.invokelatest(run_prepared_service, options, declarations, graph)
    finally
        close_execution(graph)
    end
    return nothing
end

function run_prepared_service(options, declarations, graph)
    execution = something(options.capture_backend, options.execution)
    common_options = (;
        name=options.name,
        remote=options.remote,
        rate=options.rate,
        connect=false,
        run_control=options.session_run_control,
        channel_units=options.channel_units,
    )
    node = if isnothing(options.capture_backend)
        FilterGraphPipeWire.PipeWireNode(
            graph;
            common_options...,
            boundary_layout=options.boundary_layout,
            fifo_inputs=options.fifo_inputs,
            feedback=options.feedback,
        )
    else
        FilterGraphPipeWire.PipeWireNode(graph; common_options...)
    end

    try
        if isnothing(options.capture_backend) && !isempty(options.warmup_outputs)
            FilterGraphPipeWire.prepare!(node;
                execution=options.execution == "row-block" ? :row_block : :complete_frame,
                required_outputs=options.warmup_outputs,
            )
        end
        run_connected_service(node, options, declarations, execution)
    finally
        close(node)
        if hasproperty(node.owner, :callback_metrics)
            FilterGraphPipeWire._write_callback_metrics(node.owner.callback_metrics)
        end
    end
    return nothing
end

function run_connected_service(node, options, declarations, execution)
    if options.check
        println(
            "JULIA_FILTER_GRAPH_CHECKED graph=$(options.graph) " *
            "algorithms=$(length(declarations)) " *
            "execution=$execution",
        )
        return nothing
    end

    if !isnothing(options.connect_request)
        signal_event(options.prepared_event)
        println(
            "JULIA_FILTER_GRAPH_PREPARED name=$(options.name) " *
            "graph=$(options.graph) execution=$execution",
        )
        flush(stdout)
        if !wait_for_connect_request(options.connect_request, options.quit_request)
            return nothing
        end
    end
    FilterGraphPipeWire.connect!(node)

    remote = something(options.remote, "default")
    signal_event(options.connect_reply)
    println(
        "JULIA_FILTER_GRAPH_CONNECT_ACCEPTED name=$(options.name) " *
        "remote=$remote graph=$(options.graph) " *
        "execution=$execution",
    )
    flush(stdout)
    finished = Threads.Atomic{Bool}(false)
    ready_monitor = start_ready_monitor(node, options, finished)
    quit_monitor = if isnothing(options.quit_request)
        nothing
    else
        monitor_quit_request(node, options.quit_request, finished)
    end
    try
        FilterGraphPipeWire.run!(node)
    finally
        finished[] = true
        wait(ready_monitor)
        isnothing(quit_monitor) || wait(quit_monitor)
    end
    return nothing
end
