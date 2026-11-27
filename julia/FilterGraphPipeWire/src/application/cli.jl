const USAGE = """
usage: julia --project=julia/FilterGraphPipeWire -m FilterGraphPipeWire [options]

Required:
  --graph PATH                 PipeWire filter.graph SPA-JSON configuration

Options:
  --name NAME                  Published Node name (default: julia.filter-graph-island)
  --remote NAME                PipeWire remote (default: PIPEWIREAO_REMOTE or native default)
  --rate NUM[/DEN]             Graph base rate, for example 1000 or 1000/1
  --channel-unit PORT CHANNEL UNITS
                               Declare source units for a named output channel (repeat)
  --boundary-layout LAYOUT     Published layout: graph, row-major, or column-major
  --fifo-inputs                Preserve every queued progressive frame-data block
  --session-run-control        Start stopped and accept RTC owner-control requests
  --algorithm-package PACKAGE  Add PACKAGE.algorithms(); may be repeated
  --algorithm MODULE.TYPE      Add an exported Algorithm declaration (repeat)
  --no-builtins                Do not include FilterGraphAlgorithms.algorithms()
  --parameter PORT TYPE DIMS PATH
                               Bind a little-endian row-major array before publication
                               TYPE: Float32, UInt16, UInt32, UInt8, Bool, etc.; DIMS: 2,3
  --feedback INPUT OUTPUT      Retain successful-frame feedback (one pair supported)
  --warmup-output NAME         Opt into synthetic validation of this output (repeat)
  --execution MODE             complete-frame (default) or row-block
  --row-workers N              Persistent row workers (default: 0)
  --matrix-layout LAYOUT       shared (default) or sharded row-worker matrix
  --pin-cpus LIST              One CPU per default Julia thread, excluding CPUs 0/1
                               --threads=N,0: thread 1 main, 2 monitors, 3+ workers
  --capture-backend BACKEND    Prepare native capture for amdgpu or cuda
  --quit-request PATH          Call quit! when PATH appears
  --connect-request PATH       Prepare disconnected; connect when PATH appears
  --prepared-event PATH        Create PATH after disconnected preparation
  --connect-reply PATH         Create PATH after connect! succeeds
  --check                      Prepare and validate without connecting
  -h, --help                   Show this help
"""

function usage_error(message::AbstractString)
    throw(ArgumentError("$message\n\n$USAGE"))
end

function option_value(args, index::Int, option::AbstractString)
    argument = args[index]
    prefix = "$option="
    if startswith(argument, prefix)
        value = argument[(lastindex(prefix) + 1):end]
        isempty(value) && usage_error("$option requires a value")
        return value, index
    end
    index == length(args) && usage_error("$option requires a value")
    return args[index + 1], index + 1
end

function parse_rate(text::AbstractString)
    terms = split(text, '/'; keepempty=true)
    length(terms) in (1, 2) || usage_error("--rate must be NUM or NUM/DEN")
    numerator = tryparse(Int, terms[1])
    denominator = length(terms) == 1 ? 1 : tryparse(Int, terms[2])
    isnothing(numerator) && usage_error("--rate numerator must be an integer")
    isnothing(denominator) && usage_error("--rate denominator must be an integer")
    numerator > 0 && denominator > 0 || usage_error("--rate must be positive")
    return numerator // denominator
end

function parse_boundary_layout(text::AbstractString)
    text == "graph" && return nothing
    text == "row-major" && return FilterGraphAlgorithms.RowMajorLayout()
    text == "column-major" && return FilterGraphAlgorithms.ColumnMajorLayout()
    usage_error("--boundary-layout must be graph, row-major, or column-major")
end

"""
    parse_arguments(args)

Validate launcher arguments and return the graph, execution, startup parameter,
and lifecycle options. Paths are made absolute and startup payload extents are
checked before graph preparation. `--help` prints usage and exits.
"""
function parse_arguments(args)
    graph = nothing
    name = "julia.filter-graph-island"
    remote = get(ENV, "PIPEWIREAO_REMOTE", nothing)
    rate = nothing
    boundary_layout = nothing
    channel_units = Dict{String,Dict{String,String}}()
    packages = String[]
    algorithms = String[]
    builtins = true
    capture_backend = nothing
    quit_request = get(ENV, "JULIA_FILTER_GRAPH_QUIT_REQUEST", nothing)
    connect_request = nothing
    prepared_event = nothing
    connect_reply = nothing
    check = false
    session_run_control = false
    fifo_inputs = false
    parameters = StartupParameter[]
    feedback = nothing
    warmup_outputs = Symbol[]
    execution = "complete-frame"
    row_workers = 0
    matrix_layout = :shared
    pin_cpus = nothing
    row_options = false

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "-h" || argument == "--help"
            println(USAGE)
            exit(0)
        elseif argument == "--check"
            check = true
        elseif argument == "--session-run-control"
            session_run_control = true
        elseif argument == "--fifo-inputs"
            fifo_inputs = true
        elseif argument == "--no-builtins"
            builtins = false
        elseif argument == "--parameter"
            index + 4 <= length(args) || usage_error(
                "--parameter requires PORT TYPE DIMS PATH",
            )
            parameter = startup_parameter(args[(index + 1):(index + 4)]...)
            any(value -> value.port === parameter.port, parameters) &&
                usage_error("Parameter Port $(parameter.port) is assigned more than once")
            push!(parameters, parameter)
            index += 4
        elseif argument == "--channel-unit"
            index + 3 <= length(args) || usage_error("--channel-unit requires PORT CHANNEL UNITS")
            port, channel, units = args[(index + 1):(index + 3)]
            all(value -> isvalid(value) && !isempty(strip(value)) && ncodeunits(value) <= 255 &&
                !any(iscntrl, value), (port, channel, units)) ||
                usage_error("--channel-unit values must be valid UTF-8, nonempty bounded text without controls")
            occursin(r"^[a-z][a-z0-9_]*$", channel) || usage_error("invalid --channel-unit channel name")
            channels = get!(channel_units, port, Dict{String,String}())
            haskey(channels, channel) && usage_error("--channel-unit $port $channel is repeated")
            channels[channel] = units
            length(channel_units) <= 64 && length(channels) <= 64 ||
                usage_error("--channel-unit exceeds 64 Ports or channels")
            index += 3
        elseif argument == "--feedback"
            index + 2 <= length(args) || usage_error("--feedback requires INPUT OUTPUT")
            isnothing(feedback) || usage_error(
                "FilterGraphPipeWire supports only one feedback pair",
            )
            all(!isempty, args[(index + 1):(index + 2)]) ||
                usage_error("--feedback Port names must not be empty")
            feedback = Symbol(args[index + 1]) => Symbol(args[index + 2])
            index += 2
        elseif argument == "--warmup-output" || startswith(argument, "--warmup-output=")
            value, index = option_value(args, index, "--warmup-output")
            isempty(value) && usage_error("--warmup-output must not be empty")
            output_name = Symbol(value)
            output_name in warmup_outputs && usage_error("--warmup-output $output_name is repeated")
            push!(warmup_outputs, output_name)
        elseif argument == "--execution" || startswith(argument, "--execution=")
            execution, index = option_value(args, index, "--execution")
            execution in ("complete-frame", "row-block") ||
                usage_error("--execution must be complete-frame or row-block")
        elseif argument == "--row-workers" || startswith(argument, "--row-workers=")
            value, index = option_value(args, index, "--row-workers")
            selected = tryparse(Int, value)
            !isnothing(selected) && selected >= 0 ||
                usage_error("--row-workers must be a nonnegative integer")
            row_workers = selected
            row_options = true
        elseif argument == "--matrix-layout" || startswith(argument, "--matrix-layout=")
            value, index = option_value(args, index, "--matrix-layout")
            value in ("shared", "sharded") ||
                usage_error("--matrix-layout must be shared or sharded")
            matrix_layout = Symbol(value)
            row_options = true
        elseif argument == "--pin-cpus" || startswith(argument, "--pin-cpus=")
            value, index = option_value(args, index, "--pin-cpus")
            pin_cpus = parse_pin_cpus(value)
        elseif argument == "--capture-backend" ||
               startswith(argument, "--capture-backend=")
            capture_backend, index = option_value(args, index, "--capture-backend")
            capture_backend in ("amdgpu", "cuda") || usage_error(
                "--capture-backend must be amdgpu or cuda",
            )
        elseif argument == "--graph" || startswith(argument, "--graph=")
            graph, index = option_value(args, index, "--graph")
        elseif argument == "--name" || startswith(argument, "--name=")
            name, index = option_value(args, index, "--name")
        elseif argument == "--remote" || startswith(argument, "--remote=")
            remote, index = option_value(args, index, "--remote")
        elseif argument == "--rate" || startswith(argument, "--rate=")
            value, index = option_value(args, index, "--rate")
            rate = parse_rate(value)
        elseif argument == "--boundary-layout" ||
               startswith(argument, "--boundary-layout=")
            value, index = option_value(args, index, "--boundary-layout")
            boundary_layout = parse_boundary_layout(value)
        elseif argument == "--quit-request" || startswith(argument, "--quit-request=")
            quit_request, index = option_value(args, index, "--quit-request")
        elseif argument == "--connect-request" ||
               startswith(argument, "--connect-request=")
            connect_request, index = option_value(args, index, "--connect-request")
        elseif argument == "--prepared-event" ||
               startswith(argument, "--prepared-event=")
            prepared_event, index = option_value(args, index, "--prepared-event")
        elseif argument == "--connect-reply" || startswith(argument, "--connect-reply=")
            connect_reply, index = option_value(args, index, "--connect-reply")
        elseif argument == "--algorithm-package" ||
               startswith(argument, "--algorithm-package=")
            value, index = option_value(args, index, "--algorithm-package")
            push!(packages, value)
        elseif argument == "--algorithm" || startswith(argument, "--algorithm=")
            value, index = option_value(args, index, "--algorithm")
            algorithm_identifier(value)
            push!(algorithms, value)
        else
            usage_error("unknown argument: $argument")
        end
        index += 1
    end

    isnothing(graph) && usage_error("--graph is required")
    isempty(name) && usage_error("--name must not be empty")
    !isnothing(remote) && isempty(remote) && usage_error("--remote must not be empty")
    builtins || !isempty(packages) || !isempty(algorithms) || usage_error(
        "--no-builtins requires --algorithm-package or --algorithm",
    )
    row_options && execution != "row-block" &&
        usage_error("--row-workers and --matrix-layout require --execution row-block")
    !isnothing(capture_backend) && execution != "complete-frame" &&
        usage_error("--execution row-block is not supported with --capture-backend")
    !isnothing(capture_backend) && !isempty(warmup_outputs) &&
        usage_error("--warmup-output is not supported with --capture-backend")
    !isnothing(capture_backend) && !isnothing(feedback) &&
        usage_error("--feedback is not supported with --capture-backend")
    !isnothing(capture_backend) && !isnothing(boundary_layout) && usage_error(
        "--boundary-layout is not supported with --capture-backend",
    )
    !isnothing(capture_backend) && fifo_inputs && usage_error(
        "--fifo-inputs is not supported with --capture-backend",
    )
    graph_path = abspath(graph)
    isfile(graph_path) || usage_error("Graph configuration does not exist: $graph_path")
    quit_request_path = isnothing(quit_request) ? nothing : abspath(quit_request)
    connect_request_path =
        isnothing(connect_request) ? nothing : abspath(connect_request)
    prepared_event_path = isnothing(prepared_event) ? nothing : abspath(prepared_event)
    connect_reply_path = isnothing(connect_reply) ? nothing : abspath(connect_reply)
    for (label, path) in (
        ("Quit request", quit_request_path),
        ("Connect request", connect_request_path),
        ("Prepared event", prepared_event_path),
        ("Connect reply", connect_reply_path),
    )
        !isnothing(path) && ispath(path) && usage_error("$label already exists: $path")
    end
    if isnothing(connect_request_path)
        isnothing(prepared_event_path) ||
            usage_error("--prepared-event requires --connect-request")
        isnothing(connect_reply_path) ||
            usage_error("--connect-reply requires --connect-request")
    else
        check && usage_error("--connect-request cannot be combined with --check")
        isnothing(quit_request_path) &&
            usage_error("--connect-request requires --quit-request")
    end
    return (;
        graph=graph_path,
        name,
        remote,
        rate,
        boundary_layout,
        channel_units,
        packages=Tuple(packages),
        algorithms=Tuple(algorithms),
        builtins,
        capture_backend,
        quit_request=quit_request_path,
        connect_request=connect_request_path,
        prepared_event=prepared_event_path,
        connect_reply=connect_reply_path,
        check,
        session_run_control,
        fifo_inputs,
        parameters=Tuple(parameters),
        feedback,
        warmup_outputs=Tuple(warmup_outputs),
        execution,
        row_workers,
        matrix_layout,
        pin_cpus,
    )
end
