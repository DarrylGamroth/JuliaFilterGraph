# Generic startup preparation. Scientific operations and parameter validation
# remain with the declared Algorithms and FilterGraphs.
const MAX_STARTUP_PARAMETER_BYTES = 512 * 1024 * 1024
const STARTUP_ELEMENT_TYPES = Dict(
    "Bool" => Bool, "Int8" => Int8, "UInt8" => UInt8,
    "Int16" => Int16, "UInt16" => UInt16, "Int32" => Int32,
    "UInt32" => UInt32, "Int64" => Int64, "UInt64" => UInt64,
    "Float32" => Float32, "Float64" => Float64,
    "F32" => Float32, "F64" => Float64, "U8" => UInt8,
    "U16" => UInt16, "U32" => UInt32, "U64" => UInt64,
    "I8" => Int8, "I16" => Int16, "I32" => Int32, "I64" => Int64,
)

struct StartupParameter
    port::Symbol
    element_type::DataType
    dimensions::Tuple{Vararg{Int}}
    path::String
    elements::Int
    bytes::Int
end

function startup_parameter(port, type_name, dimensions, path)
    isempty(port) && usage_error("--parameter Port name must not be empty")
    element_type = get(STARTUP_ELEMENT_TYPES, type_name, nothing)
    isnothing(element_type) && usage_error("Unsupported --parameter element type: $type_name")
    terms = split(dimensions, ','; keepempty=true)
    1 <= length(terms) <= 64 || usage_error("--parameter dimensions require rank 1:64")
    dims = map(terms) do term
        value = tryparse(Int, term)
        !isnothing(value) && value > 0 ||
            usage_error("--parameter dimensions must be positive integers separated by commas")
        value
    end
    elements, bytes = try
        count = foldl(Base.checked_mul, dims; init=1)
        count, Base.checked_mul(count, sizeof(element_type))
    catch error
        error isa OverflowError || rethrow()
        usage_error("--parameter byte extent overflows Int")
    end
    bytes <= MAX_STARTUP_PARAMETER_BYTES || usage_error(
        "--parameter payload exceeds $MAX_STARTUP_PARAMETER_BYTES bytes",
    )
    parameter_path = abspath(path)
    isfile(parameter_path) || usage_error("Parameter file does not exist: $parameter_path")
    filesize(parameter_path) == bytes || usage_error(
        "Parameter file $parameter_path must contain exactly $bytes bytes",
    )
    return StartupParameter(Symbol(port), element_type, Tuple(dims), parameter_path, elements, bytes)
end

function read_startup_parameter(parameter::StartupParameter)
    # Recheck after opening: reject changed file lengths before allocating.
    values = open(parameter.path, "r") do io
        stat(io).size == parameter.bytes || throw(DimensionMismatch(
            "Parameter file $(parameter.path) changed byte extent after admission",
        ))
        T = parameter.element_type === Bool ? UInt8 : parameter.element_type
        data = Vector{T}(undef, parameter.elements)
        read!(io, data)
        eof(io) || throw(DimensionMismatch("Parameter file grew during preparation"))
        data
    end
    if parameter.element_type === Bool
        all(value -> value <= 1, values) || throw(ArgumentError(
            "Bool parameter $(parameter.port) requires bytes 0 or 1",
        ))
        values = Bool.(values)
    elseif parameter.element_type === Float32 || parameter.element_type === Float64
        words = reinterpret(parameter.element_type === Float32 ? UInt32 : UInt64, values)
        map!(ltoh, words, words)
    else
        map!(ltoh, values, values)
    end
    # Files use the same last-index-contiguous order as standard ndarray
    # artifacts. Algorithms receive ordinary Julia arrays with logical axes.
    length(parameter.dimensions) == 1 && return values
    axes_order = reverse(ntuple(identity, length(parameter.dimensions)))
    return permutedims(reshape(values, reverse(parameter.dimensions)), axes_order)
end

function parse_pin_cpus(text)
    cpus = map(split(text, ','; keepempty=true)) do term
        cpu = tryparse(Int, term)
        !isnothing(cpu) && cpu >= 2 ||
            usage_error("--pin-cpus requires CPU IDs >= 2 separated by commas")
        cpu
    end
    return cpus
end

"""
    configure_threads(options)

Validate and optionally pin the default Julia threads. Reserve thread 1 for the
main task, thread 2 for monitors, and threads 3 onward for row workers. Return the
worker thread IDs, or `nothing` when no row workers are requested.
"""
function configure_threads(options)
    ids = FilterGraphs.ThreadPinning.threadids(; threadpool=:default)
    if !isnothing(options.pin_cpus) || options.row_workers > 0
        ids == collect(1:length(ids)) || throw(ArgumentError(
            "Placed deployment threads require --threads=N,0",
        ))
        length(ids) >= options.row_workers + 2 || throw(ArgumentError(
            "Deployment requires one main thread, one monitor thread and $(options.row_workers) worker threads",
        ))
    elseif !options.check
        length(ids) >= 2 || throw(ArgumentError(
            "Live deployment requires at least two default Julia threads (--threads=2,0)",
        ))
    end
    if !isnothing(options.pin_cpus)
        length(options.pin_cpus) == length(ids) || throw(ArgumentError(
            "--pin-cpus must list one CPU per default Julia thread",
        ))
        for (threadid, cpu) in zip(ids, options.pin_cpus)
            FilterGraphs.ThreadPinning.pinthread(cpu; threadid)
            actual = findall(!iszero, FilterGraphs.ThreadPinning.getaffinity(; threadid)) .- 1
            actual == [cpu] || error("Julia thread $threadid was not pinned to CPU $cpu")
        end
    end
    return options.row_workers == 0 ? nothing : collect(3:(options.row_workers + 2))
end

function spawn_monitor(callback)
    ids = FilterGraphs.ThreadPinning.threadids(; threadpool=:default)
    length(ids) >= 2 || throw(ArgumentError("A deployment monitor requires a second default Julia thread"))
    return FilterGraphs.ThreadPinning.@spawnat ids[2] callback()
end

function deployment_sleep(seconds::Real)
    microseconds = round(Cuint, seconds * 1e6)
    @ccall gc_safe=true usleep(microseconds::Cuint)::Cint
    # Both readiness and quit monitoring share the reserved monitor thread.
    yield()
    return nothing
end

"""
    prepare_service_graph(path; algorithms=FilterGraphAlgorithms.algorithms(),
        execution=:complete_frame, workers=0, matrix_layout=:shared,
        worker_thread_ids=nothing, initial_parameters=(), parameters=(),
        capture_backend=nothing)

Prepare a graph without publishing a PipeWire node. `execution` selects
`:complete_frame` or `:row_block`; `capture_backend` is `nothing`, `:cuda`, or
`:amdgpu`. Workers and matrix layout apply to row-block execution. The caller
owns the returned execution and releases it with [`close_execution`](@ref).

`initial_parameters` supplies `Symbol => Array` pairs in logical Julia axes,
or a function of the prepared parameter formats returning those pairs. Values
are validated and copied before GPU capture. `parameters` accepts the checked
raw-file descriptors returned by [`parse_arguments`](@ref); overlapping or
repeated bindings fail before adoption. Optional callback metrics initialize
only when explicitly enabled through their environment configuration.
"""
function prepare_service_graph(path::AbstractString;
        algorithms=FilterGraphAlgorithms.algorithms(), execution=:complete_frame,
        workers=0, matrix_layout=:shared, worker_thread_ids=nothing,
        initial_parameters=(), parameters=(), capture_backend=nothing)
    return _prepare_service_graph(path; declarations=algorithms, execution,
        workers, matrix_layout, worker_thread_ids, initial_parameters, parameters,
        capture_backend)
end

function _prepare_service_graph(options; declarations=_algorithm_declarations(options),
        worker_thread_ids=nothing, initial_parameters=())
    return _prepare_service_graph(options.graph; declarations,
        execution=options.execution == "row-block" ? :row_block : :complete_frame,
        workers=options.row_workers, matrix_layout=options.matrix_layout,
        worker_thread_ids, initial_parameters, parameters=options.parameters,
        capture_backend=isnothing(options.capture_backend) ? nothing : Symbol(options.capture_backend),
        feedback=options.feedback)
end

function _prepare_service_graph(path::AbstractString; declarations, execution,
        workers, matrix_layout, worker_thread_ids, initial_parameters, parameters,
        capture_backend, feedback=nothing)
    execution in (:complete_frame, :row_block) || throw(ArgumentError("unsupported execution mode: $execution"))
    capture_backend in (nothing, :cuda, :amdgpu) || throw(ArgumentError("unsupported capture backend: $capture_backend"))
    if execution === :complete_frame
        workers == 0 && matrix_layout === :shared && worker_thread_ids === nothing ||
            throw(ArgumentError("worker placement and matrix_layout require row-block execution"))
    end
    capture_backend === nothing || execution === :complete_frame ||
        throw(ArgumentError("GPU capture requires complete-frame execution"))
    if get(ENV, "JULIA_FGPW_CALLBACK_METRICS_CAPACITY", "0") != "0"
        enable_callback_metrics!()
    end
    composition = haskey(FilterGraphs._parse_spa_json(read(path, String)), "filter.composition")
    if composition
        execution === :complete_frame && feedback === nothing ||
            throw(ArgumentError("composition owns its feedback and requires complete-frame execution"))
    end
    graph = if execution === :row_block
        Base.invokelatest(FilterGraphs.prepare_cpu_progressive_graph,
            path; algorithms=declarations, workers=workers,
            matrix_layout=matrix_layout, worker_thread_ids,
        )
    elseif composition
        Base.invokelatest(FilterGraphs.prepare_composition, path; algorithms=declarations)
    else
        Base.invokelatest(FilterGraphs.prepare_graph, path; algorithms=declarations)
    end
    try
        checked = checked_graph(graph)
        ports = Set{Symbol}()
        for parameter in parameters
            parameter.port in ports && throw(ArgumentError("Duplicate startup parameter $(parameter.port)"))
            push!(ports, parameter.port)
            haskey(checked.parameter_formats, parameter.port) || throw(ArgumentError(
                "Graph has no external Parameter Port '$(parameter.port)'",
            ))
            format = getproperty(checked.parameter_formats, parameter.port)
            format.element_type === parameter.element_type || throw(ArgumentError(
                "Parameter $(parameter.port) requires $(format.element_type)",
            ))
            format.shape == parameter.dimensions || throw(DimensionMismatch(
                "Parameter $(parameter.port) requires logical dimensions $(format.shape)",
            ))
        end
        owned = collect(initial_parameters isa Function ?
            initial_parameters(FilterGraphs.parameter_formats(graph)) : initial_parameters)
        for binding in owned
            binding isa Pair && first(binding) isa Symbol && last(binding) isa Array ||
                throw(ArgumentError("initial_parameters requires Symbol => owned Array pairs"))
            port, data = binding
            port in ports && throw(ArgumentError("Duplicate startup parameter $port"))
            push!(ports, port)
            haskey(checked.parameter_formats, port) ||
                throw(ArgumentError("Graph has no external Parameter Port '$port'"))
            format = getproperty(checked.parameter_formats, port)
            eltype(data) === format.element_type ||
                throw(ArgumentError("Parameter $port requires $(format.element_type)"))
            size(data) == format.shape ||
                throw(DimensionMismatch("Parameter $port requires logical dimensions $(format.shape)"))
        end
        if !isempty(parameters) || !isempty(owned)
            updates = Pair{Symbol,Array}[parameter.port => read_startup_parameter(parameter) for parameter in parameters]
            append!(updates, owned)
            Base.invokelatest(FilterGraphs.replace_parameters!, graph, updates...)
        end
        if !isnothing(capture_backend)
            return Base.invokelatest(
                FilterGraphs.prepare_captured_graph, graph,
                capture_device(String(capture_backend)),
            )
        end
        return graph
    catch
        close_execution(graph)
        rethrow()
    end
end

# Prepared CPU owners retain workers until service teardown; ordinary checked
# Graphs own no independently running execution tasks.
checked_graph(graph::FilterGraphs.PreparedGraph) = graph
checked_graph(graph::FilterGraphs.PreparedCpuProgressiveGraph) = graph.graph
"""
    close_execution(graph)

Release the execution resources owned by a prepared service graph, including
persistent row workers. Ordinary and captured prepared graphs require no teardown.
"""
close_execution(::FilterGraphs.PreparedGraph) = nothing
close_execution(graph::FilterGraphs.PreparedCpuProgressiveGraph) = close(graph)
close_execution(::FilterGraphs.PreparedCapturedGraph) = nothing

checked_graph(graph::FilterGraphs.PreparedComposition) = graph.graph
close_execution(::FilterGraphs.PreparedComposition) = nothing
