# Run with a selected-source environment containing FGA and FilterGraphs:
# JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 julia --startup-file=no \
#   --project=<environment> julia/FilterGraphAlgorithms/benchmark/array_statistics.jl /tmp/statistics
# The environment needs JSON.jl and FilterGraphs as benchmark dependencies;
# neither is added to the FGA core. The output prefix receives raw timings, JSON
# summaries and concrete
# LLVM/native numerical-kernel listings. This is not an offered-load/SLO test.

using FilterGraphAlgorithms, FilterGraphs, Statistics, JSON, InteractiveUtils, LinearAlgebra, SHA
import FilterGraphAlgorithms: process!, reset!
const FGA = FilterGraphAlgorithms
const NAMES = (:mean, :variance, :stddev, Symbol("sample-count"), :complete, Symbol("variance-valid"))
const AVAILABLE = NamedTuple{NAMES}((true, true, true, true, true, true))
function prepare_case(shape, mode)
    text = """
    filter.graph = {
      nodes = [ { type = ndarray name = statistics label = array-statistics-f32
        config = {
          size = [$(join(shape, ' '))] mode = $mode window_samples = $(mode == "finite" ? 1000000000 : 16)
          input_schema = "example/sample-f32/1" mean_schema = "example/mean-f32/1"
          variance_schema = "example/variance-f32/1" stddev_schema = "example/stddev-f32/1"
        } } ] links = [] inputs = [ "statistics:sample" ]
      outputs = [ "statistics:mean" "statistics:variance" "statistics:stddev"
                  "statistics:sample-count" "statistics:complete" "statistics:variance-valid" ]
    }
    """
    graph = mktemp() do path, io
        write(io, text); flush(io)
        prepare_graph(path; algorithms=algorithms())
    end
    inputs = ntuple(4) do frame
        (sample=reshape(Float32[1000 + ((i*17 + frame*11)%31)/8 for i in 1:prod(shape)], shape),)
    end
    out = NamedTuple{NAMES}((zeros(Float32, shape), zeros(Float32, shape), zeros(Float32, shape),
        zeros(UInt64, 1), zeros(Bool, 1), zeros(Bool, 1)))
    return graph, inputs, out
end
function warm!(out, graph, inputs, count)
    for k in 1:count
        process!(out, graph, inputs[mod1(k, 4)]) == AVAILABLE || error("Graph warmup failed")
    end
end
function timed!(times, out, graph, inputs)
    for k in eachindex(times)
        input = inputs[mod1(k, 4)]
        started = time_ns()
        result = process!(out, graph, input)
        times[k] = time_ns() - started
        result == AVAILABLE || error("Graph measurement failed: $result")
    end
end
allocation_bytes(out, graph, input) = @allocated process!(out, graph, input)
function run(prefix)
    BLAS.set_num_threads(1)
    metadata = Dict("julia"=>string(VERSION), "cpu"=>Sys.CPU_NAME,
        "threads"=>Threads.nthreads(), "cpu_target"=>get(ENV,"JULIA_CPU_TARGET","default"),
        "fga_source"=>pathof(FGA),
        "statistics_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(FGA)), "algorithms/statistics/array_statistics.jl")))),
        "kernel"=>string(Sys.KERNEL), "llvm"=>string(Base.libllvm_version),
        "blas_threads"=>BLAS.get_num_threads(), "gc"=>"enabled; GC.gc outside each repetition", "active_project"=>Base.active_project(),
        "shape_description"=>"1024 vector and 64x96 rectangular array",
        "measurement"=>"closed-loop checked Graph process!; 500 warmups then 5x1024 samples; per-sample time_ns clock; deterministic four-frame cycle; no fixed-rate or queue latency claim")
    results = Dict[]
    raw = open(prefix*"-raw.csv", "w")
    println(raw,"shape,mode,repetition,sample,nanoseconds")
    for shape in ((1024,), (64,96)), mode in ("finite", "sliding", "exponential")
        graph, inputs, out = prepare_case(shape, mode)
        warm!(out, graph, inputs, 500)
        times = zeros(UInt64, 1024)
        timed!(times, out, graph, inputs)
        allocation_bytes(out, graph, inputs[1])
        bytes = allocation_bytes(out, graph, inputs[1])
        alltimes = UInt64[]
        medians = Float64[]
        for repetition in 1:5
            GC.gc()
            timed!(times, out, graph, inputs)
            push!(medians, median(times))
            append!(alltimes,times)
            for k in eachindex(times)
                println(raw,join((join(shape,'x'),mode,repetition,k,times[k]),','))
            end
        end
        row = Dict("shape"=>collect(shape),"mode"=>mode,"allocation_bytes"=>bytes,
            "p50_ns"=>median(alltimes),"p90_ns"=>quantile(alltimes,.9),
            "p99_ns"=>quantile(alltimes,.99),"max_ns"=>maximum(alltimes),
            "repetition_medians_ns"=>medians,"checksum"=>sum(Float64,out.mean)+sum(Float64,out.variance))
        push!(results,row)
        println(row)
        if shape == (64,96) && mode == "sliding"
            p = first(graph.nodes).prepared
            sig = Tuple{map(typeof,values(out))..., typeof(p.workspace),typeof(p.plan),typeof(inputs[1].sample)}
            open(prefix*"-process.ll","w") do io
                code_llvm(io,FGA._process_array_statistics!,sig;debuginfo=:none)
            end
            open(prefix*"-process.s","w") do io
                code_native(io,FGA._process_array_statistics!,sig;debuginfo=:none,syntax=:intel)
            end
            println("concrete_method=",which(FGA._process_array_statistics!,sig))
            println("concrete_signature=",sig)
            for name in (:_array_statistics_finite!, :_array_statistics_exponential!, :_array_statistics_sliding!)
                isdefined(FGA,name) || continue
                fn=getfield(FGA,name)
                helpersig=Tuple{typeof(p.workspace),typeof(p.plan),UInt64}
                open(prefix*"-"*string(name)*".ll","w") do io
                    code_llvm(io,fn,helpersig;debuginfo=:none)
                end
                open(prefix*"-"*string(name)*".s","w") do io
                    code_native(io,fn,helpersig;debuginfo=:none,syntax=:intel)
                end
            end
        end
    end
    close(raw)
    open(prefix*".json","w") do io
        JSON.json(io,Dict("environment"=>metadata,"results"=>results); pretty=true)
    end
end
run(only(ARGS))
