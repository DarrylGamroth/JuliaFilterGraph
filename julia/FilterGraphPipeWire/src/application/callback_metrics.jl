module CallbackMetricsSampler
using Libdl
import ..FilterGraphPipeWire

struct Sampler
    library::Ptr{Cvoid} # Retain library for the process lifetime: pre-GC hook stays registered.
    sample::Ptr{Cvoid}
end

function (sampler::Sampler)()
    return ccall(sampler.sample, FilterGraphPipeWire.CallbackAllocationSample, ())
end

function from_environment(input_count::Int)
    capacity = parse(Int, get(ENV, "JULIA_FGPW_CALLBACK_METRICS_CAPACITY", "0"))
    capacity >= 0 || throw(ArgumentError("callback metrics capacity must be nonnegative"))
    iszero(capacity) && return nothing
    input_count == 1 || throw(ArgumentError("callback metrics require one public data input"))
    path = ENV["JULIA_FGPW_CALLBACK_METRICS_PATH"]
    isempty(path) && throw(ArgumentError("callback metrics path must not be empty"))
    (ispath(path) || islink(path)) && throw(ArgumentError("callback metrics file exists: $path"))
    isdir(dirname(path)) || throw(ArgumentError("callback metrics directory does not exist"))
    library = Libdl.dlopen(ENV["JULIA_FGPW_CALLBACK_METRICS_LIBRARY"])
    initialize = Libdl.dlsym(library, :jfg_metrics_init)
    ccall(initialize, Cint, ()) == 1 || throw(ArgumentError("callback metrics Julia header/runtime version mismatch"))
    sampler = Sampler(library, Libdl.dlsym(library, :jfg_metrics_sample))
    sampler() # resolve and compile before callback publication
    return FilterGraphPipeWire.GraphCallbackMetrics(path, capacity, sampler)
end
end

"""
    enable_callback_metrics!()

Register the optional callback metrics sampler for subsequent node construction.
Registration performs no native calls; enabled nodes read the metrics environment
and load the instrumentation library when their sampler is created.
"""
function enable_callback_metrics!()
    FilterGraphPipeWire._callback_metrics_factory[] = CallbackMetricsSampler.from_environment
    return nothing
end
