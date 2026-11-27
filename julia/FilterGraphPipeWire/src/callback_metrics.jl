# Opt-in investigative trace. One serial native callback owns all array writes.
# Read/export only after native close, which joins the callback thread.
struct CallbackAllocationSample
    epoch::UInt64
    allocated::Int64
    thread_id::Int32
    gc_enabled::Int32
end

mutable struct GraphCallbackMetrics{Sampler}
    sample::Sampler
    path::String
    sequence::Vector{UInt64}
    offset::Vector{Int64}
    start_ns::Vector{UInt64}
    end_ns::Vector{UInt64}
    allocation_start::Vector{CallbackAllocationSample}
    allocation_end::Vector{CallbackAllocationSample}
    properties_changed::Vector{Bool}
    parameters_changed::Vector{Bool}
    succeeded::Vector{Bool}
    count::Int
    omitted::UInt64
end

# Factory registration is a cold deployment hook for investigative instrumentation.
# The returned metrics and sampler become concrete fields of PreparedGraphOwner.
const _callback_metrics_factory = Ref{Union{Nothing,Function}}(nothing)
function _callback_metrics_from_environment(input_count::Int)
    factory = _callback_metrics_factory[]
    return isnothing(factory) ? nothing : factory(input_count)
end

function GraphCallbackMetrics(path::String, capacity::Int, sampler)
    capacity > 0 || throw(ArgumentError("callback metrics capacity must be positive"))
    empty_sample = CallbackAllocationSample(0, 0, -1, 0)
    return GraphCallbackMetrics(sampler, path,
        zeros(UInt64, capacity), zeros(Int64, capacity),
        zeros(UInt64, capacity), zeros(UInt64, capacity),
        fill(empty_sample, capacity), fill(empty_sample, capacity),
        fill(false, capacity), fill(false, capacity), fill(false, capacity), 0, UInt64(0))
end

@inline _begin_callback_metrics!(::Nothing, inputs) = 0
@inline function _begin_callback_metrics!(metrics::GraphCallbackMetrics, inputs)
    return _begin_callback_metrics!(metrics, PipeWireAO.buffer_header(inputs[1]))
end
@inline function _begin_callback_metrics!(metrics::GraphCallbackMetrics,
    header::PipeWireAO.BufferHeader)
    index = metrics.count + 1
    if index > length(metrics.sequence)
        metrics.omitted += UInt64(1)
        return 0
    end
    metrics.sequence[index] = header.sequence
    metrics.offset[index] = header.offset
    metrics.start_ns[index] = time_ns()
    metrics.allocation_start[index] = metrics.sample()
    metrics.count = index
    return index
end

@inline _end_callback_metrics!(::Nothing, index, properties, parameters, succeeded) = nothing
@inline function _end_callback_metrics!(metrics::GraphCallbackMetrics, index::Int,
    properties::Bool, parameters::Bool, succeeded::Bool)
    if !iszero(index)
        metrics.allocation_end[index] = metrics.sample()
        metrics.end_ns[index] = time_ns()
        metrics.properties_changed[index] = properties
        metrics.parameters_changed[index] = parameters
        metrics.succeeded[index] = succeeded
    end
    return nothing
end

_callback_allocation_valid(a, b) =
    a.epoch == b.epoch && a.thread_id == b.thread_id &&
    a.gc_enabled == b.gc_enabled == 1 && b.allocated >= a.allocated

function _write_callback_metrics(metrics::GraphCallbackMetrics)
    # Caller must have closed the native node before reading this serial trace.
    (ispath(metrics.path) || islink(metrics.path)) && error("callback metrics file exists: $(metrics.path)")
    open(metrics.path, "w") do io
        println(io, "callback,sequence,offset,start_ns,end_ns,allocated_bytes,allocation_valid,gc_epoch_start,gc_epoch_end,thread_id,properties_changed,parameters_changed,succeeded")
        for i in 1:metrics.count
            a, b = metrics.allocation_start[i], metrics.allocation_end[i]
            valid = _callback_allocation_valid(a, b)
            print(io, i, ',', metrics.sequence[i], ',', metrics.offset[i], ',',
                metrics.start_ns[i], ',', metrics.end_ns[i], ',',
                valid ? string(b.allocated - a.allocated) : "", ',', valid, ',',
                a.epoch, ',', b.epoch, ',', a.thread_id, ',',
                metrics.properties_changed[i], ',', metrics.parameters_changed[i], ',',
                metrics.succeeded[i], '\n')
        end
    end
    # Summary printed once, after callback teardown. Overflow never goes unnoticed.
    println("JULIA_FGPW_CALLBACK_METRICS path=$(metrics.path) capacity=$(length(metrics.sequence)) records=$(metrics.count) omitted=$(metrics.omitted)")
    return nothing
end
_write_callback_metrics(::Nothing) = nothing
