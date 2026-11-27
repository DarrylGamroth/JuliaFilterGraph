import FilterGraphAlgorithms: PreparedAlgorithm, InputPort, OutputPort, FrameData,
    algorithm_ports, algorithm_formats, complete!, enqueue!, preflight, reset!

struct CapturedKernel{K,A}
    kernel::K
    arguments::A
    extent::Int
end

function enqueue!(adapter::CapturedKernel)
    adapter.kernel(adapter.arguments...; ndrange=adapter.extent)
    return nothing
end
preflight(::CapturedKernel) = nothing
complete!(::CapturedKernel, ::Int32) = nothing
reset!(adapter::CapturedKernel) = adapter

function _check_capture_ports(device, prepared::PreparedAlgorithm{D}, outputs, inputs, status;
        input_overlap=:none) where {D}
    _require_current_device(device)
    _require_capture_array("captured frame status", status, device)
    eltype(status) === Int32 && axes(status) == (Base.OneTo(1),) ||
        throw(ArgumentError("captured frame status must be one Int32"))
    ports = algorithm_ports(D)
    formats = algorithm_formats(prepared)
    for (direction, arrays) in ((InputPort, inputs), (OutputPort, outputs))
        bindings = [(port, format) for (port, format) in zip(ports, formats)
            if port.direction isa direction && port.role isa FrameData]
        length(bindings) == length(arrays) ||
            throw(ArgumentError("unexpected count of captured Algorithm ports"))
        for ((port, format), array) in zip(bindings, arrays)
            FilterGraphAlgorithms._validate_declared_array(format, array)
            _require_capture_array(string(port.name), array, device)
            _capture_arrays_overlap(array, status) &&
                throw(ArgumentError("captured port must be disjoint from frame status"))
        end
    end
    for i in eachindex(outputs)
        for j in 1:(i - 1)
            _capture_arrays_overlap(outputs[i], outputs[j]) &&
                throw(ArgumentError("captured outputs must be disjoint"))
        end
        if input_overlap !== :all
            for input in inputs
                (input_overlap === :exact && outputs[i] === input) ||
                    !_capture_arrays_overlap(outputs[i], input) ||
                    throw(ArgumentError("captured output overlaps input"))
            end
        end
    end
    return KernelAbstractions.get_backend(status)
end
