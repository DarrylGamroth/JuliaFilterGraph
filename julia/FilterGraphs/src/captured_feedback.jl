_select_captured_ports(ports::NamedTuple, names::Tuple) =
    NamedTuple{names}(map(name -> getfield(ports, name), names))

function _captured_feedback_ports(input_formats, output_formats, metadata_sources, feedback, outputs)
    selected = Tuple(outputs)
    all(name -> name isa Symbol && haskey(output_formats, name), selected) ||
        throw(GraphConfigurationError("captured output selection contains an unknown Port"))
    length(unique(selected)) == length(selected) ||
        throw(GraphConfigurationError("captured output selection contains duplicate Ports"))
    if isempty(selected) && (!isempty(output_formats) || !isempty(feedback))
        throw(GraphConfigurationError("captured output selection must retain a public output"))
    end
    for (input, output) in pairs(feedback)
        haskey(input_formats, input) ||
            throw(GraphConfigurationError("unknown captured feedback input '$input'"))
        output isa Symbol && haskey(output_formats, output) ||
            throw(GraphConfigurationError("unknown captured feedback output '$output'"))
        destination, source = input_formats[input], output_formats[output]
        compatible = destination.element_type === source.element_type &&
            destination.shape == source.shape && typeof(destination.layout) === typeof(source.layout) &&
            destination.schema == source.schema && destination.rate == source.rate
        compatible || throw(GraphConfigurationError(
            "captured feedback '$input' <- '$output' has incompatible Port formats",
        ))
    end
    inputs = Tuple(name for name in keys(input_formats) if !haskey(feedback, name))
    isempty(inputs) && !isempty(feedback) &&
        throw(GraphConfigurationError("resident captured feedback needs a public input"))
    sources = map(selected) do output
        index = findfirst(==(output), keys(output_formats))
        source = metadata_sources[index]
        iszero(source) && return 0
        name = keys(input_formats)[source]
        haskey(feedback, name) && throw(GraphConfigurationError(
            "selected output '$output' derives metadata from feedback input '$name'",
        ))
        return something(findfirst(==(name), inputs))
    end
    return (; inputs, outputs=selected, metadata_sources=sources)
end

function _require_captured_feedback_support(device, feedback)
    isempty(feedback) || throw(GraphConfigurationError(
        "resident captured feedback is not qualified for $(typeof(device))",
    ))
    return nothing
end

function _prepare_captured_feedback(device, feedback, device_inputs, device_outputs, frame_status)
    _require_captured_feedback_support(device, feedback)
    return ()
end

_enqueue_captured_feedback!(::Tuple{}) = nothing
function _enqueue_captured_feedback!(feedback::Tuple)
    _enqueue_captured_feedback!(first(feedback))
    return _enqueue_captured_feedback!(Base.tail(feedback))
end

_reset_captured_feedback!(executable) = nothing
_reset_captured_feedback!(::Tuple{}) = nothing
function _reset_captured_feedback!(feedback::Tuple)
    _reset_captured_feedback!(first(feedback))
    return _reset_captured_feedback!(Base.tail(feedback))
end

# Empty feedback preserves existing backend recording implementations.
function _record_captured_graph(context, frame_status, nodes::Tuple, feedback::Tuple)
    isempty(feedback) || throw(GraphConfigurationError(
        "resident captured feedback recording is not qualified for $(typeof(context))",
    ))
    return _record_captured_graph(context, frame_status, nodes)
end

function _captured_result_types(::GraphResultTypes{Availability,Failure}, output_formats) where {Availability,Failure}
    selected = NamedTuple{keys(output_formats),NTuple{length(output_formats),Bool}}
    return GraphResultTypes{selected,Failure}()
end
