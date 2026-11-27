"""
    prepare!(node; execution=:complete_frame, required_outputs=(), samples=nothing)

Explicitly validate an unpublished CPU Graph Node with two synthetic complete
samples by default, then restore its initial Graph state and feedback. Construct the Node with
`connect=false`, call this method from its serialized owner, and connect only
after it returns. The first native Start retains its ordinary preparation and
control adoption behavior. No external source is read or advanced.

`execution` is `:complete_frame` or `:row_block`. Row preparation requires matrix
data inputs with matching integer block rates relative to the Graph base rate.
The input shape supplies the row stride; the declared rate supplies the number
of blocks. Synthetic unsigned inputs use their maximum value; other inputs use
one. Algorithms retain ownership of scientific validation and execution.

`required_outputs` is a tuple of canonical output Port Symbols. Every named
output must be available on each synthetic terminal sample, including outputs
declared conditional. Unknown and repeated names are rejected. With no required
outputs, conditional publication remains optional. Preparation rejects expected
Algorithm failures and restores state on success and failure.

The framework uses the borrowed array and admitted metadata types of its live
Graph path. This is startup preparation, not allocation, native metadata
transport, or latency qualification. Captured Graphs and joined input owners
are outside this CPU API.

`samples`, when supplied, is a nonempty tuple of named tuples with exactly
`inputs`, `metadata`, and `expected` fields. Inputs and `SampleMetadata` records
name the Graph's data input Ports. `expected=nothing` requires successful
completion and the selected `required_outputs`; an exact Algorithm failure enum
requires that rejection. The final sample must require success. This opt-in
sequence supports labelled optical priming without accepting unexpected
rejections. It requires complete-frame execution without retained feedback.

This synthetic preparation does not invoke input `ingress` callbacks or publish
acquisition data. Callers supply synthetic identity and optical annotations
explicitly. Normal startup does not require synthetic execution; real input
frames establish scientific state. State and scratch are reset in `finally`.
"""
function prepare!(
    node::PipeWireNode{<:NdArrayFilter,<:PreparedGraphOwner};
    execution::Symbol=:complete_frame,
    required_outputs::Tuple{Vararg{Symbol}}=(),
    samples=nothing,
)
    node_state(node) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED || throw(
        InvalidStateException("CPU Graph preparation requires an unpublished Node", :connected),
    )
    execution in (:complete_frame, :row_block) || throw(
        ArgumentError("execution must be :complete_frame or :row_block"),
    )
    owner = node.owner
    execution === :row_block && !isnothing(owner.frame_observer) && throw(
        ArgumentError("frame observation requires complete-frame execution"),
    )
    execution === :row_block && !isempty(owner.ingress) && throw(
        ArgumentError("input ingress requires complete-frame execution"),
    )
    samples === nothing || (execution === :complete_frame &&
        (isnothing(owner.feedback) || owner.join isa FilterGraphs.PreparedComposition)) ||
        throw(ArgumentError("Explicit synthetic samples require a supported complete-frame boundary"))
    _prepare_cpu_graph!(owner, owner.join, Val(execution), required_outputs; samples)
    return node
end

function _prepare_cpu_graph!(owner, join, execution, required_outputs; samples=nothing)
    throw(ArgumentError("CPU Graph preparation does not support joined input owners"))
end

_warmup_block_count(owner, ::Val{:complete_frame}) = 1

function _warmup_block_count(owner, ::Val{:row_block})
    formats = map(name -> getproperty(owner.graph.input_formats, name),
                  keys(_public_input_bindings(owner)))
    progressive = filter(format -> format.rate[1] > format.rate[2], formats)
    isempty(progressive) && throw(ArgumentError(
        "Row preparation requires an input whose declared rate gives blocks per frame",
    ))
    all(format -> format.rate[2] == 1 && length(format.shape) == 2, progressive) ||
        throw(ArgumentError("Row preparation requires integer-rate matrix row inputs"))
    blocks = Int(first(progressive).rate[1])
    all(format -> format.rate[1] == blocks, progressive) || throw(ArgumentError(
        "Row preparation requires matching input block rates",
    ))
    return blocks
end

_warmup_inputs(owner, ::Nothing, borrowed) = borrowed
_warmup_outputs(owner, ::Nothing, borrowed) = borrowed

function _warmup_inputs(owner, bridge::PreparedFeedbackBridge, borrowed)
    public = NamedTuple{keys(bridge.public_input_bindings)}(map(
        name -> getproperty(borrowed, name), keys(bridge.public_input_bindings),
    ))
    return _feedback_inputs(owner, bridge, public)
end

function _warmup_outputs(owner, bridge::PreparedFeedbackBridge, borrowed)
    public = NamedTuple{keys(bridge.public_output_bindings)}(map(
        name -> getproperty(borrowed, name), keys(bridge.public_output_bindings),
    ))
    return _feedback_outputs(owner, bridge, public)
end

_warmup_metadata(::Nothing, metadata) = metadata

function _warmup_metadata(bridge::PreparedFeedbackBridge{Input}, metadata) where {Input}
    return NamedTuple{keys(metadata)}(map(keys(metadata)) do name
        name === Input ? FilterGraphAlgorithms._NO_SAMPLE_METADATA : getproperty(metadata, name)
    end)
end

function _validate_warmup_result(result::NamedTuple, graph, required_outputs, terminal)
    terminal || return nothing
    availability = result.availability
    if !isempty(required_outputs)
        result.terminal || throw(ArgumentError("Synthetic startup sample did not complete"))
        for name in required_outputs
            getproperty(availability, name) || throw(ArgumentError(
                "Synthetic terminal sample omitted required output '$name'",
            ))
        end
    end
    all(published || conditional for (published, conditional) in
        zip(values(availability), values(graph.output_conditional))) || throw(ArgumentError(
        "Synthetic startup sample omitted an unconditional output",
    ))
    return nothing
end

function _validate_warmup_result(result::Enum, graph, required_outputs, terminal)
    throw(ArgumentError("Synthetic startup sample was rejected: $result"))
end

_advance_warmup_feedback!(::Nothing, local_output, result) = nothing

function _advance_warmup_feedback!(bridge::PreparedFeedbackBridge, local_output, result::NamedTuple)
    result.terminal || return nothing
    # Match the live feedback bridge's complete terminal publication contract.
    all(values(_native_outputs(local_output, result.availability))) || throw(ArgumentError(
        "Synthetic terminal feedback sample omitted an output",
    ))
    copyto!(bridge.slot, bridge.scratch)
    return nothing
end

function _prepare_cpu_graph!(
    owner::PreparedGraphOwner, ::Nothing, execution, required_outputs; samples=nothing,
)
    length(unique(required_outputs)) == length(required_outputs) || throw(
        ArgumentError("required_outputs must contain distinct Port names"),
    )
    for name in required_outputs
        haskey(owner.graph.output_formats, name) || throw(ArgumentError(
            "Graph has no output Port '$name'",
        ))
    end
    blocks = _warmup_block_count(owner, execution)
    try
        samples === nothing || _reset_owner!(owner)
        _prepare_owner(owner)
        for storage in values(owner.warm_inputs)
            value = eltype(storage) <: Unsigned ? typemax(eltype(storage)) : one(eltype(storage))
            fill!(storage, value)
        end
        GC.@preserve owner begin
            inputs = _warmup_inputs(owner, owner.feedback,
                _borrow_warm_buffers(owner.input_bindings, owner.warm_inputs, Val(false)))
            outputs = _warmup_outputs(owner, owner.feedback,
                _borrow_warm_buffers(owner.output_bindings, owner.warm_outputs, Val(true)))
            if samples !== nothing
                _prepare_explicit_samples!(owner, inputs, outputs, samples, required_outputs)
            else
                for sequence in 0:1, block in 0:(blocks - 1)
                    metadata = NamedTuple{keys(owner.graph.input_formats)}(map(values(owner.graph.input_formats)) do format
                        offset = blocks == 1 ? 0 : block * first(format.shape)
                        sample = SampleMetadata(sequence, offset; terminal=block == blocks - 1)
                        FilterGraphAlgorithms._admitted_sample_metadata(sample)
                    end)
                    metadata = _warmup_metadata(owner.feedback, metadata)
                    result = FilterGraphs._process_candidate_admitted!(
                        outputs, owner.graph, inputs, metadata,
                    )
                    _validate_warmup_result(result, owner.graph, required_outputs, block == blocks - 1)
                    _advance_warmup_feedback!(owner.feedback, owner.local_output, result)
                end
            end
        end
    finally
        try
            _reset_owner!(owner)
        finally
            _deactivate_owner(owner)
            foreach(storage -> fill!(storage, 0), values(owner.warm_inputs))
            foreach(storage -> fill!(storage, 0), values(owner.warm_outputs))
        end
    end
    return nothing
end

function _prepare_explicit_samples!(owner, inputs, outputs, samples, required_outputs)
    samples isa Tuple && !isempty(samples) ||
        throw(ArgumentError("Explicit synthetic samples must be a nonempty tuple"))
    for sample in samples
        sample isa NamedTuple && Set(keys(sample)) == Set((:inputs, :metadata, :expected)) ||
            throw(ArgumentError("Synthetic samples require inputs, metadata and expected fields"))
        sample.inputs isa NamedTuple && sample.metadata isa NamedTuple ||
            throw(ArgumentError("Synthetic inputs and metadata must name the Graph input Ports"))
        sample.expected === nothing || sample.expected isa Enum ||
            throw(ArgumentError("Synthetic expected result must be nothing or an exact Algorithm failure enum"))
    end
    last(samples).expected === nothing ||
        throw(ArgumentError("The final synthetic sample must require successful completion"))
    for sample in samples
        values = FilterGraphs._validate_external_arrays(:input, owner.graph.input_formats, sample.inputs)
        metadata = FilterGraphs._validate_external_metadata(owner.graph.input_formats, sample.metadata)
        for name in keys(owner.warm_inputs)
            copyto!(getproperty(owner.warm_inputs, name), getproperty(values, name))
        end
        admitted = map(FilterGraphAlgorithms._admitted_sample_metadata, metadata)
        result = FilterGraphs._process_candidate_admitted!(outputs, owner.graph, inputs, admitted)
        if sample.expected === nothing
            _validate_warmup_result(result, owner.graph, required_outputs, true)
        else
            result === sample.expected || throw(ArgumentError(
                "Synthetic sample expected $(sample.expected), observed $result"))
        end
    end
    return nothing
end
