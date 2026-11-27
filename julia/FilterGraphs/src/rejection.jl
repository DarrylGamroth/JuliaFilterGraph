"""
    GraphRejection

An immutable diagnostic snapshot of an expected Algorithm rejection. `node` is
its configured Node name, `reason` is the unchanged Algorithm enum, and `inputs`
contains the supplied external input metadata (`SampleMetadata` or `nothing`).
Input metadata need not share a sequence and is not inferred from old outputs.
This record does not describe an unexpected exception or undo earlier commits.
"""
struct GraphRejection{R,M}
    node::Symbol
    reason::R
    inputs::M
end

# Only host storage; recording a rejection adds no device work or success scan.
mutable struct RejectionState{R,M}
    node::Int
    reason::Union{Nothing,R}
    inputs::M
end

function _rejection_state(::GraphResultTypes{A,R}, formats::NamedTuple) where {A,R}
    M = NamedTuple{keys(formats),NTuple{length(formats),Union{Nothing,FilterGraphAlgorithms.SampleMetadata}}}
    return RejectionState{R,M}(0, nothing, M(map(_ -> nothing, values(formats))))
end

@inline function _clear_rejection!(state::RejectionState)
    state.node = 0
    state.reason = nothing
    return nothing
end

@inline function _record_rejection!(state::RejectionState, index::Int, reason)
    state.node = index
    state.reason = reason
    return nothing
end

@inline function _rejection_inputs!(state::RejectionState{R,M}, metadata::NamedTuple) where {R,M}
    state.inputs = M(map(_rejection_input_metadata, metadata))
    return nothing
end

function _rejection_context(state::RejectionState, names::Tuple)
    iszero(state.node) && return nothing
    return GraphRejection(names[state.node], state.reason, state.inputs)
end

"""
    rejection_context(graph)

Return `nothing` or an immutable [`GraphRejection`](@ref) for the most recent
admitted call on an ordinary, CPU/GPU progressive, GPU readout or fully captured
Graph. On a composition, a branch or shared-stage rejection records the inputs
admitted to that stage, including retained metadata from earlier arrivals.
For GPU readout device failures, metadata belongs to the terminal
reporting call; it does not identify which earlier block first failed. A new
admitted call or successful `reset!` clears it;
checked argument errors leave the previous diagnostic intact. Serialize this
query with processing, control updates and reset, as required by the owner.
Constructing this diagnostic snapshot may allocate; ordinary execution records
only bounded, typed host storage.

For a captured pipeline request `take_captured_result!(outputs, pipeline;
diagnostics=true)`: the opt-in receipt includes `rejection` for that frame and
remains valid after slot reuse. The default receipt is unchanged.
"""
function rejection_context end

rejection_context(owner::PreparedCpuProgressiveGraph) = rejection_context(owner.graph)
rejection_context(owner::PreparedGpuProgressiveGraph) = rejection_context(owner.graph)

_rejection_input_metadata(value::FilterGraphAlgorithms.AdmittedSampleMetadata) =
    FilterGraphAlgorithms._sample_metadata(value)
_rejection_input_metadata(value::Union{Nothing,FilterGraphAlgorithms.SampleMetadata}) = value

rejection_context(owner::PreparedGpuReadoutGraph) =
    _rejection_context(owner.rejection, map(node -> node.name, owner.graph.nodes))

# Retain the existing preparation constructor while giving every Graph its own
# diagnostic storage, including progressive Graphs rebuilt from checked sources.
function PreparedGraph(
    nodes, failure_types, outputs, available, metadata, inflight, routes,
    parameters, output_routes, inputs, parameter_formats, output_formats,
    conditional, published, default_metadata, results,
)
    return PreparedGraph(
        nodes, failure_types, outputs, available, metadata, inflight, routes,
        parameters, output_routes, inputs, parameter_formats, output_formats,
        conditional, published, default_metadata, results,
        _rejection_state(results, inputs),
    )
end

rejection_context(graph::PreparedGraph) =
    _rejection_context(graph.rejection, map(node -> node.name, graph.nodes))
