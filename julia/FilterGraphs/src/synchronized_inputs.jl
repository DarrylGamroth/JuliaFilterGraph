# Acquisition matching pins one producer domain for the owner's lifetime.
mutable struct SynchronizedAcquisitionMatch
    has_identity::Bool
    domain::NTuple{16,UInt8}
    generation::UInt64
    domain_changed::Bool
end

SynchronizedAcquisitionMatch() = SynchronizedAcquisitionMatch(
    false, ntuple(_ -> UInt8(0), 16), UInt64(0), false,
)

_synchronized_match(::Val{:sequence}) = Val(:sequence)
_synchronized_match(::Val{:acquisition}) = SynchronizedAcquisitionMatch()
_synchronized_match(::Val) = throw(ArgumentError("synchronized match must be :sequence or :acquisition"))

mutable struct SynchronizedInputState{R}
    active::Bool
    sequence::UInt64
    has_last_sequence::Bool
    last_sequence::UInt64
    corrupted::Bool
    discontinuity::Bool
    received::R
end

function SynchronizedInputState(input_formats::NamedTuple)
    received = map(_ -> Ref(false), values(input_formats))
    return SynchronizedInputState(
        false,
        UInt64(0),
        false,
        UInt64(0),
        false,
        false,
        received,
    )
end

mutable struct SynchronizedInputCounters
    arrivals::UInt64
    completed_windows::UInt64
    expired_windows::UInt64
    corrupted_windows::UInt64
    rejected_windows::UInt64
    aborted_windows::UInt64
    stale_arrivals::UInt64
    duplicate_arrivals::UInt64
    discontinuities::UInt64
    incoherent_arrivals::UInt64
    domain_changes::UInt64
end

SynchronizedInputCounters() =
    SynchronizedInputCounters(ntuple(_ -> UInt64(0), Val(11))...)

"""
    PreparedSynchronizedInputs

One prepared all-required input window for a Graph whose external data Ports
have the same rate. Each Port owns one preallocated sample slot. Arrivals may
occur in any order, but the Graph runs only after every Port has supplied the
same sample sequence.
"""
struct PreparedSynchronizedInputs{G,B,M,S,Match}
    graph::G
    input_buffers::B
    input_metadata::M
    state::S
    counters::SynchronizedInputCounters
    match::Match
end

@inline _any_true(::Tuple{}) = false

@inline function _any_true(values::Tuple)
    first(values) && return true
    return _any_true(Base.tail(values))
end

@inline _has_conditional_node_output(::Tuple{}) = false

@inline function _has_conditional_node_output(nodes::Tuple)
    _any_true(first(nodes).output_conditional) && return true
    return _has_conditional_node_output(Base.tail(nodes))
end

function _require_synchronized_input_rates(formats::Tuple)
    rate = first(formats).rate
    for format in Base.tail(formats)
        format.rate == rate || throw(
            ArgumentError("all synchronized Graph input Ports must have the same rate"),
        )
    end
    return rate
end

"""
    prepare_synchronized_inputs(graph; match=:sequence)

Prepare an equal-rate, all-required arrival window for a Graph with at least
two external data inputs. The returned owner copies each arrival into bounded
storage and performs no waiting. A host timer may abandon an incomplete window
with [`expire!`](@ref).

`match=:sequence` associates arrivals by sample sequence. `match=:acquisition`
matches the full source domain, generation and physical acquisition sequence.
Publication sequences and Header PTS may differ between contributors. The first valid identity pins the
source domain for this owner's lifetime. A different domain permanently closes
admission; reconstruct the owner after producer replacement. Generations must
increase, and the Graph resets once on the first valid arrival from each newer
generation. Late older generations and backward sequences are dropped, even
when marked discontinuous.

This first synchronized-input slice admits only terminal Algorithms. General
progressive scheduling remains a separate per-Port/edge concern.
"""
function prepare_synchronized_inputs(graph::PreparedGraph; match::Symbol=:sequence)
    matching = _synchronized_match(Val(match))
    length(graph.input_formats) >= 2 || throw(
        ArgumentError("synchronized input processing requires at least two external data Ports"),
    )
    _require_synchronized_input_rates(values(graph.input_formats))
    _has_conditional_node_output(graph.nodes) && throw(
        ArgumentError("synchronized input processing requires terminal Algorithm outputs"),
    )

    input_buffers = _external_buffers(graph.input_formats)
    input_metadata = _external_output_metadata(graph.input_formats)
    synchronized = PreparedSynchronizedInputs(
        graph,
        input_buffers,
        input_metadata,
        SynchronizedInputState(graph.input_formats),
        SynchronizedInputCounters(),
        matching,
    )
    output_type = _external_buffer_type(graph.output_formats)
    for input in values(input_buffers)
        precompile(
            process!,
            (
                output_type,
                typeof(synchronized),
                Symbol,
                typeof(input),
                FilterGraphAlgorithms.SampleMetadata,
            ),
        )
    end
    precompile(expire!, (typeof(synchronized),))
    precompile(reset!, (typeof(synchronized),))
    precompile(synchronized_input_counters, (typeof(synchronized),))
    return synchronized
end

@inline _clear_received!(::Tuple{}) = nothing

@inline function _clear_received!(received::Tuple)
    first(received)[] = false
    return _clear_received!(Base.tail(received))
end

@inline _all_received(::Tuple{}) = true

@inline function _all_received(received::Tuple)
    first(received)[] || return false
    return _all_received(Base.tail(received))
end

@inline _clear_metadata!(::Tuple{}) = nothing

@inline function _clear_metadata!(metadata::Tuple)
    first(metadata).present = false
    return _clear_metadata!(Base.tail(metadata))
end

@inline function _open_synchronized_window!(
    synchronized::PreparedSynchronizedInputs,
    sequence::UInt64,
    discontinuity::Bool,
)
    state = synchronized.state
    _clear_received!(state.received)
    state.active = true
    state.sequence = sequence
    state.corrupted = false
    state.discontinuity = discontinuity
    return nothing
end

@inline function _close_synchronized_window!(synchronized::PreparedSynchronizedInputs)
    state = synchronized.state
    state.has_last_sequence = true
    state.last_sequence = state.sequence
    state.active = false
    state.corrupted = false
    state.discontinuity = false
    _clear_received!(state.received)
    return nothing
end

@inline function _admit_synchronized_sequence!(
    synchronized::PreparedSynchronizedInputs,
    metadata::FilterGraphAlgorithms.SampleMetadata,
)
    state = synchronized.state
    counters = synchronized.counters
    sequence = metadata.sequence

    if metadata.discontinuity
        if state.active && sequence == state.sequence
            if !state.discontinuity
                counters.discontinuities += UInt64(1)
                state.discontinuity = true
            end
            return true
        end
        if state.active
            counters.expired_windows += UInt64(1)
            _close_synchronized_window!(synchronized)
        end
        counters.discontinuities += UInt64(1)
        _open_synchronized_window!(synchronized, sequence, true)
        return true
    end

    if state.active
        sequence < state.sequence && (counters.stale_arrivals += UInt64(1); return false)
        if sequence > state.sequence
            counters.expired_windows += UInt64(1)
            counters.discontinuities += UInt64(1)
            _close_synchronized_window!(synchronized)
            _open_synchronized_window!(synchronized, sequence, true)
        end
        return true
    end

    if state.has_last_sequence
        sequence <= state.last_sequence &&
            (counters.stale_arrivals += UInt64(1); return false)
        contiguous = sequence - state.last_sequence == UInt64(1)
        !contiguous && (counters.discontinuities += UInt64(1))
        _open_synchronized_window!(synchronized, sequence, !contiguous)
        return true
    end

    _open_synchronized_window!(synchronized, sequence, false)
    return true
end

@inline _admit_synchronized_match!(synchronized, metadata, ::Val{:sequence}) =
    _admit_synchronized_sequence!(synchronized, metadata)

@inline function _reject_incoherent_arrival!(synchronized)
    synchronized.counters.incoherent_arrivals += UInt64(1)
    if synchronized.state.active
        synchronized.counters.expired_windows += UInt64(1)
        _close_synchronized_window!(synchronized)
    end
    return false
end

@inline function _admit_synchronized_match!(
    synchronized,
    metadata::FilterGraphAlgorithms.SampleMetadata,
    matching::SynchronizedAcquisitionMatch,
)
    matching.domain_changed && return _reject_incoherent_arrival!(synchronized)
    identity = FilterGraphAlgorithms.sample_acquisition_identity(metadata)
    isnothing(identity) &&
        return _reject_incoherent_arrival!(synchronized)

    if matching.has_identity
        if identity.domain != matching.domain
            matching.domain_changed = true
            synchronized.counters.domain_changes += UInt64(1)
            return _reject_incoherent_arrival!(synchronized)
        end
        if identity.generation < matching.generation
            synchronized.counters.stale_arrivals += UInt64(1)
            return false
        end
    end

    # A corrupt arrival can retire an old window but cannot adopt a generation
    # or reset scientific state using an untrusted Header.
    if metadata.corrupted &&
            (!matching.has_identity || identity.generation > matching.generation)
        synchronized.counters.corrupted_windows += UInt64(1)
        if synchronized.state.active
            synchronized.counters.expired_windows += UInt64(1)
            _close_synchronized_window!(synchronized)
        end
        return false
    end

    state = synchronized.state
    if !matching.has_identity
        matching.has_identity = true
        matching.domain = identity.domain
        matching.generation = identity.generation
    elseif identity.generation > matching.generation
        if state.active
            synchronized.counters.expired_windows += UInt64(1)
            _close_synchronized_window!(synchronized)
        end
        reset!(synchronized.graph)
        matching.generation = identity.generation
        state.has_last_sequence = false
        synchronized.counters.discontinuities += UInt64(1)
        _open_synchronized_window!(synchronized, identity.sequence, true)
        return true
    end

    # A discontinuity flag cannot authorize a rewind inside an acquisition
    # generation. Only a strictly newer generation permits a sequence reset.
    sequence_metadata = FilterGraphAlgorithms.SampleMetadata(metadata;
        sequence=identity.sequence, discontinuity=false)
    _admit_synchronized_sequence!(synchronized, sequence_metadata) || return false
    if metadata.discontinuity && !state.discontinuity
        synchronized.counters.discontinuities += UInt64(1)
        state.discontinuity = true
    end
    return true
end

@inline function _synchronized_metadata(synchronized::PreparedSynchronizedInputs)
    if synchronized.state.discontinuity
        for slot in values(synchronized.input_metadata)
            slot.present && (slot.value = _metadata_with_discontinuity(slot.value))
        end
    end
    admitted = _admitted_metadata_values(Base.values(synchronized.input_metadata))
    return NamedTuple{keys(synchronized.graph.input_formats)}(admitted)
end

@inline function _finish_synchronized_window_candidate!(
    outputs::NamedTuple,
    synchronized::PreparedSynchronizedInputs,
)
    counters = synchronized.counters
    state = synchronized.state
    if state.corrupted
        counters.corrupted_windows += UInt64(1)
        _close_synchronized_window!(synchronized)
        return _unavailable_result(synchronized.graph)
    end

    result = try
        _process_candidate_admitted!(
            outputs,
            synchronized.graph,
            synchronized.input_buffers,
            _synchronized_metadata(synchronized),
        )
    catch
        counters.aborted_windows += UInt64(1)
        _close_synchronized_window!(synchronized)
        rethrow()
    end
    if !(result isa NamedTuple)
        counters.rejected_windows += UInt64(1)
        _close_synchronized_window!(synchronized)
        return result
    end
    if !result.terminal
        counters.aborted_windows += UInt64(1)
        _close_synchronized_window!(synchronized)
        error("a prepared synchronized-input Graph produced deferred output")
    end

    _close_synchronized_window!(synchronized)
    return result
end

@inline function _admit_synchronized_input!(
    synchronized::PreparedSynchronizedInputs,
    input,
    metadata::FilterGraphAlgorithms.SampleMetadata,
    ::Val{Index},
) where {Index}
    counters = synchronized.counters
    state = synchronized.state
    counters.arrivals += UInt64(1)
    _admit_synchronized_match!(synchronized, metadata, synchronized.match) ||
        return false

    received = getfield(state.received, Index)
    if received[]
        counters.duplicate_arrivals += UInt64(1)
        return false
    end

    if !metadata.corrupted
        copyto!(getfield(synchronized.input_buffers, Index), input)
    end
    stored_metadata = state.discontinuity ? _metadata_with_discontinuity(metadata) : metadata
    _store_metadata!(getfield(synchronized.input_metadata, Index), stored_metadata)
    received[] = true
    state.corrupted |= metadata.corrupted
    return true
end

@inline function _process_synchronized_candidate_admitted!(outputs::NamedTuple,
    synchronized::PreparedSynchronizedInputs,input,metadata::FilterGraphAlgorithms.SampleMetadata,index::Val)
    _admit_synchronized_input!(synchronized,input,metadata,index) ||
        return _unavailable_result(synchronized.graph)
    _all_received(synchronized.state.received) ||
        return _unavailable_result(synchronized.graph)
    return _finish_synchronized_window_candidate!(outputs, synchronized)
end

@inline function _process_synchronized_named!(
    outputs,
    synchronized,
    input_name::Symbol,
    input,
    metadata,
    ::Tuple{},
    ::Val,
)
    throw(ArgumentError("unknown synchronized Graph input Port '$input_name'"))
end

@inline function _process_synchronized_named!(
    outputs,
    synchronized,
    input_name::Symbol,
    input,
    metadata,
    input_names::Tuple,
    ::Val{Index},
) where {Index}
    if input_name === first(input_names)
        format = getfield(synchronized.graph.input_formats, Index)
        FilterGraphAlgorithms._validate_declared_array(format, input, ("Graph input", input_name))
        return _process_synchronized_candidate_admitted!(
            outputs,
            synchronized,
            input,
            metadata,
            Val(Index),
        )
    end
    return _process_synchronized_named!(
        outputs,
        synchronized,
        input_name,
        input,
        metadata,
        Base.tail(input_names),
        Val(Index + 1),
    )
end

"""
    process!(outputs, synchronized, input_name, input, metadata)

Accept one complete external input arrival. Incomplete, stale, duplicate, or
corrupted windows return an all-false output-availability `NamedTuple`. The
arrival that completes a valid window processes the Graph and returns its
availability or declared Algorithm failure.

`metadata` is required because an all-required window must associate arrivals
with an explicit sample sequence. The input is copied into prepared storage;
the caller retains ownership and may reuse its buffer after this call returns.
"""
function process!(
    outputs::NamedTuple,
    synchronized::PreparedSynchronizedInputs,
    input_name::Symbol,
    input::AbstractArray,
    metadata::FilterGraphAlgorithms.SampleMetadata,
)
    outputs = _validate_external_arrays(:output, synchronized.graph.output_formats, outputs)
    _validate_external_output_aliases!(keys(synchronized.graph.output_formats), values(outputs))
    result = _process_synchronized_named!(
        outputs,
        synchronized,
        input_name,
        input,
        metadata,
        keys(synchronized.graph.input_formats),
        Val(1),
    )
    result isa NamedTuple || return result
    result.terminal && (synchronized.counters.completed_windows += UInt64(1))
    return result.availability
end

"""
    expire!(synchronized) -> Bool

Abandon the active incomplete input window at a serialized host timer boundary.
Returns `true` when a window was active. This function never waits and never
invokes the Graph.
"""
function expire!(synchronized::PreparedSynchronizedInputs)
    synchronized.state.active || return false
    synchronized.counters.expired_windows += UInt64(1)
    _close_synchronized_window!(synchronized)
    return true
end

"""
    synchronized_input_counters(synchronized::PreparedSynchronizedInputs)

Return an allocation-free snapshot of synchronized-input counters.
"""
@inline function synchronized_input_counters(synchronized::PreparedSynchronizedInputs)
    counters = synchronized.counters
    return (
        arrivals=counters.arrivals,
        completed_windows=counters.completed_windows,
        expired_windows=counters.expired_windows,
        corrupted_windows=counters.corrupted_windows,
        rejected_windows=counters.rejected_windows,
        aborted_windows=counters.aborted_windows,
        stale_arrivals=counters.stale_arrivals,
        duplicate_arrivals=counters.duplicate_arrivals,
        discontinuities=counters.discontinuities,
        incoherent_arrivals=counters.incoherent_arrivals,
        domain_changes=counters.domain_changes,
    )
end

function _clear_synchronized_input_counters!(synchronized::PreparedSynchronizedInputs)
    counters = synchronized.counters
    counters.arrivals = UInt64(0)
    counters.completed_windows = UInt64(0)
    counters.expired_windows = UInt64(0)
    counters.corrupted_windows = UInt64(0)
    counters.rejected_windows = UInt64(0)
    counters.aborted_windows = UInt64(0)
    counters.stale_arrivals = UInt64(0)
    counters.duplicate_arrivals = UInt64(0)
    counters.discontinuities = UInt64(0)
    counters.incoherent_arrivals = UInt64(0)
    counters.domain_changes = UInt64(0)
    return synchronized
end

function _reset_synchronized_sequence!(state, ::Val{:sequence})
    state.has_last_sequence = false
    state.last_sequence = UInt64(0)
    return nothing
end

function _reset_synchronized_sequence!(state, ::SynchronizedAcquisitionMatch)
    if state.active
        state.has_last_sequence = true
        state.last_sequence = state.sequence
    end
    return nothing
end

function reset!(synchronized::PreparedSynchronizedInputs)
    reset!(synchronized.graph)
    state = synchronized.state
    _reset_synchronized_sequence!(state, synchronized.match)
    state.active = false
    state.sequence = UInt64(0)
    state.corrupted = false
    state.discontinuity = false
    _clear_received!(state.received)
    _clear_metadata!(values(synchronized.input_metadata))
    return synchronized
end
