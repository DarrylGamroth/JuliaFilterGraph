const _LATEST_HOLD_EMPTY = UInt8(0)
const _LATEST_HOLD_VALID = UInt8(1)
const _LATEST_HOLD_INVALID = UInt8(2)

mutable struct LatestHoldState
    status::UInt8
    fresh::Bool
    primary_cycles::UInt32
    has_sequence::Bool
    last_sequence::UInt64
    last_identity::Union{Nothing,FilterGraphAlgorithms.SampleAcquisitionIdentity}
    metadata::SampleMetadataSlot
end

LatestHoldState() = LatestHoldState(
    _LATEST_HOLD_EMPTY,
    false,
    UInt32(0),
    false,
    UInt64(0),
    nothing,
    SampleMetadataSlot(),
)

mutable struct LatestHoldCounters
    primary_arrivals::UInt64
    held_arrivals::UInt64
    new_uses::UInt64
    retained_uses::UInt64
    skipped_primary::UInt64
    invalid_primary::UInt64
    invalid_held::UInt64
    stale_held::UInt64
    discontinuities::UInt64
    timeouts::UInt64
    overruns::UInt64
end

LatestHoldCounters() = LatestHoldCounters(ntuple(_ -> UInt64(0), Val(11))...)

"""
    PreparedLatestHold

One prepared two-input Graph data plane in which a faster primary input clocks
processing and a slower input supplies the latest retained complete sample.
One buffer retains the latest sample and a second stages each arriving sample.
"""
mutable struct PreparedLatestHold{PrimaryIndex,HeldIndex,G,B}
    const graph::G
    held_buffer::B
    staging_buffer::B
    const state::LatestHoldState
    const counters::LatestHoldCounters
    const max_primary_cycles::UInt32
end

PreparedLatestHold{PrimaryIndex,HeldIndex}(
    graph::G,
    held_buffer::B,
    state::LatestHoldState,
    counters::LatestHoldCounters,
    max_primary_cycles::UInt32,
) where {PrimaryIndex,HeldIndex,G,B} = PreparedLatestHold{
    PrimaryIndex,
    HeldIndex,
    G,
    B,
}(graph, held_buffer, similar(held_buffer), state, counters, max_primary_cycles)

function _latest_hold_rate_ratio(primary, held)
    primary_numerator, primary_denominator = primary.rate
    held_numerator, held_denominator = held.rate
    numerator = UInt64(primary_numerator) * UInt64(held_denominator)
    denominator = UInt64(primary_denominator) * UInt64(held_numerator)
    numerator > denominator || throw(
        ArgumentError("the primary Port rate must be greater than the held Port rate"),
    )
    iszero(rem(numerator, denominator)) || throw(
        ArgumentError("the first latest/hold slice requires an integer Port-rate ratio"),
    )
    ratio = div(numerator, denominator)
    ratio <= typemax(UInt32) || throw(ArgumentError("the Port-rate ratio must fit UInt32"))
    return UInt32(ratio)
end

"""
    prepare_latest_hold(graph, primary, held)

Prepare the concrete bounded join used by a Graph with exactly two external
data inputs. `primary` names the faster input that clocks Graph processing;
`held` names the slower input whose latest valid sample is copied into one
Graph-owned slot and reused for the declared integer rate ratio.

This is a host/data-plane lifecycle boundary. It does not change the
scientist-facing Algorithm interface or the authored FGN Graph vocabulary.
"""
function prepare_latest_hold(graph::PreparedGraph, primary::Symbol, held::Symbol)
    input_names = keys(graph.input_formats)
    length(input_names) == 2 || throw(
        ArgumentError("the first latest/hold slice requires exactly two external data Ports"),
    )
    primary == held && throw(ArgumentError("primary and held Ports must be different"))
    primary_index = findfirst(==(primary), input_names)
    isnothing(primary_index) && throw(ArgumentError("unknown primary Graph input Port '$primary'"))
    held_index = findfirst(==(held), input_names)
    isnothing(held_index) && throw(ArgumentError("unknown held Graph input Port '$held'"))
    any(values(graph.output_conditional)) && throw(
        ArgumentError("the first latest/hold slice requires terminal Graph outputs"),
    )
    primary_format = getfield(graph.input_formats, primary_index)
    held_format = getfield(graph.input_formats, held_index)
    max_primary_cycles = _latest_hold_rate_ratio(primary_format, held_format)
    held_buffer = _allocate_port_buffer(held_format)
    join = PreparedLatestHold{primary_index,held_index}(
        graph,
        held_buffer,
        LatestHoldState(),
        LatestHoldCounters(),
        max_primary_cycles,
    )
    primary_type = _dense_buffer_type(primary_format)
    output_type = _external_buffer_type(graph.output_formats)
    precompile(
        process!,
        (
            output_type,
            typeof(join),
            primary_type,
        ),
    )
    precompile(
        process!,
        (
            output_type,
            typeof(join),
            primary_type,
            Nothing,
        ),
    )
    precompile(
        accept_held!,
        (typeof(join), typeof(held_buffer), Nothing),
    )
    precompile(reset!, (typeof(join),))
    return join
end

@inline function _metadata_with_discontinuity(metadata::FilterGraphAlgorithms.SampleMetadata)
    metadata.discontinuity && return metadata
    return FilterGraphAlgorithms.SampleMetadata(metadata; discontinuity=true)
end

@inline function _accept_held_admitted!(
    join::PreparedLatestHold,
    held,
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    state = join.state
    counters = join.counters
    counters.held_arrivals += UInt64(1)

    if metadata.present
        sample = metadata.value
        identity = FilterGraphAlgorithms.sample_acquisition_identity(sample)
        previous = state.last_identity
        if identity === nothing && previous === nothing
            stale = state.has_sequence && !sample.discontinuity && sample.sequence <= state.last_sequence
            discontinuity = sample.discontinuity ||
                (state.has_sequence && sample.sequence != state.last_sequence + UInt64(1))
        elseif identity === nothing
            stale, discontinuity = true, false
        elseif previous === nothing
            stale, discontinuity = false, sample.discontinuity
        else
            same_domain = identity.domain == previous.domain
            same_generation = identity.generation == previous.generation
            stale = !same_domain || identity.generation < previous.generation ||
                (same_generation && identity.sequence <= previous.sequence)
            discontinuity = sample.discontinuity || !same_generation ||
                (same_generation && identity.sequence != previous.sequence + UInt64(1))
        end
        stale && (counters.stale_held += UInt64(1); return false)
        sample.corrupted || copyto!(join.staging_buffer, held)
        state.has_sequence = true
        state.last_sequence = sample.sequence
        state.last_identity = identity
        if sample.corrupted
            counters.invalid_held += UInt64(1)
            discontinuity && (counters.discontinuities += UInt64(1))
            state.status = _LATEST_HOLD_INVALID
            state.fresh = false
            state.primary_cycles = UInt32(0)
            _store_metadata!(state.metadata, metadata)
            return false
        end
        if discontinuity
            counters.discontinuities += UInt64(1)
            metadata = FilterGraphAlgorithms.AdmittedSampleMetadata(
                true,
                _metadata_with_discontinuity(sample),
            )
        end
    else
        copyto!(join.staging_buffer, held)
    end

    state.status == _LATEST_HOLD_VALID && state.fresh &&
        (counters.overruns += UInt64(1))
    join.held_buffer, join.staging_buffer = join.staging_buffer, join.held_buffer
    _store_metadata!(state.metadata, metadata)
    state.status = _LATEST_HOLD_VALID
    state.fresh = true
    state.primary_cycles = UInt32(0)
    return true
end

"""
    accept_held!(join, held[, metadata]) -> Bool

Copy one slower input arrival into the prepared one-sample retention slot.
Returns `true` when the sample becomes the retained value. Corrupted or
out-of-order arrivals return `false`, update counters, and never reach an
Algorithm.
"""
function accept_held!(
    join::PreparedLatestHold{PrimaryIndex,HeldIndex},
    held,
    metadata::Union{Nothing,FilterGraphAlgorithms.SampleMetadata}=nothing,
) where {PrimaryIndex,HeldIndex}
    held_format = getfield(join.graph.input_formats, HeldIndex)
    FilterGraphAlgorithms._validate_declared_array(
        held_format, held, ("Graph input", keys(join.graph.input_formats)[HeldIndex]),
    )
    return _accept_held_admitted!(
        join,
        held,
        FilterGraphAlgorithms._admitted_sample_metadata(metadata),
    )
end

@inline function _latest_hold_values(join::PreparedLatestHold{1,2}, primary)
    return (primary, join.held_buffer)
end

@inline function _latest_hold_values(join::PreparedLatestHold{2,1}, primary)
    return (join.held_buffer, primary)
end

@inline function _latest_hold_metadata(
    join::PreparedLatestHold{1,2},
    primary::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    return (primary, _admitted_metadata(join.state.metadata))
end

@inline function _latest_hold_metadata(
    join::PreparedLatestHold{2,1},
    primary::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    return (_admitted_metadata(join.state.metadata), primary)
end

@inline function _latest_hold_inputs(join::PreparedLatestHold, primary)
    return NamedTuple{keys(join.graph.input_formats)}(_latest_hold_values(join, primary))
end

@inline function _latest_hold_input_metadata(
    join::PreparedLatestHold,
    primary::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    return NamedTuple{keys(join.graph.input_formats)}(_latest_hold_metadata(join, primary))
end

@inline function _unavailable_result(graph::PreparedGraph)
    availability = NamedTuple{keys(graph.output_formats)}(
        ntuple(_ -> false, length(graph.output_formats)),
    )
    return (availability=availability, terminal=false)
end

@inline function _process_primary_candidate_admitted!(
    outputs::NamedTuple,
    join::PreparedLatestHold,
    primary,
    primary_metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
)
    counters = join.counters
    state = join.state
    counters.primary_arrivals += UInt64(1)

    if state.status != _LATEST_HOLD_VALID
        primary_metadata.present && primary_metadata.value.corrupted &&
            (counters.invalid_primary += UInt64(1))
        counters.skipped_primary += UInt64(1)
        return _unavailable_result(join.graph)
    end
    if state.primary_cycles >= join.max_primary_cycles
        counters.timeouts += UInt64(1)
        counters.skipped_primary += UInt64(1)
        state.status = _LATEST_HOLD_EMPTY
        state.fresh = false
        return _unavailable_result(join.graph)
    end
    state.primary_cycles += UInt32(1)
    if primary_metadata.present && primary_metadata.value.corrupted
        counters.invalid_primary += UInt64(1)
        counters.skipped_primary += UInt64(1)
        return _unavailable_result(join.graph)
    end

    if state.fresh
        counters.new_uses += UInt64(1)
        state.fresh = false
    else
        counters.retained_uses += UInt64(1)
    end
    inputs = _latest_hold_inputs(join, primary)
    metadata = _latest_hold_input_metadata(join, primary_metadata)
    return _process_candidate_admitted!(outputs, join.graph, inputs, metadata)
end

"""
    process!(outputs, join::PreparedLatestHold, primary[, metadata])

Process one primary arrival with the current retained held input. A missing,
invalid, or rate-expired held input produces an all-false availability result
without invoking the Graph. Expected Algorithm failures retain their declared
enum result and the Graph's per-Algorithm state semantics.
"""
function process!(
    outputs::NamedTuple,
    join::PreparedLatestHold{PrimaryIndex},
    primary,
    metadata::Union{Nothing,FilterGraphAlgorithms.SampleMetadata}=nothing,
) where {PrimaryIndex}
    outputs = _validate_external_arrays(:output, join.graph.output_formats, outputs)
    _validate_external_output_aliases!(keys(join.graph.output_formats), values(outputs))
    primary_format = getfield(join.graph.input_formats, PrimaryIndex)
    FilterGraphAlgorithms._validate_declared_array(
        primary_format, primary, ("Graph input", keys(join.graph.input_formats)[PrimaryIndex]),
    )
    result = _process_primary_candidate_admitted!(
        outputs,
        join,
        primary,
        FilterGraphAlgorithms._admitted_sample_metadata(metadata),
    )
    result isa NamedTuple || return result
    return result.availability
end

"""
    latest_hold_counters(join::PreparedLatestHold)

Return an allocation-free snapshot of the latest/hold data-plane counters.
"""
@inline function latest_hold_counters(join::PreparedLatestHold)
    counters = join.counters
    return (
        primary_arrivals=counters.primary_arrivals,
        held_arrivals=counters.held_arrivals,
        new_uses=counters.new_uses,
        retained_uses=counters.retained_uses,
        skipped_primary=counters.skipped_primary,
        invalid_primary=counters.invalid_primary,
        invalid_held=counters.invalid_held,
        stale_held=counters.stale_held,
        discontinuities=counters.discontinuities,
        timeouts=counters.timeouts,
        overruns=counters.overruns,
    )
end

function _clear_latest_hold_counters!(join::PreparedLatestHold)
    counters = join.counters
    counters.primary_arrivals = UInt64(0)
    counters.held_arrivals = UInt64(0)
    counters.new_uses = UInt64(0)
    counters.retained_uses = UInt64(0)
    counters.skipped_primary = UInt64(0)
    counters.invalid_primary = UInt64(0)
    counters.invalid_held = UInt64(0)
    counters.stale_held = UInt64(0)
    counters.discontinuities = UInt64(0)
    counters.timeouts = UInt64(0)
    counters.overruns = UInt64(0)
    return join
end

function reset!(join::PreparedLatestHold)
    reset!(join.graph)
    state = join.state
    state.status = _LATEST_HOLD_EMPTY
    state.fresh = false
    state.primary_cycles = UInt32(0)
    state.has_sequence = false
    state.last_sequence = UInt64(0)
    state.last_identity = nothing
    _store_metadata!(state.metadata, FilterGraphAlgorithms._NO_SAMPLE_METADATA)
    return join
end
