"The source-reported electrical trigger classification associated with a sample."
@enum SampleTriggerState::UInt8 begin
    SAMPLE_TRIGGER_ABSENT = 0
    SAMPLE_TRIGGER_LOW = 1
    SAMPLE_TRIGGER_HIGH = 2
    SAMPLE_TRIGGER_MIXED = 3
end

@doc "No electrical trigger classification was supplied." SAMPLE_TRIGGER_ABSENT
@doc "The source reported a low trigger level." SAMPLE_TRIGGER_LOW
@doc "The source reported a high trigger level." SAMPLE_TRIGGER_HIGH
@doc "The source reported both trigger levels during the acquisition." SAMPLE_TRIGGER_MIXED

"""Application-assigned optical role of one SCC chopped image."""
@enum ChoppedImageKind::UInt8 begin
    ChoppedImageFringed = 0
    ChoppedImageUnfringed = 1
    ChoppedImageMixed = 2
    ChoppedImageAbsent = 3
end

@doc "The SCC reference fringes are present in this image." ChoppedImageFringed
@doc "The SCC reference fringes are absent from this image." ChoppedImageUnfringed
@doc "The image contains both SCC optical phases and cannot form a pure pair." ChoppedImageMixed
@doc "The source explicitly reports no optical classification for this exposure." ChoppedImageAbsent

"Transport-neutral physical acquisition identity, including its source cohort."
struct SampleAcquisitionIdentity
    domain::NTuple{16,UInt8}
    generation::UInt64
    sequence::UInt64

    function SampleAcquisitionIdentity(domain::NTuple{16,UInt8}, generation::Integer,
                                       sequence::Integer)
        all(iszero, domain) && throw(ArgumentError("acquisition domain must be nonzero"))
        0 <= generation <= typemax(UInt64) || throw(ArgumentError("generation must fit UInt64"))
        0 <= sequence <= typemax(UInt64) || throw(ArgumentError("sequence must fit UInt64"))
        return new(domain, UInt64(generation), UInt64(sequence))
    end
end

"The declared clock used by exposure and reference timestamps."
@enum SampleClock::UInt32 begin
    SAMPLE_CLOCK_MONOTONIC = 1
    SAMPLE_CLOCK_REALTIME = 2
    SAMPLE_CLOCK_TAI = 3
    SAMPLE_CLOCK_MODEL = 4
end

@doc "Linux monotonic clock; no synchronization is implied." SAMPLE_CLOCK_MONOTONIC
@doc "Realtime epoch; this label alone asserts no synchronization." SAMPLE_CLOCK_REALTIME
@doc "TAI time scale." SAMPLE_CLOCK_TAI
@doc "Simulation model time, independent of execution clocks." SAMPLE_CLOCK_MODEL

"""
    SampleExposureStart(nanoseconds; clock=SAMPLE_CLOCK_MONOTONIC)

A physical exposure start in the declared clock, represented as integer
nanoseconds. A clock label makes no synchronization or accuracy claim.
Model time remains distinct from execution wall time.
"""
struct SampleExposureStart
    nanoseconds::Int64
    clock::SampleClock

    function SampleExposureStart(nanoseconds::Integer;
            clock::SampleClock=SAMPLE_CLOCK_MONOTONIC)
        typemin(Int64) <= nanoseconds <= typemax(Int64) || throw(
            ArgumentError("exposure start must fit Int64"))
        return new(Int64(nanoseconds), clock)
    end
end

"""
    SampleReferenceTimestamp(timestamp_nanoseconds; clock=SAMPLE_CLOCK_REALTIME,
                             duration_nanoseconds=nothing)

One source-associated timestamp in the declared clock and an optional duration.
The timestamp is unsigned integer nanoseconds; UInt64 maximum is reserved.
Clock labels alone establish neither synchronization nor accuracy. This fact
follows the triggering input through derived outputs and carries no event history.
"""
struct SampleReferenceTimestamp
    timestamp_nanoseconds::UInt64
    clock::SampleClock
    has_duration::Bool
    duration_nanoseconds::UInt64

    function SampleReferenceTimestamp(timestamp_nanoseconds::Integer;
            clock::SampleClock=SAMPLE_CLOCK_REALTIME,
            duration_nanoseconds::Union{Nothing,Integer}=nothing)
        0 <= timestamp_nanoseconds < typemax(UInt64) || throw(
            ArgumentError("reference timestamp must be in 0:UInt64_MAX-1"))
        duration = if duration_nanoseconds === nothing
            typemax(UInt64)
        else
            0 <= duration_nanoseconds < typemax(UInt64) || throw(
                ArgumentError("reference duration must be in 0:UInt64_MAX-1"))
            UInt64(duration_nanoseconds)
        end
        return new(UInt64(timestamp_nanoseconds), clock, duration_nanoseconds !== nothing, duration)
    end
end

const _EMPTY_SAMPLE_REFERENCE_TIMESTAMP = SampleReferenceTimestamp(0)

const _EMPTY_SAMPLE_ACQUISITION_IDENTITY = SampleAcquisitionIdentity(
    ntuple(_ -> UInt8(1), 16), 0, 0)
const _EMPTY_SAMPLE_EXPOSURE_START = SampleExposureStart(0)

"""
    SampleMetadata(sequence, offset=0; timestamp_nanoseconds=nothing,
                   terminal=false, discontinuity=false, corrupted=false,
                   trigger_high=missing, trigger_state=missing,
                   chopped_image_kind=nothing,
                   exposure_duration_nanoseconds=nothing,
                   acquisition_identity=nothing, exposure_start=nothing,
                   reference_timestamp=nothing)

Owned, transport-neutral identity and lifecycle facts for one input or output
artifact. `offset` is a zero-based semantic offset defined by its schema.
Absence of metadata is represented by `nothing`.

`timestamp_nanoseconds` is the publication timestamp projected from SPA Header;
it is not a physical exposure start. `exposure_start` preserves the physical
clock. `acquisition_identity` retains the indivisible
source domain, generation, and physical acquisition sequence. `sequence` is the
publication sequence; a held acquisition retains its identity while publication
advances. These counters need not be equal.
Exposure duration is positive when present.

`has_trigger_classification` distinguishes a missing property from an explicitly
reported `SAMPLE_TRIGGER_ABSENT`. `trigger_state=missing` is the default missing
property; explicit enum values supply a classification. `trigger_state` distinguishes
absent, low, high, and mixed source classifications.
Transported trigger classification applies to the whole exposure. A raw camera
tag alone does not establish that guarantee.
The `trigger_high` keyword is a convenience for low/high classification and
`nothing` explicitly clears that classification and its presence. Specifying both keywords
requires them to agree. `chopped_image_kind` is an independent application optical
role; `ChoppedImageAbsent` preserves an explicitly supplied absent optical fact,
while `nothing` means no optical property. Roles are assigned through an explicit
polarity mapping or supplied directly by the
application. Electrical and optical facts are transported independently. All storage is concrete and isbits.
"""
struct SampleMetadata
    sequence::UInt64
    offset::UInt32
    timestamp_nanoseconds::Int64
    has_timestamp::Bool
    terminal::Bool
    discontinuity::Bool
    corrupted::Bool
    has_trigger_classification::Bool
    trigger_state::SampleTriggerState
    has_chopped_image_kind::Bool
    chopped_image_kind::ChoppedImageKind
    has_exposure_duration::Bool
    exposure_duration_nanoseconds::UInt64
    has_acquisition_identity::Bool
    acquisition_identity::SampleAcquisitionIdentity
    has_exposure_start::Bool
    exposure_start::SampleExposureStart
    has_reference_timestamp::Bool
    reference_timestamp::SampleReferenceTimestamp

    @inline function SampleMetadata(
        sequence::UInt64, offset::UInt32, timestamp_nanoseconds::Int64,
        has_timestamp::Bool, terminal::Bool, discontinuity::Bool, corrupted::Bool,
        has_trigger_classification::Bool, trigger_state::SampleTriggerState, has_chopped_image_kind::Bool,
        chopped_image_kind::ChoppedImageKind, has_exposure_duration::Bool,
        exposure_duration_nanoseconds::UInt64, has_acquisition_identity::Bool,
        acquisition_identity::SampleAcquisitionIdentity, has_exposure_start::Bool,
        exposure_start::SampleExposureStart,
        has_reference_timestamp::Bool, reference_timestamp::SampleReferenceTimestamp,
    )
        has_exposure_duration && iszero(exposure_duration_nanoseconds) && throw(
            ArgumentError("exposure duration must be positive"))
        return new(sequence, offset, timestamp_nanoseconds, has_timestamp, terminal,
            discontinuity, corrupted, has_trigger_classification, trigger_state, has_chopped_image_kind,
            chopped_image_kind, has_exposure_duration,
            exposure_duration_nanoseconds, has_acquisition_identity,
            acquisition_identity, has_exposure_start, exposure_start,
            has_reference_timestamp, reference_timestamp)
    end
end

@inline function SampleMetadata(
    sequence::Integer,
    offset::Integer=0;
    timestamp_nanoseconds::Union{Nothing,Integer}=nothing,
    terminal::Bool=false,
    discontinuity::Bool=false,
    corrupted::Bool=false,
    trigger_state::Union{Missing,SampleTriggerState}=missing,
    trigger_high::Union{Nothing,Bool,Missing}=missing,
    chopped_image_kind::Union{Nothing,ChoppedImageKind}=nothing,
    exposure_duration_nanoseconds::Union{Nothing,Integer}=nothing,
    acquisition_identity::Union{Nothing,SampleAcquisitionIdentity}=nothing,
    exposure_start::Union{Nothing,SampleExposureStart}=nothing,
    reference_timestamp::Union{Nothing,SampleReferenceTimestamp}=nothing,
)
    0 <= sequence <= typemax(UInt64) || throw(ArgumentError("sequence must fit UInt64"))
    0 <= offset <= typemax(UInt32) || throw(ArgumentError("offset must fit UInt32"))
    has_timestamp = !isnothing(timestamp_nanoseconds)
    timestamp = if has_timestamp
        typemin(Int64) <= timestamp_nanoseconds <= typemax(Int64) || throw(
            ArgumentError("timestamp_nanoseconds must fit Int64"))
        Int64(timestamp_nanoseconds)
    else
        Int64(0)
    end
    has_trigger_classification = !ismissing(trigger_state)
    state = ismissing(trigger_state) ? SAMPLE_TRIGGER_ABSENT : trigger_state
    if !ismissing(trigger_high)
        pure_state = isnothing(trigger_high) ? SAMPLE_TRIGGER_ABSENT :
                     trigger_high ? SAMPLE_TRIGGER_HIGH : SAMPLE_TRIGGER_LOW
        ismissing(trigger_state) || state == pure_state || throw(
            ArgumentError("trigger_high and trigger_state disagree"))
        state = pure_state
        has_trigger_classification = !isnothing(trigger_high)
    end
    has_duration = !isnothing(exposure_duration_nanoseconds)
    if has_duration
        0 < exposure_duration_nanoseconds <= typemax(UInt64) || throw(
            ArgumentError("exposure_duration_nanoseconds must be positive and fit UInt64"))
    end
    return SampleMetadata(UInt64(sequence), UInt32(offset), timestamp, has_timestamp,
        terminal, discontinuity, corrupted, has_trigger_classification, state, !isnothing(chopped_image_kind),
        isnothing(chopped_image_kind) ? ChoppedImageFringed : chopped_image_kind, has_duration,
        has_duration ? UInt64(exposure_duration_nanoseconds) : UInt64(0),
        !isnothing(acquisition_identity),
        isnothing(acquisition_identity) ? _EMPTY_SAMPLE_ACQUISITION_IDENTITY : acquisition_identity,
        !isnothing(exposure_start),
        isnothing(exposure_start) ? _EMPTY_SAMPLE_EXPOSURE_START : exposure_start,
        reference_timestamp !== nothing,
        something(reference_timestamp, _EMPTY_SAMPLE_REFERENCE_TIMESTAMP))
end

"""
Copy sample metadata while replacing selected fields. Electrical and optical
annotations are independent: replacing a trigger preserves an assigned optical
role. If that role was derived from the trigger, rerun `chopped_image_metadata`
after replacing the electrical annotation.
"""
@inline function SampleMetadata(
    metadata::SampleMetadata;
    sequence::Integer=metadata.sequence,
    offset::Integer=metadata.offset,
    timestamp_nanoseconds::Union{Nothing,Integer}=(
        metadata.has_timestamp ? metadata.timestamp_nanoseconds : nothing),
    terminal::Bool=metadata.terminal,
    discontinuity::Bool=metadata.discontinuity,
    corrupted::Bool=metadata.corrupted,
    trigger_state::Union{Missing,SampleTriggerState}=missing,
    trigger_high::Union{Nothing,Bool,Missing}=missing,
    chopped_image_kind::Union{Nothing,ChoppedImageKind}=sample_chopped_image_kind(metadata),
    exposure_duration_nanoseconds::Union{Nothing,Integer}=sample_exposure_duration_nanoseconds(metadata),
    acquisition_identity::Union{Nothing,SampleAcquisitionIdentity}=sample_acquisition_identity(metadata),
    exposure_start::Union{Nothing,SampleExposureStart}=sample_exposure_start(metadata),
    reference_timestamp::Union{Nothing,SampleReferenceTimestamp}=sample_reference_timestamp(metadata),
)
    # A trigger_high replacement also replaces the enum inherited from the source.
    state = ismissing(trigger_state) && ismissing(trigger_high) ?
        (metadata.has_trigger_classification ? metadata.trigger_state : missing) : trigger_state
    return SampleMetadata(sequence, offset; timestamp_nanoseconds, terminal,
        discontinuity, corrupted, trigger_state=state, trigger_high, chopped_image_kind,
        exposure_duration_nanoseconds, acquisition_identity, exposure_start, reference_timestamp)
end

"""
    sample_trigger_state(metadata::SampleMetadata)

Return the source-reported electrical trigger classification.
"""
sample_trigger_state(metadata::SampleMetadata) = metadata.trigger_state

"""
    sample_trigger_high(metadata::SampleMetadata)

Return the reported trigger level, or `nothing` for absent/mixed classification.
"""
sample_trigger_high(metadata::SampleMetadata) =
    metadata.trigger_state == SAMPLE_TRIGGER_HIGH ? true :
    metadata.trigger_state == SAMPLE_TRIGGER_LOW ? false : nothing

"""
    sample_chopped_image_kind(metadata::SampleMetadata)

Return the application-assigned SCC optical role, or `nothing` when absent.
"""
sample_chopped_image_kind(metadata::SampleMetadata) =
    metadata.has_chopped_image_kind ? metadata.chopped_image_kind : nothing

"""
    sample_exposure_duration_nanoseconds(metadata::SampleMetadata)

Return physical exposure duration in nanoseconds, or `nothing` when absent.
"""
sample_exposure_duration_nanoseconds(metadata::SampleMetadata) =
    metadata.has_exposure_duration ? metadata.exposure_duration_nanoseconds : nothing

"""
    sample_acquisition_identity(metadata::SampleMetadata)

Return the indivisible acquisition identity, or `nothing` when absent.
"""
sample_acquisition_identity(metadata::SampleMetadata) =
    metadata.has_acquisition_identity ? metadata.acquisition_identity : nothing

"""
    sample_exposure_start(metadata::SampleMetadata)

Return the clock-qualified physical exposure start, or `nothing` when absent.
"""
sample_exposure_start(metadata::SampleMetadata) =
    metadata.has_exposure_start ? metadata.exposure_start : nothing

"Return the source-associated reference timestamp, or `nothing` when unavailable."
sample_reference_timestamp(metadata::SampleMetadata) =
    metadata.has_reference_timestamp ? metadata.reference_timestamp : nothing

"""
    alternating_trigger_metadata(metadata; even_is_high=true)

Annotate a sample's in-band digital trigger from the parity of its `sequence`
frame identity. This ingress helper is useful for simulated or untagged
alternating acquisition and does not create a Graph data Port.
"""
function alternating_trigger_metadata(
    metadata::SampleMetadata;
    even_is_high::Bool=true,
)
    even = iseven(metadata.sequence)
    return SampleMetadata(metadata; trigger_high=(even == even_is_high))
end

const _EMPTY_SAMPLE_METADATA = SampleMetadata(0)

"""
    AdmittedSampleMetadata(present::Bool, value::SampleMetadata)

Concrete optional sample metadata for prepared execution and frame observation.
Inspect `present` before reading `value`; when false, `value` is unspecified and
must not establish source identity, time, continuity or sample validity. When
true, `value` contains the actual public [`SampleMetadata`](@ref). The immutable
record is isbits and can be copied without retaining borrowed array storage.
"""
struct AdmittedSampleMetadata
    present::Bool
    value::SampleMetadata
end

const _NO_SAMPLE_METADATA = AdmittedSampleMetadata(false, _EMPTY_SAMPLE_METADATA)

@inline _admitted_sample_metadata(::Nothing) = _NO_SAMPLE_METADATA
@inline _admitted_sample_metadata(metadata::SampleMetadata) =
    AdmittedSampleMetadata(true, metadata)
@inline _admitted_sample_metadata(metadata::AdmittedSampleMetadata) = metadata
@inline _sample_metadata(metadata::AdmittedSampleMetadata) =
    metadata.present ? metadata.value : nothing

# Header sequence/offset belong to publication. A row series with physical
# identity must use the full authority tuple and unchanged exposure facts.
@inline function _same_row_block_metadata(a::SampleMetadata,b::SampleMetadata)
    a.has_acquisition_identity==b.has_acquisition_identity || return false
    if a.has_acquisition_identity
        a.acquisition_identity==b.acquisition_identity || return false
    else
        a.sequence==b.sequence || return false
    end
    return a.has_exposure_duration==b.has_exposure_duration &&
        (!a.has_exposure_duration || a.exposure_duration_nanoseconds==b.exposure_duration_nanoseconds) &&
        a.has_exposure_start==b.has_exposure_start && (!a.has_exposure_start || a.exposure_start==b.exposure_start) &&
        a.has_trigger_classification==b.has_trigger_classification && (!a.has_trigger_classification || a.trigger_state==b.trigger_state) &&
        a.has_chopped_image_kind==b.has_chopped_image_kind && (!a.has_chopped_image_kind || a.chopped_image_kind==b.chopped_image_kind) &&
        a.has_reference_timestamp==b.has_reference_timestamp &&
        (!a.has_reference_timestamp || a.reference_timestamp==b.reference_timestamp)
end
