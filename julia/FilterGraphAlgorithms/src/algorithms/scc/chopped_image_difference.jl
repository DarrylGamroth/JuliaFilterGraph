"""
    chopped_image_metadata(metadata; trigger_high_is_fringed)

Interpret the electrical trigger for an SCC application using explicitly supplied
polarity. Preserve the original trigger and acquisition facts, and replace the
optical annotation. An absent trigger produces no optical role; known mixed
classification produces [`ChoppedImageMixed`](@ref). Camera decoding alone does
not select this application meaning or establish whole-exposure timing. Call
this mapper after electrical annotation and rerun it if that annotation changes.
"""
function chopped_image_metadata(metadata::SampleMetadata; trigger_high_is_fringed::Bool)
    trigger = sample_trigger_state(metadata)
    kind = if trigger == SAMPLE_TRIGGER_ABSENT
        nothing
    elseif trigger == SAMPLE_TRIGGER_MIXED
        ChoppedImageMixed
    elseif (trigger == SAMPLE_TRIGGER_HIGH) == trigger_high_is_fringed
        ChoppedImageFringed
    else
        ChoppedImageUnfringed
    end
    return SampleMetadata(metadata; chopped_image_kind=kind)
end

"""Publication status of one rolling SCC chopped-image result."""
@enum ChoppedImagePairStatus::UInt8 begin
    ChoppedImagePriming = 0
    ChoppedImageComplete = 1
end

@doc "No adjacent opposite-labeled pair is available."
ChoppedImagePriming
@doc "An adjacent opposite-labeled pair produced the output difference."
ChoppedImageComplete

"""
    ChoppedImageDifferencePlan(T, pixel_count, trigger_high_is_fringed)
    ChoppedImageDifferencePlan(pixel_count, trigger_high_is_fringed=true)

Prepare rolling adjacent SCC chopped-image differencing for one nonempty
image extent. `input_schema` identifies the pixel units without applying any
calibration. The Algorithm prefers an explicit in-band optical role.
The non-strict array mode can map an electrical trigger using the Plan's explicit
polarity when no optical role is supplied. In that mode a mixed optical role is rejected
without changing history.
The direct Bool processing interface uses the Plan's polarity. Once primed, an
opposite-labeled image produces `fringed - unfringed`; a repeated label emits
priming output and replaces the candidate retained image.

With `require_acquisition_identity=true`, declaration processing requires the
source domain, generation and physical acquisition sequence, positive duration
and explicit optical classification. Pair durations must agree. Publication
counters do not govern pairing. Gaps, a new cohort or discontinuity prime
using the current pure exposure. Mixed or corrupted exposures consume identity
and clear history; two new pure alternating exposures are then required.
Older generations within the same authority and repeated or decreasing sequence
within a cohort reject without changing history.
The explicit `false` and direct Bool processing retain the host-supervised
sequence/reset contract. Neither mode compensates different exposure commands.
"""
struct ChoppedImageDifferencePlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    pixel_count::Int
    trigger_high_kind::ChoppedImageKind
    require_acquisition_identity::Bool

    function ChoppedImageDifferencePlan(
        ::Type{T},
        pixel_count::Integer,
        trigger_high_is_fringed::Bool;
        require_acquisition_identity::Bool=false,
    ) where {T<:AbstractFloat}
        pixels = Int(pixel_count)
        pixels > 0 ||
            throw(ArgumentError("chopped image pixel_count must be positive"))
        high_kind = trigger_high_is_fringed ?
                    ChoppedImageFringed : ChoppedImageUnfringed
        return new{T}(pixels, high_kind, require_acquisition_identity)
    end
end

ChoppedImageDifferencePlan(
    ::Type{T},
    pixel_count::Integer,
) where {T<:AbstractFloat} = ChoppedImageDifferencePlan(T, pixel_count, true)

ChoppedImageDifferencePlan(
    pixel_count::Integer,
    trigger_high_is_fringed::Bool=true,
) = ChoppedImageDifferencePlan(
    Float32,
    pixel_count,
    trigger_high_is_fringed,
)

"""
    ChoppedImageDifferenceWorkspace(plan)

Own the committed preceding chopped image, its candidate replacement, and
fixed output staging for one Node. Processing never changes committed history.
[`commit!`](@ref) advances the retained image after this Algorithm succeeds;
[`discard!`](@ref) abandons the candidate after its rejection or abort.
"""
mutable struct ChoppedImageDifferenceWorkspace{T<:AbstractFloat}
    committed_primed::Bool
    candidate_ready::Bool
    candidate_primed::Bool
    committed_identity::Union{Nothing,SampleAcquisitionIdentity}
    candidate_identity::Union{Nothing,SampleAcquisitionIdentity}
    committed_duration::UInt64
    candidate_duration::UInt64
    committed_kind::ChoppedImageKind
    candidate_kind::ChoppedImageKind
    committed_image::Vector{T}
    candidate_image::Vector{T}
    committed_validity::Vector{Bool}
    candidate_validity::Vector{Bool}
    staged_difference::Vector{T}
    staged_validity::Vector{Bool}
end

function ChoppedImageDifferenceWorkspace(
    plan::ChoppedImageDifferencePlan{T},
) where {T}
    committed_image = zeros(T, plan.pixel_count)
    committed_validity = fill(false, plan.pixel_count)
    return ChoppedImageDifferenceWorkspace{T}(
        false,
        false,
        false,
        nothing,
        nothing,
        UInt64(0),
        UInt64(0),
        ChoppedImageFringed,
        ChoppedImageFringed,
        committed_image,
        similar(committed_image),
        committed_validity,
        similar(committed_validity),
        similar(committed_image),
        similar(committed_validity),
    )
end

"""
    chopped_image_is_primed(workspace::ChoppedImageDifferenceWorkspace)

Return whether one preceding chopped image is committed.
"""
chopped_image_is_primed(workspace::ChoppedImageDifferenceWorkspace) =
    workspace.committed_primed

"""
    chopped_image_retained_kind(workspace::ChoppedImageDifferenceWorkspace)

Return the committed chopped-image role, or `nothing` before priming.
"""
chopped_image_retained_kind(workspace::ChoppedImageDifferenceWorkspace) =
    workspace.committed_primed ? workspace.committed_kind : nothing

"""
    chopped_image_retained_image(workspace::ChoppedImageDifferenceWorkspace)

Return a copy of the committed preceding image, or `nothing` before priming.
"""
chopped_image_retained_image(workspace::ChoppedImageDifferenceWorkspace) =
    workspace.committed_primed ? copy(workspace.committed_image) : nothing

"""
    chopped_image_retained_validity(workspace::ChoppedImageDifferenceWorkspace)

Return committed normalized validity, or `nothing` before priming.
"""
chopped_image_retained_validity(workspace::ChoppedImageDifferenceWorkspace) =
    workspace.committed_primed ? copy(workspace.committed_validity) : nothing

function commit!(workspace::ChoppedImageDifferenceWorkspace)
    workspace.candidate_ready || return workspace
    workspace.committed_image, workspace.candidate_image =
        workspace.candidate_image, workspace.committed_image
    workspace.committed_validity, workspace.candidate_validity =
        workspace.candidate_validity, workspace.committed_validity
    workspace.committed_kind = workspace.candidate_kind
    workspace.committed_primed = workspace.candidate_primed
    workspace.committed_identity = workspace.candidate_identity
    workspace.committed_duration = workspace.candidate_duration
    workspace.candidate_ready = false
    return workspace
end

function discard!(workspace::ChoppedImageDifferenceWorkspace)
    workspace.candidate_ready = false
    return workspace
end

function reset!(workspace::ChoppedImageDifferenceWorkspace{T}) where {T}
    workspace.committed_primed = false
    workspace.candidate_ready = false
    workspace.candidate_primed = false
    workspace.committed_identity = nothing
    workspace.candidate_identity = nothing
    workspace.committed_duration = UInt64(0)
    workspace.candidate_duration = UInt64(0)
    workspace.committed_kind = ChoppedImageFringed
    workspace.candidate_kind = ChoppedImageFringed
    fill!(workspace.committed_image, zero(T))
    fill!(workspace.candidate_image, zero(T))
    fill!(workspace.committed_validity, false)
    fill!(workspace.candidate_validity, false)
    fill!(workspace.staged_difference, zero(T))
    fill!(workspace.staged_validity, false)
    return workspace
end

"Clear obsolete F/U history while retaining the acquisition admission watermark."
function recover!(workspace::ChoppedImageDifferenceWorkspace)
    identity, duration = workspace.committed_identity, workspace.committed_duration
    reset!(workspace)
    workspace.committed_identity = identity
    workspace.candidate_identity = identity
    workspace.committed_duration = duration
    workspace.candidate_duration = duration
    return workspace
end

"""Expected data rejection from rolling SCC chopped-image differencing."""
@enum ChoppedImageDifferenceFailure::UInt8 begin
    ChoppedImageMetadataMissing = 1
    ChoppedImageTriggerMissing = 2
    ChoppedImageSequenceInvalid = 3
    ChoppedImageExposureMismatch = 4
end

@doc "The image lacks sample metadata or its required acquisition identity."
ChoppedImageMetadataMissing
@doc "The image has a mixed optical role or lacks both optical role and usable fallback trigger."
ChoppedImageTriggerMissing

@doc "The exposure duration differs from the retained acquisition."
ChoppedImageExposureMismatch

@doc "The acquisition sequence repeats or decreases within one source cohort."
ChoppedImageSequenceInvalid

function _chopped_workspace_extent_is_valid(
    workspace::ChoppedImageDifferenceWorkspace,
    plan::ChoppedImageDifferencePlan,
)
    expected = plan.pixel_count
    return length(workspace.committed_image) == expected &&
           length(workspace.candidate_image) == expected &&
           length(workspace.committed_validity) == expected &&
           length(workspace.candidate_validity) == expected &&
           length(workspace.staged_difference) == expected &&
           length(workspace.staged_validity) == expected
end

function _chopped_aliases_workspace(
    values::AbstractArray,
    workspace::ChoppedImageDifferenceWorkspace,
)
    return Base.mightalias(values, workspace.committed_image) ||
           Base.mightalias(values, workspace.candidate_image) ||
           Base.mightalias(values, workspace.committed_validity) ||
           Base.mightalias(values, workspace.candidate_validity) ||
           Base.mightalias(values, workspace.staged_difference) ||
           Base.mightalias(values, workspace.staged_validity)
end

function _check_chopped_image_storage(
    difference::AbstractVector,
    output_validity::AbstractVector,
    pair_status::AbstractVector,
    workspace::ChoppedImageDifferenceWorkspace,
    plan::ChoppedImageDifferencePlan,
    image::AbstractVector,
    input_validity::AbstractVector,
)
    Base.require_one_based_indexing(
        difference,
        output_validity,
        pair_status,
        image,
        input_validity,
    )
    expected = plan.pixel_count
    length(image) == expected ||
        throw(DimensionMismatch("chopped image extent is invalid"))
    length(input_validity) == expected ||
        throw(DimensionMismatch("chopped-image input validity extent is invalid"))
    length(difference) == expected ||
        throw(DimensionMismatch("chopped-image difference extent is invalid"))
    length(output_validity) == expected ||
        throw(DimensionMismatch("chopped-image output validity extent is invalid"))
    length(pair_status) == 1 ||
        throw(DimensionMismatch("chopped-image pair-status extent is invalid"))
    _chopped_workspace_extent_is_valid(workspace, plan) ||
        throw(DimensionMismatch("chopped-image Workspace extent is invalid"))
    (
        Base.mightalias(difference, output_validity) ||
        Base.mightalias(difference, pair_status) ||
        Base.mightalias(output_validity, pair_status)
    ) && throw(ArgumentError("chopped-image outputs must not overlap"))
    (
        _chopped_aliases_workspace(difference, workspace) ||
        _chopped_aliases_workspace(output_validity, workspace) ||
        _chopped_aliases_workspace(pair_status, workspace) ||
        _chopped_aliases_workspace(image, workspace) ||
        _chopped_aliases_workspace(input_validity, workspace)
    ) && throw(
        ArgumentError("chopped-image arrays must not alias Workspace storage"),
    )
    return nothing
end

@inline function _scc_storage_disjoint(a::Vector, b::Vector)
    start_a = UInt(pointer(a))
    start_b = UInt(pointer(b))
    return start_a + UInt(sizeof(a)) <= start_b || start_b + UInt(sizeof(b)) <= start_a
end

@inline function _scc_finite_difference_operand(value::Float32)
    magnitude = reinterpret(UInt32, value) & 0x7fffffff
    # Nonzero magnitudes in [2⁻¹⁰³, 2¹²⁷) are separated by at least the
    # smallest normal Float32. Their differences cannot overflow or be subnormal.
    return iszero(magnitude) | ((magnitude >= 0x0c000000) & (magnitude < 0x7f000000))
end

function _stage_chopped_pair!(workspace, image, input_validity, current_is_fringed, indices)
    @inbounds for index in indices
        current_valid = input_validity[index] && isfinite(image[index])
        valid = current_valid && workspace.committed_validity[index]
        if valid
            fringed = current_is_fringed ? image[index] : workspace.committed_image[index]
            unfringed = current_is_fringed ? workspace.committed_image[index] : image[index]
            value = fringed - unfringed
            valid = isfinite(value)
            workspace.staged_difference[index] = valid ? value : zero(eltype(image))
            workspace.staged_validity[index] = valid
        else
            workspace.staged_difference[index] = zero(eltype(image))
            workspace.staged_validity[index] = false
        end
    end
    return nothing
end

function _stage_chopped_candidate!(workspace, image, input_validity, indices)
    @inbounds for index in indices
        value = image[index]
        workspace.candidate_image[index] = value
        workspace.candidate_validity[index] =
            input_validity[index] && isfinite(value)
    end
    return nothing
end

@inline _chopped_writes_disjoint(::Tuple{}, reads) = true

@inline function _chopped_writes_disjoint(writes::Tuple, reads)
    buffer = first(writes)
    remaining = Base.tail(writes)
    return all(input -> _scc_storage_disjoint(buffer, input), reads) &&
        all(output -> _scc_storage_disjoint(buffer, output), remaining) &&
        _chopped_writes_disjoint(remaining, reads)
end

function _stage_chopped_pair!(
    workspace::ChoppedImageDifferenceWorkspace,
    image::AbstractVector,
    input_validity::AbstractVector{Bool},
    current_is_fringed::Bool,
)
    indices = eachindex(workspace.staged_difference)
    _stage_chopped_pair!(workspace, image, input_validity, current_is_fringed, indices)
    return _stage_chopped_candidate!(workspace, image, input_validity, indices)
end

function _stage_chopped_pair!(
    workspace::ChoppedImageDifferenceWorkspace{Float32},
    image::Vector{Float32},
    input_validity::Vector{Bool},
    current_is_fringed::Bool,
)
    (; committed_image, committed_validity, candidate_image, candidate_validity,
        staged_difference, staged_validity) = workspace
    writes = (candidate_image, candidate_validity, staged_difference, staged_validity)
    reads = (image, input_validity, committed_image, committed_validity)
    # Distinct unsafe_wrap Vectors can overlap even when mightalias is false.
    if !_chopped_writes_disjoint(writes, reads)
        indices = eachindex(image)
        _stage_chopped_pair!(workspace, image, input_validity, current_is_fringed, indices)
        _stage_chopped_candidate!(workspace, image, input_validity, indices)
        return nothing
    end
    fringed, unfringed = current_is_fringed ?
                        (image, committed_image) : (committed_image, image)

    # Check bounded blocks so an invalid pixel keeps its scalar evaluation
    # rules while the surrounding image can use packed subtraction.
    block_size = 256
    normalize_candidate = false
    for first in 1:block_size:length(image)
        indices = first:min(first + block_size - 1, length(image))
        eligible = true
        @inbounds @simd for index in indices
            eligible &= input_validity[index] & committed_validity[index] &
                _scc_finite_difference_operand(image[index]) &
                _scc_finite_difference_operand(committed_image[index])
        end
        if eligible
            @inbounds for index in indices
                staged_difference[index] = fringed[index] - unfringed[index]
                staged_validity[index] = true
                candidate_image[index] = image[index]
                candidate_validity[index] = true
            end
        else
            _stage_chopped_pair!(workspace, image, input_validity, current_is_fringed, indices)
            normalize_candidate = true
        end
    end
    if normalize_candidate
        # Preserve the order of subtraction and potentially exceptional
        # classification. The eligible stores above use finite operands.
        _stage_chopped_candidate!(workspace, image, input_validity, eachindex(image))
    end
    return nothing
end

"""
    process!(difference, output_validity, pair_status, workspace, plan,
             image, input_validity, trigger_high)

Process one acquisition-ordered chopped image. The Plan maps `trigger_high` to
fringed or unfringed. Non-finite or upstream-invalid pixels are published as
zero and invalid without rejecting the pair. `pair_status[1]` is priming `0`
or complete `1`.

The candidate retained image is not committed by this method. The checked
declaration call commits automatically; admitted Graph execution commits when
this Algorithm completes successfully.
"""
function process!(
    difference::AbstractVector{T},
    output_validity::AbstractVector{Bool},
    pair_status::AbstractVector{UInt8},
    workspace::ChoppedImageDifferenceWorkspace{T},
    plan::ChoppedImageDifferencePlan{T},
    image::AbstractVector{T},
    input_validity::AbstractVector{Bool},
    trigger_high::Bool;
    adjacent::Bool=true,
) where {T<:AbstractFloat}
    _check_chopped_image_storage(
        difference,
        output_validity,
        pair_status,
        workspace,
        plan,
        image,
        input_validity,
    )
    workspace.candidate_ready = false
    current_kind = trigger_high ? plan.trigger_high_kind :
                   (
                       plan.trigger_high_kind == ChoppedImageFringed ?
                       ChoppedImageUnfringed : ChoppedImageFringed
                   )

    complete = adjacent && workspace.committed_primed &&
               workspace.committed_kind != current_kind
    current_is_fringed = current_kind == ChoppedImageFringed
    if complete
        _stage_chopped_pair!(workspace, image, input_validity, current_is_fringed)
    else
        fill!(workspace.staged_difference, zero(T))
        fill!(workspace.staged_validity, false)
        _stage_chopped_candidate!(workspace, image, input_validity, eachindex(workspace.staged_difference))
    end
    workspace.candidate_kind = current_kind
    workspace.candidate_primed = true
    workspace.candidate_identity = nothing
    workspace.candidate_duration = UInt64(0)
    copyto!(difference, workspace.staged_difference)
    copyto!(output_validity, workspace.staged_validity)
    pair_status[1] = UInt8(
        complete ? ChoppedImageComplete : ChoppedImagePriming,
    )
    workspace.candidate_ready = true
    return nothing
end

@inline function _chopped_image_admission(plan, workspace, image_sample)
    isnothing(image_sample) && return ChoppedImageMetadataMissing
    kind = sample_chopped_image_kind(image_sample)
    kind == ChoppedImageAbsent && return ChoppedImageTriggerMissing
    adjacent = true
    if plan.require_acquisition_identity
        identity = sample_acquisition_identity(image_sample)
        isnothing(identity) && return ChoppedImageMetadataMissing
        duration = sample_exposure_duration_nanoseconds(image_sample)
        isnothing(duration) && return ChoppedImageMetadataMissing
        (isnothing(kind) || kind == ChoppedImageAbsent) && return ChoppedImageTriggerMissing
        previous = workspace.committed_identity
        same_domain = previous !== nothing && previous.domain == identity.domain
        same_domain && identity.generation < previous.generation &&
            return ChoppedImageSequenceInvalid
        same_cohort = same_domain && previous.generation == identity.generation
        same_cohort && identity.sequence <= previous.sequence &&
            return ChoppedImageSequenceInvalid
        same_cohort && duration != workspace.committed_duration &&
            return ChoppedImageExposureMismatch
        adjacent = same_cohort && identity.sequence - previous.sequence == 1 &&
            !image_sample.discontinuity && !image_sample.corrupted
    end
    trigger_high = if plan.require_acquisition_identity && image_sample.corrupted
        true
    elseif isnothing(kind)
        sample_trigger_high(image_sample)
    elseif kind == ChoppedImageMixed
        plan.require_acquisition_identity ? true : nothing
    else
        kind == plan.trigger_high_kind
    end
    isnothing(trigger_high) && return ChoppedImageTriggerMissing
    return (; adjacent, trigger_high)
end

"""
Float32 rolling SCC chopped-image difference. The Node emits a fixed-sign
`fringed - unfringed` difference for each adjacent opposite-labeled pair and
advances retained history after its own successful transition. An explicit
optical role takes precedence over the configured raw-trigger mapping; a mixed
role returns `ChoppedImageTriggerMissing` in legacy mode. Acquisition-aware mode
consumes mixed identities and clears retained history. No exposure-command
compensation is inferred.
"""
@algorithm ChoppedImageDifferenceF32 begin
    label = "chopped-image-difference-f32"
    failure = ChoppedImageDifferenceFailure

    """Configuration for [`ChoppedImageDifferenceF32`](@ref)."""
    @configuration ChoppedImageDifferenceF32Configuration begin
        pixel_count::Int => graph_rebuild
        trigger_high_is_fringed::Bool => construction
        (input_schema::String = "pipewireao/calibrated-pixels-f32/1") => graph_rebuild
        (require_acquisition_identity::Bool = false) => construction
    end

    ports(configuration) = (
        input(
            :image,
            schema_element_type(CALIBRATED_PIXELS_V1),
            (configuration.pixel_count,),
            configuration.input_schema;
            sample=:image_sample,
        ),
        input(
            :input_validity,
            schema_element_type(CALIBRATED_PIXEL_VALIDITY_V1),
            (configuration.pixel_count,),
            CALIBRATED_PIXEL_VALIDITY_V1;
            name="input-validity",
        ),
        output(
            :difference,
            schema_element_type(SCC_CHOPPED_IMAGE_DIFFERENCE_V1),
            (configuration.pixel_count,),
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1;
            metadata_from=:image,
        ),
        output(
            :output_validity,
            schema_element_type(SCC_CHOPPED_DIFFERENCE_VALIDITY_V1),
            (configuration.pixel_count,),
            SCC_CHOPPED_DIFFERENCE_VALIDITY_V1;
            name="output-validity",
            metadata_from=:image,
        ),
        output(
            :pair_status,
            schema_element_type(SCC_CHOPPED_DIFFERENCE_STATUS_V1),
            (1,),
            SCC_CHOPPED_DIFFERENCE_STATUS_V1;
            name="pair-status",
            metadata_from=:image,
        ),
    )

    prepare(configuration) = begin
        plan = ChoppedImageDifferencePlan(
            Float32,
            configuration.pixel_count,
            configuration.trigger_high_is_fringed;
            require_acquisition_identity=configuration.require_acquisition_identity,
        )
        (plan=plan, workspace=ChoppedImageDifferenceWorkspace(plan))
    end

    process!(plan, workspace) = _process_chopped_image!(
        difference, output_validity, pair_status, workspace, plan, image,
        input_validity, image_sample,
    )
end

required_sample_metadata(prepared::PreparedAlgorithm{ChoppedImageDifferenceF32},input_name::Symbol)=
    input_name===:image && prepared.plan.require_acquisition_identity ?
        (:acquisition_identity,:exposure_duration_nanoseconds,:chopped_image_kind) : ()


function _process_chopped_image!(
    difference, output_validity, pair_status, workspace, plan, image,
    input_validity, image_sample,
)
    admission = _chopped_image_admission(plan, workspace, image_sample)
    admission isa ChoppedImageDifferenceFailure && return admission
    (; adjacent, trigger_high) = admission
    process!(
        difference, output_validity, pair_status, workspace, plan, image,
        input_validity, trigger_high; adjacent,
    )
    if plan.require_acquisition_identity
        workspace.candidate_identity = sample_acquisition_identity(image_sample)
        workspace.candidate_duration = sample_exposure_duration_nanoseconds(image_sample)
        if sample_chopped_image_kind(image_sample) == ChoppedImageMixed || image_sample.corrupted
            workspace.candidate_primed = false
            fill!(difference, 0f0)
            fill!(output_validity, false)
            pair_status[1] = UInt8(ChoppedImagePriming)
        end
    end
    return nothing
end

"""
Full detector ROI for SCC inspection. A complete pair is fringed minus unfringed.
Invalid pixels and priming images contain NaN. A complete result uses the more
recent acquisition as its provenance anchor. The other member has the same
domain and generation and the immediately preceding sequence. Neither member's
optical role, trigger, or exposure clock describes the combined image.
"""
@algorithm ChoppedDetectorDifferenceF32 begin
    label = "chopped-detector-difference-f32"
    failure = ChoppedImageDifferenceFailure

    """Detector shape in [row, column] order and the input's calibrated pixel schema."""
    @configuration ChoppedDetectorDifferenceF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        input_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:image, Float32, configuration.image_size, configuration.input_schema;
            sample=:image_sample),
        output(:difference, Float32, configuration.image_size,
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1),
        output(:pair_status, UInt8, (1,), SCC_CHOPPED_DIFFERENCE_STATUS_V1;
            name="pair-status"),
    )

    prepare(configuration) = begin
        all(>(0), configuration.image_size) ||
            throw(ArgumentError("chopped detector dimensions must be positive"))
        count = Base.checked_mul(configuration.image_size...)
        # Strict admission reads only the explicit optical classification.
        pair = ChoppedImageDifferencePlan(Float32, count, true; require_acquisition_identity=true)
        plan = (; pair, input_validity=trues(count), output_validity=falses(count))
        (; plan, workspace=ChoppedImageDifferenceWorkspace(pair))
    end

    process!(plan, workspace) = begin
        pixels = vec(difference)
        result = _process_chopped_image!(
            pixels, plan.output_validity, pair_status, workspace, plan.pair, vec(image),
            plan.input_validity, image_sample,
        )
        result === nothing || return result
        for index in eachindex(pixels, plan.output_validity)
            plan.output_validity[index] || (pixels[index] = NaN32)
        end
        return AlgorithmProcessComplete(SampleMetadata(
            image_sample;
            trigger_high=nothing,
            chopped_image_kind=nothing,
            exposure_start=nothing,
            reference_timestamp=nothing,
            corrupted=false,
        ))
    end
end

required_sample_metadata(::PreparedAlgorithm{ChoppedDetectorDifferenceF32}, input_name::Symbol) =
    input_name === :image ?
    (:acquisition_identity, :exposure_duration_nanoseconds, :chopped_image_kind) : ()
