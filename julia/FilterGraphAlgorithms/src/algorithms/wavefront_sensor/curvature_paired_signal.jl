"""Expected rejection from a complete paired-branch curvature signal."""
@enum CurvaturePairedSignalFailure::UInt8 begin
    CurvaturePairedSignalCalibrationMismatch = 1
    CurvaturePairedSignalNonFiniteInput = 2
    CurvaturePairedSignalNonFiniteOutput = 3
end

@doc "The acquired frame calibration signature differs from the active curvature calibration."
CurvaturePairedSignalCalibrationMismatch
@doc "The complete acquired curvature input contains a non-finite sample."
CurvaturePairedSignalNonFiniteInput
@doc "A computed curvature signal component is non-finite."
CurvaturePairedSignalNonFiniteOutput

"""
Immutable calibration and geometry for a paired-branch curvature image.

The two branch origins are zero-based `(row, column)` coordinates in positive-
and negative-defocus order. Each rectangular branch is summed into evenly sized
signal-grid tiles. Signal, support, and reference matrices use Julia
column-major FGA `(row, column)` order.
"""
struct CurvaturePairedImagePlan{
    T<:AbstractFloat,
    Support<:AbstractMatrix{Bool},
    Reference<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    signal_size::NTuple{2,Int}
    branch_size::NTuple{2,Int}
    branch_origins::NTuple{2,NTuple{2,Int}}
    support::Support
    reference_signal::Reference
    branch_scales::NTuple{2,T}
    calibration_signature::UInt64

    function CurvaturePairedImagePlan(
        image_size::NTuple{2,Int}, signal_size::NTuple{2,Int}, branch_size::NTuple{2,Int},
        branch_origins::NTuple{2,NTuple{2,Int}}, support::Support,
        reference_signal::Reference, branch_scales::NTuple{2,T}, calibration_signature::UInt64,
        ::Val{:owned},
    ) where {T<:AbstractFloat,Support<:AbstractMatrix{Bool},Reference<:AbstractMatrix{T}}
        return new{T,Support,Reference}(
            image_size, signal_size, branch_size, branch_origins, support, reference_signal,
            branch_scales, calibration_signature,
        )
    end
end

"""
Immutable calibration and geometry for paired curvature branch channels.

`branch_rows` gives the zero-based positive- and negative-defocus rows of the
complete branch-by-channel input. `channel_offsets` maps each full FGA signal
coordinate in Julia column-major order to a zero-based input-channel column.
"""
struct CurvaturePairedChannelPlan{
    T<:AbstractFloat,
    Support<:AbstractMatrix{Bool},
    Reference<:AbstractMatrix{T},
    Offsets<:AbstractVector{Int},
} <: AbstractAlgorithmPlan
    input_size::NTuple{2,Int}
    signal_size::NTuple{2,Int}
    branch_rows::NTuple{2,Int}
    channel_offsets::Offsets
    support::Support
    reference_signal::Reference
    branch_scales::NTuple{2,T}
    calibration_signature::UInt64

    function CurvaturePairedChannelPlan(
        input_size::NTuple{2,Int}, signal_size::NTuple{2,Int}, branch_rows::NTuple{2,Int},
        channel_offsets::Offsets, support::Support, reference_signal::Reference,
        branch_scales::NTuple{2,T}, calibration_signature::UInt64, ::Val{:owned},
    ) where {
        T<:AbstractFloat,Support<:AbstractMatrix{Bool},Reference<:AbstractMatrix{T},
        Offsets<:AbstractVector{Int},
    }
        return new{T,Support,Reference,Offsets}(
            input_size, signal_size, branch_rows, channel_offsets, support, reference_signal,
            branch_scales, calibration_signature,
        )
    end
end

"""Caller-output staging shared by the paired-image and paired-channel estimators."""
mutable struct CurvaturePairedSignalWorkspace{T<:AbstractFloat,Signal<:AbstractVector{T}}
    staged_signal::Signal
end

function _curvature_paired_signal_calibration(
    signal_size::NTuple{2,Int},
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    branch_scales::NTuple{2,T},
) where {T<:AbstractFloat}
    all(>(0), signal_size) || throw(ArgumentError("curvature signal dimensions must be positive"))
    size(support) == signal_size || throw(
        DimensionMismatch("curvature support extent must match signal size"),
    )
    size(reference_signal) == signal_size || throw(
        DimensionMismatch("curvature reference extent must match signal size"),
    )
    all(isfinite, reference_signal) || throw(ArgumentError("curvature reference must be finite"))
    all(scale -> isfinite(scale) && scale > zero(T), branch_scales) || throw(
        ArgumentError("curvature branch scales must be finite and positive"),
    )
    return Matrix{Bool}(support), Matrix{T}(reference_signal)
end

function CurvaturePairedImagePlan(
    image_size::NTuple{2,Int},
    signal_size::NTuple{2,Int},
    branch_size::NTuple{2,Int},
    branch_origins::NTuple{2,NTuple{2,Int}},
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    branch_scales::NTuple{2,T},
    calibration_signature::UInt64,
) where {T<:AbstractFloat}
    image_rows, image_columns = image_size
    branch_rows, branch_columns = branch_size
    image_rows > 0 && image_columns > 0 || throw(
        ArgumentError("curvature image dimensions must be positive"),
    )
    branch_rows > 0 && branch_columns > 0 || throw(
        ArgumentError("curvature branch dimensions must be positive"),
    )
    rem(branch_rows, signal_size[1]) == 0 && rem(branch_columns, signal_size[2]) == 0 ||
        throw(DimensionMismatch("curvature branch dimensions must reduce evenly to signal size"))
    maximum_row, maximum_column = image_rows - branch_rows, image_columns - branch_columns
    for (row, column) in branch_origins
        0 <= row <= maximum_row && 0 <= column <= maximum_column || throw(
            DimensionMismatch("curvature branch region exceeds detector image"),
        )
    end
    owned_support, owned_reference = _curvature_paired_signal_calibration(
        signal_size, support, reference_signal, branch_scales,
    )
    return CurvaturePairedImagePlan(
        image_size, signal_size, branch_size, branch_origins, owned_support, owned_reference,
        branch_scales, calibration_signature, Val(:owned),
    )
end

function CurvaturePairedChannelPlan(
    input_size::NTuple{2,Int},
    signal_size::NTuple{2,Int},
    branch_rows::NTuple{2,Int},
    channel_offsets::AbstractVector{<:Integer},
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    branch_scales::NTuple{2,T},
    calibration_signature::UInt64,
) where {T<:AbstractFloat}
    input_rows, input_channels = input_size
    input_rows > 0 && input_channels > 0 || throw(
        ArgumentError("curvature channel input dimensions must be positive"),
    )
    all(row -> 0 <= row < input_rows, branch_rows) || throw(
        DimensionMismatch("curvature branch row is outside the input"),
    )
    signal_count = Base.checked_mul(signal_size...)
    length(channel_offsets) == signal_count || throw(
        DimensionMismatch("curvature channel offsets must map every signal coordinate"),
    )
    # Normalize the caller's logical axis into owned, one-based execution storage.
    owned_offsets = Vector{Int}(undef, length(channel_offsets))
    for (index, offset) in enumerate(channel_offsets)
        owned_offsets[index] = Int(offset)
    end
    all(offset -> 0 <= offset < input_channels, owned_offsets) || throw(
        DimensionMismatch("curvature channel offset is outside the input"),
    )
    owned_support, owned_reference = _curvature_paired_signal_calibration(
        signal_size, support, reference_signal, branch_scales,
    )
    return CurvaturePairedChannelPlan(
        input_size, signal_size, branch_rows, owned_offsets, owned_support, owned_reference,
        branch_scales, calibration_signature, Val(:owned),
    )
end

function CurvaturePairedSignalWorkspace(plan::Union{
    CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T},
}) where {T<:AbstractFloat}
    return CurvaturePairedSignalWorkspace(zeros(T, Base.checked_mul(plan.signal_size...)))
end

commit!(workspace::CurvaturePairedSignalWorkspace) = workspace
discard!(workspace::CurvaturePairedSignalWorkspace) = workspace
reset!(workspace::CurvaturePairedSignalWorkspace) = workspace

function _replace_curvature_paired_signal_calibration(
    plan::P,
    support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T},
    branch_scales::AbstractVector{T},
    calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat,P<:Union{CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T}}}
    return _rebuild_curvature_paired_signal_calibration(
        plan, support, reference_signal,
        _curvature_branch_scales(branch_scales),
        _curvature_calibration_signature(calibration_signature),
    )
end

function _curvature_branch_scales(branch_scales)
    length(branch_scales) == 2 || throw(DimensionMismatch("curvature branch scales extent is invalid"))
    return (branch_scales[1], branch_scales[2])
end

function _curvature_calibration_signature(calibration_signature)
    length(calibration_signature) == 1 || throw(
        DimensionMismatch("curvature calibration signature extent is invalid"),
    )
    return calibration_signature[1]
end

function _rebuild_curvature_paired_signal_calibration(
    plan::CurvaturePairedImagePlan{T}, support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T}, branch_scales::NTuple{2,T},
    calibration_signature::UInt64,
) where {T<:AbstractFloat}
    return CurvaturePairedImagePlan(
        plan.image_size, plan.signal_size, plan.branch_size, plan.branch_origins, support,
        reference_signal, branch_scales, calibration_signature,
    )
end

function _rebuild_curvature_paired_signal_calibration(
    plan::CurvaturePairedChannelPlan{T}, support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T}, branch_scales::NTuple{2,T},
    calibration_signature::UInt64,
) where {T<:AbstractFloat}
    return CurvaturePairedChannelPlan(
        plan.input_size, plan.signal_size, plan.branch_rows, plan.channel_offsets, support,
        reference_signal, branch_scales, calibration_signature,
    )
end

function _replace_curvature_paired_signal_support(plan, values::AbstractMatrix{Bool})
    return _rebuild_curvature_paired_signal_calibration(
        plan, values, plan.reference_signal, plan.branch_scales, plan.calibration_signature,
    )
end

function _replace_curvature_paired_signal_reference(plan::P, values::AbstractMatrix{T}) where {
    T<:AbstractFloat,P<:Union{CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T}},
}
    return _rebuild_curvature_paired_signal_calibration(
        plan, plan.support, values, plan.branch_scales, plan.calibration_signature,
    )
end

function _replace_curvature_paired_signal_scales(plan::P, values::AbstractVector{T}) where {
    T<:AbstractFloat,P<:Union{CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T}},
}
    return _rebuild_curvature_paired_signal_calibration(
        plan, plan.support, plan.reference_signal, _curvature_branch_scales(values),
        plan.calibration_signature,
    )
end

function _replace_curvature_paired_signal_signature(plan, values::AbstractVector{UInt64})
    return _rebuild_curvature_paired_signal_calibration(
        plan, plan.support, plan.reference_signal, plan.branch_scales,
        _curvature_calibration_signature(values),
    )
end

function _check_curvature_paired_signal_execution(
    signal::AbstractVector{T},
    workspace::CurvaturePairedSignalWorkspace{T},
    plan::Union{CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T}},
    input::AbstractMatrix{<:Real},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(
        signal, workspace.staged_signal, input, frame_calibration_signature,
    )
    size(input) == _curvature_paired_signal_input_size(plan) ||
        throw(DimensionMismatch("curvature input extent is invalid"))
    signal_count = Base.checked_mul(plan.signal_size...)
    length(workspace.staged_signal) == signal_count || throw(
        DimensionMismatch("curvature workspace signal extent is invalid"),
    )
    length(signal) == signal_count || throw(
        DimensionMismatch("curvature signal output extent is invalid"),
    )
    length(frame_calibration_signature) == 1 || throw(
        DimensionMismatch("curvature frame calibration signature extent is invalid"),
    )
    (Base.mightalias(signal, input) || Base.mightalias(signal, frame_calibration_signature)) &&
        throw(ArgumentError("curvature caller arrays must not overlap"))
    (Base.mightalias(signal, plan.support) ||
     Base.mightalias(signal, plan.reference_signal) ||
     Base.mightalias(signal, workspace.staged_signal) ||
     Base.mightalias(input, plan.support) ||
     Base.mightalias(input, plan.reference_signal) ||
     Base.mightalias(input, workspace.staged_signal)) && throw(
        ArgumentError("curvature caller arrays must not alias Plan or Workspace storage"),
    )
    return nothing
end

_curvature_paired_signal_input_size(plan::CurvaturePairedImagePlan) = plan.image_size
_curvature_paired_signal_input_size(plan::CurvaturePairedChannelPlan) = plan.input_size

@inline function _curvature_paired_signal_value(
    plus::T, minus::T, reference::T,
) where {T<:AbstractFloat}
    isfinite(plus) && isfinite(minus) || return T(NaN)
    total = plus + minus
    total <= eps(T) && return zero(T)
    difference = plus - minus
    if isfinite(total) && isfinite(difference)
        return difference / (total + eps(T)) - reference
    end
    # Preserve the low-total policy and epsilon in the original units while
    # avoiding overflow in either sum or difference of finite branch values.
    scale = max(abs(plus), abs(minus))
    scaled_plus, scaled_minus = plus / scale, minus / scale
    return (scaled_plus - scaled_minus) /
           (scaled_plus + scaled_minus + eps(T) / scale) - reference
end

function _stage_curvature_paired_image!(
    workspace::CurvaturePairedSignalWorkspace{T}, plan::CurvaturePairedImagePlan{T},
    image::AbstractMatrix{<:Real}, frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    frame_calibration_signature[1] == plan.calibration_signature ||
        return CurvaturePairedSignalCalibrationMismatch
    all(isfinite, image) || return CurvaturePairedSignalNonFiniteInput
    image_rows_per_signal = div(plan.branch_size[1], plan.signal_size[1])
    image_columns_per_signal = div(plan.branch_size[2], plan.signal_size[2])
    positive_origin, negative_origin = plan.branch_origins
    fill!(workspace.staged_signal, zero(T))
    @inbounds for column in 1:plan.signal_size[2], row in 1:plan.signal_size[1]
        index = row + (column - 1) * plan.signal_size[1]
        plan.support[index] || continue
        positive = zero(T)
        negative = zero(T)
        for local_column in 1:image_columns_per_signal, local_row in 1:image_rows_per_signal
            branch_row = (row - 1) * image_rows_per_signal + local_row
            branch_column = (column - 1) * image_columns_per_signal + local_column
            positive += T(image[positive_origin[1] + branch_row, positive_origin[2] + branch_column])
            negative += T(image[negative_origin[1] + branch_row, negative_origin[2] + branch_column])
        end
        value = _curvature_paired_signal_value(
            plan.branch_scales[1] * positive, plan.branch_scales[2] * negative,
            plan.reference_signal[index],
        )
        isfinite(value) || return CurvaturePairedSignalNonFiniteOutput
        workspace.staged_signal[index] = value
    end
    return nothing
end

function _stage_curvature_paired_channels!(
    workspace::CurvaturePairedSignalWorkspace{T}, plan::CurvaturePairedChannelPlan{T},
    channels::AbstractMatrix{<:Real}, frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    frame_calibration_signature[1] == plan.calibration_signature ||
        return CurvaturePairedSignalCalibrationMismatch
    all(isfinite, channels) || return CurvaturePairedSignalNonFiniteInput
    positive_row, negative_row = plan.branch_rows
    fill!(workspace.staged_signal, zero(T))
    @inbounds for index in eachindex(workspace.staged_signal)
        plan.support[index] || continue
        channel = plan.channel_offsets[index] + 1
        value = _curvature_paired_signal_value(
            plan.branch_scales[1] * T(channels[positive_row + 1, channel]),
            plan.branch_scales[2] * T(channels[negative_row + 1, channel]),
            plan.reference_signal[index],
        )
        isfinite(value) || return CurvaturePairedSignalNonFiniteOutput
        workspace.staged_signal[index] = value
    end
    return nothing
end

function _publish_curvature_paired_signal!(
    signal::AbstractVector{T}, workspace::CurvaturePairedSignalWorkspace{T},
) where {T<:AbstractFloat}
    copyto!(signal, workspace.staged_signal)
    return nothing
end

"""Process one complete paired-branch curvature image with caller-owned output."""
function process!(
    signal::AbstractVector{T}, workspace::CurvaturePairedSignalWorkspace{T},
    plan::CurvaturePairedImagePlan{T}, image::AbstractMatrix{<:Real},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    _check_curvature_paired_signal_execution(
        signal, workspace, plan, image, frame_calibration_signature,
    )
    result = _stage_curvature_paired_image!(workspace, plan, image, frame_calibration_signature)
    result === nothing || return result
    return _publish_curvature_paired_signal!(signal, workspace)
end

"""Process one complete paired-branch curvature channel matrix with caller-owned output."""
function process!(
    signal::AbstractVector{T}, workspace::CurvaturePairedSignalWorkspace{T},
    plan::CurvaturePairedChannelPlan{T}, channels::AbstractMatrix{<:Real},
    frame_calibration_signature::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    _check_curvature_paired_signal_execution(
        signal, workspace, plan, channels, frame_calibration_signature,
    )
    result = _stage_curvature_paired_channels!(
        workspace, plan, channels, frame_calibration_signature,
    )
    result === nothing || return result
    return _publish_curvature_paired_signal!(signal, workspace)
end

function _configured_curvature_support(values::Union{Nothing,Vector{Bool}}, rows::Int, columns::Int)
    count = Base.checked_mul(rows, columns)
    isnothing(values) && return trues(rows, columns)
    length(values) == count || throw(
        DimensionMismatch("curvature support must contain $count column-major values"),
    )
    return reshape(copy(values), rows, columns)
end

function _configured_curvature_reference(
    values::Union{Nothing,Vector{Float32}}, rows::Int, columns::Int,
)
    count = Base.checked_mul(rows, columns)
    isnothing(values) && return zeros(Float32, rows, columns)
    length(values) == count || throw(
        DimensionMismatch("curvature reference must contain $count column-major values"),
    )
    return reshape(copy(values), rows, columns)
end

function _configured_curvature_origin(values::Vector{Int}, name::AbstractString)
    length(values) == 2 || throw(DimensionMismatch("$name must contain row and column"))
    values[1] >= 0 && values[2] >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return (values[1], values[2])
end

function _configured_curvature_scales(values::Union{Nothing,Vector{Float32}})
    isnothing(values) && return (1.0f0, 1.0f0)
    length(values) == 2 || throw(DimensionMismatch("curvature branch scales must contain two values"))
    return (values[1], values[2])
end

"""Float32 paired-branch curvature signal reduced from one complete detector image."""
@algorithm CurvaturePairedImageF32 begin
    label = "curvature-paired-image-f32"
    failure = CurvaturePairedSignalFailure

    """Configuration for CurvaturePairedImageF32."""
    @configuration CurvaturePairedImageF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        signal_size::NTuple{2,Int} => graph_rebuild
        branch_size::NTuple{2,Int} => graph_rebuild
        positive_branch_origin::Vector{Int} => graph_rebuild
        negative_branch_origin::Vector{Int} => graph_rebuild
        curvature_support::Union{Nothing,Vector{Bool}} => construction
        initial_reference_signal::Union{Nothing,Vector{Float32}} => construction
        initial_branch_scales::Union{Nothing,Vector{Float32}} => construction
        initial_calibration_signature::UInt64 => construction
    end

    ports(configuration) = (
        input(:image, schema_element_type(CURVATURE_PAIRED_IMAGE_V1), configuration.image_size, CURVATURE_PAIRED_IMAGE_V1),
        input(:frame_calibration_signature, schema_element_type(CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1), (1,), CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1; name="frame-calibration-signature"),
        output(:signal, schema_element_type(CURVATURE_PAIRED_SIGNAL_V1), (Base.checked_mul(configuration.signal_size...),), CURVATURE_PAIRED_SIGNAL_V1; metadata_from=:image),
        parameter(:curvature_support, schema_element_type(CURVATURE_PAIRED_SUPPORT_V1), configuration.signal_size, CURVATURE_PAIRED_SUPPORT_V1; name="curvature-support", replace=(plan, values) -> _replace_curvature_paired_signal_support(plan, values)),
        parameter(:reference_signal, schema_element_type(CURVATURE_PAIRED_REFERENCE_SIGNAL_V1), configuration.signal_size, CURVATURE_PAIRED_REFERENCE_SIGNAL_V1; name="reference-signal", replace=(plan, values) -> _replace_curvature_paired_signal_reference(plan, values)),
        parameter(:branch_scales, schema_element_type(CURVATURE_PAIRED_BRANCH_SCALES_V1), (2,), CURVATURE_PAIRED_BRANCH_SCALES_V1; name="branch-scales", replace=(plan, values) -> _replace_curvature_paired_signal_scales(plan, values)),
        parameter(:calibration_signature, schema_element_type(CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1), (1,), CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1; name="calibration-signature", replace=(plan, values) -> _replace_curvature_paired_signal_signature(plan, values)),
    )

    prepare(configuration) = begin
        signal_size = configuration.signal_size
        plan = CurvaturePairedImagePlan(
            configuration.image_size, signal_size,
            configuration.branch_size,
            (_configured_curvature_origin(configuration.positive_branch_origin, "positive branch origin"), _configured_curvature_origin(configuration.negative_branch_origin, "negative branch origin")),
            _configured_curvature_support(configuration.curvature_support, signal_size...),
            _configured_curvature_reference(configuration.initial_reference_signal, signal_size...),
            _configured_curvature_scales(configuration.initial_branch_scales),
            configuration.initial_calibration_signature,
        )
        (plan=plan, workspace=CurvaturePairedSignalWorkspace(plan))
    end

    process!(plan, workspace) = process!(signal, workspace, plan, image, frame_calibration_signature)
end

"""Float32 paired-branch curvature signal unpacked from one complete branch-by-channel matrix."""
@algorithm CurvaturePairedChannelF32 begin
    label = "curvature-paired-channel-f32"
    failure = CurvaturePairedSignalFailure

    """Configuration for CurvaturePairedChannelF32."""
    @configuration CurvaturePairedChannelF32Configuration begin
        input_branch_rows::Int => graph_rebuild
        input_channels::Int => graph_rebuild
        signal_size::NTuple{2,Int} => graph_rebuild
        positive_branch_row::Int => graph_rebuild
        negative_branch_row::Int => graph_rebuild
        channel_offsets::Vector{Int} => graph_rebuild
        curvature_support::Union{Nothing,Vector{Bool}} => construction
        initial_reference_signal::Union{Nothing,Vector{Float32}} => construction
        initial_branch_scales::Union{Nothing,Vector{Float32}} => construction
        initial_calibration_signature::UInt64 => construction
    end

    ports(configuration) = (
        input(:channels, schema_element_type(CURVATURE_PAIRED_CHANNELS_V1), (configuration.input_branch_rows, configuration.input_channels), CURVATURE_PAIRED_CHANNELS_V1),
        input(:frame_calibration_signature, schema_element_type(CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1), (1,), CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1; name="frame-calibration-signature"),
        output(:signal, schema_element_type(CURVATURE_PAIRED_SIGNAL_V1), (Base.checked_mul(configuration.signal_size...),), CURVATURE_PAIRED_SIGNAL_V1; metadata_from=:channels),
        parameter(:curvature_support, schema_element_type(CURVATURE_PAIRED_SUPPORT_V1), configuration.signal_size, CURVATURE_PAIRED_SUPPORT_V1; name="curvature-support", replace=(plan, values) -> _replace_curvature_paired_signal_support(plan, values)),
        parameter(:reference_signal, schema_element_type(CURVATURE_PAIRED_REFERENCE_SIGNAL_V1), configuration.signal_size, CURVATURE_PAIRED_REFERENCE_SIGNAL_V1; name="reference-signal", replace=(plan, values) -> _replace_curvature_paired_signal_reference(plan, values)),
        parameter(:branch_scales, schema_element_type(CURVATURE_PAIRED_BRANCH_SCALES_V1), (2,), CURVATURE_PAIRED_BRANCH_SCALES_V1; name="branch-scales", replace=(plan, values) -> _replace_curvature_paired_signal_scales(plan, values)),
        parameter(:calibration_signature, schema_element_type(CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1), (1,), CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1; name="calibration-signature", replace=(plan, values) -> _replace_curvature_paired_signal_signature(plan, values)),
    )

    prepare(configuration) = begin
        signal_size = configuration.signal_size
        plan = CurvaturePairedChannelPlan(
            (configuration.input_branch_rows, configuration.input_channels), signal_size,
            (configuration.positive_branch_row, configuration.negative_branch_row),
            configuration.channel_offsets,
            _configured_curvature_support(configuration.curvature_support, signal_size...),
            _configured_curvature_reference(configuration.initial_reference_signal, signal_size...),
            _configured_curvature_scales(configuration.initial_branch_scales),
            configuration.initial_calibration_signature,
        )
        (plan=plan, workspace=CurvaturePairedSignalWorkspace(plan))
    end

    process!(plan, workspace) = process!(signal, workspace, plan, channels, frame_calibration_signature)
end


"""
    replace_curvature_signal_calibration!(prepared, support, reference, branch_scales, signature)

Validate a complete paired-curvature calibration replacement and publish its Plan once.
"""
function replace_curvature_signal_calibration!(
    prepared::PreparedAlgorithm{D,P}, support::AbstractMatrix{Bool},
    reference_signal::AbstractMatrix{T}, branch_scales::AbstractVector{T},
    calibration_signature::AbstractVector{UInt64},
) where {
    D<:Union{CurvaturePairedImageF32,CurvaturePairedChannelF32},T<:AbstractFloat,
    P<:Union{CurvaturePairedImagePlan{T},CurvaturePairedChannelPlan{T}},
}
    prepared.plan = _replace_curvature_paired_signal_calibration(
        prepared.plan, support, reference_signal, branch_scales, calibration_signature,
    )
    return prepared
end
