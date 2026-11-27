include("pdm_command_policies.jl")

"""
    PdmCommandPlan(lower_limits, upper_limits, slew_limits=nothing;
                   quantization_origin=nothing, quantization_step=nothing,
                   dither_seed=nothing)

Prepare physical-deformable-mirror (PDM) range and optional per-tick slew
conditioning. Slew limits apply independently to each actuator.
Set the legacy quantization keywords to `nothing` or omit them. The legacy
`dither_seed` keyword is accepted for compatibility and has no effect. Configure
DAC quantization and dither in the device driver.

All prepared arrays are copied into dense Julia storage. Limits must be
nonempty, finite, and compatible. Plan arrays are logically read-only. Rebuild
the Plan when configuration changes. Construction selects the slew policy and
the Workspace holds command state.
"""
struct PdmCommandPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    lower_limits::Vector{T}
    upper_limits::Vector{T}
    slew_limits::Union{Nothing,Vector{T}}
    slew::Union{NoSlew,IndependentSlew}
end

function PdmCommandPlan(
    lower_limits::AbstractVector{T},
    upper_limits::AbstractVector{T},
    slew_limits::Union{Nothing,AbstractVector{T}}=nothing;
    quantization_origin=nothing,
    quantization_step=nothing,
    dither_seed::Union{Nothing,Integer}=nothing,
) where {T<:Union{Float32,Float64}}
    if !(isnothing(quantization_origin) && isnothing(quantization_step))
        throw(ArgumentError(
            "Configure DAC quantization and dither in the device driver. " *
            "Omit quantization_origin and quantization_step or set both to null.",
        ))
    end
    if !isnothing(dither_seed) && !(0 <= dither_seed <= typemax(UInt64))
        throw(ArgumentError("PDM dither seed must fit UInt64"))
    end
    extent = length(lower_limits)
    extent > 0 || throw(ArgumentError("PDM actuator extent must be nonempty"))
    length(upper_limits) == extent ||
        throw(DimensionMismatch("PDM upper-limit extent is invalid"))
    isnothing(slew_limits) || length(slew_limits) == extent ||
        throw(DimensionMismatch("PDM slew-limit extent is invalid"))

    lower = Vector{T}(lower_limits)
    upper = Vector{T}(upper_limits)
    slew = isnothing(slew_limits) ? nothing : Vector{T}(slew_limits)
    all(isfinite, lower) && all(isfinite, upper) &&
        (isnothing(slew) || all(isfinite, slew)) ||
        throw(ArgumentError("PDM command configuration must be finite"))
    all(index -> lower[index] <= upper[index], eachindex(lower, upper)) ||
        throw(ArgumentError("PDM lower limits must not exceed upper limits"))
    isnothing(slew) || all(limit -> limit >= zero(T), slew) ||
        throw(ArgumentError("PDM slew limits must be nonnegative"))

    return PdmCommandPlan{T}(lower, upper, slew, _pdm_slew_policy(slew))
end

"""
    PdmCommandWorkspace(plan, initial_current)

Own the initial, committed, and candidate PDM slew state and fixed output
staging for one Algorithm instance.
The initial command is copied and must satisfy the prepared actuator limits.
"""
mutable struct PdmCommandWorkspace{T,V<:AbstractVector{T}}
    initial_current::V
    committed_current::V
    candidate_current::V
    candidate_demanded::V
    candidate_constraint_feedback::V
end

function PdmCommandWorkspace(
    plan::PdmCommandPlan{T},
    initial_current::AbstractVector{T},
) where {T}
    extent = length(plan.lower_limits)
    length(initial_current) == extent ||
        throw(DimensionMismatch("initial PDM command extent is invalid"))
    initial = Vector{T}(initial_current)
    for index in eachindex(initial, plan.lower_limits, plan.upper_limits)
        value = initial[index]
        isfinite(value) ||
            throw(ArgumentError("initial PDM command must be finite"))
        plan.lower_limits[index] <= value <= plan.upper_limits[index] || throw(
            ArgumentError("initial PDM command violates actuator limits"),
        )
    end
    return PdmCommandWorkspace(
        initial,
        copy(initial),
        copy(initial),
        zeros(T, extent),
        zeros(T, extent),
    )
end

"""
    pdm_command_state(workspace::PdmCommandWorkspace)

Return the committed PDM slew state as a read-only borrowed
reference to workspace storage. Commit can swap this buffer, and later
operations can reuse it; this reference is not a retained snapshot. Use `copy`
on the analysis or control path when a snapshot is needed, and serialize access
with workspace use.
"""
pdm_command_state(workspace::PdmCommandWorkspace) = workspace.committed_current

function commit!(workspace::PdmCommandWorkspace)
    workspace.committed_current, workspace.candidate_current =
        workspace.candidate_current, workspace.committed_current
    return workspace
end

function discard!(workspace::PdmCommandWorkspace)
    copyto!(workspace.candidate_current, workspace.committed_current)
    return workspace
end

recover!(workspace::PdmCommandWorkspace{T,Vector{T}}) where {T} = discard!(workspace)

function reset!(workspace::PdmCommandWorkspace)
    copyto!(workspace.committed_current, workspace.initial_current)
    copyto!(workspace.candidate_current, workspace.initial_current)
    fill!(workspace.candidate_demanded, zero(eltype(workspace.initial_current)))
    fill!(workspace.candidate_constraint_feedback, zero(eltype(workspace.initial_current)))
    return workspace
end

"""Expected numerical rejection from PDM command conditioning."""
@enum PdmCommandFailure::UInt8 begin
    PdmCommandNonFiniteInput = 1
    PdmCommandNonFiniteOutput = 2
end

@doc "A requested physical-actuator command contains a non-finite component." PdmCommandNonFiniteInput
@doc "Stroke or slew conditioning produced a non-finite command or feedback." PdmCommandNonFiniteOutput

@inline function _move_toward(current::T, target::T, limit::T) where {T}
    if target > current
        limited = current + limit
        return !isfinite(limited) || limited >= target ? target : limited
    elseif target < current
        limited = current - limit
        return !isfinite(limited) || limited <= target ? target : limited
    end
    return current
end

function process!(
    demanded::AbstractVector{T},
    constraint_feedback::AbstractVector{T},
    workspace::PdmCommandWorkspace{T,Vector{T}},
    plan::PdmCommandPlan{T},
    requested::AbstractVector{T},
) where {T}
    return _process_pdm_command!(demanded, constraint_feedback, workspace, plan, requested,
        Val(true))
end

function _process_pdm_command!(
    demanded::AbstractVector{T},
    constraint_feedback::AbstractVector{T},
    workspace::PdmCommandWorkspace{T,Vector{T}},
    plan::PdmCommandPlan{T},
    requested::AbstractVector{T},
    ::Val{CheckInput},
    check_output::Val=Val(true),
) where {T,CheckInput}
    Base.require_one_based_indexing(requested, demanded, constraint_feedback)
    extent = length(plan.lower_limits)
    length(requested) == extent ||
        throw(DimensionMismatch("requested PDM command extent is invalid"))
    length(demanded) == extent ||
        throw(DimensionMismatch("demanded PDM command extent is invalid"))
    length(constraint_feedback) == extent ||
        throw(DimensionMismatch("PDM constraint-feedback extent is invalid"))
    length(workspace.committed_current) == extent &&
        length(workspace.candidate_current) == extent &&
        length(workspace.candidate_demanded) == extent &&
        length(workspace.candidate_constraint_feedback) == extent || throw(
        DimensionMismatch("PDM command Workspace extent is invalid"),
    )
    Base.mightalias(demanded, constraint_feedback) && throw(
        ArgumentError("demanded command and constraint feedback must not alias"),
    )
    if CheckInput
        for value in requested
            isfinite(value) || return PdmCommandNonFiniteInput
        end
    end

    return _process_pdm_command!(
        demanded, constraint_feedback, workspace, plan, requested,
        plan.slew, Val(CheckInput), check_output,
    )
end

_pdm_slew_limit(::NoSlew, plan, index) = zero(eltype(plan.lower_limits))
_pdm_slew_limit(::IndependentSlew, plan, index) =
    (plan.slew_limits::Vector)[index]

function _process_pdm_command!(
    demanded::Vector{T}, constraint_feedback::Vector{T},
    workspace::PdmCommandWorkspace{T,Vector{T}}, plan::PdmCommandPlan{T},
    requested::Vector{T}, ::NoSlew, ::Val{false}, ::Val{CheckOutput},
) where {T,CheckOutput}
    # The admitted composition owns these buffers. Without slew, the staged
    # current is already the demanded command.
    current = workspace.candidate_current
    feedback = workspace.candidate_constraint_feedback
    lower, upper = plan.lower_limits, plan.upper_limits
    valid = true
    @inbounds @simd for index in eachindex(requested, lower, upper, current, feedback)
        request = requested[index]
        value = clamp(request, lower[index], upper[index])
        difference = request - value
        if CheckOutput
            valid &= isfinite(value) & isfinite(difference)
        end
        current[index] = value
        feedback[index] = difference
    end
    valid || return PdmCommandNonFiniteOutput

    copyto!(demanded, current)
    copyto!(constraint_feedback, feedback)
    return nothing
end

function _process_pdm_command!(
    demanded, constraint_feedback, workspace::PdmCommandWorkspace{T},
    plan::PdmCommandPlan{T}, requested, slew::S, ::Val, ::Val,
) where {T,S}
    for index in eachindex(
        requested,
        plan.lower_limits,
        plan.upper_limits,
        workspace.committed_current,
        workspace.candidate_current,
        workspace.candidate_demanded,
        workspace.candidate_constraint_feedback,
    )
        safe_target = clamp(
            requested[index],
            plan.lower_limits[index],
            plan.upper_limits[index],
        )
        current = _pdm_current(
            slew, workspace.committed_current[index], safe_target,
            _pdm_slew_limit(slew, plan, index),
        )
        isfinite(current) || return PdmCommandNonFiniteOutput
        workspace.candidate_current[index] = current

        value = clamp(current, plan.lower_limits[index], plan.upper_limits[index])
        feedback = requested[index] - value
        isfinite(value) && isfinite(feedback) ||
            return PdmCommandNonFiniteOutput
        workspace.candidate_demanded[index] = value
        workspace.candidate_constraint_feedback[index] = feedback
    end

    copyto!(demanded, workspace.candidate_demanded)
    copyto!(constraint_feedback, workspace.candidate_constraint_feedback)
    return nothing
end

function _expand_command_limit(
    value::T,
    extent::Int,
    name::AbstractString,
) where {T<:AbstractFloat}
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return fill(value, extent)
end

function _expand_command_limit(
    value::AbstractVector{T},
    extent::Int,
    name::AbstractString,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(value)
    length(value) == extent ||
        throw(DimensionMismatch("$name extent must match the command extent"))
    return Vector{T}(value)
end

function _validate_pdm_command_extent(actuator_count::Int)
    actuator_count > 0 || throw(ArgumentError("actuator_count must be positive"))
    UInt64(actuator_count) <= div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32))) ||
        throw(ArgumentError("PDM command byte extent must fit UInt32"))
    return nothing
end

function _prepare_command_limits(count, limits, slew, initial; kwargs...)
    lower = _expand_command_limit(limits[1], count, "lower command limit")
    upper = _expand_command_limit(limits[2], count, "upper command limit")
    slew_limits = isnothing(slew) ? nothing : _expand_command_limit(slew, count, "slew limit")
    plan = PdmCommandPlan(lower, upper, slew_limits; kwargs...)
    current = isnothing(initial) ? zeros(Float32, count) : initial
    return (plan=plan, workspace=PdmCommandWorkspace(plan, current))
end

"""
Turn a requested physical deformable-mirror (PDM) command into the demanded
command after configured stroke bounds and slew limits.
Publishes constraint feedback as `requested - demanded`, in command units.
"""
@algorithm PdmCommandF32 begin
    label = "pdm-command-f32"
    failure = PdmCommandFailure

    """
    Set actuator count and per-actuator or scalar lower/upper stroke limits.
    Optional per-actuator or scalar slew limits and the initial current use the
    same units as the requested command. The demanded schema selects µm wavefront OPD
    (default), metres wavefront OPD, or native actuator command values. The
    limiter uses the supplied values directly for all three ports. Omit the legacy quantization fields or set
    them to `nothing` (`null` in graph configuration). The legacy `dither_seed`
    field is accepted for compatibility and has no effect.
    """
    @configuration PdmCommandF32Configuration begin
        actuator_count::Int => graph_rebuild
        command_limits::NTuple{2,Union{Float32,Vector{Float32}}} => construction
        slew_limit::Union{Nothing,Float32,Vector{Float32}} => construction
        initial_current::Union{Nothing,Vector{Float32}} => construction
        (quantization_origin::Union{Nothing,Float32} = nothing) => construction
        (quantization_step::Union{Nothing,Float32} = nothing) => construction
        (dither_seed::Union{Nothing,UInt64} = nothing) => construction
        (demanded_schema::String = "pipewireao/dm-command-opd-um-f32/1") => graph_rebuild
    end

    ports(c) = (
        input(:requested, Float32, (c.actuator_count,), REQUESTED_PDM_COMMAND_V1),
        output(:demanded, Float32, (c.actuator_count,), c.demanded_schema; metadata_from=:requested),
        output(:constraint_feedback, Float32, (c.actuator_count,), PDM_CONSTRAINT_FEEDBACK_V1;
            name="constraint-feedback", metadata_from=:requested),
    )

    prepare(c) = begin
        _validate_pdm_command_extent(c.actuator_count)
        c.demanded_schema in (DEMANDED_PDM_COMMAND_V1, ACTUATOR_COMMAND_OPD_M_V1, NATIVE_PDM_COMMAND_V1) ||
            throw(ArgumentError("unsupported physical actuator command schema: $(c.demanded_schema)"))
        _prepare_command_limits(c.actuator_count, c.command_limits, c.slew_limit, c.initial_current;
            quantization_origin=c.quantization_origin, quantization_step=c.quantization_step,
            dither_seed=c.dither_seed)
    end

    process!(plan, workspace) = process!(
        demanded,
        constraint_feedback,
        workspace,
        plan,
        requested,
    )
end

# Preserve the positional constructor from before the demanded schema was selectable.
PdmCommandF32Configuration(count, limits, slew, initial, origin, step, seed) =
    PdmCommandF32Configuration(count, limits, slew, initial, origin, step, seed, DEMANDED_PDM_COMMAND_V1)

"""Limit the two normalized-pupil (X, Y) tip/tilt OPD coefficients in metres.
The constraint feedback is requested minus demanded in the same coefficient basis.
"""
@algorithm TipTiltCommandF32 begin
    label = "tip-tilt-command-f32"
    failure = PdmCommandFailure

    """Set lower and upper OPD limits in metres, optional per-sample slew, and initial (X, Y) coefficients."""
    @configuration TipTiltCommandF32Configuration begin
        command_limits::NTuple{2,Union{Float32,Vector{Float32}}} => construction
        slew_limit::Union{Nothing,Float32,Vector{Float32}} => construction
        initial_current::Union{Nothing,Vector{Float32}} => construction
    end

    ports(c) = (
        input(:requested, Float32, (2,), TIP_TILT_OPD_COEFFICIENTS_M_V1),
        output(:demanded, Float32, (2,), TIP_TILT_OPD_COEFFICIENTS_M_V1; metadata_from=:requested),
        output(:constraint_feedback, Float32, (2,), TIP_TILT_OPD_COEFFICIENTS_M_V1;
            name="constraint-feedback", metadata_from=:requested),
    )

    prepare(c) = _prepare_command_limits(2, c.command_limits, c.slew_limit, c.initial_current)

    process!(plan, workspace) = process!(demanded, constraint_feedback, workspace, plan, requested)
end

function _process_finite_input!(demanded, feedback, prepared::PreparedAlgorithm{PdmCommandF32},
        requested, _metadata)
    return _process_pdm_command!(demanded, feedback, prepared.workspace, prepared.plan,
        requested, Val(false))
end

# For finite requests and intervals containing zero, the clamped value lies
# between zero and the request, so both the value and request - value are finite.
_finite_conditioning_result(plan::PdmCommandPlan{Float32}) =
    _finite_conditioning_result(plan, plan.slew)

_finite_conditioning_result(plan, ::IndependentSlew) = false

_finite_conditioning_result(plan, ::NoSlew) =
    all(index -> plan.lower_limits[index] <= 0f0 <= plan.upper_limits[index],
        eachindex(plan.lower_limits, plan.upper_limits))

# Selected at CPU shared-stage preparation for proven finite dense inputs and
# copied, finite limits satisfying _finite_conditioning_result.
function _process_finite_conditioning!(demanded::Vector{Float32}, feedback::Vector{Float32},
        prepared::PreparedAlgorithm{PdmCommandF32}, requested::Vector{Float32}, _metadata)
    return _process_pdm_command!(demanded, feedback, prepared.workspace, prepared.plan,
        requested, Val(false), Val(false))
end
