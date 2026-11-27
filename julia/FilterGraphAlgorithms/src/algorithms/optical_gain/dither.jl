"""Minimum supported sample count in one optical-gain dither period."""
const MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD = 4

"""Timing and commanded amplitude of one modal optical-gain dither."""
struct OpticalGainDitherMode{T<:AbstractFloat}
    amplitude::T
    frames_per_period::Int
    periods_per_estimate::Int

    function OpticalGainDitherMode(
        amplitude::T,
        frames_per_period::Integer,
        periods_per_estimate::Integer,
    ) where {T<:AbstractFloat}
        return new{T}(
            amplitude,
            Int(frames_per_period),
            Int(periods_per_estimate),
        )
    end
end

function _dither_mode_value(values::AbstractDict, name::String)
    haskey(values, name) && return values[name]
    symbol = Symbol(name)
    haskey(values, symbol) && return values[symbol]
    throw(ArgumentError("optical-gain dither mode requires '$name'"))
end

function Base.convert(
    ::Type{OpticalGainDitherMode{T}},
    values::AbstractDict,
) where {T<:AbstractFloat}
    allowed = Set(("amplitude", "frames_per_period", "periods_per_estimate"))
    all(key -> String(key) in allowed, keys(values)) ||
        throw(ArgumentError("optical-gain dither mode has an unknown field"))
    return OpticalGainDitherMode(
        convert(T, _dither_mode_value(values, "amplitude")),
        convert(Int, _dither_mode_value(values, "frames_per_period")),
        convert(Int, _dither_mode_value(values, "periods_per_estimate")),
    )
end

"""
    OpticalGainDitherSchedule(modes)

Validate a synchronized modal dither schedule and prepare every sine and
cosine reference outside repeated execution. Every mode must span the same
estimate-window length.
"""
struct OpticalGainDitherSchedule{T<:AbstractFloat}
    modes::Vector{OpticalGainDitherMode{T}}
    offsets::Vector{Int}
    cosines::Vector{T}
    sines::Vector{T}
    frames_per_estimate::Int
end

function OpticalGainDitherSchedule(
    modes::AbstractVector{OpticalGainDitherMode{T}},
) where {T<:AbstractFloat}
    isempty(modes) &&
        throw(ArgumentError("optical-gain dither schedule must be nonempty"))
    prepared_modes = Vector{OpticalGainDitherMode{T}}(modes)
    frames_per_estimate = 0
    table_length = 0
    for (mode_index, mode) in pairs(prepared_modes)
        isfinite(mode.amplitude) && mode.amplitude > zero(T) || throw(
            ArgumentError(
                "optical-gain dither mode $(mode_index - 1) amplitude must be finite and positive",
            ),
        )
        mode.frames_per_period >= MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD || throw(
            ArgumentError(
                "optical-gain dither mode $(mode_index - 1) period is too short",
            ),
        )
        mode.periods_per_estimate > 0 || throw(
            ArgumentError(
                "optical-gain dither mode $(mode_index - 1) must complete at least one period",
            ),
        )
        window = Base.Checked.checked_mul(
            mode.frames_per_period,
            mode.periods_per_estimate,
        )
        if isone(mode_index)
            frames_per_estimate = window
        elseif window != frames_per_estimate
            throw(
                ArgumentError(
                    "optical-gain dither modes must share one estimate window",
                ),
            )
        end
        table_length = Base.Checked.checked_add(
            table_length,
            mode.frames_per_period,
        )
    end

    offsets = Vector{Int}(undef, length(prepared_modes))
    cosines = Vector{T}(undef, table_length)
    sines = similar(cosines)
    destination = 1
    two_pi = T(2) * T(pi)
    for (mode_index, mode) in pairs(prepared_modes)
        offsets[mode_index] = destination
        period = T(mode.frames_per_period)
        isfinite(period) || throw(
            ArgumentError(
                "optical-gain dither mode $(mode_index - 1) timing is not representable",
            ),
        )
        @inbounds for phase_index in 0:(mode.frames_per_period - 1)
            phase = T(phase_index)
            angle = two_pi * phase / period
            cosine, sine = cos(angle), sin(angle)
            isfinite(cosine) && isfinite(sine) || throw(
                ArgumentError(
                    "optical-gain dither mode $(mode_index - 1) has a non-finite quadrature",
                ),
            )
            cosines[destination] = cosine
            sines[destination] = sine
            destination += 1
        end
    end
    return OpticalGainDitherSchedule{T}(
        prepared_modes,
        offsets,
        cosines,
        sines,
        frames_per_estimate,
    )
end

"""
    optical_gain_dither_modes(schedule::OpticalGainDitherSchedule)

Return a copy of the prepared modal dither descriptors.
"""
optical_gain_dither_modes(schedule::OpticalGainDitherSchedule) =
    copy(schedule.modes)

"""
    optical_gain_dither_cosines(schedule::OpticalGainDitherSchedule, mode::Integer)

Return a copy of one mode's prepared cosine quadrature.
"""
function optical_gain_dither_cosines(
    schedule::OpticalGainDitherSchedule,
    mode::Integer,
)
    1 <= mode <= length(schedule.modes) || throw(BoundsError(schedule, mode))
    descriptor = schedule.modes[mode]
    first_index = schedule.offsets[mode]
    return copy(
        @view(
            schedule.cosines[
                first_index:(first_index + descriptor.frames_per_period - 1)
            ]
        ),
    )
end

"""
    optical_gain_dither_sines(schedule::OpticalGainDitherSchedule, mode::Integer)

Return a copy of one mode's prepared sine quadrature.
"""
function optical_gain_dither_sines(
    schedule::OpticalGainDitherSchedule,
    mode::Integer,
)
    1 <= mode <= length(schedule.modes) || throw(BoundsError(schedule, mode))
    descriptor = schedule.modes[mode]
    first_index = schedule.offsets[mode]
    return copy(
        @view(
            schedule.sines[
                first_index:(first_index + descriptor.frames_per_period - 1)
            ]
        ),
    )
end

@inline function _optical_gain_dither_cosine(
    schedule::OpticalGainDitherSchedule,
    mode::Int,
    phase::Int,
)
    return @inbounds schedule.cosines[schedule.offsets[mode] + phase]
end

"""Prepared source of synchronized commanded modal optical-gain sinusoids."""
struct OpticalGainDitherPlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    schedule::OpticalGainDitherSchedule{T}
end

"""
    OpticalGainDitherWorkspace(plan)

Own committed and candidate modal phase plus estimate-window state. A Graph
advances the source only when the complete sample is committed.
"""
mutable struct OpticalGainDitherWorkspace
    committed_phases::Vector{Int}
    candidate_phases::Vector{Int}
    committed_frame::Int
    candidate_frame::Int
    candidate_ready::Bool
end

function OpticalGainDitherWorkspace(plan::OpticalGainDitherPlan)
    phases = zeros(Int, length(plan.schedule.modes))
    return OpticalGainDitherWorkspace(
        phases,
        similar(phases),
        0,
        0,
        false,
    )
end

"""
    optical_gain_dither_phases(workspace::OpticalGainDitherWorkspace)

Return a copy of the committed phase used by the next dither sample.
"""
optical_gain_dither_phases(workspace::OpticalGainDitherWorkspace) =
    copy(workspace.committed_phases)

"""
    optical_gain_dither_frame(workspace::OpticalGainDitherWorkspace)

Return the committed zero-based frame used by the next estimate sample.
"""
optical_gain_dither_frame(workspace::OpticalGainDitherWorkspace) =
    workspace.committed_frame

function commit!(workspace::OpticalGainDitherWorkspace)
    workspace.candidate_ready || return workspace
    workspace.committed_phases, workspace.candidate_phases =
        workspace.candidate_phases, workspace.committed_phases
    workspace.committed_frame = workspace.candidate_frame
    workspace.candidate_ready = false
    return workspace
end

function discard!(workspace::OpticalGainDitherWorkspace)
    workspace.candidate_ready = false
    return workspace
end

function reset!(workspace::OpticalGainDitherWorkspace)
    fill!(workspace.committed_phases, 0)
    fill!(workspace.candidate_phases, 0)
    workspace.committed_frame = 0
    workspace.candidate_frame = 0
    workspace.candidate_ready = false
    return workspace
end

@inline function _optical_gain_dither_aliases_owned_storage(
    output::AbstractVector,
    workspace::OpticalGainDitherWorkspace,
    plan::OpticalGainDitherPlan,
)
    schedule = plan.schedule
    return Base.mightalias(output, workspace.committed_phases) ||
           Base.mightalias(output, workspace.candidate_phases) ||
           Base.mightalias(output, schedule.modes) ||
           Base.mightalias(output, schedule.offsets) ||
           Base.mightalias(output, schedule.cosines) ||
           Base.mightalias(output, schedule.sines)
end

function _check_optical_gain_dither_storage(
    commanded_modes::AbstractVector,
    window_frame::AbstractVector,
    phase_indices::AbstractVector,
    completes_window::AbstractVector,
    workspace::OpticalGainDitherWorkspace,
    plan::OpticalGainDitherPlan,
)
    Base.require_one_based_indexing(
        commanded_modes,
        window_frame,
        phase_indices,
        completes_window,
    )
    mode_count = length(plan.schedule.modes)
    length(commanded_modes) == mode_count ||
        throw(DimensionMismatch("optical-gain dither command extent is invalid"))
    length(window_frame) == 1 ||
        throw(DimensionMismatch("optical-gain dither window-frame extent is invalid"))
    length(phase_indices) == mode_count ||
        throw(DimensionMismatch("optical-gain dither phase extent is invalid"))
    length(completes_window) == 1 || throw(
        DimensionMismatch("optical-gain dither completion extent is invalid"),
    )
    length(workspace.committed_phases) == mode_count &&
        length(workspace.candidate_phases) == mode_count || throw(
        DimensionMismatch("optical-gain dither Workspace extent is invalid"),
    )
    0 <= workspace.committed_frame < plan.schedule.frames_per_estimate || throw(
        ArgumentError("optical-gain dither Workspace frame is invalid"),
    )
    @inbounds for mode in 1:mode_count
        0 <= workspace.committed_phases[mode] <
             plan.schedule.modes[mode].frames_per_period || throw(
            ArgumentError("optical-gain dither Workspace phase is invalid"),
        )
    end
    (Base.mightalias(commanded_modes, window_frame) ||
     Base.mightalias(commanded_modes, phase_indices) ||
     Base.mightalias(commanded_modes, completes_window) ||
     Base.mightalias(window_frame, phase_indices) ||
     Base.mightalias(window_frame, completes_window) ||
     Base.mightalias(phase_indices, completes_window)) && throw(
        ArgumentError("optical-gain dither outputs must not overlap"),
    )
    (_optical_gain_dither_aliases_owned_storage(commanded_modes, workspace, plan) ||
     _optical_gain_dither_aliases_owned_storage(window_frame, workspace, plan) ||
     _optical_gain_dither_aliases_owned_storage(phase_indices, workspace, plan) ||
     _optical_gain_dither_aliases_owned_storage(completes_window, workspace, plan)) &&
        throw(
            ArgumentError(
                "optical-gain dither outputs must not overlap prepared or Workspace storage",
            ),
        )
    return nothing
end

function process!(
    commanded_modes::AbstractVector{T},
    window_frame::AbstractVector{UInt64},
    phase_indices::AbstractVector{UInt64},
    completes_window::AbstractVector{Bool},
    workspace::OpticalGainDitherWorkspace,
    plan::OpticalGainDitherPlan{T},
) where {T<:AbstractFloat}
    _check_optical_gain_dither_storage(
        commanded_modes,
        window_frame,
        phase_indices,
        completes_window,
        workspace,
        plan,
    )
    workspace.candidate_ready = false
    schedule = plan.schedule
    mode_count = length(schedule.modes)
    @inbounds for mode in 1:mode_count
        phase = workspace.committed_phases[mode]
        descriptor = schedule.modes[mode]
        commanded_modes[mode] = descriptor.amplitude *
                                _optical_gain_dither_cosine(
            schedule,
            mode,
            phase,
        )
        phase_indices[mode] = UInt64(phase)
        next_phase = phase + 1
        workspace.candidate_phases[mode] =
            next_phase == descriptor.frames_per_period ? 0 : next_phase
    end
    frame = workspace.committed_frame
    window_frame[1] = UInt64(frame)
    completes_window[1] = frame + 1 == schedule.frames_per_estimate
    next_frame = frame + 1
    workspace.candidate_frame =
        next_frame == schedule.frames_per_estimate ? 0 : next_frame
    workspace.candidate_ready = true
    return nothing
end

"""
Float32 source of synchronized modal optical-gain dithers. Phase and window
state advance after this Algorithm succeeds.
"""
@algorithm OpticalGainDitherF32 begin
    label = "optical-gain-dither-f32"

    """Configuration for [`OpticalGainDitherF32`](@ref)."""
    @configuration OpticalGainDitherF32Configuration begin
        modes::Vector{OpticalGainDitherMode{Float32}} => graph_rebuild
    end

    ports(configuration) = (
        output(
            :commanded_modes,
            schema_element_type(OPTICAL_GAIN_DITHER_COMMAND_V1),
            (length(configuration.modes),),
            OPTICAL_GAIN_DITHER_COMMAND_V1;
            name="commanded-modes",
        ),
        output(
            :window_frame,
            schema_element_type(OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1),
            (1,),
            OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1;
            name="window-frame",
        ),
        output(
            :phase_indices,
            schema_element_type(OPTICAL_GAIN_DITHER_PHASE_INDICES_V1),
            (length(configuration.modes),),
            OPTICAL_GAIN_DITHER_PHASE_INDICES_V1;
            name="phase-indices",
        ),
        output(
            :completes_window,
            schema_element_type(OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1),
            (1,),
            OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1;
            name="completes-window",
        ),
    )

    prepare(configuration) = begin
        schedule = OpticalGainDitherSchedule(configuration.modes)
        plan = OpticalGainDitherPlan(schedule)
        (plan=plan, workspace=OpticalGainDitherWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(
            commanded_modes,
            window_frame,
            phase_indices,
            completes_window,
            workspace,
            plan,
        )
    end
end
