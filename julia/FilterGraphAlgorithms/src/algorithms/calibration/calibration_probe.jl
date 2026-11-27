"""Global sign of one structured calibration command."""
@enum CalibrationProbePolarity::UInt8 begin
    CalibrationProbePositive = 0
    CalibrationProbeNegative = 1
end

@doc "Positive command in a structured calibration push/pull pair." CalibrationProbePositive
@doc "Exact negative of the positive command in a structured calibration pair." CalibrationProbeNegative

const _AOCProbeBases = AdaptiveOpticsCalibration.ProbeBases

function _complete_probe_commands(method)
    calibration_plan = AdaptiveOpticsCalibration.prepare(method, nothing)
    product = AdaptiveOpticsCalibration.allocate_result(calibration_plan)
    workspace = AdaptiveOpticsCalibration.allocate_workspace(calibration_plan)
    AdaptiveOpticsCalibration.process!(product, workspace, calibration_plan, nothing)
    return permutedims(_AOCProbeBases.probe_commands(product))
end

"""Zero-based pattern coordinate and polarity published with one probe command."""
struct CalibrationProbeStep
    pattern_index::UInt64
    polarity::CalibrationProbePolarity
    completes_cycle::Bool
end

"""
    ModalPushPullPlan(mode_count, corrector_coordinate_count, positive_commands)
    ModalPushPullPlan(positive_commands)

Prepare an arbitrary modal positive/negative calibration sequence.

The vector form accepts the frozen FGN configuration representation: modal
rows concatenated in row-major order. The matrix form accepts ordinary Julia
logical indexing with shape `(mode_count, corrector_coordinate_count)`. Every
row is the exact positive corrector command, including probe amplitude. Each
processed sample emits a row and then its exact negative before advancing to
the next mode. Preparation rejects empty dimensions, non-finite coefficients,
and all-zero rows.
"""
struct ModalPushPullPlan{
    T<:Union{Float32,Float64},
    M<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    commands::M

    function ModalPushPullPlan(
        commands::M,
        ::Val{:prepared},
    ) where {T<:Union{Float32,Float64},M<:AbstractMatrix{T}}
        Base.require_one_based_indexing(commands)
        return new{T,M}(commands)
    end
end

function ModalPushPullPlan(
    positive_commands::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(positive_commands)
    commands = _complete_probe_commands(
        _AOCProbeBases.ModalPushPull(positive_commands),
    )
    return ModalPushPullPlan(commands, Val(:prepared))
end

function ModalPushPullPlan(
    mode_count::Integer,
    corrector_coordinate_count::Integer,
    positive_commands::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    modes = Int(mode_count)
    correctors = Int(corrector_coordinate_count)
    modes > 0 && correctors > 0 ||
        throw(ArgumentError("modal push/pull dimensions must be positive"))
    coefficient_count =
        _structured_checked_product("modal positive-command", modes, correctors)
    _structured_checked_product("modal push/pull cycle", 2, modes)
    length(positive_commands) == coefficient_count || throw(
        DimensionMismatch(
            "modal positive-command extent $(length(positive_commands)) " *
            "does not match $coefficient_count",
        ),
    )
    Base.require_one_based_indexing(positive_commands)
    commands = Matrix{T}(undef, modes, correctors)
    for mode in 1:modes
        row_start = (mode - 1) * correctors
        for corrector_coordinate in 1:correctors
            commands[mode, corrector_coordinate] =
                positive_commands[row_start + corrector_coordinate]
        end
    end
    return ModalPushPullPlan(commands)
end

"""
    modal_positive_commands(plan::ModalPushPullPlan{T}) where {T}

Return a copy of the prepared modal positive-command matrix.
"""
function modal_positive_commands(plan::ModalPushPullPlan{T}) where {T}
    commands = Matrix{T}(
        undef,
        calibration_probe_pattern_count(plan),
        calibration_probe_corrector_coordinate_count(plan),
    )
    for mode in axes(commands, 1)
        for corrector in axes(commands, 2)
            commands[mode, corrector] = plan.commands[corrector, 2 * mode - 1]
        end
    end
    return commands
end

"""
    calibration_probe_pattern_count(plan::ModalPushPullPlan)

Return the number of modal patterns in one prepared Plan.
"""
calibration_probe_pattern_count(plan::ModalPushPullPlan) =
    size(plan.commands, 2) ÷ 2

"""
    calibration_probe_corrector_coordinate_count(plan::ModalPushPullPlan)

Return the corrector-coordinate extent of one prepared probe Plan.
"""
calibration_probe_corrector_coordinate_count(plan::ModalPushPullPlan) =
    size(plan.commands, 1)

"""
    calibration_probe_cycle_length(plan::ModalPushPullPlan)

Return the number of commands in one complete positive/negative cycle.
"""
calibration_probe_cycle_length(plan::ModalPushPullPlan) =
    2 * calibration_probe_pattern_count(plan)

"""
    ZonalPushPullPlan(amplitudes)

Prepare exact positive/negative single-coordinate probe commands. Every
amplitude must be finite and positive.
"""
struct ZonalPushPullPlan{
    T<:Union{Float32,Float64},
    M<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    commands::M

    function ZonalPushPullPlan(
        commands::M,
        ::Val{:prepared},
    ) where {T<:Union{Float32,Float64},M<:AbstractMatrix{T}}
        Base.require_one_based_indexing(commands)
        return new{T,M}(commands)
    end
end

function ZonalPushPullPlan(
    amplitudes::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(amplitudes)
    commands = _complete_probe_commands(
        _AOCProbeBases.ZonalPushPull(amplitudes),
    )
    return ZonalPushPullPlan(commands, Val(:prepared))
end

"""
    HadamardPushPullPlan(amplitudes)

Prepare a balanced Sylvester-Hadamard positive/negative probe sequence. The
smallest order greater than the corrector count is used so the all-ones DC
column can be excluded; corrector coordinate `i` uses Hadamard column `i`.
Every amplitude must be finite and positive.
"""
struct HadamardPushPullPlan{
    T<:Union{Float32,Float64},
    M<:AbstractMatrix{T},
} <: AbstractAlgorithmPlan
    commands::M
    order::Int

    function HadamardPushPullPlan(
        commands::M,
        order::Int,
        ::Val{:prepared},
    ) where {T<:Union{Float32,Float64},M<:AbstractMatrix{T}}
        Base.require_one_based_indexing(commands)
        return new{T,M}(commands, order)
    end
end

function HadamardPushPullPlan(
    amplitudes::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(amplitudes)
    commands = _complete_probe_commands(
        _AOCProbeBases.HadamardPushPull(amplitudes),
    )
    return HadamardPushPullPlan(
        commands,
        size(commands, 2) ÷ 2,
        Val(:prepared),
    )
end

"""
    calibration_probe_amplitudes(plan::ZonalPushPullPlan{T}) where {T}

Return a copy of the prepared per-coordinate probe amplitudes.
"""
calibration_probe_amplitudes(
    plan::ZonalPushPullPlan{T},
) where {T} = T[
    plan.commands[coordinate, 2 * coordinate - 1] for
    coordinate in 1:calibration_probe_corrector_coordinate_count(plan)
]

calibration_probe_amplitudes(
    plan::HadamardPushPullPlan{T},
) where {T} = copy(@view plan.commands[:, 1])

calibration_probe_pattern_count(plan::ZonalPushPullPlan) =
    size(plan.commands, 2) ÷ 2
calibration_probe_pattern_count(plan::HadamardPushPullPlan) = plan.order
calibration_probe_corrector_coordinate_count(
    plan::Union{ZonalPushPullPlan,HadamardPushPullPlan},
) = size(plan.commands, 1)
calibration_probe_cycle_length(
    plan::Union{ZonalPushPullPlan,HadamardPushPullPlan},
) = 2 * calibration_probe_pattern_count(plan)

"""
    CalibrationProbeWorkspace(plan)

Own committed and candidate sequence position plus fixed output staging for one
structured calibration source. Processing reads only the committed position.
[`commit!`](@ref) advances it after this Algorithm succeeds; [`discard!`](@ref)
abandons its candidate; [`reset!`](@ref) returns to positive pattern zero.
"""
mutable struct CalibrationProbeWorkspace{T<:Union{Float32,Float64}}
    pattern_count::Int
    corrector_coordinate_count::Int
    committed_pattern_index::Int
    candidate_pattern_index::Int
    committed_polarity::CalibrationProbePolarity
    candidate_polarity::CalibrationProbePolarity
    staged_excitation::Vector{T}
end

function CalibrationProbeWorkspace(
    plan::P,
) where {
    T,
    P<:Union{
        ModalPushPullPlan{T},
        ZonalPushPullPlan{T},
        HadamardPushPullPlan{T},
    },
}
    return CalibrationProbeWorkspace{T}(
        calibration_probe_pattern_count(plan),
        calibration_probe_corrector_coordinate_count(plan),
        0,
        0,
        CalibrationProbePositive,
        CalibrationProbePositive,
        zeros(T, calibration_probe_corrector_coordinate_count(plan)),
    )
end

"""
    calibration_probe_next_step(workspace::CalibrationProbeWorkspace)

Return the next committed probe step without advancing it.
"""
function calibration_probe_next_step(workspace::CalibrationProbeWorkspace)
    return CalibrationProbeStep(
        UInt64(workspace.committed_pattern_index),
        workspace.committed_polarity,
        workspace.committed_polarity === CalibrationProbeNegative &&
        workspace.committed_pattern_index + 1 == workspace.pattern_count,
    )
end

function commit!(workspace::CalibrationProbeWorkspace)
    workspace.committed_pattern_index = workspace.candidate_pattern_index
    workspace.committed_polarity = workspace.candidate_polarity
    return workspace
end

function discard!(workspace::CalibrationProbeWorkspace)
    workspace.candidate_pattern_index = workspace.committed_pattern_index
    workspace.candidate_polarity = workspace.committed_polarity
    return workspace
end

function reset!(workspace::CalibrationProbeWorkspace)
    workspace.committed_pattern_index = 0
    workspace.candidate_pattern_index = 0
    workspace.committed_polarity = CalibrationProbePositive
    workspace.candidate_polarity = CalibrationProbePositive
    fill!(workspace.staged_excitation, zero(eltype(workspace.staged_excitation)))
    return workspace
end

_plan_adopted!(workspace::CalibrationProbeWorkspace) = discard!(workspace)

function _calibration_probe_outputs_overlap(
    excitation::AbstractVector,
    pattern_index::AbstractVector,
    polarity::AbstractVector,
    completes_cycle::AbstractVector,
)
    return Base.mightalias(excitation, pattern_index) ||
           Base.mightalias(excitation, polarity) ||
           Base.mightalias(excitation, completes_cycle) ||
           Base.mightalias(pattern_index, polarity) ||
           Base.mightalias(pattern_index, completes_cycle) ||
           Base.mightalias(polarity, completes_cycle)
end

function _validate_calibration_probe_storage(
    excitation::AbstractVector,
    pattern_index::AbstractVector,
    polarity::AbstractVector,
    completes_cycle::AbstractVector,
    workspace::CalibrationProbeWorkspace,
    plan_storage::AbstractArray,
    pattern_count::Int,
    corrector_coordinate_count::Int,
)
    Base.require_one_based_indexing(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
    )
    length(excitation) == corrector_coordinate_count || throw(
        DimensionMismatch("calibration probe excitation extent is invalid"),
    )
    length(pattern_index) == 1 ||
        throw(DimensionMismatch("calibration probe pattern-index extent is invalid"))
    length(polarity) == 1 ||
        throw(DimensionMismatch("calibration probe polarity extent is invalid"))
    length(completes_cycle) == 1 || throw(
        DimensionMismatch("calibration probe cycle-completion extent is invalid"),
    )
    workspace.pattern_count == pattern_count &&
    workspace.corrector_coordinate_count == corrector_coordinate_count &&
    length(workspace.staged_excitation) == corrector_coordinate_count &&
    0 <= workspace.committed_pattern_index < pattern_count || throw(
        DimensionMismatch("calibration probe Workspace is incompatible with the Plan"),
    )
    _calibration_probe_outputs_overlap(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
    ) && throw(ArgumentError("calibration probe outputs must not overlap"))
    (
        Base.mightalias(excitation, workspace.staged_excitation) ||
        Base.mightalias(pattern_index, workspace.staged_excitation) ||
        Base.mightalias(polarity, workspace.staged_excitation) ||
        Base.mightalias(completes_cycle, workspace.staged_excitation)
    ) && throw(
        ArgumentError("calibration probe outputs must not alias Workspace staging"),
    )
    (
        Base.mightalias(excitation, plan_storage) ||
        Base.mightalias(pattern_index, plan_storage) ||
        Base.mightalias(polarity, plan_storage) ||
        Base.mightalias(completes_cycle, plan_storage)
    ) && throw(ArgumentError("calibration probe outputs must not alias Plan storage"))
    return nothing
end

function _stage_probe_step!(workspace::CalibrationProbeWorkspace)
    pattern_index = workspace.committed_pattern_index
    polarity = workspace.committed_polarity
    completes_cycle = polarity === CalibrationProbeNegative &&
                      pattern_index + 1 == workspace.pattern_count
    if polarity === CalibrationProbePositive
        workspace.candidate_pattern_index = pattern_index
        workspace.candidate_polarity = CalibrationProbeNegative
    else
        workspace.candidate_pattern_index =
            completes_cycle ? 0 : pattern_index + 1
        workspace.candidate_polarity = CalibrationProbePositive
    end
    return CalibrationProbeStep(UInt64(pattern_index), polarity, completes_cycle)
end

function _publish_calibration_probe!(
    excitation::AbstractVector,
    pattern_index::AbstractVector{UInt64},
    polarity::AbstractVector{UInt8},
    completes_cycle::AbstractVector{Bool},
    workspace::CalibrationProbeWorkspace,
)
    step = _stage_probe_step!(workspace)
    copyto!(excitation, workspace.staged_excitation)
    pattern_index[1] = step.pattern_index
    polarity[1] = UInt8(step.polarity)
    completes_cycle[1] = step.completes_cycle
    return nothing
end

"""
    process!(excitation, pattern_index, polarity, completes_cycle,
             workspace, plan)

Produce one structured calibration command from the Plan's committed sequence
position. Outputs use zero-based pattern indices and polarity `0` for positive
or `1` for negative. The sequence advances only if the owner later calls
[`commit!`](@ref). Checked shape, axis, alias, or Workspace errors throw before
any caller-owned output changes. Strided one-based output vectors are
supported.
"""
function process!(
    excitation::AbstractVector{T},
    pattern_index::AbstractVector{UInt64},
    polarity::AbstractVector{UInt8},
    completes_cycle::AbstractVector{Bool},
    workspace::CalibrationProbeWorkspace{T},
    plan::Union{
        ModalPushPullPlan{T},
        ZonalPushPullPlan{T},
        HadamardPushPullPlan{T},
    },
) where {T}
    pattern_count = calibration_probe_pattern_count(plan)
    corrector_count = calibration_probe_corrector_coordinate_count(plan)
    _validate_calibration_probe_storage(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
        workspace,
        plan.commands,
        pattern_count,
        corrector_count,
    )
    row = 2 * workspace.committed_pattern_index +
          (workspace.committed_polarity === CalibrationProbePositive ? 1 : 2)
    for corrector_coordinate in 1:corrector_count
        workspace.staged_excitation[corrector_coordinate] =
            plan.commands[corrector_coordinate, row]
    end
    return _publish_calibration_probe!(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
        workspace,
    )
end

function _replace_modal_positive_commands(
    ::ModalPushPullPlan{T},
    positive_commands::AbstractMatrix{T},
) where {T}
    return ModalPushPullPlan(positive_commands)
end

function _replace_zonal_amplitudes(
    ::ZonalPushPullPlan{T},
    amplitudes::AbstractVector{T},
) where {T}
    return ZonalPushPullPlan(amplitudes)
end

function _replace_hadamard_amplitudes(
    ::HadamardPushPullPlan{T},
    amplitudes::AbstractVector{T},
) where {T}
    return HadamardPushPullPlan(amplitudes)
end

"""Float32 arbitrary modal positive/negative calibration source."""
@algorithm ModalPushPullF32 begin
    label = "modal-push-pull-f32"

    """Configuration for [`ModalPushPullF32`](@ref)."""
    @configuration ModalPushPullF32Configuration begin
        mode_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        initial_positive_commands::Vector{Float32} => construction
        excitation_schema::String => graph_rebuild
    end

    ports(configuration) = (
        output(
            :excitation,
            Float32,
            (configuration.corrector_coordinate_count,),
            configuration.excitation_schema,
        ),
        output(:pattern_index, schema_element_type(CALIBRATION_PATTERN_INDEX_V1), (1,), CALIBRATION_PATTERN_INDEX_V1;
            name="pattern-index"),
        output(:polarity, schema_element_type(CALIBRATION_PROBE_POLARITY_V1), (1,), CALIBRATION_PROBE_POLARITY_V1),
        output(:completes_cycle, schema_element_type(CALIBRATION_CYCLE_COMPLETE_V1), (1,), CALIBRATION_CYCLE_COMPLETE_V1;
            name="completes-cycle"),
        parameter(
            :positive_commands,
            schema_element_type(MODAL_POSITIVE_COMMANDS_V1),
            (configuration.mode_count, configuration.corrector_coordinate_count),
            MODAL_POSITIVE_COMMANDS_V1;
            name="positive-commands",
            replace=_replace_modal_positive_commands,
        ),
    )

    prepare(configuration) = begin
        plan = ModalPushPullPlan(
            configuration.mode_count,
            configuration.corrector_coordinate_count,
            configuration.initial_positive_commands,
        )
        (plan=plan, workspace=CalibrationProbeWorkspace(plan))
    end

    process!(plan, workspace) = process!(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
        workspace,
        plan,
    )
end

"""Float32 exact single-coordinate positive/negative calibration source."""
@algorithm ZonalPushPullF32 begin
    label = "zonal-push-pull-f32"

    """Configuration for [`ZonalPushPullF32`](@ref)."""
    @configuration ZonalPushPullF32Configuration begin
        amplitudes::Vector{Float32} => graph_rebuild
        excitation_schema::String => graph_rebuild
    end

    ports(configuration) = (
        output(
            :excitation,
            Float32,
            (length(configuration.amplitudes),),
            configuration.excitation_schema,
        ),
        output(:pattern_index, schema_element_type(CALIBRATION_PATTERN_INDEX_V1), (1,), CALIBRATION_PATTERN_INDEX_V1;
            name="pattern-index"),
        output(:polarity, schema_element_type(CALIBRATION_PROBE_POLARITY_V1), (1,), CALIBRATION_PROBE_POLARITY_V1),
        output(:completes_cycle, schema_element_type(CALIBRATION_CYCLE_COMPLETE_V1), (1,), CALIBRATION_CYCLE_COMPLETE_V1;
            name="completes-cycle"),
        parameter(
            :amplitudes,
            schema_element_type(CALIBRATION_PROBE_AMPLITUDES_V1),
            (length(configuration.amplitudes),),
            CALIBRATION_PROBE_AMPLITUDES_V1;
            replace=_replace_zonal_amplitudes,
        ),
    )

    prepare(configuration) = begin
        plan = ZonalPushPullPlan(configuration.amplitudes)
        (plan=plan, workspace=CalibrationProbeWorkspace(plan))
    end

    process!(plan, workspace) = process!(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
        workspace,
        plan,
    )
end

"""Float32 balanced Sylvester-Hadamard positive/negative calibration source."""
@algorithm HadamardPushPullF32 begin
    label = "hadamard-push-pull-f32"

    """Configuration for [`HadamardPushPullF32`](@ref)."""
    @configuration HadamardPushPullF32Configuration begin
        amplitudes::Vector{Float32} => graph_rebuild
        excitation_schema::String => graph_rebuild
    end

    ports(configuration) = (
        output(
            :excitation,
            Float32,
            (length(configuration.amplitudes),),
            configuration.excitation_schema,
        ),
        output(:pattern_index, schema_element_type(CALIBRATION_PATTERN_INDEX_V1), (1,), CALIBRATION_PATTERN_INDEX_V1;
            name="pattern-index"),
        output(:polarity, schema_element_type(CALIBRATION_PROBE_POLARITY_V1), (1,), CALIBRATION_PROBE_POLARITY_V1),
        output(:completes_cycle, schema_element_type(CALIBRATION_CYCLE_COMPLETE_V1), (1,), CALIBRATION_CYCLE_COMPLETE_V1;
            name="completes-cycle"),
        parameter(
            :amplitudes,
            schema_element_type(CALIBRATION_PROBE_AMPLITUDES_V1),
            (length(configuration.amplitudes),),
            CALIBRATION_PROBE_AMPLITUDES_V1;
            replace=_replace_hadamard_amplitudes,
        ),
    )

    prepare(configuration) = begin
        plan = HadamardPushPullPlan(configuration.amplitudes)
        (plan=plan, workspace=CalibrationProbeWorkspace(plan))
    end

    process!(plan, workspace) = process!(
        excitation,
        pattern_index,
        polarity,
        completes_cycle,
        workspace,
        plan,
    )
end
