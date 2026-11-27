"""
    ClosedLoopCorrectionPlan(...)

Prepare leaky integration, delayed constraint-feedback anti-windup, and
hidden-mode removal for one controller-coordinate vector. The keyword `closed`
defaults to `true`. With `closed=false`, successful samples republish the committed
correction and mathematical controller state without residual, anti-windup, or
hidden-mode arithmetic. Shape, alias, finite-input and delayed-feedback contracts
still apply. The initial held correction/state is zero. Closing again resumes
from the held preceding correction without clearing state. Downstream temporal
filters and physical response continue settling.

On the CPU, anti-windup and hidden-mode subtraction use fused multiply-add.
CPU correction uses
`fma(gain, residual_error, pole * state)`, rounding the bounded pole product
before the fused input contribution. Fused cancellation can keep a result
finite when an independently rounded intermediate product would overflow.
"""
struct ClosedLoopCorrectionPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    gain::T
    pole::T
    anti_windup_gain::T
    hidden_mode_gain::T
    closed::Bool
    state_length::Int
    hidden_mode_count::Int
    shape_to_hidden::Matrix{T}
    hidden_to_shape::Matrix{T}
end

function _closed_loop_correction_plan(
    gain::T,
    pole::T,
    anti_windup_gain::T,
    hidden_mode_gain::T,
    state_length::Int,
    shape_to_hidden::Matrix{T},
    hidden_to_shape::Matrix{T},
    closed::Bool,
) where {T<:Union{Float32,Float64}}
    state_length > 0 || throw(ArgumentError("state length must be positive"))
    isfinite(gain) || throw(ArgumentError("gain must be finite"))
    isfinite(pole) && zero(T) <= pole <= one(T) ||
        throw(ArgumentError("pole must be finite and in [0, 1]"))
    isfinite(anti_windup_gain) ||
        throw(ArgumentError("anti-windup gain must be finite"))
    isfinite(hidden_mode_gain) ||
        throw(ArgumentError("hidden-mode gain must be finite"))

    hidden_mode_count = size(shape_to_hidden, 1)
    size(shape_to_hidden, 2) == state_length || throw(
        DimensionMismatch("shape-to-hidden projection has the wrong state extent"),
    )
    size(hidden_to_shape) == (state_length, hidden_mode_count) || throw(
        DimensionMismatch("hidden-to-shape projection has the wrong extent"),
    )
    all(isfinite, shape_to_hidden) ||
        throw(ArgumentError("shape-to-hidden projection must be finite"))
    all(isfinite, hidden_to_shape) ||
        throw(ArgumentError("hidden-to-shape projection must be finite"))

    return ClosedLoopCorrectionPlan{T}(
        gain,
        pole,
        anti_windup_gain,
        hidden_mode_gain,
        closed,
        state_length,
        hidden_mode_count,
        shape_to_hidden,
        hidden_to_shape,
    )
end

function ClosedLoopCorrectionPlan(
    gain::T,
    pole::T,
    anti_windup_gain::T,
    hidden_mode_gain::T,
    state_length::Integer,
    shape_to_hidden::AbstractMatrix{T},
    hidden_to_shape::AbstractMatrix{T};
    closed::Bool=true,
) where {T<:Union{Float32,Float64}}
    return _closed_loop_correction_plan(
        gain,
        pole,
        anti_windup_gain,
        hidden_mode_gain,
        Int(state_length),
        Matrix(shape_to_hidden),
        Matrix(hidden_to_shape),
        closed,
    )
end

"""
    ClosedLoopCorrectionWorkspace(plan)

Own committed and candidate controller state, the preceding correction, and
fixed hidden-mode scratch for one prepared closed-loop correction Algorithm.
"""
mutable struct ClosedLoopCorrectionWorkspace{T<:Union{Float32,Float64}}
    committed_state::Vector{T}
    committed_output::Vector{T}
    committed_has_output::Bool
    candidate_state::Vector{T}
    candidate_output::Vector{T}
    candidate_has_output::Bool
    realized::Vector{T}
    hidden_modes::Vector{T}
    hidden_in_state::Vector{T}
end

function ClosedLoopCorrectionWorkspace(plan::ClosedLoopCorrectionPlan{T}) where {T}
    state_length = plan.state_length
    return ClosedLoopCorrectionWorkspace{T}(
        zeros(T, state_length),
        zeros(T, state_length),
        false,
        zeros(T, state_length),
        zeros(T, state_length),
        false,
        zeros(T, state_length),
        zeros(T, plan.hidden_mode_count),
        zeros(T, state_length),
    )
end

"""
    closed_loop_correction_state(workspace::ClosedLoopCorrectionWorkspace)

Return the committed feedback-corrected controller state as a read-only borrowed
reference to workspace storage. Commit can swap this buffer, and later
operations can reuse it; this reference is not a retained snapshot. Use `copy`
on the analysis or control path when a snapshot is needed, and serialize access
with workspace use.
"""
closed_loop_correction_state(workspace::ClosedLoopCorrectionWorkspace) =
    workspace.committed_state

"""
    closed_loop_correction_last_output(workspace::ClosedLoopCorrectionWorkspace)

Return the preceding committed correction, or `nothing` before the first call.
The array is a read-only borrowed reference to workspace storage. Commit can
swap this buffer, and later operations can reuse it; it is not a retained
snapshot. Use `copy` on the analysis or control path when an array snapshot is
needed, and serialize access with workspace use.
"""
closed_loop_correction_last_output(workspace::ClosedLoopCorrectionWorkspace) =
    workspace.committed_has_output ? workspace.committed_output : nothing

function commit!(workspace::ClosedLoopCorrectionWorkspace)
    workspace.committed_state, workspace.candidate_state =
        workspace.candidate_state, workspace.committed_state
    workspace.committed_output, workspace.candidate_output =
        workspace.candidate_output, workspace.committed_output
    workspace.committed_has_output = workspace.candidate_has_output
    return workspace
end

function discard!(workspace::ClosedLoopCorrectionWorkspace)
    copyto!(workspace.candidate_state, workspace.committed_state)
    copyto!(workspace.candidate_output, workspace.committed_output)
    workspace.candidate_has_output = workspace.committed_has_output
    return workspace
end

function reset!(workspace::ClosedLoopCorrectionWorkspace)
    fill!(workspace.committed_state, zero(eltype(workspace.committed_state)))
    fill!(workspace.committed_output, zero(eltype(workspace.committed_output)))
    fill!(workspace.candidate_state, zero(eltype(workspace.candidate_state)))
    fill!(workspace.candidate_output, zero(eltype(workspace.candidate_output)))
    fill!(workspace.realized, zero(eltype(workspace.realized)))
    fill!(workspace.hidden_modes, zero(eltype(workspace.hidden_modes)))
    fill!(workspace.hidden_in_state, zero(eltype(workspace.hidden_in_state)))
    workspace.committed_has_output = false
    workspace.candidate_has_output = false
    return workspace
end

_plan_adopted!(workspace::ClosedLoopCorrectionWorkspace) = discard!(workspace)

recover!(workspace::ClosedLoopCorrectionWorkspace) = reset!(workspace)

"""Expected numerical rejection from closed-loop correction."""
@enum ClosedLoopCorrectionFailure::UInt8 begin
    ClosedLoopCorrectionMissingFeedback = 1
    ClosedLoopCorrectionUnexpectedFeedback = 2
    ClosedLoopCorrectionNonFiniteInput = 3
    ClosedLoopCorrectionNonFiniteFeedback = 4
    ClosedLoopCorrectionNonFiniteOutput = 5
end

@doc "A previous correction was committed, but its delayed constraint feedback is absent." ClosedLoopCorrectionMissingFeedback
@doc "Constraint feedback was supplied before any correction was committed." ClosedLoopCorrectionUnexpectedFeedback
@doc "The current residual-error input has a non-finite component." ClosedLoopCorrectionNonFiniteInput
@doc "The delayed constraint-feedback vector has a non-finite component." ClosedLoopCorrectionNonFiniteFeedback
@doc "Anti-windup, hidden-mode removal, or correction produced a non-finite state/output." ClosedLoopCorrectionNonFiniteOutput

@inline function _closed_loop_correction_output_aliases_owned(output, workspace, plan)
    return Base.mightalias(output, workspace.committed_state) ||
        Base.mightalias(output, workspace.committed_output) ||
        Base.mightalias(output, workspace.candidate_state) ||
        Base.mightalias(output, workspace.candidate_output) ||
        Base.mightalias(output, workspace.realized) ||
        Base.mightalias(output, workspace.hidden_modes) ||
        Base.mightalias(output, workspace.hidden_in_state) ||
        Base.mightalias(output, plan.shape_to_hidden) ||
        Base.mightalias(output, plan.hidden_to_shape)
end

"""
    process!(correction, controller_state, workspace::ClosedLoopCorrectionWorkspace,
             plan::ClosedLoopCorrectionPlan, residual_error, constraint_feedback)

Stage correction and controller-state outputs in disjoint caller-owned vectors.
Outputs must not alias prepared projections or workspace storage. Ordinary
input/output overlap is supported because both results are staged before
publication. A rejected call preserves caller outputs and committed history.
"""
function process!(
    correction::AbstractVector{T},
    controller_state::AbstractVector{T},
    workspace::ClosedLoopCorrectionWorkspace{T},
    plan::ClosedLoopCorrectionPlan{T},
    residual_error::AbstractVector{T},
    constraint_feedback::Union{Nothing,AbstractVector{T}},
) where {T}
    Base.require_one_based_indexing(correction, controller_state, residual_error)
    isnothing(constraint_feedback) ||
        Base.require_one_based_indexing(constraint_feedback)
    state_axes = axes(workspace.committed_state)
    axes(correction) == state_axes &&
        axes(controller_state) == state_axes &&
        axes(residual_error) == state_axes || throw(
        DimensionMismatch("closed-loop data and Workspace axes must match"),
    )
    if !isnothing(constraint_feedback) && axes(constraint_feedback) != state_axes
        throw(DimensionMismatch("constraint-feedback axes must match controller state"))
    end
    Base.mightalias(correction, controller_state) && throw(
        ArgumentError("closed-loop correction and controller-state outputs must not overlap"),
    )
    (_closed_loop_correction_output_aliases_owned(correction, workspace, plan) ||
        _closed_loop_correction_output_aliases_owned(controller_state, workspace, plan)) && throw(
        ArgumentError("closed-loop outputs must not alias plan or workspace storage"),
    )
    for value in residual_error
        isfinite(value) || return ClosedLoopCorrectionNonFiniteInput
    end

    workspace.candidate_has_output = false
    if workspace.committed_has_output
        isnothing(constraint_feedback) &&
            return ClosedLoopCorrectionMissingFeedback
        for value in constraint_feedback
            isfinite(value) || return ClosedLoopCorrectionNonFiniteFeedback
        end
    else
        isnothing(constraint_feedback) ||
            return ClosedLoopCorrectionUnexpectedFeedback
    end

    if !plan.closed
        copyto!(workspace.candidate_state, workspace.committed_state)
        copyto!(workspace.candidate_output, workspace.committed_output)
        copyto!(correction, workspace.candidate_output)
        copyto!(controller_state, workspace.candidate_state)
        workspace.candidate_has_output = true
        return nothing
    end

    if workspace.committed_has_output
        for index in eachindex(
            workspace.realized,
            workspace.candidate_state,
            workspace.committed_output,
            constraint_feedback,
        )
            output = workspace.committed_output[index]
            feedback = constraint_feedback[index]
            workspace.realized[index] = output - feedback
            workspace.candidate_state[index] =
                fma(-plan.anti_windup_gain, feedback, output)
        end
        if !iszero(plan.hidden_mode_count)
            mul!(
                workspace.hidden_modes,
                plan.shape_to_hidden,
                workspace.realized,
            )
            mul!(
                workspace.hidden_in_state,
                plan.hidden_to_shape,
                workspace.hidden_modes,
            )
            for index in eachindex(
                workspace.candidate_state,
                workspace.hidden_in_state,
            )
                workspace.candidate_state[index] = fma(
                    -plan.hidden_mode_gain,
                    workspace.hidden_in_state[index],
                    workspace.candidate_state[index],
                )
            end
        end
    else
        copyto!(workspace.candidate_state, workspace.committed_state)
    end

    # Rejected candidate scratch is disposable; validate before publication.
    finite_output = true
    for index in eachindex(
        workspace.candidate_output,
        workspace.candidate_state,
        residual_error,
    )
        state = workspace.candidate_state[index]
        output = fma(plan.gain, residual_error[index], plan.pole * state)
        finite_output = finite_output & (isfinite(state) & isfinite(output))
        workspace.candidate_output[index] = output
    end
    finite_output || return ClosedLoopCorrectionNonFiniteOutput
    copyto!(correction, workspace.candidate_output)
    copyto!(controller_state, workspace.candidate_state)
    workspace.candidate_has_output = true
    return nothing
end

const CLOSED_LOOP_CORRECTION_GAIN_PROPERTY = UInt32(0)
const CLOSED_LOOP_CORRECTION_POLE_PROPERTY = UInt32(1)
const CLOSED_LOOP_CORRECTION_ANTI_WINDUP_GAIN_PROPERTY = UInt32(2)
const CLOSED_LOOP_CORRECTION_HIDDEN_MODE_GAIN_PROPERTY = UInt32(3)
const CLOSED_LOOP_CORRECTION_CLOSED_PROPERTY = UInt32(4)
const _CLOSED_LOOP_CORRECTION_PROPERTY_NAMES = (
    :gain,
    :pole,
    Symbol("anti-windup-gain"),
    Symbol("hidden-mode-gain"),
    :closed,
)

function property_schema(::Type{ClosedLoopCorrectionPlan{T}}) where {T}
    finite_range = PropertyRange(-floatmax(T), floatmax(T))
    return (
        runtime_property(
            CLOSED_LOOP_CORRECTION_GAIN_PROPERTY,
            :gain,
            T;
            default=one(T),
            description="Leaky-integrator input gain g",
            unit="1",
            constraint=finite_range,
        ),
        runtime_property(
            CLOSED_LOOP_CORRECTION_POLE_PROPERTY,
            :pole,
            T;
            default=zero(T),
            description="Retained correction per update p; 1 means no leak. Opening holds controller state.",
            unit="1",
            constraint=PropertyRange(zero(T), one(T)),
        ),
        runtime_property(
            CLOSED_LOOP_CORRECTION_ANTI_WINDUP_GAIN_PROPERTY,
            Symbol("anti-windup-gain"),
            T;
            default=zero(T),
            description="Clipping/constraint feedback gain k_aw: removes conditioned command error from the integrator. Zero disables this compensation, not command limits or device protection.",
            unit="1",
            constraint=finite_range,
        ),
        runtime_property(
            CLOSED_LOOP_CORRECTION_HIDDEN_MODE_GAIN_PROPERTY,
            Symbol("hidden-mode-gain"),
            T;
            default=zero(T),
            description="Hidden-mode removal gain k_h",
            unit="1",
            constraint=finite_range,
        ),
        runtime_property(
            CLOSED_LOOP_CORRECTION_CLOSED_PROPERTY,
            :closed,
            Bool;
            default=true,
            description="Close feedback integration; false holds committed correction and state",
            unit="1",
        ),
    )
end

function property_values(plan::ClosedLoopCorrectionPlan)
    return NamedTuple{_CLOSED_LOOP_CORRECTION_PROPERTY_NAMES}(
        (
            plan.gain,
            plan.pole,
            plan.anti_windup_gain,
            plan.hidden_mode_gain,
            plan.closed,
        ),
    )
end

function replace_properties(
    plan::ClosedLoopCorrectionPlan{T},
    values::NamedTuple{
        _CLOSED_LOOP_CORRECTION_PROPERTY_NAMES,
        Tuple{T,T,T,T,Bool},
    },
) where {T}
    return _closed_loop_correction_plan(
        values.gain,
        values.pole,
        getproperty(values, Symbol("anti-windup-gain")),
        getproperty(values, Symbol("hidden-mode-gain")),
        plan.state_length,
        plan.shape_to_hidden,
        plan.hidden_to_shape,
        values.closed,
    )
end

function _replace_shape_to_hidden(plan::ClosedLoopCorrectionPlan, values)
    return _closed_loop_correction_plan(
        plan.gain,
        plan.pole,
        plan.anti_windup_gain,
        plan.hidden_mode_gain,
        plan.state_length,
        Matrix(values),
        plan.hidden_to_shape,
        plan.closed,
    )
end

function _replace_hidden_to_shape(plan::ClosedLoopCorrectionPlan, values)
    return _closed_loop_correction_plan(
        plan.gain,
        plan.pole,
        plan.anti_windup_gain,
        plan.hidden_mode_gain,
        plan.state_length,
        plan.shape_to_hidden,
        Matrix(values),
        plan.closed,
    )
end

function _validate_closed_loop_correction_dimensions(
    state_length::Int,
    hidden_mode_count::Int,
)
    state_length > 0 || throw(ArgumentError("state_length must be positive"))
    hidden_mode_count > 0 ||
        throw(ArgumentError("hidden_mode_count must be positive"))
    max_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    encoded_state_length = UInt64(state_length)
    encoded_hidden_mode_count = UInt64(hidden_mode_count)
    encoded_state_length <= max_elements ||
        throw(ArgumentError("controller state byte extent must fit UInt32"))
    encoded_hidden_mode_count <= div(max_elements, encoded_state_length) ||
        throw(ArgumentError("hidden-mode projection byte extent must fit UInt32"))
    return nothing
end

"""
Float32 feedback-aware closed-loop correction. The candidate controller state
uses the preceding correction and delayed constraint feedback for anti-windup,
optionally removes projected hidden modes, then forms each correction as
`fma(gain, residual_error, pole * state)`. State and correction have controller
command units; gain, anti-windup gain, hidden-mode gain, and pole are
dimensionless.
"""
@algorithm ClosedLoopCorrectionF32 begin
    label = "closed-loop-correction-f32"
    failure = ClosedLoopCorrectionFailure

    """
    Set the controller `state_length` and `hidden_mode_count`. The two projection
    parameter ports have shapes `(hidden_mode_count, state_length)` and
    `(state_length, hidden_mode_count)`; gain, pole, anti-windup gain, and hidden-mode
    gain are runtime properties.
    """
    @configuration ClosedLoopCorrectionF32Configuration begin
        state_length::Int => graph_rebuild
        hidden_mode_count::Int => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(
            :residual_error,
            schema_element_type(CONTROLLER_RESIDUAL_ERROR_V1),
            (configuration.state_length,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            name="residual-error",
        ),
        input(
            :constraint_feedback,
            schema_element_type(CONTROLLER_CONSTRAINT_FEEDBACK_V1),
            (configuration.state_length,),
            CONTROLLER_CONSTRAINT_FEEDBACK_V1;
            name="constraint-feedback",
        ),
        output(
            :correction,
            schema_element_type(CONTROLLER_COMMAND_V1),
            (configuration.state_length,),
            CONTROLLER_COMMAND_V1;
            metadata_from=:residual_error,
        ),
        output(
            :controller_state,
            schema_element_type(CONTROLLER_STATE_V1),
            (configuration.state_length,),
            CONTROLLER_STATE_V1;
            name="controller-state",
            metadata_from=:residual_error,
        ),
        parameter(
            :shape_to_hidden,
            schema_element_type(SHAPE_TO_HIDDEN_PROJECTION_V1),
            (configuration.hidden_mode_count, configuration.state_length),
            SHAPE_TO_HIDDEN_PROJECTION_V1;
            name="shape-to-hidden",
            replace=(plan, values) -> _replace_shape_to_hidden(plan, values),
        ),
        parameter(
            :hidden_to_shape,
            schema_element_type(HIDDEN_TO_SHAPE_PROJECTION_V1),
            (configuration.state_length, configuration.hidden_mode_count),
            HIDDEN_TO_SHAPE_PROJECTION_V1;
            name="hidden-to-shape",
            replace=(plan, values) -> _replace_hidden_to_shape(plan, values),
        ),
    )

    prepare(configuration) = begin
        _validate_closed_loop_correction_dimensions(
            configuration.state_length,
            configuration.hidden_mode_count,
        )
        plan = ClosedLoopCorrectionPlan(
            1.0f0,
            0.0f0,
            0.0f0,
            0.0f0,
            configuration.state_length,
            zeros(
                Float32,
                configuration.hidden_mode_count,
                configuration.state_length,
            ),
            zeros(
                Float32,
                configuration.state_length,
                configuration.hidden_mode_count,
            ),
        )
        (plan=plan, workspace=ClosedLoopCorrectionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        feedback = isnothing(closed_loop_correction_last_output(workspace)) ?
                   nothing : constraint_feedback
        process!(
            correction,
            controller_state,
            workspace,
            plan,
            residual_error,
            feedback,
        )
    end
end
