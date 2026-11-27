"""
    CorrectorStagePlan(to_command, to_controller; coefficients=nothing,
                       lower_limits, upper_limits, slew_limits=nothing, update_period=1)

Prepare one sequential residual corrector. Projection rows are output coordinates:
`to_command` maps controller residual to device commands, and `to_controller`
maps conditioned commands back to controller coordinates. Matrices are copied.
An optional base-rate SOS cascade precedes range and per-update slew conditioning.
Periods greater than one require explicit coefficients; stability checks alone do
not qualify antialias filtering. Commands start at zero, so limits must contain zero.
"""
struct CorrectorStagePlan{T<:Union{Float32,Float64}}
    to_command::Matrix{T}
    to_controller::Matrix{T}
    filter::BiquadCascadePlan{T}
    conditioner::PdmCommandPlan{T}
    update_period::Int
end

function CorrectorStagePlan(P::AbstractMatrix{T}, H::AbstractMatrix{T};
    coefficients=nothing, lower_limits::AbstractVector{T}, upper_limits::AbstractVector{T},
    slew_limits=nothing, update_period::Integer=1,
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(P, H)
    n, m = size(P)
    n > 0 && m > 0 || throw(ArgumentError("corrector dimensions must be positive"))
    size(H) == (m, n) || throw(DimensionMismatch("corrector realization dimensions are invalid"))
    all(isfinite, P) && all(isfinite, H) || throw(ArgumentError("corrector projections must be finite"))
    update_period > 0 || throw(ArgumentError("corrector update period must be positive"))
    isnothing(coefficients) && update_period != 1 && throw(ArgumentError("slower correctors require explicit base-rate coefficients"))
    bank = if isnothing(coefficients)
        identity = zeros(T, 5, 1, 1)
        identity[1, 1, 1] = one(T)
        identity
    else
        coefficients
    end
    conditioner = PdmCommandPlan(lower_limits, upper_limits, slew_limits)
    length(lower_limits) == n || throw(DimensionMismatch("corrector limits have invalid length"))
    all(i -> lower_limits[i] <= zero(T) <= upper_limits[i], eachindex(lower_limits)) ||
        throw(ArgumentError("corrector limits must contain zero"))
    return CorrectorStagePlan{T}(Matrix{T}(P), Matrix{T}(H), BiquadCascadePlan(bank, n), conditioner, Int(update_period))
end

"""An ordered, nonempty collection of correctors sharing controller coordinates."""
struct CorrectorAllocationPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    stages::Vector{CorrectorStagePlan{T}}
    controller_command_count::Int
    command_count::Int
end

function CorrectorAllocationPlan(stages::AbstractVector{CorrectorStagePlan{T}}) where {T}
    isempty(stages) && throw(ArgumentError("at least one corrector stage is required"))
    m = size(first(stages).to_command, 2)
    all(s -> size(s.to_command, 2) == m, stages) || throw(DimensionMismatch("corrector controller extents differ"))
    return CorrectorAllocationPlan{T}(collect(stages), m, sum(s -> size(s.to_command, 1), stages))
end

mutable struct _CorrectorStageWorkspace{T}
    filter::BiquadCascadeWorkspace{T}
    conditioner::PdmCommandWorkspace{T,Vector{T}}
    requested::Vector{T}
    filtered::Vector{T}
    commanded::Vector{T}
    feedback::Vector{T}
    held::Vector{T}
    counter::Int
    candidate_counter::Int
end

"""Own all committed state and candidate scratch for a corrector allocation."""
struct CorrectorAllocationWorkspace{T}
    stages::Vector{_CorrectorStageWorkspace{T}}
    residual::Vector{T}
    commands::Vector{T}
end

function CorrectorAllocationWorkspace(plan::CorrectorAllocationPlan{T}) where {T}
    stages = _CorrectorStageWorkspace{T}[]
    for s in plan.stages
        n = size(s.to_command, 1)
        push!(stages, _CorrectorStageWorkspace(BiquadCascadeWorkspace(s.filter),
            PdmCommandWorkspace(s.conditioner, zeros(T, n)), zeros(T,n), zeros(T,n),
            zeros(T,n), zeros(T,n), zeros(T,n), 0, 0))
    end
    return CorrectorAllocationWorkspace(stages, zeros(T,plan.controller_command_count), zeros(T,plan.command_count))
end

function commit!(workspace::CorrectorAllocationWorkspace)
    for w in workspace.stages
        commit!(w.filter)
        iszero(w.counter) && commit!(w.conditioner)
        copyto!(w.held, w.commanded)
        w.counter = w.candidate_counter
    end
    return workspace
end
function discard!(workspace::CorrectorAllocationWorkspace)
    for w in workspace.stages
        discard!(w.filter)
        discard!(w.conditioner)
        copyto!(w.commanded, w.held)
        w.candidate_counter = w.counter
    end
    return workspace
end
function reset!(workspace::CorrectorAllocationWorkspace{T}) where {T}
    for w in workspace.stages
        reset!(w.filter)
        reset!(w.conditioner)
        for buffer in (w.requested, w.filtered, w.commanded, w.feedback, w.held)
            fill!(buffer, zero(T))
        end
        w.counter = w.candidate_counter = 0
    end
    fill!(workspace.residual, zero(T))
    fill!(workspace.commands, zero(T))
    return workspace
end

function recover!(workspace::CorrectorAllocationWorkspace{T}) where {T}
    offset = 0
    for w in workspace.stages
        recover!(w.filter)
        recover!(w.conditioner)
        for buffer in (w.requested, w.filtered, w.feedback)
            fill!(buffer, zero(T))
        end
        copyto!(w.commanded, w.held)
        w.candidate_counter = w.counter
        copyto!(workspace.commands, offset + 1, w.held, 1, length(w.held))
        offset += length(w.held)
    end
    fill!(workspace.residual, zero(T))
    return workspace
end

"""Expected numerical rejection from sequential corrector allocation."""
@enum CorrectorAllocationFailure::UInt8 begin
    CorrectorAllocationNonFiniteInput = 1
    CorrectorAllocationNonFiniteOutput = 2
end
@doc "Reject nonfinite controller demand." CorrectorAllocationNonFiniteInput
@doc "Reject nonfinite projection, filter, conditioning, or residual arithmetic." CorrectorAllocationNonFiniteOutput

function _corrector_noalias(external, owned)
    Base.mightalias(external, owned) && throw(ArgumentError("corrector caller buffers must not alias owned storage"))
    return nothing
end
_corrector_plan_buffers(s) = (s.to_command, s.to_controller, s.filter.coefficients,
    s.filter.mode_coefficients, s.conditioner.lower_limits, s.conditioner.upper_limits)
function _corrector_check_slew_aliases(buffer, plan)
    for s in plan.stages
        isnothing(s.conditioner.slew_limits) || _corrector_noalias(buffer,s.conditioner.slew_limits)
    end
    return nothing
end
_corrector_workspace_buffers(w) = (w.filter.committed_state, w.filter.candidate_state, w.filter.candidate_output,
    w.conditioner.initial_current, w.conditioner.committed_current, w.conditioner.candidate_current,
    w.conditioner.candidate_demanded, w.conditioner.candidate_constraint_feedback,
    w.requested, w.filtered, w.commanded, w.feedback, w.held)
_corrector_buffers(s,w) = (_corrector_plan_buffers(s)..., _corrector_workspace_buffers(w)...)

_corrector_check_one(buffer, ::Tuple{}) = nothing
function _corrector_check_one(buffer, others::Tuple)
    _corrector_noalias(buffer, first(others))
    _corrector_check_one(buffer, Base.tail(others))
    return nothing
end
_corrector_cross(::Tuple{}, other::Tuple) = nothing
function _corrector_cross(buffers::Tuple, other::Tuple)
    _corrector_check_one(first(buffers), other)
    _corrector_cross(Base.tail(buffers), other)
    return nothing
end
_corrector_disjoint(::Tuple{}) = nothing
function _corrector_disjoint(buffers::Tuple)
    first_buffer = first(buffers)
    rest = Base.tail(buffers)
    foreach(buffer -> _corrector_noalias(first_buffer, buffer), rest)
    _corrector_disjoint(rest)
    return nothing
end
function _validate_corrector_buffers(commands, feedback, workspace, plan, demand)
    Base.require_one_based_indexing(commands, feedback, demand)
    length(commands) == plan.command_count && length(feedback) == length(demand) == plan.controller_command_count ||
        throw(DimensionMismatch("corrector caller buffer extents are invalid"))
    length(workspace.stages) == length(plan.stages) && length(workspace.commands) == plan.command_count && length(workspace.residual) == plan.controller_command_count ||
        throw(DimensionMismatch("corrector workspace extents are invalid"))
    Base.mightalias(commands, feedback) || Base.mightalias(commands, demand) || Base.mightalias(feedback, demand) ?
        throw(ArgumentError("corrector caller buffers must be disjoint")) : nothing
    _corrector_noalias(workspace.commands, workspace.residual)
    _corrector_check_slew_aliases(workspace.commands, plan)
    _corrector_check_slew_aliases(workspace.residual, plan)
    for i in eachindex(plan.stages)
        s, w = plan.stages[i], workspace.stages[i]
        n = size(s.to_command, 1)
        m = plan.controller_command_count
        size(s.to_controller) == (m,n) && size(s.to_command,2) == m &&
            all(buffer -> length(buffer) == n, (w.requested,w.filtered,w.commanded,w.feedback,w.held)) ||
            throw(DimensionMismatch("corrector stage workspace dimensions are invalid"))
        buffers = _corrector_buffers(s,w)
        _corrector_disjoint(buffers)
        foreach(buffer -> _corrector_check_slew_aliases(buffer, plan),_corrector_workspace_buffers(w))
        foreach(buffer -> _corrector_noalias(workspace.commands, buffer), buffers)
        foreach(buffer -> _corrector_noalias(workspace.residual, buffer), buffers)
        for j in (i+1):length(plan.stages)
            other = _corrector_buffers(plan.stages[j],workspace.stages[j])
            _corrector_cross(_corrector_workspace_buffers(w), other)
            _corrector_cross(_corrector_plan_buffers(s), _corrector_workspace_buffers(workspace.stages[j]))
        end
    end
    for external in (commands, feedback, demand)
        _corrector_noalias(external, workspace.commands)
        _corrector_noalias(external, workspace.residual)
        for i in eachindex(plan.stages)
            s, w = plan.stages[i], workspace.stages[i]
            foreach(owned -> _corrector_noalias(external, owned), _corrector_buffers(s, w))
            isnothing(s.conditioner.slew_limits) || _corrector_noalias(external, s.conditioner.slew_limits)
        end
    end
    return nothing
end

function _process_corrector_allocation!(commands::AbstractVector{T}, feedback::AbstractVector{T},
    workspace::CorrectorAllocationWorkspace{T}, plan::CorrectorAllocationPlan{T},
    demand::AbstractVector{T}) where {T}
    _validate_corrector_buffers(commands, feedback, workspace, plan, demand)
    for v in demand
        isfinite(v) || return CorrectorAllocationNonFiniteInput
    end
    copyto!(workspace.residual, demand)
    offset = 0
    for i in eachindex(plan.stages)
        s, w = plan.stages[i], workspace.stages[i]
        fill!(w.requested, zero(T))
        # Traverse contiguous matrix columns, retaining increasing-column
        # accumulation order independently for every output coordinate.
        # Validation above proves dense extents and disjoint owned buffers.
        # SIMD spans independent rows; each row retains ordered column FMA.
        for col in axes(s.to_command, 2)
            @inbounds @simd ivdep for row in axes(s.to_command, 1)
                w.requested[row] = fma(s.to_command[row,col], workspace.residual[col], w.requested[row])
            end
        end
        result = _process_biquad_cascade!(w.filtered, w.filter, s.filter, w.requested)
        result === nothing || return CorrectorAllocationNonFiniteOutput
        if iszero(w.counter)
            result = process!(w.commanded, w.feedback, w.conditioner, s.conditioner, w.filtered)
            result === nothing || return CorrectorAllocationNonFiniteOutput
        else
            copyto!(w.commanded, w.held)
        end
        for col in axes(s.to_controller, 2)
            @inbounds @simd ivdep for row in axes(s.to_controller, 1)
                workspace.residual[row] = fma(-s.to_controller[row,col], w.commanded[col], workspace.residual[row])
            end
        end
        # Residual scratch is private; reject before advancing to another stage
        # or publishing either output, just as in the row-wise traversal.
        for value in workspace.residual
            isfinite(value) || return CorrectorAllocationNonFiniteOutput
        end
        for j in eachindex(w.commanded)
            workspace.commands[offset+j] = w.commanded[j]
        end
        offset += length(w.commanded)
        w.candidate_counter = w.counter == s.update_period - 1 ? 0 : w.counter + 1
    end
    copyto!(commands, workspace.commands)
    copyto!(feedback, workspace.residual)
    return nothing
end

"""
    process!(commands, feedback, workspace, plan, demand)

Stage sequential residual allocation in controller coordinates. Every filter
advances at the base rate; each conditioned command updates at tick zero and
then every stage's positive integer period, holding between updates. Subtract
conditioned/held commands before allocating the next stage. Feedback
is the final unserved controller residual. Call `commit!` after success or
`discard!` to abandon staging. Expected numerical failure preserves both caller
outputs and committed state and restores candidates. Commands concatenate each
stage's device coordinates in configured stage order. Caller buffers must be
disjoint one-based CPU vectors and must not be concurrently mutated. Calibration
declares coordinate meaning and device units. This transaction covers this allocation only.
"""
function process!(commands::AbstractVector{T}, feedback::AbstractVector{T},
    workspace::CorrectorAllocationWorkspace{T}, plan::CorrectorAllocationPlan{T},
    demand::AbstractVector{T}) where {T}
    result = _process_corrector_allocation!(commands, feedback, workspace, plan, demand)
    result === nothing || discard!(workspace)
    return result
end

function _prepare_corrector_allocation(c)
    c.controller_command_count > 0 && !isempty(c.stage_command_counts) && all(>(0), c.stage_command_counts) ||
        throw(ArgumentError("corrector extents must be positive"))
    length(c.update_periods) == length(c.stage_command_counts) || throw(DimensionMismatch("corrector period count is invalid"))
    max_elements = div(UInt128(typemax(UInt32)), UInt128(sizeof(Float32)))
    total = sum(UInt128, c.stage_command_counts)
    total <= max_elements && UInt128(c.controller_command_count) <= max_elements ||
        throw(ArgumentError("corrector port byte extents must fit UInt32"))
    n, m = Int(total), c.controller_command_count
    UInt128(n)*UInt128(m) <= max_elements || throw(ArgumentError("corrector projection byte extents must fit UInt32"))
    c.section_count > 0 || throw(ArgumentError("corrector section count must be positive"))
    UInt128(5)*UInt128(c.section_count)*UInt128(n) <= max_elements || throw(ArgumentError("corrector coefficient byte extents must fit UInt32"))
    length(c.command_projection) == length(c.realization_projection) == n*m || throw(DimensionMismatch("corrector projection lengths are invalid"))
    length(c.command_limits[1]) == length(c.command_limits[2]) == n || throw(DimensionMismatch("corrector limit lengths are invalid"))
    isnothing(c.slew_limits) || length(c.slew_limits) == n || throw(DimensionMismatch("corrector slew length is invalid"))
    isnothing(c.initial_coefficients) || length(c.initial_coefficients) == 5*c.section_count*n || throw(DimensionMismatch("corrector coefficient length is invalid"))
    P = permutedims(reshape(c.command_projection, m, n))
    H = permutedims(reshape(c.realization_projection, n, m))
    bank = if isnothing(c.initial_coefficients)
        all(==(1), c.update_periods) || throw(ArgumentError("slower correctors require explicit base-rate coefficients"))
        identity = zeros(Float32, 5, c.section_count, n)
        identity[1,:,:] .= 1f0
        identity
    else
        reshape(c.initial_coefficients, 5, c.section_count, n)
    end
    stages = CorrectorStagePlan{Float32}[]
    offset = 0
    for i in eachindex(c.stage_command_counts)
        indices = (offset+1):(offset+c.stage_command_counts[i])
        coefficients = bank[:,:,indices]
        push!(stages, CorrectorStagePlan(P[indices,:], H[:,indices]; coefficients,
            lower_limits=c.command_limits[1][indices], upper_limits=c.command_limits[2][indices],
            slew_limits=isnothing(c.slew_limits) ? nothing : c.slew_limits[indices], update_period=c.update_periods[i]))
        offset += c.stage_command_counts[i]
    end
    plan = CorrectorAllocationPlan(stages)
    return (plan=plan, workspace=CorrectorAllocationWorkspace(plan))
end

"""CPU sequential residual corrector allocation with ordered stage commands and controller-coordinate constraint feedback. See [`CorrectorStagePlan`](@ref)."""
@algorithm CorrectorAllocationF32 begin
    label = "corrector-allocation-f32"
    failure = CorrectorAllocationFailure
    """Configure per-stage command counts, integer update periods, row-major command/realization projections, flattened column-major `(5, section_count, total_commands)` SOS coefficients, limits, and caller-declared command schema. Slower stages require explicit base-rate coefficients. Coordinates and device units are established by caller calibration."""
    @configuration CorrectorAllocationF32Configuration begin
        controller_command_count::Int => graph_rebuild
        stage_command_counts::Vector{Int} => graph_rebuild
        update_periods::Vector{Int} => construction
        command_projection::Vector{Float32} => construction
        realization_projection::Vector{Float32} => construction
        section_count::Int => construction
        initial_coefficients::Union{Nothing,Vector{Float32}} => construction
        command_limits::NTuple{2,Vector{Float32}} => construction
        slew_limits::Union{Nothing,Vector{Float32}} => construction
        command_schema::String => graph_rebuild
    end
    ports(configuration) = (
        input(:controller_command, schema_element_type(CONTROLLER_COMMAND_V1), (configuration.controller_command_count,), CONTROLLER_COMMAND_V1; name="controller-command"),
        output(:commands, Float32, (sum(configuration.stage_command_counts),), configuration.command_schema; metadata_from=:controller_command),
        output(:constraint_feedback, schema_element_type(CONTROLLER_CONSTRAINT_FEEDBACK_V1), (configuration.controller_command_count,), CONTROLLER_CONSTRAINT_FEEDBACK_V1; name="controller-constraint-feedback", metadata_from=:controller_command),
    )
    prepare(configuration) = _prepare_corrector_allocation(configuration)
    process!(plan, workspace) = _process_corrector_allocation!(commands, constraint_feedback, workspace, plan, controller_command)
end
