"""
    FrameDelayPlan(delay_frames)

Prepare a fixed delay measured in samples successfully adopted by this
Algorithm. A delay of zero copies the current input. A positive delay publishes
the input from `delay_frames` successful local adoptions earlier. A later
Algorithm rejecting the same Graph sample does not roll back this history.
"""
struct FrameDelayPlan <: AbstractAlgorithmPlan
    delay_frames::Int

    function FrameDelayPlan(delay_frames::Integer)
        delay_frames >= 0 || throw(ArgumentError("delay_frames must be nonnegative"))
        return new(Int(delay_frames))
    end
end

"""
    FrameDelayWorkspace(initial_state, plan)

Own the fixed history, staged input, and ring position for one frame delay.
Processing does not change committed history. [`commit!`](@ref) advances the
history by one frame; [`discard!`](@ref) leaves it unchanged.
"""
mutable struct FrameDelayWorkspace{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
}
    initial_state::V
    history::M
    candidate_input::V
    head::Int
end

function FrameDelayWorkspace(
    initial_state::AbstractVector{T},
    plan::FrameDelayPlan,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(initial_state)
    isempty(initial_state) && throw(ArgumentError("state length must be positive"))
    all(isfinite, initial_state) ||
        throw(ArgumentError("initial delay state must be finite"))
    initial = Vector{T}(initial_state)
    history = Matrix{T}(undef, length(initial), plan.delay_frames)
    for frame in axes(history, 2), index in eachindex(initial)
        history[index, frame] = initial[index]
    end
    candidate = copy(initial)
    return FrameDelayWorkspace{T,typeof(initial),typeof(history)}(
        initial,
        history,
        candidate,
        1,
    )
end

function FrameDelayWorkspace(
    ::Type{T},
    extent::Integer,
    plan::FrameDelayPlan;
    initial_value::T=zero(T),
) where {T<:AbstractFloat}
    extent > 0 || throw(ArgumentError("extent must be positive"))
    return FrameDelayWorkspace(fill(initial_value, Int(extent)), plan)
end

"""
    frame_delay_history(workspace::FrameDelayWorkspace)

Return the committed frame-delay history in ring-storage order as a read-only
borrowed reference to workspace storage. Later commits update this buffer; this
reference is not a retained snapshot. Use `copy` on the analysis or control path
when a snapshot is needed, and serialize access with workspace use.
"""
frame_delay_history(workspace::FrameDelayWorkspace) = workspace.history

"""
    frame_delay_head(workspace::FrameDelayWorkspace)

Return the ring column that will leave the delay on the next valid frame.
"""
frame_delay_head(workspace::FrameDelayWorkspace) = workspace.head

function commit!(workspace::FrameDelayWorkspace)
    delay_frames = size(workspace.history, 2)
    if !iszero(delay_frames)
        head = workspace.head
        for index in eachindex(workspace.candidate_input)
            workspace.history[index, head] = workspace.candidate_input[index]
        end
        workspace.head = head == delay_frames ? 1 : head + 1
    end
    return workspace
end

discard!(workspace::FrameDelayWorkspace) = workspace

function reset!(workspace::FrameDelayWorkspace)
    for frame in axes(workspace.history, 2), index in eachindex(workspace.initial_state)
        workspace.history[index, frame] = workspace.initial_state[index]
    end
    copyto!(workspace.candidate_input, workspace.initial_state)
    workspace.head = 1
    return workspace
end

"""Expected numerical rejection from a fixed frame delay."""
@enum FrameDelayFailure::UInt8 begin
    FrameDelayNonFiniteInput = 1
end

@doc "The delayed Float32 vector input contains a non-finite value." FrameDelayNonFiniteInput

function _process_frame_delay!(
    output::AbstractVector{T},
    workspace::FrameDelayWorkspace{T},
    plan::FrameDelayPlan,
    input::AbstractVector{T},
)::Union{Nothing,FrameDelayFailure} where {T}
    Base.require_one_based_indexing(output, input)
    axes(output) == axes(input) == axes(workspace.initial_state) ||
        throw(DimensionMismatch("frame-delay input, output, and state axes must match"))
    size(workspace.history) == (length(input), plan.delay_frames) ||
        throw(DimensionMismatch("frame-delay history has an incompatible extent"))
    axes(workspace.candidate_input) == axes(input) ||
        throw(DimensionMismatch("frame-delay candidate input has incompatible axes"))
    !iszero(plan.delay_frames) && Base.mightalias(output, input) && throw(
        ArgumentError("positive-delay input and output must not alias"),
    )

    for value in input
        isfinite(value) || return FrameDelayNonFiniteInput
    end

    if iszero(plan.delay_frames)
        copyto!(output, input)
        return nothing
    end

    head = workspace.head
    for index in eachindex(output, input, workspace.candidate_input)
        output[index] = workspace.history[index, head]
        workspace.candidate_input[index] = input[index]
    end
    return nothing
end

function process!(
    output::AbstractVector{T},
    workspace::FrameDelayWorkspace{T},
    plan::FrameDelayPlan,
    input::AbstractVector{T},
) where {T}
    result = _process_frame_delay!(output, workspace, plan, input)
    return result === nothing ? output : result
end

function _validate_frame_delay_configuration(extent::Int, delay_frames::Int)
    extent > 0 || throw(ArgumentError("extent must be positive"))
    delay_frames >= 0 || throw(ArgumentError("delay_frames must be nonnegative"))
    elements = UInt128(extent) * UInt128(delay_frames)
    elements <= div(UInt128(typemax(UInt32)), UInt128(sizeof(Float32))) || throw(
        ArgumentError("frame-delay history byte extent must fit UInt32"),
    )
    return nothing
end

"""
Delay an entire Float32 vector by `delay_frames` successful local adoptions.
Before history is available, publish `initial_value`; zero delay passes the
input through. Output values have the same units as input values.
"""
@algorithm FrameDelayF32 begin
    label = "frame-delay-f32"
    failure = FrameDelayFailure

    """
    Set positive vector `extent`, nonnegative `delay_frames`, finite
    `initial_value`, and versioned input/output schema identifiers. Delay is
    measured in successful local adoptions, not elapsed time.
    """
    @configuration FrameDelayF32Configuration begin
        extent::Int => graph_rebuild
        delay_frames::Int => graph_rebuild
        initial_value::Float32 => construction
        input_schema::String => graph_rebuild
        output_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            configuration.input_schema,
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            configuration.output_schema;
            metadata_from=:input,
        ),
    )

    prepare(configuration) = begin
        _validate_frame_delay_configuration(
            configuration.extent,
            configuration.delay_frames,
        )
        isfinite(configuration.initial_value) ||
            throw(ArgumentError("initial_value must be finite"))
        plan = FrameDelayPlan(configuration.delay_frames)
        (
            plan=plan,
            workspace=FrameDelayWorkspace(
                Float32,
                configuration.extent,
                plan;
                initial_value=configuration.initial_value,
            ),
        )
    end

    process!(plan, workspace) = begin
        _process_frame_delay!(output, workspace, plan, input)
    end
end
