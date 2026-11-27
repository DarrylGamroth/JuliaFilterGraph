"""
    SccPushPullResponsePlan(T, pixel_count)
    SccPushPullResponsePlan(pixel_count)

Prepare the Thompson four-image self-coherent-camera (SCC) calibration
response for one nonempty image extent. The response at each pixel is

```text
(push_fringed - push_unfringed) - (pull_fringed - pull_unfringed)
```

The four inputs must already be calibrated and associated with the named probe
and SCC-reference states. This Algorithm does not infer acquisition order,
divide by probe amplitude, or apply a spatial filter.
"""
struct SccPushPullResponsePlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    pixel_count::Int

    function SccPushPullResponsePlan(
        ::Type{T},
        pixel_count::Integer,
    ) where {T<:AbstractFloat}
        pixels = Int(pixel_count)
        pixels > 0 || throw(
            ArgumentError("SCC push-pull response pixel_count must be positive"),
        )
        return new{T}(pixels)
    end
end

SccPushPullResponsePlan(pixel_count::Integer) =
    SccPushPullResponsePlan(Float32, pixel_count)

"""
    SccPushPullResponseWorkspace(plan)

Allocate fixed staging for one SCC response Node. Processing publishes the
caller-owned response only after all four images and every double difference
are finite. The Workspace contains disposable scratch but no mathematical
history, so `commit!` and `discard!` are no-ops.
"""
mutable struct SccPushPullResponseWorkspace{T<:AbstractFloat}
    staged_response::Vector{T}
end

SccPushPullResponseWorkspace(plan::SccPushPullResponsePlan{T}) where {T} =
    SccPushPullResponseWorkspace{T}(zeros(T, plan.pixel_count))

commit!(workspace::SccPushPullResponseWorkspace) = workspace
discard!(workspace::SccPushPullResponseWorkspace) = workspace

function reset!(workspace::SccPushPullResponseWorkspace{T}) where {T}
    fill!(workspace.staged_response, zero(T))
    return workspace
end

"""Expected numerical rejection from SCC four-image differencing."""
@enum SccPushPullResponseFailure::UInt8 begin
    SccPushPullNonFinitePushFringed = 1
    SccPushPullNonFinitePushUnfringed = 2
    SccPushPullNonFinitePullFringed = 3
    SccPushPullNonFinitePullUnfringed = 4
    SccPushPullNonFiniteCalculation = 5
end

@doc "The fringed positive-probe image contains a non-finite pixel."
SccPushPullNonFinitePushFringed
@doc "The unfringed positive-probe image contains a non-finite pixel."
SccPushPullNonFinitePushUnfringed
@doc "The fringed negative-probe image contains a non-finite pixel."
SccPushPullNonFinitePullFringed
@doc "The unfringed negative-probe image contains a non-finite pixel."
SccPushPullNonFinitePullUnfringed
@doc "An SCC four-image double difference is non-finite."
SccPushPullNonFiniteCalculation

function _check_scc_push_pull_storage(
    response::AbstractVector,
    workspace::SccPushPullResponseWorkspace,
    plan::SccPushPullResponsePlan,
    push_fringed::AbstractVector,
    push_unfringed::AbstractVector,
    pull_fringed::AbstractVector,
    pull_unfringed::AbstractVector,
)
    Base.require_one_based_indexing(
        response,
        push_fringed,
        push_unfringed,
        pull_fringed,
        pull_unfringed,
    )
    expected = plan.pixel_count
    length(push_fringed) == expected || throw(
        DimensionMismatch("SCC push-fringed image extent is invalid"),
    )
    length(push_unfringed) == expected || throw(
        DimensionMismatch("SCC push-unfringed image extent is invalid"),
    )
    length(pull_fringed) == expected || throw(
        DimensionMismatch("SCC pull-fringed image extent is invalid"),
    )
    length(pull_unfringed) == expected || throw(
        DimensionMismatch("SCC pull-unfringed image extent is invalid"),
    )
    length(response) == expected || throw(
        DimensionMismatch("SCC push-pull response extent is invalid"),
    )
    length(workspace.staged_response) == expected || throw(
        DimensionMismatch("SCC push-pull Workspace extent is invalid"),
    )
    arrays = (
        response,
        push_fringed,
        push_unfringed,
        pull_fringed,
        pull_unfringed,
    )
    any(array -> Base.mightalias(array, workspace.staged_response), arrays) &&
        throw(ArgumentError("SCC arrays must not alias Workspace storage"))
    return nothing
end

"""
    process!(response, workspace, plan, push_fringed, push_unfringed,
             pull_fringed, pull_unfringed)

Compute one complete SCC four-image response into caller-owned storage. Exact
shape, one-based axes, and Workspace ownership are checked before mutation.
Inputs and output may be ordinary CPU vectors or one-based vector views, and
the output may safely alias an input because the result is staged first.
Expected non-finite input or arithmetic rejection returns a
[`SccPushPullResponseFailure`](@ref) and preserves the response.
"""
function process!(
    response::AbstractVector{T},
    workspace::SccPushPullResponseWorkspace{T},
    plan::SccPushPullResponsePlan{T},
    push_fringed::AbstractVector{T},
    push_unfringed::AbstractVector{T},
    pull_fringed::AbstractVector{T},
    pull_unfringed::AbstractVector{T},
) where {T<:AbstractFloat}
    _check_scc_push_pull_storage(
        response,
        workspace,
        plan,
        push_fringed,
        push_unfringed,
        pull_fringed,
        pull_unfringed,
    )

    for value in push_fringed
        isfinite(value) || return SccPushPullNonFinitePushFringed
    end
    for value in push_unfringed
        isfinite(value) || return SccPushPullNonFinitePushUnfringed
    end
    for value in pull_fringed
        isfinite(value) || return SccPushPullNonFinitePullFringed
    end
    for value in pull_unfringed
        isfinite(value) || return SccPushPullNonFinitePullUnfringed
    end

    @inbounds for index in eachindex(workspace.staged_response)
        push_difference = push_fringed[index] - push_unfringed[index]
        pull_difference = pull_fringed[index] - pull_unfringed[index]
        value = push_difference - pull_difference
        (
            isfinite(push_difference) &&
            isfinite(pull_difference) &&
            isfinite(value)
        ) || return SccPushPullNonFiniteCalculation
        workspace.staged_response[index] = value
    end
    copyto!(response, workspace.staged_response)
    return nothing
end

"""
Float32 Thompson four-image SCC response. Four complete calibrated images are
associated by named Ports and one response is published only after every
double difference is finite.
"""
@algorithm SccPushPullResponseF32 begin
    label = "scc-push-pull-response-f32"
    failure = SccPushPullResponseFailure

    """Configuration for [`SccPushPullResponseF32`](@ref)."""
    @configuration SccPushPullResponseF32Configuration begin
        pixel_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :push_fringed,
            schema_element_type(CALIBRATED_PIXELS_V1),
            (configuration.pixel_count,),
            CALIBRATED_PIXELS_V1;
            name="push-fringed",
        ),
        input(
            :push_unfringed,
            schema_element_type(CALIBRATED_PIXELS_V1),
            (configuration.pixel_count,),
            CALIBRATED_PIXELS_V1;
            name="push-unfringed",
        ),
        input(
            :pull_fringed,
            schema_element_type(CALIBRATED_PIXELS_V1),
            (configuration.pixel_count,),
            CALIBRATED_PIXELS_V1;
            name="pull-fringed",
        ),
        input(
            :pull_unfringed,
            schema_element_type(CALIBRATED_PIXELS_V1),
            (configuration.pixel_count,),
            CALIBRATED_PIXELS_V1;
            name="pull-unfringed",
        ),
        output(
            :response,
            schema_element_type(SCC_PUSH_PULL_RESPONSE_V1),
            (configuration.pixel_count,),
            SCC_PUSH_PULL_RESPONSE_V1;
            metadata_from=:push_fringed,
        ),
    )

    prepare(configuration) = begin
        plan = SccPushPullResponsePlan(Float32, configuration.pixel_count)
        (plan=plan, workspace=SccPushPullResponseWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(
            response,
            workspace,
            plan,
            push_fringed,
            push_unfringed,
            pull_fringed,
            pull_unfringed,
        )
    end
end
