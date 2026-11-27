"""Marker supertype for static progressive-processing support traits."""
abstract type ProgressSupport end

"""Trait value indicating that a prepared plan has no progressive support."""
struct NoProgress <: ProgressSupport end

"""Trait value indicating that a prepared plan supports progressive work."""
struct SupportsProgress <: ProgressSupport end

"""
    ProgressResult(length::Int=0)

Report the length of the greatest immutable output semantic-unit prefix.

A positive length `n` promises that output units `1:n` are complete and will
not be modified by later ranged calls in the same progressive execution. The
unit is declared by the concrete plan and need not be a scalar-array index.
This value does not represent transport bytes, sample validity, or terminal
publication. A negative length is direct API misuse.
"""
struct ProgressResult
    length::Int

    function ProgressResult(length::Int=0)
        length >= 0 || throw(ArgumentError("length must be nonnegative"))
        return new(length)
    end
end

Base.length(result::ProgressResult) = result.length
Base.isempty(result::ProgressResult) = iszero(length(result))

"""
    input_progress(::Type{P}) -> ProgressSupport
    input_progress(plan::AbstractAlgorithmPlan) -> ProgressSupport

Return whether a prepared plan type supports progressive input processing.

The default is [`NoProgress`](@ref). A plan opts in with a more-specific type
method returning [`SupportsProgress`](@ref) and implements [`start!`](@ref),
ranged [`process!`](@ref), [`finish!`](@ref), and [`abort!`](@ref). Capability
selected by a prepared option must be encoded in the concrete plan type or its
type parameters.
"""
input_progress(::Type{<:AbstractAlgorithmPlan}) = NoProgress()
input_progress(plan::AbstractAlgorithmPlan) = input_progress(typeof(plan))

"""
    output_progress(::Type{P}) -> ProgressSupport
    output_progress(plan::AbstractAlgorithmPlan) -> ProgressSupport

Return whether a prepared plan type exposes immutable progressive output.

The default is [`NoProgress`](@ref). This trait is independent of
[`input_progress`](@ref): a plan may consume input progressively while making
only its terminal output immutable. A supporting plan reports a monotonic
prefix in every ranged [`process!`](@ref) result.
"""
output_progress(::Type{<:AbstractAlgorithmPlan}) = NoProgress()
output_progress(plan::AbstractAlgorithmPlan) = output_progress(typeof(plan))

"""
    start!(workspace, plan) -> workspace

Initialize bounded in-progress state for one progressive execution without
changing previously committed mathematical history.

Sample identity, time, transport progress, lifecycle, and publication state
remain caller-owned.
"""
function start! end

"""
    finish!(output, workspace, plan) -> output

Finalize a complete progressive output and commit fully processed mathematical
history.

The authoritative caller owns the completion decision and terminal sample
publication.
"""
function finish! end

"""
    abort!(workspace, plan) -> workspace

Discard in-progress state while preserving mathematical history committed
before the matching [`start!`](@ref).

Caller-owned output may remain dirty and must not be published as a completed
sample.
"""
function abort! end
