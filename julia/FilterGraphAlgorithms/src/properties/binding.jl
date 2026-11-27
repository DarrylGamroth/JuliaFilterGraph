"""A property transaction cannot be staged while another update is pending."""
struct PropertyUpdateBusy <: Exception end

"""No later property generation can be represented by the binding."""
struct PropertyGenerationExhausted <: Exception end

function Base.showerror(io::IO, ::PropertyUpdateBusy)
    return print(io, "a property update is already pending")
end

function Base.showerror(io::IO, ::PropertyGenerationExhausted)
    return print(io, "property generation is exhausted")
end

"""
    PropertyBinding(plan)

Own the active, staged, requested, and retired instances of one concrete plan
type. At most one update may be staged or requested. The prior active plan is
retained after adoption so resource destruction can be reclaimed off the
processing path.
"""
mutable struct PropertyBinding{P<:AbstractAlgorithmPlan}
    active::P
    staged::Union{Nothing,P}
    requested::Union{Nothing,P}
    retired::Union{Nothing,P}
    requested_generation::UInt64
    active_generation::UInt64
end

function PropertyBinding(plan::P) where {P<:AbstractAlgorithmPlan}
    return PropertyBinding{P}(plan, nothing, nothing, nothing, UInt64(0), UInt64(0))
end

"""
    active_plan(binding::PropertyBinding)

Return the plan currently selected for processing.
"""
active_plan(binding::PropertyBinding) = binding.active

"""
    requested_plan(binding::PropertyBinding)

Return the latest requested plan, or the active plan when none is pending.
"""
requested_plan(binding::PropertyBinding) = something(binding.requested, binding.active)

"""
    requested_generation(binding::PropertyBinding)

Return the latest generation published for process-boundary adoption.
"""
requested_generation(binding::PropertyBinding) = binding.requested_generation

"""
    active_generation(binding::PropertyBinding)

Return the generation currently selected for processing.
"""
active_generation(binding::PropertyBinding) = binding.active_generation

"""
    prepare_properties!(binding::PropertyBinding, assignments)

Prepare one replacement plan without publishing it for adoption.
"""
function prepare_properties!(binding::PropertyBinding, assignments)
    isempty(assignments) && return binding
    if !isnothing(binding.staged) || !isnothing(binding.requested)
        throw(PropertyUpdateBusy())
    end
    binding.requested_generation == typemax(UInt64) &&
        throw(PropertyGenerationExhausted())
    binding.retired = nothing
    binding.staged = prepare_properties(binding.active, assignments)
    return binding
end

"""
    commit_properties!(binding::PropertyBinding)

Publish a prepared replacement plan as the next requested generation.
"""
function commit_properties!(binding::PropertyBinding)
    isnothing(binding.staged) && return binding
    binding.requested = binding.staged
    binding.staged = nothing
    binding.requested_generation += UInt64(1)
    return binding
end

"""
    discard_properties!(binding::PropertyBinding)

Discard a prepared but unpublished replacement plan.
"""
function discard_properties!(binding::PropertyBinding)
    binding.staged = nothing
    return binding
end

"""
    adopt_properties!(binding) -> Bool

Adopt a whole requested generation at a caller-selected process boundary.
Returns `true` exactly when the active plan changed.
"""
function adopt_properties!(binding::PropertyBinding{P}) where {P}
    isnothing(binding.requested) && return false
    previous = binding.active
    binding.active = binding.requested::P
    binding.requested = nothing
    binding.retired = previous
    binding.active_generation = binding.requested_generation
    return true
end

"""
    reclaim_retired_properties!(binding::PropertyBinding)

Release a retired plan on the non-real-time control path.
"""
function reclaim_retired_properties!(binding::PropertyBinding)
    binding.retired = nothing
    return binding
end
