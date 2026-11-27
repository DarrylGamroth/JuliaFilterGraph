struct AddOnePlan <: AbstractAlgorithmPlan end

function FilterGraphAlgorithms.process!(
    output,
    ::NoWorkspace,
    ::AddOnePlan,
    input,
)
    axes(output) == axes(input) ||
        throw(DimensionMismatch("input and output axes must match"))
    for index in eachindex(output, input)
        output[index] = input[index] + one(eltype(input))
    end
    return output
end

struct ProgressiveAddOnePlan{PublishProgress} <: AbstractAlgorithmPlan
    item_count::Int
end

mutable struct ProgressiveAddOneWorkspace
    active::Bool
    processed_items::Int
end

FilterGraphAlgorithms.input_progress(::Type{<:ProgressiveAddOnePlan}) =
    SupportsProgress()
FilterGraphAlgorithms.output_progress(
    ::Type{<:ProgressiveAddOnePlan{true}},
) = SupportsProgress()

function FilterGraphAlgorithms.start!(
    workspace::ProgressiveAddOneWorkspace,
    ::ProgressiveAddOnePlan,
)
    workspace.active && throw(ArgumentError("processing has already started"))
    workspace.active = true
    workspace.processed_items = 0
    return workspace
end

function FilterGraphAlgorithms.process!(
    output,
    workspace::ProgressiveAddOneWorkspace,
    plan::ProgressiveAddOnePlan{PublishProgress},
    input,
    workrange::AbstractUnitRange{Int},
) where {PublishProgress}
    workspace.active || throw(ArgumentError("processing has not started"))
    axes(output) == axes(input) ||
        throw(DimensionMismatch("input and output axes must match"))
    isempty(workrange) && throw(ArgumentError("workrange must be nonempty"))
    first(workrange) == workspace.processed_items + 1 ||
        throw(ArgumentError("workrange must be contiguous"))
    last(workrange) <= plan.item_count || throw(BoundsError(input, workrange))

    for index in workrange
        output[index] = input[index] + one(eltype(input))
    end
    workspace.processed_items = last(workrange)
    prefix_length = PublishProgress ? last(workrange) : 0
    return ProgressResult(prefix_length)
end

function FilterGraphAlgorithms.finish!(
    output,
    workspace::ProgressiveAddOneWorkspace,
    plan::ProgressiveAddOnePlan,
)
    workspace.active || throw(ArgumentError("processing has not started"))
    workspace.processed_items == plan.item_count ||
        throw(ArgumentError("not all work items have been processed"))
    workspace.active = false
    return output
end

function FilterGraphAlgorithms.abort!(
    workspace::ProgressiveAddOneWorkspace,
    ::ProgressiveAddOnePlan,
)
    workspace.active || throw(ArgumentError("processing has not started"))
    workspace.active = false
    workspace.processed_items = 0
    return workspace
end

struct StatefulSumPlan <: AbstractAlgorithmPlan
    item_count::Int
end

mutable struct StatefulSumWorkspace
    active::Bool
    processed_items::Int
    committed_sum::Float64
    in_progress_sum::Float64
end

FilterGraphAlgorithms.input_progress(::Type{StatefulSumPlan}) = SupportsProgress()

function FilterGraphAlgorithms.start!(
    workspace::StatefulSumWorkspace,
    ::StatefulSumPlan,
)
    workspace.active && throw(ArgumentError("processing has already started"))
    workspace.active = true
    workspace.processed_items = 0
    workspace.in_progress_sum = workspace.committed_sum
    return workspace
end

function FilterGraphAlgorithms.process!(
    output,
    workspace::StatefulSumWorkspace,
    plan::StatefulSumPlan,
    input,
    workrange::AbstractUnitRange{Int},
)
    workspace.active || throw(ArgumentError("processing has not started"))
    isempty(workrange) && throw(ArgumentError("workrange must be nonempty"))
    first(workrange) == workspace.processed_items + 1 ||
        throw(ArgumentError("workrange must be contiguous"))
    last(workrange) <= plan.item_count || throw(BoundsError(input, workrange))

    for index in workrange
        workspace.in_progress_sum += input[index]
    end
    workspace.processed_items = last(workrange)
    output[] = workspace.in_progress_sum
    return ProgressResult()
end

function FilterGraphAlgorithms.finish!(
    output,
    workspace::StatefulSumWorkspace,
    plan::StatefulSumPlan,
)
    workspace.active || throw(ArgumentError("processing has not started"))
    workspace.processed_items == plan.item_count ||
        throw(ArgumentError("not all work items have been processed"))
    workspace.committed_sum = workspace.in_progress_sum
    workspace.active = false
    return output
end

function FilterGraphAlgorithms.abort!(
    workspace::StatefulSumWorkspace,
    ::StatefulSumPlan,
)
    workspace.active || throw(ArgumentError("processing has not started"))
    workspace.in_progress_sum = workspace.committed_sum
    workspace.processed_items = 0
    workspace.active = false
    return workspace
end
