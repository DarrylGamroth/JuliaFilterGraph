# Generated snapshot from examples/stateful-algorithm/run.jl.
# Regenerate with julia test/update_snapshots.jl examples/stateful-algorithm/run.jl.
module StatefulAlgorithmExample

import FilterGraphAlgorithms
using FilterGraphAlgorithms: @algorithm, prepare_algorithm
import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct AccumulatorPlan
    gain::Float32
    function AccumulatorPlan(gain::Float32)
        isfinite(gain) || throw(ArgumentError("gain must be finite"))
        return new(gain)
    end
end

mutable struct AccumulatorWorkspace
    committed::Vector{Float32}
    candidate::Vector{Float32}
end

"""Expected rejection of a nonfinite input or accumulated result."""
@enum AccumulatorFailure::UInt8 begin
    NonfiniteInput = 1
    NonfiniteOutput = 2
end

function commit!(workspace::AccumulatorWorkspace)
    workspace.committed, workspace.candidate = workspace.candidate, workspace.committed
    return workspace
end
function discard!(workspace::AccumulatorWorkspace)
    copyto!(workspace.candidate, workspace.committed)
    return workspace
end
function reset!(workspace::AccumulatorWorkspace)
    fill!(workspace.committed, 0.0f0)
    fill!(workspace.candidate, 0.0f0)
    return workspace
end

"""
    process!(output, workspace::AccumulatorWorkspace, plan::AccumulatorPlan, input)

Stage `next = committed + gain * input` and publish output only if every value
is finite. Return `nothing`, `NonfiniteInput`, or `NonfiniteOutput`. The low-level
caller must call `commit!` after success or `discard!` after rejection; the
prepared Algorithm wrapper does this automatically. Workspace storage is private.
"""
function process!(output, workspace::AccumulatorWorkspace, plan::AccumulatorPlan, input)
    Base.require_one_based_indexing(output, input)
    axes(output) == axes(input) == axes(workspace.committed) || throw(
        DimensionMismatch("input, output, and accumulator state axes must match"),
    )
    for buffer in (workspace.committed, workspace.candidate)
        (Base.mightalias(output, buffer) || Base.mightalias(input, buffer)) && throw(
            ArgumentError("input and output must not alias private Workspace storage"),
        )
    end
    all(isfinite, input) || return NonfiniteInput
    @inbounds @simd for index in eachindex(input, workspace.candidate)
        workspace.candidate[index] = workspace.committed[index] + plan.gain * input[index]
    end
    all(isfinite, workspace.candidate) || return NonfiniteOutput
    copyto!(output, workspace.candidate)
    return nothing
end

@algorithm AccumulatorF32 begin
    label = "example-accumulator-f32"
    failure = AccumulatorFailure

    @configuration AccumulatorConfiguration begin
        extent::Int => graph_rebuild
        gain::Float32 => construction
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "example.signal.f32/1"),
        output(:output, Float32, (configuration.extent,), "example.signal.f32/1"),
    )

    prepare(configuration) = (
        plan=AccumulatorPlan(configuration.gain),
        workspace=AccumulatorWorkspace(
            zeros(Float32, configuration.extent), zeros(Float32, configuration.extent),
        ),
    )

    process!(plan, workspace) = process!(output, workspace, plan, input)
end

"""
    prepare_accumulator(extent; gain=0.5f0)

Prepare a checked stateful Algorithm with caller-owned input and output arrays.
The gain is fixed at preparation; a plain Plan needs no abstract supertype.
"""
function prepare_accumulator(extent; gain=0.5f0)
    extent > 0 || throw(ArgumentError("extent must be positive"))
    return prepare_algorithm(AccumulatorF32, AccumulatorConfiguration(extent, gain))
end

"""
    run_example()

Accumulate two samples with a rejected sample between them. Copy borrowed state
to retain a snapshot across later commits, which swap and reuse Workspace arrays.
"""
function run_example()
    prepared = prepare_accumulator(2)
    output = zeros(Float32, 2)
    process!(output, prepared, Float32[2, 4]) === nothing || error("first sample rejected")
    first_state = copy(prepared.workspace.committed)
    rejection = process!(output, prepared, Float32[NaN, 1])
    rejection === NonfiniteInput || error("expected nonfinite input rejection")
    process!(output, prepared, Float32[4, 2]) === nothing || error("second sample rejected")
    return (; first_state, rejection, final_state=copy(prepared.workspace.committed))
end

function (@main)(args)
    result = run_example()
    println("First committed state: ", result.first_state)
    println("Rejected sample: ", result.rejection)
    println("Final committed state: ", result.final_state)
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .StatefulAlgorithmExample: main
end
