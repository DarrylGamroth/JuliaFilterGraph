module PreparationValidationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct GainPlan
    gain::Float32
end

mutable struct ScratchWorkspace
    commits::Int
    discards::Int
    resets::Int
end
ScratchWorkspace() = ScratchWorkspace(0, 0, 0)

function commit!(workspace::ScratchWorkspace)
    workspace.commits += 1
    return workspace
end
function discard!(workspace::ScratchWorkspace)
    workspace.discards += 1
    return workspace
end
function reset!(workspace::ScratchWorkspace)
    workspace.resets += 1
    return workspace
end

struct IncompleteWorkspace{missing} end
commit!(workspace::IncompleteWorkspace{:discard}) = workspace
commit!(workspace::IncompleteWorkspace{:reset}) = workspace
discard!(workspace::IncompleteWorkspace{:commit}) = workspace
discard!(workspace::IncompleteWorkspace{:reset}) = workspace
reset!(workspace::IncompleteWorkspace{:commit}) = workspace
reset!(workspace::IncompleteWorkspace{:discard}) = workspace

function preparation(variant, gain)
    workspace = ScratchWorkspace()
    variant === :ordered && return (plan=gain, workspace=workspace)
    variant === :reversed && return (workspace=workspace, plan=gain)
    variant === :missing_plan && return (workspace=workspace,)
    variant === :missing_workspace && return (plan=gain,)
    variant === :wrong_names && return (gain=gain, scratch=workspace)
    variant === :extra && return (plan=gain, workspace=workspace, extra=true)
    variant === :unnamed && return (gain, workspace)
    variant === :scalar && return gain
    variant === :struct && return GainPlan(gain)
    variant === :named && return (gain=gain,)
    variant === :missing_commit && return (plan=gain, workspace=IncompleteWorkspace{:commit}())
    variant === :missing_discard && return (plan=gain, workspace=IncompleteWorkspace{:discard}())
    variant === :missing_reset && return (plan=gain, workspace=IncompleteWorkspace{:reset}())
    error("unknown test variant")
end

gain(plan::Float32) = plan
gain(plan::GainPlan) = plan.gain
gain(plan::NamedTuple{(:gain,)}) = plan.gain
gain(plan::NamedTuple{(:plan,)}) = plan.plan
gain(plan::NamedTuple{(:workspace, :plan)}) = plan.plan
gain(plan::NamedTuple{(:plan, :workspace, :extra)}) = plan.plan

@algorithm WorkspaceGain begin
    label = "workspace-gain-test"
    @configuration WorkspaceGainConfiguration begin
        variant::Symbol => construction
        gain::Float32 => construction
    end
    ports(config) = (
        input(:source, Float32, (1,), "test.samples.f32/1"),
        output(:sink, Float32, (1,), "test.samples.f32/1"),
    )
    prepare(config) = preparation(config.variant, config.gain)
    process!(plan, workspace) = begin
        sink .= source .* plan
        nothing
    end
end

@algorithm StatelessGain begin
    label = "stateless-gain-test"
    @configuration StatelessGainConfiguration begin
        variant::Symbol => construction
        gain::Float32 => construction
    end
    ports(config) = (
        input(:source, Float32, (1,), "test.samples.f32/1"),
        output(:sink, Float32, (1,), "test.samples.f32/1"),
    )
    prepare(config) = preparation(config.variant, config.gain)
    process!(plan) = begin
        sink .= source .* gain(plan)
        nothing
    end
end

prepare_workspace(variant) = prepare_algorithm(
    WorkspaceGain,
    WorkspaceGainConfiguration(variant=variant, gain=2f0),
)
prepare_stateless(variant) = prepare_algorithm(
    StatelessGain,
    StatelessGainConfiguration(variant=variant, gain=2f0),
)

@testset "stateful preparation validates named fields before processing" begin
    for variant in (:ordered, :reversed)
        prepared = prepare_workspace(variant)
        @test prepared.plan === 2f0
        @test prepared.workspace isa ScratchWorkspace
        @test (prepared.workspace.commits, prepared.workspace.discards,
               prepared.workspace.resets) == (0, 0, 0)
        output = zeros(Float32, 1)
        @test process!(output, prepared, Float32[3]) === nothing
        @test output == Float32[6]
        @test prepared.workspace.commits == 1
        @test reset!(prepared) === prepared
        @test prepared.workspace.resets == 1
    end
    for variant in (:missing_plan, :missing_workspace, :wrong_names, :extra, :unnamed, :scalar)
        @test_throws r"WorkspaceGain.*exactly.*plan and workspace" prepare_workspace(variant)
    end
end

@testset "one-argument bodies retain ordinary Plans and existing bundles" begin
    for variant in (:scalar, :struct, :named, :missing_workspace, :reversed, :extra, :ordered)
        prepared = prepare_stateless(variant)
        if variant === :ordered
            @test prepared.plan === 2f0
            @test prepared.workspace isa ScratchWorkspace
        else
            @test prepared.workspace isa FilterGraphAlgorithms.EmptyWorkspace
            if variant in (:named, :missing_workspace, :reversed, :extra)
                @test prepared.plan isa NamedTuple
            end
        end
        output = zeros(Float32, 1)
        @test process!(output, prepared, Float32[3]) === nothing
        @test output == Float32[6]
        @test reset!(prepared) === prepared
    end
end

@testset "missing Workspace lifecycle hooks fail during preparation" begin
    for (variant, hook) in ((:missing_commit, "commit!"),
                            (:missing_discard, "discard!"),
                            (:missing_reset, "reset!"))
        for prepare in (prepare_workspace, prepare_stateless)
            @test_throws Regex("must implement $hook") prepare(variant)
        end
    end
end

end
