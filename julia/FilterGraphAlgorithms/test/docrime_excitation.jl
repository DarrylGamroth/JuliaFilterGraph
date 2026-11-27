module DocrimeExcitationTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms:
    commit!, discard!, process!, reset!, splitmix64_advance

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "docrime_excitation.toml"),
)
const SEED = UInt64(FIXTURE["seed"])

struct WrappedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::WrappedVector) = size(vector.values)
Base.axes(vector::WrappedVector) = axes(vector.values)
Base.IndexStyle(::Type{<:WrappedVector}) = IndexLinear()
Base.getindex(vector::WrappedVector, index::Int) = vector.values[index]
Base.setindex!(vector::WrappedVector, value, index::Int) =
    setindex!(vector.values, value, index)

struct ZeroBasedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::ZeroBasedVector) = size(vector.values)
Base.axes(vector::ZeroBasedVector) = (0:(length(vector.values) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedVector, index::Int) = vector.values[index + 1]
Base.setindex!(vector::ZeroBasedVector, value, index::Int) =
    setindex!(vector.values, value, index + 1)

function allocation_bytes(output, workspace, plan)
    process!(output, workspace, plan)
    commit!(workspace)
    return @allocated begin
        process!(output, workspace, plan)
        commit!(workspace)
    end
end

function declared_allocation_bytes(output, prepared)
    process!(output, prepared)
    return @allocated process!(output, prepared)
end

@testset "continuous excitation matches the frozen FGN sequence" begin
    amplitudes = Float64.(FIXTURE["amplitudes"])
    plan = DocrimeExcitationPlan(amplitudes, SEED)
    workspace = DocrimeExcitationWorkspace(plan)
    output = zeros(Float64, length(amplitudes))

    for expected in FIXTURE["continuous"]
        @test process!(output, workspace, plan) === nothing
        @test output == Float64.(expected)
        commit!(workspace)
    end
end

@testset "binary excitation matches the frozen FGN sequence" begin
    amplitudes = Float64.(FIXTURE["amplitudes"])
    plan = DocrimeBinaryExcitationPlan(amplitudes, SEED)
    workspace = DocrimeExcitationWorkspace(plan)
    output = zeros(Float64, length(amplitudes))

    for expected in FIXTURE["binary"]
        @test process!(output, workspace, plan) === nothing
        @test output == Float64.(expected)
        commit!(workspace)
    end
end

@testset "Float32 and Float64 share unit draws and binary signs" begin
    continuous32 = DocrimeExcitationPlan(Float32[0.25, 2.0], SEED)
    continuous64 = DocrimeExcitationPlan(Float64[0.25, 2.0], SEED)
    binary32 = DocrimeBinaryExcitationPlan(Float32[0.25, 2.0], SEED)
    binary64 = DocrimeBinaryExcitationPlan(Float64[0.25, 2.0], SEED)
    plans = (continuous32, continuous64, binary32, binary64)
    workspaces = map(
        DocrimeExcitationWorkspace,
        plans,
    )
    outputs = (
        zeros(Float32, 2),
        zeros(Float64, 2),
        zeros(Float32, 2),
        zeros(Float64, 2),
    )

    for _ in 1:256
        for index in eachindex(workspaces)
            process!(outputs[index], workspaces[index], plans[index])
            commit!(workspaces[index])
        end
        @test Float64.(outputs[1]) == outputs[2]
        @test Float64.(outputs[3]) == outputs[4]
        @test abs(outputs[1][1]) < 0.25f0
        @test abs(outputs[1][2]) < 2.0f0
        @test abs(outputs[3][1]) == 0.25f0
        @test abs(outputs[3][2]) == 2.0f0
    end
end

@testset "candidate random state commits and discards transactionally" begin
    plan = DocrimeExcitationPlan(Float32[1, 2], SEED)
    workspace = DocrimeExcitationWorkspace(plan)
    first = zeros(Float32, 2)
    repeated = similar(first)

    process!(first, workspace, plan)
    @test docrime_excitation_state(workspace) == SEED
    process!(repeated, workspace, plan)
    @test repeated == first
    @test docrime_excitation_state(workspace) == SEED

    discard!(workspace)
    process!(repeated, workspace, plan)
    @test repeated == first
    commit!(workspace)
    @test docrime_excitation_state(workspace) ==
          splitmix64_advance(SEED, UInt64(2))

    second = copy(repeated)
    process!(second, workspace, plan)
    @test second != first
    discard!(workspace)
    process!(repeated, workspace, plan)
    @test repeated == second

    reset!(workspace)
    @test docrime_excitation_state(workspace) == SEED
    process!(repeated, workspace, plan)
    @test repeated == first
end

@testset "preparation validates amplitudes, seed, axes, and extent" begin
    for Plan in (DocrimeExcitationPlan, DocrimeBinaryExcitationPlan)
        @test_throws ArgumentError Plan(Float32[], 0)
        for amplitude in (0.0f0, -1.0f0, Float32(NaN), Float32(Inf))
            @test_throws ArgumentError Plan(Float32[1, amplitude], 0)
        end
        @test_throws ArgumentError Plan(Float32[1], -1)
        plan = Plan(Float32[1, 2], SEED)
        workspace = DocrimeExcitationWorkspace(plan)
        @test_throws DimensionMismatch process!(zeros(Float32, 1), workspace, plan)
        @test_throws ArgumentError process!(
            ZeroBasedVector(zeros(Float32, 2)),
            workspace,
            plan,
        )
        @test_throws MethodError prepare_docrime_excitation_plan(nothing, plan)
    end
end

@testset "host vector wrappers and admitted calls allocate no heap" begin
    for Plan in (DocrimeExcitationPlan, DocrimeBinaryExcitationPlan)
        plan = Plan(Float32[0.25, 2.0], SEED)
        workspace = DocrimeExcitationWorkspace(plan)
        storage = zeros(Float32, 2)
        output = WrappedVector(storage)
        @test process!(output, workspace, plan) === nothing
        @test all(isfinite, storage)
        commit!(workspace)
        @test allocation_bytes(output, workspace, plan) == 0
    end
end

@testset "declarations preserve FGN Ports, replacement, and reset" begin
    cases = (
        (
            declaration=DocrimeExcitationF32,
            configuration=DocrimeExcitationF32Configuration,
            Plan=DocrimeExcitationPlan,
            label="docrime-excitation-f32",
        ),
        (
            declaration=DocrimeBinaryExcitationF32,
            configuration=DocrimeBinaryExcitationF32Configuration,
            Plan=DocrimeBinaryExcitationPlan,
            label="docrime-binary-excitation-f32",
        ),
    )

    for case in cases
        @test algorithm_label(case.declaration) == case.label
        @test FilterGraphAlgorithms.algorithm_failure_type(case.declaration) ===
              DocrimeExcitationFailure
        configuration = algorithm_configuration(case.declaration)
        @test map(field -> field.name, configuration) == (:amplitudes, :seed)
        @test configuration[1].update isa GraphRebuildConfiguration
        @test configuration[2].update isa ConstructionConfiguration

        ports = algorithm_ports(case.declaration)
        @test map(port -> port.name, ports) == (:excitation, :amplitudes)
        @test ports[1].direction isa FilterGraphAlgorithms.OutputPort
        @test ports[1].role isa FilterGraphAlgorithms.FrameData
        @test ports[2].direction isa FilterGraphAlgorithms.InputPort
        @test ports[2].role isa FilterGraphAlgorithms.SparseParameter

        prepared = prepare_algorithm(
            case.declaration,
            case.configuration(amplitudes=Float32[1, 2], seed=SEED),
        )
        formats = algorithm_formats(prepared)
        @test map(format -> format.shape, formats) == ((2,), (2,))
        @test map(format -> format.schema, formats) == (
            DOCRIME_EXCITATION_V1,
            DOCRIME_EXCITATION_AMPLITUDE_V1,
        )
        output = zeros(Float32, 2)
        @test process!(output, prepared) === nothing
        first = copy(output)
        @test docrime_excitation_state(prepared.workspace) ==
              splitmix64_advance(SEED, UInt64(2))

        @test replace_parameter!(prepared, :amplitudes, Float32[0.5, 1.0]) ===
              prepared
        prior_state = docrime_excitation_state(prepared.workspace)
        @test process!(output, prepared) === nothing
        @test docrime_excitation_state(prepared.workspace) ==
              splitmix64_advance(prior_state, UInt64(2))
        @test docrime_excitation_amplitudes(prepared.plan) == Float32[0.5, 1.0]

        reset!(prepared)
        @test docrime_excitation_state(prepared.workspace) == SEED
        process!(output, prepared)
        @test output == first .* Float32[0.5, 0.5]
        @test declared_allocation_bytes(output, prepared) == 0
    end
end

end
