module PdmStrokeOptimizationTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: _process_admitted!, commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pdm_stroke_optimization.toml"),
)

struct ZeroBasedOptimizationVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedOptimizationVector) = size(vector.parent)
Base.axes(vector::ZeroBasedOptimizationVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedOptimizationVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedOptimizationVector, index::Int) =
    vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedOptimizationVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

function optimization_prepared(case)
    return prepare_algorithm(
        PdmStrokeOptimizationF32,
        PdmStrokeOptimizationF32Configuration(
            actuator_count=case["pdm_size"],
            null_mode_count=case["null_mode_count"],
            preferred_command=Float32.(case["preferred_command"]),
            command_to_null_modes=Float32.(case["command_to_null_modes"]),
            null_modes_to_command=Float32.(case["null_modes_to_command"]),
        ),
    )
end

function optimization_matrix(values, rows::Int, columns::Int)
    return permutedims(reshape(Float32.(values), columns, rows))
end

function optimization_checked_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function optimization_checked_allocation_bytes(output, workspace, plan, input)
    return @allocated FilterGraphAlgorithms._process_pdm_stroke_optimization!(
        output, workspace, plan, input,
    )
end

function optimization_admitted_allocation_bytes(output, prepared, input)
    return @allocated _process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

optimization_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
optimization_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
optimization_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "PDM stroke-optimization declaration preserves the FGN contract" begin
    @test algorithm_label(PdmStrokeOptimizationF32) ==
          "pdm-stroke-optimization-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PdmStrokeOptimizationF32,
    ) === PdmStrokeOptimizationFailure
    @test algorithm_property_policy(PdmStrokeOptimizationF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(PdmStrokeOptimizationF32)
    @test map(field -> field.name, configuration) == (
        :actuator_count,
        :null_mode_count,
        :preferred_command,
        :command_to_null_modes,
        :null_modes_to_command,
    )
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].update isa GraphRebuildConfiguration
    @test all(
        field -> field.update isa ConstructionConfiguration,
        configuration[3:5],
    )

    ports = algorithm_ports(PdmStrokeOptimizationF32)
    @test map(port -> port.name, ports) == (
        :requested,
        :optimized,
        Symbol("preferred-command"),
        Symbol("command-to-null-modes"),
        Symbol("null-modes-to-command"),
    )
    @test ports[2].metadata_source === :requested
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[3:5],
    )

    prepared = optimization_prepared(FIXTURE["cases"][4])
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == (
        (3,),
        (3,),
        (3,),
        (2, 3),
        (3, 2),
    )
    @test getproperty.(formats, :schema) == (
        REQUESTED_PDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
        PREFERRED_PDM_COMMAND_V1,
        PDM_COMMAND_TO_NULL_MODES_V1,
        PDM_NULL_MODES_TO_COMMAND_V1,
    )

end

@testset "PDM stroke optimization exactly matches the frozen Rust oracle" begin
    for case in FIXTURE["cases"]
        prepared = optimization_prepared(case)
        input = Float32.(case["requested"])
        output = fill(71.0f0, length(input))

        @test process!(output, prepared, input) === nothing
        @test output == Float32.(case["expected"])
        @test reinterpret(UInt32, output) == UInt32.(case["expected_bits"])
    end
end

@testset "independently calibrated maps preserve their matrix orientation" begin
    preferred = [0.0, 0.0]
    command_to_null = [0.25 0.75]
    null_to_command = reshape([2.0, 2.0 / 3.0], 2, 1)
    plan = PdmStrokeOptimizationPlan(
        preferred,
        command_to_null,
        null_to_command,
    )
    workspace = PdmStrokeOptimizationWorkspace(plan)
    output = zeros(2)
    @test FilterGraphAlgorithms._process_pdm_stroke_optimization!(
        output,
        workspace,
        plan,
        [4.0, 0.0],
    ) === nothing
    @test output[1] == 2.0
    @test output[2] == -2.0 / 3.0
end

@testset "PDM optimization preparation and parameters are failure-atomic" begin
    ranged_plan = FilterGraphAlgorithms._configured_pdm_stroke_optimization_plan(
        2,
        1,
        range(0.0, 1.0; length=2),
        range(0.25, 0.75; length=2),
        range(2.0, 2.0 / 3.0; length=2),
    )
    @test ranged_plan.preferred_command == [0.0, 1.0]
    @test ranged_plan.preferred_command isa Vector{Float64}

    @test_throws ArgumentError PdmStrokeOptimizationPlan(
        Float32[],
        zeros(Float32, 1, 0),
        zeros(Float32, 0, 1),
    )
    @test_throws ArgumentError PdmStrokeOptimizationPlan(
        zeros(Float32, 2),
        zeros(Float32, 0, 2),
        zeros(Float32, 2, 0),
    )
    @test_throws DimensionMismatch PdmStrokeOptimizationPlan(
        zeros(Float32, 2),
        zeros(Float32, 1, 1),
        zeros(Float32, 2, 1),
    )
    @test_throws DimensionMismatch PdmStrokeOptimizationPlan(
        zeros(Float32, 2),
        zeros(Float32, 1, 2),
        zeros(Float32, 1, 2),
    )
    for artifacts in (
        (Float32[NaN, 0], zeros(Float32, 1, 2), zeros(Float32, 2, 1)),
        (zeros(Float32, 2), Float32[NaN 0], zeros(Float32, 2, 1)),
        (zeros(Float32, 2), zeros(Float32, 1, 2), reshape(Float32[Inf, 0], 2, 1)),
    )
        @test_throws ArgumentError PdmStrokeOptimizationPlan(artifacts...)
    end

    case = FIXTURE["cases"][4]
    prepared = optimization_prepared(case)
    input = Float32.(case["requested"])
    output = zeros(Float32, 3)
    process!(output, prepared, input)

    original = prepared.plan
    preferred_storage = Float32[83, 0.5, 83, -0.25, 83, 0.75]
    preferred = @view preferred_storage[2:2:6]
    @test replace_parameter!(prepared, "preferred-command", preferred) === prepared
    @test prepared.plan !== original
    @test prepared.plan.command_to_null_modes ===
          original.command_to_null_modes
    @test prepared.plan.null_modes_to_command ===
          original.null_modes_to_command

    after_preferred = prepared.plan
    command_to_null = optimization_matrix(
        case["command_to_null_modes"],
        2,
        3,
    )
    @test replace_parameter!(
        prepared,
        "command-to-null-modes",
        command_to_null,
    ) === prepared
    @test prepared.plan.preferred_command === after_preferred.preferred_command
    @test prepared.plan.null_modes_to_command ===
          after_preferred.null_modes_to_command

    after_command_to_null = prepared.plan
    null_to_command = optimization_matrix(
        case["null_modes_to_command"],
        3,
        2,
    )
    @test replace_parameter!(
        prepared,
        "null-modes-to-command",
        null_to_command,
    ) === prepared
    @test prepared.plan.preferred_command ===
          after_command_to_null.preferred_command
    @test prepared.plan.command_to_null_modes ===
          after_command_to_null.command_to_null_modes

    active = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "preferred-command",
        zeros(Float32, 2),
    )
    invalid_map = copy(command_to_null)
    invalid_map[1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "command-to-null-modes",
        invalid_map,
    )
    @test prepared.plan === active
end

@testset "PDM optimization rejection preserves caller-owned output" begin
    case = FIXTURE["cases"][1]
    prepared = optimization_prepared(case)
    output = fill(71.0f0, 3)
    @test_throws DimensionMismatch process!(output, prepared, Float32[0, 0])
    @test output == fill(71.0f0, 3)
    @test_throws DimensionMismatch process!(zeros(Float32, 2), prepared, zeros(Float32, 3))
    @test output == fill(71.0f0, 3)

    @test process!(output, prepared, Float32[NaN, 0, 0]) ===
          PdmStrokeOptimizationNonFiniteInput
    @test output == fill(71.0f0, 3)

    overflow_plan = PdmStrokeOptimizationPlan(
        Float32[-floatmax(Float32)],
        ones(Float32, 1, 1),
        ones(Float32, 1, 1),
    )
    overflow_workspace = PdmStrokeOptimizationWorkspace(overflow_plan)
    overflow_output = Float32[71]
    @test FilterGraphAlgorithms._process_pdm_stroke_optimization!(
        overflow_output,
        overflow_workspace,
        overflow_plan,
        Float32[floatmax(Float32)],
    ) === PdmStrokeOptimizationNonFiniteOutput
    @test overflow_output == Float32[71]

    @test_throws ArgumentError process!(
        prepared.workspace.centered_command,
        prepared,
        zeros(Float32, 3),
    )
    @test_throws ArgumentError process!(
        output,
        prepared,
        prepared.workspace.centered_command,
    )
    zero_based = ZeroBasedOptimizationVector(zeros(Float32, 3))
    @test_throws ArgumentError process!(output, prepared, zero_based)
    @test_throws ArgumentError process!(zero_based, prepared, zeros(Float32, 3))
end

@testset "PDM final subtraction rejects without publication and recovers" begin
    extent = 277
    command_to_null_modes = zeros(Float32, 1, extent)
    command_to_null_modes[1, end] = 0.5f0
    null_modes_to_command = zeros(Float32, extent, 1)
    null_modes_to_command[end, 1] = -1f0
    plan = PdmStrokeOptimizationPlan(
        zeros(Float32, extent), command_to_null_modes, null_modes_to_command,
    )
    workspace = PdmStrokeOptimizationWorkspace(plan)
    output = fill(71f0, extent)
    requested = zeros(Float32, extent)
    requested[end] = floatmax(Float32)
    @test FilterGraphAlgorithms._process_pdm_stroke_optimization!(
        output, workspace, plan, requested,
    ) === PdmStrokeOptimizationNonFiniteOutput
    optimization_checked_allocation_bytes(output, workspace, plan, requested)
    @test optimization_checked_allocation_bytes(output, workspace, plan, requested) == 0
    @test all(isfinite, workspace.null_mode_coefficients)
    @test all(==(71f0), output)
    fill!(requested, 0f0)
    @test FilterGraphAlgorithms._process_pdm_stroke_optimization!(
        output, workspace, plan, requested,
    ) === nothing
    @test all(iszero, output)
end

@testset "PDM optimization supports strided and in-place commands" begin
    prepared = optimization_prepared(FIXTURE["cases"][1])
    input_storage = fill(83.0f0, 6)
    output_storage = fill(71.0f0, 6)
    requested = @view input_storage[1:2:5]
    optimized = @view output_storage[2:2:6]
    requested .= Float32[4, 1, -2]

    @test process!(optimized, prepared, requested) === nothing
    @test optimized == Float32[3, 0, -3]
    @test output_storage[1:2:5] == fill(71.0f0, 3)

    command = Float32[4, 1, -2]
    @test process!(command, prepared, command) === nothing
    @test command == Float32[3, 0, -3]
end

@testset "PDM optimization warmed paths allocate no Julia heap memory" begin
    prepared = optimization_prepared(FIXTURE["cases"][1])
    output = zeros(Float32, 3)
    valid = Float32[4, 1, -2]
    invalid = Float32[NaN, 0, 0]

    process!(output, prepared, valid)
    @test optimization_checked_allocation_bytes(output, prepared, valid) == 0
    process!(output, prepared, invalid)
    @test optimization_checked_allocation_bytes(output, prepared, invalid) == 0

    @test @inferred PdmStrokeOptimizationFailure _process_admitted!(
        output,
        prepared,
        valid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test optimization_admitted_allocation_bytes(output, prepared, valid) == 0
    _process_admitted!(
        output,
        prepared,
        invalid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test optimization_admitted_allocation_bytes(output, prepared, invalid) == 0

    workspace = prepared.workspace
    @test optimization_commit_allocation_bytes(workspace) == 0
    @test optimization_discard_allocation_bytes(workspace) == 0
    reset!(workspace)
    @test iszero(workspace.centered_command)
    @test iszero(workspace.null_mode_coefficients)
    @test optimization_reset_allocation_bytes(workspace) == 0
end

end
