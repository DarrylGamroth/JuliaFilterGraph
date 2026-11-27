module VdmToPdmTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "vdm_to_pdm.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_parameters()
    configuration = FIXTURE["configuration"]
    parameters = FIXTURE["parameters"]
    active_to_full = row_major_matrix(
        Float32,
        parameters["active_to_full"],
        configuration["full_vdm_size"],
        configuration["active_vdm_size"],
    )
    vdm_to_pdm = row_major_matrix(
        Float32,
        parameters["vdm_to_pdm"],
        configuration["physical_actuator_extent"],
        configuration["full_vdm_size"],
    )
    return active_to_full, vdm_to_pdm
end

function fixture_prepared()
    configuration = FIXTURE["configuration"]
    prepared = prepare_algorithm(
        VdmToPdmF32,
        VdmToPdmF32Configuration(
            active_vdm_count=configuration["active_vdm_size"],
            full_vdm_count=configuration["full_vdm_size"],
            physical_actuator_count=configuration["physical_actuator_extent"],
        ),
    )
    active_to_full, vdm_to_pdm = fixture_parameters()
    replace_parameter!(prepared, "active-to-full", active_to_full)
    replace_parameter!(prepared, "vdm-to-pdm", vdm_to_pdm)
    return prepared
end

function vdm_to_pdm_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

# Literal names must specialize even when the declaration mixes vector and matrix formats.
command_format(prepared) = algorithm_format(prepared, Symbol("vdm-command"))
expansion_format(prepared) = algorithm_format(prepared, Symbol("active-to-full"))

@testset "VDM-to-PDM declaration preserves the FGN contract" begin
    @test algorithm_label(VdmToPdmF32) == "vdm-to-pdm-f32"
    @test algorithm_property_policy(VdmToPdmF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    @test map(field -> field.name, algorithm_configuration(VdmToPdmF32)) == (
        :active_vdm_count,
        :full_vdm_count,
        :physical_actuator_count,
    )

    ports = algorithm_ports(VdmToPdmF32)
    @test map(port -> port.name, ports) == (
        Symbol("vdm-command"),
        Symbol("requested-pdm-command"),
        Symbol("correction-role"),
        Symbol("active-to-full"),
        Symbol("vdm-to-pdm"),
    )
    @test ports[2].metadata_source == Symbol("vdm-command")
    @test ports[3].metadata_source == Symbol("vdm-command")
    @test ports[4].role isa FilterGraphAlgorithms.SparseParameter
    @test ports[5].role isa FilterGraphAlgorithms.SparseParameter

    prepared = fixture_prepared()
    formats = algorithm_formats(prepared)
    @test (@inferred command_format(prepared)) === formats[1]
    @test (@inferred expansion_format(prepared)) === formats[4]
    @test getproperty.(formats, :shape) == ((2,), (2,), (2,), (3, 2), (2, 3))
    @test getproperty.(formats, :schema) == (
        VDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
        PDM_CORRECTION_ROLE_V1,
        ACTIVE_TO_FULL_VDM_PROJECTION_V1,
        VDM_TO_PDM_PROJECTION_V1,
    )

    output = zeros(Float32, 2)
    for sample in FIXTURE["samples"]
        input = Float32.(sample["vdm_command"])
        @test process!(output, prepared, input) === nothing
        @test output ≈ Float32.(sample["requested_pdm_command"])
    end
end

@testset "VDM-to-PDM accepts logical CPU views and in-place commands" begin
    prepared = fixture_prepared()
    input_storage = fill(99.0f0, 4)
    output_storage = fill(71.0f0, 4)
    input = @view input_storage[1:2:3]
    output = @view output_storage[2:2:4]
    input .= Float32[0.75, -0.4]

    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.905, -0.6325]
    @test output_storage[1:2:3] == fill(71.0f0, 2)

    command = Float32[0.75, -0.4]
    @test process!(command, prepared, command) === nothing
    @test command ≈ Float32[0.905, -0.6325]

    active_to_full, vdm_to_pdm = fixture_parameters()
    active_storage = fill(NaN32, 5, 4)
    active_view = @view active_storage[1:2:5, 2:2:4]
    active_view .= active_to_full
    pdm_storage = fill(NaN32, 4, 5)
    pdm_view = @view pdm_storage[1:2:3, 1:2:5]
    pdm_view .= vdm_to_pdm
    @test replace_parameter!(prepared, "active-to-full", active_view) === prepared
    @test replace_parameter!(prepared, "vdm-to-pdm", pdm_view) === prepared
    input .= Float32[0.75, -0.4]
    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.905, -0.6325]
end

@testset "VDM-to-PDM replacement and rejection are failure-atomic" begin
    @test_throws ArgumentError VdmToPdmPlan(
        zeros(Float32, 0, 2),
        zeros(Float32, 2, 0),
    )
    @test_throws DimensionMismatch VdmToPdmPlan(
        zeros(Float32, 3, 2),
        zeros(Float32, 2, 2),
    )
    @test_throws ArgumentError prepare_algorithm(
        VdmToPdmF32,
        VdmToPdmF32Configuration(
            active_vdm_count=0,
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        VdmToPdmF32,
        VdmToPdmF32Configuration(
            active_vdm_count=typemax(Int),
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )

    prepared = fixture_prepared()
    original_plan = prepared.plan
    invalid_active_to_full = first(fixture_parameters())
    invalid_active_to_full[1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "active-to-full",
        invalid_active_to_full,
    )
    @test prepared.plan === original_plan
    invalid_vdm_to_pdm = last(fixture_parameters())
    invalid_vdm_to_pdm[1] = Inf32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "vdm-to-pdm",
        invalid_vdm_to_pdm,
    )
    @test prepared.plan === original_plan

    output = fill(71.0f0, 2)
    @test process!(output, prepared, Float32[NaN, 0]) ===
          VdmToPdmNonFiniteInput
    @test output == fill(71.0f0, 2)

    overflowing = prepare_algorithm(
        VdmToPdmF32,
        VdmToPdmF32Configuration(
            active_vdm_count=2,
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "active-to-full",
        fill(floatmax(Float32), 3, 2),
    )
    replace_parameter!(overflowing, "vdm-to-pdm", last(fixture_parameters()))
    @test process!(output, overflowing, Float32[2, 2]) ===
          VdmToPdmNonFiniteOutput
end

@testset "VDM-to-PDM warmed paths allocate no Julia heap memory" begin
    prepared = fixture_prepared()
    workspace = prepared.workspace
    output = zeros(Float32, 2)
    valid = Float32[0.75, -0.4]
    nonfinite = Float32[NaN, 0]

    process!(output, prepared, valid)
    @test vdm_to_pdm_allocation_bytes(output, prepared, valid) == 0
    process!(output, prepared, nonfinite)
    @test vdm_to_pdm_allocation_bytes(output, prepared, nonfinite) == 0
    @test workspace_commit_allocation_bytes(workspace) == 0
    @test workspace_discard_allocation_bytes(workspace) == 0
    @test workspace_reset_allocation_bytes(workspace) == 0

    overflowing = prepare_algorithm(
        VdmToPdmF32,
        VdmToPdmF32Configuration(
            active_vdm_count=2,
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "active-to-full",
        fill(floatmax(Float32), 3, 2),
    )
    replace_parameter!(overflowing, "vdm-to-pdm", last(fixture_parameters()))
    large = Float32[2, 2]
    process!(output, overflowing, large)
    @test vdm_to_pdm_allocation_bytes(output, overflowing, large) == 0
end

end

module PdmCorrectionRoleTests
using Test, FilterGraphAlgorithms, LinearAlgebra
import FilterGraphAlgorithms: process!
bytes(output, roles, prepared, input) = @allocated process!(output, roles, prepared, input)
@testset "Correction provenance follows prepared projection replacement" begin
    completion = Float32[1 0; 0 1; 0.5 0.5; 0 0; 1 0]
    projection = Matrix{Float32}(I, 5, 5)
    plan = VdmToPdmPlan(completion, projection)
    @test plan.correction_roles == UInt8[1,1,2,0,2]
    mixed = copy(projection); mixed[1,2] = 0.5
    @test VdmToPdmPlan(completion, mixed).correction_roles == UInt8[3,1,2,0,2]
    @test VdmToPdmPlan(Float32[1 0; 0.5 0.5], Matrix{Float32}(I,2,2)).correction_roles == UInt8[3,3]
    prepared = prepare_algorithm(VdmToPdmF32, VdmToPdmF32Configuration(
        active_vdm_count=2, full_vdm_count=5, physical_actuator_count=5))
    replace_parameter!(prepared, "active-to-full", completion)
    replace_parameter!(prepared, "vdm-to-pdm", projection)
    output, roles, input = zeros(Float32,5), zeros(UInt8,5), Float32[0.2,-0.1]
    @test process!(output, roles, prepared, input) === nothing
    @test roles == UInt8[1,1,2,0,2]
    bytes(output,roles,prepared,input)
    @test bytes(output,roles,prepared,input) == 0
    replace_parameter!(prepared, "vdm-to-pdm", mixed)
    @test process!(output, roles, prepared, input) === nothing
    @test roles == UInt8[3,1,2,0,2]
    roles .= 7
    @test process!(output, roles, prepared, Float32[NaN,0]) == VdmToPdmNonFiniteInput
    @test roles == fill(UInt8(7),5)
end
@testset "Correction-role output cannot overwrite prepared calibration or scratch" begin
    completion = Float32[1 0; 0 1; 0.5 0.5; 0 0; 1 0]
    projection = Matrix{Float32}(I, 5, 5)
    for storage in (p -> p.plan.active_to_full, p -> p.plan.vdm_to_pdm,
                    p -> p.workspace.full_vdm, p -> p.workspace.extrapolated_vdm)
        prepared = prepare_algorithm(VdmToPdmF32, VdmToPdmF32Configuration(
            active_vdm_count=2, full_vdm_count=5, physical_actuator_count=5))
        replace_parameter!(prepared, "active-to-full", completion)
        replace_parameter!(prepared, "vdm-to-pdm", projection)
        array = storage(prepared)
        before = copy(array)
        roles = view(reinterpret(UInt8, vec(array)), 1:5)
        output = fill(71.0f0, 5)
        @test_throws ArgumentError process!(output, roles, prepared, Float32[0.2,-0.1])
        @test array == before
        @test output == fill(71.0f0,5)
    end
end
@testset "Correction-role output cannot overwrite prepared execution metadata" begin
    completion = vcat(Matrix{Float32}(I,5,5),fill(0.2f0,2,5))
    projection = Matrix{Float32}(I,7,7)[[2,1,3,4,5,6,7],:]
    for storage in (p -> p.plan.active_to_full_execution.coefficients,
                    p -> p.plan.active_to_full_execution.controlled_rows,
                    p -> p.plan.active_to_full_execution.extrapolated_rows,
                    p -> p.plan.vdm_to_pdm_execution.columns)
        prepared = prepare_algorithm(VdmToPdmF32, VdmToPdmF32Configuration(
            active_vdm_count=5, full_vdm_count=7, physical_actuator_count=7))
        replace_parameter!(prepared,"active-to-full",completion)
        replace_parameter!(prepared,"vdm-to-pdm",projection)
        array = storage(prepared)
        before = copy(array)
        roles = view(reinterpret(UInt8,vec(array)),1:7)
        output = fill(71.0f0,7)
        @test_throws ArgumentError process!(output,roles,prepared,ones(Float32,5))
        @test array == before
        @test output == fill(71.0f0,7)
    end
end
end
