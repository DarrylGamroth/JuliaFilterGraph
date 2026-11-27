module ThinPlateVdmCompletionTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

function prepare(plan, calibration)
    output = VdmCompletionArtifacts(plan, calibration)
    workspace = VdmCompletionWorkspace(plan, calibration)
    process!(output, workspace, plan, calibration)
    return output
end

function complete(output, partition, controlled)
    full = copy(output.full_vdm_offset)
    full[controlled_vdm_indices(partition)] .= controlled
    full[extrapolated_vdm_indices(partition)] .+=
        output.extrapolated_from_controlled * controlled
    return full
end

@testset "VDM actuator partitions are exhaustive and canonical" begin
    roles = VdmActuatorRole[
        ExtrapolatedVdmActuator,
        ControlledVdmActuator,
        FixedVdmActuator,
        ControlledVdmActuator,
    ]
    partition = VdmActuatorPartition(@view roles[1:4])
    roles[1] = ControlledVdmActuator

    @test full_vdm_count(partition) == 4
    @test vdm_actuator_roles(partition) == VdmActuatorRole[
        ExtrapolatedVdmActuator,
        ControlledVdmActuator,
        FixedVdmActuator,
        ControlledVdmActuator,
    ]
    @test controlled_vdm_indices(partition) == [2, 4]
    @test extrapolated_vdm_indices(partition) == [1]
    @test fixed_vdm_indices(partition) == [3]
    returned_indices = controlled_vdm_indices(partition)
    returned_indices[1] = 1
    @test controlled_vdm_indices(partition) == [2, 4]
    @test_throws ArgumentError VdmActuatorPartition(VdmActuatorRole[])
    @test_throws ArgumentError VdmActuatorPartition(
        fill(ExtrapolatedVdmActuator, 3),
    )
end

@testset "thin-plate completion reproduces piston and tilt" begin
    partition = VdmActuatorPartition(
        VdmActuatorRole[
            ControlledVdmActuator,
            ControlledVdmActuator,
            ControlledVdmActuator,
            ControlledVdmActuator,
            ExtrapolatedVdmActuator,
            ExtrapolatedVdmActuator,
        ],
    )
    positions = [
        0.0 0.0
        1.0 0.0
        0.0 1.0
        1.0 1.0
        2.0 0.5
        -1.0 0.25
    ]
    calibration = VdmCompletionCalibration(partition, positions)
    plan = ThinPlateVdmCompletionPlan()
    output = VdmCompletionArtifacts(plan, calibration)
    workspace = VdmCompletionWorkspace(plan, calibration)
    weight_storage = output.extrapolated_from_controlled
    offset_storage = output.full_vdm_offset

    @test process!(output, workspace, plan, calibration) === output
    @test output.extrapolated_from_controlled === weight_storage
    @test output.full_vdm_offset === offset_storage
    @test sizeof(workspace) == 0
    @test size(output.extrapolated_from_controlled) == (2, 4)
    @test iszero(output.full_vdm_offset)

    expected = 2 .+ 3 .* positions[:, 1] .- 4 .* positions[:, 2]
    completed = complete(
        output,
        partition,
        expected[controlled_vdm_indices(partition)],
    )
    @test completed ≈ expected atol = 1.0e-12
    @test output.diagnostics.solver_relative_residual <= sqrt(eps())
    @test output.diagnostics.maximum_affine_reproduction_error <=
          sqrt(eps())
    @test isfinite(output.diagnostics.maximum_absolute_support_weight)
    @test input_progress(plan) == NoProgress()
    @test output_progress(plan) == NoProgress()

    regularized = prepare(
        ThinPlateVdmCompletionPlan(regularization=1.0e-3),
        calibration,
    )
    @test complete(
        regularized,
        partition,
        expected[controlled_vdm_indices(partition)],
    ) ≈ expected atol = 1.0e-12
end

@testset "fixed coordinates contribute an explicit full-VDM offset" begin
    partition = VdmActuatorPartition(
        VdmActuatorRole[
            ControlledVdmActuator,
            FixedVdmActuator,
            ControlledVdmActuator,
            ExtrapolatedVdmActuator,
        ],
    )
    positions = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    expected = 1 .+ 2 .* positions[:, 1] .+ 3 .* positions[:, 2]
    calibration = VdmCompletionCalibration(
        partition,
        positions,
        expected[fixed_vdm_indices(partition)],
    )

    output = prepare(ThinPlateVdmCompletionPlan(), calibration)
    completed = complete(
        output,
        partition,
        expected[controlled_vdm_indices(partition)],
    )

    @test completed ≈ expected atol = 1.0e-12
    @test output.full_vdm_offset[fixed_vdm_indices(partition)] ==
          expected[fixed_vdm_indices(partition)]
    @test !iszero(
        output.full_vdm_offset[only(extrapolated_vdm_indices(partition))],
    )
end

@testset "completion preserves generic floating representations" begin
    roles = VdmActuatorRole[
        ControlledVdmActuator,
        ControlledVdmActuator,
        ControlledVdmActuator,
        ControlledVdmActuator,
        ExtrapolatedVdmActuator,
    ]
    partition = VdmActuatorPartition(roles)
    for T in (Float16, Float32, Float64, BigFloat)
        positions = T[0 0; 1 0; 0 1; 1 1; 2 0.5]
        calibration = VdmCompletionCalibration(partition, positions)
        plan = ThinPlateVdmCompletionPlan{T}()

        output = prepare(plan, calibration)
        expected = T(2) .+ T(3) .* positions[:, 1] .- T(4) .* positions[:, 2]
        completed = complete(
            output,
            partition,
            expected[controlled_vdm_indices(partition)],
        )

        @test output.extrapolated_from_controlled isa Matrix{T}
        @test output.full_vdm_offset isa Vector{T}
        @test output.diagnostics isa VdmCompletionDiagnostics{T}
        @test completed ≈ expected rtol = 32eps(T) atol = 32eps(T)
    end
end

@testset "noncontiguous calibration views retain logical coordinate order" begin
    partition = VdmActuatorPartition(
        VdmActuatorRole[
            ControlledVdmActuator,
            FixedVdmActuator,
            ControlledVdmActuator,
            ExtrapolatedVdmActuator,
        ],
    )
    positions = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    position_storage = fill(NaN, 8, 4)
    position_view = @view position_storage[1:2:7, 1:2:3]
    position_view .= positions
    fixed_storage = fill(NaN, 3)
    fixed_view = @view fixed_storage[2:2]
    fixed_view .= 3.0
    calibration = VdmCompletionCalibration(
        partition,
        position_view,
        fixed_view,
    )

    output = prepare(ThinPlateVdmCompletionPlan(), calibration)
    expected = 1 .+ 2 .* positions[:, 1]
    controlled = expected[controlled_vdm_indices(partition)]

    @test complete(output, partition, controlled) ≈ expected atol = 1.0e-12
    @test axes(output.extrapolated_from_controlled) ==
          (Base.OneTo(1), Base.OneTo(2))
end

@testset "no extrapolated coordinates require no geometric solve" begin
    partition = VdmActuatorPartition(
        VdmActuatorRole[ControlledVdmActuator, FixedVdmActuator],
    )
    calibration = VdmCompletionCalibration(
        partition,
        [0.0 0.0; 1.0 0.0],
        [4.0],
    )

    output = prepare(ThinPlateVdmCompletionPlan(), calibration)

    @test size(output.extrapolated_from_controlled) == (0, 1)
    @test output.full_vdm_offset == [0.0, 4.0]
    @test iszero(output.diagnostics.solver_relative_residual)
end

@testset "invalid preparation preserves caller-owned artifacts" begin
    for arguments in ((-1.0, 1.0e-6), (NaN, 1.0e-6), (0.0, -1.0), (0.0, 1.0))
        @test_throws ArgumentError ThinPlateVdmCompletionPlan(
            regularization=arguments[1],
            validation_tolerance=arguments[2],
        )
    end

    partition = VdmActuatorPartition(
        VdmActuatorRole[
            ControlledVdmActuator,
            ControlledVdmActuator,
            ControlledVdmActuator,
            ExtrapolatedVdmActuator,
        ],
    )
    valid = VdmCompletionCalibration(
        partition,
        [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0],
    )
    plan = ThinPlateVdmCompletionPlan()
    output = VdmCompletionArtifacts(plan, valid)
    workspace = VdmCompletionWorkspace(plan, valid)
    fill!(output.extrapolated_from_controlled, 7.0)
    fill!(output.full_vdm_offset, 8.0)
    previous_diagnostics = output.diagnostics

    invalid_calibrations = (
        (VdmCompletionCalibration(partition, ones(3, 2)), DimensionMismatch),
        (
            VdmCompletionCalibration(
                partition,
                [0.0 0.0; 1.0 0.0; 0.0 1.0; NaN 1.0],
            ),
            ArgumentError,
        ),
        (
            VdmCompletionCalibration(
                partition,
                [0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0],
            ),
            ArgumentError,
        ),
        (
            VdmCompletionCalibration(
                partition,
                [0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0],
            ),
            ArgumentError,
        ),
    )
    for (invalid, exception) in invalid_calibrations
        @test_throws exception process!(output, workspace, plan, invalid)
        @test output.extrapolated_from_controlled == fill(7.0, 1, 3)
        @test output.full_vdm_offset == fill(8.0, 4)
        @test output.diagnostics === previous_diagnostics
    end

    fixed_partition = VdmActuatorPartition(
        VdmActuatorRole[
            ControlledVdmActuator,
            FixedVdmActuator,
            ControlledVdmActuator,
            ExtrapolatedVdmActuator,
        ],
    )
    wrong_fixed = VdmCompletionCalibration(
        fixed_partition,
        [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0],
        Float64[],
    )
    fixed_output = VdmCompletionArtifacts(plan, wrong_fixed)
    fixed_workspace = VdmCompletionWorkspace(plan, wrong_fixed)
    @test_throws DimensionMismatch process!(
        fixed_output,
        fixed_workspace,
        plan,
        wrong_fixed,
    )
end

@testset "VDM completion public API exports" begin
    for name in (
        :VdmActuatorRole,
        :ControlledVdmActuator,
        :ExtrapolatedVdmActuator,
        :FixedVdmActuator,
        :VdmActuatorPartition,
        :vdm_actuator_roles,
        :controlled_vdm_indices,
        :extrapolated_vdm_indices,
        :fixed_vdm_indices,
        :full_vdm_count,
        :VdmCompletionCalibration,
        :ThinPlateVdmCompletionPlan,
        :VdmCompletionDiagnostics,
        :VdmCompletionArtifacts,
        :VdmCompletionWorkspace,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
