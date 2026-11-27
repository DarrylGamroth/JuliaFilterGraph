module SccReconstructorPreparationTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test

import FilterGraphAlgorithms: process!

function identity_calibration(response; candidate_mode_to_vdm=I)
    pixel_count, candidate_mode_count = size(response)
    vdm_projection = candidate_mode_to_vdm === I ?
                     Matrix{eltype(response)}(I, candidate_mode_count, candidate_mode_count) :
                     candidate_mode_to_vdm
    return SccReconstructorCalibration(
        response,
        vdm_projection,
        Matrix{eltype(response)}(I, pixel_count, pixel_count),
        Matrix{eltype(response)}(I, candidate_mode_count, candidate_mode_count),
        Matrix{eltype(response)}(I, candidate_mode_count, candidate_mode_count),
    )
end

@testset "full retained SCC factors reproduce the Tikhonov oracle" begin
    calibration = identity_calibration([4.0 0.0; 0.0 2.0])
    plan = SccReconstructorPreparationPlan(2, 1.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)
    pixel_to_controller_storage = output.pixel_to_controller
    controller_to_vdm_storage = output.controller_to_vdm
    fused_reconstructor_storage = output.fused_reconstructor
    singular_value_storage = output.singular_values
    gain_storage = output.tikhonov_gains

    @test process!(output, workspace, plan, calibration) === output
    @test output.pixel_to_controller === pixel_to_controller_storage
    @test output.controller_to_vdm === controller_to_vdm_storage
    @test output.fused_reconstructor === fused_reconstructor_storage
    @test output.singular_values === singular_value_storage
    @test output.tikhonov_gains === gain_storage
    @test sizeof(workspace) == 0
    expected = [4 / 17 0.0; 0.0 2 / 5]
    @test output.fused_reconstructor ≈ expected atol = 4eps(Float64)
    @test output.controller_to_vdm * output.pixel_to_controller ≈
          output.fused_reconstructor atol = 4eps(Float64)
    @test output.singular_values ≈ [4.0, 2.0]
    @test output.tikhonov_gains ≈ [4 / 17, 2 / 5]

    diagnostics = output.diagnostics
    @test diagnostics.numerical_rank == 2
    @test diagnostics.retained_rank == 2
    @test diagnostics.response_condition_number == 2.0
    @test diagnostics.retained_condition_number == 2.0
    @test diagnostics.whitened_response_residual == 0.0
    @test diagnostics.fused_reconstructor_residual == 0.0
    @test diagnostics.vdm_noise_amplification ≈
          hypot(4 / 17, 2 / 5)
    @test diagnostics.llowfs_authority_leakage == 0.0
    @test diagnostics.projector_symmetry_error == 0.0
    @test diagnostics.projector_idempotence_error == 0.0
end

@testset "noise and stroke weighting preserve the exact full-rank inverse" begin
    calibration = SccReconstructorCalibration(
        [2.0 0.0; 0.0 3.0],
        [1.0 0.0; 0.0 2.0],
        [4.0 0.0; 0.0 9.0],
        [4.0 0.0; 0.0 1.0],
        Matrix{Float64}(I, 2, 2),
    )
    plan = SccReconstructorPreparationPlan(2, 0.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)

    process!(output, workspace, plan, calibration)

    expected = [0.5 0.0; 0.0 2 / 3]
    @test output.singular_values ≈ [1.0, 0.5]
    @test output.fused_reconstructor ≈ expected atol = 8eps(Float64)
    @test output.controller_to_vdm * output.pixel_to_controller ≈
          expected atol = 8eps(Float64)
end

@testset "non-diagonal weighting matches the regularized normal equation" begin
    response = [1.0 2.0; -1.0 0.5; 0.25 -0.75]
    candidate_mode_to_vdm = [1.0 0.5; 0.0 2.0; -0.25 1.0]
    pixel_noise_covariance = [2.0 0.25 0.0; 0.25 1.5 0.1; 0.0 0.1 1.0]
    candidate_mode_stroke_metric = [3.0 0.4; 0.4 1.25]
    projector = Matrix{Float64}(I, 2, 2)
    calibration = SccReconstructorCalibration(
        response,
        candidate_mode_to_vdm,
        pixel_noise_covariance,
        candidate_mode_stroke_metric,
        projector,
    )
    tikhonov_scale = 0.3
    plan = SccReconstructorPreparationPlan(2, tikhonov_scale)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)

    process!(output, workspace, plan, calibration)

    noise_whitener =
        cholesky(Symmetric(pixel_noise_covariance)).L \
        Matrix{Float64}(I, 3, 3)
    stroke_inverse_sqrt =
        cholesky(Symmetric(candidate_mode_stroke_metric)).U \
        Matrix{Float64}(I, 2, 2)
    whitened_response =
        noise_whitener * response * stroke_inverse_sqrt
    expected =
        candidate_mode_to_vdm *
        stroke_inverse_sqrt *
        (
            (
                transpose(whitened_response) * whitened_response +
                tikhonov_scale^2 * I
            ) \
            transpose(whitened_response)
        ) *
        noise_whitener
    @test output.fused_reconstructor ≈ expected rtol = 2e-14
    @test output.controller_to_vdm * output.pixel_to_controller ≈
          expected rtol = 2e-14
end

@testset "the LLOWFS projector excludes its candidate-mode authority" begin
    response = Matrix{Float64}(I, 3, 3)
    projector = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    calibration = SccReconstructorCalibration(
        response,
        Matrix{Float64}(I, 3, 3),
        Matrix{Float64}(I, 3, 3),
        Matrix{Float64}(I, 3, 3),
        projector,
    )
    plan = SccReconstructorPreparationPlan(2, 0.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)

    process!(output, workspace, plan, calibration)

    @test output.fused_reconstructor ≈ projector atol = 8eps(Float64)
    @test output.controller_to_vdm * output.pixel_to_controller ≈
          projector atol = 8eps(Float64)
    @test output.diagnostics.numerical_rank == 2
    @test isinf(output.diagnostics.response_condition_number)
    @test output.diagnostics.retained_condition_number == 1.0
    @test output.diagnostics.llowfs_authority_leakage == 0.0
end

@testset "retained rank reports both response and fused residuals" begin
    calibration = identity_calibration([4.0 0.0; 0.0 2.0])
    plan = SccReconstructorPreparationPlan(1, 1.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)

    process!(output, workspace, plan, calibration)

    selected = output.controller_to_vdm * output.pixel_to_controller
    @test selected ≈ [4 / 17 0.0; 0.0 0.0] atol = 4eps(Float64)
    @test output.fused_reconstructor ≈
          [4 / 17 0.0; 0.0 2 / 5] atol = 4eps(Float64)
    @test output.diagnostics.whitened_response_residual ≈ 2 / hypot(4, 2)
    @test output.diagnostics.fused_reconstructor_residual ≈
          (2 / 5) / hypot(4 / 17, 2 / 5)
end

@testset "response-singular signs follow candidate-mode coordinates" begin
    calibration = identity_calibration([-4.0 0.0; 0.0 2.0])
    plan = SccReconstructorPreparationPlan(2, 1.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)

    process!(output, workspace, plan, calibration)

    @test output.controller_to_vdm ≈ Matrix{Float64}(I, 2, 2)
    @test output.fused_reconstructor ≈ [-4 / 17 0.0; 0.0 2 / 5]
end

@testset "preparation preserves the selected floating representation" begin
    for T in (Float16, Float32, Float64, BigFloat)
        calibration = identity_calibration(T[3 0; 0 1])
        plan = SccReconstructorPreparationPlan(2, T(0.25))
        output = SccReconstructorArtifacts(plan, calibration)
        workspace = SccReconstructorPreparationWorkspace(plan, calibration)

        process!(output, workspace, plan, calibration)

        @test eltype(output.pixel_to_controller) == T
        @test eltype(output.controller_to_vdm) == T
        @test eltype(output.fused_reconstructor) == T
        @test eltype(output.singular_values) == T
        @test output.diagnostics isa SccReconstructorDiagnostics{T}
        @test output.controller_to_vdm * output.pixel_to_controller ≈
              output.fused_reconstructor rtol = 32eps(T)
    end
end

@testset "invalid SCC preparation inputs fail before output mutation" begin
    @test_throws ArgumentError SccReconstructorPreparationPlan(0, 0.0)
    @test_throws ArgumentError SccReconstructorPreparationPlan(1, -1.0)
    @test_throws ArgumentError SccReconstructorPreparationPlan(1, Inf)
    @test_throws ArgumentError SccReconstructorPreparationPlan(
        1,
        0.0;
        validation_tolerance=1.0,
    )

    calibration = identity_calibration([2.0 0.0; 0.0 1.0])
    oversized_rank = SccReconstructorPreparationPlan(3, 0.0)
    @test_throws DimensionMismatch SccReconstructorArtifacts(
        oversized_rank,
        calibration,
    )

    plan = SccReconstructorPreparationPlan(2, 0.0)
    output = SccReconstructorArtifacts(plan, calibration)
    workspace = SccReconstructorPreparationWorkspace(plan, calibration)
    fill!(output.fused_reconstructor, 9.0)
    output.pixel_to_controller = zeros(1, 1)
    @test_throws DimensionMismatch process!(
        output,
        workspace,
        plan,
        calibration,
    )
    @test all(==(9.0), output.fused_reconstructor)

    bad_projection = SccReconstructorCalibration(
        calibration.selected_pixel_response,
        ones(1, 1),
        calibration.pixel_noise_covariance,
        calibration.candidate_mode_stroke_metric,
        calibration.llowfs_authority_projector,
    )
    @test_throws DimensionMismatch SccReconstructorArtifacts(
        plan,
        bad_projection,
    )

    nonsymmetric_noise = SccReconstructorCalibration(
        calibration.selected_pixel_response,
        calibration.candidate_mode_to_vdm,
        [1.0 0.5; 0.0 1.0],
        calibration.candidate_mode_stroke_metric,
        calibration.llowfs_authority_projector,
    )
    nonsymmetric_output = SccReconstructorArtifacts(plan, nonsymmetric_noise)
    nonsymmetric_workspace =
        SccReconstructorPreparationWorkspace(plan, nonsymmetric_noise)
    @test_throws ArgumentError process!(
        nonsymmetric_output,
        nonsymmetric_workspace,
        plan,
        nonsymmetric_noise,
    )

    nonprojector = SccReconstructorCalibration(
        calibration.selected_pixel_response,
        calibration.candidate_mode_to_vdm,
        calibration.pixel_noise_covariance,
        calibration.candidate_mode_stroke_metric,
        [1.0 0.0; 0.0 0.5],
    )
    nonprojector_output = SccReconstructorArtifacts(plan, nonprojector)
    nonprojector_workspace =
        SccReconstructorPreparationWorkspace(plan, nonprojector)
    @test_throws ArgumentError process!(
        nonprojector_output,
        nonprojector_workspace,
        plan,
        nonprojector,
    )

    nonpositive_noise = SccReconstructorCalibration(
        calibration.selected_pixel_response,
        calibration.candidate_mode_to_vdm,
        [1.0 0.0; 0.0 0.0],
        calibration.candidate_mode_stroke_metric,
        calibration.llowfs_authority_projector,
    )
    nonpositive_output = SccReconstructorArtifacts(plan, nonpositive_noise)
    nonpositive_workspace =
        SccReconstructorPreparationWorkspace(plan, nonpositive_noise)
    @test_throws PosDefException process!(
        nonpositive_output,
        nonpositive_workspace,
        plan,
        nonpositive_noise,
    )

    nonfinite_response = SccReconstructorCalibration(
        [2.0 0.0; 0.0 NaN],
        calibration.candidate_mode_to_vdm,
        calibration.pixel_noise_covariance,
        calibration.candidate_mode_stroke_metric,
        calibration.llowfs_authority_projector,
    )
    nonfinite_output = SccReconstructorArtifacts(plan, nonfinite_response)
    nonfinite_workspace =
        SccReconstructorPreparationWorkspace(plan, nonfinite_response)
    @test_throws ArgumentError process!(
        nonfinite_output,
        nonfinite_workspace,
        plan,
        nonfinite_response,
    )
end

@testset "SCC preparation public API exports" begin
    for name in (
        :SccReconstructorPreparationPlan,
        :SccReconstructorCalibration,
        :SccReconstructorDiagnostics,
        :SccReconstructorArtifacts,
        :SccReconstructorPreparationWorkspace,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "covariance symmetry is independent of scale" begin
    id = Matrix{Float32}(I,2,2)
    nonsymmetric = 1f-11 .* Float32[1 0.5; 0 1]
    calibration = SccReconstructorCalibration(id, id, nonsymmetric, id, id)
    plan = SccReconstructorPreparationPlan(2, 0f0)
    output = SccReconstructorArtifacts(plan, calibration)
    ws = SccReconstructorPreparationWorkspace(plan, calibration)
    @test_throws ArgumentError process!(output, ws, plan, calibration)
    unit_scale = SccReconstructorCalibration(id, id, nonsymmetric .* 1f11, id, id)
    @test_throws ArgumentError process!(output, ws, plan, unit_scale)
end

end
