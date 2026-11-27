module SccWeightedRegularizedReconstructionTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test

import FilterGraphAlgorithms: process!

const INTERACTION = Float32[1 0; 0 1; 1 1; 2 -1]
const REGULARIZATION = Matrix{Float32}(I, 2, 2)

function weighted_plan(; iterations=8, relative_tolerance=1.0f-5, regularization=0.1f0)
    return SccWeightedRegularizedReconstructionPlan(
        INTERACTION,
        REGULARIZATION;
        regularization,
        iterations,
        relative_tolerance,
        minimum_diagonal=1.0f-6,
    )
end

function prepared_weighted_reconstructor(; iterations=8, relative_tolerance=1.0f-5)
    return prepare_algorithm(
        SccWeightedRegularizedReconstructorF32,
        SccWeightedRegularizedReconstructorF32Configuration(
            measurement_count=4,
            controller_coordinate_count=2,
            regularization_row_count=2,
            measurement_schema=SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
            initial_interaction_matrix=collect(vec(transpose(INTERACTION))),
            initial_regularization_operator=collect(vec(transpose(REGULARIZATION))),
            regularization=0.1f0,
            iterations=iterations,
            relative_tolerance=relative_tolerance,
            minimum_diagonal=1.0f-6,
        ),
    )
end

function weighted_allocation_bytes(output, prepared, measurements, weights)
    return @allocated process!(output, prepared, measurements, weights)
end

@testset "weighted SCC declaration preserves measurement and controller spaces" begin
    @test algorithm_label(SccWeightedRegularizedReconstructorF32) ==
          "scc-weighted-regularized-reconstructor-f32"
    @test algorithm_operation(SccWeightedRegularizedReconstructorF32) isa
          SccWeightedRegularizedReconstructionF32Operation
    @test FilterGraphAlgorithms.algorithm_failure_type(
        SccWeightedRegularizedReconstructorF32,
    ) === SccWeightedRegularizedReconstructionFailure
    ports = algorithm_ports(SccWeightedRegularizedReconstructorF32)
    @test map(port -> port.name, ports) == (
        :measurements,
        :weights,
        Symbol("controller-residual-error"),
        Symbol("interaction-matrix"),
        Symbol("regularization-operator"),
    )
    formats = algorithm_formats(prepared_weighted_reconstructor())
    @test getproperty.(formats, :shape) == ((4,), (4,), (2,), (4, 2), (2, 2))
    @test getproperty.(formats, :schema) == (
        SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
        SCC_MEASUREMENT_WEIGHTS_V1,
        CONTROLLER_RESIDUAL_ERROR_V1,
        SCC_INTERACTION_MATRIX_V1,
        SCC_REGULARIZATION_OPERATOR_V1,
    )
end

@testset "bounded PCG matches the independent regularized normal equation" begin
    plan = weighted_plan()
    workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
    desired = Float32[2, -1]
    measurements = INTERACTION * desired
    weights = Float32[1, 0.5, 1.5, 2]
    output = zeros(Float32, 2)
    @test process!(output, workspace, plan, measurements, weights) === nothing
    weighted = Diagonal(weights)
    expected = inv(
        transpose(INTERACTION) * weighted * INTERACTION +
        plan.regularization * transpose(REGULARIZATION) * REGULARIZATION,
    ) * (transpose(INTERACTION) * weighted * measurements)
    @test output ≈ expected rtol = 2.0f-5 atol = 2.0f-6

    weights[2] = 0
    measurements[2] = NaN32
    @test process!(output, workspace, plan, measurements, weights) === nothing
end

@testset "controller partitions reproduce complete right-hand-side preparation" begin
    plan = weighted_plan()
    complete = SccWeightedRegularizedReconstructionWorkspace(plan)
    partitioned = SccWeightedRegularizedReconstructionWorkspace(plan)
    measurements = INTERACTION * Float32[2, -1]
    weights = Float32[1, 0.5, 1.5, 2]
    @test FilterGraphAlgorithms._scc_weighted_rhs_and_diagonal!(
        complete,
        plan,
        measurements,
        weights,
    ) === nothing
    @test FilterGraphAlgorithms._scc_weighted_rhs_and_diagonal!(
        partitioned,
        plan,
        measurements,
        weights,
        1:1,
    ) === nothing
    @test FilterGraphAlgorithms._scc_weighted_rhs_and_diagonal!(
        partitioned,
        plan,
        measurements,
        weights,
        2:2,
    ) === nothing
    @test partitioned.right_hand_side == complete.right_hand_side
    @test partitioned.diagonal == complete.diagonal
end

@testset "weighted failures preserve caller output" begin
    plan = weighted_plan()
    workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
    measurements = INTERACTION * Float32[2, -1]
    output = fill(71.0f0, 2)
    @test process!(
        output,
        workspace,
        plan,
        measurements,
        Float32[1, -1, 1, 1],
    ) === SccWeightedRegularizedReconstructionInvalidWeight
    @test output == fill(71.0f0, 2)
    measurements[1] = NaN32
    @test process!(output, workspace, plan, measurements, ones(Float32, 4)) ===
          SccWeightedRegularizedReconstructionInvalidMeasurement
    @test output == fill(71.0f0, 2)

    one_iteration = weighted_plan(iterations=1, relative_tolerance=eps(Float32))
    one_workspace = SccWeightedRegularizedReconstructionWorkspace(one_iteration)
    measurements = INTERACTION * Float32[2, -1]
    @test process!(output, one_workspace, one_iteration, measurements, ones(Float32, 4)) ===
          SccWeightedRegularizedReconstructionNotConverged
    @test output == fill(71.0f0, 2)

    singular = weighted_plan(regularization=0.0f0)
    singular_workspace = SccWeightedRegularizedReconstructionWorkspace(singular)
    @test process!(
        output,
        singular_workspace,
        singular,
        zeros(Float32, 4),
        zeros(Float32, 4),
    ) === SccWeightedRegularizedReconstructionSingularSystem
end

@testset "weighted calibration replacement and warmed processing are bounded" begin
    prepared = prepared_weighted_reconstructor()
    measurements = INTERACTION * Float32[2, -1]
    weights = ones(Float32, 4)
    output = zeros(Float32, 2)
    process!(output, prepared, measurements, weights)
    @test weighted_allocation_bytes(output, prepared, measurements, weights) == 0

    replacement = copy(INTERACTION)
    replacement[1, 1] = 2
    @test replace_parameter!(prepared, "interaction-matrix", replacement) === prepared
    @test prepared.plan.interaction_matrix == replacement
    original = prepared.plan
    broken = copy(replacement)
    broken[1, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(prepared, "interaction-matrix", broken)
    @test prepared.plan === original
end

@testset "weighted SCC public API exports" begin
    for name in (
        :SccWeightedRegularizedReconstructorF32,
        :SccWeightedRegularizedReconstructorF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


@testset "PCG convergence is invariant under measurement rescaling" begin
    for T in (Float32, Float64)
        H = T[1 0; 0 2; 1 1]
        L = Matrix{T}(I, 2, 2)
        plan = SccWeightedRegularizedReconstructionPlan(H, L;
            regularization=T(0.125), iterations=8, relative_tolerance=T(1e-5), minimum_diagonal=T(1e-6))
        workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
        weights = T[1, 0.5, 2]
        base = H * T[0.5, -0.25]
        # Wide arithmetic makes the residual oracle independent of the PCG products.
        wide_H, wide_L, wide_W = BigFloat.(H), BigFloat.(L), Diagonal(BigFloat.(weights))
        A = wide_H' * wide_W * wide_H + BigFloat(plan.regularization) * wide_L' * wide_L
        exponents = T === Float32 ? (-120, -90, -30, 0, 30, 90, 120) : (-1000, -600, -100, 0, 100, 600, 1000)
        for exponent in exponents
            measurements = ldexp.(base, exponent)
            output = fill(T(71), 2)
            @test process!(output, workspace, plan, measurements, weights) === nothing
            b = wide_H' * wide_W * BigFloat.(measurements)
            @test norm(A * BigFloat.(output) - b) / norm(b) <= BigFloat(2e-5)
            @test (@allocated process!(output, workspace, plan, measurements, weights)) == 0
        end
        output = fill(T(71), 2)
        @test process!(output, workspace, plan, zeros(T, 3), weights) === nothing
        @test output == zeros(T, 2)
    end
end


@testset "PCG handles representable RHS endpoints and rejects unrepresentable solutions" begin
    for T in (Float32, Float64)
        plan = SccWeightedRegularizedReconstructionPlan(ones(T, 1, 1), ones(T, 1, 1);
            regularization=zero(T), iterations=2, relative_tolerance=T(1e-5), minimum_diagonal=T(1e-6))
        workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
        output = T[71]
        for value in (nextfloat(zero(T)), floatmin(T), floatmax(T))
            @test process!(output, workspace, plan, T[value], T[1]) === nothing
            @test output == T[value]
        end
        small = T === Float32 ? T(1e-10) : T(1e-100)
        large = T === Float32 ? T(1e30) : T(1e300)
        plan = SccWeightedRegularizedReconstructionPlan(fill(small, 1, 1), ones(T, 1, 1);
            regularization=zero(T), iterations=2, relative_tolerance=T(1e-5), minimum_diagonal=small * small / T(2))
        fill!(output, T(71))
        @test process!(output, SccWeightedRegularizedReconstructionWorkspace(plan), plan, T[large], T[1]) ===
            SccWeightedRegularizedReconstructionSingularSystem
        @test output == T[71]
    end
end


@testset "PCG verifies the residual before publishing" begin
    H = Float32[0.5587443 -0.44319695; -0.5483226 0.4367202]
    plan = SccWeightedRegularizedReconstructionPlan(H, zeros(Float32, 1, 2);
        regularization=0f0, iterations=100, relative_tolerance=1f-6, minimum_diagonal=1f-12)
    workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
    output = fill(71f0, 2)
    @test process!(output, workspace, plan, Float32[-0.725702, -0.739205], ones(Float32, 2)) ===
        SccWeightedRegularizedReconstructionNotConverged
    @test output == fill(71f0, 2)
end

@testset "PCG excludes zero-weight rows and zero regularization" begin
    for T in (Float32, Float64)
        H = reshape(T[1, 1, floatmax(T)], 3, 1)
        L = ones(T, 1, 1)
        plan = SccWeightedRegularizedReconstructionPlan(H, L;
            regularization=one(T), iterations=2, relative_tolerance=T(1e-5),
            minimum_diagonal=T(1e-6))
        workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
        output = T[71]
        # The second row contributes in this frame and is masked in the next.
        @test process!(output, workspace, plan, T[4, 6, NaN], T[1, 2, 0]) === nothing
        @test output == T[4]
        for (measurement, weight) in ((floatmax(T), zero(T)), (T(NaN), -zero(T)))
            @test process!(output, workspace, plan, T[4, NaN, measurement], T[1, 0, weight]) === nothing
            @test output == T[2]
        end
        for (measurement, weight, failure) in (
            (one(T), one(T), SccWeightedRegularizedReconstructionSingularSystem),
            (T(NaN), one(T), SccWeightedRegularizedReconstructionInvalidMeasurement),
            (T(NaN), -one(T), SccWeightedRegularizedReconstructionInvalidWeight),
        )
            fill!(output, T(71))
            @test process!(output, workspace, plan, T[4, NaN, measurement], T[1, 0, weight]) === failure
            @test output == T[71]
        end

        H = ones(T, 1, 1)
        L = fill(floatmax(T), 1, 1)
        for lambda in (zero(T), -zero(T))
            plan = SccWeightedRegularizedReconstructionPlan(H, L;
                regularization=lambda, iterations=2, relative_tolerance=T(1e-5),
                minimum_diagonal=T(1e-6))
            workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
            fill!(output, T(71))
            @test process!(output, workspace, plan, T[2], T[1]) === nothing
            @test output == T[2]
        end
    end
end

@testset "PCG rejects active products that overflow during residual verification" begin
    for T in (Float32, Float64)
        exponent = T === Float32 ? 65 : 513
        for (H, L, lambda, measurements, weights) in (
            # The forward product overflows before its active weight can scale it.
            (reshape(T[1, 2], 2, 1), ones(T, 1, 1), zero(T),
                T[floatmax(T) * T(0.875), 0], T[1, 0.0625]),
            # The regularization adjoint overflows before lambda can scale it.
            (ones(T, 1, 1), fill(ldexp(one(T), exponent), 1, 1),
                ldexp(one(T), 7 - 2 * exponent), T[1024], T[1]),
        )
            plan = SccWeightedRegularizedReconstructionPlan(H, L;
                regularization=lambda, iterations=2, relative_tolerance=T(1e-5),
                minimum_diagonal=T(1e-6))
            workspace = SccWeightedRegularizedReconstructionWorkspace(plan)
            output = T[71]
            @test process!(output, workspace, plan, measurements, weights) ===
                  SccWeightedRegularizedReconstructionSingularSystem
            @test isfinite(only(workspace.candidate)) && only(workspace.candidate) > one(T)
            @test !isfinite(only(workspace.residual))
            @test output == T[71]
        end
    end
end

end
