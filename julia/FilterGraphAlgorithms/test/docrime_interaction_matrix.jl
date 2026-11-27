module DocrimeInteractionMatrixTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test
using TOML

import FilterGraphAlgorithms:
    _process_admitted!,
    commit!,
    discard!,
    prepare_properties,
    process!,
    reset!

criteria64() = DocrimeConditioningCriteria(0.1, 0.9, 0.01)

const SHARED_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "docrime_interaction_matrix.toml"),
)

function row_major_matrix(values, rows, columns, ::Type{T}=Float64) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_criteria(::Type{T}=Float64) where {T}
    fixture = SHARED_FIXTURE["conditioning_criteria"]
    return DocrimeConditioningCriteria{T}(
        fixture["minimum_corrector_variance"],
        fixture["maximum_absolute_corrector_correlation"],
        fixture["minimum_factorization_pivot_ratio"],
    )
end

function fixture_covariances(::Type{T}=Float64) where {T}
    measurements = SHARED_FIXTURE["measurement_count"]
    correctors = SHARED_FIXTURE["corrector_coordinate_count"]
    return DocrimeCovariances(
        SHARED_FIXTURE["sample_count"],
        row_major_matrix(
            SHARED_FIXTURE["measurement_corrector_cross_covariance_sum"],
            measurements,
            correctors,
            T,
        ),
        row_major_matrix(
            SHARED_FIXTURE["corrector_covariance_sum"],
            correctors,
            correctors,
            T,
        ),
    )
end

function solve(plan, covariances)
    output = DocrimeInteractionMatrixArtifacts(plan)
    workspace = DocrimeInteractionMatrixWorkspace(plan)
    process!(output, workspace, plan, covariances)
    return output
end

const DOCRIME_CROSS_SCHEMA = "example/test-cross-covariance/1"
const DOCRIME_CORRECTOR_COVARIANCE_SCHEMA =
    "example/test-corrector-covariance/1"
const DOCRIME_INTERACTION_SCHEMA = "example/test-interaction-matrix/1"

function declared_docrime_outputs(::Type{T}=Float32) where {T}
    return (
        fill(T(NaN), 2, 2),
        fill(typemax(UInt8), 1),
        fill(typemax(UInt64), 2),
        fill(typemax(UInt64), 1),
        fill(T(NaN), 2),
        fill(T(NaN), 1),
        fill(T(NaN), 1),
    )
end

function declared_docrime_prepared(declaration)
    configuration = if declaration === DocrimeOpenLoopInteractionMatrixF32
        DocrimeOpenLoopInteractionMatrixF32Configuration(
            measurement_count=2,
            corrector_coordinate_count=2,
            minimum_corrector_variance=0.1f0,
            cross_covariance_schema=DOCRIME_CROSS_SCHEMA,
            corrector_covariance_schema=DOCRIME_CORRECTOR_COVARIANCE_SCHEMA,
            interaction_matrix_schema=DOCRIME_INTERACTION_SCHEMA,
        )
    else
        DocrimeClosedLoopInteractionMatrixF32Configuration(
            measurement_count=2,
            corrector_coordinate_count=2,
            minimum_corrector_variance=0.1f0,
            cross_covariance_schema=DOCRIME_CROSS_SCHEMA,
            corrector_covariance_schema=DOCRIME_CORRECTOR_COVARIANCE_SCHEMA,
            interaction_matrix_schema=DOCRIME_INTERACTION_SCHEMA,
        )
    end
    return prepare_algorithm(declaration, configuration)
end

function direct_docrime_allocation_bytes(output, workspace, plan, covariances)
    return @allocated process!(output, workspace, plan, covariances)
end

function admitted_docrime_allocation_bytes(outputs, prepared, inputs)
    absent = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated _process_admitted!(
        outputs...,
        prepared,
        inputs...,
        absent,
        absent,
        absent,
    )
end

@testset "open-loop DO-CRIME solve matches the repository fixture" begin
    measurements = SHARED_FIXTURE["measurement_count"]
    correctors = SHARED_FIXTURE["corrector_coordinate_count"]
    plan = DocrimeOpenLoopInteractionMatrixPlan(
        measurements,
        correctors,
        fixture_criteria(),
    )
    covariances = fixture_covariances()
    output = DocrimeInteractionMatrixArtifacts(plan)
    workspace = DocrimeInteractionMatrixWorkspace(plan)
    interaction_matrix_storage = output.interaction_matrix

    @test process!(output, workspace, plan, covariances) === output
    @test output.interaction_matrix === interaction_matrix_storage
    @test workspace.calibration_product isa
          FilterGraphAlgorithms.AdaptiveOpticsCalibration.DOCRIME.InteractionMatrixProduct
    calibration_workspace = workspace.calibration_workspace
    @test calibration_workspace isa
          FilterGraphAlgorithms.AdaptiveOpticsCalibration.DOCRIME.InteractionMatrixWorkspace
    @test size(calibration_workspace.corrector_covariance) ==
          (correctors, correctors)
    @test size(calibration_workspace.factor) == (correctors, correctors)
    @test size(calibration_workspace.staged_interaction_matrix) ==
          (measurements, correctors)
    @test length(calibration_workspace.solve_scratch) == correctors
    expected = row_major_matrix(
        SHARED_FIXTURE["expected_open_loop_interaction_matrix"],
        measurements,
        correctors,
    )
    tolerance = SHARED_FIXTURE["absolute_tolerance_f64"]
    @test output.interaction_matrix ≈ expected atol = tolerance
    diagnostics = output.diagnostics
    @test diagnostics.status == DocrimeInteractionMatrixValid
    @test diagnostics.sample_count == 3
    expected_diagnostics = SHARED_FIXTURE["expected_diagnostics"]
    @test diagnostics.minimum_corrector_variance ==
          expected_diagnostics["minimum_corrector_variance"]
    @test diagnostics.maximum_corrector_variance ==
          expected_diagnostics["maximum_corrector_variance"]
    @test diagnostics.maximum_absolute_corrector_correlation ≈
          expected_diagnostics["maximum_absolute_corrector_correlation"] atol =
              tolerance
    @test diagnostics.minimum_factorization_pivot_ratio ≈
          expected_diagnostics["minimum_factorization_pivot_ratio"] atol = tolerance
    @test diagnostics.first_corrector_coordinate == 0
    @test diagnostics.second_corrector_coordinate == 0
end

@testset "closed-loop DO-CRIME applies the complete-increment loop gain" begin
    measurements = SHARED_FIXTURE["measurement_count"]
    correctors = SHARED_FIXTURE["corrector_coordinate_count"]
    plan = DocrimeClosedLoopInteractionMatrixPlan(
        measurements,
        correctors,
        Float32(SHARED_FIXTURE["closed_loop_gain"]),
        fixture_criteria(Float32),
    )
    covariances = fixture_covariances(Float32)

    output = solve(plan, covariances)

    expected = row_major_matrix(
        SHARED_FIXTURE["expected_closed_loop_interaction_matrix"],
        measurements,
        correctors,
        Float32,
    )
    @test output.interaction_matrix ≈ expected atol =
          SHARED_FIXTURE["absolute_tolerance_f32"]
    @test output.diagnostics.status == DocrimeInteractionMatrixValid
    @test output.diagnostics isa DocrimeInteractionMatrixDiagnostics{Float32}
end

@testset "DO-CRIME preserves the selected floating representation" begin
    for T in (Float16, Float32, Float64, BigFloat)
        criteria = DocrimeConditioningCriteria(T(0.1), T(0.9), T(0.01))
        plan = DocrimeOpenLoopInteractionMatrixPlan(2, 2, criteria)
        covariances = DocrimeCovariances(
            3,
            T[4 0; 0 2],
            T[2 0; 0 2],
        )

        output = solve(plan, covariances)

        @test output.interaction_matrix == T[2 0; 0 1]
        @test output.interaction_matrix isa Matrix{T}
        @test output.diagnostics isa DocrimeInteractionMatrixDiagnostics{T}
        @test output.diagnostics.status == DocrimeInteractionMatrixValid
    end
end

@testset "conditioning outcomes publish a zero interaction matrix" begin
    cases = (
        (
            DocrimeCovariances(1, zeros(1, 2), Matrix{Float64}(I, 2, 2)),
            criteria64(),
            DocrimeInteractionMatrixInsufficientSamples,
            0,
            0,
        ),
        (
            DocrimeCovariances(2, zeros(1, 2), [0.01 0.0; 0.0 1.0]),
            criteria64(),
            DocrimeInteractionMatrixInsufficientCorrectorVariance,
            1,
            0,
        ),
        (
            DocrimeCovariances(2, zeros(1, 2), [1.0 0.95; 0.95 1.0]),
            criteria64(),
            DocrimeInteractionMatrixExcessiveCorrectorCorrelation,
            1,
            2,
        ),
        (
            DocrimeCovariances(
                2,
                zeros(1, 3),
                [1.0 -0.75 -0.75; -0.75 1.0 -0.75; -0.75 -0.75 1.0],
            ),
            criteria64(),
            DocrimeInteractionMatrixNonPositiveDefinite,
            3,
            0,
        ),
        (
            DocrimeCovariances(2, zeros(1, 2), [1.0 0.99; 0.99 1.0]),
            DocrimeConditioningCriteria(0.1, 0.999, 0.05),
            DocrimeInteractionMatrixInsufficientFactorizationPivot,
            2,
            0,
        ),
    )

    for (covariances, criteria, status, first_coordinate, second_coordinate) in
        cases
        corrector_count = size(covariances.corrector_covariance_sum, 1)
        plan = DocrimeOpenLoopInteractionMatrixPlan(
            1,
            corrector_count,
            criteria,
        )
        output = solve(plan, covariances)
        @test output.diagnostics.status == status
        @test output.diagnostics.first_corrector_coordinate == first_coordinate
        @test output.diagnostics.second_corrector_coordinate == second_coordinate
        @test iszero(output.interaction_matrix)
        @test output.diagnostics.sample_count == covariances.sample_count
    end
end

@testset "DO-CRIME preparation rejects malformed configuration and input" begin
    for arguments in (
        (0.0, 0.9, 0.1),
        (NaN, 0.9, 0.1),
        (0.1, -0.1, 0.1),
        (0.1, 1.0, 0.1),
        (0.1, 0.9, 0.0),
        (0.1, 0.9, 1.1),
    )
        @test_throws ArgumentError DocrimeConditioningCriteria(arguments...)
    end
    criteria = criteria64()
    @test_throws ArgumentError DocrimeOpenLoopInteractionMatrixPlan(
        0,
        1,
        criteria,
    )
    @test_throws ArgumentError DocrimeOpenLoopInteractionMatrixPlan(
        1,
        0,
        criteria,
    )
    for loop_gain in (0.0, -0.1, NaN, Inf)
        @test_throws ArgumentError DocrimeClosedLoopInteractionMatrixPlan(
            1,
            1,
            loop_gain,
            criteria,
        )
    end
    @test_throws ArgumentError DocrimeCovariances(-1, zeros(1, 1), ones(1, 1))

    plan = DocrimeOpenLoopInteractionMatrixPlan(1, 2, criteria)
    valid = DocrimeCovariances(2, [1.0 0.0], [1.0 0.0; 0.0 1.0])
    output = DocrimeInteractionMatrixArtifacts(plan)
    workspace = DocrimeInteractionMatrixWorkspace(plan)
    fill!(output.interaction_matrix, 7.0)
    original_diagnostics = output.diagnostics

    malformed_extents = (
        DocrimeCovariances(2, ones(1, 1), [1.0 0.0; 0.0 1.0]),
        DocrimeCovariances(2, [1.0 0.0], ones(1, 1)),
    )
    for covariances in malformed_extents
        @test_throws DimensionMismatch process!(
            output,
            workspace,
            plan,
            covariances,
        )
        @test all(==(7.0), output.interaction_matrix)
        @test output.diagnostics == original_diagnostics
    end

    malformed_values = (
        (
            DocrimeCovariances(2, [NaN 0.0], [1.0 0.0; 0.0 1.0]),
            DocrimeInteractionMatrixNonFiniteCrossCovariance,
        ),
        (
            DocrimeCovariances(2, [1.0 0.0], [1.0 0.5; 0.25 1.0]),
            DocrimeInteractionMatrixNonSymmetricCorrectorCovariance,
        ),
    )
    for (covariances, failure) in malformed_values
        @test process!(
            output,
            workspace,
            plan,
            covariances,
        ) === failure
        @test all(==(7.0), output.interaction_matrix)
        @test output.diagnostics == original_diagnostics
    end

    output.interaction_matrix = zeros(1, 1)
    @test_throws DimensionMismatch process!(
        output,
        workspace,
        plan,
        valid,
    )

    criteria32 = DocrimeConditioningCriteria(0.1f0, 0.9f0, 0.01f0)
    plan32 = DocrimeOpenLoopInteractionMatrixPlan(1, 1, criteria32)
    output32 = DocrimeInteractionMatrixArtifacts(plan32)
    workspace32 = DocrimeInteractionMatrixWorkspace(plan32)
    unrepresentable_count = 16_777_218
    @test process!(
        output32,
        workspace32,
        plan32,
        DocrimeCovariances(
            unrepresentable_count,
            ones(Float32, 1, 1),
            ones(Float32, 1, 1),
        ),
    ) === DocrimeInteractionMatrixSampleCountNotRepresentable
end

@testset "DO-CRIME declarations preserve the frozen FGN contract" begin
    cases = (
        (
            declaration=DocrimeOpenLoopInteractionMatrixF32,
            label="docrime-open-loop-interaction-matrix-f32",
            property_names=(
                Symbol("maximum-absolute-corrector-correlation"),
                Symbol("minimum-factorization-pivot-ratio"),
            ),
        ),
        (
            declaration=DocrimeClosedLoopInteractionMatrixF32,
            label="docrime-closed-loop-interaction-matrix-f32",
            property_names=(
                Symbol("loop-gain"),
                Symbol("maximum-absolute-corrector-correlation"),
                Symbol("minimum-factorization-pivot-ratio"),
            ),
        ),
    )
    expected_ports = (
        Symbol("measurement-corrector-cross-covariance-sum"),
        Symbol("corrector-covariance-sum"),
        Symbol("sample-count"),
        Symbol("interaction-matrix"),
        :status,
        Symbol("status-coordinates"),
        Symbol("diagnostic-sample-count"),
        Symbol("corrector-variance-range"),
        Symbol("maximum-absolute-corrector-correlation"),
        Symbol("minimum-factorization-pivot-ratio"),
    )
    expected_schemas = (
        DOCRIME_CROSS_SCHEMA,
        DOCRIME_CORRECTOR_COVARIANCE_SCHEMA,
        DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
        DOCRIME_INTERACTION_SCHEMA,
        DOCRIME_INTERACTION_MATRIX_STATUS_V1,
        DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1,
        DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
        DOCRIME_CORRECTOR_VARIANCE_RANGE_V1,
        DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1,
        DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1,
    )

    for case in cases
        @test algorithm_label(case.declaration) == case.label
        @test FilterGraphAlgorithms.algorithm_failure_type(case.declaration) ===
              DocrimeInteractionMatrixFailure
        @test algorithm_property_policy(case.declaration) isa
              FilterGraphAlgorithms.RuntimeDeclaredProperties
        fields = algorithm_configuration(case.declaration)
        @test map(field -> field.name, fields) == (
            :measurement_count,
            :corrector_coordinate_count,
            :minimum_corrector_variance,
            :cross_covariance_schema,
            :corrector_covariance_schema,
            :interaction_matrix_schema,
        )
        @test fields[3].update isa ConstructionConfiguration
        @test all(
            field -> field.update isa GraphRebuildConfiguration,
            (fields[1], fields[2], fields[4], fields[5], fields[6]),
        )
        @test map(port -> port.name, algorithm_ports(case.declaration)) ==
              expected_ports

        prepared = declared_docrime_prepared(case.declaration)
        formats = algorithm_formats(prepared)
        @test getproperty.(formats, :shape) == (
            (2, 2),
            (2, 2),
            (1,),
            (2, 2),
            (1,),
            (2,),
            (1,),
            (2,),
            (1,),
            (1,),
        )
        @test getproperty.(formats, :schema) == expected_schemas
        @test getproperty.(property_infos(prepared.plan), :name) ==
              case.property_names
    end
end

@testset "declared solves publish complete diagnostics" begin
    inputs = (
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
        UInt64[3],
    )
    for (declaration, scale) in (
        (DocrimeOpenLoopInteractionMatrixF32, 1.0f0),
        (DocrimeClosedLoopInteractionMatrixF32, 0.4f0),
    )
        prepared = declared_docrime_prepared(declaration)
        if declaration === DocrimeClosedLoopInteractionMatrixF32
            prepared.plan = prepare_properties(
                prepared.plan,
                (PropertyAssignment(0, scale),),
            )
        end
        outputs = declared_docrime_outputs()
        @test process!(outputs..., prepared, inputs...) === nothing
        @test outputs[1] ≈ Float32[2 -1; 0.5 3] .* scale atol = 2.0f-6
        @test outputs[2] == UInt8[0]
        @test outputs[3] == UInt64[0, 0]
        @test outputs[4] == UInt64[3]
        @test outputs[5] == Float32[2, 2]
        @test outputs[6] == Float32[0.5]
        @test outputs[7] == Float32[0.75]
    end
end

@testset "declared conditioning status preserves frozen wire encoding" begin
    prepared = declared_docrime_prepared(DocrimeOpenLoopInteractionMatrixF32)
    outputs = declared_docrime_outputs()
    inputs = (
        zeros(Float32, 2, 2),
        Float32[1 0.95; 0.95 1],
        UInt64[2],
    )

    @test process!(outputs..., prepared, inputs...) === nothing
    @test iszero(outputs[1])
    @test outputs[2] == UInt8[3]
    @test outputs[3] == UInt64[0, 1]
    @test outputs[4] == UInt64[2]
    @test outputs[5] == Float32[1, 1]
    @test outputs[6] == Float32[0.95]
    @test outputs[7] == Float32[0]
end

@testset "runtime solve criteria replace atomically" begin
    open = declared_docrime_prepared(DocrimeOpenLoopInteractionMatrixF32)
    @test property_values(open.plan) == NamedTuple{
        (
            Symbol("maximum-absolute-corrector-correlation"),
            Symbol("minimum-factorization-pivot-ratio"),
        ),
    }((0.9f0, 0.01f0))
    replacement = prepare_properties(
        open.plan,
        (PropertyAssignment(0, 0.4f0),),
    )
    @test property_value(replacement, 0) === 0.4f0
    @test property_value(open.plan, 0) === 0.9f0
    @test_throws PropertyConstraintError prepare_properties(
        open.plan,
        (PropertyAssignment(0, 1.0f0),),
    )

    closed = declared_docrime_prepared(DocrimeClosedLoopInteractionMatrixF32)
    replacement = prepare_properties(
        closed.plan,
        (
            PropertyAssignment(0, 0.4f0),
            PropertyAssignment(1, 0.8f0),
            PropertyAssignment(2, 0.02f0),
        ),
    )
    @test property_values(replacement) == NamedTuple{
        (
            Symbol("loop-gain"),
            Symbol("maximum-absolute-corrector-correlation"),
            Symbol("minimum-factorization-pivot-ratio"),
        ),
    }((0.4f0, 0.8f0, 0.02f0))
    @test property_values(closed.plan) == (
        var"loop-gain"=1.0f0,
        var"maximum-absolute-corrector-correlation"=0.9f0,
        var"minimum-factorization-pivot-ratio"=0.01f0,
    )
end

@testset "declared rejection preserves every caller-owned output" begin
    prepared = declared_docrime_prepared(DocrimeOpenLoopInteractionMatrixF32)
    outputs = declared_docrime_outputs()
    valid = (
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
        UInt64[3],
    )
    process!(outputs..., prepared, valid...)
    retained = map(copy, outputs)
    failures = (
        (
            (Float32[NaN 0; 0 0], valid[2], valid[3]),
            DocrimeInteractionMatrixNonFiniteCrossCovariance,
        ),
        (
            (valid[1], Float32[4 Inf; Inf 4], valid[3]),
            DocrimeInteractionMatrixNonFiniteCorrectorCovariance,
        ),
        (
            (valid[1], Float32[4 1; 2 4], valid[3]),
            DocrimeInteractionMatrixNonSymmetricCorrectorCovariance,
        ),
        (
            (valid[1], valid[2], UInt64[16_777_218]),
            DocrimeInteractionMatrixSampleCountNotRepresentable,
        ),
        (
            (valid[1], valid[2], UInt64[typemax(UInt64)]),
            DocrimeInteractionMatrixSampleCountOutOfRange,
        ),
    )
    for (inputs, failure) in failures
        @test process!(outputs..., prepared, inputs...) === failure
        @test outputs == retained
    end

    @test_throws DimensionMismatch process!(
        outputs...,
        prepared,
        zeros(Float32, 1, 2),
        valid[2],
        valid[3],
    )
    @test outputs == retained
    @test_throws ArgumentError process!(
        prepared.workspace.calibration_product.matrix,
        outputs[2:end]...,
        prepared,
        valid...,
    )
end

@testset "DO-CRIME solve paths allocate no heap after warmup" begin
    criteria = DocrimeConditioningCriteria(0.1f0, 0.9f0, 0.01f0)
    plan = DocrimeOpenLoopInteractionMatrixPlan(2, 2, criteria)
    workspace = DocrimeInteractionMatrixWorkspace(plan)
    output = DocrimeInteractionMatrixArtifacts(plan)
    valid = DocrimeCovariances(
        3,
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
    )
    rejected = DocrimeCovariances(
        3,
        Float32[NaN 0; 0 0],
        Float32[4 2; 2 4],
    )
    process!(output, workspace, plan, valid)
    @test direct_docrime_allocation_bytes(output, workspace, plan, valid) == 0
    @test process!(output, workspace, plan, rejected) ===
          DocrimeInteractionMatrixNonFiniteCrossCovariance
    @test direct_docrime_allocation_bytes(output, workspace, plan, rejected) == 0
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0

    prepared = declared_docrime_prepared(DocrimeOpenLoopInteractionMatrixF32)
    outputs = declared_docrime_outputs()
    inputs = (
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
        UInt64[3],
    )
    process!(outputs..., prepared, inputs...)
    @test admitted_docrime_allocation_bytes(outputs, prepared, inputs) == 0
    nonfinite = (Float32[NaN 0; 0 0], inputs[2], inputs[3])
    @test admitted_docrime_allocation_bytes(
        outputs,
        prepared,
        nonfinite,
    ) == 0
end

@testset "DO-CRIME Julia public API exports" begin
    for name in (
        :DocrimeConditioningCriteria,
        :DocrimeInteractionMatrixStatus,
        :DocrimeInteractionMatrixValid,
        :DocrimeInteractionMatrixInsufficientSamples,
        :DocrimeInteractionMatrixInsufficientCorrectorVariance,
        :DocrimeInteractionMatrixExcessiveCorrectorCorrelation,
        :DocrimeInteractionMatrixNonPositiveDefinite,
        :DocrimeInteractionMatrixInsufficientFactorizationPivot,
        :DocrimeInteractionMatrixDiagnostics,
        :DocrimeInteractionMatrixFailure,
        :DocrimeCovariances,
        :DocrimeOpenLoopInteractionMatrixPlan,
        :DocrimeClosedLoopInteractionMatrixPlan,
        :DocrimeInteractionMatrixArtifacts,
        :DocrimeInteractionMatrixWorkspace,
        :DocrimeOpenLoopInteractionMatrixF32,
        :DocrimeOpenLoopInteractionMatrixF32Configuration,
        :DocrimeClosedLoopInteractionMatrixF32,
        :DocrimeClosedLoopInteractionMatrixF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
