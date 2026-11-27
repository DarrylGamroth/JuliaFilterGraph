module StructuredInteractionMatrixTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "structured_interaction_matrix.toml"),
)

function row_major_matrix(values, rows, columns, ::Type{T}=Float64) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_matrices(name::String, ::Type{T}=Float64) where {T}
    fixture = FIXTURE[name]
    measurements = fixture["measurement_count"]
    correctors = fixture["corrector_coordinate_count"]
    response_rows = name == "zonal" ? 2 * correctors : 2 * fixture["order"]
    responses = row_major_matrix(
        fixture["responses"],
        response_rows,
        measurements,
        T,
    )
    expected = row_major_matrix(
        fixture["expected_interaction_matrix"],
        measurements,
        correctors,
        T,
    )
    return responses, expected
end

function estimate_interaction_matrix(plan, responses)
    output = zeros(
        eltype(responses),
        plan.measurement_count,
        plan.corrector_coordinate_count,
    )
    workspace = StructuredInteractionMatrixWorkspace(plan)
    process!(output, workspace, plan, responses)
    return output
end

function structured_estimator_allocation_bytes(output, workspace, plan, responses)
    process!(output, workspace, plan, responses)
    return @allocated process!(output, workspace, plan, responses)
end

function declared_estimator_allocation_bytes(output, prepared, responses)
    process!(output, prepared, responses)
    return @allocated process!(output, prepared, responses)
end

@testset "zonal estimation matches the repository fixture" begin
    fixture = FIXTURE["zonal"]
    responses, expected = fixture_matrices("zonal")
    plan = ZonalInteractionMatrixPlan(
        fixture["measurement_count"],
        fixture["amplitudes"],
    )
    workspace = StructuredInteractionMatrixWorkspace(plan)
    output = fill(NaN, size(expected))

    @test FIXTURE["schema"] ==
          "test.julia-filter-graph/structured-interaction-matrix/1"
    @test process!(output, workspace, plan, responses) === nothing
    @test output ≈ expected atol = FIXTURE["absolute_tolerance_f64"]
    @test plan.measurement_count == 2
    @test plan.corrector_coordinate_count == 3
    @test plan.response_count == 12
    @test plan.amplitudes == Tuple(fixture["amplitudes"])
    @test plan.calibration_plan isa
          FilterGraphAlgorithms.AdaptiveOpticsCalibration.InteractionMatrices.ZonalPushPullPlan
    @test workspace.calibration_workspace isa
          FilterGraphAlgorithms.AdaptiveOpticsCalibration.InteractionMatrices.StructuredWorkspace
    @test input_progress(plan) == NoProgress()
    @test output_progress(plan) == NoProgress()
end

@testset "Hadamard estimation matches the repository fixture" begin
    fixture = FIXTURE["hadamard"]
    responses, expected = fixture_matrices("hadamard")
    plan = HadamardInteractionMatrixPlan(
        fixture["measurement_count"],
        fixture["amplitudes"],
    )

    output = estimate_interaction_matrix(plan, responses)

    @test output ≈ expected atol = FIXTURE["absolute_tolerance_f64"]
    @test plan.order == fixture["order"]
    @test plan.corrector_coordinate_count == 3
    @test plan.response_count == 16
    @test plan.amplitudes == Tuple(fixture["amplitudes"])
    @test plan.calibration_plan isa
          FilterGraphAlgorithms.AdaptiveOpticsCalibration.InteractionMatrices.HadamardPushPullPlan
    @test input_progress(plan) == NoProgress()
    @test output_progress(plan) == NoProgress()
end

@testset "structured probe sources recover known interaction matrices" begin
    cases = (
        (
            probe=ZonalPushPullPlan(Float64[0.5, 2, 1.5]),
            estimator=ZonalInteractionMatrixPlan(2, Float64[0.5, 2, 1.5]),
            interaction=Float64[1 2 -1; 0.5 -2 4],
            offset=Float64[10, -7],
        ),
        (
            probe=HadamardPushPullPlan(Float64[0.25, 2, 0.5]),
            estimator=HadamardInteractionMatrixPlan(2, Float64[0.25, 2, 0.5]),
            interaction=Float64[1 -3 2; 0.5 4 -2],
            offset=Float64[100, -30],
        ),
    )

    for case in cases
        probe_workspace = CalibrationProbeWorkspace(case.probe)
        destinations = (
            excitation=zeros(Float64, 3),
            pattern_index=zeros(UInt64, 1),
            polarity=zeros(UInt8, 1),
            completes_cycle=zeros(Bool, 1),
        )
        responses = zeros(
            Float64,
            calibration_probe_cycle_length(case.probe),
            2,
        )
        for response_step in axes(responses, 1)
            process!(
                destinations.excitation,
                destinations.pattern_index,
                destinations.polarity,
                destinations.completes_cycle,
                probe_workspace,
                case.probe,
            )
            mul!(
                @view(responses[response_step, :]),
                case.interaction,
                destinations.excitation,
            )
            @view(responses[response_step, :]) .+= case.offset
            commit!(probe_workspace)
        end

        actual = zeros(Float64, 2, 3)
        @test process!(
            actual,
            StructuredInteractionMatrixWorkspace(case.estimator),
            case.estimator,
            responses,
        ) === nothing
        @test actual ≈ case.interaction atol = 1.0e-12
    end
end

@testset "structured estimator declarations preserve the frozen FGN contract" begin
    cases = (
        (
            declaration=ZonalInteractionMatrixF32,
            configuration=ZonalInteractionMatrixF32Configuration(
                measurement_count=2,
                amplitudes=Float32[0.5, 2],
                response_schema="example/test-zonal-response-cycle/1",
                interaction_matrix_schema="example/test-interaction-matrix/1",
            ),
            label="zonal-interaction-matrix-f32",
            response_shape=(4, 2),
        ),
        (
            declaration=HadamardInteractionMatrixF32,
            configuration=HadamardInteractionMatrixF32Configuration(
                measurement_count=2,
                amplitudes=Float32[0.5, 2],
                response_schema="example/test-hadamard-response-cycle/1",
                interaction_matrix_schema="example/test-interaction-matrix/1",
            ),
            label="hadamard-interaction-matrix-f32",
            response_shape=(8, 2),
        ),
    )

    for case in cases
        @test algorithm_label(case.declaration) == case.label
        @test FilterGraphAlgorithms.algorithm_failure_type(case.declaration) ===
              StructuredInteractionMatrixFailure
        configuration = algorithm_configuration(case.declaration)
        @test map(field -> field.name, configuration) == (
            :measurement_count,
            :amplitudes,
            :response_schema,
            :interaction_matrix_schema,
        )
        @test all(
            field -> field.update isa GraphRebuildConfiguration,
            configuration,
        )
        ports = algorithm_ports(case.declaration)
        @test map(port -> port.name, ports) == (
            :responses,
            Symbol("interaction-matrix"),
            :amplitudes,
        )
        @test ports[1].direction isa FilterGraphAlgorithms.InputPort
        @test ports[1].role isa FilterGraphAlgorithms.FrameData
        @test ports[2].direction isa FilterGraphAlgorithms.OutputPort
        @test ports[2].metadata_source === :responses
        @test ports[3].role isa FilterGraphAlgorithms.SparseParameter

        prepared = prepare_algorithm(case.declaration, case.configuration)
        formats = algorithm_formats(prepared)
        @test map(format -> format.shape, formats) == (
            case.response_shape,
            (2, 2),
            (2,),
        )
        @test formats[1].schema == case.configuration.response_schema
        @test formats[2].schema ==
              case.configuration.interaction_matrix_schema
        @test formats[3].schema == CALIBRATION_PROBE_AMPLITUDES_V1
    end
end

@testset "declared estimators match frozen results and adopt amplitudes" begin
    zonal = prepare_algorithm(
        ZonalInteractionMatrixF32,
        ZonalInteractionMatrixF32Configuration(
            measurement_count=2,
            amplitudes=Float32[0.5, 2],
            response_schema="example/test-zonal-response-cycle/1",
            interaction_matrix_schema="example/test-interaction-matrix/1",
        ),
    )
    zonal_responses = Float32[3 5; 1 1; 9 -1; 1 3]
    zonal_output = zeros(Float32, 2, 2)
    @test process!(zonal_output, zonal, zonal_responses) === nothing
    @test zonal_output == Float32[2 2; 4 -1]
    previous_zonal = zonal.plan
    replacement = Float32[1, 4]
    @test replace_parameter!(zonal, :amplitudes, replacement) === zonal
    fill!(replacement, 0)
    @test previous_zonal.amplitudes === (0.5f0, 2.0f0)
    @test zonal.plan.amplitudes === (1.0f0, 4.0f0)
    adopted_zonal = zonal.plan
    @test_throws ArgumentError replace_parameter!(zonal, :amplitudes, Float32[1, 0])
    @test_throws DimensionMismatch replace_parameter!(zonal, :amplitudes, Float32[1, 2, 3])
    @test zonal.plan === adopted_zonal
    @test process!(zonal_output, zonal, zonal_responses) === nothing
    @test zonal_output == Float32[1 1; 2 -0.5]

    hadamard = prepare_algorithm(
        HadamardInteractionMatrixF32,
        HadamardInteractionMatrixF32Configuration(
            measurement_count=1,
            amplitudes=Float32[0.5, 2],
            response_schema="example/test-hadamard-response-cycle/1",
            interaction_matrix_schema="example/test-interaction-matrix/1",
        ),
    )
    hadamard_responses = reshape(
        Float32[6, -6, -2, 2, 4, -4, 0, 0],
        8,
        1,
    )
    hadamard_output = zeros(Float32, 1, 2)
    @test process!(hadamard_output, hadamard, hadamard_responses) === nothing
    @test hadamard_output == Float32[6 0]
    previous_hadamard = hadamard.plan
    replacement = Float32[1, 4]
    @test replace_parameter!(hadamard, :amplitudes, replacement) === hadamard
    fill!(replacement, 0)
    @test previous_hadamard.amplitudes === (0.5f0, 2.0f0)
    @test hadamard.plan.amplitudes === (1.0f0, 4.0f0)
    adopted_hadamard = hadamard.plan
    @test_throws ArgumentError replace_parameter!(hadamard, :amplitudes, Float32[1, 0])
    @test_throws DimensionMismatch replace_parameter!(hadamard, :amplitudes, Float32[1, 2, 3])
    @test hadamard.plan === adopted_hadamard
    @test process!(hadamard_output, hadamard, hadamard_responses) === nothing
    @test hadamard_output == Float32[3 0]
end

@testset "structured estimator preparation uses runtime actuator counts" begin
    for (constructor, declaration, configuration, response_rows, expected) in (
        (
            ZonalInteractionMatrixPlan, ZonalInteractionMatrixF32,
            ZonalInteractionMatrixF32Configuration, 8, Float32[1 0 0 0],
        ),
        (
            HadamardInteractionMatrixPlan, HadamardInteractionMatrixF32,
            HadamardInteractionMatrixF32Configuration, 16, fill(0.125f0, 1, 4),
        ),
    )
        options = (
            measurement_count=1,
            response_schema="example/test-structured-response-cycle/1",
            interaction_matrix_schema="example/test-interaction-matrix/1",
        )
        small = @inferred prepare_algorithm(
            declaration, configuration(; options..., amplitudes=Float32[1]),
        )
        large = @inferred prepare_algorithm(
            declaration, configuration(; options..., amplitudes=ones(Float32, 4)),
        )
        @test typeof(large) === typeof(small)
        one_corrector = @inferred constructor(1, Float32[1])
        amplitudes = fill(1.0f0, 4)
        four_correctors = @inferred constructor(1, amplitudes)
        @test typeof(four_correctors) === typeof(one_corrector)
        fill!(amplitudes, 0)
        @test hasproperty(four_correctors, :amplitudes)
        @test four_correctors.amplitudes === (1.0f0, 1.0f0, 1.0f0, 1.0f0)
        @test four_correctors.corrector_coordinate_count == 4
        @test four_correctors.response_count == response_rows

        responses = zeros(Float32, response_rows, 1)
        responses[1] = 2
        output = similar(expected)
        workspace = @inferred StructuredInteractionMatrixWorkspace(four_correctors)
        @test @inferred(
            Union{Nothing,StructuredInteractionMatrixFailure},
            process!(output, workspace, four_correctors, responses),
        ) === nothing
        @test output == expected
        @test structured_estimator_allocation_bytes(
            output, workspace, four_correctors, responses,
        ) == 0
    end
end

@testset "structured estimators preserve Float32" begin
    for (name, constructor) in (
        ("zonal", ZonalInteractionMatrixPlan),
        ("hadamard", HadamardInteractionMatrixPlan),
    )
        fixture = FIXTURE[name]
        responses, expected = fixture_matrices(name, Float32)
        plan = constructor(
            fixture["measurement_count"],
            Float32.(fixture["amplitudes"]),
        )

        output = estimate_interaction_matrix(plan, responses)

        @test output isa Matrix{Float32}
        @test output ≈ expected atol = FIXTURE["absolute_tolerance_f32"]
        @test plan.amplitudes isa Tuple{Vararg{Float32}}
    end
end

@testset "structured estimators preserve generic floating representations" begin
    for T in (Float16, Float32, Float64, BigFloat)
        zonal = ZonalInteractionMatrixPlan(1, T[1])
        zonal_output = estimate_interaction_matrix(zonal, reshape(T[3, -1], 2, 1))
        @test zonal_output == fill(T(2), 1, 1)
        @test zonal_output isa Matrix{T}
        @test zonal.amplitudes isa Tuple{Vararg{T}}

        hadamard = HadamardInteractionMatrixPlan(1, T[1])
        hadamard_output = estimate_interaction_matrix(
            hadamard,
            reshape(T[2, -2, -2, 2], 4, 1),
        )
        @test hadamard_output == fill(T(2), 1, 1)
        @test hadamard_output isa Matrix{T}
        @test hadamard.amplitudes isa Tuple{Vararg{T}}
    end
end

@testset "noncontiguous views retain logical response order" begin
    fixture = FIXTURE["hadamard"]
    responses, expected = fixture_matrices("hadamard")
    response_storage = fill(NaN, 2 * size(responses, 1), 2 * size(responses, 2))
    response_view = @view response_storage[
        1:2:(2 * size(responses, 1)),
        1:2:(2 * size(responses, 2)),
    ]
    response_view .= responses
    output_storage = fill(NaN, 2 * size(expected, 1), 2 * size(expected, 2))
    output_view = @view output_storage[
        1:2:(2 * size(expected, 1)),
        1:2:(2 * size(expected, 2)),
    ]
    plan = HadamardInteractionMatrixPlan(
        fixture["measurement_count"],
        fixture["amplitudes"],
    )

    @test process!(
        output_view,
        StructuredInteractionMatrixWorkspace(plan),
        plan,
        response_view,
    ) === nothing
    @test output_view ≈ expected atol = FIXTURE["absolute_tolerance_f64"]
    @test structured_estimator_allocation_bytes(
        output_view,
        StructuredInteractionMatrixWorkspace(plan),
        plan,
        response_view,
    ) == 0
    expected_output = copy(output_view)
    response_view[1] = NaN
    view_workspace = StructuredInteractionMatrixWorkspace(plan)
    @test process!(output_view, view_workspace, plan, response_view) ===
          StructuredNonFiniteResponse
    @test output_view == expected_output
    @test structured_estimator_allocation_bytes(
        output_view,
        view_workspace,
        plan,
        response_view,
    ) == 0
    response_view .= responses
    response_view[1] = floatmax(Float64)
    response_view[2] = -floatmax(Float64)
    @test process!(output_view, view_workspace, plan, response_view) ===
          StructuredNonFiniteCalculation
    @test output_view == expected_output
    @test structured_estimator_allocation_bytes(
        output_view,
        view_workspace,
        plan,
        response_view,
    ) == 0
end

@testset "preparation rejects invalid dimensions and scales" begin
    for constructor in (
        ZonalInteractionMatrixPlan,
        HadamardInteractionMatrixPlan,
    )
        @test_throws ArgumentError constructor(0, [1.0])
        @test_throws ArgumentError constructor(1, Float64[])
        for amplitude in (0.0, -1.0, NaN, Inf)
            @test_throws ArgumentError constructor(1, [amplitude])
        end
    end
    @test_throws ArgumentError ZonalInteractionMatrixPlan(
        1,
        [nextfloat(0.0f0)],
    )
    @test_throws ArgumentError ZonalInteractionMatrixPlan(
        typemax(Int),
        [1.0],
    )
end

@testset "execution failures preserve caller-owned output" begin
    plan = ZonalInteractionMatrixPlan(1, [1.0])
    workspace = StructuredInteractionMatrixWorkspace(plan)
    output = fill(17.0, 1, 1)
    valid_responses = reshape([3.0, -1.0], 2, 1)

    @test_throws DimensionMismatch process!(
        output,
        workspace,
        plan,
        zeros(1, 1),
    )
    @test output == fill(17.0, 1, 1)
    @test_throws DimensionMismatch process!(
        zeros(1, 2),
        workspace,
        plan,
        valid_responses,
    )
    @test output == fill(17.0, 1, 1)

    wrong_calibration_workspace =
        FilterGraphAlgorithms.AdaptiveOpticsCalibration.InteractionMatrices.StructuredWorkspace(
            zeros(1, 2),
        )
    wrong_workspace = StructuredInteractionMatrixWorkspace{
        Float64,
        typeof(wrong_calibration_workspace),
    }(wrong_calibration_workspace)
    @test_throws DimensionMismatch process!(
        output,
        wrong_workspace,
        plan,
        valid_responses,
    )
    @test output == fill(17.0, 1, 1)

    @test process!(
        output,
        workspace,
        plan,
        reshape([NaN, 0.0], 2, 1),
    ) === StructuredNonFiniteResponse
    @test output == fill(17.0, 1, 1)
    @test process!(
        output,
        workspace,
        plan,
        reshape([floatmax(Float64), -floatmax(Float64)], 2, 1),
    ) === StructuredNonFiniteCalculation
    @test output == fill(17.0, 1, 1)
    @test_throws ArgumentError process!(
        workspace.calibration_workspace.staged_matrix,
        workspace,
        plan,
        valid_responses,
    )

    @test process!(output, workspace, plan, valid_responses) === nothing
    @test output == fill(2.0, 1, 1)

    hadamard = HadamardInteractionMatrixPlan(1, [1.0])
    hadamard_workspace = StructuredInteractionMatrixWorkspace(hadamard)
    fill!(output, 23.0)
    @test process!(
        output,
        hadamard_workspace,
        hadamard,
        reshape(
            [floatmax(Float64), -floatmax(Float64), 0.0, 0.0],
            4,
            1,
        ),
    ) === StructuredNonFiniteCalculation
    @test output == fill(23.0, 1, 1)
    @test process!(
        output,
        hadamard_workspace,
        hadamard,
        reshape([2.0, -2.0, -2.0, 2.0], 4, 1),
    ) === nothing
    @test output == fill(2.0, 1, 1)
end

@testset "structured estimator outcomes allocate no heap after warmup" begin
    cases = (
        (
            plan=ZonalInteractionMatrixPlan(1, Float32[1]),
            declaration=ZonalInteractionMatrixF32,
            configuration=ZonalInteractionMatrixF32Configuration(
                measurement_count=1,
                amplitudes=Float32[1],
                response_schema="example/test-zonal-response-cycle/1",
                interaction_matrix_schema="example/test-interaction-matrix/1",
            ),
            valid=reshape(Float32[3, -1], 2, 1),
            overflow=reshape(
                Float32[floatmax(Float32), -floatmax(Float32)],
                2,
                1,
            ),
        ),
        (
            plan=HadamardInteractionMatrixPlan(1, Float32[1]),
            declaration=HadamardInteractionMatrixF32,
            configuration=HadamardInteractionMatrixF32Configuration(
                measurement_count=1,
                amplitudes=Float32[1],
                response_schema="example/test-hadamard-response-cycle/1",
                interaction_matrix_schema="example/test-interaction-matrix/1",
            ),
            valid=reshape(Float32[2, -2, -2, 2], 4, 1),
            overflow=reshape(
                Float32[floatmax(Float32), -floatmax(Float32), 0, 0],
                4,
                1,
            ),
        ),
    )

    for case in cases
        output = zeros(Float32, 1, 1)
        workspace = StructuredInteractionMatrixWorkspace(case.plan)
        nonfinite = copy(case.valid)
        nonfinite[1] = NaN32
        mixed_failure = copy(case.overflow)
        mixed_failure[1] = NaN32
        @test @inferred(
            Union{Nothing,StructuredInteractionMatrixFailure},
            process!(output, workspace, case.plan, case.valid),
        ) === nothing
        expected_output = copy(output)
        @test @inferred(
            Union{Nothing,StructuredInteractionMatrixFailure},
            process!(output, workspace, case.plan, nonfinite),
        ) === StructuredNonFiniteResponse
        @test @inferred(
            Union{Nothing,StructuredInteractionMatrixFailure},
            process!(output, workspace, case.plan, case.overflow),
        ) === StructuredNonFiniteCalculation
        @test @inferred(
            Union{Nothing,StructuredInteractionMatrixFailure},
            process!(output, workspace, case.plan, mixed_failure),
        ) === StructuredNonFiniteResponse
        @test output == expected_output
        @test structured_estimator_allocation_bytes(
            output,
            workspace,
            case.plan,
            case.valid,
        ) == 0
        @test process!(output, workspace, case.plan, nonfinite) ===
              StructuredNonFiniteResponse
        @test structured_estimator_allocation_bytes(
            output,
            workspace,
            case.plan,
            nonfinite,
        ) == 0
        @test process!(output, workspace, case.plan, case.overflow) ===
              StructuredNonFiniteCalculation
        @test structured_estimator_allocation_bytes(
            output,
            workspace,
            case.plan,
            case.overflow,
        ) == 0
        @test structured_estimator_allocation_bytes(
            output,
            workspace,
            case.plan,
            mixed_failure,
        ) == 0
        @test @allocated(commit!(workspace)) == 0
        @test @allocated(discard!(workspace)) == 0
        @test @allocated(reset!(workspace)) == 0

        prepared = prepare_algorithm(case.declaration, case.configuration)
        @test declared_estimator_allocation_bytes(output, prepared, case.valid) == 0
        @test declared_estimator_allocation_bytes(output, prepared, nonfinite) == 0
        @test declared_estimator_allocation_bytes(output, prepared, case.overflow) == 0
    end
end

@testset "structured-estimator public API exports" begin
    for name in (
        :ZonalInteractionMatrixPlan,
        :HadamardInteractionMatrixPlan,
        :StructuredInteractionMatrixWorkspace,
        :StructuredInteractionMatrixFailure,
        :ZonalInteractionMatrixF32,
        :ZonalInteractionMatrixF32Configuration,
        :HadamardInteractionMatrixF32,
        :HadamardInteractionMatrixF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
