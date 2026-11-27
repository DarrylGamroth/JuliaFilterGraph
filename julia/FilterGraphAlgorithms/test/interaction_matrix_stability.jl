module InteractionMatrixStabilityTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms:
    commit!, discard!, prepare_properties, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "interaction_matrix_stability.toml"),
)

const STABILITY_MATRIX_SCHEMA =
    "example/test-interaction-matrix/1"

function stability_prepared()
    return prepare_algorithm(
        InteractionMatrixStabilityF32,
        InteractionMatrixStabilityF32Configuration(
            measurement_count=FIXTURE["measurement_count"],
            corrector_coordinate_count=FIXTURE["corrector_coordinate_count"],
            minimum_column_norm=Float32(FIXTURE["minimum_column_norm"]),
            interaction_matrix_schema=STABILITY_MATRIX_SCHEMA,
        ),
    )
end

function stability_outputs()
    return (
        fill(typemax(UInt8), 1),
        fill(typemax(UInt64), 1),
        fill(71.0f0, 2),
        fill(72.0f0, 1),
        fill(typemax(UInt64), 1),
        fill(73.0f0, 1),
    )
end

function stability_checked_allocation_bytes(
    status,
    status_coordinate,
    minimum_column_norms,
    maximum_relative_column_difference,
    maximum_relative_column_difference_coordinate,
    relative_frobenius_difference,
    prepared,
    first_window,
    second_window,
)
    return @allocated process!(
        status,
        status_coordinate,
        minimum_column_norms,
        maximum_relative_column_difference,
        maximum_relative_column_difference_coordinate,
        relative_frobenius_difference,
        prepared,
        first_window,
        second_window,
    )
end

function stability_admitted_allocation_bytes(
    status,
    status_coordinate,
    minimum_column_norms,
    maximum_relative_column_difference,
    maximum_relative_column_difference_coordinate,
    relative_frobenius_difference,
    prepared,
    first_window,
    second_window,
)
    metadata = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated FilterGraphAlgorithms._process_admitted!(
        status,
        status_coordinate,
        minimum_column_norms,
        maximum_relative_column_difference,
        maximum_relative_column_difference_coordinate,
        relative_frobenius_difference,
        prepared,
        first_window,
        second_window,
        metadata,
        metadata,
    )
end

stability_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
stability_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
stability_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

function row_major_matrix(values, rows, columns, ::Type{T}=Float64) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_plan(::Type{T}=Float64) where {T}
    criteria = InteractionMatrixStabilityCriteria{T}(
        FIXTURE["minimum_column_norm"],
        FIXTURE["maximum_relative_column_difference"],
        FIXTURE["maximum_relative_frobenius_difference"],
    )
    return InteractionMatrixStabilityPlan(
        FIXTURE["measurement_count"],
        FIXTURE["corrector_coordinate_count"],
        criteria,
    )
end

function fixture_input(::Type{T}=Float64) where {T}
    measurements = FIXTURE["measurement_count"]
    correctors = FIXTURE["corrector_coordinate_count"]
    return InteractionMatrixStabilityInput(
        row_major_matrix(
            FIXTURE["first_window"],
            measurements,
            correctors,
            T,
        ),
        row_major_matrix(
            FIXTURE["second_window"],
            measurements,
            correctors,
            T,
        ),
    )
end

function diagnostic_values(diagnostics)
    return (
        diagnostics.status,
        diagnostics.status_corrector_coordinate,
        diagnostics.minimum_first_window_column_norm,
        diagnostics.minimum_second_window_column_norm,
        diagnostics.maximum_relative_column_difference,
        diagnostics.maximum_relative_column_difference_coordinate,
        diagnostics.relative_frobenius_difference,
    )
end

@testset "stability diagnostics match the repository fixture" begin
    @test FIXTURE["schema"] ==
          "test.julia-filter-graph/interaction-matrix-stability/1"
    @test FIXTURE["expected_status"] ==
          "excessive_relative_column_difference"
    plan = fixture_plan()
    diagnostics = InteractionMatrixStabilityDiagnostics(plan)

    workspace = InteractionMatrixStabilityWorkspace(plan)
    calibration = FilterGraphAlgorithms.AdaptiveOpticsCalibration.Diagnostics
    @test plan.calibration_plan isa calibration.InteractionMatrixStabilityPlan{Float64}
    @test workspace.product isa calibration.InteractionMatrixStabilityProduct{Float64}
    @test process!(diagnostics, workspace, plan, fixture_input()) ===
          diagnostics
    tolerance = FIXTURE["absolute_tolerance_f64"]
    @test diagnostics.status ==
          InteractionMatrixStabilityExcessiveRelativeColumnDifference
    @test diagnostics.status_corrector_coordinate ==
          FIXTURE["expected_status_corrector_coordinate_zero_based"] + 1
    @test diagnostics.minimum_first_window_column_norm ≈
          FIXTURE["expected_minimum_first_window_column_norm"] atol = tolerance
    @test diagnostics.minimum_second_window_column_norm ≈
          FIXTURE["expected_minimum_second_window_column_norm"] atol = tolerance
    @test diagnostics.maximum_relative_column_difference ≈
          FIXTURE["expected_maximum_relative_column_difference"] atol = tolerance
    @test diagnostics.maximum_relative_column_difference_coordinate ==
          FIXTURE[
              "expected_maximum_relative_column_difference_coordinate_zero_based"
          ] + 1
    @test diagnostics.relative_frobenius_difference ≈
          FIXTURE["expected_relative_frobenius_difference"] atol = tolerance
    @test input_progress(plan) == NoProgress()
    @test output_progress(plan) == NoProgress()
end

@testset "relative differences are symmetric and ties use the first coordinate" begin
    plan = InteractionMatrixStabilityPlan(
        2,
        2,
        InteractionMatrixStabilityCriteria(0.1, 2.0, 2.0),
    )
    input = fixture_input()
    forward = InteractionMatrixStabilityDiagnostics(plan)
    reverse = InteractionMatrixStabilityDiagnostics(plan)

    process!(forward, InteractionMatrixStabilityWorkspace(plan), plan, input)
    process!(
        reverse,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        InteractionMatrixStabilityInput(
            input.second_window,
            input.first_window,
        ),
    )

    @test forward.status == InteractionMatrixStabilityValid
    @test forward.status == reverse.status
    @test forward.minimum_first_window_column_norm ==
          reverse.minimum_second_window_column_norm
    @test forward.minimum_second_window_column_norm ==
          reverse.minimum_first_window_column_norm
    @test forward.maximum_relative_column_difference ==
          reverse.maximum_relative_column_difference
    @test forward.relative_frobenius_difference ==
          reverse.relative_frobenius_difference
    @test forward.maximum_relative_column_difference_coordinate == 1
end

@testset "status order is norm, column, Frobenius, then valid" begin
    criteria = InteractionMatrixStabilityCriteria(0.1, 0.5, 0.5)
    plan = InteractionMatrixStabilityPlan(2, 2, criteria)
    diagnostics = InteractionMatrixStabilityDiagnostics(plan)

    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        InteractionMatrixStabilityInput(
            [0.0 0.0; 0.0 1.0],
            [2.0 0.0; 0.0 1.0],
        ),
    )
    @test diagnostics.status ==
          InteractionMatrixStabilityInsufficientColumnNorm
    @test diagnostics.status_corrector_coordinate == 1
    @test iszero(diagnostics.maximum_relative_column_difference)

    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        fixture_input(),
    )
    @test diagnostics.status ==
          InteractionMatrixStabilityExcessiveRelativeColumnDifference
    @test diagnostics.status_corrector_coordinate == 1

    frobenius_plan = InteractionMatrixStabilityPlan(
        2,
        2,
        InteractionMatrixStabilityCriteria(0.1, 1.0, 0.5),
    )
    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(frobenius_plan),
        frobenius_plan,
        fixture_input(),
    )
    @test diagnostics.status ==
          InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference
    @test iszero(diagnostics.status_corrector_coordinate)

    identical = InteractionMatrixStabilityInput(
        [3.0 0.0; 4.0 2.0],
        [3.0 0.0; 4.0 2.0],
    )
    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        identical,
    )
    @test diagnostics.status == InteractionMatrixStabilityValid
    @test diagnostics.minimum_first_window_column_norm == 2.0
    @test iszero(diagnostics.maximum_relative_column_difference)
    @test iszero(diagnostics.relative_frobenius_difference)
end

@testset "stability diagnostics preserve Float32" begin
    plan = fixture_plan(Float32)
    diagnostics = InteractionMatrixStabilityDiagnostics(plan)

    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        fixture_input(Float32),
    )

    @test diagnostics isa InteractionMatrixStabilityDiagnostics{Float32}
    @test diagnostics.status ==
          InteractionMatrixStabilityExcessiveRelativeColumnDifference
    @test diagnostics.maximum_relative_column_difference ≈
          Float32(FIXTURE["expected_maximum_relative_column_difference"]) atol =
              FIXTURE["absolute_tolerance_f32"]
    @test diagnostics.relative_frobenius_difference ≈
          Float32(FIXTURE["expected_relative_frobenius_difference"]) atol =
              FIXTURE["absolute_tolerance_f32"]
end

@testset "stability diagnostics preserve generic floating representations" begin
    for T in (Float16, Float32, Float64, BigFloat)
        criteria = InteractionMatrixStabilityCriteria(T(0.1), T(0.5), T(0.5))
        plan = InteractionMatrixStabilityPlan(2, 2, criteria)
        diagnostics = InteractionMatrixStabilityDiagnostics(plan)
        matrix = T[3 0; 4 2]

        process!(
            diagnostics,
            InteractionMatrixStabilityWorkspace(plan),
            plan,
            InteractionMatrixStabilityInput(matrix, copy(matrix)),
        )

        @test diagnostics isa InteractionMatrixStabilityDiagnostics{T}
        @test diagnostics.status == InteractionMatrixStabilityValid
        @test diagnostics.minimum_first_window_column_norm == T(2)
        @test iszero(diagnostics.maximum_relative_column_difference)
        @test iszero(diagnostics.relative_frobenius_difference)
    end
end

@testset "noncontiguous views retain logical matrix orientation" begin
    input = fixture_input()
    first_storage = fill(NaN, 4, 4)
    second_storage = fill(NaN, 4, 4)
    first_view = @view first_storage[1:2:4, 1:2:4]
    second_view = @view second_storage[1:2:4, 1:2:4]
    first_view .= input.first_window
    second_view .= input.second_window
    plan = fixture_plan()
    diagnostics = InteractionMatrixStabilityDiagnostics(plan)

    process!(
        diagnostics,
        InteractionMatrixStabilityWorkspace(plan),
        plan,
        InteractionMatrixStabilityInput(first_view, second_view),
    )

    @test diagnostics.maximum_relative_column_difference ≈
          FIXTURE["expected_maximum_relative_column_difference"] atol =
              FIXTURE["absolute_tolerance_f64"]
    @test diagnostics.maximum_relative_column_difference_coordinate == 1
end

@testset "invalid preparation and inputs preserve diagnostics" begin
    valid_criteria = InteractionMatrixStabilityCriteria(0.1, 0.5, 0.5)
    @test_throws ArgumentError InteractionMatrixStabilityPlan(
        0,
        1,
        valid_criteria,
    )
    @test_throws ArgumentError InteractionMatrixStabilityPlan(
        1,
        0,
        valid_criteria,
    )
    @test_throws ArgumentError InteractionMatrixStabilityPlan(
        typemax(Int),
        2,
        valid_criteria,
    )
    for minimum in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError InteractionMatrixStabilityCriteria(
            minimum,
            1.0,
            1.0,
        )
    end
    for maximum in (-1.0, 2.1, NaN, Inf)
        @test_throws ArgumentError InteractionMatrixStabilityCriteria(
            0.1,
            maximum,
            1.0,
        )
        @test_throws ArgumentError InteractionMatrixStabilityCriteria(
            0.1,
            1.0,
            maximum,
        )
    end
    plan = fixture_plan(Float32)
    diagnostics = InteractionMatrixStabilityDiagnostics(plan)
    workspace = InteractionMatrixStabilityWorkspace(plan)
    process!(diagnostics, workspace, plan, fixture_input(Float32))
    original = diagnostic_values(diagnostics)
    valid = fixture_input(Float32)
    invalid_inputs = (
        (
            InteractionMatrixStabilityInput(
                zeros(Float32, 1, 2),
                valid.second_window,
            ),
            DimensionMismatch,
        ),
        (
            InteractionMatrixStabilityInput(
                valid.first_window,
                zeros(Float32, 1, 2),
            ),
            DimensionMismatch,
        ),
        (
            InteractionMatrixStabilityInput(
                Float32[NaN 0; 0 1],
                valid.second_window,
            ),
            ArgumentError,
        ),
        (
            InteractionMatrixStabilityInput(
                valid.first_window,
                Float32[1 0; 0 Inf],
            ),
            ArgumentError,
        ),
        (
            InteractionMatrixStabilityInput(
                Float32[floatmax(Float32) 0; 0 1],
                Float32[-floatmax(Float32) 0; 0 2],
            ),
            ArgumentError,
        ),
    )
    for (input, error) in invalid_inputs
        @test_throws error process!(
            diagnostics,
            workspace,
            plan,
            input,
        )
        @test diagnostic_values(diagnostics) == original
    end
end

@testset "interaction-matrix stability declaration preserves the FGN contract" begin
    @test algorithm_label(InteractionMatrixStabilityF32) ==
          "interaction-matrix-stability-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        InteractionMatrixStabilityF32,
    ) === InteractionMatrixStabilityFailure
    @test algorithm_property_policy(InteractionMatrixStabilityF32) isa
          RuntimeDeclaredProperties

    configuration = algorithm_configuration(InteractionMatrixStabilityF32)
    @test map(field -> field.name, configuration) == (
        :measurement_count,
        :corrector_coordinate_count,
        :minimum_column_norm,
        :interaction_matrix_schema,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        configuration[[1, 2, 4]],
    )
    @test configuration[3].update isa ConstructionConfiguration

    ports = algorithm_ports(InteractionMatrixStabilityF32)
    @test map(port -> port.name, ports) == (
        Symbol("first-window"),
        Symbol("second-window"),
        :status,
        Symbol("status-coordinate"),
        Symbol("minimum-column-norms"),
        Symbol("maximum-relative-column-difference"),
        Symbol("maximum-relative-column-difference-coordinate"),
        Symbol("relative-frobenius-difference"),
    )
    @test getproperty.(ports, :element_type) == (
        Float32,
        Float32,
        UInt8,
        UInt64,
        Float32,
        Float32,
        UInt64,
        Float32,
    )
    @test all(port -> isnothing(port.metadata_source), ports[1:2])
    @test all(
        port -> port.metadata_source === Symbol("second-window"),
        ports[3:8],
    )

    prepared = stability_prepared()
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == (
        (2, 2),
        (2, 2),
        (1,),
        (1,),
        (2,),
        (1,),
        (1,),
        (1,),
    )
    @test getproperty.(formats, :schema) == (
        STABILITY_MATRIX_SCHEMA,
        STABILITY_MATRIX_SCHEMA,
        INTERACTION_MATRIX_STABILITY_STATUS_V1,
        CORRECTOR_COORDINATE_V1,
        INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
        INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
        CORRECTOR_COORDINATE_V1,
        INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1,
    )

    properties = property_infos(prepared.plan)
    @test getproperty.(properties, :id) == (UInt32(0), UInt32(1))
    @test getproperty.(properties, :name) == (
        Symbol("maximum-relative-column-difference"),
        Symbol("maximum-relative-frobenius-difference"),
    )
    @test all(property -> property.unit == "1", properties)
    @test all(property -> property.update isa RuntimeProperty, properties)
    @test all(property -> property.default_value === 0.5f0, properties)
    @test all(property -> property.constraint == PropertyRange(0.0f0, 2.0f0), properties)
end

@testset "declared stability outputs match the frozen Rust oracle" begin
    prepared = stability_prepared()
    input = fixture_input(Float32)
    outputs = stability_outputs()

    @test process!(outputs..., prepared, input.first_window, input.second_window) ===
          nothing
    @test outputs[1] == UInt8[2]
    @test outputs[2] == UInt64[0]
    @test outputs[3] == Float32[1, 2]
    @test reinterpret(UInt32, outputs[3]) == UInt32[0x3f800000, 0x40000000]
    @test reinterpret(UInt32, outputs[4]) == UInt32[0x3f2aaaab]
    @test outputs[5] == UInt64[0]
    @test reinterpret(UInt32, outputs[6]) == UInt32[0x3f2aaaab]
end

@testset "runtime stability thresholds replace atomically" begin
    prepared = stability_prepared()
    original = prepared.plan
    replacement = prepare_properties(
        original,
        (
            PropertyAssignment(0, 1.0f0),
            PropertyAssignment(1, 0.5f0),
        ),
    )
    @test replacement !== original
    @test property_values(original) == NamedTuple{
        (
            Symbol("maximum-relative-column-difference"),
            Symbol("maximum-relative-frobenius-difference"),
        ),
    }((0.5f0, 0.5f0))
    @test property_value(replacement, 0) === 1.0f0
    @test property_value(replacement, 1) === 0.5f0

    prepared.plan = replacement
    input = fixture_input(Float32)
    outputs = stability_outputs()
    @test process!(outputs..., prepared, input.first_window, input.second_window) ===
          nothing
    @test outputs[1] == UInt8[3]
    @test outputs[2] == UInt64[0]

    active = prepared.plan
    @test_throws PropertyConstraintError prepare_properties(
        active,
        (PropertyAssignment(0, 2.1f0),),
    )
    @test_throws PropertyTypeError prepare_properties(
        active,
        (PropertyAssignment(1, 0.5),),
    )
    @test prepared.plan === active
end

@testset "declared stability rejection preserves every output" begin
    prepared = stability_prepared()
    valid = fixture_input(Float32)
    outputs = stability_outputs()
    original = map(copy, outputs)

    rejected_inputs = (
        (
            Float32[NaN 0; 0 1],
            valid.second_window,
            InteractionMatrixStabilityNonFiniteFirstWindow,
        ),
        (
            valid.first_window,
            Float32[1 0; 0 Inf],
            InteractionMatrixStabilityNonFiniteSecondWindow,
        ),
        (
            Float32[floatmax(Float32) 0; 0 1],
            Float32[-floatmax(Float32) 0; 0 2],
            InteractionMatrixStabilityNonFiniteCalculation,
        ),
    )
    for (first_window, second_window, failure) in rejected_inputs
        @test process!(outputs..., prepared, first_window, second_window) === failure
        @test outputs == original
    end

    @test_throws DimensionMismatch process!(
        outputs...,
        prepared,
        zeros(Float32, 1, 2),
        valid.second_window,
    )
    @test outputs == original

    shared = fill(71.0f0, 2)
    aliased_outputs = (
        outputs[1],
        outputs[2],
        shared,
        @view(shared[1:1]),
        outputs[5],
        outputs[6],
    )
    @test_throws ArgumentError process!(
        aliased_outputs...,
        prepared,
        valid.first_window,
        valid.second_window,
    )
    @test shared == fill(71.0f0, 2)
end

@testset "declared stability supports strided storage and safe input-output aliasing" begin
    prepared = stability_prepared()
    input = fixture_input(Float32)
    first_storage = fill(83.0f0, 4, 4)
    second_storage = fill(84.0f0, 4, 4)
    first_window = @view first_storage[1:2:4, 2:2:4]
    second_window = @view second_storage[2:2:4, 1:2:4]
    first_window .= input.first_window
    second_window .= input.second_window

    status_storage = fill(typemax(UInt8), 3)
    coordinate_storage = fill(typemax(UInt64), 3)
    norms_storage = fill(71.0f0, 4)
    column_storage = fill(72.0f0, 3)
    column_coordinate_storage = fill(typemax(UInt64), 3)
    frobenius_storage = fill(73.0f0, 3)
    outputs = (
        @view(status_storage[2:2]),
        @view(coordinate_storage[2:2]),
        @view(norms_storage[1:2:3]),
        @view(column_storage[2:2]),
        @view(column_coordinate_storage[2:2]),
        @view(frobenius_storage[2:2]),
    )
    @test process!(outputs..., prepared, first_window, second_window) === nothing
    @test only(outputs[1]) == UInt8(2)
    @test only(outputs[2]) == UInt64(0)
    @test collect(outputs[3]) == Float32[1, 2]
    @test reinterpret(UInt32, collect(outputs[4])) == UInt32[0x3f2aaaab]
    @test only(outputs[5]) == UInt64(0)
    @test reinterpret(UInt32, collect(outputs[6])) == UInt32[0x3f2aaaab]

    aliased_first = Float32[1 0; 0 1]
    maximum_column = @view vec(aliased_first)[1:1]
    aliased_outputs = (
        zeros(UInt8, 1),
        zeros(UInt64, 1),
        zeros(Float32, 2),
        maximum_column,
        zeros(UInt64, 1),
        zeros(Float32, 1),
    )
    @test process!(
        aliased_outputs...,
        prepared,
        aliased_first,
        Float32[2 0; 0 2],
    ) === nothing
    @test reinterpret(UInt32, aliased_outputs[4]) == UInt32[0x3f2aaaab]
end

@testset "stability checked and admitted paths allocate nothing after warmup" begin
    prepared = stability_prepared()
    valid = fixture_input(Float32)
    outputs = stability_outputs()
    rejected_inputs = (
        (Float32[NaN 0; 0 1], valid.second_window),
        (valid.first_window, Float32[1 0; 0 Inf]),
        (
            Float32[floatmax(Float32) 0; 0 1],
            Float32[-floatmax(Float32) 0; 0 2],
        ),
    )

    process!(outputs..., prepared, valid.first_window, valid.second_window)
    @test stability_checked_allocation_bytes(
        outputs...,
        prepared,
        valid.first_window,
        valid.second_window,
    ) == 0
    for (first_window, second_window) in rejected_inputs
        process!(outputs..., prepared, first_window, second_window)
        @test stability_checked_allocation_bytes(
            outputs...,
            prepared,
            first_window,
            second_window,
        ) == 0
        FilterGraphAlgorithms._process_admitted!(
            outputs...,
            prepared,
            first_window,
            second_window,
            FilterGraphAlgorithms._NO_SAMPLE_METADATA,
            FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        )
        @test stability_admitted_allocation_bytes(
            outputs...,
            prepared,
            first_window,
            second_window,
        ) == 0
    end
    @test @inferred InteractionMatrixStabilityFailure FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        valid.first_window,
        valid.second_window,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test stability_admitted_allocation_bytes(
        outputs...,
        prepared,
        valid.first_window,
        valid.second_window,
    ) == 0

    workspace = prepared.workspace
    @test stability_commit_allocation_bytes(workspace) == 0
    @test stability_discard_allocation_bytes(workspace) == 0
    @test stability_reset_allocation_bytes(workspace) == 0
end

@testset "stability public API exports" begin
    for name in (
        :CORRECTOR_COORDINATE_V1,
        :INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
        :INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
        :INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1,
        :INTERACTION_MATRIX_STABILITY_STATUS_V1,
        :InteractionMatrixStabilityCriteria,
        :InteractionMatrixStabilityDiagnostics,
        :InteractionMatrixStabilityF32,
        :InteractionMatrixStabilityF32Configuration,
        :InteractionMatrixStabilityFailure,
        :InteractionMatrixStabilityInput,
        :InteractionMatrixStabilityNonFiniteCalculation,
        :InteractionMatrixStabilityNonFiniteFirstWindow,
        :InteractionMatrixStabilityNonFiniteSecondWindow,
        :InteractionMatrixStabilityPlan,
        :InteractionMatrixStabilityWorkspace,
        :InteractionMatrixStabilityStatus,
        :InteractionMatrixStabilityValid,
        :InteractionMatrixStabilityInsufficientColumnNorm,
        :InteractionMatrixStabilityExcessiveRelativeColumnDifference,
        :InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
