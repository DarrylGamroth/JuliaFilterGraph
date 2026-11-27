module DocrimeCovarianceTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: _process_admitted!, commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "docrime_covariance.toml"),
)

const MEASUREMENTS_SCHEMA = "example/test-measurements/1"
const CORRECTOR_SAMPLE_SCHEMA = "example/test-corrector-sample/1"
const CROSS_COVARIANCE_SCHEMA =
    "example/test-measurement-corrector-covariance/1"
const CORRECTOR_COVARIANCE_SCHEMA =
    "example/test-corrector-covariance/1"

row_major_matrix(values, rows, columns, ::Type{T}) where {T} =
    permutedims(reshape(T.(values), columns, rows))

function covariance_outputs(::Type{T}=Float32) where {T}
    return (
        fill(T(NaN), 2, 2),
        fill(T(NaN), 2, 2),
        fill(typemax(UInt64), 1),
        fill(true, 1),
    )
end

function covariance_prepared(sample_limit=UInt64(3))
    return prepare_algorithm(
        DocrimeCovarianceF32,
        DocrimeCovarianceF32Configuration(
            measurement_count=2,
            corrector_coordinate_count=2,
            sample_limit=sample_limit,
            measurements_schema=MEASUREMENTS_SCHEMA,
            corrector_sample_schema=CORRECTOR_SAMPLE_SCHEMA,
            cross_covariance_schema=CROSS_COVARIANCE_SCHEMA,
            corrector_covariance_schema=CORRECTOR_COVARIANCE_SCHEMA,
        ),
    )
end

function direct_allocation_bytes(outputs, workspace, plan, measurements, corrector)
    return @allocated process!(
        outputs...,
        workspace,
        plan,
        measurements,
        corrector,
    )
end

function admitted_allocation_bytes(outputs, prepared, measurements, corrector)
    absent = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated _process_admitted!(
        outputs...,
        prepared,
        measurements,
        corrector,
        absent,
        absent,
    )
end

@testset "centered covariance matches the frozen FGN oracle" begin
    for T in (Float32, Float64)
        plan = DocrimeCovariancePlan(T, 2, 2)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = covariance_outputs(T)
        for frame in FIXTURE["frames"]
            @test process!(
                outputs...,
                workspace,
                plan,
                T.(frame["measurements"]),
                T.(frame["corrector_sample"]),
            ) === nothing
            @test outputs[1] == row_major_matrix(
                frame["cross_covariance_sum"],
                2,
                2,
                T,
            )
            @test outputs[2] == row_major_matrix(
                frame["corrector_covariance_sum"],
                2,
                2,
                T,
            )
            @test outputs[3] == UInt64[frame["sample_count"]]
            @test outputs[4] == Bool[false]
            @test docrime_covariance_sample_count(workspace) ==
                  UInt64(frame["sample_count"] - 1)
            commit!(workspace)
            @test docrime_measurement_mean(workspace) ==
                  T.(frame["measurement_mean"])
            @test docrime_corrector_sample_mean(workspace) ==
                  T.(frame["corrector_sample_mean"])
        end
        @test docrime_measurement_corrector_cross_covariance_sum(workspace) ==
              row_major_matrix(
            FIXTURE["frames"][3]["cross_covariance_sum"],
            2,
            2,
            T,
        )
        @test docrime_corrector_covariance_sum(workspace) ==
              row_major_matrix(
            FIXTURE["frames"][3]["corrector_covariance_sum"],
            2,
            2,
            T,
        )
    end
end

@testset "candidate history commits, discards, and resets" begin
    plan = DocrimeCovariancePlan(Float32, 2, 2; sample_limit=2)
    workspace = DocrimeCovarianceWorkspace(plan)
    outputs = covariance_outputs()
    first = (Float32[10, 4.5], Float32[1, 2])
    second = (Float32[16, -0.5], Float32[3, 0])

    process!(outputs..., workspace, plan, first...)
    @test outputs[4] == [false]
    @test docrime_covariance_sample_count(workspace) == 0
    discard!(workspace)
    @test docrime_covariance_sample_count(workspace) == 0
    @test docrime_measurement_mean(workspace) == zeros(Float32, 2)

    process!(outputs..., workspace, plan, first...)
    commit!(workspace)
    @test docrime_covariance_sample_count(workspace) == 1
    retained_mean = docrime_measurement_mean(workspace)
    retained_cross = docrime_measurement_corrector_cross_covariance_sum(workspace)

    process!(outputs..., workspace, plan, second...)
    @test outputs[4] == [true]
    discard!(workspace)
    @test docrime_covariance_sample_count(workspace) == 1
    @test docrime_measurement_mean(workspace) == retained_mean
    @test docrime_measurement_corrector_cross_covariance_sum(workspace) ==
          retained_cross

    process!(outputs..., workspace, plan, second...)
    commit!(workspace)
    @test docrime_covariance_sample_count(workspace) == 2
    retained_outputs = map(copy, outputs)
    @test process!(
        outputs...,
        workspace,
        plan,
        Float32[NaN, 0],
        Float32[Inf, 0],
    ) === DocrimeCovarianceSampleLimitReached
    @test outputs == retained_outputs

    @test reset!(workspace) === workspace
    @test docrime_covariance_sample_count(workspace) == 0
    @test docrime_measurement_mean(workspace) == zeros(Float32, 2)
    @test docrime_corrector_covariance_sum(workspace) == zeros(Float32, 2, 2)
end

@testset "constant measurement offsets preserve centered sums" begin
    plan = DocrimeCovariancePlan(Float64, 1, 2)
    first = DocrimeCovarianceWorkspace(plan)
    shifted = DocrimeCovarianceWorkspace(plan)
    first_outputs = (zeros(1, 2), zeros(2, 2), zeros(UInt64, 1), falses(1))
    shifted_outputs = map(similar, first_outputs)

    for (measurement, corrector) in (
        (2.0, [1.0, -1.0]),
        (8.0, [2.0, 1.0]),
        (7.0, [4.0, -2.0]),
    )
        process!(first_outputs..., first, plan, [measurement], corrector)
        process!(shifted_outputs..., shifted, plan, [measurement + 100], corrector)
        commit!(first)
        commit!(shifted)
    end
    @test first_outputs[1] == shifted_outputs[1]
    @test first_outputs[2] == shifted_outputs[2]
end

@testset "strided views preserve covariance orientation without allocation" begin
    plan = DocrimeCovariancePlan(Float32, 2, 2)
    workspace = DocrimeCovarianceWorkspace(plan)
    measurement_storage = Float32[10, 71, 4.5, 72]
    corrector_storage = Float32[1, 73, 2, 74]
    measurements = @view measurement_storage[1:2:3]
    corrector = @view corrector_storage[1:2:3]
    cross_storage = fill(NaN32, 4, 4)
    corrector_covariance_storage = fill(NaN32, 4, 4)
    count_storage = fill(typemax(UInt64), 2)
    complete_storage = fill(true, 2)
    outputs = (
        @view(cross_storage[1:2:3, 1:2:3]),
        @view(corrector_covariance_storage[1:2:3, 1:2:3]),
        @view(count_storage[1:1]),
        @view(complete_storage[1:1]),
    )

    @test process!(
        outputs...,
        workspace,
        plan,
        measurements,
        corrector,
    ) === nothing
    @test outputs[1] == zeros(Float32, 2, 2)
    @test outputs[2] == zeros(Float32, 2, 2)
    commit!(workspace)
    @test direct_allocation_bytes(
        outputs,
        workspace,
        plan,
        measurements,
        corrector,
    ) == 0
end

@testset "checked storage and preparation reject invalid use" begin
    @test_throws ArgumentError DocrimeCovariancePlan(Float32, 0, 1)
    @test_throws ArgumentError DocrimeCovariancePlan(Float32, 1, 0)
    @test_throws ArgumentError DocrimeCovariancePlan(
        Float32,
        typemax(Int),
        2,
    )
    @test_throws ArgumentError DocrimeCovariancePlan(
        Float32,
        typemax(UInt64),
        1,
    )
    @test_throws ArgumentError DocrimeCovariancePlan(
        Float32,
        1,
        1;
        sample_limit=0,
    )
    plan = DocrimeCovariancePlan(Float32, 2, 2)
    workspace = DocrimeCovarianceWorkspace(plan)
    outputs = covariance_outputs()
    measurements = Float32[10, 4.5]
    corrector = Float32[1, 2]
    @test_throws DimensionMismatch process!(
        outputs...,
        workspace,
        plan,
        zeros(Float32, 1),
        corrector,
    )
    @test_throws DimensionMismatch process!(
        zeros(Float32, 1, 2),
        outputs[2:end]...,
        workspace,
        plan,
        measurements,
        corrector,
    )
    @test_throws ArgumentError process!(
        workspace.candidate_cross_covariance_sum,
        outputs[2:end]...,
        workspace,
        plan,
        measurements,
        corrector,
    )
    @test_throws ArgumentError process!(
        outputs[1],
        outputs[1],
        outputs[3],
        outputs[4],
        workspace,
        plan,
        measurements,
        corrector,
    )
    maximum_float_extent = div(typemax(UInt32), UInt32(sizeof(Float32)))
    @test_throws ArgumentError prepare_algorithm(
        DocrimeCovarianceF32,
        DocrimeCovarianceF32Configuration(
            measurement_count=Int(maximum_float_extent) + 1,
            corrector_coordinate_count=1,
            sample_limit=nothing,
            measurements_schema=MEASUREMENTS_SCHEMA,
            corrector_sample_schema=CORRECTOR_SAMPLE_SCHEMA,
            cross_covariance_schema=CROSS_COVARIANCE_SCHEMA,
            corrector_covariance_schema=CORRECTOR_COVARIANCE_SCHEMA,
        ),
    )
end

@testset "expected failures preserve outputs and committed history" begin
    plan = DocrimeCovariancePlan(Float32, 1, 1)
    workspace = DocrimeCovarianceWorkspace(plan)
    outputs = (
        fill(71.0f0, 1, 1),
        fill(72.0f0, 1, 1),
        fill(UInt64(73), 1),
        fill(true, 1),
    )
    process!(outputs..., workspace, plan, Float32[1], Float32[1])
    commit!(workspace)
    retained_outputs = map(copy, outputs)
    retained_mean = docrime_measurement_mean(workspace)

    for (measurements, corrector, failure) in (
        (
            Float32[NaN],
            Float32[0],
            DocrimeCovarianceNonFiniteMeasurements,
        ),
        (
            Float32[0],
            Float32[Inf],
            DocrimeCovarianceNonFiniteCorrectorSample,
        ),
        (
            Float32[-floatmax(Float32)],
            Float32[-floatmax(Float32)],
            DocrimeCovarianceNonFiniteCalculation,
        ),
    )
        @test process!(
            outputs...,
            workspace,
            plan,
            measurements,
            corrector,
        ) === failure
        @test outputs == retained_outputs
        discard!(workspace)
        @test docrime_measurement_mean(workspace) == retained_mean
    end

    workspace.committed_sample_count = UInt64(1) << 24
    workspace.candidate_sample_count = workspace.committed_sample_count
    @test process!(
        outputs...,
        workspace,
        plan,
        Float32[0],
        Float32[0],
    ) === DocrimeCovarianceSampleCountNotRepresentable
    workspace.committed_sample_count = typemax(UInt64)
    workspace.candidate_sample_count = typemax(UInt64)
    @test process!(
        outputs...,
        workspace,
        plan,
        Float32[0],
        Float32[0],
    ) === DocrimeCovarianceSampleCountOverflow
end

@testset "declaration preserves FGN Ports, metadata, and finite window" begin
    @test algorithm_label(DocrimeCovarianceF32) == "docrime-covariance-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(DocrimeCovarianceF32) ===
          DocrimeCovarianceFailure
    configuration = algorithm_configuration(DocrimeCovarianceF32)
    @test map(field -> field.name, configuration) == (
        :measurement_count,
        :corrector_coordinate_count,
        :sample_limit,
        :measurements_schema,
        :corrector_sample_schema,
        :cross_covariance_schema,
        :corrector_covariance_schema,
    )
    @test configuration[3].update isa ConstructionConfiguration
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        (
            configuration[1],
            configuration[2],
            configuration[4],
            configuration[5],
            configuration[6],
            configuration[7],
        ),
    )
    ports = algorithm_ports(DocrimeCovarianceF32)
    @test map(port -> port.name, ports) == (
        :measurements,
        Symbol("corrector-sample"),
        Symbol("measurement-corrector-cross-covariance-sum"),
        Symbol("corrector-covariance-sum"),
        Symbol("sample-count"),
        :complete,
    )
    @test getproperty.(ports, :element_type) ==
          (Float32, Float32, Float32, Float32, UInt64, Bool)
    @test all(port -> port.metadata_source === :measurements, ports[3:6])

    prepared = covariance_prepared(UInt64(2))
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) ==
          ((2,), (2,), (2, 2), (2, 2), (1,), (1,))
    @test getproperty.(formats, :schema) == (
        MEASUREMENTS_SCHEMA,
        CORRECTOR_SAMPLE_SCHEMA,
        CROSS_COVARIANCE_SCHEMA,
        CORRECTOR_COVARIANCE_SCHEMA,
        DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
        DOCRIME_COVARIANCE_COMPLETE_V1,
    )
    outputs = covariance_outputs()
    @test process!(outputs..., prepared, Float32[10, 4.5], Float32[1, 2]) ===
          nothing
    @test outputs[3] == UInt64[1]
    @test outputs[4] == [false]
    @test docrime_covariance_sample_count(prepared.workspace) == 1
    @test process!(outputs..., prepared, Float32[16, -0.5], Float32[3, 0]) ===
          nothing
    @test outputs[3] == UInt64[2]
    @test outputs[4] == [true]
    retained = map(copy, outputs)
    @test process!(outputs..., prepared, Float32[0, 0], Float32[0, 0]) ===
          DocrimeCovarianceSampleLimitReached
    @test outputs == retained
    @test reset!(prepared) === prepared
    @test docrime_covariance_sample_count(prepared.workspace) == 0
end

@testset "all admitted outcomes allocate no heap after warmup" begin
    plan = DocrimeCovariancePlan(Float32, 1, 1)
    workspace = DocrimeCovarianceWorkspace(plan)
    outputs = (zeros(Float32, 1, 1), zeros(Float32, 1, 1), zeros(UInt64, 1), falses(1))
    valid = (Float32[1], Float32[1])
    failures = (
        (Float32[NaN], Float32[0], DocrimeCovarianceNonFiniteMeasurements),
        (Float32[0], Float32[Inf], DocrimeCovarianceNonFiniteCorrectorSample),
    )
    process!(outputs..., workspace, plan, valid...)
    commit!(workspace)
    @test direct_allocation_bytes(outputs, workspace, plan, valid...) == 0
    for (measurements, corrector, failure) in failures
        @test process!(
            outputs...,
            workspace,
            plan,
            measurements,
            corrector,
        ) === failure
        @test direct_allocation_bytes(
            outputs,
            workspace,
            plan,
            measurements,
            corrector,
        ) == 0
    end
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0

    prepared = covariance_prepared(nothing)
    declared_outputs = covariance_outputs()
    measurements = Float32[10, 4.5]
    corrector = Float32[1, 2]
    process!(declared_outputs..., prepared, measurements, corrector)
    @test admitted_allocation_bytes(
        declared_outputs,
        prepared,
        measurements,
        corrector,
    ) == 0
    nonfinite = Float32[NaN, 0]
    @test _process_admitted!(
        declared_outputs...,
        prepared,
        nonfinite,
        corrector,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === DocrimeCovarianceNonFiniteMeasurements
    @test admitted_allocation_bytes(
        declared_outputs,
        prepared,
        nonfinite,
        corrector,
    ) == 0
end

@testset "covariance public API exports" begin
    for name in (
        :DocrimeCovariancePlan,
        :DocrimeCovarianceWorkspace,
        :DocrimeCovarianceFailure,
        :DocrimeCovarianceF32,
        :DocrimeCovarianceF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end


# A finite Float64 product and addend fit within this exact intermediate span.
function fma_oracle(a::T, b::T, c::T) where {T<:Union{Float32,Float64}}
    return setprecision(BigFloat, 4096) do
        T(BigFloat(a) * BigFloat(b) + BigFloat(c))
    end
end

@testset "pointwise covariance agrees with the unfused scalar reference" begin
    for T in (Float32, Float64), (measurements_count, correctors_count) in
        ((4, 3), (253, 5), (277, 9), (512, 17))
        plan = DocrimeCovariancePlan(T, measurements_count, correctors_count)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, measurements_count, correctors_count),
                   zeros(T, correctors_count, correctors_count), zeros(UInt64, 1), falses(1))
        mean_measurement = zeros(T, measurements_count)
        mean_corrector = zeros(T, correctors_count)
        expected_cross = zeros(T, measurements_count, correctors_count)
        expected_corrector = zeros(T, correctors_count, correctors_count)
        cross_scale = zeros(T, measurements_count, correctors_count)
        corrector_scale = zeros(T, correctors_count, correctors_count)
        U = T === Float32 ? UInt32 : UInt64
        for frame in 1:3
            measurements = T[T((index % 19) - 9) / T(7) + T(frame) / T(11)
                             for index in 1:measurements_count]
            corrector = T[T((index % 11) - 5) / T(13) - T(frame) / T(17)
                          for index in 1:correctors_count]
            count = T(frame)
            weight = (count - one(T)) / count
            measurement_delta = measurements - mean_measurement
            corrector_delta = corrector - mean_corrector
            for index in eachindex(mean_measurement)
                mean_measurement[index] = mean_measurement[index] + measurement_delta[index] / count
            end
            for index in eachindex(mean_corrector)
                mean_corrector[index] = mean_corrector[index] + corrector_delta[index] / count
            end
            for column in 1:correctors_count, row in 1:measurements_count
                expected_cross[row, column] = expected_cross[row, column] +
                    weight * measurement_delta[row] * corrector_delta[column]
                cross_scale[row, column] +=
                    abs(weight * measurement_delta[row]) * abs(corrector_delta[column])
            end
            for column in 1:correctors_count, row in 1:column
                value = expected_corrector[row, column] +
                    weight * corrector_delta[row] * corrector_delta[column]
                expected_corrector[row, column] = value
                expected_corrector[column, row] = value
                scale = corrector_scale[row, column] +
                    abs(weight * corrector_delta[row]) * abs(corrector_delta[column])
                corrector_scale[row, column] = scale
                corrector_scale[column, row] = scale
            end
            @test process!(outputs..., workspace, plan, measurements, corrector) === nothing
            # With x = rounded(weight*delta), both paths accumulate x*y. Their
            # difference is bounded by (gamma_n + gamma_2n)*sum(abs(x*y)).
            # 4gamma_2n also covers rounding the positive scale in these normal,
            # finite fixtures (n <= 3); it does not impose a broad absolute floor.
            u = eps(T) / T(2)
            gamma = T(2 * frame) * u / (one(T) - T(2 * frame) * u)
            @test all(abs.(outputs[1] .- expected_cross) .<= T(4) * gamma .* cross_scale)
            @test all(abs.(outputs[2] .- expected_corrector) .<= T(4) * gamma .* corrector_scale)
            @test outputs[3] == UInt64[frame]
            commit!(workspace)
            @test reinterpret(U, workspace.committed_measurement_mean) == reinterpret(U, mean_measurement)
            @test reinterpret(U, workspace.committed_corrector_sample_mean) == reinterpret(U, mean_corrector)
        end
    end
end

@testset "covariance FMA trajectory matches the high-precision rounding oracle" begin
    for T in (Float32, Float64)
        measurements_count, correctors_count = 9, 5
        plan = DocrimeCovariancePlan(T, measurements_count, correctors_count)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, measurements_count, correctors_count),
            zeros(T, correctors_count, correctors_count), zeros(UInt64, 1), falses(1))
        measurement_mean, corrector_mean = zeros(T, measurements_count), zeros(T, correctors_count)
        expected_cross = zeros(T, measurements_count, correctors_count)
        expected_corrector = zeros(T, correctors_count, correctors_count)
        U = T === Float32 ? UInt32 : UInt64
        for frame in 1:12
            measurements = T[T((index % 7) - 3) / T(11) + T(frame) / T(13)
                             for index in 1:measurements_count]
            corrector = T[T((index % 5) - 2) / T(17) - T(frame) / T(19)
                          for index in 1:correctors_count]
            measurement_delta = measurements - measurement_mean
            corrector_delta = corrector - corrector_mean
            count = T(frame)
            weight = (count - one(T)) / count
            previous_cross, previous_corrector = copy(expected_cross), copy(expected_corrector)
            # Delta subtraction, division, mean addition, and the first weighted
            # product retain their T rounding; only the second product/add fuse.
            measurement_mean = T[measurement_mean[index] + measurement_delta[index] / count
                                 for index in 1:measurements_count]
            corrector_mean = T[corrector_mean[index] + corrector_delta[index] / count
                              for index in 1:correctors_count]
            for column in 1:correctors_count, row in 1:measurements_count
                expected_cross[row, column] = fma_oracle(weight * measurement_delta[row],
                    corrector_delta[column], expected_cross[row, column])
            end
            for column in 1:correctors_count, row in 1:column
                value = fma_oracle(weight * corrector_delta[row],
                    corrector_delta[column], expected_corrector[row, column])
                expected_corrector[row, column] = value
                expected_corrector[column, row] = value
            end
            @test process!(outputs..., workspace, plan, measurements, corrector) === nothing
            @test reinterpret(U, vec(outputs[1])) == reinterpret(U, vec(expected_cross))
            @test reinterpret(U, vec(outputs[2])) == reinterpret(U, vec(expected_corrector))
            @test reinterpret(U, vec(workspace.committed_cross_covariance_sum)) ==
                  reinterpret(U, vec(previous_cross))
            @test reinterpret(U, vec(workspace.committed_corrector_covariance_sum)) ==
                  reinterpret(U, vec(previous_corrector))
            @test outputs[3] == UInt64[frame]
            @test docrime_covariance_sample_count(workspace) == frame - 1
            if frame == 3
                discard!(workspace)
                @test process!(outputs..., workspace, plan, measurements, corrector) === nothing
                @test reinterpret(U, vec(outputs[1])) == reinterpret(U, vec(expected_cross))
                @test reinterpret(U, vec(outputs[2])) == reinterpret(U, vec(expected_corrector))
            end
            commit!(workspace)
            @test reinterpret(U, workspace.committed_measurement_mean) ==
                  reinterpret(U, measurement_mean)
            @test reinterpret(U, workspace.committed_corrector_sample_mean) ==
                  reinterpret(U, corrector_mean)
            @test reinterpret(U, vec(workspace.committed_cross_covariance_sum)) ==
                  reinterpret(U, vec(expected_cross))
            @test reinterpret(U, vec(workspace.committed_corrector_covariance_sum)) ==
                  reinterpret(U, vec(expected_corrector))
        end
    end
end

@testset "cross-covariance SIMD preserves overlapping inputs and shared candidate scratch" begin
    for T in (Float32, Float64), overlap in (:caller, :scratch)
        count = 277
        plan = DocrimeCovariancePlan(T, count, 1)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, count, 1), zeros(T, 1, 1), zeros(UInt64, 1), falses(1))
        for frame in 1:2
            measurements = T[T(index % 19) / T(13) + T(frame) / T(17) for index in 1:count]
            @test process!(outputs..., workspace, plan, measurements, T[frame / 7]) === nothing
            commit!(workspace)
        end
        reference = deepcopy(workspace)
        reference_outputs = map(copy, outputs)
        measurements = T[T(index % 23) / T(11) - T(3) / T(17) for index in 1:count]
        corrector = T[3 / 7]
        @test process!(reference_outputs..., reference, plan, measurements, corrector) === nothing
        if overlap === :caller
            measurement_input = @view vec(outputs[1])[1:count]
            corrector_input = @view vec(outputs[2])[1:1]
            copyto!(measurement_input, measurements)
            copyto!(corrector_input, corrector)
        else
            # Public mutable scratch can share backing storage. This call must
            # retain scalar alias semantics; no ivdep assumption is permitted.
            # Do not commit this manually shared layout: a buffer swap would
            # make the delta scratch alias committed covariance storage.
            workspace.candidate_cross_covariance_sum =
                reshape(workspace.measurement_delta, count, 1)
            measurement_input, corrector_input = measurements, corrector
        end
        @test process!(outputs..., workspace, plan, measurement_input, corrector_input) === nothing
        @test isequal(outputs, reference_outputs)
        for field in fieldnames(typeof(workspace))
            startswith(string(field), "committed_") || continue
            @test isequal(getfield(workspace, field), getfield(reference, field))
        end
        if overlap === :caller
            # Restore input values overwritten by successful publication.
            copyto!(measurement_input, measurements)
            copyto!(corrector_input, corrector)
        end
        @test direct_allocation_bytes(outputs, workspace, plan, measurement_input, corrector_input) == 0
    end
end

@testset "cross-covariance FMA cancels an overflowing product in a valid trajectory" begin
    for T in (Float32, Float64)
        plan = DocrimeCovariancePlan(T, 1, 1)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, 1, 1), zeros(T, 1, 1), zeros(UInt64, 1), falses(1))
        first_measurement = zero(T)
        second_measurement = floatmax(T) / T(2)
        for (measurement, corrector) in ((first_measurement, zero(T)), (second_measurement, T(-2)))
            @test process!(outputs..., workspace, plan, T[measurement], T[corrector]) === nothing
            commit!(workspace)
        end
        measurement, corrector = T(0.75) * floatmax(T), T(3)
        measurement_delta = measurement - only(workspace.committed_measurement_mean)
        corrector_delta = corrector - only(workspace.committed_corrector_sample_mean)
        weight = (T(3) - one(T)) / T(3)
        weighted_delta = weight * measurement_delta
        previous = only(workspace.committed_cross_covariance_sum)
        rounded_product = weighted_delta * corrector_delta
        @test !isfinite(rounded_product)
        @test !isfinite(previous + rounded_product)
        expected = fma_oracle(weighted_delta, corrector_delta, previous)
        @test isfinite(expected)
        @test process!(outputs..., workspace, plan, T[measurement], T[corrector]) === nothing
        @test only(outputs[1]) == expected
        @test only(workspace.committed_cross_covariance_sum) == previous
        @test docrime_covariance_sample_count(workspace) == 2
        commit!(workspace)
        @test only(workspace.committed_cross_covariance_sum) == expected
        @test docrime_covariance_sample_count(workspace) == 3
    end
end

@testset "pointwise covariance overflow preserves history and recovers" begin
    count = 277
    for T in (Float32, Float64), stage in (:measurement_delta, :corrector_delta, :cross),
        index in (1, 4, 5, 8, 9, count), recovery in (:discard, :retry)
        plan = DocrimeCovariancePlan(T, count, count)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, count, count), zeros(T, count, count),
                   zeros(UInt64, 1), falses(1))
        measurements = zeros(T, count)
        corrector = ones(T, count)
        if stage === :measurement_delta
            measurements[index] = -floatmax(T)
        elseif stage === :corrector_delta
            corrector[index] = -floatmax(T)
        end
        @test process!(outputs..., workspace, plan, measurements, corrector) === nothing
        commit!(workspace)
        retained_mean = docrime_measurement_mean(workspace)
        retained_corrector_mean = docrime_corrector_sample_mean(workspace)
        retained_cross = docrime_measurement_corrector_cross_covariance_sum(workspace)
        retained_corrector = docrime_corrector_covariance_sum(workspace)
        measurements[index] = stage === :corrector_delta ? zero(T) : floatmax(T)
        if stage === :corrector_delta
            corrector[index] = floatmax(T)
        elseif stage === :cross
            corrector .= T(4)
        end
        fill!(outputs[1], T(71))
        fill!(outputs[2], T(72))
        outputs[3][1] = UInt64(73)
        outputs[4][1] = true
        fill!(workspace.measurement_delta, T(NaN))
        fill!(workspace.corrector_sample_delta, T(NaN))
        fill!(workspace.candidate_cross_covariance_sum, T(NaN))
        fill!(workspace.candidate_corrector_covariance_sum, T(NaN))
        @test process!(outputs..., workspace, plan, measurements, corrector) ===
              DocrimeCovarianceNonFiniteCalculation
        @test direct_allocation_bytes(outputs, workspace, plan, measurements, corrector) == 0
        @test outputs[1] == fill(T(71), count, count)
        @test outputs[2] == fill(T(72), count, count)
        @test outputs[3] == UInt64[73]
        @test outputs[4] == Bool[true]
        @test docrime_covariance_sample_count(workspace) == 1
        @test docrime_measurement_mean(workspace) == retained_mean
        @test docrime_corrector_sample_mean(workspace) == retained_corrector_mean
        @test docrime_measurement_corrector_cross_covariance_sum(workspace) == retained_cross
        @test docrime_corrector_covariance_sum(workspace) == retained_corrector
        # Do not consume a rejected candidate in a later dependent stage.
        if stage === :measurement_delta
            @test all(isnan, workspace.corrector_sample_delta)
        elseif stage === :corrector_delta
            @test all(isnan, workspace.candidate_cross_covariance_sum)
        end
        @test all(isnan, workspace.candidate_corrector_covariance_sum)
        recovery === :discard && discard!(workspace)
        # Retrying without discard must also overwrite rejected candidate scratch.
        copyto!(measurements, retained_mean)
        copyto!(corrector, retained_corrector_mean)
        @test process!(outputs..., workspace, plan, measurements, corrector) === nothing
        @test outputs[1] == retained_cross
        @test outputs[2] == retained_corrector
        @test outputs[3] == UInt64[2]
        @test outputs[4] == Bool[false]
        @test direct_allocation_bytes(outputs, workspace, plan, measurements, corrector) == 0
        commit!(workspace)
        @test docrime_covariance_sample_count(workspace) == 2
        @test docrime_measurement_mean(workspace) == retained_mean
        @test docrime_corrector_sample_mean(workspace) == retained_corrector_mean
    end
end

@testset "covariance SIMD columns match the independent FMA oracle" begin
    for T in (Float32, Float64)
        n, m = 9, 33
        plan = DocrimeCovariancePlan(T, n, m)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, n, m), zeros(T, m, m), zeros(UInt64, 1), falses(1))
        mean_x, mean_y = zeros(T, n), zeros(T, m)
        cross, covariance = zeros(T, n, m), zeros(T, m, m)
        for frame in 1:10
            x = T[T(mod(i * frame, 31) - 15) / T(13) for i in 1:n]
            y = T[T(mod(i * i + 3 * frame * i, 37) - 18) / T(17) for i in 1:m]
            dx, dy = x - mean_x, y - mean_y
            previous_covariance = copy(covariance)
            mean_x = mean_x + dx / T(frame)
            mean_y = mean_y + dy / T(frame)
            weight = (T(frame) - one(T)) / T(frame)
            for column in 1:m, row in 1:n
                cross[row, column] = fma_oracle(weight * dx[row], dy[column], cross[row, column])
            end
            for column in 1:m, row in 1:column
                value = fma_oracle(weight * dy[row], dy[column], covariance[row, column])
                covariance[row, column] = covariance[column, row] = value
            end
            @test process!(outputs..., workspace, plan, x, y) === nothing
            @test isequal(outputs[1], cross)
            @test isequal(outputs[2], covariance)
            @test isequal(workspace.committed_corrector_covariance_sum, previous_covariance)
            @test direct_allocation_bytes(outputs, workspace, plan, x, y) == 0
            if frame == 4
                discard!(workspace)
                @test process!(outputs..., workspace, plan, x, y) === nothing
                @test isequal(outputs[2], covariance)
            end
            commit!(workspace)
            @test isequal(workspace.committed_corrector_sample_mean, mean_y)
            @test isequal(workspace.committed_measurement_mean, mean_x)
            @test docrime_covariance_sample_count(workspace) == frame
        end
    end
end

@testset "covariance validates every extent used by unchecked column loops" begin
    for T in (Float32, Float64), field in (
        :candidate_corrector_covariance_sum, :committed_corrector_covariance_sum,
        :corrector_sample_delta,
    ), difference in (-1, 1), dimension in (1, 2)
        field === :corrector_sample_delta && dimension == 2 && continue
        m = 9
        plan = DocrimeCovariancePlan(T, 1, m)
        workspace = DocrimeCovarianceWorkspace(plan)
        bad = if field === :corrector_sample_delta
            zeros(T, m + difference)
        elseif dimension == 1
            zeros(T, m + difference, m)
        else
            zeros(T, m, m + difference)
        end
        setfield!(workspace, field, bad)
        before = deepcopy(workspace)
        outputs = (fill(T(71), 1, m), fill(T(72), m, m), UInt64[73], Bool[true])
        @test_throws DimensionMismatch process!(outputs..., workspace, plan, T[0], zeros(T, m))
        @test outputs == (fill(T(71), 1, m), fill(T(72), m, m), UInt64[73], Bool[true])
        @test all(isequal(getfield(workspace, f), getfield(before, f)) for f in fieldnames(typeof(workspace)))
    end
end

@testset "covariance finite masks reject early, vector-boundary and final lanes" begin
    for T in (Float32, Float64), index in (1, 4, 5, 8, 9, 33), recovery in (:discard, :retry, :reset)
        m = 33
        plan = DocrimeCovariancePlan(T, 1, m)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, 1, m), zeros(T, m, m), zeros(UInt64, 1), falses(1))
        @test process!(outputs..., workspace, plan, T[0], zeros(T, m)) === nothing
        commit!(workspace)
        # Inject a finite committed sum whose final-column update overflows.
        workspace.committed_corrector_covariance_sum[index, m] = floatmax(T)
        workspace.committed_corrector_covariance_sum[m, index] = floatmax(T)
        retained = deepcopy(workspace)
        fill!(outputs[1], T(71)); fill!(outputs[2], T(72))
        outputs[3][1] = UInt64(73); outputs[4][1] = true
        original = map(copy, outputs)
        y = fill(T(1.25) * sqrt(floatmax(T)), m)
        @test process!(outputs..., workspace, plan, T[0], y) === DocrimeCovarianceNonFiniteCalculation
        @test isequal(outputs, original)
        @test all(isequal(getfield(workspace, f), getfield(retained, f))
                  for f in fieldnames(typeof(workspace)) if startswith(string(f), "committed_"))
        @test workspace.candidate_sample_count == 1
        @test direct_allocation_bytes(outputs, workspace, plan, T[0], y) == 0
        if recovery === :reset
            reset!(workspace)
        elseif recovery === :discard
            discard!(workspace)
        end
        @test process!(outputs..., workspace, plan, T[0], zeros(T, m)) === nothing
        expected = recovery === :reset ? zeros(T, m, m) : retained.committed_corrector_covariance_sum
        @test isequal(outputs[2], expected)
        commit!(workspace)
        @test docrime_covariance_sample_count(workspace) == (recovery === :reset ? 1 : 2)
    end
end

@testset "covariance shared storage retains checked scalar traversal" begin
    for T in (Float32, Float64)
        plan = DocrimeCovariancePlan(T, 1, 1)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, 1, 1), zeros(T, 1, 1), zeros(UInt64, 1), falses(1))
        @test process!(outputs..., workspace, plan, T[0], T[0]) === nothing
        commit!(workspace)
        workspace.candidate_corrector_covariance_sum = reshape(workspace.corrector_sample_delta, 1, 1)
        y = T(2) * sqrt(floatmax(T))
        @test process!(outputs..., workspace, plan, T[0], T[y]) === DocrimeCovarianceNonFiniteCalculation
        # Scalar rejection precedes the aliased write, retaining the finite delta.
        @test only(workspace.corrector_sample_delta) == y
        @test workspace.committed_corrector_covariance_sum == zeros(T, 1, 1)
        @test process!(outputs..., workspace, plan, T[0], T[3]) === nothing
        @test only(outputs[2]) == fma_oracle(T(0.5) * T(3), T(3), zero(T))
        @test direct_allocation_bytes(outputs, workspace, plan, T[0], T[3]) == 0
        # Do not commit the manually shared scratch layout.

        m = 9
        plan = DocrimeCovariancePlan(T, 1, m)
        workspace = DocrimeCovarianceWorkspace(plan)
        outputs = (zeros(T, 1, m), zeros(T, m, m), zeros(UInt64, 1), falses(1))
        @test process!(outputs..., workspace, plan, T[0], zeros(T, m)) === nothing
        commit!(workspace)
        # Preserve legacy in-place updates even for manually shared covariance.
        workspace.candidate_corrector_covariance_sum = workspace.committed_corrector_covariance_sum
        y = T[T(i) / T(7) for i in 1:m]
        expected = zeros(T, m, m)
        for column in 1:m, row in 1:column
            value = fma_oracle(T(0.5) * y[row], y[column], expected[row, column])
            expected[row, column] = expected[column, row] = value
        end
        @test process!(outputs..., workspace, plan, T[0], y) === nothing
        @test isequal(outputs[2], expected)
        @test isequal(workspace.committed_corrector_covariance_sum, expected)
    end
end

@testset "failed covariance retains aliased committed history" begin
    for T in (Float32, Float64)
        plan = DocrimeCovariancePlan(T, 1, 1)
        w = DocrimeCovarianceWorkspace(plan)
        out = (zeros(T,1,1), zeros(T,1,1), zeros(UInt64,1), falses(1))
        @test process!(out..., w, plan, T[0], T[0]) === nothing
        commit!(w)
        w.committed_corrector_covariance_sum[1,1] = floatmax(T)
        w.candidate_corrector_covariance_sum = w.committed_cross_covariance_sum
        original = map(copy,out)
        @test process!(out...,w,plan,T[0],T[T(1.25)*sqrt(floatmax(T))]) === DocrimeCovarianceNonFiniteCalculation
        @test isequal(out,original)
        @test iszero(only(w.committed_cross_covariance_sum))
    end
end

end
