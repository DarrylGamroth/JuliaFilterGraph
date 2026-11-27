module MatchedCalibrationDifferencingTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "matched_calibration_differencing.toml"),
)

const MEASUREMENTS_SCHEMA = "example/test-measurements/1"
const CORRECTOR_SAMPLE_SCHEMA = "example/test-corrector-sample/1"
const FILTERED_MEASUREMENTS_SCHEMA =
    "example/test-filtered-measurements/1"
const FILTERED_CORRECTOR_SAMPLE_SCHEMA =
    "example/test-filtered-corrector-sample/1"

struct ZeroBasedMatchedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedMatchedVector) = size(vector.parent)
Base.axes(vector::ZeroBasedMatchedVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedMatchedVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedMatchedVector, index::Int) =
    vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedMatchedVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

function matched_prepared(
    measurement_count::Int=FIXTURE["measurement_count"],
    corrector_coordinate_count::Int=FIXTURE["corrector_sample_count"],
)
    return prepare_algorithm(
        MatchedCalibrationDifferencingF32,
        MatchedCalibrationDifferencingF32Configuration(
            measurement_count=measurement_count,
            corrector_coordinate_count=corrector_coordinate_count,
            measurements_schema=MEASUREMENTS_SCHEMA,
            corrector_sample_schema=CORRECTOR_SAMPLE_SCHEMA,
            filtered_measurements_schema=FILTERED_MEASUREMENTS_SCHEMA,
            filtered_corrector_sample_schema=FILTERED_CORRECTOR_SAMPLE_SCHEMA,
        ),
    )
end

function matched_outputs(measurement_count::Int=2, corrector_coordinate_count::Int=2)
    return (
        fill(71.0f0, measurement_count),
        fill(72.0f0, corrector_coordinate_count),
        fill(true, 1),
    )
end

function matched_checked_allocation_bytes(
    filtered_measurements,
    filtered_corrector_sample,
    filtered,
    prepared,
    measurements,
    corrector_sample,
)
    return @allocated process!(
        filtered_measurements,
        filtered_corrector_sample,
        filtered,
        prepared,
        measurements,
        corrector_sample,
    )
end

function matched_checked_allocation_bytes(
    filtered_measurements, filtered_corrector_sample, filtered,
    workspace, plan, measurements, corrector_sample,
)
    return @allocated process!(
        filtered_measurements, filtered_corrector_sample, filtered,
        workspace, plan, measurements, corrector_sample,
    )
end

function matched_admitted_allocation_bytes(
    filtered_measurements,
    filtered_corrector_sample,
    filtered,
    prepared,
    measurements,
    corrector_sample,
)
    metadata = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated FilterGraphAlgorithms._process_admitted!(
        filtered_measurements,
        filtered_corrector_sample,
        filtered,
        prepared,
        measurements,
        corrector_sample,
        metadata,
        metadata,
    )
end

matched_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
matched_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
matched_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "matched calibration declaration preserves the frozen FGN contract" begin
    @test algorithm_label(MatchedCalibrationDifferencingF32) ==
          "matched-calibration-differencing-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        MatchedCalibrationDifferencingF32,
    ) === MatchedCalibrationDifferencingFailure
    @test algorithm_property_policy(MatchedCalibrationDifferencingF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(MatchedCalibrationDifferencingF32)
    @test map(field -> field.name, configuration) == (
        :measurement_count,
        :corrector_coordinate_count,
        :measurements_schema,
        :corrector_sample_schema,
        :filtered_measurements_schema,
        :filtered_corrector_sample_schema,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        configuration,
    )

    ports = algorithm_ports(MatchedCalibrationDifferencingF32)
    @test map(port -> port.name, ports) == (
        :measurements,
        Symbol("corrector-sample"),
        Symbol("filtered-measurements"),
        Symbol("filtered-corrector-sample"),
        :filtered,
    )
    @test getproperty.(ports, :element_type) ==
          (Float32, Float32, Float32, Float32, Bool)
    @test all(port -> isnothing(port.metadata_source), ports[1:2])
    @test all(
        port -> port.metadata_source === :measurements,
        ports[3:5],
    )

    prepared = matched_prepared(2, 1)
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2,), (1,), (2,), (1,), (1,))
    @test getproperty.(formats, :schema) == (
        MEASUREMENTS_SCHEMA,
        CORRECTOR_SAMPLE_SCHEMA,
        FILTERED_MEASUREMENTS_SCHEMA,
        FILTERED_CORRECTOR_SAMPLE_SCHEMA,
        MATCHED_CALIBRATION_FILTERED_V1,
    )
end

@testset "matched first differences exactly match the frozen Rust oracle" begin
    prepared = matched_prepared()
    filtered_measurements, filtered_corrector, filtered = matched_outputs()

    for frame in FIXTURE["frames"]
        @test process!(
            filtered_measurements,
            filtered_corrector,
            filtered,
            prepared,
            Float32.(frame["measurements"]),
            Float32.(frame["corrector_sample"]),
        ) === nothing
        @test filtered_measurements == Float32.(frame["filtered_measurements"])
        @test reinterpret(UInt32, filtered_measurements) ==
              UInt32.(frame["filtered_measurement_bits"])
        @test filtered_corrector == Float32.(frame["filtered_corrector_sample"])
        @test reinterpret(UInt32, filtered_corrector) ==
              UInt32.(frame["filtered_corrector_sample_bits"])
        @test only(filtered) === frame["filtered"]
    end
end

@testset "candidate pair history commits and discards transactionally" begin
    plan = MatchedCalibrationDifferencingPlan(Float64, 2, 2)
    workspace = MatchedCalibrationDifferencingWorkspace(plan)
    filtered_measurements = fill(71.0, 2)
    filtered_corrector = fill(72.0, 2)
    filtered = fill(true, 1)
    first_measurements = [10.0, 20.0]
    first_corrector = [1.0, -2.0]

    @test process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        first_measurements,
        first_corrector,
    ) === nothing
    @test filtered_measurements == zeros(2)
    @test filtered_corrector == zeros(2)
    @test filtered == [false]
    @test !matched_calibration_is_primed(workspace)
    @test discard!(workspace) === workspace
    @test !matched_calibration_is_primed(workspace)

    process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        first_measurements,
        first_corrector,
    )
    @test commit!(workspace) === workspace
    @test matched_calibration_is_primed(workspace)
    @test matched_calibration_previous_measurements(workspace) == first_measurements
    @test matched_calibration_previous_corrector_sample(workspace) == first_corrector

    second_measurements = [13.0, 14.0]
    second_corrector = [0.5, 4.0]
    process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        second_measurements,
        second_corrector,
    )
    @test filtered_measurements == [3.0, -6.0]
    @test filtered_corrector == [-0.5, 6.0]
    @test filtered == [true]
    @test matched_calibration_previous_measurements(workspace) == first_measurements
    @test discard!(workspace) === workspace
    @test matched_calibration_previous_measurements(workspace) == first_measurements

    process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        second_measurements,
        second_corrector,
    )
    commit!(workspace)
    @test matched_calibration_previous_measurements(workspace) == second_measurements
    @test matched_calibration_previous_corrector_sample(workspace) == second_corrector

    @test reset!(workspace) === workspace
    @test !matched_calibration_is_primed(workspace)
    @test matched_calibration_previous_measurements(workspace) == zeros(2)
    @test matched_calibration_previous_corrector_sample(workspace) == zeros(2)
end

@testset "matched rejection preserves outputs and committed history" begin
    @test_throws ArgumentError MatchedCalibrationDifferencingPlan(Float32, 0, 1)
    @test_throws ArgumentError MatchedCalibrationDifferencingPlan(Float32, 1, 0)
    @test_throws ArgumentError matched_prepared(0, 1)

    plan = MatchedCalibrationDifferencingPlan(Float32, 1, 1)
    workspace = MatchedCalibrationDifferencingWorkspace(plan)
    outputs = matched_outputs(1, 1)
    process!(outputs..., workspace, plan, Float32[floatmax(Float32)], Float32[1])
    commit!(workspace)
    committed_measurements = matched_calibration_previous_measurements(workspace)
    committed_corrector = matched_calibration_previous_corrector_sample(workspace)
    fill!(outputs[1], 71.0f0)
    fill!(outputs[2], 72.0f0)
    fill!(outputs[3], true)
    retained_outputs = map(copy, outputs)

    rejected_inputs = (
        (
            Float32[NaN],
            Float32[0],
            MatchedCalibrationNonFiniteMeasurements,
        ),
        (
            Float32[0],
            Float32[Inf],
            MatchedCalibrationNonFiniteCorrectorSample,
        ),
        (
            Float32[-floatmax(Float32)],
            Float32[0],
            MatchedCalibrationNonFiniteDifference,
        ),
    )
    for (measurements, corrector_sample, failure) in rejected_inputs
        @test process!(
            outputs...,
            workspace,
            plan,
            measurements,
            corrector_sample,
        ) === failure
        @test outputs == retained_outputs
        @test matched_calibration_previous_measurements(workspace) ==
              committed_measurements
        @test matched_calibration_previous_corrector_sample(workspace) ==
              committed_corrector
        discard!(workspace)
    end

    @test_throws DimensionMismatch process!(
        outputs...,
        workspace,
        plan,
        Float32[],
        Float32[0],
    )
    @test outputs == retained_outputs
    wrong_workspace = MatchedCalibrationDifferencingWorkspace(
        MatchedCalibrationDifferencingPlan(Float32, 2, 1),
    )
    @test_throws DimensionMismatch process!(
        outputs...,
        wrong_workspace,
        plan,
        Float32[0],
        Float32[0],
    )
    @test outputs == retained_outputs
end

@testset "matched corrector overflow preserves history and recovers" begin
    extent = 277
    plan = MatchedCalibrationDifferencingPlan(Float32, extent, extent)
    workspace = MatchedCalibrationDifferencingWorkspace(plan)
    outputs = matched_outputs(extent, extent)
    measurements = zeros(Float32, extent)
    corrector_sample = zeros(Float32, extent)
    corrector_sample[end] = floatmax(Float32)
    @test process!(outputs..., workspace, plan, measurements, corrector_sample) === nothing
    commit!(workspace)
    committed_measurements = matched_calibration_previous_measurements(workspace)
    committed_corrector = matched_calibration_previous_corrector_sample(workspace)
    fill!(outputs[1], 71f0)
    fill!(outputs[2], 72f0)
    fill!(outputs[3], true)
    retained_outputs = map(copy, outputs)
    fill!(measurements, 1f0)
    corrector_sample[end] = -floatmax(Float32)
    @test process!(outputs..., workspace, plan, measurements, corrector_sample) ===
          MatchedCalibrationNonFiniteDifference
    matched_checked_allocation_bytes(outputs..., workspace, plan, measurements, corrector_sample)
    @test matched_checked_allocation_bytes(
        outputs..., workspace, plan, measurements, corrector_sample,
    ) == 0
    @test outputs == retained_outputs
    @test matched_calibration_previous_measurements(workspace) == committed_measurements
    @test matched_calibration_previous_corrector_sample(workspace) == committed_corrector
    discard!(workspace)
    @test matched_calibration_is_primed(workspace)
    @test matched_calibration_previous_measurements(workspace) == committed_measurements
    @test matched_calibration_previous_corrector_sample(workspace) == committed_corrector

    fill!(measurements, 2f0)
    copyto!(corrector_sample, committed_corrector)
    @test process!(outputs..., workspace, plan, measurements, corrector_sample) === nothing
    @test all(==(2f0), outputs[1])
    @test all(iszero, outputs[2])
    @test only(outputs[3])
    @test matched_calibration_previous_measurements(workspace) == committed_measurements
    commit!(workspace)
    @test matched_calibration_previous_measurements(workspace) == measurements
    @test matched_calibration_previous_corrector_sample(workspace) == corrector_sample
end

@testset "matched differencing supports views and staged input aliasing" begin
    plan = MatchedCalibrationDifferencingPlan(Float32, 2, 2)
    workspace = MatchedCalibrationDifferencingWorkspace(plan)
    measurement_storage = Float32[83, 10, 83, 20, 83]
    corrector_storage = Float32[84, 1, 84, -2, 84]
    measurements = @view measurement_storage[2:2:4]
    corrector_sample = @view corrector_storage[2:2:4]
    filtered_measurement_storage = fill(71.0f0, 5)
    filtered_corrector_storage = fill(72.0f0, 5)
    filtered_measurements = @view filtered_measurement_storage[2:2:4]
    filtered_corrector = @view filtered_corrector_storage[2:2:4]
    filtered_storage = Bool[true, true, true]
    filtered = @view filtered_storage[2:2]

    @test process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        measurements,
        corrector_sample,
    ) === nothing
    @test collect(filtered_measurements) == zeros(Float32, 2)
    @test collect(filtered_corrector) == zeros(Float32, 2)
    commit!(workspace)

    measurements .= Float32[13, 14]
    corrector_sample .= Float32[0.5, 4]
    @test process!(
        measurements,
        corrector_sample,
        filtered,
        workspace,
        plan,
        measurements,
        corrector_sample,
    ) === nothing
    @test collect(measurements) == Float32[3, -6]
    @test collect(corrector_sample) == Float32[-0.5, 6]
    @test only(filtered)
    commit!(workspace)
    @test matched_calibration_previous_measurements(workspace) == Float32[13, 14]
    @test matched_calibration_previous_corrector_sample(workspace) == Float32[0.5, 4]

    shared = fill(71.0f0, 3)
    @test_throws ArgumentError process!(
        @view(shared[1:2]),
        @view(shared[2:3]),
        filtered,
        workspace,
        plan,
        Float32[0, 0],
        Float32[0, 0],
    )
    @test shared == fill(71.0f0, 3)

    zero_based = ZeroBasedMatchedVector(Float32[0, 0])
    @test_throws ArgumentError process!(
        filtered_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        zero_based,
        Float32[0, 0],
    )
    @test_throws ArgumentError process!(
        workspace.committed_measurements,
        filtered_corrector,
        filtered,
        workspace,
        plan,
        Float32[0, 0],
        Float32[0, 0],
    )
end

@testset "matched checked and admitted paths allocate nothing after warmup" begin
    prepared = matched_prepared(2, 2)
    outputs = matched_outputs(2, 2)
    priming_measurements = Float32[10, 20]
    priming_corrector = Float32[1, -2]
    measurements = Float32[13, 14]
    corrector_sample = Float32[0.5, 4]
    rejected_measurements = Float32[NaN, 0]
    rejected_corrector = Float32[0, Inf]
    large_measurements = Float32[floatmax(Float32), 14]
    overflowing_measurements = Float32[-floatmax(Float32), 14]

    process!(outputs..., prepared, priming_measurements, priming_corrector)
    process!(outputs..., prepared, measurements, corrector_sample)
    @test matched_checked_allocation_bytes(
        outputs...,
        prepared,
        measurements,
        corrector_sample,
    ) == 0
    process!(outputs..., prepared, large_measurements, corrector_sample)
    for (first, second, failure) in (
        (
            rejected_measurements,
            corrector_sample,
            MatchedCalibrationNonFiniteMeasurements,
        ),
        (
            measurements,
            rejected_corrector,
            MatchedCalibrationNonFiniteCorrectorSample,
        ),
        (
            overflowing_measurements,
            corrector_sample,
            MatchedCalibrationNonFiniteDifference,
        ),
    )
        @test process!(outputs..., prepared, first, second) === failure
        @test matched_checked_allocation_bytes(
            outputs...,
            prepared,
            first,
            second,
        ) == 0
        @test FilterGraphAlgorithms._process_admitted!(
            outputs...,
            prepared,
            first,
            second,
            FilterGraphAlgorithms._NO_SAMPLE_METADATA,
            FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        ) === failure
        @test matched_admitted_allocation_bytes(
            outputs...,
            prepared,
            first,
            second,
        ) == 0
        discard!(prepared.workspace)
    end
    @test @inferred MatchedCalibrationDifferencingFailure FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        measurements,
        corrector_sample,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test matched_admitted_allocation_bytes(
        outputs...,
        prepared,
        measurements,
        corrector_sample,
    ) == 0

    workspace = prepared.workspace
    @test matched_commit_allocation_bytes(workspace) == 0
    @test matched_discard_allocation_bytes(workspace) == 0
    @test matched_reset_allocation_bytes(workspace) == 0
end

@testset "matched calibration public API exports" begin
    for name in (
        :MATCHED_CALIBRATION_FILTERED_V1,
        :MatchedCalibrationDifferencingF32,
        :MatchedCalibrationDifferencingF32Configuration,
        :MatchedCalibrationDifferencingFailure,
        :MatchedCalibrationDifferencingPlan,
        :MatchedCalibrationDifferencingWorkspace,
        :MatchedCalibrationNonFiniteCorrectorSample,
        :MatchedCalibrationNonFiniteDifference,
        :MatchedCalibrationNonFiniteMeasurements,
        :matched_calibration_is_primed,
        :matched_calibration_previous_corrector_sample,
        :matched_calibration_previous_measurements,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
