module PseudoOpenLoopReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: _process_admitted!, commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pseudo_open_loop_reconstruction.toml"),
)

struct ZeroBasedPseudoOpenLoopVector{T,V<:AbstractVector{T}} <:
       AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedPseudoOpenLoopVector) = size(vector.parent)
Base.axes(vector::ZeroBasedPseudoOpenLoopVector) =
    (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedPseudoOpenLoopVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedPseudoOpenLoopVector, index::Int) =
    vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedPseudoOpenLoopVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

function pseudo_open_loop_configuration(case)
    return PseudoOpenLoopReconstructionF32Configuration(
        measurement_count=case["measurement_count"],
        corrector_shape_count=case["corrector_shape_count"],
        residual_error_count=case["residual_error_count"],
        initial_interaction_matrix=Float32.(case["interaction_matrix"]),
        initial_reconstructor=Float32.(case["reconstructor"]),
        initial_corrector_to_error=Float32.(case["corrector_to_error"]),
        residual_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
        corrector_shape_schema=REQUESTED_PDM_COMMAND_V1,
        pseudo_open_loop_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
        residual_error_schema=CONTROLLER_RESIDUAL_ERROR_V1,
    )
end

pseudo_open_loop_prepared(case=FIXTURE["cases"][1]) =
    prepare_algorithm(
        PseudoOpenLoopReconstructionF32,
        pseudo_open_loop_configuration(case),
    )

function pseudo_open_loop_allocation_bytes(
    measurements,
    residual_error,
    prepared,
    residual_measurements,
    corrector_shape,
)
    return @allocated process!(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    )
end

function pseudo_open_loop_admitted_allocation_bytes(
    measurements,
    residual_error,
    prepared,
    residual_measurements,
    corrector_shape,
)
    return @allocated _process_admitted!(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "pseudo-open-loop declaration preserves the frozen FGN contract" begin
    @test algorithm_label(PseudoOpenLoopReconstructionF32) ==
          "pseudo-open-loop-reconstruction-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PseudoOpenLoopReconstructionF32,
    ) === PseudoOpenLoopReconstructionFailure
    @test algorithm_property_policy(PseudoOpenLoopReconstructionF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(PseudoOpenLoopReconstructionF32)
    @test map(field -> field.name, configuration) == (
        :measurement_count,
        :corrector_shape_count,
        :residual_error_count,
        :initial_interaction_matrix,
        :initial_reconstructor,
        :initial_corrector_to_error,
        :residual_measurements_schema,
        :corrector_shape_schema,
        :pseudo_open_loop_measurements_schema,
        :residual_error_schema,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        configuration[[1, 2, 3, 7, 8, 9, 10]],
    )
    @test all(
        field -> field.update isa ConstructionConfiguration,
        configuration[4:6],
    )

    ports = algorithm_ports(PseudoOpenLoopReconstructionF32)
    @test map(port -> port.name, ports) == (
        Symbol("residual-measurements"),
        Symbol("corrector-shape"),
        Symbol("pseudo-open-loop-measurements"),
        Symbol("residual-error"),
        Symbol("interaction-matrix"),
        :reconstructor,
        Symbol("corrector-to-error"),
    )
    @test ports[3].metadata_source === Symbol("residual-measurements")
    @test ports[4].metadata_source === Symbol("residual-measurements")
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[5:7],
    )

    prepared = pseudo_open_loop_prepared()
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == (
        (2,),
        (2,),
        (2,),
        (2,),
        (2, 2),
        (2, 2),
        (2, 2),
    )
    @test getproperty.(formats, :schema) == (
        SHACK_HARTMANN_SLOPES_V1,
        REQUESTED_PDM_COMMAND_V1,
        SHACK_HARTMANN_SLOPES_V1,
        CONTROLLER_RESIDUAL_ERROR_V1,
        PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1,
        PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
        PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
    )

end

@testset "pseudo-open-loop reconstruction matches the pinned Rust oracle" begin
    for case in FIXTURE["cases"]
        prepared = pseudo_open_loop_prepared(case)
        residual_measurements = Float32.(case["residual_measurements"])
        corrector_shape = Float32.(case["corrector_shape"])
        measurements = fill(71.0f0, case["measurement_count"])
        residual_error = fill(83.0f0, case["residual_error_count"])

        @test process!(
            measurements,
            residual_error,
            prepared,
            residual_measurements,
            corrector_shape,
        ) === nothing
        @test measurements == Float32.(case["pseudo_open_loop_measurements"])
        @test reinterpret(UInt32, measurements) ==
              UInt32.(case["pseudo_open_loop_measurement_bits"])
        @test residual_error == Float32.(case["residual_error"])
        @test reinterpret(UInt32, residual_error) ==
              UInt32.(case["residual_error_bits"])
    end
end

@testset "independent pseudo-open-loop maps retain their orientation" begin
    interaction = Float64[1 2; -1 0.5]
    reconstructor = Float64[2 -1; 0.5 3]
    corrector_to_error = Float64[0.25 -1; 2 0]
    plan = PseudoOpenLoopReconstructionPlan(
        interaction,
        reconstructor,
        corrector_to_error,
    )
    workspace = PseudoOpenLoopReconstructionWorkspace(plan)
    measurements = zeros(2)
    residual_error = zeros(2)
    @test FilterGraphAlgorithms._process_pseudo_open_loop_reconstruction!(
        measurements,
        residual_error,
        workspace,
        plan,
        [4.0, 2.5],
        [0.5, -2.0],
    ) === nothing
    @test measurements == [0.5, 1.0]
    @test residual_error == [-2.125, 2.25]
end

@testset "pseudo-open-loop preparation and replacement are failure-atomic" begin
    @test_throws ArgumentError PseudoOpenLoopReconstructionPlan(
        zeros(Float32, 0, 1),
        zeros(Float32, 1, 0),
        zeros(Float32, 1, 1),
    )
    @test_throws ArgumentError PseudoOpenLoopReconstructionPlan(
        zeros(Float32, 1, 0),
        zeros(Float32, 1, 1),
        zeros(Float32, 1, 0),
    )
    @test_throws ArgumentError PseudoOpenLoopReconstructionPlan(
        zeros(Float32, 1, 1),
        zeros(Float32, 0, 1),
        zeros(Float32, 0, 1),
    )
    @test_throws DimensionMismatch PseudoOpenLoopReconstructionPlan(
        zeros(Float32, 2, 1),
        zeros(Float32, 1, 1),
        zeros(Float32, 1, 1),
    )
    @test_throws DimensionMismatch PseudoOpenLoopReconstructionPlan(
        zeros(Float32, 2, 1),
        zeros(Float32, 1, 2),
        zeros(Float32, 1, 2),
    )
    for artifacts in (
        (Float32[NaN 0], zeros(Float32, 1, 1), zeros(Float32, 1, 2)),
        (zeros(Float32, 1, 2), Float32[NaN;;], zeros(Float32, 1, 2)),
        (zeros(Float32, 1, 2), zeros(Float32, 1, 1), Float32[0 Inf]),
    )
        @test_throws ArgumentError PseudoOpenLoopReconstructionPlan(artifacts...)
    end

    prepared = pseudo_open_loop_prepared(FIXTURE["cases"][2])
    original = prepared.plan
    interaction = Float32[0; 0;;]
    @test replace_parameter!(
        prepared,
        "interaction-matrix",
        interaction,
    ) === prepared
    @test prepared.plan !== original
    @test prepared.plan.reconstructor === original.reconstructor
    @test prepared.plan.corrector_to_error === original.corrector_to_error

    after_interaction = prepared.plan
    reconstructor = Float32[1 0]
    @test replace_parameter!(prepared, :reconstructor, reconstructor) === prepared
    @test prepared.plan.interaction_matrix ===
          after_interaction.interaction_matrix
    @test prepared.plan.corrector_to_error ===
          after_interaction.corrector_to_error

    active = prepared.plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :reconstructor,
        Float32[NaN 0],
    )
    @test prepared.plan === active
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "corrector-to-error",
        zeros(Float32, 1, 2),
    )
    @test prepared.plan === active

    zero_plan = prepare_algorithm(
        PseudoOpenLoopReconstructionF32,
        PseudoOpenLoopReconstructionF32Configuration(
            measurement_count=2,
            corrector_shape_count=1,
            residual_error_count=1,
            initial_interaction_matrix=nothing,
            initial_reconstructor=nothing,
            initial_corrector_to_error=nothing,
            residual_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
            corrector_shape_schema=REQUESTED_PDM_COMMAND_V1,
            pseudo_open_loop_measurements_schema=SHACK_HARTMANN_SLOPES_V1,
            residual_error_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
    @test all(iszero, zero_plan.plan.interaction_matrix)
    @test all(iszero, zero_plan.plan.reconstructor)
    @test all(iszero, zero_plan.plan.corrector_to_error)
end

@testset "pseudo-open-loop rejection preserves both outputs" begin
    prepared = pseudo_open_loop_prepared(FIXTURE["cases"][3])
    measurements = Float32[71]
    residual_error = Float32[83]
    @test process!(measurements, residual_error, prepared, Float32[NaN], Float32[0]) ===
          PseudoOpenLoopNonFiniteResidualMeasurements
    @test measurements == Float32[71]
    @test residual_error == Float32[83]
    @test process!(measurements, residual_error, prepared, Float32[0], Float32[Inf]) ===
          PseudoOpenLoopNonFiniteCorrectorShape
    @test measurements == Float32[71]
    @test residual_error == Float32[83]

    cases = (
        (Float32[floatmax(Float32);;], Float32[0;;], Float32[0;;], 0.0f0, 2.0f0),
        (Float32[1;;], Float32[0;;], Float32[0;;], floatmax(Float32), floatmax(Float32)),
        (Float32[0;;], Float32[2;;], Float32[0;;], floatmax(Float32), 1.0f0),
        (Float32[0;;], Float32[0;;], Float32[floatmax(Float32);;], 0.0f0, 2.0f0),
        (Float32[0;;], Float32[1;;], Float32[-1;;], floatmax(Float32), floatmax(Float32)),
    )
    for (interaction, reconstructor, corrector_to_error, residual, corrector) in cases
        plan = PseudoOpenLoopReconstructionPlan(
            interaction,
            reconstructor,
            corrector_to_error,
        )
        candidate = prepare_algorithm(
            PseudoOpenLoopReconstructionF32,
            pseudo_open_loop_configuration(FIXTURE["cases"][3]),
        )
        candidate.plan = plan
        candidate.workspace = PseudoOpenLoopReconstructionWorkspace(plan)
        @test process!(
            measurements,
            residual_error,
            candidate,
            Float32[residual],
            Float32[corrector],
        ) === PseudoOpenLoopNonFiniteOutput
        @test measurements == Float32[71]
        @test residual_error == Float32[83]
    end
end

@testset "pseudo-open-loop supports views and staged input aliasing" begin
    prepared = pseudo_open_loop_prepared(FIXTURE["cases"][2])
    residual_storage = Float32[91, 4, 91, 3, 91]
    corrector_storage = Float32[92, 1, 92]
    measurement_storage = fill(93.0f0, 5)
    error_storage = fill(94.0f0, 3)
    residual_measurements = @view residual_storage[2:2:4]
    corrector_shape = @view corrector_storage[2:2]
    measurements = @view measurement_storage[2:2:4]
    residual_error = @view error_storage[2:2]
    @test process!(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    ) === nothing
    @test measurements == Float32[6, 4]
    @test residual_error == Float32[5]

    aliased_measurements = Float32[4, 3]
    aliased_error = Float32[1]
    @test process!(
        aliased_measurements,
        aliased_error,
        prepared,
        aliased_measurements,
        aliased_error,
    ) === nothing
    @test aliased_measurements == Float32[6, 4]
    @test aliased_error == Float32[5]

    overlapping = zeros(Float32, 3)
    @test_throws ArgumentError process!(
        view(overlapping, 1:2),
        view(overlapping, 2:2),
        prepared,
        Float32[4, 3],
        Float32[1],
    )
    @test_throws ArgumentError process!(
        view(prepared.plan.interaction_matrix, :, 1),
        zeros(Float32, 1),
        prepared,
        Float32[4, 3],
        Float32[1],
    )
    @test_throws ArgumentError process!(
        zeros(Float32, 2),
        zeros(Float32, 1),
        prepared,
        prepared.workspace.predicted_measurements,
        Float32[1],
    )
    @test_throws ArgumentError process!(
        ZeroBasedPseudoOpenLoopVector(zeros(Float32, 2)),
        zeros(Float32, 1),
        prepared,
        Float32[4, 3],
        Float32[1],
    )
end

@testset "pseudo-open-loop warmed paths allocate no Julia heap" begin
    prepared = pseudo_open_loop_prepared(FIXTURE["cases"][2])
    residual_measurements = Float32[4, 3]
    corrector_shape = Float32[1]
    measurements = zeros(Float32, 2)
    residual_error = zeros(Float32, 1)
    process!(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    )
    @test pseudo_open_loop_allocation_bytes(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    ) == 0
    @test @inferred PseudoOpenLoopReconstructionFailure _process_admitted!(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test pseudo_open_loop_admitted_allocation_bytes(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    ) == 0

    residual_measurements[1] = NaN32
    @test pseudo_open_loop_allocation_bytes(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    ) == 0
    residual_measurements[1] = 4.0f0
    prepared.plan.interaction_matrix[1] = floatmax(Float32)
    corrector_shape[1] = 2.0f0
    @test pseudo_open_loop_admitted_allocation_bytes(
        measurements,
        residual_error,
        prepared,
        residual_measurements,
        corrector_shape,
    ) == 0
    @test @allocated(commit!(prepared.workspace)) == 0
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
end

end
