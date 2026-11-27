module AlgorithmDeclarationTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, process!, reset!

struct WrappedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::WrappedVector) = size(vector.values)
Base.axes(vector::WrappedVector) = axes(vector.values)
Base.IndexStyle(::Type{<:WrappedVector}) = IndexLinear()
Base.getindex(vector::WrappedVector, index::Int) = vector.values[index]
Base.setindex!(vector::WrappedVector, value, index::Int) =
    setindex!(vector.values, value, index)

struct ZeroBasedMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::A
end

Base.size(array::ZeroBasedMatrix) = size(array.parent)
Base.axes(array::ZeroBasedMatrix) = map(axis -> 0:(length(axis) - 1), axes(array.parent))
Base.IndexStyle(::Type{<:ZeroBasedMatrix}) = IndexCartesian()
Base.getindex(array::ZeroBasedMatrix, row::Int, column::Int) =
    array.parent[row + 1, column + 1]

struct MatrixGainPlan
    gain::Matrix{Float32}
end

@algorithm MatrixGainF32 begin
    label = "matrix-gain-f32-test"

    @configuration MatrixGainF32Configuration begin
        rows::Int => graph_rebuild
        columns::Int => graph_rebuild
        initial_gain::Union{Nothing,Float32} => construction
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.rows, configuration.columns),
            "test.matrix.f32/1";
            name="matrix-input",
        ),
        output(
            :output,
            Float32,
            (configuration.rows, configuration.columns),
            "test.matrix.f32/1";
            name="matrix-output",
            metadata_from=:input,
        ),
        parameter(
            :gain,
            Float32,
            (configuration.rows, configuration.columns),
            "test.matrix-gain.f32/1";
            name="gain-map",
            replace=(plan, values) -> MatrixGainPlan(copy(values)),
        ),
    )

    prepare(configuration) = MatrixGainPlan(
        fill(
            something(configuration.initial_gain, 1.0f0),
            configuration.rows,
            configuration.columns,
        ),
    )

    process!(plan) = begin
        output .= input .* plan.gain
        nothing
    end
end

@algorithm SourceF32 begin
    label = "source-f32-test"

    @configuration SourceF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) =
        (output(:output, Float32, (configuration.extent,), "test.source.f32/1"),)

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        fill!(output, 2.0f0)
        nothing
    end
end

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "leaky_integrator_declaration.toml"),
)

@testset "leaky integrator declaration metadata matches the FGN contract" begin
    @test algorithm_label(LeakyIntegratorF32) == FIXTURE["declaration"]["label"]
    @test algorithm_property_policy(LeakyIntegratorF32) isa RuntimeDeclaredProperties
    @test algorithm_operation(LeakyIntegratorF32) isa GeneralOperation
    @test FilterGraphAlgorithms.algorithm_failure_type(LeakyIntegratorF32) ===
          LeakyIntegratorFailure

    configuration = algorithm_configuration(LeakyIntegratorF32)
    @test length(configuration) == 4
    @test configuration[1].name === :extent
    @test configuration[1].julia_type === Int
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].name === :initial_state
    @test configuration[2].julia_type === Float32
    @test configuration[2].update isa ConstructionConfiguration
    @test configuration[3].name === :input_schema
    @test configuration[3].julia_type === String
    @test configuration[3].update isa GraphRebuildConfiguration
    @test configuration[4].name === :output_schema
    @test configuration[4].julia_type === String
    @test configuration[4].update isa GraphRebuildConfiguration
    for (actual, expected) in zip(configuration, FIXTURE["configuration"])
        @test String(actual.name) == expected["name"]
        @test actual.update isa (
            expected["update"] == "construction" ?
            ConstructionConfiguration : GraphRebuildConfiguration
        )
    end

    ports = algorithm_ports(LeakyIntegratorF32)
    @test length(ports) == 2
    @test ports[1].name === :input
    @test ports[1].direction isa FilterGraphAlgorithms.InputPort
    @test ports[1].role isa FilterGraphAlgorithms.FrameData
    @test ports[1].element_type === Float32
    @test isnothing(ports[1].metadata_source)
    @test ports[2].name === :output
    @test ports[2].direction isa FilterGraphAlgorithms.OutputPort
    @test ports[2].role isa FilterGraphAlgorithms.FrameData
    @test ports[2].element_type === Float32
    @test ports[2].metadata_source === :input
    for (index, (actual, expected)) in enumerate(zip(ports, FIXTURE["ports"]))
        @test String(actual.name) == expected["name"]
        @test expected["role"] == "data"
        @test expected["element_type"] == "float32"
        @test expected["shape"] == ["extent"]
        @test expected["schema"] == (
            index == 1 ? CONTROLLER_RESIDUAL_ERROR_V1 : CONTROLLER_COMMAND_V1
        )
        @test actual.metadata_source ==
              (haskey(expected, "metadata_from") ? Symbol(expected["metadata_from"]) : nothing)
    end
end

@testset "operation markers describe mathematical shape without execution policy" begin
    @test algorithm_operation(SccReconstructorF32) isa DenseF32Operation
    @test algorithm_operation(DenseVectorReconstructorF32) isa DenseF32Operation
    @test algorithm_operation(ShackHartmannReconstructorF32) isa DenseF32Operation
    @test algorithm_operation(PyramidReconstructorF32) isa DenseF32Operation
    @test algorithm_operation(ShackHartmannImageF32) isa
          ShackHartmannImageF32Operation
    @test algorithm_operation(ShackHartmannRowReconstructorF32) isa
          ShackHartmannRowReconstructionF32Operation
    @test algorithm_operation(PyramidRowReconstructorF32) isa
          PyramidRowReconstructionF32Operation
    @test fieldcount(typeof(algorithm_operation(SccReconstructorF32))) == 0
    @test all(
        declaration -> algorithm_operation(declaration) isa AlgorithmOperation,
        algorithms(),
    )
end

@testset "leaky numerical rejection preserves output and committed state" begin
    plan = LeakyIntegratorPlan(floatmax(Float32), 1.0f0)
    workspace = LeakyIntegratorWorkspace(Float32[0, 0])
    output = fill(71.0f0, 2)

    @test process!(output, workspace, plan, Float32[NaN, 1]) ===
          LeakyIntegratorNonFiniteInput
    @test output == fill(71.0f0, 2)
    @test leaky_integrator_state(workspace) == Float32[0, 0]

    @test process!(output, workspace, plan, Float32[2, 1]) ===
          LeakyIntegratorNonFiniteOutput
    @test output == fill(71.0f0, 2)
    @test leaky_integrator_state(workspace) == Float32[0, 0]

    valid_plan = LeakyIntegratorPlan(0.5f0, 1.0f0)
    @test process!(output, workspace, valid_plan, Float32[2, -2]) === output
    @test output == Float32[1, -1]
    @test commit!(workspace) === workspace
    @test leaky_integrator_state(workspace) == output

    declared = prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=2,
            initial_state=0.0f0,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
    declared.plan = plan
    fill!(output, 83.0f0)
    @test process!(output, declared, Float32[2, 1]) ===
          LeakyIntegratorNonFiniteOutput
    @test output == fill(83.0f0, 2)
    @test leaky_integrator_state(declared.workspace) == Float32[0, 0]
end

@testset "native declaration matches direct stateful execution" begin
    native = FIXTURE["native"]
    configuration = LeakyIntegratorF32Configuration(
        extent=native["extent"],
        initial_state=Float32(native["initial_state"]),
        input_schema=native["input_schema"],
        output_schema=native["output_schema"],
    )
    declared = prepare_algorithm(LeakyIntegratorF32, configuration)
    formats = algorithm_formats(declared)

    @test length(formats) == 2
    for (format, schema) in zip(
        formats,
        (CONTROLLER_RESIDUAL_ERROR_V1, CONTROLLER_COMMAND_V1),
    )
        @test format.element_type === Float32
        @test format.shape == (native["extent"],)
        @test format.schema == schema
        @test format.rate == (0x00000001, 0x00000001)
        @test !hasproperty(format, :profile)
    end

    plan = LeakyIntegratorPlan(1.0f0, 0.0f0)
    workspace = LeakyIntegratorWorkspace(
        fill(Float32(native["initial_state"]), native["extent"]),
    )
    expected = zeros(Float32, native["extent"])
    actual = zeros(Float32, native["extent"])

    for (input_values, output_values) in zip(native["inputs"], native["outputs"])
        input = Float32.(input_values)
        process!(expected, workspace, plan, input)
        commit!(workspace)
        @test process!(actual, declared, input) === nothing
        @test actual == expected
        @test actual == Float32.(output_values)
        @test leaky_integrator_state(declared.workspace) == leaky_integrator_state(workspace)
    end

    @test reset!(declared) === declared
    @test leaky_integrator_state(declared.workspace) ==
          fill(Float32(native["initial_state"]), native["extent"])
end

@testset "leaky integrator accepts transport-neutral vector wrappers" begin
    prepared = prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=4,
            initial_state=0.0f0,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
    input_storage = Float32[1, 2, 3, 4]
    output_storage = zeros(Float32, 4)
    input = WrappedVector(input_storage)
    output = WrappedVector(output_storage)

    @test process!(output, prepared, input) === nothing
    @test output_storage == input_storage
    @test leaky_integrator_state(prepared.workspace) == input_storage
    @test @allocated(process!(output, prepared, input)) == 0
end

@testset "ordinary Julia matrices and sparse parameters remain native" begin
    configuration = algorithm_configuration(MatrixGainF32)
    @test configuration[3].julia_type === Union{Nothing,Float32}

    prepared = prepare_algorithm(
        MatrixGainF32,
        MatrixGainF32Configuration(rows=2, columns=3, initial_gain=nothing),
    )
    @test all(format -> format.layout isa ColumnMajorLayout, algorithm_formats(prepared))
    formats = algorithm_formats(prepared)
    @test algorithm_format(prepared, "matrix-input") === formats[1]
    @test algorithm_format(prepared, Symbol("matrix-output")) === formats[2]
    @test (@inferred algorithm_format(prepared, Val(Symbol("gain-map")))) === formats[3]
    @test_throws ArgumentError algorithm_format(prepared, "input")
    @test_throws ArgumentError algorithm_format(prepared, "unknown-port")
    ports = algorithm_ports(MatrixGainF32)
    @test map(port -> port.name, ports) == (
        Symbol("matrix-input"),
        Symbol("matrix-output"),
        Symbol("gain-map"),
    )
    @test ports[2].metadata_source === Symbol("matrix-input")
    @test ports[3].direction isa FilterGraphAlgorithms.InputPort
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter

    input = reshape(Float32.(1:6), 2, 3)
    output = similar(input)
    @test process!(output, prepared, input) === nothing
    @test output == input
    @test_throws ArgumentError process!(output, prepared, ZeroBasedMatrix(input))

    gain = fill(3.0f0, 2, 3)
    @test replace_parameter!(prepared, "gain-map", gain) === prepared
    @test process!(output, prepared, input) === nothing
    @test output == 3.0f0 .* input
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "gain-map",
        ones(Float32, 3, 2),
    )
end

@testset "source declarations do not invent an input metadata source" begin
    prepared = prepare_algorithm(SourceF32, SourceF32Configuration(extent=4))
    port = only(algorithm_ports(SourceF32))
    @test port.direction isa FilterGraphAlgorithms.OutputPort
    @test isnothing(port.metadata_source)
    output = zeros(Float32, 4)
    @test process!(output, prepared) === nothing
    @test output == fill(2.0f0, 4)
end

@testset "declaration validation and warmed processing" begin
    @test_throws ArgumentError prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=0,
            initial_state=0.0f0,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )

    for schema in ("", "missing-version", "schema/0", "schema/not-a-number")
        @test_throws ArgumentError prepare_algorithm(
            LeakyIntegratorF32,
            LeakyIntegratorF32Configuration(
                extent=2,
                initial_state=0.0f0,
                input_schema=schema,
                output_schema=CONTROLLER_COMMAND_V1,
            ),
        )
    end

    declared = prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=2,
            initial_state=0.0f0,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
    input = Float32[1, 2]
    output = zeros(Float32, 2)
    process!(output, declared, input)
    @test @allocated(process!(output, declared, input)) == 0
    @test_throws ArgumentError process!(output, declared, Float64[1, 2])
    @test_throws DimensionMismatch process!(zeros(Float32, 1), declared, input)
end

@testset "built-in Algorithm inventory is explicit and unique" begin
    declarations = @inferred algorithms()
    @test declarations == (
        LeakyIntegratorF32,
        BiquadCascadeF32,
        VectorLinearCombinationF32,
        VectorOffsetF32,
        PhysicalOffsetReferenceF32,
        CorrectorAllocationF32,
        FrameDelayF32,
        ArrayStatisticsF32,
        DocrimeExcitationF32,
        DocrimeBinaryExcitationF32,
        DocrimeCovarianceF32,
        DocrimeOpenLoopInteractionMatrixF32,
        DocrimeClosedLoopInteractionMatrixF32,
        PixelCalibrationU16F32,
        PixelCalibrationF32,
        PixelCalibrationRowU16F32,
        PixelCalibrationReadoutU16F32,
        DetectorFrameAssemblerF32,
        PixelSelectionF32,
        PixelValidityF32,
        RegionExtractionF32,
        SccPushPullResponseF32,
        ChoppedImageDifferenceF32,
        ChoppedDetectorDifferenceF32,
        SccPixelResidualF32,
        SccSpatialCarrierDemodulationF32,
        SccWeightedRegularizedReconstructorF32,
        SccReconstructorF32,
        SccCorrectionF32,
        MlLlowfsF32,
        CenterOfGravityF32,
        WeightedCenterOfGravityF32,
        ShackHartmannImageF32,
        ShackHartmannSummaryF32,
        ShackHartmannWeightedCentroidF32,
        ShackHartmannMeasurementBlockF32,
        ShackHartmannReadoutMeasurementBlockF32,
        ShackHartmannWeightedMeasurementBlockF32,
        ShackHartmannReconstructorF32,
        ShackHartmannRowReconstructorF32,
        ShackHartmannReadoutReconstructorF32,
        PyramidPixelRowF32,
        PyramidImageF32,
        PyramidPupilImageF32,
        PyramidPupilFluxU16F64,
        PyramidPupilReconstructorF32,
        BiOEdgeImageF32,
        ZernikePupilSignalF32,
        CurvaturePairedImageF32,
        CurvaturePairedChannelF32,
        PyramidMeasurementBlockF32,
        PyramidReconstructorF32,
        PyramidRowReconstructorF32,
        PyramidReadoutReconstructorF32,
        PyramidReadoutMeasurementBlockF32,
        IncrementalDenseReconstructorF32,
        DenseVectorReconstructorF32,
        ClosedLoopCorrectionF32,
        ClosedLoopIntegratorF32,
        ValidityGatedClosedLoopIntegratorF32,
        PseudoOpenLoopReconstructionF32,
        OpticalGainCorrectionF32,
        OpticalGainDitherF32,
        OpticalGainEstimatorF32,
        OpticalGainUpdateF32,
        InteractionMatrixStabilityF32,
        MatchedCalibrationDifferencingF32,
        ModalPushPullF32,
        ZonalPushPullF32,
        HadamardPushPullF32,
        ZonalInteractionMatrixF32,
        HadamardInteractionMatrixF32,
        ControllerToVdmF32,
        VdmToPdmF32,
        PdmSystemFlatF32,
        PdmEngineeringOffsetF32,
        VdmShapeOffsetF32,
        PdmCommandPowerLimitF32,
        InterActuatorStrokeLimitF32,
        PdmStrokeOptimizationF32,
        PdmCommandF32,
        TipTiltCommandF32,
        PdmFeedbackToVdmF32,
        VdmFeedbackToControllerF32,
    )
    @test all(
        declaration -> declaration <: AbstractAlgorithmDeclaration,
        declarations,
    )
    labels = map(algorithm_label, declarations)
    @test length(unique(labels)) == length(labels)
    @test count(==("pdm-command-f32"), labels) == 1
    @test "vector-conditioning-f32" ∉ labels
    @test !isdefined(FilterGraphAlgorithms, :VectorConditioningF32)
    @test !isdefined(FilterGraphAlgorithms, :VectorConditioningF32Configuration)
end

end
