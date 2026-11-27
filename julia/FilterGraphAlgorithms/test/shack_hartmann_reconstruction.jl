module ShackHartmannReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function pair_matrix(::Type{T}, values) where {T}
    matrix = Matrix{T}(undef, length(values), 2)
    for index in eachindex(values)
        matrix[index, 1] = T(values[index][1])
        matrix[index, 2] = T(values[index][2])
    end
    return matrix
end

function prepared_fixture(; coefficient_storage="fp32")
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=configuration["actuator_count"],
            subaperture_count=configuration["subaperture_count"],
            initial_reconstructor=Float32.(
                configuration["initial_reconstructor"],
            ),
            reconstructed_schema=configuration["reconstructed_schema"],
            coefficient_storage=coefficient_storage,
        ),
    )
end

function checked_allocation_bytes(reconstructed, prepared, slopes)
    return @allocated process!(reconstructed, prepared, slopes)
end

function admitted_allocation_bytes(reconstructed, prepared, slopes)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        reconstructed,
        prepared,
        slopes,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "Shack-Hartmann reconstruction preserves the FGN declaration" begin
    prepared = prepared_fixture()
    configuration = FIXTURE["configuration"]

    @test algorithm_label(ShackHartmannReconstructorF32) ==
          "shwfs-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannReconstructorF32,
    ) === DenseReconstructionFailure
    @test algorithm_property_policy(ShackHartmannReconstructorF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    fields = algorithm_configuration(ShackHartmannReconstructorF32)
    @test map(field -> field.name, fields) == (
        :reconstructed_count,
        :subaperture_count,
        :initial_reconstructor,
        :reconstructed_schema,
        :coefficient_storage,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        (fields[1], fields[2], fields[4], fields[5]),
    )
    @test fields[3].update isa ConstructionConfiguration

    ports = algorithm_ports(ShackHartmannReconstructorF32)
    @test map(port -> port.name, ports) ==
          (:slopes, :reconstructed, :reconstructor)
    @test ports[2].metadata_source === :slopes
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2, 2), (3,), (3, 4))
    @test getproperty.(formats, :schema) == (
        SHACK_HARTMANN_SLOPES_V1,
        configuration["reconstructed_schema"],
        SHWFS_RECONSTRUCTOR_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, formats)
    @test prepared.plan isa DenseReconstructorPlan{Float32}
    @test prepared.workspace isa DenseReconstructorWorkspace{Float32}
end

@testset "pair-interleaved slopes match the frozen FGN oracle" begin
    prepared = prepared_fixture()
    slopes = pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    reconstructed = zeros(Float32, 3)
    @test process!(reconstructed, prepared, slopes) === nothing
    @test reconstructed == Float32.(FIXTURE["sample"]["reconstructed"])

    replacement = FIXTURE["replacement"]
    matrix = row_major_matrix(
        Float32,
        replacement["reconstructor"],
        3,
        4,
    )
    original = prepared.plan
    @test replace_parameter!(prepared, :reconstructor, matrix) === prepared
    @test prepared.plan !== original
    @test process!(reconstructed, prepared, slopes) === nothing
    @test reconstructed == Float32.(replacement["reconstructed"])
end

@testset "construction and replacement reject invalid matrices atomically" begin
    configuration = FIXTURE["configuration"]
    @test_throws ArgumentError prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=0,
            subaperture_count=2,
            initial_reconstructor=nothing,
            reconstructed_schema=configuration["reconstructed_schema"],
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=3,
            subaperture_count=2,
            initial_reconstructor=zeros(Float32, 11),
            reconstructed_schema=configuration["reconstructed_schema"],
        ),
    )

    prepared = prepared_fixture()
    original = prepared.plan
    invalid = zeros(Float32, 3, 4)
    invalid[2, 3] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :reconstructor,
        invalid,
    )
    @test prepared.plan === original
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :reconstructor,
        zeros(Float32, 4, 3),
    )
    @test prepared.plan === original
end

@testset "numerical rejection and storage rules are explicit" begin
    prepared = prepared_fixture()
    slopes = pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    reconstructed = fill(71.0f0, 3)
    slopes[2, 1] = NaN32
    @test process!(reconstructed, prepared, slopes) ===
          DenseReconstructionNonFiniteInput
    @test reconstructed == fill(71.0f0, 3)

    overflow_matrix = fill(floatmax(Float32), 3, 4)
    @test replace_parameter!(prepared, :reconstructor, overflow_matrix) === prepared
    fill!(slopes, 2.0f0)
    @test process!(reconstructed, prepared, slopes) ===
          DenseReconstructionNonFiniteOutput
    @test all(isinf, reconstructed)

    prepared = prepared_fixture()
    slopes = pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    aliased_output = @view vec(slopes)[1:3]
    @test_throws ArgumentError process!(aliased_output, prepared, slopes)
    plan_output = @view prepared.plan.matrix[1:3]
    @test_throws ArgumentError process!(plan_output, prepared, slopes)
    scratch_input = reshape(prepared.workspace.ordered_input, 2, 2)
    @test_throws ArgumentError process!(zeros(Float32, 3), prepared, scratch_input)
end

@testset "views and REVOLT-sized warmed execution allocate no Julia heap" begin
    reconstructed_count = 277
    subaperture_count = 256
    slope_count = 2 * subaperture_count
    matrix = zeros(Float32, reconstructed_count, slope_count)
    for coordinate in 1:reconstructed_count
        matrix[coordinate, mod1(37 * coordinate, slope_count)] = 0.01f0
        matrix[coordinate, mod1(37 * coordinate + 1, slope_count)] = -0.005f0
    end
    prepared = prepare_algorithm(
        ShackHartmannReconstructorF32,
        ShackHartmannReconstructorF32Configuration(
            reconstructed_count=reconstructed_count,
            subaperture_count=subaperture_count,
            initial_reconstructor=nothing,
            reconstructed_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
    replace_parameter!(prepared, :reconstructor, matrix)
    slope_storage = zeros(Float32, subaperture_count * 2, 2)
    slopes = @view slope_storage[1:2:end, :]
    for subaperture in 1:subaperture_count
        slopes[subaperture, 1] = Float32(subaperture) / 256.0f0
        slopes[subaperture, 2] = -Float32(subaperture) / 512.0f0
    end
    output_storage = zeros(Float32, 2 * reconstructed_count)
    reconstructed = @view output_storage[1:2:end]

    process!(reconstructed, prepared, slopes)
    @test any(value -> !iszero(value), reconstructed)
    @test checked_allocation_bytes(reconstructed, prepared, slopes) == 0
    @test @inferred DenseReconstructionFailure FilterGraphAlgorithms._process_admitted!(
        reconstructed,
        prepared,
        slopes,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test admitted_allocation_bytes(reconstructed, prepared, slopes) == 0
    @test workspace_commit_allocation_bytes(prepared.workspace) == 0
    @test workspace_discard_allocation_bytes(prepared.workspace) == 0
    @test workspace_reset_allocation_bytes(prepared.workspace) == 0
end

@testset "FP16 storage survives calibration replacement" begin
    prepared = prepared_fixture(; coefficient_storage="fp16")
    @test_throws ArgumentError prepared_fixture(; coefficient_storage="invalid")
    slopes = pair_matrix(Float32, FIXTURE["sample"]["slopes"])
    output = zeros(Float32, 3)
    @test prepared.plan.matrix isa FilterGraphAlgorithms._ScaledFloat16Reconstructor
    @test process!(output, prepared, slopes) === nothing
    @test output ≈ Float32.(FIXTURE["sample"]["reconstructed"]) rtol=1f-3
    rows, columns = size(prepared.plan.matrix)
    replacement = row_major_matrix(Float32, FIXTURE["replacement"]["reconstructor"], rows, columns)
    @test replace_parameter!(prepared, :reconstructor, replacement) === prepared
    @test prepared.plan.matrix isa FilterGraphAlgorithms._ScaledFloat16Reconstructor
    @test process!(output, prepared, slopes) === nothing
    @test output ≈ Float32.(FIXTURE["replacement"]["reconstructed"]) rtol=1f-3
    @test checked_allocation_bytes(output, prepared, slopes) == 0
end

end
