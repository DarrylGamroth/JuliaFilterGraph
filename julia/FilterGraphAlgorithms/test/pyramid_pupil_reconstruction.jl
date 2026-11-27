module PyramidPupilReconstructionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_pupil_reconstruction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_fixture(; coefficient_storage="fp32")
    configuration = FIXTURE["configuration"]
    return prepare_algorithm(
        PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(
            reconstructed_count=configuration["reconstructed_count"],
            selected_pixel_count=configuration["selected_pixel_count"],
            initial_reconstructor=Float32.(
                configuration["initial_reconstructor"],
            ),
            reconstructed_schema=configuration["reconstructed_schema"],
            coefficient_storage=coefficient_storage,
        ),
    )
end

function checked_allocation_bytes(reconstructed, prepared, pixels)
    return @allocated process!(reconstructed, prepared, pixels)
end

function admitted_allocation_bytes(reconstructed, prepared, pixels)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        reconstructed,
        prepared,
        pixels,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "pyramid reconstruction preserves the FGN declaration" begin
    prepared = prepared_fixture()
    configuration = FIXTURE["configuration"]

    @test algorithm_label(PyramidPupilReconstructorF32) == "pwfs-pupil-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PyramidPupilReconstructorF32) ===
          DenseReconstructionFailure
    fields = algorithm_configuration(PyramidPupilReconstructorF32)
    @test map(field -> field.name, fields) == (
        :reconstructed_count,
        :selected_pixel_count,
        :initial_reconstructor,
        :reconstructed_schema,
        :coefficient_storage,
    )
    @test all(
        field -> field.update isa GraphRebuildConfiguration,
        (fields[1], fields[2], fields[4], fields[5]),
    )
    @test fields[3].update isa ConstructionConfiguration

    ports = algorithm_ports(PyramidPupilReconstructorF32)
    @test map(port -> port.name, ports) == (
        Symbol("reconstruction-pixels"),
        :reconstructed,
        :reconstructor,
    )
    @test ports[2].metadata_source === Symbol("reconstruction-pixels")
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((4, 2), (3,), (3, 8))
    @test getproperty.(formats, :schema) == (
        PYRAMID_RECONSTRUCTION_PIXELS_V1,
        configuration["reconstructed_schema"],
        PWFS_RECONSTRUCTOR_V1,
    )
end

@testset "pupil-major pixels match the frozen FGN oracle" begin
    configuration = FIXTURE["configuration"]
    pixels = row_major_matrix(
        Float32,
        FIXTURE["sample"]["reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        configuration["selected_pixel_count"],
    )
    reconstructed = zeros(Float32, configuration["reconstructed_count"])
    prepared = prepared_fixture()
    @test process!(reconstructed, prepared, pixels) === nothing
    @test reconstructed == Float32.(FIXTURE["sample"]["reconstructed"])

    replacement = FIXTURE["replacement"]
    matrix = row_major_matrix(
        Float32,
        replacement["reconstructor"],
        configuration["reconstructed_count"],
        PYRAMID_PUPIL_COUNT * configuration["selected_pixel_count"],
    )
    @test replace_parameter!(prepared, :reconstructor, matrix) === prepared
    @test process!(reconstructed, prepared, pixels) === nothing
    @test reconstructed == Float32.(replacement["reconstructed"])
end

@testset "pyramid rejection, views, and warmed allocation contract" begin
    prepared = prepared_fixture()
    pixels = row_major_matrix(
        Float32,
        FIXTURE["sample"]["reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        2,
    )
    reconstructed = fill(29.0f0, 3)
    pixels[3, 2] = Inf32
    @test process!(reconstructed, prepared, pixels) ===
          DenseReconstructionNonFiniteInput
    @test reconstructed == fill(29.0f0, 3)

    prepared = prepared_fixture()
    pixel_storage = zeros(Float32, 8, 2)
    pixel_view = @view pixel_storage[1:2:end, :]
    pixel_view .= row_major_matrix(
        Float32,
        FIXTURE["sample"]["reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        2,
    )
    output_storage = zeros(Float32, 6)
    output_view = @view output_storage[1:2:end]
    process!(output_view, prepared, pixel_view)
    @test output_view == Float32.(FIXTURE["sample"]["reconstructed"])
    @test checked_allocation_bytes(output_view, prepared, pixel_view) == 0
    @test admitted_allocation_bytes(output_view, prepared, pixel_view) == 0

    alias_storage = zeros(Float32, 8)
    aliased_pixels = reshape(alias_storage, 4, 2)
    @test_throws ArgumentError process!(
        @view(alias_storage[1:3]),
        prepared,
        aliased_pixels,
    )
end

@testset "invalid pyramid preparation is rejected" begin
    schema = FIXTURE["configuration"]["reconstructed_schema"]
    @test_throws ArgumentError prepare_algorithm(
        PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(
            reconstructed_count=3,
            selected_pixel_count=0,
            initial_reconstructor=nothing,
            reconstructed_schema=schema,
        ),
    )
    @test_throws DimensionMismatch prepare_algorithm(
        PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(
            reconstructed_count=3,
            selected_pixel_count=2,
            initial_reconstructor=zeros(Float32, 23),
            reconstructed_schema=schema,
        ),
    )
end

function prepared_matrix(matrix; coefficient_storage="fp32")
    return prepare_algorithm(
        PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(
            reconstructed_count=size(matrix, 1),
            selected_pixel_count=div(size(matrix, 2), 4),
            initial_reconstructor=vec(permutedims(matrix)),
            reconstructed_schema=FIXTURE["configuration"]["reconstructed_schema"],
            coefficient_storage=coefficient_storage,
        ),
    )
end

function dense_reference!(output, matrix, pixels)
    plan = DenseReconstructorPlan(matrix)
    workspace = DenseReconstructorWorkspace(plan)
    return FilterGraphAlgorithms._process_pyramid_pupil_reconstruction!(
        output, workspace, plan, pixels,
    )
end

@testset "prepared CPU pupil packing preserves full-column semantics" begin
    matrix = zeros(Float32, 128, 2048)
    for column in axes(matrix, 2)
        matrix[5, column] = Float32(mod(column, 17) - 8) / 64
        matrix[71, column] = Float32(mod(column, 11) - 5) / 128
    end
    prepared = prepared_matrix(matrix)
    execution = prepared.plan.pupil_execution
    @test execution isa FilterGraphAlgorithms._PackedPupilReconstruction{Float32}
    @test execution.active_rows == [5, 71]
    @test execution.coefficients == matrix[[5, 71], :]
    @test size(execution.coefficients, 2) == size(matrix, 2)
    # Every original column remains visible to finite-input validation even
    # when both active rows have a structural zero in that column.
    pixels = reshape(Float32.(mod.(1:2048, 29) .- 14) ./ 32, 512, 4)'
    pixels = Matrix(pixels)
    output = zeros(Float32, 128)
    reference = similar(output)
    @test dense_reference!(reference, matrix, pixels) === nothing
    @test process!(output, prepared, pixels) === nothing
    @test output ≈ reference rtol=2f-5 atol=2f-6
    matching_workspace = DenseReconstructorWorkspace(prepared.plan)
    @test length(matching_workspace.pupil_output) == size(matrix, 1)
    @test FilterGraphAlgorithms._process_pyramid_pupil_reconstruction!(
        output, matching_workspace, prepared.plan, pixels,
    ) === nothing
    @test output ≈ reference rtol=2f-5 atol=2f-6
    @test all(!signbit, output[setdiff(1:128, [5, 71])])
    process!(output, prepared, pixels)
    @test checked_allocation_bytes(output, prepared, pixels) == 0
    @test admitted_allocation_bytes(output, prepared, pixels) == 0

    for value in (0.0f0, -0.0f0, nextfloat(0.0f0))
        fill!(pixels, value)
        @test dense_reference!(reference, matrix, pixels) === nothing
        @test process!(output, prepared, pixels) === nothing
        @test reinterpret(UInt32, output) == reinterpret(UInt32, reference)
    end
    zero_column = findfirst(column -> all(iszero, @view(matrix[:, column])), axes(matrix, 2))
    @test !isnothing(zero_column)
    for value in (NaN32, Inf32, -Inf32)
        fill!(pixels, 1f0)
        ordered = reshape(permutedims(pixels), :)
        ordered[zero_column] = value
        pixels .= permutedims(reshape(ordered, 512, 4))
        fill!(output, 29f0)
        @test process!(output, prepared, pixels) === DenseReconstructionNonFiniteInput
        @test output == fill(29f0, 128)
    end

    # An extreme finite input must execute the original GEMV, including its
    # reduction order and disposable supplied-output writes on rejection.
    matrix[5, :] .= 2f0
    replace_parameter!(prepared, :reconstructor, matrix)
    @test prepared.plan.pupil_execution !== execution
    fill!(pixels, floatmax(Float32))
    @test dense_reference!(reference, matrix, pixels) === DenseReconstructionNonFiniteOutput
    @test process!(output, prepared, pixels) === DenseReconstructionNonFiniteOutput
    @test isequal(output, reference)
    fill!(pixels, 1f0)
    @test process!(output, prepared, pixels) === nothing
    fill!(prepared.workspace.pupil_output, 11f0)
    FilterGraphAlgorithms.reset!(prepared.workspace)
    @test all(iszero, prepared.workspace.ordered_input)
    @test all(iszero, prepared.workspace.pupil_output)

    # Replacement reuses fixed full-row Workspace storage across dense and
    # packed plans and across changes in active row cardinality.
    scratch = prepared.workspace.pupil_output
    replace_parameter!(prepared, :reconstructor, ones(Float32, size(matrix)))
    @test isnothing(prepared.plan.pupil_execution)
    @test process!(output, prepared, pixels) === nothing
    @test output == fill(2048f0, 128)
    matrix[92, :] .= -1f0
    replace_parameter!(prepared, :reconstructor, matrix)
    @test prepared.plan.pupil_execution.active_rows == [5, 71, 92]
    @test prepared.workspace.pupil_output === scratch
    @test process!(output, prepared, pixels) === nothing
    @test dense_reference!(reference, matrix, pixels) === nothing
    @test output ≈ reference
    @test_throws ArgumentError process!(scratch, prepared, pixels)
    @test_throws ArgumentError process!(
        @view(vec(prepared.plan.pupil_execution.coefficients)[1:128]), prepared, pixels,
    )

    # Reinterpreted scratch must not overwrite the prepared row indices before
    # the scatter. The input scratch needs the same concrete vector type.
    ordered = reinterpret(Float32, zeros(Int, 1024))
    row_indices = prepared.plan.pupil_execution.active_rows
    aliased_scratch = reinterpret(Float32, row_indices)
    aliased_workspace = DenseReconstructorWorkspace(ordered, aliased_scratch)
    saved_indices = copy(row_indices)
    @test_throws ArgumentError FilterGraphAlgorithms._process_pyramid_pupil_reconstruction!(
        output, aliased_workspace, prepared.plan, pixels,
    )
    @test row_indices == saved_indices

    alias_matrix = zeros(Float32, 128, 2048)
    alias_matrix[5, :] .= 1f0
    alias_prepared = prepared_matrix(alias_matrix)
    coefficients = alias_prepared.plan.pupil_execution.coefficients
    aliased_workspace = DenseReconstructorWorkspace(vec(coefficients), zeros(Float32, 128))
    @test_throws ArgumentError FilterGraphAlgorithms._process_pyramid_pupil_reconstruction!(
        output, aliased_workspace, alias_prepared.plan, pixels,
    )
    @test coefficients == alias_matrix[[5], :]

    # Checked views still use packed execution without allocating.
    pixel_storage = zeros(Float32, 8, 512)
    pixel_view = @view pixel_storage[1:2:end, :]
    pixel_view .= pixels
    output_storage = zeros(Float32, 256)
    output_view = @view output_storage[1:2:end]
    @test process!(output_view, prepared, pixel_view) === nothing
    @test output_view ≈ reference
    @test checked_allocation_bytes(output_view, prepared, pixel_view) == 0
end

@testset "pupil row classification uses stored coefficient bits" begin
    matrix = zeros(Float32, 128, 2048)
    matrix[5, 1] = 1f0
    matrix[17, 2] = nextfloat(0f0)
    original_mode = get_zero_subnormals()
    try
        for mode in (false, true)
            set_zero_subnormals(mode)
            prepared = prepared_matrix(matrix)
            @test prepared.plan.pupil_execution.active_rows == [5, 17]
            @test reinterpret(UInt32, prepared.plan.pupil_execution.coefficients[2, 2]) == 1
        end
    finally
        set_zero_subnormals(original_mode)
    end
    matrix[18, 3] = -0.0f0
    @test isnothing(prepared_matrix(matrix).plan.pupil_execution)
    matrix[18, 3] = 0.0f0
    matrix[65:128, :] .= 1f0
    @test isnothing(prepared_matrix(matrix).plan.pupil_execution)
end

@testset "pupil packing uses saved coefficient traffic across shapes" begin
    pixels = ones(Float32, 4, 512)
    for rows in (64, 65, 127, 513)
        matrix = zeros(Float32, rows, 2048)
        matrix[end, :] .= -0.125f0
        prepared = prepared_matrix(matrix)
        execution = prepared.plan.pupil_execution
        if rows == 64
            @test isnothing(execution)
        else
            @test execution isa FilterGraphAlgorithms._PackedPupilReconstruction{Float32}
            @test execution.active_rows == [rows]
        end
        output = zeros(Float32, rows)
        reference = similar(output)
        @test dense_reference!(reference, matrix, pixels) === nothing
        @test process!(output, prepared, pixels) === nothing
        @test output == reference
        @test checked_allocation_bytes(output, prepared, pixels) == 0
    end
end

@testset "pupil packing falls back after host arithmetic mode changes" begin
    matrix = zeros(Float32, 128, 2048)
    matrix[5, :] .= 1f0
    prepared = prepared_matrix(matrix)
    @test prepared.plan.pupil_execution isa FilterGraphAlgorithms._PackedPupilReconstruction
    pixels = fill(-1f0, 4, 512)
    output = zeros(Float32, 128)
    reference = similar(output)
    original_subnormal_mode = get_zero_subnormals()
    original_rounding = Base.Rounding.rounding_raw(Float32)
    try
        for subnormal_mode in (false, true)
            set_zero_subnormals(subnormal_mode)
            for mode in (RoundNearest, RoundDown, RoundUp, RoundToZero)
                # Float32 exposes only a getter in the public Julia API.
                # The test uses the same runtime setter as Base's own setup.
                Base.Rounding.setrounding_raw(Float32, Base.Rounding.to_fenv(mode))
                @test dense_reference!(reference, matrix, pixels) === nothing
                @test process!(output, prepared, pixels) === nothing
                @test reinterpret(UInt32, output) == reinterpret(UInt32, reference)
            end
        end
    finally
        Base.Rounding.setrounding_raw(Float32, original_rounding)
        set_zero_subnormals(original_subnormal_mode)
    end
end

@testset "row-major pupil reconstruction tails, fallback, and replacement" begin
    matrix = [Float32(mod(row * 13 + column * 7, 37) - 18) for row in 1:5, column in 1:20]
    @test isnothing(prepared_matrix(matrix).plan.pupil_execution)
    prepared = prepared_matrix(matrix; coefficient_storage="fp32-row-major")
    execution = prepared.plan.pupil_execution
    @test execution isa FilterGraphAlgorithms._RowMajorPupilReconstruction
    @test execution.coefficients == permutedims(matrix)
    workspace = prepared.workspace
    pixels = reshape(Float32.(mod.(1:20, 11) .- 5), 4, 5) .* 1f-9
    output = zeros(Float32, 5)
    reference = similar(output)
    @test maximum(abs, pixels) <= execution.safe_input_limit
    @test process!(output, prepared, pixels) === nothing
    @test dense_reference!(reference, matrix, pixels) === nothing
    @test output ≈ reference rtol=2f-5
    @test checked_allocation_bytes(output, prepared, pixels) == 0
    @test admitted_allocation_bytes(output, prepared, pixels) == 0
    @test_throws ArgumentError process!(@view(execution.coefficients[1:5]), prepared, pixels)

    # Indexed destinations may repeat a storage location. Preserve the
    # established mul! accumulation for that view.
    repeated = [1, 1, 2, 3, 4]
    expected_view = view(zeros(Float32, 4), repeated)
    output_view = view(zeros(Float32, 4), repeated)
    @test dense_reference!(expected_view, matrix, pixels) === nothing
    @test process!(output_view, prepared, pixels) === nothing
    @test reinterpret(UInt32, collect(output_view)) == reinterpret(UInt32, collect(expected_view))

    original_rounding = Base.Rounding.rounding_raw(Float32)
    original_subnormal_mode = get_zero_subnormals()
    try
        for (round_mode, flush_subnormals) in ((RoundDown, false), (RoundNearest, true))
            set_zero_subnormals(flush_subnormals)
            Base.Rounding.setrounding_raw(Float32, Base.Rounding.to_fenv(round_mode))
            @test process!(output, prepared, pixels) === nothing
            @test dense_reference!(reference, matrix, pixels) === nothing
            @test reinterpret(UInt32, output) == reinterpret(UInt32, reference)
        end
    finally
        Base.Rounding.setrounding_raw(Float32, original_rounding)
        set_zero_subnormals(original_subnormal_mode)
    end

    fill!(output, 29f0)
    pixels[end] = NaN32
    @test process!(output, prepared, pixels) === DenseReconstructionNonFiniteInput
    @test output == fill(29f0, 5)
    fill!(pixels, floatmax(Float32))
    @test process!(output, prepared, pixels) === DenseReconstructionNonFiniteOutput
    @test dense_reference!(reference, matrix, pixels) === DenseReconstructionNonFiniteOutput
    @test isequal(output, reference)

    fill!(pixels, 1f-9)
    matrix[5, 20] = 2f0
    @test replace_parameter!(prepared, :reconstructor, matrix) === prepared
    @test prepared.workspace === workspace
    @test prepared.plan.pupil_execution !== execution
    @test process!(output, prepared, pixels) === nothing
    @test dense_reference!(reference, matrix, pixels) === nothing
    @test output ≈ reference rtol=2f-5
    replacement = prepared.plan
    matrix[1, 1] = NaN32
    @test_throws ArgumentError replace_parameter!(prepared, :reconstructor, matrix)
    @test prepared.plan === replacement
end

@testset "FP16 storage survives calibration replacement" begin
    prepared = prepared_fixture(; coefficient_storage="fp16")
    @test_throws ArgumentError prepared_fixture(; coefficient_storage="invalid")
    pixels = row_major_matrix(Float32, FIXTURE["sample"]["reconstruction_pixels"], 4, 2)
    output = zeros(Float32, 3)
    @test prepared.plan.matrix isa FilterGraphAlgorithms._ScaledFloat16Reconstructor
    @test process!(output, prepared, pixels) === nothing
    @test output ≈ Float32.(FIXTURE["sample"]["reconstructed"]) rtol=1f-3
    rows, columns = size(prepared.plan.matrix)
    replacement = row_major_matrix(Float32, FIXTURE["replacement"]["reconstructor"], rows, columns)
    @test replace_parameter!(prepared, :reconstructor, replacement) === prepared
    @test prepared.plan.matrix isa FilterGraphAlgorithms._ScaledFloat16Reconstructor
    @test process!(output, prepared, pixels) === nothing
    @test output ≈ Float32.(FIXTURE["replacement"]["reconstructed"]) rtol=1f-3
    @test checked_allocation_bytes(output, prepared, pixels) == 0
end

end
