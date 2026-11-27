module IncrementalDenseReconstructionTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const MEASUREMENTS_SCHEMA = "org.example.ao.measurement-block/1"
const COLUMNS_SCHEMA = "org.example.ao.reconstructor-column-indices/1"
const COUNT_SCHEMA = "org.example.ao.measurement-block-count/1"
const RECONSTRUCTED_SCHEMA = "org.example.ao.reconstructed/1"
const RECONSTRUCTOR_SCHEMA = "org.example.ao.reconstructor/1"

struct ReadCountingVector{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Vector{Int}
end

Base.size(vector::ReadCountingVector) = size(vector.values)

function Base.getindex(vector::ReadCountingVector, index::Int)
    vector.reads[index] += 1
    return vector.values[index]
end

function prepared_fixture(; reconstructor=Float32[1, 2, 3, 4, -1, 0.5, 2, -2],
        coefficient_storage="fp32", measurement_count=4, measurement_block_capacity=3,
        reconstructed_count=2)
    return prepare_algorithm(
        IncrementalDenseReconstructorF32,
        IncrementalDenseReconstructorF32Configuration(;
            measurement_count,
            measurement_block_capacity,
            blocks_per_sample=2,
            reconstructed_count,
            initial_reconstructor=reconstructor,
            measurements_schema=MEASUREMENTS_SCHEMA,
            column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA,
            reconstructed_schema=RECONSTRUCTED_SCHEMA,
            reconstructor_schema=RECONSTRUCTOR_SCHEMA,
            coefficient_storage=coefficient_storage,
        ),
    )
end

@testset "Incremental coefficient storage selection and replacement" begin
    @test_throws ArgumentError prepared_fixture(; coefficient_storage="bf16")
    initial = Float32[1, 2, 3, 4, -1, 0.5, 2, -2] .* 2f0^-30
    prepared = prepared_fixture(; reconstructor=initial, coefficient_storage="fp16")
    expected = permutedims(reshape(initial, 4, 2))
    @test Matrix(prepared.plan.matrix) == expected
    @test eltype(prepared.plan.matrix.coefficients) === Float16
    @test algorithm_formats(prepared)[5].element_type === Float32
    replace_parameter!(prepared, :reconstructor, 2f0 .* expected)
    @test Matrix(prepared.plan.matrix) == 2f0 .* expected
    @test eltype(prepared.plan.matrix.coefficients) === Float16
    retained = prepared.plan
    invalid = copy(expected)
    invalid[1] = NaN32
    @test_throws ArgumentError replace_parameter!(prepared, :reconstructor, invalid)
    @test prepared.plan === retained
end

function process_allocation_bytes(
    reconstructed,
    prepared,
    measurements,
    columns,
    count,
    metadata,
)
    return @allocated process!(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        metadata,
    )
end

@testset "incremental prepared storage is concrete and precision-parametric" begin
    prepared = prepared_fixture()
    @test typeof(prepared.plan) ===
          DenseReconstructorPlan{Float32,Matrix{Float32}}
    @test typeof(prepared.workspace) === IncrementalDenseReconstructionWorkspace{
        Float32,
        Vector{Float32},
        Vector{Bool},
    }
    @test fieldtype(typeof(prepared.plan), :matrix) === Matrix{Float32}
    @test fieldtype(typeof(prepared.plan), :pupil_execution) === Union{
        Nothing,FilterGraphAlgorithms._PackedPupilReconstruction{Float32},
        FilterGraphAlgorithms._RowMajorPupilReconstruction,
    }
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))

    plan = DenseReconstructorPlan(Float64[1 2 3 4; -1 0.5 2 -2])
    workspace = IncrementalDenseReconstructionWorkspace(plan)
    @test typeof(plan) === DenseReconstructorPlan{Float64,Matrix{Float64}}
    @test typeof(workspace) === IncrementalDenseReconstructionWorkspace{
        Float64,
        Vector{Float64},
        Vector{Bool},
    }

    reconstructed = Float64[71, 72]
    first = FilterGraphAlgorithms._process_incremental_dense_reconstruction!(
        reconstructed,
        workspace,
        plan,
        Float64[3, 1],
        UInt32[2, 0],
        UInt32[2],
        SampleMetadata(9, 0),
    )
    @test first == AlgorithmProcessDeferred()
    result = @inferred Union{
        AlgorithmProcessResult,
        IncrementalDenseReconstructionFailure,
    } FilterGraphAlgorithms._process_incremental_dense_reconstruction!(
        reconstructed,
        workspace,
        plan,
        Float64[4, 2],
        UInt32[3, 1],
        UInt32[2],
        SampleMetadata(9, 1; terminal=true),
    )
    @test result == AlgorithmProcessComplete(
        SampleMetadata(9, 0; terminal=true),
    )
    @test reconstructed == Float64[30, -2]
end

@testset "incremental dense reconstruction accepts arbitrary column order" begin
    prepared = prepared_fixture()
    @test algorithm_label(IncrementalDenseReconstructorF32) ==
          "incremental-dense-reconstructor-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        IncrementalDenseReconstructorF32,
    ) === IncrementalDenseReconstructionFailure
    @test map(port -> port.name, algorithm_ports(IncrementalDenseReconstructorF32)) == (
        :measurements,
        :columns,
        :count,
        :reconstructed,
        :reconstructor,
    )
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3,), (3,), (1,), (2,), (2, 4))
    @test all(format -> format.rate == (0x00000002, 0x00000001), formats[1:3])
    @test algorithm_ports(IncrementalDenseReconstructorF32)[4].conditional

    reconstructed = Float32[71, 72]
    measurements = Float32[3, 1, 0]
    columns = UInt32[2, 0, 0]
    count = UInt32[2]
    first = SampleMetadata(10, 0)
    @test process!(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        (measurements=first, columns=nothing, count=nothing),
    ) == AlgorithmProcessDeferred()
    @test reconstructed == Float32[71, 72]
    @test prepared.workspace.seen_count == 2

    measurements .= Float32[4, 2, 0]
    columns .= UInt32[3, 1, 0]
    count[1] = 2
    terminal = SampleMetadata(10, 1; terminal=true)
    result = process!(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        (measurements=terminal, columns=nothing, count=nothing),
    )
    @test result == AlgorithmProcessComplete(SampleMetadata(terminal; offset=0))
    @test reconstructed == Float32[30, -2]
    @test !prepared.workspace.active
end

@testset "incremental rejection is explicit and failure-atomic" begin
    reconstructed = Float32[71, 72]
    measurements = Float32[1, 2, 3]
    columns = UInt32[0, 1, 2]
    count = UInt32[2]

    duplicate = prepared_fixture()
    process!(
        reconstructed,
        duplicate,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(20, 0),
            columns=nothing,
            count=nothing,
        ),
    )
    columns .= UInt32[1, 3, 0]
    @test process!(
        reconstructed,
        duplicate,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(20, 1; terminal=true),
            columns=nothing,
            count=nothing,
        ),
    ) === IncrementalDenseReconstructionDuplicateColumn
    @test reconstructed == Float32[71, 72]
    @test !duplicate.workspace.active

    out_of_range = prepared_fixture()
    columns .= UInt32[4, 0, 0]
    count[1] = 1
    @test process!(
        reconstructed,
        out_of_range,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(21, 0),
            columns=nothing,
            count=nothing,
        ),
    ) === IncrementalDenseReconstructionColumnOutOfRange

    nonfinite = prepared_fixture()
    measurements[1] = Inf32
    columns[1] = 0
    @test process!(
        reconstructed,
        nonfinite,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(22, 0),
            columns=nothing,
            count=nothing,
        ),
    ) === IncrementalDenseReconstructionNonFiniteInput

    incomplete = prepared_fixture()
    measurements[1:2] .= Float32[1, 2]
    columns[1:2] .= UInt32[0, 1]
    count[1] = 2
    @test process!(
        reconstructed,
        incomplete,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(23, 0; terminal=true),
            columns=nothing,
            count=nothing,
        ),
    ) === IncrementalDenseReconstructionIncompleteSample
    @test reconstructed == Float32[71, 72]

    excessive = prepared_fixture()
    count[1] = 4
    @test process!(
        reconstructed,
        excessive,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(24, 0),
            columns=nothing,
            count=nothing,
        ),
    ) === IncrementalDenseReconstructionCountOutOfRange
end

@testset "incremental outcomes and lifecycle allocate no Julia heap" begin
    prepared = prepared_fixture()
    reconstructed = zeros(Float32, 2)
    measurements = Float32[3, 1, 0]
    columns = UInt32[2, 0, 0]
    count = UInt32[2]
    first_metadata = (
        measurements=SampleMetadata(30, 0),
        columns=nothing,
        count=nothing,
    )

    process!(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        first_metadata,
    )
    discard!(prepared.workspace)
    @test process_allocation_bytes(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(31, 0),
            columns=nothing,
            count=nothing,
        ),
    ) == 0
    measurements .= Float32[4, 2, 0]
    columns .= UInt32[3, 1, 0]
    @test process_allocation_bytes(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(31, 1; terminal=true),
            columns=nothing,
            count=nothing,
        ),
    ) == 0
    @test reconstructed == Float32[30, -2]

    reset!(prepared.workspace)
    measurements .= Float32[1, 2, 0]
    columns .= UInt32[0, 0, 0]
    @test process_allocation_bytes(
        reconstructed,
        prepared,
        measurements,
        columns,
        count,
        (
            measurements=SampleMetadata(32, 0),
            columns=nothing,
            count=nothing,
        ),
    ) == 0
    @test @allocated(commit!(prepared.workspace)) == 0
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
end


# Scientific oracle: exact products and sum in 256-bit arithmetic, independently
# of the candidate's ordered single-rounding recurrence.
function reconstruction_reference(matrix, measurements, columns)
    return setprecision(BigFloat, 256) do
        [sum(BigFloat(matrix[row, column]) * BigFloat(measurements[index])
             for (index, column) in enumerate(columns)) for row in axes(matrix, 1)]
    end
end

@testset "incremental ordered FMA precision, tails and independent output partitions" begin
    for T in (Float32, Float64), outputs in (1, 7, 17, 277), mode in (:normal, :nearzero, :cancellation)
        count = 33
        matrix = T[sin(0.17row + 0.31column) for row in 1:outputs, column in 1:count]
        values = T[cos(0.27column) for column in 1:count]
        mode === :nearzero && (values .*= T(1e-30))
        if mode === :cancellation
            matrix[:, 2:2:32] .= matrix[:, 1:2:31]
            values[2:2:32] .= -values[1:2:31]
            values[end] = eps(T)
        end
        order = vcat(collect(2:2:count), collect(1:2:count))
        candidate = zeros(T, outputs)
        partitioned = copy(candidate)
        ordered = copy(candidate)
        for column in order
            FilterGraphAlgorithms._accumulate_incremental_dense_reconstruction_rows!(
                candidate, matrix, column, values[column], eachindex(candidate))
            split = div(outputs, 2)
            for rows in (1:split, (split + 1):outputs)
                FilterGraphAlgorithms._accumulate_incremental_dense_reconstruction_rows!(
                    partitioned, matrix, column, values[column], rows)
            end
            for row in eachindex(ordered)
                ordered[row] = setprecision(BigFloat, 256) do
                    T(BigFloat(matrix[row, column]) * BigFloat(values[column]) + BigFloat(ordered[row]))
                end
            end
        end
        @test reinterpret(T === Float32 ? UInt32 : UInt64, candidate) == reinterpret(T === Float32 ? UInt32 : UInt64, ordered)
        @test reinterpret(T === Float32 ? UInt32 : UInt64, partitioned) == reinterpret(T === Float32 ? UInt32 : UInt64, ordered)
        reference = reconstruction_reference(matrix, values[order], order)
        for row in eachindex(candidate)
            scale = sum(abs(BigFloat(matrix[row, column]) * BigFloat(values[column])) for column in order)
            @test abs(BigFloat(candidate[row]) - reference[row]) <=
                  2count * eps(T) * scale + count * BigFloat(nextfloat(zero(T)))
        end
    end
end

@testset "incremental FMA cancellation and overflow change is explicit" begin
    for T in (Float32, Float64)
        delta = eps(T)
        matrix = fill(one(T) + delta, 17, 1)
        candidate = fill(-one(T), 17)
        FilterGraphAlgorithms._accumulate_incremental_dense_reconstruction_rows!(
            candidate, matrix, 1, one(T) - delta, eachindex(candidate))
        @test candidate == fill(-(delta * delta), 17)
        @test (one(T) + delta) * (one(T) - delta) - one(T) == zero(T)

        maximum = floatmax(T)
        plan = DenseReconstructorPlan(reshape(T[-maximum, maximum], 1, 2))
        workspace = IncrementalDenseReconstructionWorkspace(plan)
        output = T[71]
        status = FilterGraphAlgorithms._process_incremental_dense_reconstruction!(
            output, workspace, plan, T[1, 2], UInt32[0, 1], UInt32[2],
            SampleMetadata(80, 0; terminal=true))
        @test status isa AlgorithmProcessResult && status.complete
        @test output == T[maximum]
        commit!(workspace)
        @test !workspace.active && all(iszero, workspace.candidate_reconstructed)
        status = FilterGraphAlgorithms._process_incremental_dense_reconstruction!(
            output, workspace, plan, T[2, 1], UInt32[1, 0], UInt32[2],
            SampleMetadata(81, 0; terminal=true))
        @test status === IncrementalDenseReconstructionNonFiniteOutput
        @test output == T[maximum]
        @test !workspace.active && all(iszero, workspace.candidate_reconstructed)
        reset!(workspace)
        @test !workspace.pending_discontinuity && iszero(workspace.sequence)
    end
end

@testset "incremental blocks preserve ordered FMA through views" begin
    rows, complete_count, capacity = 65, 13, 7
    matrix = Float32[sin(0.17row + 0.31column) for row in 1:rows, column in 1:complete_count]
    for storage in ("fp32", "fp16")
        prepared = prepared_fixture(; reconstructor=vec(permutedims(matrix)),
            coefficient_storage=storage, measurement_count=complete_count,
            measurement_block_capacity=capacity, reconstructed_count=rows)
        output = view(fill(71f0, 2rows), 1:2:2rows)
        measurements = view(fill(NaN32, 2capacity), 2:2:2capacity)
        columns = view(fill(typemax(UInt32), 2capacity), 2:2:2capacity)
        count = UInt32[capacity]
        order = UInt32[12, 3, 7, 0, 4, 6, 9, 8, 1, 10, 2, 5, 11]
        expected = zeros(Float32, rows)
        for (block, indices) in enumerate((1:capacity, (capacity + 1):complete_count))
            count[1] = length(indices)
            fill!(measurements, NaN32)
            fill!(columns, typemax(UInt32))
            for (item, index) in enumerate(indices)
                measurements[item] = Float32(cos(0.27index))
                columns[item] = order[index]
            end
            for row in eachindex(expected), item in 1:count[1]
                coefficient = prepared.plan.matrix[row, Int(columns[item]) + 1]
                expected[row] = fma(coefficient, measurements[item], expected[row])
            end
            sample = SampleMetadata(90, block - 1; terminal=(block == 2))
            result = process!(output, prepared, measurements, columns, count,
                (; measurements=sample, columns=nothing, count=nothing))
            if block == 1
                @test result == AlgorithmProcessDeferred()
                @test all(==(71f0), output)
                @test isequal(prepared.workspace.candidate_reconstructed, expected)
                @test findall(prepared.workspace.seen) == sort(Int.(columns) .+ 1)
            else
                @test result == AlgorithmProcessComplete(SampleMetadata(sample; offset=0))
                @test isequal(output, expected)
            end
        end
        columns .= UInt32[0, 1, 2, 3, 4, 4, 6]
        count[1] = capacity
        measurements[6] = NaN32
        @test process!(output, prepared, measurements, columns, count,
            (; measurements=SampleMetadata(91), columns=nothing, count=nothing)) ===
            IncrementalDenseReconstructionDuplicateColumn
        @test isequal(output, expected)
        @test iszero(prepared.workspace.seen_count) && !any(prepared.workspace.seen)

        aliased_columns = reinterpret(UInt32, view(prepared.workspace.candidate_reconstructed, 1:capacity))
        fill!(measurements, 1f0)
        @test process!(output, prepared, measurements, aliased_columns, count,
            (; measurements=SampleMetadata(92), columns=nothing, count=nothing)) ===
            IncrementalDenseReconstructionColumnOutOfRange
        @test isequal(output, expected)
    end
end

@testset "incremental generic measurement and column vectors are read once" begin
    prepared = prepared_fixture(; measurement_block_capacity=4)
    output = zeros(Float32, 2)
    measurements = Float32[4, 2, 1, 3]
    columns = UInt32[3, 1, 0, 2]
    count = UInt32[4]
    reads = zeros(Int, 4)
    for (block_measurements, block_columns) in (
        (ReadCountingVector(measurements, reads), columns),
        (measurements, ReadCountingVector(columns, reads)),
    )
        fill!(reads, 0)
        sample = SampleMetadata(93; terminal=true)
        @test process!(output, prepared, block_measurements, block_columns, count,
            (; measurements=sample, columns=nothing, count=nothing)) == AlgorithmProcessComplete(sample)
        @test output == Float32[30, -2]
        @test reads == ones(Int, 4)
    end
end

@testset "PWFS and SHWFS helpers preserve grouped ordered FMA" begin
    outputs = 17
    matrix = fill(1.0f0 + eps(Float32), outputs, 4)
    pixels = fill(1.0f0 - eps(Float32), 4, 1)
    candidate = fill(-1.0f0, outputs)
    FilterGraphAlgorithms._accumulate_pyramid_row_reconstruction_rows!(
        candidate, matrix, pixels, 1, 1, 1, 1, eachindex(candidate))
    @test candidate == fill(-eps(Float32)^2, outputs)
    slopes = Float32[1.0f0-eps(Float32) 0]
    fill!(candidate, -1)
    FilterGraphAlgorithms._accumulate_shack_hartmann_reconstruction_rows!(
        candidate, matrix, slopes, 1:1, eachindex(candidate))
    @test candidate == fill(-eps(Float32)^2, outputs)
end

@testset "row blocks use physical identity independently of Header publication" begin
    FGA=FilterGraphAlgorithms
    domain=ntuple(UInt8,16)
    id=SampleAcquisitionIdentity(domain,7,typemax(UInt64))
    time=SampleReferenceTimestamp(123_000_000_456)
    initial=SampleMetadata(100,0;acquisition_identity=id,exposure_duration_nanoseconds=10,reference_timestamp=time)
    terminal=SampleMetadata(initial;sequence=101,offset=1,terminal=true)
    frame=FGA.RowBlockFrame()
    @test first(FGA._accept_row_block!(frame,initial,0,1,2))
    @test first(FGA._accept_row_block!(frame,terminal,1,1,2))
    for changed in (SampleMetadata(terminal;acquisition_identity=SampleAcquisitionIdentity(domain,8,typemax(UInt64))),
            SampleMetadata(terminal;exposure_duration_nanoseconds=11),
            SampleMetadata(terminal;reference_timestamp=SampleReferenceTimestamp(124_000_000_456)))
        frame=FGA.RowBlockFrame()
        @test first(FGA._accept_row_block!(frame,initial,0,1,2))
        @test !first(FGA._accept_row_block!(frame,changed,1,1,2))
    end
end

end
