module VectorLinearCombinationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!, commit!, discard!

const SCHEMA = "org.example.ao.modal-correction/1"

function prepared_fixture(; extent=3, left_scale=1.0f0, right_scale=-1.0f0,
    schema=SCHEMA)
    return prepare_algorithm(VectorLinearCombinationF32,
        VectorLinearCombinationF32Configuration(; extent, left_scale, right_scale, schema))
end

allocated_process(output, prepared, left, right) =
    @allocated process!(output, prepared, left, right)

@testset "vector combination declaration and calibrated coordinates" begin
    prepared = prepared_fixture()
    @test algorithm_label(VectorLinearCombinationF32) == "vector-linear-combination-f32"
    @test VectorLinearCombinationF32 in algorithms()
    @test capture_support(VectorLinearCombinationF32, nothing) isa NoCapture
    @test FilterGraphAlgorithms.algorithm_failure_type(VectorLinearCombinationF32) ===
        VectorLinearCombinationFailure
    ports = algorithm_ports(VectorLinearCombinationF32)
    @test getproperty.(ports, :name) == (:left, :right, :output)
    @test ports[3].metadata_source === :left
    @test getproperty.(algorithm_formats(prepared), :shape) == ((3,), (3,), (3,))
    @test getproperty.(algorithm_formats(prepared), :schema) == (SCHEMA, SCHEMA, SCHEMA)
    fields = algorithm_configuration(VectorLinearCombinationF32)
    @test getproperty.(fields, :name) == (:extent, :left_scale, :right_scale, :schema)
    @test fields[1].update isa GraphRebuildConfiguration
    @test fields[2].update isa ConstructionConfiguration
    @test fields[3].update isa ConstructionConfiguration
    @test fields[4].update isa GraphRebuildConfiguration
end

@testset "linear combinations retain signs and reject atomically" begin
    left = Float32[4, -2, 0.5]
    right = Float32[1, 3, -0.25]
    output = zeros(Float32, 3)
    for (a, b, expected) in (
        (1.0f0, -1.0f0, Float32[3, -5, 0.75]),
        (1.0f0, 1.0f0, Float32[5, 1, 0.25]),
        (-0.5f0, 2.0f0, Float32[0, 7, -0.75]),
    )
        prepared = prepared_fixture(; left_scale=a, right_scale=b)
        @test process!(output, prepared, left, right) === nothing
        @test output == expected
    end
    prepared = prepared_fixture()
    for invalid in (NaN32, Inf32, -Inf32), side in (:left, :right)
        l, r = copy(left), copy(right)
        (side === :left ? l : r)[3] = invalid
        fill!(output, 71)
        @test process!(output, prepared, l, r) === VectorLinearCombinationNonFiniteInput
        @test output == fill(71.0f0, 3)
    end
    for (a, b, l, r) in (
        (2.0f0, 0.0f0, Float32[1, 2, floatmax(Float32)], right),
        (1.0f0, 1.0f0, Float32[1, 2, floatmax(Float32)],
            Float32[1, 2, floatmax(Float32)]),
    )
        prepared = prepared_fixture(; left_scale=a, right_scale=b)
        fill!(output, 71)
        @test process!(output, prepared, l, r) === VectorLinearCombinationNonFiniteOutput
        @test output == fill(71.0f0, 3)
    end
    # A later invalid input takes priority over an earlier finite-input overflow.
    prepared = prepared_fixture(; left_scale=2.0f0, right_scale=0.0f0)
    @test process!(output, prepared, Float32[floatmax(Float32), 2, 3],
        Float32[1, 2, NaN]) === VectorLinearCombinationNonFiniteInput
    @test output == fill(71.0f0, 3)
    @test process!(output, prepared_fixture(; left_scale=0.0f0),
        Float32[1, 2, NaN], right) === VectorLinearCombinationNonFiniteInput
end

@testset "staging permits overlap and reset reuses storage" begin
    prepared = prepared_fixture()
    storage = Float32[1, 99, 2, 99, 4, 99, 8]
    left = @view storage[1:2:5]
    output = @view storage[3:2:7]
    right = Float32[0.5, 1, 2]
    @test process!(output, prepared, left, right) === nothing
    @test storage == Float32[1, 99, 0.5, 99, 1, 99, 2]
    scratch = prepared.workspace.candidate_output
    @test reset!(prepared) === prepared
    @test prepared.workspace.candidate_output === scratch
    @test scratch == zeros(Float32, 3)
    @test commit!(prepared.workspace) === prepared.workspace
    @test discard!(prepared.workspace) === prepared.workspace
    @test_throws ArgumentError process!(scratch, prepared, left, right)
    @test_throws ArgumentError process!(output, prepared, scratch, right)
    @test_throws DimensionMismatch process!(output, prepared, Float32[1, 2], right)
    l = Float32[3, 2, 1]
    @test process!(output, prepared, l, right) === nothing
    @test allocated_process(output, prepared, l, right) == 0
    l[3] = NaN32
    process!(output, prepared, l, right)
    @test allocated_process(output, prepared, l, right) == 0
end

@testset "proven finite combinations retain Float32 bits and staging" begin
    FGA = FilterGraphAlgorithms
    metadata = FGA._NO_SAMPLE_METADATA
    largest = floatmax(Float32)
    tiny = nextfloat(0f0)
    extent = 259
    values = Float32[largest, -largest, 0f0, -0f0, tiny, -tiny, 1f0, -1f0]
    for (a, b) in ((0.5f0, 0f0), (-1f0, -0f0), (-0f0, 1f0))
        prepared = prepared_fixture(; extent, left_scale=a, right_scale=b)
        @test FGA._finite_combination_result(prepared.plan)
        storage = zeros(Float32, 2extent + 1)
        left = view(storage, 1:2:(2extent - 1))
        output = view(storage, 3:2:(2extent + 1))
        left .= (values[mod1(i, length(values))] for i in 1:extent)
        expected = [fma(a, v, b * v) for v in left]
        scratch = prepared.workspace.candidate_output
        @test FGA._process_finite_combination!(output, prepared, left, left, metadata, metadata) === nothing
        @test isequal(output, expected)
        @test isequal(scratch, expected)
        @test_throws ArgumentError FGA._process_finite_combination!(
            scratch, prepared, left, left, metadata, metadata)
        @test_throws DimensionMismatch FGA._process_finite_combination!(
            output, prepared, view(left, 1:2), left, metadata, metadata)

        # Public callers still validate both operands, including a zero-scaled one.
        invalid = copy(output)
        invalid[end] = NaN32
        @test process!(output, prepared, iszero(a) ? invalid : left,
            iszero(b) ? invalid : left) === VectorLinearCombinationNonFiniteInput
        @test isequal(output, expected)
    end
    for (a, b) in ((nextfloat(1f0), 0f0), (0f0, -2f0), (0.75f0, 0.75f0))
        prepared = prepared_fixture(; left_scale=a, right_scale=b)
        @test !FGA._finite_combination_result(prepared.plan)
    end
end

@testset "invalid preparation never allocates oversized storage" begin
    for extent in (0, -1, Int(typemax(UInt32) ÷ sizeof(Float32)) + 1)
        @test_throws ArgumentError prepared_fixture(; extent)
    end
    for scale in (NaN32, Inf32, -Inf32)
        @test_throws ArgumentError prepared_fixture(; left_scale=scale)
        @test_throws ArgumentError prepared_fixture(; right_scale=scale)
    end
    @test_throws ArgumentError prepared_fixture(; schema="invalid-schema")
end

@testset "AO extents retain scalar bits, tails, and recovery" begin
    for extent in (4, 253, 277, 512, 4096, 16384)
        prepared = prepared_fixture(; extent, left_scale=0.75f0, right_scale=-0.25f0)
        left = Float32[Float32(mod(i, 31) - 15) / 7f0 for i in 1:extent]
        right = Float32[Float32(mod(i, 23) - 11) / 3f0 for i in 1:extent]
        expected = similar(left)
        for i in eachindex(expected)
            expected[i] = setprecision(BigFloat,4096) do
                Float32(BigFloat(.75f0)*BigFloat(left[i])+BigFloat(-.25f0*right[i]))
            end
        end
        output = fill(71f0, extent)
        @test process!(output, prepared, left, right) === nothing
        @test reinterpret(UInt32, output) == reinterpret(UInt32, expected)
        @test allocated_process(output, prepared, left, right) == 0
        saved = left[end]
        left[end] = NaN32
        fill!(output, 71f0)
        @test process!(output, prepared, left, right) === VectorLinearCombinationNonFiniteInput
        @test all(==(71f0), output)
        left[end] = saved
        @test process!(output, prepared, left, right) === nothing
        @test reinterpret(UInt32, output) == reinterpret(UInt32, expected)
    end
end

@testset "validation preserves rejection precedence across the vector" begin
    extent = 4097
    prepared = prepared_fixture(; extent, left_scale=2f0, right_scale=0f0)
    right = ones(Float32, extent)
    for overlap in (false, true)
        if overlap
            storage = fill(71f0, 2 * extent + 1)
            left = view(storage, 1:2:(2 * extent - 1))
            output = view(storage, 3:2:(2 * extent + 1))
        else
            left = zeros(Float32, extent)
            output = fill(71f0, extent)
        end
        left .= 1:extent
        expected = 2f0 .* left
        for (overflow_index, invalid_index) in ((1, extent), (1024, 1025), (extent, 1))
            saved = left[overflow_index]
            left[overflow_index] = floatmax(Float32)
            previous = copy(output)
            right[invalid_index] = NaN32
            @test process!(output, prepared, left, right) === VectorLinearCombinationNonFiniteInput
            @test output == previous
            right[invalid_index] = 1f0
            @test process!(output, prepared, left, right) === VectorLinearCombinationNonFiniteOutput
            @test output == previous
            left[overflow_index] = saved
        end
        @test process!(output, prepared, left, right) === nothing
        @test output == expected
    end
end

@testset "vector FMA rounds cancellation once and preserves validation" begin
    largest = floatmax(Float32)
    @test !isfinite(2f0 * largest)
    for extent in (1, 4097)
        prepared = prepared_fixture(; extent, left_scale=2f0, right_scale=-1f0)
        left = fill(largest, extent)
        right = copy(left)
        output = fill(71f0, extent)
        @test process!(output, prepared, left, right) === nothing
        @test output == right
        @test allocated_process(output, prepared, left, right) == 0
        for invalid in (NaN32, Inf32, -Inf32)
            left[end] = invalid
            fill!(output, 71f0)
            @test process!(output, prepared, left, right) === VectorLinearCombinationNonFiniteInput
            @test all(==(71f0), output)
        end
    end
    prepared=prepared_fixture(;extent=277,left_scale=.63f0,right_scale=-.27f0)
    left=Float32[Float32(sin(i))/7f0 for i in 1:277]
    right=Float32[Float32(cos(i))/3f0 for i in 1:277]
    output=similar(left)
    rounded=setprecision(BigFloat,4096) do
        Float32[Float32(BigFloat(.63f0)*BigFloat(left[i])+BigFloat(-.27f0*right[i])) for i in 1:277]
    end
    unfused=.63f0 .* left .+ -.27f0 .* right
    @test process!(output,prepared,left,right) === nothing
    @test isequal(output,rounded)
    # Each separate operation contributes at most eps times the sum of product
    # magnitudes; include one extra eps for the fused result's final rounding.
    @test all(abs.(output.-unfused) .<= 4eps(Float32).*(abs.(.63f0 .* left).+abs.(-.27f0 .* right)))
    @test allocated_process(output,prepared,left,right)==0
end

end
