module InterActuatorStrokeLimitTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: _process_admitted!, commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "inter_actuator_stroke_limit.toml"),
)

struct ZeroBasedStrokeVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedStrokeVector) = size(vector.parent)
Base.axes(vector::ZeroBasedStrokeVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedStrokeVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedStrokeVector, index::Int) = vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedStrokeVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

function stroke_limit_prepared(edges, max_differences)
    actuator_count = maximum(maximum, edges) + 1
    return prepare_algorithm(
        InterActuatorStrokeLimitF32,
        InterActuatorStrokeLimitF32Configuration(
            actuator_count=actuator_count,
            edges=edges,
            max_differences=max_differences,
        ),
    )
end

function stroke_limit_checked_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function stroke_limit_admitted_allocation_bytes(output, prepared, input)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

stroke_limit_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
stroke_limit_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
stroke_limit_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "inter-actuator declaration preserves the FGN contract" begin
    @test algorithm_label(InterActuatorStrokeLimitF32) ==
          "inter-actuator-stroke-limit-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        InterActuatorStrokeLimitF32,
    ) === InterActuatorStrokeLimitFailure
    @test algorithm_property_policy(InterActuatorStrokeLimitF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(InterActuatorStrokeLimitF32)
    @test map(field -> field.name, configuration) == (
        :actuator_count,
        :edges,
        :max_differences,
    )
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].update isa GraphRebuildConfiguration
    @test configuration[3].update isa ConstructionConfiguration
    @test configuration[1].julia_type === Int
    @test configuration[2].julia_type === Vector{Vector{Int}}
    @test configuration[3].julia_type === Vector{Float32}

    ports = algorithm_ports(InterActuatorStrokeLimitF32)
    @test map(port -> port.name, ports) == (
        :requested,
        Symbol("stroke-limited"),
        Symbol("max-differences"),
    )
    @test ports[1].direction isa FilterGraphAlgorithms.InputPort
    @test ports[2].direction isa FilterGraphAlgorithms.OutputPort
    @test ports[3].direction isa FilterGraphAlgorithms.InputPort
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    @test ports[2].metadata_source === :requested

    prepared = stroke_limit_prepared([[0, 1], [1, 2]], Float32[1, 0.25])
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3,), (3,), (2,))
    @test getproperty.(formats, :schema) == (
        REQUESTED_PDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
        INTER_ACTUATOR_STROKE_LIMIT_V1,
    )

end

function exact_stroke_difference(first::Float32, second::Float32)
    return abs(Rational{BigInt}(first) - Rational{BigInt}(second))
end

const RESTRICTIVE_OUTPUT = Float32[
    prevfloat(1f0 / 6f0), -prevfloat(1f0 / 6f0), prevfloat(1f0 / 12f0),
]

@testset "inter-actuator projection against frozen Rust cases" begin
    for case in FIXTURE["cases"]
        input = Float32.(case["requested"])
        prepared = stroke_limit_prepared(
            case["edges"],
            Float32.(case["max_differences"]),
        )
        output = fill(71.0f0, length(input))

        @test process!(output, prepared, input) === nothing
        if case["name"] == "restrictive"
            # The Rust result exceeds 0.25 by one quarter of a Float32 ULP.
            @test exact_stroke_difference(
                Float32(case["expected"][2]),
                Float32(case["expected"][3]),
            ) > 1 // 4
            @test output == RESTRICTIVE_OUTPUT
        elseif case["name"] == "large-opposite"
            @test output == Float32[0.5, -0.5]
        else
            @test output == Float32.(case["expected"])
            @test reinterpret(UInt32, output) == UInt32.(case["expected_bits"])
        end
        for ((first, second), limit) in zip(case["edges"], case["max_differences"])
            @test exact_stroke_difference(output[first + 1], output[second + 1]) <=
                  Rational{BigInt}(Float32(limit))
        end
    end
end

@testset "opposite-sign difference remains finite for the floating range" begin
    plan = InterActuatorStrokeLimitPlan(2, [[0, 1]], [1.0])
    output = zeros(2)
    input = [floatmax(Float64), -floatmax(Float64)]
    @test FilterGraphAlgorithms._process_inter_actuator_stroke_limit!(
        output,
        plan,
        input,
    ) === nothing
    @test all(isfinite, output)
    @test output[1] ≈ 0.5 rtol=8eps(Float64)
    @test output[2] ≈ -0.5 rtol=8eps(Float64)
end

@testset "inter-actuator topology and limits reject before adoption" begin
    @test_throws ArgumentError InterActuatorStrokeLimitPlan(
        0,
        [[0, 1]],
        Float32[1],
    )
    @test_throws ArgumentError InterActuatorStrokeLimitPlan(2, [], Float32[])
    @test_throws DimensionMismatch InterActuatorStrokeLimitPlan(
        3,
        [[0, 1], [1, 2]],
        Float32[1],
    )
    @test_throws DimensionMismatch InterActuatorStrokeLimitPlan(
        2,
        [[0, 1, 2]],
        Float32[1],
    )
    @test_throws ArgumentError InterActuatorStrokeLimitPlan(
        2,
        [[0.0, 1.0]],
        Float32[1],
    )
    for edges in ([[-1, 1]], [[0, 2]], [[1, 1]], [[0, 1], [1, 0]])
        @test_throws ArgumentError InterActuatorStrokeLimitPlan(
            2,
            edges,
            ones(Float32, length(edges)),
        )
    end
    for maximum in (-1.0f0, NaN32, Inf32)
        @test_throws ArgumentError InterActuatorStrokeLimitPlan(
            2,
            [[0, 1]],
            Float32[maximum],
        )
    end

    prepared = stroke_limit_prepared([[0, 1], [1, 2]], Float32[1, 0.25])
    requested = Float32[2, -2, 1]
    output = zeros(Float32, 3)
    @test process!(output, prepared, requested) === nothing
    @test output == RESTRICTIVE_OUTPUT

    original = prepared.plan
    relaxed_storage = Float32[83, 4, 83, 4]
    relaxed = @view relaxed_storage[2:2:4]
    @test replace_parameter!(prepared, "max-differences", relaxed) === prepared
    @test prepared.plan !== original
    @test process!(output, prepared, requested) === nothing
    @test output == requested

    active = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "max-differences",
        Float32[4],
    )
    @test prepared.plan === active
    for invalid in (Float32[4, -1], Float32[4, NaN])
        @test_throws ArgumentError replace_parameter!(
            prepared,
            "max-differences",
            invalid,
        )
        @test prepared.plan === active
        @test process!(output, prepared, requested) === nothing
        @test output == requested
    end

    @test replace_parameter!(prepared, "max-differences", Float32[1, 0.25]) === prepared
    @test process!(output, prepared, requested) === nothing
    @test output == RESTRICTIVE_OUTPUT
    @test exact_stroke_difference(output[2], output[3]) <= 1 // 4
end

@testset "inter-actuator processing preserves output on checked failure" begin
    prepared = stroke_limit_prepared([[0, 1]], Float32[1])
    output = fill(71.0f0, 2)
    @test_throws DimensionMismatch process!(output, prepared, Float32[0])
    @test output == fill(71.0f0, 2)
    @test_throws DimensionMismatch process!(zeros(Float32, 1), prepared, zeros(Float32, 2))
    @test output == fill(71.0f0, 2)

    for invalid in (Float32[NaN, 0], Float32[Inf, 0], Float32[-Inf, 0])
        fill!(output, 71.0f0)
        @test process!(output, prepared, invalid) ===
              InterActuatorStrokeLimitNonFiniteInput
        @test output == fill(71.0f0, 2)
    end

    storage = Float32[1, -1, 71]
    requested = @view storage[1:2]
    overlapping = @view storage[2:3]
    @test_throws ArgumentError process!(overlapping, prepared, requested)
    @test storage == Float32[1, -1, 71]

    zero_based = ZeroBasedStrokeVector(Float32[1, -1])
    @test_throws ArgumentError process!(output, prepared, zero_based)
    @test_throws ArgumentError process!(zero_based, prepared, Float32[1, -1])
end

@testset "inter-actuator processing supports strided and in-place storage" begin
    prepared = stroke_limit_prepared([[0, 1], [1, 2]], Float32[1, 0.25])
    input_storage = fill(83.0f0, 6)
    output_storage = fill(71.0f0, 6)
    requested = @view input_storage[1:2:5]
    stroke_limited = @view output_storage[2:2:6]
    requested .= Float32[2, -2, 1]

    @test process!(stroke_limited, prepared, requested) === nothing
    @test stroke_limited == RESTRICTIVE_OUTPUT
    @test output_storage[1:2:5] == fill(71.0f0, 3)

    command = Float32[2, -2, 1]
    @test process!(command, prepared, command) === nothing
    @test command == RESTRICTIVE_OUTPUT
end

@testset "inter-actuator warmed paths allocate no Julia heap memory" begin
    prepared = stroke_limit_prepared([[0, 1], [1, 2]], Float32[1, 0.25])
    output = zeros(Float32, 3)
    valid = Float32[2, -2, 1]
    invalid = Float32[NaN, 0, 0]

    process!(output, prepared, valid)
    @test stroke_limit_checked_allocation_bytes(output, prepared, valid) == 0
    process!(output, prepared, invalid)
    @test stroke_limit_checked_allocation_bytes(output, prepared, invalid) == 0

    @test @inferred InterActuatorStrokeLimitFailure _process_admitted!(
        output,
        prepared,
        valid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test stroke_limit_admitted_allocation_bytes(output, prepared, valid) == 0
    _process_admitted!(
        output,
        prepared,
        invalid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test stroke_limit_admitted_allocation_bytes(output, prepared, invalid) == 0

    workspace = prepared.workspace
    @test stroke_limit_commit_allocation_bytes(workspace) == 0
    @test stroke_limit_discard_allocation_bytes(workspace) == 0
    @test stroke_limit_reset_allocation_bytes(workspace) == 0
end


@testset "stroke limit at Float32 rounding boundaries" begin
    smallest = nextfloat(0f0)
    largest = floatmax(Float32)
    adjacent = prevfloat(1f0)
    narrow_limit = prevfloat(1f0 - adjacent)
    near_one = prevfloat(adjacent)
    for (requested, limit, expected) in (
        (Float32[1, adjacent], narrow_limit, Float32[near_one, near_one]),
        (Float32[-1, -adjacent], narrow_limit, Float32[-near_one, -near_one]),
        (Float32[largest, -smallest], largest, Float32[prevfloat(largest), -smallest]),
        (Float32[1f38, 0], 1f-7, Float32[1f-7, 0]),
        (Float32[largest, 0], smallest, Float32[smallest, 0]),
        (Float32[largest, -largest / 2f0], smallest, Float32[smallest, -0.0]),
        (Float32[largest, -largest], smallest, Float32[0, -0.0]),
    )
        prepared = stroke_limit_prepared([[0, 1]], Float32[limit])
        output = similar(requested)
        @test process!(output, prepared, requested) === nothing
        @test isequal(output, expected)
        @test exact_stroke_difference(output[1], output[2]) <= Rational{BigInt}(limit)
        @test stroke_limit_checked_allocation_bytes(output, prepared, requested) == 0
    end
end

@testset "coupled rounding boundaries retain a useful common gain" begin
    ordinary = Float32[
        -0.021576913, 0.90252525, -0.5801698, -0.6412854, -0.6437721, 0.19386649,
        -0.86075133,
    ]
    close_pairs = Float32[1.25, nextfloat(1.25f0), 1.5, nextfloat(1.5f0)]
    for (requested, edges, limits, minimum_gain) in (
        (
            ordinary,
            [[i, i + 1] for i in 0:5],
            Float32[0.9241021, 1.8533686, 0.06111562, 0.0024867055, 1.0470482, 1.054618],
            0.9999,
        ),
        (close_pairs, [[0, 1], [2, 3]], fill(prevfloat(eps(Float32)), 2), 0.625),
    )
        prepared = stroke_limit_prepared(edges, limits)
        output = similar(requested)
        @test process!(output, prepared, requested) === nothing
        @test all(abs(Float64(y)) >= minimum_gain * abs(Float64(x)) for (x, y) in zip(requested, output))
        for ((first, second), limit) in zip(edges, limits)
            @test exact_stroke_difference(output[first + 1], output[second + 1]) <=
                  Rational{BigInt}(limit)
        end
        @test stroke_limit_checked_allocation_bytes(output, prepared, requested) == 0
    end
end

@testset "gain crosses the stored stroke rounding boundary" begin
    tiny = nextfloat(0f0)
    for value in (1f0, 1.25f0, floatmax(Float32)),
        upper in (tiny, nextfloat(tiny), prevfloat(floatmin(Float32)),
                  floatmin(Float32), 0.5f0, nextfloat(0.5f0), 1f0)
        # At an exact midpoint the low bit of upper decides which side owns
        # the tie. A non-power-of-two divisor also tests rounded quotients.
        gain = FilterGraphAlgorithms._inter_actuator_rounding_gain(
            Float64(value), upper, Val(false),
        )
        @test Float32(BigFloat(value) * BigFloat(gain)) < upper
        @test Float32(BigFloat(value) * BigFloat(nextfloat(gain))) >= upper
    end
end

@testset "stroke checks at full and partial edge-block boundaries" begin
    # A wide subtraction rounds 1 + tiny to 1. The exact represented edge
    # still needs correction, wherever it falls in the physical edge order.
    tiny = nextfloat(0f0)
    original_mode = get_zero_subnormals()
    try
        for flush in (false, true)
            set_zero_subnormals(flush) || continue
            for active in (1, 31, 32, 33, 63, 64, 65)
                edges = [[i - 1, i] for i in 1:65]
                limits = fill(2f0, length(edges))
                limits[active] = 1f0
                requested = ones(Float32, 66)
                requested[active + 1] = -tiny
                plan = stroke_limit_prepared(edges, limits)
                @test process!(requested, plan, requested) === nothing
                expected = fill(prevfloat(1f0), 66)
                expected[active + 1] = -tiny
                @test reinterpret(UInt32, requested) == reinterpret(UInt32, expected)
            end
        end
    finally
        set_zero_subnormals(original_mode)
    end
end

@testset "stroke limits apply to stored subnormals with flushing enabled" begin
    smallest = nextfloat(0f0)
    original_mode = get_zero_subnormals()
    try
        for mode in (false, true)
            set_zero_subnormals(mode) || continue
            for (requested, limit, expected_bits) in (
                (
                    Float32[3, 1.25] .* floatmin(Float32),
                    floatmin(Float32),
                    UInt32[0x00db_6db7, 0x005b_6db7],
                ),
                (Float32[1, -smallest], 1f0, UInt32[0x3f7f_ffff, 0x8000_0001]),
                (Float32[smallest, -smallest], smallest, UInt32[0x0000_0000, 0x8000_0000]),
                (Float32[floatmax(Float32), 0], smallest, UInt32[0x0000_0001, 0x0000_0000]),
                (Float32[smallest, 0], smallest, UInt32[0x0000_0001, 0x0000_0000]),
            )
                prepared = stroke_limit_prepared([[0, 1]], Float32[limit])
                output = similar(requested)
                @test process!(output, prepared, requested) === nothing
                @test reinterpret(UInt32, output) == expected_bits
                @test stroke_limit_checked_allocation_bytes(output, prepared, requested) == 0
            end
            @test_throws ArgumentError stroke_limit_prepared([[0, 1]], Float32[-smallest])
            prepared = stroke_limit_prepared([[0, 1]], Float32[0])
            @test_throws ArgumentError replace_parameter!(
                prepared, "max-differences", Float32[-smallest],
            )
            @test get_zero_subnormals() == mode
        end
    finally
        set_zero_subnormals(original_mode)
    end
end

end
