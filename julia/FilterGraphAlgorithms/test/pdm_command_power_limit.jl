module PdmCommandPowerLimitTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pdm_command_power_limit.toml"),
)

struct ZeroBasedPowerVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedPowerVector) = size(vector.parent)
Base.axes(vector::ZeroBasedPowerVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedPowerVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedPowerVector, index::Int) = vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedPowerVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

function power_limit_prepared(actuator_count::Int, max_command_power::Float32)
    return prepare_algorithm(
        PdmCommandPowerLimitF32,
        PdmCommandPowerLimitF32Configuration(
            actuator_count=actuator_count,
            max_command_power=max_command_power,
        ),
    )
end

function power_limit_checked_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function power_limit_admitted_allocation_bytes(output, prepared, input)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

power_limit_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
power_limit_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
power_limit_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

exact_command_power(command) = sum(value -> abs2(Rational{BigInt}(value)), command)

@testset "PDM command-power declaration preserves the FGN contract" begin
    @test algorithm_label(PdmCommandPowerLimitF32) ==
          "pdm-command-power-limit-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        PdmCommandPowerLimitF32,
    ) === PdmCommandPowerLimitFailure
    @test algorithm_property_policy(PdmCommandPowerLimitF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(PdmCommandPowerLimitF32)
    @test map(field -> field.name, configuration) == (
        :actuator_count,
        :max_command_power,
    )
    @test configuration[1].julia_type === Int
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].julia_type === Float32
    @test configuration[2].update isa ConstructionConfiguration

    ports = algorithm_ports(PdmCommandPowerLimitF32)
    @test map(port -> port.name, ports) == (:requested, Symbol("power-limited"))
    @test ports[1].direction isa FilterGraphAlgorithms.InputPort
    @test ports[2].direction isa FilterGraphAlgorithms.OutputPort
    @test all(port -> port.role isa FilterGraphAlgorithms.FrameData, ports)
    @test all(port -> port.element_type === Float32, ports)
    @test isnothing(ports[1].metadata_source)
    @test ports[2].metadata_source === :requested

    prepared = power_limit_prepared(277, 4.0f0)
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((277,), (277,))
    @test getproperty.(formats, :schema) == (
        REQUESTED_PDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
    )

end

@testset "PDM command-power projection from frozen Rust inputs" begin
    for case in FIXTURE["cases"]
        input = Float32.(case["requested"])
        prepared = power_limit_prepared(
            length(input),
            Float32(case["max_command_power"]),
        )
        output = fill(71.0f0, length(input))

        @test process!(output, prepared, input) === nothing
        power = Float32(case["max_command_power"])
        @test exact_command_power(output) <= Rational{BigInt}(power)
        if case["name"] in ("projected", "large-finite")
            gain = sqrt(Float64(power) / sum(value -> abs2(Float64(value)), input))
            @test output ≈ Float64.(input) .* gain rtol=2eps(Float32)
        else
            @test reinterpret(UInt32, output) == UInt32.(case["expected_bits"])
        end
    end
end

@testset "Float32 command power stays within the exact configured bound" begin
    tiny = nextfloat(0.0f0)
    for exponent in (-148, -126, -100, -64, -2, 0, 64, 100, 126)
        value = ldexp(1.0f0, exponent ÷ 2)
        power = abs2(value)
        prepared = power_limit_prepared(3, power)
        for input in (Float32[value, tiny, -tiny], Float32[tiny, -tiny, -value])
            @test sum(x -> abs2(Float64(x)), input) == Float64(power)
            @test exact_command_power(input) > Rational{BigInt}(power)
            output = copy(input)
            @test process!(output, prepared, output) === nothing
            @test exact_command_power(output) <= Rational{BigInt}(power)
            @test maximum(abs, output) == prevfloat(value)
            @test all(signbit.(output) .== signbit.(input))
        end

        input = Float32[-0.0, -value, 0.0]
        output = similar(input)
        @test process!(output, prepared, input) === nothing
        @test isequal(output, input)
    end

    # 4^-28 + 3 * sum(4^-j, j=1:28) equals 1. The first square is smaller
    # than the rounding interval when subtracting it from Float64(1).
    boundary = vcat(ldexp(1.0f0, -28), repeat(ldexp.(1.0f0, -1:-1:-28); inner=3))
    @test exact_command_power(boundary) == 1
    output = similar(boundary)
    prepared = power_limit_prepared(length(boundary), 1.0f0)
    @test process!(output, prepared, boundary) === nothing
    @test isequal(output, boundary)

    for (power, input) in (
        (0.0f0, Float32[tiny, -tiny, -0.0]),
        (tiny, Float32[floatmax(Float32), -floatmax(Float32), 0.0]),
        (floatmax(Float32), Float32[floatmax(Float32), -0.0, tiny]),
        (1.0f0, Float32[floatmax(Float32), -floatmax(Float32), -1.0f23]),
        (0.8125f0, Float32[-0.75, 0.5, tiny]),
        (0.8125f0, Float32[-0.75, 0.5, -0.0]),
    )
        prepared = power_limit_prepared(length(input), power)
        output = similar(input)
        @test process!(output, prepared, input) === nothing
        @test exact_command_power(output) <= Rational{BigInt}(power)
        @test all(signbit.(output) .== signbit.(input))
        if exact_command_power(input) <= Rational{BigInt}(power)
            @test isequal(output, input)
        else
            gain = sqrt(Float64(power) / sum(value -> abs2(Float64(value)), input))
            @test output ≈ Float64.(input) .* gain rtol=2eps(Float32)
        end
    end
end

@testset "Float32 command power uses stored subnormals in either host mode" begin
    tiny = nextfloat(0.0f0)
    cases = (
        (Float32[1, -tiny], 1.0f0, Float32[prevfloat(1.0f0), -tiny]),
        (Float32[tiny], 0.0f0, Float32[0.0]),
        (Float32[-tiny], tiny, Float32[-tiny]),
        (Float32[1, -2tiny], 0.25f0, Float32[prevfloat(0.5f0), -tiny]),
    )
    outputs = map(case -> similar(case[1]), cases)
    original_mode = get_zero_subnormals()
    try
        set_zero_subnormals(true)
        active_mode = get_zero_subnormals()
        @test_throws ArgumentError PdmCommandPowerLimitPlan(1, -tiny)
        for ((input, power, _), output) in zip(cases, outputs)
            prepared = power_limit_prepared(length(input), power)
            @test process!(output, prepared, input) === nothing
            @test get_zero_subnormals() == active_mode
            @test power_limit_checked_allocation_bytes(output, prepared, input) == 0
        end
    finally
        set_zero_subnormals(original_mode)
    end
    for ((_, power, expected), output) in zip(cases, outputs)
        @test reinterpret(UInt32, output) == reinterpret(UInt32, expected)
        @test exact_command_power(output) <= Rational{BigInt}(power)
    end
end

@testset "scaled norm remains finite for the complete floating range" begin
    plan = PdmCommandPowerLimitPlan(2, 1.0)
    output = zeros(2)
    input = [floatmax(Float64), -floatmax(Float64)]
    @test FilterGraphAlgorithms._process_pdm_command_power_limit!(
        output,
        plan,
        input,
    ) === nothing
    expected = inv(sqrt(2.0))
    @test all(isfinite, output)
    @test output[1] ≈ expected rtol=8eps(Float64)
    @test output[2] ≈ -expected rtol=8eps(Float64)
    @test sum(abs2, output) ≈ 1.0 rtol=16eps(Float64)
end

@testset "PDM command-power checked boundary preserves caller-owned output" begin
    for maximum in (-1.0f0, NaN32, Inf32)
        @test_throws ArgumentError PdmCommandPowerLimitPlan(2, maximum)
    end
    @test_throws ArgumentError PdmCommandPowerLimitPlan(0, 1.0f0)
    @test_throws ArgumentError power_limit_prepared(0, 1.0f0)

    prepared = power_limit_prepared(2, 1.0f0)
    output = fill(71.0f0, 2)
    @test_throws DimensionMismatch process!(output, prepared, Float32[0])
    @test output == fill(71.0f0, 2)
    @test_throws DimensionMismatch process!(zeros(Float32, 1), prepared, zeros(Float32, 2))
    @test output == fill(71.0f0, 2)

    for invalid in (Float32[NaN, 0], Float32[Inf, 0], Float32[-Inf, 0])
        fill!(output, 71.0f0)
        @test process!(output, prepared, invalid) ===
              PdmCommandPowerLimitNonFiniteInput
        @test output == fill(71.0f0, 2)
    end

    invalid = Float32[3, NaN]
    @test process!(invalid, prepared, invalid) === PdmCommandPowerLimitNonFiniteInput
    @test isequal(invalid, Float32[3, NaN])

    storage = Float32[3, 4, 71]
    requested = @view storage[1:2]
    overlapping = @view storage[2:3]
    @test_throws ArgumentError process!(overlapping, prepared, requested)
    @test storage == Float32[3, 4, 71]

    zero_based = ZeroBasedPowerVector(Float32[3, 4])
    @test_throws ArgumentError process!(output, prepared, zero_based)
    @test_throws ArgumentError process!(zero_based, prepared, Float32[3, 4])
end

@testset "PDM command-power processing supports strided and in-place storage" begin
    prepared = power_limit_prepared(3, 4.0f0)
    input_storage = fill(83.0f0, 6)
    output_storage = fill(71.0f0, 6)
    requested = @view input_storage[1:2:5]
    power_limited = @view output_storage[2:2:6]
    requested .= Float32[3, 4, 0]

    @test process!(power_limited, prepared, requested) === nothing
    @test exact_command_power(power_limited) <= 4
    @test power_limited ≈ Float32[1.2, 1.6, 0] rtol=2eps(Float32)
    @test output_storage[1:2:5] == fill(71.0f0, 3)

    command = Float32[3, 4, 0]
    @test process!(command, prepared, command) === nothing
    @test command == power_limited

    view_storage = Float32[83, 3, 83, 4, 83, 0]
    command_view = @view view_storage[2:2:6]
    @test process!(command_view, prepared, command_view) === nothing
    @test command_view == power_limited
end

@testset "PDM command-power warmed paths allocate no Julia heap memory" begin
    prepared = power_limit_prepared(3, 4.0f0)
    output = zeros(Float32, 3)
    valid = Float32[3, 4, 0]
    invalid = Float32[NaN, 0, 0]

    process!(output, prepared, valid)
    @test power_limit_checked_allocation_bytes(output, prepared, valid) == 0
    process!(output, prepared, invalid)
    @test power_limit_checked_allocation_bytes(output, prepared, invalid) == 0

    @test @inferred PdmCommandPowerLimitFailure FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        valid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test power_limit_admitted_allocation_bytes(output, prepared, valid) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        invalid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test power_limit_admitted_allocation_bytes(output, prepared, invalid) == 0

    boundary = Float32[nextfloat(0.0f0), 2.0, -0.0]
    process!(output, prepared, boundary)
    @test power_limit_checked_allocation_bytes(output, prepared, boundary) == 0

    workspace = prepared.workspace
    @test power_limit_commit_allocation_bytes(workspace) == 0
    @test power_limit_discard_allocation_bytes(workspace) == 0
    @test power_limit_reset_allocation_bytes(workspace) == 0
end


@testset "command power with a subnormal projection gain" begin
    power = nextfloat(0f0)
    power_limiter = prepare_algorithm(PdmCommandPowerLimitF32,
        PdmCommandPowerLimitF32Configuration(actuator_count=1, max_command_power=power))
    power_output = zeros(Float32,1)
    @test process!(power_output, power_limiter, Float32[1f22]) === nothing
    @test 0 < Float64(only(power_output))^2 <= Float64(power)
    @test only(power_output) ≈ sqrt(power)
end

end
