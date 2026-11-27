module MlLlowfsTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!
import FilterGraphAlgorithms.SimpleChains as SC

const MODEL = MlLlowfsModel(SC.SimpleChain((2, 3, 1), SC.Flatten(3), SC.TurboDense(identity, 2)))
const MODES_SCHEMA = "test/llowfs-modes-f32/1"
const PARAMETERS_SCHEMA = "test/llowfs-parameters-f32/1"
const PARAMETER_PORT = Symbol("model-parameters")

function prepared_llowfs(model=MODEL)
    return prepare_algorithm(MlLlowfsF32, MlLlowfsF32Configuration(;
        model, modes_schema=MODES_SCHEMA, parameters_schema=PARAMETERS_SCHEMA))
end

struct NoStridesArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.IndexStyle(::Type{<:NoStridesArray}) = IndexCartesian()
Base.size(array::NoStridesArray) = size(array.parent)
Base.getindex(array::NoStridesArray, indices::Vararg{Int}) = getindex(array.parent, indices...)

function checked_allocation_bytes(modes, prepared, region)
    process!(modes, prepared, region)
    return @allocated process!(modes, prepared, region)
end

@testset "ML LLOWFS derives the port formats from the model" begin
    @test MlLlowfsF32 in algorithms()
    @test algorithm_label(MlLlowfsF32) == "ml-llowfs-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(MlLlowfsF32) === MlLlowfsFailure
    for (rows, columns, outputs) in ((2, 3, 2), (3, 4, 1))
        model = MlLlowfsModel(SC.SimpleChain((rows, columns, 1), SC.Flatten(3), SC.TurboDense(identity, outputs)))
        prepared = prepared_llowfs(model)
        formats = algorithm_formats(prepared)
        @test formats[1].shape == (1, rows, columns)
        @test formats[1].schema == REGION_PIXELS_V1
        @test formats[2].shape == (outputs,)
        @test formats[2].schema == MODES_SCHEMA
        @test formats[3].shape == (outputs * (rows * columns + 1),)
        @test formats[3].schema == PARAMETERS_SCHEMA
        @test model.parameter_count == SC.numparam(model.chain)
        @test algorithm_ports(MlLlowfsF32)[2].metadata_source == Symbol("region-pixels")
    end
end

@testset "Model preparation rejects unsupported or invalid geometry" begin
    dense = SC.TurboDense(identity, 2)
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((1, 1, 1), SC.Flatten(3), dense))
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((2, 3, 2), SC.Flatten(3), dense))
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((2, 3, 1), dense))
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((2, 3, 1), SC.Conv(tanh, (3, 2), 2),
        SC.Flatten(3), dense))
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((2, 3, 1), SC.MaxPool((0, 1)),
        SC.Flatten(3), dense))
    @test_throws ArgumentError MlLlowfsModel(SC.SimpleChain((2, 3, 1), SC.Flatten(3),
        SC.TurboDense(abs, 2)))
    @test_throws OverflowError MlLlowfsModel(SC.SimpleChain((typemax(Int), 2, 1), SC.Flatten(3), dense))
end

@testset "Synthetic inference uses the training normalization and adopts owned parameters" begin
    prepared = prepared_llowfs()
    region = reshape(Float32[1, 2, 4, 8, 16, 32], 1, 2, 3)
    modes = fill(42f0, 2)
    @test process!(modes, prepared, region) === MlLlowfsParametersMissing
    @test modes == fill(42f0, 2)

    values = Float64.(vec(region))
    centered = values .- sum(values) / length(values)
    expected_input = centered ./ sqrt(sum(abs2, centered) / (length(values) - 1))
    normalized = zeros(Float32, 2, 3, 1)
    @test normalize_llowfs!(normalized, region) === nothing
    @test vec(normalized) ≈ expected_input atol=2f-7
    @test_throws DimensionMismatch normalize_llowfs!(zeros(Float32, 3, 2, 1), region)

    # Two independent modes and biases in SimpleChains column-major parameter order.
    weights = Float32[1 0 0 0 0 0; 0 0 0 0 0 -2]
    biases = Float32[0.125, -0.5]
    params = vcat(vec(weights), biases)
    @test replace_parameter!(prepared, PARAMETER_PORT, params) === prepared
    fill!(params, 0f0)
    expected = weights * expected_input + biases
    @test process!(modes, prepared, region) === nothing
    @test modes ≈ expected atol=6f-7
    @test checked_allocation_bytes(modes, prepared, region) == 0
    @test process!(modes, prepared, NoStridesArray(region)) === nothing
    @test modes ≈ expected atol=6f-7
    @test_throws DimensionMismatch replace_parameter!(prepared, PARAMETER_PORT, zeros(Float32, 13))
    @test_throws ArgumentError replace_parameter!(prepared, PARAMETER_PORT, fill(NaN32, 14))
    @test process!(modes, prepared, region) === nothing
    @test modes ≈ expected atol=6f-7
    reset!(prepared)
    @test process!(modes, prepared, region) === nothing
    @test modes ≈ expected atol=6f-7
end

@testset "Rejected frames preserve caller output" begin
    prepared = prepared_llowfs()
    params = zeros(Float32, MODEL.parameter_count)
    params[1] = 1f0
    replace_parameter!(prepared, PARAMETER_PORT, params)
    region = reshape(Float32[1, 2, 4, 8, 16, 32], 1, 2, 3)
    modes = fill(42f0, 2)
    for (value, failure) in ((NaN32, MlLlowfsNonFiniteInput), (0f0, MlLlowfsInvalidFlux),
                             (1f0, MlLlowfsInvalidStandardDeviation))
        @test process!(modes, prepared, fill(value, size(region))) === failure
        @test modes == fill(42f0, 2)
    end
    cancelling = reshape(Float32[1f8, 1, -1f8, 0, 0, 0], size(region))
    @test process!(modes, prepared, cancelling) === nothing
    @test modes[1] > 0f0
    @test_throws ArgumentError process!(view(region, 1:2), prepared, region)
    @test_throws DimensionMismatch process!(zeros(Float32, 3), prepared, region)
    @test_throws DimensionMismatch process!(modes, prepared, zeros(Float32, 1, 3, 2))

    params[11] = floatmax(Float32)
    params[13] = floatmax(Float32)
    replace_parameter!(prepared, PARAMETER_PORT, params)
    fill!(modes, 42f0)
    @test process!(modes, prepared, region) === MlLlowfsNonFiniteOutput
    @test modes == fill(42f0, 2)
end

end
