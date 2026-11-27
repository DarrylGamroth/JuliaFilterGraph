using FilterGraphAlgorithms
using LinearAlgebra: Symmetric
using Random: MersenneTwister, rand
import FilterGraphAlgorithms: process!
import SimpleChains as SC
using TOML

const MODES_SCHEMA = "examples/synthetic-llowfs-modes-f32/1"
const PARAMETERS_SCHEMA = "examples/synthetic-llowfs-parameters-f32/1"

function dataset(seed, count)
    rng = MersenneTwister(seed)
    regions = 1f0 .+ rand(rng, Float32, 1, 2, 3, count)
    input = Array{Float32}(undef, 2, 3, 1, count)
    modes = Matrix{Float32}(undef, 2, count)
    for i in 1:count
        frame = view(input, :, :, :, i)
        normalize_llowfs!(frame, view(regions, :, :, :, i)) === nothing ||
            error("synthetic region failed LLOWFS preprocessing")
        modes[1, i] = 0.6f0 * frame[1, 1, 1] - 0.3f0 * frame[2, 3, 1] + 0.1f0
        modes[2, i] = -0.4f0 * frame[2, 1, 1] + 0.7f0 * frame[1, 3, 1]
    end
    return regions, input, modes
end

function train_and_check()
    model = MlLlowfsModel(SC.SimpleChain((2, 3, 1), SC.Flatten(3), SC.TurboDense(identity, 2)))
    _, input, labels = dataset(1, 128)
    regions, validation, expected = dataset(2, 48)

    parameters = zeros(Float32, model.parameter_count)
    mse = p -> sum(abs2, model.chain(validation, p) .- expected) / length(expected)
    initial_mse = mse(parameters)
    features = reshape(Float64.(input), :, size(input, 4))
    design = vcat(features, ones(1, size(features, 2)))
    gram = design * transpose(design)
    # Regularize the pixel coefficients and leave the bias unpenalized.
    for i in axes(features, 1)
        gram[i, i] += 1e-4
    end
    fitted = (Float64.(labels) * transpose(design)) / Symmetric(gram)
    weights, bias = last(SC.params(model.chain, parameters))
    weights .= view(fitted, :, 1:size(features, 1))
    bias .= view(fitted, :, size(fitted, 2))
    trained_mse = mse(parameters)
    trained_mse < initial_mse / 100 || error("validation MSE reached $trained_mse from $initial_mse")

    prepared = prepare_algorithm(MlLlowfsF32, MlLlowfsF32Configuration(;
        model, modes_schema=MODES_SCHEMA, parameters_schema=PARAMETERS_SCHEMA))
    replace_parameter!(prepared, Symbol("model-parameters"), collect(parameters))
    output = zeros(Float32, model.mode_count)
    for i in axes(regions, 4)
        process!(output, prepared, view(regions, :, :, :, i)) === nothing ||
            error("prepared inference rejected a validation region")
        reference = model.chain(view(validation, :, :, :, i), parameters)
        isapprox(output, reference; atol=1f-6, rtol=0) ||
            error("prepared inference disagrees with the trained chain")
    end
    println("Held-out modal MSE before training = ", initial_mse)
    println("Held-out modal MSE after training = ", trained_mse)
    return collect(parameters)
end

length(ARGS) <= 1 || error("usage: train.jl [parameters.toml]")
parameters = train_and_check()
if !isempty(ARGS)
    open(only(ARGS), "w") do io
        TOML.print(io, Dict("parameters_schema" => PARAMETERS_SCHEMA, "parameters" => parameters))
    end
end
