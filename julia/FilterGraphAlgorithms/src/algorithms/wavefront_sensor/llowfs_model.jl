"""
    MlLlowfsModel(chain::SimpleChains.SimpleChain)

An inference chain for one calibrated LLOWFS region. The chain declares a fixed
`(rows, columns, 1)` input shape and produces a vector of modal coefficients.
Supported layers are two-dimensional convolution and max pooling, flattening
of all three image dimensions, and dense layers with biases. Supported
activations are `identity`, `tanh`, and `SimpleChains.relu`.

Use `model.chain` with SimpleChains to train the same topology outside the
Graph. `model.image_size`, `model.mode_count`, and `model.parameter_count`
describe the prepared port extents. The model contains no parameter values.
"""
struct MlLlowfsModel{C<:SimpleChains.SimpleChain}
    chain::C
    image_size::NTuple{2,Int}
    mode_count::Int
    parameter_count::Int

    function MlLlowfsModel(chain::SimpleChains.SimpleChain)
        input_size = chain.inputdim
        input_size isa Tuple && length(input_size) == 3 && input_size[3] == 1 ||
            throw(ArgumentError("ML LLOWFS requires a fixed (rows, columns, 1) input shape"))
        shape = map(Int, input_size)
        Base.checked_mul(shape[1], shape[2]) > 1 && all(>(0), shape) ||
            throw(ArgumentError("ML LLOWFS requires at least two input pixels"))
        image_size = (shape[1], shape[2])
        count = 0
        for layer in chain.layers
            shape, parameters = _llowfs_layer_extents(layer, shape)
            _checked_llowfs_bytes(shape)
            count = Base.checked_add(count, parameters)
        end
        length(shape) == 1 && count > 0 ||
            throw(ArgumentError("ML LLOWFS requires a parameterized model with a vector output"))
        _checked_llowfs_bytes((image_size..., 1))
        _checked_llowfs_bytes((count,))
        # Use static input dimensions for the repeated CPU inference call.
        concrete = SimpleChains.SimpleChain(map(SimpleChains.static, (image_size..., 1)), chain.layers)
        return new{typeof(concrete)}(concrete, image_size, only(shape), count)
    end
end

_checked_llowfs_bytes(shape) = foldl(Base.checked_mul, shape; init=sizeof(Float32))

function _check_llowfs_activation(activation)
    activation in (identity, tanh, SimpleChains.relu) ||
        throw(ArgumentError("unsupported ML LLOWFS layer activation"))
end

function _llowfs_layer_extents(layer::SimpleChains.Conv, shape::NTuple{3,Int})
    window = map(Int, layer.dim)
    channels = Int(layer.outputdim)
    length(window) == 2 && all(>(0), window) && window[1] <= shape[1] &&
        window[2] <= shape[2] && channels > 0 ||
        throw(ArgumentError("invalid ML LLOWFS convolution extents"))
    _check_llowfs_activation(layer.f)
    weights = foldl(Base.checked_mul, (window..., shape[3], channels))
    return ((shape[1] - window[1] + 1, shape[2] - window[2] + 1, channels),
        Base.checked_add(weights, channels))
end

function _llowfs_layer_extents(::SimpleChains.MaxPool{W}, shape::NTuple{3,Int}) where {W}
    length(W) == 2 && all(>(0), W) && W[1] <= shape[1] && W[2] <= shape[2] ||
        throw(ArgumentError("invalid ML LLOWFS pooling window"))
    return ((shape[1] ÷ W[1], shape[2] ÷ W[2], shape[3]), 0)
end

_llowfs_layer_extents(::SimpleChains.Flatten{3}, shape::NTuple{3,Int}) =
    ((foldl(Base.checked_mul, shape),), 0)

function _llowfs_layer_extents(layer::SimpleChains.TurboDense{true}, shape::Tuple{Int})
    outputs = Int(layer.outputdim)
    outputs > 0 || throw(ArgumentError("ML LLOWFS dense output count must be positive"))
    _check_llowfs_activation(layer.f)
    return ((outputs,), Base.checked_mul(Base.checked_add(only(shape), 1), outputs))
end

function _llowfs_layer_extents(layer, shape)
    throw(ArgumentError("$(typeof(layer)) is not supported for ML LLOWFS input shape $shape"))
end

function _llowfs_fields(config::AbstractDict, fields)
    length(config) == length(fields) && all(name -> haskey(config, name), fields) ||
        throw(ArgumentError("ML LLOWFS configuration requires exactly $(join(fields, ", "))"))
end

function _llowfs_size(value)
    value isa Union{Tuple,AbstractVector} && length(value) == 2 ||
        throw(ArgumentError("ML LLOWFS image and window sizes require rows and columns"))
    return (Int(value[1]), Int(value[2]))
end

function _llowfs_activation(value)
    value == "identity" && return identity
    value == "tanh" && return tanh
    value == "relu" && return SimpleChains.relu
    throw(ArgumentError("unknown ML LLOWFS layer activation: $value"))
end

function _llowfs_layer(config::AbstractDict)
    kind = get(config, "type", nothing)
    if kind == "flatten"
        _llowfs_fields(config, ("type",))
        return SimpleChains.Flatten(3)
    elseif kind == "dense"
        _llowfs_fields(config, ("type", "activation", "outputs"))
        return SimpleChains.TurboDense(_llowfs_activation(config["activation"]), Int(config["outputs"]))
    elseif kind == "convolution"
        _llowfs_fields(config, ("type", "activation", "window", "channels"))
        return SimpleChains.Conv(_llowfs_activation(config["activation"]),
            _llowfs_size(config["window"]), Int(config["channels"]))
    elseif kind == "max-pool"
        _llowfs_fields(config, ("type", "window"))
        return SimpleChains.MaxPool(_llowfs_size(config["window"]))
    end
    throw(ArgumentError("unknown ML LLOWFS layer type: $kind"))
end

function Base.convert(::Type{MlLlowfsModel}, config::AbstractDict)
    _llowfs_fields(config, ("image_size", "layers"))
    rows, columns = _llowfs_size(config["image_size"])
    config["layers"] isa AbstractVector ||
        throw(ArgumentError("ML LLOWFS layers must be an ordered array"))
    layers = Tuple(map(_llowfs_layer, config["layers"]))
    return MlLlowfsModel(SimpleChains.SimpleChain((rows, columns, 1), layers))
end
