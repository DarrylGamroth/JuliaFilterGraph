const _CHANNELS_PROPERTY = "pipewireao.ndarray.channels"
const _MAX_CHANNELS_BYTES = 16384

function _channel_text(value, kind)
    value isa AbstractString && isvalid(value) && !isempty(strip(value)) &&
        ncodeunits(value) <= 255 && !any(iscntrl, value) ||
        throw(ArgumentError("$kind must be valid UTF-8, nonempty and without controls, at most 255 bytes"))
    return String(value)
end

function _channel_name(value, kind)
    value isa Union{AbstractString,Symbol} || throw(ArgumentError("$kind must be text or a Symbol"))
    return _channel_text(string(value), kind)
end

function _channel_unit_declarations(channel_units)
    channel_units isa Union{AbstractDict,NamedTuple} ||
        throw(ArgumentError("channel_units must map public output Ports to channel units"))
    length(channel_units) <= 64 || throw(ArgumentError("channel_units exceeds 64 outputs"))
    result = Dict{String,Dict{String,String}}()
    for (port, channels) in pairs(channel_units)
        name = _channel_name(port, "channel_units Port")
        haskey(result, name) && throw(ArgumentError("duplicate channel_units Port '$name'"))
        channels isa Union{AbstractDict,NamedTuple} && length(channels) <= 64 ||
            throw(ArgumentError("channel_units Port '$name' must map at most 64 channels to units"))
        units = Dict{String,String}()
        for (channel, unit) in pairs(channels)
            channel_name = _channel_name(channel, "channel_units channel")
            haskey(units, channel_name) && throw(ArgumentError("duplicate channel '$channel_name'"))
            units[channel_name] = _channel_text(unit, "channel '$channel_name' units")
        end
        result[name] = units
    end
    return result
end

# All strings are valid UTF-8, bounded and control-free before encoding. JSON needs
# only quote and backslash escaping; numerical values are native integers.
_channel_json_string(io, value) =
    print(io, '"', replace(value, "\\" => "\\\\", "\"" => "\\\""), '"')

function _source_channels(ports, channel_units=NamedTuple())
    annotations = _channel_unit_declarations(channel_units)
    outputs = filter(port -> port.direction == PipeWireAO.DIRECTION_OUTPUT, ports)
    for (name, channels) in annotations
        port = findfirst(port -> port.name == name, outputs)
        isnothing(port) && throw(ArgumentError("channel_units has unknown public output Port '$name'"))
        schema = outputs[port].schema
        declarations = isnothing(schema) ? () : FilterGraphAlgorithms.schema_channels(schema)
        for channel in keys(channels)
            any(declaration -> declaration.name == channel, declarations) ||
                throw(ArgumentError("output '$name' has no declared channel '$channel'"))
        end
    end
    io = IOBuffer()
    print(io, "{\"version\":1,\"outputs\":[")
    count = 0
    names = Set{String}()
    for port in outputs
        isnothing(port.schema) && continue
        channels = FilterGraphAlgorithms.schema_channels(port.schema)
        isempty(channels) && continue
        port.format.shape == FilterGraphAlgorithms.schema_channel_shape(port.schema) ||
            throw(ArgumentError("output '$(port.name)' shape differs from its channel schema"))
        port.format.element_type == _element_type(FilterGraphAlgorithms.schema_element_type(port.schema)) ||
            throw(ArgumentError("output '$(port.name)' element type differs from its channel schema"))
        _channel_text(port.name, "channel output Port")
        port.name ∉ names || throw(ArgumentError("duplicate channel output Port '$(port.name)'"))
        push!(names, port.name)
        count += 1
        count <= 64 || throw(ArgumentError("source channels exceeds 64 outputs"))
        count > 1 && print(io, ',')
        print(io, "{\"port\":")
        _channel_json_string(io, port.name)
        print(io, ",\"schema\":")
        _channel_json_string(io, port.schema)
        print(io, ",\"element_type\":", UInt32(port.format.element_type),
            ",\"layout\":", UInt32(port.format.layout), ",\"shape\":[")
        join(io, port.format.shape, ',')
        print(io, "],\"channels\":[")
        units = get(annotations, port.name, Dict{String,String}())
        for (index, channel) in enumerate(channels)
            unit = get(units, channel.name, channel.units)
            !isnothing(channel.units) && unit != channel.units && throw(ArgumentError(
                "output '$(port.name)' channel '$(channel.name)' has fixed units '$(channel.units)'"))
            index > 1 && print(io, ',')
            print(io, "{\"name\":")
            _channel_json_string(io, channel.name)
            print(io, ",\"index\":", channel.index, ",\"title\":")
            _channel_json_string(io, channel.title)
            print(io, ",\"units\":")
            isnothing(unit) ? print(io, "null") : _channel_json_string(io, unit)
            print(io, '}')
            position(io) <= _MAX_CHANNELS_BYTES ||
                throw(ArgumentError("source channels exceeds 16384 UTF-8 bytes"))
        end
        print(io, "]}")
    end
    count == 0 && return nothing
    print(io, "]}")
    position(io) <= _MAX_CHANNELS_BYTES ||
        throw(ArgumentError("source channels exceeds 16384 UTF-8 bytes"))
    return PipeWireAO.Pod(String(take!(io)))
end

function _source_channel_info(channels)
    return PipeWireAO.SPA.PropInfo(
        _CHANNELS_PROPERTY, channels;
        description="Immutable named scalar channels for public ndarray outputs",
        display_name="Source ndarray channels", group="Source",
        flags=PipeWireAO.SPA.PROPERTY_READONLY, params=true,
    )
end
