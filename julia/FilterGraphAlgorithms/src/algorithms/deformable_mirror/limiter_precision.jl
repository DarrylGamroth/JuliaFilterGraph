# With subnormal flushing enabled, a native conversion can discard a stored
# Float32 value on input or output. The limit applies to the stored bits.
@inline _dm_widen(value::Float32, ::Val{false}) = Float64(value)
@inline _dm_narrow(value::Float64, ::Val{false}) = Float32(value)

@inline function _dm_widen(value::Float32, ::Val{true})
    bits = reinterpret(UInt32, value)
    magnitude = bits & 0x7fff_ffff
    magnitude >= 0x0080_0000 && return Float64(value)
    wide = Float64(magnitude) * 0x1p-149
    return signbit(value) ? -wide : wide
end

@inline function _dm_narrow(value::Float64, ::Val{true})
    abs(value) >= 0x1p-126 && return Float32(value)
    magnitude = round(UInt32, abs(value) * 0x1p149, RoundNearest)
    bits = signbit(value) ? magnitude | 0x8000_0000 : magnitude
    return reinterpret(Float32, bits)
end
