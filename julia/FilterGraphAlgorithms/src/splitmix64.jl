const _SPLITMIX64_GAMMA = UInt64(0x9e3779b97f4a7c15)
const _SPLITMIX64_MIX_1 = UInt64(0xbf58476d1ce4e5b9)
const _SPLITMIX64_MIX_2 = UInt64(0x94d049bb133111eb)
const _SPLITMIX64_UNIT_DENOMINATOR = UInt32(1) << 24

"""
    splitmix64_at(state::UInt64, draw_index::UInt64) -> UInt64

Return indexed SplitMix64 draw `draw_index` from `state`. Index zero is the
first draw. The calculation uses only fixed-width integer operations so the
same function can be called from CPU and GPU kernels.
"""
@inline function splitmix64_at(state::UInt64, draw_index::UInt64)
    offset = _SPLITMIX64_GAMMA * (draw_index + one(UInt64))
    value = state + offset
    value = (value ⊻ (value >> 30)) * _SPLITMIX64_MIX_1
    value = (value ⊻ (value >> 27)) * _SPLITMIX64_MIX_2
    return value ⊻ (value >> 31)
end

"""
    splitmix64_advance(state::UInt64, draw_count::UInt64) -> UInt64

Return the SplitMix64 state following `draw_count` indexed draws. A stateful
Algorithm stages this value in its Workspace and publishes it only through
that Algorithm's commit.
"""
@inline splitmix64_advance(state::UInt64, draw_count::UInt64) =
    state + _SPLITMIX64_GAMMA * draw_count

"""
    splitmix64_unit_at(T, state::UInt64, draw_index::UInt64) -> T

Return an indexed value in `[0, 1)` for `T === Float32` or `T === Float64`.
The top 24 random bits convert exactly to both admitted element types.
"""
@inline function splitmix64_unit_at(
    ::Type{T},
    state::UInt64,
    draw_index::UInt64,
) where {T<:Union{Float32,Float64}}
    numerator = T(splitmix64_at(state, draw_index) >> 40)
    return numerator / T(_SPLITMIX64_UNIT_DENOMINATOR)
end
