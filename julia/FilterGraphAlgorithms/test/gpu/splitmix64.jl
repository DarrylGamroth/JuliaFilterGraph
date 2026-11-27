import FilterGraphAlgorithms:
    splitmix64_advance,
    splitmix64_at,
    splitmix64_unit_at

@kernel function splitmix64_kernel!(draws, units, state)
    index = @index(Global, Linear)
    draw_index = UInt64(index - 1)
    @inbounds draws[index] = splitmix64_at(state, draw_index)
    @inbounds units[index] = splitmix64_unit_at(Float32, state, draw_index)
end

@testset "AMDGPU SplitMix64 matches the CPU sequence" begin
    state = UInt64(0x123456789abcdef0)
    draws = AMDGPU.zeros(UInt64, 6)
    units = AMDGPU.zeros(Float32, 6)
    kernel! = splitmix64_kernel!(KernelAbstractions.get_backend(draws), 6)
    kernel!(draws, units, state; ndrange=6)
    AMDGPU.synchronize()

    expected_draws = [splitmix64_at(state, UInt64(index)) for index in 0:5]
    expected_units = [
        splitmix64_unit_at(Float32, state, UInt64(index)) for index in 0:5
    ]
    @test Array(draws) == expected_draws
    @test Array(units) == expected_units
    @test splitmix64_advance(state, UInt64(6)) == UInt64(0xc78130d1967bc76e)
end
