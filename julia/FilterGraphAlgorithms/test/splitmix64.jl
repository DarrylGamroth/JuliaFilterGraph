using FilterGraphAlgorithms:
    splitmix64_advance,
    splitmix64_at,
    splitmix64_unit_at

@testset "SplitMix64 frozen sequence" begin
    state = UInt64(0x123456789abcdef0)
    expected = UInt64[
        0x161922c645ce50e8,
        0xad760cafa1697b60,
        0x3501ff44902ca50d,
        0x417cb9a826d831df,
        0x99af6f9b0c4476b6,
        0x5d51f5f75b762c59,
    ]

    @test [splitmix64_at(state, UInt64(index - 1)) for index in 1:6] ==
          expected
    @test splitmix64_advance(state, UInt64(6)) == UInt64(0xc78130d1967bc76e)

    for T in (Float32, Float64)
        values = [splitmix64_unit_at(T, state, UInt64(index)) for index in 0:5]
        expected_units = T.(expected .>> 40) ./ T(UInt32(1) << 24)
        @test values == expected_units
        @test all(value -> zero(T) <= value < one(T), values)
    end
end

@testset "SplitMix64 admitted scalar calls allocate nothing" begin
    state = UInt64(0x123456789abcdef0)
    splitmix64_at(state, UInt64(0))
    splitmix64_advance(state, UInt64(1))
    splitmix64_unit_at(Float32, state, UInt64(0))

    @test @allocated(splitmix64_at(state, UInt64(0))) == 0
    @test @allocated(splitmix64_advance(state, UInt64(1))) == 0
    @test @allocated(splitmix64_unit_at(Float32, state, UInt64(0))) == 0
end

@testset "Algorithm sources do not call Julia's stateful random API" begin
    source_root = dirname(pathof(parentmodule(splitmix64_at)))
    package_root = dirname(source_root)
    forbidden_call = r"\b(?:rand|rand!|randn|randn!)\s*\("
    forbidden_import = r"(?m)^\s*(?:using|import)\s+Random\b"

    for tree in (source_root, joinpath(package_root, "ext"))
        for (root, _, files) in walkdir(tree), file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            source = read(path, String)
            @test isnothing(match(forbidden_call, source))
            @test isnothing(match(forbidden_import, source))
        end
    end
end
