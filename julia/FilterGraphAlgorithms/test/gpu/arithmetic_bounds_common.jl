function run_bound_math_tests(extension)
    @testset "cold Float32 calibration lattice covers every UInt16 value" begin
        for (flat, background) in ((1.0f0, 0.0625f0), (-0.75f0, 123.125f0),
                (0.0f0, -2.0f0), (nextfloat(0.0f0), 0.0f0),
                (0.125f0, -nextfloat(0.0f0)), (floatmin(Float32), 0.25f0))
            bound = extension._pixel_bounds([flat], [background])
            @test bound !== nothing
            values = (FilterGraphAlgorithms._calibrated_pixel(raw, flat, background)
                for raw in typemin(UInt16):typemax(UInt16))
            @test all(values) do value
                abs(Float64(value)) <= bound.magnitude &&
                    (iszero(value) || (!isnothing(bound.quantum_exponent) &&
                        isinteger(ldexp(Float64(value), -bound.quantum_exponent))))
            end
        end
        @test isnothing(extension._pixel_bounds([floatmax(Float32)], [0.0f0]))
        regular = extension._pixel_bounds([1.0f0], [0.0625f0])
        @test extension._pupil_bound(regular, 0.125f0, 0.0f0).magnitude < floatmax(Float32)
        @test isnothing(extension._pupil_bound(regular, 0.125f0, Inf32))
        @test isnothing(extension._pupil_bound(regular, 0.125f0, floatmax(Float32)))
        @test isnothing(extension._pupil_bound(extension._PixelBounds(1.0, 127), 4.0f0, 0.0f0))
        tiny = extension._pixel_bounds([nextfloat(0.0f0)], [0.0f0])
        @test isnothing(extension._pupil_bound(tiny, 0.125f0, 0.0f0))
        zero = extension._pixel_bounds([0.0f0], [0.0f0])
        @test extension._pupil_bound(zero, 0.125f0, 2.0f0).magnitude == 0
        input = extension._FiniteBound(1024.0)
        @test extension._projection_bound(0.25, 3600, input) !== nothing
        @test isnothing(extension._projection_bound(Float64(floatmax(Float32)), 3600, input))
        @test isnothing(extension._projection_bound(0.25, (1 << 20) + 1, input))
        mode = get_zero_subnormals()
        try
            set_zero_subnormals(true)
            @test isnothing(extension._pixel_bounds([1.0f0], [0.0f0]))
            @test isnothing(extension._pupil_bound(regular, 0.125f0, 0.0f0))
            @test isnothing(extension._projection_bound(0.25, 8, input))
            @test extension._coefficient_bound(ones(Float32, 2, 2)) == Inf
        finally
            set_zero_subnormals(mode)
        end
    end
end
