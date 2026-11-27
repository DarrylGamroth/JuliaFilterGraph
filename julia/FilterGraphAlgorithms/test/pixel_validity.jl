module PixelValidityTests
using FilterGraphAlgorithms, Test
import FilterGraphAlgorithms: process!
@testset "finite pixel validity preserves scientific metadata" begin
    prepared = prepare_algorithm(PixelValidityF32,
        PixelValidityF32Configuration(extent=6, input_schema="test.relative-intensity-f32/1"))
    pixels = Float32[0, -1, 2, NaN, Inf, -Inf]
    validity = trues(6)
    @test process!(validity, prepared, pixels) === nothing
    @test validity == Bool[true, true, true, false, false, false]
    @test algorithm_formats(prepared)[1].schema == "test.relative-intensity-f32/1"
    @test algorithm_formats(prepared)[2].schema == CALIBRATED_PIXEL_VALIDITY_V1
    @test last(algorithm_ports(PixelValidityF32)).metadata_source === :input
    overlapping = view(reinterpret(Bool, pixels), 1:6)
    retained = copy(pixels)
    @test_throws ArgumentError process!(overlapping, prepared, pixels)
    @test isequal(pixels, retained)
end

end
