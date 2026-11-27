module ShackHartmannSummaryTests

using FilterGraphAlgorithms
using Test
import FilterGraphAlgorithms: process!

prepare_summary(n) = prepare_algorithm(
    ShackHartmannSummaryF32,
    ShackHartmannSummaryF32Configuration(subaperture_count=n),
)

function summary_allocation_bytes(summary, prepared, slopes, flux, validity)
    return @allocated process!(summary, prepared, slopes, flux, validity)
end

function summary_oracle(slopes, flux, validity)
    selected = findall(validity)
    n = length(selected)
    x = BigFloat.(slopes[selected, 1])
    y = BigFloat.(slopes[selected, 2])
    f = BigFloat.(flux[selected])
    rx = sqrt(sum(abs2, x) / n)
    ry = sqrt(sum(abs2, y) / n)
    return Float32[length(validity), n, sum(f) / n, rx, ry,
        sqrt(rx^2 + ry^2), sum(x) / n, sum(y) / n]
end

@testset "summary declaration and preparation" begin
    @test ShackHartmannSummaryF32 in algorithms()
    @test algorithm_label(ShackHartmannSummaryF32) == "shack-hartmann-summary-f32"
    @test capture_support(ShackHartmannSummaryF32, nothing) isa NoCapture
    @test FilterGraphAlgorithms.algorithm_failure_type(ShackHartmannSummaryF32) === Union{}
    configuration = only(algorithm_configuration(ShackHartmannSummaryF32))
    @test configuration.name === :subaperture_count
    @test configuration.update isa GraphRebuildConfiguration
    @test_throws ArgumentError prepare_summary(0)
    @test_throws ArgumentError prepare_summary(-1)
    @test_throws ArgumentError prepare_summary(2^24 + 1)
    @test algorithm_format(prepare_summary(2^24), :validity).shape == (2^24,)
    prepared = prepare_summary(3)
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((3, 2), (3,), (3,), (8,))
    @test getproperty.(formats, :element_type) == (Float32, Float32, Bool, Float32)
    @test getproperty.(formats, :schema) == (SHACK_HARTMANN_SLOPES_V1,
        SHACK_HARTMANN_FLUX_V1, SHACK_HARTMANN_VALIDITY_V1, SHACK_HARTMANN_SUMMARY_V1)
    @test last(algorithm_ports(ShackHartmannSummaryF32)).metadata_source === :slopes
end

@testset "independent numerical oracle and immutable inputs" begin
    for (slopes, flux, validity) in (
        (Float32[3 4; -3 -4; 90 80], Float32[10, 30, 999], Bool[1, 1, 0]),
        (Float32[-2 7], Float32[5], Bool[1]),
        (Float32[16777216 0; 1 -2; -16777216 3], Float32[16777216, 1, -16777216], trues(3)),
        (Float32[1e30 -2e30; -3e30 4e30], Float32[1e30, 2e30], trues(2)),
    )
        prepared = prepare_summary(length(validity))
        summary = fill(-1.0f0, 8)
        before = (copy(slopes), copy(flux), copy(validity))
        expected = summary_oracle(slopes, flux, validity)
        @test process!(summary, prepared, slopes, flux, validity) === nothing
        @test all(isapprox.(summary, expected; rtol=2eps(Float32), atol=0))
        @test isequal((slopes, flux, validity), before)
        @test summary_allocation_bytes(summary, prepared, slopes, flux, validity) == 0
    end
    # Uncentered RMS stays nonzero for constant, nonzero slopes.
    summary = zeros(Float32, 8)
    @test process!(summary, prepare_summary(2), Float32[3 4; 3 4], Float32[1, 1], trues(2)) === nothing
    @test summary == Float32[2, 2, 1, 3, 4, 5, 3, 4]
end

@testset "invalid diagnostics complete successfully and replay without stale fields" begin
    prepared = prepare_summary(3)
    summary = zeros(Float32, 8)
    slopes = Float32[3 4; NaN Inf; -3 -4]
    flux = Float32[10, NaN, 30]
    validity = Bool[1, 0, 1]
    @test process!(summary, prepared, slopes, flux, validity) === nothing
    @test summary == Float32[3, 2, 20, 3, 4, 5, 0, 0]
    fill!(validity, false)
    @test process!(summary, prepared, slopes, flux, validity) === nothing
    @test summary[1:2] == Float32[3, 0]
    @test all(isnan, summary[3:8])
    @test summary_allocation_bytes(summary, prepared, slopes, flux, validity) == 0
    fill!(validity, true)
    for nonfinite in (NaN32, Inf32, -Inf32), field in 1:3
        slopes .= Float32[3 4; -3 -4; 0 0]
        flux .= Float32[10, 30, 20]
        field == 1 ? (flux[1] = nonfinite) : (slopes[1, field - 1] = nonfinite)
        before = (copy(slopes), copy(flux), copy(validity))
        @test process!(summary, prepared, slopes, flux, validity) === nothing
        @test summary[1:2] == Float32[3, 3]
        @test all(isnan, summary[3:8])
        @test isequal((slopes, flux, validity), before)
        @test summary_allocation_bytes(summary, prepared, slopes, flux, validity) == 0
    end
    slopes .= Float32[3 4; -3 -4; 0 0]
    @test process!(summary, prepared, slopes, Float32[10, 30, 20], validity) === nothing
    @test all(isapprox.(summary, summary_oracle(slopes, Float32[10, 30, 20], validity)))
end

@testset "checked shapes, aliases, and strided storage" begin
    prepared = prepare_summary(3)
    summary = zeros(Float32, 8)
    slopes, flux, validity = zeros(Float32, 3, 2), ones(Float32, 3), trues(3)
    @test_throws DimensionMismatch process!(summary, prepared, zeros(Float32, 2, 3), flux, validity)
    @test_throws DimensionMismatch process!(summary, prepared, slopes, ones(Float32, 2), validity)
    @test_throws DimensionMismatch process!(summary, prepared, slopes, flux, trues(2))
    @test_throws DimensionMismatch process!(zeros(Float32, 7), prepared, slopes, flux, validity)
    @test_throws ArgumentError process!(summary, prepared, Float64.(slopes), flux, validity)
    shared = zeros(Float32, 16)
    @test_throws ArgumentError process!(view(shared, 1:8), prepare_summary(8),
        reshape(shared, 8, 2), ones(Float32, 8), trues(8))
    @test_throws ArgumentError process!(view(shared, 1:8), prepare_summary(8),
        zeros(Float32, 8, 2), view(shared, 1:8), trues(8))
    slopes_view = view(zeros(Float32, 6, 4), 1:2:6, 1:2:4)
    slopes_view .= Float32[3 4; -3 -4; 9 10]
    flux_view = view(Float32[10, 0, 30, 0, 99, 0], 1:2:6)
    validity_view = view(Bool[1, 0, 1, 0, 0, 0], 1:2:6)
    summary_view = view(zeros(Float32, 16), 1:2:16)
    @test process!(summary_view, prepared, slopes_view, flux_view, validity_view) === nothing
    @test summary_view == Float32[3, 2, 20, 3, 4, 5, 0, 0]
    @test summary_allocation_bytes(summary_view, prepared, slopes_view, flux_view, validity_view) == 0
end

end
