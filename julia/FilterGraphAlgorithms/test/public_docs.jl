module PublicDocsTests

using FilterGraphAlgorithms
using Test

@testset "Every public algorithm binding has documentation" begin
    # Keep public type names available when numerical tests use concrete types.
    @test all(
        name -> isdefined(FilterGraphAlgorithms, name) &&
                Base.ispublic(FilterGraphAlgorithms, name),
        (
            :CalibrationProbePolarity,
            :CaptureSupport,
            :CompleteTransition,
            :ProgressSupport,
            :PyramidReadoutReconstructionPlan,
            :PyramidReadoutReconstructionWorkspace,
        ),
    )
    public_names = filter(
        name -> Base.ispublic(FilterGraphAlgorithms, name),
        names(FilterGraphAlgorithms; all=true, imported=false),
    )
    missing = filter(
        name -> !Base.Docs.hasdoc(FilterGraphAlgorithms, name),
        public_names,
    )
    @test isempty(missing)

    for (doc, phrase) in (
        (@doc(FilterGraphAlgorithms.LeakyIntegratorF32), "pole * y[k-1] + gain * input[k]"),
        (@doc(FilterGraphAlgorithms.FrameDelayF32Configuration), "successful local adoptions"),
        (@doc(FilterGraphAlgorithms.CompleteTransition), "guarded local state commitment"),
        (@doc(FilterGraphAlgorithms.ShackHartmannMeasurementBlockMetadataMissing), "sample metadata"),
        (@doc(FilterGraphAlgorithms.PyramidReadoutMeasurementBlockWorkspace), "concurrent process calls"),
    )
        rendered = sprint(show, MIME("text/plain"), doc)
        @test occursin(phrase, join(split(rendered), " "))
    end
end

end
