module ChoppedPixelControllerTests
using FilterGraphAlgorithms, FilterGraphs, Test
import FilterGraphAlgorithms: process!, reset!

const GRAPH = """
filter.graph = {
 nodes = [
  { name=pixels type=ndarray label=pixel-selection-f32 config={
    image_size=[2 3] mask=[true true true true true true]
    input_schema=test.relative-intensity-image-f32/1
    output_schema=test.relative-intensity-pixels-f32/1 } }
  { name=validity type=ndarray label=pixel-validity-f32 config={
    extent=6 input_schema=test.relative-intensity-pixels-f32/1 } }
  { name=pair type=ndarray label=chopped-image-difference-f32 config={
    pixel_count=6 trigger_high_is_fringed=true require_acquisition_identity=true
    input_schema=test.relative-intensity-pixels-f32/1 } }
  { name=residual type=ndarray label=scc-pixel-residual-f32 config={
    pixel_count=6 mask=[true true true true true true]
    initial_reference_image=null minimum_valid_fraction=1 } }
  { name=reconstruction type=ndarray label=dense-vector-reconstructor-f32 config={
    input_count=6 output_count=2
    initial_reconstructor=[1 0 0 0 0 0  0 0 0 0 0 1]
    input_schema=pipewireao/scc-selected-pixel-residuals-f32/1
    output_schema=test.modal-residual-f32/1 reconstructor_schema=test.modal-reconstructor-f32/1 } }
  { name=correction type=ndarray label=leaky-integrator-f32 config={
    extent=2 initial_state=0 input_schema=test.modal-residual-f32/1
    output_schema=test.modal-command-f32/1 } props={gain=-0.05 pole=1} }
  { name=projection type=ndarray label=dense-vector-reconstructor-f32 config={
    input_count=2 output_count=3 initial_reconstructor=[1 0 0 1 1 1]
    input_schema=test.modal-command-f32/1 output_schema=test.physical-command-f32/1
    reconstructor_schema=test.command-projection-f32/1 } }
 ]
 links = [
  {output="pixels:selected-pixels" input="validity:input"}
  {output="pixels:selected-pixels" input="pair:image"}
  {output="validity:validity" input="pair:input-validity"}
  {output="pair:difference" input="residual:difference"}
  {output="pair:output-validity" input="residual:difference-validity"}
  {output="pair:pair-status" input="residual:pair-status"}
  {output="residual:selected-residuals" input="reconstruction:ordered-input"}
  {output="reconstruction:output" input="correction:input"}
  {output="correction:output" input="projection:ordered-input"}
 ]
 inputs=["pixels:image"]
 outputs=["projection:output" "pair:pair-status"]
}
"""

function sample(sequence, kind)
    domain = ntuple(index -> UInt8(index), 16)
    SampleMetadata(sequence; terminal=true, chopped_image_kind=kind,
        exposure_duration_nanoseconds=10_000_000,
        acquisition_identity=SampleAcquisitionIdentity(domain, 1, sequence))
end

@testset "real chopped frontend retains priming history and updates overlapping pairs" begin
    graph = mktemp() do path, io
        write(io, GRAPH)
        flush(io)
        prepare_graph(path; algorithms=FilterGraphAlgorithms.algorithms())
    end
    outputs = (; output=fill(71f0, 3), var"pair-status"=UInt8[99])
    fringed = Float32[10 20 30; 40 50 60]
    unfringed = Float32[1 2 3; 4 5 6]
    @test process!(outputs, graph, (; image=fringed), (; image=sample(1, ChoppedImageFringed))) ===
        SccPixelResidualIncompletePair
    @test outputs.output == fill(71f0, 3) && outputs.var"pair-status" == UInt8[99]
    pair = only(filter(node -> node.name === :pair, graph.nodes)).prepared.workspace
    @test chopped_image_retained_image(pair) == Float32[10, 20, 30, 40, 50, 60]
    @test all(isnothing, values(FilterGraphs.output_metadata(graph)))
    @test all(values(process!(outputs, graph, (; image=unfringed),
        (; image=sample(2, ChoppedImageUnfringed)))))
    @test outputs.output ≈ Float32[-0.45, -2.7, -3.15]
    @test outputs.var"pair-status" == UInt8[1]
    @test all(values(process!(outputs, graph, (; image=fringed .+ 1),
        (; image=sample(3, ChoppedImageFringed)))))
    @test outputs.output ≈ Float32[-0.95, -5.45, -6.4]
    @test FilterGraphs.output_metadata(graph).output.sequence == 3
    reset!(graph)
    @test !chopped_image_is_primed(pair)
end
end
