module BoundedValidationCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, Test
import FilterGraphAlgorithms: process!, reset!, preserves_inputs

const CUDAX = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
const JFGX = Base.get_extension(FilterGraphs, :FilterGraphsCUDAExt)
const Fixtures = parentmodule(@__MODULE__)

include("arithmetic_bounds_common.jl")
run_bound_math_tests(CUDAX)

@testset "CUDA fast math retains checked projection output" begin
    input = CUDAX._FiniteBound(1024.0)
    projection = CUDAX.CUBLASProjection(nothing, nothing, nothing, 0.25)
    math_mode = CUDA.CUDACore.math_mode()
    try
        CUDA.CUDACore.math_mode!(CUDA.CUDACore.FAST_MATH)
        @test isnothing(CUDAX._projection_bound(projection, input, 8))
    finally
        CUDA.CUDACore.math_mode!(math_mode)
    end
end

struct BoundProducer end
struct BoundConsumer
    bounded::Bool
end
struct BoundWriter end
struct OpaqueWriter end
preserves_inputs(::Union{BoundProducer,BoundConsumer,BoundWriter}) = true
CUDAX._capture_output_bound(::BoundProducer, ::Val{1}) = CUDAX._PixelBounds(1.0, 0)
CUDAX._prepare_bounded_validation(c::BoundConsumer, inputs::Tuple) =
    BoundConsumer(only(inputs) isa CUDAX._PixelBounds)
node(adapter) = FilterGraphs.CapturedNode{typeof(adapter),typeof(adapter)}(adapter)

@testset "range certificates expire at opaque writers and aliased outputs" begin
    source, intermediate, result = zeros(Float32, 4), zeros(Float32, 4), zeros(Float32, 4)
    nodes = (node(BoundProducer()), node(BoundConsumer(false)))
    routes = ((FilterGraphs.ExternalInput{1}(),), (FilterGraphs.NodeOutput{1,1}(),))
    outputs = ((intermediate,), (result,))
    refined = JFGX._reuse_captured_bounds(nodes, routes, outputs, (source=source,), outputs)
    @test refined[2].adapter.bounded
    for writer in (OpaqueWriter(), BoundWriter())
        middle = writer isa BoundWriter ? intermediate : similar(intermediate)
        chain = (node(BoundProducer()), node(writer), node(BoundConsumer(false)))
        links = (routes[1], routes[1], routes[2])
        buffers = ((intermediate,), (middle,), (result,))
        refined = JFGX._reuse_captured_bounds(chain, links, buffers, (source=source,), buffers)
        @test !refined[3].adapter.bounded
    end
    overlap = ((intermediate,), (view(intermediate, :),))
    refined = JFGX._reuse_captured_bounds(nodes, routes, overlap, (source=source,), overlap)
    @test !refined[2].adapter.bounded
end

function calibrated_graph(; flat=1.0f0, background=0.0625f0, coefficient=0.25f0, matrix=nothing)
    text = replace(Fixtures.CAPTURED_PYRAMID_PUPIL_GRAPH_TEXT,
        "3.0e38" => string(coefficient))
    document = FilterGraphs._parse_spa_json(text)["filter.graph"]
    pixels = matrix === nothing ? 2 : size(matrix, 2) ÷ 4
    image_size = [2, 2 * pixels]
    if matrix !== nothing
        sensing = document["nodes"][1]["config"]
        sensing["image_size"] = image_size
        sensing["pupil_size"] = [1, pixels]
        sensing["pupil_origins"] = [[0, 0], [1, 0], [0, pixels], [1, pixels]]
        reconstruction = document["nodes"][2]["config"]
        reconstruction["reconstructed_count"] = size(matrix, 1)
        reconstruction["selected_pixel_count"] = pixels
        reconstruction["initial_reconstructor"] = nothing
    end
    pushfirst!(document["nodes"], Dict("type" => "ndarray", "name" => "calibrate",
        "label" => "pixel-calibration-u16-f32", "config" => Dict(
            "image_size" => image_size,
            "initial_flat" => fill(flat, prod(image_size)),
            "initial_background" => fill(background, prod(image_size)))))
    pushfirst!(document["links"], Dict("output" => "calibrate:calibrated", "input" => "measure:image"))
    document["inputs"] = ["calibrate:raw", "calibrate:flat", "calibrate:background"]
    matrix === nothing || push!(document["inputs"], "reconstruct:reconstructor")
    graph = FilterGraphs._prepare_graph(FilterGraphs._graph_spec(document);
        algorithms=(PixelCalibrationU16F32, PyramidPupilImageF32, PyramidPupilReconstructorF32))
    matrix === nothing || replace_parameters!(graph, :reconstructor => matrix)
    return graph
end

frame_bytes(output, graph, input) = @allocated process!(output, graph, input)

@testset "CUDA captured pupil graph packs from an upstream magnitude bound" begin
    matrix = 64f0 * Fixtures.PupilProjectionCaptureTests.matrix_fixture()
    cpu, source = calibrated_graph(; matrix), calibrated_graph(; matrix)
    gpu = prepare_captured_graph(source, CUDA.device())
    pupil, reconstructor = gpu.nodes[2].adapter, gpu.nodes[3].adapter
    @test pupil.validate! isa CUDAX._BoundedPupil
    @test reconstructor.projection isa CUDAX.PupilProjection
    @test reconstructor.validate_output! isa CUDAX._BoundedOutput
    expected = (reconstructed=zeros(Float32, size(matrix, 1)),)
    actual = (reconstructed=similar(expected.reconstructed),)
    raw = reshape(UInt16[mod(i, 17) for i in 1:size(matrix, 2)], 2, :)
    input = (; raw)
    for _ in 1:3
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test actual.reconstructed ≈ expected.reconstructed rtol=8eps(Float32)
    end
    @test any(!iszero, actual.reconstructed)
end

@testset "captured bounded pupil and reconstructor preserve state and storage" begin
    cpu = calibrated_graph()
    gpu = prepare_captured_graph(calibrated_graph(), CUDA.device())
    pupil, reconstructor = gpu.nodes[2].adapter, gpu.nodes[3].adapter
    @test pupil.validate! isa CUDAX._BoundedPupil
    @test reconstructor.validate_output! isa CUDAX._BoundedOutput
    @test reconstructor.validate_input! isa CUDAX._FiniteInput
    expected, actual = (reconstructed=zeros(Float32, 1),), (reconstructed=zeros(Float32, 1),)
    input = (raw=zeros(UInt16, 2, 4),)
    addresses = (pointer(pupil.previous_gain), pointer(pupil.candidate_gain),
        pointer(reconstructor.output), pointer(reconstructor.ordered_input))
    for raw in (UInt16[0 0 0 0; 0 0 0 0], UInt16[1 0 0 0; 0 0 0 0],
            fill(typemax(UInt16), 2, 4), UInt16[7 0 4 2; 0 3 0 1])
        copyto!(input.raw, raw)
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test actual.reconstructed ≈ expected.reconstructed rtol=8eps(Float32)
        @test only(Array(pupil.previous_gain)) ≈
              pyramid_normalization_gain(cpu.nodes[2].prepared.workspace) rtol=8eps(Float32)
    end
    frame_bytes(actual, gpu, input)
    @test frame_bytes(actual, gpu, input) == 0
    reset!(cpu)
    reset!(gpu)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test all(iszero, actual.reconstructed)
    @test addresses == (pointer(pupil.previous_gain), pointer(pupil.candidate_gain),
        pointer(reconstructor.output), pointer(reconstructor.ordered_input))

    retained = only(Array(pupil.previous_gain))
    fill!(pupil.frame_status, Int32(77))
    FilterGraphAlgorithms.enqueue!(pupil)
    CUDA.synchronize()
    @test only(Array(pupil.frame_status)) == 77
    @test only(Array(pupil.previous_gain)) == retained

    # Replacement prepares fresh bounds; no proof from the old calibration is
    # transferred into a new owner. A tiny lattice and unsafe coefficients fall back.
    for (flat, background) in ((nextfloat(0.0f0), 0.0f0), (floatmax(Float32), 0.0f0))
        replacement = calibrated_graph()
        replace_parameters!(replacement, :flat => fill(flat, 2, 4),
            :background => fill(background, 2, 4))
        fallback = prepare_captured_graph(replacement, CUDA.device())
        @test !(fallback.nodes[2].adapter.validate! isa CUDAX._BoundedPupil)
        @test !(fallback.nodes[3].adapter.validate_output! isa CUDAX._BoundedOutput)
        replacement_output = (reconstructed=fill(71.0f0, 1),)
        cpu_output = (reconstructed=fill(71.0f0, 1),)
        rejected = (raw=fill(UInt16(2), 2, 4),)
        @test process!(replacement_output, fallback, rejected) ===
              process!(cpu_output, replacement, rejected)
        @test replacement_output.reconstructed == [71.0f0]
    end
    @test pupil.validate! isa CUDAX._BoundedPupil
end

@testset "bounded pupil retains downstream overflow and immediate gain commitment" begin
    cpu = calibrated_graph(; background=0.0f0, coefficient=3.0f38)
    gpu = prepare_captured_graph(calibrated_graph(; background=0.0f0, coefficient=3.0f38), CUDA.device())
    @test gpu.nodes[2].adapter.validate! isa CUDAX._BoundedPupil
    @test !(gpu.nodes[3].adapter.validate_output! isa CUDAX._BoundedOutput)
    expected, actual = (reconstructed=zeros(Float32, 1),), (reconstructed=zeros(Float32, 1),)
    input = (raw=UInt16[1 0 0 0; 0 0 0 0],)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    fill!(actual.reconstructed, 71)
    @test process!(actual, gpu, input) === DenseReconstructionNonFiniteOutput
    @test process!(expected, cpu, input) === DenseReconstructionNonFiniteOutput
    @test actual.reconstructed == [71.0f0]
    @test only(Array(gpu.nodes[2].adapter.previous_gain)) == 8.0f0
    fill!(input.raw, 0)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test actual.reconstructed == expected.reconstructed == [0.0f0]
end

end # module
