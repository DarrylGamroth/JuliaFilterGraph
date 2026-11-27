module PupilLayoutCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, Test
import FilterGraphAlgorithms: process!, reset!, prepare_capture, enqueue!, complete!

const FGA = FilterGraphAlgorithms
const CUDAX = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)
const Fixtures = parentmodule(@__MODULE__)

"""Prepare the same masked pupil/reconstruction chain in either physical layout."""
function layout_case(matrix; pupil_major=false)
    image = CUDA.fill(1.0f0, 6, 8)
    pixels = pupil_major ? PermutedDimsArray(CUDA.zeros(Float32, 3, 4), (2, 1)) : CUDA.zeros(Float32, 4, 3)
    mean, status, reconstructed = CUDA.zeros(Float32, 1), CUDA.zeros(Int32, 1), CUDA.zeros(Float32, size(matrix, 1))
    pupil = prepare_capture(CompleteTransition(), CUDA.device(), Fixtures._prepared_captured_pupil_image(),
        (pixels, mean), (image,), status, Int32(1))
    reconstructor = prepare_capture(CompleteTransition(), CUDA.device(), Fixtures._prepared_captured_pupil_reconstructor(matrix),
        (reconstructed,), (pixels,), status, Int32(2))
    return (; image, pixels, mean, status, reconstructed, pupil, reconstructor)
end

function step!(case)
    fill!(case.status, Int32(0))
    enqueue!(case.pupil)
    enqueue!(case.reconstructor)
    return nothing
end

"""Warm, reset, and record the exact direct-adapter sequence before replay."""
function record!(case)
    step!(case)
    CUDA.synchronize()
    reset!(case.pupil)
    reset!(case.reconstructor)
    CUDA.synchronize()
    graph = CUDA.capture(() -> step!(case))
    return (; graph, executable=CUDA.instantiate(graph))
end

function replay!(case, recorded, image)
    copyto!(case.image, image)
    CUDA.launch(recorded.executable)
    CUDA.synchronize()
    return only(Array(case.status))
end

logical_pixels(case) = case.pixels isa CUDA.CuArray ? Array(case.pixels) : permutedims(Array(parent(case.pixels)))
bits(array) = collect(reinterpret(UInt32, vec(array)))
frame_bytes(output, owner, inputs) = @allocated process!(output, owner, inputs)

@testset "CUDA pupil-major layout preserves captured arithmetic and gain history" begin
    matrix = reshape(Float32.(1:36) ./ 32.0f0, 3, 12)
    dense = layout_case(matrix)
    major = layout_case(matrix; pupil_major=true)
    @test size(major.pixels) == (4, 3)
    @test size(parent(major.pixels)) == (3, 4)
    @test CUDAX._pupil_mightalias(major.pixels, parent(major.pixels))
    @test CUDAX._pupil_mightalias(major.reconstructor.ordered_input, major.pixels)
    @test FilterGraphs._captured_mightalias(major.pixels, parent(major.pixels))
    @test FilterGraphs._captured_mightalias(major.reconstructor.ordered_input, major.pixels)
    @test !FilterGraphs._captured_safe_overlap(
        major.reconstructor, (major.reconstructor.ordered_input,), (major.pixels,))
    overlapping = Fixtures._prepared_captured_pupil_reconstructor(ones(Float32, 12, 12))
    @test_throws ArgumentError prepare_capture(CompleteTransition(), CUDA.device(), overlapping,
        (reshape(parent(major.pixels), 12),), (major.pixels,), major.status, Int32(2))
    @test_throws MethodError CUDAX._require_capture_array(
        "unqualified wrapper", major.pixels, CUDA.device())
    @test major.pupil.output === major.reconstructor.input === major.pixels
    @test pointer(major.reconstructor.ordered_input) == pointer(parent(major.pixels))
    @test major.reconstructor.order! isa CUDAX._PupilMajorOrder
    @test !(dense.reconstructor.order! isa CUDAX._PupilMajorOrder)
    @test bits(Array(dense.reconstructor.projection.matrix)) == bits(Array(major.reconstructor.projection.matrix))
    records = (record!(dense), record!(major))
    addresses = (pointer(parent(major.pixels)), pointer(major.pupil.previous_gain))
    image = reshape(Float32.(1:48), 6, 8)
    for sequence in 1:4
        image .= reshape(Float32.(1:48), 6, 8) .* Float32(sequence) .+ Float32(sequence - 1)
        @test replay!(dense, records[1], image) == replay!(major, records[2], image) == 0
        @test bits(logical_pixels(dense)) == bits(logical_pixels(major))
        @test bits(Array(parent(major.pixels))) == bits(permutedims(Array(dense.pixels)))
        @test bits(Array(dense.reconstructed)) == bits(Array(major.reconstructed))
        @test Array(dense.mean) == Array(major.mean)
        @test Array(dense.pupil.previous_gain) == Array(major.pupil.previous_gain)
        if sequence == 1
            @test all(iszero, logical_pixels(major))
        end
    end
    image[3, 4] = NaN32 # Detector border outside all four pupils is ignored.
    @test replay!(dense, records[1], image) == replay!(major, records[2], image) == 0
    @test bits(Array(dense.reconstructed)) == bits(Array(major.reconstructed))
    retained_gain = only(Array(major.pupil.previous_gain))
    image[1, 2] = NaN32 # Mask-excluded samples still participate in the full mean.
    for (case, record) in zip((dense, major), records)
        status = replay!(case, record, image)
        @test complete!(case.pupil, status) === PyramidPixelImageNonFiniteInput
        @test only(Array(case.pupil.previous_gain)) == retained_gain
    end
    for input_first in (false, true)
        fill!(image, 0.0f0)
        image[1, 1] = input_first ? NaN32 : floatmax(Float32)
        image[1, 2] = floatmax(Float32)
        image[1, 3] = input_first ? floatmax(Float32) : NaN32
        expected = input_first ? PyramidPixelImageNonFiniteInput : PyramidPixelImageNonFiniteMean
        for (case, record) in zip((dense, major), records)
            @test complete!(case.pupil, replay!(case, record, image)) === expected
            @test only(Array(case.pupil.previous_gain)) == retained_gain
        end
    end
    fill!(image, 4.0f0)
    @test replay!(dense, records[1], image) == replay!(major, records[2], image) == 0
    @test bits(Array(dense.reconstructed)) == bits(Array(major.reconstructed))
    for (case, record) in zip((dense, major), records)
        reset!(case.pupil)
        reset!(case.reconstructor)
        CUDA.synchronize()
        @test replay!(case, record, image) == 0
        @test all(iszero, logical_pixels(case))
        @test only(Array(case.pupil.previous_gain)) == 0.25f0
    end
    @test addresses == (pointer(parent(major.pixels)), pointer(major.pupil.previous_gain))
end

"""Prepare the established complete pupil graph with optional exported pixel output."""
function pupil_graph(; public_pixels=false)
    text = replace(Fixtures.CAPTURED_PYRAMID_PUPIL_GRAPH_TEXT, "3.0e38" => "0.25")
    if public_pixels
        text = replace(text, "\"reconstruct:reconstructed\" ]" =>
            "\"reconstruct:reconstructed\" \"measure:reconstruction-pixels\" \"measure:mean-pupil-intensity\" ]")
    end
    return mktemp() do path, io
        write(io, text)
        flush(io)
        prepare_graph(path; algorithms=(PyramidPupilImageF32, PyramidPupilReconstructorF32))
    end
end

@testset "CUDA internal pupil layout preserves exported pixels, metadata and allocation" begin
    source = pupil_graph()
    internal = prepare_captured_graph(source, CUDA.device())
    public_source = pupil_graph(; public_pixels=true)
    exported = prepare_captured_graph(public_source, CUDA.device())
    cpu = pupil_graph(; public_pixels=true)
    pupil, reconstructor = internal.nodes[1].adapter, internal.nodes[2].adapter
    image_alias = CUDA.zeros(Float32, 2, 4)
    @test_throws ArgumentError prepare_capture(CompleteTransition(), CUDA.device(), source.nodes[1].prepared,
        (PermutedDimsArray(image_alias, (2, 1)), CUDA.zeros(Float32, 1)),
        (image_alias,), CUDA.zeros(Int32, 1), Int32(1))
    @test pupil.output isa CUDAX._PupilMajorPixels
    @test reconstructor.input === pupil.output
    @test pointer(reconstructor.ordered_input) == pointer(parent(pupil.output))
    @test reconstructor.order! isa CUDAX._PupilMajorOrder
    @test exported.nodes[1].adapter.output isa CUDA.CuArray
    @test exported.device_outputs.var"reconstruction-pixels" === exported.nodes[1].adapter.output
    @test !(exported.nodes[2].adapter.order! isa CUDAX._PupilMajorOrder)
    expected = map(format -> zeros(format.element_type, format.shape), cpu.output_formats)
    actual = map(copy, expected)
    output = (reconstructed=zeros(Float32, 1),)
    image = reshape(Float32.(1:8), 2, 4)
    inputs = (image=image,)
    for sequence in 1:3
        image .= reshape(Float32.(1:8), 2, 4) .* Float32(sequence)
        metadata = SampleMetadata(sequence)
        @test process!(actual, exported, inputs, (image=metadata,)) == process!(expected, cpu, inputs, (image=metadata,))
        @test process!(output, internal, inputs, (image=metadata,)) == (reconstructed=true,)
        @test bits(output.reconstructed) == bits(actual.reconstructed)
        @test actual.var"reconstruction-pixels" ≈ expected.var"reconstruction-pixels" rtol=8eps(Float32)
        @test actual.var"mean-pupil-intensity" ≈ expected.var"mean-pupil-intensity" rtol=8eps(Float32)
        @test all(value -> value === metadata, values(FilterGraphs.output_metadata(exported)))
        @test FilterGraphs.output_metadata(internal) == (reconstructed=metadata,)
    end
    process!(output, internal, inputs)
    @test frame_bytes(output, internal, inputs) == 0
    retained = parent(pupil.output)
    @test reset!(internal) === internal
    @test parent(pupil.output) === retained
    @test process!(output, internal, inputs) == (reconstructed=true,)
    @test all(iszero, output.reconstructed)

    # A second real reconstructor consumer prevents the private-layout rewrite.
    nodes = (exported.nodes..., exported.nodes[2])
    routes = (public_source.input_routes..., public_source.input_routes[2])
    originals = ((exported.nodes[1].adapter.output, exported.nodes[1].adapter.mean_output),
        (exported.nodes[2].adapter.output,), (CUDA.zeros(Float32, 1),))
    planned = FilterGraphs._prepare_captured_layouts(CUDA.device(), nodes,
        routes, (), originals)
    @test planned[1][1] === originals[1][1]
    @test planned[1][1] isa CUDA.CuArray
end

end # module
