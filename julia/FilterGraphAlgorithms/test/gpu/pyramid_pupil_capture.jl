using FilterGraphs

import FilterGraphAlgorithms:
                              complete!,
                              enqueue!,
                              prepare_capture,
                              reset!,
                              process!

function _prepared_captured_pupil_image()
    return prepare_algorithm(
        PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(
            image_size=(6, 8),
            pupil_size=(2, 3),
            pupil_origins = [[0, 0], [3, 0], [0, 4], [3, 4]],
            pupil_mask = Bool[true, false, true, false, true, false]
        )
    )
end

function _prepared_captured_pupil_reconstructor(matrix)
    return prepare_algorithm(
        PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(
            reconstructed_count = size(matrix, 1),
            selected_pixel_count = div(size(matrix, 2), 4),
            initial_reconstructor = collect(vec(permutedims(matrix))),
            reconstructed_schema = "org.example.ao.pupil-reconstruction/1"
        )
    )
end

const CAPTURED_PYRAMID_PUPIL_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = pyramid-pupil-image-f32
            config = {
                image_size = [2, 4]
                pupil_size = [1, 2]
                pupil_origins = [ [0 0] [1 0] [0 2] [1 2] ]
                pupil_mask = null
            }
        }
        {
            type = ndarray
            name = reconstruct
            label = pwfs-pupil-reconstructor-f32
            config = {
                reconstructed_count = 1
                selected_pixel_count = 2
                initial_reconstructor = [ 3.0e38 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ]
                reconstructed_schema = org.example.ao.pupil-reconstruction/1
            }
        }
    ]
    links = [
        { output = "measure:reconstruction-pixels" input = "reconstruct:reconstruction-pixels" }
    ]
    inputs = [ "measure:image" ]
    outputs = [ "reconstruct:reconstructed" ]
}
"""

function _captured_pupil_replay_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs.process!(outputs, graph, inputs)
end

function run_pyramid_pupil_capture_tests(
        device, to_device, device_zeros, synchronize, used_memory
)
    @testset "legacy four-pupil capture uses previous-frame full-pupil gain" begin
        prepared = _prepared_captured_pupil_image()
        host_image = reshape(Float32.(1:48), 6, 8)
        image = to_device(host_image)
        pixels = device_zeros(Float32, 4, 3)
        mean = device_zeros(Float32, 1)
        status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (pixels, mean), (image,), status, Int32(1)
        )
        @test capture_support(PyramidPupilImageF32, device) isa SupportsCapture
        if !FilterGraphAlgorithms.preserves_inputs(captured)
            extension = parentmodule(typeof(captured))
            @test extension._prepare_bounded_validation(captured,
                (extension._PixelBounds(1.0, 0),)) === captured
            @test isnothing(extension._capture_output_bound(captured, Val(1)))
        end
        @test FilterGraphAlgorithms.preflight(captured) === nothing
        expected = (zeros(Float32, 4, 3), zeros(Float32, 1))
        addresses = (pointer(captured.previous_gain), pointer(captured.candidate_gain),
            pointer(captured.pupil_indices), pointer(captured.selected_indices))
        for frame in 1:4
            host_image .= reshape(Float32.(1:48), 6, 8) .* Float32(frame) .+
                          Float32(frame - 1)
            copyto!(image, host_image)
            @test process!(expected..., prepared, host_image) === nothing
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test Array(pixels) ≈ expected[1] rtol=8eps(Float32)
            @test Array(mean) ≈ expected[2] rtol=8eps(Float32)
            @test only(Array(captured.previous_gain)) ≈
                  pyramid_normalization_gain(prepared.workspace) rtol=8eps(Float32)
            @test frame == 1 ? all(iszero, Array(pixels)) : any(!iszero, Array(pixels))
        end

        # The excluded detector border is not part of any pupil.
        host_image[3, 4] = NaN32
        copyto!(image, host_image)
        @test process!(expected..., prepared, host_image) === nothing
        enqueue!(captured)
        synchronize()
        @test only(Array(status)) == 0
        @test Array(pixels) ≈ expected[1] rtol=8eps(Float32)

        retained_gain = only(Array(captured.previous_gain))
        # The mask excludes this sample from reconstruction, but the complete
        # four-pupil mean must still read and validate it.
        host_image[1, 2] = NaN32
        copyto!(image, host_image)
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              PyramidPixelImageNonFiniteInput
        @test only(Array(captured.previous_gain)) == retained_gain

        reset!(captured)
        fill!(status, Int32(0))
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              PyramidPixelImageNonFiniteInput
        @test iszero(only(Array(captured.previous_gain)))

        fill!(host_image, 4.0f0)
        copyto!(image, host_image)
        fill!(status, Int32(0))
        enqueue!(captured)
        synchronize()
        @test all(iszero, Array(pixels))
        @test Array(mean) == Float32[4]
        @test only(Array(captured.previous_gain)) == 0.25f0
        @test addresses ==
              (pointer(captured.previous_gain), pointer(captured.candidate_gain),
            pointer(captured.pupil_indices), pointer(captured.selected_indices))

        # An upstream failure cannot advance this Node's gain.
        fill!(image, 8.0f0)
        fill!(status, Int32(42))
        enqueue!(captured)
        synchronize()
        @test only(Array(status)) == 42
        @test only(Array(captured.previous_gain)) == 0.25f0
        @test complete!(captured, Int32(42)) === nothing
    end

    @testset "legacy pupil failure identity and priority match serial CPU" begin
        prepared = _prepared_captured_pupil_image()
        image = device_zeros(Float32, 6, 8)
        outputs = (device_zeros(Float32, 4, 3), device_zeros(Float32, 1))
        status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, outputs, (image,), status, Int32(2)
        )
        cpu_outputs = (zeros(Float32, 4, 3), zeros(Float32, 1))
        host_image = zeros(Float32, 6, 8)
        for input_first in (false, true)
            fill!(host_image, 0.0f0)
            host_image[1, 1] = input_first ? NaN32 : floatmax(Float32)
            host_image[1, 2] = floatmax(Float32)
            host_image[1, 3] = input_first ? floatmax(Float32) : NaN32
            copyto!(image, host_image)
            fill!(status, Int32(0))
            expected_failure = process!(cpu_outputs..., prepared, host_image)
            @test expected_failure === (input_first ? PyramidPixelImageNonFiniteInput :
                   PyramidPixelImageNonFiniteMean)
            enqueue!(captured)
            synchronize()
            @test complete!(captured, only(Array(status))) ===
                  expected_failure
            @test iszero(only(Array(captured.previous_gain)))
        end

        # Positive reciprocal overflow is a separate declared disposition.
        fill!(host_image, nextfloat(0.0f0))
        copyto!(image, host_image)
        fill!(status, Int32(0))
        @test process!(cpu_outputs..., prepared, host_image) ===
              PyramidPixelImageNonFiniteNormalizationGain
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              PyramidPixelImageNonFiniteNormalizationGain

        # A finite tiny prime frame makes the next normalized output overflow.
        fill!(host_image, floatmin(Float32))
        copyto!(image, host_image)
        fill!(status, Int32(0))
        @test process!(cpu_outputs..., prepared, host_image) === nothing
        enqueue!(captured)
        synchronize()
        retained_gain = only(Array(captured.previous_gain))
        fill!(host_image, 64.0f0)
        copyto!(image, host_image)
        @test process!(cpu_outputs..., prepared, host_image) ===
              PyramidPixelImageNonFiniteOutput
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              PyramidPixelImageNonFiniteOutput
        @test only(Array(captured.previous_gain)) == retained_gain

        # Mean/input rejection precedes a possible selected-output overflow.
        host_image[1, 2] = NaN32
        copyto!(image, host_image)
        fill!(status, Int32(0))
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              PyramidPixelImageNonFiniteInput

        reset!(captured)
        for value in (-4.0f0, 0.0f0, 8.0f0)
            fill!(image, value)
            fill!(status, Int32(0))
            enqueue!(captured)
            synchronize()
            @test iszero(only(Array(status)))
            @test Array(outputs[2]) == Float32[value]
            @test all(iszero, Array(outputs[1]))
            @test only(Array(captured.previous_gain)) == (value > 0 ? inv(value) : 0.0f0)
        end
    end

    @testset "exceptional cancellation retains a successful serial rescan sum" begin
        prepared = prepare_algorithm(
            PyramidPupilImageF32,
            PyramidPupilImageF32Configuration(
                image_size=(2, 4), pupil_size=(1, 2),
                pupil_origins = [[0, 0], [1, 0], [0, 2], [1, 2]],
                pupil_mask = nothing
            )
        )
        maximum = floatmax(Float32)
        # CPU order is [M,-M,M,-M,M,-M,M,-M], whose prefixes are finite.
        # The parallel tree combines the four positive terms before their
        # negative counterparts, producing Inf + -Inf before the rescan.
        host_image = Float32[maximum -maximum maximum -maximum;
                             maximum -maximum maximum -maximum]
        image = to_device(host_image)
        outputs = (device_zeros(Float32, 4, 2), device_zeros(Float32, 1))
        cpu_outputs = (zeros(Float32, 4, 2), zeros(Float32, 1))
        status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, outputs, (image,), status, Int32(2)
        )
        for value in (nothing, 2.0f0, 4.0f0)
            if !isnothing(value)
                fill!(host_image, value)
                copyto!(image, host_image)
            end
            fill!(status, Int32(0))
            @test process!(cpu_outputs..., prepared, host_image) === nothing
            enqueue!(captured)
            synchronize()
            @test iszero(only(Array(status)))
            @test Array(outputs[1]) == cpu_outputs[1]
            @test Array(outputs[2]) == cpu_outputs[2]
            @test only(Array(captured.previous_gain)) ==
                  pyramid_normalization_gain(prepared.workspace)
        end
        @test Array(outputs[1]) == fill(2.0f0, 4, 2)
    end

    @testset "captured signed pupil cancellation preserves disposition and next gain" begin
        description = (;
            nodes=((name=:pupil, label="pyramid-pupil-image-f32",
                config=(image_size=(2, 2), pupil_size=(1, 1),
                    pupil_origins=[[0, 0], [1, 0], [0, 1], [1, 1]], pupil_mask=nothing)),),
            links=(), inputs=(image="pupil:image",),
            outputs=(pixels="pupil:reconstruction-pixels", mean="pupil:mean-pupil-intensity"),
        )
        cpu = prepare_graph(description; algorithms=(PyramidPupilImageF32,))
        gpu = prepare_captured_graph(cpu, device)
        expected = (pixels=zeros(Float32, 4, 1), mean=zeros(Float32, 1))
        actual = (pixels=similar(expected.pixels), mean=similar(expected.mean))
        for (frame, disposition) in (
            (Float32[2^24, 1, -2^24, 0], (pixels=true, mean=true)),
            (ones(Float32, 4), (pixels=true, mean=true)),
            (Float32[1, eps(Float32) / 2, -1, 1f-39], PyramidPixelImageNonFiniteNormalizationGain),
            (fill(2f0, 4), (pixels=true, mean=true)),
        )
            inputs = (image=reshape(frame, 2, 2),)
            @test FilterGraphs.process!(expected, cpu, inputs) === disposition
            @test FilterGraphs.process!(actual, gpu, inputs) === disposition
            @test actual == expected
        end
    end

    @testset "pupil gain finalization follows every selected-pixel producer" begin
        for selected in (65, 129)
            prepared = prepare_algorithm(PyramidPupilImageF32,
                PyramidPupilImageF32Configuration(
                    image_size=(2, 2selected),
                    pupil_size=(1, selected),
                    pupil_origins = [[0, 0], [1, 0], [0, selected], [1, selected]],
                    pupil_mask = nothing))
            host_image = fill(floatmin(Float32), 2, 2selected)
            image = to_device(host_image)
            pixels, mean = device_zeros(Float32, 4, selected), device_zeros(Float32, 1)
            status = device_zeros(Int32, 1)
            captured = prepare_capture(CompleteTransition(), device, prepared,
                (pixels, mean), (image,), status, Int32(1))
            addresses = map(pointer, (pixels, mean, captured.previous_gain, captured.candidate_gain))
            expected = (zeros(Float32, 4, selected), zeros(Float32, 1))
            @test process!(expected..., prepared, host_image) === nothing
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test all(iszero, Array(pixels))
            retained_gain = only(Array(captured.previous_gain))

            fill!(host_image, 1.0f0)
            host_image[end] = 64.0f0
            copyto!(image, host_image)
            @test process!(expected..., prepared, host_image) === PyramidPixelImageNonFiniteOutput
            enqueue!(captured)
            synchronize()
            @test complete!(captured, only(Array(status))) === PyramidPixelImageNonFiniteOutput
            @test only(Array(captured.previous_gain)) == retained_gain
            # Direct capture retains its existing speculative supplied outputs;
            # the graph owner suppresses their external publication on failure.
            @test isinf(Array(pixels)[end])
            @test only(Array(mean)) > 1.0f0

            fill!(status, Int32(0))
            fill!(host_image, 2.0f0)
            copyto!(image, host_image)
            @test process!(expected..., prepared, host_image) === nothing
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == 0
            @test Array(pixels) == expected[1]
            @test only(Array(captured.previous_gain)) == 0.5f0
            saved = (Array(pixels), Array(mean), Array(captured.candidate_gain))
            fill!(status, Int32(0x103))
            fill!(image, 8.0f0)
            enqueue!(captured)
            synchronize()
            @test only(Array(status)) == Int32(0x103)
            @test (Array(pixels), Array(mean), Array(captured.candidate_gain)) == saved
            @test only(Array(captured.previous_gain)) == 0.5f0
            reset!(captured)
            synchronize()
            @test only(Array(captured.previous_gain)) == 0
            @test addresses == map(pointer, (pixels, mean, captured.previous_gain, captured.candidate_gain))
        end
    end

    @testset "legacy pupil reconstructor respects pupil-major scientific columns" begin
        # Every matrix column and input coordinate differs. Public 4×3 storage
        # interleaves pupils; reconstruction consumes all of pupil 1 first.
        matrix = Float32[row + 3column for row in 1:5, column in 1:12] ./ 32.0f0
        prepared = _prepared_captured_pupil_reconstructor(matrix)
        host_pixels = Float32[100pupil + selected for pupil in 1:4, selected in 1:3]
        pixels = to_device(host_pixels)
        output = device_zeros(Float32, 5)
        status = device_zeros(Int32, 1)
        captured = prepare_capture(
            CompleteTransition(),
            device, prepared, (output,), (pixels,), status, Int32(3)
        )
        @test_throws DimensionMismatch prepare_capture(
            CompleteTransition(),
            device, prepared, (output,), (device_zeros(Float32, 3, 4),), status, Int32(3)
        )
        @test capture_support(PyramidPupilReconstructorF32, device) isa SupportsCapture
        if !FilterGraphAlgorithms.preserves_inputs(captured)
            extension = parentmodule(typeof(captured))
            @test FilterGraphAlgorithms.prepare_validation(captured, (true,)) === captured
            @test extension._prepare_bounded_validation(captured,
                (extension._FiniteBound(1.0),)) === captured
        end
        expected = zeros(Float32, 5)
        for frame in 1:3
            host_pixels .+= Float32(frame)
            copyto!(pixels, host_pixels)
            @test process!(expected, prepared, host_pixels) === nothing
            enqueue!(captured)
            synchronize()
            @test Array(output) ≈ expected rtol=8eps(Float32)
            @test Array(captured.ordered_input) == vec(permutedims(host_pixels))
            @test iszero(only(Array(status)))
        end
        host_pixels[4, 3] = NaN32
        copyto!(pixels, host_pixels)
        enqueue!(captured)
        synchronize()
        @test complete!(captured, only(Array(status))) ===
              DenseReconstructionNonFiniteInput

        overflow = _prepared_captured_pupil_reconstructor(fill(floatmax(Float32), 1, 12))
        overflow_pixels = to_device(fill(2.0f0, 4, 3))
        overflow_output = device_zeros(Float32, 1)
        overflow_capture = prepare_capture(
            CompleteTransition(),
            device, overflow, (overflow_output,), (overflow_pixels,), status, Int32(4)
        )
        fill!(status, Int32(0))
        enqueue!(overflow_capture)
        synchronize()
        @test complete!(overflow_capture, only(Array(status))) ===
              DenseReconstructionNonFiniteOutput
        host_pixels .= 2.0f0
        host_pixels[4, 3] = NaN32
        copyto!(overflow_pixels, host_pixels)
        fill!(status, Int32(0))
        enqueue!(overflow_capture)
        synchronize()
        @test complete!(overflow_capture, only(Array(status))) ===
              DenseReconstructionNonFiniteInput
    end

    @testset "native pupil graph adopts gain before downstream failure and replays without allocation" begin
        captured = mktemp() do path, io
            write(io, CAPTURED_PYRAMID_PUPIL_GRAPH_TEXT)
            flush(io)
            checked = FilterGraphs.prepare_graph(
                path; algorithms = (PyramidPupilImageF32, PyramidPupilReconstructorF32)
            )
            FilterGraphs.prepare_captured_graph(checked, device)
        end
        image = fill(1.0f0, 2, 4)
        inputs = (image = image,)
        outputs = (reconstructed = fill(71.0f0, 1),)
        @test FilterGraphs.process!(outputs, captured, inputs) ==
              (reconstructed = true,)
        @test outputs.reconstructed == Float32[0]
        fill!(image, 2.0f0)
        @test FilterGraphs.process!(outputs, captured, inputs) ===
              DenseReconstructionNonFiniteOutput
        @test outputs.reconstructed == Float32[0]
        # The rejected frame still adopted mean=2 at the successful WFS Node.
        fill!(image, 0.5f0)
        @test FilterGraphs.process!(outputs, captured, inputs) ==
              (reconstructed = true,)
        @test outputs.reconstructed ≈ Float32[0.25f0 * 3.0f38] rtol=8eps(Float32)

        # Invalid WFS frames retain its previous gain and suppress publication.
        published = copy(outputs.reconstructed)
        image[1, 2] = NaN32
        @test FilterGraphs.process!(outputs, captured, inputs) ===
              PyramidPixelImageNonFiniteInput
        @test outputs.reconstructed == published
        @test _captured_pupil_replay_bytes(outputs, captured, inputs) == 0
        @test FilterGraphs.reset!(captured) === captured
        fill!(image, 1.0f0)
        @test FilterGraphs.process!(outputs, captured, inputs) ==
              (reconstructed = true,)
        @test outputs.reconstructed == Float32[0]
        @test FilterGraphs.process!(outputs, captured, inputs) ==
              (reconstructed = true,)
        @test _captured_pupil_replay_bytes(outputs, captured, inputs) == 0
        device_bytes = used_memory()
        for _ in 1:16
            @test FilterGraphs.process!(outputs, captured, inputs) ==
                  (reconstructed = true,)
        end
        @test used_memory() == device_bytes
    end
    return nothing
end
