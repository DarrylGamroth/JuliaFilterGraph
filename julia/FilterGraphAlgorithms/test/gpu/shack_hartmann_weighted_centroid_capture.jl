using FilterGraphs

import FilterGraphAlgorithms:
    complete!,
    enqueue!,
    prepare_capture,
    process!

const CAPTURED_WEIGHTED_SHACK_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = measure
            label = shack-hartmann-weighted-centroid-f32
            config = {
                image_size = [2, 3]
                image_schema = org.example.shwfs-frame/1
                subaperture_size = [2, 3]
                subaperture_count = 1
                initial_subaperture_origins = [ [0 0] ]
                coordinate_scale = 1.0
                pixel_threshold = 2.0
                flux_threshold = 4.0
                reference_slopes = [ 0.25 -0.5 ]
                active = [ true ]
                initial_weights = [ [0.0 1.0 2.0 3.0 0.5 1.0] ]
            }
        }
    ]
    links = []
    inputs = [ "measure:image" ]
    outputs = [ "measure:slopes" "measure:flux" "measure:validity" ]
}
"""

function _captured_weighted_shack_replay_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

function run_shack_hartmann_weighted_capture_tests(
    device,
    to_device,
    device_zeros,
    synchronize,
    used_memory,
)
    @testset "captured weighted Shack-Hartmann matches CPU and preserves the unweighted path" begin
        weights = [
            Float32[100, 1, 2, 3, 0.5, 1],
            Float32[1, 0, 2, 0, 3, 4],
        ]
        weighted = prepare_algorithm(
            ShackHartmannWeightedCentroidF32,
            ShackHartmannWeightedCentroidF32Configuration(
                image_size=(4, 6),
                image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(2, 3),
                subaperture_count=2,
                initial_subaperture_origins=[[0, 0], [2, 3]],
                coordinate_scale=1.0f0,
                pixel_threshold=2.0f0,
                flux_threshold=4.0f0,
                reference_slopes=Float32[0.25, -0.5, -0.75, 0.25],
                active=Bool[true, false],
                initial_weights=weights,
            ),
        )
        unweighted = prepare_algorithm(
            ShackHartmannImageF32,
            ShackHartmannImageF32Configuration(
                image_size=(4, 6),
                image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(2, 3),
                subaperture_count=2,
                initial_subaperture_origins=[[0, 0], [2, 3]],
                coordinate_scale=1.0f0,
                pixel_threshold=2.0f0,
                flux_threshold=4.0f0,
                reference_slopes=Float32[0.25, -0.5, -0.75, 0.25],
                active=Bool[true, false],
            ),
        )
        @test capture_support(ShackHartmannWeightedCentroidF32, device) isa SupportsCapture
        @test capture_support(ShackHartmannImageF32, device) isa SupportsCapture

        host_image = Float32[
            1 2 3 4 5 6
            7 8 9 10 11 12
            13 14 15 16 17 18
            19 20 21 22 23 24
        ]
        image = to_device(host_image)
        frame_status = device_zeros(Int32, 1)
        weighted_outputs = (
            device_zeros(Float32, 2, 2),
            device_zeros(Float32, 2),
            device_zeros(Bool, 2),
        )
        unweighted_outputs = (
            device_zeros(Float32, 2, 2),
            device_zeros(Float32, 2),
            device_zeros(Bool, 2),
        )
        weighted_capture = prepare_capture(
            CompleteTransition(),
            device,
            weighted,
            weighted_outputs,
            (image,),
            frame_status,
            Int32(1),
        )
        unweighted_capture = prepare_capture(
            CompleteTransition(),
            device,
            unweighted,
            unweighted_outputs,
            (image,),
            frame_status,
            Int32(2),
        )
        aliased = prepare_algorithm(
            ShackHartmannWeightedCentroidF32,
            ShackHartmannWeightedCentroidF32Configuration(
                image_size=(1, 2),
                image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(1, 2),
                subaperture_count=1,
                initial_subaperture_origins=[[0, 0]],
                coordinate_scale=1.0f0,
                pixel_threshold=0.0f0,
                flux_threshold=0.0f0,
                reference_slopes=Float32[0, 0],
                active=Bool[true],
                initial_weights=[Float32[1, 1]],
            ),
        )
        aliased_image = to_device(reshape(Float32[1, 2], 1, 2))
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(),
            device,
            aliased,
            (aliased_image, device_zeros(Float32, 1), device_zeros(Bool, 1)),
            (aliased_image,),
            frame_status,
            Int32(3),
        )
        unweighted_aliased = prepare_algorithm(
            ShackHartmannImageF32,
            ShackHartmannImageF32Configuration(
                image_size=(1, 2),
                image_schema=CALIBRATED_PIXELS_V1,
                subaperture_size=(1, 2),
                subaperture_count=1,
                initial_subaperture_origins=[[0, 0]],
                coordinate_scale=1.0f0,
                pixel_threshold=0.0f0,
                flux_threshold=0.0f0,
                reference_slopes=Float32[0, 0],
                active=Bool[true],
            ),
        )
        @test_throws ArgumentError prepare_capture(
            CompleteTransition(),
            device,
            unweighted_aliased,
            (aliased_image, device_zeros(Float32, 1), device_zeros(Bool, 1)),
            (aliased_image,),
            frame_status,
            Int32(4),
        )
        @test Array(weighted_capture.weights) == weighted.plan.weights

        function compare_cpu(prepared, captured_outputs)
            expected = (zeros(Float32, 2, 2), zeros(Float32, 2), falses(2))
            process!(expected..., prepared, host_image)
            enqueue!(
                prepared === weighted ? weighted_capture : unweighted_capture,
            )
            synchronize()
            actual = Array.(captured_outputs)
            @test actual[1] ≈ expected[1] rtol=2.0f-5 atol=2.0f-6 nans=true
            @test actual[2] ≈ expected[2] rtol=2.0f-5 atol=2.0f-6 nans=true
            @test actual[3] == expected[3]
        end

        compare_cpu(weighted, weighted_outputs)
        compare_cpu(unweighted, unweighted_outputs)
        thresholded_flux = Array(weighted_outputs[2])[1]
        @test Array(weighted_outputs[2]) != Array(unweighted_outputs[2])
        @test Array(weighted_outputs[3]) == Bool[true, true]
        @test all(iszero, Array(weighted_outputs[1])[2, :])

        # Threshold applies to the calibrated pixel before its large weight.
        host_image[1, 1] = 2.0f0
        copyto!(image, host_image)
        compare_cpu(weighted, weighted_outputs)
        @test Array(weighted_outputs[2])[1] ≈ thresholded_flux + 200.0f0

        # A non-finite selected pixel remains invalid even when its weight is zero.
        host_image[3, 5] = NaN32
        copyto!(image, host_image)
        compare_cpu(weighted, weighted_outputs)
        @test !Array(weighted_outputs[3])[2]
        @test isnan(Array(weighted_outputs[2])[2])

        # Flux threshold uses the weighted sum after pixel selection.
        fill!(host_image, 0.0f0)
        host_image[1, 2] = 2.0f0
        copyto!(image, host_image)
        compare_cpu(weighted, weighted_outputs)
        @test Array(weighted_outputs[2])[1] == 2.0f0
        @test !Array(weighted_outputs[3])[1]

        # A prior Graph failure suppresses writes by either adapter.
        fill!(frame_status, Int32(1))
        fill!(weighted_outputs[1], 71.0f0)
        fill!(weighted_outputs[2], 73.0f0)
        fill!(weighted_outputs[3], true)
        enqueue!(weighted_capture)
        synchronize()
        @test Array(weighted_outputs[1]) == fill(71.0f0, 2, 2)
        @test Array(weighted_outputs[2]) == fill(73.0f0, 2)
        @test all(Array(weighted_outputs[3]))
        @test complete!(weighted_capture, Int32(1)) === nothing
        @test FilterGraphAlgorithms.preflight(weighted_capture) ===
              nothing
        @test FilterGraphAlgorithms.reset!(weighted_capture) ===
              weighted_capture
    end

    @testset "weighted Shack-Hartmann replay has zero Julia heap and stable device footprint" begin
        graph = mktemp() do path, io
            write(io, CAPTURED_WEIGHTED_SHACK_GRAPH_TEXT)
            flush(io)
            checked = FilterGraphs.prepare_graph(
                path;
                algorithms=(ShackHartmannWeightedCentroidF32,),
            )
            FilterGraphs.prepare_captured_graph(checked, device)
        end
        @test FilterGraphs.captured_graph_state(graph) ===
              FilterGraphs.CapturedGraphReady
        image = Float32[1 2 3; 7 8 9]
        inputs = (image=image,)
        outputs = (
            slopes=zeros(Float32, 1, 2),
            flux=zeros(Float32, 1),
            validity=falses(1),
        )
        @test process!(outputs, graph, inputs) ==
              (slopes=true, flux=true, validity=true)
        @test outputs.slopes ≈ Float32[-1 / 7 - 0.25 13 / 42 + 0.5]
        @test outputs.flux == Float32[42]
        @test outputs.validity == Bool[true]
        @test _captured_weighted_shack_replay_bytes(outputs, graph, inputs) == 0
        device_bytes = used_memory()
        for _ in 1:8
            @test process!(outputs, graph, inputs) ==
                  (slopes=true, flux=true, validity=true)
        end
        @test used_memory() == device_bytes

        image[1, 1] = NaN32
        @test process!(outputs, graph, inputs) ==
              (slopes=true, flux=true, validity=true)
        @test outputs.slopes == zeros(Float32, 1, 2)
        @test isnan(only(outputs.flux))
        @test outputs.validity == Bool[false]
        @test _captured_weighted_shack_replay_bytes(outputs, graph, inputs) == 0
        @test FilterGraphs.reset!(graph) === graph
        image[1, 1] = 1.0f0
        @test process!(outputs, graph, inputs) ==
              (slopes=true, flux=true, validity=true)
        @test outputs.flux == Float32[42]
        @test outputs.validity == Bool[true]
    end
    return nothing
end
