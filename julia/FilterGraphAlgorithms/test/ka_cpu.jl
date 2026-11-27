module KACPUTests

using FilterGraphAlgorithms
using KernelAbstractions
using LinearAlgebra
using Test

import FilterGraphAlgorithms:
    CompleteTransition, commit!, complete!, discard!, enqueue!, process!, reset!

# Load the maintained portable source without loading CUDA or AMDGPU. This
# private test device grants no capability to the real KA.CPU backend. Tests
# launch kernels on KA.CPU; they do not test native graph capture.
struct TestDevice end

const GPU_SOURCE = joinpath(@__DIR__, "..", "ext", "gpu")
include(joinpath(GPU_SOURCE, "linear_algebra", "arithmetic_bounds.jl"))
include(joinpath(GPU_SOURCE, "input_proofs.jl"))
include(joinpath(GPU_SOURCE, "linear_algebra", "finite_values.jl"))
include(joinpath(GPU_SOURCE, "linear_algebra", "projection.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "image", "pixel_calibration.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "control", "leaky_integrator.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "control", "closed_loop_correction.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "control", "frame_delay.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "shack_hartmann_image.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_pupil_image.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_pupil_reconstruction.jl"))

include("gpu/arithmetic_bounds_common.jl")
run_bound_math_tests(@__MODULE__)

const BACKEND = KernelAbstractions.CPU()

function leaky_adapter(initial, input, gain, pole)
    return LeakyIntegrator(
        _leaky_step!(BACKEND, VALIDATION_WORKGROUP_SIZE),
        _adopt_leaky_state!(BACKEND, PROJECTION_WORKGROUP_SIZE),
        _reset_leaky_state!(BACKEND, PROJECTION_WORKGROUP_SIZE),
        similar(input), zeros(Int32, 1), copy(initial), copy(initial), input,
        gain, pole, true, Int32(101), Int32(102),
    )
end

function delay_adapter(workspace, plan, input)
    return FrameDelay(
        _validate_projection_values!(BACKEND, VALIDATION_WORKGROUP_SIZE),
        _frame_delay_output!(BACKEND, PROJECTION_WORKGROUP_SIZE),
        _adopt_frame_delay_history!(BACKEND, PROJECTION_WORKGROUP_SIZE),
        _advance_frame_delay_head!(BACKEND, 1),
        _reset_frame_delay_history!(BACKEND, PROJECTION_WORKGROUP_SIZE),
        _reset_frame_delay_head!(BACKEND, 1),
        fill(71.0f0, length(input)), zeros(Int32, 1), copy(workspace.history),
        copy(workspace.history), Int32[workspace.head], input,
        Int32(plan.delay_frames), Int32(103),
    )
end

function pupil_prepared(rows, columns; origins=[[0, 0], [0, columns], [rows, 0], [rows, columns]])
    return prepare_algorithm(PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(
            image_size=(2rows, 2columns), pupil_size=(rows, columns), pupil_origins=origins,
            pupil_mask=Bool[mod(index, 3) != 0 for index in 1:(rows * columns)]))
end

function pupil_buffers(plan)
    regions = plan.pupil_regions
    rows, columns = regions.region_size
    linear = LinearIndices(regions.image_size)
    indices = [linear[first_row + row, first_column + column]
        for (first_row, first_column) in regions.origins
        for row in 1:rows for column in 1:columns]
    return (pixels=fill(71.0f0, 4, size(plan.selected_indices, 1)),
        mean=Float32[73], status=zeros(Int32, 1), previous=zeros(Float32, 1),
        candidate=zeros(Float32, 1), indices=indices,
        selected=vec(permutedims(plan.selected_indices)))
end

function pupil_step!(buffers, plan, image)
    _pyramid_pupil_mean!(BACKEND, PYRAMID_PUPIL_LANES)(
        buffers.mean, buffers.candidate, buffers.status, image, buffers.indices,
        plan.normalization.mean_scale, Int32(PyramidPixelImageNonFiniteInput),
        Int32(PyramidPixelImageNonFiniteMean), Int32(PyramidPixelImageNonFiniteNormalizationGain);
        ndrange=PYRAMID_PUPIL_LANES)
    _pyramid_pupil_pixels!(BACKEND, PROJECTION_WORKGROUP_SIZE)(
        buffers.pixels, buffers.status, image, buffers.selected, buffers.previous;
        ndrange=length(buffers.pixels))
    _finalize_pyramid_pupil!(BACKEND, VALIDATION_WORKGROUP_SIZE)(
        buffers.previous, buffers.candidate, buffers.status, buffers.pixels,
        Int32(PyramidPixelImageNonFiniteOutput); ndrange=VALIDATION_WORKGROUP_SIZE)
    KernelAbstractions.synchronize(BACKEND)
    return nothing
end

@testset "portable KA kernels on CPU (no graph capture)" begin
    @test capture_support(LeakyIntegratorF32, BACKEND) isa NoCapture
    @test capture_support(FrameDelayF32, BACKEND) isa NoCapture
    @test capture_support(ShackHartmannImageF32, BACKEND) isa NoCapture
    @test capture_support(PyramidPupilImageF32, BACKEND) isa NoCapture

    @testset "identity copying and pupil ordering preserve bits" begin
        samples = Float32[0, -0.0, nextfloat(0.0f0), -nextfloat(0.0f0),
            Inf, -Inf, reinterpret(Float32, UInt32(0x7fc00001)), -3]
        input = [samples[mod1(index, length(samples))] for index in 1:513]
        original = copy(input)
        output = similar(input)
        status = zeros(Int32, 1)
        copy! = _identity_projection!(BACKEND, PROJECTION_WORKGROUP_SIZE)
        copy!(output, status, input; ndrange=length(output))
        KernelAbstractions.synchronize(BACKEND)
        @test reinterpret(UInt32, output) == reinterpret(UInt32, original)
        @test reinterpret(UInt32, input) == reinterpret(UInt32, original)
        copy!(input, status, input; ndrange=length(input))
        KernelAbstractions.synchronize(BACKEND)
        @test reinterpret(UInt32, input) == reinterpret(UInt32, original)
        status[1] = 9
        fill!(output, 71)
        copy!(output, status, input; ndrange=length(output))
        KernelAbstractions.synchronize(BACKEND)
        @test output == fill(71.0f0, length(output))

        pixels = reshape(repeat(samples, 65), 4, :)
        ordered = similar(pixels, length(pixels))
        expected = vec(permutedims(pixels))
        order! = _order_pyramid_pupil_pixels!(BACKEND, PROJECTION_WORKGROUP_SIZE)
        status[1] = 0
        order!(ordered, status, pixels; ndrange=length(ordered))
        KernelAbstractions.synchronize(BACKEND)
        @test reinterpret(UInt32, ordered) == reinterpret(UInt32, expected)
        status[1] = 11
        fill!(ordered, 73)
        order!(ordered, status, pixels; ndrange=length(ordered))
        KernelAbstractions.synchronize(BACKEND)
        @test ordered == fill(73.0f0, length(ordered))
    end

    @testset "pupil mean tree, previous-frame gain and reset" begin
        # More than 256 samples, masked pixels and a repeated region exercise
        # strided loads and overlap multiplicity in the full-pupil mean.
        prepared = pupil_prepared(17, 19;
            origins=[[0, 0], [0, 19], [17, 0], [0, 0]])
        buffers = pupil_buffers(prepared.plan)
        expected = (similar(buffers.pixels), similar(buffers.mean))
        base = reshape(Float32[mod(index, 23) / 4 for index in 1:(34 * 38)], 34, 38)
        for image in (base, 2 .* base, zeros(Float32, size(base)), -ones(Float32, size(base)))
            # Force the existing logical-order scalar CPU path. Dyadic inputs
            # make successful sums exact before the shared Float32 mean scale.
            oracle_image = PermutedDimsArray(permutedims(image), (2, 1))
            @test process!(expected..., prepared, oracle_image) === nothing
            fill!(buffers.status, 0)
            pupil_step!(buffers, prepared.plan, image)
            @test buffers.status == Int32[0]
            @test buffers.pixels == expected[1]
            @test buffers.mean == expected[2]
            @test only(buffers.previous) == pyramid_normalization_gain(prepared.workspace)
        end
        buffers.previous[1], buffers.candidate[1] = 3, 7
        _reset_pyramid_pupil_gain!(BACKEND, 1)(buffers.previous, buffers.candidate; ndrange=1)
        reset!(prepared)
        KernelAbstractions.synchronize(BACKEND)
        @test buffers.previous == buffers.candidate == Float32[0]
        pupil_step!(buffers, prepared.plan, base)
        @test process!(expected..., prepared, base) === nothing
        @test buffers.pixels == expected[1] == zeros(Float32, size(buffers.pixels))
        @test only(buffers.previous) == pyramid_normalization_gain(prepared.workspace)
    end

    @testset "pupil exceptional rejection priority and held gain" begin
        prepared = pupil_prepared(1, 2)
        buffers = pupil_buffers(prepared.plan)
        expected = (fill(71.0f0, size(buffers.pixels)), Float32[73])
        maximum = floatmax(Float32)
        # Write in the actual pupil-major order, independently of detector layout.
        cases = (
            (Float32[maximum, maximum, NaN, 0, 0, 0, 0, 0], PyramidPixelImageNonFiniteMean),
            (Float32[NaN, maximum, maximum, 0, 0, 0, 0, 0], PyramidPixelImageNonFiniteInput),
            (Float32[maximum, -maximum, maximum, -maximum, 0, 0, 0, 0], nothing),
            (fill(nextfloat(0.0f0), 8), PyramidPixelImageNonFiniteNormalizationGain),
        )
        for (samples, failure) in cases
            reset!(prepared)
            fill!(buffers.previous, 0)
            fill!(buffers.candidate, 0)
            fill!(buffers.status, 0)
            fill!(buffers.pixels, 71)
            fill!(buffers.mean, 73)
            image = zeros(Float32, prepared.plan.pupil_regions.image_size)
            image[buffers.indices] = samples
            oracle_image = PermutedDimsArray(permutedims(image), (2, 1))
            @test process!(expected..., prepared, oracle_image) === failure
            pupil_step!(buffers, prepared.plan, image)
            @test only(buffers.status) == (isnothing(failure) ? 0 : Int32(failure))
            @test only(buffers.previous) == pyramid_normalization_gain(prepared.workspace)
            if failure === nothing
                @test buffers.pixels == expected[1]
                @test buffers.mean == expected[2]
            else
                @test buffers.pixels == fill(71.0f0, size(buffers.pixels))
                @test buffers.mean == Float32[73]
                @test buffers.candidate == Float32[0]
            end
        end

        # Output overflow occurs after a successful mean. Kernel outputs are
        # graph-owned candidates; only gain adoption and status are atomic here.
        image = fill(2.0f0, prepared.plan.pupil_regions.image_size)
        prepared.workspace.reciprocal_previous_frame_mean[1] = maximum
        buffers.previous[1] = maximum
        fill!(buffers.status, 0)
        @test process!(expected..., prepared, image) === PyramidPixelImageNonFiniteOutput
        pupil_step!(buffers, prepared.plan, image)
        @test buffers.status == Int32[Int32(PyramidPixelImageNonFiniteOutput)]
        @test only(buffers.previous) == pyramid_normalization_gain(prepared.workspace) == maximum

        buffers.status[1] = 37
        before = map(copy, (buffers.pixels, buffers.mean, buffers.previous, buffers.candidate))
        pupil_step!(buffers, prepared.plan, image)
        @test all(isequal.(buffers.pixels, before[1]))
        @test buffers.mean == before[2]
        @test buffers.previous == before[3]
        @test buffers.candidate == before[4]
        @test buffers.status == Int32[37]
    end

    @testset "projection reduction trees and held output" begin
        for T in (Float32, Float64), lanes in (32, 64, 128, 256), extent in (1, 33, 513)
            # A high-precision BLAS-independent oracle also exposes omissions at
            # tile boundaries without reproducing the kernel's reduction tree.
            matrix = T[sin(row + column / 7) / 16 for row in 1:3, column in 1:extent]
            input = T[cos(column / 5) for column in 1:extent]
            expected = BigFloat.(matrix) * BigFloat.(input)
            output = fill(T(71), 3)
            status = zeros(Int32, 1)
            projection = KAProjection(
                _transposed_projection!(BACKEND, (lanes, 1)),
                permutedims(matrix), Val(lanes),
            )
            _enqueue_projection!(projection, output, status, input)
            KernelAbstractions.synchronize(BACKEND)
            @test output ≈ T.(expected) rtol=32eps(T) atol=32eps(T)

            fill!(output, T(71))
            status[1] = 19
            _enqueue_projection!(projection, output, status, input)
            KernelAbstractions.synchronize(BACKEND)
            @test output == fill(T(71), 3)
            @test status == Int32[19]
        end
        @test_throws ArgumentError _require_reduction_lanes(48)
    end

    @testset "finite scan includes every lane and strided tail" begin
        validate! = _validate_projection_values!(BACKEND, VALIDATION_WORKGROUP_SIZE)
        values = ones(Float32, 513)
        roles = Int32[2, 7, 3]
        for bad_index in (nothing, 1, 32, 128, 129, 256, 257, 512, 513)
            fill!(values, 1.0f0)
            bad_index === nothing || (values[bad_index] = Float32(NaN))
            status = zeros(Int32, 1)
            role_output = fill(Int32(-1), 3)
            validate!(status, values, Int32(17), role_output, roles;
                ndrange=VALIDATION_WORKGROUP_SIZE)
            KernelAbstractions.synchronize(BACKEND)
            valid = all(isfinite, values)
            @test status == Int32[valid ? 0 : 17]
            @test role_output == (valid ? roles : fill(Int32(-1), 3))
        end
        status = Int32[23]
        role_output = fill(Int32(-1), 3)
        validate!(status, values, Int32(17), role_output, roles;
            ndrange=VALIDATION_WORKGROUP_SIZE)
        KernelAbstractions.synchronize(BACKEND)
        @test status == Int32[23]
        @test role_output == fill(Int32(-1), 3)
    end

    @testset "leaky recurrence, failure precedence, state adoption and reset" begin
        for T in (Float32, Float64)
            initial = T[sin(index / 9) for index in 1:513]
            input = T[cos(index / 13) for index in 1:513]
            plan = LeakyIntegratorPlan(T(0.3), T(0.7))
            workspace = LeakyIntegratorWorkspace(initial)
            expected = similar(input)
            captured = leaky_adapter(initial, input, plan.gain, plan.pole)
            @test !Base.mightalias(captured.committed_state, initial)
            @test !Base.mightalias(captured.initial_state, captured.committed_state)
            @test !Base.mightalias(captured.output, captured.committed_state)
            # The maintained portable source retains separate arithmetic. Keep
            # its exact oracle, and compare the new CPU FMA recurrence with a
            # propagated rounding bound instead of asserting backend bit identity.
            separate_state = copy(initial)
            rounding_bound = zeros(T, length(input))
            for _ in 1:3
                state_terms = plan.pole .* separate_state
                input_terms = plan.gain .* input
                separate_state = state_terms .+ input_terms
                rounding_bound = abs(plan.pole) * (one(T) + eps(T)) .* rounding_bound .+
                    T(8) * eps(T) .* (abs.(state_terms) .+ abs.(input_terms))
                @test process!(expected, workspace, plan, input) === expected
                commit!(workspace)
                enqueue!(captured)
                KernelAbstractions.synchronize(BACKEND)
                @test complete!(captured, only(captured.frame_status)) === nothing
                @test isequal(captured.output, separate_state)
                @test isequal(captured.committed_state, separate_state)
                @test all(abs.(captured.output .- expected) .<= rounding_bound)
                @test all(abs.(captured.committed_state .- leaky_integrator_state(workspace)) .<= rounding_bound)
            end
            reset!(captured)
            reset!(workspace)
            KernelAbstractions.synchronize(BACKEND)
            @test captured.committed_state == initial
            @test captured.output == initial

            # The graph owns candidate output: failed work may modify it, but
            # must never adopt it as committed state. The CPU API is the failure
            # and state oracle, with an input failure taking priority over overflow.
            overflow_plan = LeakyIntegratorPlan(floatmax(T), one(T))
            captured = leaky_adapter(initial, input, overflow_plan.gain, overflow_plan.pole)
            for nonfinite in (T(NaN), T(Inf), T(-Inf), nothing)
                fill!(input, T(2))
                nonfinite === nothing || (input[end - 1] = nonfinite)
                expected_failure = process!(expected, workspace, overflow_plan, input)
                fill!(captured.frame_status, 0)
                enqueue!(captured)
                KernelAbstractions.synchronize(BACKEND)
                @test complete!(captured, only(captured.frame_status)) === expected_failure
                @test captured.committed_state == leaky_integrator_state(workspace) == initial
            end
            fill!(captured.output, T(71))
            captured.frame_status[1] = 29
            enqueue!(captured)
            KernelAbstractions.synchronize(BACKEND)
            @test captured.output == fill(T(71), length(input))
            @test captured.committed_state == initial
            @test captured.frame_status == Int32[29]
        end
    end

    @testset "frame delay ring ordering, rejected frames and reset" begin
        for delay in (0, 1, 3)
            initial = Float32.(1:513)
            plan = FrameDelayPlan(delay)
            workspace = FrameDelayWorkspace(initial, plan)
            input = similar(initial)
            expected = fill(71.0f0, length(input))
            captured = delay_adapter(workspace, plan, input)
            @test !Base.mightalias(captured.history, workspace.history)
            @test !Base.mightalias(captured.history, captured.initial_history)
            for frame in 1:9
                fill!(input, Float32(frame))
                frame in (2, 5) && (input[end] = Float32(NaN))
                fill!(captured.frame_status, 0)
                result = process!(expected, workspace, plan, input)
                result === expected && commit!(workspace)
                enqueue!(captured)
                KernelAbstractions.synchronize(BACKEND)
                @test complete!(captured, only(captured.frame_status)) ===
                      (result === expected ? nothing : result)
                @test captured.output == expected
                @test captured.history == frame_delay_history(workspace)
                @test only(captured.head) == frame_delay_head(workspace)
            end
            reset!(captured)
            reset!(workspace)
            KernelAbstractions.synchronize(BACKEND)
            @test captured.history == workspace.history
            @test only(captured.head) == workspace.head == 1
        end
    end

    @testset "Shack-Hartmann moment tree matches serial measurement" begin
        # 65 rows forces each lane to handle more than one pixel row. Distinct
        # regions exercise workgroup isolation; zero weights, inactive regions,
        # threshold equality and a non-finite pixel exercise validity semantics.
        rows, columns, count = 65, 3, 3
        plan = ShackHartmannImagePlan(
            (rows, count * columns), (rows, columns), [(0, 0), (0, 3), (0, 6)],
            0.5f0, 2.0f0, 4.0f0;
            reference_slopes=Float32[0.25, -0.5, -0.75, 0.25, 0, 0],
            active=Bool[true, false, true],
        )
        image = Float32[mod(row + 2column, 13) for row in 1:rows, column in 1:(count * columns)]
        weights = Float32[mod(region + pixel, 4) / 2 for region in 1:count, pixel in 1:(rows * columns)]
        weights[count, end] = 0.0f0
        weighted = ShackHartmannWeightedCentroidPlan(plan, weights)
        measure! = _shack_hartmann_image!(BACKEND, SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE)
        for pixel_weights in (nothing, weighted.weights), sample in (:finite, :dark, :nonfinite)
            frame = copy(image)
            sample === :dark && fill!(frame, 0)
            sample === :nonfinite && (frame[65, 9] = Float32(NaN))
            slopes, flux, validity = zeros(Float32, count, 2), zeros(Float32, count), fill(false, count)
            expected = (similar(slopes), similar(flux), similar(validity))
            if pixel_weights === nothing
                # This wrapper forces the existing scalar CPU measurement path.
                oracle_frame = PermutedDimsArray(permutedims(frame), (2, 1))
                FilterGraphAlgorithms._measure_shack_hartmann_range!(expected..., plan, oracle_frame, 1:count)
            else
                FilterGraphAlgorithms._measure_shack_hartmann_weighted_centroid!(expected..., weighted, frame)
            end
            status = zeros(Int32, 1)
            args = (slopes, flux, validity, status, frame,
                first.(plan.regions.origins), last.(plan.regions.origins),
                plan.x_coordinates, plan.y_coordinates,
                plan.reference_slopes[1:2:end], plan.reference_slopes[2:2:end],
                plan.pixel_thresholds, plan.flux_thresholds, plan.active,
                pixel_weights, rows, columns, nothing)
            measure!(args...; ndrange=(SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE, count))
            KernelAbstractions.synchronize(BACKEND)
            @test slopes ≈ expected[1] rtol=32eps(Float32) atol=32eps(Float32)
            @test all(isequal.(flux, expected[2]))
            @test validity == expected[3]
            @test status == Int32[0]
            before = map(copy, (slopes, flux, validity))
            status[1] = 31
            measure!(args...; ndrange=(SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE, count))
            KernelAbstractions.synchronize(BACKEND)
            @test all(isequal.(slopes, before[1]))
            @test all(isequal.(flux, before[2]))
            @test validity == before[3]
        end
    end
end

const TEST_DEVICE = TestDevice()
_capture_array(::TestDevice, values) = copy(values)
_capture_zeros(::TestDevice, ::Type{T}, dimensions...) where {T} = zeros(T, dimensions...)
_require_capture_array(name, array::Array, ::TestDevice) = nothing
_require_capture_workgroup(::TestDevice, lanes::Integer; shared_memory::Integer=0) = nothing

@testset "shared controller preparation does not qualify CPU validation reuse" begin
    prepared = prepare_algorithm(ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(state_length=5, hidden_mode_count=2))
    input, feedback, correction, state = ntuple(_ -> zeros(Float32, 5), 4)
    status = zeros(Int32, 1)
    # Preparation only: identity is not a CPU workgroup collective.
    adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
        (correction, state), (input, feedback), status, Int32(1), identity)
    @test capture_support(ClosedLoopCorrectionF32, BACKEND) isa NoCapture
    @test !FilterGraphAlgorithms.preserves_inputs(adapter)
    @test !FilterGraphAlgorithms.finite_output(adapter, Val(1))
    @test FilterGraphAlgorithms.prepare_validation(adapter, (true, true)) === adapter
    @test adapter.kernels.vote === identity
    @test !Base.mightalias(adapter.vectors.committed_state, prepared.workspace.committed_state)
    @test !Base.mightalias(adapter.vectors.committed_output, prepared.workspace.committed_output)
    @test_throws DimensionMismatch _require_expansion(zeros(Float32, 2, 1), zeros(Float32, 3), zeros(Float32, 1))
    @test_throws ArgumentError _require_expansion(zeros(Float64, 2, 1), zeros(Float32, 2), zeros(Float32, 1))
end

@testset "unqualified pixel capture does not construct a certificate" begin
    prepared = prepare_algorithm(PixelCalibrationU16F32,
        PixelCalibrationU16F32Configuration(image_size=(2, 3),
            initial_flat=nothing, initial_background=nothing))
    input, output = zeros(UInt16, 2, 3), zeros(Float32, 2, 3)
    adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
        (output,), (input,), zeros(Int32, 1), Int32(1))
    @test adapter.bounds === nothing
    @test _capture_output_bound(adapter, Val(1)) === nothing
end

@testset "shared pupil preparation and state on private CPU test device" begin
    prepared = pupil_prepared(2, 3)
    buffers = pupil_buffers(prepared.plan)
    image = fill(2.0f0, prepared.plan.pupil_regions.image_size)
    outputs = (buffers.pixels, buffers.mean)
    expected = map(copy, outputs)
    adapter = _prepare_capture(CompleteTransition(), TEST_DEVICE, prepared,
        outputs, (image,), buffers.status, Int32(1))
    @test !FilterGraphAlgorithms.preserves_inputs(adapter)
    @test !FilterGraphAlgorithms.finite_output(adapter, Val(1))
    @test _prepare_bounded_validation(adapter, (_PixelBounds(1.0, 0),)) === adapter
    @test isnothing(_capture_output_bound(adapter, Val(1)))
    reconstruction = prepare_algorithm(PyramidPupilReconstructorF32,
        PyramidPupilReconstructorF32Configuration(reconstructed_count=2,
            selected_pixel_count=size(buffers.pixels, 2),
            initial_reconstructor=zeros(Float32, 2length(buffers.pixels)),
            reconstructed_schema="org.example.pupil/1"))
    reconstructor = _prepare_capture(CompleteTransition(), TEST_DEVICE, reconstruction,
        (zeros(Float32, 2),), (buffers.pixels,), zeros(Int32, 1), Int32(2))
    @test FilterGraphAlgorithms.prepare_validation(reconstructor, (true,)) === reconstructor
    @test _prepare_bounded_validation(reconstructor, (_FiniteBound(1.0),)) === reconstructor
    for frame in 1:3
        fill!(image, Float32(2frame))
        @test process!(expected..., prepared, image) === nothing
        enqueue!(adapter)
        KernelAbstractions.synchronize(BACKEND)
        @test complete!(adapter, only(buffers.status)) === nothing
        @test outputs == expected
        @test only(adapter.previous_gain) == pyramid_normalization_gain(prepared.workspace)
    end
    gain = copy(adapter.previous_gain)
    image[1] = NaN32
    enqueue!(adapter)
    KernelAbstractions.synchronize(BACKEND)
    @test complete!(adapter, only(buffers.status)) === PyramidPixelImageNonFiniteInput
    @test adapter.previous_gain == gain
    reset!(adapter)
    KernelAbstractions.synchronize(BACKEND)
    @test iszero(only(adapter.previous_gain))
end

@testset "portable KA additional domains" begin
    @test capture_support(LeakyIntegratorF32, TEST_DEVICE) isa NoCapture
    include("ka_cpu_projection.jl")
    include("ka_cpu_frontends.jl")
    include("ka_cpu_eager.jl")
    include("ka_cpu_deformable_mirror.jl")
    include("ka_cpu_reconstruction.jl")
end

end
