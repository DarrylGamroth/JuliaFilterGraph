include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "shack_hartmann_reconstruction.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_reconstruction.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "shack_hartmann_readout_measurement_block.jl"))
include(joinpath(GPU_SOURCE, "algorithms", "wavefront_sensor", "pyramid_readout_measurement_block.jl"))

# The scientific kernels accept the packet element constructor; this plain test
# value has the same value/column representation without importing a GPU engine.
struct ReadoutEntry
    value::Float32
    column::UInt32
end

@testset "portable KA WFS reconstruction and readout" begin
    for (name, kernel!, cpu!) in (
        (:shack_hartmann, _shack_hartmann_reconstruction!, FilterGraphAlgorithms._process_shack_hartmann_reconstruction!),
        (:pyramid, _pyramid_reconstruction!, FilterGraphAlgorithms._process_pyramid_reconstruction!),
    )
        @testset "$name projection" begin
            count, lanes = 257, 128
            matrix = Float32[mod(row + column, 9) / 32 for row in 1:3, column in 1:(2count)]
            plan = DenseReconstructorPlan(matrix)
            workspace = DenseReconstructorWorkspace(plan)
            input = Float32[mod(row + component, 7) / 4 for row in 1:count, component in 1:2]
            expected, actual, status = zeros(Float32, 3), fill(71.0f0, 3), zeros(Int32, 1)
            @test cpu!(expected, workspace, plan, input) === nothing
            args = name === :shack_hartmann ?
                (permutedims(matrix[:, 1:2:end]), permutedims(matrix[:, 2:2:end]), input, Val(lanes)) :
                (permutedims(matrix), input, Val(lanes))
            launch! = kernel!(BACKEND, (lanes, 1))
            launch!(actual, status, args...; ndrange=(lanes, length(actual)))
            KernelAbstractions.synchronize(BACKEND)
            @test actual ≈ expected rtol=32eps(Float32)
            status[1] = 17
            fill!(actual, 71)
            launch!(actual, status, args...; ndrange=(lanes, length(actual)))
            KernelAbstractions.synchronize(BACKEND)
            @test actual == fill(71.0f0, 3)
        end
    end

    @testset "cooperative Shack-Hartmann packets match scalar CPU measurement" begin
        rows, columns = 65, 3
        plan = ShackHartmannImagePlan((rows, 2columns), (rows, columns), [(0, 0), (0, columns)],
            0.5f0, 2.0f0, 4.0f0; reference_slopes=Float32[0.25, -0.5, -0.75, 0.25])
        detector = Float32[mod(row + 2column, 13) for row in 1:rows, column in 1:(2columns)]
        slopes, flux, validity = zeros(Float32, 2, 2), zeros(Float32, 2), fill(false, 2)
        FilterGraphAlgorithms._measure_shack_hartmann_range!(slopes, flux, validity, plan,
            PermutedDimsArray(permutedims(detector), (2, 1)), 1:2)
        ready = Int32[2, 1]
        origins = Int32[0 0; 0 columns]
        packet, status = fill(ReadoutEntry(71, 73), 5), Int32[19]
        _shack_hartmann_readout_cooperative!(BACKEND, (32, 1))(
            packet, status, detector, ready, origins, plan.x_coordinates, plan.y_coordinates,
            plan.reference_slopes, plan.pixel_thresholds, plan.flux_thresholds, plan.active,
            rows, columns, nothing; ndrange=(32, length(ready)))
        KernelAbstractions.synchronize(BACKEND)
        @test status == Int32[0]
        @test packet[1].column == 4
        @test [entry.value for entry in packet[2:end]] ≈ vec(permutedims(slopes[ready, :])) rtol=32eps(Float32)
        @test [entry.column for entry in packet[2:end]] == UInt32[2, 3, 0, 1]
    end

    @testset "pyramid readout terminal mean and failure ordering" begin
        detector = reshape(Float32[mod(index, 17) / 4 for index in 1:516], 2, :)
        indices = collect(1:length(detector))
        gain, candidate, mean = Float32[0.5], Float32[0], Float32[73]
        columns = Int32.(indices)
        flags, status = zeros(Int32, length(indices)), Int32[19]
        packet = fill(ReadoutEntry(71, 73), length(indices) + 1)
        _pyramid_readout_pixels!(BACKEND, PROJECTION_WORKGROUP_SIZE)(
            packet, flags, detector, indices, columns, gain; ndrange=length(indices))
        _finish_pyramid_readout!(BACKEND, PYRAMID_READOUT_LANES)(
            packet, status, flags, detector, indices, inv(Float32(length(indices))),
            candidate, mean, true; ndrange=PYRAMID_READOUT_LANES)
        KernelAbstractions.synchronize(BACKEND)
        expected_mean = Float32(sum(BigFloat.(detector))) * inv(Float32(length(indices)))
        @test status == Int32[0]
        @test mean == Float32[expected_mean]
        @test candidate == Float32[inv(expected_mean)]
        @test [entry.value for entry in packet[2:end]] == vec(detector) .* only(gain)
        @test [entry.column for entry in packet[2:end]] == UInt32.(0:(length(indices) - 1))
        flags[1], flags[end] = PYRAMID_ROW_OUTPUT_FAILURE, PYRAMID_ROW_INPUT_FAILURE
        _finish_pyramid_readout!(BACKEND, PYRAMID_READOUT_LANES)(
            packet, status, flags, detector, indices, inv(Float32(length(indices))),
            candidate, mean, true; ndrange=PYRAMID_READOUT_LANES)
        KernelAbstractions.synchronize(BACKEND)
        @test status == Int32[PYRAMID_ROW_OUTPUT_FAILURE]
    end
end
