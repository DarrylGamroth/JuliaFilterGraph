module ReadoutMeasurementCaptureTests

using CUDA, FilterGraphAlgorithms, KernelAbstractions, Test

const EXTENSION = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)

struct MeasurementPacket
    value::Float32
    column::UInt32
end

function finish_buffers(values; flags = Int32[], scale = inv(Float32(length(values))))
    detector = CUDA.CuArray(values)
    return (;
        detector,
        indices = CUDA.CuArray(Int32.(eachindex(values))),
        packet = CUDA.fill(MeasurementPacket(0.0f0, UInt32(0)), length(flags) + 1),
        flags = CUDA.CuArray(flags),
        status = CUDA.fill(Int32(71), 1),
        gain = CUDA.fill(19.0f0, 1),
        mean = CUDA.fill(23.0f0, 1),
        scale,
        kernels = EXTENSION._readout_kernels(KernelAbstractions.get_backend(detector))
    )
end

function finish!(buffers; terminal = true)
    buffers.kernels.finish!(
        buffers.packet, buffers.status, buffers.flags, buffers.detector, buffers.indices,
        buffers.scale, buffers.gain, buffers.mean, terminal; ndrange = 256
    )
    CUDA.synchronize()
    return only(Array(buffers.status))
end

@testset "parallel readout normalization uses every pupil pixel" begin
    cases = (
        vcat(fill(1.0f-20, 511), fill(1.0f20, 513)),
        vcat(fill(10.0f0, 257), fill(-1.0f0, 255)),
        vcat(fill(-10.0f0, 257), fill(1.0f0, 255)),
        vcat(ones(Float32, 256), -ones(Float32, 256))
    )
    for values in cases
        buffers = finish_buffers(values; flags = zeros(Int32, 3))
        @test finish!(buffers) == 0
        expected = Float32(sum(Float64, values) * Float64(buffers.scale))
        @test only(Array(buffers.mean))≈expected rtol=8eps(Float32) atol=0
        gain = expected > 0 ? inv(expected) : 0.0f0
        @test only(Array(buffers.gain))≈gain rtol=8eps(Float32) atol=0
        @test first(Array(buffers.packet)).column == 3
    end
    # Empty readiness groups still clear stale status and publish an empty packet.
    buffers = finish_buffers(ones(Float32, 513))
    @test finish!(buffers; terminal = false) == 0
    @test first(Array(buffers.packet)).column == 0
    @test only(Array(buffers.mean)) == 23.0f0
    @test only(Array(buffers.gain)) == 19.0f0
    @test finish!(buffers) == 0
    @test only(Array(buffers.mean)) == 1.0f0
    @test only(Array(buffers.gain)) == 1.0f0

    # Qualified cancellation policy: the fixed GPU tree can change the sign
    # relative to serial Float32 addition, and therefore the next-frame gain.
    values = zeros(Float32, 256)
    values[1], values[2], values[129], values[130] = 2.0f0^25, -1.0f0, -2.0f0^25, 0.5f0
    @test foldl(+, values; init = 0.0f0) == 0.5f0
    buffers = finish_buffers(values)
    @test finish!(buffers) == 0
    @test only(Array(buffers.mean)) == -0.5f0 / 256
    @test only(Array(buffers.gain)) == 0.0f0
end

@testset "parallel readout normalization reports numerical failures" begin
    for invalid in (NaN32, Inf32, -Inf32)
        values = ones(Float32, 513)
        values[513] = invalid # Excluded from the selected-pixel packet, included in mean.
        buffers = finish_buffers(values; flags = zeros(Int32, 1))
        @test finish!(buffers) == 3
        @test only(Array(buffers.gain)) == 19.0f0
        @test only(Array(buffers.mean)) == 23.0f0
    end
    for indices in ((1, 257), (1, 2))
        values = zeros(Float32, 512)
        values[collect(indices)] .= floatmax(Float32)
        buffers = finish_buffers(values)
        @test finish!(buffers) == 4 # Lane accumulation, then cross-lane tree overflow.
        @test only(Array(buffers.gain)) == 19.0f0
    end
    values = zeros(Float32, 513)
    values[1] = values[257] = floatmax(Float32)
    values[513] = NaN32
    buffers = finish_buffers(values)
    @test finish!(buffers) == 3 # Nonfinite input takes priority over lane overflow.
    buffers = finish_buffers(fill(nextfloat(0.0f0), 256))
    @test finish!(buffers) == 5 # Positive subnormal mean has nonfinite reciprocal.
    @test only(Array(buffers.gain)) == 19.0f0

    flags = zeros(Int32, 1025)
    flags[257], flags[512] = 6, 3
    buffers = finish_buffers(ones(Float32, 513); flags)
    @test finish!(buffers; terminal = false) == 6
    @test first(Array(buffers.packet)).column == 1025
    copyto!(buffers.detector, fill(NaN32, 513))
    @test finish!(buffers) == 6 # Earlier measurement failure precedes terminal mean.
    flags[1] = 3
    copyto!(buffers.flags, flags)
    @test finish!(buffers) == 3
end

@testset "parallel readout normalization retains previous-frame gain adoption" begin
    buffers = finish_buffers(fill(2.0f0, 256))
    committed = CUDA.zeros(Float32, 1)
    global_status = CUDA.fill(Int32(1), 1)
    @test finish!(buffers) == 0
    buffers.kernels.adopt!(committed, buffers.gain, global_status; ndrange = 1)
    CUDA.synchronize()
    @test only(Array(committed)) == 0.0f0
    fill!(global_status, Int32(0))
    buffers.kernels.adopt!(committed, buffers.gain, global_status; ndrange = 1)
    CUDA.synchronize()
    @test only(Array(committed)) == 0.5f0

    pixels = CUDA.CuArray(Int32[1, 2])
    flags = CUDA.zeros(Int32, 2)
    packet = CUDA.fill(MeasurementPacket(0.0f0, UInt32(0)), 3)
    buffers.kernels.pixels!(
        packet, flags, buffers.detector, pixels, pixels, committed; ndrange = 2)
    CUDA.synchronize()
    @test [entry.value for entry in Array(packet)[2:3]] == Float32[1, 1]
    @test all(iszero, Array(flags))

    fill!(committed, 0.0f0) # Reset seed; selected nonfinite pixels must still fail.
    values = fill(2.0f0, 256)
    values[1] = NaN32
    copyto!(buffers.detector, values)
    buffers.kernels.pixels!(
        packet, flags, buffers.detector, pixels, pixels, committed; ndrange = 2)
    CUDA.synchronize()
    @test Array(flags) == Int32[3, 0]
    @test Array(packet)[3].value == 0.0f0
end

function centroid_buffers(rows, columns)
    detector = [Float32(20 + mod(7row + 3column, 31))
                for row in 1:(rows + 2), column in 1:(columns + 3)]
    origins = Int32[0 1 2 0; 0 2 1 3]
    ready = Int32[3, 1, 4, 2] # Packet order differs from scientific subaperture order.
    x = Float32[column - (columns + 1) / 2 for row in 1:rows for column in 1:columns]
    y = Float32[row - (rows + 1) / 2 for row in 1:rows for column in 1:columns]
    reference = Float32[0.01i for i in 1:8]
    pixel_thresholds = Float32[20, 30, 40, 1.0f6]
    flux_thresholds = Float32[1, 1, 1, 1]
    active = Bool[true, true, false, true]
    host = (; detector, origins, ready, x, y, reference,
        pixel_thresholds, flux_thresholds, active)
    device = map(CUDA.CuArray, host)
    packet = CUDA.fill(MeasurementPacket(71.0f0, UInt32(71)), 9)
    status = CUDA.fill(Int32(71), 1)
    return (; host, device, packet, status, rows, columns)
end

function serial_centroids(buffers)
    h = buffers.host
    result = MeasurementPacket[MeasurementPacket(0.0f0, UInt32(2length(h.ready)))]
    for subaperture in h.ready
        x, y, flux = 0.0f0, 0.0f0, 0.0f0
        for row in 1:buffers.rows, column in 1:buffers.columns

            value = h.detector[h.origins[1, subaperture] + row, h.origins[2, subaperture] + column]
            coordinate = (row - 1) * buffers.columns + column
            if !isfinite(value) || value >= h.pixel_thresholds[subaperture]
                x += value * h.x[coordinate]
                y += value * h.y[coordinate]
                flux += value
            end
        end
        index = 2subaperture - 1
        x, y, _, valid = FilterGraphAlgorithms._finish_center_of_gravity(
            x, y, flux, h.reference[index], h.reference[index + 1],
            h.flux_thresholds[subaperture])
        if !valid || !h.active[subaperture]
            x = y = 0.0f0
        end
        push!(result, MeasurementPacket(x, UInt32(index - 1)), MeasurementPacket(y, UInt32(index)))
    end
    return result
end

@testset "cooperative readout centroids retain coordinates and invalid-zero policy" begin
    for (rows, columns) in ((1, 1), (2, 2), (4, 8), (6, 6), (8, 8), (22, 22), (33, 17))
        buffers = centroid_buffers(rows, columns)
        kernels = EXTENSION._readout_kernels(KernelAbstractions.get_backend(buffers.device.detector))
        for invalid in (nothing, NaN32, Inf32, -Inf32, floatmax(Float32))
            buffers.host.detector[1, 1] = isnothing(invalid) ? 20.0f0 : invalid
            copyto!(buffers.device.detector, buffers.host.detector)
            expected = serial_centroids(buffers)
            for cooperative in (false, true)
                kernel! = cooperative ? kernels.shack_hartmann_cooperative! :
                          kernels.shack_hartmann!
                d = buffers.device
                kernel!(buffers.packet, buffers.status, d.detector, d.ready, d.origins,
                    d.x, d.y, d.reference, d.pixel_thresholds, d.flux_thresholds, d.active,
                    rows, columns;
                    ndrange = cooperative ? (32, length(d.ready)) : length(d.ready))
                CUDA.synchronize()
                actual = Array(buffers.packet)
                @test only(Array(buffers.status)) == 0
                @test getfield.(actual, :column) == getfield.(expected, :column)
                @test getfield.(actual, :value)≈getfield.(expected, :value) rtol=3.0f-5 atol=5.0f-6
            end
        end
        # Empty completion groups must still replace a stale count/status.
        d = buffers.device
        kernels.shack_hartmann_cooperative!(buffers.packet, buffers.status, d.detector,
            CUDA.CuArray(Int32[]), d.origins, d.x, d.y, d.reference, d.pixel_thresholds,
            d.flux_thresholds, d.active, rows, columns; ndrange = (32, 1))
        CUDA.synchronize()
        @test first(Array(buffers.packet)).column == 0
        @test only(Array(buffers.status)) == 0
    end
end

end
