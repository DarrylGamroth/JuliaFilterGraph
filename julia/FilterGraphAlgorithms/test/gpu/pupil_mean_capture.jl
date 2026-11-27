module PupilMeanCaptureTests

using CUDA, FilterGraphAlgorithms, Test
const CUDAX = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)

function pupil_case(ir, ic, rows, columns, packed; overlap=false, sparse=false)
    origins = overlap ? [[0, 0] for _ in 1:4] :
        [[0, 0], [ir - rows, 0], [0, ic - columns], [ir - rows, ic - columns]]
    mask = sparse ? [i == 1 for i in 1:rows*columns] :
        [!iszero(mod(i, 3)) for i in 1:rows*columns]
    prepared = prepare_algorithm(PyramidPupilImageF32, PyramidPupilImageF32Configuration(
        image_size=(ir, ic), pupil_size=(rows, columns),
        pupil_origins=origins, pupil_mask=mask))
    count = sum(mask)
    output = packed ? PermutedDimsArray(CUDA.zeros(Float32, count, 4), (2, 1)) :
        CUDA.zeros(Float32, 4, count)
    return FilterGraphAlgorithms.prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output, CUDA.zeros(Float32, 1)), (CUDA.zeros(Float32, ir, ic),),
        CUDA.zeros(Int32, 1), Int32(2))
end

same_bits(a::Float32, b::Float32) = reinterpret(UInt32, a) == reinterpret(UInt32, b)
same_bits(a::Array{Float32}, b::Array{Float32}) =
    size(a) == size(b) && reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b))

function snapshot(c)
    return (; pixels=Array(CUDAX._pupil_pixel_storage(c.output)),
        mean=only(Array(c.mean_output)), previous=only(Array(c.previous_gain)),
        candidate=only(Array(c.candidate_gain)), status=only(Array(c.frame_status)))
end

function mean_reference(image, indices, scale; serial=false)
    return setprecision(BigFloat, 384) do
        total, magnitude = BigFloat(0), BigFloat(0)
        for index in indices
            value = BigFloat(image[index])
            total += value
            magnitude += abs(value)
        end
        count = length(indices)
        # At most ceil(N/256) lane additions and eight binary reduction levels.
        # Include one final Float32 multiplication by the stored mean_scale.
        depth = (serial ? count : cld(count, 256) + 8) + 1
        unit = BigFloat(2)^(-24)
        gamma = depth * unit / (1 - depth * unit)
        alpha = BigFloat(scale)
        # A conservative absolute allowance for gradual underflow: one smallest
        # subnormal per addition/product, propagated through the same depth.
        quantum = BigFloat(nextfloat(0.0f0))
        underflow = quantum * (alpha * (count + 255) + 1) / (1 - depth * unit)
        return (; mean=alpha * total, bound=gamma * alpha * magnitude + underflow)
    end
end

function check_frame!(a, b, image; expected=Int32(0), status=Int32(0), rescan=false)
    cases = (a, b)
    for c in cases
        copyto!(c.image, image)
        fill!(c.frame_status, status)
        fill!(CUDAX._pupil_pixel_storage(c.output), -17.0f0)
        fill!(c.mean_output, -19.0f0)
    end
    before = map(snapshot, cases)
    foreach(FilterGraphAlgorithms.enqueue!, cases)
    CUDA.synchronize()
    after = map(snapshot, cases)
    disposition = iszero(status) ? expected : status
    for (c, old, actual) in zip(cases, before, after)
        @test actual.status == disposition
        before_pixels = !iszero(status) || disposition in
            (c.input_failure, c.mean_failure, c.gain_failure)
        if before_pixels
            @test same_bits(actual.pixels, old.pixels)
            @test same_bits(actual.mean, old.mean)
            @test same_bits(actual.candidate, old.candidate)
            @test same_bits(actual.previous, old.previous)
        else
            @test disposition in (Int32(0), c.output_failure)
            @test isfinite(actual.mean)
            @test isfinite(actual.candidate)
            reference = mean_reference(image, Array(c.pupil_indices), c.mean_scale;
                serial=rescan)
            @test setprecision(BigFloat, 384) do
                abs(BigFloat(actual.mean) - reference.mean) <= reference.bound
            end
            gain = actual.mean > 0.0f0 ? inv(actual.mean) : 0.0f0
            @test same_bits(actual.candidate, gain)
            selected = Array(c.selected_indices)
            pixels = reshape(Float32[old.previous * image[index] for index in selected],
                size(old.pixels))
            @test same_bits(actual.pixels, pixels)
            if iszero(disposition)
                @test all(isfinite, actual.pixels)
                @test same_bits(actual.previous, actual.candidate)
            else
                @test any(!isfinite, actual.pixels)
                @test same_bits(actual.previous, old.previous)
            end
        end
    end
    # Exceptional successful serial rescans retain their original arithmetic;
    # ordinary reductions are qualified numerically, without cross-path bits.
    if rescan && disposition in (Int32(0), a.output_failure)
        @test same_bits(after[1].mean, after[2].mean)
    end
    return nothing
end

function reset_pair!(a, b)
    FilterGraphAlgorithms.reset!(a)
    FilterGraphAlgorithms.reset!(b)
    CUDA.synchronize()
    for c in (a, b)
        @test same_bits(only(Array(c.previous_gain)), 0.0f0)
        @test same_bits(only(Array(c.candidate_gain)), 0.0f0)
    end
end

function adversarial_frames!(a, b, image, indices)
    @testset "absolute mean stability under cancellation and exponent spread" begin
        quantum = nextfloat(0.0f0)
        patterns = ((ldexp(1.0f0, 24), 1.0f0, -ldexp(1.0f0, 24), -1.0f0),
            (ldexp(1.0f0, 100), ldexp(1.0f0, -100), -ldexp(1.0f0, 100), 0.0f0),
            (ldexp(1.0f0, 20), -ldexp(1.0f0, 20), ldexp(1.0f0, -30), -ldexp(1.0f0, -30)),
            (-0.0f0, 0.0f0, -0.0f0, 0.0f0),
            (quantum, -quantum, 2quantum, -2quantum))
        for pattern in patterns
            reset_pair!(a, b)
            fill!(image, 0.0f0)
            for (entry, index) in enumerate(indices)
                image[index] = pattern[mod1(entry, length(pattern))]
            end
            check_frame!(a, b, image)
        end
        # Stay a factor of two away from the reciprocal overflow boundary, so
        # these expected dispositions do not depend on ordinary sum rounding.
        for (value, expected) in ((ldexp(1.0f0, -129), a.gain_failure),
                (ldexp(1.0f0, -127), Int32(0)))
            reset_pair!(a, b)
            fill!(image, value)
            check_frame!(a, b, image; expected)
        end
    end
end

function run_tests(reference_mean!)
    @testset "sorted pupil load map preserves multiplicity and checks index width" begin
        indices = [Int(typemax(Int32)), 1, 17, 65537, 65537]
        mapping = CUDAX._pupil_load_map(indices)
        @test mapping == sort(Int32.(indices))
        @test eltype(mapping) == Int32
        narrow = CUDAX._pupil_load_map([65535, 1, 17, 17])
        @test narrow == [1, 17, 17, 65535]
        @test eltype(narrow) == Int32
        @test CUDAX._pupil_load_map([65536]) == Int32[65536]
        @test eltype(CUDAX._pupil_load_map([65536])) == Int32
        @test CUDAX._pupil_load_map([65537]) == Int32[65537]
        @test CUDAX._pupil_load_map([Int(typemax(Int32)) + 1]) === nothing
        @test CUDAX._pupil_load_map([0]) === nothing
        @test CUDAX._pupil_load_map([-1]) === nothing
        @test CUDAX._pupil_load_map(Int[]) === nothing
        @test CUDAX._pupil_load_map(ones(Int, 4097)) === nothing
    end
    @testset "pupil mean accuracy and exact state transitions" begin
        shapes = [(64, 64, 7, 8), (64, 64, 8, 7), (64, 64, 8, 8), (64, 64, 12, 13), (64, 64, 16, 16),
            (64, 64, 16, 17), (64, 64, 30, 30), (64, 64, 32, 32),
            (64, 64, 33, 8), (66, 66, 32, 33), (256, 256, 8, 8),
            (256, 256, 8, 128), (256, 256, 128, 8), (257, 257, 8, 8),
            (2, 1800, 1, 900)]
        for (ir, ic, rows, columns) in shapes, packed in (false, true)
            overlap = rows == columns == 30
            sparse = rows * columns > 1024
            a = pupil_case(ir, ic, rows, columns, packed; overlap, sparse)
            b = pupil_case(ir, ic, rows, columns, packed; overlap, sparse)
            backend = CUDAX.KernelAbstractions.get_backend(b.previous_gain)
            fields = ntuple(fieldcount(typeof(b))) do i
                fieldname(typeof(b), i) == :mean! ? reference_mean!(backend, 256) : getfield(b, i)
            end
            b = CUDAX.PyramidPupilImage(fields...)
            indices = Array(a.pupil_indices)
            eligible = length(indices) <= 4096 && maximum(indices) <= typemax(Int32)
            @test (a.mean! isa CUDAX._PupilMean) == eligible
            image = reshape(Float32[1.1f0 + (mod(17i, 97) - 48) * 0.03125f0
                for i in 1:ir*ic], ir, ic)
            for frame in 1:4
                image .+= Float32(frame) / 16
                check_frame!(a, b, image)
            end
            image[first(indices)] = NaN32
            check_frame!(a, b, image; status=Int32(0x301))
            for (value, expected) in ((0.0f0, Int32(0)), (-4.0f0, Int32(0)),
                    (nextfloat(0.0f0), a.gain_failure), (floatmin(Float32), Int32(0)))
                fill!(image, value)
                check_frame!(a, b, image; expected)
            end
            for input_first in (false, true)
                fill!(image, 0.0f0)
                image[indices[1]] = input_first ? NaN32 : floatmax(Float32)
                image[indices[2]] = floatmax(Float32)
                image[indices[3]] = input_first ? floatmax(Float32) : NaN32
                expected = input_first ? a.input_failure : a.mean_failure
                check_frame!(a, b, image; expected)
            end
            for (entry, index) in enumerate(indices)
                image[index] = isodd(entry) ? floatmax(Float32) : -floatmax(Float32)
            end
            check_frame!(a, b, image; expected=a.output_failure, rescan=true)
            reset_pair!(a, b)
            fill!(image, floatmin(Float32))
            check_frame!(a, b, image)
            fill!(image, 2.0f0)
            image[last(Array(a.selected_indices))] = 64.0f0
            check_frame!(a, b, image; expected=a.output_failure)
            reset_pair!(a, b)
            fill!(image, 4.0f0)
            check_frame!(a, b, image)
            selected = Set(Array(a.selected_indices))
            excluded = first(index for index in indices if !(index in selected))
            image[excluded] = Inf32
            check_frame!(a, b, image; expected=a.input_failure)
            if ir == ic == 64 && rows == columns == 8
                adversarial_frames!(a, b, image, indices)
            end
        end
    end
end

end
