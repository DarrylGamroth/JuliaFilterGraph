module ValidationCaptureTests

using CUDA, FilterGraphAlgorithms, KernelAbstractions, Test

const CUDAX = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)

function validation_arrays(::Type{T}, count, role_count) where {T}
    values = CUDA.zeros(T, count)
    status = CUDA.zeros(Int32, 1)
    roles = CUDA.CuArray(Int32[mod(index, 3) for index in 1:role_count])
    role_output = CUDA.fill(Int32(-17), role_count)
    return (; values, status, roles, role_output)
end

function validate!(kernel, arrays, roles)
    if roles
        kernel(arrays.status, arrays.values, Int32(513), arrays.role_output,
            arrays.roles; ndrange = 256)
    else
        kernel(arrays.status, arrays.values, Int32(513); ndrange = 256)
    end
    return nothing
end

function check_validation!(kernel, arrays, values, prior, roles)
    copyto!(arrays.values, values)
    fill!(arrays.status, prior)
    fill!(arrays.role_output, Int32(-17))
    validate!(kernel, arrays, roles)
    CUDA.synchronize()
    valid = iszero(prior) && all(isfinite, values)
    expected_status = iszero(prior) && !valid ? Int32(513) : prior
    @test only(Array(arrays.status)) == expected_status
    expected_roles = roles && valid ? Array(arrays.roles) :
                     fill(Int32(-17), length(arrays.role_output))
    @test Array(arrays.role_output) == expected_roles
    @test reinterpret(bit_type(eltype(values)), Array(arrays.values)) ==
          reinterpret(bit_type(eltype(values)), values)
end

# Float types have matching unsigned bit representations for exact-copy checks.
bit_type(::Type{Float32}) = UInt32
bit_type(::Type{Float64}) = UInt64

@testset "CUDA validation warp/workgroup boundaries" begin
    backend = KernelAbstractions.get_backend(CUDA.zeros(Float32, 1))
    kernel = CUDAX._validate_projection_values!(backend, 256)
    @test isconcretetype(typeof(kernel))
    for T in (Float32, Float64),
        count in (0, 1, 31, 32, 33, 253, 255, 256, 257, 277, 3600, 8193)

        arrays = validation_arrays(T, count, 513)
        edge_values = (zero(T), -zero(T), nextfloat(zero(T)), -nextfloat(zero(T)),
            floatmax(T), -floatmax(T), one(T), -one(T))
        finite = T[edge_values[mod1(index, length(edge_values))] for index in 1:count]
        for roles in (false, true)
            check_validation!(kernel, arrays, finite, Int32(0), roles)
            check_validation!(kernel, arrays, finite, Int32(257), roles)
            indices = unique(filter(index -> 1 <= index <= count,
                [1, 31, collect(32:32:256)..., collect(33:32:257)..., 512, 513, count]))
            for index in indices, bad in (T(NaN), T(Inf), T(-Inf))

                values = copy(finite)
                values[index] = bad
                check_validation!(kernel, arrays, values, Int32(0), roles)
            end
            if count > 0
                values = fill(T(NaN), count)
                check_validation!(kernel, arrays, values, Int32(0), roles)
                check_validation!(kernel, arrays, values, Int32(257), roles)
            end
        end
    end
end

replay_bytes(executable) = @allocated CUDA.launch(executable)

@testset "CUDA validation captured replay, failure and recovery" begin
    arrays = validation_arrays(Float32, 277, 513)
    kernel = CUDAX._validate_projection_values!(
        KernelAbstractions.get_backend(arrays.values), 256)
    step! = () -> validate!(kernel, arrays, true)
    step!()
    CUDA.synchronize()
    executable = CUDA.instantiate(CUDA.capture(step!))
    finite = fill(1.0f0, 277)
    for (prior, bad) in ((Int32(0), true), (Int32(257), false), (Int32(0), false))
        values = copy(finite)
        bad && (values[end] = NaN32)
        copyto!(arrays.values, values)
        fill!(arrays.status, prior)
        fill!(arrays.role_output, Int32(-17))
        CUDA.launch(executable)
        CUDA.synchronize()
        @test only(Array(arrays.status)) == (bad ? Int32(513) : prior)
        @test Array(arrays.role_output) ==
              (!bad && iszero(prior) ? Array(arrays.roles) : fill(Int32(-17), 513))
    end
    replay_bytes(executable)
    CUDA.synchronize()
    @test replay_bytes(executable) == 0
    CUDA.synchronize()
end

@testset "CUDA status-only validation preserves the first failure in captured replay" begin
    status = CUDA.CuArray{Int32,1,CUDA.DeviceMemory}(undef, 1)
    left = CUDA.zeros(Float32, 513)
    right = CUDA.zeros(Float32, 32, 17)
    kernel = CUDAX._validate_projection_values!(KernelAbstractions.get_backend(status), 256)
    function step!()
        fill!(status, Int32(0))
        kernel(status, left, Int32(257); ndrange=256)
        kernel(status, right, Int32(513); ndrange=256)
        return nothing
    end

    step!()
    CUDA.synchronize()
    executable = CUDA.instantiate(CUDA.capture(step!))
    left_values = ones(Float32, size(left))
    right_values = ones(Float32, size(right))
    for (bad_left, bad_right, expected) in ((true, true, Int32(257)),
            (false, true, Int32(513)), (false, false, Int32(0)))
        left_values[512] = bad_left ? Inf32 : 1.0f0
        left_values[513] = bad_left ? -Inf32 : 1.0f0
        right_values[end] = bad_right ? NaN32 : 1.0f0
        copyto!(left, left_values)
        copyto!(right, right_values)
        CUDA.launch(executable)
        CUDA.synchronize()
        @test only(Array(status)) == expected
    end
end

@testset "CUDA validation accepts overlapping input and status storage" begin
    input = CUDA.zeros(Float32, 513)
    status = view(reinterpret(Int32, input), 1:1)
    kernel = CUDAX._validate_projection_values!(KernelAbstractions.get_backend(input), 256)
    values = zeros(Float32, 513)
    values[end] = Inf32
    for prior in (Int32(0), Int32(257))
        copyto!(input, values)
        fill!(status, prior)
        kernel(status, input, Int32(513); ndrange=256)
        CUDA.synchronize()
        @test only(Array(status)) == (iszero(prior) ? Int32(513) : prior)
    end
end

@testset "CUDA native validation matches the retained portable tree" begin
    backend = KernelAbstractions.get_backend(CUDA.zeros(Float32, 1))
    native = CUDAX._validate_projection_values!(backend, 256)
    portable = CUDAX._validate_projection_values_portable!(backend, 256)
    for count in (0, 33, 257, 1810), prior in (Int32(0), Int32(257)),
            roles in (false, true), rejected in (false, true)
        values = fill(-0.0f0, count)
        rejected && count > 0 && (values[end] = Inf32)
        for kernel in (portable, native)
            arrays = validation_arrays(Float32, count, 513)
            check_validation!(kernel, arrays, values, prior, roles)
        end
    end
end

end # module
