@testset "shared structured projection preparation and portable collectives" begin
    count = 523
    columns = [mod(17row, count) + 1 for row in 1:277]
    columns[end] = count
    matrix = zeros(Float32, length(columns), count)
    for row in eachindex(columns)
        matrix[row, columns[row]] = 1
    end
    values = collect(reinterpret(Float32, UInt32[
        mod(index, 4) == 0 ? 0x80000001 : mod(index, 4) == 1 ? 0x80000000 :
        mod(index, 4) == 2 ? 0x00000001 : 0x00000000 for index in 1:count]))
    input, output, status = copy(values), zeros(Float32, length(columns)), zeros(Int32, 1)
    projection = _prepare_projection(TEST_DEVICE, matrix, output, input, 128)
    @test projection isa SelectionProjection
    @test _projection_matrix(projection) === nothing
    fill!(matrix, 7)
    _enqueue_projection!(projection, output, status, input)
    KernelAbstractions.synchronize(BACKEND)
    @test reinterpret(UInt32, output) == reinterpret(UInt32, values[columns])
    unselected = findlast(index -> index ∉ columns, eachindex(values))
    @test !isnothing(unselected)
    for value in (NaN32, Inf32, -Inf32)
        copyto!(input, values)
        input[unselected] = value
        _enqueue_projection!(projection, output, status, input)
        KernelAbstractions.synchronize(BACKEND)
        @test all(isnan, output)
        @test iszero(only(status))
    end
    status[1] = 103
    fill!(output, 71)
    _enqueue_projection!(projection, output, status, input)
    KernelAbstractions.synchronize(BACKEND)
    @test all(==(71), output)
    @test only(status) == 103
    identity = Matrix{Float32}(I, count, count)
    copied = zeros(Float32, count)
    copy_projection = _prepare_projection(TEST_DEVICE, identity, copied, input, 128)
    @test copy_projection isa IdentityProjection
    @test _prepare_projection(TEST_DEVICE, identity, input, input, 128).copy! === nothing
    @test _capture_projection_input(TEST_DEVICE, identity, input, copied) !== input
    @test _capture_projection_input(TEST_DEVICE, identity, input, input) !== input
    status[1] = 0
    copyto!(input, values)
    _enqueue_projection!(copy_projection, copied, status, input)
    KernelAbstractions.synchronize(BACKEND)
    @test reinterpret(UInt32, copied) == reinterpret(UInt32, values)
    @test _prepare_projection(TEST_DEVICE, 2identity, copied, input, 128) isa KAProjection
    @test !FilterGraphAlgorithms._supports_validation_reuse(BACKEND)
end
