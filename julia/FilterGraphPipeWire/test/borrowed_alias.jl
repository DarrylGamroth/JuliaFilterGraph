@testset "Borrowed array byte-span alias detection" begin
    storage = zeros(UInt32, 8)
    separate = zeros(UInt32, 8)
    GC.@preserve storage separate begin
        borrowed = FilterGraphPipeWire.BorrowedArray(pointer(storage), (4,), Val(true))
        row_major = FilterGraphPipeWire.RowMajorBorrowedArray{UInt32,1,true}(
            pointer(storage, 4), (4,), 4)
        adjacent = FilterGraphPipeWire.BorrowedArray(pointer(storage, 5), (4,), Val(false))
        byte = FilterGraphPipeWire.BorrowedArray(Ptr{UInt8}(pointer(storage, 4)) + 3,
            (1,), Val(false))
        unsafe = FilterGraphPipeWire.UnsafeArray(pointer(storage), (8,))
        unsafe_separate = FilterGraphPipeWire.UnsafeArray(pointer(separate), (8,))
        for array in (borrowed, row_major, adjacent, byte)
            @test Base.mightalias(array, storage)
            @test Base.mightalias(storage, array)
            @test !Base.mightalias(array, separate)
            @test !Base.mightalias(separate, array)
            @test !Base.mightalias(array, UInt32[])
            @test !Base.mightalias(UInt32[], array)
            # Unknown wrappers remain conservative, including noncontiguous views.
            @test Base.mightalias(array, view(separate, 1:2:8))
            @test Base.mightalias(view(separate, 1:2:8), array)
            @test Base.mightalias(array, unsafe)
            @test Base.mightalias(unsafe, array)
            @test Base.mightalias(array, view(unsafe, 1:2:8))
            @test Base.mightalias(view(unsafe, 1:2:8), array)
            @test !Base.mightalias(array, unsafe_separate)
            @test !Base.mightalias(unsafe_separate, array)
            @test !Base.mightalias(array, view(unsafe_separate, 1:2:8))
            @test !Base.mightalias(view(unsafe_separate, 1:2:8), array)
        end
        @test Base.mightalias(borrowed, row_major)
        @test Base.mightalias(row_major, borrowed)
        @test Base.mightalias(borrowed, byte)
        @test Base.mightalias(byte, borrowed)
        @test !Base.mightalias(borrowed, adjacent)
        @test !Base.mightalias(adjacent, borrowed)
        @test Base.mightalias(row_major, adjacent)
        @test Base.mightalias(adjacent, row_major)
        extension = Base.get_extension(FilterGraphPipeWire, :FilterGraphPipeWireForwardDiffExt)
        @test extension !== nothing
        partials = extension.Partials((1.0, 2.0))
        @test !Base.mightalias(borrowed, partials)
        @test !Base.mightalias(partials, borrowed)
    end
end

@testset "Row-major borrowed pixel selection preserves payload order and bits" begin
    dims = (5, 37)
    bits = UInt32.(1:prod(dims))
    bits[36:41] .= (0x7fc00543, 0x7f800543, 0x80000000, 0x7f800000, 0xff800000, 0x00000000)
    buffer = collect(reinterpret(Float32, bits))
    selected_pixels = falses(length(buffer))
    selected_pixels[36:109] .= true
    selected_pixels[122:158] .= true
    separate = zeros(Float32, length(buffer))
    fgpw = FilterGraphPipeWire

    GC.@preserve buffer separate for mask in (trues(length(buffer)), selected_pixels)
        prepared = prepare_algorithm(
            PixelSelectionF32,
            PixelSelectionF32Configuration(
                image_size=dims, mask=mask,
                input_schema="test.detector/1", output_schema="test.selected-pixels/1",
            ),
        )
        format = first(algorithm_formats(prepared))
        binding = fgpw.PortBinding(format, fgpw.RowMajorLayout())
        borrowed = fgpw._borrow_pointer(binding, pointer(buffer), Val(false))
        output = zeros(Float32, count(mask))
        @test FilterGraphAlgorithms.process!(output, prepared, borrowed) === nothing
        @test reinterpret(UInt32, output) == bits[mask]

        runs = getfield(prepared.plan, :_row_major_runs)
        run = first(runs)
        for malformed in (first(run):(length(buffer) + 1), first(run):(last(run) - 1))
            runs[1] = malformed
            @test FilterGraphAlgorithms.process!(output, prepared, borrowed) === nothing
            @test reinterpret(UInt32, output) == bits[mask]
        end
        runs[1] = run

        offset = all(mask) ? 1 : 2
        aliased = fgpw.BorrowedArray(pointer(buffer, offset), (length(output),), Val(true))
        @test_throws ArgumentError FilterGraphAlgorithms.process!(aliased, prepared, borrowed)
        destination = fgpw.BorrowedArray(pointer(separate), (length(output),), Val(true))
        @test FilterGraphAlgorithms.process!(destination, prepared, borrowed) === nothing
        @test map(i -> reinterpret(UInt32, destination[i]), eachindex(destination)) == bits[mask]

        getfield(prepared.plan, :selected_rows)[1] = dims[1]
        getfield(prepared.plan, :selected_columns)[1] = dims[2]
        @test FilterGraphAlgorithms.process!(output, prepared, borrowed) === nothing
        expected = bits[mask]
        expected[1] = bits[end]
        @test reinterpret(UInt32, output) == expected
    end
end
