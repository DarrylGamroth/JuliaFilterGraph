module PupilProjectionCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, KernelAbstractions, Test
import FilterGraphAlgorithms: prepare_capture, enqueue!, complete!, process!, reset!

const FGA = FilterGraphAlgorithms
const CUDAX = Base.get_extension(FGA, :FilterGraphAlgorithmsCUDAExt)
const Fixtures = parentmodule(@__MODULE__)

function matrix_fixture(rows=253, columns=1056; active=100)
    matrix = zeros(Float32, rows, columns)
    for row in 2:active+1, column in 1:columns-1
        matrix[row, column] = Float32(mod(row + column, 5)) / 16384.0f0
    end
    return matrix
end

function adapter(matrix; dense=false, pupil_major=false)
    prepared = Fixtures._prepared_captured_pupil_reconstructor(matrix)
    columns = size(matrix, 2)
    pixels = pupil_major ? PermutedDimsArray(CUDA.zeros(Float32, columns ÷ 4, 4), (2, 1)) :
             CUDA.zeros(Float32, 4, columns ÷ 4)
    output, status = CUDA.zeros(Float32, size(matrix, 1)), CUDA.zeros(Int32, 1)
    captured = prepare_capture(CompleteTransition(), CUDA.device(), prepared,
        (output,), (pixels,), status, Int32(3))
    if dense
        projection = CUDAX._prepare_projection(CUDA.device(), matrix,
            output, captured.ordered_input, CUDAX.COOPERATIVE_PROJECTION_DEFAULT_LANES)
        captured = CUDAX.PyramidPupilReconstruction(
            captured.validate_input!, captured.order!, projection, captured.validate_output!,
            output, status, pixels, captured.ordered_input,
            captured.input_failure, captured.output_failure)
    end
    enqueue!(captured)
    CUDA.synchronize()
    executable = CUDA.instantiate(CUDA.capture(() -> enqueue!(captured)))
    return (; prepared, captured, pixels, output, status, executable)
end

function replay!(case, values; status=Int32(0))
    pixels = permutedims(reshape(values, length(values) ÷ 4, 4))
    if case.pixels isa PermutedDimsArray
        copyto!(parent(case.pixels), permutedims(pixels))
    else
        copyto!(case.pixels, pixels)
    end
    fill!(case.status, status)
    CUDA.launch(case.executable)
    CUDA.synchronize()
    return only(Array(case.status))
end

bits(array) = collect(reinterpret(UInt32, vec(array)))
replay_bytes(executable) = @allocated CUDA.launch(executable)
frame_bytes(outputs, graph, inputs) = @allocated process!(outputs, graph, inputs)

@testset "CUDA pupil preparation preserves subnormal coefficient rows under host DAZ" begin
    matrix = zeros(Float32, 128, 2048)
    matrix[2, 1] = reinterpret(Float32, UInt32(1))
    matrix[3, 1] = 0.125f0
    output, input = CUDA.zeros(Float32, size(matrix, 1)), CUDA.zeros(Float32, size(matrix, 2))
    original = get_zero_subnormals()
    try
        for flush_subnormals in (false, true)
            set_zero_subnormals(flush_subnormals)
            projection = CUDAX._prepare_pupil_projection(CUDA.device(), matrix, output, input, 256)
            @test projection isa CUDAX.PupilProjection
            @test findall(!iszero, Array(projection.row_map)) == [2, 3]
        end
    finally
        set_zero_subnormals(original)
    end
    @test get_zero_subnormals() == original
end

@testset "CUDA pupil row packing preserves original columns and cold fallback" begin
    matrix = matrix_fixture()
    case = adapter(matrix)
    projection = case.captured.projection
    @test projection isa CUDAX.PupilProjection
    @test isconcretetype(typeof(projection))
    @test size(projection.projection.matrix) == (size(matrix, 2), 100)
    @test bits(permutedims(Array(projection.projection.matrix))) == bits(matrix[2:101, :])
    @test Array(projection.row_map) == vcat(Int32[0], Int32.(1:100), zeros(Int32, 152))
    @test !Base.mightalias(projection.packed_output, case.output)
    @test !Base.mightalias(projection.packed_output, case.captured.ordered_input)
    @test projection.projection.maxabs === Float64(maximum(abs, matrix))
    # The same large sparse matrix remains dense outside the pupil factory.
    @test CUDAX._prepare_projection(CUDA.device(), matrix, case.output,
        case.captured.ordered_input, 256) isa CUDAX.CUBLASProjection
    for (rows, columns, active) in ((127, 2068, 60), (128, 2048, 64), (512, 512, 256))
        boundary = matrix_fixture(rows, columns; active)
        edge = adapter(boundary)
        @test edge.captured.projection isa CUDAX.PupilProjection
        @test replay!(edge, ones(Float32, columns)) == 0
        @test Array(edge.output) == boundary * ones(Float32, columns)
    end
    negative_zero = matrix_fixture(128, 2048; active=64)
    negative_zero[end, end] = -0.0f0
    for fallback in (matrix_fixture(513, 512; active=100),
            matrix_fixture(128, 2044; active=64),
            matrix_fixture(128, 2048; active=65), zeros(Float32, 128, 2048), negative_zero)
        output, input = CUDA.zeros(Float32, size(fallback, 1)), CUDA.zeros(Float32, size(fallback, 2))
        @test CUDAX._prepare_pupil_projection(CUDA.device(), fallback,
            output, input, 256) isa CUDAX.CUBLASProjection
    end
    @test_throws ArgumentError prepare_capture(CompleteTransition(), CUDA.device(), case.prepared,
        (view(vec(case.pixels), 1:size(matrix, 1)),), (case.pixels,), case.status, Int32(3))
end

@testset "CUDA packed pupil projection matches dense success, failure, reset and replay" begin
    matrix = matrix_fixture()
    count = size(matrix, 2)
    ieee = [reinterpret(Float32, (0x00000000, 0x80000000, 0x00000001, 0x80000001)[mod1(i, 4)])
            for i in 1:count]
    finite_cases = (zeros(Float32, count), fill(-0.0f0, count), ieee,
        Float32[mod(i, 17) - 8 for i in 1:count] ./ 16.0f0,
        Float32[sin(i / 23) for i in 1:count])
    for pupil_major in (false, true)
        @testset "pupil_major=$pupil_major" begin
            packed, dense = adapter(matrix; pupil_major), adapter(matrix; dense=true, pupil_major)
            addresses = map(pointer, (packed.output, packed.captured.ordered_input,
                packed.captured.projection.packed_output, packed.captured.projection.row_map))
            expected = zeros(Float32, size(matrix, 1))
            for (index, values) in enumerate(finite_cases)
                @test replay!(packed, values) == replay!(dense, values) == 0
                actual, baseline = Array(packed.output), Array(dense.output)
                @test actual ≈ baseline rtol=8eps(Float32)
                # Dyadic fixtures have an exact representable dot sum; the
                # changing sinusoidal fixture retains the checked CPU tolerance.
                index < length(finite_cases) && @test bits(actual) == bits(baseline)
                pixels = permutedims(reshape(values, count ÷ 4, 4))
                @test process!(expected, packed.prepared, pixels) === nothing
                @test actual ≈ expected rtol=8eps(Float32)
                @test all(iszero, bits(actual[[1; 102:253]]))
            end
            # The coefficient bound keeps both reduction orders finite at the
            # edge of the accepted input range.
            extreme = fill(floatmax(Float32), count)
            @test replay!(packed, extreme) == replay!(dense, extreme) == 0
            @test all(isapprox.(Array(packed.output), Array(dense.output); rtol=8eps(Float32)))
            @test bits(Array(packed.output)[[1; 102:253]]) == bits(Array(dense.output)[[1; 102:253]])
            # Failure suppresses ordering in ordinary storage, but pupil-major
            # ordered scratch aliases input. Exercise both old finite scratch
            # and unselected-column nonfinite scratch under the same status.
            good = finite_cases[4]
            for nonfinite in (NaN32, Inf32, -Inf32)
                @test replay!(packed, good) == replay!(dense, good) == 0
                bad = copy(good)
                bad[end] = nonfinite # Every coefficient in this column is zero.
                for upstream in (Int32(0), Int32(0x103))
                    first = replay!(packed, bad; status=upstream)
                    second = replay!(dense, bad; status=upstream)
                    @test first == second
                    @test upstream == 0 ? complete!(packed.captured, first) ===
                          DenseReconstructionNonFiniteInput : first == upstream
                    @test isfinite.(Array(packed.output)) == isfinite.(Array(dense.output))
                    finite = isfinite.(Array(dense.output))
                    @test bits(Array(packed.output)[finite]) == bits(Array(dense.output)[finite])
                end
            end
            @test reset!(packed.captured) === packed.captured
            @test replay!(packed, good) == replay!(dense, good) == 0
            @test bits(Array(packed.output)) == bits(Array(dense.output))
            @test addresses == map(pointer, (packed.output, packed.captured.ordered_input,
                packed.captured.projection.packed_output, packed.captured.projection.row_map))
            replay_bytes(packed.executable)
            CUDA.synchronize()
            @test replay_bytes(packed.executable) == 0
            CUDA.synchronize()
        end
    end
end

@testset "CUDA pupil packing preserves dense overflow, failure priority and recovery" begin
    matrix = zeros(Float32, 127, 2068)
    columns = [175, 265, 397, 629, 643, 813, 1042, 1136, 1326, 1400, 1522, 1599]
    matrix[1, columns] .= Float32[-1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1] .* (floatmax(Float32) / 2)
    candidate, dense = adapter(matrix), adapter(matrix; dense=true)
    @test candidate.captured.projection isa CUDAX.PupilRowPacking

    # On the measured cuBLAS implementation the dense accumulation overflows a
    # partial sum, while a packed GEMV can cancel it. Compare with the current
    # vendor's original layout in case it uses a different reduction order.
    cancelling = ones(Float32, size(matrix, 2))
    expected = replay!(dense, cancelling)
    @test replay!(candidate, cancelling) == expected
    @test bits(Array(candidate.output)) == bits(Array(dense.output))

    overflowing = zeros(Float32, size(matrix, 2))
    overflowing[first(columns)] = floatmax(Float32)
    nonfinite = copy(overflowing)
    nonfinite[end] = NaN32
    recovered = zeros(Float32, size(matrix, 2))
    recovered[first(columns)] = 0.125f0
    for (values, failure) in (
        (overflowing, DenseReconstructionNonFiniteOutput),
        (nonfinite, DenseReconstructionNonFiniteInput),
        (recovered, nothing),
    )
        expected = replay!(dense, values)
        @test replay!(candidate, values) == expected
        @test complete!(candidate.captured, expected) === complete!(dense.captured, expected) === failure
        @test bits(Array(candidate.output)) == bits(Array(dense.output))
    end
end

function graph(matrix)
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ { type = ndarray name = reconstruct label = pwfs-pupil-reconstructor-f32
            config = { reconstructed_count = $(size(matrix, 1))
                       selected_pixel_count = $(size(matrix, 2) ÷ 4)
                       initial_reconstructor = null
                       reconstructed_schema = org.example.ao.pupil-reconstruction/1 } } ]
          inputs = [ "reconstruct:reconstruction-pixels" "reconstruct:reconstructor" ]
          outputs = [ "reconstruct:reconstructed" ]
        }
        """)
        flush(io)
        checked = prepare_graph(path; algorithms=(PyramidPupilReconstructorF32,))
        replace_parameters!(checked, :reconstructor => matrix)
        return checked
    end
end

@testset "CUDA packed reconstruction retains publication, metadata and calibration snapshot" begin
    matrix = matrix_fixture()
    cpu, source = graph(matrix), graph(matrix)
    gpu = prepare_captured_graph(source, CUDA.device())
    @test only(gpu.nodes).adapter.projection isa CUDAX.PupilProjection
    expected, actual = (reconstructed=zeros(Float32, 253),), (reconstructed=zeros(Float32, 253),)
    pixels = reshape(Float32[mod(i, 17) - 8 for i in 1:length(matrix[1, :])], 4, :) ./ 16.0f0
    input = (var"reconstruction-pixels"=pixels,)
    metadata = SampleMetadata(17)
    for _ in 1:3
        @test process!(actual, gpu, input, (var"reconstruction-pixels"=metadata,)) ==
              process!(expected, cpu, input, (var"reconstruction-pixels"=metadata,))
        @test actual.reconstructed ≈ expected.reconstructed rtol=8eps(Float32)
        @test FilterGraphs.output_metadata(gpu) == (reconstructed=metadata,)
    end
    published = copy(actual.reconstructed)
    pixels[end] = NaN32
    @test process!(actual, gpu, input) === DenseReconstructionNonFiniteInput
    @test bits(actual.reconstructed) == bits(published)
    @test FilterGraphs.output_metadata(gpu) == (reconstructed=metadata,)
    dense_gpu = prepare_captured_graph(graph(64f0 * matrix), CUDA.device())
    fill!(pixels, 0.5f0)
    @test process!(actual, dense_gpu, input, (var"reconstruction-pixels"=metadata,)) == (reconstructed=true,)
    published = copy(actual.reconstructed)
    fill!(pixels, floatmax(Float32))
    @test process!(actual, dense_gpu, input) === DenseReconstructionNonFiniteOutput
    @test bits(actual.reconstructed) == bits(published)
    @test FilterGraphs.output_metadata(dense_gpu) == (reconstructed=metadata,)
    fill!(pixels, 0.5f0)
    # Replacement on the source graph cannot change the retained coefficients.
    replace_parameters!(source, :reconstructor => 2matrix)
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    @test actual.reconstructed ≈ expected.reconstructed rtol=8eps(Float32)
    @test reset!(gpu) === gpu
    @test process!(actual, gpu, input) == process!(expected, cpu, input)
    frame_bytes(actual, gpu, input)
    @test frame_bytes(actual, gpu, input) == 0
end

end
