module PdmOffsetCaptureTests

using CUDA, FilterGraphAlgorithms, FilterGraphs, KernelAbstractions, Test
import FilterGraphAlgorithms: process!, reset!

function offset_pair(flat, offset)
    text = """
    filter.graph = {
        nodes = [
          { type = ndarray name = flat label = pdm-system-flat-f32 config = { actuator_count = $(length(flat)) } }
          { type = ndarray name = offset label = pdm-engineering-offset-f32 config = { actuator_count = $(length(flat)) } }
        ]
        links = [ { output = "flat:requested-with-system-flat" input = "offset:requested" } ]
        inputs = [ "flat:requested" "flat:system-flat" "offset:engineering-offset" ]
        outputs = [ "flat:requested-with-system-flat" "offset:requested-with-offset" "offset:current-offset" ]
    }
    """
    function prepare()
        graph = mktemp() do path, io
            write(io, text)
            flush(io)
            prepare_graph(path; algorithms = (PdmSystemFlatF32, PdmEngineeringOffsetF32))
        end
        replace_parameters!(graph, Symbol("system-flat")=>flat, Symbol("engineering-offset")=>offset)
        return graph
    end
    cpu = prepare()
    gpu = prepare_captured_graph(prepare(), CUDA.device())
    output() = (var"requested-with-system-flat" = zeros(Float32, length(flat)),
        var"requested-with-offset" = zeros(Float32, length(flat)),
        var"current-offset" = zeros(Float32, length(flat)))
    return cpu, gpu, output(), output()
end

@testset "captured physical offsets preserve sums and current offset" begin
    for count in (3, 277, 513)
        flat = Float32[mod(i, 9) * 0.03 for i in 1:count]
        offset = Float32[mod(i, 7) * -0.02 for i in 1:count]
        cpu, gpu, expected, actual = offset_pair(flat, offset)
        try
            GC.gc(true)
            for sequence in 1:4
                input = (requested = Float32[(i-sequence) * 0.001 for i in 1:count],)
                @test process!(actual, gpu, input) == process!(expected, cpu, input)
                @test all(name -> actual[name] == expected[name], keys(expected))
                @test actual.var"current-offset" == offset
            end
            reset!(gpu)
            reset!(cpu)
            input = (requested = fill(0.1f0, count),)
            @test process!(actual, gpu, input) == process!(expected, cpu, input)
            @test all(name -> actual[name] == expected[name], keys(expected))
        finally
            @test captured_graph_state(gpu) == CapturedGraphReady
        end
    end
end

@testset "captured physical offset failure identity follows first actuator" begin
    for node in (:flat, :offset), first in (:input, :output), count in (3, 513)
        flat, offset = zeros(Float32, count), zeros(Float32, count)
        input = zeros(Float32, count)
        if first == :output
            input[1] = floatmax(Float32)
            (node == :flat ? flat : offset)[1] = floatmax(Float32)
            input[end] = NaN32
        else
            input[1] = NaN32
            input[end] = floatmax(Float32)
            (node == :flat ? flat : offset)[end] = floatmax(Float32)
        end
        # An offset-stage input failure cannot reach offset through a failing
        # flat node. Offset overflow is the meaningful downstream rejection.
        node == :offset && first == :input && continue
        cpu, gpu, expected, actual = offset_pair(flat, offset)
        try
            foreach(array -> fill!(array, 71), values(actual))
            result = process!(actual, gpu, (requested = input,))
            reference = process!(expected, cpu, (requested = input,))
            @test result == reference
            @test !(result isa NamedTuple)
            @test all(array -> all(==(71), array), values(actual))
        finally
            @test captured_graph_state(gpu) == CapturedGraphReady
        end
    end
end

function engineering_graph(offset)
    return mktemp() do path, io
        write(io, """
        filter.graph = {
          nodes = [ { type = ndarray name = offset label = pdm-engineering-offset-f32
            config = { actuator_count = $(length(offset)) } } ]
          inputs = [ "offset:requested" "offset:engineering-offset" ]
          outputs = [ "offset:requested-with-offset" "offset:current-offset" ]
        }
        """)
        flush(io)
        graph = prepare_graph(path; algorithms = (PdmEngineeringOffsetF32,))
        replace_parameters!(graph, Symbol("engineering-offset")=>offset)
        return graph
    end
end

@testset "zero engineering offset preserves signed zero" begin
    offset = Float32[0, -0.0, -0.0]
    cpu = engineering_graph(offset)
    gpu = prepare_captured_graph(engineering_graph(offset), CUDA.device())
    expected = (var"requested-with-offset" = zeros(Float32, 3),
        var"current-offset" = zeros(Float32, 3))
    actual = deepcopy(expected)
    try
        input = (requested = Float32[-0.0, 0.0, -0.0],)
        @test process!(actual, gpu, input) == process!(expected, cpu, input)
        @test reinterpret(UInt32, actual.var"requested-with-offset") ==
              reinterpret(UInt32, input.requested)
        @test reinterpret(UInt32, actual.var"current-offset") == reinterpret(UInt32, offset)
    finally
        @test captured_graph_state(gpu) == CapturedGraphReady
    end
end

@testset "engineering failure ordering across lanes" begin
    for output_first in (false, true)
        offset = zeros(Float32, 513)
        input = zeros(Float32, 513)
        if output_first
            offset[1] = input[1] = floatmax(Float32)
            input[513] = NaN32
        else
            input[1] = NaN32
            offset[513] = input[513] = floatmax(Float32)
        end
        cpu = engineering_graph(offset)
        gpu = prepare_captured_graph(engineering_graph(offset), CUDA.device())
        expected = (var"requested-with-offset" = zeros(Float32, 513),
            var"current-offset" = zeros(Float32, 513))
        actual = deepcopy(expected)
        try
            result = process!(actual, gpu, (requested = input,))
            @test result == process!(expected, cpu, (requested = input,))
            @test !(result isa NamedTuple)
        finally
            @test captured_graph_state(gpu) == CapturedGraphReady
        end
    end
end

@testset "captured physical offsets support exact in-place command storage" begin
    for (declaration, configuration) in (
        (PdmSystemFlatF32, PdmSystemFlatF32Configuration(actuator_count = 3)),
        (PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration(actuator_count = 3)))
        prepared = prepare_algorithm(declaration, configuration)
        values = Float32[-0.0, 0.0, -0.0]
        input = CUDA.CuArray(values)
        current = CUDA.zeros(Float32, 3)
        status = CUDA.zeros(Int32, 1)
        engineering = declaration === PdmEngineeringOffsetF32
        outputs = engineering ? (input, current) : (input,)
        captured = FilterGraphAlgorithms.prepare_capture(
            CompleteTransition(),
            CUDA.device(), prepared, outputs, (input,), status, Int32(1))
        FilterGraphAlgorithms.enqueue!(captured)
        CUDA.synchronize()
        expected = engineering ? values : values .+ 0.0f0
        @test only(Array(status)) == 0
        @test reinterpret(UInt32, Array(input)) == reinterpret(UInt32, expected)
        engineering && @test all(iszero, Array(current))
    end
end

function direct_offset_case(engineering, offset; in_place = false)
    n = length(offset)
    declaration, configuration, parameter = engineering ?
        (PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration(actuator_count = n), "engineering-offset") :
        (PdmSystemFlatF32, PdmSystemFlatF32Configuration(actuator_count = n), "system-flat")
    prepared = prepare_algorithm(declaration, configuration)
    replace_parameter!(prepared, parameter, offset)
    input = CUDA.zeros(Float32, n)
    output = in_place ? input : CUDA.zeros(Float32, n)
    outputs = engineering ? (output, CUDA.zeros(Float32, n)) : (output,)
    status = CUDA.zeros(Int32, 1)
    captured = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), CUDA.device(),
        prepared, outputs, (input,), status, Int32(1))
    return (; prepared, input, outputs, status, captured)
end

offset_bits(arrays) = map(array -> reinterpret(UInt32, vec(Array(array))), arrays)

function compare_offset_step!(case, reference, staged!, sample)
    expected = map(array -> zeros(Float32, length(array)), case.outputs)
    result = process!(expected..., case.prepared, sample)
    for current in (case, reference)
        foreach(array -> fill!(array, 71.0f0), current.outputs)
        copyto!(current.input, sample)
        fill!(current.status, Int32(0))
    end
    supplied = offset_bits(case.outputs)
    FilterGraphAlgorithms.enqueue!(case.captured)
    c = reference.captured
    staged!(c.output, c.current_offset, c.input, c.offset, c.frame_status,
        c.input_failure, c.output_failure; ndrange = 256)
    CUDA.synchronize()
    @test only(Array(case.status)) == only(Array(reference.status))
    @test FilterGraphAlgorithms.complete!(case.captured, only(Array(case.status))) === result
    @test offset_bits(case.outputs) == offset_bits(reference.outputs)
    @test offset_bits(case.outputs) == (isnothing(result) ? offset_bits(expected) : supplied)
    return result
end

@testset "bounded offset publication matches staged arithmetic and alias semantics" begin
    extension = Base.get_extension(FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
    for n in (1, 255, 256, 257, 277, 511, 512, 513), engineering in (false, true),
            in_place in (false, true)
        offset = Float32[mod(i, 7) * -0.02f0 for i in 1:n]
        offset[1] = -0.0f0
        sample = Float32[mod(i, 9) * 0.03f0 for i in 1:n]
        sample[1] = -0.0f0
        case = direct_offset_case(engineering, offset; in_place)
        reference = direct_offset_case(engineering, offset; in_place)
        staged! = extension._pdm_offset!(KernelAbstractions.get_backend(reference.input), 256)
        addresses = map(pointer, (case.outputs..., case.input, case.captured.offset, case.status))
        @test compare_offset_step!(case, reference, staged!, sample) === nothing
        fill!(case.status, Int32(0x103))
        supplied = offset_bits(case.outputs)
        FilterGraphAlgorithms.enqueue!(case.captured)
        CUDA.synchronize()
        @test only(Array(case.status)) == Int32(0x103)
        @test offset_bits(case.outputs) == supplied
        @test reset!(case.captured) === case.captured
        @test map(pointer, (case.outputs..., case.input, case.captured.offset, case.status)) == addresses
    end
    for n in (257, 512, 513), engineering in (false, true), in_place in (false, true)
        middle = cld(n, 2)
        offset = zeros(Float32, n)
        offset[[1, middle, n]] .= floatmax(Float32)
        case = direct_offset_case(engineering, offset; in_place)
        reference = direct_offset_case(engineering, offset; in_place)
        staged! = extension._pdm_offset!(KernelAbstractions.get_backend(reference.input), 256)
        for (overflow, nonfinite) in ((1, n), (n, 1), (middle, n), (n, middle), (n, n))
            sample = zeros(Float32, n)
            sample[overflow] = floatmax(Float32)
            sample[nonfinite] = NaN32
            @test !isnothing(compare_offset_step!(case, reference, staged!, sample))
        end
        @test compare_offset_step!(case, reference, staged!, zeros(Float32, n)) === nothing
    end
end

end # module
