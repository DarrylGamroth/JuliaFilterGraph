module SccLlowfsGraphTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

function prepare_graph_text(text; algorithms)
    return mktemp() do path, stream
        write(stream, text)
        flush(stream)
        prepare_graph(path; algorithms)
    end
end

import FilterGraphAlgorithms: process!

const REGION_INPUT_NAME = Symbol("region-pixels")
const PARAMETER_INPUT_NAME = Symbol("model-parameters")

function graph_allocation_bytes(outputs, graph, inputs)
    process!(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "ML LLOWFS graph reads a model supplied by the caller" begin
    path = joinpath(@__DIR__, "data", "examples", "ml-llowfs/graph.conf")
    graph = prepare_graph(path; algorithms=FilterGraphAlgorithms.algorithms())
    @test graph.input_formats[REGION_INPUT_NAME].shape == (1, 2, 3)
    @test graph.parameter_formats[PARAMETER_INPUT_NAME].shape == (14,)
    @test graph.output_formats.modes.shape == (2,)
    @test graph.output_formats.modes.schema == "examples/synthetic-llowfs-modes-f32/1"
    @test graph.parameter_formats[PARAMETER_INPUT_NAME].schema ==
          "examples/synthetic-llowfs-parameters-f32/1"

    region = reshape(Float32[1, 2, 4, 8, 16, 32], 1, 2, 3)
    modes = fill(42f0, 2)
    inputs = NamedTuple{(REGION_INPUT_NAME,)}((region,))
    outputs = (modes=modes,)

    @test process!(outputs, graph, inputs) === MlLlowfsParametersMissing
    @test modes == fill(42f0, 2)

    parameters = zeros(Float32, 14)
    parameters[1] = 2f0
    parameters[end] = -0.5f0
    @test replace_parameters!(graph, PARAMETER_INPUT_NAME => parameters) === graph
    fill!(parameters, 0f0)
    @test process!(outputs, graph, inputs) == (modes=true,)
    @test modes[1] < 0f0
    @test modes[2] == -0.5f0
    @test graph_allocation_bytes(outputs, graph, inputs) == 0
    @test_throws GraphConfigurationError prepare_graph_text(
        replace(read(path, String), "outputs = 2" => "outputs = 0"); algorithms=(MlLlowfsF32,))
end

const SCC_DOMAIN = ntuple(index -> UInt8(index), 16)
const SCC_DURATION_NS = Int64(1_000_000)
const SCC_IMAGE_SCHEMA = "org.subaru.spiders.synthetic-scc-counts-xy/1"

function scc_calibration(; sign=:command_error, bounds=1.0f0,
        reference=zeros(Float32, 2, 2), bias=fill(0.01f0, 2), scale=1.0f0,
        coordinate_order=Tuple(1:2), command_unit="normalized-bax307",
        fringed_response=zeros(Float32, 2, 2), unfringed_response=zeros(Float32, 2, 2))
    mask = Bool[true false; false true]
    reconstructor = zeros(Float32, 2, 2)
    reconstructor[1, 1] = scale
    reconstructor[2, 2] = 2.0f0 * scale
    return SccCalibration(; frame_shape=(2, 2), reference_difference=reference,
        correction_mask=mask, reconstructor, coordinate_order, fringed_response, unfringed_response,
        command_unit, reconstructor_sign=sign,
        exposure_duration_nanoseconds=SCC_DURATION_NS, image_schema=SCC_IMAGE_SCHEMA,
        bias_command=bias, lower_limits=fill(-bounds, 2), upper_limits=fill(bounds, 2))
end

function scc_metadata(sequence; trigger=SAMPLE_TRIGGER_HIGH, duration=SCC_DURATION_NS,
        domain=SCC_DOMAIN, generation=UInt64(1), discontinuity=false, trigger_high_is_fringed=true)
    raw = SampleMetadata(sequence; trigger_state=trigger,
        exposure_duration_nanoseconds=duration,
        acquisition_identity=SampleAcquisitionIdentity(domain, generation, UInt64(sequence)),
        discontinuity)
    return chopped_image_metadata(raw; trigger_high_is_fringed)
end

function scc_graph_text(configuration)
    value_text(value::Number) = string(value)
    value_text(value::String) = repr(value)
    value_text(value::Union{Tuple,AbstractVector}) = "[ " * join(value_text.(value), " ") * " ]"
    fields = join((string(field, " = ", value_text(getfield(configuration, field)))
        for field in fieldnames(typeof(configuration))), "\n")
    return "filter.graph = { nodes = [ { type = ndarray label = scc-correction-f32 name = scc config = { " *
        fields * " } } ] inputs = [ scc:frame ] outputs = [ scc:command ] }"
end

@testset "SCC frame size preserves width and height and independent command bounds" begin
    calibration = SccCalibration(; frame_shape=(3, 2),
        reference_difference=zeros(Float32, 3, 2),
        correction_mask=Bool[false true; true false; false false],
        reconstructor=Float32[1 0; 0 1], coordinate_order=(1, 2),
        command_unit=SCC_IMAGE_SCHEMA, reconstructor_sign=:correction,
        fringed_response=zeros(Float32, 2, 2), unfringed_response=zeros(Float32, 2, 2),
        exposure_duration_nanoseconds=SCC_DURATION_NS, image_schema=SCC_IMAGE_SCHEMA,
        bias_command=zeros(Float32, 2), lower_limits=fill(-1f0, 2), upper_limits=fill(1f0, 2))
    config = SccCorrectionF32Configuration(calibration; gain=1f0, pole=0f0)
    text = scc_graph_text(config)
    @test config.frame_size === (3, 2)
    original_limits = "command_limits = [ [ -1.0 -1.0 ] [ 1.0 1.0 ] ]"
    for limits in ("[-0.5 0.5]", "[[-0.5 -0.6] [0.5 0.6]]",
            "[-0.5 [0.5 0.6]]", "[[-0.5 -0.6] 0.5]")
        graph = prepare_graph_text(replace(text, original_limits => "command_limits = $limits");
            algorithms=(SccCorrectionF32,))
        @test graph.input_formats.frame.shape === (3, 2)
        command = zeros(Float32, 2)
        process!((command=command,), graph, (frame=Float32[0 0.4; 0.2 0; 0 0],),
            (frame=scc_metadata(1),))
        process!((command=command,), graph, (frame=zeros(Float32, 3, 2),),
            (frame=scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW),))
        @test command ≈ Float32[0.2, 0.4]
    end
    for limits in ("[]", "[-1]", "[-1 1 2]", "[[-1] 1]", "[-1 [1]]",
            "[1 -1]", "[-1 [1e100 1]]")
        @test_throws GraphConfigurationError prepare_graph_text(
            replace(text, original_limits => "command_limits = $limits"); algorithms=(SccCorrectionF32,))
    end
    for size in ("[3]", "[3 2 1]", "[3.5 2]", "[0 2]")
        @test_throws GraphConfigurationError prepare_graph_text(
            replace(text, "frame_size = [ 3 2 ]" => "frame_size = $size"); algorithms=(SccCorrectionF32,))
    end
end

@testset "SCC Graph publishes current metadata for held commands" begin
    configuration = SccCorrectionF32Configuration(scc_calibration(); gain=0.5f0, pole=0.8f0)
    graph = prepare_graph_text(scc_graph_text(configuration); algorithms=(SccCorrectionF32,))
    frame = Float32[0.1, 0, 0, -0.1]
    output = zeros(Float32, 2)
    for (sequence, trigger) in ((1, SAMPLE_TRIGGER_HIGH), (2, SAMPLE_TRIGGER_MIXED),
            (3, SAMPLE_TRIGGER_HIGH), (4, SAMPLE_TRIGGER_LOW), (5, SAMPLE_TRIGGER_HIGH))
        metadata = scc_metadata(sequence; trigger)
        image = reshape(trigger == SAMPLE_TRIGGER_LOW ? zeros(Float32, 4) : frame, 2, 2)
        availability = process!((command=output,), graph, (frame=image,), (frame=metadata,))
        @test availability == (command=true,)
        @test FilterGraphs.output_metadata(graph).command == metadata
        @test all(isfinite, output)
        @test length(output) == 2
    end
end

end
