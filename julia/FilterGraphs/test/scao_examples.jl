module ScaoExampleTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

const EXAMPLE_DIRECTORY = normpath(
    joinpath(@__DIR__, "data", "examples"),
)
const CONTROLLER_FEEDBACK = Symbol("controller-constraint-feedback")
const PYRAMID_DIVISOR = Symbol("normalization-divisor")
const SHACK_HARTMANN_RESULT = Union{
    NamedTuple{(:demanded, CONTROLLER_FEEDBACK, :flux, :validity),
        NTuple{4,Bool}},
    DenseReconstructionFailure,
    LeakyIntegratorFailure,
    ControllerToVdmFailure,
    VdmToPdmFailure,
    PdmCommandFailure,
    PdmFeedbackToVdmFailure,
    VdmFeedbackToControllerFailure,
}

function prepare_example(name)
    return prepare_graph(
        joinpath(EXAMPLE_DIRECTORY, name);
        algorithms=algorithms(),
    )
end

function process_allocation_bytes(outputs, graph, inputs)
    return @allocated FilterGraphs.process!(outputs, graph, inputs)
end

function process_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated FilterGraphs.process!(
        outputs,
        graph,
        inputs,
        metadata,
    )
end

function process_progressive_example!(
    output,
    graph,
    image,
    row_block_rows,
    sequence,
)
    row_block = Symbol("row-block")
    final_offset = size(image, 1) - row_block_rows
    for first_row in 0:row_block_rows:final_offset
        block = @view image[
            (first_row + 1):(first_row + row_block_rows),
            :,
        ]
        terminal = first_row == final_offset
        inputs = NamedTuple{(row_block,)}((block,))
        metadata = NamedTuple{(row_block,)}((
            SampleMetadata(sequence, first_row; terminal),
        ))
        @test FilterGraphs.process!(
            (reconstructed=output,),
            graph,
            inputs,
            metadata,
        ) == (reconstructed=terminal,)
    end
    return output
end

@testset "representative pyramid SCAO Graph prepares and processes" begin
    graph = prepare_example("pyramid/scao.conf")

    @test length(graph.nodes) == 9
    @test Set(node.name for node in graph.nodes) == Set((
        :calibrate,
        :measure,
        :reconstruct,
        :law,
        Symbol("controller-to-vdm"),
        Symbol("vdm-to-pdm"),
        :command,
        Symbol("pdm-feedback-to-vdm"),
        Symbol("vdm-feedback-to-controller"),
    ))
    @test keys(graph.input_formats) == (:raw,)
    @test graph.input_formats.raw.shape == (64, 64)
    @test graph.parameter_formats.reconstructor.shape == (221, 1_800)
    @test getproperty(
        graph.parameter_formats,
        Symbol("active-to-full"),
    ).shape == (289, 221)
    @test getproperty(graph.parameter_formats, Symbol("vdm-to-pdm")).shape ==
          (277, 289)
    @test getproperty(graph.parameter_formats, Symbol("i4q-support")).shape ==
          (30, 30)
    @test graph.output_formats.demanded.shape == (277,)
    @test getproperty(graph.output_formats, CONTROLLER_FEEDBACK).shape == (221,)
    @test getproperty(graph.output_formats, PYRAMID_DIVISOR).shape == (1,)

    raw = fill(UInt16(1), 64, 64)
    demanded = zeros(Float32, 277)
    feedback = zeros(Float32, 221)
    normalization_divisor = zeros(Float32, 1)
    outputs = NamedTuple{(:demanded, CONTROLLER_FEEDBACK, PYRAMID_DIVISOR)}((
        demanded,
        feedback,
        normalization_divisor,
    ))
    inputs = (raw=raw,)

    @test all(values(FilterGraphs.process!(outputs, graph, inputs)))
    @test normalization_divisor == Float32[4]
    @test all(iszero, demanded)
    @test all(iszero, feedback)
    @test process_allocation_bytes(outputs, graph, inputs) == 0
end

@testset "representative Shack-Hartmann SCAO Graph prepares and processes" begin
    graph = prepare_example("shack-hartmann/scao.conf")

    @test length(graph.nodes) == 9
    @test keys(graph.input_formats) == (:raw,)
    @test graph.input_formats.raw.shape == (352, 352)
    @test graph.parameter_formats.reconstructor.shape == (221, 376)
    @test getproperty(
        graph.parameter_formats,
        Symbol("full-to-active"),
    ).shape == (221, 289)
    @test getproperty(graph.parameter_formats, Symbol("pdm-to-vdm")).shape ==
          (289, 277)
    @test getproperty(
        graph.parameter_formats,
        Symbol("subaperture-origins"),
    ).shape == (188, 2)
    @test graph.output_formats.demanded.shape == (277,)
    @test getproperty(graph.output_formats, CONTROLLER_FEEDBACK).shape == (221,)
    @test graph.output_formats.flux.shape == (188,)
    @test graph.output_formats.validity.shape == (188,)

    raw = fill(UInt16(1), 352, 352)
    demanded = zeros(Float32, 277)
    feedback = zeros(Float32, 221)
    flux = zeros(Float32, 188)
    validity = fill(false, 188)
    outputs = NamedTuple{(:demanded, CONTROLLER_FEEDBACK, :flux, :validity)}((
        demanded,
        feedback,
        flux,
        validity,
    ))
    inputs = (raw=raw,)

    @test all(values(FilterGraphs.process!(outputs, graph, inputs)))
    @test all(==(484.0f0), flux)
    @test all(validity)
    @test all(iszero, demanded)
    @test all(iszero, feedback)
    @test process_allocation_bytes(outputs, graph, inputs) == 0

    metadata = SampleMetadata(
        17;
        timestamp_nanoseconds=1_700_000,
        terminal=true,
    )
    input_metadata = (raw=metadata,)
    @test @inferred(SHACK_HARTMANN_RESULT,
        FilterGraphs.process!(outputs, graph, inputs, input_metadata),
    ) == (demanded=true, var"controller-constraint-feedback"=true,
        flux=true, validity=true)
    published = FilterGraphs.output_metadata(graph)
    @test published.demanded == metadata
    @test getproperty(published, CONTROLLER_FEEDBACK) == metadata
    @test published.flux == metadata
    @test published.validity == metadata
    @test process_allocation_bytes(outputs, graph, inputs, input_metadata) == 0
end

@testset "progressive WFS reconstruction examples prepare and process" begin
    shack = prepare_example("shack-hartmann/progressive.conf")
    @test map(node -> node.name, shack.nodes) == (:measure, :reconstruct)
    @test getproperty(shack.input_formats, Symbol("row-block")).shape == (1, 352)
    @test shack.parameter_formats.reconstructor.shape == (221, 376)
    @test shack.output_formats.reconstructed.shape == (221,)
    shack_output = fill(71.0f0, 221)
    process_progressive_example!(
        shack_output,
        shack,
        ones(Float32, 352, 352),
        1,
        1,
    )
    @test all(iszero, shack_output)

    pyramid = prepare_example("pyramid/progressive.conf")
    @test map(node -> node.name, pyramid.nodes) == (:measure, :reconstruct)
    @test getproperty(pyramid.input_formats, Symbol("row-block")).shape == (2, 64)
    @test pyramid.parameter_formats.reconstructor.shape == (221, 3_600)
    @test pyramid.output_formats.reconstructed.shape == (221,)
    pyramid_output = fill(71.0f0, 221)
    process_progressive_example!(
        pyramid_output,
        pyramid,
        ones(Float32, 64, 64),
        2,
        2,
    )
    @test all(iszero, pyramid_output)
end

@testset "detector readout examples publish only completed frames" begin
    for (name, shape, blocks, second_frame) in (
        ("pyramid/readout.conf", (2, 1, 4), 4,
            Float32[-0.4375, 0.0625, 0.21875]),
        ("shack-hartmann/readout.conf", (2, 1, 6), 3,
            zeros(Float32, 3)),
    )
        graph = prepare_example(name)
        output = fill(71f0, 3)
        input = (var"readout-block"=ones(Float32, shape),)
        for sequence in 1:2
            for ordinal in 0:(blocks - 1)
                terminal = ordinal == blocks - 1
                metadata = (var"readout-block"=
                    SampleMetadata(sequence, ordinal; terminal),)
                @test FilterGraphs.process!((reconstructed=output,), graph,
                    input, metadata) == (reconstructed=terminal,)
            end
            # Uniform illumination has zero SHWFS slopes. Pyramid pixels use
            # zero initial normalization, then unity gain on the second frame;
            # its expected reconstruction is the configured matrix's row sums.
            @test output ≈ (sequence == 1 ? zeros(Float32, 3) : second_frame)
        end
    end
end

end
