@enum ChoppedDifferenceGateFailure::UInt8 begin
    ChoppedDifferenceGateRejected = 1
end

struct ChoppedDifferenceGatePlan
    extent::Int
end

@algorithm ChoppedDifferenceGateF32 begin
    label = "chopped-difference-gate-f32-test"
    failure = ChoppedDifferenceGateFailure

    @configuration ChoppedDifferenceGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :difference,
            Float32,
            (configuration.extent,),
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        ),
        output(
            :accepted_difference,
            Float32,
            (configuration.extent,),
            SCC_CHOPPED_IMAGE_DIFFERENCE_V1;
            name="accepted-difference",
            metadata_from=:difference,
        ),
    )

    prepare(configuration) = ChoppedDifferenceGatePlan(configuration.extent)

    process!(plan) = begin
        difference[1] < 0.0f0 && return ChoppedDifferenceGateRejected
        copyto!(accepted_difference, difference)
        nothing
    end
end

const CHOPPED_METADATA_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = chopped
            label = chopped-image-difference-f32
            config = {
                pixel_count = 2
                trigger_high_is_fringed = true
            }
        }
        {
            type = ndarray
            name = gate
            label = chopped-difference-gate-f32-test
            config = { extent = 2 }
        }
    ]
    links = [
        {
            output = "chopped:difference"
            input = "gate:difference"
        }
    ]
    inputs = [
        "chopped:image"
        "chopped:input-validity"
    ]
    outputs = [
        "gate:accepted-difference"
        "chopped:output-validity"
        "chopped:pair-status"
    ]
}
"""

const CHOPPED_INPUT_NAMES = (:image, Symbol("input-validity"))
const CHOPPED_OUTPUT_NAMES = (
    Symbol("accepted-difference"),
    Symbol("output-validity"),
    Symbol("pair-status"),
)

chopped_graph_inputs(image, validity) =
    NamedTuple{CHOPPED_INPUT_NAMES}((image, validity))

chopped_graph_outputs(difference, validity, status) =
    NamedTuple{CHOPPED_OUTPUT_NAMES}((difference, validity, status))

chopped_graph_metadata(metadata) =
    NamedTuple{CHOPPED_INPUT_NAMES}((metadata, nothing))

function tagged_raw_image(tag::UInt16)
    image = zeros(UInt16, 8)
    image[4] = tag
    return image
end

function cred2_chopped_metadata(sequence::Integer, tag::UInt16)
    return cred2_trigger_metadata(
        SampleMetadata(sequence, 0; terminal=true),
        tagged_raw_image(tag),
    )
end

function chopped_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function chopped_graph_admitted_allocation_bytes(
    outputs,
    graph,
    inputs,
    metadata,
)
    admitted = map(
        FilterGraphAlgorithms._admitted_sample_metadata,
        metadata,
    )
    return @allocated FilterGraphs._process_admitted!(
        outputs,
        graph,
        inputs,
        admitted,
    )
end

function prepare_chopped_metadata_graph()
    return prepare_graph_text(
        CHOPPED_METADATA_GRAPH_TEXT;
        algorithms=(ChoppedImageDifferenceF32, ChoppedDifferenceGateF32),
    )
end

@testset "C-RED trigger metadata drives chopped differencing in-band" begin
    graph = prepare_chopped_metadata_graph()
    @test map(node -> node.name, graph.nodes) == (:chopped, :gate)
    @test graph.node_failure_types == (
        ChoppedImageDifferenceFailure,
        ChoppedDifferenceGateFailure,
    )
    @test keys(graph.input_formats) == CHOPPED_INPUT_NAMES
    @test keys(graph.output_formats) == CHOPPED_OUTPUT_NAMES
    @test isempty(graph.parameter_formats)

    difference = fill(71.0f0, 2)
    validity = fill(true, 2)
    status = fill(typemax(UInt8), 1)
    outputs = chopped_graph_outputs(difference, validity, status)

    fringed = chopped_graph_inputs(Float32[10, 20], Bool[true, true])
    fringed_metadata = cred2_chopped_metadata(81, 0x3fff)
    @test fringed_metadata isa SampleMetadata
    @test process!(
        outputs,
        graph,
        fringed,
        chopped_graph_metadata(fringed_metadata),
    ) == NamedTuple{CHOPPED_OUTPUT_NAMES}((true, true, true))
    @test difference == zeros(Float32, 2)
    @test validity == Bool[false, false]
    @test status == UInt8[0]
    workspace = graph.nodes[1].prepared.workspace
    @test chopped_image_retained_kind(workspace) == ChoppedImageFringed
    @test chopped_image_retained_image(workspace) == Float32[10, 20]
    @test all(
        value -> value == fringed_metadata,
        values(FilterGraphs.output_metadata(graph)),
    )

    unfringed = chopped_graph_inputs(Float32[3, 8], Bool[true, true])
    unfringed_metadata = cred2_chopped_metadata(82, 0x3ffe)
    @test process!(
        outputs,
        graph,
        unfringed,
        chopped_graph_metadata(unfringed_metadata),
    ) == NamedTuple{CHOPPED_OUTPUT_NAMES}((true, true, true))
    @test difference == Float32[7, 12]
    @test validity == Bool[true, true]
    @test status == UInt8[1]
end

@testset "frame identity supplies the same metadata to the same Graph" begin
    graph = prepare_chopped_metadata_graph()
    outputs = chopped_graph_outputs(
        zeros(Float32, 2),
        fill(false, 2),
        zeros(UInt8, 1),
    )
    even = chopped_graph_inputs(Float32[10, 20], Bool[true, true])
    odd = chopped_graph_inputs(Float32[3, 8], Bool[true, true])
    even_metadata = alternating_trigger_metadata(
        SampleMetadata(20, 0; terminal=true),
    )
    odd_metadata = alternating_trigger_metadata(
        SampleMetadata(21, 0; terminal=true),
    )

    process!(outputs, graph, even, chopped_graph_metadata(even_metadata))
    process!(outputs, graph, odd, chopped_graph_metadata(odd_metadata))
    @test getproperty(outputs, Symbol("accepted-difference")) ==
          Float32[7, 12]
    @test getproperty(outputs, Symbol("output-validity")) == Bool[true, true]
    @test getproperty(outputs, Symbol("pair-status")) == UInt8[1]
end

@testset "downstream rejection preserves outputs and advances chopped history" begin
    graph = prepare_chopped_metadata_graph()
    outputs = chopped_graph_outputs(
        fill(71.0f0, 2),
        fill(true, 2),
        fill(typemax(UInt8), 1),
    )
    fringed = chopped_graph_inputs(Float32[10, 20], Bool[true, true])
    fringed_metadata = cred2_chopped_metadata(81, 0x3fff)
    process!(outputs, graph, fringed, chopped_graph_metadata(fringed_metadata))
    workspace = graph.nodes[1].prepared.workspace

    rejected = chopped_graph_inputs(Float32[30, 30], Bool[true, true])
    unfringed_metadata = cred2_chopped_metadata(82, 0x3ffe)
    retained_outputs = map(copy, outputs)
    retained_metadata = FilterGraphs.output_metadata(graph)
    @test process!(
        outputs,
        graph,
        rejected,
        chopped_graph_metadata(unfringed_metadata),
    ) === ChoppedDifferenceGateRejected
    @test outputs == retained_outputs
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test chopped_image_retained_kind(workspace) == ChoppedImageUnfringed
    @test chopped_image_retained_image(workspace) == Float32[30, 30]

    accepted = chopped_graph_inputs(Float32[3, 8], Bool[true, true])
    @test process!(
        outputs,
        graph,
        accepted,
        chopped_graph_metadata(unfringed_metadata),
    ) == NamedTuple{CHOPPED_OUTPUT_NAMES}((true, true, true))
    @test getproperty(outputs, Symbol("accepted-difference")) ==
          zeros(Float32, 2)
    @test chopped_image_retained_kind(workspace) == ChoppedImageUnfringed
    @test chopped_image_retained_image(workspace) == Float32[3, 8]

    @test cred2_chopped_metadata(83, 0x0001) === Cred2UnexpectedImageTag
    @test chopped_image_retained_image(workspace) == Float32[3, 8]
end

@testset "in-band chopped Graph outcomes allocate nothing after warmup" begin
    graph = prepare_chopped_metadata_graph()
    outputs = chopped_graph_outputs(
        zeros(Float32, 2),
        fill(false, 2),
        zeros(UInt8, 1),
    )
    fringed = chopped_graph_inputs(Float32[10, 20], Bool[true, true])
    accepted = chopped_graph_inputs(Float32[3, 8], Bool[true, true])
    rejected = chopped_graph_inputs(Float32[30, 30], Bool[true, true])
    fringed_metadata = chopped_graph_metadata(cred2_chopped_metadata(81, 0x3fff))
    unfringed_metadata = chopped_graph_metadata(cred2_chopped_metadata(82, 0x3ffe))
    missing_trigger = chopped_graph_metadata(SampleMetadata(83))
    process!(outputs, graph, fringed, fringed_metadata)
    process!(outputs, graph, accepted, unfringed_metadata)
    process!(outputs, graph, rejected, unfringed_metadata)
    process!(outputs, graph, accepted, missing_trigger)

    @test chopped_graph_allocation_bytes(
        outputs,
        graph,
        accepted,
        unfringed_metadata,
    ) == 0
    @test chopped_graph_allocation_bytes(
        outputs,
        graph,
        rejected,
        unfringed_metadata,
    ) == 0
    @test chopped_graph_allocation_bytes(
        outputs,
        graph,
        accepted,
        missing_trigger,
    ) == 0
    @test chopped_graph_admitted_allocation_bytes(
        outputs,
        graph,
        accepted,
        unfringed_metadata,
    ) == 0
end
