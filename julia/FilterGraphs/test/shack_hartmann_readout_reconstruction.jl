@enum ShackReadoutGateFailure::UInt8 begin
    ShackReadoutGateRejected = 1
end

struct ShackReadoutGatePlan end

@algorithm ShackReadoutGateF32 begin
    label = "shack-readout-gate-f32"
    failure = ShackReadoutGateFailure

    @configuration ShackReadoutGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :reconstructed,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            sample=:reconstructed_sample,
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            metadata_from=:reconstructed,
        ),
    )

    prepare(_configuration) = ShackReadoutGatePlan()

    process!(_plan) = begin
        !isnothing(reconstructed_sample) && isodd(reconstructed_sample.sequence) &&
            return ShackReadoutGateRejected
        copyto!(accepted, reconstructed)
        nothing
    end
end

const SHACK_READOUT_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = shwfs-readout-reconstructor-f32
            config = {
                detector_size = [6, 6]
                region_lines = 3
                block_lines = 1
                region_samples = 6
                detector_origins = [ [0 0] [5 0] ]
                detector_line_steps = [ [1 0] [-1 0] ]
                detector_sample_steps = [ [0 1] [0 1] ]
                subaperture_size = [2, 2]
                subaperture_count = 4
                reconstructed_count = 3
                initial_subaperture_origins = [ [0 0] [0 3] [3 0] [3 3] ]
                coordinate_scale = 1.0
                pixel_threshold = 0.0
                flux_threshold = 0.0
                reference_slopes = [ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ]
                active = [ true true true true ]
                initial_reconstructor = [
                    0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
                    0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16
                    0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
        {
            type = ndarray
            name = gate
            label = shack-readout-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "reconstruct:reconstructed" input = "gate:reconstructed" }
    ]
    inputs = [
        "reconstruct:readout-block"
    ]
    outputs = [ "gate:accepted" ]
}
"""

const SHACK_READOUT_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_readout_reconstruction.toml"),
)

const SHACK_READOUT_BLOCK_PORT = Symbol("readout-block")
const SHACK_READOUT_INPUT_NAMES = (SHACK_READOUT_BLOCK_PORT,)

shack_readout_inputs(block) = NamedTuple{SHACK_READOUT_INPUT_NAMES}((block,))
shack_readout_metadata(sample) = NamedTuple{SHACK_READOUT_INPUT_NAMES}((sample,))

function shack_readout_block(::Type{T}, values) where {T}
    block = Array{T}(undef, 2, 1, 6)
    source = 1
    for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        block[region, line, sample] = T(values[source])
        source += 1
    end
    return block
end

shack_readout_blocks() = map(SHACK_READOUT_FIXTURE["blocks"]) do block
    shack_readout_block(Float32, block["values"])
end

function process_shack_readout_graph_frame!(outputs, graph, sequence)
    availability = nothing
    blocks = shack_readout_blocks()
    for (index, block) in enumerate(blocks)
        ordinal = index - 1
        sample = SampleMetadata(
            sequence,
            ordinal;
            timestamp_nanoseconds=5000 + ordinal,
            terminal=index == length(blocks),
        )
        availability = process!(
            outputs,
            graph,
            shack_readout_inputs(block),
            shack_readout_metadata(sample),
        )
        @test availability == (accepted=index == length(blocks),)
    end
    return availability
end

function shack_readout_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function shack_readout_graph_without_metadata_allocation_bytes(outputs, graph, inputs)
    return @allocated process!(outputs, graph, inputs)
end

@testset "readout Shack-Hartmann reconstruction is a conditional Graph path" begin
    graph = prepare_graph_text(
        SHACK_READOUT_GRAPH_TEXT;
        algorithms=(ShackHartmannReadoutReconstructorF32, ShackReadoutGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:reconstruct, :gate)
    @test graph.node_failure_types == (
        ShackHartmannReadoutReconstructionFailure,
        ShackReadoutGateFailure,
    )
    @test keys(graph.input_formats) == SHACK_READOUT_INPUT_NAMES
    @test getproperty(graph.input_formats, SHACK_READOUT_BLOCK_PORT).shape == (2, 1, 6)
    @test getproperty(graph.input_formats, SHACK_READOUT_BLOCK_PORT).rate ==
          (0x00000003, 0x00000001)
    @test graph.output_conditional == (accepted=true,)
    @test isempty(graph.parameter_formats)

    accepted = fill(-1.0f0, 3)
    outputs = (accepted=accepted,)
    process_shack_readout_graph_frame!(outputs, graph, 100)
    @test accepted ≈ Float32.(
        SHACK_READOUT_FIXTURE["expected"]["reconstructed"],
    ) atol = 2.0f-6
    @test FilterGraphs.output_metadata(graph).accepted == SampleMetadata(
        100,
        0;
        timestamp_nanoseconds=5002,
        terminal=true,
    )
end

@testset "downstream rejection preserves outputs after readout reconstruction commits" begin
    graph = prepare_graph_text(
        SHACK_READOUT_GRAPH_TEXT;
        algorithms=(ShackHartmannReadoutReconstructorF32, ShackReadoutGateF32),
    )
    accepted = zeros(Float32, 3)
    outputs = (accepted=accepted,)
    process_shack_readout_graph_frame!(outputs, graph, 110)
    retained = copy(accepted)
    retained_metadata = FilterGraphs.output_metadata(graph)
    blocks = shack_readout_blocks()
    for index in 1:2
        @test process!(
            outputs,
            graph,
            shack_readout_inputs(blocks[index]),
            shack_readout_metadata(SampleMetadata(111, index - 1)),
        ) == (accepted=false,)
    end
    rejection = process!(
        outputs,
        graph,
        shack_readout_inputs(blocks[3]),
        shack_readout_metadata(SampleMetadata(111, 2; terminal=true)),
    )
    @test rejection === ShackReadoutGateRejected
    @test accepted == retained
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test !graph.nodes[1].prepared.workspace.frame.active

    process_shack_readout_graph_frame!(outputs, graph, 112)
    @test !FilterGraphs.output_metadata(graph).accepted.discontinuity
end

@testset "readout Graph success, rejection, and deferral allocate nothing" begin
    graph = prepare_graph_text(
        SHACK_READOUT_GRAPH_TEXT;
        algorithms=(ShackHartmannReadoutReconstructorF32, ShackReadoutGateF32),
    )
    blocks = shack_readout_blocks()
    inputs = map(shack_readout_inputs, blocks)
    outputs = (accepted=zeros(Float32, 3),)

    @test shack_readout_graph_allocation_bytes(
        outputs,
        graph,
        inputs[1],
        shack_readout_metadata(SampleMetadata(120, 0)),
    ) == 0
    process!(
        outputs,
        graph,
        inputs[2],
        shack_readout_metadata(SampleMetadata(120, 1)),
    )
    @test shack_readout_graph_allocation_bytes(
        outputs,
        graph,
        inputs[3],
        shack_readout_metadata(SampleMetadata(120, 2; terminal=true)),
    ) == 0

    for index in 1:2
        process!(
            outputs,
            graph,
            inputs[index],
            shack_readout_metadata(SampleMetadata(121, index - 1)),
        )
    end
    @test shack_readout_graph_allocation_bytes(
        outputs,
        graph,
        inputs[3],
        shack_readout_metadata(SampleMetadata(121, 2; terminal=true)),
    ) == 0
    @test shack_readout_graph_without_metadata_allocation_bytes(
        outputs,
        graph,
        inputs[1],
    ) == 0
end
