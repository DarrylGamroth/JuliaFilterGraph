@enum PyramidReadoutGateFailure::UInt8 begin
    PyramidReadoutGateRejected = 1
end

struct PyramidReadoutGatePlan end

@algorithm PyramidReadoutGateF32 begin
    label = "pyramid-readout-gate-f32"
    failure = PyramidReadoutGateFailure

    @configuration PyramidReadoutGateF32Configuration begin
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

    prepare(_configuration) = PyramidReadoutGatePlan()

    process!(_plan) = begin
        !isnothing(reconstructed_sample) && isodd(reconstructed_sample.sequence) &&
            return PyramidReadoutGateRejected
        copyto!(accepted, reconstructed)
        nothing
    end
end

const PYRAMID_READOUT_RECONSTRUCTION_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = reconstruct
            label = pwfs-readout-reconstructor-f32
            config = {
                detector_size = [8, 4]
                region_lines = 4
                block_lines = 1
                region_samples = 4
                detector_origins = [ [0 0] [7 0] ]
                detector_line_steps = [ [1 0] [-1 0] ]
                detector_sample_steps = [ [0 1] [0 1] ]
                pupil_size = [1, 2]
                pupil_origins = [ [7 0] [0 0] [4 2] [2 2] ]
                pupil_mask = [ true false ]
                reconstructed_count = 3
                initial_reconstructor = [
                    -0.15625 -0.125 -0.09375 -0.0625
                    -0.03125 0.0 0.03125 0.0625
                     0.09375 0.125 0.15625 -0.15625
                ]
                reconstructed_schema = pipewireao/controller-residual-error-f32/1
            }
        }
        {
            type = ndarray
            name = gate
            label = pyramid-readout-gate-f32
            config = { extent = 3 }
        }
    ]
    links = [
        { output = "reconstruct:reconstructed" input = "gate:reconstructed" }
    ]
    inputs = [ "reconstruct:readout-block" ]
    outputs = [
        "reconstruct:reconstruction-pixels"
        "reconstruct:mean-pupil-intensity"
        "gate:accepted"
    ]
}
"""

const PYRAMID_READOUT_RECONSTRUCTION_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_readout_reconstruction.toml"),
)

const PYRAMID_RECONSTRUCTION_READOUT_BLOCK = Symbol("readout-block")
const PYRAMID_READOUT_RECONSTRUCTION_PIXELS = Symbol("reconstruction-pixels")
const PYRAMID_READOUT_RECONSTRUCTION_MEAN = Symbol("mean-pupil-intensity")
const PYRAMID_READOUT_RECONSTRUCTION_INPUT_NAMES =
    (PYRAMID_RECONSTRUCTION_READOUT_BLOCK,)
const PYRAMID_READOUT_RECONSTRUCTION_OUTPUT_NAMES = (
    PYRAMID_READOUT_RECONSTRUCTION_PIXELS,
    PYRAMID_READOUT_RECONSTRUCTION_MEAN,
    :accepted,
)

pyramid_readout_reconstruction_inputs(readout_block) =
    NamedTuple{PYRAMID_READOUT_RECONSTRUCTION_INPUT_NAMES}((readout_block,))
pyramid_readout_reconstruction_metadata(sample) =
    NamedTuple{PYRAMID_READOUT_RECONSTRUCTION_INPUT_NAMES}((sample,))
pyramid_readout_reconstruction_outputs(pixels, mean, accepted) =
    NamedTuple{PYRAMID_READOUT_RECONSTRUCTION_OUTPUT_NAMES}((pixels, mean, accepted))
pyramid_readout_reconstruction_availability(pixels, mean, accepted) =
    NamedTuple{PYRAMID_READOUT_RECONSTRUCTION_OUTPUT_NAMES}((pixels, mean, accepted))

function pyramid_readout_reconstruction_block(::Type{T}, values) where {T}
    block = Array{T}(undef, 2, 1, 4)
    source = 1
    for region in axes(block, 1), line in axes(block, 2), sample in axes(block, 3)
        block[region, line, sample] = T(values[source])
        source += 1
    end
    return block
end

function pyramid_readout_reconstruction_blocks(frame)
    packed = frame["packed"]
    return ntuple(4) do index
        first = (index - 1) * 8 + 1
        pyramid_readout_reconstruction_block(
            Float32,
            @view(packed[first:(first + 7)]),
        )
    end
end

function process_pyramid_readout_reconstruction_frame!(
    outputs,
    graph,
    frame,
    sequence,
)
    availability = nothing
    blocks = pyramid_readout_reconstruction_blocks(frame)
    for (index, block) in enumerate(blocks)
        ordinal = index - 1
        sample = SampleMetadata(
            sequence,
            ordinal;
            timestamp_nanoseconds=8000 + ordinal,
            terminal=index == length(blocks),
        )
        availability = process!(
            outputs,
            graph,
            pyramid_readout_reconstruction_inputs(block),
            pyramid_readout_reconstruction_metadata(sample),
        )
        if iseven(sequence)
            expected = index == length(blocks) ?
                pyramid_readout_reconstruction_availability(true, true, true) :
                pyramid_readout_reconstruction_availability(false, false, false)
            @test availability == expected
        elseif index < length(blocks)
            @test availability ==
                  pyramid_readout_reconstruction_availability(false, false, false)
        end
    end
    return availability
end

function pyramid_readout_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function pyramid_readout_graph_without_metadata_allocation_bytes(
    outputs,
    graph,
    inputs,
)
    return @allocated process!(outputs, graph, inputs)
end

@testset "readout pyramid reconstruction is a conditional Graph path" begin
    graph = prepare_graph_text(
        PYRAMID_READOUT_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidReadoutReconstructorF32, PyramidReadoutGateF32),
    )
    @test map(node -> node.name, graph.nodes) == (:reconstruct, :gate)
    @test graph.node_failure_types == (
        PyramidReadoutReconstructionFailure,
        PyramidReadoutGateFailure,
    )
    @test keys(graph.input_formats) == PYRAMID_READOUT_RECONSTRUCTION_INPUT_NAMES
    input_format = getproperty(
        graph.input_formats,
        PYRAMID_RECONSTRUCTION_READOUT_BLOCK,
    )
    @test input_format.shape == (2, 1, 4)
    @test input_format.rate == (0x00000004, 0x00000001)
    @test graph.output_conditional ==
          pyramid_readout_reconstruction_availability(true, true, true)
    @test isempty(graph.parameter_formats)

    pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT, 1)
    mean = Float32[-2]
    accepted = fill(-3.0f0, 3)
    outputs = pyramid_readout_reconstruction_outputs(pixels, mean, accepted)
    first_frame, second_frame = PYRAMID_READOUT_RECONSTRUCTION_FIXTURE["frames"]

    process_pyramid_readout_reconstruction_frame!(outputs, graph, first_frame, 100)
    @test pixels == zeros(Float32, PYRAMID_PUPIL_COUNT, 1)
    @test mean == Float32[15.75]
    @test accepted == zeros(Float32, 3)
    expected_metadata = SampleMetadata(
        100,
        0;
        timestamp_nanoseconds=8003,
        terminal=true,
    )
    @test FilterGraphs.output_metadata(graph) == NamedTuple{
        PYRAMID_READOUT_RECONSTRUCTION_OUTPUT_NAMES,
    }((expected_metadata, expected_metadata, expected_metadata))

    process_pyramid_readout_reconstruction_frame!(outputs, graph, second_frame, 102)
    @test reinterpret(UInt32, vec(pixels)) ==
          UInt32.(second_frame["expected_reconstruction_pixel_bits"])
    @test reinterpret(UInt32, mean)[1] ==
          UInt32(second_frame["expected_mean_pupil_intensity_bits"])
    @test reinterpret(UInt32, accepted) ==
          UInt32.(second_frame["expected_reconstructed_bits"])
end

@testset "downstream rejection preserves outputs after readout reconstruction commits" begin
    graph = prepare_graph_text(
        PYRAMID_READOUT_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidReadoutReconstructorF32, PyramidReadoutGateF32),
    )
    pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 1)
    mean = zeros(Float32, 1)
    accepted = zeros(Float32, 3)
    outputs = pyramid_readout_reconstruction_outputs(pixels, mean, accepted)
    first_frame, second_frame = PYRAMID_READOUT_RECONSTRUCTION_FIXTURE["frames"]
    process_pyramid_readout_reconstruction_frame!(outputs, graph, first_frame, 110)
    retained_pixels = copy(pixels)
    retained_mean = copy(mean)
    retained_accepted = copy(accepted)
    retained_metadata = FilterGraphs.output_metadata(graph)
    workspace = graph.nodes[1].prepared.workspace
    committed_gain = workspace.committed_gain

    rejection = process_pyramid_readout_reconstruction_frame!(
        outputs,
        graph,
        second_frame,
        111,
    )
    @test rejection === PyramidReadoutGateRejected
    @test pixels == retained_pixels
    @test mean == retained_mean
    @test accepted == retained_accepted
    @test FilterGraphs.output_metadata(graph) == retained_metadata
    @test workspace.committed_gain != committed_gain
    @test !workspace.frame.active
    @test iszero(workspace.candidate_reconstructed)
    @test iszero(workspace.candidate_reconstruction_pixels)

    process_pyramid_readout_reconstruction_frame!(outputs, graph, second_frame, 112)
    metadata = FilterGraphs.output_metadata(graph)
    @test !getproperty(metadata, PYRAMID_READOUT_RECONSTRUCTION_PIXELS).discontinuity
    @test !getproperty(metadata, PYRAMID_READOUT_RECONSTRUCTION_MEAN).discontinuity
    @test !metadata.accepted.discontinuity
    @test accepted != retained_accepted
end

@testset "readout pyramid Graph outcomes allocate nothing after warmup" begin
    graph = prepare_graph_text(
        PYRAMID_READOUT_RECONSTRUCTION_GRAPH_TEXT;
        algorithms=(PyramidReadoutReconstructorF32, PyramidReadoutGateF32),
    )
    first_frame, second_frame = PYRAMID_READOUT_RECONSTRUCTION_FIXTURE["frames"]
    first_blocks = pyramid_readout_reconstruction_blocks(first_frame)
    second_blocks = pyramid_readout_reconstruction_blocks(second_frame)
    first_inputs = map(pyramid_readout_reconstruction_inputs, first_blocks)
    second_inputs = map(pyramid_readout_reconstruction_inputs, second_blocks)
    outputs = pyramid_readout_reconstruction_outputs(
        zeros(Float32, PYRAMID_PUPIL_COUNT, 1),
        zeros(Float32, 1),
        zeros(Float32, 3),
    )

    @test pyramid_readout_graph_allocation_bytes(
        outputs,
        graph,
        first_inputs[1],
        pyramid_readout_reconstruction_metadata(SampleMetadata(120, 0)),
    ) == 0
    for index in 2:3
        process!(
            outputs,
            graph,
            first_inputs[index],
            pyramid_readout_reconstruction_metadata(
                SampleMetadata(120, index - 1),
            ),
        )
    end
    @test pyramid_readout_graph_allocation_bytes(
        outputs,
        graph,
        first_inputs[4],
        pyramid_readout_reconstruction_metadata(
            SampleMetadata(120, 3; terminal=true),
        ),
    ) == 0

    for index in 1:3
        process!(
            outputs,
            graph,
            second_inputs[index],
            pyramid_readout_reconstruction_metadata(
                SampleMetadata(121, index - 1),
            ),
        )
    end
    @test pyramid_readout_graph_allocation_bytes(
        outputs,
        graph,
        second_inputs[4],
        pyramid_readout_reconstruction_metadata(
            SampleMetadata(121, 3; terminal=true),
        ),
    ) == 0
    @test pyramid_readout_graph_allocation_bytes(
        outputs,
        graph,
        second_inputs[2],
        pyramid_readout_reconstruction_metadata(SampleMetadata(122, 1)),
    ) == 0
    @test pyramid_readout_graph_without_metadata_allocation_bytes(
        outputs,
        graph,
        second_inputs[1],
    ) == 0
end
