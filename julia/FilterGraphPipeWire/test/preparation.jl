module CpuPreparationTests

using FilterGraphAlgorithms
using FilterGraphPipeWire
using FilterGraphs
using PipeWireAO
using Test

const conditional_process_count = Ref(0)

@algorithm WarmupConditionalF32 begin
    label = "warmup-conditional-f32"

    @configuration WarmupConditionalF32Configuration begin
        publish::Bool => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (1,), "test.warmup.signal/1"),
        output(:output, Float32, (1,), "test.warmup.signal/1";
               publication=:conditional, metadata_from=:input),
    )

    prepare(configuration) = configuration.publish
    process!(publish) = begin
        conditional_process_count[] += 1
        publish || return AlgorithmProcessDeferred()
        copyto!(output, input)
        nothing
    end
end

function chopped_preparation_node()
    graph = mktemp() do path, io
        write(io, """
            filter.graph = {
                nodes = [
                    { type=ndarray name=pair label=chopped-image-difference-f32
                      config={ pixel_count=6 trigger_high_is_fringed=true require_acquisition_identity=true } }
                    { type=ndarray name=residual label=scc-pixel-residual-f32
                      config={ pixel_count=6 mask=[true true true true true true]
                        initial_reference_image=null minimum_valid_fraction=1.0 } }
                ]
                links = [
                    { output="pair:difference" input="residual:difference" }
                    { output="pair:output-validity" input="residual:difference-validity" }
                    { output="pair:pair-status" input="residual:pair-status" }
                ]
                inputs = [ "pair:image" "pair:input-validity" ]
                outputs = [ { name=measurements endpoint="residual:selected-residuals" } ]
            }
            """)
        close(io)
        FilterGraphs.prepare_graph(path; algorithms=FilterGraphAlgorithms.algorithms())
    end
    return PipeWireNode(graph; connect=false)
end

function chopped_preparation_sample(sequence, kind, value, expected)
    identity = SampleAcquisitionIdentity(ntuple(_ -> UInt8(0xa5), 16), UInt64(1), UInt64(sequence))
    sample = SampleMetadata(sequence; terminal=true, chopped_image_kind=kind,
        acquisition_identity=identity, exposure_duration_nanoseconds=10_000_000)
    return (inputs=(image=fill(value, 6), var"input-validity"=fill(true, 6)),
        metadata=(image=sample, var"input-validity"=nothing), expected=expected)
end

@testset "explicit optical preparation primes both SCC pair orders then resets" begin
    node = chopped_preparation_node()
    samples = (
        chopped_preparation_sample(1, ChoppedImageFringed, 4f0, SccPixelResidualIncompletePair),
        chopped_preparation_sample(2, ChoppedImageUnfringed, 1f0, nothing),
        chopped_preparation_sample(3, ChoppedImageFringed, 6f0, nothing),
    )
    try
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; required_outputs=(:measurements,))
        @test FilterGraphPipeWire.prepare!(node; samples, required_outputs=(:measurements,)) === node
        @test !node.owner.active
        @test node.owner.process_count == 0
        @test node_state(node) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        @test all(storage -> all(iszero, storage), values(node.owner.warm_inputs))
        @test all(storage -> all(iszero, storage), values(node.owner.warm_outputs))
        @test all(==(4f0), samples[1].inputs.image)
        outputs = (measurements=zeros(Float32, 6),)
        @test FilterGraphs.process!(outputs, node.owner.graph, samples[1].inputs,
            samples[1].metadata) === SccPixelResidualIncompletePair
        @test FilterGraphs.process!(outputs, node.owner.graph, samples[2].inputs,
            samples[2].metadata).measurements
        @test outputs.measurements == fill(3f0, 6)
        @test FilterGraphPipeWire.prepare!(node; samples, required_outputs=(:measurements,)) === node
        wrong = merge(samples[1], (expected=ChoppedImageTriggerMissing,))
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; samples=(wrong, samples[2]))
        @test !node.owner.active
        @test FilterGraphs.process!(outputs, node.owner.graph, samples[1].inputs,
            samples[1].metadata) === SccPixelResidualIncompletePair
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; samples=())
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; samples=(samples[1],))
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; samples, execution=:row_block)
        malformed = merge(samples[2], (inputs=(image=ones(Float32, 5), var"input-validity"=fill(true, 6)),))
        @test_throws DimensionMismatch FilterGraphPipeWire.prepare!(node; samples=(samples[1], malformed))
        @test !node.owner.active
        @test FilterGraphs.process!(outputs, node.owner.graph, samples[1].inputs,
            samples[1].metadata) === SccPixelResidualIncompletePair
    finally
        close(node)
    end
end

function conditional_node(publish)
    return mktemp() do path, io
        write(io, """
            filter.graph = {
                nodes = [ { type=ndarray name=conditional label=warmup-conditional-f32
                            config={ publish=$publish } } ]
                links = []
                inputs = [ "conditional:input" ]
                outputs = [ "conditional:output" ]
            }
            """)
        close(io)
        graph = FilterGraphs.prepare_graph(path; algorithms=(WarmupConditionalF32,))
        return PipeWireNode(graph; connect=false)
    end
end

@testset "public CPU preparation requires selected conditional outputs" begin
    suppressed = conditional_node(false)
    published = conditional_node(true)
    try
        # Without a selected obligation, optional publication remains legal.
        @test FilterGraphPipeWire.prepare!(suppressed) === suppressed
        # The same suppressed conditional output fails explicit admission.
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(suppressed;
            required_outputs=(:output,))
        @test !suppressed.owner.active
        @test suppressed.owner.process_count == 0
        @test all(iszero, suppressed.owner.warm_inputs.input)
        @test all(iszero, suppressed.owner.warm_outputs.output)
        @test node_state(suppressed) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
        @test isnothing(last_error(suppressed))
        # Enabling the producer satisfies the same admission obligation.
        @test FilterGraphPipeWire.prepare!(published;
            required_outputs=(:output,)) === published
        @test !published.owner.active
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(published;
            required_outputs=(:unknown,))
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(published;
            required_outputs=(:output, :output))
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(published;
            execution=:unsupported)
        @test_throws ArgumentError FilterGraphPipeWire.prepare!(published;
            execution=:row_block)
        @test FilterGraphPipeWire.prepare!(published; required_outputs=(:output,)) === published
    finally
        close(suppressed)
        close(published)
    end
end

@testset "native CPU start does not execute or reset the graph" begin
    node = conditional_node(true)
    try
        conditional_process_count[] = 0
        FilterGraphPipeWire._prepare_owner(node.owner)
        @test node.owner.active
        @test conditional_process_count[] == 0
        FilterGraphPipeWire._deactivate_owner(node.owner)
        FilterGraphPipeWire._prepare_owner(node.owner)
        @test conditional_process_count[] == 0
        @test node.owner.process_count == 0
    finally
        close(node)
    end

    graph = FilterGraphs.prepare_graph(joinpath(@__DIR__, "data", "leaky-graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms())
    node = PipeWireNode(graph; connect=false)
    try
        inputs = (input=ones(Float32, 8),)
        outputs = (output=zeros(Float32, 8),)
        FilterGraphPipeWire._prepare_owner(node.owner)
        FilterGraphs.process!(outputs, graph, inputs)
        @test outputs.output == fill(2.0f0, 8)
        FilterGraphPipeWire._deactivate_owner(node.owner)
        FilterGraphPipeWire._prepare_owner(node.owner)
        FilterGraphs.process!(outputs, graph, inputs)
        @test outputs.output == fill(3.0f0, 8)
    finally
        close(node)
    end
end

@algorithm WarmupFeedbackF32 begin
    label = "warmup-feedback-f32"

    @configuration WarmupFeedbackF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:raw, Float32, (1,), "test.warmup.signal/1"),
        input(:feedback, Float32, (1,), CONTROLLER_CONSTRAINT_FEEDBACK_V1;
              name="constraint-feedback"),
        output(:demanded, Float32, (1,), "test.warmup.signal/1"),
        output(:feedback_output, Float32, (1,), CONTROLLER_CONSTRAINT_FEEDBACK_V1;
               name="controller-constraint-feedback"),
    )

    prepare(configuration) = nothing
    process!(_plan) = begin
        demanded[1] = raw[1] + feedback[1]
        feedback_output[1] = raw[1] + 1.0f0
        nothing
    end
end

@testset "public CPU preparation resets retained feedback" begin
    graph = mktemp() do path, io
        write(io, """
            filter.graph = {
                nodes = [ { type=ndarray name=feedback label=warmup-feedback-f32
                            config={ extent=1 } } ]
                links = []
                inputs = [ "feedback:raw" "feedback:constraint-feedback" ]
                outputs = [ "feedback:demanded" "feedback:controller-constraint-feedback" ]
            }
            """)
        close(io)
        FilterGraphs.prepare_graph(path; algorithms=(WarmupFeedbackF32,))
    end
    node = PipeWireNode(graph; connect=false,
        feedback=Symbol("constraint-feedback") => Symbol("controller-constraint-feedback"))
    try
        node.owner.feedback.slot[1] = 42.0f0
        node.owner.feedback.scratch[1] = -42.0f0
        node.owner.feedback.failed = true
        @test FilterGraphPipeWire.prepare!(node;
            required_outputs=(:demanded, Symbol("controller-constraint-feedback"))) === node
        @test all(iszero, node.owner.feedback.slot)
        @test all(iszero, node.owner.feedback.scratch)
        @test !node.owner.feedback.failed
        @test !node.owner.active
        @test node.owner.process_count == 0
    finally
        close(node)
    end
end

end
