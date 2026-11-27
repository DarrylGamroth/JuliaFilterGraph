module CompositionTests
using Test, FilterGraphAlgorithms, FilterGraphs
include("composition_fixture.jl")
using .CompositionFixture: CompositionGateF32, composition_fixture
const FG = FilterGraphs
const FGA = FilterGraphAlgorithms

@testset "cold composition selection preserves concrete fields and identity" begin
    buffer = Float32[1, 2]
    flag = Ref(false)
    values = (buffer, flag, 3, (:signal,))
    indices = [2, 1, 4, 1]
    selected = FG._composition_select(values, indices)
    @test typeof(selected) === Tuple{typeof(flag),typeof(buffer),Tuple{Symbol},typeof(buffer)}
    @test isconcretetype(typeof(selected))
    @test all(isconcretetype, fieldtypes(typeof(selected)))
    @test selected[1] === flag
    @test selected[2] === buffer
    @test selected[3] === values[4]
    @test selected[4] === buffer
    @test indices == [2, 1, 4, 1]
    @test FG._composition_select(values, Int[]) === ()
    @test_throws BoundsError FG._composition_select(values, [0])
end

allocate_outputs(c) = map(f -> zeros(f.element_type,f.shape),FG.output_formats(c))
function composition_allocation_bytes(outputs,c,arrivals,metadata,previous)
    # Reuse the same owned metadata while admitting a new sample; metadata
    # construction and caller setup are outside the processing measurement.
    fill!(c.last_inputs,previous)
    return @allocated FG.process!(outputs,c,arrivals,metadata)
end
function step!(outputs,c,name,value,sequence; generation=1,domain=UInt8(name == :a ? 0x51 : 0x52),corrupted=false)
    camera = Symbol(name,"-image")
    metadata = SampleMetadata(sequence;terminal=true,corrupted,
        acquisition_identity=SampleAcquisitionIdentity(ntuple(_ -> domain,16),generation,sequence),
        reference_timestamp=SampleReferenceTimestamp(UInt64(domain)*1_000_000_000+sequence))
    return FG.process!(outputs,c,NamedTuple{(camera,)}(([Float32(value)],)),NamedTuple{(camera,)}((metadata,)))
end

@testset "independent branches and shared feedback" begin
    mktempdir() do directory
        path = composition_fixture(directory)
        c = FG.prepare_composition(path;algorithms=(FGA.algorithms()...,CompositionGateF32))
        outputs = allocate_outputs(c)
        if isdefined(FGA, :_process_finite_combination!)
            condition = isdefined(FGA, :_process_finite_conditioning!) ?
                FGA._process_finite_conditioning! : FGA._process_finite_input!
            @test c.shared_stage.execution === (FGA._process_admitted!, FGA._process_finite_input!,
                condition, FGA._process_finite_input!,
                FGA._process_finite_combination!, FGA._process_finite_combination!)
        end
        @test keys(FG.input_formats(c)) == (Symbol("a-image"),Symbol("b-image"))
        @test FG.output_formats(c).command.schema == FGA.DEMANDED_PDM_COMMAND_V1
        @test FG.output_formats(c).requested.schema == FGA.CONTROLLER_COMMAND_V1
        sources=NamedTuple{keys(FG.output_formats(c))}(FG.output_metadata_source_inputs(c))
        @test sources.command==0
        @test getproperty(sources,Symbol("a-correction"))==1
        @test getproperty(sources,Symbol("b-correction"))==2
        @test !(Symbol("a-returned") in keys(FG.output_formats(c)))
        # B can run before A has supplied even a priming sample.
        result = step!(outputs,c,:b,.2,1)
        @test result.command
        @test outputs.command ≈ [.2f0]
        @test !result[Symbol("a-correction")]
        @test !step!(outputs,c,:a,-99,1).command
        @test FG.composition_observation(c).sequence == 1
        @test step!(outputs,c,:b,.2,2).command
        @test outputs.command ≈ [.4f0]
        @test step!(outputs,c,:a,.4,2).command
        @test outputs.requested ≈ [.8f0]
        @test outputs.command ≈ [.5f0]
        held_b = copy(outputs[Symbol("b-correction")])
        # Only A consumes its newest clipping share; B remains held.
        @test step!(outputs,c,:a,0,3).command
        @test outputs[Symbol("a-correction")] ≈ [.25f0]
        @test outputs[Symbol("b-correction")] == held_b
        @test !step!(outputs,c,:a,0,3).command # duplicate must not replay feedback
        @test step!(outputs,c,:a,0,4).command
        @test outputs[Symbol("a-correction")] ≈ [.175f0]
        @test step!(outputs,c,:b,-1,3).command
        @test outputs[Symbol("b-correction")] == held_b
        observed = FG.composition_observation(c)
        @test observed.trigger == (:b,)
        @test observed.contributors.a.sequence == 4
        @test observed.contributors.b.sequence == 3
        @test FG.output_metadata(c).command.has_acquisition_identity
        @test FG.output_metadata(c).command.acquisition_identity.domain != observed.contributors.a.acquisition_identity.domain
        @test FG.output_metadata(c).command.sequence == observed.sequence
        @test sample_reference_timestamp(FG.output_metadata(c).command) ==
            sample_reference_timestamp(observed.contributors.b)
        FG.replace_properties!(c,Symbol("a-integrator:closed")=>false)
        @test step!(outputs,c,:a,.2,5).command
        @test outputs[Symbol("a-correction")] ≈ [.175f0]
        FG.reset!(c)
        @test FG.composition_observation(c).sequence == 0
        @test all(isnothing,values(FG.composition_observation(c).contributors))
        @test step!(outputs,c,:b,0,1).command
        @test outputs.command == [0f0]
        FG.replace_properties!(c,Symbol("a-integrator:closed")=>true)
        FG.reset!(c)
        @test step!(outputs,c,:a,0,10;generation=2).command
        @test !step!(outputs,c,:a,.1,11;generation=1).command
        @test !step!(outputs,c,:a,.1,9;generation=2).command
        @test !step!(outputs,c,:a,.1,11;generation=2,domain=0x61).command
        @test step!(outputs,c,:a,.1,1;generation=3).command
        @test !step!(outputs,c,:a,.1,2;generation=3,corrupted=true).command
        @test FG.composition_observation(c).contributors.a.acquisition_identity.generation == 3
        @test outputs.command ≈ [.1f0]
        FG.reset!(c)
        @test FG.replace_parameters!(c,Symbol("a-reference")=>[.1f0]) === c
        @test step!(outputs,c,:b,0,1).command
        @test outputs.command == [0f0]
        @test step!(outputs,c,:a,0,1).command
        @test outputs.command ≈ [.1f0]
        @test_throws DimensionMismatch FG.replace_parameters!(c,Symbol("a-reference")=>[0f0,1f0])
        @test FG.replace_parameters!(c,Symbol("a-reference")=>[0f0]) === c
        FG.reset!(c)
        # Non-priming branch rejection requires a global reset.
        @test !step!(outputs,c,:a,-99,2).command
        priming = FG.rejection_context(c)
        @test priming.inputs.var"a-image".sequence == 2
        bad = SampleMetadata(3; reference_timestamp=SampleReferenceTimestamp(300))
        @test_throws InvalidStateException FG.process!(outputs, c, (;var"a-image"=Float32[Inf]),
            (;var"a-image"=bad))
        rejected = FG.rejection_context(c)
        @test rejected.node === Symbol("a-reference")
        @test rejected.reason === VectorLinearCombinationNonFiniteInput
        @test rejected.inputs == (;var"a-image"=bad, var"a-feedback"=nothing,
            var"b-image"=nothing, var"b-feedback"=nothing)
        @test FG.composition_observation(c).failed
        @test_throws InvalidStateException step!(outputs,c,:b,0,2)
        FG.reset!(c)
        @test step!(outputs,c,:b,0,1).command
        FG.reset!(c)
        sample_a = SampleMetadata(10; reference_timestamp=SampleReferenceTimestamp(1000))
        sample_b = SampleMetadata(20; reference_timestamp=SampleReferenceTimestamp(2000))
        largest = Float32[floatmax(Float32)]
        @test FG.process!(outputs, c, (;var"a-image"=largest), (;var"a-image"=sample_a)).command
        old_identity = FG.composition_observation(c).identity
        # Both branches commit finite corrections, but their summed demand
        # overflows. No partially advanced command may be published or resumed.
        @test_throws InvalidStateException FG.process!(outputs, c, (;var"b-image"=largest),
            (;var"b-image"=sample_b))
        shared = FG.rejection_context(c)
        @test shared.node === :sum
        @test shared.reason === VectorLinearCombinationNonFiniteOutput
        @test shared.inputs == (;var"a-image"=sample_a, var"a-feedback"=nothing,
            var"b-image"=sample_b, var"b-feedback"=nothing)
        @test priming.inputs.var"a-image".sequence == 2
        @test FG.composition_observation(c).failed
        @test FG.composition_observation(c).identity == old_identity
        @test FG.composition_observation(c).contributors.b === nothing
        @test outputs.command == [.5f0]
        @test_throws InvalidStateException step!(outputs,c,:a,0,2)
        FG.reset!(c)
        @test step!(outputs,c,:b,0,1).command
        @test outputs.command == [0f0]
        FG.reset!(c)
        canonical = allocate_outputs(c)
        reversed = NamedTuple{reverse(keys(canonical))}(reverse(values(canonical)))
        FG.process!(reversed,c,(;var"a-image"=[.4f0],var"b-image"=[.4f0]))
        @test reversed.command ≈ [.5f0]
        @test reversed.requested ≈ [.8f0]
        @test reversed[Symbol("a-correction")] ≈ [.4f0]
        @test reversed[Symbol("b-correction")] ≈ [.4f0]
    end
end

@testset "potentially overflowing return remains checked" begin
    mktempdir() do directory
        descriptor = composition_fixture(directory)
        shared = joinpath(directory, "shared.conf")
        write(shared, replace(read(shared, String),
            "left_scale = 0.5 right_scale = 0.0" => "left_scale = 2.0 right_scale = 0.0"; count=1))
        c = FG.prepare_composition(descriptor; algorithms=(FGA.algorithms()..., CompositionGateF32))
        if isdefined(FGA, :_process_finite_combination!)
            condition = isdefined(FGA, :_process_finite_conditioning!) ?
                FGA._process_finite_conditioning! : FGA._process_finite_input!
            @test c.shared_stage.execution === (FGA._process_admitted!, FGA._process_finite_input!,
                condition, FGA._process_finite_input!,
                FGA._process_admitted!, FGA._process_finite_combination!)
        end
        outputs = allocate_outputs(c)
        @test step!(outputs, c, :a, 0.75, 1).command
        @test c.branches[1].feedback_buffer == [0.5f0]
        @test c.branches[2].feedback_buffer == [0.125f0]
        previous = FG.composition_observation(c)
        previous_outputs = deepcopy(outputs)
        previous_metadata = FG.output_metadata(c)
        previous_feedback = map(branch -> copy(branch.feedback_buffer), c.branches)

        @test_throws InvalidStateException step!(outputs, c, :b, floatmax(Float32), 1)
        rejection = FG.rejection_context(c)
        @test rejection.node === Symbol("a-return")
        @test rejection.reason === VectorLinearCombinationNonFiniteOutput
        @test FG.composition_observation(c) == merge(previous, (; failed=true))
        @test isequal(outputs, previous_outputs)
        @test FG.output_metadata(c) == previous_metadata
        @test map(branch -> branch.feedback_buffer, c.branches) == previous_feedback
    end
end

@testset "finite-return preparation requires proof for the zero-scaled operand" begin
    mktemp() do path, io
        schema = FGA.CONTROLLER_CONSTRAINT_FEEDBACK_V1
        write(io, """
        filter.graph = {
          nodes = [
            {name = validated type = ndarray label = vector-linear-combination-f32
             config = {extent = 1 left_scale = 1.0 right_scale = 1.0 schema = "$schema"}}
            {name = transform type = ndarray label = vector-linear-combination-f32
             config = {extent = 1 left_scale = 0.5 right_scale = 0.0 schema = "$schema"}}
          ]
          links = [{output = validated:output input = transform:left}]
          inputs = [{name = left endpoint = validated:left}
                    {name = right endpoint = validated:right}
                    {name = unproven endpoint = transform:right}]
          outputs = [{name = result endpoint = transform:output}]
        }
        """)
        close(io)
        graph = FG.prepare_graph(path; algorithms=FGA.algorithms())
        inputs = FG._external_buffers(graph.input_formats)
        stage = FG._prepare_cpu_shared_stage(graph, (1, 2), inputs)
        @test stage.execution === nothing
    end
end

@testset "limiter overflow preserves the last shared publication and feedback" begin
    mktempdir() do directory
        descriptor = composition_fixture(directory)
        shared = joinpath(directory, "shared.conf")
        write(shared, replace(read(shared, String),
            "command_limits = [-0.5,0.5]" => "command_limits = [-3e38,-3e38]",
            "initial_current = null" => "initial_current = [-3e38]"))
        c = FG.prepare_composition(descriptor; algorithms=(FGA.algorithms()..., CompositionGateF32))
        if isdefined(FGA, :_process_finite_input!)
            limiter = findfirst(node -> node.name === :limit, c.shared_stage.nodes)
            @test c.shared_stage.execution[limiter] === FGA._process_finite_input!
        end
        outputs = allocate_outputs(c)
        @test step!(outputs, c, :b, 0, 1).command
        previous_outputs = deepcopy(outputs)
        previous_metadata = FG.output_metadata(c)
        previous = FG.composition_observation(c)
        feedback = map(branch -> copy(branch.feedback_buffer), c.branches)
        @test all(slot -> only(slot) > 0f0, feedback)

        # The sum is finite. Subtracting the negative limit from it overflows.
        @test_throws InvalidStateException step!(outputs, c, :a, floatmax(Float32), 1)
        rejection = FG.rejection_context(c)
        @test rejection.node === :limit
        @test rejection.reason === PdmCommandNonFiniteOutput
        @test FG.composition_observation(c) == merge(previous, (; failed=true))
        @test FG.output_metadata(c) == previous_metadata
        @test isequal(outputs, previous_outputs)
        @test map(branch -> branch.feedback_buffer, c.branches) == feedback

        FG.reset!(c)
        @test step!(outputs, c, :b, 0, 1).command
        @test isequal(outputs, previous_outputs)
    end
end

@testset "limiter specialization requires a proven route and disabled slew" begin
    mktempdir() do directory
        descriptor = composition_fixture(directory)
        shared = joinpath(directory, "shared.conf")
        source = read(shared, String)
        algorithms = (FGA.algorithms()..., CompositionGateF32)
        c = FG.prepare_composition(descriptor; algorithms)
        limiter = findfirst(node -> node.name === :limit, c.shared_stage.nodes)
        # A stage starting at the limiter has no successful writer to prove its input.
        limiter_only = FG._prepare_cpu_shared_stage(c.graph, (c.shared_stage.indices[limiter],), c.inputs)
        @test limiter_only.execution === nothing
        write(shared, replace(source, "slew_limit = null" => "slew_limit = 0.25"))
        checked = FG.prepare_composition(descriptor; algorithms)
        if isdefined(FGA, :_process_finite_input!)
            @test checked.shared_stage.execution[limiter] === FGA._process_finite_input!
        end
    end
end

@testset "composition retains feedback across caller output slots" begin
    mktempdir() do directory
        c = FG.prepare_composition(composition_fixture(directory);
            algorithms=(FGA.algorithms()..., CompositionGateF32))
        # The prepared owner needs no second set of all exported payloads.
        @test !hasfield(typeof(c), :outputs)
        first_slot = allocate_outputs(c)
        samples = (; var"a-image"=SampleMetadata(10), var"b-image"=SampleMetadata(7))
        published = FG.process!(first_slot, c,
            (; var"a-image"=[0.4f0], var"b-image"=[0.4f0]), samples)
        @test all(values(published))
        @test first_slot.requested ≈ [0.8f0]
        @test first_slot.command == [0.5f0]

        # The caller recycles every published payload before A arrives again.
        foreach(array -> fill!(array, -17f0), first_slot)
        next_slot = map(array -> fill(42f0, size(array)), first_slot)
        published = step!(next_slot, c, :a, 0, 11)
        @test published.var"a-correction" && published.requested && published.command
        @test !published.var"b-correction"
        @test next_slot.var"a-correction" ≈ [0.25f0]  # A consumed its half of the clipping feedback.
        @test next_slot.requested ≈ [0.65f0]          # B's retained 0.4 still reaches the sum.
        @test next_slot.command == [0.5f0]
        @test next_slot.var"b-correction" == [42f0]
        @test FG.output_metadata(c).var"b-correction" == samples.var"b-image"
        @test FG.output_metadata(c).command.sequence == 2
        @test FG.composition_observation(c).trigger == (:a,)
        @test all(array -> only(array) == -17f0, first_slot)

        FG.reset!(c)
        @test next_slot.command == [0.5f0]
        @test step!(next_slot, c, :b, 0, 1).command
        @test next_slot.command == [0f0]
    end
end

@testset "incomplete-pair recovery before correction allocates zero bytes" begin
    mktempdir() do directory
        descriptor = composition_fixture(directory)
        branch = joinpath(directory, "a.conf")
        config = read(branch, String)
        input_node = "{name=a-input type=ndarray label=vector-offset-f32 config={extent=1 schema=\"$(FGA.CONTROLLER_RESIDUAL_ERROR_V1)\"}}"
        write(branch, replace(config,
            "nodes = [" => "nodes = [$input_node ",
            "links = [" => "links = [{output=a-input:output input=a-gate:input} ",
            "name = a-image endpoint = a-gate:input" => "name = a-image endpoint = a-input:requested"))
        c = FG.prepare_composition(descriptor; algorithms=(FGA.algorithms()..., CompositionGateF32))
        outputs = allocate_outputs(c)
        @test step!(outputs, c, :a, 0.2, 1).command
        previous = first(c.last_inputs)
        camera = Symbol("a-image")
        arrivals = NamedTuple{(camera,)}((Float32[-99],))
        sample = SampleMetadata(2; terminal=true,
            acquisition_identity=SampleAcquisitionIdentity(ntuple(_ -> UInt8(0x51), 16), 1, 2))
        metadata = NamedTuple{(camera,)}((sample,))
        composition_allocation_bytes(outputs, c, arrivals, metadata, previous)
        @test composition_allocation_bytes(outputs, c, arrivals, metadata, previous) == 0
        @test outputs.command == [0.2f0]
        @test FG.rejection_context(c.graph).node == Symbol("a-gate")
        @test FG.rejection_context(c.graph).reason === SccPixelResidualIncompletePair
        @test FG.composition_observation(c).sequence == 1
        @test step!(outputs, c, :a, 0.1, 3).command
        @test outputs.command ≈ [0.3f0]
    end
end

@testset "noninteger event cadence, one publication per simultaneous arrival batch" begin
    mktempdir() do directory
        path = composition_fixture(directory)
        write(path,replace(read(path,String),"name = a graph"=>"name = a rate = [7,1] graph", "name = b graph"=>"name = b rate = [3,1] graph"))
        c = FG.prepare_composition(path;algorithms=(FGA.algorithms()...,CompositionGateF32))
        @test c.branches[1].rate == 7//1
        @test c.branches[2].rate == 3//1
        outputs = allocate_outputs(c)
        # 7:3 cadence is represented as independent events, not paired callbacks.
        for sequence in 1:7
            step!(outputs,c,:a,.01,sequence)
            sequence in (2,4,7) && step!(outputs,c,:b,.02,count(x -> x <= sequence,(2,4,7)))
        end
        @test outputs[Symbol("a-correction")] ≈ [.07f0]
        @test outputs[Symbol("b-correction")] ≈ [.06f0]
        @test FG.composition_observation(c).sequence == 10
        a=SampleMetadata(8;reference_timestamp=SampleReferenceTimestamp(900;clock=SAMPLE_CLOCK_TAI))
        b=SampleMetadata(4;reference_timestamp=SampleReferenceTimestamp(100;clock=SAMPLE_CLOCK_REALTIME))
        result = FG.process!(outputs,c,(;var"b-image"=[.02f0],var"a-image"=[.01f0]),
            (;var"b-image"=b,var"a-image"=a))
        @test result.command
        @test FG.composition_observation(c).sequence == 11
        @test FG.composition_observation(c).trigger == (:a,:b)
        @test sample_reference_timestamp(FG.output_metadata(c).command) == sample_reference_timestamp(b)
        @test FG.composition_observation(c).contributors.a.reference_timestamp == a.reference_timestamp
        @test FG.composition_observation(c).contributors.b.reference_timestamp == b.reference_timestamp
        @test outputs.command ≈ [.16f0]
        # Missing timing on the final trigger clears the prior publication fact.
        FG.process!(outputs,c,(;var"a-image"=[.01f0],var"b-image"=[.02f0]),
            (;var"a-image"=SampleMetadata(9;reference_timestamp=a.reference_timestamp),
              var"b-image"=SampleMetadata(5)))
        @test sample_reference_timestamp(FG.output_metadata(c).command) === nothing
        @test_throws ArgumentError FG.process!(outputs,c,(unknown=[0f0],))
        @test !FG.composition_observation(c).failed
    end
end

@testset "composition warmed processing allocation contract" begin
    mktempdir() do directory
        c = FG.prepare_composition(composition_fixture(directory);algorithms=(FGA.algorithms()...,CompositionGateF32))
        outputs = allocate_outputs(c)
        a,b = [0f0],[0f0]
        sample = SampleMetadata(1;terminal=true,reference_timestamp=SampleReferenceTimestamp(7;clock=SAMPLE_CLOCK_MODEL))
        previous = SampleMetadata(0;terminal=true,reference_timestamp=SampleReferenceTimestamp(6;clock=SAMPLE_CLOCK_MODEL))
        for (arrivals,metadata) in (
            ((;var"a-image"=a),(;var"a-image"=sample)),
            ((;var"b-image"=b),(;var"b-image"=sample)),
            ((;var"a-image"=a,var"b-image"=b),(;var"a-image"=sample,var"b-image"=sample)),
        )
            composition_allocation_bytes(outputs,c,arrivals,metadata,previous)
            @test composition_allocation_bytes(outputs,c,arrivals,metadata,previous) == 0
        end
    end
end
end

@testset "composition freshness follows physical acquisition identity" begin
    domain=ntuple(UInt8,16)
    identity(sequence,generation=1)=SampleAcquisitionIdentity(domain,generation,sequence)
    first=SampleMetadata(10;acquisition_identity=identity(7))
    held=SampleMetadata(99;acquisition_identity=identity(7))
    fresh=SampleMetadata(11;acquisition_identity=identity(8))
    @test !FilterGraphs._composition_new_input(first,held)
    @test FilterGraphs._composition_new_input(first,fresh)
    @test FilterGraphs._composition_new_input(first,SampleMetadata(0;acquisition_identity=identity(0,2)))
end
