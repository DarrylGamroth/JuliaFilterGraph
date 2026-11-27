module ControllerGraphTests
using Test, FilterGraphs, FilterGraphAlgorithms
using FilterGraphs: input_formats, output_formats, parameter_formats
import FilterGraphAlgorithms: process!, reset!
include("controller_fixture.jl")
const F = ControllerFixture

mutable struct Publisher
    calls::Int
    sequence::UInt64
end
function (publisher::Publisher)(command, metadata)
    publisher.calls += 1
    publisher.sequence = metadata === nothing ? 0 : metadata.sequence
    return nothing
end

prepare_source(source, device, resident) = device === nothing ? source :
    prepare_captured_graph(source, device; feedback=resident ? (prior=:feedback,) : NamedTuple())

function run_controller_tests(device=nothing)
    device === nothing && run_borrowed_buffer_tests()
    residents = device === nothing ? (false,) : (false, true)
    @testset "controller default publication trajectories" begin
        for publication in (:early, :complete), resident in residents,
                n in (1,277), pole in (.5f0,.9f0,1f0), aw in unique((pole,1f0))
            reference = prepare_source(F.graph(n,pole,aw;slew=true,flat=.125f0,offset=.25f0),device,resident)
            graph = prepare_source(F.graph(n,pole,aw;slew=true,flat=.125f0,offset=.25f0),device,resident)
            expected=F.buffers(n);actual=F.buffers(n);held=F.buffers(n)
            error=zeros(Float32,n);prior=zeros(Float32,n)
            inputs=resident ? (;error) : (;error,prior)
            sequence=Ref(0);calls=Ref(0)
            publisher=(command,metadata)->begin
                calls[]+=1
                @test command === actual.demanded
                @test isequal(command,expected.demanded)
                @test metadata.sequence==sequence[]
                if publication===:early
                    @test actual.feedback==held.feedback && actual.state==held.state && actual.correction==held.correction
                    if device===nothing
                        @test all(==(-123f0),graph.node_outputs[7][1])
                    end
                else
                    @test isequal(actual,expected)
                end
            end
            # Exercise the omitted keyword, not just an explicit :early option.
            owner=publication===:early ? prepare_controller_graph(graph,publisher) :
                prepare_controller_graph(graph,publisher;publication)
            @test input_formats(owner)==input_formats(graph)
            @test output_formats(owner)==output_formats(graph)
            @test parameter_formats(owner)==parameter_formats(graph)
            @test FilterGraphs.required_input_sample_metadata(owner)==FilterGraphs.required_input_sample_metadata(graph)
            for frame in eachindex(F.DRIVE)
                sequence[]=frame;F.drive!(error,frame)
                metadata=resident ? (error=SampleMetadata(frame),) :
                    (error=SampleMetadata(frame),prior=SampleMetadata(frame))
                @test all(values(process!(expected,reference,inputs,metadata)))
                foreach(copyto!,values(held),values(actual))
                device===nothing && fill!(graph.node_outputs[7][1],-123f0)
                @test all(values(process!(actual,owner,inputs,metadata)))
                @test isequal(actual,expected)
                @test FilterGraphs.output_metadata(owner)==FilterGraphs.output_metadata(reference)
                if device===nothing
                    for (a,b) in zip(graph.nodes,reference.nodes), name in propertynames(a.prepared.workspace)
                        startswith(string(name),"committed_") || continue
                        @test isequal(getproperty(a.prepared.workspace,name),getproperty(b.prepared.workspace,name))
                    end
                else
                    for name in (:committed_output,:committed_state)
                        @test Array(getproperty(graph.nodes[1].adapter.vectors,name))==
                            Array(getproperty(reference.nodes[1].adapter.vectors,name))
                    end
                    @test Array(graph.nodes[6].adapter.committed_current)==Array(reference.nodes[6].adapter.committed_current)
                end
                resident || copyto!(prior,actual.feedback)
            end
            @test calls[]==length(F.DRIVE)
        end
    end
    @testset "controller failure and reset" begin
        for projection in (:pv,:vc), publication in (:early,:complete), resident in residents
            source=F.graph(1,1f0,0f0)
            replace_parameters!(source,projection=>fill(floatmax(Float32),1,1))
            graph=prepare_source(source,device,resident)
            publisher=Publisher(0,0)
            owner=publication===:early ? prepare_controller_graph(graph,publisher) :
                prepare_controller_graph(graph,publisher;publication)
            outputs=F.buffers(1);error=Float32[1.5];prior=zeros(Float32,1)
            inputs=resident ? (;error) : (;error,prior)
            metadata(i)=resident ? (error=SampleMetadata(i),) : (error=SampleMetadata(i),prior=SampleMetadata(i))
            @test all(values(process!(outputs,owner,inputs,metadata(1))))
            copyto!(prior,outputs.feedback)
            held=deepcopy(outputs);held_metadata=FilterGraphs.output_metadata(owner)
            error[1]=2.5f0
            result=try process!(outputs,owner,inputs,metadata(2)) catch e;e end
            reason=projection===:pv ? PdmFeedbackToVdmNonFiniteOutput : VdmFeedbackToControllerNonFiniteOutput
            if publication===:early
                @test result isa (device===nothing ? ControllerFault : CapturedGraphFault)
                fault=device===nothing ? result : result.cause
                @test fault isa ControllerFault && fault.cause===reason
                @test owner.command_published && owner.faulted && !owner.running
                @test publisher.calls==2 && publisher.sequence==2
                @test FilterGraphs.output_metadata(owner).demanded.sequence==2
                @test_throws (device===nothing ? ControllerFault : CapturedGraphFault) process!(outputs,owner,inputs,metadata(3))
                device===nothing || @test captured_graph_state(graph)===CapturedGraphFaulted
            else
                @test result===reason
                @test !owner.faulted && !owner.command_published
                @test publisher.calls==1 && publisher.sequence==1
                @test FilterGraphs.output_metadata(owner)==held_metadata
            end
            @test outputs.feedback==held.feedback && outputs.state==held.state && outputs.correction==held.correction
            @test FilterGraphs.rejection_context(owner).node===projection
            @test FilterGraphs.rejection_context(owner).reason===reason
            @test reset!(owner)===owner
            @test !owner.faulted && !owner.command_published
            error[1]=NaN32
            calls=publisher.calls
            @test process!(outputs,owner,inputs,metadata(4))===ClosedLoopCorrectionNonFiniteInput
            @test !owner.faulted && !owner.command_published && publisher.calls==calls
            error[1]=0;fill!(prior,0)
            @test all(values(process!(outputs,owner,inputs,metadata(5))))
        end
    end
    @testset "controller publisher and admission" begin
        for publication in (:early,:complete)
            graph=prepare_source(F.graph(1,1f0,0f0),device,false)
            output=F.buffers(1);inputs=(error=zeros(Float32,1),prior=zeros(Float32,1))
            ref=Ref{Any}()
            publisher=(command,metadata)->begin
                @test_throws InvalidStateException reset!(ref[])
                @test_throws InvalidStateException process!(output,ref[],inputs)
                throw(ErrorException("publisher failed after command escape"))
            end
            owner=prepare_controller_graph(graph,publisher;publication);ref[]=owner
            @test_throws (device===nothing ? ControllerFault : CapturedGraphFault) process!(output,owner,inputs)
            @test owner.faulted && owner.command_published && !owner.running
            reset!(owner)
            @test !owner.faulted && !owner.command_published
            @test_throws GraphConfigurationError prepare_controller_graph(graph,publisher;command=:correction)
            @test_throws ArgumentError prepare_controller_graph(graph,publisher;publication=:unknown)
            # Reversed caller field order is supported by the ordinary validators.
            counter=Publisher(0,0);owner=prepare_controller_graph(graph,counter;publication)
            reversed=(feedback=output.feedback,demanded=output.demanded,state=output.state,correction=output.correction)
            @test all(values(process!(reversed,owner,(prior=inputs.prior,error=inputs.error))))
            @test counter.calls==1 && counter.sequence==0
            reset!(owner)
            alias_output=merge(output,(demanded=inputs.error,))
            @test_throws ArgumentError process!(alias_output,owner,inputs)
            @test !owner.faulted && !owner.command_published
            for i in 1:10;process!(output,owner,inputs);end
            @test allocation_bytes(output,owner,inputs)==0
        end
    end
end

allocation_bytes(outputs, owner, inputs) = @allocated process!(outputs,owner,inputs)

function run_borrowed_buffer_tests()
    @testset "graph rejects overlapping external outputs before committing" begin
        graph = F.graph(2, 1f0, 0f0)
        workspace = graph.nodes[1].prepared.workspace
        inputs = (error=Float32[0.25, 0.5], prior=zeros(Float32, 2))
        metadata = (error=SampleMetadata(1), prior=nothing)
        expected = F.buffers(2)
        availability = process!(expected, graph, inputs, metadata)
        @test all(values(availability))
        storage = fill(-17f0, 8)
        GC.@preserve storage begin
            left = unsafe_wrap(Array, pointer(storage), 3; own=false)
            overlap = unsafe_wrap(Array, pointer(storage, 2), 2; own=false)
            right = unsafe_wrap(Array, pointer(storage, 3), 3; own=false)
            reshaped = view(reshape(view(left, 1:2), 2, 1), :, 1)
            odd_parent = unsafe_wrap(Array, pointer(storage), 4; own=false)
            even_parent = unsafe_wrap(Array, pointer(storage, 2), 5; own=false)
            permuted = map((1, 2, 3)) do start
                matrix = unsafe_wrap(Array, pointer(storage, start), (2, 2); own=false)
                view(PermutedDimsArray(matrix, (2, 1)), 1, :)
            end
            for (correction, demanded) in (
                (view(left, 1:2), view(overlap, 1:2)),
                (view(left, 1:2), overlap),
                (overlap, view(left, 1:2)),
                (view(left, 3:-2:1), view(overlap, 1:2)),
                (reshaped, view(overlap, 1:2)),
                (view(odd_parent, 1:3:4), view(even_parent, 3:-2:1)),
                (permuted[1], permuted[2]),
            )
                reset!(graph)
                fill!(storage, -17f0)
                outputs = merge(F.buffers(2), (; correction, demanded))
                before = FilterGraphs.output_metadata(graph)
                @test_throws ArgumentError process!(outputs, graph, inputs, metadata)
                @test all(==(-17f0), storage)
                @test isnothing(FilterGraphAlgorithms.closed_loop_correction_last_output(workspace))
                @test FilterGraphs.output_metadata(graph) == before
            end
            for (correction, demanded) in (
                (view(left, 1:2), view(right, 1:2)),
                (view(storage, 1:2), view(storage, 3:4)),
                (view(storage, 1:2:3), view(storage, 2:2:4)),
                (reshaped, view(right, 1:2)),
                (view(odd_parent, 1:2:3), view(even_parent, 1:2:3)),
                (view(odd_parent, 1:3:4), view(even_parent, 5:-4:1)),
                (permuted[1], permuted[3]),
            )
                reset!(graph)
                outputs = merge(F.buffers(2), (; correction, demanded))
                @test process!(outputs, graph, inputs, metadata) == availability
                @test outputs == expected
                process!(outputs, graph, inputs)
                @test allocation_bytes(outputs, graph, inputs) == 0
            end
        end
    end
    @testset "controller rejects commands overlapping borrowed inputs" begin
        for publication in (:early, :complete)
            graph = F.graph(2, 1f0, 0f0)
            workspace = graph.nodes[1].prepared.workspace
            publisher = Publisher(0, 0)
            owner = prepare_controller_graph(graph, publisher; publication)
            storage = Float32[0.25, 0.5, -17, -17, -17]
            saved = copy(storage)
            prior = zeros(Float32, 2)
            GC.@preserve storage begin
                left = unsafe_wrap(Array, pointer(storage), 2; own=false)
                overlap = unsafe_wrap(Array, pointer(storage, 2), 2; own=false)
                right = unsafe_wrap(Array, pointer(storage, 3), 3; own=false)
                words = unsafe_wrap(Array, Ptr{UInt32}(pointer(storage)), 2; own=false)
                reinterpreted = reinterpret(Float32, words)
                for (error, demanded) in (
                    (left, overlap),
                    (view(left, 1:2), overlap),
                    (reinterpreted, view(overlap, 1:2)),
                )
                    reset!(owner)
                    publisher.calls = 0
                    copyto!(storage, saved)
                    outputs = merge(F.buffers(2), (; demanded))
                    @test_throws ArgumentError process!(outputs, owner, (; error, prior))
                    @test storage == saved
                    @test publisher.calls == 0
                    @test isnothing(FilterGraphAlgorithms.closed_loop_correction_last_output(workspace))
                    @test !owner.running && !owner.command_published && !owner.faulted
                end
                reset!(owner)
                publisher.calls = 0
                copyto!(storage, saved)
                outputs = merge(F.buffers(2), (demanded=view(right, 1:2),))
                inputs = (error=reinterpreted, prior=prior)
                @test all(values(process!(outputs, owner, inputs)))
                @test outputs.demanded == inputs.error == saved[1:2]
                @test publisher.calls == 1
                @test allocation_bytes(outputs, owner, inputs) == 0
            end
        end
    end
end

end
