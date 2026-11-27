module CompositionAdapterTests
using Test, FilterGraphAlgorithms, FilterGraphPipeWire, FilterGraphs, PipeWireAO
include(joinpath(@__DIR__, "data", "support", "composition_fixture.jl"))
using .CompositionFixture: CompositionGateF32, composition_fixture
const JFG = FilterGraphs
const FGP = FilterGraphPipeWire
const FGA = FilterGraphAlgorithms

# Native callback helpers are shared with the adapter's existing metadata tests.
const native_callback_buffer = parentmodule(@__MODULE__).native_callback_buffer
const native_callback_metadata_valid = parentmodule(@__MODULE__).native_callback_metadata_valid
const native_callback_header = parentmodule(@__MODULE__).native_callback_header

function composition_callback_bytes(node,inputs,outputs,previous)
    fill!(node.owner.join.last_inputs,previous)
    return @allocated node.filter.callbacks.on_process(node.filter,inputs,outputs)
end

@testset "composition independent native arrivals and private feedback" begin
    mktempdir() do directory
        path = composition_fixture(directory)
        write(path,replace(read(path,String),"name = a graph"=>"name = a rate = [7,1] graph", "name = b graph"=>"name = b rate = [3,1] graph"))
        composition = JFG.prepare_composition(path;
            algorithms=(FGA.algorithms()...,CompositionGateF32))
        ports = FGP._composition_ports(composition,100//1)
        @test ports[1].format.rate == SPA.Fraction(7,1)
        @test ports[2].format.rate == SPA.Fraction(3,1)
        @test length(ports) == 7
        @test only(port for port in ports if port.name == "command").schema == FGA.DEMANDED_PDM_COMMAND_V1
        node = FGP.PipeWireNode(composition;connect=false,rate=100//1,fifo_inputs=true)
        try
            @test keys(FGP._public_input_bindings(node.owner)) == (Symbol("a-image"),Symbol("b-image"))
            @test node.owner.graph === composition.graph
            @test length(node.owner.graph.nodes) == 11
            samples = (
                (inputs=(;var"a-image"=[-99f0]), metadata=(;var"a-image"=SampleMetadata(1;terminal=true)),
                 expected=SccPixelResidualIncompletePair),
                (inputs=(;var"b-image"=[.2f0]), metadata=(;var"b-image"=SampleMetadata(1;terminal=true)),expected=nothing),
            )
            @test FGP.prepare!(node;samples,required_outputs=(:command,)) === node
            @test !node.owner.active
            @test JFG.composition_observation(composition).sequence == 0
            FGP._prepare_owner(node.owner)
            a,b = [.1f0],[.2f0]
            owned_outputs = map(f -> zeros(f.element_type,f.shape),JFG.output_formats(composition))
            native_outputs = vcat((native_callback_buffer(array;header_available=true,properties_available=true)
                for array in values(owned_outputs))...)
            function native_step!(which,sequence)
                header = BufferHeader(UInt32(0),UInt32(0),Int64(sequence),Int64(0),UInt64(sequence))
                native_inputs = vcat(native_callback_buffer(a;present=which !== :b,header,properties_available=true),
                    native_callback_buffer(b;present=which !== :a,header,properties_available=true))
                GC.@preserve a b owned_outputs native_inputs native_outputs begin
                    inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_inputs)),UInt32(2))
                    outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_outputs)),UInt32(length(native_outputs)))
                    for index in 1:2
                        input_available(inputs[index]) || continue
                        properties = metadata_records(NdArrayFilterBuffer{true}(Ptr{Cvoid}(pointer(native_inputs,index))))
                        set_metadata_bytes!(properties,UInt32(0x900),ntuple(i -> UInt8((UInt64(sequence+100) >> (8*(i-1))) & 0xff),8))
                    end
                    node.filter.callbacks.on_process(node.filter,inputs,outputs)
                    for branch in (which === :both ? (:a,:b) : (which,))
                        name = Symbol(string(branch) * "-correction")
                        index = findfirst(==(name),keys(owned_outputs))
                        properties = metadata_records(outputs[index])
                        @test parentmodule(@__MODULE__).native_callback_custom_value(native_outputs,index) == sequence + 100
                    end
                    command_index = findfirst(==(:command),keys(owned_outputs))
                    @test parentmodule(@__MODULE__).native_callback_custom_value(native_outputs,command_index) === nothing
                end
                return nothing
            end
            @test native_step!(:b,1) === nothing
            @test owned_outputs.command ≈ [.2f0]
            @test native_step!(:a,1) === nothing
            @test owned_outputs.command ≈ [.3f0]
            @test native_step!(:a,2) === nothing
            @test owned_outputs.command ≈ [.4f0]
            command_index = findfirst(==(:command),keys(owned_outputs))
            @test native_callback_header(native_outputs,command_index).sequence == 3
            @test JFG.composition_observation(composition).trigger == (:a,)
            FGP._reset_owner!(node.owner)
            @test JFG.composition_observation(composition).sequence == 0
            a .= 0
            @test native_step!(:a,1) === nothing
            @test owned_outputs.command == [0f0]
            # The actual callback sees one fixed camera tuple plus a presence
            # mask. Payloads, metadata, and native buffers are prepared here.
            header = BufferHeader(UInt32(0),UInt32(0),Int64(1),Int64(0),UInt64(1))
            available_a = native_callback_buffer(a;header,acquisition_sequence=UInt64(1))
            available_b = native_callback_buffer(b;header,acquisition_sequence=UInt64(1))
            absent_a = native_callback_buffer(a;present=false)
            absent_b = native_callback_buffer(b;present=false)
            previous = SampleMetadata(0;terminal=true,
                acquisition_identity=SampleAcquisitionIdentity(ntuple(index -> UInt8(index),16),7,0))
            GC.@preserve a b owned_outputs native_outputs begin
                output_buffers = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_outputs)),UInt32(length(native_outputs)))
                for storage in (vcat(available_a,absent_b),vcat(absent_a,available_b),vcat(available_a,available_b))
                    GC.@preserve storage begin
                        input_buffers = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(storage)),UInt32(2))
                        composition_callback_bytes(node,input_buffers,output_buffers,previous)
                        @test composition_callback_bytes(node,input_buffers,output_buffers,previous) == 0
                    end
                end
            end
        finally
            close(node)
        end
    end
end
end
