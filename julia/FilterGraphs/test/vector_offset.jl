module VectorOffsetGraphTests
using Test, LinearAlgebra, FilterGraphs, FilterGraphAlgorithms
import FilterGraphs: process!, reset!, output_formats, output_metadata_source_inputs,
    output_metadata, replace_parameters!

@testset "offset readback follows successful requested frames" begin
    mktemp() do path,io
        write(io,"""
        filter.graph = {
          nodes = [{ type=ndarray name=offset label=vector-offset-f32
            config={ extent=3 schema="pipewireao/test-vector-f32/1" } }]
          links = []
          inputs = [{ name=requested endpoint="offset:requested" }
            { name=offset endpoint="offset:offset" }]
          outputs = [{ name=command endpoint="offset:output" }
            { name=current_offset endpoint="offset:current-offset" }]
        }
        """)
        close(io)
        graph=prepare_graph(path;algorithms=FilterGraphAlgorithms.algorithms())
        @test keys(output_formats(graph))==(:command,:current_offset)
        @test output_metadata_source_inputs(graph)==(1,1)
        outputs=(command=zeros(Float32,3),current_offset=fill(99f0,3))
        requested=Float32[1,2,3]
        metadata=SampleMetadata(17;timestamp_nanoseconds=123456,discontinuity=true)
        @test process!(outputs,graph,(;requested),(requested=metadata,))==(command=true,current_offset=true)
        @test outputs.command==requested && all(iszero,outputs.current_offset)
        @test output_metadata(graph)==(command=metadata,current_offset=metadata)
        replace_parameters!(graph,:offset=>Float32[0.5,-1,2])
        metadata=SampleMetadata(18;timestamp_nanoseconds=123789)
        @test process!(outputs,graph,(;requested),(requested=metadata,))==(command=true,current_offset=true)
        @test outputs.current_offset==Float32[0.5,-1,2]
        @test outputs.command==Float32[1.5,1,5]
        previous=deepcopy(outputs);previous_metadata=output_metadata(graph)
        requested[end]=NaN32
        @test process!(outputs,graph,(;requested),(requested=SampleMetadata(19),))===VectorLinearCombinationNonFiniteInput
        @test outputs==previous && output_metadata(graph)==previous_metadata
        reset!(graph);requested[end]=3
        @test process!(outputs,graph,(;requested),(requested=SampleMetadata(1),))==(command=true,current_offset=true)
        @test outputs.current_offset==Float32[0.5,-1,2]
        @test output_metadata(graph).current_offset==SampleMetadata(1)
    end
end

@testset "projected shapes complete before physical limits and retain request metadata" begin
    mktemp() do path,io
        write(io,"""
        filter.graph = {
          nodes = [
            { type=ndarray name=shape label=vdm-shape-offset-f32
              config={active_vdm_count=2 full_vdm_count=5 physical_actuator_count=5} }
            { type=ndarray name=forward label=vdm-to-pdm-f32
              config={active_vdm_count=2 full_vdm_count=5 physical_actuator_count=5} }
            { type=ndarray name=limit label=pdm-command-f32
              config={actuator_count=5 command_limits=[-1e-6,1e-6] slew_limit=null
                quantization_step=null quantization_origin=null dither_seed=null initial_current=null} }
          ]
          links = [{output="shape:output" input="forward:vdm-command"}
            {output="forward:requested-pdm-command" input="limit:requested"}]
          inputs = [{name=requested endpoint="shape:requested"}
            {name=shape_offset endpoint="shape:shape-offset"}
            {name=shape_completion endpoint="shape:active-to-full"}
            {name=shape_inverse endpoint="shape:pdm-to-vdm"}
            {name=completion endpoint="forward:active-to-full"}
            {name=projection endpoint="forward:vdm-to-pdm"}]
          outputs = [{name=command endpoint="limit:demanded"}
            {name=current_shape endpoint="shape:current-offset"}]
        }
        """)
        close(io)
        graph=prepare_graph(path;algorithms=FilterGraphAlgorithms.algorithms())
        completion=Float32[1 0;0.5 0.5;0 1;-1 2;0 0]
        projection=Float32.(Matrix{Int}(LinearAlgebra.I,5,5))
        offset=Float32[0.9e-6,0,-0.9e-6,0,0]
        replace_parameters!(graph,:shape_offset=>offset,:shape_completion=>completion,
            :shape_inverse=>projection,:completion=>completion,:projection=>projection)
        output=(command=zeros(Float32,5),current_shape=zeros(Float32,5))
        metadata=SampleMetadata(42;timestamp_nanoseconds=123456)
        @test process!(output,graph,(requested=zeros(Float32,2),),(requested=metadata,))==
            (command=true,current_shape=true)
        @test output.command[[1,3]]==offset[[1,3]]
        @test output.command[4]==-1f-6 && iszero(output.command[5])
        @test output.current_shape==offset
        @test output_metadata(graph)==(command=metadata,current_shape=metadata)
        previous=deepcopy(output)
        @test_throws ArgumentError replace_parameters!(graph,:shape_offset=>fill(Inf32,5))
        @test_throws ArgumentError replace_parameters!(graph,:shape_offset=>zeros(Float32,5),
            :shape_completion=>zeros(Float32,5,2))
        @test process!(output,graph,(requested=zeros(Float32,2),),(requested=SampleMetadata(43),))==
            (command=true,current_shape=true)
        @test output==previous
    end
end
end
