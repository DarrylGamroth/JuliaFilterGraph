module PhysicalOffsetReferenceGraphTests
using Test, FilterGraphs, FilterGraphAlgorithms
import FilterGraphs: process!, reset!, output_metadata, output_metadata_source_inputs, replace_parameters!

@testset "physical offset and reference compensation share adoption and primary metadata" begin
    mktemp() do path,io
        write(io,"""
        filter.graph = {
          nodes = [{ type=ndarray name=reference label=physical-offset-reference-f32
            config={residual_count=3 physical_count=4 mode_count=2 reference_sign=-1
              residual_schema="test.residual/1" physical_schema="test.physical/1"
              physical_to_modes_schema="test.physical-to-modes/1"
              modes_to_reference_schema="test.modes-to-reference/1"} }]
          links = []
          inputs = [{name=residual endpoint="reference:reconstructed-residual"}
            {name=offset endpoint="reference:offset"}
            {name=q endpoint="reference:physical-to-modes"}
            {name=c endpoint="reference:modes-to-reference"}]
          outputs = [{name=compensated endpoint="reference:compensated-residual"}
            {name=physical endpoint="reference:physical-offset"}]
        }
        """)
        close(io)
        graph=prepare_graph(path;algorithms=FilterGraphAlgorithms.algorithms())
        q=Float32[1 0 2 0;0 1 0 -1];c=Float32[1 2;-1 0;0 3]
        offset=Float32[2,-1,0.5,1]
        replace_parameters!(graph,:q=>q,:c=>c,:offset=>offset)
        output=(compensated=zeros(Float32,3),physical=zeros(Float32,4))
        residual=c*(q*offset)
        metadata=SampleMetadata(17;timestamp_nanoseconds=123456,discontinuity=true)
        @test output_metadata_source_inputs(graph)==(1,1)
        @test process!(output,graph,(;residual),(residual=metadata,))==(compensated=true,physical=true)
        @test output.compensated==zeros(Float32,3) && output.physical==offset
        @test output_metadata(graph)==(compensated=metadata,physical=metadata)
        previous=deepcopy(output);previous_metadata=output_metadata(graph)
        # A rejected multi-parameter transaction cannot adopt only the offset.
        @test_throws ArgumentError replace_parameters!(graph,:offset=>zeros(Float32,4),:q=>fill(NaN32,2,4))
        @test process!(output,graph,(;residual),(residual=metadata,))==(compensated=true,physical=true)
        @test output==previous
        replace_parameters!(graph,:offset=>zeros(Float32,4))
        bad=fill(Inf32,3)
        @test process!(output,graph,(residual=bad,),(residual=SampleMetadata(18),))===VectorLinearCombinationNonFiniteInput
        @test output==previous && output_metadata(graph)==previous_metadata
        @test process!(output,graph,(;residual),(residual=SampleMetadata(19),))==(compensated=true,physical=true)
        @test output.compensated==residual && output.physical==zeros(Float32,4)
        @test output_metadata(graph)==(compensated=SampleMetadata(19),physical=SampleMetadata(19))
        reset!(graph)
        @test process!(output,graph,(;residual),(residual=SampleMetadata(1),))==(compensated=true,physical=true)
        @test output.compensated==residual && output.physical==zeros(Float32,4)
    end
end
end
