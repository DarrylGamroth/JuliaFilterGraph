module VectorOffsetTests
using Test, FilterGraphAlgorithms
import FilterGraphAlgorithms: process!, replace_parameter!, reset!
allocated_step(y,p,x)=@allocated process!(y,p,x)
allocated_readback(y,current,p,x)=@allocated process!(y,current,p,x)
@testset "same-schema offset parameters reuse vector arithmetic" begin
    p=prepare_algorithm(VectorOffsetF32,VectorOffsetF32Configuration(extent=3,schema="pipewireao/test-vector-f32/1"))
    x=Float32[1,2,3];y=zeros(Float32,3)
    @test VectorOffsetF32 in algorithms()
    @test process!(y,p,x)===nothing
    @test y==x
    offset=Float32[2,-1,0.5]
    replace_parameter!(p,:offset,offset);fill!(offset,99)
    @test process!(y,p,x)===nothing
    @test y==Float32[3,1,3.5]
    @test allocated_step(y,p,x)==0
    @test_throws ArgumentError replace_parameter!(p,:offset,Float32[NaN,0,0])
    @test_throws DimensionMismatch replace_parameter!(p,:offset,Float32[0,0])
    @test process!(y,p,x)===nothing
    @test y==Float32[3,1,3.5]
    reset!(p)
    @test process!(y,p,x)===nothing
    @test y==Float32[3,1,3.5] # Reset scratch, retain adopted calibration.
    x[end]=NaN32
    @test process!(y,p,x)===VectorLinearCombinationNonFiniteInput
    @test y==Float32[3,1,3.5]
end

@testset "adopted offset readback shares the configured schema and commits atomically" begin
    p=prepare_algorithm(VectorOffsetF32,VectorOffsetF32Configuration(extent=3,schema="pipewireao/test-vector-f32/1"))
    ports=algorithm_ports(VectorOffsetF32)
    @test getproperty.(ports,:name)==(:requested,:output,Symbol("current-offset"),:offset)
    @test ports[2].metadata_source===ports[3].metadata_source===:requested
    @test all(format->format.schema=="pipewireao/test-vector-f32/1",algorithm_formats(p))
    x=Float32[-0.0,2,3];y=zeros(Float32,3);current=fill(99f0,3);single=similar(y)
    offset=Float32[-0.0,-1,0.5]
    replace_parameter!(p,:offset,offset)
    @test process!(y,current,p,x)===nothing
    @test process!(single,p,x)===nothing
    @test reinterpret(UInt32,y)==reinterpret(UInt32,single)
    @test reinterpret(UInt32,current)==reinterpret(UInt32,offset)
    @test allocated_readback(y,current,p,x)==0
    previous=copy(y);previous_offset=copy(current)
    replace_parameter!(p,:offset,Float32[1,2,3])
    x[end]=NaN32
    @test process!(y,current,p,x)===VectorLinearCombinationNonFiniteInput
    @test y==previous && isequal(current,previous_offset)
    x[end]=floatmax(Float32)
    replace_parameter!(p,:offset,fill(floatmax(Float32),3))
    @test process!(y,current,p,x)===VectorLinearCombinationNonFiniteOutput
    @test y==previous && isequal(current,previous_offset)
    replace_parameter!(p,:offset,offset);x[end]=3
    reset!(p)
    @test process!(y,current,p,x)===nothing
    @test isequal(current,offset)
    @test_throws DimensionMismatch process!(y,zeros(Float32,2),p,x)
    @test_throws ArgumentError process!(y,y,p,x)
    @test_throws ArgumentError process!(y,x,p,x)
    @test_throws ArgumentError process!(y,p.plan.offset,p,x)
    @test_throws ArgumentError process!(p.plan.offset,current,p,x)
end
end
