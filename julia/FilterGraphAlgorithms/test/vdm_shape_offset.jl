module VdmShapeOffsetTests
using Test, LinearAlgebra, FilterGraphAlgorithms
import FilterGraphAlgorithms: process!, replace_parameter!, reset!

allocated_shape(y,current,prepared,x) = @allocated process!(y,current,prepared,x)

@testset "shape requests enter independent coordinates before completion" begin
    completion=Float32[1 0; 0.5 0.5; 0 1; -1 2; 0 0]
    inverse=Matrix{Float32}(I,5,5)
    offset=Float32[0.4,0,-0.2,0,0]
    prepared=prepare_algorithm(VdmShapeOffsetF32,VdmShapeOffsetF32Configuration(
        active_vdm_count=2,full_vdm_count=5,physical_actuator_count=5))
    replace_parameter!(prepared,Symbol("active-to-full"),completion)
    replace_parameter!(prepared,Symbol("pdm-to-vdm"),inverse)
    replace_parameter!(prepared,Symbol("shape-offset"),offset)
    @test prepared.plan.controlled_actuators==[true,false,true,false,false]
    @test prepared.plan.shape_projection*completion==Matrix{Float32}(I,2,2)
    @test prepared.plan.active_offset==offset[[1,3]]
    x=zeros(Float32,2);y=fill(99f0,2);current=fill(99f0,5)
    @test process!(y,current,prepared,x)===nothing
    @test y==offset[[1,3]] && current==offset
    physical=completion*y
    @test physical[[1,3]]==offset[[1,3]]
    @test physical[2]≈0.1f0 && physical[4]≈-0.8f0
    @test iszero(physical[5])
    # The superseded post-completion addition leaves guards stale.
    @test (completion*x+offset)[[2,4]]!=physical[[2,4]]
    @test allocated_shape(y,current,prepared,x)==0
    previous=copy(y);observed=copy(current)
    x[1]=NaN32
    @test process!(y,current,prepared,x)===VectorLinearCombinationNonFiniteInput
    @test y==previous && current==observed
    @test_throws ArgumentError replace_parameter!(prepared,Symbol("shape-offset"),Float32[Inf,0,0,0,0])
    @test prepared.plan.shape_offset==offset
    @test_throws ArgumentError replace_parameter!(prepared,Symbol("active-to-full"),fill(0.5f0,5,2))
    @test_throws DimensionMismatch replace_parameter!(prepared,Symbol("shape-offset"),zeros(Float32,4))
    x .= [-0.0f0,0.0f0];replace_parameter!(prepared,Symbol("shape-offset"),zeros(Float32,5))
    @test process!(y,current,prepared,x)===nothing
    @test reinterpret(UInt32,y)==reinterpret(UInt32,x)
    replace_parameter!(prepared,Symbol("shape-offset"),offset);reset!(prepared);fill!(x,0)
    @test process!(y,current,prepared,x)===nothing && current==offset
    @test_throws ArgumentError process!(y,prepared.plan.shape_offset,prepared,x)
    @test_throws DimensionMismatch process!(y,zeros(Float32,4),prepared,x)
    @test_throws ArgumentError VdmShapeOffsetPlan(zeros(Float32,5,2),inverse,offset)
    @test_throws DimensionMismatch VdmShapeOffsetPlan(completion,zeros(Float32,4,5),offset)
    @test_throws ArgumentError VdmShapeOffsetPlan(completion,fill(floatmax(Float32),5,5),fill(2f0,5))
end

@testset "dense physical projection retains the intended shape" begin
    q=Float32[1 0.2 0;0 1 0.3;0.1 0 1]
    completion=Matrix{Float32}(I,3,3)
    offset=Float32[0.2,-0.1,0.3]
    plan=VdmShapeOffsetPlan(completion,inv(q),offset)
    @test all(plan.controlled_actuators)
    @test q*plan.active_offset≈offset rtol=3f-7
    @test plan.active_offset!=offset
    @test plan.shape_projection*completion≈inv(q)
    @test offset==Float32[0.2,-0.1,0.3]
end
end
