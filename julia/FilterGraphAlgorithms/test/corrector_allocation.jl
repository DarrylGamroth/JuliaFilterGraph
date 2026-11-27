module CorrectorAllocationTests
using FilterGraphAlgorithms, Test, LinearAlgebra
import FilterGraphAlgorithms: process!, commit!, discard!, reset!

identitybank(T, b0=one(T), b1=zero(T)) = reshape(T[b0,b1,0,0,0],5,1,1)
function stage(P,H; limit=1.0, period=1, slew=nothing, coefficients=nothing)
    T=eltype(P); n=size(P,1)
    CorrectorStagePlan(P,H; lower_limits=fill(T(-limit),n), upper_limits=fill(T(limit),n),
        update_period=period, slew_limits=isnothing(slew) ? nothing : fill(T(slew),n), coefficients)
end
function tick!(out,fb,w,p,q)
    result=process!(out,fb,w,p,q)
    result === nothing && commit!(w)
    return result
end

@testset "Sequential corrector residual oracle" begin
    for T in (Float32,Float64), count in (1,2,5)
        stages=CorrectorStagePlan{T}[]
        for i in 1:count
            P=i%2==1 ? T[1 0; 0 0.5] : reshape(T[0.25,0.75],1,2)
            H=i%2==1 ? T[0.8 0; 0 1.2] : reshape(T[0.5,1],2,1)
            push!(stages,stage(P,H;limit=0.3))
        end
        p=CorrectorAllocationPlan(stages); w=CorrectorAllocationWorkspace(p)
        q=T[2,-1]; out=zeros(T,p.command_count); fb=zeros(T,2)
        residual=copy(q); expected=T[]
        for s in stages
            commanded=clamp.(s.to_command*residual,T(-0.3),T(0.3))
            append!(expected,commanded); residual-=s.to_controller*commanded
        end
        @test @inferred(Union{Nothing,CorrectorAllocationFailure}, process!(out,fb,w,p,q)) === nothing
        @test out ≈ expected
        @test fb ≈ residual
        @test q ≈ fb + sum((s.to_controller * out[sum(size(stages[j].to_command,1) for j in 1:(i-1);init=0)+1:sum(size(stages[j].to_command,1) for j in 1:i)] for (i,s) in enumerate(stages)))
        commit!(w)
    end
    p=CorrectorAllocationPlan([stage(ones(1,1),ones(1,1);limit=0.25),stage(ones(1,1),ones(1,1);limit=2)])
    w=CorrectorAllocationWorkspace(p); out=zeros(2);fb=zeros(1)
    @test tick!(out,fb,w,p,[1.0]) === nothing
    @test out == [0.25,0.75]
    @test fb == [0.0]
end

@testset "Base-rate filters, holds and per-update slew" begin
    p=CorrectorAllocationPlan([stage(ones(1,1),ones(1,1); limit=10,period=3,slew=0.5,coefficients=identitybank(Float64,1,1))])
    w=CorrectorAllocationWorkspace(p);out=zeros(1);fb=zeros(1)
    for (q, expected) in zip((1.,2.,3.,4.,5.,6.,7.),(0.5,0.5,0.5,1.,1.,1.,1.5))
        @test tick!(out,fb,w,p,[q]) === nothing
        @test out == [expected]
        @test fb == [q-expected]
    end
    @test w.stages[1].filter.committed_state[1,1,1] == 7
    reset!(w)
    @test w.stages[1].counter == 0
    @test all(iszero,w.stages[1].held)
    @test tick!(out,fb,w,p,[0.25]) === nothing
    @test out == [0.25]
    # A delayed numerator proves held samples still advance filter history.
    p=CorrectorAllocationPlan([stage(ones(1,1),ones(1,1);limit=100,period=2,coefficients=identitybank(Float64,0,1))])
    w=CorrectorAllocationWorkspace(p)
    for (q,expected) in zip((2.,3.,4.),(0.,0.,3.))
        @test tick!(out,fb,w,p,[q]) === nothing
        @test out == [expected]
    end
end

@testset "Explicit base-rate filtering before slower updates" begin
    p=CorrectorAllocationPlan([stage(ones(1,1),ones(1,1);limit=10,period=2,coefficients=identitybank(Float64,0.5,0.5))])
    w=CorrectorAllocationWorkspace(p);out=zeros(1);fb=zeros(1)
    for (q,expected) in zip((1.,-1.,1.,-1.,1.),(0.5,0.5,0.,0.,0.))
        @test tick!(out,fb,w,p,[q]) === nothing
        @test out == [expected]
    end
end

@testset "Validation, views and transactional rejection" begin
    @test_throws ArgumentError stage(ones(1,1),ones(1,1);period=2)
    @test_throws ArgumentError stage(ones(1,1),ones(1,1);period=0)
    @test_throws DimensionMismatch stage(ones(2,1),ones(2,1))
    @test_throws ArgumentError stage(fill(NaN,1,1),ones(1,1))
    @test_throws ArgumentError CorrectorAllocationPlan(CorrectorStagePlan{Float64}[])
    p=CorrectorAllocationPlan([stage(ones(1,1),ones(1,1)),stage(fill(floatmax(Float64),1,1),zeros(1,1))])
    w=CorrectorAllocationWorkspace(p);out=fill(8.,2);fb=[9.]
    @test process!(out,fb,w,p,[NaN]) === CorrectorAllocationNonFiniteInput
    @test out == [8,8] && fb == [9]
    @test process!(out,fb,w,p,[4.]) === CorrectorAllocationNonFiniteOutput
    @test out == [8,8] && fb == [9]
    @test all(s -> all(iszero,s.filter.committed_state) && s.counter==0 && all(iszero,s.held),w.stages)
    @test all(s -> s.filter.candidate_state == s.filter.committed_state,w.stages)
    @test process!(out,fb,w,p,[0.5]) === nothing
    discard!(w)
    @test all(s -> all(iszero,s.held),w.stages)
    @test_throws DimensionMismatch process!(zeros(1),fb,w,p,[1.])
    @test_throws ArgumentError process!(out,fb,w,p,fb)
    @test_throws ArgumentError process!(w.commands,fb,w,p,[1.])
    @test_throws ArgumentError process!(out,fb,w,p,w.residual)
    storage=zeros(7);commands=view(storage,1:2);feedback=view(storage,4:4);demand=view(storage,6:6);demand[1]=0.25
    @test process!(commands,feedback,w,p,demand) === nothing
end

@testset "Internal ownership and nonzero history rollback" begin
    s = stage(ones(1,1),ones(1,1); limit=10,period=2,coefficients=identitybank(Float64,1,0.25))
    second = stage(fill(2.,1,1),ones(1,1);limit=10,period=2,coefficients=identitybank(Float64,1,0.25))
    p = CorrectorAllocationPlan([s,second])
    w = CorrectorAllocationWorkspace(p); out=zeros(2); fb=zeros(1)
    @test tick!(out,fb,w,p,[0.5]) === nothing
    @test all(x -> x.counter == 1,w.stages)
    saved = [(copy(x.filter.committed_state),copy(x.conditioner.committed_current),copy(x.held),x.counter) for x in w.stages]
    out .= 8; fb .= 9
    @test process!(out,fb,w,p,[floatmax(Float64)]) === CorrectorAllocationNonFiniteOutput
    @test out == [8,8] && fb == [9]
    for (x,old) in zip(w.stages,saved)
        @test x.filter.committed_state == old[1]
        @test x.conditioner.committed_current == x.conditioner.candidate_current == old[2]
        @test x.held == old[3] && x.counter == old[4]
        @test x.filter.candidate_state == old[1]
        @test all(iszero, x.filter.candidate_output)
    end
    @test tick!(out,fb,w,p,[0.25]) === nothing
    @test all(x -> x.counter == 0,w.stages)
    @test process!(out,fb,w,p,[0.25]) === nothing
    @test process!(out,fb,w,p,[NaN]) === CorrectorAllocationNonFiniteInput
    @test all(x -> x.filter.candidate_state == x.filter.committed_state,w.stages)
    # Fail-before reproduction: this alias formerly overwrote committed hold history.
    w.stages[1].requested = w.stages[1].held
    held = copy(w.stages[1].held)
    @test_throws ArgumentError process!(out,fb,w,p,[0.5])
    @test w.stages[1].held == held
end

@testset "Declared configuration and packing" begin
    c=CorrectorAllocationF32Configuration(controller_command_count=2,stage_command_counts=[1,2],update_periods=[1,1],
        command_projection=Float32[1,2,3,4,5,6], realization_projection=Float32[1,2,3,4,5,6],
        section_count=1,initial_coefficients=nothing,command_limits=(fill(-100f0,3), fill(100f0,3)),slew_limits=nothing,
        command_schema="org.test.corrector-command/1")
    a=prepare_algorithm(CorrectorAllocationF32,c)
    @test a.plan.stages[1].to_command == Float32[1 2]
    @test a.plan.stages[2].to_command == Float32[3 4;5 6]
    @test a.plan.stages[1].to_controller == reshape(Float32[1,4],2,1)
    @test a.plan.stages[2].to_controller == Float32[2 3;5 6]
    @test CorrectorAllocationF32 in algorithms()
    commands, feedback = zeros(Float32, 3), zeros(Float32, 2)
    @test process!(commands, feedback, a, Float32[1, 0]) === nothing
    input = Float32[1, NaN32]
    filter = a.workspace.stages[1].filter
    @test process!(commands, feedback, a, input) === CorrectorAllocationNonFiniteInput
    @test all(iszero, filter.candidate_output)
end

function allocation_bytes(out,fb,w,p,q)
    process!(out,fb,w,p,q)
    return @allocated process!(out,fb,w,p,q)
end

@testset "Column traversal retains ordered FMA projection" begin
    for T in (Float32, Float64), (n,m) in ((1,1), (3,9), (9,277), (277,253))
        P = [T(mod(7i+3j,23)-11)/T(13) for i in 1:n, j in 1:m]
        H = [T(mod(5i+11j,19)-9)/T(17) for i in 1:m, j in 1:n]
        q = [T(mod(13i,31)-15)/T(7) for i in 1:m]
        p = CorrectorAllocationPlan([stage(P,H;limit=0.5)])
        w = CorrectorAllocationWorkspace(p)
        out,fb = zeros(T,n),zeros(T,m)
        @test process!(out,fb,w,p,q) === nothing
        expected = zeros(T,n)
        for row in 1:n
            value = zero(T)
            for col in 1:m
                value = fma(P[row,col],q[col],value)
            end
            expected[row] = value
        end
        @test reinterpret(UInt8,w.stages[1].requested) == reinterpret(UInt8,expected)
        residual = copy(q)
        for row in 1:m
            value = q[row]
            for col in 1:n
                value = fma(-H[row,col],out[col],value)
            end
            residual[row] = value
        end
        @test reinterpret(UInt8,fb) == reinterpret(UInt8,residual)
        @test allocation_bytes(out,fb,w,p,q) == 0
    end
end

@testset "Column traversal residual overflow remains transactional" begin
    for T in (Float32,Float64)
        p = CorrectorAllocationPlan([stage(ones(T,2,3),fill(floatmax(T),3,2))])
        w = CorrectorAllocationWorkspace(p)
        out,fb = fill(T(8),2),fill(T(9),3)
        @test process!(out,fb,w,p,ones(T,3)) === CorrectorAllocationNonFiniteOutput
        @test out == fill(T(8),2) && fb == fill(T(9),3)
        s = only(w.stages)
        @test all(iszero,s.held) && s.counter == s.candidate_counter == 0
        @test s.filter.candidate_state == s.filter.committed_state
        @test s.conditioner.candidate_current == s.conditioner.committed_current
        @test process!(out,fb,w,p,zeros(T,3)) === nothing
        @test all(iszero,out) && all(iszero,fb)
    end
end
@testset "Warmed allocation contract" begin
    p=CorrectorAllocationPlan([stage(ones(Float32,1,1),ones(Float32,1,1);period=2,coefficients=identitybank(Float32))])
    w=CorrectorAllocationWorkspace(p); out=zeros(Float32,1);fb=zeros(Float32,1);q=ones(Float32,1)
    @test allocation_bytes(out,fb,w,p,q) == 0
    commit!(w)
    @test allocation_bytes(out,fb,w,p,q) == 0
    q[1]=NaN32
    @test allocation_bytes(out,fb,w,p,q) == 0
    stages=[stage(ones(Float32,1,1),ones(Float32,1,1);period=2,coefficients=identitybank(Float32)) for _ in 1:4]
    p4=CorrectorAllocationPlan(stages);w4=CorrectorAllocationWorkspace(p4)
    out4=zeros(Float32,4);q[1]=2
    @test allocation_bytes(out4,fb,w4,p4,q) == 0
    commit!(w4)
    @test allocation_bytes(out4,fb,w4,p4,q) == 0
    failureplan=CorrectorAllocationPlan([stage(ones(Float32,1,1),ones(Float32,1,1)),stage(fill(floatmax(Float32),1,1),zeros(Float32,1,1))])
    failurework=CorrectorAllocationWorkspace(failureplan);failureout=zeros(Float32,2);q[1]=4
    @test allocation_bytes(failureout,fb,failurework,failureplan,q) == 0
end

function packed_cache_fixture(::Type{T}, kind) where T
    m=kind==:residual ? 10 : 2
    counts=kind==:commands ? (2,8) : kind==:cross_stage ? (2,10) : (2,)
    stages=CorrectorStagePlan{T}[]
    for n in counts
        coefficients=zeros(T,5,1,n)
        coefficients[1,1,:].=one(T)
        coefficients[2,1,:].=T(.125)
        push!(stages,stage(fill(T(.125),n,m),fill(T(.125),m,n);limit=10,coefficients))
    end
    p=CorrectorAllocationPlan(stages);w=CorrectorAllocationWorkspace(p)
    out,fb=zeros(T,p.command_count),zeros(T,m)
    q=fill(T(.5),m)
    process!(out,fb,w,p,q)===nothing || error("packed-cache fixture rejected")
    commit!(w)
    return out,fb,w,p,q
end
function packed_cache_snapshot(out,fb,w,p,q)
    buffers=(out,fb,q,w.commands,w.residual,
        (buffer for (s,sw) in zip(p.stages,w.stages)
            for buffer in (s.filter.mode_coefficients,FilterGraphAlgorithms._corrector_buffers(s,sw)...))...)
    counters=[(sw.counter,sw.candidate_counter) for sw in w.stages]
    return (map(copy,buffers),counters)
end
@testset "Corrector allocation protects packed filter coefficients" begin
    for T in (Float32,Float64), port in (:commands,:feedback,:demand)
        out,fb,w,p,q=packed_cache_fixture(T,:external)
        cache=vec(p.stages[1].filter.mode_coefficients)
        alias=view(cache,1:2)
        port==:commands ? (out=alias) : port==:feedback ? (fb=alias) : (q=alias)
        saved=packed_cache_snapshot(out,fb,w,p,q)
        @test_throws ArgumentError process!(out,fb,w,p,q)
        @test packed_cache_snapshot(out,fb,w,p,q)==saved
    end
    for T in (Float32,Float64), buffer in (:commands,:residual,:cross_stage)
        out,fb,w,p,q=packed_cache_fixture(T,buffer)
        cache=vec(p.stages[1].filter.mode_coefficients)
        # These aliases retain valid public extents: the complete workspace
        # commands/controller residual, or another stage's requested vector.
        if buffer==:commands
            w=CorrectorAllocationWorkspace(w.stages,w.residual,cache)
        elseif buffer==:residual
            w=CorrectorAllocationWorkspace(w.stages,cache,w.commands)
        else
            w.stages[2].requested=cache
        end
        saved=packed_cache_snapshot(out,fb,w,p,q)
        @test_throws ArgumentError process!(out,fb,w,p,q)
        @test packed_cache_snapshot(out,fb,w,p,q)==saved
    end
    for T in (Float32,Float64)
        out,fb,w,p,q=packed_cache_fixture(T,:commands)
        @test allocation_bytes(out,fb,w,p,q)==0
    end
end

@testset "FMA projections: precision, cancellation and finite classification" begin
    for T in (Float32,Float64)
        # Separate product overflows; the grouped fused accumulation is finite.
        largest = floatmax(T)
        p = CorrectorAllocationPlan([stage(reshape(T[-largest,largest],1,2),zeros(T,2,1);limit=largest)])
        w = CorrectorAllocationWorkspace(p);out=zeros(T,1);fb=zeros(T,2)
        @test process!(out,fb,w,p,T[1,2]) === nothing
        @test only(out) == largest
        @test fb == T[1,2]
        @test !isfinite(largest*T(2)-largest)
        # Residual fma likewise avoids overflow of a separately rounded product.
        p = CorrectorAllocationPlan([stage(fill(T(.5),1,1),fill(T(3),1,1);limit=largest)])
        w = CorrectorAllocationWorkspace(p);out=zeros(T,1);fb=zeros(T,1)
        @test process!(out,fb,w,p,T[largest]) === nothing
        @test only(out) == largest/T(2)
        @test only(fb) == -largest/T(2)
        @test !isfinite(largest-T(3)*(largest/T(2)))
        # General ordered dot products have a conservative absolute forward bound.
        for (n,m) in ((3,9),(9,277),(277,253))
            P=[T(mod(7r+3c,23)-11)/T(13) for r in 1:n,c in 1:m]
            q=[T(mod(13i,31)-15)/T(7) for i in 1:m]
            p=CorrectorAllocationPlan([stage(P,zeros(T,m,n);limit=largest)])
            w=CorrectorAllocationWorkspace(p);out=zeros(T,n);fb=zeros(T,m)
            @test process!(out,fb,w,p,q) === nothing
            setprecision(256) do
                u=BigFloat(eps(T))/2;gamma=m*u/(1-m*u)
                for row in 1:n
                    terms=[BigFloat(P[row,col])*BigFloat(q[col]) for col in 1:m]
                    reference=sum(terms)
                    @test abs(BigFloat(w.stages[1].requested[row])-reference) <= gamma*sum(abs,terms)
                end
            end
        end
        # Cancellation can cross a very small range boundary because FMA retains
        # the exact product residual that separate arithmetic rounds to zero.
        e=eps(T);tiny=e*e/T(2)
        P=reshape(T[-1,one(T)+e],1,2);q=T[1,one(T)-e]
        p=CorrectorAllocationPlan([stage(P,zeros(T,2,1);limit=tiny)])
        w=CorrectorAllocationWorkspace(p);out=zeros(T,1);fb=zeros(T,2)
        @test process!(out,fb,w,p,q) === nothing
        @test only(w.stages).requested==T[-e*e]
        @test only(out)==-tiny
        @test P[1,1]*q[1]+P[1,2]*q[2]==zero(T)
        # Limits still clamp at the same representable boundary, including neighbors.
        p=CorrectorAllocationPlan([stage(ones(T,1,1),zeros(T,1,1);limit=T(.5))])
        w=CorrectorAllocationWorkspace(p);out=zeros(T,1);fb=zeros(T,1)
        for q in (prevfloat(T(.5)),T(.5),nextfloat(T(.5)))
            @test process!(out,fb,w,p,T[q]) === nothing
            @test only(out)==min(q,T(.5))
        end
    end
end


@testset "Owned-loop bounds are checked before staging writes" begin
    for T in (Float32,Float64), which in (:requested,:residual)
        p=CorrectorAllocationPlan([stage(ones(T,3,9),zeros(T,9,3))])
        w=CorrectorAllocationWorkspace(p);out=fill(T(7),3);fb=fill(T(9),9)
        which==:requested ? resize!(w.stages[1].requested,2) : resize!(w.residual,8)
        @test_throws DimensionMismatch process!(out,fb,w,p,ones(T,9))
        @test out==fill(T(7),3) && fb==fill(T(9),9)
        @test all(iszero,w.stages[1].filter.committed_state)
        @test all(iszero,w.stages[1].held)
    end
end

end
