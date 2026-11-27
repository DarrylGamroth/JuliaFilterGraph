module PhysicalOffsetReferenceRecoveryTests
using Test, FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire
import FilterGraphs: process!, replace_parameters!

function graph_fixture()
    text="""
    filter.graph = {
      nodes = [
        {type=ndarray name=reference label=physical-offset-reference-f32
         config={residual_count=1 physical_count=1 mode_count=1 reference_sign=-1
          residual_schema="$(CONTROLLER_RESIDUAL_ERROR_V1)" physical_schema="$(REQUESTED_PDM_COMMAND_V1)"
          physical_to_modes_schema="test.q/1" modes_to_reference_schema="test.c/1"}}
        {type=ndarray name=correction label=closed-loop-correction-f32
         config={state_length=1 hidden_mode_count=1}
         props={gain=1 pole=0.99 anti-windup-gain=0.99 hidden-mode-gain=0}}
        {type=ndarray name=projection label=dense-vector-reconstructor-f32
         config={input_count=1 output_count=1 initial_reconstructor=[1]
          input_schema="$(CONTROLLER_COMMAND_V1)" output_schema="$(REQUESTED_PDM_COMMAND_V1)"
          reconstructor_schema="test.projection/1"}}
        {type=ndarray name=sum label=vector-linear-combination-f32
         config={extent=1 left_scale=1 right_scale=1 schema="$(REQUESTED_PDM_COMMAND_V1)"}}
        {type=ndarray name=command label=pdm-command-f32
         config={actuator_count=1 command_limits=[-0.8,0.8] slew_limit=0.1
          initial_current=[0.6] quantization_origin=null quantization_step=null dither_seed=null}}
        {type=ndarray name=feedback label=dense-vector-reconstructor-f32
         config={input_count=1 output_count=1 initial_reconstructor=[1]
          input_schema="$(PDM_CONSTRAINT_FEEDBACK_V1)" output_schema="$(CONTROLLER_CONSTRAINT_FEEDBACK_V1)"
          reconstructor_schema="test.feedback/1"}}
      ]
      links = [
        {output="reference:compensated-residual" input="correction:residual-error"}
        {output="correction:correction" input="projection:ordered-input"}
        {output="projection:output" input="sum:left"}
        {output="reference:physical-offset" input="sum:right"}
        {output="sum:output" input="command:requested"}
        {output="command:constraint-feedback" input="feedback:ordered-input"}
      ]
      inputs = [{name=residual endpoint="reference:reconstructed-residual"}
        {name=offset endpoint="reference:offset"} {name=q endpoint="reference:physical-to-modes"}
        {name=c endpoint="reference:modes-to-reference"}
        {name=constraint-feedback endpoint="correction:constraint-feedback"}
        "correction:shape-to-hidden" "correction:hidden-to-shape"]
      outputs = [{name=demanded endpoint="command:demanded"}
        {name=controller-constraint-feedback endpoint="feedback:output"}]
    }
    """
    graph=mktemp() do path,io
        write(io,text);close(io)
        prepare_graph(path;algorithms=algorithms())
    end
    replace_parameters!(graph,:q=>ones(Float32,1,1),:c=>ones(Float32,1,1),:offset=>Float32[0.2],
        Symbol("shape-to-hidden")=>zeros(Float32,1,1),Symbol("hidden-to-shape")=>zeros(Float32,1,1))
    return graph
end
node_index(graph,name)=findfirst(n->n.name===name,graph.nodes)
function successful_frame!(owner)
    output=(demanded=zeros(Float32,1),var"controller-constraint-feedback"=owner.feedback.scratch)
    inputs=(residual=zeros(Float32,1),var"constraint-feedback"=owner.feedback.slot)
    @test all(values(process!(output,owner.graph,inputs)))
    copyto!(owner.feedback.slot,owner.feedback.scratch)
    FilterGraphPipeWire._observe_recovery_reference!(owner)
    owner.process_count+=UInt64(1)
    return output.demanded
end

@testset "recovery admits only the adopted physical offset output" begin
    graph=graph_fixture()
    owner=FilterGraphPipeWire.PreparedGraphOwner(graph,nothing;
        feedback=Symbol("constraint-feedback")=>Symbol("controller-constraint-feedback"))
    scope=prepare_recovery(owner;controller=:correction,participants=(:command,),reference_timeout_ms=60000)
    initial_status=FilterGraphPipeWire.parameter_status(owner,:reference)
    @test initial_status.requested_parameter_sequence==initial_status.active_parameter_sequence
    @test !initial_status.closure_inhibited
    @test_throws ArgumentError FilterGraphPipeWire.parameter_status(owner,:missing)
    reference_index=node_index(graph,:reference)
    prepared=graph.nodes[reference_index].prepared
    calibration=prepared.plan
    required=falses(length(graph.nodes))
    physical=FilterGraphs.NodeOutput{reference_index,2}()
    compensated=FilterGraphs.NodeOutput{reference_index,1}()
    @test FilterGraphPipeWire._recovery_reference_route(graph,scope.controller_index,scope.selected,required,physical)==UInt8(1)
    @test_throws ArgumentError FilterGraphPipeWire._recovery_reference_route(
        graph,scope.controller_index,scope.selected,required,compensated)
    temporal=FilterGraphAlgorithms.BiquadCascadePlan(reshape(Float32[0,1,0,0,0],5,1,1),1)
    @test_throws ArgumentError FilterGraphPipeWire._recovery_reference_transform(temporal,UInt8(1),true)
    clear=FilterGraphs._prepare_parameter_transaction(graph,:offset=>Float32[0])
    nonzero=FilterGraphs._prepare_parameter_transaction(graph,:offset=>Float32[0.1])
    factors=FilterGraphs._prepare_parameter_transaction(graph,:offset=>Float32[0],:q=>ones(Float32,1,1))
    @test FilterGraphPipeWire._recovery_offset_clear_candidate(graph,clear)
    @test !FilterGraphPipeWire._recovery_offset_clear_candidate(graph,nonzero)
    @test !FilterGraphPipeWire._recovery_offset_clear_candidate(graph,factors)
    generation=stage_recovery!(scope)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    for value in (0.1f0,NaN32,Inf32)
        @test_throws InvalidStateException FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[value])
    end
    @test_throws InvalidStateException FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:q=>ones(Float32,1,1))
    @test_throws InvalidStateException FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,
        :offset=>Float32[0],:q=>ones(Float32,1,1))
    for _ in 1:8
        successful_frame!(owner)
    end
    @test recovery_status(scope).reference_reached
    @test prepared.plan===calibration
    @test prepared.plan.offset==Float32[0.2] && prepared.plan.correction==Float32[-0.2]
    @test only(graph.nodes[node_index(graph,:command)].prepared.workspace.committed_current)≈0.2f0
    # A stale physical offset can be cleared while ordinary mutation remains
    # inhibited. Its new paired zero correction must settle before release.
    @test recovery_status(scope).closure_inhibited
    @test FilterGraphPipeWire.parameter_status(owner,:reference).closure_inhibited
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0])
    pending=FilterGraphPipeWire.parameter_status(owner,:reference)
    @test pending.requested_parameter_sequence>pending.active_parameter_sequence
    @test_throws InvalidStateException FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,
        :offset=>Float32[0.1];supersede=true)
    @test FilterGraphPipeWire.parameter_status(owner,:reference)==pending
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0];supersede=true)
    pending=FilterGraphPipeWire.parameter_status(owner,:reference)
    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
    adopted=FilterGraphPipeWire.parameter_status(owner,:reference)
    @test adopted.requested_parameter_sequence==adopted.active_parameter_sequence
    @test !recovery_status(scope).reference_reached
    @test_throws InvalidStateException stage_recovery_release!(scope,generation)
    @test prepared.plan!==calibration
    @test prepared.plan.physical_to_modes===calibration.physical_to_modes
    @test prepared.plan.modes_to_reference===calibration.modes_to_reference
    @test all(iszero,prepared.plan.offset) && all(iszero,prepared.plan.correction)
    successful_frame!(owner)
    @test !recovery_status(scope).reference_reached # slew-limited current still settling
    for _ in 1:8
        successful_frame!(owner)
    end
    @test recovery_status(scope).reference_reached
    @test iszero(only(graph.nodes[node_index(graph,:command)].prepared.workspace.committed_current))
    stage_recovery_release!(scope,generation)
    FilterGraphPipeWire._adopt_pending_recovery!(owner)
    @test !recovery_status(scope).closure_inhibited
    current=prepared.plan
    @test_throws ArgumentError FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,
        :offset=>Float32[0.1],:q=>fill(NaN32,1,1))
    @test prepared.plan===current
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.1])
    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
    @test !recovery_status(scope).reference_reached
    successful_frame!(owner)
    @test recovery_status(scope).reference_reached
    snapshot=@atomic :acquire owner.control_snapshot
    @test snapshot.active_parameter_sequences[reference_index]==snapshot.requested_parameter_sequences[reference_index]
end
@testset "pending physical offsets can be superseded before their first frame" begin
    for replacement in (0f0,0.05f0)
        graph=graph_fixture();owner=FilterGraphPipeWire.PreparedGraphOwner(graph)
        prepared=graph.nodes[node_index(graph,:reference)].prepared
        @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.3])
        original=prepared.plan
        pending=FilterGraphPipeWire.parameter_status(owner,:reference)
        @test pending.requested_parameter_sequence>pending.active_parameter_sequence
        @test !FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[replacement])
        @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[replacement];supersede=true)
        latest=FilterGraphPipeWire.parameter_status(owner,:reference)
        @test latest.requested_parameter_sequence==pending.requested_parameter_sequence+1
        @test latest.active_parameter_sequence==pending.active_parameter_sequence
        @test prepared.plan===original
        @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
        input=(residual=zeros(Float32,1),var"constraint-feedback"=zeros(Float32,1))
        output=(demanded=zeros(Float32,1),var"controller-constraint-feedback"=zeros(Float32,1))
        @test all(values(process!(output,graph,input)))
        reference_output=graph.node_outputs[node_index(graph,:reference)]
        @test only(reference_output[2])==replacement
        @test only(reference_output[1])==-replacement
        @test FilterGraphPipeWire.parameter_status(owner,:reference).active_parameter_sequence==latest.requested_parameter_sequence
        @test owner.parameter_adoption_count==1
    end
end

@testset "failed supersession retains pending data and capacity-one semantics" begin
    graph=graph_fixture();owner=FilterGraphPipeWire.PreparedGraphOwner(graph)
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.3])
    transaction=owner.parameter_transaction
    snapshot=@atomic :acquire owner.control_snapshot
    pending_snapshot=owner.pending_control_snapshot
    for (failure,updates) in ((ArgumentError,(:offset=>Float32[NaN],)),
            (DimensionMismatch,(:offset=>Float32[0,0],)),(ArgumentError,(:q=>ones(Float32,1,1),)),
            (ArgumentError,(:offset=>Float32[0],:q=>ones(Float32,1,1))))
        @test_throws failure FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,updates...;supersede=true)
        @test owner.parameter_transaction===transaction
        @test owner.pending_control_snapshot===pending_snapshot
        @test (@atomic :acquire owner.control_snapshot)===snapshot
        @test owner.control_state[]==FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_PENDING
    end
    # Once a callback claims adoption, supersession must return busy without
    # modifying either snapshot or exposing half of the parameter handoff.
    owner.control_state[]=FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_ADOPTING
    @test !FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0];supersede=true)
    @test owner.parameter_transaction===transaction
    owner.control_state[]=FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_PENDING
    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
    @test only(graph.nodes[node_index(graph,:reference)].prepared.plan.offset)==0.3f0
    # A pending calibration/offset batch must retain its original transaction;
    # a later offset-only request cannot discard that calibration replacement.
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.1],:q=>ones(Float32,1,1))
    transaction=owner.parameter_transaction;snapshot=@atomic :acquire owner.control_snapshot
    @test_throws ArgumentError FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0];supersede=true)
    @test owner.parameter_transaction===transaction && (@atomic :acquire owner.control_snapshot)===snapshot
    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
end

struct BlockingOffset <: AbstractVector{Float32}
    values::Vector{Float32}
    entered::Channel{Nothing}
    release::Channel{Nothing}
    fail::Bool
end
Base.size(x::BlockingOffset)=size(x.values)
Base.getindex(x::BlockingOffset,i::Int)=x.values[i]
Base.IndexStyle(::Type{BlockingOffset})=IndexLinear()
function Base.copy(x::BlockingOffset)
    put!(x.entered,nothing)
    take!(x.release)
    x.fail && throw(ArgumentError("injected cold offset copy failure"))
    return copy(x.values)
end

@testset "cold supersession and callback adoption have exclusive ownership" begin
    # Two default threads additionally exercise parallel preparation; Channels
    # retain the same deterministic ownership checkpoints on a single thread.
    for fail in (false,true)
        graph=graph_fixture();owner=FilterGraphPipeWire.PreparedGraphOwner(graph)
        prepared=graph.nodes[node_index(graph,:reference)].prepared
        active=prepared.plan
        @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.3])
        prior=owner.parameter_transaction;status=FilterGraphPipeWire.parameter_status(owner,:reference)
        entered=Channel{Nothing}(1);release=Channel{Nothing}(1)
        update=BlockingOffset(Float32[0.05],entered,release,fail)
        task=Threads.@spawn try
            FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>update;supersede=true)
        catch error
            error
        end
        take!(entered)
        @test owner.control_state[]==FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_PREPARING
        @test !FilterGraphPipeWire._adopt_pending_parameters!(owner)
        @test prepared.plan===active
        @test FilterGraphPipeWire.parameter_status(owner,:reference)==status
        input=(residual=zeros(Float32,1),var"constraint-feedback"=zeros(Float32,1))
        output=(demanded=zeros(Float32,1),var"controller-constraint-feedback"=zeros(Float32,1))
        @test all(values(process!(output,graph,input)))
        @test only(graph.node_outputs[node_index(graph,:reference)][2])==0.2f0
        put!(release,nothing)
        result=fetch(task)
        if fail
            @test result isa ArgumentError
            @test owner.parameter_transaction===prior
            @test FilterGraphPipeWire.parameter_status(owner,:reference)==status
        else
            @test result===true
            @test owner.parameter_transaction!==prior
            @test FilterGraphPipeWire.parameter_status(owner,:reference).active_parameter_sequence==status.active_parameter_sequence
        end
        @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
        @test only(prepared.plan.offset)==(fail ? 0.3f0 : 0.05f0)
    end
end

allocated_adoption(owner)=@allocated FilterGraphPipeWire._adopt_pending_parameters!(owner)
@testset "callback parameter adoption retains its zero allocation budget" begin
    graph=graph_fixture();owner=FilterGraphPipeWire.PreparedGraphOwner(graph)
    allocated_adoption(owner)
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.3])
    @test allocated_adoption(owner)==0
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0.1])
    @test FilterGraphPipeWire._stage_parameter_transaction!(owner,nothing,:offset=>Float32[0];supersede=true)
    @test allocated_adoption(owner)==0
end

end
