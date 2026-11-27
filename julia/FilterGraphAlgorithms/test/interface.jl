module InterfaceTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms:
                              NoWorkspace, abort!, commit!, discard!, finish!, process!,
                              reset!, start!

include("support/interface_fixture.jl")

@testset "plan capabilities are scoped and static" begin
    plan = AddOnePlan()
    @test @inferred(input_progress(AddOnePlan)) isa NoProgress
    @test @inferred(output_progress(AddOnePlan)) isa NoProgress
    @test @inferred(input_progress(plan)) isa NoProgress
    @test @inferred(output_progress(plan)) isa NoProgress
    @test @inferred(property_infos(plan)) == ()
    @test_throws MethodError input_progress(Int)
    @test_throws MethodError output_progress(Int)

    input_only_type = ProgressiveAddOnePlan{false}
    publishable_type = ProgressiveAddOnePlan{true}
    @test @inferred(input_progress(input_only_type)) isa SupportsProgress
    @test @inferred(output_progress(input_only_type)) isa NoProgress
    @test @inferred(input_progress(publishable_type)) isa SupportsProgress
    @test @inferred(output_progress(publishable_type)) isa SupportsProgress
end

@testset "captured execution is conservative and backend-specific" begin
    configuration = DocrimeExcitationF32Configuration(
        amplitudes = Float32[1, 2],
        seed = UInt64(1)
    )
    prepared = prepare_algorithm(DocrimeExcitationF32, configuration)

    @test @inferred(capture_support(DocrimeExcitationF32, nothing)) isa NoCapture
    @test @inferred(capture_support(prepared, nothing)) isa NoCapture
    @test isbitstype(NoCapture)
    @test isbitstype(SupportsCapture)
    @test isbitstype(CompleteTransition)
    @test fieldcount(CompleteTransition) == 0
end

struct ProfileCaptureDevice end

function FilterGraphAlgorithms.prepare_capture(
    ::CompleteTransition, ::ProfileCaptureDevice, prepared::FilterGraphAlgorithms.PreparedAlgorithm,
    outputs::Tuple, inputs::Tuple, status, index::Integer,
)
    return (; prepared, outputs, inputs, status, index)
end

struct LegacyCaptureDevice end
const LEGACY_PREPARATIONS = Ref(0)
function FilterGraphAlgorithms.prepare_capture(
    ::LegacyCaptureDevice, prepared::FilterGraphAlgorithms.PreparedAlgorithm,
    outputs::Tuple, inputs::Tuple, status, index::Integer,
)
    LEGACY_PREPARATIONS[] += 1
    return (; prepared, outputs, inputs, status, index)
end

@testset "capture profiles opt into complete transitions during preparation" begin
    prepared = prepare_algorithm(DocrimeExcitationF32,
        DocrimeExcitationF32Configuration(amplitudes=Float32[1,2], seed=UInt64(1)))
    args = (ProfileCaptureDevice(), prepared, (zeros(Float32, 2),),
        (ones(Float32, 2),), Ref(Int32(0)), Int32(1))
    adapter = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), args...)
    profiled = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), args..., (; lanes=128))
    @test profiled === adapter
    @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(args...)
    @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(args..., nothing)
    for protocol in (Val(0), Val(1), Val(2), Val(3), Val(:unknown))
        @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(protocol, args...)
        @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(protocol, args..., nothing)
    end
    legacy = (LegacyCaptureDevice(), Base.tail(args)...)
    @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(CompleteTransition(), legacy...)
    @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(CompleteTransition(), legacy..., nothing)
    for protocol in (Val(1), Val(2), Val(:unknown)), workgroups in ((), (nothing,))
        @test_throws FilterGraphAlgorithms.CaptureProtocolError FilterGraphAlgorithms.prepare_capture(
            protocol, legacy..., workgroups...)
    end
    @test LEGACY_PREPARATIONS[] == 0
end

@testset "optional capture guarantees are conservative" begin
    adapter = NoWorkspace()
    @test @inferred(FilterGraphAlgorithms.preserves_inputs(adapter)) === false
    @test @inferred(FilterGraphAlgorithms.finite_output(adapter, Val(1))) === false
    @test @inferred(FilterGraphAlgorithms.prepare_validation(adapter, (true, false))) ===
          adapter
    @test !isdefined(FilterGraphAlgorithms, :adopt!)
    @test_throws MethodError FilterGraphAlgorithms.preflight(adapter)
    @test_throws MethodError FilterGraphAlgorithms.enqueue!(adapter)
    @test_throws MethodError FilterGraphAlgorithms.complete!(adapter, Int32(0))
end

@testset "progress results identify immutable output prefixes" begin
    @test isbitstype(ProgressResult)
    @test isempty(ProgressResult())
    @test @inferred(length(ProgressResult())) == 0
    @test !isempty(ProgressResult(12))
    @test @inferred(length(ProgressResult(12))) == 12
    @test_throws ArgumentError ProgressResult(-1)
end

@testset "complete processing uses caller-owned storage" begin
    input = Float32[1, 2, 3]
    output = zeros(Float32, 3)
    plan = AddOnePlan()
    workspace = NoWorkspace()

    @test process!(output, workspace, plan, input) === output
    @test output == Float32[2, 3, 4]
    @test isbitstype(NoWorkspace)
    @test sizeof(workspace) == 0

    short_output = zeros(Float32, 2)
    @test_throws DimensionMismatch(
        "input and output axes must match",
    ) process!(short_output, workspace, plan, input)
    @test iszero(short_output)
end

@testset "progressive processing reports immutable prefixes" begin
    plan = ProgressiveAddOnePlan{true}(4)
    workspace = ProgressiveAddOneWorkspace(false, 0)
    input = Float32[1, 2, 3, 4]
    output = zeros(Float32, 4)

    @test start!(workspace, plan) === workspace
    @test process!(output, workspace, plan, input, 1:2) == ProgressResult(2)
    @test output == Float32[2, 3, 0, 0]
    @test process!(output, workspace, plan, input, 3:4) == ProgressResult(4)
    @test finish!(output, workspace, plan) === output
    @test output == Float32[2, 3, 4, 5]

    input_only_plan = ProgressiveAddOnePlan{false}(4)
    start!(workspace, input_only_plan)
    @test process!(output, workspace, input_only_plan, input, 1:2) ==
          ProgressResult()
    @test abort!(workspace, input_only_plan) === workspace
end

@testset "progressive misuse is rejected before progress advances" begin
    plan = ProgressiveAddOnePlan{true}(4)
    workspace = ProgressiveAddOneWorkspace(false, 0)
    input = Float32[1, 2, 3, 4]
    output = zeros(Float32, 4)

    @test_throws ArgumentError process!(output, workspace, plan, input, 1:1)
    start!(workspace, plan)
    @test_throws ArgumentError start!(workspace, plan)
    @test_throws ArgumentError process!(output, workspace, plan, input, 1:0)
    @test_throws ArgumentError process!(output, workspace, plan, input, 2:2)
    @test_throws BoundsError process!(output, workspace, plan, input, 1:5)
    @test workspace.processed_items == 0
    @test abort!(workspace, plan) === workspace
    @test_throws ArgumentError finish!(output, workspace, plan)
    @test_throws ArgumentError abort!(workspace, plan)
end

@testset "abort preserves committed mathematical history" begin
    plan = StatefulSumPlan(3)
    workspace = StatefulSumWorkspace(false, 0, 10.0, 10.0)
    input = Float64[1, 2, 3]
    output = Ref(0.0)

    start!(workspace, plan)
    @test process!(output, workspace, plan, input, 1:2) == ProgressResult()
    @test output[] == 13.0
    @test workspace.committed_sum == 10.0
    @test abort!(workspace, plan) === workspace
    @test workspace.committed_sum == 10.0
    @test workspace.in_progress_sum == 10.0

    start!(workspace, plan)
    process!(output, workspace, plan, input, 1:3)
    @test finish!(output, workspace, plan) === output
    @test output[] == 16.0
    @test workspace.committed_sum == 16.0
end

@testset "public interface visibility" begin
    @test !Base.ispublic(FilterGraphAlgorithms, :NoWorkspace)
    @test Base.ispublic(FilterGraphAlgorithms, :CompleteTransition)
    @test Base.isexported(FilterGraphAlgorithms, :CompleteTransition)
    for name in (:process!, :commit!, :discard!, :reset!, :start!, :finish!, :abort!)
        @test Base.ispublic(FilterGraphAlgorithms, name)
        @test !Base.isexported(FilterGraphAlgorithms, name)
    end
    for name in (
        :prepare_capture,
        :CaptureProtocolError,
        :preserves_inputs,
        :finite_output,
        :prepare_validation,
        :preflight,
        :enqueue!,
        :complete!,
        :reset!
    )
        @test Base.ispublic(FilterGraphAlgorithms, name)
        @test !Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
