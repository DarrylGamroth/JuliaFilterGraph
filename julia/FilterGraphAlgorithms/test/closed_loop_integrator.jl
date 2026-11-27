module ClosedLoopIntegratorTests
using Test, FilterGraphAlgorithms
import FilterGraphAlgorithms: process!, reset!

@testset "closed-loop integration without hidden wire ports" begin
    p=prepare_algorithm(ClosedLoopIntegratorF32,ClosedLoopIntegratorF32Configuration(state_length=2))
    @test p.plan.hidden_mode_count == 0
    @test size(p.plan.shape_to_hidden) == (0,2)
    @test length(algorithm_ports(ClosedLoopIntegratorF32)) == 4
    @test all(port->port.role isa FilterGraphAlgorithms.FrameData,algorithm_ports(ClosedLoopIntegratorF32))
    p.plan=FilterGraphAlgorithms.prepare_properties(p.plan,
        (PropertyAssignment(0,0.3f0),PropertyAssignment(1,0.8f0),PropertyAssignment(2,0.8f0)))
    output=zeros(Float32,2);state=similar(output)
    residual=Float32[1,-2];feedback=Float32[NaN,NaN]
    @test process!(output,state,p,residual,feedback) === nothing
    @test output ≈ 0.3f0 .* residual
    previous=copy(output);feedback .= Float32[0.25,-0.5]
    @test process!(output,state,p,residual,feedback) === nothing
    @test output ≈ 0.8f0 .* (previous .- 0.8f0 .* feedback) .+ 0.3f0 .* residual
    previous=copy(output);committed=copy(closed_loop_correction_state(p.workspace))
    feedback[end]=NaN32
    @test process!(output,state,p,residual,feedback) === ClosedLoopCorrectionNonFiniteFeedback
    @test output == previous
    @test closed_loop_correction_state(p.workspace) == committed
    reset!(p)
    @test closed_loop_correction_last_output(p.workspace) === nothing
    @test process!(output,state,p,residual,feedback) === nothing
    @test output ≈ 0.3f0 .* residual
    feedback .= 0
    measure(output,state,p,residual,feedback)=@allocated process!(output,state,p,residual,feedback)
    measure(output,state,p,residual,feedback)
    @test measure(output,state,p,residual,feedback) == 0
    held_output = copy(output); held_state = copy(state)
    p.plan = FilterGraphAlgorithms.prepare_properties(p.plan, (PropertyAssignment(4, false),))
    residual .= floatmax(Float32); feedback .= floatmax(Float32)
    @test process!(output,state,p,residual,feedback) === nothing
    @test output == held_output
    @test state == held_state
    @test measure(output,state,p,residual,feedback) == 0
    @test_throws ArgumentError prepare_algorithm(ClosedLoopIntegratorF32,ClosedLoopIntegratorF32Configuration(state_length=0))
    @test_throws ArgumentError prepare_algorithm(ClosedLoopIntegratorF32,ClosedLoopIntegratorF32Configuration(state_length=typemax(Int)))
end

@testset "invalid measurements hold feedback-aware integration" begin
    p = prepare_algorithm(ValidityGatedClosedLoopIntegratorF32,
        ValidityGatedClosedLoopIntegratorF32Configuration(state_length=2,
            validity_length=1, validity_schema=CENTER_OF_GRAVITY_VALIDITY_V1))
    p.plan = FilterGraphAlgorithms.prepare_properties(p.plan,
        (PropertyAssignment(0,0.3f0),PropertyAssignment(1,0.8f0),PropertyAssignment(2,0.5f0)))
    output = fill(99f0,2); state = similar(output)
    residual = fill(NaN32,2); feedback = fill(NaN32,2); validity = [false]
    @test process!(output,state,p,residual,feedback,validity) === nothing
    @test output == state == zeros(Float32,2)
    validity[1] = true; residual .= Float32[1,-2]; feedback .= 0
    @test process!(output,state,p,residual,feedback,validity) === nothing
    held = copy(output); held_state = copy(state)
    validity[1] = false; residual .= NaN32
    for value in (1f0, 2f0, NaN32)
        feedback .= value
        @test process!(output,state,p,residual,feedback,validity) === nothing
        @test output == held && state == held_state
        @test closed_loop_correction_last_output(p.workspace) == held
        @test closed_loop_correction_state(p.workspace) == held_state
    end
    # The owner replaces skipped feedback; consume only the latest predecessor.
    validity[1] = true; residual .= 0; feedback .= 0.1f0
    @test process!(output,state,p,residual,feedback,validity) === nothing
    @test output ≈ 0.8f0 .* (held .- 0.5f0 .* feedback)
    validity[1] = false
    measure(output,state,p,residual,feedback,validity) = @allocated process!(output,state,p,residual,feedback,validity)
    measure(output,state,p,residual,feedback,validity)
    @test measure(output,state,p,residual,feedback,validity) == 0
    reset!(p)
    @test closed_loop_correction_last_output(p.workspace) === nothing
    @test process!(output,state,p,residual,feedback,validity) === nothing
    @test output == state == zeros(Float32,2)
end
end
