module LeakyIntegratorTests
using FilterGraphAlgorithms, Test
import FilterGraphAlgorithms: process!, commit!, discard!, reset!

allocation_bytes(output, workspace, plan, input) = @allocated process!(output, workspace, plan, input)

@testset "leaky state inputs materialize owned mutable vectors" begin
    for T in (Float32, Float64), state in (T[1, 2, 3], view(T[0, 1, 2, 3, 4], 2:4), T(1):T(3))
        original = collect(state)
        workspace = LeakyIntegratorWorkspace(state)
        @test workspace.initial_state isa Vector{T}
        @test !Base.mightalias(workspace.initial_state, state)
        plan = LeakyIntegratorPlan(T(0.25), T(0.5))
        input, output = T[4, -4, 0], fill(T(71), 3)
        expected = T[1.5, 0, 1.5]
        @test process!(output, workspace, plan, input) === output
        @test output == expected
        @test leaky_integrator_state(workspace) == original
        @test discard!(workspace) === workspace
        @test workspace.candidate_state == original
        process!(output, workspace, plan, input)
        commit!(workspace)
        @test leaky_integrator_state(workspace) == expected
        reset!(workspace)
        @test leaky_integrator_state(workspace) == original
        @test collect(state) == original
        @test allocation_bytes(output, workspace, plan, input) == 0
    end
end

@testset "leaky candidate arithmetic preserves IEEE values and rejection precedence" begin
    for T in (Float32, Float64)
        input = repeat(T[zero(T), -zero(T), nextfloat(zero(T)), -nextfloat(zero(T)), 1, -1, 0.1, -0.7], 128)
        state = reverse(input)
        plan = LeakyIntegratorPlan(T(0.3), T(0.7))
        workspace = LeakyIntegratorWorkspace(state)
        output = similar(input)
        # Retain the historical separate-arithmetic reference for compatibility.
        state_terms = plan.pole .* state
        input_terms = plan.gain .* input
        expected = state_terms .+ input_terms
        @test process!(output, workspace, plan, input) === output
        @test isapprox(output, expected; atol=4eps(T), rtol=4eps(T))
        rounded = setprecision(BigFloat, 4096) do
            T[ T(BigFloat(plan.gain) * BigFloat(input[i]) +
                 BigFloat(state_terms[i])) for i in eachindex(input)]
        end
        @test isequal(output, rounded)
        @test allocation_bytes(output, workspace, plan, input) == 0
        overflow_plan = LeakyIntegratorPlan(floatmax(T), one(T))
        fill!(output, T(71))
        input[1], input[end] = T(2), T(NaN)
        @test process!(output, workspace, overflow_plan, input) === LeakyIntegratorNonFiniteInput
        @test output == fill(T(71), length(output))
        @test isequal(leaky_integrator_state(workspace), state)
        @test isequal(workspace.candidate_state, state)
        input[end] = zero(T)
        @test process!(output, workspace, overflow_plan, input) === LeakyIntegratorNonFiniteOutput
        @test output == fill(T(71), length(output))
        @test isequal(workspace.candidate_state, state)
        @test allocation_bytes(output, workspace, overflow_plan, input) == 0
    end
end

@testset "leaky checked processing restores candidate state after rejection" begin
    prepared = prepare_algorithm(LeakyIntegratorF32, LeakyIntegratorF32Configuration(;
        extent=2, initial_state=1.0f0, input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        output_schema=CONTROLLER_RESIDUAL_ERROR_V1))
    output = zeros(Float32, 2)
    @test process!(output, prepared, Float32[2, 3]) === nothing
    input = Float32[4, NaN32]
    @test process!(output, prepared, input) === LeakyIntegratorNonFiniteInput
    @test output == Float32[2, 3]
    @test prepared.workspace.candidate_state == prepared.workspace.committed_state == output
    @test process!(output, prepared, Float32[4, 5]) === nothing
    @test output == Float32[4, 5]
end

@testset "leaky external overlap is staged and owned aliases are rejected" begin
    plan = LeakyIntegratorPlan(0.5f0, 0.25f0)
    for shift in (-1, 0, 1)
        storage = Float32[1, 2, 3, 4, 5]
        input = view(storage, 2:4)
        output = view(storage, (2 + shift):(4 + shift))
        workspace = LeakyIntegratorWorkspace(Float32[4, 4, 4])
        expected = 1f0 .+ 0.5f0 .* copy(input)
        @test process!(output, workspace, plan, input) === output
        @test output == expected
    end
    workspace = LeakyIntegratorWorkspace(Float32[1, 2, 3])
    input, output = ones(Float32, 3), fill(71f0, 3)
    for owned in (workspace.initial_state, workspace.committed_state, workspace.candidate_state)
        @test_throws ArgumentError process!(owned, workspace, plan, input)
        @test_throws ArgumentError process!(output, workspace, plan, owned)
        @test leaky_integrator_state(workspace) == Float32[1, 2, 3]
        @test output == fill(71f0, 3)
    end
end
@testset "leaky FMA retains finite cancellation and follows high precision trajectories" begin
    for T in (Float32,Float64)
        plan=LeakyIntegratorPlan(T(2),one(T))
        workspace=LeakyIntegratorWorkspace(T[-floatmax(T)])
        output=T[0];input=T[floatmax(T)]
        @test !isfinite(plan.gain * input[1])
        @test process!(output,workspace,plan,input) === output
        @test output == T[floatmax(T)]
        @test leaky_integrator_state(workspace) == T[-floatmax(T)]
        @test allocation_bytes(output,workspace,plan,input)==0
        discard!(workspace)
        @test leaky_integrator_state(workspace) == T[-floatmax(T)]
        plan=LeakyIntegratorPlan(T(.37),T(.91))
        state=T[.1,-.3,.7];workspace=LeakyIntegratorWorkspace(state)
        output=similar(state);old=copy(state);rounded=copy(state)
        setprecision(BigFloat,4096) do
            for frame in 1:512
                input=T[T(sin(frame+i))/T(3) for i in 1:3]
                rounded=T[T(BigFloat(plan.gain)*BigFloat(input[i])+BigFloat(plan.pole*rounded[i])) for i in 1:3]
                old=plan.pole .* old .+ plan.gain .* input
                @test process!(output,workspace,plan,input) === output
                @test isequal(output,rounded)
                # Geometric roundoff accumulation bound for pole=.91, with
                # O(eps) local error and |state|, |input| below one.
                @test all(abs.(output.-old) .<= T(8)*eps(T)/(one(T)-plan.pole))
                commit!(workspace)
                @test isequal(leaky_integrator_state(workspace),rounded)
            end
        end
    end
end

end
