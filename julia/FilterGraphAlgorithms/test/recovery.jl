module RecoveryTests

using FilterGraphAlgorithms, Test
import FilterGraphAlgorithms: commit!, discard!, process!, recover!, reset!

recovery_bytes(workspace) = @allocated recover!(workspace)

@testset "SCC recovery clears exposure history without resetting its watermark" begin
    workspace = ChoppedImageDifferenceWorkspace(ChoppedImageDifferencePlan(Float32, 2, true;
        require_acquisition_identity=true))
    identity = SampleAcquisitionIdentity(ntuple(_ -> UInt8(0x51), 16), 7, 31)
    workspace.committed_identity = identity
    workspace.committed_duration = 100
    workspace.committed_primed = true
    workspace.candidate_ready = true
    fill!(workspace.committed_image, 9)
    fill!(workspace.committed_validity, true)
    @test recover!(workspace) === workspace
    @test workspace.committed_identity === identity
    @test workspace.candidate_identity === identity
    @test workspace.committed_duration == workspace.candidate_duration == 100
    @test !workspace.committed_primed && !workspace.candidate_primed && !workspace.candidate_ready
    @test all(iszero, workspace.committed_image) && all(iszero, workspace.candidate_image)
    @test !any(workspace.committed_validity) && !any(workspace.candidate_validity)
    @test recovery_bytes(workspace) == 0
    reset!(workspace)
    @test workspace.committed_identity === nothing
end

@testset "Protective recovery clears correction and temporal history" begin
    @test Base.ispublic(FilterGraphAlgorithms, :recover!)
    @test_throws MethodError recover!(nothing)
    for T in (Float32, Float64)
        controller = ClosedLoopCorrectionPlan(one(T), one(T), zero(T), zero(T), 2,
            zeros(T, 0, 2), zeros(T, 2, 0))
        workspace = ClosedLoopCorrectionWorkspace(controller)
        correction, state = zeros(T, 2), zeros(T, 2)
        residual, feedback = T[20, -20], zeros(T, 2)
        @test process!(correction, state, workspace, controller, residual, nothing) === nothing
        commit!(workspace)
        @test process!(correction, state, workspace, controller, residual, feedback) === nothing
        commit!(workspace)
        @test correction == T[40, -40]
        # The same zero residual retains the saturation before recovery.
        residual .= zero(T)
        @test process!(correction, state, workspace, controller, residual, feedback) === nothing
        @test correction == T[40, -40]
        # Recovery must remove uncommitted work as well as saturated history.
        @test recover!(workspace) === workspace
        @test !workspace.committed_has_output && !workspace.candidate_has_output
        for buffer in (workspace.committed_state, workspace.candidate_state,
                workspace.committed_output, workspace.candidate_output,
                workspace.realized, workspace.hidden_modes, workspace.hidden_in_state)
            @test all(iszero, buffer)
        end
        @test process!(correction, state, workspace, controller, residual, nothing) === nothing
        @test all(iszero, correction) && all(iszero, state)
        commit!(workspace)
        for _ in 1:3
            @test process!(correction, state, workspace, controller, residual, feedback) === nothing
            @test all(iszero, correction)
            commit!(workspace)
        end
        recover!(workspace)
        @test recovery_bytes(workspace) == 0

        # A one-sample delay makes obsolete filter history directly observable.
        plan = BiquadCascadePlan(reshape(T[0, 1, 0, 0, 0], 5, 1, 1), 2)
        filter = BiquadCascadeWorkspace(plan)
        input, output = T[7, -9], zeros(T, 2)
        @test process!(output, filter, plan, input) === output
        commit!(filter)
        input .= zero(T)
        @test process!(output, filter, plan, input) === output
        @test output == T[7, -9] # Same next sample without recovery carries old history.
        @test recover!(filter) === filter
        @test all(iszero, filter.committed_state)
        @test all(iszero, filter.candidate_state)
        @test all(iszero, filter.candidate_output)
        @test process!(output, filter, plan, input) === output
        @test all(iszero, output)
        recover!(filter)
        @test recovery_bytes(filter) == 0
    end
end

@testset "Leaky recovery clears correction while reset retains configured initial state" begin
    for T in (Float32, Float64)
        initial = T[0.25, -0.5]
        plan = LeakyIntegratorPlan(T(-0.05), one(T))
        workspace = LeakyIntegratorWorkspace(initial)
        input, output = T[-20, 20], zeros(T, 2)
        @test process!(output, workspace, plan, input) === output
        commit!(workspace)
        @test process!(output, workspace, plan, input) === output
        @test workspace.committed_state != initial
        @test workspace.candidate_state != workspace.committed_state
        @test recover!(workspace) === workspace
        @test all(iszero, workspace.committed_state)
        @test all(iszero, workspace.candidate_state)
        @test workspace.initial_state == initial
        @test plan.gain == T(-0.05) && plan.pole == one(T) && plan.closed
        held = LeakyIntegratorPlan(plan.gain, plan.pole; closed=false)
        @test process!(output, workspace, held, input) === output
        @test all(iszero, output)
        commit!(workspace)
        @test process!(output, workspace, plan, zeros(T, 2)) === output
        @test all(iszero, output)
        recover!(workspace)
        @test recovery_bytes(workspace) == 0
        reset!(workspace)
        @test workspace.initial_state == workspace.committed_state == workspace.candidate_state == initial
    end
end

@testset "Conditioner recovery retains nonzero current and slew continuity" begin
    for T in (Float32, Float64)
        plan = PdmCommandPlan(fill(T(-2), 2), fill(T(2), 2), fill(T(0.125), 2))
        workspace = PdmCommandWorkspace(plan, T[0.5, -0.5])
        output, feedback = zeros(T, 2), zeros(T, 2)
        requested = T[4, -4]
        for _ in 1:4
            @test process!(output, feedback, workspace, plan, requested) === nothing
            commit!(workspace)
        end
        current = copy(workspace.committed_current)
        @test current == T[1, -1]
        expected = deepcopy(workspace)
        reference = T[0.25, -0.375]
        @test process!(output, feedback, workspace, plan, reference) === nothing
        @test workspace.candidate_current != current
        @test recover!(workspace) === workspace
        @test workspace.committed_current == workspace.candidate_current == current
        expected_output, expected_feedback = similar(output), similar(feedback)
        @test process!(expected_output, expected_feedback, expected, plan, reference) === nothing
        @test process!(output, feedback, workspace, plan, reference) === nothing
        @test output == expected_output == T[0.875, -0.875]
        @test feedback == expected_feedback == reference - output
        @test abs.(workspace.candidate_current .- current) == fill(T(0.125), 2)
        recover!(workspace)
        @test recovery_bytes(workspace) == 0
        reset!(workspace)
        @test workspace.committed_current == workspace.candidate_current == T[0.5, -0.5]
    end
end

@testset "Allocation recovery preserves slew and unequal update phases" begin
    for T in (Float32, Float64)
        bank = reshape(T[0.5, 0.5, 0, 0, 0], 5, 1, 1)
        periods = (2, 3)
        plan = CorrectorAllocationPlan([
            CorrectorStagePlan(reshape(T[1, 0], 1, 2), reshape(T[1, 0], 2, 1);
                coefficients=bank, lower_limits=T[-1], upper_limits=T[1],
                slew_limits=T[0.125], update_period=periods[1]),
            CorrectorStagePlan(reshape(T[0, 1], 1, 2), reshape(T[0, 1], 2, 1);
                coefficients=bank, lower_limits=T[-1.5], upper_limits=T[1.5],
                slew_limits=T[0.125], update_period=periods[2]),
        ])
        workspace = CorrectorAllocationWorkspace(plan)
        output, feedback = zeros(T, 2), zeros(T, 2)
        saturated = T[20, -20]
        for _ in 1:38
            @test process!(output, feedback, workspace, plan, saturated) === nothing
            commit!(workspace)
        end
        @test output == T[1, -1.5]
        @test Tuple(w.counter for w in workspace.stages) == (0, 2)
        saved = [(current=copy(w.conditioner.committed_current), held=copy(w.held),
            counter=w.counter) for w in workspace.stages]
        # Stage an opposite command; recovery must abandon it without a commit.
        @test process!(output, feedback, workspace, plan, -saturated) === nothing
        @test recover!(workspace) === workspace
        @test workspace.commands == T[1, -1.5]
        @test all(iszero, workspace.residual)
        for (w, old) in zip(workspace.stages, saved)
            @test w.conditioner.committed_current == w.conditioner.candidate_current == old.current
            @test w.commanded == w.held == old.held
            @test w.counter == w.candidate_counter == old.counter
            @test all(iszero, w.filter.committed_state)
            @test all(iszero, w.filter.candidate_state)
            @test all(iszero, w.filter.candidate_output)
            @test all(buffer -> all(iszero, buffer), (w.requested, w.filtered, w.feedback))
        end
        recover!(workspace)
        @test recovery_bytes(workspace) == 0

        reference = T[0.25, -0.375]
        previous = T[1, -1.5]
        for tick in 1:48
            counters = Tuple(w.counter for w in workspace.stages)
            @test process!(output, feedback, workspace, plan, reference) === nothing
            for i in eachindex(periods)
                if iszero(counters[i])
                    @test abs(output[i] - previous[i]) <= T(0.125)
                else
                    @test output[i] == previous[i]
                end
                @test plan.stages[i].conditioner.lower_limits[1] <= output[i] <=
                      plan.stages[i].conditioner.upper_limits[1]
            end
            commit!(workspace)
            @test Tuple(w.counter for w in workspace.stages) ==
                  Tuple(mod(counters[i] + 1, periods[i]) for i in eachindex(periods))
            copyto!(previous, output)
        end
        @test output == reference
        @test all(iszero, feedback)

        # Explicit close with zero residual cannot resurrect the old saturation.
        controller = ClosedLoopCorrectionPlan(one(T), one(T), one(T), zero(T), 2,
            zeros(T, 0, 2), zeros(T, 2, 0))
        correction_workspace = ClosedLoopCorrectionWorkspace(controller)
        correction, state = zeros(T, 2), zeros(T, 2)
        @test process!(correction, state, correction_workspace, controller, saturated, nothing) === nothing
        commit!(correction_workspace)
        recover!(correction_workspace)
        open_controller = FilterGraphAlgorithms.prepare_properties(controller,
            (PropertyAssignment(4, false),))
        residual = zeros(T, 2)
        @test process!(correction, state, correction_workspace, open_controller, residual, nothing) === nothing
        commit!(correction_workspace)
        for _ in 1:6
            @test process!(correction, state, correction_workspace, controller, residual, feedback) === nothing
            commit!(correction_workspace)
            @test all(iszero, correction)
            @test process!(output, feedback, workspace, plan, reference + correction) === nothing
            commit!(workspace)
            @test output == reference
            @test all(iszero, feedback)
        end
    end
end

end
