module ClosedLoopCorrectionTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "closed_loop_correction.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_plan()
    configuration = FIXTURE["configuration"]
    properties = FIXTURE["properties"]
    parameters = FIXTURE["parameters"]
    state_length = configuration["state_length"]
    hidden_mode_count = configuration["hidden_mode_count"]
    return ClosedLoopCorrectionPlan(
        Float32(properties["gain"]),
        Float32(properties["pole"]),
        Float32(properties["anti_windup_gain"]),
        Float32(properties["hidden_mode_gain"]),
        state_length,
        row_major_matrix(
            Float32,
            parameters["shape_to_hidden"],
            hidden_mode_count,
            state_length,
        ),
        row_major_matrix(
            Float32,
            parameters["hidden_to_shape"],
            state_length,
            hidden_mode_count,
        ),
    )
end

function process_allocation_bytes(
    correction,
    controller_state,
    workspace,
    plan,
    residual_error,
    constraint_feedback,
)
    return @allocated process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        constraint_feedback,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

function declared_process_allocation_bytes(
    correction,
    controller_state,
    prepared,
    residual_error,
    constraint_feedback,
)
    return @allocated process!(
        correction,
        controller_state,
        prepared,
        residual_error,
        constraint_feedback,
    )
end

function fixture_prepared()
    configuration = FIXTURE["configuration"]
    properties = FIXTURE["properties"]
    parameters = FIXTURE["parameters"]
    prepared = prepare_algorithm(
        ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(
            state_length=configuration["state_length"],
            hidden_mode_count=configuration["hidden_mode_count"],
        ),
    )
    replace_parameter!(
        prepared,
        "shape-to-hidden",
        row_major_matrix(
            Float32,
            parameters["shape_to_hidden"],
            configuration["hidden_mode_count"],
            configuration["state_length"],
        ),
    )
    replace_parameter!(
        prepared,
        "hidden-to-shape",
        row_major_matrix(
            Float32,
            parameters["hidden_to_shape"],
            configuration["state_length"],
            configuration["hidden_mode_count"],
        ),
    )
    prepared.plan = FilterGraphAlgorithms.prepare_properties(
        prepared.plan,
        (
            PropertyAssignment(0, Float32(properties["gain"])),
            PropertyAssignment(1, Float32(properties["pole"])),
            PropertyAssignment(2, Float32(properties["anti_windup_gain"])),
            PropertyAssignment(3, Float32(properties["hidden_mode_gain"])),
        ),
    )
    return prepared
end

@testset "closed-loop correction matches the repository fixture" begin
    plan = fixture_plan()
    workspace = ClosedLoopCorrectionWorkspace(plan)
    state_length = FIXTURE["configuration"]["state_length"]
    correction = fill(-1.0f0, state_length)
    controller_state = fill(-2.0f0, state_length)
    expected_committed_state = zeros(Float32, state_length)
    expected_last_output = nothing

    for frame in FIXTURE["frames"]
        residual_error = Float32.(frame["residual_error"])
        constraint_feedback = frame["has_prior_feedback"] ?
                              Float32.(frame["constraint_feedback"]) : nothing

        @test process!(
            correction,
            controller_state,
            workspace,
            plan,
            residual_error,
            constraint_feedback,
        ) === nothing
        @test closed_loop_correction_state(workspace) == expected_committed_state
        @test closed_loop_correction_last_output(workspace) == expected_last_output
        @test correction == Float32.(frame["correction"])
        @test controller_state == Float32.(frame["controller_state"])

        @test commit!(workspace) === workspace
        expected_committed_state = Float32.(frame["controller_state"])
        expected_last_output = Float32.(frame["correction"])
        @test closed_loop_correction_state(workspace) == expected_committed_state
        @test closed_loop_correction_last_output(workspace) == expected_last_output
    end

    @test reset!(workspace) === workspace
    @test closed_loop_correction_state(workspace) == zeros(Float32, state_length)
    @test closed_loop_correction_last_output(workspace) === nothing
end

@testset "closed-loop correction rejects without changing committed state" begin
    plan = fixture_plan()
    workspace = ClosedLoopCorrectionWorkspace(plan)
    correction = fill(71.0f0, 2)
    controller_state = fill(72.0f0, 2)
    residual_error = Float32[1, -1]
    feedback = Float32[0.25, -0.5]

    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        feedback,
    ) === ClosedLoopCorrectionUnexpectedFeedback
    @test correction == fill(71.0f0, 2)
    @test controller_state == fill(72.0f0, 2)
    @test closed_loop_correction_state(workspace) == zeros(Float32, 2)
    @test closed_loop_correction_last_output(workspace) === nothing
    discard!(workspace)

    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    ) === nothing
    commit!(workspace)
    committed_state = copy(closed_loop_correction_state(workspace))
    committed_output = copy(closed_loop_correction_last_output(workspace))

    correction .= 71.0f0
    controller_state .= 72.0f0
    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    ) === ClosedLoopCorrectionMissingFeedback
    @test correction == fill(71.0f0, 2)
    @test controller_state == fill(72.0f0, 2)
    @test closed_loop_correction_state(workspace) == committed_state
    @test closed_loop_correction_last_output(workspace) == committed_output
    discard!(workspace)

    nonfinite_input = Float32[NaN, 0]
    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        nonfinite_input,
        feedback,
    ) === ClosedLoopCorrectionNonFiniteInput
    discard!(workspace)

    nonfinite_feedback = Float32[Inf, 0]
    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nonfinite_feedback,
    ) === ClosedLoopCorrectionNonFiniteFeedback
    discard!(workspace)
    @test closed_loop_correction_state(workspace) == committed_state
    @test closed_loop_correction_last_output(workspace) == committed_output
end

@testset "closed-loop correction admitted calls allocate no Julia heap memory" begin
    plan = fixture_plan()
    workspace = ClosedLoopCorrectionWorkspace(plan)
    correction = zeros(Float32, 2)
    controller_state = zeros(Float32, 2)
    residual_error = Float32[1, -1]
    feedback = Float32[0.25, -0.5]

    process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    )
    discard!(workspace)
    @test process_allocation_bytes(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    ) == 0
    commit!(workspace)

    process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        feedback,
    )
    discard!(workspace)
    @test process_allocation_bytes(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        feedback,
    ) == 0

    process!(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    )
    discard!(workspace)
    @test process_allocation_bytes(
        correction,
        controller_state,
        workspace,
        plan,
        residual_error,
        nothing,
    ) == 0

    @test workspace_commit_allocation_bytes(workspace) == 0
    @test workspace_discard_allocation_bytes(workspace) == 0
    @test workspace_reset_allocation_bytes(workspace) == 0
end

@testset "closed-loop correction validates preparation and supports Float64" begin
    @test_throws ArgumentError ClosedLoopCorrectionPlan(
        0.5f0,
        1.1f0,
        0.75f0,
        0.5f0,
        2,
        reshape(Float32[1, -1], 1, 2),
        reshape(Float32[0.5, -0.5], 2, 1),
    )
    @test_throws DimensionMismatch ClosedLoopCorrectionPlan(
        0.5f0,
        0.75f0,
        0.75f0,
        0.5f0,
        2,
        zeros(Float32, 1, 1),
        zeros(Float32, 2, 1),
    )

    plan = ClosedLoopCorrectionPlan(
        0.5,
        0.75,
        0.75,
        0.0,
        2,
        zeros(0, 2),
        zeros(2, 0),
    )
    workspace = ClosedLoopCorrectionWorkspace(plan)
    correction = zeros(2)
    controller_state = zeros(2)
    @test process!(
        correction,
        controller_state,
        workspace,
        plan,
        [2.0, -2.0],
        nothing,
    ) === nothing
    @test correction == [1.0, -1.0]
end

@testset "closed-loop correction declaration preserves the FGN contract" begin
    @test algorithm_label(ClosedLoopCorrectionF32) ==
          "closed-loop-correction-f32"
    @test algorithm_property_policy(ClosedLoopCorrectionF32) isa
          RuntimeDeclaredProperties
    @test FilterGraphAlgorithms.algorithm_failure_type(ClosedLoopCorrectionF32) ===
          ClosedLoopCorrectionFailure

    configuration = algorithm_configuration(ClosedLoopCorrectionF32)
    @test map(field -> field.name, configuration) ==
          (:state_length, :hidden_mode_count)
    @test all(field -> field.update isa GraphRebuildConfiguration, configuration)

    ports = algorithm_ports(ClosedLoopCorrectionF32)
    @test map(port -> port.name, ports) == (
        Symbol("residual-error"),
        Symbol("constraint-feedback"),
        :correction,
        Symbol("controller-state"),
        Symbol("shape-to-hidden"),
        Symbol("hidden-to-shape"),
    )
    @test all(port -> port.element_type === Float32, ports)
    @test all(
        port -> port.role isa FilterGraphAlgorithms.FrameData,
        ports[1:4],
    )
    @test all(
        port -> port.role isa FilterGraphAlgorithms.SparseParameter,
        ports[5:6],
    )
    @test ports[3].metadata_source === Symbol("residual-error")
    @test ports[4].metadata_source === Symbol("residual-error")

    prepared = fixture_prepared()
    formats = algorithm_formats(prepared)
    @test map(format -> format.shape, formats) ==
          ((2,), (2,), (2,), (2,), (1, 2), (2, 1))
    @test map(format -> format.schema, formats) == (
        CONTROLLER_RESIDUAL_ERROR_V1,
        CONTROLLER_CONSTRAINT_FEEDBACK_V1,
        CONTROLLER_COMMAND_V1,
        CONTROLLER_STATE_V1,
        SHAPE_TO_HIDDEN_PROJECTION_V1,
        HIDDEN_TO_SHAPE_PROJECTION_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, formats)

    infos = property_infos(prepared.plan)
    @test map(info -> info.id, infos) == (UInt32(0), UInt32(1), UInt32(2), UInt32(3), UInt32(4))
    @test map(info -> info.name, infos) == (
        :gain,
        :pole,
        Symbol("anti-windup-gain"),
        Symbol("hidden-mode-gain"),
        :closed,
    )
    @test property_values(prepared.plan) == (
        gain=0.5f0,
        pole=0.75f0,
        var"anti-windup-gain"=0.75f0,
        var"hidden-mode-gain"=0.5f0,
        closed=true,
    )

    correction = fill(-1.0f0, 2)
    controller_state = fill(-2.0f0, 2)
    for frame in FIXTURE["frames"]
        @test process!(
            correction,
            controller_state,
            prepared,
            Float32.(frame["residual_error"]),
            Float32.(frame["constraint_feedback"]),
        ) === nothing
        @test correction == Float32.(frame["correction"])
        @test controller_state == Float32.(frame["controller_state"])
        @test closed_loop_correction_state(prepared.workspace) == controller_state
        @test closed_loop_correction_last_output(prepared.workspace) == correction
    end

    @test reset!(prepared) === prepared
    @test closed_loop_correction_state(prepared.workspace) == zeros(Float32, 2)
    @test closed_loop_correction_last_output(prepared.workspace) === nothing
end

@testset "declared closed-loop replacement and processing are failure-atomic" begin
    prepared = fixture_prepared()
    active_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "shape-to-hidden",
        zeros(Float32, 2, 1),
    )
    @test prepared.plan === active_plan
    invalid_projection = Float32[NaN -1]
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "shape-to-hidden",
        invalid_projection,
    )
    @test prepared.plan === active_plan

    first = FIXTURE["frames"][1]
    correction = zeros(Float32, 2)
    controller_state = zeros(Float32, 2)
    @test process!(
        correction,
        controller_state,
        prepared,
        Float32.(first["residual_error"]),
        Float32.(first["constraint_feedback"]),
    ) === nothing
    committed_state = copy(closed_loop_correction_state(prepared.workspace))
    committed_output = copy(closed_loop_correction_last_output(prepared.workspace))

    correction .= 71.0f0
    controller_state .= 72.0f0
    rejection = process!(
        correction,
        controller_state,
        prepared,
        Float32[0, 0],
        Float32[NaN, 0],
    )
    @test rejection === ClosedLoopCorrectionNonFiniteFeedback
    @test correction == fill(71.0f0, 2)
    @test controller_state == fill(72.0f0, 2)
    @test closed_loop_correction_state(prepared.workspace) == committed_state
    @test closed_loop_correction_last_output(prepared.workspace) == committed_output
end

@testset "declared closed-loop processing allocates no Julia heap after warmup" begin
    prepared = fixture_prepared()
    correction = zeros(Float32, 2)
    controller_state = zeros(Float32, 2)
    residual_error = Float32[1, -1]
    feedback = Float32[0.25, -0.5]
    process!(
        correction,
        controller_state,
        prepared,
        residual_error,
        feedback,
    )
    process!(
        correction,
        controller_state,
        prepared,
        residual_error,
        feedback,
    )
    @test declared_process_allocation_bytes(
        correction,
        controller_state,
        prepared,
        residual_error,
        feedback,
    ) == 0

    nonfinite_feedback = Float32[NaN, 0]
    process!(
        correction,
        controller_state,
        prepared,
        residual_error,
        nonfinite_feedback,
    )
    @test declared_process_allocation_bytes(
        correction,
        controller_state,
        prepared,
        residual_error,
        nonfinite_feedback,
    ) == 0
end

@testset "closed-loop declaration rejects invalid graph dimensions" begin
    @test_throws ArgumentError prepare_algorithm(
        ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(
            state_length=0,
            hidden_mode_count=1,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(
            state_length=1,
            hidden_mode_count=0,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        ClosedLoopCorrectionF32,
        ClosedLoopCorrectionF32Configuration(
            state_length=typemax(Int),
            hidden_mode_count=2,
        ),
    )
end


@testset "closed-loop outputs reject overlap and borrowed owned storage" begin
    plan = ClosedLoopCorrectionPlan(2f0, 0f0, 0f0, 0f0, 2, zeros(Float32, 1, 2), zeros(Float32, 2, 1))
    workspace = ClosedLoopCorrectionWorkspace(plan)
    storage = fill(71f0, 3)
    @test_throws ArgumentError process!(view(storage, 1:2), view(storage, 2:3), workspace, plan, ones(Float32, 2), nothing)
    @test storage == fill(71f0, 3)
    for owned in (workspace.committed_state, workspace.candidate_state, workspace.committed_output,
        workspace.candidate_output, workspace.realized, workspace.hidden_in_state, vec(plan.shape_to_hidden))
        output = fill(71f0, 2)
        @test_throws ArgumentError process!(owned, output, workspace, plan, ones(Float32, 2), nothing)
        @test output == fill(71f0, 2)
        @test closed_loop_correction_state(workspace) == zeros(Float32, 2)
    end
end


# A Float64 product and addend have an exponent span below 4096 bits, so this
# evaluates finite Float32/Float64 product-plus-sum exactly before one T rounding.
function fma_oracle(a::T, b::T, c::T) where {T<:Union{Float32,Float64}}
    return setprecision(BigFloat, 4096) do
        T(BigFloat(a) * BigFloat(b) + BigFloat(c))
    end
end

@testset "closed-loop pointwise stages agree with the unfused scalar reference" begin
    for T in (Float32, Float64), count in (4, 253, 277, 512)
        plan = ClosedLoopCorrectionPlan(
            T(0.625), T(0.75), T(0.375), zero(T), count,
            zeros(T, 0, count), zeros(T, count, 0),
        )
        workspace = ClosedLoopCorrectionWorkspace(plan)
        residual = T[T((index % 19) - 9) / T(7) for index in 1:count]
        feedback = T[T((index % 11) - 5) / T(13) for index in 1:count]
        correction = zeros(T, count)
        state = zeros(T, count)
        expected_first = T[plan.pole * zero(T) + plan.gain * value for value in residual]
        @test process!(correction, state, workspace, plan, residual, nothing) === nothing
        # For these normal finite fixtures, 4eps = 8u conservatively covers the
        # product/add rounding differences; operand magnitudes handle cancellation.
        @test all(abs.(correction .- expected_first) .<=
                  T(4) * eps(T) .* abs.(plan.gain .* residual))
        commit!(workspace)
        expected_state = T[expected_first[index] - plan.anti_windup_gain * feedback[index]
                           for index in 1:count]
        expected_output = T[plan.pole * expected_state[index] + plan.gain * residual[index]
                            for index in 1:count]
        state_bound = T(4) * eps(T) .* (
            abs.(expected_first) .+ abs.(plan.anti_windup_gain .* feedback)
        )
        output_bound = abs(plan.pole) .* state_bound .+ T(4) * eps(T) .* (
            abs.(plan.pole .* expected_state) .+ abs.(plan.gain .* residual)
        )
        @test process!(correction, state, workspace, plan, residual, feedback) === nothing
        @test all(abs.(state .- expected_state) .<= state_bound)
        @test all(abs.(correction .- expected_output) .<= output_bound)
        # Ordinary input/output overlap must still publish only after staging.
        copyto!(correction, residual)
        @test process!(correction, state, workspace, plan, correction, feedback) === nothing
        @test all(abs.(correction .- expected_output) .<= output_bound)
    end
end

@testset "closed-loop FMA trajectory matches the high-precision rounding oracle" begin
    for T in (Float32, Float64)
        count = 9
        identity = T[row == column for row in 1:count, column in 1:count]
        plan = ClosedLoopCorrectionPlan(
            T(0.1), T(0.8), T(0.3), T(0.2), count, identity, identity,
        )
        workspace = ClosedLoopCorrectionWorkspace(plan)
        correction, state = zeros(T, count), zeros(T, count)
        previous_state, previous_output = zeros(T, count), zeros(T, count)
        U = T === Float32 ? UInt32 : UInt64
        for frame in 1:12
            residual = T[T((index % 7) - 3) / T(11) + T(frame) / T(13)
                         for index in 1:count]
            feedback = frame == 1 ? nothing :
                T[T((index % 5) - 2) / T(29) - T(frame) / T(31) for index in 1:count]
            expected_state = zeros(T, count)
            if frame > 1
                for index in 1:count
                    realized = previous_output[index] - feedback[index]
                    anti_windup = fma_oracle(-plan.anti_windup_gain,
                        feedback[index], previous_output[index])
                    # Identity projections introduce no additional rounding.
                    expected_state[index] = fma_oracle(-plan.hidden_mode_gain,
                        realized, anti_windup)
                end
            end
            expected_output = T[fma_oracle(plan.gain, residual[index],
                plan.pole * expected_state[index]) for index in 1:count]
            if iseven(frame)
                copyto!(correction, residual)
            end
            input = iseven(frame) ? correction : residual
            @test process!(correction, state, workspace, plan, input, feedback) === nothing
            @test reinterpret(U, state) == reinterpret(U, expected_state)
            @test reinterpret(U, correction) == reinterpret(U, expected_output)
            @test reinterpret(U, closed_loop_correction_state(workspace)) ==
                  reinterpret(U, previous_state)
            if frame == 1
                @test closed_loop_correction_last_output(workspace) === nothing
            else
                @test reinterpret(U, closed_loop_correction_last_output(workspace)) ==
                      reinterpret(U, previous_output)
            end
            if frame == 3
                discard!(workspace)
                @test process!(correction, state, workspace, plan, residual, feedback) === nothing
                @test reinterpret(U, correction) == reinterpret(U, expected_output)
                @test reinterpret(U, state) == reinterpret(U, expected_state)
            end
            commit!(workspace)
            @test reinterpret(U, closed_loop_correction_state(workspace)) ==
                  reinterpret(U, expected_state)
            @test reinterpret(U, closed_loop_correction_last_output(workspace)) ==
                  reinterpret(U, expected_output)
            previous_state, previous_output = expected_state, expected_output
        end
    end
end

@testset "closed-loop FMA retains cancellation lost by rounded products" begin
    for T in (Float32, Float64)
        empty_forward, empty_backward = zeros(T, 0, 1), zeros(T, 1, 0)
        initial_plan = ClosedLoopCorrectionPlan(
            one(T), zero(T), zero(T), zero(T), 1, empty_forward, empty_backward,
        )
        plan = ClosedLoopCorrectionPlan(
            one(T) + eps(T), one(T), zero(T), zero(T), 1, empty_forward, empty_backward,
        )
        workspace = ClosedLoopCorrectionWorkspace(plan)
        correction, state = zeros(T, 1), zeros(T, 1)
        @test process!(correction, state, workspace, initial_plan, T[-1], nothing) === nothing
        commit!(workspace)
        residual = one(T) - eps(T)
        rounded_product = plan.gain * residual
        @test rounded_product - one(T) == zero(T)
        expected = fma_oracle(plan.gain, residual, -one(T))
        @test expected == -eps(T)^2
        @test process!(correction, state, workspace, plan, T[residual], T[0]) === nothing
        @test only(correction) == expected
    end
end

@testset "closed-loop FMA cancellation accepts finite results and rejects true overflow" begin
    for T in (Float32, Float64), stage in (:correction, :anti_windup, :hidden)
        forward = stage === :hidden ? ones(T, 1, 1) : zeros(T, 0, 1)
        backward = stage === :hidden ? ones(T, 1, 1) : zeros(T, 1, 0)
        initial_plan = ClosedLoopCorrectionPlan(
            one(T), zero(T), zero(T), zero(T), 1, forward, backward,
        )
        plan = ClosedLoopCorrectionPlan(
            stage === :correction ? T(2) : one(T),
            stage === :correction ? one(T) : zero(T),
            stage === :anti_windup ? T(2) : zero(T),
            stage === :hidden ? T(2) : zero(T), 1, forward, backward,
        )
        workspace = ClosedLoopCorrectionWorkspace(plan)
        correction, state = zeros(T, 1), zeros(T, 1)
        initial = stage === :correction ? -floatmax(T) : floatmax(T)
        @test process!(correction, state, workspace, initial_plan, T[initial], nothing) === nothing
        commit!(workspace)
        residual = stage === :correction ? floatmax(T) : zero(T)
        feedback = stage === :anti_windup ? floatmax(T) : zero(T)
        # The old separately rounded multiplication overflows in every case.
        rounded_product = T(2) * floatmax(T)
        @test !isfinite(rounded_product)
        expected_state = -floatmax(T)
        expected_output = fma_oracle(plan.gain, residual, plan.pole * expected_state)
        @test isfinite(expected_output)
        @test process!(correction, state, workspace, plan, T[residual], T[feedback]) === nothing
        @test only(state) == expected_state
        @test only(correction) == expected_output
        @test closed_loop_correction_last_output(workspace) == T[initial]
        commit!(workspace)
        held_state = copy(closed_loop_correction_state(workspace))
        held_output = copy(closed_loop_correction_last_output(workspace))
        fill!(correction, T(71))
        fill!(state, T(72))
        overflow_residual = stage === :correction ? floatmax(T) : zero(T)
        overflow_feedback = stage === :correction ? zero(T) : -floatmax(T)
        @test process!(correction, state, workspace, plan,
            T[overflow_residual], T[overflow_feedback]) === ClosedLoopCorrectionNonFiniteOutput
        @test correction == T[71]
        @test state == T[72]
        @test closed_loop_correction_state(workspace) == held_state
        @test closed_loop_correction_last_output(workspace) == held_output
    end
end

@testset "closed-loop pointwise overflow rejects, discards, and recovers" begin
    count = 277
    for T in (Float32, Float64), recovery in (:discard, :retry)
        plan = ClosedLoopCorrectionPlan(
            T(2), one(T), T(2), zero(T), count,
            zeros(T, 0, count), zeros(T, count, 0),
        )
        workspace = ClosedLoopCorrectionWorkspace(plan)
        correction = zeros(T, count)
        state = zeros(T, count)
        residual = ones(T, count)
        feedback = zeros(T, count)
        @test process!(correction, state, workspace, plan, residual, nothing) === nothing
        commit!(workspace)
        committed_state = copy(closed_loop_correction_state(workspace))
        committed_output = copy(closed_loop_correction_last_output(workspace))
        for failure_stage in (:output, :state), index in (1, 4, 5, 8, 9, count)
            fill!(residual, one(T))
            fill!(feedback, zero(T))
            failure_stage === :output ? (residual[index] = floatmax(T)) :
                                      (feedback[index] = -floatmax(T))
            fill!(correction, T(71))
            fill!(state, T(72))
            fill!(workspace.candidate_output, T(NaN))
            @test process!(correction, state, workspace, plan, residual, feedback) ===
                  ClosedLoopCorrectionNonFiniteOutput
            @test process_allocation_bytes(correction, state, workspace, plan, residual, feedback) == 0
            @test correction == fill(T(71), count)
            @test state == fill(T(72), count)
            @test !workspace.candidate_has_output
            @test closed_loop_correction_state(workspace) == committed_state
            @test closed_loop_correction_last_output(workspace) == committed_output
            recovery === :discard && discard!(workspace)
            fill!(residual, one(T))
            fill!(feedback, zero(T))
            @test process!(correction, state, workspace, plan, residual, feedback) === nothing
            @test correction == fill(T(4), count)
            @test state == fill(T(2), count)
            @test process_allocation_bytes(correction, state, workspace, plan, residual, feedback) == 0
        end
        commit!(workspace)
        @test closed_loop_correction_last_output(workspace) == fill(T(4), count)
        @test closed_loop_correction_state(workspace) == fill(T(2), count)
    end
end


end
