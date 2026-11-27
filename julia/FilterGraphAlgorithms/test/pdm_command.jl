module PdmCommandTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FGA = FilterGraphAlgorithms

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pdm_command.toml"),
)

function fixture_configuration()
    configuration = FIXTURE["configuration"]
    return PdmCommandF32Configuration(
        actuator_count=configuration["actuator_count"],
        command_limits=(Float32(configuration["lower_limit"]), Float32(configuration["upper_limit"])),
        slew_limit=Float32.(configuration["slew_limit"]),
        initial_current=Float32.(configuration["initial_current"]),
        quantization_origin=nothing,
        quantization_step=nothing,
        dither_seed=nothing,
    )
end

function pdm_command_allocation_bytes(
    demanded,
    constraint_feedback,
    prepared,
    requested,
)
    return @allocated process!(
        demanded,
        constraint_feedback,
        prepared,
        requested,
    )
end

function pdm_command_admitted_allocation_bytes(
    demanded,
    constraint_feedback,
    prepared,
    requested,
)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        demanded,
        constraint_feedback,
        prepared,
        requested,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

pdm_command_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
pdm_command_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
pdm_command_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "Command limits preserve calibrated units and coordinates" begin
    bounds = (-1f-6, 1f-6)
    requested = Float32[2f-6, -0.75f-6]
    for schema in (DEMANDED_PDM_COMMAND_V1, ACTUATOR_COMMAND_OPD_M_V1, NATIVE_PDM_COMMAND_V1)
        config = PdmCommandF32Configuration(; actuator_count=2, demanded_schema=schema,
            command_limits=bounds, slew_limit=nothing, initial_current=nothing)
        prepared = prepare_algorithm(PdmCommandF32, config)
        @test getproperty.(algorithm_formats(prepared), :schema) ==
            (REQUESTED_PDM_COMMAND_V1, schema, PDM_CONSTRAINT_FEEDBACK_V1)
        demanded, feedback = similar(requested), similar(requested)
        @test isnothing(process!(demanded, feedback, prepared, requested))
        @test demanded == [bounds[2], requested[2]]
        @test feedback == requested .- demanded
    end
    @test_throws ArgumentError prepare_algorithm(PdmCommandF32, PdmCommandF32Configuration(;
        actuator_count=2, demanded_schema=TIP_TILT_OPD_COEFFICIENTS_M_V1,
        command_limits=bounds, slew_limit=nothing, initial_current=nothing))
    legacy = PdmCommandF32Configuration(2, bounds, nothing, nothing, nothing, nothing, nothing)
    @test legacy.demanded_schema == DEMANDED_PDM_COMMAND_V1

    config = TipTiltCommandF32Configuration(; command_limits=bounds, slew_limit=0.5f-6,
        initial_current=zeros(Float32, 2))
    prepared = prepare_algorithm(TipTiltCommandF32, config)
    @test all(f -> f.schema == TIP_TILT_OPD_COEFFICIENTS_M_V1 && f.shape == (2,),
        algorithm_formats(prepared))
    demanded, feedback = similar(requested), similar(requested)
    for expected in (Float32[0.5f-6, -0.5f-6], Float32[bounds[2], requested[2]])
        @test isnothing(process!(demanded, feedback, prepared, requested))
        @test demanded == expected == pdm_command_state(prepared.workspace)
        @test feedback == requested .- expected
    end
end

@testset "PDM command declaration preserves the FGN contract" begin
    @test algorithm_label(PdmCommandF32) == "pdm-command-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PdmCommandF32) ===
          PdmCommandFailure
    @test algorithm_property_policy(PdmCommandF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(PdmCommandF32)
    @test map(field -> field.name, configuration) == (
        :actuator_count,
        :command_limits,
        :slew_limit,
        :initial_current,
        :quantization_origin,
        :quantization_step,
        :dither_seed,
        :demanded_schema,
    )
    @test configuration[end].update isa GraphRebuildConfiguration
    @test configuration[1].update isa GraphRebuildConfiguration
    @test all(
        field -> field.update isa ConstructionConfiguration,
        configuration[2:end-1],
    )

    ports = algorithm_ports(PdmCommandF32)
    @test map(port -> port.name, ports) == (
        :requested,
        :demanded,
        Symbol("constraint-feedback"),
    )
    @test ports[2].metadata_source == :requested
    @test ports[3].metadata_source == :requested

    prepared = prepare_algorithm(PdmCommandF32, fixture_configuration())
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2,), (2,), (2,))
    @test getproperty.(formats, :schema) == (
        REQUESTED_PDM_COMMAND_V1,
        DEMANDED_PDM_COMMAND_V1,
        PDM_CONSTRAINT_FEEDBACK_V1,
    )

    demanded = zeros(Float32, 2)
    constraint_feedback = zeros(Float32, 2)
    for sample in FIXTURE["samples"]
        requested = Float32.(sample["requested"])
        @test process!(
            demanded,
            constraint_feedback,
            prepared,
            requested,
        ) === nothing
        @test demanded ≈ Float32.(sample["demanded"])
        @test constraint_feedback ≈ Float32.(sample["constraint_feedback"])
    end
    @test pdm_command_state(prepared.workspace) ≈ Float32[0.4, -0.2]

    @test reset!(prepared) === prepared
    @test iszero(pdm_command_state(prepared.workspace))
end

@testset "PDM command limits each actuator independently" begin
    lower = fill(-1.0f0, 3)
    upper = fill(1.0f0, 3)
    limits = Float32[0.2, 0.1, 0.4]
    requested = Float32[0.5, -0.5, 0.5]

    plan = PdmCommandPlan(lower, upper, limits)
    workspace = PdmCommandWorkspace(plan, zeros(Float32, 3))
    demanded = zeros(Float32, 3)
    feedback = zeros(Float32, 3)
    @test process!(demanded, feedback, workspace, plan, requested) === nothing
    @test demanded == Float32[0.2, -0.1, 0.4]
    @test feedback == requested - demanded
    @test commit!(workspace) === workspace
    @test process!(demanded, feedback, workspace, plan, requested) === nothing
    @test demanded == Float32[0.4, -0.2, 0.5]

    no_slew_plan = PdmCommandPlan(lower, upper)
    no_slew_workspace = PdmCommandWorkspace(no_slew_plan, zeros(Float32, 3))
    @test process!(
        demanded,
        feedback,
        no_slew_workspace,
        no_slew_plan,
        Float32[2, -2, 0.5],
    ) === nothing
    @test demanded == Float32[1, -1, 0.5]

    @test_throws MethodError PdmCommandPlan(
        lower,
        upper,
        limits;
        slew_mode="vector-preserving",
    )
end

@testset "PDM slew handles full finite actuator ranges" begin
    for T in (Float32, Float64)
        bound = floatmax(T)
        plan = PdmCommandPlan(T[-bound], T[bound], T[bound])
        workspace = PdmCommandWorkspace(plan, T[-bound])
        demanded, feedback = zeros(T, 1), zeros(T, 1)
        @test process!(demanded, feedback, workspace, plan, T[bound]) === nothing
        @test iszero(only(demanded))
        @test only(feedback) == bound
    end
end

@testset "PDM plan accepts legacy disabled DAC settings and rejects grids" begin
    lower, upper, slew = [-0.5], [0.5], [0.02]
    for keywords in (NamedTuple(),
            (; quantization_origin=nothing, quantization_step=nothing, dither_seed=nothing),
            (; dither_seed=0x1234))
        plan = PdmCommandPlan(lower, upper, slew; keywords...)
        workspace = PdmCommandWorkspace(plan, [0.0])
        @test !hasfield(typeof(plan), :quantization)
        @test !hasfield(typeof(plan), :dither_seed)
        @test !hasfield(typeof(workspace), :committed_quantization_error)
        @test !hasfield(typeof(workspace), :committed_random_state)
        demanded, feedback, requested = zeros(1), zeros(1), [0.3]
        @test process!(demanded, feedback, workspace, plan, requested) === nothing
        @test demanded == [0.02]
        @test feedback == requested - demanded
    end
    for keywords in ((; quantization_origin=0.0), (; quantization_step=0.125),
            (; quantization_origin=0.0, quantization_step=0.125, dither_seed=0x1234))
        failure = try
            PdmCommandPlan(lower, upper, slew; keywords...)
            nothing
        catch exception
            exception
        end
        @test failure isa ArgumentError
        @test occursin("driver", lowercase(sprint(showerror, failure)))
    end
end

@testset "PDM command accepts logical CPU views and safe input aliasing" begin
    plan = PdmCommandPlan(Float32[-0.25, -0.25], Float32[0.25, 0.25])
    workspace = PdmCommandWorkspace(plan, zeros(Float32, 2))
    requested_storage = Float32[0.45, 99, -0.3, 99]
    demanded_storage = fill(71.0f0, 4)
    feedback_storage = fill(72.0f0, 4)
    requested = @view requested_storage[1:2:3]
    demanded = @view demanded_storage[2:2:4]
    constraint_feedback = @view feedback_storage[1:2:3]

    @test process!(
        demanded,
        constraint_feedback,
        workspace,
        plan,
        requested,
    ) === nothing
    @test demanded == Float32[0.25, -0.25]
    @test constraint_feedback ≈ Float32[0.2, -0.05]
    @test demanded_storage[1:2:3] == fill(71.0f0, 2)
    @test feedback_storage[2:2:4] == fill(72.0f0, 2)

    aliased_requested = Float32[0.45, -0.3]
    @test process!(
        aliased_requested,
        constraint_feedback,
        workspace,
        plan,
        aliased_requested,
    ) === nothing
    @test aliased_requested == Float32[0.25, -0.25]
    @test_throws ArgumentError process!(
        demanded,
        demanded,
        workspace,
        plan,
        requested,
    )
end

@testset "PDM command preparation and rejection are failure-atomic" begin
    @test_throws ArgumentError PdmCommandPlan(Float32[], Float32[])
    @test_throws DimensionMismatch PdmCommandPlan(Float32[-1], Float32[1, 1])
    @test_throws DimensionMismatch PdmCommandPlan(
        Float32[-1],
        Float32[1],
        Float32[1, 1],
    )
    @test_throws ArgumentError PdmCommandPlan(Float32[NaN], Float32[1])
    @test_throws ArgumentError PdmCommandPlan(Float32[1], Float32[-1])
    @test_throws ArgumentError PdmCommandPlan(
        Float32[-1],
        Float32[1],
        Float32[-0.1],
    )
    @test_throws ArgumentError PdmCommandPlan(
        Float32[-1],
        Float32[1];
        dither_seed=-1,
    )

    plan = PdmCommandPlan(Float32[-1], Float32[1])
    @test_throws DimensionMismatch PdmCommandWorkspace(plan, Float32[])
    @test_throws ArgumentError PdmCommandWorkspace(plan, Float32[NaN])
    @test_throws ArgumentError PdmCommandWorkspace(plan, Float32[2])

    @test_throws ArgumentError prepare_algorithm(
        PdmCommandF32,
        PdmCommandF32Configuration(
            actuator_count=0,
            command_limits=(-1.0f0, 1.0f0),
            slew_limit=nothing,
            initial_current=nothing,
            quantization_origin=nothing,
            quantization_step=nothing,
            dither_seed=nothing,
        ),
    )

    prepared = prepare_algorithm(PdmCommandF32, fixture_configuration())
    demanded = fill(71.0f0, 2)
    constraint_feedback = fill(72.0f0, 2)
    committed = copy(pdm_command_state(prepared.workspace))
    @test process!(
        demanded,
        constraint_feedback,
        prepared,
        Float32[NaN, 0],
    ) === PdmCommandNonFiniteInput
    @test demanded == fill(71.0f0, 2)
    @test constraint_feedback == fill(72.0f0, 2)
    @test pdm_command_state(prepared.workspace) == committed

    overflow_plan = PdmCommandPlan(
        Float32[-floatmax(Float32)],
        Float32[floatmax(Float32)],
        Float32[0],
    )
    overflow_workspace = PdmCommandWorkspace(
        overflow_plan,
        Float32[-floatmax(Float32)],
    )
    overflow_demanded = Float32[71]
    overflow_feedback = Float32[72]
    @test process!(
        overflow_demanded,
        overflow_feedback,
        overflow_workspace,
        overflow_plan,
        Float32[floatmax(Float32)],
    ) === PdmCommandNonFiniteOutput
    @test overflow_demanded == Float32[71]
    @test overflow_feedback == Float32[72]
    @test pdm_command_state(overflow_workspace) ==
          Float32[-floatmax(Float32)]
end

@testset "PDM command warmed paths allocate no Julia heap memory" begin
    prepared = prepare_algorithm(PdmCommandF32, fixture_configuration())
    workspace = prepared.workspace
    demanded = zeros(Float32, 2)
    constraint_feedback = zeros(Float32, 2)
    requested = Float32[0.5, -0.5]
    nonfinite = Float32[NaN, 0]

    process!(demanded, constraint_feedback, prepared, requested)
    @test pdm_command_allocation_bytes(
        demanded,
        constraint_feedback,
        prepared,
        requested,
    ) == 0
    process!(demanded, constraint_feedback, prepared, nonfinite)
    @test pdm_command_allocation_bytes(
        demanded,
        constraint_feedback,
        prepared,
        nonfinite,
    ) == 0

    FilterGraphAlgorithms._process_admitted!(
        demanded,
        constraint_feedback,
        prepared,
        requested,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    discard!(workspace)
    @test pdm_command_admitted_allocation_bytes(
        demanded,
        constraint_feedback,
        prepared,
        requested,
    ) == 0
    @test pdm_command_commit_allocation_bytes(workspace) == 0
    @test pdm_command_discard_allocation_bytes(workspace) == 0
    @test pdm_command_reset_allocation_bytes(workspace) == 0
end

pdm_command_core_bytes(demanded, feedback, workspace, plan, requested) =
    @allocated process!(demanded, feedback, workspace, plan, requested)

@testset "PDM concrete policies preserve CPU conditioning and transactions" begin
    for T in (Float32, Float64), mode in (:none, :independent)
        @testset "$T $mode" begin
            limits = mode === :none ? nothing : T[0.2, 0.1, 0.4]
            plan = PdmCommandPlan(fill(T(-1), 3), fill(T(1), 3), limits)
            policy = mode === :none ? FGA.NoSlew : FGA.IndependentSlew
            @test plan.slew isa policy
            @test isconcretetype(typeof(plan))
            @test typeof(plan) === PdmCommandPlan{T}
            workspace = PdmCommandWorkspace(plan, zeros(T, 3))
            demanded, feedback = zeros(T, 3), zeros(T, 3)
            requested = T[1.125, -0.5, 0.3]
            @test @inferred(Union{Nothing,PdmCommandFailure},
                process!(demanded, feedback, workspace, plan, requested)) === nothing
            motion = mode === :none ? T[1, -0.5, 0.3] : T[0.2, -0.1, 0.3]
            @test demanded == workspace.candidate_current == motion
            @test iszero(workspace.committed_current)
            @test feedback == requested - demanded
            discard!(workspace)
            @test workspace.candidate_current == workspace.committed_current
            @test process!(demanded, feedback, workspace, plan, requested) === nothing
            @test demanded == motion
            commit!(workspace)
            @test workspace.committed_current == motion
            @test process!(demanded, feedback, workspace, plan, requested) === nothing
            next_motion = mode === :none ? motion : T[0.4, -0.2, 0.3]
            @test demanded == next_motion
            @test feedback == requested - demanded
            @test pdm_command_core_bytes(demanded, feedback, workspace, plan, requested) == 0
            for nonfinite in (T(NaN), T(Inf), T(-Inf))
                bad = T[0, 0, nonfinite]
                fill!(demanded, 71)
                fill!(feedback, 72)
                @test process!(demanded, feedback, workspace, plan, bad) === PdmCommandNonFiniteInput
                @test demanded == fill(T(71), 3)
                @test feedback == fill(T(72), 3)
                @test workspace.committed_current == motion
                pdm_command_core_bytes(demanded, feedback, workspace, plan, bad)
                @test pdm_command_core_bytes(demanded, feedback, workspace, plan, bad) == 0
            end
            reset!(workspace)
            @test iszero(workspace.committed_current)
            @test process!(demanded, feedback, workspace, plan, requested) === nothing
            @test demanded == motion

            bound = floatmax(T)
            overflow = PdmCommandPlan(T[-bound], T[-bound],
                mode === :none ? nothing : T[0])
            overflow_workspace = PdmCommandWorkspace(overflow, T[-bound])
            output, correction = T[71], T[72]
            @test process!(output, correction, overflow_workspace, overflow,
                T[bound]) === PdmCommandNonFiniteOutput
            @test output == T[71]
            @test correction == T[72]
            @test overflow_workspace.committed_current == T[-bound]
        end
    end
end

@testset "PDM zero slew preserves Float32 and Float64 motion" begin
    for T in (Float32, Float64)
        plan = PdmCommandPlan(fill(-one(T), 2), fill(one(T), 2), T[0, 0.1])
        workspace = PdmCommandWorkspace(plan, zeros(T, 2))
        demanded, feedback = zeros(T, 2), zeros(T, 2)
        @test process!(demanded, feedback, workspace, plan, T[0.5, 0.5]) === nothing
        @test demanded == T[0, 0.1]
        @test process!(demanded, feedback, workspace, plan, T[0, 0.5]) === nothing
        @test demanded == T[0, 0.1]
    end
end

@testset "Finite PDM input preserves command bits and transactional feedback" begin
    # Repeated values exercise packed lanes and a scalar tail.
    count = 69
    tile(values) = [values[mod1(i, length(values))] for i in 1:count]
    config = PdmCommandF32Configuration(; actuator_count=count, slew_limit=nothing,
        command_limits=(tile(Float32[-0.5, -2, -floatmax(Float32), -2]),
                        tile(Float32[0.5, 2, -floatmax(Float32), 2])),
        initial_current=tile(Float32[0, 0, -floatmax(Float32), 0]))
    admitted = prepare_algorithm(PdmCommandF32, config)
    checked = prepare_algorithm(PdmCommandF32, config)
    demanded, feedback = zeros(Float32, count), zeros(Float32, count)
    expected, expected_feedback = similar(demanded), similar(feedback)
    run_admitted!(requested) = FGA._process_finite_input!(
        demanded, feedback, admitted, requested, nothing)
    for requested in (tile(Float32[1, -0.0, 0, nextfloat(0f0)]),
                      tile(Float32[-1, 0.0, -1, -nextfloat(0f0)]))
        @test process!(expected, expected_feedback, checked, requested) === nothing
        @test run_admitted!(requested) === nothing
        commit!(admitted.workspace)
        @test isequal(demanded, expected)
        @test isequal(feedback, expected_feedback)
        @test isequal(pdm_command_state(admitted.workspace), pdm_command_state(checked.workspace))
    end

    @test !FGA._finite_conditioning_result(admitted.plan)
    requested = ones(Float32, count)
    requested[end - 2] = floatmax(Float32)
    @test run_admitted!(requested) === PdmCommandNonFiniteOutput
    discard!(admitted.workspace)
    @test isequal(demanded, expected)
    @test isequal(feedback, expected_feedback)
    @test isequal(pdm_command_state(admitted.workspace), pdm_command_state(checked.workspace))

    requested[end] = NaN32
    @test process!(expected, expected_feedback, checked, requested) === PdmCommandNonFiniteInput
    @test isequal(demanded, expected)
    @test isequal(feedback, expected_feedback)
end

@testset "Proven finite PDM conditioning preserves results and buffer checks" begin
    count = 69
    tile(values) = [values[mod1(i, length(values))] for i in 1:count]
    largest = floatmax(Float32)
    subnormal = nextfloat(0f0)
    lower = tile(Float32[-0.5, -largest, -0.0, -2, -subnormal])
    upper = tile(Float32[0.5, largest, 1, 0.0, subnormal])
    config = PdmCommandF32Configuration(; actuator_count=count, slew_limit=nothing,
        command_limits=(lower, upper), initial_current=zeros(Float32, count))
    admitted = prepare_algorithm(PdmCommandF32, config)
    proven = prepare_algorithm(PdmCommandF32, config)
    workspace = proven.workspace
    @test FGA._finite_conditioning_result(proven.plan)
    fill!(lower, 0.25f0)
    fill!(upper, Inf32)
    @test FGA._finite_conditioning_result(proven.plan)

    demanded, feedback = zeros(Float32, count), zeros(Float32, count)
    expected, expected_feedback = similar(demanded), similar(feedback)
    requested = tile(Float32[largest, -largest, -0.0, subnormal, -1])
    for request in (requested, -requested)
        @test FGA._process_finite_input!(expected, expected_feedback, admitted, request, nothing) === nothing
        @test FGA._process_finite_conditioning!(demanded, feedback, proven, request, nothing) === nothing
        @test isequal(demanded, expected)
        @test isequal(feedback, expected_feedback)
        commit!(admitted.workspace)
        commit!(workspace)
        @test isequal(pdm_command_state(workspace), pdm_command_state(admitted.workspace))
    end

    @test_throws DimensionMismatch FGA._process_finite_conditioning!(
        demanded, feedback, proven, requested[1:end-1], nothing)
    @test_throws ArgumentError FGA._process_finite_conditioning!(
        demanded, demanded, proven, requested, nothing)
    requested[end] = Inf32
    @test process!(demanded, feedback, proven, requested) === PdmCommandNonFiniteInput
    @test isequal(demanded, expected)
    @test isequal(feedback, expected_feedback)

    requested[end] = 0f0
    @test FGA._process_finite_conditioning!(demanded, feedback, proven, requested, nothing) === nothing
    discard!(workspace)
    @test isequal(workspace.candidate_current, workspace.committed_current)
    reset!(proven)
    @test isequal(workspace.committed_current, workspace.initial_current)
    @test FGA._process_finite_conditioning!(demanded, feedback, proven, requested, nothing) === nothing
    commit!(workspace)
    @test isequal(workspace.committed_current, demanded)
end

end
