import FilterGraphAlgorithms:
    commit!, discard!, process!, reset!, splitmix64_advance

const DOCRIME_TEST_SEED = UInt64(0x123456789abcdef0)

function run_docrime_excitation_tests(device, device_zeros, deviceof, used_memory)
    function _assert_docrime_cpu_gpu_sequence(Plan, ::Type{T}) where {T}
        cpu_plan = Plan(T[1, 2, 0.5], DOCRIME_TEST_SEED)
        gpu_plan = prepare_docrime_excitation_plan(device, cpu_plan)
        cpu_workspace = DocrimeExcitationWorkspace(cpu_plan)
        gpu_workspace = DocrimeExcitationWorkspace(gpu_plan)
        cpu_output = zeros(T, 3)
        gpu_output = device_zeros(T, 3)

        for _ in 1:4
            @test process!(cpu_output, cpu_workspace, cpu_plan) === nothing
            @test process!(gpu_output, gpu_workspace, gpu_plan) === nothing
            @test Array(gpu_output) == cpu_output
            @test docrime_excitation_state(gpu_workspace) ==
                  docrime_excitation_state(cpu_workspace)
            commit!(cpu_workspace)
            commit!(gpu_workspace)
        end

        prior_state = docrime_excitation_state(gpu_workspace)
        prior_output = Array(gpu_output)
        @test_throws DimensionMismatch process!(device_zeros(T, 2), gpu_workspace, gpu_plan)
        @test docrime_excitation_state(gpu_workspace) == prior_state
        @test Array(gpu_output) == prior_output
        reset!(cpu_workspace); reset!(gpu_workspace)
        process!(cpu_output, cpu_workspace, cpu_plan)
        process!(gpu_output, gpu_workspace, gpu_plan)
        @test Array(gpu_output) == cpu_output
        discard!(cpu_workspace); discard!(gpu_workspace)
        @test docrime_excitation_state(gpu_workspace) == DOCRIME_TEST_SEED
        return gpu_plan, gpu_workspace, gpu_output
    end

    @testset "GPU DO-CRIME excitation matches CPU bit for bit" begin
        for Plan in (DocrimeExcitationPlan, DocrimeBinaryExcitationPlan)
            for T in (Float32, Float64)
                gpu_plan, gpu_workspace, gpu_output =
                    _assert_docrime_cpu_gpu_sequence(Plan, T)
                @test deviceof(gpu_plan.amplitudes) == device
                @test deviceof(gpu_workspace.staged_excitation) ==
                      device
                @test deviceof(gpu_output) == device
            end
        end
    end

    @testset "GPU DO-CRIME state remains transactional" begin
        cpu_plan = DocrimeExcitationPlan(Float32[1, 2], DOCRIME_TEST_SEED)
        plan = prepare_docrime_excitation_plan(device, cpu_plan)
        workspace = DocrimeExcitationWorkspace(plan)
        first = device_zeros(Float32, 2)
        repeated = similar(first)

        process!(first, workspace, plan)
        process!(repeated, workspace, plan)
        @test Array(repeated) == Array(first)
        @test docrime_excitation_state(workspace) == DOCRIME_TEST_SEED

        discard!(workspace)
        process!(repeated, workspace, plan)
        @test Array(repeated) == Array(first)
        commit!(workspace)
        @test docrime_excitation_state(workspace) ==
              splitmix64_advance(DOCRIME_TEST_SEED, UInt64(2))

        reset!(workspace)
        process!(repeated, workspace, plan)
        @test Array(repeated) == Array(first)
        @test_throws MethodError process!(zeros(Float32, 2), workspace, plan)
    end

    @testset "GPU DO-CRIME parameter adoption preserves sequence" begin
        host_prepared = prepare_algorithm(
            DocrimeExcitationF32,
            DocrimeExcitationF32Configuration(
                amplitudes=Float32[1, 2],
                seed=DOCRIME_TEST_SEED,
            ),
        )
        plan = prepare_docrime_excitation_plan(device, host_prepared.plan)
        prepared = FilterGraphAlgorithms.PreparedAlgorithm(
            DocrimeExcitationF32,
            plan,
            DocrimeExcitationWorkspace(plan),
            host_prepared.formats,
        )
        output = device_zeros(Float32, 2)

        process!(output, prepared)
        prior_state = docrime_excitation_state(prepared.workspace)
        replace_parameter!(prepared, :amplitudes, Float32[0.5, 1.0])
        @test deviceof(prepared.plan.amplitudes) == device
        process!(output, prepared)
        @test docrime_excitation_state(prepared.workspace) ==
              splitmix64_advance(prior_state, UInt64(2))
        @test Array(prepared.plan.amplitudes) == Float32[0.5, 1.0]
    end

    @testset "GPU DO-CRIME repeated storage is bounded" begin
        cpu_plan = DocrimeBinaryExcitationPlan(
            fill(0.25f0, 1024),
            DOCRIME_TEST_SEED,
        )
        plan = prepare_docrime_excitation_plan(device, cpu_plan)
        workspace = DocrimeExcitationWorkspace(plan)
        output = device_zeros(Float32, 1024)

        process!(output, workspace, plan)
        commit!(workspace)
        device_bytes = used_memory()
        for _ in 1:16
            process!(output, workspace, plan)
            commit!(workspace)
        end
        @test used_memory() == device_bytes
    end

end
