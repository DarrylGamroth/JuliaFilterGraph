import FilterGraphAlgorithms:
    DocrimeExcitationWorkspace,
    _replace_docrime_amplitudes,
    prepare_docrime_excitation_plan,
    process!,
    splitmix64_advance,
    splitmix64_at,
    splitmix64_unit_at

const DOCRIME_EXCITATION_WORKGROUP_SIZE = 256

struct DocrimeExcitationBackend{Device,ComputeKernel,PublishKernel}
    device::Device
    compute_kernel::ComputeKernel
    publish_kernel::PublishKernel
end

@inline function _docrime_continuous_value(::Type{T}, amplitude, state, draw_index) where {T}
    unit = splitmix64_unit_at(T, state, draw_index)
    return amplitude * (unit + unit - one(T))
end

@inline _docrime_binary_value(::Type, amplitude, state, draw_index) =
    iszero(splitmix64_at(state, draw_index) >> 63) ? -amplitude : amplitude

@kernel function _docrime_continuous_kernel!(staged, amplitudes, state)
    index = @index(Global, Linear)
    if index <= length(staged)
        draw_index = UInt64(index - 1)
        @inbounds staged[index] = _docrime_continuous_value(eltype(staged), amplitudes[index], state, draw_index)
    end
end

@kernel function _docrime_binary_kernel!(staged, amplitudes, state)
    index = @index(Global, Linear)
    if index <= length(staged)
        draw_index = UInt64(index - 1)
        @inbounds staged[index] = _docrime_binary_value(eltype(staged), amplitudes[index], state, draw_index)
    end
end

@kernel function _docrime_publish_kernel!(output, staged)
    index = @index(Global, Linear)
    if index <= length(output)
        @inbounds output[index] = staged[index]
    end
end

function _docrime_backend(device, amplitudes, compute_kernel)
    backend = KernelAbstractions.get_backend(amplitudes)
    return DocrimeExcitationBackend(
        device,
        compute_kernel(backend, DOCRIME_EXCITATION_WORKGROUP_SIZE),
        _docrime_publish_kernel!(
            backend,
            DOCRIME_EXCITATION_WORKGROUP_SIZE,
        ),
    )
end

function _warm_docrime_excitation_kernels!(plan)
    workspace = DocrimeExcitationWorkspace(plan)
    output = similar(plan.amplitudes)
    process!(output, workspace, plan)
    FilterGraphAlgorithms.discard!(workspace)
    return plan
end

function _prepare_docrime_excitation_plan(
    device,
    plan,
    plan_type,
    compute_kernel,
)
    return _with_eager_device(device) do
        amplitudes = _capture_array(device, plan.amplitudes)
        backend = _docrime_backend(device, amplitudes, compute_kernel)
        prepared =
            FilterGraphAlgorithms._prepared_docrime_excitation_plan(
                plan_type,
                amplitudes,
                plan.seed,
                backend,
            )
        return _warm_docrime_excitation_kernels!(prepared)
    end
end

function DocrimeExcitationWorkspace(
    plan::Union{
        DocrimeExcitationPlan{
            T,
            <:AbstractVector{T},
            <:DocrimeExcitationBackend,
        },
        DocrimeBinaryExcitationPlan{
            T,
            <:AbstractVector{T},
            <:DocrimeExcitationBackend,
        },
    },
) where {T<:Union{Float32,Float64}}
    device = plan.backend.device
    return _with_eager_device(device) do
        staged = similar(plan.amplitudes)
        fill!(staged, zero(T))
        FilterGraphAlgorithms._prepared_docrime_excitation_workspace(
            T,
            plan.seed,
            plan.seed,
            plan.seed,
            staged,
        )
    end
end

function _check_docrime_devices(output, workspace, plan)
    device = plan.backend.device
    _require_current_device(device)
    _require_array_device("plan amplitudes", plan.amplitudes, device)
    _require_array_device("output", output, device)
    _require_array_device(
        "workspace staged excitation",
        workspace.staged_excitation,
        device,
    )
    return nothing
end

function _process_docrime_excitation!(output, workspace, plan)
    FilterGraphAlgorithms._validate_docrime_execution(
        output,
        workspace,
        plan,
    )
    _check_docrime_devices(output, workspace, plan)

    random_state = workspace.committed_random_state
    plan.backend.compute_kernel(
        workspace.staged_excitation,
        plan.amplitudes,
        random_state;
        ndrange=length(plan.amplitudes),
    )
    plan.backend.publish_kernel(
        output,
        workspace.staged_excitation;
        ndrange=length(plan.amplitudes),
    )
    KernelAbstractions.synchronize(
        KernelAbstractions.get_backend(plan.amplitudes),
    )
    workspace.candidate_random_state = splitmix64_advance(
        random_state,
        UInt64(length(plan.amplitudes)),
    )
    return nothing
end

function _replace_docrime_amplitudes(
    plan::DocrimeExcitationPlan{
        T,
        <:AbstractVector{T},
        <:DocrimeExcitationBackend,
    },
    values,
) where {T<:Union{Float32,Float64}}
    cpu_plan = DocrimeExcitationPlan(values, plan.seed)
    return _with_eager_device(plan.backend.device) do
        amplitudes = _capture_array(plan.backend.device, cpu_plan.amplitudes)
        FilterGraphAlgorithms._prepared_docrime_excitation_plan(
            DocrimeExcitationPlan,
            amplitudes,
            plan.seed,
            plan.backend,
        )
    end
end

function _replace_docrime_amplitudes(
    plan::DocrimeBinaryExcitationPlan{
        T,
        <:AbstractVector{T},
        <:DocrimeExcitationBackend,
    },
    values,
) where {T<:Union{Float32,Float64}}
    cpu_plan = DocrimeBinaryExcitationPlan(values, plan.seed)
    return _with_eager_device(plan.backend.device) do
        amplitudes = _capture_array(plan.backend.device, cpu_plan.amplitudes)
        FilterGraphAlgorithms._prepared_docrime_excitation_plan(
            DocrimeBinaryExcitationPlan,
            amplitudes,
            plan.seed,
            plan.backend,
        )
    end
end
