function prepare_docrime_excitation_plan(
    device::AMDGPU.HIPDevice,
    plan::DocrimeExcitationPlan{T,Vector{T},Nothing},
) where {T<:Union{Float32,Float64}}
    return _prepare_docrime_excitation_plan(
        device,
        plan,
        DocrimeExcitationPlan,
        _docrime_continuous_kernel!,
    )
end

function prepare_docrime_excitation_plan(
    device::AMDGPU.HIPDevice,
    plan::DocrimeBinaryExcitationPlan{T,Vector{T},Nothing},
) where {T<:Union{Float32,Float64}}
    return _prepare_docrime_excitation_plan(
        device,
        plan,
        DocrimeBinaryExcitationPlan,
        _docrime_binary_kernel!,
    )
end

function process!(
    output::AMDGPU.ROCVector{T},
    workspace::DocrimeExcitationWorkspace{T,<:AMDGPU.ROCVector{T}},
    plan::DocrimeExcitationPlan{
        T,
        <:AMDGPU.ROCVector{T},
        <:DocrimeExcitationBackend,
    },
) where {T<:Union{Float32,Float64}}
    return _process_docrime_excitation!(output, workspace, plan)
end

function process!(
    output::AMDGPU.ROCVector{T},
    workspace::DocrimeExcitationWorkspace{T,<:AMDGPU.ROCVector{T}},
    plan::DocrimeBinaryExcitationPlan{
        T,
        <:AMDGPU.ROCVector{T},
        <:DocrimeExcitationBackend,
    },
) where {T<:Union{Float32,Float64}}
    return _process_docrime_excitation!(output, workspace, plan)
end
