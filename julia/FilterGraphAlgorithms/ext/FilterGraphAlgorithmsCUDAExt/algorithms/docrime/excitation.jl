function prepare_docrime_excitation_plan(
    device::CUDA.CuDevice,
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
    device::CUDA.CuDevice,
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
    output::CUDA.CuVector{T},
    workspace::DocrimeExcitationWorkspace{T,<:CUDA.CuVector{T}},
    plan::DocrimeExcitationPlan{
        T,
        <:CUDA.CuVector{T},
        <:DocrimeExcitationBackend,
    },
) where {T<:Union{Float32,Float64}}
    return _process_docrime_excitation!(output, workspace, plan)
end

function process!(
    output::CUDA.CuVector{T},
    workspace::DocrimeExcitationWorkspace{T,<:CUDA.CuVector{T}},
    plan::DocrimeBinaryExcitationPlan{
        T,
        <:CUDA.CuVector{T},
        <:DocrimeExcitationBackend,
    },
) where {T<:Union{Float32,Float64}}
    return _process_docrime_excitation!(output, workspace, plan)
end
