function prepare_pyramid_pixel_plan(device::AMDGPU.HIPDevice, plan::PyramidPixelPlan{T,Vector{Bool},Vector{Int},Nothing}) where {T<:Union{Float32,Float64}}
    return _prepare_pyramid_pixel_plan(device, plan)
end

function pyramid_normalization_gain(
    workspace::PyramidPixelWorkspace{
        T,
        <:AMDGPU.ROCVector{T},
        <:AMDGPU.ROCVector{Int32},
    },
) where {T<:Union{Float32,Float64}}
    return Array(workspace.reciprocal_previous_frame_mean)[1]
end

function process!(
    output::PyramidPixelOutput{
        T,
        <:AMDGPU.ROCVector{T},
        <:AMDGPU.ROCVector{T},
    },
    workspace::PyramidPixelWorkspace{
        T,
        <:AMDGPU.ROCVector{T},
        <:AMDGPU.ROCVector{Int32},
        Vector{Int32},
    },
    plan::PyramidPixelPlan{
        T,
        <:AMDGPU.ROCVector{Bool},
        <:AMDGPU.ROCVector{Int},
        <:PyramidPixelBackend,
    },
    input::AMDGPU.ROCMatrix{T},
) where {T<:Union{Float32,Float64}}
    return _process_pyramid_pixel!(output, workspace, plan, input)
end
