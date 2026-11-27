function prepare_pyramid_pixel_plan(device::CUDA.CuDevice, plan::PyramidPixelPlan{T,Vector{Bool},Vector{Int},Nothing}) where {T<:Union{Float32,Float64}}
    return _prepare_pyramid_pixel_plan(device, plan)
end

function pyramid_normalization_gain(
    workspace::PyramidPixelWorkspace{
        T,
        <:CUDA.CuVector{T},
        <:CUDA.CuVector{Int32},
    },
) where {T<:Union{Float32,Float64}}
    return Array(workspace.reciprocal_previous_frame_mean)[1]
end

function process!(
    output::PyramidPixelOutput{
        T,
        <:CUDA.CuVector{T},
        <:CUDA.CuVector{T},
    },
    workspace::PyramidPixelWorkspace{
        T,
        <:CUDA.CuVector{T},
        <:CUDA.CuVector{Int32},
        Vector{Int32},
    },
    plan::PyramidPixelPlan{
        T,
        <:CUDA.CuVector{Bool},
        <:CUDA.CuVector{Int},
        <:PyramidPixelBackend,
    },
    input::CUDA.CuMatrix{T},
) where {T<:Union{Float32,Float64}}
    return _process_pyramid_pixel!(output, workspace, plan, input)
end
