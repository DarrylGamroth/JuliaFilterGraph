import FilterGraphAlgorithms:
    PyramidPixelWorkspace,
    prepare_pyramid_pixel_plan,
    process!,
    pyramid_normalization_gain

const PYRAMID_WORKGROUP_SIZE = 256

struct PyramidPixelBackend{Device,ComputeKernel,PublishKernel}
    device::Device
    compute_kernel::ComputeKernel
    publish_kernel::PublishKernel
end

@inline function _merge_pyramid_status(left::Int32, right::Int32)
    iszero(left) && return right
    iszero(right) && return left
    return min(left, right)
end

@inline function _reduce_pyramid_sum_status!(
    local_sums,
    local_statuses,
    local_index::Int,
    reduction_stride::Int,
)
    if local_index <= reduction_stride
        @inbounds combined_sum =
            local_sums[local_index] +
            local_sums[local_index + reduction_stride]
        @inbounds combined_status = _merge_pyramid_status(
            local_statuses[local_index],
            local_statuses[local_index + reduction_stride],
        )
        if !isfinite(combined_sum)
            combined_status = _merge_pyramid_status(
                combined_status,
                Int32(PyramidPixelNonFiniteMean),
            )
            combined_sum = zero(eltype(local_sums))
        end
        @inbounds local_sums[local_index] = combined_sum
        @inbounds local_statuses[local_index] = combined_status
    end
    return nothing
end

@inline function _reduce_pyramid_status!(
    local_statuses,
    local_index::Int,
    reduction_stride::Int,
)
    if local_index <= reduction_stride
        @inbounds local_statuses[local_index] = max(
            local_statuses[local_index],
            local_statuses[local_index + reduction_stride],
        )
    end
    return nothing
end

@kernel function _pyramid_compute_kernel!(
    staged_pixels,
    staged_mean_pupil_intensity,
    staged_reciprocal_current_frame_mean,
    status,
    reciprocal_previous_frame_mean,
    selected_pixel_offsets,
    input,
    mean_scale,
    selected_pixel_count,
)
    local_index = @index(Local, Linear)
    local_sums = @localmem eltype(input) (PYRAMID_WORKGROUP_SIZE,)
    local_statuses = @localmem Int32 (PYRAMID_WORKGROUP_SIZE,)

    sum_intensity = zero(eltype(input))
    local_status = Int32(PyramidPixelValid)
    input_index = Int(local_index)
    while input_index <= length(input)
        @inbounds value = input[input_index]
        if isfinite(value)
            sum_intensity += value
            if !isfinite(sum_intensity)
                local_status = _merge_pyramid_status(
                    local_status,
                    Int32(PyramidPixelNonFiniteMean),
                )
                sum_intensity = zero(eltype(input))
            end
        else
            local_status = _merge_pyramid_status(
                local_status,
                Int32(PyramidPixelNonFiniteInput),
            )
        end
        input_index += PYRAMID_WORKGROUP_SIZE
    end
    @inbounds local_sums[Int(local_index)] = sum_intensity
    @inbounds local_statuses[Int(local_index)] = local_status
    @synchronize

    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        128,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        64,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        32,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        16,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        8,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        4,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        2,
    )
    @synchronize
    _reduce_pyramid_sum_status!(
        local_sums,
        local_statuses,
        Int(local_index),
        1,
    )
    @synchronize

    if Int(local_index) == 1
        @inbounds operation_status = local_statuses[1]
        if operation_status == Int32(PyramidPixelValid)
            @inbounds mean_pupil_intensity = local_sums[1] * mean_scale
            if !isfinite(mean_pupil_intensity)
                operation_status = Int32(PyramidPixelNonFiniteMean)
            else
                reciprocal_current_frame_mean =
                    mean_pupil_intensity > zero(eltype(input)) ?
                    inv(mean_pupil_intensity) : zero(eltype(input))
                if !isfinite(reciprocal_current_frame_mean)
                    operation_status =
                        Int32(PyramidPixelNonFiniteNormalizationGain)
                else
                    @inbounds staged_mean_pupil_intensity[1] =
                        mean_pupil_intensity
                    @inbounds staged_reciprocal_current_frame_mean[1] =
                        reciprocal_current_frame_mean
                end
            end
        end
        @inbounds status[1] = operation_status
    end
    @synchronize

    gather_status = Int32(PyramidPixelValid)
    if @inbounds(status[1]) == Int32(PyramidPixelValid)
        output_index = Int(local_index)
        pupil_pixel_count = size(input, 1)
        gain = @inbounds reciprocal_previous_frame_mean[1]
        while output_index <= length(staged_pixels)
            pupil_offset = 0
            selected_pixel = output_index
            while selected_pixel > selected_pixel_count
                selected_pixel -= selected_pixel_count
                pupil_offset += 1
            end
            @inbounds input_index =
                pupil_offset * pupil_pixel_count +
                selected_pixel_offsets[selected_pixel] + 1
            @inbounds value = gain * input[input_index]
            if isfinite(value)
                @inbounds staged_pixels[output_index] = value
            else
                gather_status = Int32(PyramidPixelNonFiniteOutput)
            end
            output_index += PYRAMID_WORKGROUP_SIZE
        end
    end
    @inbounds local_statuses[Int(local_index)] = gather_status
    @synchronize

    _reduce_pyramid_status!(local_statuses, Int(local_index), 128)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 64)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 32)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 16)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 8)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 4)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 2)
    @synchronize
    _reduce_pyramid_status!(local_statuses, Int(local_index), 1)
    @synchronize
    if Int(local_index) == 1 &&
       @inbounds(status[1]) == Int32(PyramidPixelValid)
        @inbounds status[1] = local_statuses[1]
    end
end

@kernel function _pyramid_publish_kernel!(
    output_pixels,
    output_mean_pupil_intensity,
    staged_pixels,
    staged_mean_pupil_intensity,
    status,
)
    local_index = @index(Local, Linear)
    if @inbounds(status[1]) == Int32(PyramidPixelValid)
        output_index = Int(local_index)
        while output_index <= length(output_pixels)
            @inbounds output_pixels[output_index] = staged_pixels[output_index]
            output_index += PYRAMID_WORKGROUP_SIZE
        end
        if Int(local_index) == 1
            @inbounds output_mean_pupil_intensity[1] =
                staged_mean_pupil_intensity[1]
        end
    end
end

function _check_pyramid_devices(
    output::PyramidPixelOutput,
    workspace::PyramidPixelWorkspace,
    plan::PyramidPixelPlan,
    input,
)
    device = plan.backend.device
    _require_current_device(device)
    _require_array_device("plan mask", plan.mask, device)
    _require_array_device(
        "plan selected pixel offsets",
        plan.selected_pixel_offsets,
        device,
    )
    _require_array_device("input", input, device)
    _require_array_device("output pixels", output.pixels, device)
    _require_array_device(
        "mean pupil intensity output",
        output.mean_pupil_intensity,
        device,
    )
    _require_array_device(
        "workspace committed gain",
        workspace.reciprocal_previous_frame_mean,
        device,
    )
    _require_array_device(
        "workspace staged pixels",
        workspace.staged_pixels,
        device,
    )
    _require_array_device(
        "workspace staged mean",
        workspace.staged_mean_pupil_intensity,
        device,
    )
    _require_array_device(
        "workspace staged gain",
        workspace.staged_reciprocal_current_frame_mean,
        device,
    )
    _require_array_device("workspace status", workspace.status, device)
    return nothing
end

function PyramidPixelWorkspace(
    plan::PyramidPixelPlan{
        T,
        <:AbstractVector{Bool},
        <:AbstractVector{Int},
        <:PyramidPixelBackend,
    },
) where {T<:Union{Float32,Float64}}
    device = plan.backend.device
    return _with_eager_device(device) do
        FilterGraphAlgorithms._prepared_pyramid_pixel_workspace(
            T,
            _capture_zeros(device, T, 1),
            _capture_zeros(device, T, pyramid_output_pixel_count(plan)),
            _capture_zeros(device, T, 1),
            _capture_zeros(device, T, 1),
            _capture_zeros(device, Int32, 1),
            zeros(Int32, 1),
        )
    end
end

function _warm_pyramid_kernels!(plan::PyramidPixelPlan{T}) where {T}
    workspace = PyramidPixelWorkspace(plan)
    input = _capture_zeros(
        plan.backend.device,
        T,
        pyramid_pupil_pixel_count(plan),
        PYRAMID_PUPIL_COUNT,
    )
    output = PyramidPixelOutput(
        _capture_zeros(plan.backend.device, T, pyramid_output_pixel_count(plan)),
        _capture_zeros(plan.backend.device, T, 1),
    )
    process!(output, workspace, plan, input)
    FilterGraphAlgorithms.commit!(workspace)
    return plan
end

function _prepare_pyramid_pixel_plan(
    device,
    plan::PyramidPixelPlan{
        T,
        Vector{Bool},
        Vector{Int},
        Nothing,
    },
) where {T<:Union{Float32,Float64}}
    return _with_eager_device(device) do
        mask = _capture_array(device, plan.mask)
        selected_pixel_offsets =
            _capture_array(device, plan.selected_pixel_offsets)
        backend = KernelAbstractions.get_backend(mask)
        prepared_backend = PyramidPixelBackend(
            device,
            _pyramid_compute_kernel!(backend, PYRAMID_WORKGROUP_SIZE),
            _pyramid_publish_kernel!(backend, PYRAMID_WORKGROUP_SIZE),
        )
        prepared_plan = FilterGraphAlgorithms._prepared_pyramid_pixel_plan(
            T,
            mask,
            selected_pixel_offsets,
            plan.output_pixel_count,
            plan.mean_scale,
            prepared_backend,
        )
        return _warm_pyramid_kernels!(prepared_plan)
    end
end

function _process_pyramid_pixel!(output, workspace, plan, input)
    FilterGraphAlgorithms._check_pyramid_pixel_execution(
        output,
        workspace,
        plan,
        input,
    )
    _check_pyramid_devices(output, workspace, plan, input)
    workspace.candidate_ready = false

    plan.backend.compute_kernel(
        workspace.staged_pixels,
        workspace.staged_mean_pupil_intensity,
        workspace.staged_reciprocal_current_frame_mean,
        workspace.status,
        workspace.reciprocal_previous_frame_mean,
        plan.selected_pixel_offsets,
        input,
        plan.mean_scale,
        pyramid_selected_pixel_count(plan);
        ndrange=PYRAMID_WORKGROUP_SIZE,
    )
    plan.backend.publish_kernel(
        output.pixels,
        output.mean_pupil_intensity,
        workspace.staged_pixels,
        workspace.staged_mean_pupil_intensity,
        workspace.status;
        ndrange=PYRAMID_WORKGROUP_SIZE,
    )
    copyto!(workspace.completion_status, workspace.status)
    status = PyramidPixelStatus(workspace.completion_status[1])
    status == PyramidPixelValid || throw(PyramidPixelNumericalError(status))
    workspace.candidate_ready = true
    return output
end
