# Eager device ownership and activation; scientific work lives in ext/gpu.
function _with_eager_device(f, device::CUDA.CuDevice)
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    return CUDA.device!(f, device)
end

function _require_current_device(device::CUDA.CuDevice)
    CUDA.device() == device || throw(ArgumentError(
        "current CUDA device does not match the prepared Algorithm Plan"))
    return nothing
end

function _require_array_device(name::AbstractString, array::CUDA.CuArray, device::CUDA.CuDevice)
    CUDA.device(array) == device || throw(ArgumentError(
        "$name is not resident on the prepared CUDA device"))
    return nothing
end

_capture_array(device::CUDA.CuDevice, values) = CUDA.CuArray(values)
_capture_zeros(device::CUDA.CuDevice, ::Type{T}, dimensions...) where {T} =
    CUDA.zeros(T, dimensions...)

function _require_capture_array(name, array::CUDA.CuArray, device::CUDA.CuDevice)
    CUDA.device(array) == device || throw(
        ArgumentError("$name is not resident on the selected CUDA device"),
    )
    return nothing
end

function _check_capture_workgroup(lanes::Integer, limits;
        shared_memory::Integer=0, warp_width::Union{Nothing,Integer}=nothing)
    lanes > 0 || throw(ArgumentError("captured workgroup lanes must be positive"))
    shared_memory >= 0 || throw(
        ArgumentError("captured workgroup shared memory must be nonnegative"),
    )
    lanes <= limits.max_threads && lanes <= limits.max_x || throw(
        ArgumentError("captured workgroup requires $lanes lanes; selected CUDA device " *
                      "supports at most $(min(limits.max_threads, limits.max_x)) in one dimension"),
    )
    shared_memory <= limits.shared_memory || throw(
        ArgumentError("captured workgroup requires $shared_memory bytes of shared memory; " *
                      "selected CUDA device supports $(limits.shared_memory) bytes per block"),
    )
    if warp_width !== nothing
        warp_width > 0 || throw(ArgumentError("required CUDA warp width must be positive"))
        warp_width == limits.warp_width || throw(
            ArgumentError("captured kernel requires CUDA warp width $warp_width; " *
                          "selected device has width $(limits.warp_width)"),
        )
    end
    return nothing
end

# Cold preparation only. Device limits are necessary conditions; the exact-path
# warmup still establishes that each compiled kernel fits its resource limits.
function _require_capture_workgroup(device::CUDA.CuDevice, lanes::Integer;
        shared_memory::Integer=0, warp_width::Union{Nothing,Integer}=nothing)
    CUDA.device() == device || throw(
        ArgumentError("captured workgroup device must be the selected CUDA device"),
    )
    limits = (
        max_threads=CUDA.attribute(device, CUDA.CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        max_x=CUDA.attribute(device, CUDA.CUDACore.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        shared_memory=shared_memory > 0 ?
            CUDA.attribute(device, CUDA.CUDACore.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) : 0,
        warp_width=warp_width === nothing ? nothing : CUDA.warpsize(device),
    )
    return _check_capture_workgroup(lanes, limits; shared_memory, warp_width)
end
