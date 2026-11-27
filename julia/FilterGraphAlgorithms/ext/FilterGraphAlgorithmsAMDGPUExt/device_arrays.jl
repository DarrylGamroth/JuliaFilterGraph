function _require_current_device(device::AMDGPU.HIPDevice)
    AMDGPU.device() == device || throw(
        ArgumentError(
            "current AMDGPU device does not match the prepared Algorithm Plan",
        ),
    )
    return nothing
end

function _require_array_device(
    name::AbstractString,
    array::AMDGPU.ROCArray,
    device::AMDGPU.HIPDevice,
)
    AMDGPU.device(array) == device || throw(
        ArgumentError("$name is not resident on the prepared AMDGPU device"),
    )
    return nothing
end

function _with_eager_device(f, device::AMDGPU.HIPDevice)
    AMDGPU.functional() || throw(ArgumentError("AMDGPU is not functional on this host"))
    return AMDGPU.device!(f, device)
end

_capture_array(device::AMDGPU.HIPDevice, values) = AMDGPU.ROCArray(values)
_capture_zeros(device::AMDGPU.HIPDevice, ::Type{T}, dimensions...) where {T} =
    AMDGPU.zeros(T, dimensions...)

function _require_capture_array(name, array::AMDGPU.ROCArray, device::AMDGPU.HIPDevice)
    AMDGPU.device(array) == device || throw(
        ArgumentError("$name is not resident on the selected AMDGPU device"),
    )
    return nothing
end

# Contiguous ROCArray views in AMDGPU 2.8 report only their starting pointer as
# dataids. Shifted overlapping views therefore need a cold byte-range check.
# Preparation already requires both arrays on the selected device. Non-bitstype
# storage can have additional selector bytes; keep its borrowing conservative.
function _capture_arrays_overlap(left::AMDGPU.ROCArray, right::AMDGPU.ROCArray)
    (isempty(left) || isempty(right)) && return false
    (isbitstype(eltype(left)) && isbitstype(eltype(right))) || return true
    left_bytes = UInt128(length(left)) * UInt128(Base.elsize(typeof(left)))
    right_bytes = UInt128(length(right)) * UInt128(Base.elsize(typeof(right)))
    (iszero(left_bytes) || iszero(right_bytes)) && return false
    GC.@preserve left right begin
        left_start = UInt128(UInt(pointer(left)))
        right_start = UInt128(UInt(pointer(right)))
        return left_start < right_start + right_bytes &&
               right_start < left_start + left_bytes
    end
end

# Cold validation of the fixed workgroup geometry used by shared adapters.
function _require_capture_workgroup(device::AMDGPU.HIPDevice, lanes::Integer;
        shared_memory::Integer=0)
    AMDGPU.device() == device || throw(
        ArgumentError("captured workgroup device must be the selected AMDGPU device"))
    lanes > 0 || throw(ArgumentError("captured workgroup lanes must be positive"))
    shared_memory >= 0 || throw(
        ArgumentError("captured workgroup shared memory must be nonnegative"))
    properties = AMDGPU.HIP.properties(device)
    limit = min(properties.maxThreadsPerBlock, first(properties.maxThreadsDim))
    lanes <= limit || throw(ArgumentError(
        "captured workgroup requires $lanes lanes; selected AMDGPU device supports at most $limit in one dimension"))
    shared_memory <= properties.sharedMemPerBlock || throw(ArgumentError(
        "captured workgroup requires $shared_memory bytes of shared memory; selected AMDGPU device supports $(properties.sharedMemPerBlock) bytes per block"))
    return nothing
end
