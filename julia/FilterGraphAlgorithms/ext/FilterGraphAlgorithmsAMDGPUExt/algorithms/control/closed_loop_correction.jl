# This is a workgroup-wide barrier/vote, including every wavefront. Convert
# AMDGPU's Cint predicate/result at the boundary to the shared Bool contract.
struct _BlockAny end
@inline (::_BlockAny)(failure) = !iszero(AMDGPU.sync_workgroup_or(Int32(failure)))

# The one-mode shortcut is qualified against CUDA's one-term cuBLAS product.
# Retain ordinary KA projection arithmetic on AMD until separately qualified.
function _prepare_expansion(device::AMDGPU.HIPDevice, matrix, output, input)
    _require_expansion(matrix, output, input)
    return _prepare_projection(device, matrix, output, input,
        COOPERATIVE_PROJECTION_DEFAULT_LANES)
end
