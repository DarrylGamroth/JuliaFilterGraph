# Internal scratch borrowing is qualified separately from validation reuse.
_projection_borrowing(::AMDGPU.HIPDevice) = Val(true)
