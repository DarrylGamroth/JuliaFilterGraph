# Synchronizes the complete workgroup, including all wavefronts.
struct _SccBarrier end
@inline (::_SccBarrier)() = AMDGPU.sync_workgroup()
_prepare_scc_execution(output::AMDGPU.ROCArray, interaction, regularization) =
    _prepare_scc_execution(_SccBarrier(), AMDGPU.device(output), output, interaction, regularization)
