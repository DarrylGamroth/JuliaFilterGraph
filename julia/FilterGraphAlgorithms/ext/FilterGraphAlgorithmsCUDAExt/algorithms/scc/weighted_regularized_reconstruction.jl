# Keep the native device barrier inside the uniform early-exit solve loop.
# Numerical stages, launch composition and the bounded selector are shared.
struct _SccBarrier end
@inline (::_SccBarrier)() = CUDA.sync_threads()
_prepare_scc_execution(output::CUDA.CuArray, interaction, regularization) =
    _prepare_scc_execution(_SccBarrier(), CUDA.device(output), output, interaction, regularization)
