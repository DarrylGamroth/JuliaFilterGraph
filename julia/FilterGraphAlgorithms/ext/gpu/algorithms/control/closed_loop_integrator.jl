# ClosedLoopIntegratorF32 uses the same delayed feedback, state adoption and
# output validation as ClosedLoopCorrectionF32, with no hidden-mode projections.
# A sentinel avoids zero-sized projection launches and preserves the CPU's
# omission of hidden-mode arithmetic when the prepared mode count is zero.
struct NoHiddenModes end

_enqueue_projection!(::NoHiddenModes, output, frame_status, input) = nothing
_expansion_values(::NoHiddenModes) = nothing
