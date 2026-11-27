# Empty policy values also pass to device kernels without array adaptation.
struct NoSlew end
struct IndependentSlew end

_pdm_slew_policy(::Nothing) = NoSlew()
_pdm_slew_policy(::AbstractVector) = IndependentSlew()

@inline _pdm_current(::NoSlew, committed, target, limit) = target
@inline _pdm_current(::IndependentSlew, committed, target, limit) = _move_toward(
    committed, target, limit)
