"""
    AbstractAlgorithmPlan

Nominal interface boundary for an admitted prepared numerical operation.

A concrete subtype contains validated dimensions and layouts, logically
read-only prepared artifacts, numerical options, and any selected backend.
Mutable scratch, in-progress state, and committed mathematical history belong
in a separate compatible workspace with one authoritative owner. Input and
output are short-lived caller-owned views and must not be retained after a
complete call or caller-managed progressive sequence.

One plan represents one independently callable numerical operation. Related
operations use separate plan types, although they may share immutable prepared
artifacts. A sufficient `LinearAlgebra` or dependency primitive remains a
direct implementation call rather than acquiring a forwarding plan solely for
uniformity.

One plan operation does not imply one implementation kernel or dependency
call. A backend implementation may use no kernel, one kernel, several kernels,
or direct dependency operations while preserving every observable result,
state transition, unit, coordinate space, failure outcome, ordering rule, and
progressive-output guarantee. More-specific methods and concrete prepared
storage may provide an optional accelerator implementation. Device buffers,
transfers, launch, synchronization, and graph scheduling remain outside this
interface.

Physical units and coordinate spaces are part of the concrete operation
contract. A unit-parametric operation preserves its documented relationships
and does not infer or introduce a conversion unless conversion is the declared
operation.
"""
abstract type AbstractAlgorithmPlan end

"""
    NoWorkspace()

Internal zero-size workspace for direct low-level operation tests with no
mutable execution storage.

Scientist declarations do not name or construct this value: returning only a
Plan from `prepare` makes the declaration machinery supply its private empty
workspace. `NoWorkspace` remains an implementation detail for the older
direct Plan interface while that interface is retired.
"""
struct NoWorkspace end

"""
    process!(output, workspace, plan, input) -> output
    process!(output, workspace, plan, input, workrange) -> ProgressResult

Execute one complete operation or one progressive range of semantic work.

Complete execution uses the four-argument method. Progressive plans accept a
nonempty, one-based inclusive `AbstractUnitRange{Int}` in their declared
semantic input-unit domain. Both forms may mutate caller-owned output and
workspace storage but must not mutate the logically read-only plan or retain
input and output views.

A failed complete operation must preserve mathematical history committed
before the call. Scratch that cannot affect a later operation may be mutated.
Expected numerical conditions belong in the concrete output, validity, status,
or caller-owned result rather than exception-driven control flow. Exceptions
indicate documented direct misuse, a broken prepared invariant, or an
unexpected failure.

Methods selected for an operating path perform bounded work without Julia heap
allocation after warmup. Preparation-only methods require their own resource
and reproducibility evidence but are not implicitly real-time operations.
An accelerator method satisfies the complete four-argument contract only when
the entire operation has finished before the method returns.
"""
function process! end

"""
    commit!(workspace) -> workspace

Publish candidate mathematical state after the owning Algorithm reaches
terminal success. An admitted Workspace must perform bounded, allocation-free,
non-failing work after warmup.
"""
function commit! end

"""
    discard!(workspace) -> workspace

Abandon candidate mathematical state when the owning Algorithm rejects or its
in-flight work is aborted, while preserving the previously committed state. An
admitted Workspace must perform bounded, allocation-free, non-failing work
after warmup.
"""
function discard! end

"""
    reset!(workspace) -> workspace
    reset!(adapter)

Restore a Workspace's documented initial mathematical state at a serialized
lifecycle boundary. Reset must reuse prepared storage for an admitted
Workspace.

Captured adapters implement the same operation to restore their initial device
state without replacing any address retained by the native executable. An
adapter method may enqueue reset work; its Graph owner supplies the completion
boundary. Call reset only when the owner is quiescent, with no concurrent replay
or external writer accessing its retained buffers.
"""
function reset! end

"""
    recover!(workspace) -> workspace

Remove obsolete correction history at a serialized protective-recovery boundary,
using existing prepared CPU storage. Supported Workspaces define the history
they clear and the committed physical-command state they preserve. There is no
fallback to [`reset!`](@ref) for unsupported Workspaces.

For a CPU `LeakyIntegratorWorkspace`, recovery clears committed and candidate
correction while preserving the initial state used by reset.
For `ClosedLoopCorrectionWorkspace` and `BiquadCascadeWorkspace`, recovery clears
controller or temporal-filter history. For a CPU `PdmCommandWorkspace`, it
discards candidates and preserves committed current.
`CorrectorAllocationWorkspace` clears its temporal histories
and obsolete scratch while retaining conditioned current, held commands and
update phase. Subsequent processing reaches the configured reference through
the existing command limits, slew limits and update periods.

The caller must serialize recovery with processing and reconcile all coupled
controller, filter, allocation and delayed-feedback owners before reuse. This
operation does not open or close a loop, change a Plan or reference, publish a
command, or replace ordinary hold/resume. It performs bounded, allocation-free,
non-failing work after warmup for admitted supported Workspaces.
"""
function recover! end

# Neumaier summation retains a small contribution when a later large term cancels.
@inline function _compensated_add(total::Float64, correction::Float64, x::Float64)
    next = total + x
    error = abs(total) >= abs(x) ? (total - next) + x : (x - next) + total
    return next, correction + error
end
