"""Marker supertype for native captured-execution support traits."""
abstract type CaptureSupport end

"""Trait value indicating that an Algorithm has no qualified capture adapter."""
struct NoCapture <: CaptureSupport end

"""Trait value indicating that an Algorithm has a qualified capture adapter."""
struct SupportsCapture <: CaptureSupport end

"""
    capture_support(::Type{Declaration}, device) -> CaptureSupport
    capture_support(prepared::PreparedAlgorithm, device) -> CaptureSupport

Return whether one Algorithm declaration has a fixed-address asynchronous
adapter for the concrete accelerator device. The conservative default is
[`NoCapture`](@ref). A backend extension opts in with a more-specific
declaration-and-device method returning [`SupportsCapture`](@ref).

Support promises that preparation binds every Plan, Workspace, input, output,
scratch, status, and kernel identity needed by
[`enqueue!`](@ref). It also promises that preflight,
completion, and reset preserve the declaration's documented outcomes without
performing eager fallback. Qualification on one backend or device type does
not imply qualification on another.
"""
function capture_support end

"""
    CaptureProtocolError

Cold preparation error indicating an unsupported or obsolete captured-execution
protocol. Update the backend adapter to the complete-transition protocol before
preparing a captured owner; ordinary checked `process!` is unchanged.
"""
struct CaptureProtocolError <: Exception
    message::String
end

Base.showerror(io::IO, error::CaptureProtocolError) = print(io, error.message)

"""
    CompleteTransition()

Preparation marker for a captured adapter whose [`enqueue!`](@ref) owns the
complete Algorithm transition: computation, numerical checks, and exactly one
guarded local state commitment. Earlier or local rejection prevents commitment;
a later Node rejection does not rewind a successful local commitment.

Pass this zero-field marker as the first argument to [`prepare_capture`](@ref).
It names the execution contract; it is not a numeric protocol version.
"""
struct CompleteTransition end

"""
    prepare_capture(
        CompleteTransition(), device, prepared, outputs, inputs, frame_status, node_index[, workgroups],
    )

Prepare one backend-specific captured Algorithm adapter bound to fixed device
arrays. [`CompleteTransition`](@ref) opts into [`enqueue!`](@ref) owning the
complete transition, including guarded local state commitment. Legacy untagged
built-in preparation and numeric or unknown protocol tags reject cold; tagged
calls never fall back to legacy adapters or checked execution.

`outputs` and `inputs` are tuples in declaration Port order with sparse
Parameter Ports omitted. `frame_status` is fixed device storage shared by all
adapters in one captured Graph; zero means every preceding Node remains valid.
An adapter records a node-qualified disposition code when it rejects its result.
`node_index` is the one-based position used to qualify those codes.

This is cold control-plane work and may allocate or throw. A supporting extension
implements the seven-argument tagged method. The optional eighth `workgroups`
argument is a cold execution profile; the default forwards only to that tagged
method. Geometry and every retained identity are fixed before native recording.
"""
function prepare_capture end

function _capture_protocol_error(protocol, device, prepared::PreparedAlgorithm{D}) where {D}
    throw(CaptureProtocolError(
        "capture protocol $protocol is unavailable for Algorithm '$(algorithm_label(D))' on " *
        "$(typeof(device)); implement prepare_capture(CompleteTransition(), device, prepared, " *
        "outputs, inputs, status, index[, workgroups]) and a complete enqueue! " *
        "that owns numerical checks and guarded local state commitment. " *
        "Reprepare the owner; legacy enqueue!/adopt! adapters cannot be reused.",
    ))
end

prepare_capture(::CompleteTransition, device, prepared::PreparedAlgorithm, outputs::Tuple, inputs::Tuple,
    status, index::Integer) = _capture_protocol_error(CompleteTransition(), device, prepared)

function prepare_capture(protocol::CompleteTransition, device, prepared::PreparedAlgorithm, outputs::Tuple,
    inputs::Tuple, status, index::Integer, workgroups)
    return prepare_capture(protocol, device, prepared, outputs, inputs, status, index)
end

prepare_capture(::Val{Protocol}, device, prepared::PreparedAlgorithm, outputs::Tuple, inputs::Tuple,
    status, index::Integer) where {Protocol} =
    _capture_protocol_error(Protocol, device, prepared)

prepare_capture(::Val{Protocol}, device, prepared::PreparedAlgorithm, outputs::Tuple, inputs::Tuple,
    status, index::Integer, workgroups) where {Protocol} =
    _capture_protocol_error(Protocol, device, prepared)

prepare_capture(device, prepared::PreparedAlgorithm, outputs::Tuple, inputs::Tuple,
    status, index::Integer) = _capture_protocol_error("legacy", device, prepared)

prepare_capture(device, prepared::PreparedAlgorithm, outputs::Tuple, inputs::Tuple,
    status, index::Integer, workgroups) =
    _capture_protocol_error("legacy", device, prepared)

"""
    preserves_inputs(adapter) -> Bool

Return whether an adapter preserves its input bindings during execution. The default
is `false`, so unqualified adapters are barriers to Graph validation reuse.

An opt-in adapter may write only its supplied outputs, shared frame status and
private state/scratch. It must not modify another Node's buffers through hidden
aliases, retain asynchronous writers outside the owner's ordered stream, or clear
an existing nonzero frame status. These promises include conditional local state commitment. Output/input overlap must be excluded separately before reusing a proof;
writing an aliased supplied output does not preserve the corresponding input.
This trait does not enable an unchecked direct-adapter execution mode.
"""
preserves_inputs(adapter) = false

"""
    alias_input(adapter, ::Val{OutputIndex}) -> Union{Nothing,Int}

Return the input index whose exact storage may replace this captured output,
or `nothing` (the default). This cold preparation trait is restricted to exact
stateless identity outputs: no arithmetic, conversion, or permutation is allowed.
Input and output tuples omit Parameter Ports, as in [`prepare_capture`](@ref).

An opt-in adapter must support preparing against that identical input/output
buffer without writing it during execution or reset. Graph preparation
must finalize downstream routes before recording and prove that every adapter
preserves borrowed inputs.
The adapter must retain its checks, failure identities and metadata behavior.
This trait grants no permission to reuse storage for arbitrary arithmetic.
"""
alias_input(adapter, ::Val) = nothing

"""
    finite_output(adapter, ::Val{OutputIndex}) -> Bool

Return whether an adapter guarantees a finite complete output whenever shared
frame status remains zero after its complete Algorithm transition.
The default is `false`. `OutputIndex` is the one-based position in the `outputs`
tuple passed to [`prepare_capture`](@ref), with Parameter Ports omitted.

An opt-in guarantee must cover every successful path, initialization, reset and
replay. All output writes must complete in the owner's ordered stream before
downstream work, and an existing nonzero status must remain unchanged. The
guarantee concerns the exact supplied output buffer, not an overlapping view.
This is a per-execution guarantee, not a query of current buffer contents.
It must follow from retained validation or an equally strong invariant; finite
Parameters alone do not establish finite results of floating-point arithmetic.
"""
finite_output(adapter, ::Val) = false

"""
    prepare_validation(adapter, finite_inputs::Tuple)

Return a captured adapter specialized for proven finite input buffers. The
default returns `adapter` unchanged. This is cold Graph preparation, before
warmup and native recording; direct adapter preparation remains fully checked.

Each Boolean in `finite_inputs` corresponds to a sparse-Parameter-free input
tuple position. A `true` entry means the exact unchanged buffer has already been
proved finite by earlier ordered Graph work whenever frame status is zero.
The Graph must establish input preservation and exclude output/input aliases
before supplying these proofs, except for an exact identity output classified
by [`alias_input`](@ref). External inputs must remain unproven.

A supporting implementation may omit redundant finite-input validation and a
finite-output scan of an exact bit-preserving identity of a proven finite input.
It must retain all other checks, scientific work, buffers, failure identities,
status guards and state adoption. It must not mutate scientific state during
preparation, add replay bookkeeping, or weaken arithmetic output/overflow
validation. These proofs are not user-facing options to disable checks.
"""
prepare_validation(adapter, finite_inputs::Tuple) = adapter

# Cold qualification hook for backend-independent validation/alias refinements.
# Sharing equations does not qualify another backend to elide its checks.
_supports_validation_reuse(backend) = false

"""
    preflight(adapter)

Perform bounded host-side admission checks that are knowable before a native
graph launch. Return `nothing` to admit the Algorithm or one of its declared
expected-failure values to suppress launch. The method must not enqueue device
work or mutate mathematical state.
"""
function preflight end

"""
    enqueue!(adapter)

Enqueue the complete fixed-address transition of one admitted Algorithm:
computation, numerical checks, and exactly one guarded local state commitment.
Earlier or local rejection prevents commitment; a later Node rejection does not
rewind this Algorithm's successful state. The owner calls this once per Node;
there is no separate public adoption phase. Multiple internally ordered kernels
are permitted when device dependencies require them.

This method is recorded inside a native CUDA or HIP Graph. It must not
synchronize, allocate execution storage, replace a captured identity, query a
device result, invoke a host callback, or mutate host scientific state.
"""
function enqueue! end

"""
    complete!(adapter, frame_status)

Interpret the copied graph-wide `frame_status` after the retained accelerator
stream has finished. Return `nothing` when the status does not belong to this
adapter, or return one of the Algorithm's declared expected-failure values.
The method must not perform a device transfer or synchronization. Unexpected
errors propagate to the captured owner and make that owner fail-stop.
"""
function complete! end
