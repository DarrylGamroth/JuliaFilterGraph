# FilterGraphAlgorithms

Scientists define an Algorithm's equations, ports, calibration, properties and
state here. FilterGraphs connects those Algorithms and owns scheduling,
transport buffers, publication and CUDA Graph replay. An Algorithm may use
several GPU kernels without becoming several scientific graph nodes.

## Reconstructor coefficient storage

Set `coefficient_storage = "fp16"` in a node's `config` to store its
reconstructor coefficients as Float16 with one power-of-two Float32 scale per
row. The default is `"fp32"`. Both settings use Float32 calibration inputs,
measurements, accumulation, and outputs. Changing the setting requires a
graph rebuild.

The option applies to `dense-vector-reconstructor-f32` and
`scc-reconstructor-f32` on CPU and in CUDA and AMDGPU captures, and to
`incremental-dense-reconstructor-f32` in the CPU algorithm.

## Find an implementation

Start with `src/algorithms/<domain>/<algorithm>.jl`. Each file keeps the scientific
Plan, Workspace, CPU equations and declaration together. Public docstrings live
beside their definitions. Use a preceding docstring on `@algorithm` for its
declaration and on the nested `@configuration` for its configuration. Types and
functions use preceding docstrings.

Define kernels in terms of the measurement and its calibration. Use multiple
dispatch for different storage types, and use geometry, memory traffic and
numerical bounds to choose an implementation. Put instrument names and fitted
dimensions in the graph or benchmark that supplies them.

GPU paths use the same domain and algorithm names:

| Responsibility | Location |
| --- | --- |
| CPU equations, calibration and declaration | `src/algorithms/<domain>/<algorithm>.jl` |
| Portable KernelAbstractions implementation | `ext/gpu/algorithms/<domain>/<algorithm>.jl` |
| CUDA implementation or refinement | `ext/FilterGraphAlgorithmsCUDAExt/algorithms/<domain>/<algorithm>.jl` |
| AMDGPU implementation or refinement | `ext/FilterGraphAlgorithmsAMDGPUExt/algorithms/<domain>/<algorithm>.jl` |

Portable KA kernels also run on the private `KernelAbstractions.CPU()` test
device. The 664-check suite passed with KA 0.9.45's Julia-task CPU backend and
KA 0.10.1's PoCL backend. These checks compare equations, reductions, failure
handling and state against independent references; the PoCL run is not native
Julia CPU-executor qualification. The maintained native CPU executor remains a
separate path.

## Process device-resident buffers

All 84 built-in Algorithm declarations have a numerical execution path for
`CUDA.CuArray` and `AMDGPU.ROCArray`. Load KernelAbstractions and the selected
GPU package. The input and output buffers, working arrays and scientific
history used during processing reside on the selected device. Configuration
and sparse calibration can be prepared on the host. Sample metadata, admission
and completion status can also live on the host.

| Execution path | Declarations | When to use it |
| --- | ---: | --- |
| Fixed-address capture | 68 | Bind the input, output and status arrays once with `prepare_capture(CompleteTransition(), ...)`. The graph owner records the transition and publishes on success. |
| Checked device execution | 16 | Use `prepare_algorithm(declaration, configuration, device)` and call `process!` with device arrays and the declaration's sample metadata. Packet assembly, conditional publication and per-exposure decisions retain their ordinary semantics. |

`capture_support(D, device)` selects between these paths for the built-in
catalog. Look for `SupportsCapture()` to record native CUDA/HIP graph work.
The 16 with `NoCapture()` are the row and readout pixel calibrators, detector
frame assembler, incremental dense reconstructor, five progressive Pyramid
declarations, five progressive Shack–Hartmann declarations,
`ChoppedDetectorDifferenceF32`, and `SccCorrectionF32`. Their device
implementation can feed its output to the next Algorithm on the same device.
Progressive processing can synchronize to report scalar progress or rejection.

This example calibrates the first detector row using distinct input and output
arrays on the same GPU.

```julia
using CUDA, KernelAbstractions, FilterGraphAlgorithms
import FilterGraphAlgorithms as FGA

config = PixelCalibrationRowU16F32Configuration(image_size=(4, 4), row_block_rows=1,
    initial_flat=nothing, initial_background=nothing)
calibrator = prepare_algorithm(PixelCalibrationRowU16F32, config, CUDA.device())
raw = CUDA.fill(UInt16(12), 1, 4)
calibrated = CUDA.zeros(Float32, 1, 4)
sample = SampleMetadata(1, 0; terminal=false)
FGA.process!(calibrated, calibrator, raw, (raw=sample,))
```

Use the corresponding ROCArray allocations and `AMDGPU.device()` for HIP.

For example, compare these files when reviewing Shack–Hartmann centroids:

- [CPU equations and declaration](src/algorithms/wavefront_sensor/shack_hartmann_image.jl)
- [Generic GPU implementation](ext/gpu/algorithms/wavefront_sensor/shack_hartmann_image.jl)
- [CUDA reduction selection](ext/FilterGraphAlgorithmsCUDAExt/algorithms/wavefront_sensor/shack_hartmann_image.jl)

For pupil processing, image formation and reconstruction have separate files,
as on CPU. Progressive pyramid and Shack–Hartmann readout measurement kernels
also have separate algorithm files. CUDA's `readout_kernels.jl` only assembles
the kernel bundle consumed by the graph's progressive executor.

Some primitives deliberately serve several Algorithms:

- `ext/gpu/linear_algebra/projection.jl`: shared identity/selection
  classification and execution plus portable matrix/vector operations.
- `ext/gpu/linear_algebra/finite_values.jl`: the common finite-vector reduction.
- `ext/FilterGraphAlgorithmsCUDAExt/linear_algebra/projection.jl`: native CUDA
  execution hooks, cuBLAS dense projection dispatch, and CUDA-specific proof and
  math-mode qualification.
- `ext/FilterGraphAlgorithmsAMDGPUExt/linear_algebra/projection.jl`: the qualified
  internal scratch-borrowing opt-in; dense arithmetic and selection reductions
  remain in shared KA source.
- `ext/gpu/algorithms/deformable_mirror/pdm_offset.jl`:
  the shared addition/validation implementation for system flat and engineering
  offset; each declaration retains its own failure identities. Its matching CUDA
  file prepares and qualifies the backend execution.

Algorithm-specific validation and proof refinements belong with that Algorithm.
The shared finite reduction supplies a numerical primitive, not an Algorithm's
validation policy. Shared arithmetic assumptions live in
`ext/gpu/linear_algebra/arithmetic_bounds.jl`; CUDA-specific proof hooks live in
`ext/FilterGraphAlgorithmsCUDAExt/input_proofs.jl`. Pixel, pupil and projection
bounds live with their owners. A CUDA proof does not qualify the HIP
implementation.

## Complete-array streaming statistics

`ArrayStatisticsF32` (`array-statistics-f32`) collects headless per-element
statistics from full Float32 arrays of any configured positive shape. It is
available in `algorithms()` for an optional analysis Graph. It has no I/O or
control-loop side effects and currently has a CPU implementation.

| Configuration | Meaning |
| --- | --- |
| `size` | Positive integer array dimensions, preserving the input shape |
| `mode = "finite"` | Collect `window_samples` accepted frames, then freeze until reset |
| `mode = "sliding"` | Update over the last `window_samples` accepted frames, with partial warmup |
| `mode = "exponential"` | Use α = −expm1(−`sample_period_s`/`time_constant_s`) |
| `max_state_bytes` | Maximum owned numeric storage, default 268435456 bytes (256 MiB) |
| `input_schema`, `mean_schema`, `variance_schema`, `stddev_schema` | Required versioned schemas for the respective quantities |

`mean`, `variance`, and `stddev` are same-shape Float32 arrays. The mean and
standard deviation have the input's units; variance has squared units. The
UInt64 `sample-count` counts all accepted frames since reset (including evicted
sliding samples). The Bool `complete` is true only for a full finite window.
The Bool `variance-valid` distinguishes warmup from a defined variance. Each
scalar output has shape `(1,)`. All outputs inherit `sample` metadata.

Finite/sliding variance is the unbiased sample variance M₂/(N−1). Before two
samples are available it is zero and invalid. Exponential variance is central
weighted population variance, initialized to zero at the first sample and
valid immediately. It follows μ′ = μ + α(x−μ),
v′ = (1−α)(v + α(x−μ)²), without an unbiased correction. The default
`window_samples`, `sample_period_s`, and `time_constant_s` are 1, 1.0, and 1.0;
time values must be finite and positive.

Means use compensated Float64 sums. Sliding variance is reconstructed from the
exact retained Float32 window with a centered second pass. Finite variance uses
a centered recurrence, and exponential means use the equivalent convex weighted
sum. This preserves small signals after large level shifts and avoids subtracting
nearly equal large centered sums during sliding eviction. Finite/exponential
storage is O(P); sliding storage is O(PW), where P is the array element count
and W is the window length. Per-frame work is O(P) for finite/exponential modes and O(PW) for sliding mode,
including validation and bounded local commitment; preparation allocates all
storage. Sliding reconstruction trades window-length work for numerical stability. Samples and
outputs must be disjoint from each other and the owned state. Non-finite
samples or statistics that cannot be published as finite Float32 reject without
changing committed history or caller outputs. A completed finite window
republishes its final products and ignores further input until reset.

The window and nominal exponential period count **accepted analysis frames**.
Copied streams and drop queues can omit detector exposures; these settings do
not assert exposure contiguity or actual wall-clock spacing. `commit!` adopts
one successful local Algorithm sample, `discard!` abandons a candidate, and
`reset!` clears all history and completion. As with other stateful Algorithms,
a later Node rejecting the Graph does not undo an earlier local adoption.

The CPU kernels use ordinary SIMD across independent elements and explicit FMA
for finite/exponential centered updates. Each pixel keeps its Neumaier addition
and history order; no fast-math reassociation is permitted. Sliding passes visit
each retained frame contiguously and use two additional Float64 scratch arrays.
Owned numeric storage is 52P bytes for finite/exponential modes and
(68 + 4W)P bytes for sliding mode, all included in `max_state_bytes`.

The [CPU benchmark](benchmark/array_statistics.jl) records JSON summaries, raw
CSV timings and concrete LLVM/native listings. Its environment needs
FilterGraphs and JSON.jl; these are benchmark dependencies, separate from the
FGA core. Use the selected-source environment and the command documented in the
script. Before/after qualification on 2026-10-28 compared `f771217` with these
SIMD kernels on an AMD Ryzen 7 6800H (AVX2/FMA), Julia 1.13.3/LLVM 20.1.8,
one Julia/BLAS thread, default `znver3` CPU target, and caller affinity CPU 2.
Two fresh-process pairs each measured five repetitions of 1024 checked Graph
calls after warmup, cycling four deterministic frames. Sliding W was 16; the
finite limit was 10⁹ so every measured finite call remained active. Every call
succeeded and all six cases allocated zero warmed bytes.

Observed ranges across the two process pairs, in microseconds:

| Shape | Mode | Before p50 | SIMD p50 | Before p99 | SIMD p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 | finite | 13.97–18.93 | 6.98–7.01 | 16.39–25.18 | 7.11–7.18 |
| 1024 | sliding | 70.85–119.99 | 15.79–15.80 | 88.67–146.63 | 16.66–19.82 |
| 1024 | exponential | 13.01–18.66 | 6.32–6.43 | 13.65–23.74 | 6.72–6.85 |
| 64 × 96 | finite | 88.51–124.54 | 39.73–40.23 | 100.11–138.99 | 46.94–47.44 |
| 64 × 96 | sliding | 465.28–473.76 | 94.25–95.04 | 505.23–744.30 | 110.47–113.14 |
| 64 × 96 | exponential | 77.84–81.71 | 36.50–37.92 | 89.39–93.52 | 44.11–45.23 |

These are closed-loop service-time observations on a shared host without a fixed
CPU frequency, not deployment latency guarantees. Initial and final runs varied;
the JSON/CSV records retain the distributions and repetition medians. They do
not measure queueing, offered-load capacity or GPU performance. The concrete
Matrix{Float32} Graph specialization emitted four-double LLVM vector operations
and AVX2 packed arithmetic in all three numerical kernels, with packed FMA in
finite/exponential updates. Compensated additions/subtractions retained strict
floating-point operations without `fast`/`reassoc` flags. BigFloat vector-lane
and tail oracles, level-shift regressions, atomic rejection, warmed allocations
and forced-bounds checks passed alongside the numerical Graph tests.

The public mathematical names follow the standard mean, centered sum of
squares M₂, sample variance, standard deviation, and exponential central
variance equations above. The declaration and its F32 suffix follow the
existing Algorithm naming contract. No detector-specific units are assumed.

## Understand the interfaces

| File or directory | Contract |
| --- | --- |
| `src/processing.jl` | Direct numerical processing and Workspace lifecycle |
| `src/progress.jl` | Incremental input and completion protocol |
| `src/capture.jl` | Fixed-storage asynchronous GPU preparation, execution and completion |
| `src/sample_metadata.jl` | Transport-neutral acquisition/sample identity |
| `src/properties/` | Scalar property schema, updates and binding ownership |
| `src/declarations/description.jl` | Algorithm/configuration descriptions and queries |
| `src/declarations/ports.jl` | Port directions, roles, layouts and formats |
| `src/declarations/prepared.jl` | Prepared Algorithm, process outcomes and admission |
| `src/declarations/parser.jl`, `algorithm_macro.jl` | Internal `@algorithm` parsing and code generation |

`recover!(workspace)` is the protective correction-history lifecycle operation
for explicitly supported CPU controller, biquad, command-conditioner and
corrector-allocation Workspaces. It preserves physical current and update phase
where applicable, while `reset!` restores initial state. Serialize it with
processing and reconcile coupled delayed feedback separately.

Scientists normally read the Algorithm file and these contracts; macro internals
are not required to write equations or implement a GPU specialization. Backend
`device_arrays.jl` files supply allocation, residency, activation and capability
hooks. Allocation is cold; eager execution also checks device ownership. The generic `src/processing.jl` owns direct numerical
processing and Workspace lifecycle; AMDGPU-specific behavior is provided by
explicit device preparation and native execution hooks, not a separate
AMDGPU `processing.jl` implementation.

Each backend binds its device type to `CUDA.CuDevice` or `AMDGPU.HIPDevice`.
Backend preparation methods and shared KernelAbstractions implementations live
beside the relevant Algorithms. This keeps
public dispatch tied to the backend without duplicating the equations or
overwriting another extension's methods. CUDA-only collectives and library
choices remain small backend specializations.

For prepared port metadata, use `algorithm_format(prepared, Symbol("vdm-command"))`
(or `:image` for a natural Julia identifier). Names are the declaration's canonical
graph names, including sparse parameter names. `@algorithm` generates direct
access from those names; adapter code need not repeat tuple positions.
`algorithm_formats(prepared)` remains the ordered tuple for iteration. String
names are accepted for dynamic preparation code; literal Symbols retain static
format inference. These queries do not run during GPU replay.

The portable DM projection policies and exact identity/selection handling live
beside their shared adapters and use a conservative backend qualification hook.
CUDA retains its existing proof/alias qualification; shared source does not opt
HIP, KA CPU or Metal into it.

## Write GPU work that is readable and bounded

Retain reusable device views and buffer descriptors during preparation. Passing
an existing device array to a Graph can avoid host payload uploads; constructing
new device view descriptors every frame can still allocate on the host.

Use ordinary helper functions for equations and multiple dispatch for behavior
selected by types. Choose execution policies during preparation and retain
concrete fields in the adapter. Specializations for identity, selection, dense
projection, slew and quantization avoid repeated runtime type routing.

Use enums for values: failure categories, changing lifecycle state, per-actuator
roles and configuration choices. A compact enum array can describe many
actuators; a different Julia object type per entry would obscure that data model.
Use concrete policy types when preparation selects different implementations,
as with projection structure, independent slew and quantization. A configuration
enum can remain scientist-facing even when preparation selects a typed policy.

Name prepared values by their scientific role. For example, controller maps are
`projections.shape_to_hidden` and `projections.hidden_to_shape`, and its outputs
are `outputs.correction` and `outputs.controller_state`. Scalar coefficients and
failure categories have named fields. Keep genuine coordinates and array
indices: `frame_status[1]` accesses a one-element device status buffer, while
`Val{OutputIndex}` in a trait identifies the declared output position. Those
indices do not select an execution protocol.

A function boundary is not a kernel boundary. Helpers can inline inside a GPU
kernel. `enqueue!` records the complete ordered scientific transition; it may
launch multiple kernels where a reduction, library operation or guarded state
publication requires it. cuBLAS cannot inline arbitrary Julia helpers. Changing
that boundary requires numerical, state and whole-chain performance evidence.

New GPU implementations for RTC graphs must be capture-friendly from the start:
allocate and bind storage during preparation, enqueue device work asynchronously,
and keep host reads and waits out of the recorded transition. Implement
`prepare_capture(CompleteTransition(), ...)` and let one `enqueue!` own computation,
checks and guarded local commitment. `CompleteTransition` names that contract;
obsolete numeric protocol tags are rejected. Keep `capture_support` conservative
until the concrete backend path is qualified. For progressive Algorithms, test
checked device execution with the actual block order and completion metadata.

Prefer `for` with a `UnitRange` or `StepRange` when the iteration domain is known:

```julia
for index in lane:lanes:length(input)
    # Each lane visits its fixed subset, in the declared arithmetic order.
end
```

This does not allocate an iterator array. A `while` loop is also appropriate for
an explicit reduction stride, an early-terminating scan or a genuinely adaptive
algorithm. Both forms can lower to equivalent machine loops. GPU scheduling
depends on dependencies, work distribution and divergent execution, not the
source keyword. Shape bounds are usually known at launch even when they are
runtime values rather than compile-time constants.

Distinguish branches before changing them:

- **Preparation decisions** run on the CPU before graph capture. Type dispatch
  expresses policy selection; scalar dimensions, aliasing and calibration values
  still need ordinary predicates.
- **Specialization conditions**, such as a `Val` workgroup size or `Nothing`
  versus a concrete weights array, can disappear during compilation. Inspect
  generated code before treating them as GPU divergence.
- **Data conditions**, including pixel thresholds, finiteness and previous-frame
  state, implement scientific behavior. A predicate is not permission to drop
  the check or evaluate an invalid array access speculatively.

Every participating lane must reach workgroup collectives and barriers, including
empty lanes and rejected frames. Guard arithmetic and publication while keeping
those collectives convergent. Preserve failure priority, invalid-zero behavior,
held outputs, reset and exactly one guarded local state adoption.

Test portable kernels on KA's CPU backend as well as the target GPU. Bind an
index directly, for example `lane = @index(Local, Linear)`, and convert it where
Int-width arithmetic is required. The installed KA CPU lowering reconstructs
these bindings between barrier segments. Use `@uniform` only for values that
are truly shared by the group; use named `@private` storage for per-lane values
that must survive a barrier. The CPU checks caught ordinary locals that worked
on CUDA but were undefined after CPU barrier splitting. These annotations need
both backend correctness checks and GPU resource inspection.

Prepare storage and fixed calibration outside replay. Use coalesced array access,
bounded per-lane work and a reduction suited to the backend. Preserve finite
checks and inspect inference, generated code and warmed allocations. Qualify
numerical outputs and retained state before interpreting component or complete
chain timings.
