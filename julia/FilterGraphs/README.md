# FilterGraphs

`FilterGraphs` prepares and processes Julia-native Filter Graph Ndarray
DAGs. Scientists define Algorithms in ordinary Julia with `@algorithm`; a
PipeWire `filter.graph` SPA-JSON configuration instantiates them as Nodes and
connects their typed Ports.

```julia
using FilterGraphAlgorithms
using FilterGraphs

import FilterGraphAlgorithms: process!

graph = prepare_graph(
    "filter-chain.conf";
    algorithms=(LeakyIntegratorF32, MyAlgorithms.algorithms()...),
)

inputs = (input=input_buffer,)
outputs = (output=output_buffer,)
result = process!(outputs, graph, inputs)
```

In-memory applications may pass the graph dictionary to
`prepare_graph(document; algorithms)`. The dictionary uses the same Node,
Link and external Port fields as the parsed configuration file.

For a native captured graph with a device-resident consumer, call
`process_device!(consume, captured, inputs[, input_metadata])`. After
completion, a successful frame gives the consumer read-only access to the
selected device arrays for the duration of the callback. The consumer
finishes all work on those arrays before returning. The captured owner
stays busy during that borrow. Expected rejections suppress the callback.

For the closed-loop physical-mirror controller chain, use
`prepare_controller_graph(graph, publisher)` on a prepared CPU Graph or CUDA
captured Graph. Its default publishes the validated limited command at command
readiness, through `publisher(command, metadata)`. Ordinary
`process!(outputs, controller, inputs, metadata)` still completes feedback before
returning or accepting another frame. Generic Graph processing is unchanged.

For a Graph fed by detector row or region blocks, choose the CPU execution
layout at the Graph level. The Algorithm `config` objects continue to describe
scientific preparation; `execution.cpu` belongs to deployment:

```text
filter.graph = {
    execution = {
        cpu = {
            row_workers = 4
            worker_thread_ids = [3 4 5 6]
            matrix_layout = sharded
        }
    }
    nodes = [ ... ]
    links = [ ... ]
    inputs = [ ... ]
    outputs = [ ... ]
}
```

Use `prepare_cpu_progressive_graph(path; algorithms)` to apply that layout.
Its worker adapter currently targets `incremental-dense-reconstructor-f32`;
upstream row or region WFS Nodes process newly ready measurements in their
ordinary Graph order. `row_workers = 0` selects the serial progressive path
and needs no extra Julia threads. Explicit preparation keywords override the
file. If both execution settings and preparation keywords are omitted, the
existing default of `max(1, Threads.nthreads() - 1)` workers applies. Worker
IDs are Julia default-pool thread IDs, not physical CPU IDs;
deployment owns physical affinity and reserves a coordinator thread. `sharded`
packs each worker's matrix rows into a private column-major matrix at Graph
preparation and Parameter replacement; `shared` retains the single prepared
matrix. The executor keeps the scientific column order in either layout.
`prepare_graph` rejects a configuration requesting workers so a deployment
cannot silently ignore the layout.

Run the pixel-pyramid example with an environment containing the packages.

```sh
julia --project=/path/to/environment --startup-file=no examples/pixel-pyramid-row/run.jl
```

The region-block examples are
[`pyramid/readout.conf`](../../examples/pyramid/readout.conf)
and
[`shack-hartmann/readout.conf`](../../examples/shack-hartmann/readout.conf).
They publish newly complete WFS measurements to the incremental reconstructor
on each detector block while retaining one terminal reconstructed output.
Captured backends use fixed storage and publish outputs atomically. Borrowed
external measurement inputs use a bounded copy.

The opt-in GPU progressive experiment offloads
only incremental reconstruction to CUDA while retaining the checked CPU WFS and
controller. `prepare_gpu_readout_graph` additionally prepares GPU pyramid or
Shack–Hartmann WFS, incremental reconstruction and a captured controller,
projection and command suffix. Intermediate data stays on-device, with one
terminal wait and final output transfer. Both profiles use bounded staging;
neither currently selects a PipeWire GPU row mode or direct-device transport.

An acquisition path can supply standard FGN sample identity without exposing
PipeWire types to an Algorithm.

```julia
metadata = (
    input=SampleMetadata(41, 0; timestamp_nanoseconds=100, terminal=true),
)
result = process!(outputs, graph, inputs, metadata)
published_metadata = FilterGraphs.output_metadata(graph)
```

The prepared Graph owns fixed routes, internal buffers, and each Node's Plan and
Workspace. Each Algorithm/Node owns its candidate-state commit boundary.
Repeated processing does not parse configuration or resolve Algorithm labels.
Expected Algorithm failure is returned as its declared enum and leaves caller
output and committed state unchanged when that Algorithm rejects locally. A
later Node rejection does not roll back earlier Nodes. The Graph still
suppresses external output for the failed call. Otherwise, `result` is a
Boolean `NamedTuple` keyed by
external output Port: a `true` value identifies a complete caller buffer, and
a `false` value leaves that buffer unchanged and unpublished.

The first slice supports complete and conditional column-major Frame Data
Ports, reduced exact Port-rate ratios, optional FGN-compatible sample metadata,
and metadata propagation in one serial execution domain. A required consumer
of an absent Port is skipped while independent ready Nodes continue. Parameter
Ports named in `filter.graph.inputs` are excluded from per-sample routing and
are updated as one checked transaction:

```julia
replace_parameters!(
    graph,
    "flat" => flat_calibration,
    "background" => background_calibration,
)
```

External `inputs` and `outputs` entries can be endpoint strings or objects with
`name` and `endpoint` fields. A string retains the Algorithm Port name. An
object assigns a unique external name when several Nodes use the same Port
name:

```text
inputs = [
    { name = woofer-input endpoint = "woofer:ordered-input" }
    { name = tweeter-input endpoint = "tweeter:ordered-input" }
    { name = woofer-matrix endpoint = "woofer:reconstructor" }
    { name = tweeter-matrix endpoint = "tweeter:reconstructor" }
]
outputs = [
    { name = woofer endpoint = "woofer:output" }
    { name = tweeter endpoint = "tweeter:output" }
]
```

The external name keys the input/output `NamedTuple`, reported formats, metadata,
and Parameter replacement calls such as `replace_parameters!(graph,
"woofer-matrix" => matrix)`. It preserves the selected Port's shape, schema,
and numerical meaning. Links continue to use endpoint strings and Algorithm
Port names. External names must be unique within inputs (including Parameters)
and within outputs.

Every replacement Plan is prepared before active state changes. Parameter and
Property replacement is a separate control-plane transaction. This control call
on a bare Graph is serialized with `process!`. `FilterGraphPipeWire` owns
the concurrent deployment handoff: it prepares away from the data loop and
adopts one pending typed transaction at the next Graph callback boundary.

The row-block Algorithms `PixelCalibrationRowU16F32` and
`PyramidPixelRowF32` consume the established FGN metadata and block conventions.
The repository Graph fixture links calibrated rows directly into pyramid
measurement and checks numerical results, metadata propagation, multi-Node
Parameter Port replacement, recovery without mixed generations, and warmed
allocation. `FilterGraphPipeWire` exercises the fixture through its borrowed
buffer callback.

Two bounded temporal owners cover the concrete controller arrivals observed so
far. `prepare_latest_hold` retains one slower input for an admitted integer
number of faster primary cycles. `prepare_synchronized_inputs` instead owns one
slot per equal-rate input and processes only after every Port supplies the same
sample sequence; it never reuses an older contributor. Set `match=:acquisition`
to require identical acquisition domain, generation, sequence, and present
publication timestamps (Header PTS). A newer generation resets Graph state
once and permits a sequence restart; stale generations cannot reopen it. A
producer-domain change closes admission permanently and requires reconstruction
of the owner. `reset!` preserves that domain pin, generation, and sequence watermark.
`incoherent_arrivals` and `domain_changes` expose rejected metadata and domain
replacement. Both owners expose explicit
expiry and counters without adding scheduling concepts to Algorithm
declarations. The synchronized owner is checked at the Julia Graph boundary;
its connected PipeWire proof drives three independently arriving inputs, checks sequence
expiry, duplicate and corruption handling, and verifies numerical output and
Header lineage across recovery.

### Independent graph composition

`filter.composition` describes a **graph composition**: independent input
**branches** followed by one **shared graph**. `PreparedComposition` is its
serialized CPU execution owner. A branch is an ordinary complete-frame graph
with one camera input, one private controller-feedback input, and one vector
correction output. The shared graph combines retained branch corrections and
produces the common command plus separate controller-feedback vectors. This
execution boundary is distinct from deployment calibration or offset composition.

A descriptor references ordinary SPA-JSON graphs relative to its own directory:

```json
{"filter.composition":{"branches":[
  {"name":"a","graph":"a.conf","input":"a-image",
   "correction":"a-correction","feedback":"a-feedback",
   "shared-input":"a-command","shared-feedback":"a-returned"}
],"shared":"shared.conf"}}
```

Names of nodes, parameter ports, and external outputs are globally unique;
ordinary `node:property` names remain intact. Camera inputs and branch telemetry
are public. Delayed feedback inputs and shared feedback outputs are private.
All declared graph files remain source inputs; exporters must copy, resolve and
hash those files alongside the descriptor.

An optional branch `rate: [numerator, denominator]` declares that camera's native
port rate in Hz. It must contain positive UInt32 integers. The independent
arrival path imposes no rate-ratio constraint. Without this declaration the
owner's base rate supplies the camera port rate. The shared command port retains
the owner's nominal rate; command publications follow accepted input events and
do not define a new periodic clock or alter the camera exposure clock.

`process!(outputs, composition, arrivals, metadata)` accepts a partial NamedTuple
of camera arrivals with matching metadata keys. A missing camera contributes
its retained correction, initially zero. Each arrived branch executes once in
descriptor order, then the shared graph executes once. SCC incomplete-pair
priming suppresses only that branch. Other rejection or exceptions latch failure
until reset, because earlier branch state may already have advanced.

After publication, the separate feedback vectors replace their branch slots.
Each branch consumes its latest slot on its next update, followed immediately
by a new shared publication. Feedback is never applied merely because another
camera arrived. Open controls retain their Algorithms' ordinary hold semantics.
Global `reset!` clears all graphs, retained corrections and feedback together.

Combined command metadata has its own UUID domain, reset generation and
publication sequence in an independent acquisition identity record when that
capability is declared and negotiated. It carries no invented exposure interval
or camera trigger annotation. A derived PipeWire publication can carry its own
absolute production event, REALTIME by default or explicit TAI with a validated
host offset. Neither the label nor offset establishes clock accuracy or physical
synchronization.
`composition_observation` exposes reset generation,
publication sequence, triggering branches and each retained camera contributor.
This CPU data-plane contract provides no physical-device or deadline qualification.
Header publication counters remain independent of retained camera acquisition
identities. Every data Port requires Header; other independent records are
explicit cold capabilities selected by its role.

The public naming authority for this boundary is this section. `filter.composition`
and `PreparedComposition` name the graph execution composition; `branch` and
`shared` retain their graph meanings. `input`, `correction`, and `feedback` refer
to existing named Algorithm ports. `domain`, `generation`, and `sequence` preserve
the adopted `SampleAcquisitionIdentity` vocabulary at the native boundary.
No instrument controller, deployment calibration, or wire protocol is defined here.

Warmed composition processing uses prepared stage and port tuples. Native
callbacks use one fixed tuple of camera wrappers plus a presence mask, so new
arrival combinations do not create new tuple types. Trigger snapshots are
constructed only by `composition_observation`.

Historical native 2.x / BufferProperties allocation evidence on Julia 1.13.3
with one default thread used two one-dimensional branches and preallocated arrays, metadata, and native buffers:

| Warmed call | Before (bytes) | After (bytes) | After median (ns) |
| --- | ---: | ---: | ---: |
| Checked FG, one camera | 34,080 | 0 | 831 |
| Checked FG, both cameras | 43,616 | 0 | 1,092 |
| Actual FGPW callback, one camera | 39,600 | 0 | 1,142 |
| Actual FGPW callback, both cameras | 49,504 | 0 | 1,423 |

Measurements use a function barrier for `@allocated` and BenchmarkTools with
100 samples and one evaluation per sample. Caller setup, compilation,
exceptions, and diagnostic snapshots are excluded. Native timing inputs carry
the former Header and BufferProperties layout. These timings have not been
remeasured for the native 3.x independent-record cohort. Current allocation
regressions retain complete physical identities independently of publication
counters and cover each camera alone and simultaneous arrivals. The historical
synthetic timings provide no physical-device or deadline qualification.
