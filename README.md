# FilterGraphs

FilterGraphs prepares scientific Algorithms as Nodes in a typed, configured
Graph and executes the Graph on the CPU or through supported CUDA and AMDGPU
capture adapters. FilterGraphPipeWire exposes a prepared Graph as one PipeWire
ndarray Node. Scientists supply ordinary Julia numerical code. The engine owns
routing, buffers, execution, and transport boundaries.

| Package | Responsibility |
| --- | --- |
| [FilterGraphAlgorithms](julia/FilterGraphAlgorithms) | Algorithm declarations, scientific implementations, Plans and Workspaces. |
| [FilterGraphs](julia/FilterGraphs) | Graph preparation, routing, CPU execution, joins, and captured execution. |
| [FilterGraphPipeWire](julia/FilterGraphPipeWire) | PipeWire integration, graph preparation, and the standalone CLI. |

Use Julia 1.13 or later. The packages use separate projects inside this
repository. From the repository root, run the CPU suites in their package test
environments.

```sh
for package in FilterGraphAlgorithms FilterGraphs FilterGraphPipeWire; do
    project="julia/$package/test"
    julia --startup-file=no --project="$project" .github/scripts/prepare_cpu_tests.jl
    julia --startup-file=no --threads=3,0 --project="$project" "$project/runtests.jl"
done
```

The environment script uses the General registry and DarrylGamroth's
[PackageRegistry](https://github.com/DarrylGamroth/PackageRegistry). It selects
the three local package sources and the public AdaptiveOpticsCalibration release.
FilterGraphPipeWire uses [PipeWireAO.jl](https://github.com/DarrylGamroth/PipeWireAO.jl)
for the native host interface.

The [ML LLOWFS example](examples/ml-llowfs/README.md) fits a synthetic model
and runs it through a prepared Algorithm.

Package tests keep copies of shared numerical and graph fixtures so they can
run from installed package trees. When editing a shared fixture, regenerate
the copies and update their hash receipts.

```sh
julia --startup-file=no test/update_snapshots.jl <source-path>
```

See [LICENSE](LICENSE). The same license accompanies each package directory.
Each Julia dependency is distributed under its own license.
