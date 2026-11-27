# FilterGraphPipeWire

Expose a prepared FilterGraphs graph as one PipeWire ndarray node, or launch it
as a standalone Julia process. The package provides both the library and CLI;
importing it only defines their methods.

## Library preparation

```julia
using FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire

graph = FilterGraphPipeWire.prepare_service_graph("graph.conf"; algorithms=FilterGraphAlgorithms.algorithms())
try
    formats = FilterGraphs.input_formats(graph)
    # Construct a PipeWireNode when the application is ready to publish it.
finally
    FilterGraphPipeWire.close_execution(graph)
end
```

`prepare_service_graph` and `close_execution` are public qualified APIs; use
`FilterGraphPipeWire.prepare_service_graph` and
`FilterGraphPipeWire.close_execution` if they are not explicitly imported.
Preparation accepts owned initial Parameter arrays or a callback receiving
parameter formats. Row-block execution and GPU capture are explicit keyword
selections. Validation and parameter adoption precede capture and publication.

## Standalone process

```sh
julia --startup-file=no --threads=2,0 --project=/path/to/environment \
  -m FilterGraphPipeWire --graph graph.conf --check
```

The installed command remains `filtergraph-island`. Remove `--check` and supply
site-owned connection options to publish the node. A site-managed core owns the
surrounding device nodes and links. PipeWire schedules the exposed node;
FilterGraphs executes its internal graph.

Scientific algorithms remain in FilterGraphAlgorithms. The native SDK requires
PipeWireAO 0.9.7 / PipeWireAO_jll 4.0.2+0 or newer within the supported majors.
Owner-loop command endpoints additionally require a native SDK exporting
`pw_ndarray_filter_get_loop`.
