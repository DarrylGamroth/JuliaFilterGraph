# Train and run a synthetic ML LLOWFS model

The [example graph](graph.conf) uses a two-mode linear model on a 2 × 3
detector region. The model is supplied by the Graph. `MlLlowfsF32` derives
its input, parameter, and output extents from that model.

## Run the example

From the repository root, prepare the example environment and train the
model entirely on synthetic data.

```sh
julia --startup-file=no --project=examples/ml-llowfs -e '
    using Pkg
    Pkg.Registry.add("General")
    Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/DarrylGamroth/PackageRegistry.git"))
    Pkg.instantiate()
'
julia --startup-file=no --project=examples/ml-llowfs examples/ml-llowfs/train.jl
```

The example fits the linear model by ridge regression, measures error on a
separate generated dataset, then checks the prepared Algorithm against the
trained SimpleChains chain. Pass a path as the final argument to save the model
parameters as TOML. Keep that file with the graph that defines its topology and
schemas.

To supply saved parameters through the Julia Graph API, use the parameter
port and verify the saved schema before replacing the active parameters.

```julia
using FilterGraphAlgorithms, FilterGraphs, TOML

graph = prepare_graph("examples/ml-llowfs/graph.conf"; algorithms=algorithms())
port = Symbol("model-parameters")
saved = TOML.parsefile("parameters.toml")
format = graph.parameter_formats[port]
saved["parameters_schema"] == format.schema || error("parameter schema mismatch")
replace_parameters!(graph, port => Float32.(saved["parameters"]))
```

A model in a graph configuration declares `image_size` in rows and columns
and an ordered `layers` array. The types are `convolution` with a two-dimensional
`window`, output `channels`, and `activation`, `max-pool` with a two-dimensional
`window`, `flatten` for the image and channel axes, and `dense` with `outputs`
and `activation`. Activations are `identity`, `tanh`, and `relu`. A Julia caller
can instead construct [`MlLlowfsModel`](../../julia/FilterGraphAlgorithms/src/algorithms/wavefront_sensor/llowfs_model.jl)
from a SimpleChains chain with a declared `(rows, columns, 1)` input.
Dimensions and schemas are fixed during Graph preparation. Replacing weights
for a live device capture requires preparing a new capture.

## Design a model for instrument data

1. Pair each calibrated detector region with the known modal perturbation
   for that exposure. Record the mode order, sign, units, and acquisition
   identity. Use bounded probes or a validated simulator to span the
   expected operating range. Hold out whole acquisition runs so neighboring
   frames from one run cannot appear in both training and validation.
2. Apply the deployed crop and `normalize_llowfs!` before training. It uses
   unit flux, zero mean, and corrected sample deviation. The model receives
   spatial structure after that normalization. If the target depends on
   absolute flux, qualify a preprocessing change before fitting the model.
3. Start with a linear map from pixels to modes as in the example. Compare
   held-out error per mode with the existing reconstructor. Try a small
   hidden dense layer when the residuals show repeatable nonlinear error.
   Add convolutions if local image features help on held-out runs and the
   resulting inference fits the latency budget. Use an identity output for
   signed modal coefficients. For a nonlinear SimpleChains model, use
   `SquaredLoss` and fit its parameters with the documented optimizer API.
4. Deploy the trained parameters with the exact topology, crop,
   normalization, and output basis used in training. Give incompatible
   basis or parameter-order changes distinct schema IDs. Verify inference on
   held-out frames with the deployment backend before closed-loop operation.

[SimpleChains training API](https://pumasai.github.io/SimpleChains.jl/)
