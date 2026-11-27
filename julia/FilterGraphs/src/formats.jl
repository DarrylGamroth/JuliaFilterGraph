"""
    input_formats(owner)

Return the ordered `NamedTuple` of logical external data-input Port formats for
a prepared Graph or execution owner. Each immutable `AlgorithmFormat` describes
the element type, layout, shape, scientific schema, and relative rate. Parameter
Ports are listed separately by [`parameter_formats`](@ref).

Captured owners report their actual external boundary: feedback-bound inputs
are excluded. Arrival-based joins report all logical inputs, including held
inputs supplied separately from the primary processing call. The result does
not specify host/device allocation or change the owner's processing interface.
"""
input_formats(graph::PreparedGraph) = graph.input_formats
input_formats(graph::PreparedCapturedGraph) = graph.input_formats
input_formats(owner::PreparedCpuProgressiveGraph) = input_formats(owner.graph)
input_formats(owner::PreparedGpuProgressiveGraph) = input_formats(owner.graph)
input_formats(owner::PreparedGpuReadoutGraph) = owner.input_formats
input_formats(owner::PreparedCapturedPipeline) = input_formats(owner.graph)
input_formats(owner::PreparedLatestHold) = input_formats(owner.graph)
input_formats(owner::PreparedSynchronizedInputs) = input_formats(owner.graph)

"""
    output_formats(owner)

Return the ordered `NamedTuple` of logical external output Port formats for a
prepared Graph or execution owner. Each immutable `AlgorithmFormat` describes
the element type, layout, shape, scientific schema, and relative rate. Captured
owners preserve the selected output names and their requested order.

Formats describe the boundary; they do not indicate whether an output is
available on a particular processing call. Inspect that call's result for
availability or rejection.
"""
output_formats(graph::PreparedGraph) = graph.output_formats
output_formats(graph::PreparedCapturedGraph) = graph.output_formats
output_formats(owner::PreparedCpuProgressiveGraph) = output_formats(owner.graph)
output_formats(owner::PreparedGpuProgressiveGraph) = output_formats(owner.graph)
output_formats(owner::PreparedGpuReadoutGraph) = owner.output_formats
output_formats(owner::PreparedCapturedPipeline) = output_formats(owner.graph)
output_formats(owner::PreparedLatestHold) = output_formats(owner.graph)
output_formats(owner::PreparedSynchronizedInputs) = output_formats(owner.graph)

"""
    parameter_formats(owner)

Return the ordered `NamedTuple` of configured external Parameter Port formats
for a prepared Graph or execution owner. Each immutable `AlgorithmFormat`
describes the element type, layout, shape, scientific schema, and relative rate;
the returned values contain no calibration arrays.

Schema inspection does not grant replacement permission. Use
[`replace_parameters!`](@ref) only on supported owners. Captured and GPU readout
owners retain their prepared calibration; prepare a new owner to change it.
"""
parameter_formats(graph::PreparedGraph) = graph.parameter_formats
parameter_formats(graph::PreparedCapturedGraph) = graph.parameter_formats
parameter_formats(owner::PreparedCpuProgressiveGraph) = parameter_formats(owner.graph)
parameter_formats(owner::PreparedGpuProgressiveGraph) = parameter_formats(owner.graph)
parameter_formats(owner::PreparedGpuReadoutGraph) = parameter_formats(owner.graph)
parameter_formats(owner::PreparedCapturedPipeline) = parameter_formats(owner.graph)
parameter_formats(owner::PreparedLatestHold) = parameter_formats(owner.graph)
parameter_formats(owner::PreparedSynchronizedInputs) = parameter_formats(owner.graph)
