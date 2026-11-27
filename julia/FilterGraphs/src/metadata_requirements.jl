"""
    required_input_sample_metadata(graph)

Resolve mandatory Algorithm sample facts through declared metadata sources to
public Graph inputs. Facts use transport-neutral SampleMetadata field names.
Captured owners retain this cold contract from their checked source Graph.
"""
function required_input_sample_metadata end

function _trace_required_fact!(facts,graph,::ExternalInput{Index},fact) where {Index}
    fact in facts[Index] || push!(facts[Index],fact)
    return nothing
end
function _trace_required_fact!(facts,graph,::NodeOutput{Node,Port},fact) where {Node,Port}
    source=graph.nodes[Node].output_metadata_sources[Port]
    iszero(source) && throw(ArgumentError("required sample fact $fact has no declared external metadata source"))
    return _trace_required_fact!(facts,graph,graph.input_routes[Node][source],fact)
end
function required_input_sample_metadata(graph::PreparedGraph)
    facts=[Symbol[] for _ in keys(graph.input_formats)]
    for (index,node) in enumerate(graph.nodes), (port,name) in enumerate(node.input_names)
        for fact in FilterGraphAlgorithms.required_sample_metadata(node.prepared,name)
            _trace_required_fact!(facts,graph,graph.input_routes[index][port],fact)
        end
    end
    return NamedTuple{keys(graph.input_formats)}(Tuple(Tuple(value) for value in facts))
end
function required_input_sample_metadata(graph::PreparedComposition)
    all_facts=required_input_sample_metadata(graph.graph)
    return NamedTuple{keys(graph.input_formats)}(map(name -> getproperty(all_facts,name),keys(graph.input_formats)))
end
required_input_sample_metadata(graph::PreparedCapturedGraph)=graph.required_sample_metadata


required_input_sample_metadata(owner::Union{PreparedCpuProgressiveGraph,PreparedGpuProgressiveGraph,
    PreparedGpuReadoutGraph,PreparedCapturedPipeline,PreparedLatestHold,PreparedSynchronizedInputs})=
    required_input_sample_metadata(owner.graph)

output_metadata_source_inputs(graph::PreparedCapturedGraph)=graph.output_metadata_sources
output_metadata_source_inputs(owner::Union{PreparedCpuProgressiveGraph,PreparedGpuProgressiveGraph,
    PreparedGpuReadoutGraph,PreparedCapturedPipeline,PreparedLatestHold,PreparedSynchronizedInputs})=
    output_metadata_source_inputs(owner.graph)
