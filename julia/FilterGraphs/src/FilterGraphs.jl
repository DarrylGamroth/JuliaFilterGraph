"""
A prepared Julia-native executor for PipeWire filter-graph ndarray DAGs.

Scientists declare Algorithms in ordinary Julia. This package owns configured
Nodes, Links, fixed routes, and Graph-owned buffers. Each Algorithm owns its
candidate-state boundary. Execution owners are selected explicitly; the engine
does not automatically place kernels or compile arbitrary Julia Graphs.
"""
module FilterGraphs

import FilterGraphAlgorithms
import ThreadPinning
import UUIDs

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

include("configuration.jl")
include("graph.jl")
include("cpu_loans.jl")
include("cpu_progressive_graph.jl")
include("captured_graph.jl")
include("captured_feedback.jl")
include("captured_aliases.jl")
include("captured_validation.jl")
include("captured_pipeline.jl")
include("gpu_progressive_graph.jl")
include("gpu_readout_graph.jl")
include("latest_hold.jl")
include("synchronized_inputs.jl")
include("parameters.jl")
include("formats.jl")
include("rejection.jl")
include("properties.jl")
include("construction.jl")
include("composition.jl")
include("metadata_requirements.jl")
include("controller_graph.jl")
include("version.jl")
export required_input_sample_metadata

public PreparedComposition, prepare_composition, composition_observation

export GraphConfigurationError
export CapturedGraphBusy
export CapturedGraphFault
export CapturedGraphFaulted
export CapturedGraphReady
export CapturedGraphRunning
export CapturedGraphState
export CapturedWorkgroupProfile
export PreparedCapturedGraph
export PreparedCapturedPipeline
export PreparedGraph
export PreparedCpuLoans, prepare_cpu_loans, process_cpu_loan!
public prepare_cpu_loan_node
export PreparedControllerGraph, prepare_controller_graph, ControllerFault
export captured_graph_fault
export captured_graph_state
export captured_workgroup_profile
export prepare_captured_graph
export process_device!
export prepare_captured_pipeline
export submit_captured_frame!
export advance_captured_pipeline!
export take_captured_result!
export captured_pipeline_counters
export prepare_graph
export prepare_cpu_progressive_graph
export PreparedCpuProgressiveGraph
export PreparedGpuProgressiveGraph
export prepare_gpu_progressive_graph
export PreparedGpuReadoutGraph
export prepare_gpu_readout_graph
export replace_parameters!
export replace_properties!

public input_formats
public output_formats
public parameter_formats
public GraphRejection
public rejection_context
public output_metadata
public output_metadata_source_inputs
public cpu_progressive_counters
public gpu_progressive_counters
public PreparedLatestHold
public prepare_latest_hold
public accept_held!
public latest_hold_counters
public PreparedSynchronizedInputs
public prepare_synchronized_inputs
public expire!
public synchronized_input_counters
public process!
public reset!
public software_version

end
