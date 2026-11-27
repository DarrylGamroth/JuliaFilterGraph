module ModalCommandAllocationExample

import FilterGraphAlgorithms as FGA
import FilterGraphs as JFG

const MODES = 5
const RESIDUAL_SCHEMA = "pipewireao/controller-residual-error-f32/1"
# Synthetic normalized commands: q is in VDM modal units; TT, woofer and
# tweeter outputs each use their own normalized device-command units.
const BASIS = Float32[1 .1 0 .2 0; 0 1 .2 0 .1; .1 0 1 0 .2; 0 .2 0 1 .1; .2 0 .1 0 1]
const TT = Float32[1 .2 .1 0 .15; -.1 1 .2 .1 0]
const WOOFER = Float32[.5 .1 .2 .3 0; 0 .4 .1 0 .2; .2 0 .3 .1 .4]
const TWEETER = Float32[.1 .2 0 .5 .1; .2 0 .3 .1 .6; 0 .3 .2 .4 .1; .4 .1 .2 0 .3]
const TT_TO_WOOFER = Float32[.4 .1; -.2 .5; .3 -.1]
const GAIN = -0.3f0
const POLE = 0.7f0

row_major(matrix) = vec(permutedims(matrix))
identity_matrix(n) = Float32[i == j for i in 1:n, j in 1:n]

"""
    layout(; low_modes_last=false)

Place the LO modes (tip, tilt, defocus) before or after the HO modes.
"""
function layout(; low_modes_last=false)
    permutation = low_modes_last ? [4, 5, 1, 2, 3] : collect(1:5)
    return (; permutation)
end

# Stable single-pole biquads, with distinct per-mode bandwidths. Coefficient
# order is b0,b1,b2,a1,a2, with denominator 1+a1*z^-1+a2*z^-2.
function coefficients(indices)
    result = Float32[]
    for mode in indices
        alpha = 0.15f0 + 0.1f0 * mode
        append!(result, (alpha, 0f0, 0f0, alpha - 1f0, 0f0))
    end
    return result
end

function bank_node(name, indices)
    return """
    { type = ndarray name = $name label = biquad-cascade-f32
      config = { extent = $(length(indices)) section_count = 1 per_mode = true
        initial_coefficients = [$(join(coefficients(indices), " "))]
        input_schema = $RESIDUAL_SCHEMA output_schema = $RESIDUAL_SCHEMA } }
    """
end

function graph_configuration(selection)
    # Independent per-mode filtering is equivalent to splitting, filtering and
    # embedding the LO/HO branches, without intermediate vectors or a join.
    nodes = [bank_node("bank", selection.permutation), """
      { type = ndarray name = law label = leaky-integrator-f32
        config = { extent = 5 initial_state = 0.0 input_schema = $RESIDUAL_SCHEMA
          output_schema = pipewireao/controller-command-f32/1 }
        props = { gain = $GAIN pole = $POLE } }
    """]
    return "filter.graph = { nodes = [" * join(nodes, "\n") * """
        ] links = [{ output = "bank:output" input = "law:input" }]
        inputs = ["bank:input"] outputs = ["law:output"] }
        """
end

function prepare_dense(matrix; input_schema, output_schema)
    return FGA.prepare_algorithm(FGA.DenseVectorReconstructorF32,
        FGA.DenseVectorReconstructorF32Configuration(input_count=size(matrix,2),
            output_count=size(matrix,1), initial_reconstructor=row_major(matrix),
            input_schema=input_schema, output_schema=output_schema,
            reconstructor_schema="org.example.modal.projection/1"))
end

function prepare_device(matrix)
    n = size(matrix,2)
    prepared = FGA.prepare_algorithm(FGA.VdmToPdmF32,
        FGA.VdmToPdmF32Configuration(active_vdm_count=n, full_vdm_count=n,
            physical_actuator_count=size(matrix,1)))
    FGA.replace_parameter!(prepared, "active-to-full", identity_matrix(n))
    FGA.replace_parameter!(prepared, "vdm-to-pdm", matrix)
    return prepared
end

"""
    prepare_example(; low_modes_last=false)

Prepare all graph state, projection plans and owned output buffers once.
"""
function prepare_example(; low_modes_last=false)
    selection = layout(; low_modes_last)
    graph = mktemp() do path, io
        write(io, graph_configuration(selection)); close(io)
        JFG.prepare_graph(path; algorithms=FGA.algorithms())
    end
    basis = FGA.prepare_algorithm(FGA.ControllerToVdmF32,
        FGA.ControllerToVdmF32Configuration(controller_command_count=MODES, active_vdm_count=MODES))
    FGA.replace_parameter!(basis, "controller-to-vdm", BASIS[:,selection.permutation])
    buffers = (controller=zeros(Float32,5), vdm=zeros(Float32,5),
        requested_tt=zeros(Float32,2), commanded_tt=zeros(Float32,2),
        requested_woofer=zeros(Float32,3), tt_woofer=zeros(Float32,3),
        woofer=zeros(Float32,3), tweeter=zeros(Float32,4))
    plans = (; basis, tt=prepare_dense(TT; input_schema=FGA.VDM_COMMAND_V1,
            output_schema="org.example.modal.tt-command/1"),
        tt_woofer=prepare_dense(TT_TO_WOOFER;
            input_schema="org.example.modal.tt-command/1",
            output_schema=FGA.REQUESTED_PDM_COMMAND_V1),
        woofer=prepare_device(WOOFER), tweeter=prepare_device(TWEETER))
    return (; graph, selection, plans, buffers, faulted=Ref(false))
end

# A conditioning callback writes the same-tick software commanded TT command.
# It must return nothing on success or a failure value. Device acknowledgement,
# output scheduling and multi-device atomicity belong to the caller.
copy_tt!(commanded, requested) = (copyto!(commanded, requested); nothing)

"""
    allocate_commands!(prepared, condition_tt! = copy_tt!)

Allocate one VDM command. A failure stops later stages; prior writes persist.
"""
function allocate_commands!(prepared, condition_tt! = copy_tt!)
    p, b = prepared.plans, prepared.buffers
    status = FGA.process!(b.vdm, p.basis, b.controller)
    status === nothing || return status
    status = FGA.process!(b.requested_tt, p.tt, b.vdm)
    status === nothing || return status
    status = condition_tt!(b.commanded_tt, b.requested_tt)
    status === nothing || return status
    status = FGA.process!(b.tt_woofer, p.tt_woofer, b.commanded_tt)
    status === nothing || return status
    status = FGA.process!(b.requested_woofer, p.woofer, b.vdm)
    status === nothing || return status
    # Validate the native subtraction before publishing any woofer component.
    for i in eachindex(b.requested_woofer, b.tt_woofer)
        isfinite(b.requested_woofer[i] - b.tt_woofer[i]) || return :nonfinite_woofer
    end
    b.woofer .= b.requested_woofer .- b.tt_woofer
    return FGA.process!(b.tweeter, p.tweeter, b.vdm)
end

"""
    process_frame!(prepared, residual, condition_tt! = copy_tt!)

Process a residual and allocate commands; consume owned buffers on success.
A failed call requires `reset!` before reuse: the temporal filter and controller
may already have committed state before command allocation fails.
"""
function process_frame!(prepared, residual, condition_tt! = copy_tt!)
    prepared.faulted[] && error("reset! is required after a failed controller frame")
    prepared.faulted[] = true
    available = JFG.process!((output=prepared.buffers.controller,), prepared.graph,
        (input=residual,))
    available isa NamedTuple || return available
    available.output || return available
    status = allocate_commands!(prepared, condition_tt!)
    status === nothing && (prepared.faulted[] = false)
    return status
end

"""
    reset!(prepared)

Reset temporal and controller state, clear command buffers, and allow reuse.
"""
function reset!(prepared)
    JFG.reset!(prepared.graph)
    foreach(buffer -> fill!(buffer, 0f0), values(prepared.buffers))
    prepared.faulted[] = false
    return prepared
end

"""
    run_example(; low_modes_last=false)

Process two synthetic samples and return owned copies of the allocated commands.
"""
function run_example(; low_modes_last=false)
    prepared = prepare_example(; low_modes_last)
    samples = (Float32[1,-.5,.25,.75,-.2], Float32[.2,.1,-.4,.3,.5])
    return map(samples) do sample
        status = process_frame!(prepared, sample[prepared.selection.permutation])
        status === nothing || error("command allocation failed: $status")
        b = prepared.buffers
        (; controller=copy(b.controller), commanded_tt=copy(b.commanded_tt),
            woofer=copy(b.woofer), tweeter=copy(b.tweeter))
    end
end

function (@main)(args)
    foreach(println, run_example())
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .ModalCommandAllocationExample: main
end
