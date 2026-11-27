module ControllerFixture
using FilterGraphAlgorithms, FilterGraphs, LinearAlgebra

function graph(n, pole, aw; slew=false, flat=0f0, offset=0f0)
    nodes = (
        (name=:controller, label="closed-loop-correction-f32",
            config=(state_length=n, hidden_mode_count=1),
            props=(gain=1f0, pole=pole, var"anti-windup-gain"=aw, var"hidden-mode-gain"=0f0)),
        (name=:cv, label="controller-to-vdm-f32", config=(controller_command_count=n, active_vdm_count=n)),
        (name=:vp, label="vdm-to-pdm-f32", config=(active_vdm_count=n, full_vdm_count=n, physical_actuator_count=n)),
        (name=:flat, label="pdm-system-flat-f32", config=(actuator_count=n,)),
        (name=:offset, label="pdm-engineering-offset-f32", config=(actuator_count=n,)),
        (name=:command, label="pdm-command-f32",
            config=(actuator_count=n, command_limits=(-1f0,1f0), slew_limit=slew ? 0.25f0 : nothing,
                initial_current=nothing, quantization_origin=nothing, quantization_step=nothing, dither_seed=nothing)),
        (name=:pv, label="pdm-feedback-to-vdm-f32", config=(active_vdm_count=n, full_vdm_count=n, physical_actuator_count=n)),
        (name=:vc, label="vdm-feedback-to-controller-f32", config=(active_vdm_count=n, controller_command_count=n)))
    links = (
        (output="controller:correction", input="cv:controller-command"),
        (output="cv:vdm-command", input="vp:vdm-command"),
        (output="vp:requested-pdm-command", input="flat:requested"),
        (output="flat:requested-with-system-flat", input="offset:requested"),
        (output="offset:requested-with-offset", input="command:requested"),
        (output="command:constraint-feedback", input="pv:pdm-constraint-feedback"),
        (output="pv:vdm-constraint-feedback", input="vc:vdm-constraint-feedback"))
    inputs = (error="controller:residual-error", prior="controller:constraint-feedback",
        hidden="controller:shape-to-hidden", expansion="controller:hidden-to-shape",
        cv="cv:controller-to-vdm", active="vp:active-to-full", vp="vp:vdm-to-pdm",
        flat="flat:system-flat", offset="offset:engineering-offset",
        full="pv:full-to-active", pv="pv:pdm-to-vdm", vc="vc:vdm-to-controller")
    outputs = (correction="controller:correction", state="controller:controller-state",
        demanded="command:demanded", feedback="vc:controller-constraint-feedback")
    prepared = prepare_graph((;nodes, links, inputs, outputs); algorithms=algorithms())
    identity = Matrix{Float32}(I,n,n)
    replace_parameters!(prepared, :hidden=>zeros(Float32,1,n), :expansion=>zeros(Float32,n,1),
        :cv=>identity, :active=>identity, :vp=>identity, :flat=>fill(flat,n),
        :offset=>fill(offset,n), :full=>identity, :pv=>identity, :vc=>identity)
    return prepared
end

buffers(n) = (correction=zeros(Float32,n), state=zeros(Float32,n),
    demanded=zeros(Float32,n), feedback=zeros(Float32,n))
const DRIVE = vcat(fill(2f0,12),zeros(Float32,16),fill(-2f0,12),zeros(Float32,16),
    Float32[2,0,0,-2,0,0,0,0])
function drive!(error, frame)
    for i in eachindex(error)
        error[i] = (isodd(i) ? 1f0 : -1f0) * DRIVE[frame]
    end
end

end
