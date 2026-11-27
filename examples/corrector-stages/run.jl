module CorrectorStagesExample

import FilterGraphAlgorithms as FGA
import FilterGraphs as JFG
import DSP # Preparation-time coefficient design; processing uses plain arrays.

const MODES = 3
const SAMPLE_RATE_HZ = 1000.0
const RESIDUAL_SCHEMA = FGA.CONTROLLER_RESIDUAL_ERROR_V1

identity_matrix(n) = Float32[i == j for i in 1:n, j in 1:n]
row_major(matrix) = vec(permutedims(matrix))
values_text(values) = join(values, " ")

identity_sections() = reshape(Float32[1, 0, 0, 0, 0], 5, 1)

"""
    lowpass(passband_hz, stopband_hz; period=1)

Design and check a Float32 Butterworth cascade at the example's base rate.
"""
function lowpass(passband_hz, stopband_hz; period=1)
    return FGA.design_lowpass(; sample_rate_hz=SAMPLE_RATE_HZ, passband_hz,
        stopband_hz, update_period=period, passband_loss_db=1.0,
        stopband_attenuation_db=40.0).coefficients
end

default_regions() = (1:2 => lowpass(80, 200), 3:3 => lowpass(120, 250))

"""
    stage(to_command, to_controller; coefficients=identity_sections(),
          lower=-0.3f0, upper=0.3f0, slew=1f0, period=1)

Declare one stage's calibrated maps, command limits, and update period.
"""
function stage(to_command, to_controller; coefficients=identity_sections(),
               lower=-0.3f0, upper=0.3f0, slew=1f0, period=1)
    n = size(to_command, 1)
    return (; to_command, to_controller, coefficients,
        lower=fill(Float32(lower), n), upper=fill(Float32(upper), n),
        slew=fill(Float32(slew), n), period)
end

"""
    default_stages()

Four synthetic stages, in priority order, with unequal command counts.
"""
function default_stages()
    return (
        stage(Float32[1 0 0], reshape(Float32[1, 0, 0], 3, 1);
            coefficients=lowpass(20, 100; period=4), period=4, slew=0.05f0),
        stage(Float32[1 0.2 0; 0 1 0], Float32[1 -0.2; 0 1; 0 0];
            coefficients=lowpass(50, 200; period=2), period=2, slew=0.1f0),
        stage(Float32[0 0 1], reshape(Float32[0, 0, 1], 3, 1);
            lower=-0.2f0, upper=0.2f0),
        stage(identity_matrix(MODES), identity_matrix(MODES);
            lower=-0.15f0, upper=0.15f0),
    )
end

function allocation_configuration(stages)
    isempty(stages) && throw(ArgumentError("at least one corrector stage is required"))
    stage_command_counts = [size(s.to_command, 1) for s in stages]
    section_count = maximum(size(s.coefficients, 2) for s in stages)
    coefficients = zeros(Float32, 5, section_count, sum(stage_command_counts))
    coefficients[1, :, :] .= 1f0
    offset = 0
    for (s, n) in zip(stages, stage_command_counts)
        for command in (offset + 1):(offset + n)
            coefficients[:, 1:size(s.coefficients, 2), command] .= s.coefficients
        end
        offset += n
    end
    return (; stage_command_counts, periods=[s.period for s in stages], section_count, coefficients,
        projection=reduce(vcat, (s.to_command for s in stages)),
        realization=reduce(hcat, (s.to_controller for s in stages)),
        lower=reduce(vcat, (s.lower for s in stages)),
        upper=reduce(vcat, (s.upper for s in stages)),
        slew=reduce(vcat, (s.slew for s in stages)))
end

function graph_configuration(regions, stages; gain, pole, anti_windup_gain,
                             hidden_mode_gain, hidden_mode_count)
    bank = FGA.BiquadCascadePlan(regions, MODES).coefficients
    c = allocation_configuration(stages)
    return """
    filter.graph = {
      nodes = [
        { type = ndarray name = temporal label = biquad-cascade-f32
          config = { extent = $MODES section_count = $(size(bank, 2)) per_mode = true
            initial_coefficients = [$(values_text(vec(bank)))]
            input_schema = $RESIDUAL_SCHEMA output_schema = $RESIDUAL_SCHEMA } }
        { type = ndarray name = law label = closed-loop-correction-f32
          config = { state_length = $MODES hidden_mode_count = $hidden_mode_count }
          props = { gain = $gain pole = $pole anti-windup-gain = $anti_windup_gain
                    hidden-mode-gain = $hidden_mode_gain } }
        { type = ndarray name = allocation label = corrector-allocation-f32
          config = { controller_command_count = $MODES
            stage_command_counts = [$(values_text(c.stage_command_counts))]
            update_periods = [$(values_text(c.periods))]
            command_projection = [$(values_text(row_major(c.projection)))]
            realization_projection = [$(values_text(row_major(c.realization)))]
            section_count = $(c.section_count)
            initial_coefficients = [$(values_text(vec(c.coefficients)))]
            command_limits = [[ $(values_text(c.lower)) ] [ $(values_text(c.upper)) ]]
            slew_limits = [$(values_text(c.slew))]
            command_schema = org.example.corrector-stages.commands/1 } }
      ]
      links = [
        { output = "temporal:output" input = "law:residual-error" }
        { output = "law:correction" input = "allocation:controller-command" }
      ]
      inputs = [ "temporal:input" "law:constraint-feedback"
                 "law:shape-to-hidden" "law:hidden-to-shape" ]
      outputs = [ "allocation:commands" "allocation:controller-constraint-feedback"
                  "law:correction" "law:controller-state" ]
    }
    """
end

"""
    prepare_example(; regions=default_regions(), stages=default_stages(),
                    gain=0.3f0, pole=0.9f0, anti_windup_gain=1f0,
                    hidden_mode_gain=0f0, shape_to_hidden=zeros(Float32, 1, MODES),
                    hidden_to_shape=zeros(Float32, MODES, 1))

Prepare a region filterbank, the existing constrained controller law, and an
ordered set of correctors. All maps and commands use synthetic normalized units.
Hidden-mode removal is enabled by supplying calibrated projections and a nonzero
`hidden_mode_gain`; the default zero maps disable it.
"""
function prepare_example(; regions=default_regions(), stages=default_stages(),
                         gain=0.3f0, pole=0.9f0, anti_windup_gain=1f0,
                         hidden_mode_gain=0f0,
                         shape_to_hidden=zeros(Float32, 1, MODES),
                         hidden_to_shape=zeros(Float32, MODES, 1))
    graph = mktemp() do path, io
        write(io, graph_configuration(regions, stages; gain, pole, anti_windup_gain,
            hidden_mode_gain, hidden_mode_count=size(shape_to_hidden, 1)))
        close(io)
        JFG.prepare_graph(path; algorithms=FGA.algorithms())
    end
    JFG.replace_parameters!(graph, "shape-to-hidden" => shape_to_hidden,
        "hidden-to-shape" => hidden_to_shape)
    commands = zeros(Float32, sum(size(s.to_command, 1) for s in stages))
    feedback = zeros(Float32, MODES)
    previous_feedback = similar(feedback)
    fill!(previous_feedback, 0f0)
    correction = zeros(Float32, MODES)
    controller_state = zeros(Float32, MODES)
    outputs = (; commands, var"controller-constraint-feedback"=feedback,
        correction, var"controller-state"=controller_state)
    return (; graph, outputs, previous_feedback, faulted=Ref(false))
end

"""
    process_frame!(prepared, residual)

Process one base-rate sample. Feedback from this successful frame is supplied to
the next controller update. Consume output buffers only after `nothing` is
returned. A failed call requires `reset!` before reuse: Graph state commits per
Node, so a downstream failure cannot be retried against the old feedback.
"""
function process_frame!(prepared, residual)
    prepared.faulted[] && error("reset! is required after a failed controller frame")
    prepared.faulted[] = true
    status = JFG.process!(prepared.outputs, prepared.graph,
        (input=residual, var"constraint-feedback"=prepared.previous_feedback))
    if status isa NamedTuple && all(values(status))
        copyto!(prepared.previous_feedback,
            prepared.outputs.var"controller-constraint-feedback")
        prepared.faulted[] = false
        return nothing
    end
    return status
end

"""
    reset!(prepared)

Reset filter, controller, limit, hold, cadence, and delayed feedback state.
"""
function reset!(prepared)
    JFG.reset!(prepared.graph)
    foreach(buffer -> fill!(buffer, 0f0), values(prepared.outputs))
    fill!(prepared.previous_feedback, 0f0)
    prepared.faulted[] = false
    return prepared
end

"""
    run_example()

Process three synthetic residuals with delayed feedback and hidden-mode removal.
"""
function run_example()
    prepared = prepare_example(hidden_mode_gain=0.25f0,
        shape_to_hidden=Float32[0 0 1],
        hidden_to_shape=reshape(Float32[0, 0, 1], MODES, 1))
    return map((Float32[2, -1, 0.4], Float32[2, -1, 0.4], zeros(Float32, MODES))) do x
        status = process_frame!(prepared, x)
        status === nothing || error("controller frame failed: $status")
        (; commands=copy(prepared.outputs.commands),
            feedback=copy(prepared.previous_feedback))
    end
end

function (@main)(args)
    foreach(println, run_example())
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .CorrectorStagesExample: main
end
