module CorrectorStagesTests

using Test
using FilterGraphAlgorithms, FilterGraphs
import FilterGraphAlgorithms: process!

function prepare_graph_text(text; algorithms)
    return mktemp() do path, stream
        write(stream, text)
        flush(stream)
        prepare_graph(path; algorithms)
    end
end

include(joinpath(@__DIR__, "data", "examples", "corrector-stages/run.jl"))
const E = CorrectorStagesExample

@testset "corrector allocation command limit pairs" begin
    text = """
    filter.graph = {
      nodes = [{type=ndarray name=allocation label=corrector-allocation-f32 config={
        controller_command_count=2 stage_command_counts=[2] update_periods=[1]
        command_projection=[1 0 0 1] realization_projection=[1 0 0 1]
        section_count=1 initial_coefficients=null command_limits=[[-0.2 -0.3] [0.4 0.5]]
        slew_limits=null command_schema=org.example.commands/1
      }}]
      inputs=[allocation:controller-command]
      outputs=[allocation:commands allocation:controller-constraint-feedback]
    }
    """
    graph = prepare_graph_text(text; algorithms=(CorrectorAllocationF32,))
    commands, feedback = zeros(Float32, 2), zeros(Float32, 2)
    outputs = (commands=commands, var"controller-constraint-feedback"=feedback)
    process!(outputs, graph, (var"controller-command"=Float32[-2, 2],))
    @test commands == Float32[-0.2, 0.5]
    @test feedback == Float32[-1.8, 1.5]
    for limits in ("[]", "[[-1 -1]]", "[[-1 -1] [1 1] [2 2]]",
            "[[-1] [1 1]]", "[[-1 -1] [1]]", "[1 -1]",
            "[[1 1] [-1 -1]]", "[[-1 -1] [1e100 1]]")
        @test_throws GraphConfigurationError prepare_graph_text(replace(text,
            "command_limits=[[-0.2 -0.3] [0.4 0.5]]" => "command_limits=$limits");
            algorithms=(CorrectorAllocationF32,))
    end
end

allocation_bytes(p, x) = @allocated E.process_frame!(p, x)

function single_stage(; limit=100f0)
    return (E.stage(E.identity_matrix(3), E.identity_matrix(3);
        lower=-limit, upper=limit, slew=100f0),)
end

identity_regions() = (1:3 => E.identity_sections(),)

# Independent direct-form-I history; runtime uses transposed direct form II.
function filter_history(coefficients, extent)
    bank = ndims(coefficients) == 2 ?
        reshape(Float64.(coefficients), 5, size(coefficients, 2), 1) :
        Float64.(coefficients)
    dimensions = (extent, size(bank, 2))
    return (; bank, x1=zeros(dimensions), x2=zeros(dimensions),
        y1=zeros(dimensions), y2=zeros(dimensions))
end

function filter_sample!(history, input)
    output = Float64.(input)
    for section in axes(history.bank, 2), mode in eachindex(output)
        set = size(history.bank, 3) == 1 ? 1 : mode
        b0, b1, b2, a1, a2 = history.bank[:, section, set]
        x = output[mode]
        y = b0*x + b1*history.x1[mode, section] + b2*history.x2[mode, section] -
            a1*history.y1[mode, section] - a2*history.y2[mode, section]
        history.x2[mode, section] = history.x1[mode, section]
        history.x1[mode, section] = x
        history.y2[mode, section] = history.y1[mode, section]
        history.y1[mode, section] = y
        output[mode] = y
    end
    return output
end

@testset "region filters, four-stage allocation, delayed feedback and hidden removal" begin
    stages = E.default_stages()
    shape_to_hidden = Float32[1 0 -1]
    hidden_to_shape = reshape(Float32[0.5, 0, -0.5], 3, 1)
    gain, pole, kaw, kh = 0.6f0, 0.9f0, 1f0, 0.4f0
    p = E.prepare_example(; stages, gain, pole, anti_windup_gain=kaw,
        hidden_mode_gain=kh, shape_to_hidden, hidden_to_shape)
    region_bank = E.FGA.BiquadCascadePlan(E.default_regions(), 3).coefficients
    history = filter_history(region_bank, 3)
    previous_demand = zeros(Float64, 3)
    previous_feedback = zeros(Float64, 3)
    stage_histories = [filter_history(s.coefficients, size(s.to_command, 1)) for s in stages]
    held = [zeros(Float64, size(s.to_command, 1)) for s in stages]
    first_held = zeros(Float32, 1)
    samples = (Float32[2, -1, 0.8], Float32[1, 0.5, -0.7],
        Float32[-2, 1, 0.3], Float32[0.1, -0.4, 0.1],
        Float32[2, 1, -1], zeros(Float32, 3), zeros(Float32, 3))

    for (frame, x) in enumerate(samples)
        filtered_error = filter_sample!(history, x)
        state = frame == 1 ? zeros(Float32, 3) :
            previous_demand - kaw * previous_feedback -
            kh * hidden_to_shape * shape_to_hidden * (previous_demand - previous_feedback)
        demand = pole * state + gain * filtered_error
        residual = copy(demand)
        for (index, s) in enumerate(stages)
            requested = s.to_command * residual
            filtered = filter_sample!(stage_histories[index], requested)
            if (frame - 1) % s.period == 0
                for actuator in eachindex(requested)
                    target = clamp(filtered[actuator], s.lower[actuator], s.upper[actuator])
                    held[index][actuator] = clamp(target,
                        held[index][actuator] - s.slew[actuator],
                        held[index][actuator] + s.slew[actuator])
                end
            end
            residual -= s.to_controller * held[index]
        end
        @test E.process_frame!(p, x) === nothing
        @test p.outputs.correction ≈ demand atol=2f-6
        @test p.outputs.var"controller-state" ≈ state atol=2f-6
        @test p.outputs.commands ≈ reduce(vcat, held) atol=2f-6
        @test p.previous_feedback ≈ residual atol=2f-6
        commanded = sum(s.to_controller * h for (s, h) in zip(stages, held))
        @test demand - p.previous_feedback ≈ commanded atol=2f-6
        if frame == 1
            first_held .= p.outputs.commands[1:1]
        elseif frame <= 4
            @test p.outputs.commands[1:1] == first_held
        end
        previous_demand .= demand
        previous_feedback .= residual
    end
    @test length(p.outputs.commands) == 7
    E.reset!(p)
    fresh = E.prepare_example(; stages, gain, pole, anti_windup_gain=kaw,
        hidden_mode_gain=kh, shape_to_hidden, hidden_to_shape)
    @test E.process_frame!(p, first(samples)) === nothing
    @test E.process_frame!(fresh, first(samples)) === nothing
    @test p.outputs == fresh.outputs
end

@testset "commanded aggregate feedback excludes deliberate offload" begin
    first = E.stage(E.identity_matrix(3), E.identity_matrix(3);
        lower=-0.1f0, upper=0.1f0)
    p = E.prepare_example(regions=identity_regions(), stages=(first, only(single_stage())),
        gain=1f0, pole=1f0)
    @test E.process_frame!(p, Float32[1, -2, 0.5]) === nothing
    @test p.outputs.commands[1:3] == Float32[0.1, -0.1, 0.1]
    @test p.outputs.commands[4:6] ≈ Float32[0.9, -1.9, 0.4]
    @test p.previous_feedback ≈ zeros(Float32, 3) atol=3f-7
end

@testset "anti-windup releases saturation and hidden removal uses commanded shape" begin
    controllers = [E.prepare_example(regions=identity_regions(), stages=single_stage(limit=0.2f0),
        gain=0.5f0, pole=0.9f0, anti_windup_gain=kaw) for kaw in (0f0, 1f0)]
    for p in controllers
        for _ in 1:20
            @test E.process_frame!(p, ones(Float32, 3)) === nothing
        end
        @test all(x -> x <= 0.2f0, p.outputs.commands)
        @test E.process_frame!(p, zeros(Float32, 3)) === nothing
    end
    @test all(x -> x > 0.2f0, controllers[1].outputs.correction)
    @test controllers[2].outputs.correction ≈ fill(0.18f0, 3)
    @test controllers[2].previous_feedback ≈ zeros(Float32, 3) atol=3f-7

    p = E.prepare_example(regions=identity_regions(), stages=single_stage(limit=0.2f0),
        gain=1f0, pole=1f0, anti_windup_gain=0f0, hidden_mode_gain=1f0,
        shape_to_hidden=Float32[1 0 0],
        hidden_to_shape=reshape(Float32[1, 0, 0], 3, 1))
    @test E.process_frame!(p, Float32[1, 0, 0]) === nothing
    @test E.process_frame!(p, zeros(Float32, 3)) === nothing
    # Removal is 0.2 (commanded), rather than the full requested hidden mode 1.0.
    @test p.outputs.var"controller-state" ≈ Float32[0.8, 0, 0]
    @test p.outputs.correction ≈ Float32[0.8, 0, 0]
end

@testset "failure requires reset and warmed base-rate processing allocates no heap" begin
    p = E.prepare_example()
    x = ones(Float32, 3)
    for _ in 1:8
        E.process_frame!(p, x)
    end
    allocation_bytes(p, x)
    @test allocation_bytes(p, x) == 0
    @test allocation_bytes(p, x) == 0 # also covers a held stage tick
    previous = copy(p.previous_feedback)
    outputs = map(copy, p.outputs)
    @test E.process_frame!(p, fill(NaN32, 3)) === E.FGA.BiquadCascadeNonFiniteInput
    @test p.previous_feedback == previous
    @test p.outputs == outputs
    @test_throws ErrorException E.process_frame!(p, x)
    E.reset!(p)
    fresh = E.prepare_example()
    @test E.process_frame!(p, x) === nothing
    @test E.process_frame!(fresh, x) === nothing
    @test p.outputs == fresh.outputs

    huge = E.stage(E.identity_matrix(3), floatmax(Float32) * E.identity_matrix(3);
        lower=-100f0, upper=100f0, slew=100f0)
    p = E.prepare_example(regions=identity_regions(), stages=(huge,), gain=1f0)
    @test E.process_frame!(p, fill(2f0, 3)) === E.FGA.CorrectorAllocationNonFiniteOutput
    @test p.previous_feedback == zeros(Float32, 3)
    @test p.outputs.commands == zeros(Float32, 3)
    @test_throws ErrorException E.process_frame!(p, zeros(Float32, 3))
    E.reset!(p)
    @test E.process_frame!(p, zeros(Float32, 3)) === nothing
end

end
