@enum TestOpticalGainDitherGateFailure::UInt8 begin
    TestOpticalGainDitherGateRejected = 1
end

struct TestOpticalGainDitherGatePlan
    mode_count::Int
end

@algorithm TestOpticalGainDitherGateF32 begin
    label = "test-optical-gain-dither-gate-f32"
    failure = TestOpticalGainDitherGateFailure

    @configuration TestOpticalGainDitherGateF32Configuration begin
        mode_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.mode_count,),
            OPTICAL_GAIN_DITHER_COMMAND_V1,
        ),
        output(
            :output,
            Float32,
            (configuration.mode_count,),
            OPTICAL_GAIN_DITHER_COMMAND_V1;
            metadata_from=:input,
        ),
    )

    prepare(configuration) = TestOpticalGainDitherGatePlan(configuration.mode_count)

    process!(plan) = begin
        input[1] < 0.0f0 && return TestOpticalGainDitherGateRejected
        copyto!(output, input)
        return nothing
    end
end

const OPTICAL_GAIN_DITHER_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = dither
            label = optical-gain-dither-f32
            config = {
                modes = [
                    {
                        amplitude = 1.0
                        frames_per_period = 4
                        periods_per_estimate = 3
                    }
                    {
                        amplitude = 0.5
                        frames_per_period = 6
                        periods_per_estimate = 2
                    }
                ]
            }
        }
    ]
    links = []
    inputs = []
    outputs = [
        "dither:commanded-modes"
        "dither:window-frame"
        "dither:phase-indices"
        "dither:completes-window"
    ]
}
"""

const OPTICAL_GAIN_DITHER_GATE_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = dither
            label = optical-gain-dither-f32
            config = {
                modes = [
                    {
                        amplitude = 1.0
                        frames_per_period = 4
                        periods_per_estimate = 1
                    }
                ]
            }
        }
        {
            type = ndarray
            name = gate
            label = test-optical-gain-dither-gate-f32
            config = { mode_count = 1 }
        }
    ]
    links = [
        { output = "dither:commanded-modes" input = "gate:input" }
    ]
    inputs = []
    outputs = [
        "gate:output"
        "dither:window-frame"
        "dither:phase-indices"
        "dither:completes-window"
    ]
}
"""

const DITHER_OUTPUT_NAMES = (
    Symbol("commanded-modes"),
    Symbol("window-frame"),
    Symbol("phase-indices"),
    Symbol("completes-window"),
)

function dither_graph_outputs(names=DITHER_OUTPUT_NAMES, modes=2)
    return NamedTuple{names}(
        (
            zeros(Float32, modes),
            zeros(UInt64, 1),
            zeros(UInt64, modes),
            zeros(Bool, 1),
        ),
    )
end

dither_graph_allocation_bytes(outputs, graph) =
    @allocated process!(outputs, graph, NamedTuple())

@testset "optical-gain dither is a SPA-JSON source Node" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_DITHER_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32,),
    )
    @test isempty(keys(graph.input_formats))
    @test keys(graph.output_formats) == DITHER_OUTPUT_NAMES
    @test getproperty.(values(graph.output_formats), :shape) ==
          ((2,), (1,), (2,), (1,))
    @test getproperty.(values(graph.output_formats), :schema) == (
        OPTICAL_GAIN_DITHER_COMMAND_V1,
        OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1,
        OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
        OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
    )
    @test optical_gain_dither_modes(only(graph.nodes).prepared.plan.schedule) == [
        OpticalGainDitherMode(1.0f0, 4, 3),
        OpticalGainDitherMode(0.5f0, 6, 2),
    ]

    outputs = dither_graph_outputs()
    availability = NamedTuple{DITHER_OUTPUT_NAMES}((true, true, true, true))
    for frame in 0:11
        @test process!(outputs, graph, NamedTuple()) == availability
        @test outputs[1][1] ≈ Float32(cos(2pi * frame / 4)) atol=4eps(Float32)
        @test outputs[1][2] ≈ Float32(0.5cos(2pi * frame / 6)) atol=4eps(Float32)
        @test outputs[2] == UInt64[frame]
        @test outputs[3] == UInt64[frame % 4, frame % 6]
        @test outputs[4] == [frame == 11]
    end
    @test optical_gain_dither_phases(only(graph.nodes).prepared.workspace) == [0, 0]
    @test FilterGraphs.output_metadata(graph) == NamedTuple{
        DITHER_OUTPUT_NAMES,
    }((nothing, nothing, nothing, nothing))
end

@testset "dither Node commits its phase on success" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_DITHER_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32,),
    )
    workspace = only(graph.nodes).prepared.workspace
    outputs = dither_graph_outputs()

    @test process!(outputs, graph, NamedTuple()) isa NamedTuple
    @test optical_gain_dither_phases(workspace) == [1, 1]
    @test reset!(graph) === graph
    @test optical_gain_dither_phases(workspace) == [0, 0]
end

@testset "downstream rejection preserves outputs but commits dither phase" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_DITHER_GATE_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, TestOpticalGainDitherGateF32),
    )
    names = (
        :output,
        Symbol("window-frame"),
        Symbol("phase-indices"),
        Symbol("completes-window"),
    )
    outputs = dither_graph_outputs(names, 1)
    workspace = first(graph.nodes).prepared.workspace
    @test process!(outputs, graph, NamedTuple()) ==
          NamedTuple{names}((true, true, true, true))
    @test outputs.output == Float32[1]
    @test optical_gain_dither_phases(workspace) == [1]

    retained = map(copy, outputs)
    @test process!(outputs, graph, NamedTuple()) ===
          TestOpticalGainDitherGateRejected
    @test outputs == retained
    @test optical_gain_dither_phases(workspace) == [2]
    @test process!(outputs, graph, NamedTuple()) ===
          TestOpticalGainDitherGateRejected
    @test outputs == retained
    @test optical_gain_dither_phases(workspace) == [3]
end

@testset "optical-gain dither Graph paths allocate no heap after warmup" begin
    graph = prepare_graph_text(
        OPTICAL_GAIN_DITHER_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32,),
    )
    outputs = dither_graph_outputs()
    process!(outputs, graph, NamedTuple())
    @test dither_graph_allocation_bytes(outputs, graph) == 0
    gate = prepare_graph_text(
        OPTICAL_GAIN_DITHER_GATE_GRAPH_TEXT;
        algorithms=(OpticalGainDitherF32, TestOpticalGainDitherGateF32),
    )
    names = (
        :output,
        Symbol("window-frame"),
        Symbol("phase-indices"),
        Symbol("completes-window"),
    )
    gate_outputs = dither_graph_outputs(names, 1)
    process!(gate_outputs, gate, NamedTuple())
    process!(gate_outputs, gate, NamedTuple())
    @test dither_graph_allocation_bytes(gate_outputs, gate) == 0
end
