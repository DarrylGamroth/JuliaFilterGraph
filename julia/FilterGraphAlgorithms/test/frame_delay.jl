module FrameDelayTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

function process_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

commit_allocation_bytes(workspace) = @allocated commit!(workspace)
discard_allocation_bytes(workspace) = @allocated discard!(workspace)
reset_allocation_bytes(workspace) = @allocated reset!(workspace)

function prepare_delay(delay_frames; extent=2, initial_value=-1.0f0)
    return prepare_algorithm(
        FrameDelayF32,
        FrameDelayF32Configuration(
            extent=extent,
            delay_frames=delay_frames,
            initial_value=initial_value,
            input_schema=CONTROLLER_COMMAND_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
end

@testset "frame delay declaration and exact committed sequence" begin
    @test algorithm_label(FrameDelayF32) == "frame-delay-f32"
    @test map(field -> field.name, algorithm_configuration(FrameDelayF32)) == (
        :extent,
        :delay_frames,
        :initial_value,
        :input_schema,
        :output_schema,
    )
    ports = algorithm_ports(FrameDelayF32)
    @test map(port -> port.name, ports) == (:input, :output)
    @test ports[2].metadata_source === :input

    prepared = prepare_delay(2)
    workspace = prepared.workspace
    output = fill(71.0f0, 2)
    expected = (
        Float32[-1, -1],
        Float32[-1, -1],
        Float32[1, 10],
        Float32[2, 20],
    )
    for (input, delayed) in zip(
        (Float32[1, 10], Float32[2, 20], Float32[3, 30], Float32[4, 40]),
        expected,
    )
        @test process!(output, prepared, input) === nothing
        @test output == delayed
    end
    @test frame_delay_head(workspace) == 1
    @test frame_delay_history(workspace) == Float32[3 4; 30 40]

    @test reset!(prepared) === prepared
    @test frame_delay_head(workspace) == 1
    @test frame_delay_history(workspace) == fill(-1.0f0, 2, 2)
    @test process!(output, prepared, Float32[5, 50]) === nothing
    @test output == Float32[-1, -1]
end

@testset "frame delay zero, one, discard, failure, and views" begin
    output = zeros(Float32, 2)
    zero_delay = prepare_delay(0; initial_value=0.0f0)
    input = Float32[2, -3]
    @test process!(output, zero_delay, input) === nothing
    @test output == input
    @test process!(input, zero_delay, input) === nothing
    @test input == Float32[2, -3]

    one_delay = prepare_delay(1; initial_value=0.0f0)
    @test FilterGraphAlgorithms._process_admitted!(
        output,
        one_delay,
        Float32[1, 2],
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test output == Float32[0, 0]
    @test discard!(one_delay.workspace) === one_delay.workspace
    @test process!(output, one_delay, Float32[3, 4]) === nothing
    @test output == Float32[0, 0]
    @test process!(output, one_delay, Float32[5, 6]) === nothing
    @test output == Float32[3, 4]

    history_before = copy(frame_delay_history(one_delay.workspace))
    head_before = frame_delay_head(one_delay.workspace)
    fill!(output, 71.0f0)
    @test process!(output, one_delay, Float32[NaN, 0]) ===
        FrameDelayNonFiniteInput
    @test output == fill(71.0f0, 2)
    @test frame_delay_history(one_delay.workspace) == history_before
    @test frame_delay_head(one_delay.workspace) == head_before

    view_delay = prepare_delay(1; initial_value=0.5f0)
    input_storage = fill(99.0f0, 4)
    output_storage = fill(71.0f0, 4)
    input_view = @view input_storage[1:2:3]
    output_view = @view output_storage[2:2:4]
    input_view .= Float32[7, 8]
    @test process!(output_view, view_delay, input_view) === nothing
    @test output_view == Float32[0.5, 0.5]
    @test output_storage[1:2:3] == fill(71.0f0, 2)

    @test_throws ArgumentError process!(input, one_delay, input)
    @test_throws DimensionMismatch process!(zeros(Float32, 1), one_delay, input)
    @test_throws ArgumentError FrameDelayPlan(-1)
    @test_throws ArgumentError prepare_delay(-1)
    @test_throws ArgumentError prepare_delay(1; extent=0)
    @test_throws ArgumentError prepare_delay(1; initial_value=NaN32)
end

@testset "frame delay workspace preserves the floating representation" begin
    plan = FrameDelayPlan(1)
    workspace = FrameDelayWorkspace(Float16[0, 0], plan)
    output = zeros(Float16, 2)
    input = Float16[1, -2]

    @test @inferred Union{FrameDelayFailure,Vector{Float16}} process!(
        output,
        workspace,
        plan,
        input,
    ) === output
    @test output == Float16[0, 0]
    @test @inferred(commit!(workspace)) === workspace
    @test eltype(frame_delay_history(workspace)) === Float16

    fill!(input, Float16(3))
    @test process!(output, workspace, plan, input) === output
    @test output == Float16[1, -2]
end

@testset "frame delay warmed paths infer and allocate zero bytes" begin
    prepared = prepare_delay(3; initial_value=0.0f0)
    output = zeros(Float32, 2)
    input = Float32[1, -2]
    @test @inferred Union{Nothing,FrameDelayFailure} process!(
        output,
        prepared,
        input,
    ) === nothing
    @test @inferred(commit!(prepared.workspace)) === prepared.workspace
    @test @inferred(discard!(prepared.workspace)) === prepared.workspace
    @test @inferred(reset!(prepared.workspace)) === prepared.workspace

    process!(output, prepared, input)
    @test process_allocation_bytes(output, prepared, input) == 0
    @test commit_allocation_bytes(prepared.workspace) == 0
    @test discard_allocation_bytes(prepared.workspace) == 0
    @test reset_allocation_bytes(prepared.workspace) == 0

    nonfinite = Float32[NaN, 0]
    process!(output, prepared, nonfinite)
    @test process_allocation_bytes(output, prepared, nonfinite) == 0
end

end
