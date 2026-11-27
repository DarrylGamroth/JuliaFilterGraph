module OpticalGainEstimatorTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct ZeroBasedEstimatorVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::ZeroBasedEstimatorVector) = size(vector.values)
Base.axes(vector::ZeroBasedEstimatorVector) = (0:(length(vector.values) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedEstimatorVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedEstimatorVector, index::Int) = vector.values[index + 1]
Base.setindex!(vector::ZeroBasedEstimatorVector, value, index::Int) =
    setindex!(vector.values, value, index + 1)

estimator_outputs(::Type{T}, modes; value=T(NaN)) where {T} = (
    command_amplitudes=fill(value, modes),
    response_amplitudes=fill(value, modes),
    optical_gains=fill(value, modes),
    phase_lags=fill(value, modes),
    valid=falses(modes),
    estimate_complete=trues(1),
)

estimator_inputs(::Type{T}, modes) where {T} = (
    realized_command_modes=zeros(T, modes),
    reconstructed_response_modes=zeros(T, modes),
    window_frame=zeros(UInt64, 1),
    phase_indices=zeros(UInt64, modes),
    completes_window=falses(1),
)

function process_estimator!(outputs, workspace, plan, inputs)
    return process!(
        outputs.command_amplitudes,
        outputs.response_amplitudes,
        outputs.optical_gains,
        outputs.phase_lags,
        outputs.valid,
        outputs.estimate_complete,
        workspace,
        plan,
        inputs.realized_command_modes,
        inputs.reconstructed_response_modes,
        inputs.window_frame,
        inputs.phase_indices,
        inputs.completes_window,
    )
end

function process_estimator!(outputs, prepared, inputs)
    return process!(
        outputs.command_amplitudes,
        outputs.response_amplitudes,
        outputs.optical_gains,
        outputs.phase_lags,
        outputs.valid,
        outputs.estimate_complete,
        prepared,
        inputs.realized_command_modes,
        inputs.reconstructed_response_modes,
        inputs.window_frame,
        inputs.phase_indices,
        inputs.completes_window,
    )
end

function set_estimator_frame!(inputs, frame, schedule, command, response)
    copyto!(inputs.realized_command_modes, command)
    copyto!(inputs.reconstructed_response_modes, response)
    inputs.window_frame[1] = UInt64(frame)
    @inbounds for mode in eachindex(inputs.phase_indices)
        inputs.phase_indices[mode] = UInt64(
            frame % schedule.modes[mode].frames_per_period,
        )
    end
    inputs.completes_window[1] = frame + 1 == schedule.frames_per_estimate
    return inputs
end

function direct_estimator_allocation_bytes(outputs, workspace, plan, inputs)
    process_estimator!(outputs, workspace, plan, inputs)
    discard!(workspace)
    return @allocated begin
        process_estimator!(outputs, workspace, plan, inputs)
        discard!(workspace)
    end
end

function prepared_estimator_allocation_bytes(outputs, prepared, inputs)
    return @allocated process_estimator!(outputs, prepared, inputs)
end

@testset "optical-gain estimator declaration preserves the frozen FGN contract" begin
    @test algorithm_label(OpticalGainEstimatorF32) ==
          "optical-gain-estimator-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        OpticalGainEstimatorF32,
    ) === OpticalGainEstimatorFailure
    configuration = algorithm_configuration(OpticalGainEstimatorF32)
    @test map(field -> field.name, configuration) ==
          (:modes, :initial_calibration_responses)
    @test map(field -> typeof(field.update), configuration) ==
          (GraphRebuildConfiguration, ConstructionConfiguration)

    ports = algorithm_ports(OpticalGainEstimatorF32)
    @test map(port -> port.name, ports) == (
        Symbol("realized-command-modes"),
        Symbol("reconstructed-response-modes"),
        Symbol("window-frame"),
        Symbol("phase-indices"),
        Symbol("completes-window"),
        Symbol("command-amplitudes"),
        Symbol("response-amplitudes"),
        Symbol("optical-gains"),
        Symbol("phase-lags"),
        :valid,
        Symbol("estimate-complete"),
        Symbol("calibration-responses"),
    )
    @test all(
        port -> port.direction isa FilterGraphAlgorithms.InputPort,
        ports[1:5],
    )
    @test all(
        port -> port.direction isa FilterGraphAlgorithms.OutputPort,
        ports[6:11],
    )
    @test ports[12].role isa FilterGraphAlgorithms.SparseParameter
    @test all(
        port -> port.metadata_source === Symbol("reconstructed-response-modes"),
        ports[6:11],
    )

    modes = [OpticalGainDitherMode(0.5f0, 4, 1)]
    prepared = prepare_algorithm(
        OpticalGainEstimatorF32,
        OpticalGainEstimatorF32Configuration(
            modes=modes,
            initial_calibration_responses=Float32[1],
        ),
    )
    formats = algorithm_formats(prepared)
    @test length(formats) == 12
    @test getproperty.(formats, :element_type) == (
        Float32,
        Float32,
        UInt64,
        UInt64,
        Bool,
        Float32,
        Float32,
        Float32,
        Float32,
        Bool,
        Bool,
        Float32,
    )
    @test getproperty.(formats, :schema) == (
        OPTICAL_GAIN_DITHER_COMMAND_V1,
        OPTICAL_GAIN_DITHER_RESPONSE_V1,
        OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1,
        OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
        OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
        OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        OPTICAL_GAIN_PHASE_LAG_V1,
        OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
        OPTICAL_GAIN_CALIBRATION_RESPONSE_V1,
    )
end

@testset "estimator preparation validates calibration normalization" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 1),
    ])
    @test_throws DimensionMismatch OpticalGainEstimatorPlan(schedule, Float32[])
    for response in (0.0f0, -1.0f0, NaN32, Inf32)
        @test_throws ArgumentError OpticalGainEstimatorPlan(
            schedule,
            Float32[response],
        )
    end
    plan = OpticalGainEstimatorPlan(schedule, Float32[1.25])
    @test optical_gain_calibration_responses(plan) == Float32[1.25]
    copy = optical_gain_calibration_responses(plan)
    copy[1] = 9
    @test optical_gain_calibration_responses(plan) == Float32[1.25]
    @test plan.two_over_window == 0.5f0
end

@testset "synchronous estimates match the frozen Rust oracle" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(0.4, 8, 3),
        OpticalGainDitherMode(0.7, 6, 4),
    ])
    calibrations = [1.25, 0.8]
    gains = [0.55, 0.9]
    lags = [0.3, -0.45]
    amplitudes = [0.4, 0.7]
    plan = OpticalGainEstimatorPlan(schedule, calibrations)
    workspace = OpticalGainEstimatorWorkspace(plan)
    outputs = estimator_outputs(Float64, 2)
    inputs = estimator_inputs(Float64, 2)
    for frame in 0:(schedule.frames_per_estimate - 1)
        command = [
            amplitudes[mode] *
            optical_gain_dither_cosines(schedule, mode)[
                frame % schedule.modes[mode].frames_per_period + 1
            ] for mode in 1:2
        ]
        response = [
            gains[mode] * calibrations[mode] * amplitudes[mode] * cos(
                2pi * frame / schedule.modes[mode].frames_per_period -
                lags[mode],
            ) for mode in 1:2
        ]
        set_estimator_frame!(inputs, frame, schedule, command, response)
        @test process_estimator!(outputs, workspace, plan, inputs) === nothing
        @test outputs.estimate_complete ==
              [frame + 1 == schedule.frames_per_estimate]
        @test optical_gain_estimator_frame(workspace) == frame
        if frame + 1 < schedule.frames_per_estimate
            @test all(isnan, outputs.optical_gains)
        end
        commit!(workspace)
    end
    @test outputs.command_amplitudes ≈ amplitudes atol=2e-15
    @test outputs.response_amplitudes ≈
          gains .* calibrations .* amplitudes atol=2e-15
    @test outputs.optical_gains ≈ gains atol=2e-15
    @test outputs.phase_lags ≈ lags atol=2e-15
    @test outputs.valid == [true, true]
    @test optical_gain_estimator_frame(workspace) == 0
end

@testset "zero command or response publishes an invalid expected estimate" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 1),
        OpticalGainDitherMode(1.0f0, 4, 1),
    ])
    plan = OpticalGainEstimatorPlan(schedule, Float32[1, 1])
    workspace = OpticalGainEstimatorWorkspace(plan)
    outputs = estimator_outputs(Float32, 2)
    inputs = estimator_inputs(Float32, 2)
    commands = ((0.0f0, 1.0f0), (0.0f0, 0.0f0), (0.0f0, -1.0f0), (0.0f0, 0.0f0))
    for frame in 0:3
        set_estimator_frame!(
            inputs,
            frame,
            schedule,
            Float32[commands[frame + 1]...],
            Float32[1, 0],
        )
        process_estimator!(outputs, workspace, plan, inputs)
        commit!(workspace)
    end
    @test outputs.estimate_complete == [true]
    @test outputs.valid == [false, false]
    @test outputs.optical_gains == Float32[0, 0]
    @test outputs.phase_lags == Float32[0, 0]
end

@testset "estimator accumulation uses an Algorithm-local candidate" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 1),
    ])
    plan = OpticalGainEstimatorPlan(schedule, Float32[1])
    workspace = OpticalGainEstimatorWorkspace(plan)
    outputs = estimator_outputs(Float32, 1)
    inputs = estimator_inputs(Float32, 1)
    set_estimator_frame!(inputs, 0, schedule, Float32[1], Float32[0.5])

    process_estimator!(outputs, workspace, plan, inputs)
    @test optical_gain_estimator_frame(workspace) == 0
    discard!(workspace)
    @test optical_gain_estimator_frame(workspace) == 0
    process_estimator!(outputs, workspace, plan, inputs)
    commit!(workspace)
    @test optical_gain_estimator_frame(workspace) == 1

    set_estimator_frame!(inputs, 1, schedule, Float32[0], Float32[0])
    process_estimator!(outputs, workspace, plan, inputs)
    inputs.phase_indices[1] = 0
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorPhaseIndexMismatch
    commit!(workspace)
    @test optical_gain_estimator_frame(workspace) == 1
    inputs.phase_indices[1] = 1
    process_estimator!(outputs, workspace, plan, inputs)
    commit!(workspace)
    @test optical_gain_estimator_frame(workspace) == 2

    reset!(workspace)
    @test optical_gain_estimator_frame(workspace) == 0
end

@testset "estimator rejects synchronization and numerical failures atomically" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0, 4, 1),
    ])
    plan = OpticalGainEstimatorPlan(schedule, [1.0])
    workspace = OpticalGainEstimatorWorkspace(plan)
    outputs = estimator_outputs(Float64, 1)
    inputs = estimator_inputs(Float64, 1)

    set_estimator_frame!(inputs, 0, schedule, [1.0], [0.5])
    inputs.window_frame[1] = typemax(UInt64)
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorWindowFrameOutOfRange
    inputs.window_frame[1] = 1
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorWindowFrameMismatch
    inputs.window_frame[1] = 0
    inputs.phase_indices[1] = 1
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorPhaseIndexMismatch
    inputs.phase_indices[1] = 0
    inputs.completes_window[1] = true
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorWindowCompletionMismatch
    inputs.completes_window[1] = false
    inputs.realized_command_modes[1] = NaN
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorNonFiniteCommand
    inputs.realized_command_modes[1] = 1
    inputs.reconstructed_response_modes[1] = NaN
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorNonFiniteResponse
    @test optical_gain_estimator_frame(workspace) == 0
    @test all(isnan, outputs.command_amplitudes)

    inputs.reconstructed_response_modes[1] = 0
    for (frame, command) in ((0, floatmax(Float64) / 2), (1, floatmax(Float64) / 2))
        set_estimator_frame!(inputs, frame, schedule, [command], [0.0])
        @test process_estimator!(outputs, workspace, plan, inputs) === nothing
        commit!(workspace)
    end
    set_estimator_frame!(inputs, 2, schedule, [-floatmax(Float64)], [0.0])
    @test process_estimator!(outputs, workspace, plan, inputs) ===
          OpticalGainEstimatorNonFiniteAccumulation
    @test optical_gain_estimator_frame(workspace) == 2
end

@testset "checked storage validation preserves caller ownership" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 1),
    ])
    plan = OpticalGainEstimatorPlan(schedule, Float32[1])
    workspace = OpticalGainEstimatorWorkspace(plan)
    outputs = estimator_outputs(Float32, 1; value=7.0f0)
    inputs = estimator_inputs(Float32, 1)
    set_estimator_frame!(inputs, 0, schedule, Float32[1], Float32[0.5])
    before = deepcopy(outputs)
    wrong = merge(outputs, (command_amplitudes=Float32[],))
    @test_throws DimensionMismatch process_estimator!(wrong, workspace, plan, inputs)
    @test outputs == before
    @test_throws ArgumentError process!(
        ZeroBasedEstimatorVector(copy(outputs.command_amplitudes)),
        outputs.response_amplitudes,
        outputs.optical_gains,
        outputs.phase_lags,
        outputs.valid,
        outputs.estimate_complete,
        workspace,
        plan,
        inputs.realized_command_modes,
        inputs.reconstructed_response_modes,
        inputs.window_frame,
        inputs.phase_indices,
        inputs.completes_window,
    )
    @test_throws ArgumentError process!(
        outputs.command_amplitudes,
        outputs.command_amplitudes,
        outputs.optical_gains,
        outputs.phase_lags,
        outputs.valid,
        outputs.estimate_complete,
        workspace,
        plan,
        inputs.realized_command_modes,
        inputs.reconstructed_response_modes,
        inputs.window_frame,
        inputs.phase_indices,
        inputs.completes_window,
    )
    @test_throws ArgumentError process!(
        plan.calibration_responses,
        outputs.response_amplitudes,
        outputs.optical_gains,
        outputs.phase_lags,
        outputs.valid,
        outputs.estimate_complete,
        workspace,
        plan,
        inputs.realized_command_modes,
        inputs.reconstructed_response_modes,
        inputs.window_frame,
        inputs.phase_indices,
        inputs.completes_window,
    )
    @test optical_gain_estimator_frame(workspace) == 0
end

@testset "calibration replacement is owned and preserves window position" begin
    modes = [OpticalGainDitherMode(0.5f0, 4, 1)]
    prepared = prepare_algorithm(
        OpticalGainEstimatorF32,
        OpticalGainEstimatorF32Configuration(
            modes=modes,
            initial_calibration_responses=Float32[1],
        ),
    )
    outputs = estimator_outputs(Float32, 1)
    inputs = estimator_inputs(Float32, 1)
    set_estimator_frame!(inputs, 0, prepared.plan.schedule, Float32[0.5], Float32[0.25])
    process_estimator!(outputs, prepared, inputs)
    @test optical_gain_estimator_frame(prepared.workspace) == 1

    replacement = Float32[0.5]
    @test replace_parameter!(prepared, "calibration-responses", replacement) ===
          prepared
    replacement[1] = 9
    @test optical_gain_calibration_responses(prepared.plan) == Float32[0.5]
    @test optical_gain_estimator_frame(prepared.workspace) == 1
    original = prepared.plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "calibration-responses",
        Float32[NaN],
    )
    @test prepared.plan === original
end

@testset "estimator views and warmed outcomes allocate no heap" begin
    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 1),
    ])
    plan = OpticalGainEstimatorPlan(schedule, Float32[1])
    workspace = OpticalGainEstimatorWorkspace(plan)
    output_storage = ntuple(_ -> fill(-9.0f0, 2), Val(4))
    bool_storage = (falses(2), falses(2))
    outputs = (
        command_amplitudes=@view(output_storage[1][1:2:1]),
        response_amplitudes=@view(output_storage[2][1:2:1]),
        optical_gains=@view(output_storage[3][1:2:1]),
        phase_lags=@view(output_storage[4][1:2:1]),
        valid=@view(bool_storage[1][1:2:1]),
        estimate_complete=@view(bool_storage[2][1:2:1]),
    )
    inputs = estimator_inputs(Float32, 1)
    set_estimator_frame!(inputs, 0, schedule, Float32[1], Float32[0.5])
    @test direct_estimator_allocation_bytes(outputs, workspace, plan, inputs) == 0

    inputs.realized_command_modes[1] = NaN
    process_estimator!(outputs, workspace, plan, inputs)
    @test @allocated(process_estimator!(outputs, workspace, plan, inputs)) == 0
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0

    prepared = prepare_algorithm(
        OpticalGainEstimatorF32,
        OpticalGainEstimatorF32Configuration(
            modes=[OpticalGainDitherMode(1.0f0, 4, 1)],
            initial_calibration_responses=Float32[1],
        ),
    )
    inputs.realized_command_modes[1] = 1
    set_estimator_frame!(inputs, 0, prepared.plan.schedule, Float32[1], Float32[0.5])
    process_estimator!(outputs, prepared, inputs)
    set_estimator_frame!(inputs, 1, prepared.plan.schedule, Float32[0], Float32[0])
    @test prepared_estimator_allocation_bytes(outputs, prepared, inputs) == 0
end

@testset "optical-gain estimator public API exports" begin
    names = (
        :OPTICAL_GAIN_CALIBRATION_RESPONSE_V1,
        :OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        :OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        :OPTICAL_GAIN_DITHER_RESPONSE_V1,
        :OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
        :OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        :OPTICAL_GAIN_PHASE_LAG_V1,
        :OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        :OpticalGainEstimatorF32,
        :OpticalGainEstimatorF32Configuration,
        :OpticalGainEstimatorFailure,
        :OpticalGainEstimatorNonFiniteAccumulation,
        :OpticalGainEstimatorNonFiniteCommand,
        :OpticalGainEstimatorNonFiniteEstimate,
        :OpticalGainEstimatorNonFiniteResponse,
        :OpticalGainEstimatorPhaseIndexMismatch,
        :OpticalGainEstimatorPlan,
        :OpticalGainEstimatorWindowCompletionMismatch,
        :OpticalGainEstimatorWindowFrameMismatch,
        :OpticalGainEstimatorWindowFrameOutOfRange,
        :OpticalGainEstimatorWorkspace,
        :optical_gain_calibration_responses,
        :optical_gain_estimator_frame,
    )
    @test all(name -> Base.isexported(FilterGraphAlgorithms, name), names)
end


@testset "finite optical gain with an overflowing amplitude product" begin
    schedule = OpticalGainDitherSchedule([OpticalGainDitherMode(1.0f0, 4, 1)])
    plan = OpticalGainEstimatorPlan(schedule, Float32[floatmax(Float32)])
    ws = OpticalGainEstimatorWorkspace(plan)
    outputs = (zeros(Float32, 1), zeros(Float32, 1), zeros(Float32, 1),
        zeros(Float32, 1), falses(1), falses(1))
    for phase in 0:3
        sample = Float32[2, 0, -2, 0][phase + 1]
        response = sample * 1f36
        @test process!(outputs..., ws, plan, Float32[sample], Float32[response],
            UInt64[phase], UInt64[phase], Bool[phase == 3]) === nothing
        FilterGraphAlgorithms.commit!(ws)
    end
    @test only(outputs[5])
    @test only(outputs[3]) ≈ Float64(only(outputs[2])) /
        (Float64(only(outputs[1])) * Float64(only(plan.calibration_responses)))
end

end
