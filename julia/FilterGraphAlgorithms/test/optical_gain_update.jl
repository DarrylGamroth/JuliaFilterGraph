module OpticalGainUpdateTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct ZeroBasedUpdateVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::ZeroBasedUpdateVector) = size(vector.values)
Base.axes(vector::ZeroBasedUpdateVector) = (0:(length(vector.values) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedUpdateVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedUpdateVector, index::Int) = vector.values[index + 1]
Base.setindex!(vector::ZeroBasedUpdateVector, value, index::Int) =
    setindex!(vector.values, value, index + 1)

criterion(::Type{T}, minimum_gain=T(0.2), maximum_gain=one(T)) where {T} =
    OpticalGainUpdateCriteria(
        T(0.1),
        T(0.05),
        minimum_gain,
        maximum_gain,
        T(0.25),
    )

update_outputs(::Type{T}, dither_modes, reconstructed_modes; value=T(NaN)) where {T} = (
    optical_gains=fill(value, reconstructed_modes),
    accepted_dither_modes=trues(dither_modes),
    updated_reconstructed_modes=trues(reconstructed_modes),
)

update_inputs(::Type{T}, dither_modes; complete=true) where {T} = (
    command_amplitudes=fill(T(0.5), dither_modes),
    response_amplitudes=fill(T(0.25), dither_modes),
    optical_gain_estimates=fill(T(0.5), dither_modes),
    phase_lags=zeros(T, dither_modes),
    valid=trues(dither_modes),
    estimate_complete=Bool[complete],
)

function process_update!(outputs, workspace, plan, inputs)
    return process!(
        outputs.optical_gains,
        outputs.accepted_dither_modes,
        outputs.updated_reconstructed_modes,
        workspace,
        plan,
        inputs.command_amplitudes,
        inputs.response_amplitudes,
        inputs.optical_gain_estimates,
        inputs.phase_lags,
        inputs.valid,
        inputs.estimate_complete,
    )
end

function process_update!(outputs, prepared, inputs)
    return process!(
        outputs.optical_gains,
        outputs.accepted_dither_modes,
        outputs.updated_reconstructed_modes,
        prepared,
        inputs.command_amplitudes,
        inputs.response_amplitudes,
        inputs.optical_gain_estimates,
        inputs.phase_lags,
        inputs.valid,
        inputs.estimate_complete,
    )
end

function direct_update_allocation_bytes(outputs, workspace, plan, inputs)
    process_update!(outputs, workspace, plan, inputs)
    discard!(workspace)
    return @allocated begin
        process_update!(outputs, workspace, plan, inputs)
        discard!(workspace)
    end
end

prepared_update_allocation_bytes(outputs, prepared, inputs) =
    @allocated process_update!(outputs, prepared, inputs)

@testset "optical-gain update declaration preserves the frozen FGN contract" begin
    @test algorithm_label(OpticalGainUpdateF32) == "optical-gain-update-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(OpticalGainUpdateF32) ===
          OpticalGainUpdateFailure
    configuration = algorithm_configuration(OpticalGainUpdateF32)
    @test map(field -> field.name, configuration) ==
          (:criteria, :assignments, :initial_optical_gains)
    @test map(field -> typeof(field.update), configuration) == (
        GraphRebuildConfiguration,
        GraphRebuildConfiguration,
        ConstructionConfiguration,
    )

    ports = algorithm_ports(OpticalGainUpdateF32)
    @test map(port -> port.name, ports) == (
        Symbol("command-amplitudes"),
        Symbol("response-amplitudes"),
        Symbol("optical-gain-estimates"),
        Symbol("phase-lags"),
        :valid,
        Symbol("estimate-complete"),
        Symbol("optical-gains"),
        Symbol("accepted-dither-modes"),
        Symbol("updated-reconstructed-modes"),
    )
    @test all(port -> port.direction isa FilterGraphAlgorithms.InputPort, ports[1:6])
    @test all(port -> port.direction isa FilterGraphAlgorithms.OutputPort, ports[7:9])
    @test all(port -> port.metadata_source === Symbol("estimate-complete"), ports[7:9])

    prepared = prepare_algorithm(
        OpticalGainUpdateF32,
        OpticalGainUpdateF32Configuration(
            criteria=[criterion(Float32)],
            assignments=OpticalGainAssignment{Float32}[
                OpticalGainAssignment(0, 0.5f0),
                OpticalGainAssignment{Float32}(),
            ],
            initial_optical_gains=Float32[0.8, 3],
        ),
    )
    formats = algorithm_formats(prepared)
    @test length(formats) == 9
    @test getproperty.(formats, :element_type) == (
        Float32,
        Float32,
        Float32,
        Float32,
        Bool,
        Bool,
        Float32,
        Bool,
        Bool,
    )
    @test getproperty.(formats, :schema) == (
        OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        OPTICAL_GAIN_PHASE_LAG_V1,
        OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
        PHYSICAL_OPTICAL_GAIN_V1,
        OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1,
        OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
    )
end

@testset "tagged assignment configuration is strict and transport-neutral" begin
    criteria = convert(
        OpticalGainUpdateCriteria{Float32},
        Dict(
            "minimum_command_amplitude" => 0.1,
            "minimum_response_amplitude" => 0.05,
            "minimum_optical_gain" => 0.2,
            "maximum_optical_gain" => 1,
            "maximum_absolute_phase_lag" => 0.25,
        ),
    )
    @test criteria == criterion(Float32)
    hold = convert(
        OpticalGainAssignment{Float32},
        Dict("kind" => "hold"),
    )
    assigned = convert(
        OpticalGainAssignment{Float32},
        Dict("kind" => "dither-mode", "dither_mode" => 1, "update_fraction" => 0.5),
    )
    @test optical_gain_assignment_dither_mode(hold) === nothing
    @test optical_gain_assignment_update_fraction(hold) === nothing
    @test optical_gain_assignment_dither_mode(assigned) == 1
    @test optical_gain_assignment_update_fraction(assigned) == 0.5f0
    @test_throws ArgumentError convert(
        OpticalGainAssignment{Float32},
        Dict("kind" => "hold", "dither_mode" => 0),
    )
    @test_throws ArgumentError convert(
        OpticalGainAssignment{Float32},
        Dict("kind" => "unknown"),
    )
end

@testset "update preparation validates criteria, assignments, and history" begin
    assigned = OpticalGainAssignment(0, 1.0f0)
    @test_throws ArgumentError OpticalGainUpdatePlan(
        OpticalGainUpdateCriteria{Float32}[],
        OpticalGainAssignment{Float32}[assigned],
    )
    @test_throws ArgumentError OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[],
    )
    for invalid in (
        OpticalGainUpdateCriteria(-0.1f0, 0.0f0, 0.2f0, 1.0f0, 0.25f0),
        OpticalGainUpdateCriteria(0.1f0, 0.0f0, 0.0f0, 1.0f0, 0.25f0),
        OpticalGainUpdateCriteria(
            0.1f0,
            0.0f0,
            nextfloat(0.0f0),
            1.0f0,
            0.25f0,
        ),
        OpticalGainUpdateCriteria(
            0.1f0,
            0.0f0,
            0.2f0,
            1.0f0,
            Float32(pi) + 0.1f0,
        ),
    )
        @test_throws ArgumentError OpticalGainUpdatePlan(
            [invalid],
            OpticalGainAssignment{Float32}[assigned],
        )
    end
    @test_throws ArgumentError OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(1, 1.0f0)],
    )
    @test_throws ArgumentError OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(0, 0.0f0)],
    )

    plan = OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[
            assigned,
            OpticalGainAssignment{Float32}(),
        ],
    )
    @test_throws DimensionMismatch OpticalGainUpdateWorkspace(plan, Float32[1])
    @test_throws ArgumentError OpticalGainUpdateWorkspace(plan, Float32[NaN, 1])
    @test_throws ArgumentError OpticalGainUpdateWorkspace(plan, Float32[1.1, 1])
    workspace = OpticalGainUpdateWorkspace(plan, Float32[1, 3])
    @test optical_gains(workspace) == Float32[1, 3]
    copied = optical_gain_update_assignments(plan)
    @test length(copied) == 2
    @test optical_gain_update_criteria(plan) == [criterion(Float32)]
end

@testset "incomplete estimates hold without reading unpublished fields" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(0, 1.0f0)],
    )
    workspace = OpticalGainUpdateWorkspace(plan, Float32[0.8])
    outputs = update_outputs(Float32, 1, 1)
    inputs = (
        command_amplitudes=Float32[NaN],
        response_amplitudes=Float32[NaN],
        optical_gain_estimates=Float32[NaN],
        phase_lags=Float32[NaN],
        valid=trues(1),
        estimate_complete=falses(1),
    )
    @test process_update!(outputs, workspace, plan, inputs) === nothing
    @test outputs.optical_gains == Float32[0.8]
    @test outputs.accepted_dither_modes == [false]
    @test outputs.updated_reconstructed_modes == [false]
    commit!(workspace)
    @test optical_gains(workspace) == Float32[0.8]
end

@testset "accepted estimates support grouped, filtered, and held assignments" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float64), criterion(Float64)],
        OpticalGainAssignment{Float64}[
            OpticalGainAssignment(0, 1.0),
            OpticalGainAssignment(0, 0.5),
            OpticalGainAssignment(1, 0.25),
            OpticalGainAssignment{Float64}(),
        ],
    )
    workspace = OpticalGainUpdateWorkspace(plan, [1.0, 0.8, 0.8, 3.0])
    outputs = update_outputs(Float64, 2, 4)
    inputs = (
        command_amplitudes=[0.5, 0.5],
        response_amplitudes=[0.25, 0.2],
        optical_gain_estimates=[0.5, 0.4],
        phase_lags=[0.1, -0.1],
        valid=[true, true],
        estimate_complete=[true],
    )
    @test process_update!(outputs, workspace, plan, inputs) === nothing
    expected = [0.5, 0.65, 0.8 + 0.25 * (0.4 - 0.8), 3.0]
    @test outputs.optical_gains == expected
    @test outputs.accepted_dither_modes == [true, true]
    @test outputs.updated_reconstructed_modes == [true, true, true, false]
    @test optical_gains(workspace) == [1.0, 0.8, 0.8, 3.0]
    commit!(workspace)
    @test optical_gains(workspace) == outputs.optical_gains

    inputs.optical_gain_estimates .= [0.8, 0.6]
    process_update!(outputs, workspace, plan, inputs)
    discard!(workspace)
    @test optical_gains(workspace) == expected
    reset!(workspace)
    @test optical_gains(workspace) == [1.0, 0.8, 0.8, 3.0]
end

@testset "every inclusive acceptance condition controls adoption" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(0, 1.0f0)],
    )
    for (gain, phase) in ((0.2f0, 0.25f0), (1.0f0, -0.25f0))
        workspace = OpticalGainUpdateWorkspace(plan, Float32[0.8])
        outputs = update_outputs(Float32, 1, 1)
        inputs = update_inputs(Float32, 1)
        inputs.command_amplitudes[1] = 0.1f0
        inputs.response_amplitudes[1] = 0.05f0
        inputs.optical_gain_estimates[1] = gain
        inputs.phase_lags[1] = phase
        process_update!(outputs, workspace, plan, inputs)
        @test outputs.optical_gains == Float32[gain]
        @test outputs.accepted_dither_modes == [true]
    end

    rejected = (
        (0.09f0, 0.5f0, 0.5f0, 0.0f0, true),
        (0.5f0, 0.04f0, 0.5f0, 0.0f0, true),
        (0.5f0, 0.5f0, 0.19f0, 0.0f0, true),
        (0.5f0, 0.5f0, 1.01f0, 0.0f0, true),
        (0.5f0, 0.5f0, 0.5f0, 0.26f0, true),
        (0.5f0, 0.5f0, 0.5f0, 0.0f0, false),
    )
    for (command, response, gain, phase, valid) in rejected
        workspace = OpticalGainUpdateWorkspace(plan, Float32[0.8])
        outputs = update_outputs(Float32, 1, 1)
        inputs = update_inputs(Float32, 1)
        inputs.command_amplitudes[1] = command
        inputs.response_amplitudes[1] = response
        inputs.optical_gain_estimates[1] = gain
        inputs.phase_lags[1] = phase
        inputs.valid[1] = valid
        process_update!(outputs, workspace, plan, inputs)
        @test outputs.optical_gains == Float32[0.8]
        @test outputs.accepted_dither_modes == [false]
        @test outputs.updated_reconstructed_modes == [false]
    end
end

@testset "numerical rejection preserves output and committed history" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float64, 0.1, 2.0)],
        OpticalGainAssignment{Float64}[OpticalGainAssignment(0, 0.5)],
    )
    workspace = OpticalGainUpdateWorkspace(plan, [1.0])
    outputs = update_outputs(Float64, 1, 1; value=7.0)
    inputs = update_inputs(Float64, 1)
    inputs.optical_gain_estimates[1] = NaN
    before = deepcopy(outputs)
    @test process_update!(outputs, workspace, plan, inputs) ===
          OpticalGainUpdateNonFiniteInput
    @test outputs == before
    commit!(workspace)
    @test optical_gains(workspace) == [1.0]

    minimum = floatmin(Float64)
    edge_plan = OpticalGainUpdatePlan(
        [criterion(Float64, minimum, floatmax(Float64))],
        OpticalGainAssignment{Float64}[OpticalGainAssignment(0, 1.0)],
    )
    edge_workspace = OpticalGainUpdateWorkspace(edge_plan, [floatmax(Float64)])
    edge_outputs = update_outputs(Float64, 1, 1)
    edge_inputs = update_inputs(Float64, 1)
    edge_inputs.optical_gain_estimates[1] = minimum
    @test process_update!(edge_outputs, edge_workspace, edge_plan, edge_inputs) === nothing
    @test edge_outputs.optical_gains == [minimum]
    @test isfinite(1 / edge_outputs.optical_gains[1])
end

@testset "checked storage supports views and safe input aliasing" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(0, 1.0f0)],
    )
    workspace = OpticalGainUpdateWorkspace(plan, Float32[0.8])
    storage = fill(-9.0f0, 2)
    bool_storage = (trues(2), trues(2))
    outputs = (
        optical_gains=@view(storage[1:2:1]),
        accepted_dither_modes=@view(bool_storage[1][1:2:1]),
        updated_reconstructed_modes=@view(bool_storage[2][1:2:1]),
    )
    inputs = update_inputs(Float32, 1)
    inputs = merge(inputs, (optical_gain_estimates=outputs.optical_gains,))
    outputs.optical_gains[1] = 0.5f0
    @test process_update!(outputs, workspace, plan, inputs) === nothing
    @test outputs.optical_gains == Float32[0.5]

    wrong = merge(outputs, (optical_gains=Float32[],))
    @test_throws DimensionMismatch process_update!(wrong, workspace, plan, inputs)
    @test_throws ArgumentError process!(
        ZeroBasedUpdateVector(copy(outputs.optical_gains)),
        outputs.accepted_dither_modes,
        outputs.updated_reconstructed_modes,
        workspace,
        plan,
        inputs.command_amplitudes,
        inputs.response_amplitudes,
        inputs.optical_gain_estimates,
        inputs.phase_lags,
        inputs.valid,
        inputs.estimate_complete,
    )
    @test_throws ArgumentError process!(
        outputs.optical_gains,
        outputs.accepted_dither_modes,
        outputs.accepted_dither_modes,
        workspace,
        plan,
        inputs.command_amplitudes,
        inputs.response_amplitudes,
        inputs.optical_gain_estimates,
        inputs.phase_lags,
        inputs.valid,
        inputs.estimate_complete,
    )
end

@testset "update outcomes and lifecycle allocate no heap after warmup" begin
    plan = OpticalGainUpdatePlan(
        [criterion(Float32)],
        OpticalGainAssignment{Float32}[OpticalGainAssignment(0, 0.5f0)],
    )
    workspace = OpticalGainUpdateWorkspace(plan, Float32[1])
    outputs = update_outputs(Float32, 1, 1)
    inputs = update_inputs(Float32, 1)
    @test direct_update_allocation_bytes(outputs, workspace, plan, inputs) == 0

    inputs.estimate_complete[1] = false
    process_update!(outputs, workspace, plan, inputs)
    @test @allocated(process_update!(outputs, workspace, plan, inputs)) == 0
    inputs.estimate_complete[1] = true
    inputs.optical_gain_estimates[1] = NaN
    process_update!(outputs, workspace, plan, inputs)
    @test @allocated(process_update!(outputs, workspace, plan, inputs)) == 0
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0

    prepared = prepare_algorithm(
        OpticalGainUpdateF32,
        OpticalGainUpdateF32Configuration(
            criteria=[criterion(Float32)],
            assignments=OpticalGainAssignment{Float32}[
                OpticalGainAssignment(0, 0.5f0),
            ],
            initial_optical_gains=Float32[1],
        ),
    )
    inputs.optical_gain_estimates[1] = 0.5f0
    process_update!(outputs, prepared, inputs)
    @test prepared_update_allocation_bytes(outputs, prepared, inputs) == 0
end

@testset "optical-gain update public API exports" begin
    names = (
        :OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1,
        :OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
        :OpticalGainAssignment,
        :OpticalGainUpdateCriteria,
        :OpticalGainUpdateF32,
        :OpticalGainUpdateF32Configuration,
        :OpticalGainUpdateFailure,
        :OpticalGainUpdateNonFiniteInput,
        :OpticalGainUpdateNonFiniteOutput,
        :OpticalGainUpdatePlan,
        :OpticalGainUpdateWorkspace,
        :optical_gain_assignment_dither_mode,
        :optical_gain_assignment_update_fraction,
        :optical_gain_update_assignments,
        :optical_gain_update_criteria,
    )
    @test all(name -> Base.isexported(FilterGraphAlgorithms, name), names)
end

end
