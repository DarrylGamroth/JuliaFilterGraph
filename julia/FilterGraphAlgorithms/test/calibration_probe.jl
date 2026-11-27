module CalibrationProbeTests

using FilterGraphAlgorithms
using LinearAlgebra
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const AdaptiveOpticsCalibration =
    FilterGraphAlgorithms.AdaptiveOpticsCalibration
const AOCProbeBases = AdaptiveOpticsCalibration.ProbeBases

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "calibration_probe.toml"),
)

struct ZeroBasedProbeVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::ZeroBasedProbeVector) = size(vector.values)
Base.axes(vector::ZeroBasedProbeVector) = (0:(length(vector.values) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedProbeVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedProbeVector, index::Int) = vector.values[index + 1]
Base.setindex!(vector::ZeroBasedProbeVector, value, index::Int) =
    setindex!(vector.values, value, index + 1)

function probe_destinations(::Type{T}, corrector_count) where {T}
    return (
        excitation=zeros(T, corrector_count),
        pattern_index=zeros(UInt64, 1),
        polarity=zeros(UInt8, 1),
        completes_cycle=zeros(Bool, 1),
    )
end

function process_probe!(destinations, workspace, plan)
    return process!(
        destinations.excitation,
        destinations.pattern_index,
        destinations.polarity,
        destinations.completes_cycle,
        workspace,
        plan,
    )
end

function assert_probe_step(destinations, expected)
    @test destinations.excitation == Float32.(expected["excitation"])
    @test collect(reinterpret(UInt32, destinations.excitation)) ==
          UInt32.(expected["bits"])
    @test only(destinations.pattern_index) == UInt64(expected["pattern_index"])
    @test only(destinations.polarity) == UInt8(expected["polarity"])
    @test only(destinations.completes_cycle) == expected["completes_cycle"]
end

function direct_probe_allocation_bytes(destinations, workspace, plan)
    process_probe!(destinations, workspace, plan)
    commit!(workspace)
    return @allocated begin
        process_probe!(destinations, workspace, plan)
        commit!(workspace)
    end
end

function prepared_probe_allocation_bytes(destinations, prepared)
    process!(
        destinations.excitation,
        destinations.pattern_index,
        destinations.polarity,
        destinations.completes_cycle,
        prepared,
    )
    return @allocated process!(
        destinations.excitation,
        destinations.pattern_index,
        destinations.polarity,
        destinations.completes_cycle,
        prepared,
    )
end

function aoc_probe_command_columns(method)
    plan = AdaptiveOpticsCalibration.prepare(method, nothing)
    result = AdaptiveOpticsCalibration.allocate_result(plan)
    workspace = AdaptiveOpticsCalibration.allocate_workspace(plan)
    AdaptiveOpticsCalibration.process!(result, workspace, plan, nothing)
    return permutedims(AOCProbeBases.probe_commands(result))
end

probe_bit_type(::Type{Float32}) = UInt32
probe_bit_type(::Type{Float64}) = UInt64

@testset "structured probe declarations preserve the frozen FGN contract" begin
    cases = (
        (
            declaration=ModalPushPullF32,
            configuration=ModalPushPullF32Configuration(
                mode_count=2,
                corrector_coordinate_count=3,
                initial_positive_commands=Float32.(
                    FIXTURE["modal"]["positive_commands"],
                ),
                excitation_schema="example/test-modal-excitation/1",
            ),
            configuration_names=(
                :mode_count,
                :corrector_coordinate_count,
                :initial_positive_commands,
                :excitation_schema,
            ),
            configuration_updates=(
                GraphRebuildConfiguration,
                GraphRebuildConfiguration,
                ConstructionConfiguration,
                GraphRebuildConfiguration,
            ),
            parameter_name=Symbol("positive-commands"),
            parameter_shape=(2, 3),
            parameter_schema=MODAL_POSITIVE_COMMANDS_V1,
        ),
        (
            declaration=ZonalPushPullF32,
            configuration=ZonalPushPullF32Configuration(
                amplitudes=Float32.(FIXTURE["zonal"]["amplitudes"]),
                excitation_schema="example/test-zonal-excitation/1",
            ),
            configuration_names=(:amplitudes, :excitation_schema),
            configuration_updates=(
                GraphRebuildConfiguration,
                GraphRebuildConfiguration,
            ),
            parameter_name=:amplitudes,
            parameter_shape=(2,),
            parameter_schema=CALIBRATION_PROBE_AMPLITUDES_V1,
        ),
        (
            declaration=HadamardPushPullF32,
            configuration=HadamardPushPullF32Configuration(
                amplitudes=Float32.(FIXTURE["hadamard"]["amplitudes"]),
                excitation_schema="example/test-hadamard-excitation/1",
            ),
            configuration_names=(:amplitudes, :excitation_schema),
            configuration_updates=(
                GraphRebuildConfiguration,
                GraphRebuildConfiguration,
            ),
            parameter_name=:amplitudes,
            parameter_shape=(3,),
            parameter_schema=CALIBRATION_PROBE_AMPLITUDES_V1,
        ),
    )

    expected_labels = (
        "modal-push-pull-f32",
        "zonal-push-pull-f32",
        "hadamard-push-pull-f32",
    )
    for (case, expected_label) in zip(cases, expected_labels)
        @test algorithm_label(case.declaration) == expected_label
        @test FilterGraphAlgorithms.algorithm_failure_type(case.declaration) ===
              Union{}
        configuration = algorithm_configuration(case.declaration)
        @test map(field -> field.name, configuration) == case.configuration_names
        @test map(field -> typeof(field.update), configuration) ==
              case.configuration_updates
        ports = algorithm_ports(case.declaration)
        @test map(port -> port.name, ports) == (
            :excitation,
            Symbol("pattern-index"),
            :polarity,
            Symbol("completes-cycle"),
            case.parameter_name,
        )
        @test all(
            port -> port.direction isa FilterGraphAlgorithms.OutputPort,
            ports[1:4],
        )
        @test ports[5].direction isa FilterGraphAlgorithms.InputPort
        @test ports[5].role isa FilterGraphAlgorithms.SparseParameter

        prepared = prepare_algorithm(case.declaration, case.configuration)
        formats = algorithm_formats(prepared)
        @test map(format -> format.element_type, formats) ==
              (Float32, UInt64, UInt8, Bool, Float32)
        @test map(format -> format.shape, formats)[2:4] == ((1,), (1,), (1,))
        @test formats[5].shape == case.parameter_shape
        @test formats[5].schema == case.parameter_schema
        @test map(format -> format.schema, formats)[2:4] == (
            CALIBRATION_PATTERN_INDEX_V1,
            CALIBRATION_PROBE_POLARITY_V1,
            CALIBRATION_CYCLE_COMPLETE_V1,
        )
    end
end

@testset "structured probe sequences exactly match the frozen Rust oracle" begin
    modal = FIXTURE["modal"]
    plans = (
        ModalPushPullPlan(
            modal["mode_count"],
            modal["corrector_coordinate_count"],
            Float32.(modal["positive_commands"]),
        ),
        ZonalPushPullPlan(Float32.(FIXTURE["zonal"]["amplitudes"])),
        HadamardPushPullPlan(Float32.(FIXTURE["hadamard"]["amplitudes"])),
    )
    fixtures = (modal, FIXTURE["zonal"], FIXTURE["hadamard"])

    for (plan, fixture) in zip(plans, fixtures)
        @test calibration_probe_cycle_length(plan) == length(fixture["steps"])
        workspace = CalibrationProbeWorkspace(plan)
        destinations = probe_destinations(
            Float32,
            calibration_probe_corrector_coordinate_count(plan),
        )
        for expected in fixture["steps"]
            @test process_probe!(destinations, workspace, plan) === nothing
            assert_probe_step(destinations, expected)
            commit!(workspace)
        end
        @test process_probe!(destinations, workspace, plan) === nothing
        assert_probe_step(destinations, first(fixture["steps"]))
    end
end

@testset "probe commands are adopted exactly from Calibration" begin
    for T in (Float32, Float64)
        modal_commands = T[0.5 -1 0; -0.25 0.5 2]
        amplitudes = T[0.5, 1, 2]
        cases = (
            (
                ModalPushPullPlan(modal_commands),
                AOCProbeBases.ModalPushPull(modal_commands),
            ),
            (
                ZonalPushPullPlan(amplitudes),
                AOCProbeBases.ZonalPushPull(amplitudes),
            ),
            (
                HadamardPushPullPlan(amplitudes),
                AOCProbeBases.HadamardPushPull(amplitudes),
            ),
        )
        for (plan, method) in cases
            expected = aoc_probe_command_columns(method)
            @test plan.commands == expected
            @test reinterpret(probe_bit_type(T), vec(plan.commands)) ==
                  reinterpret(probe_bit_type(T), vec(expected))
        end
    end
end

@testset "floating representations and Hadamard balance are preserved" begin
    plan_pairs = (
        (
            ModalPushPullPlan(2, 2, Float32[0.25, -2, 1.5, 0.5]),
            ModalPushPullPlan(2, 2, Float64[0.25, -2, 1.5, 0.5]),
        ),
        (
            ZonalPushPullPlan(Float32[0.25, 2]),
            ZonalPushPullPlan(Float64[0.25, 2]),
        ),
        (
            HadamardPushPullPlan(Float32[0.25, 2]),
            HadamardPushPullPlan(Float64[0.25, 2]),
        ),
    )
    for (plan32, plan64) in plan_pairs
        workspace32 = CalibrationProbeWorkspace(plan32)
        workspace64 = CalibrationProbeWorkspace(plan64)
        output32 = probe_destinations(Float32, 2)
        output64 = probe_destinations(Float64, 2)
        for _ in 1:calibration_probe_cycle_length(plan32)
            process_probe!(output32, workspace32, plan32)
            process_probe!(output64, workspace64, plan64)
            @test Float64.(output32.excitation) == output64.excitation
            @test output32.pattern_index == output64.pattern_index
            @test output32.polarity == output64.polarity
            @test output32.completes_cycle == output64.completes_cycle
            commit!(workspace32)
            commit!(workspace64)
        end
    end

    amplitudes = Float64[1, 0.5, 2, 1.5, 3]
    plan = HadamardPushPullPlan(amplitudes)
    @test calibration_probe_pattern_count(plan) == 8
    workspace = CalibrationProbeWorkspace(plan)
    output = probe_destinations(Float64, length(amplitudes))
    total = zeros(Float64, length(amplitudes))
    gram = zeros(Float64, length(amplitudes), length(amplitudes))
    for _ in 1:calibration_probe_cycle_length(plan)
        process_probe!(output, workspace, plan)
        total .+= output.excitation
        gram .+= output.excitation * transpose(output.excitation)
        commit!(workspace)
    end
    @test total == zeros(Float64, length(amplitudes))
    @test gram == Diagonal(
        calibration_probe_cycle_length(plan) .* amplitudes .^ 2,
    )
end

@testset "probe sequence position uses an Algorithm-local candidate" begin
    plans = (
        ModalPushPullPlan(2, 2, Float32[1, 0, 0, 2]),
        ZonalPushPullPlan(Float32[1, 2]),
        HadamardPushPullPlan(Float32[1, 2]),
    )

    for plan in plans
        workspace = CalibrationProbeWorkspace(plan)
        first_output = probe_destinations(
            Float32,
            calibration_probe_corrector_coordinate_count(plan),
        )
        repeated = probe_destinations(
            Float32,
            calibration_probe_corrector_coordinate_count(plan),
        )
        initial_step = CalibrationProbeStep(
            0,
            CalibrationProbePositive,
            false,
        )
        @test calibration_probe_next_step(workspace) == initial_step
        process_probe!(first_output, workspace, plan)
        process_probe!(repeated, workspace, plan)
        @test repeated == first_output
        @test calibration_probe_next_step(workspace) == initial_step

        discard!(workspace)
        process_probe!(repeated, workspace, plan)
        @test repeated == first_output
        commit!(workspace)
        @test calibration_probe_next_step(workspace) == CalibrationProbeStep(
            0,
            CalibrationProbeNegative,
            false,
        )

        process_probe!(repeated, workspace, plan)
        negative = copy(repeated.excitation)
        discard!(workspace)
        process_probe!(repeated, workspace, plan)
        @test repeated.excitation == negative

        reset!(workspace)
        @test calibration_probe_next_step(workspace) == initial_step
        process_probe!(repeated, workspace, plan)
        @test repeated == first_output
    end
end

@testset "parameter adoption preserves sequence position" begin
    modal = prepare_algorithm(
        ModalPushPullF32,
        ModalPushPullF32Configuration(
            mode_count=2,
            corrector_coordinate_count=2,
            initial_positive_commands=Float32[1, 0, 0, 2],
            excitation_schema="example/test-modal-excitation/1",
        ),
    )
    zonal = prepare_algorithm(
        ZonalPushPullF32,
        ZonalPushPullF32Configuration(
            amplitudes=Float32[1, 2],
            excitation_schema="example/test-zonal-excitation/1",
        ),
    )
    hadamard = prepare_algorithm(
        HadamardPushPullF32,
        HadamardPushPullF32Configuration(
            amplitudes=Float32[1, 2],
            excitation_schema="example/test-hadamard-excitation/1",
        ),
    )
    replacements = (
        Float32[3 0; 0 4],
        Float32[3, 4],
        Float32[3, 4],
    )
    parameter_names = (Symbol("positive-commands"), :amplitudes, :amplitudes)
    expected_negative = (
        Float32[-3, -0.0],
        Float32[-3, 0],
        Float32[-3, -4],
    )

    for (prepared, replacement, name, expected) in zip(
        (modal, zonal, hadamard),
        replacements,
        parameter_names,
        expected_negative,
    )
        destinations = probe_destinations(Float32, 2)
        @test process!(
            destinations.excitation,
            destinations.pattern_index,
            destinations.polarity,
            destinations.completes_cycle,
            prepared,
        ) === nothing
        @test replace_parameter!(prepared, name, replacement) === prepared
        @test calibration_probe_next_step(prepared.workspace).polarity ===
              CalibrationProbeNegative
        process!(
            destinations.excitation,
            destinations.pattern_index,
            destinations.polarity,
            destinations.completes_cycle,
            prepared,
        )
        @test destinations.excitation == expected
        @test destinations.polarity == UInt8[1]
        reset!(prepared)
        process!(
            destinations.excitation,
            destinations.pattern_index,
            destinations.polarity,
            destinations.completes_cycle,
            prepared,
        )
        @test destinations.excitation == -expected
        @test destinations.polarity == UInt8[0]
    end

    @test modal_positive_commands(modal.plan) == Float32[3 0; 0 4]
    @test calibration_probe_amplitudes(zonal.plan) == Float32[3, 4]
    @test calibration_probe_amplitudes(hadamard.plan) == Float32[3, 4]
end

@testset "preparation and storage checks preserve caller ownership" begin
    @test_throws ArgumentError ModalPushPullPlan(0, 1, Float32[])
    @test_throws DimensionMismatch ModalPushPullPlan(2, 2, Float32[1, 2, 3])
    @test_throws ArgumentError ModalPushPullPlan(
        2,
        2,
        Float32[1, 0, 0, -0.0],
    )
    @test_throws ArgumentError ModalPushPullPlan(
        2,
        2,
        Float32[1, 0, 0, NaN],
    )
    for Plan in (ZonalPushPullPlan, HadamardPushPullPlan)
        @test_throws ArgumentError Plan(Float32[])
        for invalid in (0.0f0, -1.0f0, Float32(NaN), Float32(Inf))
            @test_throws ArgumentError Plan(Float32[1, invalid])
        end
    end

    plan = ModalPushPullPlan(2, 2, Float32[1, 0, 0, 2])
    workspace = CalibrationProbeWorkspace(plan)
    destinations = (
        excitation=fill(7.0f0, 1),
        pattern_index=fill(UInt64(8), 1),
        polarity=fill(UInt8(9), 1),
        completes_cycle=fill(true, 1),
    )
    before = deepcopy(destinations)
    @test_throws DimensionMismatch process_probe!(destinations, workspace, plan)
    @test destinations == before
    @test calibration_probe_next_step(workspace) == CalibrationProbeStep(
        0,
        CalibrationProbePositive,
        false,
    )

    valid = probe_destinations(Float32, 2)
    wrong_workspace = CalibrationProbeWorkspace(
        ModalPushPullPlan(1, 1, Float32[1]),
    )
    @test_throws DimensionMismatch process_probe!(valid, wrong_workspace, plan)
    @test valid == probe_destinations(Float32, 2)
    @test_throws ArgumentError process!(
        ZeroBasedProbeVector(zeros(Float32, 2)),
        valid.pattern_index,
        valid.polarity,
        valid.completes_cycle,
        workspace,
        plan,
    )
    @test_throws ArgumentError process!(
        workspace.staged_excitation,
        valid.pattern_index,
        valid.polarity,
        valid.completes_cycle,
        workspace,
        plan,
    )
    @test_throws ArgumentError process!(
        valid.excitation,
        reinterpret(UInt64, workspace.staged_excitation),
        valid.polarity,
        valid.completes_cycle,
        workspace,
        plan,
    )
end

@testset "strided outputs and exact admitted signatures allocate no heap" begin
    plans = (
        ModalPushPullPlan(2, 2, Float32[1, 0, 0, 2]),
        ZonalPushPullPlan(Float32[1, 2]),
        HadamardPushPullPlan(Float32[1, 2]),
    )
    declarations = (ModalPushPullF32, ZonalPushPullF32, HadamardPushPullF32)
    configurations = (
        ModalPushPullF32Configuration(
            mode_count=2,
            corrector_coordinate_count=2,
            initial_positive_commands=Float32[1, 0, 0, 2],
            excitation_schema="example/test-modal-excitation/1",
        ),
        ZonalPushPullF32Configuration(
            amplitudes=Float32[1, 2],
            excitation_schema="example/test-zonal-excitation/1",
        ),
        HadamardPushPullF32Configuration(
            amplitudes=Float32[1, 2],
            excitation_schema="example/test-hadamard-excitation/1",
        ),
    )

    for (plan, declaration, configuration) in
        zip(plans, declarations, configurations)
        excitation_storage = fill(-9.0f0, 4)
        pattern_storage = fill(typemax(UInt64), 2)
        polarity_storage = fill(typemax(UInt8), 2)
        cycle_storage = fill(true, 2)
        destinations = (
            excitation=@view(excitation_storage[1:2:3]),
            pattern_index=@view(pattern_storage[1:2:1]),
            polarity=@view(polarity_storage[1:2:1]),
            completes_cycle=@view(cycle_storage[1:2:1]),
        )
        workspace = CalibrationProbeWorkspace(plan)
        @test (@inferred process_probe!(destinations, workspace, plan)) === nothing
        @test destinations.pattern_index == UInt64[0]
        @test destinations.polarity == UInt8[0]
        @test direct_probe_allocation_bytes(destinations, workspace, plan) == 0
        @test @allocated(commit!(workspace)) == 0
        @test @allocated(discard!(workspace)) == 0
        @test @allocated(reset!(workspace)) == 0

        prepared = prepare_algorithm(declaration, configuration)
        @test (@inferred process!(
            destinations.excitation,
            destinations.pattern_index,
            destinations.polarity,
            destinations.completes_cycle,
            prepared,
        )) === nothing
        @test prepared_probe_allocation_bytes(destinations, prepared) == 0
    end
end

end
