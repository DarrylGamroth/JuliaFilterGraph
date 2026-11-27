module OpticalGainDitherTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct ZeroBasedDitherVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
end

Base.size(vector::ZeroBasedDitherVector) = size(vector.values)
Base.axes(vector::ZeroBasedDitherVector) = (0:(length(vector.values) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedDitherVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedDitherVector, index::Int) = vector.values[index + 1]
Base.setindex!(vector::ZeroBasedDitherVector, value, index::Int) =
    setindex!(vector.values, value, index + 1)

dither_outputs(::Type{T}, modes) where {T} = (
    commanded_modes=zeros(T, modes),
    window_frame=zeros(UInt64, 1),
    phase_indices=zeros(UInt64, modes),
    completes_window=zeros(Bool, 1),
)

function process_dither!(outputs, workspace, plan)
    return process!(
        outputs.commanded_modes,
        outputs.window_frame,
        outputs.phase_indices,
        outputs.completes_window,
        workspace,
        plan,
    )
end

function process_dither!(outputs, prepared)
    return process!(
        outputs.commanded_modes,
        outputs.window_frame,
        outputs.phase_indices,
        outputs.completes_window,
        prepared,
    )
end

function direct_dither_allocation_bytes(outputs, workspace, plan)
    process_dither!(outputs, workspace, plan)
    commit!(workspace)
    return @allocated begin
        process_dither!(outputs, workspace, plan)
        commit!(workspace)
    end
end

function prepared_dither_allocation_bytes(outputs, prepared)
    process_dither!(outputs, prepared)
    return @allocated process_dither!(outputs, prepared)
end

@testset "optical-gain dither declaration preserves the frozen FGN contract" begin
    @test algorithm_label(OpticalGainDitherF32) == "optical-gain-dither-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(OpticalGainDitherF32) === Union{}
    configuration = algorithm_configuration(OpticalGainDitherF32)
    @test map(field -> field.name, configuration) == (:modes,)
    @test only(configuration).update isa GraphRebuildConfiguration

    ports = algorithm_ports(OpticalGainDitherF32)
    @test map(port -> port.name, ports) == (
        Symbol("commanded-modes"),
        Symbol("window-frame"),
        Symbol("phase-indices"),
        Symbol("completes-window"),
    )
    @test all(port -> port.direction isa FilterGraphAlgorithms.OutputPort, ports)

    prepared = prepare_algorithm(
        OpticalGainDitherF32,
        OpticalGainDitherF32Configuration(
            modes=[OpticalGainDitherMode(0.5f0, 4, 1)],
        ),
    )
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :element_type) == (Float32, UInt64, UInt64, Bool)
    @test getproperty.(formats, :shape) == ((1,), (1,), (1,), (1,))
    @test getproperty.(formats, :schema) == (
        OPTICAL_GAIN_DITHER_COMMAND_V1,
        OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1,
        OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
        OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
    )
end

@testset "dither schedule validates synchronized modes" begin
    @test_throws ArgumentError OpticalGainDitherSchedule(OpticalGainDitherMode{Float32}[])
    for mode in (
        OpticalGainDitherMode(0.0f0, 4, 1),
        OpticalGainDitherMode(-1.0f0, 4, 1),
        OpticalGainDitherMode(NaN32, 4, 1),
        OpticalGainDitherMode(Inf32, 4, 1),
        OpticalGainDitherMode(1.0f0, 3, 1),
        OpticalGainDitherMode(1.0f0, 4, 0),
    )
        @test_throws ArgumentError OpticalGainDitherSchedule([mode])
    end
    @test_throws ArgumentError OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, 4, 2),
        OpticalGainDitherMode(1.0f0, 5, 2),
    ])
    @test_throws OverflowError OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0f0, typemax(Int), 2),
    ])

    schedule = OpticalGainDitherSchedule([
        OpticalGainDitherMode(1.0, 4, 3),
        OpticalGainDitherMode(0.5, 6, 2),
    ])
    @test schedule.frames_per_estimate == 12
    @test optical_gain_dither_modes(schedule) == [
        OpticalGainDitherMode(1.0, 4, 3),
        OpticalGainDitherMode(0.5, 6, 2),
    ]
    @test optical_gain_dither_cosines(schedule, 1) ≈ [1.0, 0.0, -1.0, 0.0] atol=32eps()
    @test optical_gain_dither_sines(schedule, 1) ≈ [0.0, 1.0, 0.0, -1.0] atol=32eps()
    @test_throws BoundsError optical_gain_dither_cosines(schedule, 0)

    copied = optical_gain_dither_modes(schedule)
    copied[1] = OpticalGainDitherMode(9.0, 4, 3)
    @test first(optical_gain_dither_modes(schedule)).amplitude == 1.0
end

@testset "dither sequence and state match the frozen Rust oracle" begin
    plan = OpticalGainDitherPlan(
        OpticalGainDitherSchedule([
            OpticalGainDitherMode(1.0, 4, 3),
            OpticalGainDitherMode(0.5, 6, 2),
        ]),
    )
    workspace = OpticalGainDitherWorkspace(plan)
    outputs = dither_outputs(Float64, 2)
    for frame in 0:11
        @test process_dither!(outputs, workspace, plan) === nothing
        @test outputs.commanded_modes[1] ≈ cos(2pi * frame / 4) atol=32eps()
        @test outputs.commanded_modes[2] ≈ 0.5cos(2pi * frame / 6) atol=32eps()
        @test outputs.window_frame == UInt64[frame]
        @test outputs.phase_indices == UInt64[frame % 4, frame % 6]
        @test outputs.completes_window == [frame == 11]
        @test optical_gain_dither_frame(workspace) == frame
        commit!(workspace)
    end
    @test optical_gain_dither_phases(workspace) == [0, 0]
    @test optical_gain_dither_frame(workspace) == 0

    plan32 = OpticalGainDitherPlan(
        OpticalGainDitherSchedule([OpticalGainDitherMode(0.25f0, 4, 1)]),
    )
    workspace32 = OpticalGainDitherWorkspace(plan32)
    outputs32 = dither_outputs(Float32, 1)
    process_dither!(outputs32, workspace32, plan32)
    @test outputs32.commanded_modes == Float32[0.25]
    commit!(workspace32)
    process_dither!(outputs32, workspace32, plan32)
    @test abs(only(outputs32.commanded_modes)) <= 2eps(Float32)
end

@testset "dither position uses an Algorithm-local candidate" begin
    plan = OpticalGainDitherPlan(
        OpticalGainDitherSchedule([OpticalGainDitherMode(1.0f0, 4, 1)]),
    )
    workspace = OpticalGainDitherWorkspace(plan)
    outputs = dither_outputs(Float32, 1)

    process_dither!(outputs, workspace, plan)
    @test outputs.commanded_modes == Float32[1]
    @test optical_gain_dither_phases(workspace) == [0]
    @test optical_gain_dither_frame(workspace) == 0
    process_dither!(outputs, workspace, plan)
    @test outputs.commanded_modes == Float32[1]
    discard!(workspace)
    process_dither!(outputs, workspace, plan)
    @test outputs.commanded_modes == Float32[1]
    commit!(workspace)
    @test optical_gain_dither_phases(workspace) == [1]
    @test optical_gain_dither_frame(workspace) == 1

    process_dither!(outputs, workspace, plan)
    @test abs(only(outputs.commanded_modes)) <= 2eps(Float32)
    discard!(workspace)
    process_dither!(outputs, workspace, plan)
    @test abs(only(outputs.commanded_modes)) <= 2eps(Float32)
    reset!(workspace)
    @test optical_gain_dither_phases(workspace) == [0]
    @test optical_gain_dither_frame(workspace) == 0
end

@testset "dither storage checks preserve output and state" begin
    plan = OpticalGainDitherPlan(
        OpticalGainDitherSchedule([
            OpticalGainDitherMode(1.0f0, 4, 1),
            OpticalGainDitherMode(0.5f0, 4, 1),
        ]),
    )
    workspace = OpticalGainDitherWorkspace(plan)
    wrong = dither_outputs(Float32, 1)
    before = deepcopy(wrong)
    @test_throws DimensionMismatch process_dither!(wrong, workspace, plan)
    @test wrong == before
    @test optical_gain_dither_phases(workspace) == [0, 0]

    outputs = dither_outputs(Float32, 2)
    @test_throws ArgumentError process!(
        ZeroBasedDitherVector(zeros(Float32, 2)),
        outputs.window_frame,
        outputs.phase_indices,
        outputs.completes_window,
        workspace,
        plan,
    )
    @test_throws ArgumentError process!(
        outputs.commanded_modes,
        reinterpret(UInt64, outputs.commanded_modes),
        outputs.phase_indices,
        outputs.completes_window,
        workspace,
        plan,
    )
    protected = @view reinterpret(Float32, workspace.committed_phases)[1:2]
    @test_throws ArgumentError process!(
        protected,
        outputs.window_frame,
        outputs.phase_indices,
        outputs.completes_window,
        workspace,
        plan,
    )
    @test optical_gain_dither_phases(workspace) == [0, 0]
end

@testset "dither views and admitted lifecycle allocate no heap" begin
    modes = [
        OpticalGainDitherMode(1.0f0, 4, 3),
        OpticalGainDitherMode(0.5f0, 6, 2),
    ]
    plan = OpticalGainDitherPlan(OpticalGainDitherSchedule(modes))
    workspace = OpticalGainDitherWorkspace(plan)
    command_storage = fill(-9.0f0, 4)
    frame_storage = fill(typemax(UInt64), 2)
    phase_storage = fill(typemax(UInt64), 4)
    completion_storage = fill(true, 2)
    outputs = (
        commanded_modes=@view(command_storage[1:2:3]),
        window_frame=@view(frame_storage[1:2:1]),
        phase_indices=@view(phase_storage[1:2:3]),
        completes_window=@view(completion_storage[1:2:1]),
    )
    process_dither!(outputs, workspace, plan)
    commit!(workspace)
    @test outputs.commanded_modes ≈ Float32[1, 0.5] atol=4eps(Float32)
    @test direct_dither_allocation_bytes(outputs, workspace, plan) == 0
    @test @allocated(commit!(workspace)) == 0
    @test @allocated(discard!(workspace)) == 0
    @test @allocated(reset!(workspace)) == 0

    prepared = prepare_algorithm(
        OpticalGainDitherF32,
        OpticalGainDitherF32Configuration(modes=modes),
    )
    @test prepared_dither_allocation_bytes(outputs, prepared) == 0
end

@testset "optical-gain dither public API exports" begin
    names = (
        :MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD,
        :OPTICAL_GAIN_DITHER_COMMAND_V1,
        :OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
        :OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
        :OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1,
        :OpticalGainDitherF32,
        :OpticalGainDitherF32Configuration,
        :OpticalGainDitherMode,
        :OpticalGainDitherPlan,
        :OpticalGainDitherSchedule,
        :OpticalGainDitherWorkspace,
        :optical_gain_dither_cosines,
        :optical_gain_dither_frame,
        :optical_gain_dither_modes,
        :optical_gain_dither_phases,
        :optical_gain_dither_sines,
    )
    @test all(name -> Base.isexported(FilterGraphAlgorithms, name), names)
end

end
