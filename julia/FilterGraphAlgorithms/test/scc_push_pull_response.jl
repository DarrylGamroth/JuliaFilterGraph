module SccPushPullResponseTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct ZeroBasedSccVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedSccVector) = size(vector.parent)
Base.axes(vector::ZeroBasedSccVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedSccVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedSccVector, index::Int) =
    vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedSccVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

scc_prepared(pixel_count::Int=3) = prepare_algorithm(
    SccPushPullResponseF32,
    SccPushPullResponseF32Configuration(pixel_count=pixel_count),
)

function scc_checked_allocation_bytes(response, prepared, inputs...)
    return @allocated process!(response, prepared, inputs...)
end

function scc_admitted_allocation_bytes(response, prepared, inputs...)
    metadata = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated FilterGraphAlgorithms._process_admitted!(
        response,
        prepared,
        inputs...,
        metadata,
        metadata,
        metadata,
        metadata,
    )
end

@testset "SCC four-image declaration preserves the frozen FGN contract" begin
    @test algorithm_label(SccPushPullResponseF32) ==
          "scc-push-pull-response-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        SccPushPullResponseF32,
    ) === SccPushPullResponseFailure
    @test algorithm_property_policy(SccPushPullResponseF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties

    configuration = algorithm_configuration(SccPushPullResponseF32)
    @test map(field -> field.name, configuration) == (:pixel_count,)
    @test only(configuration).update isa GraphRebuildConfiguration

    ports = algorithm_ports(SccPushPullResponseF32)
    @test map(port -> port.name, ports) == (
        Symbol("push-fringed"),
        Symbol("push-unfringed"),
        Symbol("pull-fringed"),
        Symbol("pull-unfringed"),
        :response,
    )
    @test all(port -> port.element_type === Float32, ports)
    @test all(port -> isnothing(port.metadata_source), ports[1:4])
    @test ports[5].metadata_source === Symbol("push-fringed")

    formats = algorithm_formats(scc_prepared())
    @test getproperty.(formats, :shape) == ntuple(_ -> (3,), 5)
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1,
        CALIBRATED_PIXELS_V1,
        CALIBRATED_PIXELS_V1,
        CALIBRATED_PIXELS_V1,
        SCC_PUSH_PULL_RESPONSE_V1,
    )
end

@testset "SCC double difference exactly matches the frozen Rust oracle" begin
    inputs = (
        Float64[10, 5, -1],
        Float64[4, 7, -2],
        Float64[3, -1, 8],
        Float64[2, 2, 3],
    )
    plan = SccPushPullResponsePlan(Float64, 3)
    workspace = SccPushPullResponseWorkspace(plan)
    response = zeros(Float64, 3)
    @test process!(response, workspace, plan, inputs...) === nothing
    @test response == [5.0, 1.0, -4.0]

    prepared = scc_prepared()
    response32 = zeros(Float32, 3)
    inputs32 = map(values -> Float32.(values), inputs)
    @test process!(response32, prepared, inputs32...) === nothing
    @test response32 == Float32[5, 1, -4]
    @test reinterpret(UInt32, response32) ==
          UInt32[0x40a00000, 0x3f800000, 0xc0800000]
end

@testset "SCC rejection preserves output and disposable Workspace" begin
    @test_throws ArgumentError SccPushPullResponsePlan(Float32, 0)
    plan = SccPushPullResponsePlan(Float32, 2)
    workspace = SccPushPullResponseWorkspace(plan)
    response = Float32[71, 72]
    finite = Float32[1, 2]
    zero = zeros(Float32, 2)

    cases = (
        (
            (Float32[NaN, 2], finite, finite, zero),
            SccPushPullNonFinitePushFringed,
        ),
        (
            (finite, Float32[1, Inf], finite, zero),
            SccPushPullNonFinitePushUnfringed,
        ),
        (
            (finite, zero, Float32[-Inf, 1], zero),
            SccPushPullNonFinitePullFringed,
        ),
        (
            (finite, zero, finite, Float32[0, NaN]),
            SccPushPullNonFinitePullUnfringed,
        ),
        (
            (
                Float32[1, floatmax(Float32)],
                Float32[0, -floatmax(Float32)],
                zero,
                zero,
            ),
            SccPushPullNonFiniteCalculation,
        ),
    )
    for (inputs, failure) in cases
        @test process!(response, workspace, plan, inputs...) === failure
        @test response == Float32[71, 72]
    end

    @test commit!(workspace) === workspace
    @test discard!(workspace) === workspace
    @test reset!(workspace) === workspace
    @test workspace.staged_response == zeros(Float32, 2)
end

@testset "SCC storage contract supports CPU views and safe aliasing" begin
    plan = SccPushPullResponsePlan(Float32, 3)
    workspace = SccPushPullResponseWorkspace(plan)
    storage = Float32[99, 10, 5, -1, 99]
    push_fringed = @view storage[2:4]
    response_storage = fill(71.0f0, 7)
    response = @view response_storage[2:2:6]
    inputs = (
        push_fringed,
        @view(Float32[99, 4, 7, -2, 99][2:4]),
        @view(Float32[99, 3, -1, 8, 99][2:4]),
        @view(Float32[99, 2, 2, 3, 99][2:4]),
    )
    @test process!(response, workspace, plan, inputs...) === nothing
    @test response == Float32[5, 1, -4]

    aliased = copy(push_fringed)
    @test process!(aliased, workspace, plan, aliased, inputs[2:4]...) === nothing
    @test aliased == Float32[5, 1, -4]

    @test_throws DimensionMismatch process!(
        Float32[0, 0],
        workspace,
        plan,
        inputs...,
    )
    @test_throws ArgumentError process!(
        ZeroBasedSccVector(zeros(Float32, 3)),
        workspace,
        plan,
        inputs...,
    )
    @test_throws ArgumentError process!(
        workspace.staged_response,
        workspace,
        plan,
        inputs...,
    )
end

@testset "SCC checked and admitted paths allocate nothing after warmup" begin
    prepared = scc_prepared(8)
    response = zeros(Float32, 8)
    finite_inputs = (
        fill(4.0f0, 8),
        fill(1.0f0, 8),
        fill(0.5f0, 8),
        fill(1.5f0, 8),
    )
    rejected_inputs = (
        finite_inputs[1],
        finite_inputs[2],
        fill(Float32(NaN), 8),
        finite_inputs[4],
    )
    process!(response, prepared, finite_inputs...)
    process!(response, prepared, rejected_inputs...)
    FilterGraphAlgorithms._process_admitted!(
        response,
        prepared,
        finite_inputs...,
        ntuple(_ -> FilterGraphAlgorithms._NO_SAMPLE_METADATA, 4)...,
    )
    FilterGraphAlgorithms._process_admitted!(
        response,
        prepared,
        rejected_inputs...,
        ntuple(_ -> FilterGraphAlgorithms._NO_SAMPLE_METADATA, 4)...,
    )

    @test scc_checked_allocation_bytes(response, prepared, finite_inputs...) == 0
    @test scc_checked_allocation_bytes(response, prepared, rejected_inputs...) == 0
    @test scc_admitted_allocation_bytes(response, prepared, finite_inputs...) == 0
    @test scc_admitted_allocation_bytes(response, prepared, rejected_inputs...) == 0
    @test (@allocated commit!(prepared.workspace)) == 0
    @test (@allocated discard!(prepared.workspace)) == 0
    @test (@allocated reset!(prepared.workspace)) == 0
end

@testset "SCC response public API exports" begin
    for name in (
        :SCC_PUSH_PULL_RESPONSE_V1,
        :SccPushPullResponsePlan,
        :SccPushPullResponseWorkspace,
        :SccPushPullResponseFailure,
        :SccPushPullResponseF32,
        :SccPushPullResponseF32Configuration,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
