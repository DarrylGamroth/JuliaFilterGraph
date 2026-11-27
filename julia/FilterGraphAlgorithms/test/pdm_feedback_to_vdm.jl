module PdmFeedbackToVdmTests

using FilterGraphAlgorithms
using LinearAlgebra: I
using Test
using TOML

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pdm_feedback_to_vdm.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_parameters()
    configuration = FIXTURE["configuration"]
    parameters = FIXTURE["parameters"]
    full_to_active = row_major_matrix(
        Float32,
        parameters["full_to_active"],
        configuration["active_vdm_size"],
        configuration["full_vdm_size"],
    )
    pdm_to_vdm = row_major_matrix(
        Float32,
        parameters["pdm_to_vdm"],
        configuration["full_vdm_size"],
        configuration["physical_actuator_extent"],
    )
    return full_to_active, pdm_to_vdm
end

function fixture_prepared()
    configuration = FIXTURE["configuration"]
    prepared = prepare_algorithm(
        PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(
            active_vdm_count=configuration["active_vdm_size"],
            full_vdm_count=configuration["full_vdm_size"],
            physical_actuator_count=configuration["physical_actuator_extent"],
        ),
    )
    full_to_active, pdm_to_vdm = fixture_parameters()
    replace_parameter!(prepared, "full-to-active", full_to_active)
    replace_parameter!(prepared, "pdm-to-vdm", pdm_to_vdm)
    return prepared
end

function pdm_feedback_to_vdm_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function pdm_feedback_to_vdm_admitted_allocation_bytes(output, prepared, input)
    metadata = FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        metadata,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "PDM-feedback declaration preserves the FGN contract" begin
    @test algorithm_label(PdmFeedbackToVdmF32) == "pdm-feedback-to-vdm-f32"
    @test algorithm_property_policy(PdmFeedbackToVdmF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    @test map(
        field -> field.name,
        algorithm_configuration(PdmFeedbackToVdmF32),
    ) == (:active_vdm_count, :full_vdm_count, :physical_actuator_count, :sparse_feedback)
    @test PdmFeedbackToVdmF32Configuration(2, 3, 2).sparse_feedback === false

    ports = algorithm_ports(PdmFeedbackToVdmF32)
    @test map(port -> port.name, ports) == (
        Symbol("pdm-constraint-feedback"),
        Symbol("vdm-constraint-feedback"),
        Symbol("full-to-active"),
        Symbol("pdm-to-vdm"),
    )
    @test ports[2].metadata_source == Symbol("pdm-constraint-feedback")
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    @test ports[4].role isa FilterGraphAlgorithms.SparseParameter

    prepared = fixture_prepared()
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2,), (2,), (2, 3), (3, 2))
    @test getproperty.(formats, :schema) == (
        PDM_CONSTRAINT_FEEDBACK_V1,
        VDM_CONSTRAINT_FEEDBACK_V1,
        FULL_TO_ACTIVE_VDM_PROJECTION_V1,
        PDM_TO_VDM_PROJECTION_V1,
    )

    output = zeros(Float32, 2)
    for sample in FIXTURE["samples"]
        input = Float32.(sample["pdm_constraint_feedback"])
        @test process!(output, prepared, input) === nothing
        @test output ≈ Float32.(sample["vdm_constraint_feedback"])
    end
end

@testset "PDM feedback accepts logical CPU views and safe input aliasing" begin
    prepared = fixture_prepared()
    input_storage = fill(99.0f0, 4)
    output_storage = fill(71.0f0, 4)
    input = @view input_storage[1:2:3]
    output = @view output_storage[2:2:4]
    input .= Float32[0.4, -0.75]

    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.51875, -0.9875]
    @test output_storage[1:2:3] == fill(71.0f0, 2)

    feedback = Float32[0.4, -0.75]
    @test process!(feedback, prepared, feedback) === nothing
    @test feedback ≈ Float32[0.51875, -0.9875]

    full_to_active, pdm_to_vdm = fixture_parameters()
    active_storage = fill(NaN32, 4, 5)
    active_view = @view active_storage[1:2:3, 1:2:5]
    active_view .= full_to_active
    pdm_storage = fill(NaN32, 5, 4)
    pdm_view = @view pdm_storage[1:2:5, 2:2:4]
    pdm_view .= pdm_to_vdm
    @test replace_parameter!(prepared, "full-to-active", active_view) === prepared
    @test replace_parameter!(prepared, "pdm-to-vdm", pdm_view) === prepared
    input .= Float32[0.4, -0.75]
    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.51875, -0.9875]
end

@testset "PDM-feedback replacement and rejection are failure-atomic" begin
    @test_throws ArgumentError PdmFeedbackToVdmPlan(
        zeros(Float32, 2, 0),
        zeros(Float32, 0, 2),
    )
    @test_throws DimensionMismatch PdmFeedbackToVdmPlan(
        zeros(Float32, 2, 3),
        zeros(Float32, 2, 2),
    )
    @test_throws ArgumentError prepare_algorithm(
        PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(
            active_vdm_count=2,
            full_vdm_count=0,
            physical_actuator_count=2,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(
            active_vdm_count=typemax(Int),
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )

    prepared = fixture_prepared()
    original_plan = prepared.plan
    invalid_full_to_active = first(fixture_parameters())
    invalid_full_to_active[1] = NaN32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "full-to-active",
        invalid_full_to_active,
    )
    @test prepared.plan === original_plan
    invalid_pdm_to_vdm = last(fixture_parameters())
    invalid_pdm_to_vdm[1] = Inf32
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "pdm-to-vdm",
        invalid_pdm_to_vdm,
    )
    @test prepared.plan === original_plan

    output = fill(71.0f0, 2)
    @test process!(output, prepared, Float32[NaN, 0]) ===
          PdmFeedbackToVdmNonFiniteInput
    @test output == fill(71.0f0, 2)

    overflowing = prepare_algorithm(
        PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(
            active_vdm_count=2,
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "pdm-to-vdm",
        fill(floatmax(Float32), 3, 2),
    )
    replace_parameter!(overflowing, "full-to-active", first(fixture_parameters()))
    @test process!(output, overflowing, Float32[2, 2]) ===
          PdmFeedbackToVdmNonFiniteOutput
end

@testset "PDM-feedback warmed paths allocate no Julia heap memory" begin
    prepared = fixture_prepared()
    workspace = prepared.workspace
    output = zeros(Float32, 2)
    valid = Float32[0.4, -0.75]
    nonfinite = Float32[NaN, 0]

    process!(output, prepared, valid)
    @test pdm_feedback_to_vdm_allocation_bytes(output, prepared, valid) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        valid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test pdm_feedback_to_vdm_admitted_allocation_bytes(
        output,
        prepared,
        valid,
    ) == 0
    process!(output, prepared, nonfinite)
    @test pdm_feedback_to_vdm_allocation_bytes(output, prepared, nonfinite) == 0
    @test workspace_commit_allocation_bytes(workspace) == 0
    @test workspace_discard_allocation_bytes(workspace) == 0
    @test workspace_reset_allocation_bytes(workspace) == 0

    overflowing = prepare_algorithm(
        PdmFeedbackToVdmF32,
        PdmFeedbackToVdmF32Configuration(
            active_vdm_count=2,
            full_vdm_count=3,
            physical_actuator_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "pdm-to-vdm",
        fill(floatmax(Float32), 3, 2),
    )
    replace_parameter!(overflowing, "full-to-active", first(fixture_parameters()))
    large = Float32[2, 2]
    process!(output, overflowing, large)
    @test pdm_feedback_to_vdm_allocation_bytes(output, overflowing, large) == 0
end

@testset "Opt-in sparse feedback preserves dense fallback" begin
    FGA = FilterGraphAlgorithms
    count = 512
    identity = Matrix{Float32}(I, count, count)
    projection = fill(1f0, count, count)
    projection[1, 2] = -prevfloat(1f0)
    projection[1, 3] = -0.75f0
    projection[2, 3] = 0f0

    function prepare_feedback(sparse_feedback)
        configuration = PdmFeedbackToVdmF32Configuration(
            active_vdm_count=count,
            full_vdm_count=count,
            physical_actuator_count=count,
            sparse_feedback=sparse_feedback,
        )
        prepared = prepare_algorithm(PdmFeedbackToVdmF32, configuration)
        replace_parameter!(prepared, "full-to-active", projection)
        @test prepared.plan.sparse_execution === nothing
        replace_parameter!(prepared, "pdm-to-vdm", identity)
        return prepared
    end

    dense = prepare_feedback(false)
    sparse = prepare_feedback(true)
    other_worker = prepare_feedback(true)
    @test dense.workspace.sparse === nothing
    @test sparse.plan.sparse_execution !== nothing
    @test sparse.workspace.sparse.columns !== other_worker.workspace.sparse.columns
    @test sparse.workspace.sparse.sum_abs !== other_worker.workspace.sparse.sum_abs
    @test sparse.plan.full_to_active_execution isa FGA.DenseProjection
    @test sparse.plan.pdm_to_vdm_execution isa FGA.IdentityProjection

    output = zeros(Float32, count)
    expected = similar(output)
    function compare_feedback(input, admitted)
        plan = sparse.plan
        @test FGA._try_sparse_feedback!(
            output, plan.full_to_active, input, input, sparse.workspace.sparse,
            plan.sparse_execution,
        ) === admitted
        @test process!(expected, dense, input) === nothing
        @test process!(output, sparse, input) === nothing
        if admitted
            @test all(abs.(output .- expected) .<= 1f-11 .+ 2f-4 .* abs.(expected))
        else
            @test reinterpret(UInt32, output) == reinterpret(UInt32, expected)
        end
    end

    empty = zeros(Float32, count)
    compare_feedback(empty, true)
    @test all(iszero, reinterpret(UInt32, output))

    positive = copy(empty)
    positive[3] = 1f-10
    compare_feedback(positive, true)
    @test reinterpret(UInt32, output) == reinterpret(UInt32, expected)
    @test output[1] < 0f0
    @test reinterpret(UInt32, output[2]) == 0x00000000
    @test pdm_feedback_to_vdm_allocation_bytes(output, sparse, positive) == 0

    negative = copy(empty)
    negative[3] = -positive[3]
    compare_feedback(negative, false)

    valid = copy(empty)
    valid[3:66] .= 1f-10
    compare_feedback(valid, true)
    @test pdm_feedback_to_vdm_allocation_bytes(output, sparse, valid) == 0
    FGA._process_admitted!(output, sparse, valid, FGA._NO_SAMPLE_METADATA)
    @test pdm_feedback_to_vdm_admitted_allocation_bytes(output, sparse, valid) == 0

    too_many = copy(valid)
    too_many[67] = valid[3]
    compare_feedback(too_many, false)
    @test pdm_feedback_to_vdm_allocation_bytes(output, sparse, too_many) == 0
    for value in (-0f0, nextfloat(0f0), floatmin(Float32), 1f0)
        input = copy(empty)
        input[3] = value
        compare_feedback(input, false)
    end

    cancellation = copy(empty)
    cancellation[1:2] .= 1f-10
    compare_feedback(cancellation, false)
    nonfinite = copy(too_many)
    nonfinite[end] = NaN32
    fill!(output, 17f0)
    @test process!(output, sparse, nonfinite) === PdmFeedbackToVdmNonFiniteInput
    @test all(==(17f0), output)
    extreme = fill(floatmax(Float32), count)
    @test process!(expected, dense, extreme) === PdmFeedbackToVdmNonFiniteOutput
    @test process!(output, sparse, extreme) === PdmFeedbackToVdmNonFiniteOutput
    @test isequal(output, expected)

    input_storage = zeros(Float32, 2count)
    input_view = @view input_storage[1:2:end]
    input_view .= valid
    output_view = @view input_storage[2:2:end]
    @test process!(expected, dense, input_view) === nothing
    @test process!(output_view, sparse, input_view) === nothing
    @test isequal(output_view, expected)
    aliased = copy(valid)
    @test process!(aliased, sparse, aliased) === nothing
    @test all(abs.(aliased .- expected) .<= 1f-11 .+ 2f-4 .* abs.(expected))
    aliased_scratch = sparse.workspace.sparse.sum_abs
    @test process!(aliased_scratch, sparse, valid) === nothing
    @test isequal(aliased_scratch, expected)

    original_subnormals = get_zero_subnormals()
    try
        if set_zero_subnormals(true) && get_zero_subnormals()
            compare_feedback(valid, false)
            compare_feedback(positive, false)
        end
    finally
        set_zero_subnormals(original_subnormals)
    end
    Base.Rounding.setrounding_raw(Float32, Base.Rounding.to_fenv(RoundDown)) do
        compare_feedback(valid, false)
        compare_feedback(positive, false)
    end

    workspace = sparse.workspace
    invalid_calibration = copy(projection)
    invalid_calibration[1, 1] = nextfloat(0f0)
    replace_parameter!(sparse, "full-to-active", invalid_calibration)
    @test sparse.plan.sparse_execution === nothing
    invalid_calibration[1, 1] = -0f0
    replace_parameter!(sparse, "full-to-active", invalid_calibration)
    @test sparse.plan.sparse_execution === nothing
    replace_parameter!(sparse, "full-to-active", projection)
    @test sparse.plan.sparse_execution !== nothing
    replace_parameter!(sparse, "pdm-to-vdm", Float32(2) .* identity)
    @test sparse.plan.sparse_execution === nothing
    replace_parameter!(sparse, "pdm-to-vdm", identity)
    @test sparse.plan.sparse_execution !== nothing
    @test sparse.workspace === workspace
    reset!(workspace)
    compare_feedback(valid, true)
end

end
