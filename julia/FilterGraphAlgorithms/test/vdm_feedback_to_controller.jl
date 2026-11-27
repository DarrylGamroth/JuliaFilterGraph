module VdmFeedbackToControllerTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms: process!

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "vdm_feedback_to_controller.toml"),
)

function row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function fixture_projection()
    configuration = FIXTURE["configuration"]
    return row_major_matrix(
        Float32,
        FIXTURE["parameter"]["vdm_to_controller"],
        configuration["controller_command_extent"],
        configuration["vdm_size"],
    )
end

function fixture_prepared()
    configuration = FIXTURE["configuration"]
    prepared = prepare_algorithm(
        VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(
            controller_command_count=configuration["controller_command_extent"],
            active_vdm_count=configuration["vdm_size"],
        ),
    )
    replace_parameter!(prepared, "vdm-to-controller", fixture_projection())
    return prepared
end

function vdm_feedback_to_controller_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function vdm_feedback_to_controller_admitted_allocation_bytes(
    output,
    prepared,
    input,
)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "VDM-feedback declaration preserves the FGN contract" begin
    @test algorithm_label(VdmFeedbackToControllerF32) ==
          "vdm-feedback-to-controller-f32"
    @test algorithm_property_policy(VdmFeedbackToControllerF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    @test map(
        field -> field.name,
        algorithm_configuration(VdmFeedbackToControllerF32),
    ) == (:controller_command_count, :active_vdm_count)

    ports = algorithm_ports(VdmFeedbackToControllerF32)
    @test map(port -> port.name, ports) == (
        Symbol("vdm-constraint-feedback"),
        Symbol("controller-constraint-feedback"),
        Symbol("vdm-to-controller"),
    )
    @test ports[1].role isa FilterGraphAlgorithms.FrameData
    @test ports[2].role isa FilterGraphAlgorithms.FrameData
    @test ports[3].role isa FilterGraphAlgorithms.SparseParameter
    @test ports[2].metadata_source == Symbol("vdm-constraint-feedback")

    prepared = fixture_prepared()
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) == ((2,), (2,), (2, 2))
    @test getproperty.(formats, :schema) == (
        VDM_CONSTRAINT_FEEDBACK_V1,
        CONTROLLER_CONSTRAINT_FEEDBACK_V1,
        VDM_TO_CONTROLLER_PROJECTION_V1,
    )
    @test all(format -> format.layout isa ColumnMajorLayout, formats)

    output = zeros(Float32, 2)
    for sample in FIXTURE["samples"]
        input = Float32.(sample["vdm_constraint_feedback"])
        @test process!(output, prepared, input) === nothing
        @test output ≈ Float32.(sample["controller_constraint_feedback"])
    end
end

@testset "VDM feedback accepts logical CPU views" begin
    prepared = fixture_prepared()
    input_storage = fill(99.0f0, 4)
    output_storage = fill(71.0f0, 4)
    input = @view input_storage[1:2:3]
    output = @view output_storage[2:2:4]
    input .= Float32[0.3, -0.2]

    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.325, -0.175]
    @test output_storage[1:2:3] == fill(71.0f0, 2)

    projection_storage = fill(NaN32, 4, 4)
    projection = @view projection_storage[1:2:3, 2:2:4]
    projection .= fixture_projection()
    @test replace_parameter!(prepared, "vdm-to-controller", projection) === prepared
    @test process!(output, prepared, input) === nothing
    @test output ≈ Float32[0.325, -0.175]
end

@testset "VDM feedback rejects invalid values failure-atomically" begin
    @test_throws ArgumentError VdmFeedbackToControllerPlan(zeros(Float32, 0, 2))
    @test_throws ArgumentError VdmFeedbackToControllerPlan(
        Float32[1 NaN; 0 1],
    )
    @test_throws ArgumentError prepare_algorithm(
        VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(
            controller_command_count=0,
            active_vdm_count=2,
        ),
    )
    @test_throws ArgumentError prepare_algorithm(
        VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(
            controller_command_count=typemax(Int),
            active_vdm_count=2,
        ),
    )

    prepared = fixture_prepared()
    original_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "vdm-to-controller",
        zeros(Float32, 1, 2),
    )
    @test prepared.plan === original_plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        "vdm-to-controller",
        Float32[1 NaN; 0 1],
    )
    @test prepared.plan === original_plan

    output = fill(71.0f0, 2)
    @test process!(output, prepared, Float32[NaN, 0]) ===
          VdmFeedbackToControllerNonFiniteInput
    @test output == fill(71.0f0, 2)
    @test_throws DimensionMismatch process!(
        output,
        prepared,
        zeros(Float32, 1),
    )
    @test_throws ArgumentError process!(output, prepared, output)

    overflowing = prepare_algorithm(
        VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(
            controller_command_count=2,
            active_vdm_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "vdm-to-controller",
        fill(floatmax(Float32), 2, 2),
    )
    @test process!(output, overflowing, Float32[2, 2]) ===
          VdmFeedbackToControllerNonFiniteOutput
end

@testset "VDM-feedback warmed paths allocate no Julia heap memory" begin
    prepared = fixture_prepared()
    output = zeros(Float32, 2)
    valid = Float32[0.3, -0.2]
    nonfinite = Float32[NaN, 0]

    process!(output, prepared, valid)
    @test vdm_feedback_to_controller_allocation_bytes(
        output,
        prepared,
        valid,
    ) == 0
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        valid,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    @test vdm_feedback_to_controller_admitted_allocation_bytes(
        output,
        prepared,
        valid,
    ) == 0
    process!(output, prepared, nonfinite)
    @test vdm_feedback_to_controller_allocation_bytes(
        output,
        prepared,
        nonfinite,
    ) == 0

    overflowing = prepare_algorithm(
        VdmFeedbackToControllerF32,
        VdmFeedbackToControllerF32Configuration(
            controller_command_count=2,
            active_vdm_count=2,
        ),
    )
    replace_parameter!(
        overflowing,
        "vdm-to-controller",
        fill(floatmax(Float32), 2, 2),
    )
    large = Float32[2, 2]
    process!(output, overflowing, large)
    @test vdm_feedback_to_controller_allocation_bytes(
        output,
        overflowing,
        large,
    ) == 0
end

end
