module LeakyIntegratorPropertyTests

using FilterGraphAlgorithms
using Test
using TOML

import FilterGraphAlgorithms:
    adopt_properties!,
    commit!,
    commit_properties!,
    discard_properties!,
    prepare_properties,
    prepare_properties!,
    process!,
    property_schema,
    property_values,
    replace_properties

const FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "leaky_integrator_properties.toml"),
)

struct UpdateClassPlan <: AbstractAlgorithmPlan
    mode::UInt32
    construction::Int32
    graph_rebuild::Int32
end

struct DuplicateSchemaPlan <: AbstractAlgorithmPlan
    first::Int32
    second::Int32
end

struct MixedPlan <: AbstractAlgorithmPlan
    enabled::Bool
    gain::Float32
end

property_schema(::Type{MixedPlan}) = (
    runtime_property(20, :enabled, Bool; default=false, description="Enabled", unit="1"),
    runtime_property(21, :gain, Float32; default=1.0f0, description="Gain", unit="1"),
)
property_values(plan::MixedPlan) = (enabled=plan.enabled, gain=plan.gain)
replace_properties(::MixedPlan, values) = MixedPlan(values.enabled, values.gain)

struct CoupledPlan <: AbstractAlgorithmPlan
    lower::Int32
    upper::Int32

    function CoupledPlan(lower::Int32, upper::Int32)
        lower <= upper || throw(ArgumentError("lower must not exceed upper"))
        return new(lower, upper)
    end
end

property_schema(::Type{CoupledPlan}) = (
    runtime_property(
        10,
        :lower,
        Int32;
        default=Int32(0),
        description="Lower admitted value",
        unit="1",
    ),
    runtime_property(
        11,
        :upper,
        Int32;
        default=Int32(1),
        description="Upper admitted value",
        unit="1",
    ),
)
property_values(plan::CoupledPlan) = (lower=plan.lower, upper=plan.upper)
replace_properties(::CoupledPlan, values) = CoupledPlan(values.lower, values.upper)

property_schema(::Type{DuplicateSchemaPlan}) = (
    runtime_property(
        5,
        :first,
        Int32;
        default=Int32(0),
        description="First value",
        unit="1",
    ),
    runtime_property(
        5,
        :second,
        Int32;
        default=Int32(0),
        description="Second value",
        unit="1",
    ),
)
property_values(plan::DuplicateSchemaPlan) =
    (first=plan.first, second=plan.second)
replace_properties(::DuplicateSchemaPlan, values) =
    DuplicateSchemaPlan(values.first, values.second)

property_schema(::Type{UpdateClassPlan}) = (
    read_only_property(
        7,
        :mode,
        UInt32;
        default=UInt32(1),
        description="Mode selector",
        unit="1",
        constraint=PropertyChoices(
            (
                PropertyChoice(UInt32(1), :one, "first"),
                PropertyChoice(UInt32(2), :two, "second"),
            ),
        ),
    ),
    construction_property(
        8,
        :construction,
        Int32;
        default=Int32(1),
        description="Construction-only value",
        unit="1",
    ),
    graph_rebuild_property(
        9,
        :graph_rebuild,
        Int32;
        default=Int32(1),
        description="Graph-defining value",
        unit="1",
    ),
)

property_values(plan::UpdateClassPlan) = (
    mode=plan.mode,
    construction=plan.construction,
    graph_rebuild=plan.graph_rebuild,
)
replace_properties(::UpdateClassPlan, values) = UpdateClassPlan(
    values.mode,
    values.construction,
    values.graph_rebuild,
)

@testset "leaky integrator property descriptors match the repository fixture" begin
    property = FIXTURE["property"]
    plan = LeakyIntegratorPlan(Float32(0.5), Float32(0.75))
    infos = @inferred property_infos(plan)

    @test length(infos) == 3
    @test infos[1].id == UInt32(property["gain_id"])
    @test infos[1].name == Symbol(property["gain_name"])
    @test infos[1].unit == property["unit"]
    @test infos[1].update isa RuntimeProperty
    @test infos[1].default_value === 1.0f0
    @test infos[1].constraint.minimum === -floatmax(Float32)
    @test infos[1].constraint.maximum === floatmax(Float32)

    @test infos[2].id == UInt32(property["pole_id"])
    @test infos[2].name == Symbol(property["pole_name"])
    @test infos[2].unit == property["unit"]
    @test infos[2].update isa RuntimeProperty
    @test infos[2].constraint.minimum === 0.0f0
    @test infos[2].constraint.maximum === 1.0f0

    @test @inferred(Union{Bool,Float32}, property_value(plan, infos[1].id)) === 0.5f0
    @test @inferred(Union{Bool,Float32}, property_value(plan, infos[2].id)) === 0.75f0
    @test infos[3].id == UInt32(2)
    @test infos[3].name === :closed
    @test infos[3].default_value === true
    @test infos[3].update isa RuntimeProperty
    @test property_value(plan, infos[3].id) === true
    @test_throws UnknownPropertyError property_value(plan, UInt32(99))
end

@testset "property preparation is typed and failure-atomic" begin
    plan = LeakyIntegratorPlan(0.5f0, 0.75f0)

    updated = prepare_properties(
        plan,
        (PropertyAssignment(0, -0.25f0), PropertyAssignment(1, 0.5f0)),
    )
    @test property_value(updated, 0) === -0.25f0
    @test property_value(updated, 1) === 0.5f0
    @test property_value(plan, 0) === 0.5f0
    @test property_value(plan, 1) === 0.75f0

    @test_throws PropertyTypeError prepare_properties(
        plan,
        (PropertyAssignment(0, -0.25),),
    )
    @test_throws DuplicatePropertyError prepare_properties(
        plan,
        (PropertyAssignment(0, -0.25f0), PropertyAssignment(0, 0.25f0)),
    )
    @test_throws UnknownPropertyError prepare_properties(
        plan,
        (PropertyAssignment(99, 0.0f0),),
    )
    @test_throws PropertyConstraintError prepare_properties(
        plan,
        (PropertyAssignment(1, 1.25f0),),
    )
    @test property_value(plan, 0) === 0.5f0
    @test property_value(plan, 1) === 0.75f0

    coupled = CoupledPlan(Int32(0), Int32(1))
    @test_throws ArgumentError prepare_properties(
        coupled,
        (PropertyAssignment(10, Int32(3)), PropertyAssignment(11, Int32(2))),
    )
    @test property_values(coupled) == (lower=Int32(0), upper=Int32(1))
end

@testset "runtime assignment vectors preserve Tuple preparation contracts" begin
    plan = LeakyIntegratorPlan(0.5f0, 0.75f0)
    for assignments in (
        PropertyAssignment[],
        PropertyAssignment[PropertyAssignment(0, 0.25f0)],
        PropertyAssignment[PropertyAssignment(1, 0.25f0)],
        PropertyAssignment[PropertyAssignment(0, 0.25f0), PropertyAssignment(1, 0.5f0)],
        PropertyAssignment[PropertyAssignment(1, 0.5f0), PropertyAssignment(0, 0.25f0)],
    )
        @test property_values(prepare_properties(plan, assignments)) ==
              property_values(prepare_properties(plan, Tuple(assignments)))
    end
    for (error_type, assignments) in (
        (PropertyTypeError, PropertyAssignment[PropertyAssignment(0, 0.25)]),
        (DuplicatePropertyError, PropertyAssignment[PropertyAssignment(0, 0.25f0), PropertyAssignment(0, 0.5f0)]),
        (UnknownPropertyError, PropertyAssignment[PropertyAssignment(99, 0.25f0)]),
        (PropertyConstraintError, PropertyAssignment[PropertyAssignment(1, 1.25f0)]),
    )
        @test_throws error_type prepare_properties(plan, assignments)
    end
    @test property_values(plan) == (gain=0.5f0, pole=0.75f0, closed=true)

    coupled = CoupledPlan(Int32(0), Int32(1))
    @test property_values(prepare_properties(coupled, PropertyAssignment[
        PropertyAssignment(10, Int32(2)), PropertyAssignment(11, Int32(3)),
    ])) == (lower=Int32(2), upper=Int32(3))
    @test_throws ArgumentError prepare_properties(coupled, PropertyAssignment[
        PropertyAssignment(10, Int32(3)), PropertyAssignment(11, Int32(2)),
    ])
    @test property_values(coupled) == (lower=Int32(0), upper=Int32(1))

    mixed = MixedPlan(false, 1.0f0)
    for assignments in (
        PropertyAssignment[PropertyAssignment(20, true)],
        PropertyAssignment[PropertyAssignment(21, 0.25f0)],
        PropertyAssignment[PropertyAssignment(20, true), PropertyAssignment(21, 0.25f0)],
        PropertyAssignment[PropertyAssignment(21, 0.25f0), PropertyAssignment(20, true)],
    )
        @test property_values(prepare_properties(mixed, assignments)) ==
              property_values(prepare_properties(mixed, Tuple(assignments)))
    end
    update_class = UpdateClassPlan(UInt32(1), Int32(1), Int32(1))
    for assignment in (
        PropertyAssignment(7, UInt32(2)), PropertyAssignment(8, Int32(2)),
        PropertyAssignment(9, Int32(2)),
    )
        @test_throws PropertyUpdateClassError prepare_properties(
            update_class, PropertyAssignment[assignment],
        )
    end
    @test_throws PropertySchemaError prepare_properties(
        DuplicateSchemaPlan(Int32(0), Int32(0)), PropertyAssignment[PropertyAssignment(5, Int32(1))],
    )
end

@testset "schema admission rejects ambiguous and incompatible descriptors" begin
    plan = DuplicateSchemaPlan(Int32(0), Int32(0))
    error = try
        validate_property_schema(plan)
        nothing
    catch caught
        caught
    end
    @test error == PropertySchemaError(UInt32(5), :duplicate_id, 1, 2)
    @test_throws PropertySchemaError property_value(plan, 5)
    @test_throws PropertySchemaError prepare_properties(
        plan,
        (PropertyAssignment(5, Int32(1)),),
    )
    @test_throws TypeError PropertyRange{Bool}(false, true)
    @test_throws TypeError PropertyRange{String}("a", "z")
    mismatched = PropertySpec{:mismatch,Float32,RuntimeProperty,PropertyRange{Float64}}(
        PropertyInfo(
            UInt32(6),
            :mismatch,
            "Mismatched range",
            "1",
            RuntimeProperty(),
            0.5f0,
            PropertyRange(0.0, 1.0),
        ),
    )
    @test_throws PropertySchemaError FilterGraphAlgorithms._validate_property_schema(
        (mismatched,),
        1,
    )
end

@testset "portable binding adopts one generation at a process boundary" begin
    initial = FIXTURE["initial"]
    update = FIXTURE["update"]
    state = Float32.(initial["state"])
    input = Float32.(initial["input"])
    output = similar(input)
    workspace = LeakyIntegratorWorkspace(state)
    binding = PropertyBinding(
        LeakyIntegratorPlan(Float32(initial["gain"]), Float32(initial["pole"])),
    )

    @test prepare_properties!(binding, ()) === binding
    @test commit_properties!(binding) === binding
    @test requested_generation(binding) == 0
    @test active_generation(binding) == 0

    @test process!(output, workspace, active_plan(binding), input) === output
    @test output == Float32.(initial["output"])
    @test commit!(workspace) === workspace

    assignments = (
        PropertyAssignment(0, Float32(update["gain"])),
        PropertyAssignment(1, Float32(update["pole"])),
    )
    @test prepare_properties!(binding, assignments) === binding
    @test commit_properties!(binding) === binding
    @test requested_generation(binding) == UInt64(update["requested_generation"])
    @test active_generation(binding) == 0
    @test_throws PropertyUpdateBusy prepare_properties!(binding, assignments)

    @test adopt_properties!(binding)
    @test !adopt_properties!(binding)
    @test requested_plan(binding) === active_plan(binding)
    @test active_generation(binding) == UInt64(update["active_generation"])

    update_input = Float32.(update["input"])
    @test process!(output, workspace, active_plan(binding), update_input) === output
    @test output == Float32.(update["output"])
    @test commit!(workspace) === workspace
    @test leaky_integrator_state(workspace) == output

    prepare_properties!(binding, (PropertyAssignment(0, 0.125f0),))
    @test discard_properties!(binding) === binding
    @test !adopt_properties!(binding)
end

@testset "scalar property value domain remains explicit" begin
    choices = PropertyChoices(
        (
            PropertyChoice(UInt32(1), :one, "first"),
            PropertyChoice(UInt32(2), :two, "second"),
        ),
    )
    info = PropertyInfo(
        UInt32(7),
        :mode,
        "Mode selector",
        "1",
        ConstructionProperty(),
        UInt32(1),
        choices,
    )
    @test info.constraint.choices[2].value == UInt32(2)
    @test PropertyAssignment(8, "label").value == "label"
    @test_throws TypeError PropertyAssignment(9, Float32[1])
    @test NoPropertyConstraint() isa PropertyConstraint
    @test ReadOnlyProperty() isa PropertyUpdate
    @test GraphRebuildProperty() isa PropertyUpdate
    plan = UpdateClassPlan(UInt32(1), Int32(1), Int32(1))
    for assignment in (
        PropertyAssignment(7, UInt32(2)),
        PropertyAssignment(8, Int32(2)),
        PropertyAssignment(9, Int32(2)),
    )
        @test_throws PropertyUpdateClassError prepare_properties(plan, (assignment,))
    end
end

end
