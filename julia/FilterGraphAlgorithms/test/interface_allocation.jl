module InterfaceAllocationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms:
    NoWorkspace,
    abort!,
    adopt_properties!,
    commit!,
    commit_properties!,
    prepare_properties!,
    process!,
    start!

include("support/interface_fixture.jl")

function adopt_and_process!(output, workspace, binding, input)
    adopt_properties!(binding)
    result = process!(output, workspace, active_plan(binding), input)
    commit!(workspace)
    return result
end

@testset "selected operating interface paths allocate no heap after warmup" begin
    input_progress(AddOnePlan)
    output_progress(AddOnePlan)
    @test @allocated(input_progress(AddOnePlan)) == 0
    @test @allocated(output_progress(AddOnePlan)) == 0

    input = Float32[1, 2, 3]
    output = zeros(Float32, 3)
    plan = AddOnePlan()
    workspace = NoWorkspace()
    process!(output, workspace, plan, input)
    @test @allocated(process!(output, workspace, plan, input)) == 0

    progressive_plan = ProgressiveAddOnePlan{true}(4)
    progressive_workspace = ProgressiveAddOneWorkspace(false, 0)
    progressive_input = Float32[1, 2, 3, 4]
    progressive_output = zeros(Float32, 4)

    start!(progressive_workspace, progressive_plan)
    process!(
        progressive_output,
        progressive_workspace,
        progressive_plan,
        progressive_input,
        1:4,
    )
    abort!(progressive_workspace, progressive_plan)

    start!(progressive_workspace, progressive_plan)
    @test @allocated(
        process!(
            progressive_output,
            progressive_workspace,
            progressive_plan,
            progressive_input,
            1:4,
        ),
    ) == 0
    @test @allocated(abort!(progressive_workspace, progressive_plan)) == 0
end

@testset "property adoption and processing allocate no heap after warmup" begin
    input = Float32[1, 2]
    output = similar(input)
    assignments = (PropertyAssignment(0, -0.25f0), PropertyAssignment(1, 0.5f0))

    warm_binding = PropertyBinding(LeakyIntegratorPlan(0.5f0, 0.75f0))
    warm_workspace = LeakyIntegratorWorkspace(Float32[4, -4])
    prepare_properties!(warm_binding, assignments)
    commit_properties!(warm_binding)
    adopt_and_process!(output, warm_workspace, warm_binding, input)

    binding = PropertyBinding(LeakyIntegratorPlan(0.5f0, 0.75f0))
    workspace = LeakyIntegratorWorkspace(Float32[4, -4])
    prepare_properties!(binding, assignments)
    commit_properties!(binding)
    @test @allocated(adopt_and_process!(output, workspace, binding, input)) == 0
end

end
