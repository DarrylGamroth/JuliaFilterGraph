module StatefulDeclarationTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

function admitted_allocation_bytes(output, prepared, input)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

workspace_commit_allocation_bytes(workspace) = @allocated commit!(workspace)
workspace_discard_allocation_bytes(workspace) = @allocated discard!(workspace)
workspace_reset_allocation_bytes(workspace) = @allocated reset!(workspace)

@testset "stateful scientist declaration contract" begin
    initial_state = 0.25f0
    prepared = prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=2,
            initial_state=initial_state,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
    prepared.plan = LeakyIntegratorPlan(0.5f0, 0.75f0)
    workspace = prepared.workspace
    output = zeros(Float32, 2)

    discarded_input = Float32[2, -2]
    committed_before = copy(leaky_integrator_state(workspace))
    @test FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        discarded_input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test output == Float32[1.1875, -0.8125]
    @test leaky_integrator_state(workspace) == committed_before
    @test discard!(workspace) === workspace
    @test leaky_integrator_state(workspace) == committed_before

    committed_input = Float32[4, 0]
    @test FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        committed_input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    ) === nothing
    @test output == Float32[2.1875, 0.1875]
    @test leaky_integrator_state(workspace) == committed_before
    @test commit!(workspace) === workspace
    @test leaky_integrator_state(workspace) == output

    @test reset!(prepared) === prepared
    @test leaky_integrator_state(workspace) == fill(initial_state, 2)

    @test process!(output, prepared, committed_input) === nothing
    @test leaky_integrator_state(workspace) == output

    reset!(prepared)
    FilterGraphAlgorithms._process_admitted!(
        output,
        prepared,
        committed_input,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    discard!(workspace)
    @test admitted_allocation_bytes(output, prepared, committed_input) == 0
    @test workspace_discard_allocation_bytes(workspace) == 0
    @test workspace_commit_allocation_bytes(workspace) == 0
    @test workspace_reset_allocation_bytes(workspace) == 0
end

end
