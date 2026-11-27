using Test
using FilterGraphAlgorithms

# Julia coverage instrumentation can change allocation counts. The full
# package test suite, including allocation contracts, runs without coverage.
# Collect coverage from the scientific and integration fixtures listed here.
const TEST_FILES = (
    FilterGraphAlgorithms=(
        "catalog.jl",
        "interface.jl",
        "preparation_validation.jl",
        "ka_cpu.jl",
        "leaky_integrator_properties.jl",
        "lowpass_design.jl",
        "pixel_validity.jl",
        "scc_reconstructor_preparation.jl",
        "thin_plate_vdm_completion.jl",
        "pdm_command.jl",
    ),
    FilterGraphs=(
        "cpu_loans.jl",
        "chopped_pixel_controller.jl",
        "cred2_fits_replay.jl",
        "external_aliases.jl",
        "physical_offset_reference.jl",
        "stateful_example.jl",
    ),
    FilterGraphPipeWire=(
        "source_channels.jl",
        "parameter_ports.jl",
        "preparation.jl",
        "command_endpoint.jl",
    ),
)

package = only(ARGS)
test_directory = joinpath(@__DIR__, "..", "..", "julia", package, "test")
@testset "$package CPU coverage fixtures" begin
    for file in getproperty(TEST_FILES, Symbol(package))
        include(joinpath(test_directory, file))
    end
end
