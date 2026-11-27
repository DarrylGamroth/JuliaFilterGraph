# Generated snapshot from examples/pixel-pyramid-row/run.jl.
# Regenerate with julia test/update_snapshots.jl examples/pixel-pyramid-row/run.jl.
module PixelPyramidRowExample

import FilterGraphAlgorithms
import FilterGraphs

const OUTPUT_NAMES = (
    Symbol("reconstruction-pixels"),
    Symbol("mean-pupil-intensity"),
)

"""
    process_frame!(outputs, graph, image, sequence)

Process four row blocks and require terminal publication for both outputs.
"""
function process_frame!(outputs, graph, image, sequence)
    availability = nothing
    for first_row in 0:2:6
        terminal = first_row == 6
        row_block = @view image[(first_row + 1):(first_row + 2), :]
        sample = FilterGraphAlgorithms.SampleMetadata(
            sequence,
            first_row;
            timestamp_nanoseconds=sequence * 1_000 + first_row,
            terminal=terminal,
        )
        availability = FilterGraphs.process!(
            outputs,
            graph,
            (raw=row_block,),
            (raw=sample,),
        )
        availability isa NamedTuple || error(
            "the row Graph rejected sample $sequence with $availability",
        )
        all(values(availability)) == terminal || error(
            "external outputs became available at the wrong row boundary",
        )
    end
    return availability
end

"""
    run_example()

Prepare and process two detector frames through two linked Julia Nodes.
"""
function run_example()
    graph = FilterGraphs.prepare_graph(
        joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms(),
    )

    image = reshape(UInt16.(1:64), 8, 8)
    reconstruction_pixels = zeros(Float32, 4, 3)
    mean_pupil_intensity = zeros(Float32, 1)
    outputs = NamedTuple{OUTPUT_NAMES}(
        (reconstruction_pixels, mean_pupil_intensity),
    )

    first_availability = process_frame!(outputs, graph, image, 1)
    all(iszero, reconstruction_pixels) || error(
        "the first frame must use the initial zero normalization gain",
    )

    second_availability = process_frame!(outputs, graph, image, 2)
    any(!iszero, reconstruction_pixels) || error(
        "the second frame must use the first committed normalization gain",
    )

    return (
        first_availability=first_availability,
        second_availability=second_availability,
        reconstruction_pixels=reconstruction_pixels,
        mean_pupil_intensity=only(mean_pupil_intensity),
        metadata=FilterGraphs.output_metadata(graph),
    )
end

function (@main)(args)
    result = run_example()
    println("Two linked Julia Nodes completed two detector frames.")
    println("Reconstruction pixels:")
    show(stdout, "text/plain", result.reconstruction_pixels)
    println()
    println("Mean pupil intensity: ", result.mean_pupil_intensity)
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .PixelPyramidRowExample: main
end
