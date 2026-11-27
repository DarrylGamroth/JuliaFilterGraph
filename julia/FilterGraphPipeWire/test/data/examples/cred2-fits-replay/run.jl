# Generated snapshot from examples/cred2-fits-replay/run.jl.
# Regenerate with julia test/update_snapshots.jl examples/cred2-fits-replay/run.jl.
module Cred2FitsReplayExample

using FITSIO
using FilterGraphAlgorithms
import FilterGraphs

const FRAME_SHAPE = (4, 2)
const SOURCE_DOMAIN = ntuple(index -> UInt8(index), 16)
const SOURCE_GENERATION = UInt64(1)
# Explicit software-fixture exposure, matching graph.conf; no physical timing claim.
const SYNTHETIC_EXPOSURE_DURATION_NS = UInt64(1_000_000)

# This small calibration is a software fixture, not an instrument calibration.
# The first four transmitted pixels are camera tags; only pixel 5 is reconstructed.
"""
    prepare_replay_graph()

Prepare the fixture's calibration-to-command graph without a transport adapter.
"""
function prepare_replay_graph()
    return FilterGraphs.prepare_graph(joinpath(@__DIR__, "graph.conf");
        algorithms=FilterGraphAlgorithms.algorithms())
end

"""
    synthetic_cube(; trigger_high_is_fringed::Bool)

Generate four synthetic optical phases and encode their configured electrical tags.
"""
function synthetic_cube(; trigger_high_is_fringed::Bool)
    cube = fill(UInt16(100), FRAME_SHAPE..., 4)
    for ordinal in axes(cube, 3)
        frame = @view cube[:, :, ordinal]
        fringed = isodd(ordinal)
        frame[1:3] .= (UInt16(ordinal), UInt16(0), UInt16(0))
        frame[4] = fringed == trigger_high_is_fringed ? UInt16(0x3fff) : UInt16(0x3ffe)
        frame[5] = fringed ? UInt16(104) : UInt16(100)
    end
    return cube
end

"""
    write_cube(path::AbstractString, cube::Array{UInt16,3})

Write a fixture using FITSIO's standard UInt16 FITS encoding.
"""
function write_cube(path::AbstractString, cube::Array{UInt16,3})
    FITS(path, "w") do file
        write(file, cube)
    end
    return path
end

"""
    read_cube(path::AbstractString)

Read an uncompressed primary UInt16 cube in `(x, y, frame)` axis order.
"""
function read_cube(path::AbstractString)
    return FITS(path, "r") do file
        image = file[1]
        eltype(image) === UInt16 || throw(ArgumentError("replay requires UInt16 FITS pixels"))
        ndims(image) == 3 || throw(DimensionMismatch("replay requires a three-axis FITS cube"))
        size(image)[1:2] == FRAME_SHAPE ||
            throw(DimensionMismatch("fixture calibration requires a 4-by-2 detector"))
        size(image, 3) > 0 || throw(DimensionMismatch("replay requires at least one frame"))
        read(image)
    end
end

"""
    process_frame!(outputs, graph, raw, metadata::SampleMetadata;
                   trigger_high_is_fringed::Bool)

Admit one raw frame: decode its electrical tag before calibration, apply the
configured application polarity, then process the graph. Invalid tags return
`Cred2UnexpectedImageTag` without invoking the graph. No phase is inferred from
frame parity, and no transport version or measured duration is synthesized.
"""
function process_frame!(outputs, graph, raw, metadata::SampleMetadata;
        trigger_high_is_fringed::Bool)
    electrical = cred2_trigger_metadata(metadata, raw)
    electrical isa SampleMetadata || return electrical
    optical = chopped_image_metadata(electrical; trigger_high_is_fringed)
    return FilterGraphs.process!(outputs, graph, (raw=raw,), (raw=optical,))
end

"""
    replay_cube(path::AbstractString; trigger_high_is_fringed::Bool, exposure_duration_nanoseconds::Integer)

Replay a tagged fixture and retain each published command and its owned metadata.
"""
function replay_cube(path::AbstractString; trigger_high_is_fringed::Bool, exposure_duration_nanoseconds::Integer)
    0 < exposure_duration_nanoseconds <= typemax(UInt64) || throw(ArgumentError("an explicit positive exposure duration is required"))
    cube = read_cube(path)
    graph = prepare_replay_graph()
    outputs = (calibrated=zeros(Float32, FRAME_SHAPE), command=zeros(Float32, 1))
    records = map(axes(cube, 3)) do ordinal
        raw = @view cube[:, :, ordinal]
        # This bounded replay is the acquisition authority; it assigns its own identity.
        metadata = SampleMetadata(ordinal; terminal=true,exposure_duration_nanoseconds,
            acquisition_identity=SampleAcquisitionIdentity(SOURCE_DOMAIN, SOURCE_GENERATION, ordinal))
        result = process_frame!(outputs, graph, raw, metadata; trigger_high_is_fringed)
        result == (calibrated=true, command=true) ||
            error("fixture frame $ordinal was rejected: $result")
        published = FilterGraphs.output_metadata(graph).command
        (sequence=published.sequence, trigger_high=sample_trigger_high(published),
            kind=sample_chopped_image_kind(published), command=copy(outputs.command), metadata=published)
    end
    return records
end

"""
    run_example(; trigger_high_is_fringed::Bool)

Write, reopen, and replay a temporary tagged FITS cube.
"""
function run_example(; trigger_high_is_fringed::Bool)
    return mktempdir() do directory
        path = joinpath(directory, "cred2-tagged.fits")
        write_cube(path, synthetic_cube(; trigger_high_is_fringed))
        replay_cube(path; trigger_high_is_fringed,exposure_duration_nanoseconds=SYNTHETIC_EXPOSURE_DURATION_NS)
    end
end

function (@main)(arguments)
    1 <= length(arguments) <= 3 || throw(ArgumentError(
        "usage: run.jl --trigger-high-is-fringed=true|false [--exposure-duration-nanoseconds=N cube.fits]"))
    option = arguments[1]
    polarity = if option == "--trigger-high-is-fringed=true"
        true
    elseif option == "--trigger-high-is-fringed=false"
        false
    else
        throw(ArgumentError("supply --trigger-high-is-fringed=true or false explicitly"))
    end
    records = if length(arguments)==1
        run_example(;trigger_high_is_fringed=polarity)
    else
        length(arguments)==3 && startswith(arguments[2],"--exposure-duration-nanoseconds=") ||
            throw(ArgumentError("external FITS replay requires --exposure-duration-nanoseconds=N"))
        duration=parse(UInt64,split(arguments[2],'=';limit=2)[2])
        replay_cube(arguments[3];trigger_high_is_fringed=polarity,exposure_duration_nanoseconds=duration)
    end
    for record in records
        println("sequence=$(record.sequence) trigger_high=$(record.trigger_high) " *
            "optical=$(record.kind) command=$(only(record.command))")
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import .Cred2FitsReplayExample: main
end
