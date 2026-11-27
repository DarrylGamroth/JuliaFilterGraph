module ApplicationTests
using Test

import FilterGraphPipeWire
import FilterGraphAlgorithms
import FilterGraphs
import PipeWireAO
using FilterGraphPipeWire: parse_arguments, _algorithm_declarations, configure_threads,
                         _prepare_service_graph, close_execution
using FilterGraphPipeWire: run_service, parse_pin_cpus, load_algorithm_declaration,
                         startup_parameter, read_startup_parameter,
                         MAX_STARTUP_PARAMETER_BYTES, spawn_monitor, monitor_quit_request

const LEAKY = joinpath(@__DIR__, "data", "leaky-graph.conf")
const ROW_GRAPH = joinpath(@__DIR__, "data", "pixel-pyramid-row.conf")
const SCC_GRAPH = joinpath(@__DIR__, "data", "cred2-ingress.conf")

include("composition.jl")
include("source_channels.jl")

@testset "deployment validation uses real metadata only when processing" begin
    @test isnothing(run_service(parse_arguments(["--graph", SCC_GRAPH, "--check"])))
    @test_throws ArgumentError run_service(parse_arguments([
        "--graph", SCC_GRAPH, "--check", "--warmup-output", "command",
    ]))
end

@testset "retired acquisition version selection is rejected" begin
    @test !hasproperty(parse_arguments(["--graph", LEAKY, "--check"]), :acquisition_version)
    @test_throws ArgumentError parse_arguments(["--graph", LEAKY, "--check", "--acquisition-version=3"])
    for version in ("1", "2", "4", "invalid", "-1")
        @test_throws ArgumentError parse_arguments(["--graph", LEAKY, "--check", "--acquisition-version", version])
    end
end

@testset "generic deployment CLI" begin
    options = parse_arguments(["--graph", LEAKY, "--check"])
    @test options.execution == "complete-frame"
    @test options.row_workers == 0
    @test isempty(options.parameters)
    @test isnothing(options.feedback)
    @test !options.fifo_inputs
    @test isempty(options.warmup_outputs)
    @test isempty(options.algorithms)
    @test parse_arguments(["--graph", LEAKY, "--check", "--warmup-output=output"]).warmup_outputs == (:output,)
    row = parse_arguments([
        "--graph", ROW_GRAPH, "--check", "--execution=row-block",
        "--row-workers", "0", "--matrix-layout=sharded", "--fifo-inputs",
        "--rate", "1000", "--pin-cpus", "14,10",
    ])
    @test row.matrix_layout == :sharded
    @test row.fifo_inputs
    @test row.pin_cpus == [14, 10]
    for arguments in (
        ["--warmup-output", "output", "--warmup-output", "output"],
        ["--warmup-output", ""],
        ["--warmup-output", "output", "--capture-backend", "cuda"],
        ["--algorithm", "FilterGraphAlgorithms"],
        ["--algorithm", "FilterGraphAlgorithms.foo()"],
        ["--execution", "bad"], ["--row-workers", "-1"],
        ["--row-workers", "1"], ["--matrix-layout", "unknown"],
        ["--pin-cpus", "0,14"], ["--pin-cpus", "14,1"],
        ["--pin-cpus", "14,"], ["--parameter", "background", "Float32"],
        ["--feedback", "input"],
        ["--feedback", "input", "output", "--feedback", "second", "output"],
        ["--execution", "row-block", "--capture-backend", "cuda"],
        ["--feedback", "input", "output", "--capture-backend", "cuda"],
    )
        @test_throws ArgumentError parse_arguments(["--graph", LEAKY, "--check", arguments...])
    end
    @test parse_pin_cpus("14,14") == [14, 14]
end

@testset "explicit exported Algorithm selection" begin
    selected = parse_arguments([
        "--graph", LEAKY, "--check",
        "--algorithm", "FilterGraphAlgorithms.PyramidPupilImageF32",
        "--algorithm", "FilterGraphAlgorithms.PyramidPupilReconstructorF32",
        "--algorithm", "FilterGraphAlgorithms.LeakyIntegratorF32",
        "--algorithm", "FilterGraphAlgorithms.PyramidPupilImageF32",
    ])
    declarations = _algorithm_declarations(selected)
    @test declarations == FilterGraphAlgorithms.algorithms()
    @test FilterGraphAlgorithms.PyramidPupilImageF32 in declarations
    @test FilterGraphAlgorithms.PyramidPupilReconstructorF32 in declarations
    @test count(==(FilterGraphAlgorithms.LeakyIntegratorF32), declarations) == 1
    only_explicit = parse_arguments([
        "--graph", LEAKY, "--check", "--no-builtins",
        "--algorithm", "FilterGraphAlgorithms.LeakyIntegratorF32",
    ])
    @test _algorithm_declarations(only_explicit) == (FilterGraphAlgorithms.LeakyIntegratorF32,)
    @test_throws ArgumentError load_algorithm_declaration("FilterGraphAlgorithms._NO_SAMPLE_METADATA")
    @test_throws ArgumentError load_algorithm_declaration("FilterGraphAlgorithms.algorithms")
    @test_throws ArgumentError load_algorithm_declaration("FilterGraphAlgorithms.AbstractAlgorithmDeclaration")
    @test_throws ArgumentError load_algorithm_declaration("FilterGraphAlgorithms.UnknownDeclaration")
end

@testset "typed little-endian row-major startup arrays" begin
    mktempdir() do directory
        path = joinpath(directory, "matrix.f32le")
        open(path, "w") do io
            write(io, htol.(reinterpret(UInt32, Float32[1, 2, 3, 4, 5, 6])))
        end
        parameter = startup_parameter("background", "F32", "2,3", path)
        @test read_startup_parameter(parameter) == Float32[1 2 3; 4 5 6]
        @test parameter.bytes == 24
        @test_throws ArgumentError startup_parameter("background", "Code", "2,3", path)
        @test_throws ArgumentError startup_parameter("background", "F32", "0,6", path)
        @test_throws ArgumentError startup_parameter("background", "F32", "2,", path)
        @test_throws ArgumentError startup_parameter("background", "F32", "$(typemax(Int)),2", path)
        @test_throws ArgumentError startup_parameter("background", "F32", "$(MAX_STARTUP_PARAMETER_BYTES)", path)
        @test_throws ArgumentError startup_parameter("background", "F32", "2,4", path)
        @test_throws ArgumentError parse_arguments([
            "--graph", ROW_GRAPH, "--check",
            "--parameter", "background", "F32", "2,3", path,
            "--parameter", "background", "F32", "2,3", path,
        ])
        write(path, UInt8[0])
        @test_throws DimensionMismatch read_startup_parameter(parameter)

        bool_path = joinpath(directory, "active.u8")
        write(bool_path, UInt8[0, 1, 1])
        bool_parameter = startup_parameter("active", "Bool", "3", bool_path)
        @test read_startup_parameter(bool_parameter) == Bool[false, true, true]
        write(bool_path, UInt8[0, 2, 1])
        @test_throws ArgumentError read_startup_parameter(bool_parameter)

        cube_path = joinpath(directory, "cube.u16le")
        write(cube_path, htol.(UInt16.(1:12)))
        cube = read_startup_parameter(startup_parameter("cube", "UInt16", "2,2,3", cube_path))
        @test size(cube) == (2, 2, 3)
        @test cube[1, 2, 1] == 4
        @test cube[2, 1, 3] == 9
    end
end

@testset "graph parameter binding and unpublished row preparation" begin
    mktempdir() do directory
        background_path = joinpath(directory, "background.f32le")
        # A non-symmetric array detects accidental native Julia file ordering.
        background = reshape(Float32.(1:64), 8, 8)
        write(background_path, htol.(reinterpret(UInt32, vec(permutedims(background)))))
        options = parse_arguments([
            "--graph", ROW_GRAPH, "--execution", "row-block", "--row-workers", "0",
            "--matrix-layout", "shared", "--rate", "1000", "--check",
            "--boundary-layout", "row-major",
            "--parameter", "background", "Float32", "8,8", background_path,
        ])
        graph = _prepare_service_graph(options)
        node = nothing
        try
            @test graph isa FilterGraphs.PreparedCpuProgressiveGraph
            @test graph.graph.nodes[1].prepared.plan.background == background
            node = FilterGraphPipeWire.PipeWireNode(graph;
                connect=false, rate=options.rate, boundary_layout=options.boundary_layout)
            @test FilterGraphPipeWire.prepare!(node; execution=:row_block) === node
            FilterGraphPipeWire.prepare!(node; execution=:row_block)
            @test !PipeWireAO.isrunning(node)
            @test isopen(node)
            @test isnothing(FilterGraphPipeWire.last_error(node))
            @test isnothing(FilterGraphPipeWire.node_id(node))
            @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; required_outputs=(:unknown,))
            @test_throws ArgumentError FilterGraphPipeWire.prepare!(node; required_outputs=(Symbol("reconstruction-pixels"), Symbol("reconstruction-pixels")))
            # The first subsequent sample starts at zero and completes normally.
            inputs = (raw=fill(UInt16(100), 2, 8),)
            outputs = NamedTuple{keys(graph.graph.output_formats)}(map(values(graph.graph.output_formats)) do format
                zeros(format.element_type, format.shape)
            end)
            for block in 0:3
                result = FilterGraphs.process!(outputs, graph, inputs,
                    (raw=FilterGraphAlgorithms.SampleMetadata(0, 2 * block; terminal=block == 3),))
                @test result isa NamedTuple
                @test all(values(result)) == (block == 3)
            end
        finally
            isnothing(node) || close(node)
            close(graph)
        end
        bad_name = merge(options, (; parameters=(startup_parameter("unknown", "Float32", "8,8", background_path),)))
        @test_throws ArgumentError _prepare_service_graph(bad_name)
        bad_type = merge(options, (; parameters=(startup_parameter("background", "UInt32", "8,8", background_path),)))
        @test_throws ArgumentError _prepare_service_graph(bad_type)
        bad_shape = merge(options, (; parameters=(startup_parameter("background", "Float32", "4,16", background_path),)))
        @test_throws DimensionMismatch _prepare_service_graph(bad_shape)
    end
end

@testset "reserved monitor and worker thread placement" begin
    options = parse_arguments(["--graph", ROW_GRAPH, "--check", "--execution", "row-block", "--row-workers", "1"])
    if Threads.nthreads(:default) >= 3
        @test configure_threads(options) == [3]
    else
        @test_throws ArgumentError configure_threads(options)
    end
    impossible = merge(options, (; row_workers=Threads.nthreads(:default)))
    @test_throws ArgumentError configure_threads(impossible)
    if Threads.nthreads(:default) >= 2
        @test fetch(spawn_monitor(() -> Threads.threadid())) == 2
    end
end


# Child processes use the active Pkg test environment, which also works when
# the package is installed without a sibling source checkout or local lock.
@testset "quit request precedes monitor logging" begin
    project = dirname(Base.active_project())
    script = joinpath(@__DIR__, "monitor_order.jl")
    @test success(`$(Base.julia_cmd()) --startup-file=no --threads=2,0 --project=$project $script`)
end

@testset "package import and module entry point" begin
    project = dirname(Base.active_project())
    import_check = """
        using FilterGraphPipeWire
        @assert !isdefined(Main, :main)
        @assert Base.ispublic(FilterGraphPipeWire, :main)
        @assert !Base.isexported(FilterGraphPipeWire, :main)
        @assert FilterGraphPipeWire._callback_metrics_factory[] === nothing
        println("imported")
    """
    @test occursin("imported", read(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $import_check`, String))
    help = read(`$(Base.julia_cmd()) --startup-file=no --project=$project -m FilterGraphPipeWire --help`, String)
    @test occursin("--graph PATH", help)
end

@testset "cold optional Algorithm provider after precompilation" begin
    project = dirname(Base.active_project())
    graph = joinpath(@__DIR__, "data", "cold-provider.conf")
    providers = joinpath(@__DIR__, "data")
    command = `$(Base.julia_cmd()) --startup-file=no --threads=2,0 --project=$project -m FilterGraphPipeWire --graph $graph --check --no-builtins --algorithm-package ColdIslandAlgorithms`
    load_path = join(("@", providers, "@stdlib"), Sys.iswindows() ? ';' : ':')
    output = read(addenv(command, "JULIA_LOAD_PATH" => load_path), String)
    @test occursin("JULIA_FILTER_GRAPH_CHECKED", output)
    @test occursin("algorithms=1", output)
end

include("startup_initial_parameters.jl")

include("direct_preparation.jl")

end
