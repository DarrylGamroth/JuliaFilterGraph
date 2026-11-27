# Standalone cold preparation regression; no PipeWire connection is created.
using Test
import FilterGraphPipeWire, FilterGraphs, FilterGraphAlgorithms

@testset "owned startup parameters precede execution" begin
    mktempdir() do directory
        graph_path = joinpath(directory, "graph.conf")
        write(graph_path, """
            filter.graph = {
                nodes = [ { type = ndarray name = calibrate
                    label = pixel-calibration-row-u16-f32
                    config = { image_size = [8, 8] row_block_rows = 2
                        initial_flat = null initial_background = null } } ]
                links = []
                inputs = [ "calibrate:raw" "calibrate:background" "calibrate:flat" ]
                outputs = [ "calibrate:calibrated" ]
            }
            """)
        background = reshape(Float32.(1:64), 8, 8)
        for execution in (:complete_frame, :row_block)
            calls = Ref(0)
            graph = FilterGraphPipeWire.prepare_service_graph(graph_path; execution,
                initial_parameters=formats -> begin
                    calls[] += 1
                    @test formats.background.shape == (8, 8)
                    [:background => copy(background)]
                end)
            try
                @test calls[] == 1
                outputs = (; calibrated=zeros(Float32, 2, 8))
                inputs = (; raw=fill(UInt16(100), 2, 8))
                for block in 0:3
                    metadata = (; raw=FilterGraphAlgorithms.SampleMetadata(0, 2block;
                        terminal=block == 3))
                    FilterGraphs.process!(outputs, graph, inputs, metadata)
                    @test outputs.calibrated == 100f0 .- background[(2block+1):(2block+2), :]
                end
            finally
                FilterGraphPipeWire.close_execution(graph)
            end
            @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(graph_path; execution,
                initial_parameters=[:background => background, :background => background])
            @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(graph_path; execution,
                initial_parameters=[:background => Float64.(background)])
            @test_throws DimensionMismatch FilterGraphPipeWire.prepare_service_graph(graph_path; execution,
                initial_parameters=[:background => zeros(Float32, 4, 16)])
            @test_throws ErrorException FilterGraphPipeWire.prepare_service_graph(graph_path; execution,
                initial_parameters=_ -> error("artifact preparation failed"))
            expected = execution === :complete_frame ? ErrorException : ArgumentError
            @test_throws expected FilterGraphPipeWire.prepare_service_graph(graph_path;
                execution, capture_backend=:cuda,
                initial_parameters=_ -> error("artifact preparation failed"))
            raw = joinpath(directory, "background.raw")
            write(raw, zeros(UInt8, 8 * 8 * sizeof(Float32)))
            parameter = FilterGraphPipeWire.startup_parameter("background", "Float32", "8,8", raw)
            @test_throws ArgumentError FilterGraphPipeWire.prepare_service_graph(graph_path;
                execution, parameters=[parameter], initial_parameters=[:background => background])
        end
    end
end
