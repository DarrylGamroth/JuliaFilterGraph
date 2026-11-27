module SourceChannelsTests

using Test, JSON, FilterGraphAlgorithms, FilterGraphPipeWire, FilterGraphs, PipeWireAO
import PipeWireAO: pod_value, props_param

const F = FilterGraphPipeWire
const PROPERTY = "pipewireao.ndarray.channels"
const SUMMARY = SHACK_HARTMANN_SUMMARY_V1

function summary_port(name="summary"; shape=(8,), element=NdArray.F32_LE)
    return NdArrayFilterPort(name, PipeWireAO.DIRECTION_OUTPUT,
        NdArrayFormat(element, shape; layout=NdArray.ROW_MAJOR); schema=SUMMARY)
end

decode(pod) = JSON.parse(pod_value(String, pod))

@testset "source channels reject malformed UTF-8 annotations" begin
    malformed = String(UInt8[0xff])
    @test !isvalid(malformed)
    @test_throws ArgumentError F._source_channels([summary_port()],
        Dict("summary" => Dict("mean_flux" => malformed)))
    @test_throws ArgumentError F._source_channels([summary_port(malformed)])
    @test_throws ArgumentError F._channel_unit_declarations(Dict(malformed => Dict()))
end

@testset "source channels bind exact native formats and source units" begin
    port = summary_port()
    encoded = F._source_channels([port])
    document = decode(encoded)
    @test document["version"] == 1
    output = only(document["outputs"])
    @test output["port"] == "summary"
    @test output["schema"] == SUMMARY
    @test output["shape"] == [8]
    @test output["element_type"] == Int(NdArray.F32_LE)
    @test output["layout"] == Int(NdArray.ROW_MAJOR)
    @test getindex.(output["channels"], "name") == [
        "subaperture_count", "valid_count", "mean_flux", "slope_rms_x",
        "slope_rms_y", "slope_rms", "mean_x_slope", "mean_y_slope"]
    @test getindex.(output["channels"], "index") == collect(0:7)
    @test getindex.(output["channels"], "units") == ["count", "count", nothing, nothing,
        nothing, nothing, nothing, nothing]
    units = Dict(:summary => Dict(:mean_x_slope => "detector pixel",
        :mean_flux => "ADC \\\"count\\\""))
    annotated = only(decode(F._source_channels([port], units))["outputs"])["channels"]
    @test annotated[7]["units"] == "detector pixel"
    @test annotated[3]["units"] == units[:summary][:mean_flux]
    @test annotated[8]["units"] === nothing
    @test FilterGraphAlgorithms.schema_channels(SUMMARY)[7].units === nothing
    for invalid in (
        Dict("missing" => Dict("mean_flux" => "ADC")),
        Dict("summary" => Dict("missing" => "ADC")),
        Dict("summary" => Dict("valid_count" => "pixel")),
        Dict("summary" => Dict("mean_flux" => "")),
        Dict("summary" => Dict("mean_flux" => "bad\nunits")),
        Dict("summary" => Dict("mean_flux" => repeat("µ", 128))),
        Dict("summary" => Dict("mean_flux" => nothing)),
        Dict("summary" => Dict("mean_flux" => "ADC", :mean_flux => "other")),
        Dict("summary" => Dict(), :summary => Dict()),
        Dict("summary" => "ADC"),
        ["summary" => Dict("mean_flux" => "ADC")],
    )
        @test_throws ArgumentError F._source_channels([port], invalid)
    end
    @test_throws ArgumentError F._source_channels([summary_port(; shape=(1, 8))])
    @test_throws ArgumentError F._source_channels([summary_port(; element=NdArray.F64_LE)])
    @test_throws ArgumentError F._source_channels([port, port])
    @test_throws ArgumentError F._source_channels([summary_port("summary$i") for i in 1:64])
    unknown = NdArrayFilterPort("unknown", PipeWireAO.DIRECTION_OUTPUT,
        NdArrayFormat(NdArray.F32_LE, (8,); layout=NdArray.ROW_MAJOR))
    @test F._source_channels([unknown]) === nothing
    @test_throws ArgumentError F._source_channels([unknown], Dict("unknown" => Dict("mean_flux" => "ADC")))
end

@testset "external output aliases publish through the native Props getter" begin
    mktemp() do path, io
        write(io, """
            filter.graph = {
                nodes = [ { type = ndarray name = measure label = shack-hartmann-summary-f32
                    config = { subaperture_count = 2 } } ]
                links = []
                inputs = [ "measure:slopes" "measure:flux" "measure:validity" ]
                outputs = [ { name = public-summary endpoint = "measure:summary" } ]
            }
            """)
        flush(io)
        graph = FilterGraphs.prepare_graph(path; algorithms=(ShackHartmannSummaryF32,))
        source_units = Dict("public-summary" => Dict("mean_x_slope" => "detector pixel"))
        synchronized = FilterGraphs.prepare_synchronized_inputs(graph; match=:acquisition)
        for owner in (graph, synchronized)
            node = F.PipeWireNode(owner; connect=false, channel_units=source_units,
                boundary_layout=RowMajorLayout())
            try
                callbacks = node.filter.callbacks
                info = only(filter(info -> info.name == PROPERTY, callbacks.property_info))
                @test info.flags & SPA.PROPERTY_READONLY != 0
                props = callbacks.on_get_properties(node.filter)
                pod = only(filter(pair -> pair.first == PROPERTY, props.values)).second
                payload = only(decode(pod)["outputs"])
                @test payload["port"] == "public-summary"
                @test payload["channels"][7]["units"] == "detector pixel"
                source_units["public-summary"]["mean_x_slope"] = "changed caller storage"
                repeated = only(filter(pair -> pair.first == PROPERTY,
                    callbacks.on_get_properties(node.filter).values)).second
                @test repeated === pod
                wire_props = SPA.Props(Pod(props_param(props)))
                @test pod_value(String, only(filter(pair -> pair.first == PROPERTY, wire_props.values)).second) ==
                    pod_value(String, pod)
                @test_throws ArgumentError callbacks.on_properties(node.filter, SPA.Props(PROPERTY => "{}"))
            finally
                close(node)
                source_units["public-summary"]["mean_x_slope"] = "detector pixel"
            end
        end
        @test_throws ArgumentError F.PipeWireNode(graph; connect=false,
            channel_units=Dict("summary" => Dict("mean_x_slope" => "pixel")))
    end
end

@testset "GPI2 pyramid pupil flux publishes its F64 channel schema" begin
    mktemp() do path, io
        write(io, """
            filter.graph = {
                nodes = [
                    { type = ndarray name = flux label = pyramid-pupil-flux-u16-f64
                      config = { image_size = [ 4 6 ] pupil_size = [ 2 3 ]
                        pupil_origins = [ [ 0 0 ] [ 0 3 ] [ 2 0 ] [ 2 3 ] ] } }
                ]
                links = []
                inputs = [ "flux:raw" ]
                outputs = [ "flux:pupil-flux" ]
            }
            """)
        flush(io)
        graph = FilterGraphs.prepare_graph(path;
            algorithms=(FilterGraphAlgorithms.PyramidPupilFluxU16F64,))
        channel_units = Dict("pupil-flux" => Dict(
            "pupil_1" => "ADC counts per exposure",
            "pupil_2" => "ADC counts per exposure",
            "pupil_3" => "ADC counts per exposure",
            "pupil_4" => "ADC counts per exposure",
        ))
        node = F.PipeWireNode(graph; connect=false, channel_units)
        try
            props = node.filter.callbacks.on_get_properties(node.filter)
            pod = only(filter(pair -> pair.first == PROPERTY, props.values)).second
            output = only(decode(pod)["outputs"])
            @test output["port"] == "pupil-flux"
            @test output["schema"] == PYRAMID_PUPIL_FLUX_ADU_F64_V1
            @test output["shape"] == [4]
            @test output["element_type"] == Int(NdArray.F64_LE)
            @test getindex.(output["channels"], "name") ==
                ["pupil_1", "pupil_2", "pupil_3", "pupil_4"]
            @test getindex.(output["channels"], "index") == collect(0:3)
            @test getindex.(output["channels"], "title") ==
                ["Pupil 1", "Pupil 2", "Pupil 3", "Pupil 4"]
            @test getindex.(output["channels"], "units") ==
                fill("ADC counts per exposure", 4)
        finally
            close(node)
        end
    end
end

end
