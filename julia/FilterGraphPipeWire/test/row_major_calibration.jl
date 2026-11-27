import FilterGraphAlgorithms: process!, reset!

function row_major_calibration_graph(; image_size=(71, 69), pupil_size=(33, 33),
                                     consumer=:pyramid, pupil_parameters=false)
    rows, columns = image_size
    pupil_rows, pupil_columns = pupil_size
    node = consumer === :pyramid ? """
        { type = ndarray name = measure label = pyramid-pupil-image-f32
          config = { image_size = [$rows $columns]
                     pupil_size = [$pupil_rows $pupil_columns]
                     pupil_origins = [ [0 0] [$pupil_rows 0]
                                       [$pupil_rows $pupil_columns] [0 $pupil_columns] ]
                     pupil_mask = null } }
        """ : consumer === :regions ? """
        { type = ndarray name = measure label = region-extraction-f32
          config = { image_size = [$rows $columns] region_size = [1 1]
                     origins = [ [0 0] ] } }
        """ : ""
    links = consumer === :none ? "" :
        "{ output = \"calibrate:calibrated\" input = \"measure:$(consumer === :pyramid ? "image" : "calibrated")\" }"
    outputs = consumer === :pyramid ?
        "\"measure:reconstruction-pixels\" \"measure:mean-pupil-intensity\"" :
        consumer === :regions ? "\"measure:regions\"" : "\"calibrate:calibrated\""
    parameters = pupil_parameters ? "\"measure:pupil-origins\" \"measure:pupil-mask\"" : ""
    text = """
    filter.graph = {
        nodes = [
            { type = ndarray name = calibrate label = pixel-calibration-u16-f32
              config = { image_size = [$rows $columns]
                         initial_flat = null initial_background = null } }
            $node
        ]
        links = [ $links ]
        inputs = [ "calibrate:raw" "calibrate:flat" "calibrate:background" $parameters ]
        outputs = [ $outputs ]
    }
    """
    return mktemp() do path, io
        write(io, text)
        close(io)
        prepare_graph(path; algorithms=FilterGraphAlgorithms.algorithms())
    end
end

row_major_bits(array) = reinterpret(UInt32, vec(collect(array)))
row_major_outputs(graph) = FilterGraphs._external_buffers(graph.output_formats)
row_major_allocations(outputs, graph, inputs, metadata) =
    @allocated process!(outputs, graph, inputs, metadata)

@testset "Row-major calibration is detached and preserves canonical science" begin
    source = row_major_calibration_graph()
    image_size = source.input_formats.raw.shape
    raw = reshape(UInt16[mod(7919 * i, 65536) for i in 1:prod(image_size)], image_size)
    raw[1] = 0
    raw[end] = typemax(UInt16)
    flat = reshape(Float32[0.5f0 + mod(i, 17) / 8f0 for i in eachindex(raw)], image_size)
    background = reshape(Float32[mod(i, 31) - 15.5f0 for i in eachindex(raw)], image_size)
    replace_parameters!(source, :flat => flat, :background => background)
    baseline_output = row_major_outputs(source)
    # Clone an already-published graph, including its mutable previous-frame gain.
    initial_gain = copy(source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean)
    metadata = (raw=SampleMetadata(1; terminal=true),)
    @test process!(baseline_output, source, (raw=raw,), metadata) isa NamedTuple
    source_plan = source.nodes[1].prepared.plan
    source_outputs = map(outputs -> map(copy, outputs), source.node_outputs)
    source_gain = copy(source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean)
    @test source_gain != initial_gain
    source_metadata = FilterGraphs.output_metadata(source)
    prepared = prepare_row_major_calibration(source)
    @test prepared !== source
    @test source.nodes[1].prepared.plan === source_plan
    @test source.nodes[1].prepared.plan isa FilterGraphAlgorithms.PixelCalibrationPlan
    @test source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean == source_gain
    for (saved, current) in zip(source_outputs, source.node_outputs), (a, b) in zip(saved, current)
        @test row_major_bits(a) == row_major_bits(b)
    end
    @test prepared.nodes[1] isa FilterGraphs.PreparedNode{PixelCalibrationU16F32}
    @test prepared.nodes[1].prepared.workspace isa FilterGraphAlgorithms.EmptyWorkspace
    @test prepared.node_outputs[1][1] isa FilterGraphPipeWire.Transpose{Float32,Matrix{Float32}}
    @test size(prepared.node_outputs[1][1]) == image_size
    @test prepared.input_formats == source.input_formats
    @test prepared.output_formats == source.output_formats
    @test prepared.parameter_formats == source.parameter_formats
    @test prepared.nodes[1].prepared.plan.canonical.flat !== source_plan.flat
    @test prepared.nodes[2].prepared.workspace !== source.nodes[2].prepared.workspace
    @test prepared.nodes[2].prepared.workspace.reciprocal_previous_frame_mean !==
          source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean
    @test prepared.node_output_available[1][1] !== source.node_output_available[1][1]
    @test prepared.node_output_metadata[1][1] !== source.node_output_metadata[1][1]
    @test prepared.rejection !== source.rejection
    for (original, detached) in zip(source.node_outputs, prepared.node_outputs), (a, b) in zip(original, detached)
        @test row_major_bits(a) == row_major_bits(b)
        @test !Base.mightalias(a, b)
    end
    @test FilterGraphs.output_metadata(prepared) == source_metadata

    # Exercise the native pointer projection, then an ordinary dense warmup call.
    storage = vec(permutedims(raw))
    borrowed = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(
        pointer(storage), image_size, length(storage),
    )
    actual = row_major_outputs(prepared)
    GC.@preserve storage begin
        for sequence in 2:5
            metadata = (raw=SampleMetadata(sequence; terminal=true),)
            inputs = (raw=borrowed,)
            @test process!(baseline_output, source, inputs, metadata) ==
                  process!(actual, prepared, inputs, metadata)
            for (a, b) in zip(values(baseline_output), values(actual))
                @test row_major_bits(a) == row_major_bits(b)
            end
            for (a, b) in zip(source.node_outputs, prepared.node_outputs), (x, y) in zip(a, b)
                @test row_major_bits(x) == row_major_bits(y)
            end
            @test source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean ==
                  prepared.nodes[2].prepared.workspace.reciprocal_previous_frame_mean
            @test FilterGraphs.output_metadata(prepared) == FilterGraphs.output_metadata(source)
        end
        @test row_major_allocations(actual, prepared, (raw=borrowed,), metadata) == 0
    end
    @test process!(baseline_output, source, (raw=raw,), metadata) ==
          process!(actual, prepared, (raw=raw,), metadata)
    @test all(row_major_bits(a) == row_major_bits(b) for
              (a, b) in zip(values(baseline_output), values(actual)))
    # Resetting or advancing the detached graph never resets the caller's state.
    retained_gain = copy(source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean)
    reset!(prepared)
    @test source.nodes[2].prepared.workspace.reciprocal_previous_frame_mean == retained_gain
    @test source.nodes[1].prepared.plan === source_plan
end

@testset "Row-major calibration retains ordered pupil normalization" begin
    rows, columns = 31, 125
    image_size = (2rows, 2columns)
    source = row_major_calibration_graph(; image_size, pupil_size=(rows, columns),
        pupil_parameters=true)
    raw = zeros(UInt16, image_size)
    raw[1, 1:3] .= 1
    raw[end, end] = 1
    flat = ones(Float32, image_size)
    flat[1, 1] = Float32(2^25)
    replace_parameters!(source, :flat => flat)
    detached = prepare_row_major_calibration(source)
    canonical_output = row_major_outputs(source)
    detached_output = row_major_outputs(detached)
    canonical_workspace = source.nodes[2].prepared.workspace
    detached_workspace = detached.nodes[2].prepared.workspace
    mean_scale = source.nodes[2].prepared.plan.normalization.mean_scale

    # The canonical scan loses the three unit samples after the bright pixel.
    # Summing in Float64 before narrowing would add one Float32 step instead.
    expected_mean = Float32(2^25) * mean_scale
    @test Float32(2^25 + 4) * mean_scale != expected_mean
    data = vec(permutedims(raw))
    borrowed = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(
        pointer(data), image_size, length(data),
    )
    GC.@preserve data begin
        for sequence in 1:3
            if sequence == 2
                # Rebuild both plans through their unchanged parameter interfaces.
                origins = UInt32[0 0; rows 0; rows columns; 0 columns]
                mask = fill(true, rows, columns)
                for graph in (source, detached)
                    replace_parameters!(graph, :background => zeros(Float32, image_size),
                        Symbol("pupil-origins") => origins, Symbol("pupil-mask") => mask)
                end
            end
            metadata = (raw=SampleMetadata(sequence; terminal=true),)
            @test process!(canonical_output, source, (raw=borrowed,), metadata) isa NamedTuple
            @test process!(detached_output, detached, (raw=borrowed,), metadata) isa NamedTuple
            @test only(last(values(canonical_output))) === expected_mean
            @test map(row_major_bits, values(canonical_output)) ==
                  map(row_major_bits, values(detached_output))
            @test pyramid_normalization_gain(canonical_workspace) === inv(expected_mean)
            @test pyramid_normalization_gain(detached_workspace) === inv(expected_mean)
        end
    end
end

@testset "Row-major calibration replacement keeps both coefficient orientations atomic" begin
    source = row_major_calibration_graph(; image_size=(3, 5), consumer=:none)
    prepared = prepare_row_major_calibration(source)
    source_plan = source.nodes[1].prepared.plan
    function check_planes(plan)
        @test row_major_bits(plan.transposed.flat) == row_major_bits(permutedims(plan.canonical.flat))
        @test row_major_bits(plan.transposed.background) == row_major_bits(permutedims(plan.canonical.background))
    end
    for update in (:flat => fill(2.0f0, 3, 5), :background => fill(1.0f0, 3, 5))
        replace_parameters!(prepared, update)
        check_planes(prepared.nodes[1].prepared.plan)
        @test source.nodes[1].prepared.plan === source_plan
    end
    current = prepared.nodes[1].prepared.plan
    @test_throws DimensionMismatch replace_parameters!(prepared,
        :flat => fill(3.0f0, 3, 5), :background => ones(Float32, 2, 5))
    @test prepared.nodes[1].prepared.plan === current
    @test_throws ArgumentError replace_parameters!(prepared, :flat => fill(Inf32, 3, 5))
    @test prepared.nodes[1].prepared.plan === current
    replace_parameters!(prepared, :flat => fill(4.0f0, 3, 5),
                        :background => fill(2.0f0, 3, 5))
    check_planes(prepared.nodes[1].prepared.plan)
    reset!(prepared)
    check_planes(prepared.nodes[1].prepared.plan)
    @test source.nodes[1].prepared.plan === source_plan

    # Use the real native owner's control slot; preparation must not publish a cache early.
    node = PipeWireNode(prepared; boundary_layout=RowMajorLayout(), connect=false)
    try
        snapshot = @atomic :acquire node.owner.control_snapshot
        plan = prepared.nodes[1].prepared.plan
        @test FilterGraphPipeWire._stage_parameter_transaction!(node.owner,
            :flat => fill(5.0f0, 3, 5), :background => fill(3.0f0, 3, 5))
        @test prepared.nodes[1].prepared.plan === plan
        requested = @atomic :acquire node.owner.control_snapshot
        @test requested.requested_parameter_sequences == (UInt64(1),)
        @test requested.active_parameter_sequences == (UInt64(0),)
        @test snapshot.plans == (plan,)
        @test FilterGraphPipeWire._adopt_pending_parameters!(node.owner)
        @test prepared.nodes[1].prepared.plan !== plan
        check_planes(prepared.nodes[1].prepared.plan)
        active = @atomic :acquire node.owner.control_snapshot
        @test active.active_parameter_sequences == (UInt64(1),)
        @test FilterGraphPipeWire.prepare!(node) === node
        check_planes(prepared.nodes[1].prepared.plan)
        @test FilterGraphPipeWire._graph_properties(node.owner) isa PipeWireAO.SPA.Props
    finally
        close(node)
    end
end

@testset "Row-major calibration preserves checked alias behavior" begin
    graph = prepare_row_major_calibration(row_major_calibration_graph(;
        image_size=(3, 3), consumer=:none))
    plan = graph.nodes[1].prepared.plan
    raw_dense = reshape(UInt16.(1:9), 3, 3)
    storage = vec(permutedims(raw_dense))
    borrowed = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(
        pointer(storage), (3, 3), 9,
    )
    GC.@preserve storage begin
        for plane in (plan.canonical.flat, plan.canonical.background,
                      plan.transposed.flat, plan.transposed.background)
            @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration!(
                FilterGraphPipeWire.transpose(plane), plan, borrowed)
        end
        for plane in (plan.transposed.flat, plan.transposed.background)
            @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration!(plane, plan, raw_dense)
            @test_throws ArgumentError FilterGraphAlgorithms._process_pixel_calibration!(
                FilterGraphPipeWire.transpose(plane), plan, raw_dense)
        end
    end
    for offset in (0, 4, 8)
        expected = reshape(Float32.(1:9), 3, 3)
        actual = copy(expected)
        raw_expected = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(
            Ptr{UInt16}(pointer(expected) + offset), (3, 3), 9)
        raw_actual = FilterGraphPipeWire.RowMajorBorrowedArray{UInt16,2,false}(
            Ptr{UInt16}(pointer(actual) + offset), (3, 3), 9)
        GC.@preserve expected actual begin
            FilterGraphAlgorithms._process_pixel_calibration!(
                FilterGraphPipeWire.transpose(expected), plan.canonical, raw_expected)
            FilterGraphAlgorithms._process_pixel_calibration!(
                FilterGraphPipeWire.transpose(actual), plan, raw_actual)
            @test row_major_bits(actual) == row_major_bits(expected)
        end
    end
end

@testset "Row-major calibration rejects unqualified Graphs without mutation" begin
    small = row_major_calibration_graph(; image_size=(7, 7), pupil_size=(2, 2))
    plan = small.nodes[1].prepared.plan
    @test_throws ArgumentError prepare_row_major_calibration(small)
    @test small.nodes[1].prepared.plan === plan
    cutoff = row_major_calibration_graph(; image_size=(65, 65), pupil_size=(32, 32))
    @test_throws ArgumentError prepare_row_major_calibration(cutoff)
    unsupported = row_major_calibration_graph(; image_size=(3, 5), consumer=:regions)
    @test_throws ArgumentError prepare_row_major_calibration(unsupported)
    one_row = row_major_calibration_graph(; image_size=(3, 7), pupil_size=(1, 2))
    @test prepare_row_major_calibration(one_row) isa FilterGraphs.PreparedGraph
    source = row_major_calibration_graph(; image_size=(3, 5), consumer=:none)
    source.node_inflight[1][] = true
    @test_throws ArgumentError prepare_row_major_calibration(source)
    @test source.node_inflight[1][]
    source.node_inflight[1][] = false
    detached = prepare_row_major_calibration(source)
    @test_throws ArgumentError prepare_row_major_calibration(detached)
end
