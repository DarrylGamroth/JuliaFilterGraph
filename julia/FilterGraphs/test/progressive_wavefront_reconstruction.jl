@enum ProgressiveReconstructionGateFailure::UInt8 begin
    ProgressiveReconstructionGateRejected = 1
end

struct ProgressiveReconstructionGatePlan
    reject_sequence::UInt64
end

@algorithm ProgressiveReconstructionGateF32 begin
    label = "progressive-reconstruction-gate-f32"
    failure = ProgressiveReconstructionGateFailure

    @configuration ProgressiveReconstructionGateF32Configuration begin
        extent::Int => graph_rebuild
        reject_sequence::UInt64 => graph_rebuild
    end

    ports(configuration) = (
        input(
            :reconstructed,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            sample=:reconstructed_sample,
        ),
        output(
            :accepted,
            Float32,
            (configuration.extent,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            metadata_from=:reconstructed,
        ),
    )

    prepare(configuration) =
        ProgressiveReconstructionGatePlan(configuration.reject_sequence)

    process!(plan) = begin
        !isnothing(reconstructed_sample) &&
            reconstructed_sample.sequence == plan.reject_sequence &&
            return ProgressiveReconstructionGateRejected
        copyto!(accepted, reconstructed)
        nothing
    end
end

const PROGRESSIVE_MEASUREMENTS_SCHEMA = "org.example.ao.measurement-block/1"
const PROGRESSIVE_COLUMNS_SCHEMA =
    "org.example.ao.reconstructor-column-indices/1"
const PROGRESSIVE_COUNT_SCHEMA = "org.example.ao.measurement-block-count/1"

const PROGRESSIVE_SHACK_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = sense
            label = shack-hartmann-measurement-block-f32
            config = {
                image_size = [6, 6]
                row_block_rows = 1
                subaperture_size = [2, 2]
                subaperture_count = 4
                initial_subaperture_origins = [ [0 0] [0 3] [3 0] [3 3] ]
                coordinate_scale = 1.0
                pixel_threshold = 0.0
                flux_threshold = 0.0
                reference_slopes = null
                active = null
                measurements_schema = $PROGRESSIVE_MEASUREMENTS_SCHEMA
                column_indices_schema = $PROGRESSIVE_COLUMNS_SCHEMA
                measurement_count_schema = $PROGRESSIVE_COUNT_SCHEMA
            }
        }
        {
            type = ndarray
            name = reconstruct
            label = incremental-dense-reconstructor-f32
            config = {
                measurement_count = 8
                measurement_block_capacity = 8
                blocks_per_sample = 6
                reconstructed_count = 3
                initial_reconstructor = null
                measurements_schema = $PROGRESSIVE_MEASUREMENTS_SCHEMA
                column_indices_schema = $PROGRESSIVE_COLUMNS_SCHEMA
                measurement_count_schema = $PROGRESSIVE_COUNT_SCHEMA
                reconstructed_schema = $CONTROLLER_RESIDUAL_ERROR_V1
                reconstructor_schema = $SHWFS_RECONSTRUCTOR_V1
            }
        }
    ]
    links = [
        { output = "sense:measurements" input = "reconstruct:measurements" }
        { output = "sense:columns" input = "reconstruct:columns" }
        { output = "sense:count" input = "reconstruct:count" }
    ]
    inputs = [
        "sense:row-block"
        "sense:subaperture-origins"
        "sense:coordinates"
        "sense:reference-slopes"
        "sense:thresholds"
        "sense:active"
        "reconstruct:reconstructor"
    ]
    outputs = [ "reconstruct:reconstructed" ]
}
"""

const PROGRESSIVE_PYRAMID_GRAPH_TEXT = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = sense
            label = pyramid-measurement-block-f32
            config = {
                image_size = [8, 8]
                row_block_rows = 2
                pupil_size = [2, 2]
                pupil_origins = [ [6 0] [0 0] [6 4] [0 4] ]
                pupil_mask = [ true false true true ]
                measurements_schema = $PROGRESSIVE_MEASUREMENTS_SCHEMA
                column_indices_schema = $PROGRESSIVE_COLUMNS_SCHEMA
                measurement_count_schema = $PROGRESSIVE_COUNT_SCHEMA
            }
        }
        {
            type = ndarray
            name = reconstruct
            label = incremental-dense-reconstructor-f32
            config = {
                measurement_count = 12
                measurement_block_capacity = 12
                blocks_per_sample = 4
                reconstructed_count = 5
                initial_reconstructor = null
                measurements_schema = $PROGRESSIVE_MEASUREMENTS_SCHEMA
                column_indices_schema = $PROGRESSIVE_COLUMNS_SCHEMA
                measurement_count_schema = $PROGRESSIVE_COUNT_SCHEMA
                reconstructed_schema = $CONTROLLER_RESIDUAL_ERROR_V1
                reconstructor_schema = $PWFS_RECONSTRUCTOR_V1
            }
        }
        {
            type = ndarray
            name = gate
            label = progressive-reconstruction-gate-f32
            config = { extent = 5 reject_sequence = 201 }
        }
    ]
    links = [
        { output = "sense:measurements" input = "reconstruct:measurements" }
        { output = "sense:columns" input = "reconstruct:columns" }
        { output = "sense:count" input = "reconstruct:count" }
        { output = "reconstruct:reconstructed" input = "gate:reconstructed" }
    ]
    inputs = [
        "sense:row-block"
        "sense:pupil-origins"
        "sense:pupil-mask"
        "reconstruct:reconstructor"
    ]
    outputs = [ "gate:accepted" ]
}
"""

const PROGRESSIVE_SHACK_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "shack_hartmann_row_reconstruction.toml"),
)
const PROGRESSIVE_PYRAMID_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_row_reconstruction.toml"),
)
const PROGRESSIVE_ROW_BLOCK = Symbol("row-block")
const PROGRESSIVE_INPUT_NAMES = (PROGRESSIVE_ROW_BLOCK,)

progressive_inputs(row_block) =
    NamedTuple{PROGRESSIVE_INPUT_NAMES}((row_block,))
progressive_metadata(sample) =
    NamedTuple{PROGRESSIVE_INPUT_NAMES}((sample,))

function prepare_cpu_progressive_graph_text(text; algorithms, kwargs...)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        return prepare_cpu_progressive_graph(path; algorithms, kwargs...)
    end
end

function progressive_row_major_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function progressive_shack_image()
    configuration = PROGRESSIVE_SHACK_FIXTURE["configuration"]
    return progressive_row_major_matrix(
        Float32,
        PROGRESSIVE_SHACK_FIXTURE["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
end

function progressive_pyramid_image()
    configuration = PROGRESSIVE_PYRAMID_FIXTURE["configuration"]
    return progressive_row_major_matrix(
        Float32,
        PROGRESSIVE_PYRAMID_FIXTURE["image"],
        configuration["image_rows"],
        configuration["image_columns"],
    )
end

function progressive_reconstructor(fixture)
    configuration = fixture["configuration"]
    rows = haskey(configuration, "reconstructed_count") ?
        configuration["reconstructed_count"] : configuration["actuator_count"]
    columns = div(length(configuration["initial_reconstructor"]), rows)
    return progressive_row_major_matrix(
        Float32,
        configuration["initial_reconstructor"],
        rows,
        columns,
    )
end

function progressive_shack_fused_output(image)
    configuration = PROGRESSIVE_SHACK_FIXTURE["configuration"]
    prepared = prepare_algorithm(
        ShackHartmannRowReconstructorF32,
        ShackHartmannRowReconstructorF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            row_block_rows=configuration["row_block_rows"],
            subaperture_size=(configuration["subaperture_rows"], configuration["subaperture_columns"]),
            subaperture_count=configuration["subaperture_count"],
            reconstructed_count=configuration["actuator_count"],
            initial_subaperture_origins=[
                Int.(origin) for
                origin in configuration["initial_subaperture_origins"]
            ],
            coordinate_scale=Float32(configuration["coordinate_scale"]),
            pixel_threshold=Float32(configuration["pixel_threshold"]),
            flux_threshold=Float32(configuration["flux_threshold"]),
            reference_slopes=nothing,
            active=nothing,
            initial_reconstructor=Float32.(configuration["initial_reconstructor"]),
            reconstructed_schema=configuration["reconstructed_schema"],
        ),
    )
    output = zeros(Float32, configuration["actuator_count"])
    for first_row in PROGRESSIVE_SHACK_FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + 1), :]
        process!(
            output,
            prepared,
            block,
            (
                row_block=SampleMetadata(
                    100,
                    first_row;
                    terminal=first_row == configuration["image_rows"] - 1,
                ),
            ),
        )
    end
    return output
end

function progressive_pyramid_fused_output(image)
    configuration = PROGRESSIVE_PYRAMID_FIXTURE["configuration"]
    prepared = prepare_algorithm(
        PyramidRowReconstructorF32,
        PyramidRowReconstructorF32Configuration(
            image_size=(configuration["image_rows"], configuration["image_columns"]),
            row_block_rows=configuration["row_block_rows"],
            pupil_size=(configuration["pupil_rows"], configuration["pupil_columns"]),
            pupil_origins=[Int.(origin) for origin in configuration["pupil_origins"]],
            pupil_mask=Bool.(configuration["pupil_mask"]),
            reconstructed_count=configuration["reconstructed_count"],
            initial_reconstructor=Float32.(configuration["initial_reconstructor"]),
            reconstructed_schema=configuration["reconstructed_schema"],
        ),
    )
    pixels = zeros(Float32, PYRAMID_PUPIL_COUNT, 3)
    mean = zeros(Float32, 1)
    reconstructed = zeros(Float32, configuration["reconstructed_count"])
    for (sequence, frame) in (
        (200, image),
        (201, 2.0f0 .* image),
        (202, image),
    )
        for first_row in PROGRESSIVE_PYRAMID_FIXTURE["block_offsets"]
            block = @view frame[(first_row + 1):(first_row + 2), :]
            process!(
                pixels,
                mean,
                reconstructed,
                prepared,
                block,
                (
                    row_block=SampleMetadata(
                        sequence,
                        first_row;
                        terminal=first_row == 6,
                    ),
                ),
            )
        end
    end
    return reconstructed
end

function progressive_graph_allocation_bytes(outputs, graph, inputs, metadata)
    return @allocated process!(outputs, graph, inputs, metadata)
end

function process_progressive_shack_frame!(output, graph, image, sequence)
    availability = nothing
    retained = copy(output)
    for first_row in PROGRESSIVE_SHACK_FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + 1), :]
        sample = SampleMetadata(sequence, first_row; terminal=first_row == 5)
        availability = process!(
            (reconstructed=output,),
            graph,
            progressive_inputs(block),
            progressive_metadata(sample),
        )
        @test availability == (reconstructed=(first_row == 5),)
        first_row == 5 || @test output == retained
    end
    return availability
end

function process_progressive_pyramid_frame!(output, graph, image, sequence)
    availability = nothing
    retained = copy(output)
    for first_row in PROGRESSIVE_PYRAMID_FIXTURE["block_offsets"]
        block = @view image[(first_row + 1):(first_row + 2), :]
        sample = SampleMetadata(sequence, first_row; terminal=first_row == 6)
        availability = process!(
            (accepted=output,),
            graph,
            progressive_inputs(block),
            progressive_metadata(sample),
        )
        expected = first_row == 6 ? (accepted=true,) : (accepted=false,)
        first_row == 6 || @test output == retained
        if sequence != 201 || first_row != 6
            @test availability == expected
        end
    end
    return availability
end

@testset "Shack-Hartmann Graph reconstructs as measurements become ready" begin
    graph = prepare_graph_text(
        PROGRESSIVE_SHACK_GRAPH_TEXT;
        algorithms=(
            ShackHartmannMeasurementBlockF32,
            IncrementalDenseReconstructorF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) == (:sense, :reconstruct)
    @test graph.output_conditional == (reconstructed=true,)
    @test graph.nodes[2].input_formats[1].rate == (0x00000006, 0x00000001)
    @test replace_parameters!(
        graph,
        :reconstructor => progressive_reconstructor(PROGRESSIVE_SHACK_FIXTURE),
    ) === graph

    image = progressive_shack_image()
    reconstructed = fill(-1.0f0, 3)
    availability = (reconstructed=false,)
    for first_row in 0:1
        block = @view image[(first_row + 1):(first_row + 1), :]
        availability = process!(
            (reconstructed=reconstructed,),
            graph,
            progressive_inputs(block),
            progressive_metadata(SampleMetadata(99, first_row)),
        )
    end
    workspace = graph.nodes[2].prepared.workspace
    @test availability == (reconstructed=false,)
    @test reconstructed == fill(-1.0f0, 3)
    @test any(!iszero, workspace.candidate_reconstructed)
    reset!(graph)

    process_progressive_shack_frame!(reconstructed, graph, image, 100)
    @test reconstructed == progressive_shack_fused_output(image)
    @test reconstructed ≈ Float32.(
        PROGRESSIVE_SHACK_FIXTURE["expected"]["reconstructed"],
    ) atol = 2.0f-6
    @test !graph.nodes[1].prepared.workspace.frame.active
    @test !graph.nodes[2].prepared.workspace.active
end

@testset "pyramid Graph matches the fused oracle with per-Node gain commits" begin
    graph = prepare_graph_text(
        PROGRESSIVE_PYRAMID_GRAPH_TEXT;
        algorithms=(
            PyramidMeasurementBlockF32,
            IncrementalDenseReconstructorF32,
            ProgressiveReconstructionGateF32,
        ),
    )
    @test map(node -> node.name, graph.nodes) == (:sense, :reconstruct, :gate)
    @test replace_parameters!(
        graph,
        :reconstructor => progressive_reconstructor(PROGRESSIVE_PYRAMID_FIXTURE),
    ) === graph

    image = progressive_pyramid_image()
    accepted = fill(-1.0f0, 5)
    process_progressive_pyramid_frame!(accepted, graph, image, 200)
    @test all(iszero, accepted)
    workspace = graph.nodes[1].prepared.workspace
    committed_gain = pyramid_row_normalization_gain(workspace.pixels)
    @test committed_gain == inv(31.5f0)

    retained = copy(accepted)
    rejection = process_progressive_pyramid_frame!(
        accepted,
        graph,
        2.0f0 .* image,
        201,
    )
    @test rejection === ProgressiveReconstructionGateRejected
    @test accepted == retained
    @test pyramid_row_normalization_gain(workspace.pixels) != committed_gain
    @test pyramid_row_normalization_gain(workspace.pixels) == inv(63.0f0)

    process_progressive_pyramid_frame!(accepted, graph, image, 202)
    expected = 0.5f0 .* Float32.(
        PROGRESSIVE_PYRAMID_FIXTURE["frames"][2]["expected_reconstructed"],
    )
    @test accepted == progressive_pyramid_fused_output(image)
    @test accepted ≈ expected atol = 4.0f-8 rtol = 0
    @test !FilterGraphs.output_metadata(graph).accepted.discontinuity
end

@testset "composed progressive Graph calls allocate no Julia heap" begin
    shack = prepare_graph_text(
        PROGRESSIVE_SHACK_GRAPH_TEXT;
        algorithms=(
            ShackHartmannMeasurementBlockF32,
            IncrementalDenseReconstructorF32,
        ),
    )
    replace_parameters!(
        shack,
        :reconstructor => progressive_reconstructor(PROGRESSIVE_SHACK_FIXTURE),
    )
    image = progressive_shack_image()
    outputs = (reconstructed=zeros(Float32, 3),)
    @test progressive_graph_allocation_bytes(
        outputs,
        shack,
        progressive_inputs(@view(image[1:1, :])),
        progressive_metadata(SampleMetadata(300, 0)),
    ) == 0
    for first_row in 1:4
        process!(
            outputs,
            shack,
            progressive_inputs(@view(image[(first_row + 1):(first_row + 1), :])),
            progressive_metadata(SampleMetadata(300, first_row)),
        )
    end
    @test progressive_graph_allocation_bytes(
        outputs,
        shack,
        progressive_inputs(@view(image[6:6, :])),
        progressive_metadata(SampleMetadata(300, 5; terminal=true)),
    ) == 0
end

@testset "CPU progressive worker MVM trace records handoff epochs" begin
    trace = FilterGraphs.CpuProgressiveMvmTrace(1800, 2, 2, UInt64(4))
    FilterGraphs._cpu_record_mvm_trace!(trace, 1, UInt64(5), 1799)
    FilterGraphs._cpu_record_mvm_trace!(trace, 2, UInt64(5), 1800)
    FilterGraphs._cpu_record_worker_terminal_trace!(trace, 2, UInt64(5))
    FilterGraphs._cpu_record_coordinator_wait_start!(trace, UInt64(5))
    FilterGraphs._cpu_record_coordinator_wait_end!(trace, UInt64(5))
    slot = FilterGraphs._cpu_trace_index(trace, 1, 2)
    @test trace.handoff_epochs[slot] == UInt64(5)
    @test trace.timestamps_ns[slot] > 0
    @test trace.worker_terminal_timestamps_ns[slot] > 0
    @test trace.coordinator_wait_start_ns[1] > 0
    @test trace.coordinator_wait_end_ns[1] >= trace.coordinator_wait_start_ns[1]
    @test all(iszero, trace.timestamps_ns[1:(slot - 1)])

    graph = prepare_cpu_progressive_graph_text(
        PROGRESSIVE_SHACK_GRAPH_TEXT;
        algorithms=(
            ShackHartmannMeasurementBlockF32,
            IncrementalDenseReconstructorF32,
        ),
        workers=1,
    )
    try
        replace_parameters!(
            graph,
            :reconstructor => progressive_reconstructor(PROGRESSIVE_SHACK_FIXTURE),
        )
        image = progressive_shack_image()
        output = zeros(Float32, 3)
        process_progressive_shack_frame!(output, graph, image, 700)

        @test FilterGraphs.install_cpu_progressive_trace!(
            graph; capacity=2, item=8,
        ) === graph
        for first_row in 0:4
            availability = process!(
                (reconstructed=output,),
                graph,
                progressive_inputs(@view(image[(first_row + 1):(first_row + 1), :])),
                progressive_metadata(SampleMetadata(701, first_row)),
            )
            @test availability == (reconstructed=false,)
        end
        @test progressive_graph_allocation_bytes(
            (reconstructed=output,),
            graph,
            progressive_inputs(@view(image[6:6, :])),
            progressive_metadata(SampleMetadata(701, 5; terminal=true)),
        ) == 0
        exported = FilterGraphs.cpu_progressive_trace(graph)
        records = exported[:reconstruct].records
        @test exported[:reconstruct].base_handoff_epoch == UInt64(1)
        @test length(records) == 1
        @test records[1].trace_ordinal == 1
        @test records[1].handoff_epoch == UInt64(2)
        @test records[1].worker == 1
        @test records[1].item == 8
        @test records[1].timestamp_ns > 0
        @test records[1].worker_terminal_timestamp_ns > 0
        @test records[1].coordinator_wait_start_ns > 0
        @test records[1].coordinator_wait_end_ns >=
              records[1].coordinator_wait_start_ns
        @test records[1].coordinator_wait_end_ns >=
              records[1].worker_terminal_timestamp_ns
    finally
        close(graph)
    end
end

@testset "CPU row workers preserve fused SH and PW block-row graph results" begin
    for layout in (:shared, :sharded), workers in (1, 2)
        graph = prepare_cpu_progressive_graph_text(
            PROGRESSIVE_SHACK_GRAPH_TEXT;
            algorithms=(ShackHartmannMeasurementBlockF32, IncrementalDenseReconstructorF32),
            workers, matrix_layout=layout,
            worker_thread_ids=Threads.nthreads() >= workers + 1 ? collect(2:(workers + 1)) : nothing,
        )
        try
            replace_parameters!(graph, :reconstructor => progressive_reconstructor(PROGRESSIVE_SHACK_FIXTURE))
            image = progressive_shack_image()
            output = fill(-71.0f0, 3)
            process_progressive_shack_frame!(output, graph, image, 800)
            @test output == progressive_shack_fused_output(image)
            @test FilterGraphs.cpu_progressive_counters(graph)[:reconstruct].numerical_fallbacks == 0
            reset!(graph)
            process_progressive_shack_frame!(output, graph, image, 801)
            @test output == progressive_shack_fused_output(image)
        finally
            close(graph)
        end
    end
    for layout in (:shared, :sharded), workers in (1, 2, 4)
        graph = prepare_cpu_progressive_graph_text(
            PROGRESSIVE_PYRAMID_GRAPH_TEXT;
            algorithms=(PyramidMeasurementBlockF32, IncrementalDenseReconstructorF32, ProgressiveReconstructionGateF32),
            workers, matrix_layout=layout,
            worker_thread_ids=Threads.nthreads() >= workers + 1 ? collect(2:(workers + 1)) : nothing,
        )
        try
            replace_parameters!(graph, :reconstructor => progressive_reconstructor(PROGRESSIVE_PYRAMID_FIXTURE))
            image = progressive_pyramid_image()
            output = fill(-71.0f0, 5)
            process_progressive_pyramid_frame!(output, graph, image, 200)
            retained = copy(output)
            @test process_progressive_pyramid_frame!(output, graph, 2.0f0 .* image, 201) === ProgressiveReconstructionGateRejected
            @test output == retained
            process_progressive_pyramid_frame!(output, graph, image, 202)
            @test output == progressive_pyramid_fused_output(image)
            @test FilterGraphs.cpu_progressive_counters(graph)[:reconstruct].numerical_fallbacks == 0
        finally
            close(graph)
        end
    end
end

@testset "CPU progressive batches preserve ordered FMA across row shards and fallback" begin
    rows, columns, capacity = 65, 11, 6
    graph_text = """
        filter.graph = {
            nodes = [{ type=ndarray name=reconstruct label=incremental-dense-reconstructor-f32
                config={measurement_count=$columns measurement_block_capacity=$capacity
                    blocks_per_sample=2 reconstructed_count=$rows initial_reconstructor=null
                    measurements_schema=$PROGRESSIVE_MEASUREMENTS_SCHEMA
                    column_indices_schema=$PROGRESSIVE_COLUMNS_SCHEMA
                    measurement_count_schema=$PROGRESSIVE_COUNT_SCHEMA
                    reconstructed_schema=$CONTROLLER_RESIDUAL_ERROR_V1
                    reconstructor_schema=$PWFS_RECONSTRUCTOR_V1} }]
            inputs=["reconstruct:measurements" "reconstruct:columns" "reconstruct:count"
                "reconstruct:reconstructor"]
            outputs=["reconstruct:reconstructed"]
        }
        """
    matrix = Float32[sin(0.17row + 0.31column) for row in 1:rows, column in 1:columns]
    matrix[end, end] = floatmax(Float32) / 8f0
    first_block = (measurements=Float32[0.2, -0.4, 0.1, 3, 5, 19],
        columns=UInt32[9, 0, 2, 1, 7, 5], count=UInt32[6])
    last_block = (measurements=Float32[-3, 0.2, 1, -2, 1, NaN],
        columns=UInt32[8, 4, 10, 3, 6, typemax(UInt32)], count=UInt32[5])
    output = (reconstructed=zeros(Float32, rows),)
    for matrix_layout in (:shared, :sharded)
        graph = prepare_cpu_progressive_graph_text(graph_text;
            algorithms=(IncrementalDenseReconstructorF32,), workers=2, matrix_layout)
        try
            replace_parameters!(graph, :reconstructor => matrix)
            FilterGraphs.install_cpu_progressive_trace!(graph; capacity=3, item=3)
            for sequence in 1:3
                last_block.measurements[3] = (1f0, 3f0, 9f0)[sequence]
                first_metadata = (measurements=SampleMetadata(sequence), columns=nothing, count=nothing)
                terminal_metadata = (measurements=SampleMetadata(sequence, 1; terminal=true),
                    columns=nothing, count=nothing)
                @test process!(output, graph, first_block, first_metadata) == (reconstructed=false,)
                retained = copy(output.reconstructed)
                status = process!(output, graph, last_block, terminal_metadata)
                if sequence < 3
                    expected = zeros(Float32, rows)
                    for input in (first_block, last_block), row in 1:rows, item in 1:only(input.count)
                        coefficient = matrix[row, Int(input.columns[item]) + 1]
                        expected[row] = fma(coefficient, input.measurements[item], expected[row])
                    end
                    @test status == (reconstructed=true,)
                    @test isequal(output.reconstructed, expected)
                else
                    @test status === IncrementalDenseReconstructionNonFiniteOutput
                    @test isequal(output.reconstructed, retained)
                end
            end
            @test FilterGraphs.cpu_progressive_counters(graph)[:reconstruct].numerical_fallbacks == 2
            records = FilterGraphs.cpu_progressive_trace(graph)[:reconstruct].records
            @test length(records) == 6
            @test all(record -> record.item == 3 &&
                0 < record.timestamp_ns <= record.worker_terminal_timestamp_ns, records)
        finally
            close(graph)
        end
    end
end
