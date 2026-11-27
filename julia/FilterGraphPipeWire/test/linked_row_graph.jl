using TOML

const LINKED_ROW_GRAPH_FIXTURE = joinpath(@__DIR__, "data", "pixel-pyramid-row.conf")
const LINKED_ROW_NUMERICAL_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "pyramid_pixel_row.toml"),
)

prepared_linked_row_graph() = prepare_graph(
    LINKED_ROW_GRAPH_FIXTURE;
    algorithms=(PixelCalibrationRowU16F32, PyramidPixelRowF32),
)

function linked_row_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function process_linked_row!(
    node::PipeWireNode,
    row::Vector{UInt16},
    native_outputs,
    sequence::Integer,
    offset::Integer;
    terminal::Bool=false,
    metadata::Bool=true,
    measure::Bool=false,
)
    flags = terminal ? SPA.META_HEADER_FLAG_MARKER : UInt32(0)
    header = metadata ?
        BufferHeader(
            flags,
            UInt32(offset),
            Int64(3_000 + offset),
            Int64(0),
            UInt64(sequence),
        ) : nothing
    native_input = native_callback_buffer(row; header)
    callback = node.filter.callbacks.on_process
    GC.@preserve row native_input native_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_input)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_outputs)),
            UInt32(length(native_outputs)),
        )
        if measure
            return @allocated(callback(node.filter, inputs, outputs))
        end
        callback(node.filter, inputs, outputs)
    end
    return nothing
end

function process_linked_frame!(
    node::PipeWireNode,
    image::AbstractMatrix{UInt16},
    native_outputs,
    sequence::Integer;
    measure::Bool=false,
)
    unavailable =
        PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
    availability = Bool[]
    allocations = Int[]
    for offset in 0:2:6
        row = vec(copy(@view image[(offset + 1):(offset + 2), :]))
        allocation = process_linked_row!(
            node,
            row,
            native_outputs,
            sequence,
            offset;
            terminal=offset == 6,
            measure,
        )
        push!(
            availability,
            all(
                iszero(native_callback_flags(native_outputs, index) & unavailable) for
                index in eachindex(native_outputs)
            ),
        )
        measure && push!(allocations, allocation)
    end
    return (; availability, allocations)
end

@testset "linked production row Graph matches the repository fixture through the callback" begin
    graph = prepared_linked_row_graph()
    node = PipeWireNode(graph; rate=(1_000, 1), connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)

    fixture = LINKED_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = linked_row_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    selected = count(configuration["pupil_mask"])
    reconstruction_pixels = fill(-1.0f0, PYRAMID_PUPIL_COUNT * selected)
    mean_pupil_intensity = Float32[-1]
    native_reconstruction = native_callback_buffer(
        reconstruction_pixels;
        header_available=true,
    )
    native_mean = native_callback_buffer(
        mean_pupil_intensity;
        header_available=true,
    )
    native_outputs = vcat(native_reconstruction, native_mean)

    first_frame = process_linked_frame!(
        node,
        image,
        native_outputs,
        fixture["frames"][1]["sequence"],
    )
    @test first_frame.availability == [false, false, false, true]
    first_expected = linked_row_matrix(
        Float32,
        fixture["frames"][1]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reshape(reconstruction_pixels, PYRAMID_PUPIL_COUNT, selected) ==
          first_expected
    @test mean_pupil_intensity ==
          Float32[fixture["frames"][1]["expected_mean_pupil_intensity"]]

    second_frame = process_linked_frame!(
        node,
        image,
        native_outputs,
        fixture["frames"][2]["sequence"];
        measure=true,
    )
    @test second_frame.availability == [false, false, false, true]
    @test second_frame.allocations == zeros(Int, 4)
    second_expected = linked_row_matrix(
        Float32,
        fixture["frames"][2]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reshape(reconstruction_pixels, PYRAMID_PUPIL_COUNT, selected) ==
          second_expected
    @test mean_pupil_intensity ==
          Float32[fixture["frames"][2]["expected_mean_pupil_intensity"]]
    expected_header = BufferHeader(
        SPA.META_HEADER_FLAG_MARKER,
        UInt32(0),
        Int64(3_006),
        Int64(0),
        UInt64(fixture["frames"][2]["sequence"]),
    )
    @test native_callback_header(native_outputs, 1) == expected_header
    @test native_callback_header(native_outputs, 2) == expected_header
    @test node.owner.process_count == 8
    @test last_error(node) === nothing
    close(node)
end

@testset "linked row callback discards incomplete work and adopts one Parameter generation" begin
    graph = prepared_linked_row_graph()
    node = PipeWireNode(graph; rate=(1_000, 1), connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)

    fixture = LINKED_ROW_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    image = linked_row_matrix(
        UInt16,
        fixture["image"],
        configuration["image_size"][1],
        configuration["image_size"][2],
    )
    selected = count(configuration["pupil_mask"])
    reconstruction_pixels = zeros(Float32, PYRAMID_PUPIL_COUNT * selected)
    mean_pupil_intensity = zeros(Float32, 1)
    native_outputs = vcat(
        native_callback_buffer(reconstruction_pixels; header_available=true),
        native_callback_buffer(mean_pupil_intensity; header_available=true),
    )
    unavailable =
        PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

    process_linked_frame!(node, image, native_outputs, 20)
    baseline_reconstruction = copy(reconstruction_pixels)
    baseline_mean = copy(mean_pupil_intensity)

    first_row = vec(copy(@view image[1:2, :]))
    second_row = vec(copy(@view image[3:4, :]))
    process_linked_row!(node, first_row, native_outputs, 21, 0)
    @test all(
        !iszero(native_callback_flags(native_outputs, index) & unavailable) for
        index in eachindex(native_outputs)
    )
    process_linked_row!(
        node,
        second_row,
        native_outputs,
        21,
        2;
        metadata=false,
    )
    @test reconstruction_pixels == baseline_reconstruction
    @test mean_pupil_intensity == baseline_mean
    @test last_error(node) === nothing

    recovery = process_linked_frame!(node, image, native_outputs, 22; measure=true)
    @test recovery.availability == [false, false, false, true]
    @test recovery.allocations == zeros(Int, 4)
    recovery_expected = linked_row_matrix(
        Float32,
        fixture["frames"][2]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reshape(reconstruction_pixels, PYRAMID_PUPIL_COUNT, selected) ==
          recovery_expected
    @test !iszero(
        native_callback_header(native_outputs, 1).flags &
        SPA.META_HEADER_FLAG_DISCONT,
    )

    process_linked_row!(node, first_row, native_outputs, 23, 0)
    previous_plans = map(graph.nodes) do graph_node
        graph_node.prepared.plan
    end
    replacement_mask = linked_row_matrix(
        Bool,
        fixture["replacement"]["pupil_mask"],
        configuration["pupil_size"][1],
        configuration["pupil_size"][2],
    )
    replace_parameters!(
        node,
        :flat => fill(2.0f0, 8, 8),
        Symbol("pupil-mask") => replacement_mask,
    )
    @test node.owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_PENDING
    @test process_linked_row!(
        node,
        second_row,
        native_outputs,
        23,
        2;
        measure=true,
    ) == 0
    @test node.owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_AVAILABLE
    @test all(
        graph.nodes[index].prepared.plan !== previous_plans[index] for
        index in eachindex(graph.nodes)
    )
    for offset in (4, 6)
        row = vec(copy(@view image[(offset + 1):(offset + 2), :]))
        process_linked_row!(
            node,
            row,
            native_outputs,
            23,
            offset;
            terminal=offset == 6,
        )
    end
    @test reconstruction_pixels == vec(recovery_expected)

    replacement = process_linked_frame!(
        node,
        image,
        native_outputs,
        24;
        measure=true,
    )
    @test replacement.availability == [false, false, false, true]
    @test replacement.allocations == zeros(Int, 4)
    replacement_expected = 2.0f0 .* linked_row_matrix(
        Float32,
        fixture["replacement"]["expected_reconstruction_pixels"],
        PYRAMID_PUPIL_COUNT,
        selected,
    )
    @test reshape(reconstruction_pixels, PYRAMID_PUPIL_COUNT, selected) ==
          replacement_expected
    @test mean_pupil_intensity == Float32[63]
    expected_flags =
        SPA.META_HEADER_FLAG_MARKER | SPA.META_HEADER_FLAG_DISCONT
    @test native_callback_header(native_outputs, 1) == BufferHeader(
        expected_flags,
        UInt32(0),
        Int64(3_006),
        Int64(0),
        UInt64(24),
    )
    @test node.owner.process_count == 18
    @test last_error(node) === nothing
    close(node)
end
