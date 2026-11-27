using TOML

const CALLBACK_CLOSED_LOOP_GRAPH_FIXTURE = joinpath(@__DIR__, "data", "closed-loop-correction.conf")
const CALLBACK_CLOSED_LOOP_NUMERICAL_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "data", "algorithms", "closed_loop_correction.toml"),
)

function callback_closed_loop_matrix(::Type{T}, values, rows, columns) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function prepared_callback_closed_loop_graph()
    graph = prepare_graph(
        CALLBACK_CLOSED_LOOP_GRAPH_FIXTURE;
        algorithms=algorithms(),
    )
    fixture = CALLBACK_CLOSED_LOOP_NUMERICAL_FIXTURE
    configuration = fixture["configuration"]
    parameters = fixture["parameters"]
    replace_parameters!(
        graph,
        "shape-to-hidden" => callback_closed_loop_matrix(
            Float32,
            parameters["shape_to_hidden"],
            configuration["hidden_mode_count"],
            configuration["state_length"],
        ),
        "hidden-to-shape" => callback_closed_loop_matrix(
            Float32,
            parameters["hidden_to_shape"],
            configuration["state_length"],
            configuration["hidden_mode_count"],
        ),
    )
    return graph
end

function callback_closed_loop_inputs(residual_error, constraint_feedback; header)
    return vcat(
        native_callback_buffer(residual_error; header),
        native_callback_buffer(constraint_feedback),
    )
end

function invoke_native_graph_process(
    filter::T,
    native_inputs::NativeCallbackStorage{PipeWireAO.LibPipeWire.pw_ndarray_filter_buffer},
    native_outputs::NativeCallbackStorage{PipeWireAO.LibPipeWire.pw_ndarray_filter_buffer},
) where {T<:NdArrayFilter}
    events = PipeWireAO._ndarray_filter_events(filter)
    return GC.@preserve native_inputs native_outputs ccall(
        events.process,
        Cint,
        (
            Ref{T},
            Ptr{PipeWireAO.LibPipeWire.pw_ndarray_filter_buffer},
            UInt32,
            Ptr{PipeWireAO.LibPipeWire.pw_ndarray_filter_buffer},
            UInt32,
        ),
        filter,
        pointer(native_inputs),
        UInt32(length(native_inputs)),
        pointer(native_outputs),
        UInt32(length(native_outputs)),
    )
end

function callback_closed_loop_header(sequence::Integer)
    return BufferHeader(
        SPA.META_HEADER_FLAG_MARKER,
        UInt32(0),
        Int64(4_000 + sequence),
        Int64(0),
        UInt64(sequence),
    )
end

@testset "closed-loop callback preserves host feedback across rejection" begin
    graph = prepared_callback_closed_loop_graph()
    node = PipeWireNode(graph; rate=(1_000, 1), connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)
    callback = node.filter.callbacks.on_process
    fixture = CALLBACK_CLOSED_LOOP_NUMERICAL_FIXTURE
    frames = fixture["frames"]
    workspace = only(graph.nodes).prepared.workspace
    correction = fill(-1.0f0, 2)
    controller_state = fill(-2.0f0, 2)
    native_outputs = vcat(
        native_callback_buffer(correction; header_available=true),
        native_callback_buffer(controller_state; header_available=true),
    )
    unavailable =
        PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

    first = frames[1]
    first_residual = Float32.(first["residual_error"])
    first_feedback = Float32.(first["constraint_feedback"])
    first_inputs = callback_closed_loop_inputs(
        first_residual,
        first_feedback;
        header=callback_closed_loop_header(1),
    )
    GC.@preserve first_residual first_feedback correction controller_state first_inputs native_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(first_inputs)),
            UInt32(length(first_inputs)),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_outputs)),
            UInt32(length(native_outputs)),
        )
        @test @inferred(callback(node.filter, inputs, outputs)) === nothing
    end
    @test correction == Float32.(first["correction"])
    @test controller_state == Float32.(first["controller_state"])
    @test closed_loop_correction_state(workspace) ==
          Float32.(first["controller_state"])
    @test all(
        iszero(native_callback_flags(native_outputs, index) & unavailable) for
        index in eachindex(native_outputs)
    )

    second = frames[2]
    host_feedback = Float32.(second["constraint_feedback"])
    rejected_residual = Float32[NaN, 0]
    rejected_inputs = callback_closed_loop_inputs(
        rejected_residual,
        host_feedback;
        header=callback_closed_loop_header(2),
    )
    fill!(correction, 71.0f0)
    fill!(controller_state, 72.0f0)
    GC.@preserve rejected_residual host_feedback correction controller_state rejected_inputs native_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(rejected_inputs)),
            UInt32(length(rejected_inputs)),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_outputs)),
            UInt32(length(native_outputs)),
        )
        callback(node.filter, inputs, outputs)
        @test @allocated(callback(node.filter, inputs, outputs)) == 0
    end
    @test correction == fill(71.0f0, 2)
    @test controller_state == fill(72.0f0, 2)
    @test all(
        !iszero(native_callback_flags(native_outputs, index) & unavailable) for
        index in eachindex(native_outputs)
    )
    @test closed_loop_correction_state(workspace) ==
          Float32.(first["controller_state"])
    @test host_feedback == Float32.(second["constraint_feedback"])

    second_residual = Float32.(second["residual_error"])
    second_inputs = callback_closed_loop_inputs(
        second_residual,
        host_feedback;
        header=callback_closed_loop_header(2),
    )
    GC.@preserve second_residual host_feedback correction controller_state second_inputs native_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(second_inputs)),
            UInt32(length(second_inputs)),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_outputs)),
            UInt32(length(native_outputs)),
        )
        callback(node.filter, inputs, outputs)
        @test correction == Float32.(second["correction"])
        @test controller_state == Float32.(second["controller_state"])
        @test closed_loop_correction_state(workspace) ==
              Float32.(second["controller_state"])
        @test @allocated(callback(node.filter, inputs, outputs)) == 0
    end
    @test host_feedback == Float32.(second["constraint_feedback"])
    @test last_error(node) === nothing
    close(node)
end

@testset "metadata publication failure leaves scientific state unchanged" begin
    graph = prepared_callback_closed_loop_graph()
    node = PipeWireNode(graph; rate=(1_000, 1), connect=false)
    FilterGraphPipeWire._prepare_owner(node.owner)
    callback = node.filter.callbacks.on_process
    frames = CALLBACK_CLOSED_LOOP_NUMERICAL_FIXTURE["frames"]
    workspace = only(graph.nodes).prepared.workspace

    first = frames[1]
    first_residual = Float32.(first["residual_error"])
    first_feedback = Float32.(first["constraint_feedback"])
    correction = zeros(Float32, 2)
    controller_state = zeros(Float32, 2)
    first_inputs = callback_closed_loop_inputs(
        first_residual,
        first_feedback;
        header=callback_closed_loop_header(1),
    )
    publishable_outputs = vcat(
        native_callback_buffer(correction; header_available=true),
        native_callback_buffer(controller_state; header_available=true),
    )
    GC.@preserve first_residual first_feedback correction controller_state first_inputs publishable_outputs begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(first_inputs)),
            UInt32(length(first_inputs)),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(publishable_outputs)),
            UInt32(length(publishable_outputs)),
        )
        callback(node.filter, inputs, outputs)
    end
    committed_state = copy(closed_loop_correction_state(workspace))

    second = frames[2]
    second_residual = Float32.(second["residual_error"])
    host_feedback = Float32.(second["constraint_feedback"])
    second_inputs = callback_closed_loop_inputs(
        second_residual,
        host_feedback;
        header=callback_closed_loop_header(2),
    )
    unpublished_outputs = vcat(
        native_callback_buffer(correction;header_available=false),
        native_callback_buffer(controller_state;header_available=false),
    )
    result = GC.@preserve second_residual host_feedback correction controller_state second_inputs unpublished_outputs begin
        invoke_native_graph_process(node.filter, second_inputs, unpublished_outputs)
    end
    @test result == -Base.Libc.EFAULT
    @test node.filter.callback_error[] isa InvalidStateException
    @test closed_loop_correction_state(workspace) == committed_state
    @test closed_loop_correction_state(workspace) != Float32.(second["controller_state"])
    @test host_feedback == Float32.(second["constraint_feedback"])
    @test node.owner.process_count == 1
    close(node)
end
