"""One exact external Graph Port prepared for borrowed-buffer projection."""
struct PortBinding{T,N,Layout<:ArrayLayout}
    shape::NTuple{N,Int}
    elements::Int
    bytes::Int
end

function PortBinding(
    format::AlgorithmFormat{N},
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
) where {N}
    return PortBinding(format, format.element_type, boundary_layout)
end

function PortBinding(
    format::AlgorithmFormat{N},
    ::Type{T},
    boundary_layout::Union{Nothing,ArrayLayout},
) where {N,T}
    isbitstype(T) || throw(ArgumentError("PipeWire Port elements must be bitstypes"))
    format.layout isa ColumnMajorLayout || throw(
        ArgumentError("the Julia Graph PipeWire adapter requires column-major Ports"),
    )
    elements = 1
    for dimension in format.shape
        elements = Base.checked_mul(elements, dimension)
    end
    bytes = Base.checked_mul(elements, sizeof(T))
    layout = isnothing(boundary_layout) ? format.layout : boundary_layout
    return PortBinding{T,N,typeof(layout)}(format.shape, elements, bytes)
end

function _port_bindings(
    formats::NamedTuple,
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    bindings = map(values(formats)) do format
        PortBinding(format, boundary_layout)
    end
    return NamedTuple{keys(formats)}(bindings)
end

_allocate_warm_buffer(binding::PortBinding{T}) where {T} = zeros(T, binding.shape)

function _warm_buffers(bindings::NamedTuple)
    return NamedTuple{keys(bindings)}(map(_allocate_warm_buffer, values(bindings)))
end

@inline function _borrow_pointer(
    binding::PortBinding{T,N,ColumnMajorLayout},
    data::Ptr{T},
    ::Val{Writable},
) where {T,N,Writable}
    data == C_NULL && throw(ArgumentError("borrowed array pointer is null"))
    iszero(UInt(data) % UInt(Base.datatype_alignment(T))) || throw(
        ArgumentError("borrowed array pointer is not aligned for $T"),
    )
    return BorrowedArray{T,N,Writable}(
        data,
        binding.shape,
        binding.elements,
    )
end

@inline function _borrow_pointer(
    binding::PortBinding{T,N,RowMajorLayout},
    data::Ptr{T},
    ::Val{Writable},
) where {T,N,Writable}
    data == C_NULL && throw(ArgumentError("borrowed array pointer is null"))
    iszero(UInt(data) % UInt(Base.datatype_alignment(T))) || throw(
        ArgumentError("borrowed array pointer is not aligned for $T"),
    )
    return RowMajorBorrowedArray{T,N,Writable}(
        data,
        binding.shape,
        binding.elements,
    )
end

@inline function _borrow_buffer(
    binding::PortBinding{T},
    buffer::NdArrayFilterBuffer{Writable},
) where {T,Writable}
    payload_size(buffer) == binding.bytes || throw(
        DimensionMismatch("PipeWire payload size does not match the Graph Port"),
    )
    capacity(buffer) >= binding.bytes || throw(
        DimensionMismatch("PipeWire buffer capacity is smaller than the Graph Port"),
    )
    return _borrow_pointer(
        binding,
        Ptr{T}(data_pointer(buffer)),
        Val(Writable),
    )
end

@inline _borrow_buffers(::Tuple{}, buffers, index::Int) = ()

@inline function _borrow_buffers(bindings::Tuple, buffers, index::Int)
    return (
        _borrow_buffer(first(bindings), buffers[index]),
        _borrow_buffers(Base.tail(bindings), buffers, index + 1)...,
    )
end

@inline function _borrow_buffers(
    bindings::NamedTuple,
    buffers::NdArrayFilterBuffers,
)
    length(buffers) == length(bindings) || throw(
        DimensionMismatch("PipeWire callback buffer count does not match the Graph Ports"),
    )
    arrays = _borrow_buffers(values(bindings), buffers, 1)
    return NamedTuple{keys(bindings)}(arrays)
end

@inline _borrow_warm_buffers(::Tuple{}, ::Tuple{}, writable) = ()

@inline function _borrow_warm_buffers(bindings::Tuple, storage::Tuple, writable)
    array = first(storage)
    return (
        BorrowedArray(pointer(array), first(bindings).shape, writable),
        _borrow_warm_buffers(Base.tail(bindings), Base.tail(storage), writable)...,
    )
end

@inline function _borrow_warm_buffers(
    bindings::NamedTuple,
    storage::NamedTuple,
    writable,
)
    arrays = _borrow_warm_buffers(values(bindings), values(storage), writable)
    return NamedTuple{keys(bindings)}(arrays)
end

function _require_supported_graph(
    graph::Union{PreparedGraph,PreparedCapturedGraph},
)
    publication_api = if isdefined(PipeWireAO, :set_output_available!)
        hasmethod(
            PipeWireAO.buffer_header,
            Tuple{NdArrayFilterBuffer{false}},
        ) && hasmethod(
            PipeWireAO.set_buffer_header!,
            Tuple{NdArrayFilterBuffer{true},BufferHeader},
        ) && hasmethod(
            getfield(PipeWireAO, :set_output_available!),
            Tuple{NdArrayFilterBuffer{true},Bool},
        )
    else
        false
    end
    publication_api || throw(
        ArgumentError(
            "PipeWireAO does not provide ndarray output availability and Header metadata",
        ),
    )
    return graph
end

function _append_graph_ports!(
    ports,
    formats::NamedTuple,
    direction,
    base_rate;
    parameter::Bool=false,
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    for (name, format) in pairs(formats)
        push!(
            ports,
            NdArrayFilterPort(
                String(name),
                direction,
                _ndarray_format(format, base_rate; boundary_layout);
                schema=format.schema,
                parameter,
            ),
        )
    end
    return ports
end

function _graph_ports(
    graph::PreparedGraph,
    base_rate;
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    return _graph_ports(
        graph.input_formats, graph.parameter_formats, graph.output_formats,
        base_rate; boundary_layout,
    )
end

function _graph_ports(
    input_formats::NamedTuple,
    parameter_formats::NamedTuple,
    output_formats::NamedTuple,
    base_rate;
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    ports = NdArrayFilterPort[]
    sizehint!(
        ports,
        length(input_formats) + length(parameter_formats) + length(output_formats),
    )
    _append_graph_ports!(
        ports,
        input_formats,
        PipeWireAO.DIRECTION_INPUT,
        base_rate;
        boundary_layout,
    )
    _append_graph_ports!(
        ports,
        parameter_formats,
        PipeWireAO.DIRECTION_INPUT,
        nothing;
        parameter=true,
        boundary_layout,
    )
    _append_graph_ports!(
        ports,
        output_formats,
        PipeWireAO.DIRECTION_OUTPUT,
        base_rate;
        boundary_layout,
    )
    return ports
end

# Independent records are an explicit cold Port contract. Header is mandatory;
# other capabilities and preservation policy are declared by the caller.
function _metadata_ports(ports, inputs::NamedTuple, outputs::NamedTuple)
    for (direction, declarations) in ((PipeWireAO.DIRECTION_INPUT, inputs),
            (PipeWireAO.DIRECTION_OUTPUT, outputs))
        names = Set(Symbol(port.name) for port in ports if port.direction == direction && !port.parameter)
        all(name -> name in names, keys(declarations)) ||
            throw(ArgumentError("metadata records refer to an unknown data Port"))
    end
    return map(ports) do port
        declarations = port.direction == PipeWireAO.DIRECTION_INPUT ? inputs : outputs
        name = Symbol(port.name)
        haskey(declarations,name) || return port
        settings = getproperty(declarations,name)
        settings isa NamedTuple || throw(ArgumentError("metadata Port settings must be a NamedTuple"))
        all(name -> name in (:metadata, :propagation), keys(settings)) ||
            throw(ArgumentError("unknown metadata Port setting"))
        return NdArrayFilterPort(port.name, port.direction, port.format;
            schema=port.schema, parameter=port.parameter,
            settings...)
    end
end

function _captured_graph_ports(graph::PreparedCapturedGraph, base_rate)
    ports = NdArrayFilterPort[]
    sizehint!(
        ports,
        length(graph.input_formats) + length(graph.output_formats),
    )
    _append_graph_ports!(
        ports,
        graph.input_formats,
        PipeWireAO.DIRECTION_INPUT,
        base_rate;
    )
    _append_graph_ports!(
        ports,
        graph.output_formats,
        PipeWireAO.DIRECTION_OUTPUT,
        base_rate;
    )
    return ports
end

function _append_carrier_ports!(ports, formats::NamedTuple, direction, carrier_rate)
    for (name, format) in pairs(formats)
        push!(
            ports,
            NdArrayFilterPort(
                String(name),
                direction,
                NdArrayFormat(
                    _element_type(format.element_type),
                    format.shape;
                    layout=_layout(format.layout),
                    rate=carrier_rate,
                );
                schema=format.schema,
            ),
        )
    end
    return ports
end

function _latest_hold_ports(
    join::FilterGraphs.PreparedLatestHold{PrimaryIndex},
    base_rate;
) where {PrimaryIndex}
    graph = join.graph
    primary_format = getfield(graph.input_formats, PrimaryIndex)
    carrier_rate = _port_rate(primary_format, base_rate)
    ports = NdArrayFilterPort[]
    sizehint!(
        ports,
        length(graph.input_formats) +
        length(graph.parameter_formats) +
        length(graph.output_formats),
    )
    _append_carrier_ports!(
        ports,
        graph.input_formats,
        PipeWireAO.DIRECTION_INPUT,
        carrier_rate;
    )
    _append_graph_ports!(
        ports,
        graph.parameter_formats,
        PipeWireAO.DIRECTION_INPUT,
        nothing;
        parameter=true,
    )
    _append_carrier_ports!(
        ports,
        graph.output_formats,
        PipeWireAO.DIRECTION_OUTPUT,
        carrier_rate;
    )
    return ports
end

const _PARAMETER_SLOT_AVAILABLE = UInt8(0)
const _PARAMETER_SLOT_PREPARING = UInt8(1)
const _PARAMETER_SLOT_PENDING = UInt8(2)
const _PROPERTY_SLOT_AVAILABLE = UInt8(0)
const _PROPERTY_SLOT_PREPARING = UInt8(1)
const _PROPERTY_SLOT_PENDING = UInt8(2)
const _CONTROL_SLOT_AVAILABLE = UInt8(0)
const _CONTROL_SLOT_PROPERTY_PREPARING = UInt8(1)
const _CONTROL_SLOT_PROPERTY_PENDING = UInt8(2)
const _CONTROL_SLOT_PARAMETER_PREPARING = UInt8(3)
const _CONTROL_SLOT_PARAMETER_PENDING = UInt8(4)
const _CONTROL_SLOT_RECOVERY_PREPARING = UInt8(5)
const _CONTROL_SLOT_RECOVERY_PENDING = UInt8(6)
const _CONTROL_SLOT_RECOVERY_RELEASE_PENDING = UInt8(7)
const _CONTROL_SLOT_PARAMETER_ADOPTING = UInt8(8)
const _CAPTURED_CALLBACK_IDLE = UInt8(0)
const _CAPTURED_CALLBACK_ACTIVE = UInt8(1)
const _CAPTURED_REPLACEMENT_PREPARED = UInt8(0)
const _CAPTURED_REPLACEMENT_CONSUMED = UInt8(1)
struct PreparedPropertyTransaction{Plans,Touched}
    plans::Plans
    touched::Touched
    closed::Touched
end
_recovery_plan_closed(plan) = false
_recovery_plan_closed(plan::FilterGraphAlgorithms.ClosedLoopCorrectionPlan) = plan.closed
_recovery_plan_closed(plan::FilterGraphAlgorithms.LeakyIntegratorPlan) = plan.closed
PreparedPropertyTransaction(plans, touched) = PreparedPropertyTransaction(plans, touched, map(_recovery_plan_closed, plans))

struct NativeRecoveryStatus
    requested_generation::UInt64
    active_generation::UInt64
    request_token::Int64
    release_token::Int64
end
NativeRecoveryStatus() = NativeRecoveryStatus(0, 0, 0, 0)

struct GraphRecoveryStatus
    scope_id::UInt64
    controller_index::Int
    requested_generation::UInt64
    active_generation::UInt64
    status::Symbol
    closure_inhibited::Bool
    reference_supported::Bool
    native::NativeRecoveryStatus
    close_guarded::Bool
    controllers::NTuple{16,Int}
    controller_count::Int
    close_guard_mask::UInt16
end

GraphRecoveryStatus(scope_id, controller_index, requested, active, status, inhibited, supported, native, guarded) =
    GraphRecoveryStatus(scope_id, controller_index, requested, active, status, inhibited, supported, native,
        guarded, ntuple(i -> i == 1 ? controller_index : 0, 16), iszero(controller_index) ? 0 : 1,
        guarded ? UInt16(1) : UInt16(0))

GraphRecoveryStatus(scope_id, controller_index, requested, active, status, inhibited, supported, native) =
    GraphRecoveryStatus(scope_id, controller_index, requested, active, status, inhibited,
        supported, native, status === :released)

GraphRecoveryStatus() = GraphRecoveryStatus(0, 0, 0, 0, :idle, false, false, NativeRecoveryStatus())

# Mutable only so the owner can atomically publish one GC-rooted reference.
# Snapshot instances are never modified after construction.
mutable struct GraphControlSnapshot{Plans,Generations}
    plans::Plans
    requested_property_generations::Generations
    active_property_generations::Generations
    requested_parameter_sequences::Generations
    active_parameter_sequences::Generations
    recovery::GraphRecoveryStatus
end

GraphControlSnapshot(plans, requested_properties, active_properties, requested_parameters, active_parameters) =
    GraphControlSnapshot(plans, requested_properties, active_properties, requested_parameters, active_parameters, GraphRecoveryStatus())

mutable struct GraphCallbackTrace
    sequence::Vector{UInt64}
    offset::Vector{Int64}
    start_ns::Vector{UInt64}
    end_ns::Vector{UInt64}
    count::Int
    omitted::UInt64
    record_gc::Bool
    gc_pause_start::Vector{Int32}
    gc_total_time_start_ns::Vector{Int64}
    gc_pause_end::Vector{Int32}
    gc_total_time_end_ns::Vector{Int64}
    # Process-wide cumulative allocation counters. Deltas can include other
    # Julia tasks; they do not attribute bytes exclusively to this callback.
    gc_allocated_bytes_start::Vector{Int64}
    gc_allocated_bytes_end::Vector{Int64}
    gc_initial_time_ns::UInt64
    gc_initial_pause::Int32
    gc_initial_total_time_ns::Int64
end

function GraphCallbackTrace(capacity::Int)
    capacity > 0 || throw(ArgumentError("Graph callback trace capacity must be positive"))
    record_gc = get(ENV, "JULIA_RTC_TRACE_GC", "0") == "1"
    gc_capacity = record_gc ? capacity : 0
    trace = GraphCallbackTrace(
        zeros(UInt64, capacity),
        zeros(Int64, capacity),
        zeros(UInt64, capacity),
        zeros(UInt64, capacity),
        0,
        UInt64(0),
        record_gc,
        zeros(Int32, gc_capacity),
        zeros(Int64, gc_capacity),
        zeros(Int32, gc_capacity),
        zeros(Int64, gc_capacity),
        zeros(Int64, gc_capacity),
        zeros(Int64, gc_capacity),
        UInt64(0),
        Int32(0),
        Int64(0),
    )
    if record_gc
        precompile(Base.gc_num, ())
        stats = Base.gc_num()
        trace.gc_initial_time_ns = time_ns()
        trace.gc_initial_pause = stats.pause
        trace.gc_initial_total_time_ns = stats.total_time
    end
    return trace
end

# Written only by the serial native Parameter worker; read after teardown.
mutable struct GraphParameterTrace
    phase::Vector{UInt8}
    time_ns::Vector{UInt64}
    input_port::Vector{Int}
    control_state::Vector{UInt8}
    parameter_state::Vector{UInt8}
    requested::Matrix{UInt64}
    active::Matrix{UInt64}
    count::Int
    omitted::UInt64
    current_port::Int
end

function GraphParameterTrace(nodes::Int; capacity::Int=64)
    capacity > 0 || throw(ArgumentError("Parameter trace capacity must be positive"))
    return GraphParameterTrace(
        zeros(UInt8, capacity), zeros(UInt64, capacity), zeros(Int, capacity),
        zeros(UInt8, capacity), zeros(UInt8, capacity),
        zeros(UInt64, capacity, nodes), zeros(UInt64, capacity, nodes),
        0, UInt64(0), 0,
    )
end

@inline _trace_parameter_phase!(::Nothing, owner, phase::UInt8) = nothing

@inline function _trace_parameter_phase!(trace::GraphParameterTrace, owner, phase::UInt8)
    index = trace.count + 1
    if index > length(trace.phase)
        trace.omitted += UInt64(1)
        return nothing
    end
    snapshot = @atomic :acquire owner.control_snapshot
    trace.phase[index] = phase
    trace.time_ns[index] = time_ns()
    trace.input_port[index] = trace.current_port
    trace.control_state[index] = owner.control_state[]
    trace.parameter_state[index] = owner.parameter_state[]
    for node in axes(trace.requested, 2)
        trace.requested[index, node] = snapshot.requested_parameter_sequences[node]
        trace.active[index, node] = snapshot.active_parameter_sequences[node]
    end
    trace.count = index
    return nothing
end

mutable struct PreparedFeedbackBridge{Input,Output,IB,OB,MS,T}
    public_input_bindings::IB
    public_output_bindings::OB
    output_metadata_source_inputs::MS
    slot::Vector{T}
    scratch::Vector{T}
    failed::Bool
end

function _without_port(formats::NamedTuple, name::Symbol)
    names = Tuple(filter(port -> port !== name, keys(formats)))
    return NamedTuple{names}(map(port -> getproperty(formats, port), names))
end

function _public_feedback_metadata_sources(
    graph::PreparedGraph, public_inputs::NamedTuple, public_outputs::NamedTuple,
)
    full_sources = FilterGraphs.output_metadata_source_inputs(graph)
    input_names = keys(graph.input_formats)
    output_names = keys(graph.output_formats)
    public_input_names = keys(public_inputs)
    return map(keys(public_outputs)) do output_name
        output_index = findfirst(isequal(output_name), output_names)
        source_index = full_sources[something(output_index)]
        iszero(source_index) && return 0
        public_index = findfirst(isequal(input_names[source_index]), public_input_names)
        return something(public_index, 0)
    end
end

function PreparedFeedbackBridge(
    graph::PreparedGraph, input_name::Symbol, output_name::Symbol,
    boundary_layout::Union{Nothing,ArrayLayout}, local_output=nothing,
)
    haskey(graph.input_formats, input_name) || throw(
        ArgumentError("feedback input '$input_name' is not a Graph data input"),
    )
    haskey(graph.output_formats, output_name) || throw(
        ArgumentError("feedback output '$output_name' is not a Graph data output"),
    )
    input = getproperty(graph.input_formats, input_name)
    output = getproperty(graph.output_formats, output_name)
    input.element_type === Float32 && output.element_type === Float32 &&
        length(input.shape) == 1 && input.shape == output.shape &&
        typeof(input.layout) === typeof(output.layout) &&
        input.schema == output.schema && input.rate == output.rate || throw(
        ArgumentError("feedback endpoints need identical F32 vector shape, layout, schema, and rate"),
    )
    input.schema == FilterGraphAlgorithms.CONTROLLER_CONSTRAINT_FEEDBACK_V1 || throw(
        ArgumentError("feedback endpoints must carry controller constraint feedback"),
    )
    public_inputs = _port_bindings(_without_port(graph.input_formats, input_name), boundary_layout)
    public_formats = _without_port(graph.output_formats, output_name)
    public_outputs = _port_bindings(_native_outputs(local_output, public_formats), boundary_layout)
    isempty(public_inputs) && throw(ArgumentError("feedback Graph needs a public data input"))
    isempty(public_outputs) && throw(ArgumentError("feedback Graph needs a public data output"))
    metadata_sources = _public_feedback_metadata_sources(
        graph, public_inputs, public_outputs,
    )
    return PreparedFeedbackBridge{input_name,output_name,typeof(public_inputs),typeof(public_outputs),typeof(metadata_sources),Float32}(
        public_inputs, public_outputs, metadata_sources, zeros(Float32, only(input.shape)),
        zeros(Float32, only(output.shape)), false,
    )
end

_public_input_bindings(owner) = _public_input_bindings(owner, owner.feedback)
_public_input_bindings(owner, ::Nothing) = owner.input_bindings
_public_input_bindings(owner, bridge::PreparedFeedbackBridge) = bridge.public_input_bindings

_owner_feedback_bridge(graph, join, boundary_layout, ::Nothing) = nothing
_owner_feedback_bridge(graph, join, boundary_layout, feedback::Pair) =
    PreparedFeedbackBridge(graph, feedback.first, feedback.second, boundary_layout)

mutable struct LocalOutput{Name,B,S,C}
    const binding::B
    const storage::S
    const callback::C
    failed::Bool
end

_local_output(::Nothing, bindings, storage) = nothing
function _local_output(output::Pair{Symbol}, bindings, storage)
    name, callback = output
    haskey(bindings, name) || throw(ArgumentError("unknown local Graph output '$name'"))
    binding = getproperty(bindings, name)
    array = getproperty(storage, name)
    return LocalOutput{name,typeof(binding),typeof(array),typeof(callback)}(
        binding, array, callback, false,
    )
end

@inline _native_outputs(::Nothing, outputs::NamedTuple) = outputs
@inline _native_outputs(::LocalOutput{Name}, outputs::NamedTuple) where {Name} =
    Base.structdiff(outputs, NamedTuple{(Name,)})

@inline _with_local_output(::Nothing, names, native) = native
@inline function _with_local_output(local_output::LocalOutput, names, native)
    return NamedTuple{names}(map(name -> _local_or_public_output(local_output, name, native), names))
end

@inline _observe_local_output!(::Nothing, availability, metadata) = nothing
@inline _observe_local_output!(::LocalOutput{Name,B,S,Nothing}, availability, metadata) where {Name,B,S} = nothing
@inline function _observe_local_output!(local_output::LocalOutput{Name}, availability, metadata) where {Name}
    getproperty(availability, Name) || return nothing
    local_output.failed && return nothing
    try
        storage = local_output.storage
        GC.@preserve storage begin
            array = _borrow_pointer(local_output.binding, pointer(storage), Val(false))
            local_output.callback(Val(:frame), array, getproperty(metadata, Name))
        end
    catch
        local_output.failed = true
    end
    return nothing
end

@inline _reset_local_output!(::Nothing) = nothing
@inline _reset_local_output!(::LocalOutput{Name,B,S,Nothing}) where {Name,B,S} = nothing
@inline function _reset_local_output!(local_output::LocalOutput)
    local_output.failed && return nothing
    try
        local_output.callback(Val(:reset))
    catch
        local_output.failed = true
    end
    return nothing
end

mutable struct PreparedGraphOwner{G,J,MS,IB,PB,OB,WI,WO,DM,PT,RT,ST,FB,CM,IG,RF,FO,PI,PO,LO}
    graph::G
    join::J
    output_metadata_source_inputs::MS
    input_records::PI
    output_records::PO
    deferred_records::PI
    input_bindings::IB
    parameter_bindings::PB
    output_bindings::OB
    local_output::LO
    feedback::FB
    ingress::IG
    warm_inputs::WI
    warm_outputs::WO
    deferred_inputs::Union{Nothing,WI}
    deferred_metadata::DM
    defer_first_input::Bool
    first_input_held::Bool
    callback_trace::Union{Nothing,GraphCallbackTrace}
    callback_metrics::CM
    frame_observer::FO
    parameter_trace::Union{Nothing,GraphParameterTrace}
    parameter_transaction::PT
    parameter_state::Threads.Atomic{UInt8}
    property_transaction::RT
    recovery_transaction::RT
    active_recovery_selection::RF
    active_recovery_guard_selection::RF
    pending_recovery_guard_selection::RF
    pending_recovery_feedback::RF
    active_recovery_feedback::RF
    property_state::Threads.Atomic{UInt8}
    control_state::Threads.Atomic{UInt8}
    @atomic control_snapshot::ST
    pending_control_snapshot::ST
    recovery_failure_snapshot::ST
    recovery_observation_epoch::Threads.Atomic{UInt64}
    recovery_observation_generation::Threads.Atomic{UInt64}
    recovery_observation_sequence::Threads.Atomic{UInt64}
    recovery_observation_timestamp_ns::Threads.Atomic{UInt64}
    pending_recovery_reference_timeout_ns::UInt64
    recovery_reference_timeout_ns::Threads.Atomic{UInt64}
    recovery_properties_notified_sequence::UInt64
    recovery_properties_notified_timestamp_ns::UInt64
    recovery_reference_reached::Threads.Atomic{Bool}
    recovery_command_coherent::Threads.Atomic{Bool}
    recovery_branch_sequences::NTuple{16,Threads.Atomic{UInt64}}
    recovery_branch_timestamps::NTuple{16,Threads.Atomic{UInt64}}
    recovery_branch_ready::NTuple{16,Threads.Atomic{Bool}}
    recovery_branch_candidate_ready::Vector{Bool}
    parameter_adoption_count::UInt64
    metadata_fault::Union{Nothing,MetadataPublicationError}
    active::Bool
    process_count::UInt64
end

_owner_frame_observer(owner::PreparedGraphOwner) = owner.frame_observer
_frame_observer_property_info(owner::PreparedGraphOwner) =
    _frame_observer_property_info(owner.frame_observer)
_append_frame_observer_properties!(owner::PreparedGraphOwner, properties) =
    _append_frame_observer_properties!(owner.frame_observer, properties)

@inline _public_output_bindings(owner::PreparedGraphOwner) =
    isnothing(owner.feedback) ? _native_outputs(owner.local_output, owner.output_bindings) :
        owner.feedback.public_output_bindings

mutable struct PreparedCapturedGraphOwner{G,IB,OB,WI,WO,PI,PO}
    graph::G
    join::Nothing
    input_records::PI
    output_records::PO
    input_bindings::IB
    output_bindings::OB
    staged_inputs::WI
    staged_outputs::WO
    warmed::Bool
    active::Bool
    process_count::UInt64
end

_public_input_bindings(owner::PreparedCapturedGraphOwner)=owner.input_bindings
_latch_metadata_fault!(owner,error)=(owner.active=false;nothing)
function _latch_metadata_fault!(owner::PreparedGraphOwner,error)
    owner.active=false
    owner.metadata_fault=error
    return nothing
end

mutable struct PreparedCapturedGraphSlot{Owner,IF,OF}
    first::Owner
    second::Owner
    input_formats::IF
    output_formats::OF
    active_index::Threads.Atomic{UInt8}
    callback_state::Threads.Atomic{UInt8}
    control_lock::ReentrantLock
end

"""One fully warmed captured owner awaiting one control-plane swap."""
mutable struct PreparedCapturedGraphReplacement{Slot,Owner}
    target::Slot
    owner::Owner
    state::Threads.Atomic{UInt8}
end

function PreparedGraphOwner(
    graph::PreparedGraph,
    join,
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
    ; defer_first_input::Bool=false, trace_callbacks::Int=0,
    feedback::Union{Nothing,Pair{Symbol,Symbol}}=nothing,
    ingress::NamedTuple=NamedTuple(),
    frame_observer=nothing,
    local_output::Union{Nothing,Pair{Symbol}}=nothing,
)
    _require_supported_graph(graph)
    !isnothing(frame_observer) && (join !== nothing || defer_first_input) && throw(
        ArgumentError("frame observation requires an ordinary complete-frame Graph"),
    )
    input_bindings = _port_bindings(graph.input_formats, boundary_layout)
    parameter_bindings = _port_bindings(graph.parameter_formats, boundary_layout)
    output_bindings = _port_bindings(graph.output_formats, boundary_layout)
    warm_outputs = _warm_buffers(output_bindings)
    local_output === nothing || (join === nothing && !defer_first_input) || throw(
        ArgumentError("local output requires an ordinary Graph boundary"),
    )
    local_output === nothing || feedback === nothing || local_output.first !== feedback.second || throw(
        ArgumentError("local output cannot also be a private feedback output"),
    )
    prepared_local = _local_output(local_output, output_bindings, warm_outputs)
    bridge = prepared_local === nothing || feedback === nothing ?
        _owner_feedback_bridge(graph, join, boundary_layout, feedback) :
        PreparedFeedbackBridge(graph, feedback.first, feedback.second, boundary_layout, prepared_local)
    public_input_bindings = isnothing(bridge) ? input_bindings : bridge.public_input_bindings
    public_output_bindings = isnothing(bridge) ? _native_outputs(prepared_local, output_bindings) :
        bridge.public_output_bindings
    sources = NamedTuple{keys(output_bindings)}(FilterGraphs.output_metadata_source_inputs(graph))
    prepared_local === nothing || !isempty(public_output_bindings) || throw(
        ArgumentError("a Graph with a local output still needs a native output"),
    )
    if !isempty(ingress)
        (join === nothing && feedback === nothing && !defer_first_input) || throw(
            ArgumentError("input ingress requires an ordinary complete-frame Graph"),
        )
        all(name -> haskey(input_bindings, name), keys(ingress)) || throw(
            ArgumentError("input ingress names must identify external Graph data inputs"),
        )
    end
    prepared_ingress = isempty(ingress) ? ingress : NamedTuple{keys(input_bindings)}(
        map(name -> get(ingress, name, nothing), keys(input_bindings)),
    )
    plans = map(node -> node.prepared.plan, graph.nodes)
    touched = map(_ -> false, graph.nodes)
    parameter_transaction =
        FilterGraphs._parameter_transaction(graph, plans, touched)
    property_transaction = PreparedPropertyTransaction(plans, Tuple(touched))
    generations = ntuple(_ -> UInt64(0), length(graph.nodes))
    control_snapshot = GraphControlSnapshot(
        plans,
        generations,
        generations,
        generations,
        generations,
    )
    defer_first_input && (join !== nothing || length(public_input_bindings) != 1) && throw(
        ArgumentError("deferred first input requires a single-input Graph"),
    )
    trace_callbacks > 0 && length(public_input_bindings) != 1 && throw(
        ArgumentError("Graph callback tracing requires one data input"),
    )
    return PreparedGraphOwner(
        graph,
        join,
        values(_native_outputs(prepared_local, sources)),
        _metadata_snapshots(public_input_bindings),
        _metadata_snapshots(public_output_bindings),
        _metadata_snapshots(public_input_bindings),
        input_bindings,
        parameter_bindings,
        output_bindings,
        prepared_local,
        bridge,
        prepared_ingress,
        _warm_buffers(input_bindings),
        warm_outputs,
        defer_first_input ? _warm_buffers(input_bindings) : nothing,
        NamedTuple{keys(input_bindings)}(
            map(_ -> FilterGraphAlgorithms._NO_SAMPLE_METADATA, values(input_bindings)),
        ),
        defer_first_input,
        false,
        iszero(trace_callbacks) ? nothing : GraphCallbackTrace(trace_callbacks),
        _callback_metrics_from_environment(length(public_input_bindings)),
        _frame_observer(frame_observer),
        isempty(get(ENV, "JULIA_FGPW_PARAMETER_TRACE_DIR", "")) ? nothing :
            GraphParameterTrace(length(graph.nodes)),
        parameter_transaction,
        Threads.Atomic{UInt8}(_PARAMETER_SLOT_AVAILABLE),
        property_transaction,
        property_transaction,
        property_transaction.touched,
        property_transaction.touched,
        property_transaction.touched,
        property_transaction.touched,
        property_transaction.touched,
        Threads.Atomic{UInt8}(_PROPERTY_SLOT_AVAILABLE),
        Threads.Atomic{UInt8}(_CONTROL_SLOT_AVAILABLE),
        control_snapshot,
        control_snapshot,
        control_snapshot,
        Threads.Atomic{UInt64}(0),
        Threads.Atomic{UInt64}(0),
        Threads.Atomic{UInt64}(0),
        Threads.Atomic{UInt64}(0),
        UInt64(0),
        Threads.Atomic{UInt64}(0),
        UInt64(0),
        UInt64(0),
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Bool}(false),
        ntuple(_ -> Threads.Atomic{UInt64}(0), 16),
        ntuple(_ -> Threads.Atomic{UInt64}(0), 16),
        ntuple(_ -> Threads.Atomic{Bool}(false), 16),
        fill(false, 16),
        UInt64(0),
        nothing,
        false,
        UInt64(0),
    )
end

function _qualified_property(name::AbstractString)
    separator = findfirst(':', name)
    separator === nothing && throw(ArgumentError("Graph Property '$name' is not qualified"))
    separator == firstindex(name) && throw(ArgumentError("Graph Property '$name' has no Node"))
    separator == lastindex(name) && throw(
        ArgumentError("Graph Property '$name' has no algorithm-local name"),
    )
    return Symbol(SubString(name, firstindex(name), prevind(name, separator))),
    String(SubString(name, nextind(name, separator), lastindex(name)))
end

function _property_node_index(nodes::Tuple, name::Symbol)
    for index in eachindex(nodes)
        nodes[index].name == name && return index
    end
    throw(ArgumentError("Graph has no Node '$name'"))
end

function _property_info(plan, name::AbstractString)
    for info in FilterGraphAlgorithms.property_infos(plan)
        String(info.name) == name && return info
    end
    throw(ArgumentError("algorithm has no Property '$name'"))
end

_property_value(pod::PipeWireAO.Pod, ::Type{UInt32}) =
    PipeWireAO.pod_value(PipeWireAO.SPA.Id, pod).value
_property_value(pod::PipeWireAO.Pod, ::Type{T}) where {T} =
    PipeWireAO.pod_value(T, pod)

function _prepare_property_transaction(owner::PreparedGraphOwner, properties::PipeWireAO.SPA.Props)
    isempty(properties.values) && throw(ArgumentError("Property transaction must not be empty"))
    nodes = owner.graph.nodes
    assignments = [FilterGraphAlgorithms.PropertyAssignment[] for _ in nodes]
    names = Set{String}()
    for (qualified_name, pod) in properties.values
        qualified_name in names && throw(
            ArgumentError("Graph Property '$qualified_name' is assigned more than once"),
        )
        push!(names, qualified_name)
        node_name, property_name = _qualified_property(qualified_name)
        node_index = _property_node_index(nodes, node_name)
        plan = nodes[node_index].prepared.plan
        plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan || throw(
            ArgumentError("Node '$node_name' has no scalar Property contract"),
        )
        info = _property_info(plan, property_name)
        value = _property_value(pod, typeof(info.default_value))
        push!(
            assignments[node_index],
            FilterGraphAlgorithms.PropertyAssignment(info.id, value),
        )
    end

    plans = map(node -> node.prepared.plan, nodes)
    candidates = Any[plans...]
    touched = falses(length(nodes))
    for index in eachindex(nodes)
        isempty(assignments[index]) && continue
        candidates[index] = FilterGraphAlgorithms.prepare_properties(
            plans[index],
            assignments[index],
        )
        touched[index] = true
    end
    candidate_plans = Tuple(candidates)
    typeof(candidate_plans) === typeof(plans) || error(
        "prepared Property transaction changed a Plan type",
    )
    return PreparedPropertyTransaction(candidate_plans, Tuple(touched))
end

function _warm_property_preparation!(owner::PreparedGraphOwner)
    snapshot = @atomic :acquire owner.control_snapshot
    values = Pair{String,Any}[]
    for (index, node) in pairs(owner.graph.nodes)
        plan = snapshot.plans[index]
        plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan || continue
        for info in FilterGraphAlgorithms.property_infos(plan)
            info.update isa FilterGraphAlgorithms.RuntimeProperty || continue
            push!(values, "$(node.name):$(info.name)" =>
                  _spa_property_value(FilterGraphAlgorithms.property_value(plan, info.id)))
        end
    end
    isempty(values) && return nothing
    pod = PipeWireAO.Pod(PipeWireAO.props_param(PipeWireAO.SPA.Props(values)))
    properties = PipeWireAO.SPA.Props(pod)
    transaction = _prepare_property_transaction(owner, properties)
    # Compile recovery admission for the concrete dry transaction before writes.
    _validate_recovery_closure(owner, transaction, properties)
    # Dynamic field assignment is warmed without publishing this dry candidate.
    precompile(Base.setproperty!, (typeof(owner), Symbol, typeof(transaction)))
    return transaction
end

_warm_property_preparation!(owner) = nothing

_parameter_array_type(::PortBinding{T,N,ColumnMajorLayout}) where {T,N} =
    BorrowedArray{T,N,false}
_parameter_array_type(::PortBinding{T,N,RowMajorLayout}) where {T,N} =
    RowMajorBorrowedArray{T,N,false}

function _precompile_parameter_route!(
    owner::PreparedGraphOwner,
    route::FilterGraphs.ParameterInput{NodeIndex,Name},
    array_type::Type,
) where {NodeIndex,Name}
    node = getfield(owner.graph.nodes, NodeIndex)
    precompile(
        FilterGraphAlgorithms._prepare_parameter_replacement,
        (typeof(node.prepared), typeof(node.prepared.plan), Val{Name}, array_type),
    )
    precompile(
        FilterGraphs._prepare_parameter_update!,
        (Vector{Any}, BitVector, typeof(owner.graph.nodes), typeof(route), array_type),
    )
    return nothing
end

function _precompile_parameter_preparation!(owner::PreparedGraphOwner)
    for name in keys(owner.parameter_bindings)
        binding = getproperty(owner.parameter_bindings, name)
        route = getproperty(owner.graph.parameter_routes, name)
        array_type = _parameter_array_type(binding)
        _precompile_parameter_route!(owner, route, array_type)
        # Native updates arrive one port at a time. Compile the complete
        # transaction for its concrete boundary array, as well as the route.
        pair_type = Pair{Symbol,array_type}
        precompile(
            FilterGraphs._prepare_parameter_transaction,
            (typeof(owner.graph), pair_type),
        )
        precompile(
            _stage_parameter_transaction!,
            (typeof(owner), typeof(owner.parameter_trace), pair_type),
        )
    end
    snapshot = @atomic :acquire owner.control_snapshot
    precompile(_parameter_snapshots, (typeof(owner), typeof(owner.parameter_transaction)))
    precompile(
        GraphControlSnapshot,
        (
            typeof(snapshot.plans),
            typeof(snapshot.requested_property_generations),
            typeof(snapshot.active_property_generations),
            typeof(snapshot.requested_parameter_sequences),
            typeof(snapshot.active_parameter_sequences),
            GraphRecoveryStatus,
        ),
    )
    precompile(Base.setproperty!, (typeof(owner), Symbol, typeof(owner.parameter_transaction)))
    precompile(Base.setproperty!, (typeof(owner), Symbol, typeof(snapshot)))
    if owner.parameter_trace !== nothing
        _trace_parameter_phase!(owner.parameter_trace, owner, UInt8(0))
        owner.parameter_trace.count = 0
    end
    return nothing
end

function _record_property_diagnostic(event::Symbol, start_ns::UInt64)
    stamp = time_ns()
    try
        println(
            stderr,
            "JULIA_GRAPH_PROPERTY_DIAG event=", event,
            " time_ns=", stamp,
            " elapsed_ns=", stamp - start_ns,
        )
        flush(stderr)
    catch
        # A diagnostic destination must not change transaction acceptance.
    end
    return nothing
end

# Shared only by the serialized native scalar-property callbacks, never processing.
mutable struct PropertyObserver{F}
    callback::F
    owner_thread::Int
    disabled::Threads.Atomic{Bool}
end

PropertyObserver(callback) = PropertyObserver(callback, Threads.threadid(), Threads.Atomic{Bool}(false))
_property_observer(::Nothing) = nothing
_property_observer(callback) = PropertyObserver(callback)

_notify_property_observer!(::Nothing, phase, properties) = nothing
function _notify_property_observer!(observer::PropertyObserver, phase, properties)
    observer.disabled[] && return nothing
    try
        Threads.threadid() == observer.owner_thread || throw(
            InvalidStateException("property observer called outside its owner thread", :thread))
        observer.callback(phase, properties)
    catch error
        # Staging may already be committed. Never turn observer failure into a
        # rejection of that accepted write or prevent authoritative readback.
        if !Threads.atomic_xchg!(observer.disabled, true)
            try
                @error "Property observer disabled after failure" exception=(error, catch_backtrace())
            catch
                # A failing application logger cannot change write acceptance.
            end
        end
    end
    return nothing
end

_stage_property_transaction!(owner::PreparedGraphOwner, properties::PipeWireAO.SPA.Props) =
    _stage_property_transaction!(owner, properties, UInt64(0))

function _stage_property_transaction!(
    owner::PreparedGraphOwner,
    properties::PipeWireAO.SPA.Props,
    diagnostic_start_ns::UInt64,
    observer=nothing,
)
    diagnostic = !iszero(diagnostic_start_ns)
    diagnostic && _record_property_diagnostic(:stage_enter, diagnostic_start_ns)
    previous = Threads.atomic_cas!(
        owner.control_state,
        _CONTROL_SLOT_AVAILABLE,
        _CONTROL_SLOT_PROPERTY_PREPARING,
    )
    previous == _CONTROL_SLOT_AVAILABLE || throw(
        FilterGraphAlgorithms.PropertyUpdateBusy(),
    )
    owner.property_state[] = _PROPERTY_SLOT_PREPARING
    requested_properties = nothing
    observed_properties = nothing
    try
        diagnostic && _record_property_diagnostic(:prepare_enter, diagnostic_start_ns)
        transaction = _prepare_property_transaction(owner, properties)
        _validate_recovery_closure(owner, transaction, properties)
        diagnostic && _record_property_diagnostic(:prepare_completed, diagnostic_start_ns)
        current_snapshot = @atomic :acquire owner.control_snapshot
        requested_generations = map(
            transaction.touched,
            current_snapshot.requested_property_generations,
        ) do touched, generation
            touched || return generation
            generation == typemax(Int64) && throw(
                FilterGraphAlgorithms.PropertyGenerationExhausted(),
            )
            return generation + UInt64(1)
        end
        active_generations = map(
            (touched, requested, active) -> touched ? requested : active,
            transaction.touched,
            requested_generations,
            current_snapshot.active_property_generations,
        )
        requested_snapshot = GraphControlSnapshot(
            current_snapshot.plans,
            requested_generations,
            current_snapshot.active_property_generations,
            current_snapshot.requested_parameter_sequences,
            current_snapshot.active_parameter_sequences,
            current_snapshot.recovery,
        )
        pending_snapshot = GraphControlSnapshot(
            transaction.plans,
            requested_generations,
            active_generations,
            current_snapshot.requested_parameter_sequences,
            current_snapshot.active_parameter_sequences,
            _property_recovery_state(current_snapshot, transaction),
        )
        _prepare_guarded_close_failure!(owner, current_snapshot, transaction, requested_generations)
        if observer !== nothing
            # Build desired values before publishing anything; processing may
            # adopt immediately after the pending slot becomes visible.
            observed_properties = _graph_properties(owner, current_snapshot)
            requested_properties = _graph_properties(owner, GraphControlSnapshot(
                transaction.plans,
                requested_generations,
                current_snapshot.active_property_generations,
                current_snapshot.requested_parameter_sequences,
                current_snapshot.active_parameter_sequences,
                current_snapshot.recovery,
            ))
        end
        owner.property_transaction = transaction
        owner.pending_control_snapshot = pending_snapshot
        @atomic :release owner.control_snapshot = requested_snapshot
        owner.property_state[] = _PROPERTY_SLOT_PENDING
        owner.control_state[] = _CONTROL_SLOT_PROPERTY_PENDING
        diagnostic && _record_property_diagnostic(:published, diagnostic_start_ns)
    catch
        owner.property_state[] = _PROPERTY_SLOT_AVAILABLE
        owner.control_state[] = _CONTROL_SLOT_AVAILABLE
        diagnostic && _record_property_diagnostic(:failed, diagnostic_start_ns)
        rethrow()
    end
    # Reconcile an earlier adoption whose main-loop notification is still queued
    # before describing this newly accepted request.
    _notify_property_observer!(observer, Val(:observed), observed_properties)
    _notify_property_observer!(observer, Val(:requested), requested_properties)
    return nothing
end

@inline _adopt_property_plans!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

@inline function _adopt_property_plans!(
    nodes::Tuple,
    plans::Tuple,
    touched::Tuple,
)
    if first(touched)
        node = first(nodes)
        FilterGraphAlgorithms._plan_adopted!(node.prepared.workspace)
        node.prepared.plan = first(plans)
    end
    return _adopt_property_plans!(
        Base.tail(nodes),
        Base.tail(plans),
        Base.tail(touched),
    )
end

@inline function _adopt_pending_properties!(owner::PreparedGraphOwner)
    owner.control_state[] == _CONTROL_SLOT_PROPERTY_PENDING || return false
    transaction = owner.property_transaction
    snapshot = @atomic :acquire owner.control_snapshot
    if _guarded_recovery_close(snapshot, transaction) && !_current_recovery_reference_fresh(owner, snapshot.recovery, transaction)
        _invalidate_recovery_observation!(owner)
        @atomic :release owner.control_snapshot = owner.recovery_failure_snapshot
        owner.property_state[] = _PROPERTY_SLOT_AVAILABLE
        owner.control_state[] = _CONTROL_SLOT_AVAILABLE
        return true
    end
    _invalidate_recovery_observation!(owner)
    _invalidate_recovery_branch_readiness!(owner)
    FilterGraphs._discard_inflight!(owner.graph)
    _adopt_property_plans!(
        owner.graph.nodes,
        transaction.plans,
        transaction.touched,
    )
    owner.active_recovery_guard_selection = owner.pending_recovery_guard_selection
    @atomic :release owner.control_snapshot = owner.pending_control_snapshot
    owner.property_state[] = _PROPERTY_SLOT_AVAILABLE
    owner.control_state[] = _CONTROL_SLOT_AVAILABLE
    return true
end

@inline _adopt_pending_properties!(::PreparedCapturedGraphOwner) = false
@inline _adopt_pending_properties!(::PreparedCapturedGraphSlot) = false

@inline _parameter_adoption_count(owner::PreparedGraphOwner) =
    owner.parameter_adoption_count
@inline _parameter_adoption_count(::PreparedCapturedGraphOwner) = UInt64(0)
@inline _parameter_adoption_count(::PreparedCapturedGraphSlot) = UInt64(0)

function PreparedCapturedGraphOwner(graph::PreparedCapturedGraph)
    keys(graph.host_outputs) == keys(graph.output_formats) || throw(
        FilterGraphs.GraphConfigurationError(
            "captured PipeWire Nodes require host readback in public output Port order",
        ),
    )
    _require_supported_graph(graph)
    input_bindings = _port_bindings(graph.input_formats)
    output_bindings = _port_bindings(graph.output_formats)
    staged_inputs = _warm_buffers(input_bindings)
    try
        FilterGraphs._with_captured_context(graph.context) do
            try
                FilterGraphs._copy_from_captured_outputs!(
                    graph.context,
                    values(staged_inputs),
                    values(graph.device_inputs),
                )
                FilterGraphs._synchronize_captured_graph!(graph.context)
            catch
                FilterGraphs._drain_captured_storage!(
                    graph.context,
                    graph,
                    staged_inputs,
                )
                rethrow()
            end
        end
    catch error
        throw(FilterGraphs._fault_captured_graph!(graph, :warmup_input, error))
    end
    return PreparedCapturedGraphOwner(
        graph,
        nothing,
        _metadata_snapshots(input_bindings),
        _metadata_snapshots(output_bindings),
        input_bindings,
        output_bindings,
        staged_inputs,
        _warm_buffers(output_bindings),
        false,
        false,
        UInt64(0),
    )
end

function PreparedCapturedGraphSlot(owner::PreparedCapturedGraphOwner)
    graph = owner.graph
    return PreparedCapturedGraphSlot(
        owner,
        owner,
        graph.input_formats,
        graph.output_formats,
        Threads.Atomic{UInt8}(UInt8(1)),
        Threads.Atomic{UInt8}(_CAPTURED_CALLBACK_IDLE),
        ReentrantLock(),
    )
end

@inline function _active_captured_owner(slot::PreparedCapturedGraphSlot)
    return slot.active_index[] == UInt8(1) ? slot.first : slot.second
end

PreparedGraphOwner(graph::PreparedGraph) = PreparedGraphOwner(graph, nothing, nothing)

function PreparedGraphOwner(join::FilterGraphs.PreparedLatestHold)
    return PreparedGraphOwner(join.graph, join)
end

function PreparedGraphOwner(
    synchronized::FilterGraphs.PreparedSynchronizedInputs,
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    return PreparedGraphOwner(synchronized.graph, synchronized, boundary_layout)
end

# Snapshot construction depends on the prepared graph and transaction. Keep its
# tuple work out of the specializations for individual parameter buffer types.
@noinline function _parameter_snapshots(owner::PreparedGraphOwner, transaction)
    current_snapshot = @atomic :acquire owner.control_snapshot
    requested_sequences = map(
        transaction.touched,
        current_snapshot.requested_parameter_sequences,
    ) do touched, sequence
        touched || return sequence
        sequence == typemax(Int64) && throw(
            OverflowError("parameter sequence is exhausted"),
        )
        return sequence + UInt64(1)
    end
    active_sequences = map(
        (touched, requested, active) -> touched ? requested : active,
        transaction.touched,
        requested_sequences,
        current_snapshot.active_parameter_sequences,
    )
    requested_snapshot = GraphControlSnapshot(
        current_snapshot.plans,
        current_snapshot.requested_property_generations,
        current_snapshot.active_property_generations,
        requested_sequences,
        current_snapshot.active_parameter_sequences,
        current_snapshot.recovery,
    )
    pending_snapshot = GraphControlSnapshot(
        transaction.plans,
        current_snapshot.requested_property_generations,
        current_snapshot.active_property_generations,
        requested_sequences,
        active_sequences,
        current_snapshot.recovery,
    )
    return requested_snapshot, pending_snapshot
end

_stage_parameter_transaction!(owner::PreparedGraphOwner, updates::Pair...; supersede::Bool=false) =
    _stage_parameter_transaction!(owner, nothing, updates...; supersede)

function _stage_parameter_transaction!(
    owner::PreparedGraphOwner,
    trace::Union{Nothing,GraphParameterTrace},
    updates::Pair...; supersede::Bool=false,
)
    return _stage_parameter_updates!(owner, trace, updates, supersede)
end

# The owner state machine and publication are identical for every wire array
# type. Keep the typed parameter preparation calls behind this cold boundary.
@noinline Base.@nospecializeinfer function _stage_parameter_updates!(
    owner::PreparedGraphOwner,
    trace::Union{Nothing,GraphParameterTrace},
    @nospecialize(updates::Tuple{Vararg{Pair}}),
    supersede::Bool,
)
    owner.join isa FilterGraphs.PreparedComposition && FilterGraphs._require_composition_updates(owner.join)
    previous = Threads.atomic_cas!(
        owner.control_state,
        _CONTROL_SLOT_AVAILABLE,
        _CONTROL_SLOT_PARAMETER_PREPARING,
    )
    superseding = false
    if previous != _CONTROL_SLOT_AVAILABLE
        superseding = supersede && previous == _CONTROL_SLOT_PARAMETER_PENDING &&
            Threads.atomic_cas!(owner.control_state, _CONTROL_SLOT_PARAMETER_PENDING,
                _CONTROL_SLOT_PARAMETER_PREPARING) == _CONTROL_SLOT_PARAMETER_PENDING
        if !superseding
            _trace_parameter_phase!(trace, owner, UInt8(6)) # busy_return
            return false
        end
    end
    # The control slot now excludes callback adoption and all other cold writers.
    # A failed supersession must retain the previously accepted transaction.
    prior_transaction = owner.parameter_transaction
    prior_pending_snapshot = owner.pending_control_snapshot
    prior_snapshot = @atomic :acquire owner.control_snapshot
    owner.parameter_state[] = _PARAMETER_SLOT_PREPARING
    local transaction, requested_snapshot, pending_snapshot
    try
        _trace_parameter_phase!(trace, owner, UInt8(3)) # stage_prepare_start
        snapshot = @atomic :acquire owner.control_snapshot
        offset_index = supersede ? _physical_offset_update_index(owner.graph, updates) : nothing
        if supersede
            offset_index === nothing && throw(ArgumentError(
                "supersession requires one physical offset parameter update"))
            (!superseding || _physical_offset_transaction(owner.graph, prior_transaction, offset_index)) ||
                throw(ArgumentError("pending transaction is not the same physical offset without calibration changes"))
        end
        if snapshot.recovery.closure_inhibited && !_recovery_offset_clear(owner.graph, updates)
            throw(InvalidStateException(
                "parameter mutation is inhibited during recovery except clearing one physical offset", :inhibited))
        end
        transaction = @noinline FilterGraphs._prepare_parameter_transaction(owner.graph, updates...)
        if supersede && !_physical_offset_transaction(owner.graph, transaction, offset_index)
            throw(ArgumentError("physical offset supersession must retain calibration and sign"))
        end
        if snapshot.recovery.closure_inhibited && !_recovery_offset_clear_candidate(owner.graph, transaction)
            throw(InvalidStateException("recovery physical offset clear must retain calibration and adopt zero", :inhibited))
        end
        _trace_parameter_phase!(trace, owner, UInt8(4)) # stage_prepare_end
        requested_snapshot, pending_snapshot = _parameter_snapshots(owner, transaction)
    catch
        if superseding
            owner.parameter_transaction = prior_transaction
            owner.pending_control_snapshot = prior_pending_snapshot
            @atomic :release owner.control_snapshot = prior_snapshot
            owner.parameter_state[] = _PARAMETER_SLOT_PENDING
            owner.control_state[] = _CONTROL_SLOT_PARAMETER_PENDING
        else
            owner.parameter_state[] = _PARAMETER_SLOT_AVAILABLE
            owner.control_state[] = _CONTROL_SLOT_AVAILABLE
        end
        _trace_parameter_phase!(trace, owner, UInt8(7)) # failed
        rethrow()
    end
    # Publish only after all fallible cold preparation has succeeded. Once the
    # slot is PENDING the callback may adopt immediately; never roll it back.
    owner.parameter_transaction = transaction
    owner.pending_control_snapshot = pending_snapshot
    @atomic :release owner.control_snapshot = requested_snapshot
    owner.parameter_state[] = _PARAMETER_SLOT_PENDING
    owner.control_state[] = _CONTROL_SLOT_PARAMETER_PENDING
    _trace_parameter_phase!(trace, owner, UInt8(5)) # published
    return true
end

@inline function _adopt_pending_parameters!(owner::PreparedGraphOwner)
    # Exclusively claim this transaction before reading it. A superseding writer
    # can claim PENDING instead, in which case this frame uses the active Plan.
    Threads.atomic_cas!(owner.control_state, _CONTROL_SLOT_PARAMETER_PENDING,
        _CONTROL_SLOT_PARAMETER_ADOPTING) == _CONTROL_SLOT_PARAMETER_PENDING || return false
    transaction = owner.parameter_transaction
    _invalidate_recovery_observation!(owner)
    _invalidate_recovery_branch_readiness!(owner)
    FilterGraphs._adopt_parameter_transaction!(
        owner.graph,
        transaction,
    )
    owner.parameter_adoption_count += UInt64(1)
    @atomic :release owner.control_snapshot = owner.pending_control_snapshot
    owner.parameter_state[] = _PARAMETER_SLOT_AVAILABLE
    owner.control_state[] = _CONTROL_SLOT_AVAILABLE
    return true
end

const _SPA_TIME_INVALID = typemin(Int64)

@inline _buffer_metadata(::Nothing) = FilterGraphAlgorithms._NO_SAMPLE_METADATA

@inline _sample_reference_timestamp(::Nothing)=nothing
@inline function _sample_reference_timestamp(stamp::PipeWireAO.ReferenceTimestamp)
    return SampleReferenceTimestamp(stamp.timestamp_nanoseconds;
        clock=SampleClock(UInt32(stamp.clock)),
        duration_nanoseconds=stamp.has_duration ? stamp.duration_nanoseconds : nothing)
end
@inline _sample_exposure_start(::Nothing)=nothing
@inline function _sample_exposure_start(timing::PipeWireAO.ExposureTiming)
    timing.has_start || return nothing
    return SampleExposureStart(timing.start_nanoseconds;clock=SampleClock(UInt32(timing.clock)))
end
@inline function _buffer_metadata(header::BufferHeader, records=nothing)
    flags=header.flags
    records === nothing || PipeWireAO.metadata_records_valid(records) || throw(ArgumentError("invalid buffer metadata"))
    identity=records===nothing ? nothing : PipeWireAO.acquisition_identity(records)
    timing=records===nothing ? nothing : PipeWireAO.exposure_timing(records)
    exposure=_sample_exposure_start(timing)
    trigger=records===nothing ? nothing : PipeWireAO.trigger_classification(records)
    optical=records===nothing ? nothing : PipeWireAO.optical_classification(records)
    chopped=optical===nothing ? nothing : optical==PipeWireAO.OPTICAL_ABSENT ? FilterGraphAlgorithms.ChoppedImageAbsent :
        optical==PipeWireAO.OPTICAL_FRINGED ? FilterGraphAlgorithms.ChoppedImageFringed :
        optical==PipeWireAO.OPTICAL_UNFRINGED ? FilterGraphAlgorithms.ChoppedImageUnfringed : FilterGraphAlgorithms.ChoppedImageMixed
    reference=records===nothing ? nothing : _sample_reference_timestamp(PipeWireAO.reference_timestamp(records))
    return FilterGraphAlgorithms.AdmittedSampleMetadata(true,SampleMetadata(header.sequence,header.offset;
        timestamp_nanoseconds=header.pts==_SPA_TIME_INVALID ? nothing : header.pts,
        terminal=!iszero(flags&SPA.META_HEADER_FLAG_MARKER),discontinuity=!iszero(flags&SPA.META_HEADER_FLAG_DISCONT),
        corrupted=!iszero(flags&SPA.META_HEADER_FLAG_CORRUPTED),trigger_state=trigger===nothing ? missing : SampleTriggerState(UInt8(trigger)),
        chopped_image_kind=chopped,exposure_duration_nanoseconds=timing===nothing || !timing.has_duration ? nothing : timing.duration_nanoseconds,
        acquisition_identity=identity===nothing ? nothing : SampleAcquisitionIdentity(identity.domain.bytes,identity.generation,identity.sequence),
        exposure_start=exposure,reference_timestamp=reference))
end
@inline function _buffer_metadata(buffer::NdArrayFilterBuffer)
    header=PipeWireAO.buffer_header(buffer)
    header===nothing && return FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return _buffer_metadata(header,PipeWireAO.metadata_records(buffer))
end

@inline _input_metadata(::Tuple{}, buffers, index::Int) = ()

@inline function _input_metadata(bindings::Tuple, buffers, index::Int)
    return (
        _buffer_metadata(buffers[index]),
        _input_metadata(Base.tail(bindings), buffers, index + 1)...,
    )
end

@inline function _input_metadata(bindings::NamedTuple, buffers::NdArrayFilterBuffers{false})
    metadata = _input_metadata(values(bindings), buffers, 1)
    return NamedTuple{keys(bindings)}(metadata)
end

@inline _ingress_metadata(::Nothing, metadata, raw) = metadata

@inline function _ingress_metadata(callback, metadata, raw)
    metadata.present || return metadata
    result = callback(metadata.value, raw)
    result isa Enum && return result
    result isa SampleMetadata || throw(
        ArgumentError("input ingress must return SampleMetadata or an Enum rejection"),
    )
    return FilterGraphAlgorithms._admitted_sample_metadata(result)
end

@inline _ingress_metadata(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

@inline function _ingress_metadata(callbacks::Tuple, metadata::Tuple, arrays::Tuple)
    sample = _ingress_metadata(first(callbacks), first(metadata), first(arrays))
    sample isa Enum && return sample
    rest = _ingress_metadata(Base.tail(callbacks), Base.tail(metadata), Base.tail(arrays))
    rest isa Enum && return rest
    return (sample, rest...)
end

@inline _ingress_metadata(::NamedTuple{()}, metadata::NamedTuple, arrays::NamedTuple) = metadata

@inline function _ingress_metadata(callbacks::NamedTuple, metadata::NamedTuple, arrays::NamedTuple)
    result = _ingress_metadata(values(callbacks), values(metadata), values(arrays))
    result isa Enum && return result
    return NamedTuple{keys(metadata)}(result)
end

@inline function _header_flags(metadata::SampleMetadata)
    flags = UInt32(0)
    metadata.terminal && (flags |= SPA.META_HEADER_FLAG_MARKER)
    metadata.discontinuity && (flags |= SPA.META_HEADER_FLAG_DISCONT)
    metadata.corrupted && (flags |= SPA.META_HEADER_FLAG_CORRUPTED)
    return flags
end

@inline function _buffer_header(metadata::SampleMetadata)
    return BufferHeader(
        _header_flags(metadata),
        metadata.offset,
        metadata.has_timestamp ? metadata.timestamp_nanoseconds : _SPA_TIME_INVALID,
        Int64(0),
        metadata.sequence,
    )
end

@inline function _same_acquisition_facts(a::SampleMetadata, b::SampleMetadata)
    return sample_acquisition_identity(a) == sample_acquisition_identity(b) &&
        sample_exposure_start(a) == sample_exposure_start(b) &&
        sample_exposure_duration_nanoseconds(a) == sample_exposure_duration_nanoseconds(b) &&
        a.has_trigger_classification == b.has_trigger_classification &&
        sample_trigger_state(a) == sample_trigger_state(b) &&
        sample_chopped_image_kind(a) == sample_chopped_image_kind(b) &&
        sample_reference_timestamp(a) == sample_reference_timestamp(b)
end

@inline function _preserves_acquisition_facts(a::SampleMetadata,b::SampleMetadata)
    (!a.has_acquisition_identity || (b.has_acquisition_identity && a.acquisition_identity==b.acquisition_identity)) &&
    (!a.has_exposure_duration || (b.has_exposure_duration && a.exposure_duration_nanoseconds==b.exposure_duration_nanoseconds)) &&
    (!a.has_exposure_start || (b.has_exposure_start && a.exposure_start==b.exposure_start)) &&
    (!a.has_trigger_classification || (b.has_trigger_classification && a.trigger_state==b.trigger_state)) &&
    (!a.has_chopped_image_kind || (b.has_chopped_image_kind && a.chopped_image_kind==b.chopped_image_kind)) || return false
    return !a.has_reference_timestamp ||
        (b.has_reference_timestamp && a.reference_timestamp==b.reference_timestamp)
end
@inline function _reference_value(stamp::SampleReferenceTimestamp)
    return PipeWireAO.ReferenceTimestamp(stamp.timestamp_nanoseconds;
        clock=PipeWireAO.MetadataClock(UInt32(stamp.clock)),
        duration_nanoseconds=stamp.has_duration ? stamp.duration_nanoseconds : nothing)
end
@inline function _set_output_records!(buffer,metadata::SampleMetadata,snapshot::Union{Nothing,RetainedMetadata})
    if snapshot !== nothing && snapshot.present
        _publish_metadata!(buffer,snapshot)
        snapshot.augment || return nothing
    else
        PipeWireAO.initialize_metadata!(buffer)
    end
    kind=sample_chopped_image_kind(metadata)
    if kind !== nothing && PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_OPTICAL_CHOPPING)!==nothing
        optical=kind==FilterGraphAlgorithms.ChoppedImageFringed ? PipeWireAO.OPTICAL_FRINGED :
            kind==FilterGraphAlgorithms.ChoppedImageUnfringed ? PipeWireAO.OPTICAL_UNFRINGED :
            kind==FilterGraphAlgorithms.ChoppedImageAbsent ? PipeWireAO.OPTICAL_ABSENT : PipeWireAO.OPTICAL_MIXED
        PipeWireAO.set_optical_classification!(buffer,optical)
    end
    metadata.has_trigger_classification && PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_ELECTRICAL_TRIGGER)!==nothing && PipeWireAO.set_trigger_classification!(buffer,PipeWireAO.AcquisitionTriggerState(UInt8(sample_trigger_state(metadata))))
    identity=sample_acquisition_identity(metadata)
    identity===nothing || PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_ACQUISITION_IDENTITY)===nothing || PipeWireAO.set_acquisition_identity!(buffer,PipeWireAO.AcquisitionIdentity(
        PipeWireAO.AcquisitionDomain(identity.domain),identity.generation,identity.sequence))
    duration=sample_exposure_duration_nanoseconds(metadata)
    exposure=sample_exposure_start(metadata)
    if exposure!==nothing && PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_EXPOSURE_TIMING)!==nothing
        PipeWireAO.set_exposure_timing!(buffer,PipeWireAO.ExposureTiming(exposure.nanoseconds;
            clock=PipeWireAO.MetadataClock(UInt32(exposure.clock)),duration_nanoseconds=duration))
    elseif duration!==nothing && PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_EXPOSURE_TIMING)!==nothing
        PipeWireAO.set_exposure_timing!(buffer,PipeWireAO.ExposureTiming(;duration_nanoseconds=duration))
    end
    stamp=sample_reference_timestamp(metadata)
    if stamp!==nothing && PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_REFERENCE_TIMESTAMP)!==nothing
        PipeWireAO.set_reference_timestamp!(buffer,_reference_value(stamp))
    end
    return nothing
end

@inline function _set_output_metadata!(buffer,
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata,
    snapshot::Union{Nothing,RetainedMetadata}=nothing;production::Bool=true)
    try
        if metadata.present
            _set_output_records!(buffer, metadata.value, snapshot)
            production && snapshot!==nothing && _produce_output_time!(buffer,snapshot)
            PipeWireAO.set_buffer_header!(buffer, _buffer_header(metadata.value))
        elseif snapshot!==nothing && snapshot.produce_header
            PipeWireAO.initialize_metadata!(buffer)
            production && _produce_output_time!(buffer,snapshot)
            sequence=snapshot.publication_sequence+UInt64(1)
            PipeWireAO.set_buffer_header!(buffer,BufferHeader(UInt32(0),UInt32(0),_SPA_TIME_INVALID,Int64(0),sequence))
            production && (snapshot.publication_sequence=sequence)
        else
            PipeWireAO.clear_buffer_metadata!(buffer)
        end
        production && snapshot!==nothing && !PipeWireAO.metadata_records_valid(PipeWireAO.metadata_records(buffer),snapshot.requirements) &&
            throw(InvalidStateException("output lacks required metadata facts",:metadata))
    catch error
        production && throw(error isa MetadataPublicationError ? error : MetadataPublicationError(error))
        rethrow()
    end
    return nothing
end

@inline _finalize_outputs!(::Tuple{},::Tuple{},buffers,index::Int,snapshots::Tuple=()) = nothing
@inline function _finalize_outputs!(availability::Tuple,metadata::Tuple,buffers,index::Int,snapshots::Tuple=())
    available = first(availability)
    snapshot = isempty(snapshots) ? nothing : first(snapshots)
    available && _set_output_metadata!(buffers[index],first(metadata),snapshot)
    PipeWireAO.set_output_available!(buffers[index],available)
    return _finalize_outputs!(Base.tail(availability),Base.tail(metadata),buffers,index+1,
        isempty(snapshots) ? () : Base.tail(snapshots))
end

@inline _set_outputs_unavailable!(::Tuple{}, buffers, index::Int) = nothing

@inline function _set_outputs_unavailable!(bindings::Tuple, buffers, index::Int)
    index>length(buffers) && return nothing
    PipeWireAO.set_output_available!(buffers[index], false)
    return _set_outputs_unavailable!(Base.tail(bindings), buffers, index + 1)
end

@inline _all_inputs_available(::Tuple{}, buffers, index::Int) = true

@inline function _all_inputs_available(bindings::Tuple, buffers, index::Int)
    PipeWireAO.input_available(buffers[index]) || return false
    return _all_inputs_available(Base.tail(bindings), buffers, index + 1)
end

@inline _reserve_captured_output_metadata!(::Tuple{},buffers,index::Int) = nothing
@inline function _reserve_captured_output_metadata!(snapshots::Tuple,buffers,index::Int)
    snapshot = first(snapshots)
    _set_output_metadata!(buffers[index],snapshot.metadata,snapshot;production=false)
    return _reserve_captured_output_metadata!(Base.tail(snapshots),buffers,index+1)
end

@inline _publish_captured_outputs!(::Tuple{}, buffers, index::Int) = nothing

@inline function _publish_captured_outputs!(availability::Tuple, buffers, index::Int)
    PipeWireAO.set_output_available!(buffers[index], first(availability))
    return _publish_captured_outputs!(
        Base.tail(availability),
        buffers,
        index + 1,
    )
end

@inline _copy_captured_staging!(::Tuple{}, ::Tuple{}) = nothing

@inline function _copy_captured_staging!(
    destination::Tuple{Array{T,2},Vararg{Any}},
    source::Tuple{RowMajorBorrowedArray{T,2,W},Vararg{Any}},
) where {T,W}
    output, input = first(destination), first(source)
    size(output) == size(input) || throw(DimensionMismatch("captured staging shape changed"))
    rows, columns = size(input)
    # The prepared staging array and the native input loan occupy distinct
    # storage. Tile the transpose to reuse cache lines from both layouts.
    @inbounds for first_row in 1:16:rows, first_column in 1:16:columns
        last_row = min(first_row + 15, rows)
        last_column = min(first_column + 15, columns)
        for column in first_column:last_column
            @simd for row in first_row:last_row
                output[row, column] = unsafe_load(input.data, (row - 1) * columns + column)
            end
        end
    end
    return _copy_captured_staging!(Base.tail(destination), Base.tail(source))
end

@inline function _copy_captured_staging!(destination::Tuple, source::Tuple)
    copyto!(first(destination), first(source))
    return _copy_captured_staging!(
        Base.tail(destination),
        Base.tail(source),
    )
end

function _finish_graph_candidate!(owner::PreparedGraphOwner,buffers,result::NamedTuple)
    full_metadata = FilterGraphs._output_metadata_admitted(owner.graph)
    metadata = values(_native_outputs(owner.local_output, full_metadata))
    availability = values(_native_outputs(owner.local_output, result.availability))
    _cache_output_metadata!(values(owner.output_records),values(owner.input_records),
        owner.output_metadata_source_inputs,metadata)
    _finalize_outputs!(availability,metadata,buffers,1,values(owner.output_records))
    if result.terminal && all(availability)
        _observe_recovery_reference!(owner)
        _observe_frame!(owner.frame_observer, owner, buffers)
    else
        _invalidate_recovery_observation!(owner)
        _frame_observer_event!(owner.frame_observer, Val(:invalidated))
    end
    result.terminal && _observe_local_output!(owner.local_output, result.availability, full_metadata)
    return nothing
end

function _finish_ordinary_graph_candidate!(owner::PreparedGraphOwner,
    inputs::NdArrayFilterBuffers{false},outputs::NdArrayFilterBuffers{true},result::NamedTuple)
    return _finish_graph_candidate!(owner,outputs,result)
end

function _finish_graph_candidate!(owner::PreparedGraphOwner, buffers, failure::Enum)
    _invalidate_recovery_observation!(owner)
    _frame_observer_event!(owner.frame_observer, Val(:invalidated))
    _set_outputs_unavailable!(values(_public_output_bindings(owner)), buffers, 1)
    return nothing
end

function _finish_ordinary_graph_candidate!(
    owner::PreparedGraphOwner,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
    failure::Enum,
)
    return _finish_graph_candidate!(owner, outputs, failure)
end

@inline function _finish_deferred_first!(owner, outputs, result::NamedTuple)
    result.terminal && throw(
        InvalidStateException("first deferred block completed the Graph", :input),
    )
    return true
end

@inline function _finish_deferred_first!(owner, outputs, result::Enum)
    _finish_graph_candidate!(owner, outputs, result)
    return false
end

function _finish_captured_frame!(
    owner::PreparedCapturedGraphOwner,
    output_arrays,
    buffers,
    availability::NamedTuple,
)
    try
        _copy_captured_staging!(
            values(output_arrays),
            values(owner.staged_outputs),
        )
        _finalize_output_times!(values(availability),values(owner.output_records),buffers,1)
        _publish_captured_outputs!(values(availability), buffers, 1)
    catch error
        throw(
            FilterGraphs._fault_captured_graph!(
                owner.graph,
                :publication,
                error,
            ),
        )
    end
    return nothing
end

function _finish_captured_frame!(
    owner::PreparedCapturedGraphOwner,
    output_arrays,
    buffers,
    failure::Enum,
)
    return nothing
end

function _reset_owner!(owner::PreparedGraphOwner, ::Nothing)
    owner.first_input_held = false
    # RESET invalidates command-reference evidence before settling any staged
    # release or first close, whose adoption must then retain protection.
    _invalidate_recovery_observation!(owner)
    _adopt_pending_recovery!(owner)
    _adopt_pending_properties!(owner)
    _adopt_pending_parameters!(owner)
    _frame_observer_event!(owner.frame_observer, Val(:reset))
    _reset_local_output!(owner.local_output)
    result = reset!(owner.graph)
    _reset_feedback!(owner.feedback)
    return result
end

_reset_feedback!(::Nothing) = nothing
function _reset_feedback!(bridge::PreparedFeedbackBridge)
    fill!(bridge.slot, 0.0f0)
    fill!(bridge.scratch, 0.0f0)
    bridge.failed = false
    return nothing
end
function _reset_owner!(
    owner::PreparedGraphOwner,
    join::FilterGraphs.PreparedLatestHold,
)
    _adopt_pending_recovery!(owner)
    _adopt_pending_properties!(owner)
    _adopt_pending_parameters!(owner)
    reset!(join)
    FilterGraphs._clear_latest_hold_counters!(join)
    return join
end

function _reset_owner!(
    owner::PreparedGraphOwner,
    synchronized::FilterGraphs.PreparedSynchronizedInputs,
)
    _adopt_pending_recovery!(owner)
    _adopt_pending_properties!(owner)
    _adopt_pending_parameters!(owner)
    reset!(synchronized)
    FilterGraphs._clear_synchronized_input_counters!(synchronized)
    return synchronized
end

function _reset_owner!(owner::PreparedGraphOwner)
    _clear_metadata_snapshots!(owner.deferred_records)
    _clear_metadata_snapshots!(owner.input_records)
    _clear_metadata_snapshots!(owner.output_records)
    return _reset_owner!(owner,owner.join)
end
function _reset_owner!(owner::PreparedCapturedGraphOwner)
    _clear_metadata_snapshots!(owner.deferred_records)
    _clear_metadata_snapshots!(owner.input_records)
    _clear_metadata_snapshots!(owner.output_records)
    return reset!(owner.graph)
end

function _reset_owner!(slot::PreparedCapturedGraphSlot)
    lock(slot.control_lock)
    try
        slot.callback_state[] == _CAPTURED_CALLBACK_IDLE || throw(
            InvalidStateException("captured PipeWire callback is active during reset", :busy),
        )
        _reset_owner!(slot.first)
        slot.second === slot.first || _reset_owner!(slot.second)
    finally
        unlock(slot.control_lock)
    end
    return slot
end

function _prepare_owner(owner::PreparedGraphOwner)
    owner.metadata_fault===nothing || throw(owner.metadata_fault)
    _adopt_pending_recovery!(owner)
    _adopt_pending_properties!(owner)
    _adopt_pending_parameters!(owner)
    owner.active = true
    return nothing
end

function _prepare_owner(owner::PreparedCapturedGraphOwner)
    if !owner.warmed
        GC.@preserve owner begin
            try
                # Zero-filled retained inputs are sufficient to compile and
                # warm the fixed callback path, but some Algorithms correctly
                # reject them as a declared numerical result. Completion has
                # already validated that disposition; reset below removes all
                # warmup effects before the owner becomes active.
                FilterGraphs._process_captured_admitted!(
                    owner.staged_outputs,
                    owner.graph,
                    owner.staged_inputs,
                    owner.graph.default_input_metadata,
                    owner.graph.result_types,
                )
            finally
                reset!(owner.graph)
            end
        end
        owner.warmed = true
    end
    owner.active = true
    return nothing
end

function _prepare_owner(slot::PreparedCapturedGraphSlot)
    lock(slot.control_lock)
    try
        return _prepare_owner(_active_captured_owner(slot))
    finally
        unlock(slot.control_lock)
    end
end

function _process_graph_buffers!(
    owner::PreparedGraphOwner,
    ::Nothing,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    return _process_feedback_buffers!(owner, owner.feedback, inputs, outputs)
end

function _process_feedback_buffers!(
    owner::PreparedGraphOwner,
    ::Nothing,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    input_arrays = _borrow_buffers(owner.input_bindings, inputs)
    native = _borrow_buffers(_public_output_bindings(owner), outputs)
    output_arrays = _with_local_output(owner.local_output, keys(owner.output_bindings), native)
    input_metadata = _input_metadata(owner.input_bindings, inputs)
    _retain_input_metadata!(values(owner.input_records),inputs,values(input_metadata),1)
    _preflight_output_records!(values(owner.output_records),values(owner.input_records),
        owner.output_metadata_source_inputs,outputs,1)
    input_metadata = _ingress_metadata(owner.ingress, input_metadata, input_arrays)
    input_metadata isa Enum && return _finish_graph_candidate!(owner, outputs, input_metadata)
    _check_input_facts!(values(owner.input_records),values(input_metadata))
    _adopt_pending_parameters!(owner)
    if owner.defer_first_input
        held_inputs = something(owner.deferred_inputs)
        sample = only(values(input_metadata)).value
        if !sample.terminal
            sample.offset == 0 || throw(
                InvalidStateException("deferred input must start at row zero", :input),
            )
            owner.first_input_held && throw(
                InvalidStateException("deferred input has no terminal block", :input),
            )
            copyto!(only(values(held_inputs)), only(values(input_arrays)))
            _retain_metadata!(only(values(owner.deferred_records)),inputs[1],only(values(input_metadata)))
            owner.deferred_metadata = input_metadata
            owner.first_input_held = true
            _set_outputs_unavailable!(values(owner.output_bindings), outputs, 1)
            return nothing
        end
        owner.first_input_held || throw(
            InvalidStateException("terminal input has no held first block", :input),
        )
        first_sample = only(values(owner.deferred_metadata)).value
        FilterGraphAlgorithms._same_row_block_metadata(first_sample,sample) &&
            _same_frame_records(only(values(owner.deferred_records)),inputs[1]) ||
            throw(InvalidStateException("row blocks disagree on scientific metadata",:metadata))
        owner.first_input_held = false
        first_result = FilterGraphs._process_candidate_admitted!(
            output_arrays,
            owner.graph,
            held_inputs,
            owner.deferred_metadata,
        )
        _finish_deferred_first!(owner, outputs, first_result) || return nothing
    end
    result = GC.@preserve owner FilterGraphs._process_candidate_admitted!(
        output_arrays, owner.graph, input_arrays, input_metadata,
    )
    return _finish_ordinary_graph_candidate!(owner, inputs, outputs, result)
end

@inline function _feedback_inputs(
    owner::PreparedGraphOwner,
    bridge::PreparedFeedbackBridge{Input},
    public::NamedTuple,
) where {Input}
    names = keys(owner.input_bindings)
    return NamedTuple{names}(map(
        name -> name === Input ? bridge.slot : getproperty(public, name), names,
    ))
end

@inline function _feedback_outputs(
    owner::PreparedGraphOwner,
    bridge::PreparedFeedbackBridge{Input,Output},
    public::NamedTuple,
) where {Input,Output}
    names = keys(owner.output_bindings)
    return NamedTuple{names}(map(
        name -> name === Output ? bridge.scratch :
            _local_or_public_output(owner.local_output, name, public), names,
    ))
end

@inline _local_or_public_output(::Nothing, name, public) = getproperty(public, name)
@inline function _local_or_public_output(local_output::LocalOutput{Name}, name, public) where {Name}
    name === Name || return getproperty(public, name)
    return _borrow_pointer(local_output.binding, pointer(local_output.storage), Val(true))
end

@inline function _feedback_metadata(
    owner::PreparedGraphOwner,
    bridge::PreparedFeedbackBridge{Input},
    public::NamedTuple,
) where {Input}
    names = keys(owner.input_bindings)
    return NamedTuple{names}(map(
        name -> name === Input ? FilterGraphAlgorithms._NO_SAMPLE_METADATA :
            getproperty(public, name), names,
    ))
end

@inline function _public_output_values(
    full::NamedTuple, bridge::PreparedFeedbackBridge{Input,Output},
) where {Input,Output}
    return map(name -> getproperty(full, name), keys(bridge.public_output_bindings))
end

function _process_feedback_buffers!(
    owner::PreparedGraphOwner,
    bridge::PreparedFeedbackBridge,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    try
        bridge.failed && throw(InvalidStateException(
            "feedback Graph failed and requires reset", :feedback,
        ))
        public_inputs = _borrow_buffers(bridge.public_input_bindings, inputs)
        public_outputs = _borrow_buffers(bridge.public_output_bindings, outputs)
        input_arrays = _feedback_inputs(owner, bridge, public_inputs)
        output_arrays = _feedback_outputs(owner, bridge, public_outputs)
        public_metadata = _input_metadata(bridge.public_input_bindings,inputs)
        _retain_input_metadata!(values(owner.input_records),inputs,values(public_metadata),1)
        metadata = _feedback_metadata(owner,bridge,public_metadata)
        _adopt_pending_parameters!(owner)
        result = GC.@preserve owner FilterGraphs._process_candidate_admitted!(
            output_arrays, owner.graph, input_arrays, metadata,
        )
        return _finish_feedback_candidate!(owner, bridge, inputs, outputs, result)
    catch
        bridge.failed = true
        _set_outputs_unavailable!(values(bridge.public_output_bindings), outputs, 1)
        rethrow()
    end
end

function _finish_feedback_candidate!(owner, bridge, inputs, buffers, result::NamedTuple)
    if result.terminal
        all(values(_native_outputs(owner.local_output, result.availability))) || throw(InvalidStateException(
            "terminal feedback Graph omitted an output", :feedback,
        ))
    end
    public_availability = _public_output_values(result.availability, bridge)
    full_metadata = FilterGraphs._output_metadata_admitted(owner.graph)
    metadata = _public_output_values(full_metadata,bridge)
    _cache_output_metadata!(values(owner.output_records),values(owner.input_records),
        bridge.output_metadata_source_inputs,metadata)
    _finalize_outputs!(public_availability,metadata,buffers,1,values(owner.output_records))
    if result.terminal
        copyto!(bridge.slot, bridge.scratch)
        _observe_recovery_reference!(owner)
        _observe_frame!(owner.frame_observer, owner, buffers)
        _observe_local_output!(owner.local_output, result.availability, full_metadata)
    else
        _invalidate_recovery_observation!(owner)
        _frame_observer_event!(owner.frame_observer, Val(:invalidated))
    end
    return nothing
end

function _finish_feedback_candidate!(owner, bridge, inputs, buffers, failure::Enum)
    throw(InvalidStateException("feedback Graph rejected the sample: $failure", :feedback))
end

function _process_graph_buffers!(
    owner::PreparedCapturedGraphOwner,
    ::Nothing,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    length(outputs) == length(owner.output_bindings) || throw(
        DimensionMismatch(
            "PipeWire callback buffer count does not match the Graph outputs",
        ),
    )
    _set_outputs_unavailable!(values(owner.output_bindings), outputs, 1)
    output_arrays = _borrow_buffers(owner.output_bindings, outputs)

    length(inputs) == length(owner.input_bindings) || throw(
        DimensionMismatch(
            "PipeWire callback buffer count does not match the Graph inputs",
        ),
    )
    _all_inputs_available(values(owner.input_bindings), inputs, 1) ||
        return nothing
    input_arrays = _borrow_buffers(owner.input_bindings, inputs)
    _copy_captured_staging!(
        values(owner.staged_inputs),
        values(input_arrays),
    )
    input_metadata = _input_metadata(owner.input_bindings, inputs)
    _retain_input_metadata!(values(owner.input_records),inputs,values(input_metadata),1)
    output_metadata = FilterGraphs._captured_output_metadata_admitted(
        owner.graph,
        input_metadata,
    )
    _cache_output_metadata!(values(owner.output_records),values(owner.input_records),
        owner.graph.output_metadata_sources,values(output_metadata))
    _reserve_captured_output_metadata!(values(owner.output_records), outputs, 1)
    result = FilterGraphs._process_captured_admitted!(
        owner.staged_outputs,
        owner.graph,
        owner.staged_inputs,
        input_metadata,
        owner.graph.result_types,
    )
    return _finish_captured_frame!(owner, output_arrays, outputs, result)
end


function _process_graph_buffers!(
    owner::PreparedGraphOwner,
    join::FilterGraphs.PreparedLatestHold{PrimaryIndex,HeldIndex},
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
) where {PrimaryIndex,HeldIndex}
    length(inputs) == length(owner.input_bindings) || throw(
        DimensionMismatch("PipeWire callback buffer count does not match the Graph inputs"),
    )
    output_arrays = _borrow_buffers(owner.output_bindings, outputs)
    held_buffer = inputs[HeldIndex]
    primary_buffer = inputs[PrimaryIndex]
    held_available = PipeWireAO.input_available(held_buffer)
    primary_available = PipeWireAO.input_available(primary_buffer)

    if held_available
        held = _borrow_buffer(getfield(owner.input_bindings, HeldIndex), held_buffer)
        held_metadata = _buffer_metadata(held_buffer)
        if FilterGraphs._accept_held_admitted!(join,held,held_metadata)
            _retain_metadata!(getfield(values(owner.input_records),HeldIndex),held_buffer,held_metadata)
        end
    end
    if primary_available
        primary = _borrow_buffer(
            getfield(owner.input_bindings, PrimaryIndex),
            primary_buffer,
        )
        primary_metadata = _buffer_metadata(primary_buffer)
        _retain_metadata!(getfield(values(owner.input_records),PrimaryIndex),primary_buffer,primary_metadata)
        _adopt_pending_parameters!(owner)
        result = FilterGraphs._process_primary_candidate_admitted!(
            output_arrays,
            join,
            primary,
            primary_metadata,
        )
        return _finish_graph_candidate!(owner, outputs, result)
    end
    held_available || throw(
        InvalidStateException("the PipeWire callback has no input arrival", :unavailable),
    )
    _set_outputs_unavailable!(values(owner.output_bindings), outputs, 1)
    return nothing
end

@inline _any_input_available(::Tuple{}, buffers, index::Int) = false

@inline function _any_input_available(bindings::Tuple, buffers, index::Int)
    PipeWireAO.input_available(buffers[index]) && return true
    return _any_input_available(Base.tail(bindings), buffers, index + 1)
end

@inline function _process_synchronized_buffers!(
    owner,
    synchronized,
    output_arrays,
    ::Tuple{},
    inputs,
    ::Val,
)
    return FilterGraphs._unavailable_result(synchronized.graph)
end

@inline function _process_synchronized_buffers!(
    owner,
    synchronized,
    output_arrays,
    bindings::Tuple,
    inputs,
    ::Val{Index},
) where {Index}
    buffer = inputs[Index]
    if PipeWireAO.input_available(buffer)
        metadata = _buffer_metadata(buffer)
        metadata.present || throw(
            ArgumentError("a synchronized Graph input requires Header metadata"),
        )
        input = _borrow_buffer(first(bindings), buffer)
        admitted = FilterGraphs._admit_synchronized_input!(synchronized,input,metadata.value,Val(Index))
        if admitted
            _retain_metadata!(getfield(values(owner.input_records),Index),buffer,metadata)
        end
        result = admitted && FilterGraphs._all_received(synchronized.state.received) ?
            FilterGraphs._finish_synchronized_window_candidate!(output_arrays,synchronized) :
            FilterGraphs._unavailable_result(synchronized.graph)
        if result isa Enum || result.terminal
            result isa NamedTuple && _cache_output_metadata!(values(owner.output_records),
                values(owner.input_records),owner.output_metadata_source_inputs,
                values(FilterGraphs._output_metadata_admitted(owner.graph)))
            _retain_synchronized_tail!(
                owner,
                synchronized,
                output_arrays,
                Base.tail(bindings),
                inputs,
                Val(Index + 1),
            )
            return result
        end
    end
    return _process_synchronized_buffers!(
        owner,
        synchronized,
        output_arrays,
        Base.tail(bindings),
        inputs,
        Val(Index + 1),
    )
end

@inline _retain_synchronized_tail!(owner, synchronized, output_arrays, ::Tuple{}, inputs, ::Val) =
    nothing

@inline function _retain_synchronized_tail!(
    owner,
    synchronized,
    output_arrays,
    bindings::Tuple,
    inputs,
    ::Val{Index},
) where {Index}
    buffer = inputs[Index]
    if PipeWireAO.input_available(buffer)
        metadata = _buffer_metadata(buffer)
        metadata.present || throw(
            ArgumentError("a synchronized Graph input requires Header metadata"),
        )
        input = _borrow_buffer(first(bindings), buffer)
        admitted = FilterGraphs._admit_synchronized_input!(synchronized,input,metadata.value,Val(Index))
        admitted && _retain_metadata!(getfield(values(owner.input_records),Index),buffer,metadata)
        result = admitted && FilterGraphs._all_received(synchronized.state.received) ?
            FilterGraphs._finish_synchronized_window_candidate!(output_arrays,synchronized) :
            FilterGraphs._unavailable_result(synchronized.graph)
        (result isa Enum || result.terminal) && error(
            "one synchronized PipeWire callback produced more than one publication unit",
        )
    end
    return _retain_synchronized_tail!(
        owner,
        synchronized,
        output_arrays,
        Base.tail(bindings),
        inputs,
        Val(Index + 1),
    )
end

function _finish_synchronized_candidate!(
    owner::PreparedGraphOwner,
    synchronized::FilterGraphs.PreparedSynchronizedInputs,
    buffers,
    result::NamedTuple,
)
    try
        _finalize_outputs!(
            values(result.availability),
            map(snapshot -> snapshot.metadata,values(owner.output_records)),
            buffers,
            1,
            values(owner.output_records),
        )
    catch
        result.terminal && (synchronized.counters.aborted_windows += UInt64(1))
        rethrow()
    end
    result.terminal && (synchronized.counters.completed_windows += UInt64(1))
    return nothing
end

function _finish_synchronized_candidate!(
    owner::PreparedGraphOwner,
    synchronized::FilterGraphs.PreparedSynchronizedInputs,
    buffers,
    failure::Enum,
)
    _set_outputs_unavailable!(values(owner.output_bindings), buffers, 1)
    return nothing
end

function _process_graph_buffers!(
    owner::PreparedGraphOwner,
    synchronized::FilterGraphs.PreparedSynchronizedInputs,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    length(inputs) == length(owner.input_bindings) || throw(
        DimensionMismatch("PipeWire callback buffer count does not match the Graph inputs"),
    )
    _any_input_available(values(owner.input_bindings), inputs, 1) || throw(
        InvalidStateException("the PipeWire callback has no input arrival", :unavailable),
    )
    output_arrays = _borrow_buffers(owner.output_bindings, outputs)
    _adopt_pending_parameters!(owner)
    result = _process_synchronized_buffers!(
        owner,
        synchronized,
        output_arrays,
        values(owner.input_bindings),
        inputs,
        Val(1),
    )
    return _finish_synchronized_candidate!(owner, synchronized, outputs, result)
end

function _deactivate_owner(owner::PreparedGraphOwner)
    owner.active = false
    owner.first_input_held = false
    return nothing
end

function _deactivate_owner(owner::PreparedCapturedGraphOwner)
    owner.active = false
    return nothing
end

function _deactivate_owner(slot::PreparedCapturedGraphSlot)
    lock(slot.control_lock)
    try
        slot.callback_state[] == _CAPTURED_CALLBACK_IDLE || throw(
            InvalidStateException(
                "captured PipeWire callback is active during deactivation",
                :busy,
            ),
        )
        _deactivate_owner(slot.first)
        slot.second === slot.first || _deactivate_owner(slot.second)
    finally
        unlock(slot.control_lock)
    end
    return nothing
end

struct GraphPrepareCallback{Owner}
    owner::Owner
end

function (callback::GraphPrepareCallback)(::NdArrayFilter)
    return _prepare_owner(callback.owner)
end

struct GraphProcessCallback{Owner}
    owner::Owner
end

@inline _graph_callback_trace(owner::PreparedGraphOwner) = owner.callback_trace
@inline _graph_callback_trace(::PreparedCapturedGraphOwner) = nothing
@inline _graph_callback_metrics(owner::PreparedGraphOwner) = owner.callback_metrics
@inline _graph_callback_metrics(::PreparedCapturedGraphOwner) = nothing

@inline _trace_graph_callback_begin!(::Nothing, inputs) = 0

@inline function _trace_graph_callback_begin!(
    trace::GraphCallbackTrace, inputs::NdArrayFilterBuffers{false},
)
    index = trace.count + 1
    if index > length(trace.sequence)
        trace.omitted += UInt64(1)
        return 0
    end
    header = PipeWireAO.buffer_header(inputs[1])
    trace.sequence[index] = header.sequence
    trace.offset[index] = header.offset
    trace.start_ns[index] = time_ns()
    if trace.record_gc
        stats = Base.gc_num()
        trace.gc_pause_start[index] = stats.pause
        trace.gc_total_time_start_ns[index] = stats.total_time
        trace.gc_allocated_bytes_start[index] = Base.gc_total_bytes(stats)
    end
    trace.count = index
    return index
end

@inline _trace_graph_callback_end!(::Nothing, ::Int) = nothing

@inline function _trace_graph_callback_end!(trace::GraphCallbackTrace, index::Int)
    if !iszero(index)
        if trace.record_gc
            stats = Base.gc_num()
            trace.gc_pause_end[index] = stats.pause
            trace.gc_total_time_end_ns[index] = stats.total_time
            trace.gc_allocated_bytes_end[index] = Base.gc_total_bytes(stats)
        end
        trace.end_ns[index] = time_ns()
    end
    return nothing
end

@inline _preflight_callback_outputs!(owner, inputs, outputs) =
    _preflight_output_slots!(values(owner.output_records), _metadata_sources(owner), inputs, outputs, 1)

function _process_graph_callback!(owner, filter, inputs, outputs)
    owner.active || throw(
        InvalidStateException("the Julia Graph Node is inactive", :inactive),
    )
    metrics = _graph_callback_metrics(owner)
    metrics_index = _begin_callback_metrics!(metrics, inputs)
    properties_changed = false
    succeeded = false
    parameter_adoption_count = _parameter_adoption_count(owner)
    reference_before = _recovery_reference_reached(owner)
    frame_observer_failed_before = _frame_observer_failed(_owner_frame_observer(owner))
    trace = _graph_callback_trace(owner)
    trace_index = 0
    try
        length(inputs)==length(_public_input_bindings(owner)) && length(outputs)==length(owner.output_records) ||
            throw(DimensionMismatch("PipeWire callback buffer count differs from the declared ports"))
        _preflight_input_facts!(owner,inputs)
        _preflight_callback_outputs!(owner, inputs, outputs)
        recovery_changed = _adopt_pending_recovery!(owner)
        properties_changed = _adopt_pending_properties!(owner) || recovery_changed
        trace_index = _trace_graph_callback_begin!(trace, inputs)
        _process_graph_buffers!(owner, owner.join, inputs, outputs)
        owner.process_count += UInt64(1)
        succeeded = true
    catch error
        if error isa MetadataPublicationError
            _latch_metadata_fault!(owner,error)
        end
        rethrow()
    finally
        if !succeeded
            _set_outputs_unavailable!(values(owner.output_records),outputs,1)
            _invalidate_recovery_observation!(owner)
            _invalidate_recovery_branch_readiness!(owner)
            _frame_observer_event!(_owner_frame_observer(owner), Val(:invalidated))
        end
        properties_changed |= reference_before != _recovery_reference_reached(owner)
        properties_changed |= frame_observer_failed_before != _frame_observer_failed(_owner_frame_observer(owner))
        _trace_graph_callback_end!(trace, trace_index)
        parameters_changed =
            _parameter_adoption_count(owner) != parameter_adoption_count
        try
            _notify_graph_properties!(owner, filter, properties_changed || parameters_changed)
        finally
            _end_callback_metrics!(metrics, metrics_index,
                properties_changed, parameters_changed, succeeded)
        end
    end
    return nothing
end

function _process_graph_callback!(
    slot::PreparedCapturedGraphSlot,
    filter,
    inputs,
    outputs,
)
    previous = Threads.atomic_cas!(
        slot.callback_state,
        _CAPTURED_CALLBACK_IDLE,
        _CAPTURED_CALLBACK_ACTIVE,
    )
    previous == _CAPTURED_CALLBACK_IDLE || throw(
        InvalidStateException(
            "captured PipeWire callbacks must remain serialized",
            :busy,
        ),
    )
    try
        owner = _active_captured_owner(slot)
        return _process_graph_callback!(owner, filter, inputs, outputs)
    finally
        slot.callback_state[] = _CAPTURED_CALLBACK_IDLE
    end
end

function (callback::GraphProcessCallback)(
    filter::NdArrayFilter,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    return _process_graph_callback!(callback.owner, filter, inputs, outputs)
end

struct GraphParameterCallback{Owner}
    owner::Owner
end

function (callback::GraphParameterCallback)(
    ::NdArrayFilter,
    input_port::Int,
    parameter::NdArrayFilterBuffer{false},
)
    owner = callback.owner
    trace = owner.parameter_trace
    if trace !== nothing
        trace.current_port = input_port
    end
    _trace_parameter_phase!(trace, owner, UInt8(1)) # entry
    parameter_index = input_port - length(_public_input_bindings(owner))
    1 <= parameter_index <= length(owner.parameter_bindings) || throw(
        ArgumentError("PipeWire Parameter Port index does not match the Graph"),
    )
    name = keys(owner.parameter_bindings)[parameter_index]
    binding = values(owner.parameter_bindings)[parameter_index]
    values_array = _borrow_buffer(binding, parameter)
    _trace_parameter_phase!(trace, owner, UInt8(2)) # borrowed
    return _stage_parameter_transaction!(owner, trace, name => values_array)
end

struct GraphDeactivateCallback{Owner}
    owner::Owner
end

_spa_property_value(value::UInt32) = PipeWireAO.SPA.Id(value)
_spa_property_value(value) = value

function _spa_property_type(info)
    default = _spa_property_value(info.default_value)
    constraint = info.constraint
    if constraint isa FilterGraphAlgorithms.PropertyRange
        return PipeWireAO.Pod(
            PipeWireAO.SPA.Choice(
                PipeWireAO.SPA.CHOICE_RANGE,
                [
                    default,
                    _spa_property_value(constraint.minimum),
                    _spa_property_value(constraint.maximum),
                ],
            ),
        )
    end
    if constraint isa FilterGraphAlgorithms.PropertyChoices
        values = [default]
        append!(values, _spa_property_value(choice.value) for choice in constraint.choices)
        return PipeWireAO.Pod(PipeWireAO.SPA.Choice(PipeWireAO.SPA.CHOICE_ENUM, values))
    end
    return PipeWireAO.Pod(default)
end

function _graph_property_info(owner::PreparedGraphOwner)
    result = PipeWireAO.SPA.PropInfo[]
    for node in owner.graph.nodes
        plan = node.prepared.plan
        if plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan
            for info in FilterGraphAlgorithms.property_infos(plan)
                push!(
                    result,
                    PipeWireAO.SPA.PropInfo(
                        "$(node.name):$(info.name)",
                        _spa_property_type(info);
                        description=info.description,
                        display_name=string(info.name),
                        group="$(node.name)/Controls",
                        flags=info.update isa FilterGraphAlgorithms.RuntimeProperty ?
                            0 : PipeWireAO.SPA.PROPERTY_READONLY,
                        params=true,
                    ),
                )
            end
            push!(
                result,
                PipeWireAO.SPA.PropInfo(
                    "$(node.name):requested-generation",
                    PipeWireAO.Pod(Int64(0));
                    display_name="requested-generation",
                    description="Latest property generation published for processing",
                    group="$(node.name)/Status",
                    flags=PipeWireAO.SPA.PROPERTY_READONLY,
                    params=true,
                ),
                PipeWireAO.SPA.PropInfo(
                    "$(node.name):active-generation",
                    PipeWireAO.Pod(Int64(0));
                    display_name="active-generation",
                    description="Property generation used by processing",
                    group="$(node.name)/Status",
                    flags=PipeWireAO.SPA.PROPERTY_READONLY,
                    params=true,
                ),
            )
        end
        push!(
            result,
            PipeWireAO.SPA.PropInfo(
                "$(node.name):requested-parameter-sequence",
                PipeWireAO.Pod(Int64(0));
                display_name="requested-parameter-sequence",
                description="Latest accepted ndarray Parameter Port sequence",
                group="$(node.name)/Status",
                flags=PipeWireAO.SPA.PROPERTY_READONLY,
                params=true,
            ),
            PipeWireAO.SPA.PropInfo(
                "$(node.name):active-parameter-sequence",
                PipeWireAO.Pod(Int64(0));
                display_name="active-parameter-sequence",
                description="Ndarray Parameter Port sequence used by processing",
                group="$(node.name)/Status",
                flags=PipeWireAO.SPA.PROPERTY_READONLY,
                params=true,
            ),
        )
    end
    return result
end

_graph_property_info(owner) = PipeWireAO.SPA.PropInfo[]

_graph_properties(owner) = PipeWireAO.SPA.Props()

function _graph_properties(owner::PreparedGraphOwner)
    snapshot = @atomic :acquire owner.control_snapshot
    return _graph_properties(owner, snapshot)
end

function _graph_properties(owner::PreparedGraphOwner, snapshot::GraphControlSnapshot)
    result = Pair{String,Any}[]
    for (index, node) in pairs(owner.graph.nodes)
        plan = snapshot.plans[index]
        if plan isa FilterGraphAlgorithms.AbstractAlgorithmPlan
            for info in FilterGraphAlgorithms.property_infos(plan)
                push!(
                    result,
                    "$(node.name):$(info.name)" =>
                    _spa_property_value(FilterGraphAlgorithms.property_value(plan, info.id)),
                )
            end
            push!(
                result,
                "$(node.name):requested-generation" =>
                Int64(snapshot.requested_property_generations[index]),
                "$(node.name):active-generation" =>
                Int64(snapshot.active_property_generations[index]),
            )
        end
        push!(
            result,
            "$(node.name):requested-parameter-sequence" =>
            Int64(snapshot.requested_parameter_sequences[index]),
            "$(node.name):active-parameter-sequence" =>
            Int64(snapshot.active_parameter_sequences[index]),
        )
    end
    return PipeWireAO.SPA.Props(result)
end

# Scalar status is read off the processing path and never enters a Graph transaction.
struct PropertyGetter{F,Declarations}
    callback::F
    declarations::Declarations
    owner_thread::Int
end

function _property_scalar_type(pod::PipeWireAO.Pod)
    type = PipeWireAO.pod_type(pod)
    if type == PipeWireAO.LibPipeWire.SPA_TYPE_Choice
        choice = PipeWireAO.pod_value(PipeWireAO.SPA.Choice, pod)
        type = PipeWireAO.pod_type(PipeWireAO.Pod(first(choice.values)))
    end
    type in (
        PipeWireAO.LibPipeWire.SPA_TYPE_Bool,
        PipeWireAO.LibPipeWire.SPA_TYPE_Id,
        PipeWireAO.LibPipeWire.SPA_TYPE_Int,
        PipeWireAO.LibPipeWire.SPA_TYPE_Long,
        PipeWireAO.LibPipeWire.SPA_TYPE_Float,
        PipeWireAO.LibPipeWire.SPA_TYPE_Double,
        PipeWireAO.LibPipeWire.SPA_TYPE_String,
    ) || throw(ArgumentError("extra properties must have scalar SPA types"))
    return type
end

function _property_getter(property_info, callback, graph_info)
    isempty(property_info) && callback === nothing && return nothing
    isempty(property_info) && throw(ArgumentError("property_getter requires extra property_info"))
    callback === nothing && throw(ArgumentError("extra property_info requires property_getter"))
    isconcretetype(typeof(callback)) || throw(ArgumentError("property_getter must be concrete"))
    applicable(callback, PipeWireAO.SPA.Props(), true) || throw(
        ArgumentError("property_getter must accept (SPA.Props, Bool)"))
    names = Set(info.name for info in graph_info)
    declarations = map(property_info) do info
        info.flags & PipeWireAO.SPA.PROPERTY_READONLY != 0 || throw(
            ArgumentError("extra property $(info.name) must be read-only"))
        info.container === nothing || throw(ArgumentError("extra properties must be scalar"))
        (info.name in names || info.name == _CHANNELS_PROPERTY ||
         info.name == "pipewireao.run-control" || startswith(info.name, "pipewireao.run-control.") ||
         info.name == "pipewireao.reset-control" || startswith(info.name, "pipewireao.reset-control.")) &&
            throw(ArgumentError("extra property $(info.name) collides with a reserved property"))
        push!(names, info.name)
        return info.name => _property_scalar_type(info.type)
    end
    return PropertyGetter(callback, Tuple(declarations), Threads.threadid())
end

_observer_available(::Nothing) = true
_observer_available(observer::PropertyObserver) = !observer.disabled[]

_get_extra_properties(::Nothing, observer, properties) = nothing
function _get_extra_properties(getter::PropertyGetter, observer, properties)
    Threads.threadid() == getter.owner_thread || throw(
        InvalidStateException("property getter called outside its owner thread", :thread))
    extra = getter.callback(properties, _observer_available(observer))
    return _validate_extra_properties(extra, getter.declarations)
end

_validate_extra_properties(extra, declarations) =
    throw(ArgumentError("property_getter must return SPA.Props"))

function _validate_extra_properties(extra::PipeWireAO.SPA.Props, declarations)
    length(extra.values) == length(declarations) || throw(
        ArgumentError("property_getter must return exactly its declared extra properties"))
    names = Set{String}()
    for (name, value) in extra.values
        name in names && throw(ArgumentError("property_getter returned duplicate property $name"))
        push!(names, name)
        index = findfirst(pair -> pair.first == name, declarations)
        index === nothing && throw(ArgumentError("property_getter returned undeclared property $name"))
        PipeWireAO.pod_type(value) == declarations[index].second || throw(
            ArgumentError("property_getter returned the wrong SPA type for $name"))
        # Validate scalar body shape as well as its header type.
        PipeWireAO.pod_value(value)
    end
    return extra
end

_reject_extra_property_writes(::Nothing, properties) = nothing
function _reject_extra_property_writes(getter::PropertyGetter, properties)
    any(properties.values) do (name, _)
        any(pair -> pair.first == name, getter.declarations)
    end && throw(ArgumentError("extra properties are read-only"))
    return nothing
end

struct GraphGetPropertiesCallback{Owner,Observer,Getter,Recovery}
    owner::Owner
    channels::Union{Nothing,PipeWireAO.Pod}
    observer::Observer
    getter::Getter
    recovery::Recovery
end
GraphGetPropertiesCallback(owner, channels) = GraphGetPropertiesCallback(owner, channels, nothing)
GraphGetPropertiesCallback(owner, channels, observer) =
    GraphGetPropertiesCallback(owner, channels, observer, nothing)
GraphGetPropertiesCallback(owner, channels, observer, getter) =
    GraphGetPropertiesCallback(owner, channels, observer, getter, nothing)

function (callback::GraphGetPropertiesCallback)(::NdArrayFilter)
    properties = _graph_properties_for_control(callback.owner, callback.recovery)
    _append_frame_observer_properties!(callback.owner, properties)
    isnothing(callback.channels) ||
        push!(properties.values, _CHANNELS_PROPERTY => callback.channels)
    _notify_property_observer!(callback.observer, Val(:observed), properties)
    extra = _get_extra_properties(callback.getter, callback.observer, properties)
    extra === nothing && return properties
    # Keep observer facts and getter input separate from the native combined response.
    return PipeWireAO.SPA.Props(vcat(properties.values, extra.values))
end

struct GraphSetPropertiesCallback{Owner,Observer,Getter,Recovery}
    owner::Owner
    observer::Observer
    getter::Getter
    recovery::Recovery
end
GraphSetPropertiesCallback(owner) = GraphSetPropertiesCallback(owner, nothing)
GraphSetPropertiesCallback(owner, observer) = GraphSetPropertiesCallback(owner, observer, nothing)
GraphSetPropertiesCallback(owner, observer, getter) = GraphSetPropertiesCallback(owner, observer, getter, nothing)

function (callback::GraphSetPropertiesCallback)(
    ::NdArrayFilter,
    properties::PipeWireAO.SPA.Props;
    throw_on_release_rejection::Bool=false,
)
    _stage_native_recovery!(callback.recovery, properties, callback.observer;
        throw_on_release_rejection) && return nothing
    _reject_extra_property_writes(callback.getter, properties)
    any(pair -> pair.first == _CHANNELS_PROPERTY, properties.values) &&
        throw(ArgumentError("source ndarray channels are read-only"))
    callback.owner isa PreparedGraphOwner ||
        throw(ArgumentError("this Graph owner has no writable scalar properties"))
    # This control callback runs off the processing path. Opt-in timestamps
    # distinguish entry, dynamic preparation, and handoff without frame logs.
    if get(ENV, "JULIA_FGPW_PROPERTY_DIAG", "0") == "1"
        start_ns = time_ns()
        _record_property_diagnostic(:callback_enter, start_ns)
        return _stage_property_transaction!(callback.owner, properties, start_ns, callback.observer)
    end
    return _stage_property_transaction!(callback.owner, properties, UInt64(0), callback.observer)
end

struct GraphResetCallback{Owner}
    owner::Owner
end

struct GraphApplyPropertiesCallback{Owner}
    owner::Owner
end

function (callback::GraphApplyPropertiesCallback)(::NdArrayFilter)
    _adopt_pending_recovery!(callback.owner)
    _adopt_pending_properties!(callback.owner)
    return nothing
end

function (callback::GraphResetCallback)(::NdArrayFilter)
    return _reset_owner!(callback.owner)
end

function (callback::GraphDeactivateCallback)(::NdArrayFilter)
    return _deactivate_owner(callback.owner)
end

function _pipewire_graph_node(
    owner::PreparedGraphOwner,
    graph::PreparedGraph;
    ports,
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    connect::Bool=true,
    independent_inputs::Bool=false,
    fifo_inputs::Bool=false,
    parameter_ports::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    progressive=nothing,
    property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[],
    property_getter=nothing,
    recovery=nothing,
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    parameter_callback = !parameter_ports || isempty(graph.parameter_formats) ?
        nothing : GraphParameterCallback(owner)
    return _pipewire_owned_graph_node(
        owner;
        ports,
        parameter_callback,
        name,
        remote,
        connect,
        independent_inputs,
        fifo_inputs,
        run_control,
        channel_units,
        reference_clock,
        progressive,
        property_observer,
        property_info,
        property_getter,
        recovery,
    )
end

function _pipewire_owned_graph_node(
    owner;
    ports,
    parameter_callback=nothing,
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    connect::Bool=true,
    independent_inputs::Bool=false,
    fifo_inputs::Bool=false,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    progressive=nothing,
    property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[],
    property_getter=nothing,
    recovery=nothing,
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    ports = _prepare_metadata_ports(ports,owner,reference_clock)
    prepare_callback = GraphPrepareCallback(owner)
    process_callback = GraphProcessCallback(owner)
    deactivate_callback = GraphDeactivateCallback(owner)
    observer = _property_observer(property_observer)
    channels = _source_channels(ports, channel_units)
    recovery_control = _native_recovery(owner, recovery)
    graph_info = vcat(_graph_property_info(owner), _native_recovery_property_info(recovery_control),
        _frame_observer_property_info(owner))
    isnothing(channels) || push!(graph_info, _source_channel_info(channels))
    extra_info = collect(PipeWireAO.SPA.PropInfo, property_info)
    getter = _property_getter(extra_info, property_getter, graph_info)
    property_info = vcat(graph_info, extra_info)
    property_options = isempty(property_info) ?
        (; reset_control=run_control, on_reset=GraphResetCallback(owner)) :
        (;
            property_info,
            on_get_properties=GraphGetPropertiesCallback(owner, channels, observer, getter, recovery_control),
            on_properties=GraphSetPropertiesCallback(owner, observer, getter, recovery_control),
            (owner isa PreparedGraphOwner ?
                (; on_apply_properties=GraphApplyPropertiesCallback(owner)) : (;))...,
            reset_control=run_control,
            on_reset=GraphResetCallback(owner),
        )
    run_control_options = run_control ? (; run_control=true) : (;)
    fifo_input_options = fifo_inputs ? (; fifo_inputs=true) : (;)
    progressive_options = isnothing(progressive) ? (;) : (; progressive)
    if get(ENV, "JULIA_FGPW_PROPERTY_DIAG", "0") == "1"
        precompile(_record_property_diagnostic, (Symbol, UInt64))
        _record_property_diagnostic(:logger_ready, time_ns())
    end
    filter = NdArrayFilter(
        name,
        ports;
        remote,
        on_prepare=prepare_callback,
        on_process=process_callback,
        on_parameter=parameter_callback,
        on_deactivate=deactivate_callback,
        independent_inputs,
        progressive_options...,
        fifo_input_options...,
        property_options...,
        run_control_options...,
    )
    # The native Start path reaches these Julia methods on PipeWire's process
    # thread. Compile their specializations before publishing the node.
    precompile(PipeWireAO._ndarray_filter_prepare, (typeof(filter),))
    native_buffer = Ptr{PipeWireAO.LibPipeWire.pw_ndarray_filter_buffer}
    precompile(PipeWireAO._ndarray_filter_process,
               (typeof(filter), native_buffer, UInt32, native_buffer, UInt32))
    precompile(prepare_callback, (typeof(filter),))
    precompile(
        process_callback,
        (
            typeof(filter),
            NdArrayFilterBuffers{false},
            NdArrayFilterBuffers{true},
        ),
    )
    if !isnothing(parameter_callback)
        precompile(
            PipeWireAO._ndarray_filter_update_parameter,
            (typeof(filter), UInt32, native_buffer),
        )
        precompile(
            parameter_callback,
            (typeof(filter), Int, NdArrayFilterBuffer{false}),
        )
        _precompile_parameter_preparation!(owner)
    end
    if !isempty(property_info)
        if owner isa PreparedGraphOwner
            precompile(PipeWireAO._ndarray_filter_apply_properties, (typeof(filter),))
            precompile(GraphApplyPropertiesCallback(owner), (typeof(filter),))
        end
        precompile(
            PipeWireAO._ndarray_filter_set_props,
            (typeof(filter), Ptr{PipeWireAO.LibPipeWire.spa_pod}),
        )
        precompile(
            GraphSetPropertiesCallback(owner, observer, getter, recovery_control),
            (typeof(filter), PipeWireAO.SPA.Props),
        )
        _warm_property_preparation!(owner)
    end
    node = PipeWireNode(filter, owner)
    try
        # Supply cold facts even when connection is deferred, so applications
        # can validate exact property bindings before admission.
        if observer !== nothing || getter !== nothing
            GraphGetPropertiesCallback(owner, channels, observer, getter, recovery_control)(filter)
        end
        connect && connect!(node)
    catch
        close(node)
        rethrow()
    end
    return node
end

function _pipewire_captured_graph_node(
    owner::PreparedCapturedGraphSlot,
    graph::PreparedCapturedGraph;
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    connect::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    input_metadata::NamedTuple=NamedTuple(),
    output_metadata::NamedTuple=NamedTuple(),
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    return _pipewire_owned_graph_node(
        owner;
        ports=_metadata_ports(_captured_graph_ports(graph, rate),input_metadata,output_metadata),
        name,
        remote,
        connect,
        run_control,
        channel_units,
        reference_clock,
    )
end


"""
    PipeWireNode(graph::PreparedCapturedGraph;
                 name, remote=nothing, rate=nothing, connect=true,
                 run_control=false)

Publish one fixed-storage captured Graph as a synchronous, capacity-one
PipeWire ndarray Node. The captured Graph must stage every public output Port.
Every output callback buffer is reserved and marked unavailable before input
admission or native launch. Required Header metadata
is finalized before launch, and successful outputs become available only when
the exclusive callback returns.

Captured Nodes expose data Ports only. Apply calibration and other Parameter
updates to a checked `PreparedGraph` before capture, then prepare, warm,
validate, and swap a replacement captured owner at a control-plane boundary.
Set `run_control=true` only when an external RTC owns session activation. The
Node then connects stopped and accepts the PipeWireAO owner-mediated run-control
parameter; the Julia process continues to own the Node and its algorithm state.
"""
function PipeWireNode(
    graph::PreparedCapturedGraph;
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    connect::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    input_metadata::NamedTuple=NamedTuple(),
    output_metadata::NamedTuple=NamedTuple(),
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    return _pipewire_captured_graph_node(
        PreparedCapturedGraphSlot(PreparedCapturedGraphOwner(graph)),
        graph;
        name,
        remote,
        rate,
        connect,
        run_control,
        channel_units,
        reference_clock,
        input_metadata,output_metadata,
    )
end


@inline function _same_captured_port_format(left::AlgorithmFormat, right::AlgorithmFormat)
    return left.element_type === right.element_type &&
           typeof(left.layout) === typeof(right.layout) &&
           left.shape == right.shape &&
           left.schema == right.schema &&
           left.rate == right.rate
end

@inline _same_captured_port_formats(::Tuple{}, ::Tuple{}) = true
@inline _same_captured_port_formats(::Tuple{}, right::Tuple) = false
@inline _same_captured_port_formats(left::Tuple, ::Tuple{}) = false

@inline function _same_captured_port_formats(left::Tuple, right::Tuple)
    return _same_captured_port_format(first(left), first(right)) &&
           _same_captured_port_formats(Base.tail(left), Base.tail(right))
end

function _same_captured_port_formats(left::NamedTuple, right::NamedTuple)
    keys(left) == keys(right) || return false
    return _same_captured_port_formats(values(left), values(right))
end

function _require_captured_replacement(
    slot::PreparedCapturedGraphSlot{Owner},
    owner::PreparedCapturedGraphOwner,
) where {Owner}
    owner isa Owner || throw(
        ArgumentError(
            "replacement must have the same concrete captured execution shape",
        ),
    )
    _same_captured_port_formats(slot.input_formats, owner.graph.input_formats) ||
        throw(DimensionMismatch("replacement captured Graph input Ports differ"))
    _same_captured_port_formats(slot.output_formats, owner.graph.output_formats) ||
        throw(DimensionMismatch("replacement captured Graph output Ports differ"))
    owner.graph !== slot.first.graph && owner.graph !== slot.second.graph || throw(
        ArgumentError("replacement must own a distinct captured Graph"),
    )
    return owner
end

"""
    prepare_replacement(node::PipeWireNode, graph::PreparedCapturedGraph)

Validate, warm, and reset one distinct captured Graph away from the PipeWire
callback. The replacement must have exactly the Node's published input and
output Ports and the same concrete captured execution shape. Failure leaves the
active owner unchanged.

The replacement's configured reset state becomes authoritative after the swap;
this operation does not infer or migrate mutable controller state.
"""
function prepare_replacement(
    node::PipeWireNode{
        <:NdArrayFilter,
        <: PreparedCapturedGraphSlot{Owner},
    },
    graph::PreparedCapturedGraph,
) where {Owner}
    isopen(node) || throw(
        InvalidStateException("the PipeWire Node is closed", :closed),
    )
    owner = _require_captured_replacement(
        node.owner,
        PreparedCapturedGraphOwner(graph),
    )
    _copy_output_metadata_policy!(owner,_active_captured_owner(node.owner))
    _prepare_owner(owner)
    captured_graph_state(owner.graph) === CapturedGraphReady || throw(
        InvalidStateException("replacement captured Graph is not ready", :faulted),
    )
    return PreparedCapturedGraphReplacement(
        node.owner,
        owner,
        Threads.Atomic{UInt8}(_CAPTURED_REPLACEMENT_PREPARED),
    )
end

function _replace_graph!(
    slot::PreparedCapturedGraphSlot{Owner},
    replacement::PreparedCapturedGraphReplacement{<:Any,Owner},
) where {Owner}
    lock(slot.control_lock)
    try
        replacement.state[] == _CAPTURED_REPLACEMENT_PREPARED || throw(
            InvalidStateException(
                "captured Graph replacement was already consumed",
                :consumed,
            ),
        )
        slot.callback_state[] == _CAPTURED_CALLBACK_IDLE || throw(
            InvalidStateException(
                "captured PipeWire callback is active; retry the control-plane swap",
                :busy,
            ),
        )
        replacement.owner.active || throw(
            InvalidStateException("replacement captured owner is inactive", :inactive),
        )
        captured_graph_state(replacement.owner.graph) === CapturedGraphReady ||
            throw(
                InvalidStateException(
                    "replacement captured Graph is not ready",
                    :faulted,
                ),
            )

        _copy_output_metadata_policy!(replacement.owner,_active_captured_owner(slot))
        active_index = slot.active_index[]
        if active_index == UInt8(1)
            slot.second === slot.first || _deactivate_owner(slot.second)
            slot.second = replacement.owner
            slot.active_index[] = UInt8(2)
        else
            _deactivate_owner(slot.first)
            slot.first = replacement.owner
            slot.active_index[] = UInt8(1)
        end
        replacement.state[] = _CAPTURED_REPLACEMENT_CONSUMED
    finally
        unlock(slot.control_lock)
    end
    return nothing
end

"""
    replace_graph!(node::PipeWireNode, replacement::PreparedCapturedGraphReplacement)

Publish one fully prepared captured owner at a callback boundary. If a callback
is active, return a bounded `InvalidStateException` with state `:busy`; the
caller may retry from the control plane. A replacement can be consumed once.
"""
function replace_graph!(
    node::PipeWireNode{<:NdArrayFilter,<:PreparedCapturedGraphSlot},
    replacement::PreparedCapturedGraphReplacement,
)
    isopen(node) || throw(
        InvalidStateException("the PipeWire Node is closed", :closed),
    )
    replacement.target === node.owner || throw(
        ArgumentError("captured Graph replacement belongs to another PipeWire Node"),
    )
    _replace_graph!(node.owner, replacement)
    return node
end


"""
    PipeWireNode(graph::PreparedGraph; name, remote=nothing, rate=nothing,
                 boundary_layout=nothing, connect=true, fifo_inputs=false,
                 run_control=false, frame_observer=nothing, local_output=nothing, property_observer=nothing,
                 property_info=PipeWireAO.SPA.PropInfo[], property_getter=nothing)

Publish one prepared Julia Filter Graph as one standalone PipeWire ndarray
Node. `rate` is the Graph deployment base rate; each external Algorithm Port
ratio is multiplied by it using FGN semantics.

`boundary_layout` may override the layout published to PipeWire. The borrowed
array boundary maps multidimensional indices to the negotiated memory order,
so scientific Graph code retains Julia's logical indexing without copying the
PipeWire payload.

`input_metadata` and `output_metadata` declare exact per-Port record layouts,
for example `(pixels=(metadata=(PipeWireAO.MetadataRequirement(
PipeWireAO.SPA.META_ACQUISITION_IDENTITY;required=true),),),)`.
Header is required independently. `propagation` explicitly selects preserve,
produce, replace or drop. Held, joined, feedback and captured paths own bounded
snapshots before releasing input leases, including unknown optional records.
The declared `metadata_from` source governs preservation. Derived outputs retain
only scientific facts that remain true; sums preserve explicit named contributors.

`local_output=port => callback` retains one logical Graph output in private
preallocated storage and omits its native Port. A terminal publication invokes
`callback(Val(:frame), array, metadata)` after native output finalization. The
read-only array is borrowed for the call. After Stop, the last accepted observation
retains its acquisition identity and time. A stopped Graph Reset calls
`callback(Val(:reset))` to clear that observation. Pass `port => nothing` to
discard the local result. Keep callbacks bounded and nonblocking. Exceptions
disable the local observation for this owner. `frame_observer` receives native
outputs.

`frame_observer`, when supplied, is a concrete callable for ordinary CPU complete-
frame execution, including the owned delayed-feedback bridge. After a terminal
sample makes every Graph output available, it receives
`observer(Val(:frame), outputs::NamedTuple, metadata::NamedTuple)`. Both tuples
have the declared public output Port names; the bridge's private delayed-feedback
output is excluded. Arrays are read-only borrowed views, valid only until this
call returns. Do not retain them or their pointers. Each metadata value is the
actual output `FilterGraphAlgorithms.AdmittedSampleMetadata` record. Check
`present` before reading `value::SampleMetadata`; when absent, its value must not
be read as sample evidence. Sequence, publication timestamp, discontinuity/
corruption and acquisition facts are preserved. These are source facts, not a
newly generated observer clock.

`observer(Val(:invalidated))` reports a partial/rejected sample or processing
exception; `observer(Val(:reset))` precedes a stopped Graph reset. No invalidation
or reset clears a previous observer failure. Observer calls are serialized with
processing/reset. They must be bounded and must not yield, issue controls, or
mutate Graph state. The observer owns its own single-writer state and any atomic
publication to control readers; property callbacks must not mutate that state.
Exceptions disable the observer for this owner without changing accepted science,
retrying, retaining the exception, or logging from processing. Configured observers
publish read-only `pipewireao.frame-observer.state` as `"active"` or `"failed"`
through ordinary scalar readback. `"active"` describes observation availability,
not sample validity. Observer updates request coalesced property publication at
100 ms intervals; a newly failed observer requests immediate publication. With
`nothing`, there is no observation or status property.
Joined, deferred-first-input, progressive and captured owners are unsupported;
row-block startup preparation is refused. Synthetic startup preparation does not
invoke frame observation.

`property_observer`, when supplied, is a concrete callable accepting
`(Val(:requested), props::PipeWireAO.SPA.Props)` and `(Val(:observed), props)`.
After successful staging it receives the pre-publication applied snapshot,
then desired values with new requested and previous active generations.
Readback observations contain applied values from the same snapshot returned
to native clients. Calls run on the constructing thread through serialized
native scalar callbacks, never frame processing. Treat `props` as read-only;
do not reenter control callbacks. Exceptions or thread-affinity violations
log a failure and disable observation without rejecting accepted writes or
preventing native readback. See the deployment guide for identity and
notification-coalescing rules.

`property_info` and `property_getter` may supply additional read-only scalar
status properties. Supply both a nonempty collection of `SPA.PropInfo` and a
concrete callable `property_getter(graphprops::SPA.Props, observer_available::Bool)`.
The getter runs after observer reconciliation, on the constructing thread, using
the same applied Graph snapshot that native readback receives. It must return
`SPA.Props` containing exactly the declared extra names and SPA scalar types.
The native response combines those values with Graph properties; observer facts
and scientific transactions contain only Graph properties. `observer_available`
is false after a configured observer has been disabled, and true otherwise.
The getter also runs during disconnected construction. Its exceptions propagate
and fail native readback; constructing-thread violations reject the call before
entering user code. Treat its input as read-only and do not reenter callbacks.
Extra names must not collide with Graph, channel, or native run/reset controls.
Writes to extra properties are rejected before Graph transaction preparation.
Captured Graphs do not accept these optional property keywords.

The current native endpoint admits Graph data Ports, per-output availability,
typed expected Algorithm failures, FGN Header sample metadata, and sparse
Parameter Ports. Parameter preparation runs on PipeWireAO's bounded worker;
the prepared transaction becomes active at the next Graph callback boundary.
An expected failure discards the rejecting Algorithm's candidate state and
makes every output unavailable for that callback. State already committed by
earlier Nodes is retained.

Set `fifo_inputs=true` only for a loss-intolerant progressive input boundary.
It requests PipeWireAO FIFO admission: each frame-data buffer dequeued by the
native helper reaches the callback once, in order, and temporary output
starvation applies back pressure instead of silently discarding a block. The
graph driver must still present distinct PipeWire graph cycles; FIFO admission
cannot recover cycles that a driver overlaps or coalesces before the helper
sees them. Ordinary complete-frame Graphs retain the native drain-to-latest
policy by default.

`recovery=(controller=:correction, participants=(:temporal, :command))` optionally
declares a protective CPU recovery scope and fixed controls under virtual Node
`pipewireao-recovery`. Its request/release tokens use SPA Long values; recovery
and ordinary scalar writes cannot be mixed. The scope keeps closure inhibited
until the committed conditioner currents equal their internal requests and a matching explicit release is adopted. Release leaves the controller
open. See [`prepare_recovery`](@ref), [`recovery_status`](@ref) and
[`stage_properties!`](@ref). Joined, progressive, captured and unsupported
reference scopes are refused before publication.

`ingress` maps external data input names to callbacks `(metadata, raw)`. Each
callback runs on the borrowed, read-only input before any Graph Node and must
return owned `SampleMetadata` or an `Enum` rejection. Missing Header metadata
remains absent and does not invoke the callback. A rejection makes all outputs
unavailable without running the Graph. Callbacks must not retain the raw array
or its pointer. They must be bounded and allocation-free for a low-latency host.
Before publishing an ingress electrical tag as whole-exposure triggerClassification,
the source adapter must establish its temporal scope over the complete exposure.
Raw C-RED 2 tag decoding alone does not establish that scope or optical purity.
Callbacks only annotate metadata; they must not mutate scientific state, since
another input can reject the callback and ingress has no rollback lifecycle.
Ingress currently supports ordinary complete-frame Graphs only; deferred,
feedback, progressive, joined, and captured boundaries are outside this API.
Ingress leaves borrowed input records unchanged. The selected output
`metadata_from` route serializes applicable owned annotations into declared,
negotiated metadata records. Compatible source facts and opaque optional records
are retained; the output must preserve their declared physical meaning.

`parameter_ports=false` omits live Parameter Ports and their sparse-update
callback. Bind the graph's parameters before construction; their declarations
and arrays remain part of the prepared graph. The default exposes live ports.

"""
function PipeWireNode(
    graph::PreparedGraph;
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    input_metadata::NamedTuple=NamedTuple(),
    output_metadata::NamedTuple=NamedTuple(),
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
    connect::Bool=true,
    fifo_inputs::Bool=false,
    parameter_ports::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    frame_observer=nothing,
    local_output::Union{Nothing,Pair{Symbol}}=nothing,
    property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[],
    property_getter=nothing,
    defer_first_input::Bool=false,
    trace_callbacks::Int=0,
    feedback::Union{Nothing,Pair{Symbol,Symbol}}=nothing,
    ingress::NamedTuple=NamedTuple(),
    recovery=nothing,
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    feedback !== nothing && defer_first_input && throw(
        ArgumentError("feedback delay cannot be combined with deferred first input"),
    )
    owner = PreparedGraphOwner(
        graph, nothing, boundary_layout;
        defer_first_input, trace_callbacks, feedback, ingress, frame_observer, local_output,
    )
    input_formats = isnothing(feedback) ? graph.input_formats :
        _without_port(graph.input_formats, feedback.first)
    output_formats = isnothing(feedback) ? graph.output_formats :
        _without_port(graph.output_formats, feedback.second)
    output_formats = _native_outputs(owner.local_output, output_formats)
    ports = _graph_ports(
        input_formats,
        parameter_ports ? graph.parameter_formats : NamedTuple(),
        output_formats,
        rate;
        boundary_layout,
    )
    ports = _metadata_ports(ports, input_metadata, output_metadata)
    return _pipewire_graph_node(
        owner,
        graph;
        ports,
        name,
        remote,
        connect,
        fifo_inputs,
        parameter_ports,
        run_control,
        channel_units,
        reference_clock,
        property_observer,
        property_info,
        property_getter,
        recovery,
    )
end

function PipeWireNode(graph::FilterGraphs.PreparedCpuProgressiveGraph;
    ingress::NamedTuple=NamedTuple(), frame_observer=nothing,
    input_metadata::NamedTuple=NamedTuple(), output_metadata::NamedTuple=NamedTuple(), kwargs...)
    isnothing(frame_observer) || throw(ArgumentError(
        "frame observation requires an ordinary complete-frame Graph"))
    graph.closed && throw(ArgumentError("CPU progressive Graph is closed"))
    isempty(ingress) || throw(ArgumentError("input ingress requires a complete-frame Graph"))
    return PipeWireNode(graph.graph; input_metadata,output_metadata,kwargs...)
end


"""
    PipeWireNode(join::PreparedLatestHold; name, remote=nothing, rate=nothing,
                 connect=true, run_control=false)

Publish a prepared primary-plus-latest/hold Graph boundary. The native helper
admits independent input arrivals. Slower arrivals are copied into the
prepared retained slot without publishing; primary arrivals run the Graph
with the current admitted held sample.
"""
function PipeWireNode(
    join::FilterGraphs.PreparedLatestHold;
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    connect::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[],
    property_getter=nothing,
    input_metadata::NamedTuple=NamedTuple(),
    output_metadata::NamedTuple=NamedTuple(),
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    return _pipewire_graph_node(
        PreparedGraphOwner(join),
        join.graph;
        ports=_metadata_ports(_latest_hold_ports(join, rate),input_metadata,output_metadata),
        name,
        remote,
        connect,
        independent_inputs=true,
        run_control,
        channel_units,
        reference_clock,
        property_observer,
        property_info,
        property_getter,
    )
end

"""
    PipeWireNode(synchronized::PreparedSynchronizedInputs;
                 name, remote=nothing, rate=nothing, connect=true,
                 run_control=false, boundary_layout=nothing)

Publish a prepared equal-rate, all-required Graph boundary. The native helper
admits independent input arrivals and copies each one into its prepared Port
slot. Outputs remain unavailable until every Port satisfies the prepared
synchronization policy and the Graph has completed successfully.
`boundary_layout` overrides the external Port layout and borrowed array bindings,
including row-major matrix inputs, while the Graph owns column-major storage.
"""
function PipeWireNode(
    synchronized::FilterGraphs.PreparedSynchronizedInputs;
    name::AbstractString="julia.filter-graph-island",
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
    connect::Bool=true,
    run_control::Bool=false,
    channel_units=NamedTuple(),
    property_observer=nothing,
    property_info=PipeWireAO.SPA.PropInfo[],
    property_getter=nothing,
    input_metadata::NamedTuple=NamedTuple(),
    output_metadata::NamedTuple=NamedTuple(),
    reference_clock::PipeWireAO.MetadataClock=PipeWireAO.METADATA_CLOCK_REALTIME,
)
    return _pipewire_graph_node(
        PreparedGraphOwner(synchronized, boundary_layout),
        synchronized.graph;
        ports=_metadata_ports(_graph_ports(synchronized.graph, rate; boundary_layout),input_metadata,output_metadata),
        name,
        remote,
        connect,
        independent_inputs=true,
        run_control,
        channel_units,
        reference_clock,
        property_observer,
        property_info,
        property_getter,
    )
end

"""
    replace_parameters!(node::PipeWireNode, updates::Pair...; supersede=false)

Prepare one failure-atomic Parameter Port transaction away from repeated Graph
processing and stage it in the Node's bounded handoff. The transaction becomes
active at the next Graph callback boundary. A second update is rejected while
preparation or adoption is pending. With `supersede=true`, only a single
`PhysicalOffsetReferenceF32` offset update may replace an unadopted offset-only
transaction for that same Algorithm. Calibration and sign remain unchanged;
requested sequences advance while active sequences wait for callback adoption.
An invalid replacement preserves the pending transaction. Protective recovery
still permits only an exact-zero offset. Captured execution is unsupported.
"""
function replace_parameters!(
    node::PipeWireNode{<:NdArrayFilter,<:PreparedGraphOwner},
    updates::Pair...; supersede::Bool=false,
)
    isopen(node) || throw(InvalidStateException("the PipeWire Node is closed", :closed))
    _stage_parameter_transaction!(node.owner, updates...; supersede) || throw(
        InvalidStateException("a Parameter Port transaction is already pending", :busy),
    )
    return node
end

function replace_parameters!(
    node::PipeWireNode{<:NdArrayFilter,<:PreparedCapturedGraphSlot},
    updates::Pair...; supersede::Bool=false,
)
    throw(
        InvalidStateException(
            "captured Graph Parameters require prepare, warm, validate, and owner swap",
            :captured,
        ),
    )
end

"""
    parameter_status(node::PipeWireNode, algorithm::Symbol)

Read requested/active parameter sequences and protective recovery inhibition
from one acquired immutable control snapshot. This is a cold control-task
readback, not proof that a submitted transaction has already been processed.
"""
function parameter_status(node::PipeWireNode{<:NdArrayFilter,<:PreparedGraphOwner}, algorithm::Symbol)
    isopen(node) || throw(InvalidStateException("the PipeWire Node is closed", :closed))
    return parameter_status(node.owner,algorithm)
end

function parameter_status(owner::PreparedGraphOwner,algorithm::Symbol)
    index = findfirst(entry -> entry.name === algorithm,owner.graph.nodes)
    index === nothing && throw(ArgumentError("unknown Algorithm Node $algorithm"))
    snapshot = @atomic :acquire owner.control_snapshot
    return (;requested_parameter_sequence=snapshot.requested_parameter_sequences[index],
        active_parameter_sequence=snapshot.active_parameter_sequences[index],
        closure_inhibited=snapshot.recovery.closure_inhibited)
end
public parameter_status

_metadata_sources(owner::PreparedGraphOwner) = owner.feedback===nothing ? owner.output_metadata_source_inputs : owner.feedback.output_metadata_source_inputs
_metadata_sources(owner::PreparedCapturedGraphSlot) = owner.first.graph.output_metadata_sources
_metadata_sources(owner::PreparedCapturedGraphOwner) = owner.graph.output_metadata_sources
function _prepare_metadata_ports(ports,owner,reference_clock=PipeWireAO.METADATA_CLOCK_REALTIME)
    inputs=filter(port -> port.direction==PipeWireAO.DIRECTION_INPUT && !port.parameter,ports)
    outputs=filter(port -> port.direction==PipeWireAO.DIRECTION_OUTPUT && !port.parameter,ports)
    sources=_metadata_sources(owner)
    _prepare_required_facts!(owner)
    length(sources)==length(outputs) || throw(ArgumentError("metadata provenance does not cover public outputs"))
    for (position,(output,index)) in enumerate(zip(outputs,sources))
        snapshot=getfield(values(owner isa PreparedCapturedGraphSlot ? owner.first.output_records : owner.output_records),position)
        snapshot.propagation=iszero(index) ? output.propagation : output.propagation==0 ? UInt32(1) : output.propagation
        snapshot.requirements=[PipeWireAO.MetadataRequirement(r.type,r.size;required=!iszero(r.flags&1)) for r in output.metadata]
        snapshot.clock=reference_clock
        snapshot.produce_header=iszero(index)
        snapshot.produce_time=isempty(inputs) && iszero(index) && output.propagation!=3 && any(r -> r.type==PipeWireAO.SPA.META_REFERENCE_TIMESTAMP,output.metadata)
        if !isempty(inputs) && iszero(index) && any(r -> r.type==PipeWireAO.SPA.META_REFERENCE_TIMESTAMP && !iszero(r.flags&1),output.metadata)
            throw(ArgumentError("required reference timestamp needs a declared triggering input metadata source"))
        end
        if output.propagation==3 && any(r -> !iszero(r.flags&1),output.metadata)
            throw(ArgumentError("a dropping output cannot satisfy required metadata"))
        end
        (iszero(index) || snapshot.propagation!=1) && continue
        source=inputs[index]
        for requirement in source.metadata
            any(r -> r.type==requirement.type && r.size==requirement.size,output.metadata) ||
                throw(ArgumentError("output $(output.name) cannot preserve input metadata type $(requirement.type)"))
        end
    end
    return ports
end
@inline _preflight_output_records!(::Tuple{},sources::Tuple,indices::Tuple,buffers,index)=nothing
@inline function _preflight_output_records!(outputs::Tuple,sources::Tuple,indices::Tuple,buffers,index)
    source_index=first(indices)
    snapshot=first(outputs)
    if snapshot.propagation==1 && !iszero(source_index)
        source=getfield(sources,source_index)
        source.present && !PipeWireAO.metadata_records_valid(PipeWireAO.metadata_records(source.records),snapshot.requirements) &&
            throw(InvalidStateException("input cannot supply required downstream metadata",:metadata))
        source.present && !PipeWireAO.metadata_fits(buffers[index],source.records) &&
            throw(InvalidStateException("output metadata capacity cannot preserve the retained input",:metadata))
    end
    return _preflight_output_records!(Base.tail(outputs),sources,Base.tail(indices),buffers,index+1)
end

@inline _finalize_output_times!(::Tuple{},::Tuple{},buffers,index)=nothing
@inline function _finalize_output_times!(available::Tuple,snapshots::Tuple,buffers,index)
    snapshot=first(snapshots)
    if first(available)
        _produce_output_time!(buffers[index],snapshot)
        snapshot.produce_header && !snapshot.metadata.present && (snapshot.publication_sequence+=UInt64(1))
    end
    return _finalize_output_times!(Base.tail(available),Base.tail(snapshots),buffers,index+1)
end
@inline _preflight_output_slots!(::Tuple{},::Tuple{},inputs,buffers,index)=nothing
@inline function _preflight_output_slots!(snapshots::Tuple,indices::Tuple,inputs,buffers,index)
    snapshot=first(snapshots)
    PipeWireAO.metadata_record(buffers[index],UInt32(PipeWireAO.SPA.META_HEADER))===nothing &&
        throw(InvalidStateException("output Header metadata was not negotiated",:metadata))
    snapshot.produce_header && snapshot.publication_sequence==typemax(UInt64) &&
        throw(InvalidStateException("output publication sequence is exhausted",:metadata))
    if snapshot.produce_time
        record=PipeWireAO.metadata_record(buffers[index],PipeWireAO.SPA.META_REFERENCE_TIMESTAMP)
        record===nothing || PipeWireAO.reference_timestamp_now(snapshot.clock)
    end
    PipeWireAO.metadata_fits(buffers[index],snapshot.requirements) ||
        throw(InvalidStateException("output metadata layout differs from its declaration",:metadata))
    source_index=first(indices)
    if snapshot.propagation==1 && !iszero(source_index) && PipeWireAO.input_available(inputs[source_index])
        source=PipeWireAO.metadata_records(inputs[source_index])
        PipeWireAO.metadata_records_valid(source,snapshot.requirements) && PipeWireAO.metadata_fits(buffers[index],source) ||
            throw(InvalidStateException("output cannot preserve required source metadata",:metadata))
    end
    return _preflight_output_slots!(Base.tail(snapshots),Base.tail(indices),inputs,buffers,index+1)
end
function _copy_output_metadata_policy!(destination,source)
    FilterGraphs.required_input_sample_metadata(destination.graph)==FilterGraphs.required_input_sample_metadata(source.graph) ||
        throw(ArgumentError("replacement changes required sample facts and needs cold port renegotiation"))
    _prepare_required_facts!(destination)
    for (target,original) in zip(values(destination.output_records),values(source.output_records))
        target.propagation=original.propagation
        target.requirements=copy(original.requirements)
        target.produce_time=original.produce_time
        target.produce_header=original.produce_header
        target.publication_sequence=original.publication_sequence
        target.clock=original.clock
    end
    return destination
end
