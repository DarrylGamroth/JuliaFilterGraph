"""
Standalone PipeWire ndarray Nodes backed by prepared Algorithm declarations.

This package is deployment glue. Scientists continue to implement algorithms
in `FilterGraphAlgorithms`; they do not handle PipeWire callbacks, C pointers,
buffer ownership, errno values, or generated bindings. The current vertical
slice hosts a prepared Julia Filter Graph with conditional outputs, sample
metadata, expected failure, transactional Workspaces, sparse Parameter Ports,
and bounded latest/hold and all-required synchronized-input boundaries. It
also hosts parameter-free captured Graphs with prelaunch publication
reservation and synchronous capacity-one replay. It remains a JIT development
runtime and is not admitted for hard-real-time use.
"""
module FilterGraphPipeWire

using FilterGraphAlgorithms
using FilterGraphs
using PipeWireAO
using UnsafeArrays: UnsafeArray
using LinearAlgebra: Transpose, transpose

import FilterGraphAlgorithms: process!, reset!
import FilterGraphs: replace_parameters!
import PipeWireAO: command_endpoint, connect!, last_error, node_id, quit!, run!

"""
An allocation-free, non-owning, one-based array projected over validated
foreign storage. `Writable` is `false` for PipeWire inputs and `true` for
exclusively owned outputs. Scientific algorithms continue to accept ordinary
`AbstractArray` values.
"""
struct BorrowedArray{T,N,Writable} <: AbstractArray{T,N}
    data::Ptr{T}
    shape::NTuple{N,Int}
    length::Int
end

"""A logical Julia array projected over a row-major PipeWire payload."""
struct RowMajorBorrowedArray{T,N,Writable} <: AbstractArray{T,N}
    data::Ptr{T}
    shape::NTuple{N,Int}
    length::Int
end

const BorrowedStorageArray = Union{BorrowedArray,RowMajorBorrowedArray}

# Camera tags refer to transmitted words, before the row-major payload is
# projected into logical Julia coordinates. Source ingress must establish that
# the tag classifies the complete exposure before publishing it as the transport
# trigger classification; this decoder establishes neither temporal scope nor
# an optical classification.
function FilterGraphAlgorithms.cred2_trigger_metadata(
    metadata::SampleMetadata, raw_image::RowMajorBorrowedArray{UInt16,N,W},
) where {N,W}
    transmitted = BorrowedArray{UInt16,1,W}(
        raw_image.data, (length(raw_image),), length(raw_image),
    )
    return cred2_trigger_metadata(metadata, transmitted)
end

# These pointer wrappers are isbits, so Base's default mightalias would always
# return false. Both layouts occupy one contiguous byte span in foreign storage.
@inline function _byte_spans_overlap(left::UInt, left_bytes::UInt,
        right::UInt, right_bytes::UInt)
    (iszero(left_bytes) || iszero(right_bytes)) && return false
    return left <= right ? right - left < left_bytes : left - right < right_bytes
end

@inline _borrowed_bytes(array::BorrowedStorageArray) =
    Base.checked_mul(UInt(sizeof(eltype(array))), UInt(length(array)))

@inline function Base.mightalias(left::BorrowedStorageArray, right::BorrowedStorageArray)
    return _byte_spans_overlap(UInt(left.data), _borrowed_bytes(left),
        UInt(right.data), _borrowed_bytes(right))
end

@inline function Base.mightalias(left::BorrowedStorageArray, right::Array)
    GC.@preserve right begin
        return _byte_spans_overlap(UInt(left.data), _borrowed_bytes(left),
            UInt(pointer(right)), Base.checked_mul(UInt(Base.elsize(right)), UInt(length(right))))
    end
end
@inline Base.mightalias(left::Array, right::BorrowedStorageArray) = Base.mightalias(right, left)

# An arbitrary array need not expose host pointers or contiguous storage. In
# that case retain the conservative answer instead of assuming disjointness.
@inline Base.mightalias(left::BorrowedStorageArray, right::AbstractArray) =
    !isempty(left) && !isempty(right)
@inline Base.mightalias(left::AbstractArray, right::BorrowedStorageArray) =
    !isempty(left) && !isempty(right)

# UnsafeArrays also specializes both AbstractArray operand positions.
@inline function Base.mightalias(left::BorrowedStorageArray, right::UnsafeArray)
    return _byte_spans_overlap(UInt(left.data), _borrowed_bytes(left),
        UInt(pointer(right)), Base.checked_mul(UInt(sizeof(eltype(right))), UInt(length(right))))
end
@inline Base.mightalias(left::UnsafeArray, right::BorrowedStorageArray) =
    Base.mightalias(right, left)
@inline Base.mightalias(left::BorrowedStorageArray, right::SubArray{T,N,<:UnsafeArray}) where {T,N} =
    !isempty(right) && Base.mightalias(left, parent(right))
@inline Base.mightalias(left::SubArray{T,N,<:UnsafeArray}, right::BorrowedStorageArray) where {T,N} =
    Base.mightalias(right, left)

function BorrowedArray(
    data::Ptr{T},
    shape::NTuple{N,Int},
    ::Val{Writable},
) where {T,N,Writable}
    isbitstype(T) || throw(ArgumentError("borrowed array elements must be bitstypes"))
    data == C_NULL && throw(ArgumentError("borrowed array pointer is null"))
    iszero(UInt(data) % UInt(Base.datatype_alignment(T))) || throw(
        ArgumentError("borrowed array pointer is not aligned for $T"),
    )
    elements = 1
    for dimension in shape
        dimension > 0 || throw(ArgumentError("borrowed array dimensions must be positive"))
        elements = Base.checked_mul(elements, dimension)
    end
    return BorrowedArray{T,N,Writable}(data, shape, elements)
end

Base.size(array::BorrowedArray) = array.shape
Base.axes(array::BorrowedArray) = map(Base.OneTo, array.shape)
Base.length(array::BorrowedArray) = array.length
Base.IndexStyle(::Type{<:BorrowedArray}) = IndexLinear()

Base.size(array::RowMajorBorrowedArray) = array.shape
Base.axes(array::RowMajorBorrowedArray) = map(Base.OneTo, array.shape)
Base.length(array::RowMajorBorrowedArray) = array.length
Base.IndexStyle(::Type{<:RowMajorBorrowedArray}) = IndexCartesian()

@inline function Base.getindex(array::BorrowedArray{T}, index::Int) where {T}
    @boundscheck checkbounds(array, index)
    return unsafe_load(array.data, index)
end

@inline function _row_major_index(
    shape::NTuple{N,Int},
    indices::NTuple{N,Int},
) where {N}
    offset = 0
    stride = 1
    for dimension in N:-1:1
        offset += (indices[dimension] - 1) * stride
        stride *= shape[dimension]
    end
    return offset + 1
end

@inline function Base.getindex(
    array::RowMajorBorrowedArray{T,N},
    indices::Vararg{Int,N},
) where {T,N}
    @boundscheck checkbounds(array, indices...)
    return unsafe_load(array.data, _row_major_index(array.shape, indices))
end

@inline function Base.setindex!(
    array::BorrowedArray{T,N,true},
    value,
    index::Int,
) where {T,N}
    @boundscheck checkbounds(array, index)
    unsafe_store!(array.data, convert(T, value), index)
    return value
end

@inline function Base.setindex!(
    array::RowMajorBorrowedArray{T,N,true},
    value,
    indices::Vararg{Int,N},
) where {T,N}
    @boundscheck checkbounds(array, indices...)
    unsafe_store!(
        array.data,
        convert(T, value),
        _row_major_index(array.shape, indices),
    )
    return value
end

# Older compatible FilterGraphAlgorithms releases use their Cartesian selector.
if isdefined(FilterGraphAlgorithms, :_select_pixels!)
    @inline function FilterGraphAlgorithms._select_pixels!(
        output::Vector{Float32}, plan::PixelSelectionPlan, image::RowMajorBorrowedArray{Float32,2},
    )
        # The graph owns the borrowed payload for the duration of this call.
        runs = plan._row_major_runs
        if !isnothing(runs) && length(image) == prod(size(image)) &&
           FilterGraphAlgorithms._can_copy_pixel_selection_runs(output, plan, image, runs)
            GC.@preserve output begin
                return FilterGraphAlgorithms._copy_selected_pixel_runs!(
                    pointer(output), image.data, runs,
                )
            end
        end
        return FilterGraphAlgorithms._select_pixels_cartesian!(output, plan, image)
    end
end

mutable struct LeakyIntegratorOwner{P}
    prepared::P
    extent::Int
    warm_input::Vector{Float32}
    warm_output::Vector{Float32}
    warmed::Bool
    active::Bool
    process_count::UInt64
end

function LeakyIntegratorOwner(prepared)
    extent = only(first(algorithm_formats(prepared)).shape)
    return LeakyIntegratorOwner(
        prepared,
        extent,
        zeros(Float32, extent),
        zeros(Float32, extent),
        false,
        false,
        UInt64(0),
    )
end

function _prepare_owner(owner::LeakyIntegratorOwner)
    if !owner.warmed
        GC.@preserve owner begin
            input = BorrowedArray(pointer(owner.warm_input), (owner.extent,), Val(false))
            output = BorrowedArray(pointer(owner.warm_output), (owner.extent,), Val(true))
            process!(output, owner.prepared, input)
            reset!(owner.prepared)
        end
        owner.warmed = true
    end
    owner.active = true
    return nothing
end

function _deactivate_owner(owner::LeakyIntegratorOwner)
    owner.active = false
    return nothing
end

function _process_owner(
    owner::LeakyIntegratorOwner,
    input_data::Ptr{Float32},
    output_data::Ptr{Float32},
)
    owner.active || throw(InvalidStateException("the Filter Graph node is inactive", :inactive))
    input = BorrowedArray(input_data, (owner.extent,), Val(false))
    output = BorrowedArray(output_data, (owner.extent,), Val(true))
    process!(output, owner.prepared, input)
    owner.process_count += UInt64(1)
    return nothing
end

struct PrepareCallback{Owner}
    owner::Owner
end

function (callback::PrepareCallback)(::NdArrayFilter)
    return _prepare_owner(callback.owner)
end

struct ProcessCallback{Owner}
    owner::Owner
end

function (callback::ProcessCallback)(
    ::NdArrayFilter,
    inputs::NdArrayFilterBuffers{false},
    outputs::NdArrayFilterBuffers{true},
)
    length(inputs) == 1 || throw(DimensionMismatch("expected one ndarray input port"))
    length(outputs) == 1 || throw(DimensionMismatch("expected one ndarray output port"))
    input = inputs[1]
    output = outputs[1]
    expected_size = sizeof(Float32) * callback.owner.extent
    payload_size(input) == expected_size ||
        throw(DimensionMismatch("the input payload size does not match its declaration"))
    payload_size(output) == expected_size ||
        throw(DimensionMismatch("the output payload size does not match its declaration"))
    _process_owner(
        callback.owner,
        Ptr{Float32}(data_pointer(input)),
        Ptr{Float32}(data_pointer(output)),
    )
    propagate_metadata!(output, input)
    return nothing
end

struct DeactivateCallback{Owner}
    owner::Owner
end

function (callback::DeactivateCallback)(::NdArrayFilter)
    return _deactivate_owner(callback.owner)
end

function _element_type(element_type::DataType)
    element_type === Bool && return NdArray.BOOL8
    element_type === Int8 && return NdArray.I8
    element_type === UInt8 && return NdArray.U8
    element_type === Int16 && return NdArray.I16_LE
    element_type === UInt16 && return NdArray.U16_LE
    element_type === Int32 && return NdArray.I32_LE
    element_type === UInt32 && return NdArray.U32_LE
    element_type === Int64 && return NdArray.I64_LE
    element_type === UInt64 && return NdArray.U64_LE
    element_type === Float32 && return NdArray.F32_LE
    element_type === Float64 && return NdArray.F64_LE
    throw(ArgumentError("unsupported PipeWire ndarray element type $element_type"))
end

function _layout(layout::ArrayLayout)
    layout isa RowMajorLayout && return NdArray.ROW_MAJOR
    layout isa ColumnMajorLayout && return NdArray.COLUMN_MAJOR
    throw(ArgumentError("unsupported PipeWire ndarray layout $(typeof(layout))"))
end

function _boundary_layout(
    format::AlgorithmFormat,
    boundary_layout::Union{Nothing,ArrayLayout},
)
    isnothing(boundary_layout) && return _layout(format.layout)
    return _layout(boundary_layout)
end

function _rate_terms(rate)
    numerator, denominator = if rate isa Rational
        (Base.numerator(rate), Base.denominator(rate))
    elseif rate isa Tuple && length(rate) == 2
        rate
    else
        throw(ArgumentError("rate must be nothing, a Rational, or (numerator, denominator)"))
    end
    numerator isa Integer && denominator isa Integer || throw(
        ArgumentError("rate terms must be integers"),
    )
    (numerator isa Bool || denominator isa Bool) && throw(
        ArgumentError("rate terms must be integers other than Bool"),
    )
    numerator > 0 && denominator > 0 || throw(ArgumentError("rate must be positive"))
    divisor = gcd(numerator, denominator)
    numerator = div(numerator, divisor)
    denominator = div(denominator, divisor)
    numerator <= typemax(UInt32) && denominator <= typemax(UInt32) ||
        throw(ArgumentError("reduced rate must fit UInt32"))
    return (UInt32(numerator), UInt32(denominator))
end

function _port_rate(format::AlgorithmFormat, base_rate)
    ratio_numerator, ratio_denominator = format.rate
    if isnothing(base_rate)
        (ratio_numerator, ratio_denominator) == (UInt32(1), UInt32(1)) || throw(
            ArgumentError(
                "a deployment base rate is required for non-unit Algorithm Port rates",
            ),
        )
        return nothing
    end
    base_numerator, base_denominator = _rate_terms(base_rate)
    numerator = Base.checked_mul(UInt64(base_numerator), UInt64(ratio_numerator))
    denominator = Base.checked_mul(UInt64(base_denominator), UInt64(ratio_denominator))
    divisor = gcd(numerator, denominator)
    numerator = div(numerator, divisor)
    denominator = div(denominator, divisor)
    numerator <= typemax(UInt32) && denominator <= typemax(UInt32) || throw(
        ArgumentError("projected Algorithm Port rate must fit UInt32"),
    )
    return SPA.Fraction(UInt32(numerator), UInt32(denominator))
end

function _ndarray_format(
    format::AlgorithmFormat,
    base_rate;
    boundary_layout::Union{Nothing,ArrayLayout}=nothing,
)
    return NdArrayFormat(
        _element_type(format.element_type),
        format.shape;
        layout=_boundary_layout(format, boundary_layout),
        rate=_port_rate(format, base_rate),
    )
end

function _ports(prepared, rate)
    declarations = algorithm_ports(LeakyIntegratorF32)
    formats = algorithm_formats(prepared)
    length(declarations) == length(formats) || error("declaration format count changed")
    ports = NdArrayFilterPort[]
    sizehint!(ports, length(formats))
    for (declaration, format) in zip(declarations, formats)
        declaration.role isa FilterGraphAlgorithms.FrameData || throw(
            ArgumentError("the Julia client proof supports frame-data ports only"),
        )
        direction = declaration.direction isa FilterGraphAlgorithms.InputPort ?
            PipeWireAO.DIRECTION_INPUT : PipeWireAO.DIRECTION_OUTPUT
        ndarray_format = _ndarray_format(format, rate)
        push!(
            ports,
            NdArrayFilterPort(
                String(declaration.name),
                direction,
                ndarray_format;
                schema=format.schema,
            ),
        )
    end
    return ports
end

"""
One externally published PipeWire Node backed by a rooted Algorithm instance.

The current implementation accepts a prepared `LeakyIntegratorF32`. All native
calls and callback ABI details are owned by `PipeWireAO`; this wrapper contains
only Algorithm-to-ndarray declaration and processing adaptation.
"""
mutable struct PipeWireNode{Filter<:NdArrayFilter,Owner}
    filter::Filter
    owner::Owner
end

function PipeWireNode(
    prepared::FilterGraphAlgorithms.PreparedAlgorithm{LeakyIntegratorF32};
    name::AbstractString=algorithm_label(LeakyIntegratorF32),
    remote::Union{Nothing,AbstractString}=nothing,
    rate=nothing,
    connect::Bool=true,
    run_control::Bool=false,
)
    owner = LeakyIntegratorOwner(prepared)
    run_control_options = run_control ? (; run_control=true) : (;)
    filter = NdArrayFilter(
        name,
        _ports(prepared, rate);
        remote,
        on_prepare=PrepareCallback(owner),
        on_process=ProcessCallback(owner),
        on_deactivate=DeactivateCallback(owner),
        run_control_options...,
    )
    node = PipeWireNode(filter, owner)
    if connect
        try
            connect!(node)
        catch
            close(node)
            rethrow()
        end
    end
    return node
end

"""
    connect!(node::PipeWireNode)

Connect a constructed Filter Graph Node to its configured PipeWire remote.
"""
function connect!(node::PipeWireNode)
    connect!(node.filter)
    return node
end

"Run a Filter Graph Node until another task or thread calls [`quit!`](@ref)."
function run!(node::PipeWireNode)
    run!(node.filter)
    return node
end

Base.run(node::PipeWireNode) = run!(node)

"Request termination of a running Filter Graph Node's main loop."
function quit!(node::PipeWireNode)
    quit!(node.filter)
    return node
end

"""
    command_endpoint(node::PipeWireNode, callback, ::Type{T}; capacity=1)

Create a bounded command receiver on the node's constructing thread before
`run!`. `callback(message)` executes on that thread's native main loop. Send
through `PipeWireAO.commands(receiver)`; the sender retains transport state only.
Close the receiver on its owning thread after `run!` returns, before closing the
node. Queue statuses and callback failures follow PipeWireAO's command contract.
This control endpoint does not execute commands in the frame-processing callback.
Requires a native SDK exporting `pw_ndarray_filter_get_loop`.
"""
function command_endpoint(node::PipeWireNode, callback, ::Type{T}; capacity=1) where {T}
    return command_endpoint(node.filter, callback, T; capacity)
end

"""
    node_state(node::PipeWireNode)

Return the current standalone ndarray-filter lifecycle state.
"""
node_state(node::PipeWireNode) = filter_state(node.filter)

"""
    last_error(node::PipeWireNode)

Return the first asynchronous native error, or `nothing`.
"""
last_error(node::PipeWireNode) = last_error(node.filter)

"""
    node_id(node::PipeWireNode)

Return the published PipeWire node ID, or `nothing` before registration.
"""
node_id(node::PipeWireNode) = node_id(node.filter)

Base.isopen(node::PipeWireNode) = isopen(node.filter)
PipeWireAO.isrunning(node::PipeWireNode) = isrunning(node.filter)
Base.close(node::PipeWireNode) = close(node.filter)

include("callback_metrics.jl")
include("source_channels.jl")
include("frame_observer.jl")
include("metadata_records.jl")
include("graph_node.jl")
include("gpu_readout_node.jl")
include("metadata_requirements.jl")
export required_input_metadata
include("composition.jl")
include("recovery.jl")
include("recovery_scope.jl")
include("composition_recovery.jl")
include("preparation.jl")
include("row_major_calibration.jl")

export PreparedGraphRecovery
export prepare_recovery
export stage_properties!
export stage_recovery!
export recovery_status
export stage_recovery_release!
export PipeWireNode
export command_endpoint
export PreparedCapturedGraphReplacement
export connect!
export last_error
export node_id
export node_state
export prepare!
export prepare_row_major_calibration
export prepare_replacement
export quit!
export replace_graph!
export replace_parameters!
export run!

# Standalone application definitions are inert until main/run_service is called.
include("application/cli.jl")
include("application/algorithms.jl")
include("application/callback_metrics.jl")
include("application/startup.jl")
include("application/runtime.jl")

public parse_arguments, algorithm_declarations, configure_threads,
       prepare_service_graph, close_execution, enable_callback_metrics!, main

"""
    main(args)

Run a prepared graph as one PipeWire ndarray node. Invoke with
`julia --project=ENV -m FilterGraphPipeWire [options]`. Importing the package
only defines the library and application API; startup is explicit.
"""
function (@main)(args)
    run_service(parse_arguments(args))
    return 0
end

end
