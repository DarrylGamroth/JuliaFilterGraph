using FilterGraphAlgorithms
using FilterGraphPipeWire
using FilterGraphs
using PipeWireAO
using Test
using ForwardDiff

include("borrowed_alias.jl")
include("scc_correction.jl")
include("source_channels.jl")

function prepared_leaky(extent=277)
    return prepare_algorithm(
        LeakyIntegratorF32,
        LeakyIntegratorF32Configuration(
            extent=extent,
            initial_state=0.0f0,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_COMMAND_V1,
        ),
    )
end

const LEAKY_GRAPH_FIXTURE = joinpath(@__DIR__, "data", "leaky-graph.conf")
const PIXEL_CALIBRATION_GRAPH_FIXTURE = joinpath(@__DIR__, "data", "pixel-calibration-row.conf")
const CONTROL_GRAPH_FIXTURE = joinpath(@__DIR__, "data", "closed-loop-correction.conf")

prepared_leaky_graph() = prepare_graph(
    LEAKY_GRAPH_FIXTURE;
    algorithms=(LeakyIntegratorF32,),
)

prepared_pixel_calibration_graph() = prepare_graph(
    PIXEL_CALIBRATION_GRAPH_FIXTURE;
    algorithms=(PixelCalibrationRowU16F32,),
)

prepared_control_graph() = prepare_graph(
    CONTROL_GRAPH_FIXTURE;
    algorithms=algorithms(),
)

function prepare_graph_text(text; algorithms)
    return mktemp() do path, io
        write(io, text)
        flush(io)
        return prepare_graph(path; algorithms)
    end
end

@algorithm AdapterConditionalF32 begin
    label = "adapter-conditional-f32"

    @configuration AdapterConditionalF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            sample=:sample,
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
            publication=:conditional,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        (isnothing(sample) || !sample.terminal) && return AlgorithmProcessDeferred()
        copyto!(output, input)
        return nothing
    end
end

@algorithm AdapterMetadataSourceF32 begin
    label = "adapter-metadata-source-f32"

    @configuration AdapterMetadataSourceF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:left, Float32, (configuration.extent,), "test.signal.f32/1"),
        input(:right, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:right,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @. output = left + right
        nothing
    end
end

@enum AdapterFailure::UInt8 begin
    AdapterRejected = 1
end

@algorithm AdapterFailureF32 begin
    label = "adapter-failure-f32"
    failure = AdapterFailure

    @configuration AdapterFailureF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), CONTROLLER_COMMAND_V1),
        output(:output, Float32, (configuration.extent,), CONTROLLER_COMMAND_V1),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        input[1] < 0.0f0 && return AdapterRejected
        copyto!(output, input)
        nothing
    end
end

const ADAPTER_HETEROGENEOUS_PROPERTY_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = integrate
            label = leaky-integrator-f32
            config = {
                extent = 2
                initial_state = 0.0
                input_schema = pipewireao/controller-residual-error-f32/1
                output_schema = pipewireao/controller-command-f32/1
            }
            props = { gain = 2.0 pole = 0.5 }
        }
        {
            type = ndarray
            name = failure
            label = adapter-failure-f32
            config = { extent = 2 }
        }
    ]
    links = [ { output = "integrate:output" input = "failure:input" } ]
    inputs = [ "integrate:input" ]
    outputs = [ "failure:output" ]
}
"""

@algorithm AdapterLatestHoldAddF32 begin
    label = "adapter-latest-hold-add-f32"

    @configuration AdapterLatestHoldAddF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :primary,
            Float32,
            (configuration.extent,),
            "test.primary.f32/1";
            rate=(2, 1),
        ),
        input(
            :held,
            Float32,
            (configuration.extent,),
            "test.held.f32/1";
            rate=(1, 1),
        ),
        output(
            :joined,
            Float32,
            (configuration.extent,),
            "test.joined.f32/1";
            metadata_from=:held,
            rate=(2, 1),
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @inbounds @simd for index in eachindex(joined, primary, held)
            joined[index] = primary[index] + held[index]
        end
        nothing
    end
end

@algorithm AdapterSynchronizedSumF32 begin
    label = "adapter-synchronized-sum-f32"

    @configuration AdapterSynchronizedSumF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:a, Float32, (configuration.extent,), "test.signal.f32/1"),
        input(:b, Float32, (configuration.extent,), "test.signal.f32/1"),
        input(:c, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :sum,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:a,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @inbounds @simd for index in eachindex(sum, a, b, c)
            sum[index] = a[index] + b[index] + c[index]
        end
        nothing
    end
end

@algorithm AdapterSynchronizedMatrixSumF32 begin
    label = "adapter-synchronized-matrix-sum-f32"

    @configuration AdapterSynchronizedMatrixSumF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:a, Float32, (configuration.extent, 3), "test.signal.f32/1"),
        input(:b, Float32, (configuration.extent, 3), "test.signal.f32/1"),
        input(:c, Float32, (configuration.extent, 3), "test.signal.f32/1"),
        output(
            :sum,
            Float32,
            (configuration.extent, 3),
            "test.signal.f32/1";
            metadata_from=:a,
        ),
    )

    prepare(configuration) = configuration.extent
    process!(plan) = begin
        @inbounds @simd for index in eachindex(sum, a, b, c)
            sum[index] = a[index] + b[index] + c[index]
        end
        nothing
    end
end

const ADAPTER_CONDITIONAL_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = conditional
            label = adapter-conditional-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "conditional:input" ]
    outputs = [ "conditional:output" ]
}
"""

const ADAPTER_FAILURE_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = failure
            label = adapter-failure-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "failure:input" ]
    outputs = [ "failure:output" ]
}
"""

const ADAPTER_METADATA_SOURCE_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = metadata
            label = adapter-metadata-source-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "metadata:left", "metadata:right" ]
    outputs = [ "metadata:output" ]
}
"""

const ADAPTER_LATEST_HOLD_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = add
            label = adapter-latest-hold-add-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "add:primary" "add:held" ]
    outputs = [ "add:joined" ]
}
"""

const ADAPTER_SYNCHRONIZED_GRAPH = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = add
            label = adapter-synchronized-sum-f32
            config = { extent = 2 }
        }
    ]
    links = []
    inputs = [ "add:a" "add:b" "add:c" ]
    outputs = [ "add:sum" ]
}
"""

prepared_conditional_graph() = prepare_graph_text(
    ADAPTER_CONDITIONAL_GRAPH;
    algorithms=(AdapterConditionalF32,),
)

prepared_failure_graph() = prepare_graph_text(
    ADAPTER_FAILURE_GRAPH;
    algorithms=(AdapterFailureF32,),
)

prepared_metadata_source_graph() = prepare_graph_text(
    ADAPTER_METADATA_SOURCE_GRAPH;
    algorithms=(AdapterMetadataSourceF32,),
)

prepared_heterogeneous_property_graph() = prepare_graph_text(
    ADAPTER_HETEROGENEOUS_PROPERTY_GRAPH;
    algorithms=(LeakyIntegratorF32, AdapterFailureF32),
)

prepared_latest_hold_graph() = prepare_graph_text(
    ADAPTER_LATEST_HOLD_GRAPH;
    algorithms=(AdapterLatestHoldAddF32,),
)

prepared_synchronized_graph() = prepare_graph_text(
    ADAPTER_SYNCHRONIZED_GRAPH;
    algorithms=(AdapterSynchronizedSumF32,),
)

# Callback skeletons and their independent metadata owners share one lease.
struct NativeCallbackStorage{T}
    records::Vector{T}
    metadata::Vector{MetadataSnapshot}
end
Base.pointer(storage::NativeCallbackStorage,index::Int=1)=pointer(storage.records,index)
Base.length(storage::NativeCallbackStorage)=length(storage.records)
Base.eachindex(storage::NativeCallbackStorage)=eachindex(storage.records)
Base.getindex(storage::NativeCallbackStorage,index::Int)=storage.records[index]
Base.vcat(storages::NativeCallbackStorage...)=NativeCallbackStorage(vcat(map(s -> s.records,storages)...),vcat(map(s -> s.metadata,storages)...))
function callback_field(storage::NativeCallbackStorage,::Val{Field},index::Int=1) where {Field}
    T=eltype(storage.records); field=findfirst(==(Field),fieldnames(T))
    return Ptr{fieldtype(T,field)}(Ptr{UInt8}(pointer(storage,index))+fieldoffset(T,field))
end
function native_callback_buffer(data::Vector;
        header::Union{Nothing,BufferHeader}=nothing,header_available::Bool=true,
        acquisition_sequence::Union{Nothing,UInt64}=nothing,properties_available::Bool=true,
        trigger_state::Union{Nothing,AcquisitionTriggerState}=nothing,
        optical_classification::Union{Nothing,OpticalClassification}=nothing,
        extra_property::Union{Nothing,Int64}=nothing,
        exposure_duration_nanoseconds::Union{Nothing,UInt64}=nothing,
        exposure_start::Union{Nothing,ExposureTiming}=nothing,present::Bool=true)
    native=PipeWireAO.LibPipeWire
    requirements=((header!==nothing || header_available ? (MetadataRequirement(SPA.META_HEADER),) : ())...,MetadataRequirement(SPA.META_ACQUISITION_IDENTITY),
        MetadataRequirement(SPA.META_EXPOSURE_TIMING),MetadataRequirement(SPA.META_REFERENCE_TIMESTAMP),
        MetadataRequirement(SPA.META_OPTICAL_CHOPPING),MetadataRequirement(SPA.META_ELECTRICAL_TRIGGER),MetadataRequirement(0x900,8))
    snapshot=MetadataSnapshot(requirements)
    writable=writable_metadata_records(snapshot)
    header===nothing || set_buffer_header!(writable,header)
    acquisition_sequence===nothing || set_acquisition_identity!(writable,AcquisitionIdentity(AcquisitionDomain(ntuple(UInt8,16)),7,acquisition_sequence))
    trigger_state===nothing || set_trigger_classification!(writable,trigger_state)
    optical_classification===nothing || set_optical_classification!(writable,optical_classification)
    if exposure_start!==nothing
        set_exposure_timing!(writable,exposure_start)
    elseif exposure_duration_nanoseconds!==nothing
        set_exposure_timing!(writable,ExposureTiming(;duration_nanoseconds=exposure_duration_nanoseconds))
    end
    extra_property===nothing || set_metadata_bytes!(writable,UInt32(0x900),Tuple(reinterpret(UInt8,[extra_property])))
    extent=sizeof(eltype(data))*length(data)
    buffers=[native.pw_ndarray_filter_buffer(UInt32(sizeof(native.pw_ndarray_filter_buffer)),
        present ? UInt32(0) : native.PW_NDARRAY_FILTER_BUFFER_FLAG_INPUT_UNAVAILABLE,
        present ? Ptr{Cvoid}(pointer(data)) : C_NULL,present ? UInt32(extent) : UInt32(0),
        present ? UInt32(extent) : UInt32(0),present ? snapshot.count : UInt32(0),0,
        present ? pointer(snapshot.descriptors) : C_NULL)]
    return NativeCallbackStorage(buffers,[snapshot])
end
function set_native_callback_duration!(storage,duration::UInt64,index::Int=1)
    records=writable_metadata_records(storage.metadata[index])
    current=exposure_timing(records)
    value=current===nothing || !current.has_start ? ExposureTiming(;duration_nanoseconds=duration) :
        ExposureTiming(current.start_nanoseconds;clock=current.clock,duration_nanoseconds=duration)
    set_exposure_timing!(records,value)
    return nothing
end
set_native_callback_flags!(storage,flags::UInt32,index::Int=1)=(unsafe_store!(callback_field(storage,Val(:flags),index),flags);storage)
native_callback_flags(storage,index::Int=1)=unsafe_load(callback_field(storage,Val(:flags),index))
native_callback_metadata_valid(storage,index::Int=1)=UInt32(buffer_header(storage.metadata[index])!==nothing && iszero(native_callback_flags(storage,index)&PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE))
native_callback_header(storage,index::Int=1)=buffer_header(storage.metadata[index])
function native_callback_acquisition(storage,index::Int=1)
    records=storage.metadata[index]
    timing=exposure_timing(records)
    return (identity=acquisition_identity(records),duration=timing===nothing || !timing.has_duration ? nothing : timing.duration_nanoseconds,
        exposure_start=timing===nothing || !timing.has_start ? nothing : timing,
        trigger=trigger_classification(records),optical=optical_classification(records))
end

function native_callback_custom_value(storage,index::Int=1)
    record=metadata_record(storage.metadata[index],UInt32(0x900))
    record===nothing && return nothing
    raw=metadata_bytes(record)
    value=Int64(sum(UInt64(raw[i]) << (8*(i-1)) for i in eachindex(raw)))
    return iszero(value) ? nothing : value
end

@testset "FGN Header metadata mapping is transport-neutral" begin
    @test !FilterGraphPipeWire._buffer_metadata(nothing).present

    header = BufferHeader(
        SPA.META_HEADER_FLAG_MARKER |
        SPA.META_HEADER_FLAG_DISCONT |
        SPA.META_HEADER_FLAG_CORRUPTED |
        SPA.META_HEADER_FLAG_HEADER,
        UInt32(17),
        typemin(Int64),
        Int64(-11),
        UInt64(93),
    )
    admitted = FilterGraphPipeWire._buffer_metadata(header)
    @test admitted.present
    @test admitted.value == SampleMetadata(
        UInt64(93),
        UInt32(17);
        terminal=true,
        discontinuity=true,
        corrupted=true,
    )
    @test FilterGraphPipeWire._buffer_header(admitted.value) == BufferHeader(
        SPA.META_HEADER_FLAG_MARKER |
        SPA.META_HEADER_FLAG_DISCONT |
        SPA.META_HEADER_FLAG_CORRUPTED,
        UInt32(17),
        typemin(Int64),
        Int64(0),
        UInt64(93),
    )
end

@testset "acquisition projection owns qualified exposure metadata" begin
    data = Float32[1, 2]
    header = BufferHeader(UInt32(0), UInt32(0), Int64(5000), Int64(0), UInt64(11))
    for (native_state, sample_state, pure) in (
        (ACQUISITION_TRIGGER_ABSENT, SAMPLE_TRIGGER_ABSENT, nothing),
        (ACQUISITION_TRIGGER_LOW, SAMPLE_TRIGGER_LOW, false),
        (ACQUISITION_TRIGGER_HIGH, SAMPLE_TRIGGER_HIGH, true),
        (ACQUISITION_TRIGGER_MIXED, SAMPLE_TRIGGER_MIXED, nothing),
    )
        input = native_callback_buffer(data; header, acquisition_sequence=UInt64(11), trigger_state=native_state,
            exposure_duration_nanoseconds=UInt64(900),
            exposure_start=ExposureTiming(1234;clock=METADATA_CLOCK_TAI,duration_nanoseconds=900))
        output = native_callback_buffer(data; header_available=true, properties_available=true)
        GC.@preserve data input output begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input)), UInt32(1))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(output)), UInt32(1))
            admitted = @inferred FilterGraphPipeWire._buffer_metadata(inputs[1])
            sample = admitted.value
            @test sample.timestamp_nanoseconds == 5000
            @test sample_trigger_state(sample) == sample_state
            @test sample_trigger_high(sample) === pure
            @test sample_exposure_duration_nanoseconds(sample) === UInt64(900)
            @test sample_acquisition_identity(sample) == SampleAcquisitionIdentity(
                ntuple(i -> UInt8(i), 16), 7, 11)
            @test sample_exposure_start(sample) == SampleExposureStart(1234;
                clock=SAMPLE_CLOCK_TAI)
            FilterGraphPipeWire._set_output_metadata!(outputs[1], admitted)
            @test native_callback_acquisition(output) == native_callback_acquisition(input)
            # The retained sample owns values after its borrowed native input is changed.
            set_native_callback_duration!(input, UInt64(1))
            @test sample_exposure_duration_nanoseconds(sample) === UInt64(900)
            FilterGraphPipeWire._set_output_metadata!(outputs[1], FilterGraphAlgorithms._NO_SAMPLE_METADATA)
            @test acquisition_identity(metadata_records(outputs[1])) === nothing
        end
    end
    for properties_available in (false, true)
        input = native_callback_buffer(data; header, properties_available)
        GC.@preserve data input begin
            buffers = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(input)), UInt32(1))
            sample = FilterGraphPipeWire._buffer_metadata(buffers[1]).value
            @test sample_trigger_state(sample) == SAMPLE_TRIGGER_ABSENT
            @test sample_exposure_duration_nanoseconds(sample) === nothing
            @test sample_exposure_start(sample) === nothing
        end
    end
    held = native_callback_buffer(data; header, acquisition_sequence=UInt64(7))
    GC.@preserve data held begin
        buffer = NdArrayFilterBuffer{false}(Ptr{Cvoid}(pointer(held)))
        sample = FilterGraphPipeWire._buffer_metadata(buffer).value
        @test sample.sequence == 11
        @test sample_acquisition_identity(sample).sequence == 7
    end
    graph = prepared_leaky_graph()
    @test all(port -> isempty(port.metadata),
        FilterGraphPipeWire._graph_ports(graph, nothing))
end

@testset "local optical interpretation transports independent facts" begin
    data = Float32[1, 2]
    header = BufferHeader(UInt32(0), UInt32(0), Int64(5000), Int64(0), UInt64(11))
    source = native_callback_buffer(data; header, acquisition_sequence=UInt64(7),
        exposure_duration_nanoseconds=UInt64(900))
    destination = native_callback_buffer(data; header_available=true, properties_available=true)
    GC.@preserve data source destination begin
        inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(source)), UInt32(1))
        outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(destination)), UInt32(1))
        admitted = FilterGraphPipeWire._buffer_metadata(inputs[1])
        tagged = SampleMetadata(admitted.value; trigger_high=true)
        optical = chopped_image_metadata(tagged; trigger_high_is_fringed=false)
        FilterGraphPipeWire._set_output_metadata!(outputs[1],
            FilterGraphAlgorithms.AdmittedSampleMetadata(true, optical))
        transported = FilterGraphPipeWire._buffer_metadata(outputs[1]).value
        @test sample_acquisition_identity(transported) == sample_acquisition_identity(optical)
        @test sample_exposure_duration_nanoseconds(transported) == 900
        @test sample_trigger_high(transported) === true
        @test sample_chopped_image_kind(transported) === ChoppedImageUnfringed
        @test transported.sequence == 11
        @test sample_acquisition_identity(transported).sequence == 7
    end
end

@testset "joined callbacks retain complete acquisition records after ingress" begin
    data = Float32[1, 2]
    output = zeros(Float32, 2)
    for kind in (:latest_hold, :synchronized)
        graph = kind == :latest_hold ? prepared_latest_hold_graph() : prepared_synchronized_graph()
        join = kind == :latest_hold ? FilterGraphs.prepare_latest_hold(graph, :primary, :held) :
            FilterGraphs.prepare_synchronized_inputs(graph)
        owner = FilterGraphPipeWire.PreparedGraphOwner(join)
        FilterGraphPipeWire._prepare_owner(owner)
        count = kind == :latest_hold ? 2 : 3
        source_index = kind == :latest_hold ? 2 : 1
        header = BufferHeader(UInt32(0), UInt32(0), Int64(5000), Int64(0), UInt64(11))
        source = native_callback_buffer(data; header, acquisition_sequence=UInt64(11), trigger_state=ACQUISITION_TRIGGER_MIXED,
            exposure_duration_nanoseconds=UInt64(900),
            exposure_start=ExposureTiming(1234;clock=METADATA_CLOCK_TAI,duration_nanoseconds=900))
        absent = native_callback_buffer(data; present=false)
        ordinary = native_callback_buffer(data; header)
        first_arrival = vcat(ntuple(i -> i == source_index ? source : absent, count)...)
        later_arrival = vcat(ntuple(i -> i == source_index ? absent : ordinary, count)...)
        destination = native_callback_buffer(output; header_available=true, properties_available=true)
        expected = native_callback_acquisition(source)
        GC.@preserve data output source absent ordinary first_arrival later_arrival destination begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(first_arrival)), UInt32(count))
            outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(destination)), UInt32(1))
            FilterGraphPipeWire._process_graph_buffers!(owner, join, inputs, outputs)
            @test !iszero(native_callback_flags(destination) &
                PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE)
            set_native_callback_duration!(first_arrival, UInt64(1), source_index)
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(later_arrival)), UInt32(count))
            FilterGraphPipeWire._process_graph_buffers!(owner, join, inputs, outputs)
            @test output == count .* data
            @test native_callback_acquisition(destination) == expected
            @test native_callback_header(destination).sequence == 11
            @test iszero(native_callback_flags(destination) &
                PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE)
        end
    end
end

@testset "borrowed arrays preserve rank, axes, and direction ownership" begin
    storage = Float32[1, 2, 3, 4]
    GC.@preserve storage begin
        input = FilterGraphPipeWire.BorrowedArray(pointer(storage), (2, 2), Val(false))
        output = FilterGraphPipeWire.BorrowedArray(pointer(storage), (2, 2), Val(true))
        @test size(input) == (2, 2)
        @test axes(input) == (Base.OneTo(2), Base.OneTo(2))
        @test collect(input) == reshape(storage, 2, 2)
        @test_throws Base.CanonicalIndexError input[2, 1] = 9.0f0
        output[2, 1] = 8.0f0
        @test storage[2] == 8.0f0
        @test @allocated(input[1, 1] + output[2, 1]) == 0
    end
    @test_throws ArgumentError FilterGraphPipeWire.BorrowedArray(
        Ptr{Float32}(C_NULL),
        (4,),
        Val(false),
    )
    GC.@preserve storage begin
        @test_throws ArgumentError FilterGraphPipeWire.BorrowedArray(
            pointer(storage),
            (0,),
            Val(false),
        )
    end
end

@testset "borrowed-buffer processing matches direct execution" begin
    extent = 277
    prepared = prepared_leaky(extent)
    owner = FilterGraphPipeWire.LeakyIntegratorOwner(prepared)
    input = collect(Float32, 1:extent)
    output = zeros(Float32, extent)
    expected = similar(output)
    expected_prepared = prepared_leaky(extent)
    FilterGraphAlgorithms.process!(expected, expected_prepared, input)

    FilterGraphPipeWire._prepare_owner(owner)
    GC.@preserve input output begin
        FilterGraphPipeWire._process_owner(owner, pointer(input), pointer(output))
        @test output == expected
        @test owner.process_count == 1

        fill!(output, 0.0f0)
        FilterGraphPipeWire._process_owner(owner, pointer(input), pointer(output))
        @test @allocated(
            FilterGraphPipeWire._process_owner(owner, pointer(input), pointer(output))
        ) == 0
    end
    @test owner.process_count == 3
    FilterGraphPipeWire._deactivate_owner(owner)
    @test_throws InvalidStateException GC.@preserve input output begin
        FilterGraphPipeWire._process_owner(owner, pointer(input), pointer(output))
    end
end

@testset "native endpoint construction avoids direct C bindings" begin
    source_directory = joinpath(@__DIR__, "..", "src")
    source = join(
        read(joinpath(source_directory, file), String) for
        file in readdir(source_directory) if endswith(file, ".jl")
    )
    # A generated buffer type is named in the native callback precompile
    # signature; the package still makes no direct C binding calls.
    @test !occursin(r"LibPipeWire\.[A-Za-z_][A-Za-z_0-9]*\s*\(", source)
    @test !occursin("@ccall", source)
    @test !occursin(r"\bccall\s*\(", source)

    prepared = prepared_leaky(8)
    node = PipeWireNode(prepared; name="test.filter-graph.julia", connect=false)
    @test isopen(node)
    @test !isrunning(node)
    @test node_state(node) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
    @test last_error(node) === nothing
    @test node_id(node) === nothing
    close(node)
    @test !isopen(node)

    graph_node = PipeWireNode(
        prepared_leaky_graph();
        name="test.filter-graph.controls",
        connect=false,
        run_control=true,
    )
    callbacks = graph_node.filter.callbacks
    @test getproperty.(callbacks.property_info, :name) == [
        "integrate:gain",
        "integrate:pole",
        "integrate:closed",
        "integrate:requested-generation",
        "integrate:active-generation",
        "integrate:requested-parameter-sequence",
        "integrate:active-parameter-sequence",
    ]
    @test getproperty.(callbacks.property_info, :group) == [
        "integrate/Controls",
        "integrate/Controls",
        "integrate/Controls",
        "integrate/Status",
        "integrate/Status",
        "integrate/Status",
        "integrate/Status",
    ]
    @test callbacks.on_get_properties !== nothing
    @test callbacks.on_properties !== nothing
    @test callbacks.on_reset !== nothing
    @test callbacks.on_get_properties(graph_node.filter) isa SPA.Props
    @test callbacks.on_reset(graph_node.filter) === graph_node.owner.graph
    close(graph_node)
end

@testset "Graph scalar properties use the FGN-qualified control surface" begin
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_leaky_graph())
    infos = FilterGraphPipeWire._graph_property_info(owner)
    @test getproperty.(infos, :name) == [
        "integrate:gain",
        "integrate:pole",
        "integrate:closed",
        "integrate:requested-generation",
        "integrate:active-generation",
        "integrate:requested-parameter-sequence",
        "integrate:active-parameter-sequence",
    ]
    @test getproperty.(infos, :display_name) == [
        "gain",
        "pole",
        "closed",
        "requested-generation",
        "active-generation",
        "requested-parameter-sequence",
        "active-parameter-sequence",
    ]
    @test getproperty.(infos, :group) == [
        "integrate/Controls",
        "integrate/Controls",
        "integrate/Controls",
        "integrate/Status",
        "integrate/Status",
        "integrate/Status",
        "integrate/Status",
    ]
    @test all(info -> info.params, infos)
    @test infos[1].flags == UInt32(0)
    @test infos[4].flags == SPA.PROPERTY_READONLY
    for info in infos
        wire_info = SPA.PropInfo(Pod(prop_info_param(info)))
        @test wire_info.name == info.name
        @test wire_info.group == info.group
        @test wire_info.display_name == info.display_name
        @test wire_info.description == info.description
        @test wire_info.type == info.type
        @test wire_info.params == info.params
        @test wire_info.flags == info.flags
    end
    heterogeneous_owner = FilterGraphPipeWire.PreparedGraphOwner(
        prepared_heterogeneous_property_graph(),
    )
    @test Set(getproperty.(FilterGraphPipeWire._graph_property_info(heterogeneous_owner), :group)) ==
          Set(("integrate/Controls", "integrate/Status", "failure/Status"))

    initial = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Float32, initial.values[1].second) == 2.0f0
    @test PipeWireAO.pod_value(Int64, initial.values[4].second) == 0
    @test PipeWireAO.pod_value(Int64, initial.values[5].second) == 0
    @test PipeWireAO.pod_value(Int64, initial.values[6].second) == 0
    @test PipeWireAO.pod_value(Int64, initial.values[7].second) == 0

    FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:gain" => 3.0f0),
    )
    requested = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Float32, requested.values[1].second) == 2.0f0
    @test PipeWireAO.pod_value(Int64, requested.values[4].second) == 1
    @test PipeWireAO.pod_value(Int64, requested.values[5].second) == 0
    @test_throws FilterGraphAlgorithms.PropertyUpdateBusy FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:pole" => 0.25f0),
    )

    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    active = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Float32, active.values[1].second) == 3.0f0
    @test PipeWireAO.pod_value(Int64, active.values[4].second) == 1
    @test PipeWireAO.pod_value(Int64, active.values[5].second) == 1
    @test !FilterGraphPipeWire._adopt_pending_properties!(owner)

    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("gain" => 1.0f0),
    )
    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("missing:gain" => 1.0f0),
    )
    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:missing" => 1.0f0),
    )
    @test_throws ArgumentError FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:gain" => 1.0),
    )
end

@testset "Boolean closed Property adoption holds correction and generations" begin
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    graph = owner.graph
    prepared = only(graph.nodes).prepared
    output = zeros(Float32, 2); state = similar(output)
    residual = Float32[1, -2]; feedback = zeros(Float32, 2)
    FilterGraphAlgorithms.process!(output, state, prepared, residual, feedback)
    FilterGraphAlgorithms.process!(output, state, prepared, residual, feedback)
    held = (copy(output), copy(state))
    FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("closed-loop-correction:closed" => false))
    snapshot = @atomic :acquire owner.control_snapshot
    @test only(snapshot.requested_property_generations) == 1
    @test only(snapshot.active_property_generations) == 0
    @test prepared.plan.closed
    adopt_bytes(owner) = @allocated FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test adopt_bytes(owner) == 0
    @test !prepared.plan.closed
    snapshot = @atomic :acquire owner.control_snapshot
    @test only(snapshot.active_property_generations) == 1
    residual .= floatmax(Float32); feedback .= floatmax(Float32)
    @test FilterGraphAlgorithms.process!(output, state, prepared, residual, feedback) === nothing
    @test (output, state) == held
    FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("closed-loop-correction:closed" => true))
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test prepared.plan.closed
    @test prepared.workspace.committed_output == held[1]
    @test prepared.workspace.committed_state == held[2]
    FilterGraphPipeWire._stage_property_transaction!(owner,
        SPA.Props("closed-loop-correction:closed" => false))
    FilterGraphPipeWire._reset_owner!(owner)
    @test !prepared.plan.closed
    @test all(iszero, prepared.workspace.committed_output)
    @test all(iszero, prepared.workspace.committed_state)
end

include("property_observer.jl")
include("property_getter.jl")

@testset "Executor property preparation preserves active state and generations" begin
    graph = prepared_leaky_graph()
    prepared = only(graph.nodes).prepared
    prepared.workspace.committed_state .= 7.0f0
    prepared.workspace.candidate_state .= 9.0f0
    active_plan = prepared.plan
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph)
    snapshot = @atomic :acquire owner.control_snapshot
    pending_snapshot = owner.pending_control_snapshot
    recovery_failure_snapshot = owner.recovery_failure_snapshot
    property_transaction = owner.property_transaction
    parameter_transaction = owner.parameter_transaction
    property_state = owner.property_state[]
    parameter_state = owner.parameter_state[]
    control_state = owner.control_state[]
    dry = FilterGraphPipeWire._warm_property_preparation!(owner)
    @test FilterGraphAlgorithms.property_values(only(dry.plans)) == (gain=2.0f0, pole=0.5f0, closed=true)
    @test (@atomic :acquire owner.control_snapshot) === snapshot
    @test owner.pending_control_snapshot === pending_snapshot
    @test owner.recovery_failure_snapshot === recovery_failure_snapshot
    @test owner.property_transaction === property_transaction
    @test owner.parameter_transaction === parameter_transaction
    @test owner.property_state[] == property_state
    @test owner.parameter_state[] == parameter_state
    @test owner.control_state[] == control_state
    @test snapshot.requested_property_generations == (UInt64(0),)
    @test snapshot.active_property_generations == (UInt64(0),)
    @test prepared.plan === active_plan
    @test all(==(7.0f0), prepared.workspace.committed_state)
    @test all(==(9.0f0), prepared.workspace.candidate_state)

    node = PipeWireNode(graph; connect=false)
    @test prepared.plan === active_plan
    @test all(==(7.0f0), prepared.workspace.committed_state)
    @test all(==(9.0f0), prepared.workspace.candidate_state)
    properties = FilterGraphPipeWire._graph_properties(node.owner)
    @test PipeWireAO.pod_value(Int64, properties.values[4].second) == 0
    @test PipeWireAO.pod_value(Int64, properties.values[5].second) == 0
    close(node)
end

@testset "Parameter preparation precompiles both boundary layouts without publication" begin
    for layout in (ColumnMajorLayout(), RowMajorLayout())
        graph = prepared_control_graph()
        prepared = only(graph.nodes).prepared
        prepared.workspace.committed_state .= 7.0f0
        prepared.workspace.candidate_state .= 9.0f0
        owner = FilterGraphPipeWire.PreparedGraphOwner(graph, nothing, layout)
        snapshot = @atomic :acquire owner.control_snapshot
        plans = map(node -> node.prepared.plan, graph.nodes)
        parameter_transaction = owner.parameter_transaction
        property_transaction = owner.property_transaction
        control_state = owner.control_state[]
        parameter_state = owner.parameter_state[]
        @test FilterGraphPipeWire._precompile_parameter_preparation!(owner) === nothing
        for binding in values(owner.parameter_bindings)
            array_type = FilterGraphPipeWire._parameter_array_type(binding)
            @test array_type <: AbstractMatrix{Float32}
            @test (array_type <: FilterGraphPipeWire.RowMajorBorrowedArray) == (layout isa RowMajorLayout)
        end
        @test (@atomic :acquire owner.control_snapshot) === snapshot
        @test owner.parameter_transaction === parameter_transaction
        @test owner.property_transaction === property_transaction
        @test owner.control_state[] == control_state
        @test owner.parameter_state[] == parameter_state
        @test snapshot.requested_parameter_sequences == (UInt64(0),)
        @test snapshot.active_parameter_sequences == (UInt64(0),)
        @test map(node -> node.prepared.plan, graph.nodes) === plans
        @test all(==(7.0f0), prepared.workspace.committed_state)
        @test all(==(9.0f0), prepared.workspace.candidate_state)
    end
end

@testset "Callback GC diagnostics retain allocation-free bounded tracing" begin
    for enabled in (false, true)
        trace = withenv("JULIA_RTC_TRACE_GC" => (enabled ? "1" : "0")) do
            FilterGraphPipeWire.GraphCallbackTrace(2)
        end
        @test trace.record_gc == enabled
        @test length(trace.gc_pause_start) == (enabled ? 2 : 0)
        @test length(trace.gc_allocated_bytes_start) == (enabled ? 2 : 0)
        @test length(trace.gc_allocated_bytes_end) == (enabled ? 2 : 0)
        data = Float32[1]
        buffer = native_callback_buffer(data; header=BufferHeader(
            UInt32(0), UInt32(0), Int64(0), Int64(0), UInt64(42),
        ))
        GC.@preserve data buffer begin
            inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(buffer)), UInt32(1))
            @test FilterGraphPipeWire._trace_graph_callback_begin!(trace, inputs) == 1
            FilterGraphPipeWire._trace_graph_callback_end!(trace, 1)
            @test @allocated(FilterGraphPipeWire._trace_graph_callback_begin!(trace, inputs)) == 0
            @test @allocated(FilterGraphPipeWire._trace_graph_callback_end!(trace, 2)) == 0
            @test trace.sequence == UInt64[42, 42]
            @test all(trace.start_ns .<= trace.end_ns)
            if enabled
                @test trace.gc_initial_pause <= trace.gc_pause_start[1] <= trace.gc_pause_end[1]
                @test trace.gc_total_time_start_ns[1] <= trace.gc_total_time_end_ns[1]
                @test trace.gc_allocated_bytes_start[1] <= trace.gc_allocated_bytes_end[1]
            end
            @test FilterGraphPipeWire._trace_graph_callback_begin!(trace, inputs) == 0
            @test trace.omitted == 1
        end
    end
end

@testset "Scalar diagnostics preserve the pending transaction" begin
    node = PipeWireNode(prepared_leaky_graph(); connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_properties
    properties = SPA.Props("integrate:gain" => 3.0f0, "integrate:pole" => 0.25f0)
    mktemp() do path, io
        redirect_stderr(io) do
            withenv("JULIA_FGPW_PROPERTY_DIAG" => "1") do
                callback(node.filter, properties)
            end
        end
        seekstart(io)
        lines = readlines(io)
        @test length(lines) == 5
        @test [split(split(line, "event=")[2])[1] for line in lines] == [
            "callback_enter", "stage_enter", "prepare_enter", "prepare_completed", "published",
        ]
        stamps = [parse(UInt64, match(r"time_ns=(\d+)", line).captures[1]) for line in lines]
        @test issorted(stamps)
    end
    requested = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Float32, requested.values[1].second) == 2.0f0
    @test PipeWireAO.pod_value(Int64, requested.values[4].second) == 1
    @test PipeWireAO.pod_value(Int64, requested.values[5].second) == 0
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    active = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Float32, active.values[1].second) == 3.0f0
    @test PipeWireAO.pod_value(Float32, active.values[2].second) == 0.25f0
    @test PipeWireAO.pod_value(Int64, active.values[4].second) == 1
    @test PipeWireAO.pod_value(Int64, active.values[5].second) == 1

    mktemp() do path, io
        redirect_stderr(io) do
            withenv("JULIA_FGPW_PROPERTY_DIAG" => nothing) do
                callback(node.filter, SPA.Props("integrate:gain" => 4.0f0))
            end
        end
        @test filesize(path) == 0
    end
    close(node)
end

@testset "Property adoption is allocation-free for a heterogeneous Graph" begin
    owner = FilterGraphPipeWire.PreparedGraphOwner(
        prepared_heterogeneous_property_graph(),
    )
    FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:gain" => 3.0f0),
    )
    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:gain" => 4.0f0),
    )
    @test @allocated(FilterGraphPipeWire._adopt_pending_properties!(owner)) == 0
end

@testset "Property and Parameter updates share one bounded handoff" begin
    parameter = fill(2.0f0, 1, 2)

    property_first = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    FilterGraphPipeWire._stage_property_transaction!(
        property_first,
        SPA.Props("closed-loop-correction:gain" => 2.0f0),
    )
    @test !FilterGraphPipeWire._stage_parameter_transaction!(
        property_first,
        Symbol("shape-to-hidden") => parameter,
    )
    @test FilterGraphPipeWire._adopt_pending_properties!(property_first)
    @test FilterGraphPipeWire._stage_parameter_transaction!(
        property_first,
        Symbol("shape-to-hidden") => parameter,
    )
    @test FilterGraphPipeWire._adopt_pending_parameters!(property_first)
    property_first_plan = only(property_first.graph.nodes).prepared.plan
    @test FilterGraphAlgorithms.property_values(property_first_plan).gain == 2.0f0
    @test property_first_plan.shape_to_hidden == parameter

    parameter_first = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    @test FilterGraphPipeWire._stage_parameter_transaction!(
        parameter_first,
        Symbol("shape-to-hidden") => parameter,
    )
    @test_throws FilterGraphAlgorithms.PropertyUpdateBusy FilterGraphPipeWire._stage_property_transaction!(
        parameter_first,
        SPA.Props("closed-loop-correction:gain" => 2.0f0),
    )
    @test FilterGraphPipeWire._adopt_pending_parameters!(parameter_first)
    FilterGraphPipeWire._stage_property_transaction!(
        parameter_first,
        SPA.Props("closed-loop-correction:gain" => 2.0f0),
    )
    @test FilterGraphPipeWire._adopt_pending_properties!(parameter_first)
    parameter_first_plan = only(parameter_first.graph.nodes).prepared.plan
    @test FilterGraphAlgorithms.property_values(parameter_first_plan).gain == 2.0f0
    @test parameter_first_plan.shape_to_hidden == parameter
end

@testset "Reset adopts pending Graph control before clearing state" begin
    parameter = fill(2.0f0, 1, 2)
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    @test FilterGraphPipeWire._stage_parameter_transaction!(
        owner,
        Symbol("shape-to-hidden") => parameter,
    )
    FilterGraphPipeWire._reset_owner!(owner)
    @test !FilterGraphPipeWire._adopt_pending_parameters!(owner)
    @test only(owner.graph.nodes).prepared.plan.shape_to_hidden == parameter
    FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("closed-loop-correction:gain" => 2.0f0),
    )
    FilterGraphPipeWire._reset_owner!(owner)
    @test !FilterGraphPipeWire._adopt_pending_properties!(owner)
    @test FilterGraphAlgorithms.property_values(
        only(owner.graph.nodes).prepared.plan,
    ).gain == 2.0f0
end

@testset "Graph control snapshots publish coherent cross-thread state" begin
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_leaky_graph())

    FilterGraphPipeWire._stage_property_transaction!(
        owner,
        SPA.Props("integrate:gain" => 3.0f0),
    )
    requested = @atomic :acquire owner.control_snapshot
    @test FilterGraphAlgorithms.property_values(only(requested.plans)).gain == 2.0f0
    @test only(requested.requested_property_generations) == 1
    @test only(requested.active_property_generations) == 0

    @test FilterGraphPipeWire._adopt_pending_properties!(owner)
    active = @atomic :acquire owner.control_snapshot
    @test FilterGraphAlgorithms.property_values(only(active.plans)).gain == 3.0f0
    @test only(active.requested_property_generations) == 1
    @test only(active.active_property_generations) == 1

    if Threads.nthreads() > 1
        done = Threads.Atomic{Bool}(false)
        failure = Ref{Any}(nothing)
        reader = Threads.@spawn try
            while !done[]
                properties = FilterGraphPipeWire._graph_properties(owner)
                gain = PipeWireAO.pod_value(Float32, properties.values[1].second)
                requested_generation =
                    PipeWireAO.pod_value(Int64, properties.values[4].second)
                active_generation =
                    PipeWireAO.pod_value(Int64, properties.values[5].second)
                requested_generation - active_generation in (0, 1) || error(
                    "incoherent requested and active generations",
                )
                expected_gain = isodd(active_generation) ? 3.0f0 : 4.0f0
                gain == expected_gain || error(
                    "active generation and property Plan came from different snapshots",
                )
            end
        catch error
            failure[] = error
        end
        for generation in 2:500
            gain = isodd(generation) ? 3.0f0 : 4.0f0
            FilterGraphPipeWire._stage_property_transaction!(
                owner,
                SPA.Props("integrate:gain" => gain),
            )
            FilterGraphPipeWire._adopt_pending_properties!(owner)
        end
        done[] = true
        wait(reader)
        @test failure[] === nothing
    end
end

@testset "Parameter adoption notification survives a following request" begin
    owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
    parameter = fill(2.0f0, 1, 2)
    before = FilterGraphPipeWire._parameter_adoption_count(owner)
    @test FilterGraphPipeWire._stage_parameter_transaction!(
        owner,
        Symbol("shape-to-hidden") => parameter,
    )
    @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
    @test FilterGraphPipeWire._stage_parameter_transaction!(
        owner,
        Symbol("shape-to-hidden") => fill(3.0f0, 1, 2),
    )
    @test FilterGraphPipeWire._parameter_adoption_count(owner) == before + 1
    @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_PENDING
end

@testset "FGN deployment rates project each Algorithm Port ratio" begin
    unit = AlgorithmFormat(Float32, (2,), "test.signal.f32/1")
    fast = AlgorithmFormat(Float32, (2,), "test.signal.f32/1"; rate=(4, 1))

    @test FilterGraphPipeWire._port_rate(unit, nothing) === nothing
    @test_throws ArgumentError FilterGraphPipeWire._port_rate(fast, nothing)
    @test_throws ArgumentError FilterGraphPipeWire._port_rate(unit, (1.0, 1.0))
    @test_throws ArgumentError FilterGraphPipeWire._port_rate(unit, (true, 1))
    projected = FilterGraphPipeWire._port_rate(fast, (25, 2))
    @test (projected.num, projected.denom) == (UInt32(50), UInt32(1))

    graph = prepared_leaky_graph()
    ports = FilterGraphPipeWire._graph_ports(graph, (1_000, 1))
    @test map(port -> port.name, ports) == ["input", "output"]
    @test map(port -> port.direction, ports) == [
        PipeWireAO.DIRECTION_INPUT,
        PipeWireAO.DIRECTION_OUTPUT,
    ]
    @test all(port -> port.format.rate == SPA.Fraction(1_000, 1), ports)

    row_major_ports = FilterGraphPipeWire._graph_ports(
        graph,
        (1_000, 1);
        boundary_layout=RowMajorLayout(),
    )
    @test all(port -> port.format.layout == NdArray.ROW_MAJOR, row_major_ports)
    row_major_node = PipeWireNode(
        graph;
        rate=(1_000, 1),
        boundary_layout=RowMajorLayout(),
        connect=false,
    )
    @test isopen(row_major_node)
    close(row_major_node)

    parameter_graph = prepared_pixel_calibration_graph()
    row_major_parameter_ports = FilterGraphPipeWire._graph_ports(
        parameter_graph,
        (1_000, 1);
        boundary_layout=RowMajorLayout(),
    )
    @test all(
        port -> port.format.layout == NdArray.ROW_MAJOR,
        row_major_parameter_ports,
    )
    parameter_ports = FilterGraphPipeWire._graph_ports(parameter_graph, (1_000, 1))
    @test map(port -> port.name, parameter_ports) == [
        "raw",
        "flat",
        "background",
        "calibrated",
    ]
    @test map(port -> port.direction, parameter_ports) == [
        PipeWireAO.DIRECTION_INPUT,
        PipeWireAO.DIRECTION_INPUT,
        PipeWireAO.DIRECTION_INPUT,
        PipeWireAO.DIRECTION_OUTPUT,
    ]
    @test map(port -> port.parameter, parameter_ports) == [false, true, true, false]
    @test map(port -> port.format.rate, parameter_ports) == [
        SPA.Fraction(2_000, 1),
        nothing,
        nothing,
        SPA.Fraction(2_000, 1),
    ]

    latest_join = FilterGraphs.prepare_latest_hold(
        prepared_latest_hold_graph(),
        :primary,
        :held,
    )
    latest_ports = FilterGraphPipeWire._latest_hold_ports(latest_join, (1_000, 1))
    @test map(port -> port.name, latest_ports) == ["primary", "held", "joined"]
    @test all(port -> port.format.rate == SPA.Fraction(2_000, 1), latest_ports)
    @test latest_join.graph.input_formats.held.rate == (UInt32(1), UInt32(1))
end

@testset "prepared Julia Graph processes borrowed PipeWire storage" begin
    graph = prepared_leaky_graph()
    owner = FilterGraphPipeWire.PreparedGraphOwner(graph)
    input = collect(Float32, 1:8)
    output = zeros(Float32, 8)

    FilterGraphPipeWire._prepare_owner(owner)
    GC.@preserve input output begin
        input_view = FilterGraphPipeWire._borrow_pointer(
            owner.input_bindings.input,
            pointer(input),
            Val(false),
        )
        output_view = FilterGraphPipeWire._borrow_pointer(
            owner.output_bindings.output,
            pointer(output),
            Val(true),
        )
        inputs = (input=input_view,)
        outputs = (output=output_view,)

        @test FilterGraphs._process_admitted!(
            outputs,
            graph,
            inputs,
            graph.default_input_metadata,
        ) == (output=true,)
        @test output == 2.0f0 .* input
        FilterGraphs._process_admitted!(
            outputs,
            graph,
            inputs,
            graph.default_input_metadata,
        )
        @test output == 3.0f0 .* input
        @test @allocated(
            FilterGraphs._process_admitted!(
                outputs,
                graph,
                inputs,
                graph.default_input_metadata,
            )
        ) == 0
    end

    FilterGraphPipeWire._deactivate_owner(owner)
    @test !owner.active
end

@testset "complete Graph callback is allocation-free after preparation" begin
    graph = prepared_leaky_graph()
    node = PipeWireNode(graph; connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_process
    FilterGraphPipeWire._prepare_owner(owner)

    input = collect(Float32, 1:8)
    output = zeros(Float32, 8)
    native_input = native_callback_buffer(input)
    native_output = native_callback_buffer(output)
    GC.@preserve input output native_input native_output begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_input)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        @test @inferred(callback(node.filter, inputs, outputs)) === nothing
        @test output == 2.0f0 .* input
        callback(node.filter, inputs, outputs)
        @test @allocated(callback(node.filter, inputs, outputs)) == 0
    end
    @test owner.process_count == 3
    close(node)
end

@testset "ordinary Graph callback propagates declared acquisition provenance" begin
    graph = prepared_metadata_source_graph()
    node = PipeWireNode(graph; connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_process
    FilterGraphPipeWire._prepare_owner(owner)

    left = Float32[1, 2]
    right = Float32[10, 20]
    output = zeros(Float32, 2)
    left_header = BufferHeader(UInt32(0), UInt32(3), Int64(101), Int64(-1), UInt64(11))
    right_header = BufferHeader(
        SPA.META_HEADER_FLAG_MARKER | SPA.META_HEADER_FLAG_HEADER,
        UInt32(9),
        Int64(202),
        Int64(-7),
        UInt64(22),
    )
    expected_header = BufferHeader(
        SPA.META_HEADER_FLAG_MARKER,
        UInt32(9),
        Int64(202),
        Int64(0),
        UInt64(22),
    )
    native_left = native_callback_buffer(
        left;
        header=left_header,
        acquisition_sequence=UInt64(11),
    )
    native_right = native_callback_buffer(
        right;
        header=right_header,
        acquisition_sequence=UInt64(22),
    )
    native_missing = native_callback_buffer(
        right;header_available=false,
        properties_available=true,
    )
    native_header_only = native_callback_buffer(
        right;
        header=right_header,
        properties_available=true,
    )
    native_output = native_callback_buffer(
        output;
        header_available=true,
        properties_available=true,
    )
    metadata = UInt32(1)
    native_inputs = vcat(native_left, native_right)
    missing_native_inputs = vcat(native_left, native_missing)
    header_only_native_inputs = vcat(native_left, native_header_only)

    GC.@preserve left right output native_left native_right native_missing native_header_only native_output native_inputs missing_native_inputs header_only_native_inputs begin
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_inputs)),
            UInt32(2),
        )
        @test @inferred(callback(node.filter, inputs, outputs)) === nothing
        @test output == Float32[11, 22]
        @test native_callback_header(native_output) == expected_header
        @test native_callback_acquisition(native_output) ==
              native_callback_acquisition(native_right)
        @test native_callback_acquisition(native_output) !=
              native_callback_acquisition(native_left)
        @test native_callback_metadata_valid(native_output) == metadata

        missing_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(missing_native_inputs)),
            UInt32(2),
        )
        @test callback(node.filter, missing_inputs, outputs) === nothing
        @test acquisition_identity(only(native_output.metadata)) === nothing

        header_only_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(header_only_native_inputs)),
            UInt32(2),
        )
        @test callback(node.filter, header_only_inputs, outputs) === nothing
        @test native_callback_header(native_output) == expected_header
        @test native_callback_metadata_valid(native_output) == metadata

        callback(node.filter, inputs, outputs)
        @test @allocated(callback(node.filter, inputs, outputs)) == 0
    end
    @test owner.process_count == 5
    close(node)
end

@testset "PipeWire callback admits a bounded latest/hold join" begin
    graph = prepared_latest_hold_graph()
    join = FilterGraphs.prepare_latest_hold(graph, :primary, :held)
    owner = FilterGraphPipeWire.PreparedGraphOwner(join)
    FilterGraphPipeWire._prepare_owner(owner)
    @test FilterGraphs.latest_hold_counters(join) == (
        primary_arrivals=0,
        held_arrivals=0,
        new_uses=0,
        retained_uses=0,
        skipped_primary=0,
        invalid_primary=0,
        invalid_held=0,
        stale_held=0,
        discontinuities=0,
        timeouts=0,
        overruns=0,
    )

    # The callback does not inspect its filter argument. A regular unconnected
    # endpoint supplies the concrete wrapper while the latest/hold owner drives
    # the tested data plane.
    carrier = PipeWireNode(prepared_leaky_graph(); connect=false)
    callback = FilterGraphPipeWire.GraphProcessCallback(owner)
    primary = Float32[10, 20]
    held = Float32[1, 2]
    output = fill(-1.0f0, 2)
    primary_header = BufferHeader(UInt32(0), UInt32(0), 1_000, 0, 100)
    held_header = BufferHeader(UInt32(0), UInt32(0), 900, 0, 10)
    next_held_header = BufferHeader(UInt32(0), UInt32(0), 950, 0, 11)
    native_primary = native_callback_buffer(primary; header=primary_header)
    native_primary_absent = native_callback_buffer(primary; present=false)
    native_held = native_callback_buffer(held; header=held_header)
    native_next_held = native_callback_buffer(held; header=next_held_header)
    native_unsequenced_held = native_callback_buffer(held;header_available=false)
    native_held_absent = native_callback_buffer(held; present=false)
    native_output = native_callback_buffer(output; header_available=true)

    GC.@preserve primary held output native_primary native_primary_absent native_held native_next_held native_unsequenced_held native_held_absent native_output begin
        held_only_storage = vcat(native_primary_absent, native_held)
        held_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(held_only_storage)),
            UInt32(2),
        )
        primary_only_storage = vcat(native_primary, native_held_absent)
        primary_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(primary_only_storage)),
            UInt32(2),
        )
        both_storage = vcat(native_primary, native_next_held)
        both = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(both_storage)),
            UInt32(2),
        )
        unsequenced_held_storage = vcat(native_primary_absent, native_unsequenced_held)
        unsequenced_held_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(unsequenced_held_storage)),
            UInt32(2),
        )
        unsequenced_both_storage = vcat(native_primary, native_unsequenced_held)
        unsequenced_both = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(unsequenced_both_storage)),
            UInt32(2),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )

        @test callback(carrier.filter, held_only, outputs) === nothing
        @test native_callback_flags(native_output) ==
              PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
        @test output == fill(-1.0f0, 2)

        set_native_callback_flags!(native_output, UInt32(0))
        @test callback(carrier.filter, primary_only, outputs) === nothing
        @test output == Float32[11, 22]
        @test native_callback_flags(native_output) == UInt32(0)
        @test native_callback_header(native_output).sequence == 10

        primary .= Float32[20, 30]
        @test callback(carrier.filter, primary_only, outputs) === nothing
        @test output == Float32[21, 32]

        fill!(output, -2.0f0)
        @test callback(carrier.filter, primary_only, outputs) === nothing
        @test native_callback_flags(native_output) ==
              PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
        @test output == fill(-2.0f0, 2)

        held .= Float32[3, 4]
        primary .= Float32[30, 40]
        set_native_callback_flags!(native_output, UInt32(0))
        @test callback(carrier.filter, both, outputs) === nothing
        @test output == Float32[33, 44]
        @test native_callback_flags(native_output) == UInt32(0)

        # Warm each exact state before measuring the callback itself.
        set_native_callback_flags!(native_output, UInt32(0))
        callback(carrier.filter, unsequenced_held_only, outputs)
        @test @allocated(callback(carrier.filter, unsequenced_held_only, outputs)) == 0
        set_native_callback_flags!(native_output, UInt32(0))
        @test @allocated(callback(carrier.filter, primary_only, outputs)) == 0
        set_native_callback_flags!(native_output, UInt32(0))
        @test @allocated(callback(carrier.filter, unsequenced_both, outputs)) == 0
    end

    counters = FilterGraphs.latest_hold_counters(join)
    @test counters.primary_arrivals == 6
    @test counters.held_arrivals == 5
    @test counters.new_uses == 4
    @test counters.retained_uses == 1
    @test counters.skipped_primary == 1
    @test counters.timeouts == 1
    @test counters.overruns == 1
    @test owner.process_count == 9
    close(carrier)
end

@testset "PipeWire callback admits all-required synchronized inputs" begin
    graph = prepared_synchronized_graph()
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph)
    owner = FilterGraphPipeWire.PreparedGraphOwner(synchronized)
    FilterGraphPipeWire._prepare_owner(owner)
    @test FilterGraphs.synchronized_input_counters(synchronized) == (
        arrivals=0,
        completed_windows=0,
        expired_windows=0,
        corrupted_windows=0,
        rejected_windows=0,
        aborted_windows=0,
        stale_arrivals=0,
        duplicate_arrivals=0,
        discontinuities=0,
        incoherent_arrivals=0,
        domain_changes=0,
    )

    carrier = PipeWireNode(prepared_leaky_graph(); connect=false)
    callback = FilterGraphPipeWire.GraphProcessCallback(owner)
    a = Float32[1, 2]
    b = Float32[10, 20]
    c = Float32[100, 200]
    output = fill(-1.0f0, 2)
    header_10 = BufferHeader(UInt32(0), UInt32(0), 1_000, 0, 10)
    header_11 = BufferHeader(UInt32(0), UInt32(0), 1_001, 0, 11)
    corrupted_header = BufferHeader(
        PipeWireAO.SPA.META_HEADER_FLAG_CORRUPTED,
        UInt32(0),
        1_002,
        0,
        12,
    )
    present_a = native_callback_buffer(a; header=header_10)
    present_b = native_callback_buffer(b; header=header_10)
    present_c = native_callback_buffer(c; header=header_10)
    next_a = native_callback_buffer(a; header=header_11)
    next_b = native_callback_buffer(b; header=header_11)
    next_c = native_callback_buffer(c; header=header_11)
    corrupt_a = native_callback_buffer(a; header=corrupted_header)
    corrupt_b = native_callback_buffer(b; header=corrupted_header)
    corrupt_c = native_callback_buffer(c; header=corrupted_header)
    no_header_a = native_callback_buffer(a;header_available=false)
    absent_a = native_callback_buffer(a; present=false)
    absent_b = native_callback_buffer(b; present=false)
    absent_c = native_callback_buffer(c; present=false)
    native_output = native_callback_buffer(output; header_available=true)
    unavailable_output = native_callback_buffer(output;header_available=false)
    native_buffers = (
        present_a,
        present_b,
        present_c,
        next_a,
        next_b,
        next_c,
        corrupt_a,
        corrupt_b,
        corrupt_c,
        no_header_a,
        absent_a,
        absent_b,
        absent_c,
        native_output,
        unavailable_output,
    )

    GC.@preserve a b c output native_buffers begin
        a_only_storage = vcat(present_a, absent_b, absent_c)
        a_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(a_only_storage)),
            UInt32(3),
        )
        b_only_storage = vcat(absent_a, present_b, absent_c)
        b_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(b_only_storage)),
            UInt32(3),
        )
        c_only_storage = vcat(absent_a, absent_b, present_c)
        c_only = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(c_only_storage)),
            UInt32(3),
        )
        next_all_storage = vcat(next_a, next_b, next_c)
        next_all = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(next_all_storage)),
            UInt32(3),
        )
        corrupt_all_storage = vcat(corrupt_a, corrupt_b, corrupt_c)
        corrupt_all = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(corrupt_all_storage)),
            UInt32(3),
        )
        no_header_storage = vcat(no_header_a, absent_b, absent_c)
        no_header = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(no_header_storage)),
            UInt32(3),
        )
        no_arrival_storage = vcat(absent_a, absent_b, absent_c)
        no_arrival = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(no_arrival_storage)),
            UInt32(3),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        unavailable_outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(unavailable_output)),
            UInt32(1),
        )

        @test callback(carrier.filter, a_only, outputs) === nothing
        @test native_callback_flags(native_output) ==
              PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
        @test output == fill(-1.0f0, 2)

        @test callback(carrier.filter, b_only, outputs) === nothing
        @test native_callback_flags(native_output) ==
              PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

        @test callback(carrier.filter, c_only, outputs) === nothing
        @test output == Float32[111, 222]
        @test native_callback_flags(native_output) == UInt32(0)
        @test native_callback_header(native_output).sequence == 10

        a .= Float32[2, 3]
        b .= Float32[20, 30]
        c .= Float32[200, 300]
        @test callback(carrier.filter, next_all, outputs) === nothing
        @test output == Float32[222, 333]
        @test native_callback_header(native_output).sequence == 11

        fill!(output, -2.0f0)
        @test callback(carrier.filter, corrupt_all, outputs) === nothing
        @test native_callback_flags(native_output) ==
              PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
        @test output == fill(-2.0f0, 2)

        @test_throws ArgumentError callback(carrier.filter, no_header, outputs)
        @test_throws InvalidStateException callback(carrier.filter, no_arrival, outputs)

        FilterGraphAlgorithms.reset!(synchronized)
        FilterGraphs._clear_synchronized_input_counters!(synchronized)
        set_native_callback_flags!(native_output, UInt32(0))
        callback(carrier.filter, a_only, outputs)
        @test @allocated(callback(carrier.filter, a_only, outputs)) == 0
        @test @allocated(callback(carrier.filter, b_only, outputs)) == 0
        @test @allocated(callback(carrier.filter, c_only, outputs)) == 0
        @test @allocated(callback(carrier.filter, next_all, outputs)) == 0
        @test @allocated(callback(carrier.filter, corrupt_all, outputs)) == 0

        FilterGraphAlgorithms.reset!(synchronized)
        FilterGraphs._clear_synchronized_input_counters!(synchronized)
        @test_throws InvalidStateException callback(
            carrier.filter,
            next_all,
            unavailable_outputs,
        )
        @test FilterGraphs.synchronized_input_counters(synchronized).aborted_windows == 0
    end

    @test owner.process_count == 11
    close(carrier)
end

@testset "synchronized callback retains arrivals after one publication" begin
    synchronized = FilterGraphs.prepare_synchronized_inputs(
        prepared_synchronized_graph(),
    )
    owner = FilterGraphPipeWire.PreparedGraphOwner(synchronized)
    FilterGraphPipeWire._prepare_owner(owner)

    a_30 = Float32[1, 2]
    b_30 = Float32[10, 20]
    c_30 = Float32[100, 200]
    a_31 = Float32[3, 4]
    b_31 = Float32[30, 40]
    c_31 = Float32[300, 400]
    output = fill(-1.0f0, 2)
    header_30 = BufferHeader(UInt32(0), UInt32(0), 3_000, 0, 30)
    header_31 = BufferHeader(UInt32(0), UInt32(0), 3_001, 0, 31)
    absent = native_callback_buffer(Float32[0, 0]; present=false)
    native_output = native_callback_buffer(output; header_available=true)
    native_buffers = (
        native_callback_buffer(a_30; header=header_30),
        native_callback_buffer(b_30; header=header_30),
        native_callback_buffer(c_30; header=header_30),
        native_callback_buffer(a_31; header=header_31),
        native_callback_buffer(b_31; header=header_31),
        native_callback_buffer(c_31; header=header_31),
        absent,
        native_output,
    )

    GC.@preserve a_30 b_30 c_30 a_31 b_31 c_31 output native_buffers begin
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        b_only_storage = vcat(absent, native_buffers[2], absent)
        c_only_storage = vcat(absent, absent, native_buffers[3])
        completing_and_next_storage = vcat(native_buffers[1], native_buffers[5], absent)
        finish_next_storage = vcat(native_buffers[4], absent, native_buffers[6])

        for storage in (b_only_storage, c_only_storage)
            inputs = NdArrayFilterBuffers{false}(
                Ptr{Cvoid}(pointer(storage)),
                UInt32(3),
            )
            FilterGraphPipeWire._process_graph_buffers!(
                owner,
                synchronized,
                inputs,
                outputs,
            )
        end

        completing_and_next = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(completing_and_next_storage)),
            UInt32(3),
        )
        FilterGraphPipeWire._process_graph_buffers!(
            owner,
            synchronized,
            completing_and_next,
            outputs,
        )
        @test output == Float32[111, 222]
        @test native_callback_header(native_output).sequence == 30

        finish_next = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(finish_next_storage)),
            UInt32(3),
        )
        FilterGraphPipeWire._process_graph_buffers!(
            owner,
            synchronized,
            finish_next,
            outputs,
        )
        @test output == Float32[333, 444]
        @test native_callback_header(native_output).sequence == 31
    end
    counters = FilterGraphs.synchronized_input_counters(synchronized)
    @test counters.arrivals == 6
    @test counters.completed_windows == 2
    @test counters.expired_windows == 0
end

@testset "Graph callback projects FGN metadata and conditional availability" begin
    graph = prepared_conditional_graph()
    node = PipeWireNode(graph; connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_process
    FilterGraphPipeWire._prepare_owner(owner)

    input = Float32[3, 5]
    output = zeros(Float32, 2)
    deferred_header = BufferHeader(
        SPA.META_HEADER_FLAG_DISCONT,
        UInt32(4),
        Int64(123_456),
        Int64(-7),
        UInt64(41),
    )
    terminal_flags = SPA.META_HEADER_FLAG_MARKER |
                     SPA.META_HEADER_FLAG_DISCONT |
                     SPA.META_HEADER_FLAG_CORRUPTED |
                     SPA.META_HEADER_FLAG_HEADER
    terminal_header = BufferHeader(
        terminal_flags,
        UInt32(6),
        Int64(987_654),
        Int64(-9),
        UInt64(42),
    )
    expected_header = BufferHeader(
        SPA.META_HEADER_FLAG_MARKER |
        SPA.META_HEADER_FLAG_DISCONT |
        SPA.META_HEADER_FLAG_CORRUPTED,
        UInt32(6),
        Int64(987_654),
        Int64(0),
        UInt64(42),
    )
    native_deferred = native_callback_buffer(input; header=deferred_header)
    native_terminal = native_callback_buffer(input; header=terminal_header)
    native_output = native_callback_buffer(output; header_available=true)
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

    GC.@preserve input output native_deferred native_terminal native_output begin
        deferred_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_deferred)),
            UInt32(1),
        )
        terminal_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_terminal)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )

        @test @inferred(callback(node.filter, deferred_inputs, outputs)) === nothing
        @test output == zeros(Float32, 2)
        @test native_callback_flags(native_output) == unavailable
        @test @allocated(callback(node.filter, deferred_inputs, outputs)) == 0

        @test @inferred(callback(node.filter, terminal_inputs, outputs)) === nothing
        @test output == input
        @test native_callback_flags(native_output) == UInt32(0)
        @test native_callback_header(native_output) == expected_header
        @test FilterGraphs.output_metadata(graph) == (
            output=SampleMetadata(
                UInt64(42),
                UInt32(6);
                timestamp_nanoseconds=Int64(987_654),
                terminal=true,
                discontinuity=true,
                corrupted=true,
            ),
        )
        @test @allocated(callback(node.filter, terminal_inputs, outputs)) == 0
    end
    @test owner.process_count == 4
    close(node)
end

@testset "deferred Graph ingress retains two blocks without callback allocation" begin
    graph = prepared_conditional_graph()
    node = PipeWireNode(
        graph; connect=false, defer_first_input=true, trace_callbacks=4,
    )
    FilterGraphPipeWire._prepare_owner(node.owner)
    callback = node.filter.callbacks.on_process
    first_block = Float32[3, 5]
    terminal_block = Float32[7, 11]
    result = zeros(Float32, 2)
    first_buffer = native_callback_buffer(
        first_block;
        header=BufferHeader(UInt32(0), UInt32(0), Int64(0), Int64(0), UInt64(1)),
        acquisition_sequence=UInt64(42),extra_property=Int64(123),
    )
    terminal_buffer = native_callback_buffer(
        terminal_block;
        header=BufferHeader(
            SPA.META_HEADER_FLAG_MARKER, UInt32(2), Int64(0), Int64(0), UInt64(99),
        ),
        acquisition_sequence=UInt64(42),extra_property=Int64(123),
    )
    output_buffer = native_callback_buffer(result; header_available=true)
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE
    GC.@preserve first_block terminal_block result first_buffer terminal_buffer output_buffer begin
        first_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(first_buffer)), UInt32(1),
        )
        terminal_inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(terminal_buffer)), UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(output_buffer)), UInt32(1),
        )
        callback(node.filter, first_inputs, outputs)
        @test native_callback_flags(output_buffer) == unavailable
        callback(node.filter, terminal_inputs, outputs)
        @test result == terminal_block
        FilterGraphs.reset!(graph)

        @test @allocated(callback(node.filter, first_inputs, outputs)) == 0
        @test native_callback_flags(output_buffer) == unavailable
        @test @allocated(callback(node.filter, terminal_inputs, outputs)) == 0
        @test result == terminal_block
        @test node.owner.callback_trace.count == 4
        @test node.owner.callback_trace.omitted == 0
        previous=copy(result)
        callback(node.filter,first_inputs,outputs)
        set_metadata_bytes!(writable_metadata_records(terminal_buffer.metadata[1]),UInt32(0x900),(UInt8(124),ntuple(_ -> UInt8(0),7)...))
        @test_throws InvalidStateException callback(node.filter,terminal_inputs,outputs)
        @test result==previous
        @test native_callback_flags(output_buffer)==unavailable
    end
    close(node)
end

@testset "Graph callback contains expected Algorithm rejection" begin
    graph = prepared_failure_graph()
    node = PipeWireNode(graph; connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_process
    FilterGraphPipeWire._prepare_owner(owner)

    rejected_input = Float32[-1, 2]
    accepted_input = Float32[7, 11]
    output = fill(73.0f0, 2)
    native_rejected = native_callback_buffer(rejected_input)
    native_accepted = native_callback_buffer(accepted_input)
    native_output = native_callback_buffer(output)
    unavailable = PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE

    GC.@preserve rejected_input accepted_input output native_rejected native_accepted native_output begin
        rejected = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_rejected)),
            UInt32(1),
        )
        accepted = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_accepted)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )

        @test @inferred(callback(node.filter, rejected, outputs)) === nothing
        @test output == fill(73.0f0, 2)
        @test native_callback_flags(native_output) == unavailable
        @test @allocated(callback(node.filter, rejected, outputs)) == 0

        @test @inferred(callback(node.filter, accepted, outputs)) === nothing
        @test output == accepted_input
        @test native_callback_flags(native_output) == UInt32(0)
        @test @allocated(callback(node.filter, accepted, outputs)) == 0
    end
    @test owner.process_count == 4
    @test last_error(node) === nothing
    close(node)
end

@testset "Parameter Ports hand off one prepared transaction at a Graph boundary" begin
    graph = prepared_pixel_calibration_graph()
    node = PipeWireNode(graph; rate=(1_000, 1), connect=false)
    owner = node.owner
    process_callback = node.filter.callbacks.on_process
    parameter_callback = node.filter.callbacks.on_parameter
    FilterGraphPipeWire._prepare_owner(owner)

    original_plan = graph.nodes[1].prepared.plan
    flat = fill(2.0f0, 12)
    background = fill(1.0f0, 12)
    native_flat = native_callback_buffer(flat)
    native_background = native_callback_buffer(background)
    GC.@preserve flat background native_flat native_background begin
        flat_buffer = NdArrayFilterBuffer{false}(Ptr{Cvoid}(pointer(native_flat)))
        background_buffer =
            NdArrayFilterBuffer{false}(Ptr{Cvoid}(pointer(native_background)))
        @test parameter_callback(node.filter, 2, flat_buffer)
        @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_PENDING
        @test graph.nodes[1].prepared.plan === original_plan
        @test !parameter_callback(node.filter, 3, background_buffer)
        @test graph.nodes[1].prepared.plan === original_plan
    end

    raw = UInt16[1, 2, 3, 4, 5, 6]
    output = zeros(Float32, 6)
    header = BufferHeader(UInt32(0), UInt32(0), Int64(100), Int64(0), UInt64(1))
    native_raw = native_callback_buffer(raw; header)
    native_output = native_callback_buffer(output; header_available=true)
    GC.@preserve raw output native_raw native_output begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_raw)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        @test @inferred(process_callback(node.filter, inputs, outputs)) === nothing
        @test output == 2.0f0 .* Float32.(raw)
    end
    @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_AVAILABLE
    @test graph.nodes[1].prepared.plan !== original_plan
    adopted = FilterGraphPipeWire._graph_properties(owner)
    @test PipeWireAO.pod_value(Int64, adopted.values[end - 1].second) == 1
    @test PipeWireAO.pod_value(Int64, adopted.values[end].second) == 1

    @test_throws DimensionMismatch replace_parameters!(
        node,
        :flat => zeros(Float32, 3, 3),
    )
    @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_AVAILABLE

    adopted_plan = graph.nodes[1].prepared.plan
    @test replace_parameters!(
        node,
        :flat => fill(3.0f0, 4, 3),
        :background => fill(1.0f0, 4, 3),
    ) === node
    @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_PENDING
    @test graph.nodes[1].prepared.plan === adopted_plan
    @test_throws InvalidStateException replace_parameters!(
        node,
        :flat => fill(4.0f0, 4, 3),
    )

    fill!(output, 0.0f0)
    GC.@preserve raw output native_raw native_output begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_raw)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        process_callback(node.filter, inputs, outputs)
        @test output == 3.0f0 .* (Float32.(raw) .- 1.0f0)

        replace_parameters!(
            node,
            :flat => fill(4.0f0, 4, 3),
            :background => fill(2.0f0, 4, 3),
        )
        @test @allocated(process_callback(node.filter, inputs, outputs)) == 0
        @test output == 4.0f0 .* (Float32.(raw) .- 2.0f0)
    end
    @test owner.parameter_state[] == FilterGraphPipeWire._PARAMETER_SLOT_AVAILABLE
    close(node)
end

include("linked_row_graph.jl")
include("closed_loop_feedback.jl")
include("composition.jl")
include("parameter_ports.jl")
include("command_endpoint.jl")
include("feedback_bridge.jl")
include("frame_observer.jl")
include("recovery.jl")
include("recovery_leaky_integrator.jl")
include("recovery_timeout.jl")
include("recovery_publication.jl")
include("recovery_boundaries.jl")
include("recovery_feedback.jl")
include("composition_recovery.jl")
include("physical_offset_reference.jl")
include("captured_graph.jl")
include("buffer_properties_retention.jl")
include("preparation.jl")
include("chopped_preparation_callback.jl")

@testset "metadata layout preflight precedes mutable Node execution" begin
    graph = prepared_leaky_graph()
    node = PipeWireNode(graph; connect=false)
    owner = node.owner
    callback = node.filter.callbacks.on_process
    FilterGraphPipeWire._prepare_owner(owner)

    input = collect(Float32, 1:8)
    output = zeros(Float32, 8)
    header = BufferHeader(UInt32(0), UInt32(0), Int64(1), Int64(0), UInt64(1))
    native_input = native_callback_buffer(input; header)
    native_output = native_callback_buffer(output;header_available=false)
    workspace = graph.nodes[1].prepared.workspace

    GC.@preserve input output native_input native_output begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_input)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        @test_throws InvalidStateException callback(node.filter, inputs, outputs)
    end
    @test all(iszero,leaky_integrator_state(workspace))
    @test owner.process_count == 0
    close(node)
end

@testset "prepared Graph endpoint admits supported publication contracts" begin
    graph = prepared_leaky_graph()
    node = PipeWireNode(
        graph;
        name="test.julia.filter-graph-island",
        rate=(1_000, 1),
        connect=false,
    )
    @test node.owner isa FilterGraphPipeWire.PreparedGraphOwner
    @test isopen(node)
    @test node_state(node) == PipeWireAO.NDARRAY_FILTER_STATE_UNCONNECTED
    close(node)

    parameter_graph = prepared_pixel_calibration_graph()
    parameter_node = PipeWireNode(
        parameter_graph;
        rate=(1_000, 1),
        connect=false,
    )
    @test isopen(parameter_node)
    close(parameter_node)

    conditional_graph = prepared_conditional_graph()
    conditional_node = PipeWireNode(conditional_graph; connect=false)
    @test isopen(conditional_node)
    close(conditional_node)

    failure_graph = prepared_failure_graph()
    failure_node = PipeWireNode(failure_graph; connect=false)
    @test isopen(failure_node)
    close(failure_node)

    latest_join = FilterGraphs.prepare_latest_hold(
        prepared_latest_hold_graph(),
        :primary,
        :held,
    )
    latest_node = PipeWireNode(latest_join; rate=(1_000, 1), connect=false)
    @test isopen(latest_node)
    @test latest_node.owner.join === latest_join
    close(latest_node)

    synchronized = FilterGraphs.prepare_synchronized_inputs(
        prepared_synchronized_graph(),
    )
    synchronized_node = PipeWireNode(
        synchronized;
        rate=(1_000, 1),
        connect=false,
    )
    @test isopen(synchronized_node)
    @test synchronized_node.owner.join === synchronized
    close(synchronized_node)

    row_synchronized_node = PipeWireNode(synchronized;
        rate=(1_000, 1), connect=false, boundary_layout=RowMajorLayout())
    @test row_synchronized_node.owner.input_bindings.a isa FilterGraphPipeWire.PortBinding{Float32,1,RowMajorLayout}
    @test row_synchronized_node.owner.output_bindings.sum isa FilterGraphPipeWire.PortBinding{Float32,1,RowMajorLayout}
    @test all(port -> port.format.layout == PipeWireAO.NdArray.ROW_MAJOR,
        FilterGraphPipeWire._graph_ports(synchronized.graph, (1_000, 1); boundary_layout=RowMajorLayout()))
    close(row_synchronized_node)
end

@testset "acquisition synchronized row-major matrix boundary" begin
    graph = prepare_graph_text(replace(ADAPTER_SYNCHRONIZED_GRAPH,
        "adapter-synchronized-sum-f32" => "adapter-synchronized-matrix-sum-f32");
        algorithms=(AdapterSynchronizedMatrixSumF32,))
    synchronized = FilterGraphs.prepare_synchronized_inputs(graph; match=:acquisition)
    node = PipeWireNode(synchronized; connect=false, boundary_layout=RowMajorLayout())
    node.filter.callbacks.on_prepare(node.filter)
    callback = FilterGraphPipeWire.GraphProcessCallback(node.owner)
    a = Float32[1, 2, 3, 4, 5, 6]
    b, c = 10 .* a, 100 .* a
    output = fill(-1.0f0, 6)
    header = BufferHeader(UInt32(0), UInt32(0), 1000, 0, 1)
    native_a = native_callback_buffer(a; header, acquisition_sequence=UInt64(1))
    native_b = native_callback_buffer(b; header, acquisition_sequence=UInt64(1))
    native_c = native_callback_buffer(c; header, acquisition_sequence=UInt64(1))
    native_output = native_callback_buffer(output; header_available=true, properties_available=true)
    native_inputs = vcat(native_a, native_b, native_c)
    GC.@preserve a b c output native_inputs native_output begin
        inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(native_inputs)), UInt32(3))
        outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(native_output)), UInt32(1))
        callback(node.filter, inputs, outputs)
        @test output == 111 .* a
        @test synchronized.input_buffers.a == Float32[1 2 3; 4 5 6]
        @test synchronized.input_buffers.b == Float32[10 20 30; 40 50 60]
        @test PipeWireAO.buffer_header(outputs[1]) == header
        @test FilterGraphPipeWire._buffer_metadata(outputs[1]).value.acquisition_identity ==
            synchronized.input_metadata.a.value.acquisition_identity
    end
    @test FilterGraphs.synchronized_input_counters(synchronized).completed_windows == 1
    close(node)
end

@testset "row-major borrowed arrays preserve multidimensional indices" begin
    format = AlgorithmFormat(
        Float32,
        (2, 3),
        "test.image.f32/1",
        ColumnMajorLayout(),
    )
    binding = FilterGraphPipeWire.PortBinding(format, RowMajorLayout())
    storage = Float32[1, 2, 3, 4, 5, 6]
    GC.@preserve storage begin
        readable = FilterGraphPipeWire._borrow_pointer(
            binding,
            pointer(storage),
            Val(false),
        )
        @test collect(readable) == Float32[1 2 3; 4 5 6]

        writable = FilterGraphPipeWire._borrow_pointer(
            binding,
            pointer(storage),
            Val(true),
        )
        writable[2, 1] = 9
        writable[1, 3] = 8
        @test storage == Float32[1, 2, 8, 9, 5, 6]
        @test collect(writable) == Float32[1 2 8; 9 5 6]
    end
end

@testset "rank-one boundary layout preserves Graph state across deactivate" begin
    graph = prepared_leaky_graph()
    node = PipeWireNode(
        graph;
        boundary_layout=RowMajorLayout(),
        connect=false,
    )
    prepare_callback = node.filter.callbacks.on_prepare
    process_callback = node.filter.callbacks.on_process
    deactivate_callback = node.filter.callbacks.on_deactivate
    prepare_callback(node.filter)

    input = ones(Float32, 8)
    output = zeros(Float32, 8)
    native_input = native_callback_buffer(input)
    native_output = native_callback_buffer(output)
    GC.@preserve input output native_input native_output begin
        inputs = NdArrayFilterBuffers{false}(
            Ptr{Cvoid}(pointer(native_input)),
            UInt32(1),
        )
        outputs = NdArrayFilterBuffers{true}(
            Ptr{Cvoid}(pointer(native_output)),
            UInt32(1),
        )
        process_callback(node.filter, inputs, outputs)
        @test output == fill(2.0f0, 8)

        deactivate_callback(node.filter)
        prepare_callback(node.filter)
        fill!(input, 3.0f0)
        process_callback(node.filter, inputs, outputs)
        @test output == fill(7.0f0, 8)
        @test leaky_integrator_state(graph.nodes[1].prepared.workspace) == output
    end
    close(node)
end

@testset "package surface" begin
    @test isempty(Test.detect_ambiguities(FilterGraphPipeWire; recursive=true))
    for name in (
        :PipeWireNode,
        :PreparedCapturedGraphReplacement,
        :connect!,
        :last_error,
        :node_id,
        :node_state,
        :prepare_replacement,
        :quit!,
        :replace_graph!,
        :replace_parameters!,
        :run!,
    )
        @test Base.isexported(FilterGraphPipeWire, name)
        @test Base.Docs.hasdoc(FilterGraphPipeWire, name)
    end
end

@testset "Parameter callback diagnostics are bounded and allocation-free" begin
    withenv("JULIA_FGPW_PARAMETER_TRACE_DIR" => nothing) do
        owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
        @test owner.parameter_trace === nothing
        @test FilterGraphPipeWire._trace_parameter_phase!(nothing, owner, UInt8(1)) === nothing
    end
    withenv("JULIA_FGPW_PARAMETER_TRACE_DIR" => "diagnostic-test") do
        owner = FilterGraphPipeWire.PreparedGraphOwner(prepared_control_graph())
        trace = owner.parameter_trace
        @test trace isa FilterGraphPipeWire.GraphParameterTrace
        @test length(trace.phase) == 64
        @test trace.count == 0
        trace.current_port = 2
        FilterGraphPipeWire._trace_parameter_phase!(trace, owner, UInt8(1))
        @test @allocated(FilterGraphPipeWire._trace_parameter_phase!(trace, owner, UInt8(2))) == 0
        @test FilterGraphPipeWire._stage_parameter_transaction!(
            owner, trace, Symbol("shape-to-hidden") => fill(2.0f0, 1, 2),
        )
        @test trace.phase[1:5] == UInt8[1, 2, 3, 4, 5]
        @test issorted(trace.time_ns[1:trace.count])
        @test all(==(2), trace.input_port[1:trace.count])
        @test trace.control_state[3:4] == fill(FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_PREPARING, 2)
        @test trace.parameter_state[3:4] == fill(FilterGraphPipeWire._PARAMETER_SLOT_PREPARING, 2)
        @test trace.requested[5, 1] == 1
        @test trace.active[5, 1] == 0
        @test !FilterGraphPipeWire._stage_parameter_transaction!(
            owner, trace, Symbol("shape-to-hidden") => fill(3.0f0, 1, 2),
        )
        @test trace.phase[6] == 6
        @test trace.control_state[6] == FilterGraphPipeWire._CONTROL_SLOT_PARAMETER_PENDING
        @test FilterGraphPipeWire._adopt_pending_parameters!(owner)
        FilterGraphPipeWire._trace_parameter_phase!(trace, owner, UInt8(1))
        @test trace.requested[7, 1] == trace.active[7, 1] == 1
        while trace.count < length(trace.phase)
            GC.safepoint()
            FilterGraphPipeWire._trace_parameter_phase!(trace, owner, UInt8(1))
        end
        @test @allocated(FilterGraphPipeWire._trace_parameter_phase!(trace, owner, UInt8(1))) == 0
        @test trace.count == 64
        @test trace.omitted == 1
        @test_throws ArgumentError FilterGraphPipeWire.GraphParameterTrace(1; capacity=0)
    end
end

include("cred2_ingress.jl")

include("cpu_progressive_parameters.jl")

include("row_major_calibration.jl")

include("buffer_properties.jl")

include("derived_metadata.jl")

include("required_metadata.jl")

include("application/runtests.jl")

include("aqua.jl")
