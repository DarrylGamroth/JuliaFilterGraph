# Metadata facts are copied while the input lease is active. Opaque optional
# records stay in a shared bounded PipeWireAO snapshot for held/captured reuse.
mutable struct RetainedMetadata
    records::PipeWireAO.MetadataSnapshot
    present::Bool
    augment::Bool
    metadata::FilterGraphAlgorithms.AdmittedSampleMetadata
    propagation::UInt32
    requirements::Vector{PipeWireAO.MetadataRequirement}
    required_facts::Vector{Symbol}
    produce_time::Bool
    produce_header::Bool
    publication_sequence::UInt64
    clock::PipeWireAO.MetadataClock
end
function RetainedMetadata()
    return RetainedMetadata(PipeWireAO.MetadataSnapshot(),false,false,FilterGraphAlgorithms._NO_SAMPLE_METADATA,UInt32(1),
        PipeWireAO.MetadataRequirement[],Symbol[],false,false,UInt64(0),PipeWireAO.METADATA_CLOCK_REALTIME)
end
_metadata_snapshots(bindings::NamedTuple) = map(_ -> RetainedMetadata(),bindings)
function _clear_metadata_snapshot!(snapshot::RetainedMetadata)
    PipeWireAO.clear_metadata!(snapshot.records)
    snapshot.present=false
    snapshot.augment=false
    snapshot.metadata=FilterGraphAlgorithms._NO_SAMPLE_METADATA
    return nothing
end
_clear_metadata_snapshots!(snapshots::NamedTuple) = foreach(_clear_metadata_snapshot!,values(snapshots))
@inline function _retain_metadata!(snapshot::RetainedMetadata,buffer,metadata)
    PipeWireAO.capture_metadata!(snapshot.records,buffer)
    snapshot.metadata=metadata
    snapshot.present=metadata.present
    return nothing
end

@inline _retain_input_metadata!(::Tuple{},inputs,::Tuple{},index) = nothing
@inline function _retain_input_metadata!(snapshots::Tuple,inputs,metadata::Tuple,index)
    PipeWireAO.input_available(inputs[index]) &&
        _retain_metadata!(first(snapshots),inputs[index],first(metadata))
    return _retain_input_metadata!(Base.tail(snapshots),inputs,Base.tail(metadata),index+1)
end

@inline function _cache_output_metadata!(destination::RetainedMetadata,source,metadata)
    destination.metadata = metadata
    compatible=destination.propagation==1 && source!==nothing && source.present && metadata.present && source.metadata.present
    same=compatible && _same_acquisition_facts(source.metadata.value,metadata.value)
    if !compatible || (!same && !_preserves_acquisition_facts(source.metadata.value,metadata.value))
        destination.present = false
        destination.augment = false
        PipeWireAO.clear_metadata!(destination.records)
        return nothing
    end
    PipeWireAO.capture_metadata!(destination.records,source.records)
    destination.present = true
    destination.augment = !same
    return nothing
end
@inline _cache_output_metadata!(::Tuple{},sources::Tuple,::Tuple{},::Tuple{}) = nothing
@inline function _cache_output_metadata!(destinations::Tuple,sources::Tuple,indices::Tuple,metadata::Tuple)
    index = first(indices)
    source = iszero(index) ? nothing : getfield(sources,index)
    _cache_output_metadata!(first(destinations),source,first(metadata))
    return _cache_output_metadata!(Base.tail(destinations),sources,Base.tail(indices),Base.tail(metadata))
end

@inline function _publish_metadata!(buffer,snapshot::RetainedMetadata)
    PipeWireAO.copy_metadata!(buffer,snapshot.records)
    return nothing
end

function _produce_output_time!(buffer,snapshot::RetainedMetadata)
    snapshot.produce_time || return nothing
    PipeWireAO.metadata_record(buffer,PipeWireAO.SPA.META_REFERENCE_TIMESTAMP)===nothing && return nothing
    PipeWireAO.reference_timestamp(buffer)===nothing || return nothing
    stamp=try
        PipeWireAO.reference_timestamp_now(snapshot.clock)
    catch error
        throw(MetadataPublicationError(error))
    end
    PipeWireAO.set_reference_timestamp!(buffer,stamp)
    return nothing
end

struct MetadataPublicationError <: Exception
    cause::Exception
end
Base.showerror(io::IO,error::MetadataPublicationError)=(print(io,"metadata publication failed: ");showerror(io,error.cause))

# Header sequence/offset describe this transport block; all retained scientific
# records, including opaque extensions, belong to the complete row product.
function _same_frame_records(snapshot::RetainedMetadata,buffer)
    incoming=PipeWireAO.metadata_records(buffer)
    retained=PipeWireAO.metadata_records(snapshot.records)
    count=0
    for record in incoming
        type=PipeWireAO.metadata_type(record)
        (type==PipeWireAO.SPA.META_HEADER || PipeWireAO.metadata_is_transport_owned(type)) && continue
        previous=PipeWireAO.metadata_record(retained,type)
        previous===nothing && return false
        PipeWireAO.metadata_bytes(previous)==PipeWireAO.metadata_bytes(record) || return false
        count+=1
    end
    return count==length(retained)-1
end
