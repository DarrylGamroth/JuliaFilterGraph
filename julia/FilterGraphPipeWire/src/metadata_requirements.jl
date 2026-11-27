# Resolve scientific obligations at preparation time; no transport identifiers
# enter the numerical Algorithms package.
_fact_type(fact::Symbol)=fact===:acquisition_identity ? PipeWireAO.SPA.META_ACQUISITION_IDENTITY :
    fact===:exposure_duration_nanoseconds ? PipeWireAO.SPA.META_EXPOSURE_TIMING :
    fact===:chopped_image_kind ? PipeWireAO.SPA.META_OPTICAL_CHOPPING :
    throw(ArgumentError("unsupported required sample fact $fact"))

"""
    required_input_metadata(graph; produced=NamedTuple())

Resolve Algorithm-required semantic facts through declared metadata provenance.
Return a NamedTuple of required record tuples for public inputs; Header is implicit.
`produced=(raw=(:chopped_image_kind,),)` explicitly declares facts supplied by an
input adapter instead of an incoming record. Actual facts are still checked after
that adapter and before any Graph Algorithm runs.
"""
function required_input_metadata(graph;produced::NamedTuple=NamedTuple())
    facts=FilterGraphs.required_input_sample_metadata(graph)
    all(name -> haskey(facts,name),keys(produced)) || throw(ArgumentError("produced facts refer to an unknown graph input"))
    for supplied_facts in values(produced), fact in supplied_facts
        _fact_type(fact)
    end
    return map(keys(facts),values(facts)) do name,required
        supplied=haskey(produced,name) ? getproperty(produced,name) : ()
        Tuple(PipeWireAO.MetadataRequirement(_fact_type(fact);required=true) for fact in required if !(fact in supplied))
    end |> result -> NamedTuple{keys(facts)}(result)
end
_prepare_required_facts!(owner)=nothing
function _prepare_required_facts!(owner::Union{PreparedGraphOwner,PreparedCapturedGraphOwner})
    facts=FilterGraphs.required_input_sample_metadata(owner.graph)
    for (name,snapshot) in pairs(owner.input_records)
        snapshot.required_facts=collect(Symbol,getproperty(facts,name))
    end
    return nothing
end

function _prepare_required_facts!(slot::PreparedCapturedGraphSlot)
    _prepare_required_facts!(slot.first)
    _prepare_required_facts!(slot.second)
    return nothing
end

@inline _check_input_facts!(::Tuple{},::Tuple{})=nothing
@inline function _check_input_facts!(snapshots::Tuple,metadata::Tuple)
    required=first(snapshots).required_facts
    sample=first(metadata)
    for fact in required
        valid=sample.present && (fact===:acquisition_identity ? sample_acquisition_identity(sample.value)!==nothing :
            fact===:exposure_duration_nanoseconds ? sample_exposure_duration_nanoseconds(sample.value)!==nothing :
            fact===:chopped_image_kind ? (sample_chopped_image_kind(sample.value)!==nothing &&
                sample_chopped_image_kind(sample.value)!=FilterGraphAlgorithms.ChoppedImageAbsent) : false)
        valid || throw(InvalidStateException("required input sample fact $fact is unavailable",:metadata))
    end
    return _check_input_facts!(Base.tail(snapshots),Base.tail(metadata))
end
_preflight_input_facts!(owner,inputs)=nothing
function _preflight_input_facts!(owner::PreparedGraphOwner,inputs)
    owner.join===nothing && owner.feedback===nothing && return nothing
    return _preflight_input_facts!(values(owner.input_records),inputs,1)
end
@inline _preflight_input_facts!(::Tuple{},inputs,index)=nothing
@inline function _preflight_input_facts!(snapshots::Tuple,inputs,index)
    if !isempty(first(snapshots).required_facts) && PipeWireAO.input_available(inputs[index])
        _check_input_facts!((first(snapshots),),(_buffer_metadata(inputs[index]),))
    end
    return _preflight_input_facts!(Base.tail(snapshots),inputs,index+1)
end

_preflight_input_facts!(owner::PreparedCapturedGraphOwner,inputs)=
    _preflight_input_facts!(values(owner.input_records),inputs,1)

required_input_metadata(slot::PreparedCapturedGraphSlot;produced::NamedTuple=NamedTuple())=
    required_input_metadata(slot.first.graph;produced)
