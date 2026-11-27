"""Prepared plan, workspace, and exact formats for one declaration."""
mutable struct PreparedAlgorithm{D<:AbstractAlgorithmDeclaration,P,W,F}
    plan::P
    workspace::W
    formats::F
end

PreparedAlgorithm(
    ::Type{D},
    plan::P,
    workspace::W,
    formats::F,
) where {D<:AbstractAlgorithmDeclaration,P,W,F} =
    PreparedAlgorithm{D,P,W,F}(plan, workspace, formats)

"""
    required_sample_metadata(prepared, input_name)

Return mandatory semantic facts for one Algorithm input. The default is empty;
requiring SampleMetadata for offsets or terminal markers adds no acquisition fact.
Transport adapters must check these facts before advancing a containing Graph.
"""
required_sample_metadata(prepared,input_name::Symbol)=()

capture_support(::Type{<:AbstractAlgorithmDeclaration}, device) = NoCapture()
capture_support(
    ::PreparedAlgorithm{D},
    device,
) where {D<:AbstractAlgorithmDeclaration} = capture_support(D, device)

"""Compact publication result for an Algorithm with conditional output."""
struct AlgorithmProcessResult
    complete::Bool
    has_metadata::Bool
    metadata::SampleMetadata

    function AlgorithmProcessResult(
        complete::Bool,
        has_metadata::Bool,
        metadata::SampleMetadata,
    )
        complete || !has_metadata || throw(
            ArgumentError("a deferred Algorithm result cannot carry output metadata"),
        )
        return new(complete, has_metadata, metadata)
    end
end

"""
    AlgorithmProcessComplete()
    AlgorithmProcessComplete(metadata::SampleMetadata)

Construct explicit completion, optionally replacing propagated output metadata.
"""
AlgorithmProcessComplete() = AlgorithmProcessResult(true, false, _EMPTY_SAMPLE_METADATA)
AlgorithmProcessComplete(metadata::SampleMetadata) =
    AlgorithmProcessResult(true, true, metadata)

"""
    AlgorithmProcessDeferred()

Construct successful processing with every conditional output absent.
"""
AlgorithmProcessDeferred() = AlgorithmProcessResult(false, false, _EMPTY_SAMPLE_METADATA)

"""
    prepare_algorithm(::Type{<:AbstractAlgorithmDeclaration}, configuration)
    prepare_algorithm(::Type{<:AbstractAlgorithmDeclaration}, configuration, device)
    prepare_algorithm(prepared::PreparedAlgorithm, device)

Prepare a native algorithm instance from typed configuration. The declaration
must have a matching configuration type; otherwise dispatch fails with
`MethodError`. Where an eager device implementation exists, the device forms
prepare an independent instance for arrays resident on the selected device.
When starting from a prepared instance, use its current parameter values and
start with fresh workspace state.
"""
function prepare_algorithm end

function prepare_algorithm(declaration::Type{<:AbstractAlgorithmDeclaration}, configuration, device)
    return prepare_algorithm(prepare_algorithm(declaration, configuration), device)
end

# Device adapters can reuse declaration preparation before applying host-only layouts.
function _prepare_declared_algorithm end

_prepare_host_plan(declaration, plan) = plan

"""Private warmed entry used after a graph has admitted exact arrays and formats."""
function _process_admitted! end

struct EmptyWorkspace end

reset!(workspace::EmptyWorkspace) = workspace
commit!(workspace::EmptyWorkspace) = workspace
discard!(workspace::EmptyWorkspace) = workspace

_plan_adopted!(workspace) = workspace

_prepared_plan_workspace(prepared::NamedTuple{(:plan,:workspace)}) =
    (prepared.plan, prepared.workspace)
_prepared_plan_workspace(plan) = (plan, EmptyWorkspace())

# A stateful declaration requires an explicit Plan/Workspace pair. The macro
# already knows whether process! declares a workspace; no numeric dispatch tag
# is needed to distinguish this checked form from the unary stateless helper.
function _prepared_plan_workspace(declaration, prepared::NamedTuple{names}) where {names}
    if names === (:plan, :workspace) || names === (:workspace, :plan)
        return prepared.plan, prepared.workspace
    end
    return _invalid_workspace_preparation(declaration, prepared)
end

_prepared_plan_workspace(declaration, prepared) =
    _invalid_workspace_preparation(declaration, prepared)

function _invalid_workspace_preparation(declaration, prepared)
    received = prepared isa NamedTuple ? "fields $(keys(prepared))" : string(typeof(prepared))
    throw(ArgumentError(
        "$declaration prepare must return exactly the named fields plan and workspace " *
        "for process!(plan, workspace), in either order; received $received",
    ))
end

function _validate_workspace_lifecycle(declaration, workspace)
    for hook in (commit!, discard!, reset!)
        applicable(hook, workspace) || throw(ArgumentError(
            "$declaration Workspace $(typeof(workspace)) must implement $(nameof(hook))(workspace)",
        ))
    end
    return nothing
end

function _validate_failure_type(declaration::Type{<:AbstractAlgorithmDeclaration})
    failure_type = algorithm_failure_type(declaration)
    failure_type === Union{} || (failure_type isa DataType && failure_type <: Enum) ||
        throw(ArgumentError("algorithm failure must be a concrete Enum type"))
    return failure_type
end

_validate_process_result(::Type{<:AbstractAlgorithmDeclaration}, ::Nothing) = nothing

function _validate_process_result(
    declaration::Type{<:AbstractAlgorithmDeclaration},
    result::AlgorithmProcessResult,
)
    if !result.complete
        ports = algorithm_ports(declaration)
        any(port -> port.direction isa OutputPort, ports) &&
            all(port -> !(port.direction isa OutputPort) || port.conditional, ports) ||
            throw(ArgumentError(
                "AlgorithmProcessDeferred() requires every output Port to be conditional",
            ))
    end
    return result
end

function _validate_process_result(declaration::Type{<:AbstractAlgorithmDeclaration}, result)
    result isa algorithm_failure_type(declaration) ||
        throw(ArgumentError("algorithm process! returned an undeclared result $(typeof(result))"))
    return result
end

_finish_workspace!(workspace, ::Nothing) = commit!(workspace)
_finish_workspace!(workspace, result::AlgorithmProcessResult) = result.complete && commit!(workspace)
_finish_workspace!(workspace, ::Enum) = discard!(workspace)

function _finish_checked_process!(
    declaration::Type{<:AbstractAlgorithmDeclaration},
    workspace,
    result,
)
    try
        _validate_process_result(declaration, result)
    catch
        discard!(workspace)
        rethrow()
    end
    _finish_workspace!(workspace, result)
    return result
end

"""
    replace_parameter!(prepared::PreparedAlgorithm, name, values::AbstractArray)

Install a sparse ndarray parameter in a prepared native declaration. The name
must identify a declared sparse parameter, and the array must match its declared
element type and shape.
"""
function replace_parameter! end

function _prepare_parameter_replacement end

_replace_prepared_parameter(replacement, plan, values) = replacement(plan, values)

replace_parameter!(prepared::PreparedAlgorithm, name::Symbol, values::AbstractArray) =
    replace_parameter!(prepared, Val(name), values)

replace_parameter!(prepared::PreparedAlgorithm, name::AbstractString, values::AbstractArray) =
    replace_parameter!(prepared, Symbol(name), values)

"""
    algorithm_formats(prepared::PreparedAlgorithm)

Return exact prepared formats in declaration order.
"""
algorithm_formats(prepared::PreparedAlgorithm) = prepared.formats

"""
    algorithm_format(prepared::PreparedAlgorithm, name::Union{Symbol,AbstractString})

Return the prepared format for the canonical graph port `name`, including sparse
parameters. Use the port's declared `name` (for example, `"vdm-command"`), rather
than its local Julia binding. An unknown name throws `ArgumentError`.

Declarations created with [`@algorithm`](@ref) generate direct format access
for statically known Symbols, for example `algorithm_format(prepared, Symbol("vdm-command"))`.
String names are a convenience for dynamic preparation code. The tuple returned by
[`algorithm_formats`](@ref) remains available for iteration in declaration order.
"""
@inline algorithm_format(prepared::PreparedAlgorithm, name::Symbol) =
    algorithm_format(prepared, Val(name))

@inline algorithm_format(prepared::PreparedAlgorithm, name::AbstractString) =
    algorithm_format(prepared, Symbol(name))

# Support hand-written declarations as well as those generated by @algorithm.
function algorithm_format(prepared::PreparedAlgorithm{D}, ::Val{name}) where {D,name}
    ports = algorithm_ports(D)
    for (index, port) in pairs(ports)
        port.name === name && return prepared.formats[index]
    end
    throw(ArgumentError("unknown port $(repr(name)) for $(algorithm_label(D))"))
end
