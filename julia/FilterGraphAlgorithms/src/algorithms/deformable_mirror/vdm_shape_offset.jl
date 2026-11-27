"""
    VdmShapeOffsetPlan(active_to_full, pdm_to_vdm, shape_offset)

Project a physical-command-sized shape offset into independent VDM coordinates.
The first unit row for each column of `active_to_full` defines the direct
selector S, with S*C = I. The adopted active offset is S*pdm_to_vdm*shape_offset;
least-squares constraint-feedback completion inverses are deliberately separate.
Initial matrices and offsets are copied. Offset replacements share the owned,
logically immutable calibration arrays and project only the new offset. Callers
must pair these maps with the downstream completion and physical projection from
the same calibration. Projection runs during parameter adoption.
"""
struct VdmShapeOffsetPlan <: AbstractAlgorithmPlan
    active_to_full::Matrix{Float32}
    pdm_to_vdm::Matrix{Float32}
    shape_projection::Matrix{Float32}
    controlled_actuators::Vector{Bool}
    shape_offset::Vector{Float32}
    active_offset::Vector{Float32}
end

function VdmShapeOffsetPlan(completion::AbstractMatrix{Float32}, inverse::AbstractMatrix{Float32},
    offset::AbstractVector{Float32})
    Base.require_one_based_indexing(completion, inverse, offset)
    !isempty(completion) && !isempty(inverse) || throw(ArgumentError("shape projections must be nonempty"))
    size(completion, 1) == size(inverse, 1) && size(inverse, 2) == length(offset) ||
        throw(DimensionMismatch("shape completion, inverse and physical offset extents differ"))
    all(isfinite, completion) && all(isfinite, inverse) && all(isfinite, offset) ||
        throw(ArgumentError("shape projections and offset must be finite"))
    owned_completion = Matrix(completion)
    controlled, _ = _vdm_completion_roles(owned_completion)
    all(!iszero, controlled) || throw(ArgumentError("shape completion requires one direct unit row per independent coordinate"))
    owned_inverse = Matrix(inverse)
    projection = owned_inverse[controlled, :]
    mask = Vector{Bool}(vec(any(!iszero, projection; dims=1)))
    owned_offset = Vector(offset)
    active = projection * owned_offset
    all(isfinite, active) || throw(ArgumentError("projected shape offset must be finite"))
    return VdmShapeOffsetPlan(owned_completion, owned_inverse, projection, mask, owned_offset, active)
end

_replace_shape_completion(plan::VdmShapeOffsetPlan, values) =
    VdmShapeOffsetPlan(values, plan.pdm_to_vdm, plan.shape_offset)
_replace_shape_inverse(plan::VdmShapeOffsetPlan, values) =
    VdmShapeOffsetPlan(plan.active_to_full, values, plan.shape_offset)
function _replace_shape_offset(plan::VdmShapeOffsetPlan, values::AbstractVector{Float32})
    Base.require_one_based_indexing(values)
    length(values)==length(plan.shape_offset) || throw(DimensionMismatch("physical shape offset extent differs"))
    all(isfinite,values) || throw(ArgumentError("shape offset must be finite"))
    offset=Vector(values)
    active=plan.shape_projection*offset
    all(isfinite,active) || throw(ArgumentError("projected shape offset must be finite"))
    # Calibration arrays are owned and logically immutable across adoptions.
    return VdmShapeOffsetPlan(plan.active_to_full,plan.pdm_to_vdm,plan.shape_projection,
        plan.controlled_actuators,offset,active)
end

_vdm_shape_storage_alias(destination,plan::VdmShapeOffsetPlan) =
    Base.mightalias(destination,plan.active_to_full) || Base.mightalias(destination,plan.pdm_to_vdm) ||
    Base.mightalias(destination,plan.shape_projection) || Base.mightalias(destination,plan.controlled_actuators) ||
    Base.mightalias(destination,plan.shape_offset) || Base.mightalias(destination,plan.active_offset)

function _process_vdm_shape_offset!(output::AbstractVector{Float32}, current::AbstractVector{Float32},
    workspace::VectorLinearCombinationWorkspace, plan::VdmShapeOffsetPlan, requested::AbstractVector{Float32})
    Base.require_one_based_indexing(output, current, requested)
    length(output) == length(requested) == length(plan.active_offset) == length(workspace.candidate_output) &&
        length(current) == length(plan.shape_offset) || throw(DimensionMismatch("shape offset output extents differ"))
    (_vdm_shape_storage_alias(output,plan) || _vdm_shape_storage_alias(current,plan) ||
        Base.mightalias(output,workspace.candidate_output) || Base.mightalias(current,workspace.candidate_output)) &&
        throw(ArgumentError("shape outputs must not alias prepared storage"))
    (Base.mightalias(current, output) || Base.mightalias(current, requested) ||
        Base.mightalias(requested, workspace.candidate_output)) &&
        throw(ArgumentError("shape readback, commands and scratch must not alias"))
    for index in eachindex(requested, plan.active_offset, workspace.candidate_output)
        isfinite(requested[index]) || return VectorLinearCombinationNonFiniteInput
        value = _pdm_offset_sum(requested[index], plan.active_offset[index])
        isfinite(value) || return VectorLinearCombinationNonFiniteOutput
        workspace.candidate_output[index] = value
    end
    copyto!(output, workspace.candidate_output)
    copyto!(current, plan.shape_offset)
    return nothing
end

"""
Add an adopted physical shape in independent VDM coordinates before completion.
The current-offset output reports the requested physical shape, not its completed
physical command. Both outputs inherit requested metadata and commit together.
The calibrated inverse defines units; no unit conversion is inferred. Reset
clears scratch and retains the adopted shape and projections.
"""
@algorithm VdmShapeOffsetF32 begin
    label = "vdm-shape-offset-f32"
    failure = VectorLinearCombinationFailure
    @configuration VdmShapeOffsetF32Configuration begin
        active_vdm_count::Int => graph_rebuild
        full_vdm_count::Int => graph_rebuild
        physical_actuator_count::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:requested, Float32, (configuration.active_vdm_count,), VDM_COMMAND_V1),
        output(:output, Float32, (configuration.active_vdm_count,), VDM_COMMAND_V1; metadata_from=:requested),
        output(:current_offset, Float32, (configuration.physical_actuator_count,), REQUESTED_PDM_COMMAND_V1;
            name="current-offset", metadata_from=:requested),
        parameter(:shape_offset, Float32, (configuration.physical_actuator_count,), REQUESTED_PDM_COMMAND_V1;
            name="shape-offset", replace=_replace_shape_offset),
        parameter(:active_to_full, Float32, (configuration.full_vdm_count, configuration.active_vdm_count),
            ACTIVE_TO_FULL_VDM_PROJECTION_V1; name="active-to-full", replace=_replace_shape_completion),
        parameter(:pdm_to_vdm, Float32, (configuration.full_vdm_count, configuration.physical_actuator_count),
            PDM_TO_VDM_PROJECTION_V1; name="pdm-to-vdm", replace=_replace_shape_inverse),
    )
    prepare(configuration) = begin
        _validate_vdm_to_pdm_dimensions(configuration.active_vdm_count, configuration.full_vdm_count,
            configuration.physical_actuator_count)
        configuration.active_vdm_count <= configuration.full_vdm_count ||
            throw(ArgumentError("shape independent extent exceeds full VDM extent"))
        # A valid neutral builder permits individually adopted calibration ports.
        completion = Matrix{Float32}(I, configuration.full_vdm_count, configuration.active_vdm_count)
        plan = VdmShapeOffsetPlan(completion,
            zeros(Float32, configuration.full_vdm_count, configuration.physical_actuator_count),
            zeros(Float32, configuration.physical_actuator_count))
        workspace = VectorLinearCombinationWorkspace(VectorLinearCombinationPlan(configuration.active_vdm_count, 1f0, 1f0))
        (;plan, workspace)
    end
    process!(plan, workspace) = _process_vdm_shape_offset!(output, current_offset, workspace, plan, requested)
end
