# Cold preparation metadata. Codes describe correction provenance, never health.
# The first unit row for every active coordinate establishes the controlled set.
# Ambiguous completion or mixed physical projections remain unknown (3).
_pdm_role_metadata_alias(array, ::Union{DenseProjection,IdentityProjection}) = false
_pdm_role_metadata_alias(array, execution::SelectionProjection) = Base.mightalias(array, execution.columns)
_pdm_role_metadata_alias(array, execution::CompletionProjection) = _cpu_completion_mightalias(array, execution)

function _vdm_completion_roles(completion::Matrix{T}) where {T}
    rows, columns = size(completion)
    controlled = zeros(Int, columns)
    full_roles = fill(UInt8(3), rows)
    for row in 1:rows
        nonzero = findall(!iszero, view(completion, row, :))
        isempty(nonzero) && (full_roles[row] = 0; continue)
        if length(nonzero) == 1 && completion[row, only(nonzero)] == one(T)
            column = only(nonzero)
            if controlled[column] == 0
                controlled[column] = row
                full_roles[row] = 1
            end
        end
    end
    return controlled, full_roles
end

function _pdm_correction_roles(completion::Matrix{T}, projection::Matrix{T}) where {T}
    controlled, full_roles = _vdm_completion_roles(completion)
    if all(!iszero, controlled)
        for row in eachindex(full_roles)
            full_roles[row] == 3 && (full_roles[row] = 2)
        end
    else
        # An incomplete basis does not establish an actuator partition.
        for row in eachindex(full_roles)
            full_roles[row] == 1 && (full_roles[row] = 3)
        end
    end
    roles = fill(UInt8(3), size(projection, 1))
    for row in axes(projection, 1)
        nonzero = findall(!iszero, view(projection, row, :))
        isempty(nonzero) && (roles[row] = 0; continue)
        if length(nonzero) == 1 && projection[row, only(nonzero)] == one(T)
            roles[row] = full_roles[only(nonzero)]
        end
    end
    return roles
end

"""
    VdmToPdmPlan(active_to_full, vdm_to_pdm)

Prepare the active-to-full virtual-deformable-mirror (VDM) completion and the
full-VDM-to-physical-deformable-mirror (PDM) projection. Matrix rows select the
destination coordinates and columns select the source coordinates. Both
matrices are copied into dense Julia storage and must be nonempty, compatible,
and finite.
"""
struct VdmToPdmPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    active_to_full::Matrix{T}
    vdm_to_pdm::Matrix{T}
    active_to_full_execution::CpuCompletionExecution{T}
    vdm_to_pdm_execution::CpuProjectionExecution
    correction_roles::Vector{UInt8}

    function VdmToPdmPlan(
        active_to_full::AbstractMatrix{T},
        vdm_to_pdm::AbstractMatrix{T},
    ) where {T<:Union{Float32,Float64}}
        isempty(active_to_full) && throw(
            ArgumentError("active-to-full VDM projection must be nonempty"),
        )
        isempty(vdm_to_pdm) &&
            throw(ArgumentError("VDM-to-PDM projection must be nonempty"))
        size(vdm_to_pdm, 2) == size(active_to_full, 1) || throw(
            DimensionMismatch(
                "VDM-to-PDM input extent must match the full VDM extent",
            ),
        )
        all(isfinite, active_to_full) || throw(
            ArgumentError("active-to-full VDM projection must be finite"),
        )
        all(isfinite, vdm_to_pdm) ||
            throw(ArgumentError("VDM-to-PDM projection must be finite"))
        completion = Matrix(active_to_full)
        projection = Matrix(vdm_to_pdm)
        return new{T}(
            completion,
            projection,
            _cpu_completion_execution(completion, projection),
            _cpu_projection_execution(projection),
            _pdm_correction_roles(completion, projection),
        )
    end
end

"""Fixed full-VDM and extrapolation scratch owned by one Algorithm instance."""
struct VdmToPdmWorkspace{T,V<:AbstractVector{T}}
    full_vdm::V
    extrapolated_vdm::V
end

function VdmToPdmWorkspace(plan::VdmToPdmPlan{T}) where {T}
    rows, columns = size(plan.active_to_full)
    return VdmToPdmWorkspace(zeros(T, rows), zeros(T, max(rows - columns, 0)))
end

commit!(workspace::VdmToPdmWorkspace) = workspace
discard!(workspace::VdmToPdmWorkspace) = workspace

function reset!(workspace::VdmToPdmWorkspace)
    fill!(workspace.full_vdm, zero(eltype(workspace.full_vdm)))
    fill!(workspace.extrapolated_vdm, zero(eltype(workspace.extrapolated_vdm)))
    return workspace
end

"""Expected numerical rejection from VDM-to-PDM projection."""
@enum VdmToPdmFailure::UInt8 begin
    VdmToPdmNonFiniteInput = 1
    VdmToPdmNonFiniteOutput = 2
end

@doc "An active VDM command contains a non-finite component." VdmToPdmNonFiniteInput
@doc "Projection of an active VDM command to physical actuators produced a non-finite value." VdmToPdmNonFiniteOutput

function _process_vdm_to_pdm!(
    requested_pdm_command::AbstractVector{T},
    workspace::VdmToPdmWorkspace{T},
    plan::VdmToPdmPlan{T},
    vdm_command::AbstractVector{T},
    ::Val{CheckInput}=Val(true),
) where {T,CheckInput}
    Base.require_one_based_indexing(requested_pdm_command, vdm_command)
    active_to_full = plan.active_to_full
    vdm_to_pdm = plan.vdm_to_pdm
    full_vdm = workspace.full_vdm
    extrapolated_vdm = workspace.extrapolated_vdm
    length(vdm_command) == size(active_to_full, 2) ||
        throw(DimensionMismatch("active VDM command has an incompatible extent"))
    length(full_vdm) == size(active_to_full, 1) ||
        throw(DimensionMismatch("VDM-to-PDM Workspace has an incompatible extent"))
    length(extrapolated_vdm) == max(size(active_to_full, 1) - size(active_to_full, 2), 0) ||
        throw(DimensionMismatch("VDM completion Workspace has an incompatible extent"))
    length(requested_pdm_command) == size(vdm_to_pdm, 1) || throw(
        DimensionMismatch("requested PDM command has an incompatible extent"),
    )
    Base.mightalias(full_vdm, vdm_command) && throw(
        ArgumentError("VDM input must not alias full-VDM scratch"),
    )
    Base.mightalias(extrapolated_vdm, vdm_command) && throw(
        ArgumentError("VDM input must not alias completion scratch"),
    )
    Base.mightalias(requested_pdm_command, full_vdm) && throw(
        ArgumentError("PDM output must not alias full-VDM scratch"),
    )
    Base.mightalias(requested_pdm_command, extrapolated_vdm) && throw(
        ArgumentError("PDM output must not alias completion scratch"),
    )
    _cpu_completion_mightalias(requested_pdm_command, plan.active_to_full_execution) && throw(
        ArgumentError("PDM output must not alias completion metadata"),
    )
    Base.mightalias(extrapolated_vdm, vdm_to_pdm) && throw(
        ArgumentError("completion scratch must not alias the VDM-to-PDM projection"),
    )
    Base.mightalias(requested_pdm_command, active_to_full) && throw(
        ArgumentError("PDM output must not alias the active-to-full projection"),
    )
    Base.mightalias(requested_pdm_command, vdm_to_pdm) && throw(
        ArgumentError("PDM output must not alias the VDM-to-PDM projection"),
    )

    if CheckInput
        for value in vdm_command
            isfinite(value) || return VdmToPdmNonFiniteInput
        end
    end
    _cpu_complete!(
        full_vdm,
        active_to_full,
        vdm_command,
        plan.active_to_full_execution,
        extrapolated_vdm,
    )
    _cpu_project_intermediate!(
        requested_pdm_command,
        vdm_to_pdm,
        full_vdm,
        plan.vdm_to_pdm_execution,
    )
    for value in requested_pdm_command
        isfinite(value) || return VdmToPdmNonFiniteOutput
    end
    return nothing
end

function _replace_active_to_full(plan::VdmToPdmPlan, values::AbstractMatrix)
    return VdmToPdmPlan(values, plan.vdm_to_pdm)
end

function _replace_vdm_to_pdm(plan::VdmToPdmPlan, values::AbstractMatrix)
    return VdmToPdmPlan(plan.active_to_full, values)
end

function _validate_vdm_to_pdm_dimensions(
    active_vdm_count::Int,
    full_vdm_count::Int,
    physical_actuator_count::Int,
)
    active_vdm_count > 0 ||
        throw(ArgumentError("active_vdm_count must be positive"))
    full_vdm_count > 0 || throw(ArgumentError("full_vdm_count must be positive"))
    physical_actuator_count > 0 ||
        throw(ArgumentError("physical_actuator_count must be positive"))
    max_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    encoded_active_count = UInt64(active_vdm_count)
    encoded_full_count = UInt64(full_vdm_count)
    encoded_physical_count = UInt64(physical_actuator_count)
    encoded_active_count <= max_elements ||
        throw(ArgumentError("active VDM byte extent must fit UInt32"))
    encoded_full_count <= div(max_elements, encoded_active_count) || throw(
        ArgumentError("active-to-full projection byte extent must fit UInt32"),
    )
    encoded_physical_count <= div(max_elements, encoded_full_count) || throw(
        ArgumentError("VDM-to-PDM projection byte extent must fit UInt32"),
    )
    return nothing
end

"""
Expand an active VDM command into full VDM coordinates and map it to a
requested physical-actuator command. The two prepared matrices define the
coordinate and unit mappings.
"""
@algorithm VdmToPdmF32 begin
    label = "vdm-to-pdm-f32"
    failure = VdmToPdmFailure

    """
    Set active/full VDM vector lengths and physical-actuator extent. The
    `active_to_full` matrix has shape `(full_vdm_count, active_vdm_count)`; the
    `vdm_to_pdm` matrix maps full VDM to physical actuators.
    """
    @configuration VdmToPdmF32Configuration begin
        active_vdm_count::Int => graph_rebuild
        full_vdm_count::Int => graph_rebuild
        physical_actuator_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :vdm_command,
            schema_element_type(VDM_COMMAND_V1),
            (configuration.active_vdm_count,),
            VDM_COMMAND_V1;
            name="vdm-command",
        ),
        output(
            :requested_pdm_command,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.physical_actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            name="requested-pdm-command",
            metadata_from=:vdm_command,
        ),
        output(
            :correction_role,
            schema_element_type(PDM_CORRECTION_ROLE_V1),
            (configuration.physical_actuator_count,),
            PDM_CORRECTION_ROLE_V1;
            name="correction-role",
            metadata_from=:vdm_command,
        ),
        parameter(
            :active_to_full,
            schema_element_type(ACTIVE_TO_FULL_VDM_PROJECTION_V1),
            (configuration.full_vdm_count, configuration.active_vdm_count),
            ACTIVE_TO_FULL_VDM_PROJECTION_V1;
            name="active-to-full",
            replace=(plan, values) -> _replace_active_to_full(plan, values),
        ),
        parameter(
            :vdm_to_pdm,
            schema_element_type(VDM_TO_PDM_PROJECTION_V1),
            (
                configuration.physical_actuator_count,
                configuration.full_vdm_count,
            ),
            VDM_TO_PDM_PROJECTION_V1;
            name="vdm-to-pdm",
            replace=(plan, values) -> _replace_vdm_to_pdm(plan, values),
        ),
    )

    prepare(configuration) = begin
        _validate_vdm_to_pdm_dimensions(
            configuration.active_vdm_count,
            configuration.full_vdm_count,
            configuration.physical_actuator_count,
        )
        plan = VdmToPdmPlan(
            zeros(
                Float32,
                configuration.full_vdm_count,
                configuration.active_vdm_count,
            ),
            zeros(
                Float32,
                configuration.physical_actuator_count,
                configuration.full_vdm_count,
            ),
        )
        (plan=plan, workspace=VdmToPdmWorkspace(plan))
    end

    process!(plan, workspace) = _process_vdm_to_pdm_with_roles!(
        requested_pdm_command, correction_role, workspace, plan, vdm_command)
end

function _process_vdm_to_pdm_with_roles!(requested_pdm_command, correction_role,
        workspace, plan, vdm_command, check_input::Val=Val(true))
    Base.require_one_based_indexing(correction_role)
    (Base.mightalias(correction_role, requested_pdm_command) ||
     Base.mightalias(correction_role, vdm_command) ||
     Base.mightalias(correction_role, plan.correction_roles) ||
     Base.mightalias(correction_role, plan.active_to_full) ||
     Base.mightalias(correction_role, plan.vdm_to_pdm) ||
     Base.mightalias(correction_role, workspace.full_vdm) ||
     Base.mightalias(correction_role, workspace.extrapolated_vdm) ||
     _pdm_role_metadata_alias(correction_role, plan.active_to_full_execution) ||
     _pdm_role_metadata_alias(correction_role, plan.vdm_to_pdm_execution)) &&
        throw(ArgumentError("correction-role output must own separate storage"))
    length(correction_role) == length(plan.correction_roles) ||
        throw(DimensionMismatch("PDM correction-role extent is incompatible"))
    status = _process_vdm_to_pdm!(
        requested_pdm_command,
        workspace,
        plan,
        vdm_command,
        check_input,
    )
    if status === nothing
        copyto!(correction_role, plan.correction_roles)
    end
    status
end

function _process_finite_input!(requested, roles, prepared::PreparedAlgorithm{VdmToPdmF32},
        input, _metadata)
    return _process_vdm_to_pdm_with_roles!(requested, roles, prepared.workspace, prepared.plan,
        input, Val(false))
end

# Preserve the numerical single-output API for callers that do not observe roles.
function process!(output::AbstractVector, prepared::PreparedAlgorithm{VdmToPdmF32}, input::AbstractVector)
    return _process_vdm_to_pdm!(output, prepared.workspace, prepared.plan, input)
end
