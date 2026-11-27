"""
    VdmActuatorRole

Classify one coordinate in canonical full-VDM order.

`ControlledVdmActuator` is an independent controller output,
`ExtrapolatedVdmActuator` is completed from controlled coordinates, and
`FixedVdmActuator` receives a separately prepared value. These are control
roles, not physical-device health states.
"""
@enum VdmActuatorRole::UInt8 begin
    ControlledVdmActuator
    ExtrapolatedVdmActuator
    FixedVdmActuator
end

@doc "Control role for an independent VDM coordinate produced by the controller."
ControlledVdmActuator
@doc "Control role for a dependent VDM coordinate produced by completion."
ExtrapolatedVdmActuator
@doc "Control role for a VDM coordinate assigned a separately prepared value."
FixedVdmActuator

"""
    VdmActuatorPartition(roles)

Prepare one exhaustive control role for every coordinate in canonical
full-VDM order.

The packed controlled, extrapolated, and fixed indices preserve canonical
full-VDM order. Julia indices are one-based. A different controller order
requires an explicit controller-to-VDM projection rather than a hidden
permutation. The constructor copies `roles` and requires a nonempty full VDM
with at least one controlled coordinate.
"""
struct VdmActuatorPartition
    roles::Vector{VdmActuatorRole}
    controlled_indices::Vector{Int}
    extrapolated_indices::Vector{Int}
    fixed_indices::Vector{Int}
end

function VdmActuatorPartition(
    roles::AbstractVector{VdmActuatorRole},
)
    prepared_roles = collect(roles)
    isempty(prepared_roles) &&
        throw(ArgumentError("full VDM actuator partition must be nonempty"))

    controlled_indices = Int[]
    extrapolated_indices = Int[]
    fixed_indices = Int[]
    for (index, role) in enumerate(prepared_roles)
        if role == ControlledVdmActuator
            push!(controlled_indices, index)
        elseif role == ExtrapolatedVdmActuator
            push!(extrapolated_indices, index)
        else
            push!(fixed_indices, index)
        end
    end
    isempty(controlled_indices) &&
        throw(
            ArgumentError(
                "VDM actuator partition has no controlled actuator",
            ),
        )
    return VdmActuatorPartition(
        prepared_roles,
        controlled_indices,
        extrapolated_indices,
        fixed_indices,
    )
end

"""
    vdm_actuator_roles(partition::VdmActuatorPartition)

Return a copy of the control roles in canonical full-VDM order.
"""
vdm_actuator_roles(partition::VdmActuatorPartition) = copy(partition.roles)

"""
    controlled_vdm_indices(partition::VdmActuatorPartition)

Return a copy of the one-based controlled indices in full-VDM order.
"""
controlled_vdm_indices(partition::VdmActuatorPartition) =
    copy(partition.controlled_indices)

"""
    extrapolated_vdm_indices(partition::VdmActuatorPartition)

Return a copy of the one-based extrapolated indices in full-VDM order.
"""
extrapolated_vdm_indices(partition::VdmActuatorPartition) =
    copy(partition.extrapolated_indices)

"""
    fixed_vdm_indices(partition::VdmActuatorPartition)

Return a copy of the one-based fixed indices in full-VDM order.
"""
fixed_vdm_indices(partition::VdmActuatorPartition) =
    copy(partition.fixed_indices)

"""
    full_vdm_count(partition::VdmActuatorPartition)

Return the number of coordinates in the full VDM.
"""
full_vdm_count(partition::VdmActuatorPartition) = length(partition.roles)
