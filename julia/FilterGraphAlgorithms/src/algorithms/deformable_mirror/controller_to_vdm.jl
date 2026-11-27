"""
    ControllerToVdmPlan(controller_to_vdm)

Prepare the projection from concatenated controller-command coordinates to
active virtual-deformable-mirror (VDM) coordinates. Rows select VDM
coordinates and columns select controller coordinates. The projection is
copied into dense Julia storage and must be nonempty and finite.
"""
struct ControllerToVdmPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    controller_to_vdm::Matrix{T}
    controller_to_vdm_execution::CpuProjectionExecution

    function ControllerToVdmPlan(
        controller_to_vdm::AbstractMatrix{T},
    ) where {T<:Union{Float32,Float64}}
        isempty(controller_to_vdm) &&
            throw(ArgumentError("controller-to-VDM projection must be nonempty"))
        all(isfinite, controller_to_vdm) || throw(
            ArgumentError("controller-to-VDM projection must be finite"),
        )
        projection = Matrix(controller_to_vdm)
        return new{T}(projection, _cpu_projection_execution(projection))
    end
end

"""Expected numerical rejection from controller-to-VDM projection."""
@enum ControllerToVdmFailure::UInt8 begin
    ControllerToVdmNonFiniteInput = 1
    ControllerToVdmNonFiniteOutput = 2
end

@doc "The controller-command input contains a non-finite value." ControllerToVdmNonFiniteInput
@doc "Controller-to-VDM matrix multiplication produced a non-finite command." ControllerToVdmNonFiniteOutput

function _process_controller_to_vdm!(
    vdm_command::AbstractVector{T},
    plan::ControllerToVdmPlan{T},
    controller_command::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(vdm_command, controller_command)
    projection = plan.controller_to_vdm
    length(vdm_command) == size(projection, 1) ||
        throw(DimensionMismatch("VDM command has an incompatible extent"))
    length(controller_command) == size(projection, 2) ||
        throw(DimensionMismatch("controller command has an incompatible extent"))
    Base.mightalias(vdm_command, controller_command) && throw(
        ArgumentError("controller and VDM command storage must not alias"),
    )
    Base.mightalias(vdm_command, projection) && throw(
        ArgumentError("VDM command must not alias the prepared projection"),
    )

    for value in controller_command
        isfinite(value) || return ControllerToVdmNonFiniteInput
    end
    _cpu_project!(
        vdm_command,
        projection,
        controller_command,
        plan.controller_to_vdm_execution,
    )
    for value in vdm_command
        isfinite(value) || return ControllerToVdmNonFiniteOutput
    end
    return nothing
end

function _validate_controller_to_vdm_dimensions(
    controller_command_count::Int,
    active_vdm_count::Int,
)
    controller_command_count > 0 ||
        throw(ArgumentError("controller_command_count must be positive"))
    active_vdm_count > 0 || throw(ArgumentError("active_vdm_count must be positive"))
    max_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    encoded_controller_count = UInt64(controller_command_count)
    encoded_active_vdm_count = UInt64(active_vdm_count)
    encoded_controller_count <= max_elements || throw(
        ArgumentError("controller command byte extent must fit UInt32"),
    )
    encoded_active_vdm_count <= div(max_elements, encoded_controller_count) || throw(
        ArgumentError("controller-to-VDM projection byte extent must fit UInt32"),
    )
    return nothing
end

"""
Map a controller-command vector to a virtual deformable-mirror (VDM) command
by the configured `controller_to_vdm` matrix. Input and output values use their
respective command schemas; matrix coefficients supply the unit conversion.
"""
@algorithm ControllerToVdmF32 begin
    label = "controller-to-vdm-f32"
    failure = ControllerToVdmFailure

    """
    Set `controller_command_count` and `active_vdm_count`, the input and output vector
    lengths. The replacement matrix has shape
    `(active_vdm_count, controller_command_count)`.
    """
    @configuration ControllerToVdmF32Configuration begin
        controller_command_count::Int => graph_rebuild
        active_vdm_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :controller_command,
            schema_element_type(CONTROLLER_COMMAND_V1),
            (configuration.controller_command_count,),
            CONTROLLER_COMMAND_V1;
            name="controller-command",
        ),
        output(
            :vdm_command,
            schema_element_type(VDM_COMMAND_V1),
            (configuration.active_vdm_count,),
            VDM_COMMAND_V1;
            name="vdm-command",
            metadata_from=:controller_command,
        ),
        parameter(
            :controller_to_vdm,
            schema_element_type(CONTROLLER_TO_VDM_PROJECTION_V1),
            (
                configuration.active_vdm_count,
                configuration.controller_command_count,
            ),
            CONTROLLER_TO_VDM_PROJECTION_V1;
            name="controller-to-vdm",
            replace=(_plan, values) -> ControllerToVdmPlan(values),
        ),
    )

    prepare(configuration) = begin
        _validate_controller_to_vdm_dimensions(
            configuration.controller_command_count,
            configuration.active_vdm_count,
        )
        ControllerToVdmPlan(
            zeros(
                Float32,
                configuration.active_vdm_count,
                configuration.controller_command_count,
            ),
        )
    end

    process!(plan) = begin
        _process_controller_to_vdm!(vdm_command, plan, controller_command)
    end
end
