# Feedback embeddings place each input in one distinct controller coordinate
# and leave other coordinates zero. Metadata shares the Plan's owned,
# logically read-only lifetime; parameter replacement rebuilds it.
struct EmbeddingProjection
    rows::Vector{Int}
end

_cpu_feedback_mightalias(output, ::CpuProjectionExecution) = false
_cpu_feedback_mightalias(output, execution::EmbeddingProjection) =
    Base.mightalias(output, execution.rows)

function _cpu_feedback_projection_execution(projection::Matrix{T}) where {T<:Union{Float32,Float64}}
    rows, columns = size(projection)
    rows > columns || return _cpu_projection_execution(projection)
    selected_rows = Vector{Int}(undef, columns)
    # Inspect stored bits: DAZ must not classify a subnormal coefficient as
    # zero. Negative-zero coefficients retain the original dense arithmetic.
    bits = reinterpret(T === Float32 ? UInt32 : UInt64, projection)
    unit = reinterpret(eltype(bits), one(T))
    for column in 1:columns
        selected = 0
        for row in 1:rows
            coefficient = bits[row, column]
            if coefficient == unit
                iszero(selected) || return _cpu_projection_execution(projection)
                selected = row
            elseif !iszero(coefficient)
                return _cpu_projection_execution(projection)
            end
        end
        iszero(selected) && return _cpu_projection_execution(projection)
        selected_rows[column] = selected
    end
    allunique(selected_rows) || return _cpu_projection_execution(projection)
    return EmbeddingProjection(selected_rows)
end

function _cpu_project!(output, projection, input, execution::EmbeddingProjection)
    # Multiplication can flush subnormal inputs/results; a copy cannot. Retain
    # dense execution whenever the caller enables that host arithmetic mode.
    if Base.get_zero_subnormals()
        mul!(output, projection, input)
        return nothing
    end
    fill!(output, zero(eltype(output)))
    for column in eachindex(execution.rows)
        value = input[column]
        output[execution.rows[column]] = iszero(value) ? zero(value) : value
    end
    return nothing
end

"""
    VdmFeedbackToControllerPlan(vdm_to_controller)

Prepare the projection from active virtual-deformable-mirror (VDM) constraint
feedback to concatenated controller coordinates. Rows select controller
coordinates and columns select VDM coordinates. The projection is copied into
dense Julia storage and must be nonempty and finite. CPU execution prepares
exact unit-coordinate embeddings with positive-zero padding as indexed copies;
subnormal-flushing mode retains dense multiplication. GPU preparation uses the
embedding with disjoint buffers and otherwise uses the dense/identity/selection
classifier.
"""
struct VdmFeedbackToControllerPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    vdm_to_controller::Matrix{T}
    vdm_to_controller_execution::Union{CpuProjectionExecution,EmbeddingProjection}

    function VdmFeedbackToControllerPlan(
        vdm_to_controller::AbstractMatrix{T},
    ) where {T<:Union{Float32,Float64}}
        isempty(vdm_to_controller) && throw(
            ArgumentError("VDM-to-controller projection must be nonempty"),
        )
        all(isfinite, vdm_to_controller) || throw(
            ArgumentError("VDM-to-controller projection must be finite"),
        )
        projection = Matrix(vdm_to_controller)
        return new{T}(projection, _cpu_feedback_projection_execution(projection))
    end
end

"""Expected numerical rejection from VDM-to-controller feedback projection."""
@enum VdmFeedbackToControllerFailure::UInt8 begin
    VdmFeedbackToControllerNonFiniteInput = 1
    VdmFeedbackToControllerNonFiniteOutput = 2
end

@doc "VDM constraint feedback contains a non-finite component." VdmFeedbackToControllerNonFiniteInput
@doc "Projection of VDM feedback to controller coordinates produced a non-finite value." VdmFeedbackToControllerNonFiniteOutput

function _process_vdm_feedback_to_controller!(
    controller_constraint_feedback::AbstractVector{T},
    plan::VdmFeedbackToControllerPlan{T},
    vdm_constraint_feedback::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(
        controller_constraint_feedback,
        vdm_constraint_feedback,
    )
    projection = plan.vdm_to_controller
    length(controller_constraint_feedback) == size(projection, 1) || throw(
        DimensionMismatch(
            "controller constraint feedback has an incompatible extent",
        ),
    )
    length(vdm_constraint_feedback) == size(projection, 2) || throw(
        DimensionMismatch("VDM constraint feedback has an incompatible extent"),
    )
    Base.mightalias(controller_constraint_feedback, vdm_constraint_feedback) &&
        throw(
            ArgumentError(
                "VDM and controller constraint-feedback storage must not alias",
            ),
        )
    (Base.mightalias(controller_constraint_feedback, projection) ||
     _cpu_feedback_mightalias(controller_constraint_feedback, plan.vdm_to_controller_execution)) && throw(
        ArgumentError(
            "controller constraint feedback must not alias the prepared projection",
        ),
    )

    for value in vdm_constraint_feedback
        isfinite(value) || return VdmFeedbackToControllerNonFiniteInput
    end
    _cpu_project!(
        controller_constraint_feedback,
        projection,
        vdm_constraint_feedback,
        plan.vdm_to_controller_execution,
    )
    for value in controller_constraint_feedback
        isfinite(value) || return VdmFeedbackToControllerNonFiniteOutput
    end
    return nothing
end

function _validate_vdm_feedback_to_controller_dimensions(
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
        ArgumentError("controller feedback byte extent must fit UInt32"),
    )
    encoded_active_vdm_count <= div(max_elements, encoded_controller_count) || throw(
        ArgumentError("VDM-to-controller projection byte extent must fit UInt32"),
    )
    return nothing
end

"""
Map VDM constraint feedback back into controller coordinates using the
`vdm_to_controller` matrix. Matrix coefficients determine any unit conversion.
"""
@algorithm VdmFeedbackToControllerF32 begin
    label = "vdm-feedback-to-controller-f32"
    failure = VdmFeedbackToControllerFailure

    """
    Set controller-command extent and VDM size; the `vdm_to_controller` parameter
    matrix has shape `(controller_command_count, active_vdm_count)`.
    """
    @configuration VdmFeedbackToControllerF32Configuration begin
        controller_command_count::Int => graph_rebuild
        active_vdm_count::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :vdm_constraint_feedback,
            schema_element_type(VDM_CONSTRAINT_FEEDBACK_V1),
            (configuration.active_vdm_count,),
            VDM_CONSTRAINT_FEEDBACK_V1;
            name="vdm-constraint-feedback",
        ),
        output(
            :controller_constraint_feedback,
            schema_element_type(CONTROLLER_CONSTRAINT_FEEDBACK_V1),
            (configuration.controller_command_count,),
            CONTROLLER_CONSTRAINT_FEEDBACK_V1;
            name="controller-constraint-feedback",
            metadata_from=:vdm_constraint_feedback,
        ),
        parameter(
            :vdm_to_controller,
            schema_element_type(VDM_TO_CONTROLLER_PROJECTION_V1),
            (configuration.controller_command_count, configuration.active_vdm_count),
            VDM_TO_CONTROLLER_PROJECTION_V1;
            name="vdm-to-controller",
            replace=(_plan, values) -> VdmFeedbackToControllerPlan(values),
        ),
    )

    prepare(configuration) = begin
        _validate_vdm_feedback_to_controller_dimensions(
            configuration.controller_command_count,
            configuration.active_vdm_count,
        )
        VdmFeedbackToControllerPlan(
            zeros(
                Float32,
                configuration.controller_command_count,
                configuration.active_vdm_count,
            ),
        )
    end

    process!(plan) = begin
        _process_vdm_feedback_to_controller!(
            controller_constraint_feedback,
            plan,
            vdm_constraint_feedback,
        )
    end
end
