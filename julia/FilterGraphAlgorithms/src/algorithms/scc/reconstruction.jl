function _process_scc_reconstruction!(
    controller_residual_error::AbstractVector{T},
    plan::DenseReconstructorPlan{T},
    selected_residuals::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(controller_residual_error, selected_residuals)
    matrix = plan.matrix
    length(controller_residual_error) == size(matrix, 1) || throw(
        DimensionMismatch("SCC controller-residual-error extent is invalid"),
    )
    length(selected_residuals) == size(matrix, 2) ||
        throw(DimensionMismatch("SCC selected-residual extent is invalid"))
    Base.mightalias(controller_residual_error, selected_residuals) && throw(
        ArgumentError("SCC input and output storage must not alias"),
    )
    Base.mightalias(controller_residual_error, matrix) && throw(
        ArgumentError("SCC output must not alias the prepared reconstructor"),
    )
    return _multiply_dense_reconstructor!(
        controller_residual_error,
        plan,
        selected_residuals,
    )
end

"""
Float32 selected-pixel SCC reconstruction into named controller residual-error
coordinates. The prepared matrix has controller coordinates in rows and the
dark-hole pixel order from [`SccPixelResidualF32`](@ref) in columns.

Set `coefficient_storage = "fp16"` to pack the Float32 calibration with
power-of-two row scales. The default is `"fp32"`. Both settings use
Float32 inputs, accumulation and outputs.

For a calibration stored with selected pixels in rows and physical actuators
in columns, transpose the matrix and map its actuator rows into the selected
VDM/controller coordinates before use.
"""
@algorithm SccReconstructorF32 begin
    label = "scc-reconstructor-f32"
    failure = DenseReconstructionFailure
    operation = DenseF32Operation()

    """Configuration for [`SccReconstructorF32`](@ref)."""
    @configuration SccReconstructorF32Configuration begin
        controller_coordinate_count::Int => graph_rebuild
        selected_pixel_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        (coefficient_storage::String = "fp32") => graph_rebuild
    end

    ports(configuration) = (
        input(
            :selected_residuals,
            schema_element_type(SCC_SELECTED_PIXEL_RESIDUALS_V1),
            (configuration.selected_pixel_count,),
            SCC_SELECTED_PIXEL_RESIDUALS_V1;
            name="selected-residuals",
        ),
        output(
            :controller_residual_error,
            schema_element_type(CONTROLLER_RESIDUAL_ERROR_V1),
            (configuration.controller_coordinate_count,),
            CONTROLLER_RESIDUAL_ERROR_V1;
            name="controller-residual-error",
            metadata_from=:selected_residuals,
        ),
        parameter(
            :reconstructor,
            schema_element_type(SCC_RECONSTRUCTOR_V1),
            (
                configuration.controller_coordinate_count,
                configuration.selected_pixel_count,
            ),
            SCC_RECONSTRUCTOR_V1;
            replace=_replace_dense_reconstructor_storage,
        ),
    )

    prepare(configuration) = begin
        plan = DenseReconstructorPlan(
            Float32,
            configuration.controller_coordinate_count,
            configuration.selected_pixel_count,
            configuration.initial_reconstructor,
        )
        _configure_dense_reconstructor_storage(plan, configuration.coefficient_storage)
    end

    process!(plan) = begin
        _process_scc_reconstruction!(
            controller_residual_error,
            plan,
            selected_residuals,
        )
    end
end
