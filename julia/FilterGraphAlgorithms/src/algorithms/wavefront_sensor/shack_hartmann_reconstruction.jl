function _process_shack_hartmann_reconstruction!(
    reconstructed::AbstractVector{T},
    workspace::DenseReconstructorWorkspace{T},
    plan::DenseReconstructorPlan{T},
    slopes::AbstractMatrix{T},
) where {T}
    _validate_dense_reconstruction_storage(
        reconstructed,
        workspace,
        plan,
        slopes,
    )
    slope_count = size(plan.matrix, 2)
    iseven(slope_count) || throw(
        DimensionMismatch("Shack-Hartmann reconstructor must contain X/Y pairs"),
    )
    subaperture_count = div(slope_count, 2)
    size(slopes) == (subaperture_count, 2) || throw(
        DimensionMismatch("Shack-Hartmann slope input has an incompatible shape"),
    )

    for subaperture in 1:subaperture_count
        input_index = 2 * subaperture - 1
        x_slope = slopes[subaperture, 1]
        y_slope = slopes[subaperture, 2]
        isfinite(x_slope) && isfinite(y_slope) ||
            return DenseReconstructionNonFiniteInput
        workspace.ordered_input[input_index] = x_slope
        workspace.ordered_input[input_index + 1] = y_slope
    end
    return _multiply_dense_reconstructor!(reconstructed, workspace, plan)
end

"""
Multiply complete-image Shack-Hartmann X/Y slopes by a dense reconstructor to
produce a Float32 reconstructed vector. Matrix coefficients and the output
schema establish its coordinate basis and units.
"""
@algorithm ShackHartmannReconstructorF32 begin
    label = "shwfs-reconstructor-f32"
    failure = DenseReconstructionFailure
    operation = DenseF32Operation()

    """
    Set subaperture count, reconstructed count and output schema. The optional initial
    matrix has shape `(reconstructed_count, 2 * subaperture_count)` and can be replaced
    through its reconstructor parameter port. Set `coefficient_storage = "fp16"`
    for row-scaled Float16 storage with Float32 arithmetic.
    """
    @configuration ShackHartmannReconstructorF32Configuration begin
        reconstructed_count::Int => graph_rebuild
        subaperture_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        reconstructed_schema::String => graph_rebuild
        (coefficient_storage::String = "fp32") => graph_rebuild
    end

    ports(configuration) = (
        input(
            :slopes,
            schema_element_type(SHACK_HARTMANN_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_SLOPES_V1,
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:slopes,
        ),
        parameter(
            :reconstructor,
            schema_element_type(SHWFS_RECONSTRUCTOR_V1),
            (
                configuration.reconstructed_count,
                Base.Checked.checked_mul(configuration.subaperture_count, 2),
            ),
            SHWFS_RECONSTRUCTOR_V1;
            replace=_replace_dense_reconstructor_storage,
        ),
    )

    prepare(configuration) = begin
        slope_count = Base.Checked.checked_mul(configuration.subaperture_count, 2)
        plan = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            slope_count,
            configuration.initial_reconstructor,
        )
        plan = _configure_dense_reconstructor_storage(plan, configuration.coefficient_storage)
        (plan=plan, workspace=DenseReconstructorWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_shack_hartmann_reconstruction!(
            reconstructed,
            workspace,
            plan,
            slopes,
        )
    end
end
