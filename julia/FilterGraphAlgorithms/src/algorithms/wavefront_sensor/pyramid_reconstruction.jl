function _process_pyramid_reconstruction!(
    reconstructed::AbstractVector{T},
    workspace::DenseReconstructorWorkspace{T},
    plan::DenseReconstructorPlan{T},
    signal::AbstractMatrix{T},
) where {T}
    _validate_dense_reconstruction_storage(
        reconstructed,
        workspace,
        plan,
        signal,
    )
    signal_count = size(plan.matrix, 2)
    rem(signal_count, 2) == 0 || throw(
        DimensionMismatch("pyramid I4Q reconstructor must contain X and Y blocks"),
    )
    selected_coordinate_count = div(signal_count, 2)
    size(signal) == (selected_coordinate_count, 2) || throw(
        DimensionMismatch("pyramid reconstruction input has an incompatible shape"),
    )

    input_index = 1
    for component in 1:2
        for coordinate in 1:selected_coordinate_count
            value = signal[coordinate, component]
            isfinite(value) || return DenseReconstructionNonFiniteInput
            workspace.ordered_input[input_index] = value
            input_index += 1
        end
    end
    return _multiply_dense_reconstructor!(reconstructed, workspace, plan)
end

"""
Map a selected two-coordinate pyramid differential signal through a dense
reconstructor to a Float32 reconstructed vector. Matrix coefficients define
the conversion from normalized signal units to output units.
"""
@algorithm PyramidReconstructorF32 begin
    label = "pwfs-reconstructor-f32"
    failure = DenseReconstructionFailure
    operation = DenseF32Operation()

    """
    Set reconstructed count, selected coordinate count, and output schema.
    The optional initial matrix maps the flattened two-coordinate signal to the
    reconstructed vector and may be replaced at the reconstructor parameter port.
    """
    @configuration PyramidReconstructorF32Configuration begin
        reconstructed_count::Int => graph_rebuild
        selected_coordinate_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        reconstructed_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :signal,
            schema_element_type(PYRAMID_I4Q_SIGNAL_V1),
            (configuration.selected_coordinate_count, 2),
            PYRAMID_I4Q_SIGNAL_V1;
            name="signal",
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:signal,
        ),
        parameter(
            :reconstructor,
            schema_element_type(PYRAMID_I4Q_RECONSTRUCTOR_V1),
            (
                configuration.reconstructed_count,
                Base.Checked.checked_mul(
                    configuration.selected_coordinate_count,
                    2,
                ),
            ),
            PYRAMID_I4Q_RECONSTRUCTOR_V1;
            replace=(_plan, values) -> DenseReconstructorPlan(values),
        ),
    )

    prepare(configuration) = begin
        signal_count = Base.Checked.checked_mul(
            configuration.selected_coordinate_count,
            2,
        )
        plan = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            signal_count,
            configuration.initial_reconstructor,
        )
        (plan=plan, workspace=DenseReconstructorWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_pyramid_reconstruction!(
            reconstructed,
            workspace,
            plan,
            signal,
        )
    end
end
