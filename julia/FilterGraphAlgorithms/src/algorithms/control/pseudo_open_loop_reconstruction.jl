"""
    PseudoOpenLoopReconstructionPlan(
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )

Prepare pseudo-open-loop reconstruction from synchronized residual measurements
and the corrector shape associated with the same acquisition. For residual
measurements `s`, corrector shape `u`, interaction matrix `D`, reconstructor
`R`, and independently calibrated corrector-to-error map `P`, processing
computes:

```text
pseudo_open_loop_measurements = s + D * u
residual_error = R * pseudo_open_loop_measurements - P * u
```

The three matrices are copied into dense Julia storage. The Plan does not
select a corrector shape, infer temporal association, or substitute delayed
constraint feedback for the acquisition-associated shape.
"""
struct PseudoOpenLoopReconstructionPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    measurement_count::Int
    corrector_shape_count::Int
    residual_error_count::Int
    interaction_matrix::Matrix{T}
    reconstructor::Matrix{T}
    corrector_to_error::Matrix{T}
end

function _validate_pseudo_open_loop_artifacts(
    interaction_matrix::AbstractMatrix{T},
    reconstructor::AbstractMatrix{T},
    corrector_to_error::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )
    measurement_count, corrector_shape_count = size(interaction_matrix)
    residual_error_count = size(reconstructor, 1)
    measurement_count > 0 || throw(
        ArgumentError("pseudo-open-loop measurement extent must be nonempty"),
    )
    corrector_shape_count > 0 || throw(
        ArgumentError("pseudo-open-loop corrector-shape extent must be nonempty"),
    )
    residual_error_count > 0 || throw(
        ArgumentError("pseudo-open-loop residual-error extent must be nonempty"),
    )
    size(reconstructor, 2) == measurement_count || throw(
        DimensionMismatch("pseudo-open-loop reconstructor has an invalid shape"),
    )
    size(corrector_to_error) ==
        (residual_error_count, corrector_shape_count) || throw(
        DimensionMismatch("corrector-to-error map has an invalid shape"),
    )
    all(isfinite, interaction_matrix) || throw(
        ArgumentError("pseudo-open-loop interaction matrix must be finite"),
    )
    all(isfinite, reconstructor) || throw(
        ArgumentError("pseudo-open-loop reconstructor must be finite"),
    )
    all(isfinite, corrector_to_error) || throw(
        ArgumentError("corrector-to-error map must be finite"),
    )
    return measurement_count, corrector_shape_count, residual_error_count
end

function _prepared_pseudo_open_loop_plan(
    interaction_matrix::Matrix{T},
    reconstructor::Matrix{T},
    corrector_to_error::Matrix{T},
) where {T<:Union{Float32,Float64}}
    return PseudoOpenLoopReconstructionPlan{T}(
        size(interaction_matrix, 1),
        size(interaction_matrix, 2),
        size(reconstructor, 1),
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )
end

function PseudoOpenLoopReconstructionPlan(
    interaction_matrix::AbstractMatrix{T},
    reconstructor::AbstractMatrix{T},
    corrector_to_error::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    _validate_pseudo_open_loop_artifacts(
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )
    return _prepared_pseudo_open_loop_plan(
        Matrix(interaction_matrix),
        Matrix(reconstructor),
        Matrix(corrector_to_error),
    )
end

function _pseudo_open_loop_row_major_matrix(
    ::Type{T},
    rows::Int,
    columns::Int,
    values::Union{Nothing,AbstractVector},
    name::AbstractString,
) where {T}
    extent = Base.Checked.checked_mul(rows, columns)
    isnothing(values) && return zeros(T, rows, columns)
    Base.require_one_based_indexing(values)
    length(values) == extent || throw(
        DimensionMismatch("$name extent does not match its configured shape"),
    )
    matrix = Matrix{T}(undef, rows, columns)
    for row in 1:rows, column in 1:columns
        matrix[row, column] = values[(row - 1) * columns + column]
    end
    return matrix
end

function _configured_pseudo_open_loop_plan(
    ::Type{T},
    measurement_count::Int,
    corrector_shape_count::Int,
    residual_error_count::Int,
    initial_interaction_matrix::Union{Nothing,AbstractVector{T}},
    initial_reconstructor::Union{Nothing,AbstractVector{T}},
    initial_corrector_to_error::Union{Nothing,AbstractVector{T}},
) where {T<:Union{Float32,Float64}}
    measurement_count > 0 || throw(
        ArgumentError("measurement_count must be positive"),
    )
    corrector_shape_count > 0 || throw(
        ArgumentError("corrector_shape_count must be positive"),
    )
    residual_error_count > 0 || throw(
        ArgumentError("residual_error_count must be positive"),
    )
    interaction_matrix = _pseudo_open_loop_row_major_matrix(
        T,
        measurement_count,
        corrector_shape_count,
        initial_interaction_matrix,
        "pseudo-open-loop interaction matrix",
    )
    reconstructor = _pseudo_open_loop_row_major_matrix(
        T,
        residual_error_count,
        measurement_count,
        initial_reconstructor,
        "pseudo-open-loop reconstructor",
    )
    corrector_to_error = _pseudo_open_loop_row_major_matrix(
        T,
        residual_error_count,
        corrector_shape_count,
        initial_corrector_to_error,
        "corrector-to-error map",
    )
    _validate_pseudo_open_loop_artifacts(
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )
    return _prepared_pseudo_open_loop_plan(
        interaction_matrix,
        reconstructor,
        corrector_to_error,
    )
end

function _replace_pseudo_open_loop_interaction_matrix(
    plan::PseudoOpenLoopReconstructionPlan{T},
    values::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(values)
    size(values) == (plan.measurement_count, plan.corrector_shape_count) ||
        throw(DimensionMismatch("interaction matrix has an invalid shape"))
    all(isfinite, values) || throw(
        ArgumentError("pseudo-open-loop interaction matrix must be finite"),
    )
    return _prepared_pseudo_open_loop_plan(
        Matrix(values),
        plan.reconstructor,
        plan.corrector_to_error,
    )
end

function _replace_pseudo_open_loop_reconstructor(
    plan::PseudoOpenLoopReconstructionPlan{T},
    values::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(values)
    size(values) == (plan.residual_error_count, plan.measurement_count) ||
        throw(DimensionMismatch("pseudo-open-loop reconstructor has an invalid shape"))
    all(isfinite, values) || throw(
        ArgumentError("pseudo-open-loop reconstructor must be finite"),
    )
    return _prepared_pseudo_open_loop_plan(
        plan.interaction_matrix,
        Matrix(values),
        plan.corrector_to_error,
    )
end

function _replace_pseudo_open_loop_corrector_to_error(
    plan::PseudoOpenLoopReconstructionPlan{T},
    values::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(values)
    size(values) ==
        (plan.residual_error_count, plan.corrector_shape_count) || throw(
        DimensionMismatch("corrector-to-error map has an invalid shape"),
    )
    all(isfinite, values) || throw(
        ArgumentError("corrector-to-error map must be finite"),
    )
    return _prepared_pseudo_open_loop_plan(
        plan.interaction_matrix,
        plan.reconstructor,
        Matrix(values),
    )
end

"""Disposable fixed scratch for one pseudo-open-loop reconstruction Node."""
struct PseudoOpenLoopReconstructionWorkspace{T<:Union{Float32,Float64}}
    predicted_measurements::Vector{T}
    staged_pseudo_open_loop_measurements::Vector{T}
    open_loop_estimate::Vector{T}
    corrector_in_error_coordinates::Vector{T}
    staged_residual_error::Vector{T}
end

function PseudoOpenLoopReconstructionWorkspace(
    plan::PseudoOpenLoopReconstructionPlan{T},
) where {T}
    return PseudoOpenLoopReconstructionWorkspace(
        zeros(T, plan.measurement_count),
        zeros(T, plan.measurement_count),
        zeros(T, plan.residual_error_count),
        zeros(T, plan.residual_error_count),
        zeros(T, plan.residual_error_count),
    )
end

commit!(workspace::PseudoOpenLoopReconstructionWorkspace) = workspace
discard!(workspace::PseudoOpenLoopReconstructionWorkspace) = workspace

function reset!(workspace::PseudoOpenLoopReconstructionWorkspace)
    fill!(workspace.predicted_measurements, 0)
    fill!(workspace.staged_pseudo_open_loop_measurements, 0)
    fill!(workspace.open_loop_estimate, 0)
    fill!(workspace.corrector_in_error_coordinates, 0)
    fill!(workspace.staged_residual_error, 0)
    return workspace
end

"""Expected numerical rejection from pseudo-open-loop reconstruction."""
@enum PseudoOpenLoopReconstructionFailure::UInt8 begin
    PseudoOpenLoopNonFiniteResidualMeasurements = 1
    PseudoOpenLoopNonFiniteCorrectorShape = 2
    PseudoOpenLoopNonFiniteOutput = 3
end

@doc "The residual-measurement input contains a non-finite value."
PseudoOpenLoopNonFiniteResidualMeasurements
@doc "The acquisition-associated corrector shape contains a non-finite value."
PseudoOpenLoopNonFiniteCorrectorShape
@doc "Pseudo-open-loop reconstruction produced a non-finite result."
PseudoOpenLoopNonFiniteOutput

function _pseudo_open_loop_scratch_aliases(
    workspace::PseudoOpenLoopReconstructionWorkspace,
    values::AbstractArray,
)
    return Base.mightalias(values, workspace.predicted_measurements) ||
           Base.mightalias(
        values,
        workspace.staged_pseudo_open_loop_measurements,
    ) ||
           Base.mightalias(values, workspace.open_loop_estimate) ||
           Base.mightalias(
        values,
        workspace.corrector_in_error_coordinates,
    ) ||
           Base.mightalias(values, workspace.staged_residual_error)
end

function _pseudo_open_loop_plan_aliases(
    plan::PseudoOpenLoopReconstructionPlan,
    values::AbstractArray,
)
    return Base.mightalias(values, plan.interaction_matrix) ||
           Base.mightalias(values, plan.reconstructor) ||
           Base.mightalias(values, plan.corrector_to_error)
end

function _pseudo_open_loop_scratch_is_disjoint(
    workspace::PseudoOpenLoopReconstructionWorkspace,
)
    predicted = workspace.predicted_measurements
    measurements = workspace.staged_pseudo_open_loop_measurements
    estimate = workspace.open_loop_estimate
    correction = workspace.corrector_in_error_coordinates
    residual = workspace.staged_residual_error
    return !Base.mightalias(predicted, measurements) &&
           !Base.mightalias(predicted, estimate) &&
           !Base.mightalias(predicted, correction) &&
           !Base.mightalias(predicted, residual) &&
           !Base.mightalias(measurements, estimate) &&
           !Base.mightalias(measurements, correction) &&
           !Base.mightalias(measurements, residual) &&
           !Base.mightalias(estimate, correction) &&
           !Base.mightalias(estimate, residual) &&
           !Base.mightalias(correction, residual)
end

function _validate_pseudo_open_loop_storage(
    pseudo_open_loop_measurements::AbstractVector{T},
    residual_error::AbstractVector{T},
    workspace::PseudoOpenLoopReconstructionWorkspace{T},
    plan::PseudoOpenLoopReconstructionPlan{T},
    residual_measurements::AbstractVector{T},
    corrector_shape::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(
        pseudo_open_loop_measurements,
        residual_error,
        residual_measurements,
        corrector_shape,
    )
    length(residual_measurements) == plan.measurement_count || throw(
        DimensionMismatch("residual-measurement input extent is invalid"),
    )
    length(corrector_shape) == plan.corrector_shape_count || throw(
        DimensionMismatch("corrector-shape input extent is invalid"),
    )
    length(pseudo_open_loop_measurements) == plan.measurement_count || throw(
        DimensionMismatch("pseudo-open-loop measurement output extent is invalid"),
    )
    length(residual_error) == plan.residual_error_count || throw(
        DimensionMismatch("residual-error output extent is invalid"),
    )
    length(workspace.predicted_measurements) == plan.measurement_count ||
        throw(DimensionMismatch("pseudo-open-loop Workspace extent is invalid"))
    length(workspace.staged_pseudo_open_loop_measurements) ==
        plan.measurement_count || throw(
        DimensionMismatch("pseudo-open-loop Workspace extent is invalid"),
    )
    length(workspace.open_loop_estimate) == plan.residual_error_count ||
        throw(DimensionMismatch("pseudo-open-loop Workspace extent is invalid"))
    length(workspace.corrector_in_error_coordinates) ==
        plan.residual_error_count || throw(
        DimensionMismatch("pseudo-open-loop Workspace extent is invalid"),
    )
    length(workspace.staged_residual_error) == plan.residual_error_count ||
        throw(DimensionMismatch("pseudo-open-loop Workspace extent is invalid"))
    _pseudo_open_loop_scratch_is_disjoint(workspace) || throw(
        ArgumentError("pseudo-open-loop Workspace scratch must not overlap"),
    )
    Base.mightalias(pseudo_open_loop_measurements, residual_error) && throw(
        ArgumentError("pseudo-open-loop outputs must not overlap"),
    )
    _pseudo_open_loop_scratch_aliases(workspace, residual_measurements) &&
        throw(ArgumentError("inputs must not alias pseudo-open-loop Workspace"))
    _pseudo_open_loop_scratch_aliases(workspace, corrector_shape) &&
        throw(ArgumentError("inputs must not alias pseudo-open-loop Workspace"))
    _pseudo_open_loop_scratch_aliases(
        workspace,
        pseudo_open_loop_measurements,
    ) && throw(
        ArgumentError("outputs must not alias pseudo-open-loop Workspace"),
    )
    _pseudo_open_loop_scratch_aliases(workspace, residual_error) && throw(
        ArgumentError("outputs must not alias pseudo-open-loop Workspace"),
    )
    _pseudo_open_loop_plan_aliases(plan, pseudo_open_loop_measurements) &&
        throw(ArgumentError("outputs must not alias prepared artifacts"))
    _pseudo_open_loop_plan_aliases(plan, residual_error) && throw(
        ArgumentError("outputs must not alias prepared artifacts"),
    )
    return nothing
end

function _process_pseudo_open_loop_reconstruction!(
    pseudo_open_loop_measurements::AbstractVector{T},
    residual_error::AbstractVector{T},
    workspace::PseudoOpenLoopReconstructionWorkspace{T},
    plan::PseudoOpenLoopReconstructionPlan{T},
    residual_measurements::AbstractVector{T},
    corrector_shape::AbstractVector{T},
) where {T}
    _validate_pseudo_open_loop_storage(
        pseudo_open_loop_measurements,
        residual_error,
        workspace,
        plan,
        residual_measurements,
        corrector_shape,
    )
    for value in residual_measurements
        isfinite(value) || return PseudoOpenLoopNonFiniteResidualMeasurements
    end
    for value in corrector_shape
        isfinite(value) || return PseudoOpenLoopNonFiniteCorrectorShape
    end

    predicted = workspace.predicted_measurements
    mul!(predicted, plan.interaction_matrix, corrector_shape)
    for value in predicted
        isfinite(value) || return PseudoOpenLoopNonFiniteOutput
    end

    staged_measurements = workspace.staged_pseudo_open_loop_measurements
    @inbounds for index in eachindex(
        staged_measurements,
        residual_measurements,
        predicted,
    )
        value = residual_measurements[index] + predicted[index]
        staged_measurements[index] = value
    end
    for value in staged_measurements
        isfinite(value) || return PseudoOpenLoopNonFiniteOutput
    end

    estimate = workspace.open_loop_estimate
    mul!(estimate, plan.reconstructor, staged_measurements)
    for value in estimate
        isfinite(value) || return PseudoOpenLoopNonFiniteOutput
    end
    corrector_in_error = workspace.corrector_in_error_coordinates
    mul!(corrector_in_error, plan.corrector_to_error, corrector_shape)
    for value in corrector_in_error
        isfinite(value) || return PseudoOpenLoopNonFiniteOutput
    end

    staged_residual = workspace.staged_residual_error
    @inbounds for index in eachindex(
        staged_residual,
        estimate,
        corrector_in_error,
    )
        value = estimate[index] - corrector_in_error[index]
        staged_residual[index] = value
    end
    for value in staged_residual
        isfinite(value) || return PseudoOpenLoopNonFiniteOutput
    end

    copyto!(pseudo_open_loop_measurements, staged_measurements)
    copyto!(residual_error, staged_residual)
    return nothing
end

"""
Float32 pseudo-open-loop reconstruction from residual measurements and the
corrector shape synchronized to the same acquisition. Both outputs inherit
metadata from the residual-measurement input.
"""
@algorithm PseudoOpenLoopReconstructionF32 begin
    label = "pseudo-open-loop-reconstruction-f32"
    failure = PseudoOpenLoopReconstructionFailure

    """Configuration for [`PseudoOpenLoopReconstructionF32`](@ref)."""
    @configuration PseudoOpenLoopReconstructionF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_shape_count::Int => graph_rebuild
        residual_error_count::Int => graph_rebuild
        initial_interaction_matrix::Union{Nothing,Vector{Float32}} => construction
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        initial_corrector_to_error::Union{Nothing,Vector{Float32}} => construction
        residual_measurements_schema::String => graph_rebuild
        corrector_shape_schema::String => graph_rebuild
        pseudo_open_loop_measurements_schema::String => graph_rebuild
        residual_error_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :residual_measurements,
            Float32,
            (configuration.measurement_count,),
            configuration.residual_measurements_schema;
            name="residual-measurements",
        ),
        input(
            :corrector_shape,
            Float32,
            (configuration.corrector_shape_count,),
            configuration.corrector_shape_schema;
            name="corrector-shape",
        ),
        output(
            :pseudo_open_loop_measurements,
            Float32,
            (configuration.measurement_count,),
            configuration.pseudo_open_loop_measurements_schema;
            name="pseudo-open-loop-measurements",
            metadata_from=:residual_measurements,
        ),
        output(
            :residual_error,
            Float32,
            (configuration.residual_error_count,),
            configuration.residual_error_schema;
            name="residual-error",
            metadata_from=:residual_measurements,
        ),
        parameter(
            :interaction_matrix,
            schema_element_type(PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1),
            (configuration.measurement_count, configuration.corrector_shape_count),
            PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1;
            name="interaction-matrix",
            replace=(plan, values) ->
                _replace_pseudo_open_loop_interaction_matrix(plan, values),
        ),
        parameter(
            :reconstructor,
            schema_element_type(PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1),
            (configuration.residual_error_count, configuration.measurement_count),
            PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1;
            replace=(plan, values) ->
                _replace_pseudo_open_loop_reconstructor(plan, values),
        ),
        parameter(
            :corrector_to_error,
            schema_element_type(PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1),
            (
                configuration.residual_error_count,
                configuration.corrector_shape_count,
            ),
            PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1;
            name="corrector-to-error",
            replace=(plan, values) ->
                _replace_pseudo_open_loop_corrector_to_error(plan, values),
        ),
    )

    prepare(configuration) = begin
        plan = _configured_pseudo_open_loop_plan(
            Float32,
            configuration.measurement_count,
            configuration.corrector_shape_count,
            configuration.residual_error_count,
            configuration.initial_interaction_matrix,
            configuration.initial_reconstructor,
            configuration.initial_corrector_to_error,
        )
        (plan=plan, workspace=PseudoOpenLoopReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_pseudo_open_loop_reconstruction!(
            pseudo_open_loop_measurements,
            residual_error,
            workspace,
            plan,
            residual_measurements,
            corrector_shape,
        )
    end
end
