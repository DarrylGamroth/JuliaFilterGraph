"""
    VdmCompletionCalibration(partition, coordinate_positions, fixed_values=[])

Bundle zonal-VDM geometry and fixed values for completion preparation.

`coordinate_positions` has one row per full-VDM coordinate and two columns
containing positions in a common Cartesian coordinate frame. Coordinate units
may be arbitrary but must be mutually consistent. `fixed_values` is packed in
the canonical order selected by [`fixed_vdm_indices`](@ref) and has the VDM
command unit. The caller must not mutate these arrays during [`process!`](@ref).

The two matrix axes are full-VDM-coordinate and Cartesian-component
enumerations. `process!` materializes the input as dense one-based host arrays
in logical element order. CPU views and wrappers that support that conversion
are accepted; device-only arrays are outside this preparation contract.
"""
struct VdmCompletionCalibration{
    P<:AbstractMatrix{<:Real},
    F<:AbstractVector{<:Real},
}
    partition::VdmActuatorPartition
    coordinate_positions::P
    fixed_values::F
end

function VdmCompletionCalibration(
    partition::VdmActuatorPartition,
    coordinate_positions::AbstractMatrix{<:Real},
)
    return VdmCompletionCalibration(
        partition,
        coordinate_positions,
        Float64[],
    )
end

"""
    ThinPlateVdmCompletionPlan{T}(;
        regularization=zero(T),
        validation_tolerance=sqrt(eps(T)),
    )
    ThinPlateVdmCompletionPlan(;
        regularization=0.0,
        validation_tolerance=nothing,
    )

Prepare a smooth zonal-VDM completion operator from two-dimensional geometry.

The preparation fits the thin-plate spline

```text
f(x) = sum(a[j] * phi(norm(x - x[j]))) + b[1] + b[2] * x[1] + b[3] * x[2]
phi(r) = r^2 * log(r), with phi(0) = 0
```

to the controlled and fixed support coordinates, then evaluates it at the
extrapolated coordinates. The affine polynomial tail reproduces piston and
both tilt components. Coordinates are centered and scaled internally, so
`regularization` is a finite nonnegative dimensionless diagonal term applied
to the radial-kernel block. A zero value gives interpolation; a positive value
trades exact support interpolation for a smoother, better-conditioned
completion while retaining affine reproduction.

`validation_tolerance` bounds the relative solver residual and maximum affine
reproduction error. Preparation preserves any `AbstractFloat` representation
supported by its arithmetic and factorization operations. This is an offline
artifact-preparation method, not an admitted operating-path solve.
"""
struct ThinPlateVdmCompletionPlan{T<:AbstractFloat} <:
       AbstractAlgorithmPlan
    regularization::T
    validation_tolerance::T

    function ThinPlateVdmCompletionPlan{T}(;
        regularization::Real=zero(T),
        validation_tolerance::Real=sqrt(eps(T)),
    ) where {T<:AbstractFloat}
        prepared_regularization = T(regularization)
        isfinite(prepared_regularization) &&
            prepared_regularization >= zero(T) ||
            throw(
                ArgumentError(
                    "regularization must be finite and nonnegative",
                ),
            )
        prepared_tolerance = T(validation_tolerance)
        isfinite(prepared_tolerance) &&
            zero(T) <= prepared_tolerance < one(T) ||
            throw(
                ArgumentError(
                    "validation_tolerance must be finite and in [0, 1)",
                ),
            )
        return new{T}(prepared_regularization, prepared_tolerance)
    end
end

function ThinPlateVdmCompletionPlan(;
    regularization::Real=0.0,
    validation_tolerance::Union{Nothing,Real}=nothing,
)
    T = float(typeof(regularization))
    T <: AbstractFloat ||
        throw(
            ArgumentError(
                "thin-plate VDM completion requires an AbstractFloat",
            ),
        )
    tolerance = if isnothing(validation_tolerance)
        sqrt(eps(T))
    else
        validation_tolerance
    end
    return ThinPlateVdmCompletionPlan{T}(;
        regularization=regularization,
        validation_tolerance=tolerance,
    )
end

"""
    VdmCompletionDiagnostics{T}

Report numerical diagnostics for one thin-plate VDM completion preparation.

`solver_relative_residual` is the relative Frobenius residual of all support
basis solves. `maximum_affine_reproduction_error` is measured in normalized
coordinates over piston and both tilt components.
`maximum_absolute_support_weight` includes the weights of both controlled and
fixed support coordinates. Large weights warn that extrapolation may amplify
calibration or controller errors even when the solve itself is accurate.
"""
struct VdmCompletionDiagnostics{T<:AbstractFloat}
    solver_relative_residual::T
    maximum_affine_reproduction_error::T
    maximum_absolute_support_weight::T
end

function _empty_vdm_completion_diagnostics(
    ::Type{T},
) where {T<:AbstractFloat}
    return VdmCompletionDiagnostics{T}(T(Inf), T(Inf), T(Inf))
end

"""
    VdmCompletionArtifacts(plan, calibration)

Allocate caller-owned artifacts for one VDM completion preparation.

For `c` controlled, `e` extrapolated, and `v` full-VDM coordinates,
`extrapolated_from_controlled` is the `e` by `c` matrix `W` and
`full_vdm_offset` is the `v`-element vector `b`. A caller completes one packed
controlled vector `u` structurally as

```text
full[controlled]   = u
full[extrapolated] = W * u + b[extrapolated]
full[fixed]        = b[fixed]
```

The partition carries all scatter/gather indices, so neither identity-copy
rows nor full-to-controlled selection rows are stored as matrix coefficients.
Artifact serialization and activation remain caller-owned responsibilities.
"""
mutable struct VdmCompletionArtifacts{T<:AbstractFloat}
    extrapolated_from_controlled::Matrix{T}
    full_vdm_offset::Vector{T}
    diagnostics::VdmCompletionDiagnostics{T}
end

function VdmCompletionArtifacts(
    ::ThinPlateVdmCompletionPlan{T},
    calibration::VdmCompletionCalibration,
) where {T<:AbstractFloat}
    partition = calibration.partition
    return VdmCompletionArtifacts{T}(
        zeros(
            T,
            length(partition.extrapolated_indices),
            length(partition.controlled_indices),
        ),
        zeros(T, full_vdm_count(partition)),
        _empty_vdm_completion_diagnostics(T),
    )
end

"""
    VdmCompletionWorkspace(plan, calibration)

Construct the zero-size workspace for thin-plate VDM completion preparation.

The dense solver produces complete local results that are validated before
publication, so the workspace does not duplicate caller-owned artifacts.
"""
struct VdmCompletionWorkspace{T<:AbstractFloat} end

function VdmCompletionWorkspace(
    ::ThinPlateVdmCompletionPlan{T},
    ::VdmCompletionCalibration,
) where {T<:AbstractFloat}
    return VdmCompletionWorkspace{T}()
end

function _check_vdm_completion_extents(
    artifacts::VdmCompletionArtifacts,
    calibration::VdmCompletionCalibration,
)
    partition = calibration.partition
    expected_weights = (
        length(partition.extrapolated_indices),
        length(partition.controlled_indices),
    )
    size(artifacts.extrapolated_from_controlled) == expected_weights ||
        throw(
            DimensionMismatch(
                "extrapolated_from_controlled has an incompatible extent",
            ),
        )
    length(artifacts.full_vdm_offset) == full_vdm_count(partition) ||
        throw(
            DimensionMismatch(
                "full_vdm_offset has an incompatible extent",
            ),
        )
    return nothing
end

function _validated_vdm_completion_calibration(
    ::Type{T},
    calibration::VdmCompletionCalibration,
) where {T<:AbstractFloat}
    partition = calibration.partition
    expected_position_shape = (full_vdm_count(partition), 2)
    size(calibration.coordinate_positions) == expected_position_shape ||
        throw(
            DimensionMismatch(
                "coordinate_positions must have shape $expected_position_shape",
            ),
        )
    length(calibration.fixed_values) == length(partition.fixed_indices) ||
        throw(
            DimensionMismatch(
                "fixed_values must have one value per fixed VDM coordinate",
            ),
        )

    coordinate_positions = Matrix{T}(calibration.coordinate_positions)
    fixed_values = Vector{T}(calibration.fixed_values)
    all(isfinite, coordinate_positions) ||
        throw(ArgumentError("coordinate_positions must be finite"))
    all(isfinite, fixed_values) ||
        throw(ArgumentError("fixed_values must be finite"))

    coordinate_count = size(coordinate_positions, 1)
    for first_coordinate in 1:(coordinate_count - 1)
        for second_coordinate in (first_coordinate + 1):coordinate_count
            if coordinate_positions[first_coordinate, 1] ==
               coordinate_positions[second_coordinate, 1] &&
               coordinate_positions[first_coordinate, 2] ==
               coordinate_positions[second_coordinate, 2]
                throw(
                    ArgumentError(
                        "full-VDM coordinate positions must be distinct",
                    ),
                )
            end
        end
    end
    return coordinate_positions, fixed_values
end

function _normalized_vdm_positions(
    coordinate_positions::AbstractMatrix{T},
    support_indices::AbstractVector{Int},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(coordinate_positions, support_indices)
    support_count = length(support_indices)
    divisor = T(support_count)
    isfinite(divisor) && divisor == support_count ||
        throw(
            ArgumentError(
                "VDM completion support count is not exactly representable",
            ),
        )
    center_x = zero(T)
    center_y = zero(T)
    for index in support_indices
        center_x += coordinate_positions[index, 1]
        center_y += coordinate_positions[index, 2]
    end
    center_x /= divisor
    center_y /= divisor

    scale_squared = zero(T)
    for index in support_indices
        x = coordinate_positions[index, 1] - center_x
        y = coordinate_positions[index, 2] - center_y
        scale_squared += x * x + y * y
    end
    scale_squared /= divisor
    isfinite(scale_squared) && scale_squared > zero(T) ||
        throw(
            ArgumentError(
                "VDM completion support positions have zero or non-finite scale",
            ),
        )
    scale = sqrt(scale_squared)
    normalized = similar(coordinate_positions)
    for coordinate in axes(normalized, 1)
        normalized[coordinate, 1] =
            (coordinate_positions[coordinate, 1] - center_x) / scale
        normalized[coordinate, 2] =
            (coordinate_positions[coordinate, 2] - center_y) / scale
    end
    all(isfinite, normalized) ||
        throw(
            ArgumentError(
                "normalized VDM completion positions are non-finite",
            ),
        )
    return normalized
end

function _thin_plate_kernel(distance_squared::T) where {T<:AbstractFloat}
    iszero(distance_squared) && return zero(T)
    return distance_squared * log(distance_squared) / T(2)
end

function _thin_plate_support_system(
    normalized_positions::AbstractMatrix{T},
    support_indices::AbstractVector{Int},
    regularization::T,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(normalized_positions, support_indices)
    support_count = length(support_indices)
    system = zeros(T, support_count + 3, support_count + 3)
    for first_support in eachindex(support_indices)
        first_coordinate = support_indices[first_support]
        first_x = normalized_positions[first_coordinate, 1]
        first_y = normalized_positions[first_coordinate, 2]
        for second_support in eachindex(support_indices)
            second_coordinate = support_indices[second_support]
            dx = first_x - normalized_positions[second_coordinate, 1]
            dy = first_y - normalized_positions[second_coordinate, 2]
            distance_squared = dx * dx + dy * dy
            isfinite(distance_squared) ||
                throw(
                    ArgumentError(
                        "thin-plate VDM kernel distance is non-finite",
                    ),
                )
            system[first_support, second_support] =
                _thin_plate_kernel(distance_squared)
        end
        system[first_support, first_support] += regularization
        system[first_support, support_count + 1] = one(T)
        system[first_support, support_count + 2] = first_x
        system[first_support, support_count + 3] = first_y
        system[support_count + 1, first_support] = one(T)
        system[support_count + 2, first_support] = first_x
        system[support_count + 3, first_support] = first_y
    end
    all(isfinite, system) ||
        throw(ArgumentError("thin-plate VDM support system is non-finite"))
    return system
end

function _thin_plate_evaluation(
    normalized_positions::AbstractMatrix{T},
    support_indices::AbstractVector{Int},
    target_indices::AbstractVector{Int},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(
        normalized_positions,
        support_indices,
        target_indices,
    )
    support_count = length(support_indices)
    evaluation = zeros(T, length(target_indices), support_count + 3)
    for target in eachindex(target_indices)
        target_coordinate = target_indices[target]
        target_x = normalized_positions[target_coordinate, 1]
        target_y = normalized_positions[target_coordinate, 2]
        for support in eachindex(support_indices)
            support_coordinate = support_indices[support]
            dx = target_x - normalized_positions[support_coordinate, 1]
            dy = target_y - normalized_positions[support_coordinate, 2]
            distance_squared = dx * dx + dy * dy
            isfinite(distance_squared) ||
                throw(
                    ArgumentError(
                        "thin-plate VDM evaluation distance is non-finite",
                    ),
                )
            evaluation[target, support] =
                _thin_plate_kernel(distance_squared)
        end
        evaluation[target, support_count + 1] = one(T)
        evaluation[target, support_count + 2] = target_x
        evaluation[target, support_count + 3] = target_y
    end
    all(isfinite, evaluation) ||
        throw(ArgumentError("thin-plate VDM evaluation is non-finite"))
    return evaluation
end

function _maximum_absolute(matrix::AbstractArray{T}) where {T<:AbstractFloat}
    isempty(matrix) && return zero(T)
    return maximum(abs, matrix)
end

function _thin_plate_affine_reproduction_error(
    support_weights::AbstractMatrix{T},
    normalized_positions::AbstractMatrix{T},
    support_indices::AbstractVector{Int},
    target_indices::AbstractVector{Int},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(
        support_weights,
        normalized_positions,
        support_indices,
        target_indices,
    )
    isempty(target_indices) && return zero(T)
    support_count = length(support_indices)
    maximum_error = zero(T)
    for target in eachindex(target_indices)
        target_coordinate = target_indices[target]
        piston = zero(T)
        x_tilt = zero(T)
        y_tilt = zero(T)
        for support in 1:support_count
            weight = support_weights[target, support]
            support_coordinate = support_indices[support]
            piston += weight
            x_tilt +=
                weight * normalized_positions[support_coordinate, 1]
            y_tilt +=
                weight * normalized_positions[support_coordinate, 2]
        end
        maximum_error = max(maximum_error, abs(piston - one(T)))
        maximum_error = max(
            maximum_error,
            abs(
                x_tilt -
                normalized_positions[target_coordinate, 1],
            ),
        )
        maximum_error = max(
            maximum_error,
            abs(
                y_tilt -
                normalized_positions[target_coordinate, 2],
            ),
        )
    end
    return maximum_error
end

function _solve_thin_plate_completion(
    plan::ThinPlateVdmCompletionPlan{T},
    calibration::VdmCompletionCalibration,
    coordinate_positions::AbstractMatrix{T},
    fixed_values::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(coordinate_positions, fixed_values)
    partition = calibration.partition
    target_indices = partition.extrapolated_indices
    controlled_indices = partition.controlled_indices
    fixed_indices = partition.fixed_indices
    full_offset = zeros(T, full_vdm_count(partition))
    for (fixed_coordinate, full_coordinate) in enumerate(fixed_indices)
        full_offset[full_coordinate] = fixed_values[fixed_coordinate]
    end
    if isempty(target_indices)
        weights = zeros(T, 0, length(controlled_indices))
        diagnostics = VdmCompletionDiagnostics{T}(
            zero(T),
            zero(T),
            zero(T),
        )
        return weights, full_offset, diagnostics
    end

    support_indices = sort!(vcat(controlled_indices, fixed_indices))
    length(support_indices) >= 3 ||
        throw(
            ArgumentError(
                "thin-plate VDM completion requires at least three support coordinates",
            ),
        )
    normalized_positions =
        _normalized_vdm_positions(coordinate_positions, support_indices)
    system = _thin_plate_support_system(
        normalized_positions,
        support_indices,
        plan.regularization,
    )
    support_count = length(support_indices)
    right_hand_side = zeros(T, support_count + 3, support_count)
    for support in 1:support_count
        right_hand_side[support, support] = one(T)
    end
    coefficients = try
        system \ right_hand_side
    catch error
        if error isa SingularException || error isa ZeroPivotException
            throw(
                ArgumentError(
                    "thin-plate VDM support geometry is singular",
                ),
            )
        end
        rethrow()
    end
    all(isfinite, coefficients) ||
        throw(ArgumentError("thin-plate VDM solve is non-finite"))
    evaluation = _thin_plate_evaluation(
        normalized_positions,
        support_indices,
        target_indices,
    )
    support_weights = evaluation * coefficients
    all(isfinite, support_weights) ||
        throw(ArgumentError("thin-plate VDM completion weights are non-finite"))

    controlled_column = zeros(Int, full_vdm_count(partition))
    for (column, full_coordinate) in enumerate(controlled_indices)
        controlled_column[full_coordinate] = column
    end
    fixed_column = zeros(Int, full_vdm_count(partition))
    for (column, full_coordinate) in enumerate(fixed_indices)
        fixed_column[full_coordinate] = column
    end
    controlled_weights = zeros(
        T,
        length(target_indices),
        length(controlled_indices),
    )
    for (support, full_coordinate) in enumerate(support_indices)
        controlled = controlled_column[full_coordinate]
        if !iszero(controlled)
            for target in eachindex(target_indices)
                controlled_weights[target, controlled] =
                    support_weights[target, support]
            end
            continue
        end
        fixed = fixed_column[full_coordinate]
        for (target, target_coordinate) in enumerate(target_indices)
            full_offset[target_coordinate] +=
                support_weights[target, support] * fixed_values[fixed]
        end
    end

    solver_denominator = max(norm(right_hand_side), eps(T))
    solver_relative_residual =
        T(norm(system * coefficients - right_hand_side) / solver_denominator)
    affine_error = _thin_plate_affine_reproduction_error(
        support_weights,
        normalized_positions,
        support_indices,
        target_indices,
    )
    isfinite(solver_relative_residual) &&
        solver_relative_residual <= plan.validation_tolerance ||
        throw(
            ArgumentError(
                "thin-plate VDM solve exceeds validation_tolerance",
            ),
        )
    isfinite(affine_error) &&
        affine_error <= plan.validation_tolerance ||
        throw(
            ArgumentError(
                "thin-plate VDM affine reproduction exceeds validation_tolerance",
            ),
        )
    diagnostics = VdmCompletionDiagnostics{T}(
        solver_relative_residual,
        affine_error,
        _maximum_absolute(support_weights),
    )
    return controlled_weights, full_offset, diagnostics
end

"""
    process!(
        output::VdmCompletionArtifacts,
        workspace::VdmCompletionWorkspace,
        plan::ThinPlateVdmCompletionPlan,
        calibration::VdmCompletionCalibration,
    ) -> output

Prepare compact thin-plate completion weights and a full-VDM fixed offset.

All calibration input and complete local solver products are validated before
the caller-owned output is changed. A failed call therefore preserves every
published artifact field. This preparation method does not support progressive
execution and makes no operating-path allocation or timing claim.
"""
function process!(
    output::VdmCompletionArtifacts{T},
    ::VdmCompletionWorkspace{T},
    plan::ThinPlateVdmCompletionPlan{T},
    calibration::VdmCompletionCalibration,
) where {T<:AbstractFloat}
    _check_vdm_completion_extents(output, calibration)
    coordinate_positions, fixed_values =
        _validated_vdm_completion_calibration(T, calibration)
    controlled_weights, full_offset, diagnostics =
        _solve_thin_plate_completion(
            plan,
            calibration,
            coordinate_positions,
            fixed_values,
        )
    copyto!(output.extrapolated_from_controlled, controlled_weights)
    copyto!(output.full_vdm_offset, full_offset)
    output.diagnostics = diagnostics
    return output
end
