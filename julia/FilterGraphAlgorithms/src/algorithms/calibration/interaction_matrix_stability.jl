"""
    InteractionMatrixStabilityCriteria{T}(
        minimum_column_norm,
        maximum_relative_column_difference,
        maximum_relative_frobenius_difference,
    )
    InteractionMatrixStabilityCriteria(
        minimum_column_norm,
        maximum_relative_column_difference,
        maximum_relative_frobenius_difference,
    )

Prepare acceptance criteria for independent-window interaction-matrix
stability.

`minimum_column_norm` must be finite and positive. Both maximum symmetric
relative differences must be finite and in `[0, 2]`. The column norm has the
interaction-matrix column unit; the two relative criteria are dimensionless.
Preparation preserves the selected `AbstractFloat` representation.
"""
struct InteractionMatrixStabilityCriteria{
    T<:AbstractFloat,
}
    minimum_column_norm::T
    maximum_relative_column_difference::T
    maximum_relative_frobenius_difference::T

    function InteractionMatrixStabilityCriteria{T}(
        minimum_column_norm::Real,
        maximum_relative_column_difference::Real,
        maximum_relative_frobenius_difference::Real,
    ) where {T<:AbstractFloat}
        minimum_norm = T(minimum_column_norm)
        isfinite(minimum_norm) && minimum_norm > zero(T) ||
            throw(
                ArgumentError(
                    "minimum_column_norm must be finite and positive",
                ),
            )
        maximum_column_difference = T(maximum_relative_column_difference)
        isfinite(maximum_column_difference) &&
            zero(T) <= maximum_column_difference <= T(2) ||
            throw(
                ArgumentError(
                    "maximum_relative_column_difference must be finite and in [0, 2]",
                ),
            )
        maximum_frobenius_difference =
            T(maximum_relative_frobenius_difference)
        isfinite(maximum_frobenius_difference) &&
            zero(T) <= maximum_frobenius_difference <= T(2) ||
            throw(
                ArgumentError(
                    "maximum_relative_frobenius_difference must be finite and in [0, 2]",
                ),
            )
        return new{T}(
            minimum_norm,
            maximum_column_difference,
            maximum_frobenius_difference,
        )
    end
end

function InteractionMatrixStabilityCriteria(
    minimum_column_norm::Real,
    maximum_relative_column_difference::Real,
    maximum_relative_frobenius_difference::Real,
)
    T = float(
        promote_type(
            typeof(minimum_column_norm),
            typeof(maximum_relative_column_difference),
            typeof(maximum_relative_frobenius_difference),
        ),
    )
    T <: AbstractFloat ||
        throw(
            ArgumentError(
                "interaction-matrix stability preparation requires an AbstractFloat",
            ),
        )
    return InteractionMatrixStabilityCriteria{T}(
        minimum_column_norm,
        maximum_relative_column_difference,
        maximum_relative_frobenius_difference,
    )
end

"""
    InteractionMatrixStabilityStatus

Classify one independent-window interaction-matrix stability comparison.

Checks occur in this order: insufficient column norm, excessive relative
column difference, excessive relative Frobenius difference, then valid.
Coordinates associated with a status are reported by
[`InteractionMatrixStabilityDiagnostics`](@ref).
"""
@enum InteractionMatrixStabilityStatus::UInt8 begin
    InteractionMatrixStabilityValid
    InteractionMatrixStabilityInsufficientColumnNorm
    InteractionMatrixStabilityExcessiveRelativeColumnDifference
    InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference
end

@doc "Status indicating that both interaction matrices satisfy every stability criterion."
InteractionMatrixStabilityValid
@doc "Status indicating insufficient energy in at least one matrix column."
InteractionMatrixStabilityInsufficientColumnNorm
@doc "Status indicating an excessive symmetric relative difference in one matrix column."
InteractionMatrixStabilityExcessiveRelativeColumnDifference
@doc "Status indicating an excessive symmetric relative Frobenius difference."
InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference

"""Expected numerical rejection from interaction-matrix stability analysis."""
@enum InteractionMatrixStabilityFailure::UInt8 begin
    InteractionMatrixStabilityNonFiniteFirstWindow = 1
    InteractionMatrixStabilityNonFiniteSecondWindow = 2
    InteractionMatrixStabilityNonFiniteCalculation = 3
end

@doc "The first-window interaction matrix contains a non-finite coefficient."
InteractionMatrixStabilityNonFiniteFirstWindow
@doc "The second-window interaction matrix contains a non-finite coefficient."
InteractionMatrixStabilityNonFiniteSecondWindow
@doc "An interaction-matrix stability calculation produced a non-finite value."
InteractionMatrixStabilityNonFiniteCalculation

"""
    InteractionMatrixStabilityDiagnostics{T}()
    InteractionMatrixStabilityDiagnostics(plan)

Allocate caller-owned diagnostics for an independent-window comparison.

Corrector coordinates are one-based. `status_corrector_coordinate` is the
lowest insufficient-norm coordinate or the maximum-difference coordinate for
the corresponding status, and is zero when no coordinate applies.
`maximum_relative_column_difference_coordinate` is the lowest coordinate on
a tie. A failed [`process!`](@ref) call preserves every field.
"""
mutable struct InteractionMatrixStabilityDiagnostics{
    T<:AbstractFloat,
}
    status::InteractionMatrixStabilityStatus
    status_corrector_coordinate::Int
    minimum_first_window_column_norm::T
    minimum_second_window_column_norm::T
    maximum_relative_column_difference::T
    maximum_relative_column_difference_coordinate::Int
    relative_frobenius_difference::T
end

function InteractionMatrixStabilityDiagnostics{T}() where {
    T<:AbstractFloat,
}
    return InteractionMatrixStabilityDiagnostics{T}(
        InteractionMatrixStabilityInsufficientColumnNorm,
        0,
        zero(T),
        zero(T),
        zero(T),
        0,
        zero(T),
    )
end

"""
    InteractionMatrixStabilityInput(first_window, second_window)

Bundle two independently identified interaction matrices without copying.

Both matrices have the same `AbstractFloat` element type and logical
measurement-by-corrector orientation. The caller owns acquisition
independence, persistence, and coherent artifact activation and must not
mutate either matrix during [`process!`](@ref).
"""
struct InteractionMatrixStabilityInput{
    T<:AbstractFloat,
    A<:AbstractMatrix{T},
    B<:AbstractMatrix{T},
}
    first_window::A
    second_window::B
end

"""
    InteractionMatrixStabilityPlan(
        measurement_count,
        corrector_coordinate_count,
        criteria,
    )

Prepare diagnostics comparing two independent interaction-matrix windows.

Both extents must be positive and their product must fit in `Int`. The plan
retains immutable [`InteractionMatrixStabilityCriteria`](@ref) and performs no
artifact activation or truth selection.
"""
struct InteractionMatrixStabilityPlan{
    T<:AbstractFloat,
} <: AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    matrix_count::Int
    criteria::InteractionMatrixStabilityCriteria{T}
    calibration_plan::AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixStabilityPlan{T}
end

function InteractionMatrixStabilityPlan(
    measurement_count::Integer,
    corrector_coordinate_count::Integer,
    criteria::InteractionMatrixStabilityCriteria{T},
) where {T<:AbstractFloat}
    measurements = Int(measurement_count)
    correctors = Int(corrector_coordinate_count)
    measurements > 0 && correctors > 0 ||
        throw(
            ArgumentError(
                "measurement and corrector-coordinate extents must be positive",
            ),
        )
    matrix_count = try
        Base.Checked.checked_mul(measurements, correctors)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("interaction-matrix extent overflows Int"))
    end
    calibration_method =
        AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixStability{T}(
            criteria.minimum_column_norm,
            criteria.maximum_relative_column_difference,
            criteria.maximum_relative_frobenius_difference,
        )
    calibration_plan = AdaptiveOpticsCalibration.prepare(
        calibration_method,
        AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixStabilitySpecification(
            measurements,
            correctors,
        ),
    )
    return InteractionMatrixStabilityPlan{T}(
        measurements,
        correctors,
        matrix_count,
        criteria,
        calibration_plan,
    )
end

function InteractionMatrixStabilityDiagnostics(
    ::InteractionMatrixStabilityPlan{T},
) where {T<:AbstractFloat}
    return InteractionMatrixStabilityDiagnostics{T}()
end

"""
    InteractionMatrixStabilityWorkspace(plan)

Allocate the caller-owned Calibration diagnostics product used by the direct
and declared interaction-matrix stability adapter.
"""
mutable struct InteractionMatrixStabilityWorkspace{T<:AbstractFloat}
    product::AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixStabilityProduct{T}
end

function InteractionMatrixStabilityWorkspace(
    plan::InteractionMatrixStabilityPlan{T},
) where {T<:AbstractFloat}
    return InteractionMatrixStabilityWorkspace{T}(
        AdaptiveOpticsCalibration.allocate_result(plan.calibration_plan),
    )
end

reset!(workspace::InteractionMatrixStabilityWorkspace) = workspace
commit!(workspace::InteractionMatrixStabilityWorkspace) = workspace
discard!(workspace::InteractionMatrixStabilityWorkspace) = workspace

const INTERACTION_MATRIX_STABILITY_MAXIMUM_COLUMN_PROPERTY = UInt32(0)
const INTERACTION_MATRIX_STABILITY_MAXIMUM_FROBENIUS_PROPERTY = UInt32(1)
const _INTERACTION_MATRIX_STABILITY_PROPERTY_NAMES = (
    Symbol("maximum-relative-column-difference"),
    Symbol("maximum-relative-frobenius-difference"),
)

function property_schema(
    ::Type{InteractionMatrixStabilityPlan{T}},
) where {T<:Union{Float32,Float64}}
    relative_range = PropertyRange(zero(T), T(2))
    return (
        runtime_property(
            INTERACTION_MATRIX_STABILITY_MAXIMUM_COLUMN_PROPERTY,
            Symbol("maximum-relative-column-difference"),
            T;
            default=T(0.5),
            description="Largest accepted symmetric relative column difference",
            unit="1",
            constraint=relative_range,
        ),
        runtime_property(
            INTERACTION_MATRIX_STABILITY_MAXIMUM_FROBENIUS_PROPERTY,
            Symbol("maximum-relative-frobenius-difference"),
            T;
            default=T(0.5),
            description="Largest accepted symmetric relative Frobenius difference",
            unit="1",
            constraint=relative_range,
        ),
    )
end

function property_values(
    plan::InteractionMatrixStabilityPlan{T},
) where {T<:Union{Float32,Float64}}
    criteria = plan.criteria
    return NamedTuple{_INTERACTION_MATRIX_STABILITY_PROPERTY_NAMES}(
        (
            criteria.maximum_relative_column_difference,
            criteria.maximum_relative_frobenius_difference,
        ),
    )
end

function replace_properties(
    plan::InteractionMatrixStabilityPlan{T},
    values::NamedTuple{
        _INTERACTION_MATRIX_STABILITY_PROPERTY_NAMES,
        Tuple{T,T},
    },
) where {T<:Union{Float32,Float64}}
    criteria = InteractionMatrixStabilityCriteria{T}(
        plan.criteria.minimum_column_norm,
        getproperty(values, Symbol("maximum-relative-column-difference")),
        getproperty(values, Symbol("maximum-relative-frobenius-difference")),
    )
    return InteractionMatrixStabilityPlan(
        plan.measurement_count,
        plan.corrector_coordinate_count,
        criteria,
    )
end

function _check_stability_input_shape(
    plan::InteractionMatrixStabilityPlan,
    first_window::AbstractMatrix,
    second_window::AbstractMatrix,
)
    expected = (plan.measurement_count, plan.corrector_coordinate_count)
    size(first_window) == expected ||
        throw(
            DimensionMismatch(
                "first_window must have shape $expected",
            ),
        )
    size(second_window) == expected ||
        throw(
            DimensionMismatch(
                "second_window must have shape $expected",
            ),
        )
    return nothing
end

function _publish_stability_diagnostics!(
    output::InteractionMatrixStabilityDiagnostics{T},
    diagnostics::AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixStabilityProduct{T},
) where {T<:AbstractFloat}
    output.status = _fga_stability_status(diagnostics.status)
    output.status_corrector_coordinate =
        diagnostics.status_corrector_coordinate
    output.minimum_first_window_column_norm =
        diagnostics.minimum_first_window_column_norm
    output.minimum_second_window_column_norm =
        diagnostics.minimum_second_window_column_norm
    output.maximum_relative_column_difference =
        diagnostics.maximum_relative_column_difference
    output.maximum_relative_column_difference_coordinate =
        diagnostics.maximum_relative_column_difference_coordinate
    output.relative_frobenius_difference =
        diagnostics.relative_frobenius_difference
    return output
end

function _fga_stability_status(
    status::AdaptiveOpticsCalibration.Diagnostics.StabilityStatus,
)
    if status === AdaptiveOpticsCalibration.Diagnostics.Stable
        return InteractionMatrixStabilityValid
    elseif status === AdaptiveOpticsCalibration.Diagnostics.InsufficientColumnNorm
        return InteractionMatrixStabilityInsufficientColumnNorm
    elseif status ===
           AdaptiveOpticsCalibration.Diagnostics.ExcessiveRelativeColumnDifference
        return InteractionMatrixStabilityExcessiveRelativeColumnDifference
    elseif status ===
           AdaptiveOpticsCalibration.Diagnostics.ExcessiveRelativeFrobeniusDifference
        return InteractionMatrixStabilityExcessiveRelativeFrobeniusDifference
    end
    throw(ArgumentError("unexpected Calibration stability status"))
end

"""Delegate stability arithmetic to the Calibration-owned numerical plan."""
function _calculate_interaction_matrix_stability(
    plan::InteractionMatrixStabilityPlan{T},
    workspace::InteractionMatrixStabilityWorkspace{T},
    first_window::AbstractMatrix{T},
    second_window::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_stability_input_shape(plan, first_window, second_window)
    for value in first_window
        isfinite(value) ||
            return InteractionMatrixStabilityNonFiniteFirstWindow
    end
    for value in second_window
        isfinite(value) ||
            return InteractionMatrixStabilityNonFiniteSecondWindow
    end

    AdaptiveOpticsCalibration.process!(
        workspace.product,
        nothing,
        plan.calibration_plan,
        AdaptiveOpticsCalibration.Diagnostics.InteractionMatrixPair(
            first_window,
            second_window,
        ),
    )
    workspace.product.status ===
    AdaptiveOpticsCalibration.Diagnostics.NonFiniteStabilityCalculation &&
        return InteractionMatrixStabilityNonFiniteCalculation
    return workspace.product
end

function _throw_interaction_matrix_stability_failure(
    failure::InteractionMatrixStabilityFailure,
)
    if failure === InteractionMatrixStabilityNonFiniteFirstWindow
        throw(ArgumentError("first_window must be finite"))
    elseif failure === InteractionMatrixStabilityNonFiniteSecondWindow
        throw(ArgumentError("second_window must be finite"))
    end
    throw(
        ArgumentError(
            "interaction-matrix stability calculation produced a non-finite value",
        ),
    )
end

function process!(
    output::InteractionMatrixStabilityDiagnostics{T},
    workspace::InteractionMatrixStabilityWorkspace{T},
    plan::InteractionMatrixStabilityPlan{T},
    input::InteractionMatrixStabilityInput{T},
) where {T<:AbstractFloat}
    result = _calculate_interaction_matrix_stability(
        plan,
        workspace,
        input.first_window,
        input.second_window,
    )
    result isa InteractionMatrixStabilityFailure &&
        _throw_interaction_matrix_stability_failure(result)
    return _publish_stability_diagnostics!(output, result)
end

function _stability_outputs_overlap(
    status::AbstractVector,
    status_coordinate::AbstractVector,
    minimum_column_norms::AbstractVector,
    maximum_relative_column_difference::AbstractVector,
    maximum_relative_column_difference_coordinate::AbstractVector,
    relative_frobenius_difference::AbstractVector,
)
    return Base.mightalias(status, status_coordinate) ||
           Base.mightalias(status, minimum_column_norms) ||
           Base.mightalias(status, maximum_relative_column_difference) ||
           Base.mightalias(
        status,
        maximum_relative_column_difference_coordinate,
    ) ||
           Base.mightalias(status, relative_frobenius_difference) ||
           Base.mightalias(status_coordinate, minimum_column_norms) ||
           Base.mightalias(
        status_coordinate,
        maximum_relative_column_difference,
    ) ||
           Base.mightalias(
        status_coordinate,
        maximum_relative_column_difference_coordinate,
    ) ||
           Base.mightalias(status_coordinate, relative_frobenius_difference) ||
           Base.mightalias(
        minimum_column_norms,
        maximum_relative_column_difference,
    ) ||
           Base.mightalias(
        minimum_column_norms,
        maximum_relative_column_difference_coordinate,
    ) ||
           Base.mightalias(
        minimum_column_norms,
        relative_frobenius_difference,
    ) ||
           Base.mightalias(
        maximum_relative_column_difference,
        maximum_relative_column_difference_coordinate,
    ) ||
           Base.mightalias(
        maximum_relative_column_difference,
        relative_frobenius_difference,
    ) ||
           Base.mightalias(
        maximum_relative_column_difference_coordinate,
        relative_frobenius_difference,
    )
end

function _process_interaction_matrix_stability!(
    status::AbstractVector{UInt8},
    status_coordinate::AbstractVector{UInt64},
    minimum_column_norms::AbstractVector{T},
    maximum_relative_column_difference::AbstractVector{T},
    maximum_relative_column_difference_coordinate::AbstractVector{UInt64},
    relative_frobenius_difference::AbstractVector{T},
    plan::InteractionMatrixStabilityPlan{T},
    workspace::InteractionMatrixStabilityWorkspace{T},
    first_window::AbstractMatrix{T},
    second_window::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(
        status,
        status_coordinate,
        minimum_column_norms,
        maximum_relative_column_difference,
        maximum_relative_column_difference_coordinate,
        relative_frobenius_difference,
    )
    length(status) == 1 ||
        throw(DimensionMismatch("stability status output extent is invalid"))
    length(status_coordinate) == 1 || throw(
        DimensionMismatch("stability status-coordinate output extent is invalid"),
    )
    length(minimum_column_norms) == 2 || throw(
        DimensionMismatch("stability column-norm output extent is invalid"),
    )
    length(maximum_relative_column_difference) == 1 || throw(
        DimensionMismatch("relative-column output extent is invalid"),
    )
    length(maximum_relative_column_difference_coordinate) == 1 || throw(
        DimensionMismatch("relative-column coordinate output extent is invalid"),
    )
    length(relative_frobenius_difference) == 1 || throw(
        DimensionMismatch("relative-Frobenius output extent is invalid"),
    )
    _stability_outputs_overlap(
        status,
        status_coordinate,
        minimum_column_norms,
        maximum_relative_column_difference,
        maximum_relative_column_difference_coordinate,
        relative_frobenius_difference,
    ) && throw(ArgumentError("interaction-matrix stability outputs must not overlap"))

    result = _calculate_interaction_matrix_stability(
        plan,
        workspace,
        first_window,
        second_window,
    )
    result isa InteractionMatrixStabilityFailure && return result

    status[1] = UInt8(_fga_stability_status(result.status))
    status_coordinate[1] = if iszero(result.status_corrector_coordinate)
        zero(UInt64)
    else
        UInt64(result.status_corrector_coordinate - 1)
    end
    minimum_column_norms[1] = result.minimum_first_window_column_norm
    minimum_column_norms[2] = result.minimum_second_window_column_norm
    maximum_relative_column_difference[1] =
        result.maximum_relative_column_difference
    maximum_relative_column_difference_coordinate[1] =
        UInt64(result.maximum_relative_column_difference_coordinate - 1)
    relative_frobenius_difference[1] = result.relative_frobenius_difference
    return nothing
end

"""
Float32 independent-window interaction-matrix stability diagnostics. The
declaration publishes the frozen FGN status encoding and zero-based corrector
coordinates while the direct Julia diagnostics API remains one-based.
"""
@algorithm InteractionMatrixStabilityF32 begin
    label = "interaction-matrix-stability-f32"
    failure = InteractionMatrixStabilityFailure

    """Configuration for [`InteractionMatrixStabilityF32`](@ref)."""
    @configuration InteractionMatrixStabilityF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        minimum_column_norm::Float32 => construction
        interaction_matrix_schema::String => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(
            :first_window,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.interaction_matrix_schema;
            name="first-window",
        ),
        input(
            :second_window,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.interaction_matrix_schema;
            name="second-window",
        ),
        output(
            :status,
            schema_element_type(INTERACTION_MATRIX_STABILITY_STATUS_V1),
            (1,),
            INTERACTION_MATRIX_STABILITY_STATUS_V1;
            metadata_from=:second_window,
        ),
        output(
            :status_coordinate,
            schema_element_type(CORRECTOR_COORDINATE_V1),
            (1,),
            CORRECTOR_COORDINATE_V1;
            name="status-coordinate",
            metadata_from=:second_window,
        ),
        output(
            :minimum_column_norms,
            schema_element_type(INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1),
            (2,),
            INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1;
            name="minimum-column-norms",
            metadata_from=:second_window,
        ),
        output(
            :maximum_relative_column_difference,
            schema_element_type(INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1),
            (1,),
            INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1;
            name="maximum-relative-column-difference",
            metadata_from=:second_window,
        ),
        output(
            :maximum_relative_column_difference_coordinate,
            schema_element_type(CORRECTOR_COORDINATE_V1),
            (1,),
            CORRECTOR_COORDINATE_V1;
            name="maximum-relative-column-difference-coordinate",
            metadata_from=:second_window,
        ),
        output(
            :relative_frobenius_difference,
            schema_element_type(INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1),
            (1,),
            INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1;
            name="relative-frobenius-difference",
            metadata_from=:second_window,
        ),
    )

    prepare(configuration) = begin
        criteria = InteractionMatrixStabilityCriteria{Float32}(
            configuration.minimum_column_norm,
            0.5f0,
            0.5f0,
        )
        plan = InteractionMatrixStabilityPlan(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
            criteria,
        )
        (plan=plan, workspace=InteractionMatrixStabilityWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_interaction_matrix_stability!(
            status,
            status_coordinate,
            minimum_column_norms,
            maximum_relative_column_difference,
            maximum_relative_column_difference_coordinate,
            relative_frobenius_difference,
            plan,
            workspace,
            first_window,
            second_window,
        )
    end
end
