const _AOCDOCRIME = AdaptiveOpticsCalibration.DOCRIME

"""
    DocrimeConditioningCriteria{T}(
        minimum_corrector_variance,
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )
    DocrimeConditioningCriteria(
        minimum_corrector_variance,
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )

Prepare covariance-conditioning acceptance criteria for DO-CRIME.

The minimum corrector variance must be finite and positive, the maximum
absolute off-diagonal corrector correlation must be finite and in `[0, 1)`,
and the minimum Cholesky pivot ratio must be finite and in `(0, 1]`.
Variance has the square of the declared corrector-coordinate unit;
correlation and pivot ratio are dimensionless. Preparation preserves the
selected `AbstractFloat` representation.
"""
struct DocrimeConditioningCriteria{T<:AbstractFloat}
    minimum_corrector_variance::T
    maximum_absolute_corrector_correlation::T
    minimum_factorization_pivot_ratio::T

    function DocrimeConditioningCriteria{T}(
        minimum_corrector_variance::Real,
        maximum_absolute_corrector_correlation::Real,
        minimum_factorization_pivot_ratio::Real,
    ) where {T<:AbstractFloat}
        minimum_variance = T(minimum_corrector_variance)
        isfinite(minimum_variance) && minimum_variance > zero(T) ||
            throw(
                ArgumentError(
                    "minimum_corrector_variance must be finite and positive",
                ),
            )
        maximum_correlation = T(maximum_absolute_corrector_correlation)
        isfinite(maximum_correlation) &&
            zero(T) <= maximum_correlation < one(T) ||
            throw(
                ArgumentError(
                    "maximum_absolute_corrector_correlation must be finite and in [0, 1)",
                ),
            )
        minimum_pivot_ratio = T(minimum_factorization_pivot_ratio)
        isfinite(minimum_pivot_ratio) &&
            zero(T) < minimum_pivot_ratio <= one(T) ||
            throw(
                ArgumentError(
                    "minimum_factorization_pivot_ratio must be finite and in (0, 1]",
                ),
            )
        return new{T}(
            minimum_variance,
            maximum_correlation,
            minimum_pivot_ratio,
        )
    end
end

function DocrimeConditioningCriteria(
    minimum_corrector_variance::Real,
    maximum_absolute_corrector_correlation::Real,
    minimum_factorization_pivot_ratio::Real,
)
    T = float(
        promote_type(
            typeof(minimum_corrector_variance),
            typeof(maximum_absolute_corrector_correlation),
            typeof(minimum_factorization_pivot_ratio),
        ),
    )
    T <: AbstractFloat ||
        throw(
            ArgumentError(
                "DO-CRIME interaction-matrix preparation requires an AbstractFloat",
            ),
        )
    return DocrimeConditioningCriteria{T}(
        minimum_corrector_variance,
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )
end

"""
    DocrimeInteractionMatrixStatus

Classify the expected validity of one DO-CRIME interaction-matrix artifact.

The status checks occur in this order: sample count, corrector variance,
off-diagonal corrector correlation, positive-definiteness, and factorization
pivot ratio. Invalid statuses publish a zero interaction matrix rather than
raising an exception.
"""
@enum DocrimeInteractionMatrixStatus::UInt8 begin
    DocrimeInteractionMatrixValid
    DocrimeInteractionMatrixInsufficientSamples
    DocrimeInteractionMatrixInsufficientCorrectorVariance
    DocrimeInteractionMatrixExcessiveCorrectorCorrelation
    DocrimeInteractionMatrixNonPositiveDefinite
    DocrimeInteractionMatrixInsufficientFactorizationPivot
end

@doc "Status indicating that a DO-CRIME interaction matrix is valid."
DocrimeInteractionMatrixValid
@doc "Status indicating that fewer than two covariance samples are available."
DocrimeInteractionMatrixInsufficientSamples
@doc "Status indicating insufficient variance in one corrector coordinate."
DocrimeInteractionMatrixInsufficientCorrectorVariance
@doc "Status indicating excessive correlation between two corrector coordinates."
DocrimeInteractionMatrixExcessiveCorrectorCorrelation
@doc "Status indicating a non-positive-definite corrector covariance."
DocrimeInteractionMatrixNonPositiveDefinite
@doc "Status indicating an insufficient positive Cholesky pivot ratio."
DocrimeInteractionMatrixInsufficientFactorizationPivot

"""
    DocrimeInteractionMatrixDiagnostics{T}

Report DO-CRIME covariance conditioning and validity.

Corrector coordinates are one-based. `first_corrector_coordinate` is zero
when no status coordinate applies; `second_corrector_coordinate` is nonzero
only for [`DocrimeInteractionMatrixExcessiveCorrectorCorrelation`](@ref). The variance,
correlation, and pivot fields are zero when their stage was not reached.
"""
struct DocrimeInteractionMatrixDiagnostics{T<:AbstractFloat}
    status::DocrimeInteractionMatrixStatus
    sample_count::Int
    minimum_corrector_variance::T
    maximum_corrector_variance::T
    maximum_absolute_corrector_correlation::T
    minimum_factorization_pivot_ratio::T
    first_corrector_coordinate::Int
    second_corrector_coordinate::Int
end

function _initial_docrime_diagnostics(
    ::Type{T},
    sample_count::Int=0,
) where {T<:AbstractFloat}
    return DocrimeInteractionMatrixDiagnostics{T}(
        DocrimeInteractionMatrixInsufficientSamples,
        sample_count,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
    )
end

"""
    DocrimeCovariances(
        sample_count,
        measurement_corrector_cross_covariance_sum,
        corrector_covariance_sum,
    )

Bundle frozen centered covariance sums for a DO-CRIME solve without copying.

For `q` measurements and `m` corrector coordinates, `S_mc` has shape `q` by
`m` and `S_cc` has shape `m` by `m`. `sample_count` and both matrices must be
the coherent snapshot produced by one covariance accumulator. Open-loop
identification uses realized calibration excitation as the corrector sample;
closed-loop identification uses the complete exposure-associated corrector
command increment `delta_c`.
"""
struct DocrimeCovariances{
    C<:AbstractMatrix{<:Real},
    V<:AbstractMatrix{<:Real},
}
    sample_count::Int
    measurement_corrector_cross_covariance_sum::C
    corrector_covariance_sum::V

    function DocrimeCovariances(
        sample_count::Integer,
        measurement_corrector_cross_covariance_sum::C,
        corrector_covariance_sum::V,
    ) where {
        C<:AbstractMatrix{<:Real},
        V<:AbstractMatrix{<:Real},
    }
        sample_count >= 0 || throw(ArgumentError("sample_count must be nonnegative"))
        sample_count <= typemax(Int) || throw(
            ArgumentError("sample_count must fit Int"),
        )
        count = Int(sample_count)
        return new{C,V}(
            count,
            measurement_corrector_cross_covariance_sum,
            corrector_covariance_sum,
        )
    end
end

function _docrime_interaction_dimension(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("DO-CRIME $name extent must be positive"))
    value <= typemax(Int) || throw(
        ArgumentError("DO-CRIME $name extent must fit Int"),
    )
    return Int(value)
end

function _docrime_interaction_matrix_count(
    name::AbstractString,
    rows::Int,
    columns::Int,
)
    try
        return Base.Checked.checked_mul(rows, columns)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("DO-CRIME $name matrix extent overflows Int"))
    end
end

"""
    DocrimeOpenLoopInteractionMatrixPlan(
        measurement_count,
        corrector_coordinate_count,
        criteria,
    )

Prepare the open-loop DO-CRIME solve
`D = C_mc * inverse(C_cc)` without forming the inverse.
"""
struct DocrimeOpenLoopInteractionMatrixPlan{T<:AbstractFloat} <:
       AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    criteria::DocrimeConditioningCriteria{T}
    calibration_plan::_AOCDOCRIME.InteractionMatrixPlan{
        T,
        _AOCDOCRIME.OpenLoop{T},
    }

    function DocrimeOpenLoopInteractionMatrixPlan{T}(
        measurement_count::Int,
        corrector_coordinate_count::Int,
        criteria::DocrimeConditioningCriteria{T},
    ) where {T<:AbstractFloat}
        measurement_count > 0 && corrector_coordinate_count > 0 ||
            throw(
                ArgumentError(
                    "DO-CRIME measurement and corrector-coordinate extents must be positive",
                ),
            )
        _docrime_interaction_matrix_count(
            "measurement-by-corrector",
            measurement_count,
            corrector_coordinate_count,
        )
        _docrime_interaction_matrix_count(
            "corrector-by-corrector",
            corrector_coordinate_count,
            corrector_coordinate_count,
        )
        calibration_criteria = _AOCDOCRIME.ConditioningCriteria{T}(
            criteria.minimum_corrector_variance,
            criteria.maximum_absolute_corrector_correlation,
            criteria.minimum_factorization_pivot_ratio,
        )
        calibration_plan = _AOCDOCRIME.prepare(
            _AOCDOCRIME.OpenLoop(calibration_criteria),
            _AOCDOCRIME.CovarianceSpecification{T}(
                measurement_count,
                corrector_coordinate_count,
            ),
        )
        return new{T}(
            measurement_count,
            corrector_coordinate_count,
            criteria,
            calibration_plan,
        )
    end
end

function DocrimeOpenLoopInteractionMatrixPlan(
    measurement_count::Integer,
    corrector_coordinate_count::Integer,
    criteria::DocrimeConditioningCriteria{T},
) where {T<:AbstractFloat}
    measurements = _docrime_interaction_dimension(
        "measurement",
        measurement_count,
    )
    correctors = _docrime_interaction_dimension(
        "corrector-coordinate",
        corrector_coordinate_count,
    )
    return DocrimeOpenLoopInteractionMatrixPlan{T}(
        measurements,
        correctors,
        criteria,
    )
end

"""
    DocrimeClosedLoopInteractionMatrixPlan(
        measurement_count,
        corrector_coordinate_count,
        loop_gain,
        criteria,
    )

Prepare the closed-loop DO-CRIME solve
`D = g * C_m_delta_c * inverse(C_delta_c_delta_c)`.

`loop_gain` is the finite positive scalar gain `g`. The input corrector sample
is the complete exposure-associated command increment `delta_c`, not merely
the injected calibration excitation.
"""
struct DocrimeClosedLoopInteractionMatrixPlan{T<:AbstractFloat} <:
       AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    loop_gain::T
    criteria::DocrimeConditioningCriteria{T}
    calibration_plan::_AOCDOCRIME.InteractionMatrixPlan{
        T,
        _AOCDOCRIME.ClosedLoop{T},
    }

    function DocrimeClosedLoopInteractionMatrixPlan{T}(
        measurement_count::Int,
        corrector_coordinate_count::Int,
        loop_gain::T,
        criteria::DocrimeConditioningCriteria{T},
    ) where {T<:AbstractFloat}
        measurement_count > 0 && corrector_coordinate_count > 0 ||
            throw(
                ArgumentError(
                    "DO-CRIME measurement and corrector-coordinate extents must be positive",
                ),
            )
        _docrime_interaction_matrix_count(
            "measurement-by-corrector",
            measurement_count,
            corrector_coordinate_count,
        )
        _docrime_interaction_matrix_count(
            "corrector-by-corrector",
            corrector_coordinate_count,
            corrector_coordinate_count,
        )
        isfinite(loop_gain) && loop_gain > zero(T) ||
            throw(ArgumentError("loop_gain must be finite and positive"))
        calibration_criteria = _AOCDOCRIME.ConditioningCriteria{T}(
            criteria.minimum_corrector_variance,
            criteria.maximum_absolute_corrector_correlation,
            criteria.minimum_factorization_pivot_ratio,
        )
        calibration_plan = _AOCDOCRIME.prepare(
            _AOCDOCRIME.ClosedLoop(loop_gain, calibration_criteria),
            _AOCDOCRIME.CovarianceSpecification{T}(
                measurement_count,
                corrector_coordinate_count,
            ),
        )
        return new{T}(
            measurement_count,
            corrector_coordinate_count,
            loop_gain,
            criteria,
            calibration_plan,
        )
    end
end

function DocrimeClosedLoopInteractionMatrixPlan(
    measurement_count::Integer,
    corrector_coordinate_count::Integer,
    loop_gain::Real,
    criteria::DocrimeConditioningCriteria{T},
) where {T<:AbstractFloat}
    open_loop = DocrimeOpenLoopInteractionMatrixPlan(
        measurement_count,
        corrector_coordinate_count,
        criteria,
    )
    gain = T(loop_gain)
    return DocrimeClosedLoopInteractionMatrixPlan{T}(
        open_loop.measurement_count,
        open_loop.corrector_coordinate_count,
        gain,
        criteria,
    )
end

"""
    DocrimeInteractionMatrixArtifacts(plan)

Allocate caller-owned storage for one DO-CRIME interaction matrix and its
diagnostics.

`interaction_matrix` has measurement rows and corrector-coordinate columns.
Its coefficients have measurement-unit/corrector-unit. Expected invalid
conditioning publishes a zero matrix and diagnostic status. This is the
logical matrix orientation; artifact serialization order is a caller-owned
transport concern.
"""
mutable struct DocrimeInteractionMatrixArtifacts{T<:AbstractFloat}
    interaction_matrix::Matrix{T}
    diagnostics::DocrimeInteractionMatrixDiagnostics{T}
end

function DocrimeInteractionMatrixArtifacts(
    plan::Union{
        DocrimeOpenLoopInteractionMatrixPlan{T},
        DocrimeClosedLoopInteractionMatrixPlan{T},
    },
) where {T<:AbstractFloat}
    return DocrimeInteractionMatrixArtifacts{T}(
        zeros(T, plan.measurement_count, plan.corrector_coordinate_count),
        _initial_docrime_diagnostics(T),
    )
end
const DOCRIME_OPEN_LOOP_MAXIMUM_CORRELATION_PROPERTY = UInt32(0)
const DOCRIME_OPEN_LOOP_MINIMUM_PIVOT_PROPERTY = UInt32(1)
const DOCRIME_CLOSED_LOOP_GAIN_PROPERTY = UInt32(0)
const DOCRIME_CLOSED_LOOP_MAXIMUM_CORRELATION_PROPERTY = UInt32(1)
const DOCRIME_CLOSED_LOOP_MINIMUM_PIVOT_PROPERTY = UInt32(2)

const _DOCRIME_OPEN_LOOP_PROPERTY_NAMES = (
    Symbol("maximum-absolute-corrector-correlation"),
    Symbol("minimum-factorization-pivot-ratio"),
)
const _DOCRIME_CLOSED_LOOP_PROPERTY_NAMES = (
    Symbol("loop-gain"),
    Symbol("maximum-absolute-corrector-correlation"),
    Symbol("minimum-factorization-pivot-ratio"),
)

function property_schema(
    ::Type{DocrimeOpenLoopInteractionMatrixPlan{T}},
) where {T<:Union{Float32,Float64}}
    return (
        runtime_property(
            DOCRIME_OPEN_LOOP_MAXIMUM_CORRELATION_PROPERTY,
            Symbol("maximum-absolute-corrector-correlation"),
            T;
            default=T(0.9),
            description="Largest accepted absolute corrector correlation",
            unit="1",
            constraint=PropertyRange(zero(T), prevfloat(one(T))),
        ),
        runtime_property(
            DOCRIME_OPEN_LOOP_MINIMUM_PIVOT_PROPERTY,
            Symbol("minimum-factorization-pivot-ratio"),
            T;
            default=T(0.01),
            description="Smallest accepted factorization pivot ratio",
            unit="1",
            constraint=PropertyRange(nextfloat(zero(T)), one(T)),
        ),
    )
end

function property_values(
    plan::DocrimeOpenLoopInteractionMatrixPlan{T},
) where {T<:Union{Float32,Float64}}
    criteria = plan.criteria
    return NamedTuple{_DOCRIME_OPEN_LOOP_PROPERTY_NAMES}(
        (
            criteria.maximum_absolute_corrector_correlation,
            criteria.minimum_factorization_pivot_ratio,
        ),
    )
end

function replace_properties(
    plan::DocrimeOpenLoopInteractionMatrixPlan{T},
    values::NamedTuple{
        _DOCRIME_OPEN_LOOP_PROPERTY_NAMES,
        Tuple{T,T},
    },
) where {T<:Union{Float32,Float64}}
    criteria = DocrimeConditioningCriteria{T}(
        plan.criteria.minimum_corrector_variance,
        getproperty(
            values,
            Symbol("maximum-absolute-corrector-correlation"),
        ),
        getproperty(values, Symbol("minimum-factorization-pivot-ratio")),
    )
    return DocrimeOpenLoopInteractionMatrixPlan(
        plan.measurement_count,
        plan.corrector_coordinate_count,
        criteria,
    )
end

function property_schema(
    ::Type{DocrimeClosedLoopInteractionMatrixPlan{T}},
) where {T<:Union{Float32,Float64}}
    return (
        runtime_property(
            DOCRIME_CLOSED_LOOP_GAIN_PROPERTY,
            Symbol("loop-gain"),
            T;
            default=one(T),
            description="Closed-loop response scale",
            unit="1",
            constraint=PropertyRange(nextfloat(zero(T)), floatmax(T)),
        ),
        runtime_property(
            DOCRIME_CLOSED_LOOP_MAXIMUM_CORRELATION_PROPERTY,
            Symbol("maximum-absolute-corrector-correlation"),
            T;
            default=T(0.9),
            description="Largest accepted absolute corrector correlation",
            unit="1",
            constraint=PropertyRange(zero(T), prevfloat(one(T))),
        ),
        runtime_property(
            DOCRIME_CLOSED_LOOP_MINIMUM_PIVOT_PROPERTY,
            Symbol("minimum-factorization-pivot-ratio"),
            T;
            default=T(0.01),
            description="Smallest accepted factorization pivot ratio",
            unit="1",
            constraint=PropertyRange(nextfloat(zero(T)), one(T)),
        ),
    )
end

function property_values(
    plan::DocrimeClosedLoopInteractionMatrixPlan{T},
) where {T<:Union{Float32,Float64}}
    criteria = plan.criteria
    return NamedTuple{_DOCRIME_CLOSED_LOOP_PROPERTY_NAMES}(
        (
            plan.loop_gain,
            criteria.maximum_absolute_corrector_correlation,
            criteria.minimum_factorization_pivot_ratio,
        ),
    )
end

function replace_properties(
    plan::DocrimeClosedLoopInteractionMatrixPlan{T},
    values::NamedTuple{
        _DOCRIME_CLOSED_LOOP_PROPERTY_NAMES,
        Tuple{T,T,T},
    },
) where {T<:Union{Float32,Float64}}
    criteria = DocrimeConditioningCriteria{T}(
        plan.criteria.minimum_corrector_variance,
        getproperty(
            values,
            Symbol("maximum-absolute-corrector-correlation"),
        ),
        getproperty(values, Symbol("minimum-factorization-pivot-ratio")),
    )
    return DocrimeClosedLoopInteractionMatrixPlan(
        plan.measurement_count,
        plan.corrector_coordinate_count,
        getproperty(values, Symbol("loop-gain")),
        criteria,
    )
end

"""
    DocrimeInteractionMatrixWorkspace(plan)

Own the AdaptiveOpticsCalibration product and numerical workspace used by one
DO-CRIME interaction-matrix Node. The Workspace contains scratch rather than
mathematical history, so commit and discard are no-ops.
"""
mutable struct DocrimeInteractionMatrixWorkspace{
    T<:AbstractFloat,
    P<:_AOCDOCRIME.InteractionMatrixProduct{T},
    W<:_AOCDOCRIME.InteractionMatrixWorkspace{T},
}
    calibration_product::P
    calibration_workspace::W
    diagnostics::DocrimeInteractionMatrixDiagnostics{T}
end

function DocrimeInteractionMatrixWorkspace(
    plan::Union{
        DocrimeOpenLoopInteractionMatrixPlan{T},
        DocrimeClosedLoopInteractionMatrixPlan{T},
    },
) where {T<:AbstractFloat}
    product = AdaptiveOpticsCalibration.allocate_result(plan.calibration_plan)
    workspace = AdaptiveOpticsCalibration.allocate_workspace(plan.calibration_plan)
    return DocrimeInteractionMatrixWorkspace{
        T,
        typeof(product),
        typeof(workspace),
    }(
        product,
        workspace,
        _initial_docrime_diagnostics(T),
    )
end

commit!(workspace::DocrimeInteractionMatrixWorkspace) = workspace
discard!(workspace::DocrimeInteractionMatrixWorkspace) = workspace

function reset!(workspace::DocrimeInteractionMatrixWorkspace{T}) where {T}
    product = workspace.calibration_product
    calibration_workspace = workspace.calibration_workspace
    fill!(product.matrix, zero(T))
    product.diagnostics = _AOCDOCRIME.InteractionMatrixDiagnostics{T}(
        _AOCDOCRIME.InsufficientSamples,
        0,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
    )
    fill!(calibration_workspace.corrector_covariance, zero(T))
    fill!(calibration_workspace.factor, zero(T))
    fill!(calibration_workspace.staged_interaction_matrix, zero(T))
    fill!(calibration_workspace.solve_scratch, zero(T))
    calibration_workspace.diagnostics = product.diagnostics
    workspace.diagnostics = _initial_docrime_diagnostics(T)
    return workspace
end

"""Expected input or numerical rejection from a DO-CRIME solve."""
@enum DocrimeInteractionMatrixFailure::UInt8 begin
    DocrimeInteractionMatrixNonFiniteCrossCovariance = 1
    DocrimeInteractionMatrixNonFiniteCorrectorCovariance = 2
    DocrimeInteractionMatrixNonSymmetricCorrectorCovariance = 3
    DocrimeInteractionMatrixSampleCountNotRepresentable = 4
    DocrimeInteractionMatrixSampleCountOutOfRange = 5
    DocrimeInteractionMatrixNonFiniteNormalization = 6
    DocrimeInteractionMatrixNonFiniteCorrelation = 7
    DocrimeInteractionMatrixNonFiniteFactorization = 8
    DocrimeInteractionMatrixNonFiniteSolve = 9
end

@doc "The measurement--corrector covariance sum contains a non-finite value."
DocrimeInteractionMatrixNonFiniteCrossCovariance
@doc "The corrector covariance sum contains a non-finite value."
DocrimeInteractionMatrixNonFiniteCorrectorCovariance
@doc "The corrector covariance sum is not exactly symmetric."
DocrimeInteractionMatrixNonSymmetricCorrectorCovariance
@doc "The sample covariance divisor is not exactly representable."
DocrimeInteractionMatrixSampleCountNotRepresentable
@doc "The UInt64 sample count is outside Julia's native Int range."
DocrimeInteractionMatrixSampleCountOutOfRange
@doc "Sample-covariance normalization produced a non-finite value."
DocrimeInteractionMatrixNonFiniteNormalization
@doc "Corrector-correlation calculation produced a non-finite value."
DocrimeInteractionMatrixNonFiniteCorrelation
@doc "Cholesky factorization produced a non-finite value."
DocrimeInteractionMatrixNonFiniteFactorization
@doc "Triangular interaction-matrix solution produced a non-finite value."
DocrimeInteractionMatrixNonFiniteSolve

@inline function _docrime_interaction_aliases_workspace(
    array::AbstractArray,
    workspace::DocrimeInteractionMatrixWorkspace,
)
    product = workspace.calibration_product
    calibration_workspace = workspace.calibration_workspace
    return Base.mightalias(array, product.matrix) ||
           Base.mightalias(array, calibration_workspace.corrector_covariance) ||
           Base.mightalias(array, calibration_workspace.factor) ||
           Base.mightalias(
        array,
        calibration_workspace.staged_interaction_matrix,
    ) ||
           Base.mightalias(array, calibration_workspace.solve_scratch)
end

function _validate_docrime_interaction_workspace(
    workspace::DocrimeInteractionMatrixWorkspace,
    plan,
)
    product = workspace.calibration_product
    calibration_workspace = workspace.calibration_workspace
    correctors = plan.corrector_coordinate_count
    expected = (plan.measurement_count, correctors)
    size(product.matrix) == expected &&
        size(calibration_workspace.corrector_covariance) ==
        (correctors, correctors) &&
        size(calibration_workspace.factor) == (correctors, correctors) &&
        size(calibration_workspace.staged_interaction_matrix) == expected &&
        length(calibration_workspace.solve_scratch) == correctors || throw(
        DimensionMismatch("DO-CRIME interaction-matrix Workspace extent is invalid"),
    )
    return nothing
end

function _validate_docrime_interaction_inputs(
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan,
    cross_covariance_sum::AbstractMatrix,
    corrector_covariance_sum::AbstractMatrix,
) where {T}
    Base.require_one_based_indexing(
        cross_covariance_sum,
        corrector_covariance_sum,
    )
    eltype(cross_covariance_sum) === T || throw(
        ArgumentError("DO-CRIME cross covariance must use the Plan element type"),
    )
    eltype(corrector_covariance_sum) === T || throw(
        ArgumentError("DO-CRIME corrector covariance must use the Plan element type"),
    )
    size(cross_covariance_sum) ==
        (plan.measurement_count, plan.corrector_coordinate_count) || throw(
        DimensionMismatch("DO-CRIME cross-covariance extent is invalid"),
    )
    size(corrector_covariance_sum) == (
        plan.corrector_coordinate_count,
        plan.corrector_coordinate_count,
    ) || throw(
        DimensionMismatch("DO-CRIME corrector covariance extent is invalid"),
    )
    _validate_docrime_interaction_workspace(workspace, plan)
    (
        _docrime_interaction_aliases_workspace(
            cross_covariance_sum,
            workspace,
        ) ||
        _docrime_interaction_aliases_workspace(
            corrector_covariance_sum,
            workspace,
        )
    ) && throw(
        ArgumentError("DO-CRIME covariance inputs must not alias Workspace storage"),
    )
    return nothing
end

function _docrime_failure(failure::_AOCDOCRIME.InteractionMatrixFailure)
    failure === _AOCDOCRIME.NonFiniteCrossCovariance &&
        return DocrimeInteractionMatrixNonFiniteCrossCovariance
    failure === _AOCDOCRIME.NonFiniteCorrectorCovariance &&
        return DocrimeInteractionMatrixNonFiniteCorrectorCovariance
    failure === _AOCDOCRIME.NonSymmetricCorrectorCovariance &&
        return DocrimeInteractionMatrixNonSymmetricCorrectorCovariance
    failure === _AOCDOCRIME.SampleCountNotRepresentable &&
        return DocrimeInteractionMatrixSampleCountNotRepresentable
    failure === _AOCDOCRIME.SampleCountOutOfRange &&
        return DocrimeInteractionMatrixSampleCountOutOfRange
    failure === _AOCDOCRIME.NonFiniteNormalization &&
        return DocrimeInteractionMatrixNonFiniteNormalization
    failure === _AOCDOCRIME.NonFiniteCorrelation &&
        return DocrimeInteractionMatrixNonFiniteCorrelation
    failure === _AOCDOCRIME.NonFiniteFactorization &&
        return DocrimeInteractionMatrixNonFiniteFactorization
    failure === _AOCDOCRIME.NonFiniteSolve &&
        return DocrimeInteractionMatrixNonFiniteSolve
    throw(ArgumentError("unsupported AdaptiveOpticsCalibration DO-CRIME failure"))
end

function _docrime_status(status::_AOCDOCRIME.InteractionMatrixStatus)
    status === _AOCDOCRIME.Valid && return DocrimeInteractionMatrixValid
    status === _AOCDOCRIME.InsufficientSamples &&
        return DocrimeInteractionMatrixInsufficientSamples
    status === _AOCDOCRIME.InsufficientCorrectorVariance &&
        return DocrimeInteractionMatrixInsufficientCorrectorVariance
    status === _AOCDOCRIME.ExcessiveCorrectorCorrelation &&
        return DocrimeInteractionMatrixExcessiveCorrectorCorrelation
    status === _AOCDOCRIME.NonPositiveDefinite &&
        return DocrimeInteractionMatrixNonPositiveDefinite
    status === _AOCDOCRIME.InsufficientFactorizationPivot &&
        return DocrimeInteractionMatrixInsufficientFactorizationPivot
    throw(ArgumentError("unsupported AdaptiveOpticsCalibration DO-CRIME status"))
end

function _update_docrime_diagnostics!(
    workspace::DocrimeInteractionMatrixWorkspace{T},
) where {T}
    diagnostics = workspace.calibration_workspace.diagnostics
    workspace.calibration_product.diagnostics = diagnostics
    workspace.diagnostics = DocrimeInteractionMatrixDiagnostics{T}(
        _docrime_status(diagnostics.status),
        diagnostics.sample_count,
        diagnostics.minimum_corrector_variance,
        diagnostics.maximum_corrector_variance,
        diagnostics.maximum_absolute_corrector_correlation,
        diagnostics.minimum_factorization_pivot_ratio,
        diagnostics.first_corrector_coordinate,
        diagnostics.second_corrector_coordinate,
    )
    return nothing
end

function _calculate_docrime_interaction_matrix!(
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan,
    sample_count::UInt64,
    cross_covariance_sum::AbstractMatrix{T},
    corrector_covariance_sum::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _validate_docrime_interaction_inputs(
        workspace,
        plan,
        cross_covariance_sum,
        corrector_covariance_sum,
    )
    covariances = _AOCDOCRIME.CovarianceSums(
        sample_count,
        cross_covariance_sum,
        corrector_covariance_sum,
    )
    failure = AdaptiveOpticsCalibration.process!(
        workspace.calibration_product.matrix,
        workspace.calibration_workspace,
        plan.calibration_plan,
        covariances,
    )
    failure === nothing || return _docrime_failure(failure)
    _update_docrime_diagnostics!(workspace)
    return nothing
end

function _validate_docrime_artifacts(
    output::DocrimeInteractionMatrixArtifacts{T},
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan,
) where {T}
    Base.require_one_based_indexing(output.interaction_matrix)
    size(output.interaction_matrix) ==
        (plan.measurement_count, plan.corrector_coordinate_count) || throw(
        DimensionMismatch("DO-CRIME interaction-matrix output extent is invalid"),
    )
    _docrime_interaction_aliases_workspace(
        output.interaction_matrix,
        workspace,
    ) && throw(
        ArgumentError("DO-CRIME interaction-matrix output must not alias Workspace storage"),
    )
    return nothing
end

"""
    process!(output, workspace, plan, covariances)

Solve an open- or closed-loop DO-CRIME interaction matrix with fixed Workspace
storage. Expected conditioning outcomes publish a zero matrix plus diagnostics.
Malformed shapes and aliases throw at the checked boundary; numerical input or
calculation rejection returns DocrimeInteractionMatrixFailure without mutating
output.
"""
function process!(
    output::DocrimeInteractionMatrixArtifacts{T},
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan::DocrimeOpenLoopInteractionMatrixPlan{T},
    covariances::DocrimeCovariances,
) where {T<:AbstractFloat}
    _validate_docrime_artifacts(output, workspace, plan)
    result = _calculate_docrime_interaction_matrix!(
        workspace,
        plan,
        UInt64(covariances.sample_count),
        covariances.measurement_corrector_cross_covariance_sum,
        covariances.corrector_covariance_sum,
    )
    result isa DocrimeInteractionMatrixFailure && return result
    copyto!(output.interaction_matrix, workspace.calibration_product.matrix)
    output.diagnostics = workspace.diagnostics
    return output
end

function process!(
    output::DocrimeInteractionMatrixArtifacts{T},
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan::DocrimeClosedLoopInteractionMatrixPlan{T},
    covariances::DocrimeCovariances,
) where {T<:AbstractFloat}
    _validate_docrime_artifacts(output, workspace, plan)
    result = _calculate_docrime_interaction_matrix!(
        workspace,
        plan,
        UInt64(covariances.sample_count),
        covariances.measurement_corrector_cross_covariance_sum,
        covariances.corrector_covariance_sum,
    )
    result isa DocrimeInteractionMatrixFailure && return result
    copyto!(output.interaction_matrix, workspace.calibration_product.matrix)
    output.diagnostics = workspace.diagnostics
    return output
end

@inline _docrime_any_output_alias(::AbstractArray, ::Tuple{}) = false

@inline function _docrime_any_output_alias(
    array::AbstractArray,
    remaining::Tuple,
)
    return Base.mightalias(array, first(remaining)) ||
           _docrime_any_output_alias(array, Base.tail(remaining))
end

@inline _docrime_outputs_overlap(::Tuple{}) = false

@inline function _docrime_outputs_overlap(outputs::Tuple)
    return _docrime_any_output_alias(first(outputs), Base.tail(outputs)) ||
           _docrime_outputs_overlap(Base.tail(outputs))
end

function _validate_docrime_declared_outputs(
    interaction_matrix::AbstractMatrix,
    status::AbstractVector,
    status_coordinates::AbstractVector,
    diagnostic_sample_count::AbstractVector,
    corrector_variance_range::AbstractVector,
    maximum_correlation::AbstractVector,
    minimum_pivot_ratio::AbstractVector,
    workspace::DocrimeInteractionMatrixWorkspace,
    plan,
)
    Base.require_one_based_indexing(
        interaction_matrix,
        status,
        status_coordinates,
        diagnostic_sample_count,
        corrector_variance_range,
        maximum_correlation,
        minimum_pivot_ratio,
    )
    size(interaction_matrix) ==
        (plan.measurement_count, plan.corrector_coordinate_count) || throw(
        DimensionMismatch("DO-CRIME interaction-matrix output extent is invalid"),
    )
    length(status) == 1 ||
        throw(DimensionMismatch("DO-CRIME status output extent is invalid"))
    length(status_coordinates) == 2 || throw(
        DimensionMismatch("DO-CRIME status-coordinate output extent is invalid"),
    )
    length(diagnostic_sample_count) == 1 || throw(
        DimensionMismatch("DO-CRIME diagnostic sample-count extent is invalid"),
    )
    length(corrector_variance_range) == 2 || throw(
        DimensionMismatch("DO-CRIME corrector-variance output extent is invalid"),
    )
    length(maximum_correlation) == 1 || throw(
        DimensionMismatch("DO-CRIME maximum-correlation output extent is invalid"),
    )
    length(minimum_pivot_ratio) == 1 || throw(
        DimensionMismatch("DO-CRIME minimum-pivot output extent is invalid"),
    )
    outputs = (
        interaction_matrix,
        status,
        status_coordinates,
        diagnostic_sample_count,
        corrector_variance_range,
        maximum_correlation,
        minimum_pivot_ratio,
    )
    _docrime_outputs_overlap(outputs) && throw(
        ArgumentError("DO-CRIME interaction-matrix outputs must not overlap"),
    )
    (
        _docrime_interaction_aliases_workspace(interaction_matrix, workspace) ||
        _docrime_interaction_aliases_workspace(status, workspace) ||
        _docrime_interaction_aliases_workspace(status_coordinates, workspace) ||
        _docrime_interaction_aliases_workspace(
            diagnostic_sample_count,
            workspace,
        ) ||
        _docrime_interaction_aliases_workspace(
            corrector_variance_range,
            workspace,
        ) ||
        _docrime_interaction_aliases_workspace(
            maximum_correlation,
            workspace,
        ) ||
        _docrime_interaction_aliases_workspace(
            minimum_pivot_ratio,
            workspace,
        )
    ) && throw(
        ArgumentError("DO-CRIME outputs must not alias Workspace storage"),
    )
    return nothing
end

function _docrime_status_coordinates(
    diagnostics::DocrimeInteractionMatrixDiagnostics,
)
    status = diagnostics.status
    if status === DocrimeInteractionMatrixInsufficientCorrectorVariance ||
       status === DocrimeInteractionMatrixNonPositiveDefinite ||
       status === DocrimeInteractionMatrixInsufficientFactorizationPivot
        coordinate = UInt64(diagnostics.first_corrector_coordinate - 1)
        return (coordinate, coordinate)
    elseif status === DocrimeInteractionMatrixExcessiveCorrectorCorrelation
        return (
            UInt64(diagnostics.first_corrector_coordinate - 1),
            UInt64(diagnostics.second_corrector_coordinate - 1),
        )
    end
    return (zero(UInt64), zero(UInt64))
end

function _process_docrime_interaction_matrix_outputs!(
    interaction_matrix::AbstractMatrix{T},
    status::AbstractVector{UInt8},
    status_coordinates::AbstractVector{UInt64},
    diagnostic_sample_count::AbstractVector{UInt64},
    corrector_variance_range::AbstractVector{T},
    maximum_correlation::AbstractVector{T},
    minimum_pivot_ratio::AbstractVector{T},
    workspace::DocrimeInteractionMatrixWorkspace{T},
    plan,
    cross_covariance_sum::AbstractMatrix{T},
    corrector_covariance_sum::AbstractMatrix{T},
    sample_count::AbstractVector{UInt64},
) where {T<:AbstractFloat}
    _validate_docrime_declared_outputs(
        interaction_matrix,
        status,
        status_coordinates,
        diagnostic_sample_count,
        corrector_variance_range,
        maximum_correlation,
        minimum_pivot_ratio,
        workspace,
        plan,
    )
    Base.require_one_based_indexing(sample_count)
    length(sample_count) == 1 ||
        throw(DimensionMismatch("DO-CRIME sample-count input extent is invalid"))
    result = _calculate_docrime_interaction_matrix!(
        workspace,
        plan,
        sample_count[1],
        cross_covariance_sum,
        corrector_covariance_sum,
    )
    result isa DocrimeInteractionMatrixFailure && return result

    diagnostics = workspace.diagnostics
    coordinates = _docrime_status_coordinates(diagnostics)
    copyto!(interaction_matrix, workspace.calibration_product.matrix)
    status[1] = UInt8(diagnostics.status)
    status_coordinates[1] = coordinates[1]
    status_coordinates[2] = coordinates[2]
    diagnostic_sample_count[1] = UInt64(diagnostics.sample_count)
    corrector_variance_range[1] = diagnostics.minimum_corrector_variance
    corrector_variance_range[2] = diagnostics.maximum_corrector_variance
    maximum_correlation[1] =
        diagnostics.maximum_absolute_corrector_correlation
    minimum_pivot_ratio[1] = diagnostics.minimum_factorization_pivot_ratio
    return nothing
end

function _validate_docrime_interaction_node_extents(
    measurement_count::Int,
    corrector_coordinate_count::Int,
)
    measurements = _docrime_interaction_dimension(
        "measurement",
        measurement_count,
    )
    correctors = _docrime_interaction_dimension(
        "corrector-coordinate",
        corrector_coordinate_count,
    )
    max_float_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    interaction_count = UInt64(
        _docrime_interaction_matrix_count(
            "measurement-by-corrector",
            measurements,
            correctors,
        ),
    )
    covariance_count = UInt64(
        _docrime_interaction_matrix_count(
            "corrector-by-corrector",
            correctors,
            correctors,
        ),
    )
    interaction_count <= max_float_elements || throw(
        ArgumentError("DO-CRIME interaction-matrix Port byte extent must fit UInt32"),
    )
    covariance_count <= max_float_elements || throw(
        ArgumentError("DO-CRIME corrector-covariance Port byte extent must fit UInt32"),
    )
    return nothing
end

"""Float32 open-loop DO-CRIME interaction-matrix solve."""
@algorithm DocrimeOpenLoopInteractionMatrixF32 begin
    label = "docrime-open-loop-interaction-matrix-f32"
    failure = DocrimeInteractionMatrixFailure

    """Configuration for DocrimeOpenLoopInteractionMatrixF32."""
    @configuration DocrimeOpenLoopInteractionMatrixF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        minimum_corrector_variance::Float32 => construction
        cross_covariance_schema::String => graph_rebuild
        corrector_covariance_schema::String => graph_rebuild
        interaction_matrix_schema::String => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(
            :cross_covariance_sum,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.cross_covariance_schema;
            name="measurement-corrector-cross-covariance-sum",
        ),
        input(
            :corrector_covariance_sum,
            Float32,
            (
                configuration.corrector_coordinate_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.corrector_covariance_schema;
            name="corrector-covariance-sum",
        ),
        input(
            :sample_count,
            schema_element_type(DOCRIME_COVARIANCE_SAMPLE_COUNT_V1),
            (1,),
            DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
            name="sample-count",
        ),
        output(
            :interaction_matrix,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.interaction_matrix_schema;
            name="interaction-matrix",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :status,
            schema_element_type(DOCRIME_INTERACTION_MATRIX_STATUS_V1),
            (1,),
            DOCRIME_INTERACTION_MATRIX_STATUS_V1;
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :status_coordinates,
            schema_element_type(DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1),
            (2,),
            DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1;
            name="status-coordinates",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :diagnostic_sample_count,
            schema_element_type(DOCRIME_COVARIANCE_SAMPLE_COUNT_V1),
            (1,),
            DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
            name="diagnostic-sample-count",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :corrector_variance_range,
            schema_element_type(DOCRIME_CORRECTOR_VARIANCE_RANGE_V1),
            (2,),
            DOCRIME_CORRECTOR_VARIANCE_RANGE_V1;
            name="corrector-variance-range",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :maximum_correlation,
            schema_element_type(DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1),
            (1,),
            DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1;
            name="maximum-absolute-corrector-correlation",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :minimum_pivot_ratio,
            schema_element_type(DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1),
            (1,),
            DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1;
            name="minimum-factorization-pivot-ratio",
            metadata_from=:cross_covariance_sum,
        ),
    )

    prepare(configuration) = begin
        _validate_docrime_interaction_node_extents(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
        )
        criteria = DocrimeConditioningCriteria{Float32}(
            configuration.minimum_corrector_variance,
            0.9f0,
            0.01f0,
        )
        plan = DocrimeOpenLoopInteractionMatrixPlan(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
            criteria,
        )
        (plan=plan, workspace=DocrimeInteractionMatrixWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_docrime_interaction_matrix_outputs!(
            interaction_matrix,
            status,
            status_coordinates,
            diagnostic_sample_count,
            corrector_variance_range,
            maximum_correlation,
            minimum_pivot_ratio,
            workspace,
            plan,
            cross_covariance_sum,
            corrector_covariance_sum,
            sample_count,
        )
    end
end

"""Float32 closed-loop DO-CRIME interaction-matrix solve."""
@algorithm DocrimeClosedLoopInteractionMatrixF32 begin
    label = "docrime-closed-loop-interaction-matrix-f32"
    failure = DocrimeInteractionMatrixFailure

    """Configuration for DocrimeClosedLoopInteractionMatrixF32."""
    @configuration DocrimeClosedLoopInteractionMatrixF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        minimum_corrector_variance::Float32 => construction
        cross_covariance_schema::String => graph_rebuild
        corrector_covariance_schema::String => graph_rebuild
        interaction_matrix_schema::String => graph_rebuild
    end

    properties = :runtime

    ports(configuration) = (
        input(
            :cross_covariance_sum,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.cross_covariance_schema;
            name="measurement-corrector-cross-covariance-sum",
        ),
        input(
            :corrector_covariance_sum,
            Float32,
            (
                configuration.corrector_coordinate_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.corrector_covariance_schema;
            name="corrector-covariance-sum",
        ),
        input(
            :sample_count,
            schema_element_type(DOCRIME_COVARIANCE_SAMPLE_COUNT_V1),
            (1,),
            DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
            name="sample-count",
        ),
        output(
            :interaction_matrix,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.interaction_matrix_schema;
            name="interaction-matrix",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :status,
            schema_element_type(DOCRIME_INTERACTION_MATRIX_STATUS_V1),
            (1,),
            DOCRIME_INTERACTION_MATRIX_STATUS_V1;
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :status_coordinates,
            schema_element_type(DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1),
            (2,),
            DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1;
            name="status-coordinates",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :diagnostic_sample_count,
            schema_element_type(DOCRIME_COVARIANCE_SAMPLE_COUNT_V1),
            (1,),
            DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
            name="diagnostic-sample-count",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :corrector_variance_range,
            schema_element_type(DOCRIME_CORRECTOR_VARIANCE_RANGE_V1),
            (2,),
            DOCRIME_CORRECTOR_VARIANCE_RANGE_V1;
            name="corrector-variance-range",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :maximum_correlation,
            schema_element_type(DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1),
            (1,),
            DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1;
            name="maximum-absolute-corrector-correlation",
            metadata_from=:cross_covariance_sum,
        ),
        output(
            :minimum_pivot_ratio,
            schema_element_type(DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1),
            (1,),
            DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1;
            name="minimum-factorization-pivot-ratio",
            metadata_from=:cross_covariance_sum,
        ),
    )

    prepare(configuration) = begin
        _validate_docrime_interaction_node_extents(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
        )
        criteria = DocrimeConditioningCriteria{Float32}(
            configuration.minimum_corrector_variance,
            0.9f0,
            0.01f0,
        )
        plan = DocrimeClosedLoopInteractionMatrixPlan(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
            1.0f0,
            criteria,
        )
        (plan=plan, workspace=DocrimeInteractionMatrixWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_docrime_interaction_matrix_outputs!(
            interaction_matrix,
            status,
            status_coordinates,
            diagnostic_sample_count,
            corrector_variance_range,
            maximum_correlation,
            minimum_pivot_ratio,
            workspace,
            plan,
            cross_covariance_sum,
            corrector_covariance_sum,
            sample_count,
        )
    end
end
