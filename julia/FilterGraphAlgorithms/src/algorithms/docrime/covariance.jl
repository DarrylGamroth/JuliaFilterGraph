function _docrime_covariance_count(name::AbstractString, left::Int, right::Int)
    try
        return Base.Checked.checked_mul(left, right)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("DO-CRIME $name covariance extent overflows Int"))
    end
end

function _docrime_covariance_sample_limit(sample_limit)
    isnothing(sample_limit) && return nothing
    sample_limit isa Integer || throw(
        ArgumentError("DO-CRIME covariance sample_limit must be an integer or nothing"),
    )
    sample_limit > 0 || throw(
        ArgumentError("DO-CRIME covariance sample_limit must be positive"),
    )
    sample_limit <= typemax(UInt64) || throw(
        ArgumentError("DO-CRIME covariance sample_limit must fit UInt64"),
    )
    return UInt64(sample_limit)
end

function _docrime_covariance_dimension(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("DO-CRIME $name extent must be positive"))
    value <= typemax(Int) || throw(
        ArgumentError("DO-CRIME $name extent must fit Int"),
    )
    return Int(value)
end

"""
    DocrimeCovariancePlan(T, measurement_count, corrector_coordinate_count;
                          sample_limit=nothing)

Prepare centered online covariance accumulation for DO-CRIME. The measurement
and exposure-associated corrector-sample extents must be positive. A finite
`sample_limit` accepts exactly that many successful Algorithm samples.

The caller owns exposure association and any matched temporal filtering. For
open-loop identification, the corrector sample is the realized random
excitation. For closed-loop identification, it is the complete commanded
corrector increment.

Each CPU covariance update rounds `weight * delta_row`, then fuses its product
with `delta_column` and addition to the committed sum. Means retain the
separate delta division and addition. Symmetric corrector entries share one
calculated value. Fused cancellation can retain a finite covariance when an
independently rounded second product would overflow.
"""
struct DocrimeCovariancePlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    sample_limit::Union{Nothing,UInt64}

    function DocrimeCovariancePlan(
        ::Type{T},
        measurement_count::Integer,
        corrector_coordinate_count::Integer;
        sample_limit=nothing,
    ) where {T<:Union{Float32,Float64}}
        measurements = _docrime_covariance_dimension(
            "measurement",
            measurement_count,
        )
        correctors = _docrime_covariance_dimension(
            "corrector-sample",
            corrector_coordinate_count,
        )
        _docrime_covariance_count(
            "measurement-corrector cross",
            measurements,
            correctors,
        )
        _docrime_covariance_count("corrector", correctors, correctors)
        return new{T}(
            measurements,
            correctors,
            _docrime_covariance_sample_limit(sample_limit),
        )
    end
end


function _validate_docrime_covariance_node_extents(
    measurement_count::Int,
    corrector_coordinate_count::Int,
)
    max_float_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    measurements = UInt64(measurement_count)
    correctors = UInt64(corrector_coordinate_count)
    measurements <= max_float_elements || throw(
        ArgumentError("DO-CRIME measurement Port byte extent must fit UInt32"),
    )
    correctors <= max_float_elements || throw(
        ArgumentError("DO-CRIME corrector-sample Port byte extent must fit UInt32"),
    )
    cross_count = UInt64(
        _docrime_covariance_count(
            "measurement-corrector cross",
            measurement_count,
            corrector_coordinate_count,
        ),
    )
    corrector_count = UInt64(
        _docrime_covariance_count(
            "corrector",
            corrector_coordinate_count,
            corrector_coordinate_count,
        ),
    )
    cross_count <= max_float_elements || throw(
        ArgumentError("DO-CRIME cross-covariance Port byte extent must fit UInt32"),
    )
    corrector_count <= max_float_elements || throw(
        ArgumentError("DO-CRIME corrector covariance Port byte extent must fit UInt32"),
    )
    return nothing
end

"""
    DocrimeCovarianceWorkspace(plan)

Own committed and candidate means and centered covariance sums for one
DO-CRIME covariance Node. Processing never changes committed history.
[`commit!`](@ref) publishes one candidate sample after this Algorithm succeeds;
[`discard!`](@ref) preserves the preceding committed history.
"""
mutable struct DocrimeCovarianceWorkspace{T<:Union{Float32,Float64}}
    committed_sample_count::UInt64
    candidate_sample_count::UInt64
    committed_measurement_mean::Vector{T}
    candidate_measurement_mean::Vector{T}
    committed_corrector_sample_mean::Vector{T}
    candidate_corrector_sample_mean::Vector{T}
    committed_cross_covariance_sum::Matrix{T}
    candidate_cross_covariance_sum::Matrix{T}
    committed_corrector_covariance_sum::Matrix{T}
    candidate_corrector_covariance_sum::Matrix{T}
    measurement_delta::Vector{T}
    corrector_sample_delta::Vector{T}
end

function DocrimeCovarianceWorkspace(plan::DocrimeCovariancePlan{T}) where {T}
    measurement_mean = zeros(T, plan.measurement_count)
    corrector_mean = zeros(T, plan.corrector_coordinate_count)
    cross_covariance = zeros(
        T,
        plan.measurement_count,
        plan.corrector_coordinate_count,
    )
    corrector_covariance = zeros(
        T,
        plan.corrector_coordinate_count,
        plan.corrector_coordinate_count,
    )
    return DocrimeCovarianceWorkspace{T}(
        0,
        0,
        measurement_mean,
        similar(measurement_mean),
        corrector_mean,
        similar(corrector_mean),
        cross_covariance,
        similar(cross_covariance),
        corrector_covariance,
        similar(corrector_covariance),
        similar(measurement_mean),
        similar(corrector_mean),
    )
end

"""
    docrime_covariance_sample_count(workspace::DocrimeCovarianceWorkspace)

Return the number of samples in committed DO-CRIME covariance history.
"""
docrime_covariance_sample_count(workspace::DocrimeCovarianceWorkspace) =
    workspace.committed_sample_count

"""
    docrime_measurement_mean(workspace::DocrimeCovarianceWorkspace)

Return a copy of the committed measurement mean.
"""
docrime_measurement_mean(workspace::DocrimeCovarianceWorkspace) =
    copy(workspace.committed_measurement_mean)

"""
    docrime_corrector_sample_mean(workspace::DocrimeCovarianceWorkspace)

Return a copy of the committed corrector-sample mean.
"""
docrime_corrector_sample_mean(workspace::DocrimeCovarianceWorkspace) =
    copy(workspace.committed_corrector_sample_mean)

"""
    docrime_measurement_corrector_cross_covariance_sum(workspace::DocrimeCovarianceWorkspace)

Return a copy of the committed measurement--corrector covariance sum.
"""
docrime_measurement_corrector_cross_covariance_sum(
    workspace::DocrimeCovarianceWorkspace,
) = copy(workspace.committed_cross_covariance_sum)

"""
    docrime_corrector_covariance_sum(workspace::DocrimeCovarianceWorkspace)

Return a copy of the committed corrector covariance sum.
"""
docrime_corrector_covariance_sum(workspace::DocrimeCovarianceWorkspace) =
    copy(workspace.committed_corrector_covariance_sum)

function commit!(workspace::DocrimeCovarianceWorkspace)
    workspace.committed_measurement_mean,
    workspace.candidate_measurement_mean =
        workspace.candidate_measurement_mean,
        workspace.committed_measurement_mean
    workspace.committed_corrector_sample_mean,
    workspace.candidate_corrector_sample_mean =
        workspace.candidate_corrector_sample_mean,
        workspace.committed_corrector_sample_mean
    workspace.committed_cross_covariance_sum,
    workspace.candidate_cross_covariance_sum =
        workspace.candidate_cross_covariance_sum,
        workspace.committed_cross_covariance_sum
    workspace.committed_corrector_covariance_sum,
    workspace.candidate_corrector_covariance_sum =
        workspace.candidate_corrector_covariance_sum,
        workspace.committed_corrector_covariance_sum
    workspace.committed_sample_count = workspace.candidate_sample_count
    return workspace
end

function discard!(workspace::DocrimeCovarianceWorkspace)
    copyto!(
        workspace.candidate_measurement_mean,
        workspace.committed_measurement_mean,
    )
    copyto!(
        workspace.candidate_corrector_sample_mean,
        workspace.committed_corrector_sample_mean,
    )
    copyto!(
        workspace.candidate_cross_covariance_sum,
        workspace.committed_cross_covariance_sum,
    )
    copyto!(
        workspace.candidate_corrector_covariance_sum,
        workspace.committed_corrector_covariance_sum,
    )
    workspace.candidate_sample_count = workspace.committed_sample_count
    return workspace
end

function reset!(workspace::DocrimeCovarianceWorkspace{T}) where {T}
    workspace.committed_sample_count = 0
    workspace.candidate_sample_count = 0
    fill!(workspace.committed_measurement_mean, zero(T))
    fill!(workspace.candidate_measurement_mean, zero(T))
    fill!(workspace.committed_corrector_sample_mean, zero(T))
    fill!(workspace.candidate_corrector_sample_mean, zero(T))
    fill!(workspace.committed_cross_covariance_sum, zero(T))
    fill!(workspace.candidate_cross_covariance_sum, zero(T))
    fill!(workspace.committed_corrector_covariance_sum, zero(T))
    fill!(workspace.candidate_corrector_covariance_sum, zero(T))
    fill!(workspace.measurement_delta, zero(T))
    fill!(workspace.corrector_sample_delta, zero(T))
    return workspace
end

"""Expected data or numerical rejection from DO-CRIME covariance accumulation."""
@enum DocrimeCovarianceFailure::UInt8 begin
    DocrimeCovarianceNonFiniteMeasurements = 1
    DocrimeCovarianceNonFiniteCorrectorSample = 2
    DocrimeCovarianceNonFiniteCalculation = 3
    DocrimeCovarianceSampleCountOverflow = 4
    DocrimeCovarianceSampleCountNotRepresentable = 5
    DocrimeCovarianceSampleLimitReached = 6
end

@doc "The measurement vector contains a non-finite value."
DocrimeCovarianceNonFiniteMeasurements
@doc "The exposure-associated corrector sample contains a non-finite value."
DocrimeCovarianceNonFiniteCorrectorSample
@doc "A candidate mean or covariance sum is non-finite."
DocrimeCovarianceNonFiniteCalculation
@doc "The committed UInt64 sample count cannot be incremented."
DocrimeCovarianceSampleCountOverflow
@doc "The next sample count is not represented exactly by the Plan scalar type."
DocrimeCovarianceSampleCountNotRepresentable
@doc "The selected finite covariance window already contains its maximum samples."
DocrimeCovarianceSampleLimitReached

function _docrime_covariance_workspace_extent_is_valid(
    workspace::DocrimeCovarianceWorkspace,
    plan::DocrimeCovariancePlan,
)
    measurement_axes = (Base.OneTo(plan.measurement_count),)
    corrector_axes = (Base.OneTo(plan.corrector_coordinate_count),)
    cross_axes = (measurement_axes[1], corrector_axes[1])
    corrector_covariance_axes = (corrector_axes[1], corrector_axes[1])
    return axes(workspace.committed_measurement_mean) == measurement_axes &&
           axes(workspace.candidate_measurement_mean) == measurement_axes &&
           axes(workspace.measurement_delta) == measurement_axes &&
           axes(workspace.committed_corrector_sample_mean) == corrector_axes &&
           axes(workspace.candidate_corrector_sample_mean) == corrector_axes &&
           axes(workspace.corrector_sample_delta) == corrector_axes &&
           axes(workspace.committed_cross_covariance_sum) == cross_axes &&
           axes(workspace.candidate_cross_covariance_sum) == cross_axes &&
           axes(workspace.committed_corrector_covariance_sum) ==
           corrector_covariance_axes &&
           axes(workspace.candidate_corrector_covariance_sum) ==
           corrector_covariance_axes
end

@inline function _docrime_covariance_aliases_workspace(
    array::AbstractArray,
    workspace::DocrimeCovarianceWorkspace,
)
    return Base.mightalias(array, workspace.committed_measurement_mean) ||
           Base.mightalias(array, workspace.candidate_measurement_mean) ||
           Base.mightalias(array, workspace.committed_corrector_sample_mean) ||
           Base.mightalias(array, workspace.candidate_corrector_sample_mean) ||
           Base.mightalias(array, workspace.committed_cross_covariance_sum) ||
           Base.mightalias(array, workspace.candidate_cross_covariance_sum) ||
           Base.mightalias(array, workspace.committed_corrector_covariance_sum) ||
           Base.mightalias(array, workspace.candidate_corrector_covariance_sum) ||
           Base.mightalias(array, workspace.measurement_delta) ||
           Base.mightalias(array, workspace.corrector_sample_delta)
end

function _validate_docrime_covariance_storage(
    cross_covariance_sum::AbstractMatrix,
    corrector_covariance_sum::AbstractMatrix,
    sample_count::AbstractVector,
    complete::AbstractVector,
    workspace::DocrimeCovarianceWorkspace,
    plan::DocrimeCovariancePlan,
    measurements::AbstractVector,
    corrector_sample::AbstractVector,
)
    Base.require_one_based_indexing(
        cross_covariance_sum,
        corrector_covariance_sum,
        sample_count,
        complete,
        measurements,
        corrector_sample,
    )
    axes(measurements) == (Base.OneTo(plan.measurement_count),) || throw(
        DimensionMismatch("DO-CRIME measurement extent is invalid"),
    )
    axes(corrector_sample) == (Base.OneTo(plan.corrector_coordinate_count),) || throw(
        DimensionMismatch("DO-CRIME corrector-sample extent is invalid"),
    )
    axes(cross_covariance_sum) == (
        Base.OneTo(plan.measurement_count),
        Base.OneTo(plan.corrector_coordinate_count),
    ) || throw(DimensionMismatch("DO-CRIME cross-covariance extent is invalid"))
    axes(corrector_covariance_sum) == (
        Base.OneTo(plan.corrector_coordinate_count),
        Base.OneTo(plan.corrector_coordinate_count),
    ) || throw(DimensionMismatch("DO-CRIME corrector covariance extent is invalid"))
    length(sample_count) == 1 || throw(
        DimensionMismatch("DO-CRIME sample-count extent is invalid"),
    )
    length(complete) == 1 || throw(
        DimensionMismatch("DO-CRIME completion extent is invalid"),
    )
    _docrime_covariance_workspace_extent_is_valid(workspace, plan) || throw(
        DimensionMismatch("DO-CRIME covariance Workspace extent is invalid"),
    )
    (
        Base.mightalias(cross_covariance_sum, corrector_covariance_sum) ||
        Base.mightalias(cross_covariance_sum, sample_count) ||
        Base.mightalias(cross_covariance_sum, complete) ||
        Base.mightalias(corrector_covariance_sum, sample_count) ||
        Base.mightalias(corrector_covariance_sum, complete) ||
        Base.mightalias(sample_count, complete)
    ) && throw(ArgumentError("DO-CRIME covariance outputs must not overlap"))
    (
        _docrime_covariance_aliases_workspace(cross_covariance_sum, workspace) ||
        _docrime_covariance_aliases_workspace(corrector_covariance_sum, workspace) ||
        _docrime_covariance_aliases_workspace(sample_count, workspace) ||
        _docrime_covariance_aliases_workspace(complete, workspace) ||
        _docrime_covariance_aliases_workspace(measurements, workspace) ||
        _docrime_covariance_aliases_workspace(corrector_sample, workspace)
    ) && throw(
        ArgumentError("DO-CRIME covariance arrays must not alias Workspace storage"),
    )
    return nothing
end

@inline function _docrime_covariance_all_finite(values::AbstractArray)
    for value in values
        isfinite(value) || return false
    end
    return true
end

# Only the Boolean finiteness check is reduced across lanes. Each covariance
# element retains its FMA update order. Ordinary @simd allows the compiler
# to check for memory dependencies before vectorizing.
function _stage_docrime_cross_covariance!(
    workspace::DocrimeCovarianceWorkspace{Float32}, weight::Float32,
)
    finite_cross = true
    for corrector in axes(workspace.candidate_cross_covariance_sum, 2)
        corrector_delta = workspace.corrector_sample_delta[corrector]
        finite_column = true
        for measurement in axes(workspace.candidate_cross_covariance_sum, 1)
            candidate = fma(
                weight * workspace.measurement_delta[measurement],
                corrector_delta,
                workspace.committed_cross_covariance_sum[measurement, corrector],
            )
            finite_column = finite_column & isfinite(candidate)
            workspace.candidate_cross_covariance_sum[
                measurement,
                corrector,
            ] = candidate
        end
        finite_cross = finite_cross & finite_column
    end
    return finite_cross
end

function _stage_docrime_cross_covariance!(
    workspace::DocrimeCovarianceWorkspace{Float64}, weight::Float64,
)
    finite_cross = true
    for corrector in axes(workspace.candidate_cross_covariance_sum, 2)
        corrector_delta = workspace.corrector_sample_delta[corrector]
        @simd for measurement in axes(workspace.candidate_cross_covariance_sum, 1)
            candidate = fma(
                weight * workspace.measurement_delta[measurement],
                corrector_delta,
                workspace.committed_cross_covariance_sum[measurement, corrector],
            )
            finite_cross = finite_cross & isfinite(candidate)
            workspace.candidate_cross_covariance_sum[
                measurement,
                corrector,
            ] = candidate
        end
    end
    return finite_cross
end

function process!(
    cross_covariance_sum::AbstractMatrix{T},
    corrector_covariance_sum::AbstractMatrix{T},
    sample_count::AbstractVector{UInt64},
    complete::AbstractVector{Bool},
    workspace::DocrimeCovarianceWorkspace{T},
    plan::DocrimeCovariancePlan{T},
    measurements::AbstractVector{T},
    corrector_sample::AbstractVector{T},
) where {T}
    _validate_docrime_covariance_storage(
        cross_covariance_sum,
        corrector_covariance_sum,
        sample_count,
        complete,
        workspace,
        plan,
        measurements,
        corrector_sample,
    )
    committed_count = workspace.committed_sample_count
    limit = plan.sample_limit
    !isnothing(limit) && committed_count >= limit &&
        return DocrimeCovarianceSampleLimitReached
    _docrime_covariance_all_finite(measurements) ||
        return DocrimeCovarianceNonFiniteMeasurements
    _docrime_covariance_all_finite(corrector_sample) ||
        return DocrimeCovarianceNonFiniteCorrectorSample
    committed_count == typemax(UInt64) &&
        return DocrimeCovarianceSampleCountOverflow

    next_count = committed_count + one(UInt64)
    count = T(next_count)
    isfinite(count) && count == next_count ||
        return DocrimeCovarianceSampleCountNotRepresentable
    weight = (count - one(T)) / count

    # Complete each disposable candidate stage, then reject before its dependents.
    finite_measurement = true
    for index in eachindex(
        measurements,
        workspace.committed_measurement_mean,
        workspace.candidate_measurement_mean,
        workspace.measurement_delta,
    )
        delta = measurements[index] - workspace.committed_measurement_mean[index]
        candidate_mean =
            workspace.committed_measurement_mean[index] + delta / count
        finite_measurement = finite_measurement & (isfinite(delta) & isfinite(candidate_mean))
        workspace.measurement_delta[index] = delta
        workspace.candidate_measurement_mean[index] = candidate_mean
    end
    finite_measurement || return DocrimeCovarianceNonFiniteCalculation
    finite_corrector = true
    for index in eachindex(
        corrector_sample,
        workspace.committed_corrector_sample_mean,
        workspace.candidate_corrector_sample_mean,
        workspace.corrector_sample_delta,
    )
        delta = corrector_sample[index] -
                workspace.committed_corrector_sample_mean[index]
        candidate_mean =
            workspace.committed_corrector_sample_mean[index] + delta / count
        finite_corrector = finite_corrector & (isfinite(delta) & isfinite(candidate_mean))
        workspace.corrector_sample_delta[index] = delta
        workspace.candidate_corrector_sample_mean[index] = candidate_mean
    end
    finite_corrector || return DocrimeCovarianceNonFiniteCalculation

    _stage_docrime_cross_covariance!(workspace, weight) ||
        return DocrimeCovarianceNonFiniteCalculation
    covariance = workspace.candidate_corrector_covariance_sum
    committed = workspace.committed_corrector_covariance_sum
    delta = workspace.corrector_sample_delta
    if Base.mightalias(covariance, workspace.committed_measurement_mean) ||
       Base.mightalias(covariance, workspace.candidate_measurement_mean) ||
       Base.mightalias(covariance, workspace.committed_corrector_sample_mean) ||
       Base.mightalias(covariance, workspace.candidate_corrector_sample_mean) ||
       Base.mightalias(covariance, workspace.committed_cross_covariance_sum) ||
       Base.mightalias(covariance, workspace.candidate_cross_covariance_sum) ||
       Base.mightalias(covariance, committed) ||
       Base.mightalias(covariance, workspace.measurement_delta) ||
       Base.mightalias(covariance, delta)
        # Preserve interleaved reads/writes for manually shared Workspace storage.
        for column in axes(workspace.candidate_corrector_covariance_sum, 2)
            for row in firstindex(
                workspace.candidate_corrector_covariance_sum,
                1,
            ):column
                candidate = fma(
                    weight * workspace.corrector_sample_delta[row],
                    workspace.corrector_sample_delta[column],
                    workspace.committed_corrector_covariance_sum[row, column],
                )
                isfinite(candidate) || return DocrimeCovarianceNonFiniteCalculation
                workspace.candidate_corrector_covariance_sum[row, column] = candidate
                workspace.candidate_corrector_covariance_sum[column, row] = candidate
            end
        end
    else
        # Validated one-based m×m storage proves 1 ≤ row ≤ column ≤ m.
        # Calculate contiguous upper columns, then mirror in small tiles with
        # contiguous destination writes while the source columns are still hot.
        # Strict-lower destinations cannot overwrite an upper source or diagonal.
        m = size(covariance, 1)
        @inbounds for column_start in 1:16:m
            column_end = min(column_start + 15, m)
            for column in column_start:column_end
                finite_column = true
                for row in firstindex(covariance, 1):column
                    candidate = fma(weight * delta[row], delta[column], committed[row, column])
                    finite_column = finite_column & isfinite(candidate)
                    covariance[row, column] = candidate
                end
                finite_column || return DocrimeCovarianceNonFiniteCalculation
            end
            for row_start in 1:16:column_start
                row_end = min(row_start + 15, m)
                for row in row_start:row_end
                    for column in max(column_start, row + 1):column_end
                        covariance[column, row] = covariance[row, column]
                    end
                end
            end
        end
    end

    workspace.candidate_sample_count = next_count
    copyto!(cross_covariance_sum, workspace.candidate_cross_covariance_sum)
    copyto!(
        corrector_covariance_sum,
        workspace.candidate_corrector_covariance_sum,
    )
    sample_count[1] = next_count
    complete[1] = !isnothing(limit) && next_count == limit
    return nothing
end

"""Float32 centered DO-CRIME covariance accumulation."""
@algorithm DocrimeCovarianceF32 begin
    label = "docrime-covariance-f32"
    failure = DocrimeCovarianceFailure

    """Configuration for [`DocrimeCovarianceF32`](@ref)."""
    @configuration DocrimeCovarianceF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        sample_limit::Union{Nothing,UInt64} => construction
        measurements_schema::String => graph_rebuild
        corrector_sample_schema::String => graph_rebuild
        cross_covariance_schema::String => graph_rebuild
        corrector_covariance_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :measurements,
            Float32,
            (configuration.measurement_count,),
            configuration.measurements_schema,
        ),
        input(
            :corrector_sample,
            Float32,
            (configuration.corrector_coordinate_count,),
            configuration.corrector_sample_schema;
            name="corrector-sample",
        ),
        output(
            :cross_covariance_sum,
            Float32,
            (
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.cross_covariance_schema;
            name="measurement-corrector-cross-covariance-sum",
            metadata_from=:measurements,
        ),
        output(
            :corrector_covariance_sum,
            Float32,
            (
                configuration.corrector_coordinate_count,
                configuration.corrector_coordinate_count,
            ),
            configuration.corrector_covariance_schema;
            name="corrector-covariance-sum",
            metadata_from=:measurements,
        ),
        output(
            :sample_count,
            schema_element_type(DOCRIME_COVARIANCE_SAMPLE_COUNT_V1),
            (1,),
            DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
            name="sample-count",
            metadata_from=:measurements,
        ),
        output(
            :complete,
            schema_element_type(DOCRIME_COVARIANCE_COMPLETE_V1),
            (1,),
            DOCRIME_COVARIANCE_COMPLETE_V1;
            metadata_from=:measurements,
        ),
    )

    prepare(configuration) = begin
        _validate_docrime_covariance_node_extents(
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
        )
        plan = DocrimeCovariancePlan(
            Float32,
            configuration.measurement_count,
            configuration.corrector_coordinate_count;
            sample_limit=configuration.sample_limit,
        )
        (plan=plan, workspace=DocrimeCovarianceWorkspace(plan))
    end

    process!(plan, workspace) = begin
        process!(
            cross_covariance_sum,
            corrector_covariance_sum,
            sample_count,
            complete,
            workspace,
            plan,
            measurements,
            corrector_sample,
        )
    end
end
