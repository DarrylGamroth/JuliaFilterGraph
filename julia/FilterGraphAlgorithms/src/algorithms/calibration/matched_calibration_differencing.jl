"""
    MatchedCalibrationDifferencingPlan(T, measurement_count, corrector_coordinate_count)

Prepare the matched `[1, -1]` first-difference filter for exposure-associated
measurements and corrector samples. Both extents must be positive. The caller
must associate each input pair with the same sensor exposure before processing;
the Algorithm neither infers command delay nor selects among samples.
"""
struct MatchedCalibrationDifferencingPlan{
    T<:Union{Float32,Float64},
} <: AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int

    function MatchedCalibrationDifferencingPlan(
        ::Type{T},
        measurement_count::Integer,
        corrector_coordinate_count::Integer,
    ) where {T<:Union{Float32,Float64}}
        measurements = Int(measurement_count)
        correctors = Int(corrector_coordinate_count)
        measurements > 0 && correctors > 0 || throw(
            ArgumentError(
                "matched calibration measurement and corrector-sample extents must be positive",
            ),
        )
        return new{T}(measurements, correctors)
    end
end

"""
    MatchedCalibrationDifferencingWorkspace(plan)

Own the committed previous input pair, its candidate replacement, and fixed
first-difference staging for one Algorithm instance. Processing never changes
committed history. [`commit!`](@ref) advances history after this Algorithm
succeeds, while [`discard!`](@ref) abandons its rejected or aborted candidate.
"""
mutable struct MatchedCalibrationDifferencingWorkspace{
    T<:Union{Float32,Float64},
}
    committed_primed::Bool
    candidate_primed::Bool
    committed_measurements::Vector{T}
    candidate_measurements::Vector{T}
    committed_corrector_sample::Vector{T}
    candidate_corrector_sample::Vector{T}
    staged_measurements::Vector{T}
    staged_corrector_sample::Vector{T}
end

function MatchedCalibrationDifferencingWorkspace(
    plan::MatchedCalibrationDifferencingPlan{T},
) where {T}
    committed_measurements = zeros(T, plan.measurement_count)
    committed_corrector = zeros(T, plan.corrector_coordinate_count)
    return MatchedCalibrationDifferencingWorkspace{T}(
        false,
        false,
        committed_measurements,
        similar(committed_measurements),
        committed_corrector,
        similar(committed_corrector),
        similar(committed_measurements),
        similar(committed_corrector),
    )
end

"""
    matched_calibration_is_primed(workspace::MatchedCalibrationDifferencingWorkspace)

Return whether matched differencing has a committed previous input pair.
"""
matched_calibration_is_primed(
    workspace::MatchedCalibrationDifferencingWorkspace,
) = workspace.committed_primed

"""
    matched_calibration_previous_measurements(workspace::MatchedCalibrationDifferencingWorkspace)

Return a copy of the committed previous measurement vector.
"""
matched_calibration_previous_measurements(
    workspace::MatchedCalibrationDifferencingWorkspace,
) = copy(workspace.committed_measurements)

"""
    matched_calibration_previous_corrector_sample(workspace::MatchedCalibrationDifferencingWorkspace)

Return a copy of the committed previous corrector-sample vector.
"""
matched_calibration_previous_corrector_sample(
    workspace::MatchedCalibrationDifferencingWorkspace,
) = copy(workspace.committed_corrector_sample)

function commit!(workspace::MatchedCalibrationDifferencingWorkspace)
    workspace.committed_measurements, workspace.candidate_measurements =
        workspace.candidate_measurements, workspace.committed_measurements
    workspace.committed_corrector_sample,
    workspace.candidate_corrector_sample =
        workspace.candidate_corrector_sample,
        workspace.committed_corrector_sample
    workspace.committed_primed = workspace.candidate_primed
    return workspace
end

function discard!(workspace::MatchedCalibrationDifferencingWorkspace)
    copyto!(
        workspace.candidate_measurements,
        workspace.committed_measurements,
    )
    copyto!(
        workspace.candidate_corrector_sample,
        workspace.committed_corrector_sample,
    )
    workspace.candidate_primed = workspace.committed_primed
    return workspace
end

function reset!(workspace::MatchedCalibrationDifferencingWorkspace{T}) where {T}
    workspace.committed_primed = false
    workspace.candidate_primed = false
    fill!(workspace.committed_measurements, zero(T))
    fill!(workspace.candidate_measurements, zero(T))
    fill!(workspace.committed_corrector_sample, zero(T))
    fill!(workspace.candidate_corrector_sample, zero(T))
    fill!(workspace.staged_measurements, zero(T))
    fill!(workspace.staged_corrector_sample, zero(T))
    return workspace
end

"""Expected numerical rejection from matched calibration differencing."""
@enum MatchedCalibrationDifferencingFailure::UInt8 begin
    MatchedCalibrationNonFiniteMeasurements = 1
    MatchedCalibrationNonFiniteCorrectorSample = 2
    MatchedCalibrationNonFiniteDifference = 3
end

@doc "The exposure-associated measurement vector contains a non-finite value."
MatchedCalibrationNonFiniteMeasurements
@doc "The exposure-associated corrector sample contains a non-finite value."
MatchedCalibrationNonFiniteCorrectorSample
@doc "A matched first difference produced a non-finite value."
MatchedCalibrationNonFiniteDifference

function _matched_workspace_extent_is_valid(
    workspace::MatchedCalibrationDifferencingWorkspace,
    plan::MatchedCalibrationDifferencingPlan,
)
    return length(workspace.committed_measurements) == plan.measurement_count &&
           length(workspace.candidate_measurements) == plan.measurement_count &&
           length(workspace.staged_measurements) == plan.measurement_count &&
           length(workspace.committed_corrector_sample) ==
           plan.corrector_coordinate_count &&
           length(workspace.candidate_corrector_sample) ==
           plan.corrector_coordinate_count &&
           length(workspace.staged_corrector_sample) ==
           plan.corrector_coordinate_count
end

function _matched_aliases_workspace(
    values::AbstractArray,
    workspace::MatchedCalibrationDifferencingWorkspace,
)
    return Base.mightalias(values, workspace.committed_measurements) ||
           Base.mightalias(values, workspace.candidate_measurements) ||
           Base.mightalias(values, workspace.committed_corrector_sample) ||
           Base.mightalias(values, workspace.candidate_corrector_sample) ||
           Base.mightalias(values, workspace.staged_measurements) ||
           Base.mightalias(values, workspace.staged_corrector_sample)
end

function _validate_matched_calibration_storage(
    filtered_measurements::AbstractVector,
    filtered_corrector_sample::AbstractVector,
    filtered::AbstractVector,
    workspace::MatchedCalibrationDifferencingWorkspace,
    plan::MatchedCalibrationDifferencingPlan,
    measurements::AbstractVector,
    corrector_sample::AbstractVector,
)
    Base.require_one_based_indexing(
        filtered_measurements,
        filtered_corrector_sample,
        filtered,
        measurements,
        corrector_sample,
    )
    length(measurements) == plan.measurement_count || throw(
        DimensionMismatch("matched calibration measurement extent is invalid"),
    )
    length(corrector_sample) == plan.corrector_coordinate_count || throw(
        DimensionMismatch("matched calibration corrector-sample extent is invalid"),
    )
    length(filtered_measurements) == plan.measurement_count || throw(
        DimensionMismatch("filtered measurement extent is invalid"),
    )
    length(filtered_corrector_sample) == plan.corrector_coordinate_count || throw(
        DimensionMismatch("filtered corrector-sample extent is invalid"),
    )
    length(filtered) == 1 ||
        throw(DimensionMismatch("matched calibration status extent is invalid"))
    _matched_workspace_extent_is_valid(workspace, plan) || throw(
        DimensionMismatch("matched calibration Workspace extent is invalid"),
    )
    Base.mightalias(filtered_measurements, filtered_corrector_sample) && throw(
        ArgumentError("matched calibration outputs must not overlap"),
    )
    (
        _matched_aliases_workspace(filtered_measurements, workspace) ||
        _matched_aliases_workspace(filtered_corrector_sample, workspace) ||
        _matched_aliases_workspace(filtered, workspace) ||
        _matched_aliases_workspace(measurements, workspace) ||
        _matched_aliases_workspace(corrector_sample, workspace)
    ) && throw(
        ArgumentError("matched calibration arrays must not alias Workspace storage"),
    )
    return nothing
end

function process!(
    filtered_measurements::AbstractVector{T},
    filtered_corrector_sample::AbstractVector{T},
    filtered::AbstractVector{Bool},
    workspace::MatchedCalibrationDifferencingWorkspace{T},
    plan::MatchedCalibrationDifferencingPlan{T},
    measurements::AbstractVector{T},
    corrector_sample::AbstractVector{T},
) where {T}
    _validate_matched_calibration_storage(
        filtered_measurements,
        filtered_corrector_sample,
        filtered,
        workspace,
        plan,
        measurements,
        corrector_sample,
    )
    for value in measurements
        isfinite(value) || return MatchedCalibrationNonFiniteMeasurements
    end
    for value in corrector_sample
        isfinite(value) || return MatchedCalibrationNonFiniteCorrectorSample
    end

    primed = workspace.committed_primed
    if primed
        for index in eachindex(
            workspace.staged_measurements,
            measurements,
            workspace.committed_measurements,
        )
            difference =
                measurements[index] - workspace.committed_measurements[index]
            workspace.staged_measurements[index] = difference
        end
        for difference in workspace.staged_measurements
            isfinite(difference) || return MatchedCalibrationNonFiniteDifference
        end
        for index in eachindex(
            workspace.staged_corrector_sample,
            corrector_sample,
            workspace.committed_corrector_sample,
        )
            difference = corrector_sample[index] -
                         workspace.committed_corrector_sample[index]
            workspace.staged_corrector_sample[index] = difference
        end
        for difference in workspace.staged_corrector_sample
            isfinite(difference) || return MatchedCalibrationNonFiniteDifference
        end
    else
        fill!(workspace.staged_measurements, zero(T))
        fill!(workspace.staged_corrector_sample, zero(T))
    end

    copyto!(workspace.candidate_measurements, measurements)
    copyto!(workspace.candidate_corrector_sample, corrector_sample)
    copyto!(filtered_measurements, workspace.staged_measurements)
    copyto!(filtered_corrector_sample, workspace.staged_corrector_sample)
    filtered[1] = primed
    workspace.candidate_primed = true
    return nothing
end

"""
Float32 matched calibration first differencing. The first successful pair
publishes zero vectors with `filtered=false`; later successful pairs publish
coherent differences with `filtered=true`. Pair history advances only when the
Algorithm transition commits.
"""
@algorithm MatchedCalibrationDifferencingF32 begin
    label = "matched-calibration-differencing-f32"
    failure = MatchedCalibrationDifferencingFailure

    """Configuration for [`MatchedCalibrationDifferencingF32`](@ref)."""
    @configuration MatchedCalibrationDifferencingF32Configuration begin
        measurement_count::Int => graph_rebuild
        corrector_coordinate_count::Int => graph_rebuild
        measurements_schema::String => graph_rebuild
        corrector_sample_schema::String => graph_rebuild
        filtered_measurements_schema::String => graph_rebuild
        filtered_corrector_sample_schema::String => graph_rebuild
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
            :filtered_measurements,
            Float32,
            (configuration.measurement_count,),
            configuration.filtered_measurements_schema;
            name="filtered-measurements",
            metadata_from=:measurements,
        ),
        output(
            :filtered_corrector_sample,
            Float32,
            (configuration.corrector_coordinate_count,),
            configuration.filtered_corrector_sample_schema;
            name="filtered-corrector-sample",
            metadata_from=:measurements,
        ),
        output(
            :filtered,
            schema_element_type(MATCHED_CALIBRATION_FILTERED_V1),
            (1,),
            MATCHED_CALIBRATION_FILTERED_V1;
            metadata_from=:measurements,
        ),
    )

    prepare(configuration) = begin
        plan = MatchedCalibrationDifferencingPlan(
            Float32,
            configuration.measurement_count,
            configuration.corrector_coordinate_count,
        )
        (
            plan=plan,
            workspace=MatchedCalibrationDifferencingWorkspace(plan),
        )
    end

    process!(plan, workspace) = begin
        process!(
            filtered_measurements,
            filtered_corrector_sample,
            filtered,
            workspace,
            plan,
            measurements,
            corrector_sample,
        )
    end
end
