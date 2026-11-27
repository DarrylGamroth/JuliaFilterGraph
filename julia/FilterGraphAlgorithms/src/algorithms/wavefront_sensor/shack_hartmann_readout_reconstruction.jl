"""Build the immutable subaperture completion schedule for one readout map."""
function _shack_hartmann_readout_completion_groups(
    sensing::ShackHartmannImagePlan,
    readout::DetectorReadoutPlan,
)
    subaperture_count = length(sensing.regions.origins)
    completion_ordinals = Vector{Int}(undef, subaperture_count)
    completion_groups = [Int[] for _ in 1:readout.block_count]
    subaperture_rows, subaperture_columns = sensing.regions.region_size
    for subaperture in eachindex(sensing.regions.origins)
        first_row, first_column = sensing.regions.origins[subaperture]
        completion_ordinal = 0
        for row in 1:subaperture_rows, column in 1:subaperture_columns
            completion_ordinal = max(
                completion_ordinal,
                readout.arrival_ordinals[first_row + row, first_column + column],
            )
        end
        completion_ordinals[subaperture] = completion_ordinal
        push!(completion_groups[completion_ordinal + 1], subaperture)
    end
    return completion_ordinals, completion_groups
end

"""
Prepared detector-readout Shack–Hartmann sensing and incremental reconstruction.

The readout mapping determines when each subaperture becomes complete. The
subaperture declaration order continues to define pair-interleaved X/Y slope
identity and the corresponding reconstructor columns.
"""
struct ShackHartmannReadoutReconstructionPlan{
    Reconstructor<:DenseReconstructorPlan{Float32},
} <: AbstractAlgorithmPlan
    sensing::ShackHartmannImagePlan
    reconstructor::Reconstructor
    readout::DetectorReadoutPlan
    subaperture_completion_ordinals::Vector{Int}
    completion_groups::Vector{Vector{Int}}

    function ShackHartmannReadoutReconstructionPlan(
        sensing::ShackHartmannImagePlan,
        reconstructor::DenseReconstructorPlan{Float32},
        readout::DetectorReadoutPlan,
    )
        sensing.regions.image_size == readout.detector_size || throw(
            DimensionMismatch(
                "detector readout and Shack-Hartmann image extents must match",
            ),
        )
        subaperture_count = length(sensing.regions.origins)
        slope_count = Base.Checked.checked_mul(subaperture_count, 2)
        size(reconstructor.matrix, 2) == slope_count || throw(
            DimensionMismatch(
                "reconstructor columns must match pair-interleaved slopes",
            ),
        )

        completion_ordinals, completion_groups =
            _shack_hartmann_readout_completion_groups(sensing, readout)
        return new{typeof(reconstructor)}(
            sensing,
            reconstructor,
            readout,
            completion_ordinals,
            completion_groups,
        )
    end
end

"""Progressive state and unpublished output for one detector-readout frame."""
mutable struct ShackHartmannReadoutReconstructionWorkspace
    detector::Matrix{Float32}
    slopes::Matrix{Float32}
    flux::Vector{Float32}
    validity::Vector{Bool}
    candidate_reconstructed::Vector{Float32}
    frame::DetectorReadoutFrame
    measured_subapertures::Int
    terminal_candidate::Bool
end

function ShackHartmannReadoutReconstructionWorkspace(
    plan::ShackHartmannReadoutReconstructionPlan,
)
    subaperture_count = length(plan.sensing.regions.origins)
    return ShackHartmannReadoutReconstructionWorkspace(
        zeros(Float32, plan.readout.detector_size),
        zeros(Float32, subaperture_count, 2),
        zeros(Float32, subaperture_count),
        fill(false, subaperture_count),
        zeros(Float32, size(plan.reconstructor.matrix, 1)),
        DetectorReadoutFrame(),
        0,
        false,
    )
end

function _clear_shack_hartmann_readout_partial!(
    workspace::ShackHartmannReadoutReconstructionWorkspace,
)
    fill!(workspace.candidate_reconstructed, 0.0f0)
    workspace.measured_subapertures = 0
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_shack_hartmann_readout!(
    workspace::ShackHartmannReadoutReconstructionWorkspace,
)
    _abandon_detector_readout!(workspace.frame)
    return _clear_shack_hartmann_readout_partial!(workspace)
end

function commit!(workspace::ShackHartmannReadoutReconstructionWorkspace)
    if workspace.terminal_candidate
        _complete_detector_readout!(workspace.frame)
        _clear_shack_hartmann_readout_partial!(workspace)
    end
    return workspace
end

discard!(workspace::ShackHartmannReadoutReconstructionWorkspace) =
    _abandon_shack_hartmann_readout!(workspace)

function reset!(workspace::ShackHartmannReadoutReconstructionWorkspace)
    fill!(workspace.detector, 0.0f0)
    fill!(workspace.slopes, 0.0f0)
    fill!(workspace.flux, 0.0f0)
    fill!(workspace.validity, false)
    return _abandon_shack_hartmann_readout!(workspace)
end

"""Expected rejection from detector-readout Shack–Hartmann reconstruction."""
@enum ShackHartmannReadoutReconstructionFailure::UInt8 begin
    ShackHartmannReadoutMetadataMissing = 1
    ShackHartmannReadoutOffsetOverflow = 2
    ShackHartmannReadoutNonFiniteReconstruction = 3
end

@doc "A Shack-Hartmann readout reconstruction block lacks required sample metadata." ShackHartmannReadoutMetadataMissing
@doc "A Shack-Hartmann readout reconstruction offset cannot be represented safely." ShackHartmannReadoutOffsetOverflow
@doc "Readout-based Shack-Hartmann reconstruction produced a non-finite coordinate." ShackHartmannReadoutNonFiniteReconstruction

@inline function _shack_hartmann_readout_aliases_workspace(
    array,
    workspace::ShackHartmannReadoutReconstructionWorkspace,
)
    return Base.mightalias(array, workspace.detector) ||
        Base.mightalias(array, workspace.slopes) ||
        Base.mightalias(array, workspace.flux) ||
        Base.mightalias(array, workspace.validity) ||
        Base.mightalias(array, workspace.candidate_reconstructed)
end

function _validate_shack_hartmann_readout_storage(
    reconstructed::AbstractVector{Float32},
    workspace::ShackHartmannReadoutReconstructionWorkspace,
    plan::ShackHartmannReadoutReconstructionPlan,
    readout_block::AbstractArray{Float32,3},
)
    Base.require_one_based_indexing(reconstructed, readout_block)
    size(readout_block) == detector_readout_block_shape(plan.readout) || throw(
        DimensionMismatch("readout block has an incompatible detector shape"),
    )
    length(reconstructed) == size(plan.reconstructor.matrix, 1) || throw(
        DimensionMismatch("reconstructed output has an incompatible extent"),
    )
    size(workspace.detector) == plan.readout.detector_size || throw(
        DimensionMismatch("readout reconstruction Workspace has an invalid detector"),
    )
    Base.mightalias(reconstructed, readout_block) && throw(
        ArgumentError("reconstructed output must not alias the detector readout block"),
    )
    Base.mightalias(reconstructed, plan.reconstructor.matrix) && throw(
        ArgumentError("reconstructed output must not alias the prepared matrix"),
    )
    _shack_hartmann_aliases_plan(reconstructed, plan.sensing) && throw(
        ArgumentError("reconstructed output must not alias sensing calibration"),
    )
    _shack_hartmann_readout_aliases_workspace(reconstructed, workspace) && throw(
        ArgumentError("reconstructed output must not alias readout Workspace storage"),
    )
    _shack_hartmann_readout_aliases_workspace(readout_block, workspace) && throw(
        ArgumentError("readout input must not alias readout Workspace storage"),
    )
    return nothing
end

function _accumulate_shack_hartmann_readout_group!(
    workspace::ShackHartmannReadoutReconstructionWorkspace,
    plan::ShackHartmannReadoutReconstructionPlan,
    subapertures::AbstractVector{Int},
)
    matrix = plan.reconstructor.matrix
    candidate = workspace.candidate_reconstructed
    @inbounds for subaperture in subapertures
        _measure_shack_hartmann_range!(
            workspace.slopes,
            workspace.flux,
            workspace.validity,
            plan.sensing,
            workspace.detector,
            subaperture:subaperture,
        )
        x_column = 2 * subaperture - 1
        x_slope = workspace.slopes[subaperture, 1]
        @inbounds @simd for output in eachindex(candidate)
            candidate[output] = fma(matrix[output, x_column], x_slope, candidate[output])
        end
        y_column = x_column + 1
        y_slope = workspace.slopes[subaperture, 2]
        @inbounds @simd for output in eachindex(candidate)
            candidate[output] = fma(matrix[output, y_column], y_slope, candidate[output])
        end
        workspace.measured_subapertures += 1
    end
    for value in candidate
        isfinite(value) || return ShackHartmannReadoutNonFiniteReconstruction
    end
    return nothing
end

function _process_shack_hartmann_readout!(
    reconstructed::AbstractVector{Float32},
    workspace::ShackHartmannReadoutReconstructionWorkspace,
    plan::ShackHartmannReadoutReconstructionPlan,
    readout_block::AbstractArray{Float32,3},
    sample::SampleMetadata,
)
    _validate_shack_hartmann_readout_storage(
        reconstructed,
        workspace,
        plan,
        readout_block,
    )
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return ShackHartmannReadoutOffsetOverflow
    ordinal = Int(sample.offset)
    accepted = _accept_detector_readout!(
        workspace.frame,
        sample,
        plan.readout.block_count,
    )
    if !accepted
        _clear_shack_hartmann_readout_partial!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(ordinal) && _clear_shack_hartmann_readout_partial!(workspace)

    _scatter_detector_readout_block!(
        workspace.detector,
        plan.readout,
        readout_block,
        ordinal,
    )
    status = _accumulate_shack_hartmann_readout_group!(
        workspace,
        plan,
        plan.completion_groups[ordinal + 1],
    )
    if !isnothing(status)
        _abandon_shack_hartmann_readout!(workspace)
        return status
    end

    sample.terminal || return AlgorithmProcessDeferred()
    workspace.measured_subapertures == length(plan.sensing.regions.origins) ||
        error("prepared readout readiness did not complete every subaperture")
    copyto!(reconstructed, workspace.candidate_reconstructed)
    workspace.terminal_candidate = true
    return AlgorithmProcessComplete(
        SampleMetadata(
            sample;
            offset=0,
            terminal=true,
            discontinuity=workspace.frame.current_discontinuity,
            corrupted=false,
        ),
    )
end

"""
Decode successive detector readout blocks, measure Shack-Hartmann centroids,
and reconstruct the complete frame. Readout offsets locate blocks in the
detector map; the output follows the reconstructed vector schema.
"""
@algorithm ShackHartmannReadoutReconstructorF32 begin
    label = "shwfs-readout-reconstructor-f32"
    failure = ShackHartmannReadoutReconstructionFailure

    """
    Set detector-region/subaperture geometry and mappings, reconstructed count,
    optional origins, coordinate scale, thresholds, reference slopes and active
    mask, initial reconstructor, and versioned output schema.
    """
    @configuration ShackHartmannReadoutReconstructorF32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
        subaperture_size::NTuple{2,Int} => graph_rebuild
        subaperture_count::Int => graph_rebuild
        reconstructed_count::Int => graph_rebuild
        initial_subaperture_origins::Union{Nothing,Vector{Vector{Int}}} => construction
        coordinate_scale::Float32 => construction
        pixel_threshold::Float32 => construction
        flux_threshold::Float32 => construction
        reference_slopes::Union{Nothing,Vector{Float32}} => construction
        active::Union{Nothing,Vector{Bool}} => construction
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        reconstructed_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :readout_block,
            schema_element_type(CALIBRATED_PIXEL_READOUT_BLOCK_V1),
            (
                length(configuration.detector_origins),
                configuration.block_lines,
                configuration.region_samples,
            ),
            CALIBRATED_PIXEL_READOUT_BLOCK_V1;
            name="readout-block",
            sample=:readout_sample,
            rate=(div(configuration.region_lines, configuration.block_lines), 1),
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:readout_block,
            publication=:conditional,
        ),
    )

    prepare(configuration) = begin
        readout = _prepare_detector_readout(
            configuration.detector_size[1],
            configuration.detector_size[2],
            configuration.region_lines,
            configuration.block_lines,
            configuration.region_samples,
            configuration.detector_origins,
            configuration.detector_line_steps,
            configuration.detector_sample_steps,
        )
        origins = _initial_shack_hartmann_origins(
            configuration.detector_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        sensing = ShackHartmannImagePlan(
            configuration.detector_size,
            configuration.subaperture_size,
            origins,
            configuration.coordinate_scale,
            configuration.pixel_threshold,
            configuration.flux_threshold;
            reference_slopes=configuration.reference_slopes,
            active=configuration.active,
        )
        slope_count = Base.Checked.checked_mul(configuration.subaperture_count, 2)
        reconstructor = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            slope_count,
            configuration.initial_reconstructor,
        )
        plan = ShackHartmannReadoutReconstructionPlan(
            sensing,
            reconstructor,
            readout,
        )
        (
            plan=plan,
            workspace=ShackHartmannReadoutReconstructionWorkspace(plan),
        )
    end

    process!(plan, workspace) = begin
        isnothing(readout_sample) && return ShackHartmannReadoutMetadataMissing
        _process_shack_hartmann_readout!(
            reconstructed,
            workspace,
            plan,
            readout_block,
            readout_sample,
        )
    end
end
