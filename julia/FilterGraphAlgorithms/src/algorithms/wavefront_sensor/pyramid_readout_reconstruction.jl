"""
Prepared detector-readout pyramid normalization and incremental reconstruction.

Selected pixels retain pupil-major scientific identity. Detector acquisition
order determines only when each pixel is normalized and its reconstructor
column is accumulated; caller-owned outputs remain unpublished until the
terminal readout block.
"""
struct PyramidReadoutReconstructionPlan{
    Reconstructor<:DenseReconstructorPlan{Float32},
} <: AbstractAlgorithmPlan
    regions::RegionExtractionPlan
    normalization::PyramidPixelPlan{
        Float32,
        Vector{Bool},
        Vector{Int},
        Nothing,
    }
    reconstructor::Reconstructor
    readout::DetectorReadoutPlan
    selected_detector_rows::Vector{Int}
    selected_detector_columns::Vector{Int}
    pixel_completion_ordinals::Vector{Int}
    completion_groups::Vector{Vector{Int}}

    function PyramidReadoutReconstructionPlan(
        regions::RegionExtractionPlan,
        normalization::PyramidPixelPlan{
            Float32,
            Vector{Bool},
            Vector{Int},
            Nothing,
        },
        reconstructor::DenseReconstructorPlan{Float32},
        readout::DetectorReadoutPlan,
    )
        regions.image_size == readout.detector_size || throw(
            DimensionMismatch(
                "detector readout and pyramid image extents must match",
            ),
        )
        length(regions.origins) == PYRAMID_PUPIL_COUNT || throw(
            DimensionMismatch("pyramid geometry must contain four pupil regions"),
        )
        pupil_pixel_count = Base.Checked.checked_mul(regions.region_size...)
        length(normalization.mask) == pupil_pixel_count || throw(
            DimensionMismatch("pyramid mask and pupil-region extents must match"),
        )
        pixel_count = normalization.output_pixel_count
        size(reconstructor.matrix, 2) == pixel_count || throw(
            DimensionMismatch(
                "reconstructor columns must match pupil-major reconstruction pixels",
            ),
        )

        detector_rows = Vector{Int}(undef, pixel_count)
        detector_columns = Vector{Int}(undef, pixel_count)
        completion_ordinals = Vector{Int}(undef, pixel_count)
        completion_groups = Vector{Vector{Int}}(undef, readout.block_count)
        for ordinal in eachindex(completion_groups)
            completion_groups[ordinal] = Int[]
        end
        pupil_rows, pupil_columns = regions.region_size
        selected_per_pupil = length(normalization.selected_pixel_offsets)
        for pupil in 1:PYRAMID_PUPIL_COUNT
            first_row, first_column = regions.origins[pupil]
            for selected_pixel in 1:selected_per_pupil
                pixel_offset = normalization.selected_pixel_offsets[selected_pixel]
                pupil_row = div(pixel_offset, pupil_columns)
                pupil_column = rem(pixel_offset, pupil_columns)
                detector_row = first_row + pupil_row + 1
                detector_column = first_column + pupil_column + 1
                canonical_pixel = (pupil - 1) * selected_per_pupil + selected_pixel
                completion_ordinal =
                    readout.arrival_ordinals[detector_row, detector_column]
                detector_rows[canonical_pixel] = detector_row
                detector_columns[canonical_pixel] = detector_column
                completion_ordinals[canonical_pixel] = completion_ordinal
                push!(
                    completion_groups[completion_ordinal + 1],
                    canonical_pixel,
                )
            end
        end
        return new{typeof(reconstructor)}(
            regions,
            normalization,
            reconstructor,
            readout,
            detector_rows,
            detector_columns,
            completion_ordinals,
            completion_groups,
        )
    end
end

"""Progressive pyramid readout state and unpublished terminal outputs."""
mutable struct PyramidReadoutReconstructionWorkspace
    detector::Matrix{Float32}
    candidate_reconstruction_pixels::Matrix{Float32}
    candidate_reconstructed::Vector{Float32}
    committed_gain::Float32
    candidate_gain::Float32
    candidate_mean::Float32
    frame::DetectorReadoutFrame
    terminal_candidate::Bool
end

function PyramidReadoutReconstructionWorkspace(
    plan::PyramidReadoutReconstructionPlan,
)
    selected = length(plan.normalization.selected_pixel_offsets)
    return PyramidReadoutReconstructionWorkspace(
        zeros(Float32, plan.readout.detector_size),
        zeros(Float32, PYRAMID_PUPIL_COUNT, selected),
        zeros(Float32, size(plan.reconstructor.matrix, 1)),
        0.0f0,
        0.0f0,
        0.0f0,
        DetectorReadoutFrame(),
        false,
    )
end

function _clear_pyramid_readout_partial!(
    workspace::PyramidReadoutReconstructionWorkspace,
)
    fill!(workspace.candidate_reconstruction_pixels, 0.0f0)
    fill!(workspace.candidate_reconstructed, 0.0f0)
    workspace.candidate_gain = workspace.committed_gain
    workspace.candidate_mean = 0.0f0
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_pyramid_readout!(
    workspace::PyramidReadoutReconstructionWorkspace,
)
    _abandon_detector_readout!(workspace.frame)
    return _clear_pyramid_readout_partial!(workspace)
end

function commit!(workspace::PyramidReadoutReconstructionWorkspace)
    if workspace.terminal_candidate
        workspace.committed_gain = workspace.candidate_gain
        _complete_detector_readout!(workspace.frame)
        _clear_pyramid_readout_partial!(workspace)
    end
    workspace.candidate_gain = workspace.committed_gain
    return workspace
end

discard!(workspace::PyramidReadoutReconstructionWorkspace) =
    _abandon_pyramid_readout!(workspace)

function reset!(workspace::PyramidReadoutReconstructionWorkspace)
    fill!(workspace.detector, 0.0f0)
    workspace.committed_gain = 0.0f0
    workspace.candidate_gain = 0.0f0
    workspace.candidate_mean = 0.0f0
    workspace.frame.active = false
    workspace.frame.sequence = 0
    workspace.frame.expected_ordinal = 0
    workspace.frame.pending_discontinuity = false
    workspace.frame.current_discontinuity = false
    return _clear_pyramid_readout_partial!(workspace)
end

"""Expected rejection from detector-readout pyramid reconstruction."""
@enum PyramidReadoutReconstructionFailure::UInt8 begin
    PyramidReadoutMetadataMissing = 1
    PyramidReadoutOffsetOverflow = 2
    PyramidReadoutNonFiniteInput = 3
    PyramidReadoutNonFiniteMean = 4
    PyramidReadoutNonFiniteNormalizationGain = 5
    PyramidReadoutNonFiniteOutput = 6
    PyramidReadoutNonFiniteReconstruction = 7
end

@doc "The readout block did not carry required sample metadata."
PyramidReadoutMetadataMissing
@doc "The acquisition ordinal cannot be represented by the Julia target."
PyramidReadoutOffsetOverflow
@doc "A pyramid pupil contains a non-finite input pixel."
PyramidReadoutNonFiniteInput
@doc "The full-pupil mean calculation produced a non-finite value."
PyramidReadoutNonFiniteMean
@doc "The current-frame normalization gain is non-finite."
PyramidReadoutNonFiniteNormalizationGain
@doc "Previous-frame normalization produced a non-finite output pixel."
PyramidReadoutNonFiniteOutput
@doc "Incremental reconstruction produced a non-finite coordinate."
PyramidReadoutNonFiniteReconstruction

@inline function _pyramid_readout_aliases_workspace(
    array,
    workspace::PyramidReadoutReconstructionWorkspace,
)
    return Base.mightalias(array, workspace.detector) ||
        Base.mightalias(array, workspace.candidate_reconstruction_pixels) ||
        Base.mightalias(array, workspace.candidate_reconstructed)
end

function _validate_pyramid_readout_storage(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    reconstructed::AbstractVector{Float32},
    workspace::PyramidReadoutReconstructionWorkspace,
    plan::PyramidReadoutReconstructionPlan,
    readout_block::AbstractArray{Float32,3},
)
    Base.require_one_based_indexing(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        readout_block,
    )
    selected = length(plan.normalization.selected_pixel_offsets)
    size(reconstruction_pixels) == (PYRAMID_PUPIL_COUNT, selected) || throw(
        DimensionMismatch(
            "pyramid reconstruction-pixel output has an incompatible shape",
        ),
    )
    length(mean_pupil_intensity) == 1 || throw(
        DimensionMismatch("mean-pupil-intensity output must contain one value"),
    )
    length(reconstructed) == size(plan.reconstructor.matrix, 1) || throw(
        DimensionMismatch("reconstructed output has an incompatible extent"),
    )
    size(readout_block) == detector_readout_block_shape(plan.readout) || throw(
        DimensionMismatch("readout block has an incompatible detector shape"),
    )
    size(workspace.detector) == plan.readout.detector_size || throw(
        DimensionMismatch("pyramid readout Workspace has an invalid detector"),
    )
    size(workspace.candidate_reconstruction_pixels) ==
        (PYRAMID_PUPIL_COUNT, selected) || throw(
        DimensionMismatch(
            "pyramid readout Workspace has an invalid pixel extent",
        ),
    )
    length(workspace.candidate_reconstructed) ==
        size(plan.reconstructor.matrix, 1) || throw(
        DimensionMismatch(
            "pyramid readout Workspace has an invalid reconstruction extent",
        ),
    )

    Base.mightalias(reconstruction_pixels, mean_pupil_intensity) && throw(
        ArgumentError("pyramid outputs must not alias one another"),
    )
    Base.mightalias(reconstruction_pixels, reconstructed) && throw(
        ArgumentError("pyramid outputs must not alias one another"),
    )
    Base.mightalias(mean_pupil_intensity, reconstructed) && throw(
        ArgumentError("pyramid outputs must not alias one another"),
    )
    Base.mightalias(reconstruction_pixels, readout_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector readout block"),
    )
    Base.mightalias(mean_pupil_intensity, readout_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector readout block"),
    )
    Base.mightalias(reconstructed, readout_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector readout block"),
    )
    Base.mightalias(reconstruction_pixels, plan.reconstructor.matrix) && throw(
        ArgumentError("pyramid output must not alias the prepared reconstructor"),
    )
    Base.mightalias(mean_pupil_intensity, plan.reconstructor.matrix) && throw(
        ArgumentError("pyramid output must not alias the prepared reconstructor"),
    )
    Base.mightalias(reconstructed, plan.reconstructor.matrix) && throw(
        ArgumentError("pyramid output must not alias the prepared reconstructor"),
    )
    _pyramid_readout_aliases_workspace(reconstruction_pixels, workspace) &&
        throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_readout_aliases_workspace(mean_pupil_intensity, workspace) &&
        throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_readout_aliases_workspace(reconstructed, workspace) &&
        throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_readout_aliases_workspace(readout_block, workspace) &&
        throw(ArgumentError("readout input must not alias Workspace storage"))
    return nothing
end

function _accumulate_pyramid_readout_group!(
    workspace::PyramidReadoutReconstructionWorkspace,
    plan::PyramidReadoutReconstructionPlan,
    canonical_pixels::AbstractVector{Int},
)
    selected = length(plan.normalization.selected_pixel_offsets)
    gain = workspace.committed_gain
    matrix = plan.reconstructor.matrix
    candidate = workspace.candidate_reconstructed
    for canonical_pixel in canonical_pixels
        input = workspace.detector[
            plan.selected_detector_rows[canonical_pixel],
            plan.selected_detector_columns[canonical_pixel],
        ]
        isfinite(input) || return PyramidReadoutNonFiniteInput
        normalized = gain * input
        isfinite(normalized) || return PyramidReadoutNonFiniteOutput
        pupil = div(canonical_pixel - 1, selected) + 1
        selected_pixel = rem(canonical_pixel - 1, selected) + 1
        workspace.candidate_reconstruction_pixels[pupil, selected_pixel] =
            normalized
        @inbounds @simd for output in eachindex(candidate)
            candidate[output] = fma(matrix[output, canonical_pixel], normalized, candidate[output])
        end
    end
    for value in candidate
        isfinite(value) || return PyramidReadoutNonFiniteReconstruction
    end
    return nothing
end

function _finish_pyramid_readout_frame!(
    workspace::PyramidReadoutReconstructionWorkspace,
    plan::PyramidReadoutReconstructionPlan,
)
    sum = 0.0f0
    pupil_rows, pupil_columns = plan.regions.region_size
    for pupil in 1:PYRAMID_PUPIL_COUNT
        first_row, first_column = plan.regions.origins[pupil]
        for pupil_row in 1:pupil_rows, pupil_column in 1:pupil_columns
            input = workspace.detector[
                first_row + pupil_row,
                first_column + pupil_column,
            ]
            isfinite(input) || return PyramidReadoutNonFiniteInput
            sum += input
            isfinite(sum) || return PyramidReadoutNonFiniteMean
        end
    end
    mean = sum * plan.normalization.mean_scale
    isfinite(mean) || return PyramidReadoutNonFiniteMean
    candidate_gain = mean > 0.0f0 ? inv(mean) : 0.0f0
    isfinite(candidate_gain) ||
        return PyramidReadoutNonFiniteNormalizationGain
    workspace.candidate_gain = candidate_gain
    workspace.candidate_mean = mean
    workspace.terminal_candidate = true
    return nothing
end

function _process_pyramid_readout_reconstruction!(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    reconstructed::AbstractVector{Float32},
    workspace::PyramidReadoutReconstructionWorkspace,
    plan::PyramidReadoutReconstructionPlan,
    readout_block::AbstractArray{Float32,3},
    sample::SampleMetadata,
)
    _validate_pyramid_readout_storage(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        workspace,
        plan,
        readout_block,
    )
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return PyramidReadoutOffsetOverflow
    ordinal = Int(sample.offset)
    accepted = _accept_detector_readout!(
        workspace.frame,
        sample,
        plan.readout.block_count,
    )
    if !accepted
        _clear_pyramid_readout_partial!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(ordinal) && _clear_pyramid_readout_partial!(workspace)

    _scatter_detector_readout_block!(
        workspace.detector,
        plan.readout,
        readout_block,
        ordinal,
    )
    failure = _accumulate_pyramid_readout_group!(
        workspace,
        plan,
        plan.completion_groups[ordinal + 1],
    )
    if !isnothing(failure)
        _abandon_pyramid_readout!(workspace)
        return failure
    end
    sample.terminal || return AlgorithmProcessDeferred()

    failure = _finish_pyramid_readout_frame!(workspace, plan)
    if !isnothing(failure)
        _abandon_pyramid_readout!(workspace)
        return failure
    end
    copyto!(
        reconstruction_pixels,
        workspace.candidate_reconstruction_pixels,
    )
    mean_pupil_intensity[1] = workspace.candidate_mean
    copyto!(reconstructed, workspace.candidate_reconstructed)
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
Progressive Float32 pyramid-WFS readout and reconstruction Algorithm.

The declaration consumes acquisition-mapped readout blocks and conditionally
publishes canonical reconstruction pixels, the current full-pupil mean, and a
dense reconstruction at the terminal block.
"""
@algorithm PyramidReadoutReconstructorF32 begin
    label = "pwfs-readout-reconstructor-f32"
    failure = PyramidReadoutReconstructionFailure

    """Configuration for [`PyramidReadoutReconstructorF32`](@ref)."""
    @configuration PyramidReadoutReconstructorF32Configuration begin
        detector_size::NTuple{2,Int} => graph_rebuild
        region_lines::Int => graph_rebuild
        block_lines::Int => graph_rebuild
        region_samples::Int => graph_rebuild
        detector_origins::Vector{Vector{Int}} => graph_rebuild
        detector_line_steps::Vector{Vector{Int}} => graph_rebuild
        detector_sample_steps::Vector{Vector{Int}} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
        reconstructed_count::Int => graph_rebuild
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
            :reconstruction_pixels,
            schema_element_type(PYRAMID_RECONSTRUCTION_PIXELS_V1),
            (
                PYRAMID_PUPIL_COUNT,
                _configured_pyramid_selected_pixel_count(
                    configuration.pupil_mask,
                    configuration.pupil_size[1],
                    configuration.pupil_size[2],
                ),
            ),
            PYRAMID_RECONSTRUCTION_PIXELS_V1;
            name="reconstruction-pixels",
            metadata_from=:readout_block,
            publication=:conditional,
        ),
        output(
            :mean_pupil_intensity,
            schema_element_type(PYRAMID_MEAN_PUPIL_INTENSITY_V1),
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1;
            name="mean-pupil-intensity",
            metadata_from=:readout_block,
            publication=:conditional,
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
        origins = _configured_pyramid_origins(configuration.pupil_origins)
        regions = RegionExtractionPlan(
            configuration.detector_size,
            origins,
            configuration.pupil_size,
        )
        mask = _configured_pyramid_mask(
            configuration.pupil_mask,
            configuration.pupil_size[1],
            configuration.pupil_size[2],
        )
        row_major_mask = Bool[
            mask[row, column] for row in axes(mask, 1) for column in axes(mask, 2)
        ]
        normalization = PyramidPixelPlan{Float32}(row_major_mask)
        reconstructor = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            normalization.output_pixel_count,
            configuration.initial_reconstructor,
        )
        plan = PyramidReadoutReconstructionPlan(
            regions,
            normalization,
            reconstructor,
            readout,
        )
        (plan=plan, workspace=PyramidReadoutReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(readout_sample) && return PyramidReadoutMetadataMissing
        _process_pyramid_readout_reconstruction!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            workspace,
            plan,
            readout_block,
            readout_sample,
        )
    end
end
