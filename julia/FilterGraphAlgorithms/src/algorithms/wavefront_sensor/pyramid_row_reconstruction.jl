"""
Prepared row-block pyramid normalization and incremental dense reconstruction.

Pupil identity remains in declaration order even when detector rows make the
pupils available in a different order. Newly normalized pupil-major pixels are
accumulated immediately, while caller-owned outputs remain unpublished until
the terminal row block.
"""
struct PyramidRowReconstructionPlan{
    Reconstructor<:DenseReconstructorPlan{Float32},
} <: AbstractAlgorithmPlan
    pixels::PyramidPixelRowPlan
    reconstructor::Reconstructor

    function PyramidRowReconstructionPlan(
        pixels::PyramidPixelRowPlan,
        reconstructor::DenseReconstructorPlan{Float32},
    )
        pixel_count = Base.Checked.checked_mul(
            PYRAMID_PUPIL_COUNT,
            pixels.selected_pixel_count,
        )
        size(reconstructor.matrix, 2) == pixel_count || throw(
            DimensionMismatch(
                "reconstructor columns must match pupil-major reconstruction pixels",
            ),
        )
        return new{typeof(reconstructor)}(pixels, reconstructor)
    end
end

"""Progressive normalization and unpublished reconstruction for one row frame."""
mutable struct PyramidRowReconstructionWorkspace{
    Pixels<:PyramidPixelRowWorkspace,
    Reconstructed<:AbstractVector{Float32},
    Accumulated<:AbstractVector{Int},
}
    pixels::Pixels
    candidate_reconstructed::Reconstructed
    accumulated_selected::Accumulated
end

function PyramidRowReconstructionWorkspace(plan::PyramidRowReconstructionPlan)
    pixels = PyramidPixelRowWorkspace(plan.pixels)
    reconstructed = zeros(Float32, size(plan.reconstructor.matrix, 1))
    accumulated = zeros(Int, PYRAMID_PUPIL_COUNT)
    return PyramidRowReconstructionWorkspace{
        typeof(pixels),
        typeof(reconstructed),
        typeof(accumulated),
    }(pixels, reconstructed, accumulated)
end

function _clear_pyramid_row_reconstruction_partial!(
    workspace::PyramidRowReconstructionWorkspace,
)
    fill!(workspace.candidate_reconstructed, 0.0f0)
    fill!(workspace.accumulated_selected, 0)
    return workspace
end

function _abandon_pyramid_row_reconstruction!(
    workspace::PyramidRowReconstructionWorkspace,
)
    discard!(workspace.pixels)
    return _clear_pyramid_row_reconstruction_partial!(workspace)
end

function commit!(workspace::PyramidRowReconstructionWorkspace)
    terminal_candidate = workspace.pixels.terminal_candidate
    commit!(workspace.pixels)
    terminal_candidate &&
        _clear_pyramid_row_reconstruction_partial!(workspace)
    return workspace
end

discard!(workspace::PyramidRowReconstructionWorkspace) =
    _abandon_pyramid_row_reconstruction!(workspace)

function reset!(workspace::PyramidRowReconstructionWorkspace)
    reset!(workspace.pixels)
    return _clear_pyramid_row_reconstruction_partial!(workspace)
end

_plan_adopted!(workspace::PyramidRowReconstructionWorkspace) =
    _abandon_pyramid_row_reconstruction!(workspace)

"""Expected rejection from row-block pyramid reconstruction."""
@enum PyramidRowReconstructionFailure::UInt8 begin
    PyramidRowMetadataMissing = 1
    PyramidRowOffsetOverflow = 2
    PyramidRowNonFiniteInput = 3
    PyramidRowNonFiniteMean = 4
    PyramidRowNonFiniteNormalizationGain = 5
    PyramidRowNonFiniteOutput = 6
    PyramidRowNonFiniteReconstruction = 7
end

@doc "The row block did not carry required sample metadata."
PyramidRowMetadataMissing
@doc "The row-block offset cannot be represented by the Julia target."
PyramidRowOffsetOverflow
@doc "A pyramid pupil contains a non-finite input pixel."
PyramidRowNonFiniteInput
@doc "The full-pupil mean calculation produced a non-finite value."
PyramidRowNonFiniteMean
@doc "The current-frame normalization gain is non-finite."
PyramidRowNonFiniteNormalizationGain
@doc "Previous-frame normalization produced a non-finite output pixel."
PyramidRowNonFiniteOutput
@doc "Incremental reconstruction produced a non-finite coordinate."
PyramidRowNonFiniteReconstruction

@inline function _pyramid_row_reconstruction_failure(
    failure::PyramidPixelRowFailure,
)
    failure === PyramidPixelRowMetadataMissing &&
        return PyramidRowMetadataMissing
    failure === PyramidPixelRowOffsetOverflow &&
        return PyramidRowOffsetOverflow
    failure === PyramidPixelRowNonFiniteInput &&
        return PyramidRowNonFiniteInput
    failure === PyramidPixelRowNonFiniteMean &&
        return PyramidRowNonFiniteMean
    failure === PyramidPixelRowNonFiniteNormalizationGain &&
        return PyramidRowNonFiniteNormalizationGain
    failure === PyramidPixelRowNonFiniteOutput &&
        return PyramidRowNonFiniteOutput
    error("unhandled pyramid row normalization failure")
end

function _replace_pyramid_row_reconstruction_origins(
    plan::PyramidRowReconstructionPlan,
    values,
)
    return PyramidRowReconstructionPlan(
        _replace_pyramid_row_origins(plan.pixels, values),
        plan.reconstructor,
    )
end

function _replace_pyramid_row_reconstruction_mask(
    plan::PyramidRowReconstructionPlan,
    values,
)
    return PyramidRowReconstructionPlan(
        _replace_pyramid_row_mask(plan.pixels, values),
        plan.reconstructor,
    )
end

function _replace_pyramid_row_reconstructor(
    plan::PyramidRowReconstructionPlan,
    values,
)
    return PyramidRowReconstructionPlan(
        plan.pixels,
        DenseReconstructorPlan(values),
    )
end

@inline function _pyramid_row_reconstruction_aliases_workspace(
    array,
    workspace::PyramidRowReconstructionWorkspace,
)
    pixels = workspace.pixels
    return Base.mightalias(array, pixels.detector) ||
        Base.mightalias(array, pixels.staged_reconstruction_pixels) ||
        Base.mightalias(array, workspace.candidate_reconstructed)
end

function _validate_pyramid_row_reconstruction_storage(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    reconstructed::AbstractVector{Float32},
    workspace::PyramidRowReconstructionWorkspace,
    plan::PyramidRowReconstructionPlan,
    row_block::AbstractMatrix{Float32},
)
    Base.require_one_based_indexing(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        row_block,
    )
    selected = plan.pixels.selected_pixel_count
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
    size(row_block) ==
        (plan.pixels.block_rows, plan.pixels.image_columns) || throw(
        DimensionMismatch("row block has an incompatible detector shape"),
    )
    size(workspace.pixels.detector) ==
        (plan.pixels.image_rows, plan.pixels.image_columns) || throw(
        DimensionMismatch("pyramid row Workspace has an invalid detector"),
    )
    length(workspace.candidate_reconstructed) ==
        size(plan.reconstructor.matrix, 1) || throw(
        DimensionMismatch(
            "pyramid row Workspace has an invalid reconstruction extent",
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
    Base.mightalias(reconstruction_pixels, row_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector row block"),
    )
    Base.mightalias(mean_pupil_intensity, row_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector row block"),
    )
    Base.mightalias(reconstructed, row_block) && throw(
        ArgumentError("pyramid outputs must not alias the detector row block"),
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
    _pyramid_row_reconstruction_aliases_workspace(
        reconstruction_pixels,
        workspace,
    ) && throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_row_reconstruction_aliases_workspace(
        mean_pupil_intensity,
        workspace,
    ) && throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_row_reconstruction_aliases_workspace(reconstructed, workspace) &&
        throw(ArgumentError("pyramid output must not alias Workspace storage"))
    _pyramid_row_reconstruction_aliases_workspace(row_block, workspace) &&
        throw(ArgumentError("row input must not alias Workspace storage"))
    return nothing
end

"""
    _accumulate_pyramid_row_reconstruction_rows!(
        candidate, matrix, pixels, selected, pupil, first_pixel, last_pixel,
        output_rows,
    )

Accumulate the specified already-normalized pupil pixels into
`candidate[output_rows]`. `output_rows` must be a valid, disjoint partition
when called concurrently.
"""
@inline function _accumulate_pyramid_row_reconstruction_rows!(
    candidate::AbstractVector{Float32},
    matrix::AbstractMatrix{Float32},
    pixels::AbstractMatrix{Float32},
    selected::Int,
    pupil::Int,
    first_pixel::Int,
    last_pixel::Int,
    output_rows,
)
    for pixel in first_pixel:last_pixel
        column = (pupil - 1) * selected + pixel
        value = pixels[pupil, pixel]
        @inbounds @simd for output in output_rows
            candidate[output] = fma(matrix[output, column], value, candidate[output])
        end
    end
    return candidate
end

function _accumulate_pyramid_row_reconstruction!(
    workspace::PyramidRowReconstructionWorkspace,
    plan::PyramidRowReconstructionPlan,
)
    selected = plan.pixels.selected_pixel_count
    pixels = workspace.pixels.staged_reconstruction_pixels
    matrix = plan.reconstructor.matrix
    candidate = workspace.candidate_reconstructed
    for pupil in 1:PYRAMID_PUPIL_COUNT
        first_pixel = workspace.accumulated_selected[pupil] + 1
        last_pixel = workspace.pixels.processed_selected[pupil]
        _accumulate_pyramid_row_reconstruction_rows!(
            candidate,
            matrix,
            pixels,
            selected,
            pupil,
            first_pixel,
            last_pixel,
            eachindex(candidate),
        )
        workspace.accumulated_selected[pupil] = last_pixel
    end
    for value in candidate
        isfinite(value) || return PyramidRowNonFiniteReconstruction
    end
    return nothing
end

function _process_pyramid_row_reconstruction!(
    reconstruction_pixels::AbstractMatrix{Float32},
    mean_pupil_intensity::AbstractVector{Float32},
    reconstructed::AbstractVector{Float32},
    workspace::PyramidRowReconstructionWorkspace,
    plan::PyramidRowReconstructionPlan,
    row_block::AbstractMatrix{Float32},
    sample::SampleMetadata,
)
    _validate_pyramid_row_reconstruction_storage(
        reconstruction_pixels,
        mean_pupil_intensity,
        reconstructed,
        workspace,
        plan,
        row_block,
    )
    iszero(sample.offset) &&
        _clear_pyramid_row_reconstruction_partial!(workspace)
    result = _stage_pyramid_pixel_row!(
        plan.pixels,
        workspace.pixels,
        row_block,
        sample,
    )
    if result isa PyramidPixelRowFailure
        failure = _pyramid_row_reconstruction_failure(result)
        _abandon_pyramid_row_reconstruction!(workspace)
        return failure
    end
    if !workspace.pixels.frame.active
        _clear_pyramid_row_reconstruction_partial!(workspace)
        return result
    end

    failure = _accumulate_pyramid_row_reconstruction!(workspace, plan)
    if !isnothing(failure)
        _abandon_pyramid_row_reconstruction!(workspace)
        return failure
    end
    if result isa AlgorithmProcessResult && result.complete
        all(==(plan.pixels.selected_pixel_count), workspace.accumulated_selected) ||
            error("prepared pyramid readiness did not cover every reconstruction pixel")
        _publish_pyramid_pixel_row!(
            reconstruction_pixels,
            mean_pupil_intensity,
            plan.pixels,
            workspace.pixels,
        )
        copyto!(reconstructed, workspace.candidate_reconstructed)
    end
    return result
end

"""
Combine progressive pyramid row normalization and dense reconstruction in
one declaration. Accumulation progresses with each row block. The selected
pupil pixels, mean intensity and reconstructed vector are conditional outputs
published only when the complete image is available.
"""
@algorithm PyramidRowReconstructorF32 begin
    label = "pwfs-row-reconstructor-f32"
    failure = PyramidRowReconstructionFailure
    operation = PyramidRowReconstructionF32Operation()

    """
    Set image/row-block and four-pupil geometry, optional pupil mask, output
    reconstruction extent/schema, and optional initial reconstructor matrix.
    The row-block count is determined by image rows divided by block rows.
    """
    @configuration PyramidRowReconstructorF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
        pupil_mask::Union{Nothing,Vector{Bool}} => construction
        reconstructed_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        reconstructed_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :row_block,
            schema_element_type(CALIBRATED_PIXEL_ROW_BLOCK_V1),
            (configuration.row_block_rows, configuration.image_size[2]),
            CALIBRATED_PIXEL_ROW_BLOCK_V1;
            name="row-block",
            sample=:row_sample,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
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
            metadata_from=:row_block,
            publication=:conditional,
        ),
        output(
            :mean_pupil_intensity,
            schema_element_type(PYRAMID_MEAN_PUPIL_INTENSITY_V1),
            (1,),
            PYRAMID_MEAN_PUPIL_INTENSITY_V1;
            name="mean-pupil-intensity",
            metadata_from=:row_block,
            publication=:conditional,
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:row_block,
            publication=:conditional,
        ),
        parameter(
            :pupil_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (PYRAMID_PUPIL_COUNT, 2),
            REGION_ORIGINS_V1;
            name="pupil-origins",
            replace=(plan, values) ->
                _replace_pyramid_row_reconstruction_origins(plan, values),
        ),
        parameter(
            :pupil_mask,
            schema_element_type(PYRAMID_PUPIL_MASK_V1),
            configuration.pupil_size,
            PYRAMID_PUPIL_MASK_V1;
            name="pupil-mask",
            replace=(plan, values) ->
                _replace_pyramid_row_reconstruction_mask(plan, values),
        ),
        parameter(
            :reconstructor,
            schema_element_type(PWFS_RECONSTRUCTOR_V1),
            (
                configuration.reconstructed_count,
                Base.Checked.checked_mul(
                    PYRAMID_PUPIL_COUNT,
                    _configured_pyramid_selected_pixel_count(
                        configuration.pupil_mask,
                        configuration.pupil_size[1],
                        configuration.pupil_size[2],
                    ),
                ),
            ),
            PWFS_RECONSTRUCTOR_V1;
            replace=(plan, values) ->
                _replace_pyramid_row_reconstructor(plan, values),
        ),
    )

    prepare(configuration) = begin
        pixels = _prepared_pyramid_pixel_row_plan(
            configuration.image_size[1],
            configuration.image_size[2],
            configuration.row_block_rows,
            configuration.pupil_size[1],
            configuration.pupil_size[2],
            _configured_pyramid_origins(configuration.pupil_origins),
            _configured_pyramid_mask(
                configuration.pupil_mask,
                configuration.pupil_size[1],
                configuration.pupil_size[2],
            ),
        )
        reconstruction_pixel_count = Base.Checked.checked_mul(
            PYRAMID_PUPIL_COUNT,
            pixels.selected_pixel_count,
        )
        reconstructor = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            reconstruction_pixel_count,
            configuration.initial_reconstructor,
        )
        plan = PyramidRowReconstructionPlan(pixels, reconstructor)
        (plan=plan, workspace=PyramidRowReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(row_sample) && return PyramidRowMetadataMissing
        _process_pyramid_row_reconstruction!(
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
            workspace,
            plan,
            row_block,
            row_sample,
        )
    end
end
