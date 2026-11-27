"""
Prepared row-block Shack–Hartmann sensing and incremental dense reconstruction.

Subapertures retain scientific declaration order. A separate completion
schedule marks detector regions ready by their final row, while reconstruction
advances only across the greatest ready prefix in scientific order.
"""
struct ShackHartmannRowReconstructionPlan{
    Reconstructor<:DenseReconstructorPlan{Float32},
    Schedule<:AbstractVector{Int},
} <: AbstractAlgorithmPlan
    sensing::ShackHartmannImagePlan
    reconstructor::Reconstructor
    completion_schedule::Schedule
    block_rows::Int

    function ShackHartmannRowReconstructionPlan(
        sensing::ShackHartmannImagePlan,
        reconstructor::DenseReconstructorPlan{Float32},
        block_rows::Int,
    )
        image_rows = sensing.regions.image_size[1]
        0 < block_rows < image_rows || throw(
            ArgumentError(
                "row_block_rows must be positive and smaller than the image row count",
            ),
        )
        rem(image_rows, block_rows) == 0 || throw(
            ArgumentError("row_block_rows must divide the image row count"),
        )
        slope_count = Base.Checked.checked_mul(
            length(sensing.regions.origins),
            2,
        )
        size(reconstructor.matrix, 2) == slope_count || throw(
            DimensionMismatch(
                "reconstructor columns must match pair-interleaved slopes",
            ),
        )
        region_rows = sensing.regions.region_size[1]
        schedule = collect(eachindex(sensing.regions.origins))
        sort!(schedule; by=subaperture -> (
            sensing.regions.origins[subaperture][1] + region_rows,
            subaperture,
        ))
        return new{typeof(reconstructor),typeof(schedule)}(
            sensing,
            reconstructor,
            schedule,
            block_rows,
        )
    end
end

"""Progressive state and unpublished reconstruction for one row-frame owner."""
mutable struct ShackHartmannRowReconstructionWorkspace
    detector::Matrix{Float32}
    slopes::Matrix{Float32}
    flux::Vector{Float32}
    validity::Vector{Bool}
    ready::Vector{Bool}
    candidate_reconstructed::Vector{Float32}
    frame::RowBlockFrame
    schedule_cursor::Int
    processed_subapertures::Int
    terminal_candidate::Bool
end

function ShackHartmannRowReconstructionWorkspace(
    plan::ShackHartmannRowReconstructionPlan,
)
    image_rows, image_columns = plan.sensing.regions.image_size
    subaperture_count = length(plan.sensing.regions.origins)
    return ShackHartmannRowReconstructionWorkspace(
        zeros(Float32, image_rows, image_columns),
        zeros(Float32, subaperture_count, 2),
        zeros(Float32, subaperture_count),
        fill(false, subaperture_count),
        fill(false, subaperture_count),
        zeros(Float32, size(plan.reconstructor.matrix, 1)),
        RowBlockFrame(),
        0,
        0,
        false,
    )
end

function _clear_shack_hartmann_row_partial!(
    workspace::ShackHartmannRowReconstructionWorkspace,
)
    fill!(workspace.ready, false)
    fill!(workspace.candidate_reconstructed, 0.0f0)
    workspace.schedule_cursor = 0
    workspace.processed_subapertures = 0
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_shack_hartmann_row!(
    workspace::ShackHartmannRowReconstructionWorkspace,
)
    _abandon_row_block_frame!(workspace.frame)
    return _clear_shack_hartmann_row_partial!(workspace)
end

function commit!(workspace::ShackHartmannRowReconstructionWorkspace)
    if workspace.terminal_candidate
        _complete_row_block_frame!(workspace.frame)
        _clear_shack_hartmann_row_partial!(workspace)
    end
    return workspace
end

discard!(workspace::ShackHartmannRowReconstructionWorkspace) =
    _abandon_shack_hartmann_row!(workspace)

function reset!(workspace::ShackHartmannRowReconstructionWorkspace)
    fill!(workspace.detector, 0.0f0)
    fill!(workspace.slopes, 0.0f0)
    fill!(workspace.flux, 0.0f0)
    fill!(workspace.validity, false)
    _reset_row_block_frame!(workspace.frame)
    return _clear_shack_hartmann_row_partial!(workspace)
end

_plan_adopted!(workspace::ShackHartmannRowReconstructionWorkspace) =
    _abandon_shack_hartmann_row!(workspace)

"""Expected rejection from row-block Shack–Hartmann reconstruction."""
@enum ShackHartmannRowReconstructionFailure::UInt8 begin
    ShackHartmannRowMetadataMissing = 1
    ShackHartmannRowOffsetOverflow = 2
    ShackHartmannRowNonFiniteReconstruction = 3
end

@doc "A Shack-Hartmann row reconstruction block lacks required sample metadata." ShackHartmannRowMetadataMissing
@doc "A Shack-Hartmann row reconstruction offset cannot be represented safely." ShackHartmannRowOffsetOverflow
@doc "Row-based Shack-Hartmann reconstruction produced a non-finite coordinate." ShackHartmannRowNonFiniteReconstruction

function _replace_shack_hartmann_row_origins(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        _replace_shack_hartmann_origins(plan.sensing, values),
        plan.reconstructor,
        plan.block_rows,
    )
end

function _replace_shack_hartmann_row_coordinates(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        _replace_shack_hartmann_coordinates(plan.sensing, values),
        plan.reconstructor,
        plan.block_rows,
    )
end

function _replace_shack_hartmann_row_reference_slopes(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        _replace_shack_hartmann_reference_slopes(plan.sensing, values),
        plan.reconstructor,
        plan.block_rows,
    )
end

function _replace_shack_hartmann_row_thresholds(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        _replace_shack_hartmann_thresholds(plan.sensing, values),
        plan.reconstructor,
        plan.block_rows,
    )
end

function _replace_shack_hartmann_row_active(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        _replace_shack_hartmann_active(plan.sensing, values),
        plan.reconstructor,
        plan.block_rows,
    )
end

function _replace_shack_hartmann_row_reconstructor(
    plan::ShackHartmannRowReconstructionPlan,
    values,
)
    return ShackHartmannRowReconstructionPlan(
        plan.sensing,
        DenseReconstructorPlan(values),
        plan.block_rows,
    )
end

function _validate_shack_hartmann_row_storage(
    reconstructed::AbstractVector{Float32},
    workspace::ShackHartmannRowReconstructionWorkspace,
    plan::ShackHartmannRowReconstructionPlan,
    row_block::AbstractMatrix{Float32},
)
    image_rows, image_columns = plan.sensing.regions.image_size
    size(row_block) == (plan.block_rows, image_columns) || throw(
        DimensionMismatch("row block has an incompatible detector shape"),
    )
    length(reconstructed) == size(plan.reconstructor.matrix, 1) || throw(
        DimensionMismatch("reconstructed output has an incompatible extent"),
    )
    size(workspace.detector) == (image_rows, image_columns) || throw(
        DimensionMismatch("row reconstruction Workspace has an invalid detector"),
    )
    Base.mightalias(reconstructed, row_block) && throw(
        ArgumentError("reconstructed output must not alias the detector row block"),
    )
    Base.mightalias(reconstructed, plan.reconstructor.matrix) && throw(
        ArgumentError("reconstructed output must not alias the prepared matrix"),
    )
    _shack_hartmann_aliases_plan(reconstructed, plan.sensing) && throw(
        ArgumentError("reconstructed output must not alias sensing calibration"),
    )
    _shack_hartmann_row_aliases_workspace(reconstructed, workspace) && throw(
        ArgumentError("reconstructed output must not alias row Workspace storage"),
    )
    _shack_hartmann_row_aliases_workspace(row_block, workspace) && throw(
        ArgumentError("row input must not alias row Workspace storage"),
    )
    return nothing
end

@inline function _shack_hartmann_row_aliases_workspace(
    array,
    workspace::ShackHartmannRowReconstructionWorkspace,
)
    return Base.mightalias(array, workspace.detector) ||
        Base.mightalias(array, workspace.slopes) ||
        Base.mightalias(array, workspace.flux) ||
        Base.mightalias(array, workspace.validity) ||
        Base.mightalias(array, workspace.ready) ||
        Base.mightalias(array, workspace.candidate_reconstructed)
end

function _ready_shack_hartmann_prefix!(
    workspace::ShackHartmannRowReconstructionWorkspace,
    plan::ShackHartmannRowReconstructionPlan,
    completed_rows::Int,
)
    origins = plan.sensing.regions.origins
    region_rows = plan.sensing.regions.region_size[1]
    while workspace.schedule_cursor < length(plan.completion_schedule)
        cursor = workspace.schedule_cursor + 1
        subaperture = plan.completion_schedule[cursor]
        completion_row = origins[subaperture][1] + region_rows
        completion_row <= completed_rows || break
        workspace.ready[subaperture] = true
        workspace.schedule_cursor = cursor
    end

    first_subaperture = workspace.processed_subapertures + 1
    last_subaperture = workspace.processed_subapertures
    while last_subaperture < length(workspace.ready) &&
          workspace.ready[last_subaperture + 1]
        last_subaperture += 1
    end
    return first_subaperture:last_subaperture
end

"""
    _accumulate_shack_hartmann_reconstruction_rows!(
        candidate, matrix, slopes, subapertures, output_rows,
    )

Accumulate pair-interleaved slopes for `subapertures` into
`candidate[output_rows]`. `output_rows` must be a valid, disjoint partition
when called concurrently.
"""
@inline function _accumulate_shack_hartmann_reconstruction_rows!(
    candidate::AbstractVector{Float32},
    matrix::AbstractMatrix{Float32},
    slopes::AbstractMatrix{Float32},
    subapertures::AbstractUnitRange{Int},
    output_rows,
)
    @inbounds for subaperture in subapertures
        x_column = 2 * subaperture - 1
        x_slope = slopes[subaperture, 1]
        @inbounds @simd for output in output_rows
            candidate[output] = fma(matrix[output, x_column], x_slope, candidate[output])
        end
        y_column = x_column + 1
        y_slope = slopes[subaperture, 2]
        @inbounds @simd for output in output_rows
            candidate[output] = fma(matrix[output, y_column], y_slope, candidate[output])
        end
    end
    return candidate
end

function _accumulate_shack_hartmann_reconstruction!(
    workspace::ShackHartmannRowReconstructionWorkspace,
    plan::ShackHartmannRowReconstructionPlan,
    subapertures::AbstractUnitRange{Int},
)
    matrix = plan.reconstructor.matrix
    candidate = workspace.candidate_reconstructed
    _accumulate_shack_hartmann_reconstruction_rows!(
        candidate,
        matrix,
        workspace.slopes,
        subapertures,
        eachindex(candidate),
    )
    for value in candidate
        isfinite(value) || return ShackHartmannRowNonFiniteReconstruction
    end
    return nothing
end

function _process_shack_hartmann_row!(
    reconstructed::AbstractVector{Float32},
    workspace::ShackHartmannRowReconstructionWorkspace,
    plan::ShackHartmannRowReconstructionPlan,
    row_block::AbstractMatrix{Float32},
    sample::SampleMetadata,
)
    _validate_shack_hartmann_row_storage(
        reconstructed,
        workspace,
        plan,
        row_block,
    )
    UInt64(sample.offset) <= UInt64(typemax(Int)) ||
        return ShackHartmannRowOffsetOverflow
    first_row = Int(sample.offset)
    image_rows, image_columns = plan.sensing.regions.image_size
    accepted, completed_rows = _accept_row_block!(
        workspace.frame,
        sample,
        first_row,
        plan.block_rows,
        image_rows,
    )
    if !accepted
        _clear_shack_hartmann_row_partial!(workspace)
        return AlgorithmProcessDeferred()
    end
    iszero(first_row) && _clear_shack_hartmann_row_partial!(workspace)

    for column in 1:image_columns, row in 1:plan.block_rows
        workspace.detector[first_row + row, column] = row_block[row, column]
    end
    ready = _ready_shack_hartmann_prefix!(workspace, plan, completed_rows)
    if !isempty(ready)
        _measure_shack_hartmann_range!(
            workspace.slopes,
            workspace.flux,
            workspace.validity,
            plan.sensing,
            workspace.detector,
            ready,
        )
        status = _accumulate_shack_hartmann_reconstruction!(
            workspace,
            plan,
            ready,
        )
        if !isnothing(status)
            _abandon_shack_hartmann_row!(workspace)
            return status
        end
        workspace.processed_subapertures = last(ready)
    end

    sample.terminal || return AlgorithmProcessDeferred()
    if workspace.processed_subapertures != length(plan.sensing.regions.origins)
        _abandon_shack_hartmann_row!(workspace)
        return AlgorithmProcessDeferred()
    end
    copyto!(reconstructed, workspace.candidate_reconstructed)
    workspace.terminal_candidate = true
    return AlgorithmProcessComplete(
        SampleMetadata(
            sample;
            offset=0,
            terminal=true,
            discontinuity=workspace.frame.current_discontinuity,
        ),
    )
end

"""
Measure Shack-Hartmann centroids across successive row blocks and apply the
dense reconstructor when the complete image is available. The reconstructed
vector is in the output schema's units.
"""
@algorithm ShackHartmannRowReconstructorF32 begin
    label = "shwfs-row-reconstructor-f32"
    failure = ShackHartmannRowReconstructionFailure
    operation = ShackHartmannRowReconstructionF32Operation()

    """
    Set image/row-block/subaperture geometry, reconstructed count, optional origins,
    coordinate scale, thresholds, reference slopes and active mask, optional
    initial reconstructor, and versioned output schema.
    """
    @configuration ShackHartmannRowReconstructorF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        row_block_rows::Int => graph_rebuild
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
            :row_block,
            schema_element_type(CALIBRATED_PIXEL_ROW_BLOCK_V1),
            (configuration.row_block_rows, configuration.image_size[2]),
            CALIBRATED_PIXEL_ROW_BLOCK_V1;
            name="row-block",
            sample=:row_sample,
            rate=(div(configuration.image_size[1], configuration.row_block_rows), 1),
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
            :subaperture_origins,
            schema_element_type(REGION_ORIGINS_V1),
            (configuration.subaperture_count, 2),
            REGION_ORIGINS_V1;
            name="subaperture-origins",
            replace=(plan, values) ->
                _replace_shack_hartmann_row_origins(plan, values),
        ),
        parameter(
            :coordinates,
            schema_element_type(SHACK_HARTMANN_COORDINATES_V1),
            (
                Base.Checked.checked_mul(
                    configuration.subaperture_size[1],
                    configuration.subaperture_size[2],
                ),
                2,
            ),
            SHACK_HARTMANN_COORDINATES_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_row_coordinates(plan, values),
        ),
        parameter(
            :reference_slopes,
            schema_element_type(SHACK_HARTMANN_REFERENCE_SLOPES_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_REFERENCE_SLOPES_V1;
            name="reference-slopes",
            replace=(plan, values) ->
                _replace_shack_hartmann_row_reference_slopes(plan, values),
        ),
        parameter(
            :thresholds,
            schema_element_type(SHACK_HARTMANN_THRESHOLDS_V1),
            (configuration.subaperture_count, 2),
            SHACK_HARTMANN_THRESHOLDS_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_row_thresholds(plan, values),
        ),
        parameter(
            :active,
            schema_element_type(SHACK_HARTMANN_ACTIVE_V1),
            (configuration.subaperture_count,),
            SHACK_HARTMANN_ACTIVE_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_row_active(plan, values),
        ),
        parameter(
            :reconstructor,
            schema_element_type(SHWFS_RECONSTRUCTOR_V1),
            (
                configuration.reconstructed_count,
                Base.Checked.checked_mul(configuration.subaperture_count, 2),
            ),
            SHWFS_RECONSTRUCTOR_V1;
            replace=(plan, values) ->
                _replace_shack_hartmann_row_reconstructor(plan, values),
        ),
    )

    prepare(configuration) = begin
        origins = _initial_shack_hartmann_origins(
            configuration.image_size,
            configuration.subaperture_size,
            configuration.subaperture_count,
            configuration.initial_subaperture_origins,
        )
        sensing = ShackHartmannImagePlan(
            configuration.image_size,
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
        plan = ShackHartmannRowReconstructionPlan(
            sensing,
            reconstructor,
            configuration.row_block_rows,
        )
        (plan=plan, workspace=ShackHartmannRowReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(row_sample) && return ShackHartmannRowMetadataMissing
        _process_shack_hartmann_row!(
            reconstructed,
            workspace,
            plan,
            row_block,
            row_sample,
        )
    end
end
