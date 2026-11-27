"""
Progressive state for a dense reconstruction assembled from measurement blocks.

`candidate_reconstructed` is unpublished work for the current logical sample.
`seen` ensures that each zero-based reconstructor column contributes exactly
once before terminal publication.
"""
mutable struct IncrementalDenseReconstructionWorkspace{
    T<:Union{Float32,Float64},
    Values<:AbstractVector{T},
    Seen<:AbstractVector{Bool},
}
    candidate_reconstructed::Values
    seen::Seen
    seen_count::Int
    active::Bool
    sequence::UInt64
    metadata::SampleMetadata
    last_offset::UInt32
    pending_discontinuity::Bool
    current_discontinuity::Bool
    terminal_candidate::Bool
end

function _prepared_incremental_dense_reconstruction_workspace(
    candidate_reconstructed::Values,
    seen::Seen,
) where {
    T<:Union{Float32,Float64},
    Values<:AbstractVector{T},
    Seen<:AbstractVector{Bool},
}
    return IncrementalDenseReconstructionWorkspace{T,Values,Seen}(
        candidate_reconstructed,
        seen,
        0,
        false,
        0,
        _EMPTY_SAMPLE_METADATA,
        0,
        false,
        false,
        false,
    )
end

function IncrementalDenseReconstructionWorkspace(plan::DenseReconstructorPlan{T}) where {T}
    return _prepared_incremental_dense_reconstruction_workspace(
        zeros(T, size(plan.matrix, 1)),
        fill(false, size(plan.matrix, 2)),
    )
end

function _clear_incremental_dense_reconstruction!(
    workspace::IncrementalDenseReconstructionWorkspace,
)
    fill!(
        workspace.candidate_reconstructed,
        zero(eltype(workspace.candidate_reconstructed)),
    )
    fill!(workspace.seen, false)
    workspace.seen_count = 0
    workspace.active = false
    workspace.last_offset = 0
    workspace.current_discontinuity = false
    workspace.terminal_candidate = false
    return workspace
end

function _abandon_incremental_dense_reconstruction!(
    workspace::IncrementalDenseReconstructionWorkspace,
)
    workspace.pending_discontinuity = true
    return _clear_incremental_dense_reconstruction!(workspace)
end

function commit!(workspace::IncrementalDenseReconstructionWorkspace)
    if workspace.terminal_candidate
        _clear_incremental_dense_reconstruction!(workspace)
        workspace.pending_discontinuity = false
    end
    return workspace
end

discard!(workspace::IncrementalDenseReconstructionWorkspace) =
    _abandon_incremental_dense_reconstruction!(workspace)

function reset!(workspace::IncrementalDenseReconstructionWorkspace)
    _clear_incremental_dense_reconstruction!(workspace)
    workspace.sequence = 0
    workspace.pending_discontinuity = false
    return workspace
end

_plan_adopted!(workspace::IncrementalDenseReconstructionWorkspace) =
    _abandon_incremental_dense_reconstruction!(workspace)

"""Expected rejection from incremental dense reconstruction."""
@enum IncrementalDenseReconstructionFailure::UInt8 begin
    IncrementalDenseReconstructionMetadataMissing = 1
    IncrementalDenseReconstructionCountOutOfRange = 2
    IncrementalDenseReconstructionColumnOutOfRange = 3
    IncrementalDenseReconstructionDuplicateColumn = 4
    IncrementalDenseReconstructionNonFiniteInput = 5
    IncrementalDenseReconstructionNonFiniteOutput = 6
    IncrementalDenseReconstructionIncompleteSample = 7
end

@doc "An incremental measurement block lacks required sample metadata." IncrementalDenseReconstructionMetadataMissing
@doc "The block's valid-entry count exceeds its configured capacity." IncrementalDenseReconstructionCountOutOfRange
@doc "A zero-based measurement column lies outside the reconstructor's input extent." IncrementalDenseReconstructionColumnOutOfRange
@doc "The same measurement column was supplied twice in one complete sample." IncrementalDenseReconstructionDuplicateColumn
@doc "An incremental measurement value is non-finite." IncrementalDenseReconstructionNonFiniteInput
@doc "Incremental matrix accumulation produced a non-finite reconstruction coordinate." IncrementalDenseReconstructionNonFiniteOutput
@doc "The terminal sample arrived before every required measurement column was covered." IncrementalDenseReconstructionIncompleteSample

function _start_incremental_dense_reconstruction!(
    workspace::IncrementalDenseReconstructionWorkspace,
    sample::SampleMetadata,
)
    superseded = workspace.active
    superseded && _abandon_incremental_dense_reconstruction!(workspace)
    _clear_incremental_dense_reconstruction!(workspace)
    workspace.active = true
    workspace.sequence = sample.sequence
    workspace.metadata=sample
    workspace.current_discontinuity =
        workspace.pending_discontinuity || sample.discontinuity || superseded
    workspace.pending_discontinuity = false
    return workspace
end

function _accept_incremental_dense_reconstruction_block!(
    workspace::IncrementalDenseReconstructionWorkspace,
    sample::SampleMetadata,
)
    if sample.corrupted
        _abandon_incremental_dense_reconstruction!(workspace)
        return false
    end
    if iszero(sample.offset)
        _start_incremental_dense_reconstruction!(workspace, sample)
    elseif !workspace.active ||
           !_same_row_block_metadata(workspace.metadata,sample) ||
           sample.discontinuity ||
           sample.offset <= workspace.last_offset
        _abandon_incremental_dense_reconstruction!(workspace)
        return false
    end
    workspace.last_offset = sample.offset
    return true
end

function _incremental_dense_reconstruction_aliases_workspace(
    array,
    workspace::IncrementalDenseReconstructionWorkspace,
)
    return Base.mightalias(array, workspace.candidate_reconstructed) ||
        Base.mightalias(array, workspace.seen)
end

function _validate_incremental_dense_reconstruction_storage(
    reconstructed::AbstractVector{T},
    measurements::AbstractVector{T},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    workspace::IncrementalDenseReconstructionWorkspace{T},
    plan::DenseReconstructorPlan{T},
) where {T}
    Base.require_one_based_indexing(reconstructed, measurements, columns, count)
    length(reconstructed) == size(plan.matrix, 1) || throw(
        DimensionMismatch("reconstructed output has an incompatible extent"),
    )
    length(measurements) == length(columns) || throw(
        DimensionMismatch("measurement and column blocks must have equal capacity"),
    )
    length(count) == 1 || throw(
        DimensionMismatch("measurement count must contain one value"),
    )
    length(workspace.candidate_reconstructed) == size(plan.matrix, 1) || throw(
        DimensionMismatch("incremental Workspace has an invalid output extent"),
    )
    length(workspace.seen) == size(plan.matrix, 2) || throw(
        DimensionMismatch("incremental Workspace has an invalid input extent"),
    )
    Base.mightalias(reconstructed, measurements) && throw(
        ArgumentError("reconstructed output must not alias measurements"),
    )
    Base.mightalias(reconstructed, columns) && throw(
        ArgumentError("reconstructed output must not alias columns"),
    )
    Base.mightalias(reconstructed, count) && throw(
        ArgumentError("reconstructed output must not alias the measurement count"),
    )
    Base.mightalias(reconstructed, plan.matrix) && throw(
        ArgumentError("reconstructed output must not alias the Plan"),
    )
    _incremental_dense_reconstruction_aliases_workspace(
        reconstructed,
        workspace,
    ) && throw(ArgumentError("reconstructed output must not alias Workspace storage"))
    Base.mightalias(measurements, plan.matrix) && throw(
        ArgumentError("measurements must not alias the Plan"),
    )
    _incremental_dense_reconstruction_aliases_workspace(
        measurements,
        workspace,
    ) && throw(ArgumentError("measurements must not alias Workspace storage"))
    return nothing
end

"""
    _accumulate_incremental_dense_reconstruction_rows!(
        candidate, matrix, column, measurement, output_rows, row_offset=0,
    )

Add one already-validated reconstructor column to `candidate[output_rows]`.
The corresponding matrix rows are `output_rows .+ row_offset`. The candidate
rows must be disjoint when called concurrently.
"""
@inline function _accumulate_incremental_dense_reconstruction_rows!(
    candidate::AbstractVector{T},
    matrix::AbstractMatrix{T},
    column::Int,
    measurement::T,
    output_rows,
    row_offset::Int=0,
) where {T<:Union{Float32,Float64}}
    @inbounds @simd for output in output_rows
        candidate[output] = fma(matrix[output + row_offset, column], measurement, candidate[output])
    end
    return candidate
end

function _accumulate_incremental_dense_reconstruction_columns!(
    candidate::AbstractVector{T},
    matrix::AbstractMatrix{T},
    measurements::AbstractVector{T},
    columns::AbstractVector{UInt32},
    items::UnitRange{Int},
    row_offset::Int=0,
) where {T<:Union{Float32,Float64}}
    output_rows = eachindex(candidate)
    next_item = _accumulate_incremental_dense_column_groups!(
        candidate, matrix, measurements, columns, items, output_rows, row_offset,
    )
    @inbounds for item in next_item:last(items)
        _accumulate_incremental_dense_reconstruction_rows!(
            candidate, matrix, Int(columns[item]) + 1, measurements[item],
            output_rows, row_offset,
        )
    end
    return nothing
end

# Other matrix storage uses the single-column pass for every item.
_accumulate_incremental_dense_column_groups!(
    candidate, matrix::AbstractMatrix, measurements, columns, items, output_rows, row_offset,
) = first(items)

function _accumulate_incremental_dense_column_groups!(
    candidate, matrix::Matrix, measurements, columns, items, output_rows, row_offset,
)
    index = first(items)
    # Callers validate these items before accumulation. Keep each row's
    # accumulator across four columns in their original order.
    @inbounds while index <= last(items) - 3
        c1 = Int(columns[index]) + 1
        c2 = Int(columns[index + 1]) + 1
        c3 = Int(columns[index + 2]) + 1
        c4 = Int(columns[index + 3]) + 1
        m1 = measurements[index]
        m2 = measurements[index + 1]
        m3 = measurements[index + 2]
        m4 = measurements[index + 3]
        @simd for output in output_rows
            row = output + row_offset
            value = fma(matrix[row, c1], m1, candidate[output])
            value = fma(matrix[row, c2], m2, value)
            value = fma(matrix[row, c3], m3, value)
            candidate[output] = fma(matrix[row, c4], m4, value)
        end
        index += 4
    end
    return index
end

const _IncrementalDenseBlockVector = Union{Vector,SubArray{<:Any,1,<:Array,<:Tuple,true}}

_can_batch_incremental_dense_columns(matrix, measurements, columns, workspace, block_count) =
    false

function _can_batch_incremental_dense_columns(
    matrix::Matrix, measurements::_IncrementalDenseBlockVector,
    columns::_IncrementalDenseBlockVector, workspace, block_count,
)
    # The batched pass rereads entries after updating the seen mask.
    return block_count >= 4 &&
        !_incremental_dense_reconstruction_aliases_workspace(columns, workspace)
end

function _accumulate_incremental_dense_reconstruction!(
    workspace::IncrementalDenseReconstructionWorkspace{T},
    plan::DenseReconstructorPlan{T},
    measurements::AbstractVector{T},
    columns::AbstractVector{UInt32},
    block_count::Int,
) where {T}
    matrix = plan.matrix
    candidate = workspace.candidate_reconstructed
    measurement_count = size(matrix, 2)
    batch = _can_batch_incremental_dense_columns(
        matrix, measurements, columns, workspace, block_count,
    )
    output_rows = eachindex(candidate)
    @inbounds for index in 1:block_count
        column_offset = columns[index]
        UInt64(column_offset) < UInt64(measurement_count) ||
            return IncrementalDenseReconstructionColumnOutOfRange
        column = Int(column_offset) + 1
        workspace.seen[column] &&
            return IncrementalDenseReconstructionDuplicateColumn
        measurement = measurements[index]
        isfinite(measurement) ||
            return IncrementalDenseReconstructionNonFiniteInput
        if !batch
            _accumulate_incremental_dense_reconstruction_rows!(
                candidate, matrix, column, measurement, output_rows,
            )
        end
        workspace.seen[column] = true
        workspace.seen_count += 1
    end
    if batch
        _accumulate_incremental_dense_reconstruction_columns!(
            candidate, matrix, measurements, columns, 1:block_count,
        )
    end
    for value in candidate
        isfinite(value) || return IncrementalDenseReconstructionNonFiniteOutput
    end
    return nothing
end

function _process_incremental_dense_reconstruction!(
    reconstructed::AbstractVector{T},
    workspace::IncrementalDenseReconstructionWorkspace{T},
    plan::DenseReconstructorPlan{T},
    measurements::AbstractVector{T},
    columns::AbstractVector{UInt32},
    count::AbstractVector{UInt32},
    sample::SampleMetadata,
) where {T}
    _validate_incremental_dense_reconstruction_storage(
        reconstructed,
        measurements,
        columns,
        count,
        workspace,
        plan,
    )
    UInt64(count[1]) <= UInt64(length(measurements)) ||
        return IncrementalDenseReconstructionCountOutOfRange
    accepted = _accept_incremental_dense_reconstruction_block!(workspace, sample)
    accepted || return AlgorithmProcessDeferred()

    failure = _accumulate_incremental_dense_reconstruction!(
        workspace,
        plan,
        measurements,
        columns,
        Int(count[1]),
    )
    if !isnothing(failure)
        _abandon_incremental_dense_reconstruction!(workspace)
        return failure
    end
    sample.terminal || return AlgorithmProcessDeferred()
    if workspace.seen_count != length(workspace.seen)
        _abandon_incremental_dense_reconstruction!(workspace)
        return IncrementalDenseReconstructionIncompleteSample
    end

    copyto!(reconstructed, workspace.candidate_reconstructed)
    workspace.terminal_candidate = true
    metadata = SampleMetadata(
        sample;
        offset=0,
        terminal=true,
        discontinuity=workspace.current_discontinuity,
    )
    return AlgorithmProcessComplete(metadata)
end

"""
Incrementally multiply sparse blocks of measured values by a dense
reconstructor, accumulating a reconstructed vector until a complete sample.
Column indices are zero-based measurement coordinates; the `count` port gives
the number of valid entries in the current block.
"""
@algorithm IncrementalDenseReconstructorF32 begin
    label = "incremental-dense-reconstructor-f32"
    failure = IncrementalDenseReconstructionFailure

    """
    Set total measurement count, per-block capacity, blocks per sample,
    reconstructed count, and versioned port schemas. An optional initial dense
    reconstructor has shape `(reconstructed_count, measurement_count)` and can be
    replaced through its parameter port.
    Set `coefficient_storage = "fp16"` for power-of-two row-scaled Float16
    coefficients with Float32 measurements, accumulation and output.
    """
    @configuration IncrementalDenseReconstructorF32Configuration begin
        measurement_count::Int => graph_rebuild
        measurement_block_capacity::Int => graph_rebuild
        blocks_per_sample::Int => graph_rebuild
        reconstructed_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        measurements_schema::String => graph_rebuild
        column_indices_schema::String => graph_rebuild
        measurement_count_schema::String => graph_rebuild
        reconstructed_schema::String => graph_rebuild
        reconstructor_schema::String => graph_rebuild
        (coefficient_storage::String = "fp32") => graph_rebuild
    end

    ports(configuration) = (
        input(
            :measurements,
            Float32,
            (configuration.measurement_block_capacity,),
            configuration.measurements_schema;
            sample=:measurement_sample,
            rate=(configuration.blocks_per_sample, 1),
        ),
        input(
            :columns,
            UInt32,
            (configuration.measurement_block_capacity,),
            configuration.column_indices_schema;
            rate=(configuration.blocks_per_sample, 1),
        ),
        input(
            :count,
            UInt32,
            (1,),
            configuration.measurement_count_schema;
            rate=(configuration.blocks_per_sample, 1),
        ),
        output(
            :reconstructed,
            Float32,
            (configuration.reconstructed_count,),
            configuration.reconstructed_schema;
            metadata_from=:measurements,
            publication=:conditional,
        ),
        parameter(
            :reconstructor,
            Float32,
            (configuration.reconstructed_count, configuration.measurement_count),
            configuration.reconstructor_schema;
            replace=_replace_dense_reconstructor_storage,
        ),
    )

    prepare(configuration) = begin
        configuration.measurement_block_capacity > 0 || throw(
            ArgumentError("measurement_block_capacity must be positive"),
        )
        configuration.blocks_per_sample > 0 || throw(
            ArgumentError("blocks_per_sample must be positive"),
        )
        plan = DenseReconstructorPlan(
            Float32,
            configuration.reconstructed_count,
            configuration.measurement_count,
            configuration.initial_reconstructor,
        )
        plan = _configure_dense_reconstructor_storage(plan, configuration.coefficient_storage)
        (plan=plan, workspace=IncrementalDenseReconstructionWorkspace(plan))
    end

    process!(plan, workspace) = begin
        isnothing(measurement_sample) &&
            return IncrementalDenseReconstructionMetadataMissing
        _process_incremental_dense_reconstruction!(
            reconstructed,
            workspace,
            plan,
            measurements,
            columns,
            count,
            measurement_sample,
        )
    end
end
