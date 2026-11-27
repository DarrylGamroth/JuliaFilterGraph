struct IncrementalDeviceWorkspace{W,S}
    state::W
    status::S
    completion::Vector{Int64}
end

FGA.commit!(workspace::IncrementalDeviceWorkspace) = (FGA.commit!(workspace.state); workspace)
FGA.discard!(workspace::IncrementalDeviceWorkspace) = (FGA.discard!(workspace.state); workspace)
FGA.reset!(workspace::IncrementalDeviceWorkspace) = (FGA.reset!(workspace.state); workspace)
FGA._plan_adopted!(workspace::IncrementalDeviceWorkspace) = (FGA._plan_adopted!(workspace.state); workspace)

function _progressive_plan_storage(::Type{IncrementalDenseReconstructorF32}, device, plan)
    matrix = plan.matrix
    return _capture_transposed_coefficients(device, matrix, axes(matrix, 2))
end

function _progressive_workspace(::Type{IncrementalDenseReconstructorF32}, plan)
    rows, columns = size(plan.source.matrix)
    candidate = _capture_zeros(plan.device, Float32, rows)
    seen = _capture_zeros(plan.device, Bool, columns)
    state = FGA._prepared_incremental_dense_reconstruction_workspace(candidate, seen)
    return IncrementalDeviceWorkspace(state, _capture_zeros(plan.device, Int64, 2), zeros(Int64, 2))
end

function _replace_prepared_parameter(replacement, plan::ProgressiveDevicePlan{<:DenseReconstructorPlan}, values)
    return _replace_progressive_parameter(IncrementalDenseReconstructorF32, replacement, plan, values)
end

@kernel function _validate_incremental_block!(status, seen, measurements, columns, count, accepted)
    @inbounds begin
        block_count = Int(count[1])
        failure = Int64(0)
        status[2] = 0
        if block_count > length(measurements)
            failure = Int64(FGA.IncrementalDenseReconstructionCountOutOfRange)
        elseif accepted
            # Check entries in the declared order, so both duplicates and
            # multiple invalid entries preserve the CPU's failure priority.
            for entry in 1:block_count
                column = Int(columns[entry]) + 1
                if column > length(seen)
                    failure = Int64(FGA.IncrementalDenseReconstructionColumnOutOfRange)
                    break
                elseif seen[column]
                    failure = Int64(FGA.IncrementalDenseReconstructionDuplicateColumn)
                    break
                elseif !isfinite(measurements[entry])
                    failure = Int64(FGA.IncrementalDenseReconstructionNonFiniteInput)
                    break
                end
                seen[column] = true
            end
            status[2] = block_count
        end
        status[1] = failure
    end
end

@kernel function _accumulate_incremental_block!(candidate, matrix, measurements, columns, count, status)
    row = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            result = candidate[row]
            for entry in 1:Int(count[1])
                column = Int(columns[entry]) + 1
                result = fma(_projection_coefficient(matrix, column, row), measurements[entry], result)
            end
            candidate[row] = result
        end
    end
end

function _process_admitted!(output::AbstractArray,
    prepared::PreparedAlgorithm{IncrementalDenseReconstructorF32,<:ProgressiveDevicePlan},
    measurements::AbstractArray, columns::AbstractArray, count::AbstractArray,
    metadata::FGA.AdmittedSampleMetadata, ::FGA.AdmittedSampleMetadata, ::FGA.AdmittedSampleMetadata)
    plan, workspace = _progressive_admitted(prepared, (output,), (measurements, columns, count))
    sample = FGA._sample_metadata(metadata)
    isnothing(sample) && return FGA.IncrementalDenseReconstructionMetadataMissing
    state = workspace.state
    FGA._validate_incremental_dense_reconstruction_storage(
        output, measurements, columns, count, state, plan.source,
    )
    accepted = FGA._accept_incremental_dense_reconstruction_block!(state, sample)
    backend = KernelAbstractions.get_backend(output)
    _validate_incremental_block!(backend, 1)(
        workspace.status, state.seen, measurements, columns, count, accepted; ndrange=1,
    )
    if accepted
        _accumulate_incremental_block!(backend, 256)(
            state.candidate_reconstructed, plan.storage, measurements, columns, count, workspace.status;
            ndrange=length(output),
        )
        _validate_projection_values_portable!(backend, VALIDATION_WORKGROUP_SIZE)(
            workspace.status, state.candidate_reconstructed,
            Int64(FGA.IncrementalDenseReconstructionNonFiniteOutput); ndrange=VALIDATION_WORKGROUP_SIZE,
        )
    end
    failure = _progressive_status!(workspace.completion, workspace.status, IncrementalDenseReconstructionFailure)
    isnothing(failure) || return failure
    accepted || return AlgorithmProcessDeferred()
    state.seen_count += Int(workspace.completion[2])
    sample.terminal || return AlgorithmProcessDeferred()
    state.seen_count == length(state.seen) || return FGA.IncrementalDenseReconstructionIncompleteSample

    copyto!(output, state.candidate_reconstructed)
    KernelAbstractions.synchronize(backend)
    state.terminal_candidate = true
    return _progressive_complete(sample, state.current_discontinuity)
end
