import FilterGraphAlgorithms: SccCorrectionF32, SccCorrectionPlan, SccCorrectionFailure,
    SccNonFiniteInput, SccNonFiniteCorrection, SccCommandOutOfBounds,
    SampleAcquisitionIdentity, ChoppedImageKind, ChoppedImageAbsent, ChoppedImageFringed,
    ChoppedImageMixed, AdmittedSampleMetadata, _sample_metadata, _scc_exposure,
    prepare_algorithm, _process_admitted!, commit!, discard!

mutable struct SccDeviceWorkspace{D,S,C,W,H}
    device::D
    committed::S
    candidate::S
    calibration::C
    scratch::W
    completion::H
    committed_identity::Union{Nothing,SampleAcquisitionIdentity}
    candidate_identity::Union{Nothing,SampleAcquisitionIdentity}
    committed_kind::ChoppedImageKind
    candidate_kind::ChoppedImageKind
    candidate_ready::Bool
end

function commit!(ws::SccDeviceWorkspace)
    if ws.candidate_ready
        ws.committed, ws.candidate = ws.candidate, ws.committed
        ws.committed_identity = ws.candidate_identity
        ws.committed_kind = ws.candidate_kind
        ws.candidate_ready = false
    end
    return ws
end

function discard!(ws::SccDeviceWorkspace)
    ws.candidate_ready = false
    return ws
end

function reset!(ws::SccDeviceWorkspace)
    _with_eager_device(ws.device) do
        for state in (ws.committed, ws.candidate)
            fill!(state.image, 0f0)
            fill!(state.correction, 0f0)
            copyto!(state.command, ws.calibration.bias)
            copyto!(state.exposure_command, ws.calibration.bias)
        end
    end
    ws.committed_identity = ws.candidate_identity = nothing
    ws.committed_kind = ws.candidate_kind = ChoppedImageAbsent
    ws.candidate_ready = false
    return ws
end

function prepare_algorithm(prepared::PreparedAlgorithm{SccCorrectionF32,SccCorrectionPlan},
        device::DeviceType)
    return _with_eager_device(device) do
        plan = prepared.plan
        cal = plan.calibration
        status = _capture_zeros(device, Int32, 1)
        storage = (
            reference=_capture_array(device, cal.residual.reference_image),
            selected=_capture_array(device, cal.residual.selected_indices),
            reconstructor=_capture_array(device, cal.reconstructor),
            fringed_response=_capture_array(device, cal.fringed_response),
            unfringed_response=_capture_array(device, cal.unfringed_response),
            bias=_capture_array(device, cal.bias_command),
            lower=_capture_array(device, cal.lower_limits),
            upper=_capture_array(device, cal.upper_limits),
        )
        states = ntuple(2) do _
            (image=_capture_zeros(device, Float32, cal.frame_shape...),
                command=copy(storage.bias), exposure_command=copy(storage.bias),
                correction=similar(storage.bias))
        end
        scratch = (
            status=status,
            difference=_capture_zeros(device, Float32, cal.frame_shape...),
            residual=_capture_zeros(device, Float32, length(storage.selected)),
            reconstructed=similar(storage.bias), command_difference=similar(storage.bias),
            phase_correction=similar(storage.bias),
        )
        ws = SccDeviceWorkspace(device, states..., storage, scratch, Int32[0],
            nothing, nothing, ChoppedImageAbsent, ChoppedImageAbsent, false)
        _require_capture_workgroup(device, VALIDATION_WORKGROUP_SIZE;
            shared_memory=VALIDATION_WORKGROUP_SIZE * sizeof(Int32))
        reset!(ws)
        return PreparedAlgorithm(SccCorrectionF32, plan, ws, prepared.formats)
    end
end

@kernel function _select_scc_residual!(residual, status, difference, reference, indices)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            pixel = indices[i]
            residual[i] = difference[pixel] - reference[pixel]
        end
    end
end

@kernel function _stage_scc_command!(command, correction, status, reconstructed, phase_correction,
        fringed, previous, bias, gain, pole, closed)
    i = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            value = fringed ? reconstructed[i] - phase_correction[i] :
                reconstructed[i] + phase_correction[i]
            next = if !isfinite(value)
                NaN32
            elseif closed
                fma(gain, value, pole * previous[i])
            else
                previous[i]
            end
            correction[i] = next
            command[i] = bias[i] + next
        end
    end
end

@kernel function _check_scc_command!(status, command, lower, upper, calculation_failure, limits_failure)
    if @inbounds iszero(status[1])
        @inbounds for i in eachindex(command)
            value = command[i]
            if !isfinite(value)
                status[1] = calculation_failure
                break
            elseif !(lower[i] <= value <= upper[i])
                status[1] = limits_failure
                break
            end
        end
    end
end

function _enqueue_scc_correction!(ws::SccDeviceWorkspace, frame, fringed, integrator)
    committed, candidate, cal, scratch = ws.committed, ws.candidate, ws.calibration, ws.scratch
    status = scratch.status
    current, previous = fringed ? (frame, committed.image) : (committed.image, frame)
    response = fringed ? cal.unfringed_response : cal.fringed_response
    input_failure = Int32(SccNonFiniteInput)
    correction_failure = Int32(SccNonFiniteCorrection)
    enqueue!(_control_combine(scratch.difference, status, current, previous, -))
    enqueue!(_control_validate(status, scratch.difference, input_failure))
    enqueue!(_control_kernel(_select_scc_residual!,
        (scratch.residual, status, scratch.difference, cal.reference, cal.selected), length(cal.selected)))
    enqueue!(_control_validate(status, scratch.residual, input_failure))
    enqueue!(_control_kernel(_projection!,
        (scratch.reconstructed, status, cal.reconstructor, scratch.residual), length(candidate.command)))
    enqueue!(_control_combine(scratch.command_difference, status,
        committed.command, committed.exposure_command, -))
    enqueue!(_control_kernel(_projection!,
        (scratch.phase_correction, status, response, scratch.command_difference), length(candidate.command)))
    enqueue!(_control_kernel(_stage_scc_command!, (candidate.command, candidate.correction, status,
        scratch.reconstructed, scratch.phase_correction, fringed, committed.correction,
        cal.bias, integrator.gain, integrator.pole, integrator.closed), length(candidate.command)))
    enqueue!(_control_validate(status, candidate.correction, correction_failure))
    enqueue!(_control_kernel(_check_scc_command!, (status, candidate.command, cal.lower, cal.upper,
        correction_failure, Int32(SccCommandOutOfBounds)), 1; lanes=1))
    return nothing
end

function _process_admitted!(command::AbstractArray,
        prepared::PreparedAlgorithm{SccCorrectionF32,SccCorrectionPlan,<:SccDeviceWorkspace},
        frame::AbstractArray, metadata::AdmittedSampleMetadata)
    ws, plan = prepared.workspace, prepared.plan
    discard!(ws)
    _require_current_device(ws.device)
    for (name, array) in (("SCC command", command), ("SCC frame", frame))
        applicable(_require_array_device, name, array, ws.device) ||
            throw(ArgumentError("$name must be resident on the prepared device"))
        _require_array_device(name, array, ws.device)
    end
    _capture_arrays_overlap(command, frame) &&
        throw(ArgumentError("SCC input and command must not alias"))
    for owned in (ws.committed, ws.candidate, ws.scratch, ws.calibration), values in owned
        (_capture_arrays_overlap(command, values) || _capture_arrays_overlap(frame, values)) &&
            throw(ArgumentError("SCC caller arrays must not alias execution storage or calibration"))
    end
    exposure = _scc_exposure(_sample_metadata(metadata), plan.calibration.exposure_duration_nanoseconds,
        ws.committed_identity)
    exposure isa SccCorrectionFailure && return exposure
    (; identity, kind, adjacent) = exposure
    committed, candidate = ws.committed, ws.candidate
    status = ws.scratch.status
    fill!(status, Int32(0))
    enqueue!(_control_validate(status, frame, Int32(SccNonFiniteInput)))
    pure = kind != ChoppedImageMixed
    paired = pure && adjacent && ws.committed_kind != ChoppedImageMixed && ws.committed_kind != kind
    if paired
        _enqueue_scc_correction!(ws, frame, kind == ChoppedImageFringed, plan.integrator)
    else
        enqueue!(_control_copy(candidate.command, status, committed.command))
        enqueue!(_control_copy(candidate.correction, status, committed.correction))
    end
    if pure
        enqueue!(_control_copy(candidate.image, status, frame))
        enqueue!(_control_copy(candidate.exposure_command, status, committed.command))
    end
    enqueue!(_control_copy(command, status, candidate.command))
    copyto!(ws.completion, status)
    iszero(only(ws.completion)) || return SccCorrectionFailure(only(ws.completion))
    ws.candidate_identity = identity
    ws.candidate_kind = kind
    ws.candidate_ready = true
    return nothing
end
