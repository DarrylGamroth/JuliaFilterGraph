import FilterGraphAlgorithms: PixelSelectionF32, PixelValidityF32, CenterOfGravityF32,
    ChoppedImageDifferenceF32, ChoppedImageDifferenceFailure, ChoppedImageComplete,
    ChoppedImagePriming, ChoppedImageFringed, ChoppedImageUnfringed, ChoppedImageMixed,
    ValidityGatedClosedLoopIntegratorF32

include("pixel_selection.jl")

const COG_TILE = 256

@kernel unsafe_indices=true function _center_of_gravity!(
        centroid, flux, validity, pixels, x, y, settings, status, fallback)
    lane = @index(Local, Linear)
    samples = @localmem Float32 (COG_TILE,)
    x_terms = @localmem Float32 (COG_TILE,)
    y_terms = @localmem Float32 (COG_TILE,)
    selected = @localmem Bool (COG_TILE,)
    active = @inbounds iszero(status[1]) && (isnothing(fallback) || fallback[1])
    sx = sy = total = 0f0
    if active
        # Load and multiply a tile together. One lane adds the selected terms
        # in the CPU's order, including at flux and validity boundaries.
        for offset in 0:COG_TILE:(length(pixels)-1)
            pixel = offset + lane
            @inbounds if pixel <= length(pixels)
                value = pixels[pixel]
                use = !isfinite(value) || iszero(settings[4]) || value >= settings[5]
                selected[lane] = use
                if use
                    samples[lane] = value
                    x_terms[lane] = value * x[pixel]
                    y_terms[lane] = value * y[pixel]
                end
            end
            @synchronize
            if lane == 1
                @inbounds for j in 1:min(COG_TILE, length(pixels)-offset)
                    if selected[j]
                        sx += x_terms[j]
                        sy += y_terms[j]
                        total += samples[j]
                    end
                end
            end
            @synchronize
        end
        if lane == 1
            @inbounds begin
                dx, dy, measured, valid = FilterGraphAlgorithms._finish_center_of_gravity(
                    sx, sy, total, settings[1], settings[2], settings[3])
                centroid[1] = dx
                centroid[2] = dy
                flux[1] = measured
                validity[1] = valid
            end
        end
    end
end

struct CapturedCenterOfGravity{K,S}
    kernels::K
    settings::S
end

function enqueue!(adapter::CapturedCenterOfGravity)
    for kernel in adapter.kernels
        enqueue!(kernel)
    end
    return nothing
end
preflight(::CapturedCenterOfGravity) = nothing
complete!(::CapturedCenterOfGravity, ::Int32) = nothing
reset!(adapter::CapturedCenterOfGravity) = adapter

_cog_settings(plan) = (plan.reference_x, plan.reference_y, plan.flux_threshold,
    Float32(plan.pixel_threshold_enabled), plan.pixel_threshold)
_composition_settings(adapter::CapturedCenterOfGravity, plan) = _cog_settings(plan)

function _prepare_center_of_gravity(device, prepared, outputs, inputs, status; ordered)
    _check_capture_ports(device, prepared, outputs, inputs, status)
    _require_cog_workgroup(device)
    plan = prepared.plan
    pixels = only(inputs)
    backend = KernelAbstractions.get_backend(first(outputs))
    settings = _capture_array(device, collect(_cog_settings(plan)))
    x = _capture_array(device, plan.x_coordinates)
    y = _capture_array(device, plan.y_coordinates)
    fallback = ordered ? nothing : _capture_zeros(device, Bool, 1)
    exact = CapturedKernel(_center_of_gravity!(backend, COG_TILE),
        (outputs..., pixels, x, y, settings, status, fallback), COG_TILE)
    ordered && return CapturedCenterOfGravity((exact,), settings)
    return _prepare_parallel_center_of_gravity(device, backend, exact, outputs,
        pixels, x, y, settings, status, fallback)
end

@inline function _store_chopped_pixel!(i, difference, validity, pair_status, history,
        history_validity, current, valid, flags)
    @inbounds begin
        paired = !iszero(flags[1])
        value = 0f0
        output_valid = paired && valid && history_validity[i]
        if output_valid
            previous = history[i]
            value = iszero(flags[2]) ? previous - current : current - previous
            output_valid = isfinite(value)
        end
        difference[i] = output_valid ? value : 0f0
        validity[i] = output_valid
        history[i] = current
        history_validity[i] = valid
        if i == 1
            pair_status[1] = UInt8(paired ? ChoppedImageComplete : ChoppedImagePriming)
        end
    end
    return nothing
end

@kernel function _chopped_difference!(difference, validity, pair_status, history,
        history_validity, pixels, pixel_validity, flags, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        current = pixels[i]
        _store_chopped_pixel!(i, difference, validity, pair_status, history,
            history_validity, current, pixel_validity[i] && isfinite(current), flags)
    end
end

mutable struct ChoppedPairMetadata
    committed_primed::Bool
    candidate_ready::Bool
    candidate_primed::Bool
    committed_identity::Union{Nothing,FilterGraphAlgorithms.SampleAcquisitionIdentity}
    candidate_identity::Union{Nothing,FilterGraphAlgorithms.SampleAcquisitionIdentity}
    committed_duration::UInt64
    candidate_duration::UInt64
    committed_kind::ChoppedImageKind
    candidate_kind::ChoppedImageKind
end
ChoppedPairMetadata() = ChoppedPairMetadata(false,false,false,nothing,nothing,0,0,
    ChoppedImageFringed,ChoppedImageFringed)

function ChoppedPairMetadata(ws)
    return ChoppedPairMetadata(ws.committed_primed, false, false, ws.committed_identity,
        nothing, ws.committed_duration, 0, ws.committed_kind, ChoppedImageFringed)
end

function FilterGraphAlgorithms.commit!(state::ChoppedPairMetadata)
    if state.candidate_ready
        state.committed_primed = state.candidate_primed
        state.committed_kind = state.candidate_kind
        state.committed_identity = state.candidate_identity
        state.committed_duration = state.candidate_duration
        state.candidate_ready = false
    end
    return state
end

function reset!(state::ChoppedPairMetadata)
    state.committed_primed = state.candidate_ready = state.candidate_primed = false
    state.committed_identity = state.candidate_identity = nothing
    state.committed_duration = state.candidate_duration = 0
    state.committed_kind = state.candidate_kind = ChoppedImageFringed
    return state
end

struct CapturedChoppedImage{K,F,P,W}
    kernel::K
    flags::F
    plan::P
    workspace::W
end

include("pixel_preparation.jl")

# The graph owner supplies camera metadata before the native launch. Other
# execution profiles must provide the same admission boundary.
function _prepare_chopped_capture(device, prepared, outputs, inputs, status)
    _check_capture_ports(device, prepared, outputs, inputs, status)
    _require_capture_workgroup(device, 256)
    backend = KernelAbstractions.get_backend(first(outputs))
    extent = length(first(inputs))
    workspace = prepared.workspace
    FilterGraphAlgorithms._chopped_workspace_extent_is_valid(workspace, prepared.plan) &&
        extent == prepared.plan.pixel_count || throw(DimensionMismatch("invalid chopped-image workspace extent"))
    flags = _capture_zeros(device, Int32, 2)
    kernel = CapturedKernel(_chopped_difference!(backend, 256),
        (outputs..., _capture_array(device, workspace.committed_image),
            _capture_array(device, workspace.committed_validity), inputs..., flags, status), extent)
    return CapturedChoppedImage(kernel, flags, prepared.plan, ChoppedPairMetadata(workspace))
end

_admit_chopped_capture!(adapter::CapturedChoppedImage, sample) =
    _admit_chopped_sample!(adapter.plan, adapter.workspace, sample)

function _admit_chopped_sample!(plan, workspace, sample)
    admission = FilterGraphAlgorithms._chopped_image_admission(plan, workspace, sample)
    admission isa ChoppedImageDifferenceFailure && return admission
    kind = admission.trigger_high ? plan.trigger_high_kind :
        plan.trigger_high_kind == ChoppedImageFringed ? ChoppedImageUnfringed : ChoppedImageFringed
    primed = !(plan.require_acquisition_identity &&
        (FilterGraphAlgorithms.sample_chopped_image_kind(sample) == ChoppedImageMixed || sample.corrupted))
    paired = primed && admission.adjacent && workspace.committed_primed &&
        workspace.committed_kind != kind
    workspace.candidate_kind = kind
    workspace.candidate_identity = plan.require_acquisition_identity ?
        FilterGraphAlgorithms.sample_acquisition_identity(sample) : nothing
    workspace.candidate_duration = plan.require_acquisition_identity ?
        FilterGraphAlgorithms.sample_exposure_duration_nanoseconds(sample) : UInt64(0)
    workspace.candidate_primed = primed
    workspace.candidate_ready = true
    return Int32(paired), Int32(kind == ChoppedImageFringed)
end

enqueue!(adapter::CapturedChoppedImage) = enqueue!(adapter.kernel)
preflight(::CapturedChoppedImage) = nothing
function complete!(adapter::CapturedChoppedImage, ::Int32)
    # Completion visits nodes in order and stops at the first rejection. The
    # previous image is adopted even if SCC residual reports an incomplete pair.
    FilterGraphAlgorithms.commit!(adapter.workspace)
    return nothing
end
function reset!(adapter::CapturedChoppedImage)
    FilterGraphAlgorithms.reset!(adapter.workspace)
    fill!(adapter.flags, Int32(0))
    return adapter
end

@kernel function _gated_integrator!(correction, state, residual, feedback, validity,
        committed, previous, has_output, candidate_state, candidate_output, settings, status, failures)
    lane = @index(Global, Linear)
    if lane == 1 && @inbounds iszero(status[1])
        @inbounds begin
            if isnothing(validity) || all(validity)
                if !all(isfinite, residual)
                    status[1] = failures[1]
                elseif has_output[1] && !all(isfinite, feedback)
                    status[1] = failures[2]
                elseif !iszero(settings[4])
                    for i in eachindex(committed)
                        value = has_output[1] ? fma(-settings[3], feedback[i], previous[i]) : committed[i]
                        demanded = fma(settings[1], residual[i], settings[2] * value)
                        if !(isfinite(value) && isfinite(demanded))
                            status[1] = failures[3]
                            break
                        end
                        candidate_state[i] = value
                        candidate_output[i] = demanded
                    end
                    if iszero(status[1])
                        for i in eachindex(committed)
                            committed[i] = candidate_state[i]
                            previous[i] = candidate_output[i]
                        end
                    end
                end
            end
            if iszero(status[1])
                for i in eachindex(committed)
                    state[i] = committed[i]
                    correction[i] = previous[i]
                end
                has_output[1] = true
            end
        end
    end
end

struct GatedIntegrator{K,V,S}
    kernel::K
    state::V
    settings::S
end

_integrator_settings(plan) = (plan.gain, plan.pole, plan.anti_windup_gain, Float32(plan.closed))
_composition_settings(adapter::GatedIntegrator, plan) = _integrator_settings(plan)
enqueue!(adapter::GatedIntegrator) = enqueue!(adapter.kernel)
preflight(::GatedIntegrator) = nothing
function complete!(adapter::GatedIntegrator, status::Int32)
    failures = last(adapter.kernel.arguments)
    status == failures[1] && return ClosedLoopCorrectionNonFiniteInput
    status == failures[2] && return ClosedLoopCorrectionNonFiniteFeedback
    status == failures[3] && return ClosedLoopCorrectionNonFiniteOutput
    return nothing
end
function reset!(adapter::GatedIntegrator)
    for state in adapter.state
        fill!(state, zero(eltype(state)))
    end
    return adapter
end

function _prepare_composition_integrator(device, prepared, outputs, inputs, status, node_index)
    _check_capture_ports(device, prepared, outputs, inputs, status)
    plan, ws = prepared.plan, prepared.workspace
    state = (_capture_array(device, ws.committed_state), _capture_array(device, ws.committed_output),
        _capture_array(device, Bool[ws.committed_has_output]))
    settings = _capture_array(device, collect(_integrator_settings(plan)))
    scratch = (_capture_zeros(device, Float32, plan.state_length),
        _capture_zeros(device, Float32, plan.state_length))
    failures = (_failure_code(node_index, ClosedLoopCorrectionNonFiniteInput),
        _failure_code(node_index, ClosedLoopCorrectionNonFiniteFeedback),
        _failure_code(node_index, ClosedLoopCorrectionNonFiniteOutput))
    kernel = CapturedKernel(_gated_integrator!(KernelAbstractions.get_backend(first(outputs)), 1),
        (outputs..., inputs[1], inputs[2], length(inputs) == 3 ? inputs[3] : nothing,
            state..., scratch..., settings, status, failures), 1)
    return GatedIntegrator(kernel, state, settings)
end
