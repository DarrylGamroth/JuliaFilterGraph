"""
    SccCalibration(; frame_shape, reference_difference, correction_mask,
        reconstructor, coordinate_order, command_unit, fringed_response, unfringed_response, reconstructor_sign,
        exposure_duration_nanoseconds, image_schema, bias_command,
        lower_limits, upper_limits)

Snapshot one coherent SCC calibration. Images use deployed `(SizeX, SizeY)`
column-major order. Reconstructor rows follow explicit controller coordinate
order `1:n`; columns are increasing linear indices of `correction_mask`.
The command unit is explicit. `fringed_response` and `unfringed_response` are the
reconstructed phase derivatives R·Jfringed and R·Junfringed in this same coordinate basis.
`reconstructor_sign` is explicitly
`:command_error` (subtract the reconstructed error) or `:correction` (add it).
The reference difference and acquired image have the same declared image
schema and exposure duration. Every reported frame must supply positive duration
matching the calibration. No unit, sign, actuator, or
pixel conversion is inferred. Limits and bias use the same controller coordinate
order.
"""
struct SccCalibration
    frame_shape::Tuple{Int,Int}
    residual::SccPixelResidualPlan{Float32}
    reconstructor::Matrix{Float32}
    fringed_response::Matrix{Float32}
    unfringed_response::Matrix{Float32}
    command_unit::String
    correction_sign::Float32
    exposure_duration_nanoseconds::Int64
    image_schema::String
    bias_command::Vector{Float32}
    lower_limits::Vector{Float32}
    upper_limits::Vector{Float32}
end

function SccCalibration(; frame_shape::Tuple{Int,Int},
        reference_difference::AbstractMatrix{Float32},
        correction_mask::AbstractMatrix{Bool},
        reconstructor::AbstractMatrix{Float32}, coordinate_order,
        fringed_response::AbstractMatrix{Float32}, unfringed_response::AbstractMatrix{Float32},
        command_unit::AbstractString, reconstructor_sign::Symbol,
        exposure_duration_nanoseconds::Integer, image_schema::AbstractString,
        bias_command::AbstractVector{Float32},
        lower_limits::AbstractVector{Float32}, upper_limits::AbstractVector{Float32})
    Base.require_one_based_indexing(reference_difference, correction_mask,
        reconstructor, fringed_response, unfringed_response, bias_command, lower_limits, upper_limits)
    all(>(0), frame_shape) || throw(ArgumentError("SCC frame dimensions must be positive"))
    size(reference_difference) == size(correction_mask) == frame_shape ||
        throw(DimensionMismatch("SCC reference and mask must match the frame shape"))
    coordinates = size(reconstructor, 1)
    coordinates > 0 || throw(ArgumentError("SCC controller coordinates must be nonempty"))
    Tuple(coordinate_order) == ntuple(identity, coordinates) ||
        throw(ArgumentError("SCC coordinate_order must explicitly name controller coordinates 1:n"))
    isempty(command_unit) && throw(ArgumentError("SCC command units must be explicit"))
    size(fringed_response) == size(unfringed_response) == (coordinates, coordinates) ||
        throw(DimensionMismatch("SCC phase responses must map the declared controller coordinate basis"))
    all(isfinite, fringed_response) && all(isfinite, unfringed_response) ||
        throw(ArgumentError("SCC phase responses must be finite"))
    sign = if reconstructor_sign === :command_error
        -1.0f0
    elseif reconstructor_sign === :correction
        1.0f0
    else
        throw(ArgumentError("SCC reconstructor_sign must be :command_error or :correction"))
    end
    0 < exposure_duration_nanoseconds <= typemax(Int64) ||
        throw(ArgumentError("SCC exposure duration must be positive Int64 nanoseconds"))
    isempty(image_schema) && throw(ArgumentError("SCC image schema must be explicit"))
    residual = SccPixelResidualPlan(
        vec(reference_difference), vec(correction_mask), 1.0f0)
    size(reconstructor, 2) == count(correction_mask) ||
        throw(DimensionMismatch("SCC reconstructor must have one column per selected pixel"))
    matrix = DenseReconstructorPlan(reconstructor).matrix
    for values in (bias_command, lower_limits, upper_limits)
        length(values) == coordinates ||
            throw(DimensionMismatch("SCC bias and limits must contain every controller coordinate"))
        all(isfinite, values) || throw(ArgumentError("SCC bias and limits must be finite"))
    end
    for index in eachindex(bias_command, lower_limits, upper_limits)
        lower_limits[index] <= bias_command[index] <= upper_limits[index] ||
            throw(ArgumentError("SCC bias command must lie within its command limits"))
    end
    return SccCalibration(frame_shape, residual, matrix,
        Matrix(fringed_response), Matrix(unfringed_response), String(command_unit), sign,
        Int64(exposure_duration_nanoseconds), String(image_schema),
        Vector(bias_command), Vector(lower_limits), Vector(upper_limits))
end

struct SccCorrectionPlan <: AbstractAlgorithmPlan
    calibration::SccCalibration
    difference::ChoppedImageDifferencePlan{Float32}
    integrator::LeakyIntegratorPlan{Float32}
end

"Persistent pairing identities, retained images, correction history and commands."
mutable struct SccCorrectionState
    difference::ChoppedImageDifferenceWorkspace{Float32}
    committed_exposure_command::Vector{Float32}
    candidate_exposure_command::Vector{Float32}
    committed_command::Vector{Float32}
    candidate_command::Vector{Float32}
    integrator::LeakyIntegratorWorkspace{Float32,Vector{Float32}}
    committed_identity::Union{Nothing,SampleAcquisitionIdentity}
    candidate_identity::Union{Nothing,SampleAcquisitionIdentity}
    candidate_clear_history::Bool
    candidate_ready::Bool
end

"Replaceable scratch for one SCC correction call."
struct SccCorrectionWorkspace
    image::Vector{Float32}
    image_validity::Vector{Bool}
    difference::Vector{Float32}
    difference_validity::Vector{Bool}
    pair_status::Vector{UInt8}
    selected_residuals::Vector{Float32}
    selected_validity::Vector{Bool}
    valid_fraction::Vector{Float32}
    reconstructed::Vector{Float32}
    correction::Vector{Float32}
    command_difference::Vector{Float32}
    phase_correction::Vector{Float32}
end

"Exact numerical state and scratch owner passed through the framework lifecycle."
struct SccCorrectionExecution
    state::SccCorrectionState
    workspace::SccCorrectionWorkspace
    initial_command::Vector{Float32}
end

function _prepare_scc(configuration)
    shape = configuration.frame_size
    pixel_count = Base.checked_mul(shape...)
    length(configuration.reference_difference) == length(configuration.correction_mask) == pixel_count ||
        throw(DimensionMismatch("SCC configuration reference and mask extents must match its shape"))
    selected = count(identity, configuration.correction_mask)
    coordinates = configuration.controller_coordinate_count
    coordinates > 0 || throw(ArgumentError("SCC controller coordinates must be positive"))
    length(configuration.reconstructor) == Base.checked_mul(coordinates, selected) ||
        throw(DimensionMismatch("SCC configuration reconstructor extent must match its selected pixels"))
    length(configuration.fringed_response) == length(configuration.unfringed_response) ==
        Base.checked_mul(coordinates, coordinates) ||
        throw(DimensionMismatch("SCC configuration phase-response extents must match its coordinates"))
    calibration = SccCalibration(; frame_shape=shape,
        reference_difference=reshape(configuration.reference_difference, shape),
        correction_mask=reshape(configuration.correction_mask, shape),
        reconstructor=reshape(configuration.reconstructor, coordinates, selected),
        coordinate_order=configuration.coordinate_order, command_unit=configuration.command_unit,
        fringed_response=reshape(configuration.fringed_response, coordinates, coordinates),
        unfringed_response=reshape(configuration.unfringed_response, coordinates, coordinates),
        reconstructor_sign=Symbol(configuration.reconstructor_sign),
        exposure_duration_nanoseconds=configuration.exposure_duration_nanoseconds,
        image_schema=configuration.image_schema, bias_command=configuration.bias_command,
        lower_limits=_expand_command_limit(configuration.command_limits[1], coordinates, "SCC lower limit"),
        upper_limits=_expand_command_limit(configuration.command_limits[2], coordinates, "SCC upper limit"))
    configuration.gain >= 0.0f0 || throw(ArgumentError("SCC gain must be nonnegative"))
    integrator = LeakyIntegratorPlan(
        calibration.correction_sign * configuration.gain, configuration.pole)
    difference = ChoppedImageDifferencePlan(pixel_count)
    selected = length(calibration.residual.selected_indices)
    state = SccCorrectionState(ChoppedImageDifferenceWorkspace(difference),
        copy(calibration.bias_command), copy(calibration.bias_command),
        copy(calibration.bias_command), copy(calibration.bias_command),
        LeakyIntegratorWorkspace(Float32, coordinates),
        nothing, nothing, false, false)
    workspace = SccCorrectionWorkspace(zeros(Float32, pixel_count), fill(true, pixel_count),
        zeros(Float32, pixel_count), fill(false, pixel_count), UInt8[UInt8(ChoppedImagePriming)],
        zeros(Float32, selected), fill(false, selected), zeros(Float32, 1),
        zeros(Float32, coordinates), zeros(Float32, coordinates),
        zeros(Float32, coordinates), zeros(Float32, coordinates))
    return (plan=SccCorrectionPlan(calibration, difference, integrator),
        workspace=SccCorrectionExecution(state, workspace, copy(calibration.bias_command)))
end

function property_schema(::Type{SccCorrectionPlan})
    return (
        runtime_property(0, :gain, Float32; default=0.1f0,
            description="SCC input gain per usable rolling exposure", unit="1",
            constraint=PropertyRange(0.0f0, floatmax(Float32))),
        runtime_property(1, :pole, Float32; default=1.0f0,
            description="SCC integrator pole per usable rolling exposure", unit="1",
            constraint=PropertyRange(0.0f0, 1.0f0)),
    )
end

property_values(plan::SccCorrectionPlan) =
    (gain=plan.calibration.correction_sign * plan.integrator.gain, pole=plan.integrator.pole)

function replace_properties(plan::SccCorrectionPlan,
        values::NamedTuple{(:gain,:pole),Tuple{Float32,Float32}})
    values.gain >= 0.0f0 || throw(ArgumentError("SCC gain must be nonnegative"))
    return SccCorrectionPlan(plan.calibration, plan.difference,
        LeakyIntegratorPlan(
            plan.calibration.correction_sign * values.gain, values.pole))
end

function commit!(execution::SccCorrectionExecution)
    state = execution.state
    state.candidate_ready || return execution
    if state.candidate_clear_history
        reset!(state.difference)
    else
        commit!(state.difference)
        state.committed_exposure_command, state.candidate_exposure_command =
            state.candidate_exposure_command, state.committed_exposure_command
    end
    state.committed_command, state.candidate_command = state.candidate_command, state.committed_command
    state.committed_identity = state.candidate_identity
    commit!(state.integrator)
    state.candidate_ready = false
    return execution
end

function discard!(execution::SccCorrectionExecution)
    execution.state.candidate_ready = false
    discard!(execution.state.difference)
    discard!(execution.state.integrator)
    return execution
end

function reset!(execution::SccCorrectionExecution)
    state, scratch = execution.state, execution.workspace
    state.committed_identity = state.candidate_identity = nothing
    state.candidate_clear_history = false
    state.candidate_ready = false
    reset!(state.difference)
    copyto!(state.committed_exposure_command, execution.initial_command)
    copyto!(state.candidate_exposure_command, execution.initial_command)
    copyto!(state.committed_command, execution.initial_command)
    copyto!(state.candidate_command, execution.initial_command)
    reset!(state.integrator)
    for values in (scratch.image, scratch.difference, scratch.selected_residuals,
            scratch.valid_fraction, scratch.reconstructed, scratch.correction,
            scratch.command_difference, scratch.phase_correction)
        fill!(values, 0.0f0)
    end
    fill!(scratch.selected_validity, false)
    fill!(scratch.difference_validity, false)
    fill!(scratch.pair_status, UInt8(ChoppedImagePriming))
    return execution
end

"Expected data or numerical rejection from SCC correction."
@enum SccCorrectionFailure::UInt8 begin
    SccMetadataMissing = 1
    SccExposureMismatch = 2
    SccSequenceInvalid = 3
    SccNonFiniteInput = 4
    SccNonFiniteCorrection = 5
    SccCommandOutOfBounds = 6
end

@doc "The exposure lacks an acquisition identity or an explicit optical image kind."
SccMetadataMissing
@doc "The supplied per-frame exposure duration differs from the prepared calibration duration."
SccExposureMismatch
@doc "An exposure repeats or decreases its acquisition sequence within one source cohort."
SccSequenceInvalid
@doc "The exposure is corrupted or its image or selected residual is nonfinite."
SccNonFiniteInput
@doc "Reconstruction, integration, or complete-command formation produced a nonfinite value."
SccNonFiniteCorrection
@doc "A complete candidate command violates its calibrated controller-coordinate limits."
SccCommandOutOfBounds

function _scc_exposure(metadata, expected_duration, previous)
    isnothing(metadata) && return SccMetadataMissing
    kind = sample_chopped_image_kind(metadata)
    (isnothing(kind) || kind == ChoppedImageAbsent) && return SccMetadataMissing
    duration = sample_exposure_duration_nanoseconds(metadata)
    isnothing(duration) && return SccMetadataMissing
    duration == expected_duration || return SccExposureMismatch
    identity = sample_acquisition_identity(metadata)
    isnothing(identity) && return SccMetadataMissing
    metadata.corrupted && return SccNonFiniteInput
    same_domain = previous !== nothing && previous.domain == identity.domain
    same_domain && identity.generation < previous.generation && return SccSequenceInvalid
    same_cohort = same_domain && previous.generation == identity.generation
    if same_cohort && identity.sequence <= previous.sequence
        return SccSequenceInvalid
    end
    adjacent = same_cohort && identity.sequence - previous.sequence == 1 &&
        !metadata.discontinuity
    return (; identity, kind, adjacent)
end

function _process_scc!(command::AbstractVector{Float32},
        execution::SccCorrectionExecution, plan::SccCorrectionPlan,
        frame::AbstractMatrix{Float32}, metadata)
    state, scratch, calibration = execution.state, execution.workspace, plan.calibration
    discard!(execution)
    Base.require_one_based_indexing(command, frame)
    length(command) == length(calibration.bias_command) && size(frame) == calibration.frame_shape ||
        throw(DimensionMismatch("SCC input or complete command extent changed"))
    difference = state.difference
    for retained in (difference.committed_image, difference.candidate_image, difference.staged_difference,
            state.committed_exposure_command, state.candidate_exposure_command,
            state.committed_command, state.candidate_command,
            state.integrator.initial_state, state.integrator.committed_state,
            state.integrator.candidate_state, execution.initial_command,
            scratch.image, scratch.difference, scratch.selected_residuals,
            scratch.valid_fraction, scratch.reconstructed, scratch.correction,
            scratch.command_difference, scratch.phase_correction)
        (Base.mightalias(command, retained) || Base.mightalias(frame, retained)) &&
            throw(ArgumentError("SCC caller arrays must not alias execution storage"))
    end
    for immutable_values in (calibration.bias_command, calibration.lower_limits,
            calibration.upper_limits, calibration.residual.reference_image)
        Base.mightalias(command, immutable_values) &&
            throw(ArgumentError("SCC command must not alias its calibration"))
    end
    for matrix in (calibration.reconstructor, calibration.fringed_response, calibration.unfringed_response)
        Base.mightalias(command, matrix) &&
            throw(ArgumentError("SCC command must not alias its calibrated matrices"))
    end
    Base.mightalias(command, calibration.residual.selected_indices) &&
        throw(ArgumentError("SCC command must not alias its selected-pixel indices"))
    Base.mightalias(command, calibration.residual.mask) &&
        throw(ArgumentError("SCC command must not alias its correction mask"))
    for validity in (scratch.image_validity, scratch.difference_validity, scratch.selected_validity,
            difference.committed_validity, difference.candidate_validity, difference.staged_validity)
        (Base.mightalias(command, validity) || Base.mightalias(frame, validity)) &&
            throw(ArgumentError("SCC caller arrays must not alias execution validity storage"))
    end
    (Base.mightalias(command, scratch.pair_status) || Base.mightalias(frame, scratch.pair_status)) &&
        throw(ArgumentError("SCC caller arrays must not alias execution pair-status storage"))
    Base.mightalias(command, frame) && throw(ArgumentError("SCC input and command must not alias"))
    exposure = _scc_exposure(metadata, calibration.exposure_duration_nanoseconds, state.committed_identity)
    exposure isa SccCorrectionFailure && return exposure
    (; identity, kind, adjacent) = exposure
    for value in frame
        isfinite(value) || return SccNonFiniteInput
    end
    state.candidate_identity = identity
    state.candidate_clear_history = kind == ChoppedImageMixed
    copyto!(state.candidate_command, state.committed_command)
    if !state.candidate_clear_history
        fringed = kind == ChoppedImageFringed
        pixel = 0
        for column in axes(frame, 2), row in axes(frame, 1)
            pixel += 1
            scratch.image[pixel] = frame[row, column]
        end
        copyto!(state.candidate_exposure_command, state.committed_command)
        result = process!(scratch.difference,
            scratch.difference_validity, scratch.pair_status, difference, plan.difference,
            scratch.image, scratch.image_validity, fringed)
        result === nothing || return SccNonFiniteInput
        complete = adjacent && scratch.pair_status[1] == UInt8(ChoppedImageComplete)
        if complete
            all(scratch.difference_validity) || return SccNonFiniteInput
            result = process!(scratch.selected_residuals,
                scratch.selected_validity, scratch.valid_fraction, calibration.residual,
                scratch.difference, scratch.difference_validity, scratch.pair_status)
            result === nothing || return SccNonFiniteInput
            mul!(scratch.reconstructed, calibration.reconstructor, scratch.selected_residuals)
            all(isfinite, scratch.reconstructed) || return SccNonFiniteCorrection
            for index in eachindex(scratch.command_difference, state.committed_command,
                    state.committed_exposure_command)
                scratch.command_difference[index] = state.committed_command[index] -
                    state.committed_exposure_command[index]
            end
            response = fringed ? calibration.unfringed_response : calibration.fringed_response
            mul!(scratch.phase_correction, response, scratch.command_difference)
            for index in eachindex(scratch.reconstructed, scratch.phase_correction)
                value = fringed ? scratch.reconstructed[index] - scratch.phase_correction[index] :
                    scratch.reconstructed[index] + scratch.phase_correction[index]
                isfinite(value) || return SccNonFiniteCorrection
                scratch.reconstructed[index] = value
            end
            result = _process_leaky_integrator!(scratch.correction, state.integrator,
                plan.integrator, scratch.reconstructed)
            result === nothing || return SccNonFiniteCorrection
            for index in eachindex(state.candidate_command, scratch.correction)
                value = calibration.bias_command[index] + scratch.correction[index]
                isfinite(value) || return SccNonFiniteCorrection
                calibration.lower_limits[index] <= value <= calibration.upper_limits[index] ||
                    return SccCommandOutOfBounds
                state.candidate_command[index] = value
            end
        end
    end
    copyto!(command, state.candidate_command)
    state.candidate_ready = true
    return nothing
end

"""
SCC correction from rolling adjacent pure exposures with an explicit optical
fringed or unfringed image kind. Priming, mixed, repeated-phase, discontinuity and
sequence-gap exposures publish the last complete command without applying the
leak recurrence. Every usable alternating exposure updates using the preceding
and current images, with calibrated phase-response compensation for their
different exposure commands. A rejected exposure publishes nothing
and leaves committed scientific state unchanged. There is no implicit clipping.
The current exposure must have used the last successfully issued controller
command through the declared mapping. Downstream conditioning must leave that
mapping unchanged or supply qualified actual-command feedback. A downstream
Graph rejection requires fail-stop and a full reset; Nodes commit locally.
Every frame must supply positive duration matching the calibration; missing
duration is rejected without changing scientific state.
"""
@algorithm SccCorrectionF32 begin
    label = "scc-correction-f32"
    failure = SccCorrectionFailure

    """Portable coherent calibration and initial controller coefficients.

    `frame_size = (width, height)` preserves the SCC frame array axis order.
    `command_limits = (lower, upper)` accepts scalar or per-coordinate bounds.
    """
    @configuration SccCorrectionF32Configuration begin
        frame_size::NTuple{2,Int} => graph_rebuild
        controller_coordinate_count::Int => graph_rebuild
        image_schema::String => graph_rebuild
        exposure_duration_nanoseconds::Int64 => construction
        coordinate_order::Vector{Int} => construction
        command_unit::String => construction
        reconstructor_sign::String => construction
        reference_difference::Vector{Float32} => construction
        correction_mask::Vector{Bool} => construction
        reconstructor::Vector{Float32} => construction
        fringed_response::Vector{Float32} => construction
        unfringed_response::Vector{Float32} => construction
        bias_command::Vector{Float32} => construction
        command_limits::NTuple{2,Union{Float32,Vector{Float32}}} => construction
        gain::Float32 => construction
        pole::Float32 => construction
    end

    properties = :runtime

    ports(configuration) = (
        input(:frame, Float32, configuration.frame_size,
            configuration.image_schema; sample=:frame_sample),
        output(:command, schema_element_type(CONTROLLER_COMMAND_V1), (configuration.controller_coordinate_count,),
            CONTROLLER_COMMAND_V1; metadata_from=:frame),
    )

    prepare(configuration) = _prepare_scc(configuration)

    process!(plan, workspace) = _process_scc!(command, workspace, plan, frame, frame_sample)
end

"Construct portable configuration arrays from a coherent programmatic calibration."
function SccCorrectionF32Configuration(calibration::SccCalibration;
        gain::Float32, pole::Float32)
    return SccCorrectionF32Configuration(;
        frame_size=calibration.frame_shape,
        controller_coordinate_count=length(calibration.bias_command),
        image_schema=calibration.image_schema,
        exposure_duration_nanoseconds=calibration.exposure_duration_nanoseconds,
        coordinate_order=collect(eachindex(calibration.bias_command)),
        command_unit=calibration.command_unit,
        reconstructor_sign=calibration.correction_sign == -1.0f0 ? "command_error" : "correction",
        reference_difference=copy(calibration.residual.reference_image),
        correction_mask=copy(calibration.residual.mask), reconstructor=vec(copy(calibration.reconstructor)),
        fringed_response=vec(copy(calibration.fringed_response)), unfringed_response=vec(copy(calibration.unfringed_response)),
        bias_command=copy(calibration.bias_command), command_limits=(copy(calibration.lower_limits), copy(calibration.upper_limits)), gain, pole)
end

required_sample_metadata(::PreparedAlgorithm{SccCorrectionF32},input_name::Symbol)=
    input_name===:frame ? (:acquisition_identity,:exposure_duration_nanoseconds,:chopped_image_kind) : ()
