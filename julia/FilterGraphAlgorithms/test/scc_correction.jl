module SccCorrectionTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!

const SCC_DOMAIN = ntuple(index -> UInt8(index), 16)
const SCC_DURATION_NS = Int64(1_000_000)
const SCC_IMAGE_SCHEMA = "org.subaru.spiders.synthetic-scc-counts-xy/1"

function scc_calibration(; sign=:command_error, bounds=1.0f0,
        reference=zeros(Float32, 2, 2), bias=fill(0.01f0, 2), scale=1.0f0,
        coordinate_order=Tuple(1:2), command_unit="normalized-bax307",
        fringed_response=zeros(Float32, 2, 2), unfringed_response=zeros(Float32, 2, 2))
    mask = Bool[true false; false true]
    reconstructor = zeros(Float32, 2, 2)
    reconstructor[1, 1] = scale
    reconstructor[2, 2] = 2.0f0 * scale
    return SccCalibration(; frame_shape=(2, 2), reference_difference=reference,
        correction_mask=mask, reconstructor, coordinate_order, fringed_response, unfringed_response,
        command_unit, reconstructor_sign=sign,
        exposure_duration_nanoseconds=SCC_DURATION_NS, image_schema=SCC_IMAGE_SCHEMA,
        bias_command=bias, lower_limits=fill(-bounds, 2), upper_limits=fill(bounds, 2))
end

function prepared_scc(; calibration=scc_calibration(), gain=0.5f0, pole=0.8f0)
    return prepare_algorithm(SccCorrectionF32,
        SccCorrectionF32Configuration(calibration; gain, pole))
end

function scc_metadata(sequence; trigger=SAMPLE_TRIGGER_HIGH, duration=SCC_DURATION_NS,
        domain=SCC_DOMAIN, generation=UInt64(1), discontinuity=false, trigger_high_is_fringed=true)
    raw = SampleMetadata(sequence; trigger_state=trigger,
        exposure_duration_nanoseconds=duration,
        acquisition_identity=SampleAcquisitionIdentity(domain, generation, UInt64(sequence)),
        discontinuity)
    return chopped_image_metadata(raw; trigger_high_is_fringed)
end

scc_step!(output, prepared, image::AbstractMatrix, metadata) =
    process!(output, prepared, image, (frame=metadata,))
scc_step!(output, prepared, image::AbstractVector, metadata) =
    scc_step!(output, prepared, reshape(image, prepared.plan.calibration.frame_shape), metadata)

function scc_allocation_bytes(prepared, fringed, unfringed, correction;
        duration=SCC_DURATION_NS)
    fringed = reshape(fringed, prepared.plan.calibration.frame_shape)
    unfringed = reshape(unfringed, prepared.plan.calibration.frame_shape)
    output = zeros(Float32, 2)
    first = scc_metadata(1; duration)
    second = scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW, duration)
    reset!(prepared)
    scc_step!(output, prepared, fringed, first)
    scc_step!(output, prepared, unfringed, second)
    reset!(prepared)
    if correction
        scc_step!(output, prepared, fringed, first)
        return @allocated scc_step!(output, prepared, unfringed, second)
    end
    return @allocated scc_step!(output, prepared, fringed, first)
end

function scc_rolling_allocation_bytes(output, prepared, fringed, unfringed)
    first = scc_metadata(1)
    second = scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)
    third = scc_metadata(3)
    reset!(prepared)
    scc_step!(output, prepared, fringed, first)
    scc_step!(output, prepared, unfringed, second)
    scc_step!(output, prepared, fringed, third)
    reset!(prepared)
    scc_step!(output, prepared, fringed, first)
    scc_step!(output, prepared, unfringed, second)
    return @allocated scc_step!(output, prepared, fringed, third)
end

@testset "SCC declaration and calibration" begin
    @test SccCorrectionF32 in FilterGraphAlgorithms.algorithms()
    @test algorithm_label(SccCorrectionF32) == "scc-correction-f32"
    prepared = prepared_scc()
    @test algorithm_formats(prepared)[1].shape == (2, 2)
    @test algorithm_formats(prepared)[1].schema == SCC_IMAGE_SCHEMA
    @test algorithm_formats(prepared)[2].shape == (2,)
    @test algorithm_formats(prepared)[2].schema ==
        "pipewireao/controller-command-f32/1"
    @test algorithm_ports(SccCorrectionF32)[2].metadata_source == :frame
    @test_throws ArgumentError prepared_scc(; gain=-0.1f0)
    @test_throws ArgumentError prepared_scc(; gain=NaN32)
    @test_throws ArgumentError prepared_scc(; pole=1.1f0)
    @test_throws ArgumentError scc_calibration(; sign=:unknown)
    @test_throws ArgumentError scc_calibration(; coordinate_order=Tuple(2:-1:1))
    @test_throws ArgumentError scc_calibration(; command_unit="")
    @test_throws ArgumentError scc_calibration(; bounds=0.001f0)
    @test_throws ArgumentError scc_calibration(; reference=fill(NaN32, 2, 2))
    calibration = scc_calibration()
    @test calibration.residual.selected_indices == [1, 4]
    @test calibration.reconstructor[1, 1] == 1.0f0
    @test calibration.reconstructor[2, 2] == 2.0f0
end

@testset "SCC qualification of the prior priming failure" begin
    plan = SccPixelResidualPlan(zeros(Float32, 4), fill(true, 4), 1.0f0)
    @test FilterGraphAlgorithms.process!(zeros(Float32, 4), fill(false, 4),
        zeros(Float32, 1), plan, zeros(Float32, 4), fill(false, 4),
        UInt8[UInt8(ChoppedImagePriming)]) === SccPixelResidualIncompletePair
    # Passing zero reconstruction through a leaky integrator is not a hold.
    history = LeakyIntegratorWorkspace(fill(0.1f0, 2))
    output = zeros(Float32, 2)
    FilterGraphAlgorithms.process!(output, history, LeakyIntegratorPlan(1.0f0, 0.8f0),
        zeros(Float32, 2))
    @test output ≈ fill(0.08f0, 2)
    @test output != FilterGraphAlgorithms.leaky_integrator_state(history)
end

@testset "SCC rolling updates and holds" begin
    prepared = prepared_scc()
    output = fill(42.0f0, 2)
    fringed = Float32[0.1, 9, 8, -0.1]
    unfringed = zeros(Float32, 4)
    bias = copy(prepared.plan.calibration.bias_command)
    @test scc_step!(output, prepared, fringed, scc_metadata(1)) === nothing
    @test output == bias
    @test scc_step!(output, prepared, unfringed,
        scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === nothing
    expected = copy(bias)
    expected[1] -= 0.05f0
    expected[2] += 0.1f0
    @test output ≈ expected
    @test FilterGraphAlgorithms.chopped_image_is_primed(prepared.workspace.state.difference)
    previous = copy(output)
    @test scc_step!(output, prepared, fringed, scc_metadata(3)) === nothing
    @test output != previous
    @test scc_step!(output, prepared, unfringed,
        scc_metadata(4; trigger=SAMPLE_TRIGGER_LOW)) === nothing
    expected[1] = bias[1] - 0.05f0 * 2.44f0
    expected[2] = bias[2] + 0.1f0 * 2.44f0
    @test output ≈ expected
    @test scc_allocation_bytes(prepared, fringed, unfringed, false) == 0
    @test scc_allocation_bytes(prepared, fringed, unfringed, true) == 0

    positive = prepared_scc(; calibration=scc_calibration(; sign=:correction))
    scc_step!(output, positive, fringed, scc_metadata(1))
    scc_step!(output, positive, unfringed, scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW))
    @test output[1] ≈ bias[1] + 0.05f0
    @test output[2] ≈ bias[2] - 0.1f0
end

@testset "SCC requires reported exposure duration" begin
    prepared = prepared_scc()
    output = fill(42f0, 2)
    frame = zeros(Float32, 4)
    @test scc_step!(output, prepared, frame,
        scc_metadata(1; duration=nothing)) === SccMetadataMissing
    @test scc_step!(output, prepared, frame,
        scc_metadata(1; duration=SCC_DURATION_NS + 1)) === SccExposureMismatch
    @test output == fill(42f0, 2)
    @test prepared.workspace.state.committed_identity === nothing
    @test scc_step!(output, prepared, frame, SampleMetadata(scc_metadata(1);
        chopped_image_kind=ChoppedImageAbsent)) === SccMetadataMissing
    @test scc_step!(output, prepared, frame, scc_metadata(1)) === nothing
    @test scc_allocation_bytes(prepared, frame, frame, false) == 0
    @test scc_allocation_bytes(prepared, frame, frame, true) == 0
end

@testset "SCC freshness follows acquisition rather than publication" begin
    prepared = prepared_scc()
    output = zeros(Float32, 2)
    frame = zeros(Float32, 4)
    first = SampleMetadata(scc_metadata(1); sequence=100)
    @test scc_step!(output, prepared, frame, first) === nothing
    @test scc_step!(output, prepared, frame,
        SampleMetadata(first; sequence=101)) === SccSequenceInvalid
    @test scc_step!(output, prepared, frame,
        SampleMetadata(scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW); sequence=200)) === nothing
    @test prepared.workspace.state.committed_identity.sequence == 2
    @test scc_step!(output, prepared, frame, scc_metadata(1; generation=2)) === nothing
    @test scc_step!(output, prepared, frame, scc_metadata(3)) === SccSequenceInvalid
end

@testset "SCC explicit optical roles" begin
    prepared = prepared_scc()
    output = zeros(Float32, 2)
    fringed = Float32[0.1, 0, 0, -0.1]
    for (sequence, kind) in enumerate((ChoppedImageFringed, ChoppedImageUnfringed))
        metadata = SampleMetadata(sequence; chopped_image_kind=kind,
            exposure_duration_nanoseconds=SCC_DURATION_NS,
            acquisition_identity=SampleAcquisitionIdentity(SCC_DOMAIN, 1, sequence))
        frame = kind == ChoppedImageFringed ? fringed : zeros(Float32, 4)
        @test sample_trigger_state(metadata) == SAMPLE_TRIGGER_ABSENT
        @test scc_step!(output, prepared, frame, metadata) === nothing
    end
    @test output ≈ prepared.plan.calibration.bias_command + Float32[-0.05, 0.1]
end

@testset "SCC mixed, gaps, repeated phases and cohorts" begin
    prepared = prepared_scc()
    output = zeros(Float32, 2)
    frame = Float32[0.1, 0, 0, -0.1]
    baseline = copy(prepared.plan.calibration.bias_command)
    scc_step!(output, prepared, frame, scc_metadata(1))
    mixed = SampleMetadata(2; trigger_state=SAMPLE_TRIGGER_HIGH,
        chopped_image_kind=ChoppedImageMixed, exposure_duration_nanoseconds=SCC_DURATION_NS,
        acquisition_identity=SampleAcquisitionIdentity(SCC_DOMAIN, 1, 2))
    @test scc_step!(output, prepared, frame, mixed) === nothing
    @test output == baseline
    @test !FilterGraphAlgorithms.chopped_image_is_primed(prepared.workspace.state.difference)
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(3; trigger=SAMPLE_TRIGGER_LOW))
    @test output == baseline
    scc_step!(output, prepared, frame, scc_metadata(4))
    @test output[1] ≈ baseline[1] - 0.05f0
    previous = copy(output)
    scc_step!(output, prepared, frame, scc_metadata(5))
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(7; trigger=SAMPLE_TRIGGER_LOW))
    @test output == previous
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(8; trigger=SAMPLE_TRIGGER_LOW))
    @test output == previous
    # A new generation permits a sequence restart and cannot pair with old history.
    scc_step!(output, prepared, frame, scc_metadata(1; generation=UInt64(2)))
    @test output == previous
    @test prepared.workspace.state.committed_identity.generation == 2
    scc_step!(output, prepared, zeros(Float32, 4),
        scc_metadata(2; generation=UInt64(2), trigger=SAMPLE_TRIGGER_LOW, discontinuity=true))
    @test output == previous
    # A different acquisition domain similarly primes rather than crossing cohorts.
    scc_step!(output, prepared, frame,
        scc_metadata(1; domain=ntuple(_ -> UInt8(99), 16)))
    @test output == previous
end

@testset "SCC runtime coefficients preserve command history" begin
    prepared = prepared_scc()
    output = zeros(Float32, 2)
    frame = Float32[0.1, 0, 0, -0.1]
    scc_step!(output, prepared, frame, scc_metadata(1))
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW))
    previous = copy(output)
    previous_history = copy(FilterGraphAlgorithms.leaky_integrator_state(prepared.workspace.state.integrator))
    replacement = FilterGraphAlgorithms.prepare_properties(prepared.plan,
        (PropertyAssignment(0, 0.25f0), PropertyAssignment(1, 1.0f0)))
    @test FilterGraphAlgorithms.property_values(replacement) == (gain=0.25f0, pole=1.0f0)
    @test replacement.calibration === prepared.plan.calibration
    @test_throws PropertyConstraintError FilterGraphAlgorithms.prepare_properties(prepared.plan,
        (PropertyAssignment(0, -0.25f0),))
    # PreparedAlgorithm's ordinary immutable Plan replacement keeps its execution owner.
    prepared.plan = replacement
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(3; trigger=SAMPLE_TRIGGER_LOW))
    @test output == previous
    @test FilterGraphAlgorithms.leaky_integrator_state(prepared.workspace.state.integrator) == previous_history
    scc_step!(output, prepared, frame, scc_metadata(4))
    @test output[1] ≈ prepared.plan.calibration.bias_command[1] - 0.075f0
    @test output[2] ≈ prepared.plan.calibration.bias_command[2] + 0.15f0
end

@testset "SCC rejection, discard and reset" begin
    prepared = prepared_scc()
    output = fill(42.0f0, 2)
    frame = Float32[0.1, 0, 0, -0.1]
    @test process!(output, prepared, reshape(frame, 2, 2)) === SccMetadataMissing
    @test scc_step!(output, prepared, frame,
        SampleMetadata(1; trigger_state=SAMPLE_TRIGGER_HIGH,
            chopped_image_kind=ChoppedImageFringed)) === SccMetadataMissing
    @test scc_step!(output, prepared, frame,
        scc_metadata(1; duration=SCC_DURATION_NS + 1)) === SccExposureMismatch
    @test scc_step!(output, prepared, frame,
        scc_metadata(1; trigger=SAMPLE_TRIGGER_ABSENT)) === SccMetadataMissing
    @test scc_step!(output, prepared, frame, SampleMetadata(1;
        trigger_state=SAMPLE_TRIGGER_HIGH,
        exposure_duration_nanoseconds=SCC_DURATION_NS,
        acquisition_identity=SampleAcquisitionIdentity(SCC_DOMAIN, 1, 1))) === SccMetadataMissing
    @test output == fill(42.0f0, 2)
    scc_step!(output, prepared, frame, scc_metadata(1))
    previous = copy(output)
    @test scc_step!(output, prepared, frame, scc_metadata(1)) === SccSequenceInvalid
    @test scc_step!(output, prepared, Float32[NaN32, 0, 0, 0],
        scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === SccNonFiniteInput
    @test output == previous
    @test prepared.workspace.state.committed_identity.sequence == 1
    # Explicit staging followed by discard leaves mathematical history unchanged.
    @test FilterGraphAlgorithms._process_scc!(output, prepared.workspace,
        prepared.plan, zeros(Float32, 2, 2), scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === nothing
    FilterGraphAlgorithms.discard!(prepared.workspace)
    @test prepared.workspace.state.committed_identity.sequence == 1
    @test all(iszero, FilterGraphAlgorithms.leaky_integrator_state(prepared.workspace.state.integrator))
    @test prepared.workspace.state.committed_command == previous
    scc_step!(output, prepared, zeros(Float32, 4), scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW))
    @test output != previous
    reset!(prepared)
    @test prepared.workspace.state.committed_identity === nothing
    @test !FilterGraphAlgorithms.chopped_image_is_primed(prepared.workspace.state.difference)
    @test prepared.workspace.state.committed_command == prepared.plan.calibration.bias_command
    scc_step!(output, prepared, frame, scc_metadata(1))
    @test output == prepared.plan.calibration.bias_command

    bounded = prepared_scc(; calibration=scc_calibration(; bounds=0.02f0))
    scc_step!(output, bounded, frame, scc_metadata(1))
    previous = copy(output)
    @test scc_step!(output, bounded, zeros(Float32, 4),
        scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === SccCommandOutOfBounds
    @test output == previous
    @test bounded.workspace.state.committed_identity.sequence == 1
    @test all(iszero, FilterGraphAlgorithms.leaky_integrator_state(bounded.workspace.state.integrator))
    @test_throws ArgumentError scc_step!(bounded.plan.calibration.bias_command,
        bounded, frame, scc_metadata(2))
    @test_throws ArgumentError scc_step!(bounded.workspace.state.committed_command,
        bounded, frame, scc_metadata(2))

    overflowing = prepared_scc(; calibration=scc_calibration(; scale=floatmax(Float32) / 4))
    scc_step!(output, overflowing, Float32[10, 0, 0, 10], scc_metadata(1))
    previous = copy(output)
    @test scc_step!(output, overflowing, zeros(Float32, 4),
        scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === SccNonFiniteCorrection
    @test output == previous
    @test overflowing.workspace.state.committed_identity.sequence == 1

    overflowing = prepared_scc(; gain=floatmax(Float32))
    scc_step!(output, overflowing, Float32[10, 0, 0, 10], scc_metadata(1))
    previous = copy(output)
    @test scc_step!(output, overflowing, zeros(Float32, 4),
        scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === SccNonFiniteCorrection
    @test output == previous
    integrator = overflowing.workspace.state.integrator
    @test integrator.candidate_state == integrator.committed_state == zeros(Float32, 2)
end

@testset "SCC moving-command phase compensation" begin
    fringed_jacobian = Float32[2.0 0.4; 0.3 1.5]
    unfringed_jacobian = Float32[0.6 -0.2; 0.1 0.5]
    reconstructor = inv(fringed_jacobian - unfringed_jacobian)
    calibration = SccCalibration(; frame_shape=(2, 2),
        reference_difference=zeros(Float32, 2, 2), correction_mask=Bool[true false; false true],
        reconstructor, coordinate_order=(1, 2), command_unit="normalized-bax307",
        fringed_response=reconstructor * fringed_jacobian, unfringed_response=reconstructor * unfringed_jacobian,
        reconstructor_sign=:command_error, exposure_duration_nanoseconds=SCC_DURATION_NS,
        image_schema=SCC_IMAGE_SCHEMA, bias_command=zeros(Float32, 2),
        lower_limits=fill(-1f0, 2), upper_limits=fill(1f0, 2))
    disturbance = Float32[0.1, -0.05]
    target = -(reconstructor * disturbance)
    commands = Vector{Float32}[]
    for trigger_high_is_fringed in (true, false)
        prepared = prepared_scc(; calibration, gain=0.5f0, pole=1f0)
        command = zeros(Float32, 2)
        previous_image = zeros(Float32, 2)
        errors = Float32[]
        for sequence in 1:15
            current = copy(command) # This command produced the arriving exposure.
            fringed = isodd(sequence)
            pixels = fringed ? fringed_jacobian * current + disturbance : unfringed_jacobian * current
            image = Float32[pixels[1] 0; 0 pixels[2]]
            trigger = fringed == trigger_high_is_fringed ? SAMPLE_TRIGGER_HIGH : SAMPLE_TRIGGER_LOW
            metadata = scc_metadata(sequence; trigger, trigger_high_is_fringed)
            @test scc_step!(command, prepared, image, metadata) === nothing
            @test prepared.workspace.state.committed_exposure_command == current
            if trigger_high_is_fringed
                push!(commands, copy(command))
            else
                @test command == commands[sequence]
            end
            if sequence == 1
                @test command == current
            else
                exact_error = current - target
                naive_difference = fringed ? pixels - previous_image : previous_image - pixels
                naive_error = reconstructor * naive_difference
                @test prepared.workspace.workspace.reconstructed ≈ exact_error atol=3f-8
                @test command ≈ current - 0.5f0 * exact_error atol=3f-8
                push!(errors, sqrt(sum(abs2, exact_error)))
                if sequence in (3, 4)
                    # Fail-before oracle: omitting phase compensation gives the
                    # wrong current-command residual in each alternating phase.
                    @test sqrt(sum(abs2, naive_error - exact_error)) > 0.001f0
                end
            end
            previous_image = pixels
        end
        @test errors ≈ errors[1] .* (0.5f0 .^ (0:13)) atol=5f-8
        @test errors[end] < 0.001f0 * errors[1]
        @test command ≈ target atol=1f-5
        @test scc_rolling_allocation_bytes(command, prepared,
            Float32[disturbance[1] 0; 0 disturbance[2]], zeros(Float32, 2, 2)) == 0
    end
end

@testset "SCC one controller coordinate and mandatory phase calibration" begin
    calibration = SccCalibration(; frame_shape=(2, 2),
        reference_difference=zeros(Float32, 2, 2), correction_mask=Bool[true false; false false],
        reconstructor=ones(Float32, 1, 1), coordinate_order=(1,), command_unit="micrometre-opd",
        fringed_response=fill(3f0, 1, 1), unfringed_response=fill(2f0, 1, 1),
        reconstructor_sign=:command_error, exposure_duration_nanoseconds=SCC_DURATION_NS,
        image_schema=SCC_IMAGE_SCHEMA, bias_command=zeros(Float32, 1),
        lower_limits=Float32[-1], upper_limits=Float32[1])
    prepared = prepared_scc(; calibration, gain=0.5f0, pole=1f0)
    @test algorithm_formats(prepared)[2].shape == (1,)
    @test prepared.plan.calibration.command_unit == "micrometre-opd"
    command = zeros(Float32, 1)
    for sequence in 1:5
        current = only(command)
        fringed = isodd(sequence)
        image = zeros(Float32, 2, 2)
        image[1] = fringed ? 3current + 0.1f0 : 2current
        scc_step!(command, prepared, image,
            scc_metadata(sequence; trigger=fringed ? SAMPLE_TRIGGER_HIGH : SAMPLE_TRIGGER_LOW))
        @test only(command) ≈ (sequence == 1 ? current : current - 0.5f0 * (current + 0.1f0)) atol=2f-8
    end
    @test_throws DimensionMismatch scc_calibration(; fringed_response=zeros(Float32, 1, 1))
    @test_throws DimensionMismatch scc_calibration(; unfringed_response=zeros(Float32, 1, 1))
    @test_throws ArgumentError scc_calibration(; fringed_response=fill(NaN32, 2, 2))
    @test_throws ArgumentError scc_calibration(; unfringed_response=fill(Inf32, 2, 2))
    configuration = SccCorrectionF32Configuration(calibration; gain=0.5f0, pole=1f0)
    @test configuration.controller_coordinate_count == 1
    @test configuration.coordinate_order == [1]
    @test configuration.fringed_response == [3f0]
    @test configuration.unfringed_response == [2f0]
end

@testset "SCC staged history remains atomic" begin
    prepared = prepared_scc()
    command = zeros(Float32, 2)
    first = Float32[0.1 0; 0 -0.1]
    scc_step!(command, prepared, first, scc_metadata(1))
    retained = FilterGraphAlgorithms.chopped_image_retained_image(prepared.workspace.state.difference)
    exposure_command = copy(prepared.workspace.state.committed_exposure_command)
    for metadata in (scc_metadata(2; trigger=SAMPLE_TRIGGER_MIXED),
            scc_metadata(4; trigger=SAMPLE_TRIGGER_LOW))
        @test FilterGraphAlgorithms._process_scc!(command, prepared.workspace,
            prepared.plan, zeros(Float32, 2, 2), metadata) === nothing
        FilterGraphAlgorithms.discard!(prepared.workspace)
        @test prepared.workspace.state.committed_identity.sequence == 1
        @test FilterGraphAlgorithms.chopped_image_retained_image(prepared.workspace.state.difference) == retained
        @test prepared.workspace.state.committed_exposure_command == exposure_command
    end
    scc_step!(command, prepared, 2first, scc_metadata(2))
    @test FilterGraphAlgorithms.chopped_image_retained_image(prepared.workspace.state.difference) == vec(2first)
    scc_step!(command, prepared, zeros(Float32, 2, 2), scc_metadata(3; trigger=SAMPLE_TRIGGER_LOW))
    @test command ≈ prepared.plan.calibration.bias_command + Float32[-0.1, 0.2]
end

@testset "SCC protects phase calibration and exposure-command storage" begin
    prepared = prepared_scc()
    state, scratch = prepared.workspace.state, prepared.workspace.workspace
    for storage in (prepared.plan.calibration.fringed_response, prepared.plan.calibration.unfringed_response,
            state.committed_exposure_command, state.candidate_exposure_command,
            scratch.command_difference, scratch.phase_correction,
            state.difference.committed_image, state.difference.candidate_image,
            state.difference.staged_difference, scratch.image)
        command = view(vec(storage), 1:2)
        before = copy(storage)
        @test_throws ArgumentError scc_step!(command, prepared, zeros(Float32, 2, 2), scc_metadata(1))
        # Uninitialized difference scratch can contain NaNs before its first use.
        @test isequal(storage, before)
        @test state.committed_identity === nothing
    end
end

end
