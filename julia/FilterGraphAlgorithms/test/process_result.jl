module ProcessResultTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

@enum TestGateFailure::UInt8 begin
    TestGateRejected = 1
end

@enum TestMetadataFailure::UInt8 begin
    TestMetadataMissing = 1
end

struct TestGatePlan
    extent::Int
end

@algorithm TestGateF32 begin
    label = "test-gate-f32"
    failure = TestGateFailure

    @configuration TestGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] < 0.0f0 && return TestGateRejected
        copyto!(output, input)
        return nothing
    end
end

@algorithm InvalidFailureF32 begin
    label = "invalid-failure-f32"
    failure = UInt8

    @configuration InvalidFailureF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(:output, Float32, (configuration.extent,), "test.signal.f32/1"),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = nothing
end

@algorithm ConditionalGateF32 begin
    label = "conditional-gate-f32"

    @configuration ConditionalGateF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
            publication=:conditional,
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)

    process!(plan) = begin
        input[1] < 0.0f0 && return AlgorithmProcessDeferred()
        copyto!(output, input)
        return nothing
    end
end

@algorithm UndeclaredDeferredF32 begin
    label = "undeclared-deferred-f32"

    @configuration UndeclaredDeferredF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(:output, Float32, (configuration.extent,), "test.signal.f32/1"),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)
    process!(plan) = AlgorithmProcessDeferred()
end

@algorithm MetadataEchoF32 begin
    label = "metadata-echo-f32"
    failure = TestMetadataFailure

    @configuration MetadataEchoF32Configuration begin
        extent::Int => graph_rebuild
    end

    ports(configuration) = (
        input(
            :input,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            sample=:input_sample,
            rate=(4, 2),
        ),
        output(
            :output,
            Float32,
            (configuration.extent,),
            "test.signal.f32/1";
            metadata_from=:input,
            rate=(4, 2),
        ),
    )

    prepare(configuration) = TestGatePlan(configuration.extent)

    process!(plan) = begin
        isnothing(input_sample) && return TestMetadataMissing
        copyto!(output, input)
        return AlgorithmProcessComplete(SampleMetadata(input_sample; offset=0))
    end
end

function checked_allocation_bytes(output, prepared, input)
    return @allocated process!(output, prepared, input)
end

function checked_metadata_allocation_bytes(output, prepared, input, metadata)
    return @allocated process!(output, prepared, input, metadata)
end

@testset "compact Algorithm processing results" begin
    @test isbitstype(AlgorithmProcessResult)
    @test isbitstype(SampleMetadata)
    @test AlgorithmProcessComplete() != AlgorithmProcessDeferred()
    metadata = SampleMetadata(
        41,
        2;
        timestamp_nanoseconds=200,
        terminal=true,
        discontinuity=true,
        trigger_high=true,
    )
    @test metadata.sequence == 41
    @test metadata.offset == 2
    @test metadata.timestamp_nanoseconds == 200
    @test metadata.terminal
    @test metadata.discontinuity
    @test !metadata.corrupted
    @test metadata.has_timestamp
    @test sample_trigger_state(metadata) == SAMPLE_TRIGGER_HIGH
    @test sample_trigger_high(metadata) === true
    @test !SampleMetadata(1).has_timestamp
    @test !SampleMetadata(1).has_trigger_classification
    explicit_absent = SampleMetadata(1; trigger_state=SAMPLE_TRIGGER_ABSENT,
        chopped_image_kind=ChoppedImageAbsent)
    @test explicit_absent.has_trigger_classification
    @test SampleMetadata(explicit_absent; sequence=2).has_trigger_classification
    @test sample_chopped_image_kind(explicit_absent) === ChoppedImageAbsent
    @test !SampleMetadata(explicit_absent; trigger_high=nothing).has_trigger_classification
    @test !SampleMetadata(SampleMetadata(1); sequence=2).has_trigger_classification
    @test sample_trigger_state(SampleMetadata(1)) == SAMPLE_TRIGGER_ABSENT
    @test sample_trigger_high(SampleMetadata(1)) === nothing
    @test sample_trigger_high(SampleMetadata(metadata; trigger_high=nothing)) ===
          nothing
    @test sample_exposure_duration_nanoseconds(metadata) === nothing
    mixed = SampleMetadata(metadata; trigger_state=SAMPLE_TRIGGER_MIXED,
        exposure_duration_nanoseconds=UInt64(17))
    @test sample_trigger_state(mixed) == SAMPLE_TRIGGER_MIXED
    @test sample_trigger_high(mixed) === nothing
    @test sample_exposure_duration_nanoseconds(mixed) === UInt64(17)
    @test SampleMetadata(mixed; offset=3).trigger_state == SAMPLE_TRIGGER_MIXED
    @test sample_trigger_high(SampleMetadata(mixed; trigger_high=false)) === false
    @test sample_trigger_state(SampleMetadata(mixed; trigger_high=nothing)) == SAMPLE_TRIGGER_ABSENT
    @test sample_exposure_duration_nanoseconds(SampleMetadata(mixed;
        exposure_duration_nanoseconds=nothing)) === nothing
    @test_throws ArgumentError SampleMetadata(1; trigger_state=SAMPLE_TRIGGER_MIXED, trigger_high=true)
    @test_throws ArgumentError SampleMetadata(mixed; trigger_state=SAMPLE_TRIGGER_MIXED, trigger_high=true)
    @test_throws ArgumentError SampleMetadata(1; exposure_duration_nanoseconds=0)
    @test_throws ArgumentError SampleMetadata(1; exposure_duration_nanoseconds=-1)
    @test_throws ArgumentError SampleMetadata(1; exposure_duration_nanoseconds=big(typemax(UInt64)) + 1)
    identity = SampleAcquisitionIdentity(ntuple(i -> UInt8(i), 16), 7, 41)
    exposure = SampleExposureStart(900; clock=SAMPLE_CLOCK_TAI)
    acquired = SampleMetadata(mixed; acquisition_identity=identity,
        exposure_start=exposure)
    @test sample_acquisition_identity(acquired) == identity
    @test sample_exposure_start(acquired) == exposure
    @test SampleMetadata(acquired; discontinuity=true).acquisition_identity == identity
    @test SampleMetadata(acquired; sequence=42).acquisition_identity == identity
    @test_throws ArgumentError SampleAcquisitionIdentity(ntuple(_ -> UInt8(0), 16), 0, 1)
    @test SampleExposureStart(-1).nanoseconds == -1
    @test sample_trigger_high(SampleMetadata(1;
        trigger_state=SAMPLE_TRIGGER_HIGH)) === true
    optical = SampleMetadata(acquired; chopped_image_kind=ChoppedImageUnfringed)
    @test sample_chopped_image_kind(optical) == ChoppedImageUnfringed
    @test sample_chopped_image_kind(SampleMetadata(optical; offset=3)) == ChoppedImageUnfringed
    @test sample_chopped_image_kind(SampleMetadata(optical; chopped_image_kind=nothing)) === nothing
    @test sample_chopped_image_kind(metadata) === nothing
    @test SampleExposureStart(1; clock=SAMPLE_CLOCK_TAI).clock == SAMPLE_CLOCK_TAI
    stamp = SampleReferenceTimestamp(typemax(UInt64)-1; clock=SAMPLE_CLOCK_TAI, duration_nanoseconds=0)
    @test sample_reference_timestamp(SampleMetadata(acquired; reference_timestamp=stamp)) === stamp
    @test sample_reference_timestamp(SampleMetadata(SampleMetadata(1; reference_timestamp=stamp); offset=3)) === stamp
    @test sample_reference_timestamp(SampleMetadata(1)) === nothing
    @test sample_reference_timestamp(SampleMetadata(SampleMetadata(1; reference_timestamp=stamp); reference_timestamp=nothing)) === nothing
    @test SampleReferenceTimestamp(0).duration_nanoseconds == typemax(UInt64)
    @test !SampleReferenceTimestamp(0).has_duration
    @test isbitstype(SampleReferenceTimestamp)
    for retired in (:SampleAbsoluteTime, :SampleAbsoluteTimes, :SampleTimeEvent,
            :sample_absolute_times, :SampleExposureTimebase)
        @test !isdefined(FilterGraphAlgorithms, retired)
    end
    @test_throws ArgumentError SampleReferenceTimestamp(-1)
    @test_throws ArgumentError SampleReferenceTimestamp(typemax(UInt64))
    @test_throws ArgumentError SampleReferenceTimestamp(0; duration_nanoseconds=-1)
    @test_throws ArgumentError SampleReferenceTimestamp(0; duration_nanoseconds=typemax(UInt64))
    @test_throws ArgumentError SampleExposureStart(big(typemax(Int64))+1)
    @test isbitstype(SampleAcquisitionIdentity)
    @test isbitstype(SampleExposureStart)
    @test !AlgorithmProcessComplete().has_metadata
    @test AlgorithmProcessComplete(metadata).has_metadata
    @test AlgorithmProcessComplete(metadata).metadata == metadata
    @test !AlgorithmProcessDeferred().has_metadata
    @test_throws ArgumentError SampleMetadata(-1)
    @test_throws ArgumentError SampleMetadata(1, -1)
    @test_throws ArgumentError AlgorithmProcessResult(false, true, metadata)
    @test FilterGraphAlgorithms.algorithm_failure_type(TestGateF32) === TestGateFailure
    @test Base.ispublic(FilterGraphAlgorithms, :algorithm_failure_type)
    @test !Base.isexported(FilterGraphAlgorithms, :algorithm_failure_type)

    conditional_ports = algorithm_ports(ConditionalGateF32)
    @test !conditional_ports[1].conditional
    @test conditional_ports[2].conditional
    conditional = prepare_algorithm(
        ConditionalGateF32,
        ConditionalGateF32Configuration(extent=3),
    )
    conditional_output = fill(29.0f0, 3)
    @test process!(conditional_output, conditional, Float32[-1, 2, 3]) ===
          AlgorithmProcessDeferred()
    @test conditional_output == fill(29.0f0, 3)
    @test process!(conditional_output, conditional, Float32[1, 2, 3]) === nothing
    @test conditional_output == Float32[1, 2, 3]
    process!(conditional_output, conditional, Float32[-1, 2, 3])
    @test checked_allocation_bytes(
        conditional_output,
        conditional,
        Float32[-1, 2, 3],
    ) == 0

    undeclared = prepare_algorithm(
        UndeclaredDeferredF32,
        UndeclaredDeferredF32Configuration(extent=3),
    )
    @test_throws ArgumentError process!(
        conditional_output,
        undeclared,
        Float32[1, 2, 3],
    )

    prepared = prepare_algorithm(TestGateF32, TestGateF32Configuration(extent=3))
    output = fill(41.0f0, 3)
    accepted = Float32[1, 2, 3]
    rejected = Float32[-1, 2, 3]

    @test process!(output, prepared, accepted) === nothing
    @test output == accepted

    fill!(output, 41.0f0)
    @test process!(output, prepared, rejected) === TestGateRejected
    @test output == fill(41.0f0, 3)

    process!(output, prepared, accepted)
    process!(output, prepared, rejected)
    @test checked_allocation_bytes(output, prepared, accepted) == 0
    @test checked_allocation_bytes(output, prepared, rejected) == 0

    @test_throws ArgumentError prepare_algorithm(
        InvalidFailureF32,
        InvalidFailureF32Configuration(extent=3),
    )

    metadata_echo = prepare_algorithm(
        MetadataEchoF32,
        MetadataEchoF32Configuration(extent=3),
    )
    metadata_output = fill(51.0f0, 3)
    metadata_input = Float32[1, 2, 3]
    @test algorithm_formats(metadata_echo)[1].rate == (0x00000002, 0x00000001)
    @test algorithm_formats(metadata_echo)[2].rate == (0x00000002, 0x00000001)
    @test_throws ArgumentError AlgorithmFormat(
        Float32,
        (3,),
        "test.signal.f32/1";
        rate=(0, 1),
    )
    @test_throws ArgumentError AlgorithmFormat(
        Float32,
        (3,),
        "test.signal.f32/1";
        rate=(UInt64(typemax(UInt32)) + 1, 1),
    )
    @test process!(metadata_output, metadata_echo, metadata_input) === TestMetadataMissing
    @test metadata_output == fill(51.0f0, 3)
    result = process!(
        metadata_output,
        metadata_echo,
        metadata_input,
        (input=metadata,),
    )
    @test result == AlgorithmProcessComplete(SampleMetadata(metadata; offset=0))
    @test metadata_output == metadata_input
    @test_throws ArgumentError process!(
        metadata_output,
        metadata_echo,
        metadata_input,
        (wrong=metadata,),
    )
    @test_throws ArgumentError process!(
        metadata_output,
        metadata_echo,
        metadata_input,
        (input=41,),
    )
    process!(metadata_output, metadata_echo, metadata_input, (input=metadata,))
    @test checked_metadata_allocation_bytes(
        metadata_output,
        metadata_echo,
        metadata_input,
        (input=metadata,),
    ) == 0
end

end
