module Cred2ChoppedImageKindTests

using FilterGraphAlgorithms
using Test

cred2_metadata_allocation_bytes(metadata, raw) =
    @allocated cred2_trigger_metadata(metadata, raw)

alternating_metadata_allocation_bytes(metadata, even_is_high) =
    @allocated alternating_trigger_metadata(metadata; even_is_high)

chopped_metadata_allocation_bytes(metadata, polarity) =
    @allocated chopped_image_metadata(metadata; trigger_high_is_fringed=polarity)

@testset "C-RED 2 tag becomes in-band trigger metadata" begin
    metadata = SampleMetadata(
        42,
        0;
        timestamp_nanoseconds=1234,
        terminal=true,
    )
    @test sample_trigger_high(metadata) === nothing

    raw = zeros(UInt16, 8)
    raw[4] = 0x3ffe
    low = cred2_trigger_metadata(metadata, raw)
    @test low isa SampleMetadata
    @test sample_trigger_high(low) === false
    @test low.sequence == metadata.sequence
    @test low.timestamp_nanoseconds == metadata.timestamp_nanoseconds
    @test low.terminal == metadata.terminal

    raw[4] = 0x3fff
    high = cred2_trigger_metadata(metadata, raw)
    @test high isa SampleMetadata
    @test sample_trigger_high(high) === true

    raw[4] = 0x0001
    @test cred2_trigger_metadata(metadata, raw) === Cred2UnexpectedImageTag
    @test sample_trigger_high(metadata) === nothing
end

@testset "C-RED 2 electrical tags and application optical roles remain separate" begin
    domain = ntuple(i -> UInt8(i), 16)
    raw = UInt16[0, 0, 0, 0x3fff]
    let
        identity = SampleAcquisitionIdentity(domain, 7, 42)
        exposure = SampleExposureStart(900)
        metadata = SampleMetadata(42;
            acquisition_identity=identity, exposure_start=exposure,
            exposure_duration_nanoseconds=17, terminal=true)
        high = cred2_trigger_metadata(metadata, raw)
        @test sample_trigger_high(high) === true
        @test sample_chopped_image_kind(high) === nothing
        for polarity in (false, true)
            optical = chopped_image_metadata(high; trigger_high_is_fringed=polarity)
            @test sample_chopped_image_kind(optical) ==
                (polarity ? ChoppedImageFringed : ChoppedImageUnfringed)
            @test sample_trigger_high(optical) === true
            @test sample_acquisition_identity(optical) == identity
            @test sample_exposure_start(optical) == exposure
            @test sample_exposure_duration_nanoseconds(optical) == 17
            @test optical.sequence == 42 && optical.terminal
            @test chopped_metadata_allocation_bytes(high, polarity) == 0
        end
    end
    @test_throws UndefKeywordError chopped_image_metadata(SampleMetadata(1; trigger_high=true))
    for (trigger, expected) in ((SAMPLE_TRIGGER_LOW, ChoppedImageFringed),
            (SAMPLE_TRIGGER_HIGH, ChoppedImageUnfringed),
            (SAMPLE_TRIGGER_ABSENT, nothing), (SAMPLE_TRIGGER_MIXED, ChoppedImageMixed))
        source = SampleMetadata(1; trigger_state=trigger, chopped_image_kind=ChoppedImageFringed)
        optical = chopped_image_metadata(source; trigger_high_is_fringed=false)
        @test sample_chopped_image_kind(optical) === expected
        @test sample_trigger_state(optical) == trigger
    end
    raw[4] = 0x3ffe
    low = cred2_trigger_metadata(SampleMetadata(42), raw)
    @test sample_trigger_high(low) === false
    @test sample_chopped_image_kind(chopped_image_metadata(low;
        trigger_high_is_fringed=true)) == ChoppedImageUnfringed
    optical = chopped_image_metadata(SampleMetadata(42; trigger_high=true);
        trigger_high_is_fringed=true)
    retagged = cred2_trigger_metadata(optical, raw)
    @test sample_trigger_high(retagged) === false
    @test sample_chopped_image_kind(retagged) == ChoppedImageFringed
    @test sample_chopped_image_kind(chopped_image_metadata(retagged;
        trigger_high_is_fringed=true)) == ChoppedImageUnfringed
end

@testset "C-RED 2 tag validation accepts logical array views" begin
    metadata = SampleMetadata(1)
    raw_storage = UInt16[99, 0, 0, 0, 0x3fff, 0, 0, 0, 0, 99]
    raw = @view raw_storage[2:9]
    @test sample_trigger_high(cred2_trigger_metadata(metadata, raw)) === true
    @test_throws DimensionMismatch cred2_trigger_metadata(
        metadata,
        UInt16[0x3fff],
    )
end

@testset "frame identity produces the same in-band trigger metadata" begin
    for (sequence, expected) in ((0, true), (1, false), (42, true), (43, false))
        metadata = alternating_trigger_metadata(SampleMetadata(sequence))
        @test sample_trigger_high(metadata) === expected
    end
    @test sample_trigger_high(
        alternating_trigger_metadata(SampleMetadata(2); even_is_high=false),
    ) === false
    @test sample_trigger_high(
        alternating_trigger_metadata(SampleMetadata(3); even_is_high=false),
    ) === true
end

@testset "ingress trigger annotation allocates nothing after warmup" begin
    metadata = SampleMetadata(42)
    valid = zeros(UInt16, 8)
    valid[4] = 0x3fff
    invalid = copy(valid)
    invalid[4] = 0x0001
    cred2_trigger_metadata(metadata, valid)
    cred2_trigger_metadata(metadata, invalid)
    alternating_trigger_metadata(metadata)

    @test cred2_metadata_allocation_bytes(metadata, valid) == 0
    @test cred2_metadata_allocation_bytes(metadata, invalid) == 0
    @test alternating_metadata_allocation_bytes(metadata, true) == 0
    @test alternating_metadata_allocation_bytes(metadata, false) == 0
end

@testset "in-band trigger metadata public API exports" begin
    for name in (
        :Cred2ChoppedImageKindFailure,
        :Cred2UnexpectedImageTag,
        :alternating_trigger_metadata,
        :cred2_trigger_metadata,
        :sample_trigger_high,
        :sample_chopped_image_kind,
        :chopped_image_metadata,
        :ChoppedImageMixed,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

end
