module ChoppedImageDifferenceTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

struct ZeroBasedChoppedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end

Base.size(vector::ZeroBasedChoppedVector) = size(vector.parent)
Base.axes(vector::ZeroBasedChoppedVector) = (0:(length(vector.parent) - 1),)
Base.IndexStyle(::Type{<:ZeroBasedChoppedVector}) = IndexLinear()
Base.getindex(vector::ZeroBasedChoppedVector, index::Int) =
    vector.parent[index + 1]
Base.setindex!(vector::ZeroBasedChoppedVector, value, index::Int) =
    setindex!(vector.parent, value, index + 1)

chopped_prepared(
    pixel_count::Int=2;
    trigger_high_is_fringed::Bool=true,
) = prepare_algorithm(
    ChoppedImageDifferenceF32,
    ChoppedImageDifferenceF32Configuration(
        pixel_count=pixel_count,
        trigger_high_is_fringed=trigger_high_is_fringed,
    ),
)

chopped_metadata(sequence::Integer, trigger_high::Union{Nothing,Bool}) = (
    image=SampleMetadata(sequence; trigger_high),
    input_validity=nothing,
)

function chopped_checked_allocation_bytes(outputs, prepared, inputs, metadata)
    return @allocated process!(outputs..., prepared, inputs..., metadata)
end

function chopped_admitted_allocation_bytes(
    outputs,
    prepared,
    inputs,
    metadata,
)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        inputs...,
        FilterGraphAlgorithms._admitted_sample_metadata(metadata.image),
        FilterGraphAlgorithms._admitted_sample_metadata(
            metadata.input_validity,
        ),
    )
end

@testset "chopped-image declaration uses in-band trigger metadata" begin
    @test algorithm_label(ChoppedImageDifferenceF32) ==
          "chopped-image-difference-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ChoppedImageDifferenceF32,
    ) === ChoppedImageDifferenceFailure
    @test algorithm_property_policy(ChoppedImageDifferenceF32) isa
          FilterGraphAlgorithms.NoDeclaredProperties
    configuration = algorithm_configuration(ChoppedImageDifferenceF32)
    @test map(field -> field.name, configuration) ==
          (:pixel_count, :trigger_high_is_fringed, :input_schema, :require_acquisition_identity)
    @test configuration[1].update isa GraphRebuildConfiguration
    @test configuration[2].update isa ConstructionConfiguration

    ports = algorithm_ports(ChoppedImageDifferenceF32)
    @test map(port -> port.name, ports) == (
        :image,
        Symbol("input-validity"),
        :difference,
        Symbol("output-validity"),
        Symbol("pair-status"),
    )
    @test getproperty.(ports, :element_type) ==
          (Float32, Bool, Float32, Bool, UInt8)
    @test all(port -> isnothing(port.metadata_source), ports[1:2])
    @test all(port -> port.metadata_source === :image, ports[3:5])

    formats = algorithm_formats(chopped_prepared())
    @test getproperty.(formats, :shape) ==
          ((2,), (2,), (2,), (2,), (1,))
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1,
        CALIBRATED_PIXEL_VALIDITY_V1,
        SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
        SCC_CHOPPED_DIFFERENCE_STATUS_V1,
    )
end

@testset "alternating chopped frames match the frozen rolling oracle" begin
    prepared = chopped_prepared()
    difference = fill(9.0f0, 2)
    output_validity = fill(true, 2)
    pair_status = fill(typemax(UInt8), 1)
    outputs = (difference, output_validity, pair_status)
    valid = fill(true, 2)

    frames = (
        (Float32[10, 20], true, Float32[0, 0], Bool[false, false], UInt8[0]),
        (Float32[3, 8], false, Float32[7, 12], Bool[true, true], UInt8[1]),
        (Float32[12, 18], true, Float32[9, 10], Bool[true, true], UInt8[1]),
        (Float32[4, 7], false, Float32[8, 11], Bool[true, true], UInt8[1]),
    )
    for (sequence, frame) in enumerate(frames)
        image, trigger_high, expected, expected_validity, expected_status = frame
        @test process!(
            outputs...,
            prepared,
            image,
            valid,
            chopped_metadata(sequence, trigger_high),
        ) === nothing
        @test difference == expected
        @test output_validity == expected_validity
        @test pair_status == expected_status
    end
    @test chopped_image_is_primed(prepared.workspace)
    @test chopped_image_retained_kind(prepared.workspace) ==
          ChoppedImageUnfringed
    @test chopped_image_retained_image(prepared.workspace) == Float32[4, 7]
    @test chopped_image_retained_validity(prepared.workspace) == Bool[true, true]

    reverse = chopped_prepared(1)
    reverse_outputs = (zeros(Float32, 1), fill(false, 1), zeros(UInt8, 1))
    process!(
        reverse_outputs...,
        reverse,
        Float32[2],
        Bool[true],
        chopped_metadata(1, false),
    )
    process!(
        reverse_outputs...,
        reverse,
        Float32[5],
        Bool[true],
        chopped_metadata(2, true),
    )
    @test reverse_outputs[1] == Float32[3]
    @test reverse_outputs[3] == UInt8[1]

    inverted = chopped_prepared(1; trigger_high_is_fringed=false)
    inverted_outputs = (zeros(Float32, 1), fill(false, 1), zeros(UInt8, 1))
    process!(
        inverted_outputs...,
        inverted,
        Float32[2],
        Bool[true],
        chopped_metadata(1, true),
    )
    process!(
        inverted_outputs...,
        inverted,
        Float32[5],
        Bool[true],
        chopped_metadata(2, false),
    )
    @test inverted_outputs[1] == Float32[3]

    for polarity in (false, true)
        optical = chopped_prepared(1; trigger_high_is_fringed=polarity)
        optical_outputs = (zeros(Float32, 1), fill(false, 1), zeros(UInt8, 1))
        # Explicit optical roles determine the sign even when electrical tags disagree.
        for (sequence, value, kind) in ((1, 2.0f0, ChoppedImageUnfringed),
                (2, 5.0f0, ChoppedImageFringed))
            metadata = SampleMetadata(sequence; trigger_high=true, chopped_image_kind=kind)
            @test process!(optical_outputs..., optical, Float32[value], Bool[true],
                (image=metadata, input_validity=nothing)) === nothing
        end
        @test optical_outputs[1] == Float32[3]
        mixed = SampleMetadata(3; trigger_high=true, chopped_image_kind=ChoppedImageMixed)
        @test process!(optical_outputs..., optical, Float32[9], Bool[true],
            (image=mixed, input_validity=nothing)) === ChoppedImageTriggerMissing
    end
end

@testset "chopped retained image uses an Algorithm-local candidate" begin
    plan = ChoppedImageDifferencePlan(Float64, 1)
    workspace = ChoppedImageDifferenceWorkspace(plan)
    outputs = (fill(71.0, 1), fill(true, 1), fill(typemax(UInt8), 1))

    @test process!(
        outputs...,
        workspace,
        plan,
        [10.0],
        Bool[true],
        true,
    ) === nothing
    @test !chopped_image_is_primed(workspace)
    @test discard!(workspace) === workspace
    @test !chopped_image_is_primed(workspace)

    process!(outputs..., workspace, plan, [10.0], Bool[true], true)
    @test commit!(workspace) === workspace
    @test chopped_image_retained_image(workspace) == [10.0]

    process!(outputs..., workspace, plan, [3.0], Bool[true], false)
    @test outputs[1] == [7.0]
    @test discard!(workspace) === workspace
    @test chopped_image_retained_image(workspace) == [10.0]

    process!(outputs..., workspace, plan, [4.0], Bool[true], false)
    @test outputs[1] == [6.0]
    commit!(workspace)
    @test chopped_image_retained_image(workspace) == [4.0]

    process!(outputs..., workspace, plan, [5.0], Bool[true], false)
    @test outputs[1] == [0.0]
    @test outputs[3] == UInt8[0]
    commit!(workspace)
    @test chopped_image_retained_image(workspace) == [5.0]

    @test reset!(workspace) === workspace
    @test !chopped_image_is_primed(workspace)
    @test isnothing(chopped_image_retained_kind(workspace))
    @test isnothing(chopped_image_retained_image(workspace))
    @test isnothing(chopped_image_retained_validity(workspace))
end

@testset "invalid and non-finite pixels retain pair-level completion" begin
    prepared = chopped_prepared(4)
    outputs = (fill(9.0f0, 4), fill(true, 4), fill(typemax(UInt8), 1))
    process!(
        outputs...,
        prepared,
        Float32[2, NaN, floatmax(Float32), 4],
        Bool[true, true, true, false],
        chopped_metadata(1, true),
    )
    @test chopped_image_retained_validity(prepared.workspace) ==
          Bool[true, false, true, false]
    process!(
        outputs...,
        prepared,
        Float32[1, 0, -floatmax(Float32), 1],
        fill(true, 4),
        chopped_metadata(2, false),
    )
    @test outputs[1] == Float32[1, 0, 0, 0]
    @test outputs[2] == Bool[true, false, false, false]
    @test outputs[3] == UInt8[1]
end

@testset "dense Float32 chopping preserves pixel bits and retained history" begin
    count = 1031
    plan = ChoppedImageDifferencePlan(Float32, count)
    # Leave an ordinary run between the exceptional pixels at both ends.
    edge_cases = (
        (-0f0, 0f0, true, true),
        (0f0, -0f0, true, true),
        (nextfloat(0f0), 0f0, true, true),
        (floatmax(Float32), -floatmax(Float32), true, true),
        (reinterpret(Float32, 0x7fc00042), 1f0, true, true),
        (1f0, -Inf32, true, true),
        (2f0, 1f0, false, true),
        (2f0, 1f0, true, false),
        (-2f0, 3f0, true, true),
    )
    lanes = fill(last(edge_cases), count)
    lanes[1:length(edge_cases)] .= edge_cases
    lanes[end-length(edge_cases)+1:end] .= edge_cases
    fringed, unfringed = getindex.(lanes, 1), getindex.(lanes, 2)
    fringed_flags, unfringed_flags = getindex.(lanes, 3), getindex.(lanes, 4)
    expected = map(lanes) do (f, u, fv, uv)
        valid = fv && uv && isfinite(f) && isfinite(u)
        subtraction = valid ? f - u : 0f0
        valid &= isfinite(subtraction)
        return (valid ? subtraction : 0f0, valid)
    end
    expected_difference = first.(expected)
    expected_validity = last.(expected)

    for trigger_high in (false, true), dense in (false, true)
        previous, previous_flags, current, current_flags = trigger_high ?
            (unfringed, unfringed_flags, fringed, fringed_flags) :
            (fringed, fringed_flags, unfringed, unfringed_flags)
        workspace = ChoppedImageDifferenceWorkspace(plan)
        status = UInt8[99]
        process!(zeros(Float32, count), fill(false, count), status,
            workspace, plan, previous, previous_flags, !trigger_high)
        @test status == UInt8[0]
        commit!(workspace)

        # Publish into the input arrays themselves to exercise staging.
        image_storage = zeros(Float32, dense ? count : 2count)
        flag_storage = fill(false, length(image_storage))
        image = dense ? image_storage : view(image_storage, 1:2:2count)
        validity = dense ? flag_storage : view(flag_storage, 1:2:2count)
        copyto!(image, current)
        copyto!(validity, current_flags)
        process!(image, validity, status, workspace, plan, image, validity, trigger_high)
        @test status == UInt8[1]
        @test reinterpret(UInt32, collect(image)) == reinterpret(UInt32, expected_difference)
        @test validity == expected_validity
        @test reinterpret(UInt32, chopped_image_retained_image(workspace)) ==
              reinterpret(UInt32, previous)
        commit!(workspace)
        @test reinterpret(UInt32, chopped_image_retained_image(workspace)) ==
              reinterpret(UInt32, current)
        @test chopped_image_retained_validity(workspace) == current_flags .& isfinite.(current)
    end
end

@testset "shifted borrowed storage preserves scalar staging order" begin
    pixel_count = 513
    plan = ChoppedImageDifferencePlan(Float32, pixel_count)
    workspace = ChoppedImageDifferenceWorkspace(plan)
    output = (zeros(Float32, pixel_count), fill(false, pixel_count), UInt8[99])
    valid = fill(true, pixel_count)
    previous = fill(600f0, pixel_count)
    process!(output..., workspace, plan, previous, valid, true)
    commit!(workspace)

    storage = Float32.(1:(pixel_count + 1))
    expected = previous .- storage[1:pixel_count]
    GC.@preserve storage begin
        borrowed = unsafe_wrap(Vector{Float32}, pointer(storage), pixel_count)
        workspace.candidate_image = unsafe_wrap(Vector{Float32}, pointer(storage, 2), pixel_count)
        @test !Base.mightalias(borrowed, workspace.candidate_image)
        process!(output..., workspace, plan, borrowed, valid, false)
        @test output[1] == expected
        @test all(output[2])
        @test output[3] == UInt8[1]
    end
end

@testset "chopped rejection and storage checks preserve ownership" begin
    @test_throws ArgumentError ChoppedImageDifferencePlan(Float32, 0)
    prepared = chopped_prepared()
    outputs = (fill(71.0f0, 2), fill(true, 2), fill(typemax(UInt8), 1))
    process!(
        outputs...,
        prepared,
        Float32[1, 2],
        Bool[true, true],
        chopped_metadata(1, true),
    )
    retained = (
        chopped_image_retained_kind(prepared.workspace),
        chopped_image_retained_image(prepared.workspace),
        chopped_image_retained_validity(prepared.workspace),
    )
    retained_outputs = map(copy, outputs)
    @test process!(
        outputs...,
        prepared,
        Float32[0, 0],
        Bool[true, true],
    ) === ChoppedImageMetadataMissing
    @test process!(
        outputs...,
        prepared,
        Float32[0, 0],
        Bool[true, true],
        chopped_metadata(2, nothing),
    ) === ChoppedImageTriggerMissing
    @test outputs == retained_outputs
    @test (
        chopped_image_retained_kind(prepared.workspace),
        chopped_image_retained_image(prepared.workspace),
        chopped_image_retained_validity(prepared.workspace),
    ) == retained

    plan = prepared.plan
    workspace = prepared.workspace
    @test_throws DimensionMismatch process!(
        outputs...,
        workspace,
        plan,
        Float32[0],
        Bool[true, true],
        true,
    )
    @test_throws ArgumentError process!(
        outputs...,
        workspace,
        plan,
        ZeroBasedChoppedVector(Float32[0, 0]),
        Bool[true, true],
        true,
    )
    @test_throws ArgumentError process!(
        workspace.staged_difference,
        outputs[2],
        outputs[3],
        workspace,
        plan,
        Float32[0, 0],
        Bool[true, true],
        true,
    )
end

@testset "chopped views and safe input-output aliasing preserve order" begin
    plan = ChoppedImageDifferencePlan(Float32, 2)
    workspace = ChoppedImageDifferenceWorkspace(plan)
    difference_storage = fill(71.0f0, 4)
    difference = @view difference_storage[1:2:3]
    validity_storage = fill(false, 4)
    output_validity = @view validity_storage[1:2:3]
    status = zeros(UInt8, 1)
    image_storage = Float32[99, 10, 20, 99]
    image = @view image_storage[2:3]
    input_validity = @view Bool[false, true, true, false][2:3]
    process!(
        difference,
        output_validity,
        status,
        workspace,
        plan,
        image,
        input_validity,
        true,
    )
    commit!(workspace)
    aliased_image = Float32[3, 8]
    process!(
        aliased_image,
        output_validity,
        status,
        workspace,
        plan,
        aliased_image,
        input_validity,
        false,
    )
    @test aliased_image == Float32[7, 12]
    @test output_validity == Bool[true, true]
    @test status == UInt8[1]
end

@testset "chopped warmed outcomes and lifecycle allocate nothing" begin
    prepared = chopped_prepared(8)
    outputs = (zeros(Float32, 8), fill(false, 8), zeros(UInt8, 1))
    valid = (fill(2.0f0, 8), fill(true, 8))
    valid_metadata = chopped_metadata(1, true)
    missing_trigger_metadata = chopped_metadata(2, nothing)
    process!(outputs..., prepared, valid..., valid_metadata)
    process!(outputs..., prepared, valid...)
    process!(outputs..., prepared, valid..., missing_trigger_metadata)
    FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        valid...,
        FilterGraphAlgorithms._admitted_sample_metadata(valid_metadata.image),
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
    discard!(prepared.workspace)
    FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        valid...,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )

    @test chopped_checked_allocation_bytes(
        outputs,
        prepared,
        valid,
        valid_metadata,
    ) == 0
    @test (@allocated process!(outputs..., prepared, valid...)) == 0
    @test chopped_checked_allocation_bytes(
        outputs,
        prepared,
        valid,
        missing_trigger_metadata,
    ) == 0
    @test chopped_admitted_allocation_bytes(
        outputs,
        prepared,
        valid,
        valid_metadata,
    ) == 0
    discard!(prepared.workspace)
    @test chopped_admitted_allocation_bytes(
        outputs,
        prepared,
        valid,
        (image=nothing, input_validity=nothing),
    ) == 0
    @test (@allocated commit!(prepared.workspace)) == 0
    @test (@allocated discard!(prepared.workspace)) == 0
    @test (@allocated reset!(prepared.workspace)) == 0
end

@testset "chopped-image public API exports" begin
    for name in (
        :CALIBRATED_PIXEL_VALIDITY_V1,
        :SCC_CHOPPED_DIFFERENCE_STATUS_V1,
        :SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
        :SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        :ChoppedImageDifferenceF32,
        :ChoppedImageDifferenceF32Configuration,
        :ChoppedImageDifferenceFailure,
        :ChoppedImageDifferencePlan,
        :ChoppedImageDifferenceWorkspace,
        :ChoppedImageKind,
        :ChoppedImagePairStatus,
        :ChoppedImageMetadataMissing,
        :ChoppedImageTriggerMissing,
        :ChoppedImageSequenceInvalid,
        :chopped_image_is_primed,
        :chopped_image_retained_image,
        :chopped_image_retained_kind,
        :chopped_image_retained_validity,
    )
        @test Base.isexported(FilterGraphAlgorithms, name)
    end
end

@testset "full detector difference uses adjacent optical exposures" begin
    prepared = prepare_algorithm(ChoppedDetectorDifferenceF32,
        ChoppedDetectorDifferenceF32Configuration(
            image_size=(2, 3), input_schema="test.detector-electrons-f32/1"))
    difference, status = fill(99f0, 2, 3), UInt8[99]
    fringed = Float32[8 NaN 12; 9 10 13]
    unfringed = Float32[3 2 5; 1 2 6]
    domain = ntuple(UInt8, 16)
    start = typemax(UInt64) - 10
    function publish(sequence, kind, image; publication=sequence, trigger_high=true)
        metadata = SampleMetadata(publication; terminal=true, trigger_high,
            chopped_image_kind=kind, exposure_duration_nanoseconds=10,
            acquisition_identity=SampleAcquisitionIdentity(domain, 3, sequence))
        return process!(difference, status, prepared, image, (image=metadata,))
    end
    @test publish(start, ChoppedImageFringed, fringed).complete
    @test status == [0] && all(isnan, difference)
    result = publish(start + 1, ChoppedImageUnfringed, unfringed)
    @test result.complete && result.has_metadata
    @test status == [1]
    @test isequal(difference, Float32[5 NaN 7; 8 8 7])
    @test sample_acquisition_identity(result.metadata).sequence == start + 1
    @test sample_chopped_image_kind(result.metadata) === nothing
    @test sample_trigger_high(result.metadata) === nothing
    @test publish(start + 1, ChoppedImageFringed, fringed; publication=start + 2) ===
          ChoppedImageSequenceInvalid
    @test publish(start + 3, ChoppedImageFringed, fringed).complete
    @test status == [0] && all(isnan, difference)
    @test publish(start + 4, nothing, unfringed) === ChoppedImageTriggerMissing
    @test publish(start + 4, ChoppedImageUnfringed, unfringed).complete
    @test status == [1]
end

@testset "acquisition-aware chopped continuity stages real exposures" begin
    prepared = prepare_algorithm(ChoppedImageDifferenceF32,
        ChoppedImageDifferenceF32Configuration(pixel_count=2, trigger_high_is_fringed=true,
            input_schema="test.relative-intensity-f32/1", require_acquisition_identity=true))
    outputs = (fill(99f0, 2), trues(2), UInt8[99])
    domain = ntuple(index -> UInt8(index), 16)
    function accept(sequence, kind; generation=1, source_domain=domain, discontinuity=false, corrupted=false, publication=sequence, duration=17)
        metadata = SampleMetadata(publication; terminal=true, chopped_image_kind=kind,
            discontinuity, corrupted, exposure_duration_nanoseconds=duration,
            acquisition_identity=SampleAcquisitionIdentity(source_domain, generation, sequence))
        process!(outputs..., prepared, fill(Float32(sequence), 2), trues(2),
            (image=metadata, input_validity=nothing))
    end
    @test algorithm_formats(prepared)[1].schema == "test.relative-intensity-f32/1"
    @test accept(1, ChoppedImageFringed) === nothing
    @test outputs[3] == UInt8[0]
    @test accept(2, ChoppedImageUnfringed) === nothing
    @test outputs[1] == Float32[-1, -1] && outputs[3] == UInt8[1]
    retained = map(copy, outputs)
    @test accept(2, ChoppedImageFringed; publication=200) === ChoppedImageSequenceInvalid
    @test accept(3, ChoppedImageFringed; duration=18) === FilterGraphAlgorithms.ChoppedImageExposureMismatch
    @test accept(3, ChoppedImageFringed; duration=nothing) === ChoppedImageMetadataMissing
    @test accept(3, ChoppedImageAbsent) === ChoppedImageTriggerMissing
    @test outputs == retained
    @test accept(4, ChoppedImageFringed; publication=400) === nothing
    @test outputs[3] == UInt8[0]
    @test accept(5, ChoppedImageUnfringed) === nothing
    @test outputs[3] == UInt8[1]
    @test accept(6, ChoppedImageFringed; discontinuity=true) === nothing
    @test outputs[3] == UInt8[0]
    @test accept(7, ChoppedImageUnfringed) === nothing
    @test outputs[3] == UInt8[1]
    @test accept(8, ChoppedImageMixed) === nothing
    @test outputs[3] == UInt8[0] && !chopped_image_is_primed(prepared.workspace)
    @test accept(9, ChoppedImageFringed) === nothing
    @test outputs[3] == UInt8[0]
    @test accept(10, ChoppedImageUnfringed) === nothing
    @test outputs[3] == UInt8[1]
    @test accept(1, ChoppedImageFringed; generation=2) === nothing
    @test accept(11, ChoppedImageUnfringed) === ChoppedImageSequenceInvalid
    @test outputs[3] == UInt8[0]
    @test accept(2, ChoppedImageUnfringed; generation=2, corrupted=true) === nothing
    @test !chopped_image_is_primed(prepared.workspace)
    @test accept(3, ChoppedImageFringed; generation=2) === nothing
    @test outputs[3] == UInt8[0]
    @test accept(4, ChoppedImageUnfringed; generation=2) === nothing
    @test outputs[3] == UInt8[1]
    @test process!(outputs..., prepared, ones(Float32, 2), trues(2),
        (image=SampleMetadata(5; trigger_high=true), input_validity=nothing)) ===
        ChoppedImageMetadataMissing
    @test accept(5, nothing; generation=2, corrupted=true) === ChoppedImageTriggerMissing
    @test accept(5, ChoppedImageMixed; generation=2, corrupted=true) === nothing
    @test !chopped_image_is_primed(prepared.workspace)
    @test accept(1, ChoppedImageFringed; source_domain=ntuple(_ -> UInt8(9), 16)) === nothing
    @test outputs[3] == UInt8[0]
    reset!(prepared)
    @test prepared.workspace.committed_identity === nothing
end

@testset "explicit optical mixed acquisition clears history before fresh pure pairing" begin
    domain = ntuple(index -> UInt8(index), 16)
    metadata(sequence, trigger) = (image=SampleMetadata(sequence; terminal=true,
        trigger_state=trigger, exposure_duration_nanoseconds=17,
        chopped_image_kind=trigger == SAMPLE_TRIGGER_HIGH ? ChoppedImageFringed :
            trigger == SAMPLE_TRIGGER_LOW ? ChoppedImageUnfringed : ChoppedImageMixed,
        acquisition_identity=SampleAcquisitionIdentity(domain, 1, sequence)),
        input_validity=nothing)
    function prepared_pair(strict)
        prepare_algorithm(ChoppedImageDifferenceF32,
            ChoppedImageDifferenceF32Configuration(pixel_count=2,
                trigger_high_is_fringed=true, require_acquisition_identity=strict))
    end
    prepared = prepared_pair(true)
    outputs = (zeros(Float32, 2), falses(2), UInt8[0])
    @test process!(outputs..., prepared, ones(Float32, 2), trues(2),
        metadata(1, SAMPLE_TRIGGER_HIGH)) === nothing
    @test process!(outputs..., prepared, fill(2f0, 2), trues(2),
        metadata(2, SAMPLE_TRIGGER_MIXED)) === nothing
    @test !chopped_image_is_primed(prepared.workspace)
    @test prepared.workspace.committed_identity.sequence == 2
    @test outputs[3] == UInt8[UInt8(ChoppedImagePriming)]
    @test all(iszero, outputs[1]) && !any(outputs[2])
    @test process!(outputs..., prepared, fill(3f0, 2), trues(2),
        metadata(3, SAMPLE_TRIGGER_LOW)) === nothing
    @test outputs[3] == UInt8[UInt8(ChoppedImagePriming)]
    @test process!(outputs..., prepared, fill(4f0, 2), trues(2),
        metadata(4, SAMPLE_TRIGGER_HIGH)) === nothing
    @test outputs[3] == UInt8[UInt8(ChoppedImageComplete)]
    @test outputs[1] == ones(Float32, 2) && all(outputs[2])

    legacy = prepared_pair(false)
    @test process!(outputs..., legacy, ones(Float32, 2), trues(2),
        metadata(1, SAMPLE_TRIGGER_HIGH)) === nothing
    retained = map(copy, outputs)
    @test process!(outputs..., legacy, fill(2f0, 2), trues(2),
        metadata(2, SAMPLE_TRIGGER_MIXED)) === ChoppedImageTriggerMissing
    @test chopped_image_is_primed(legacy.workspace)
    @test chopped_image_retained_image(legacy.workspace) == ones(Float32, 2)
    @test outputs == retained
end


@testset "discard preserves rolling exposure duration" begin
    prepared = prepare_algorithm(ChoppedImageDifferenceF32,
        ChoppedImageDifferenceF32Configuration(pixel_count=2, trigger_high_is_fringed=true,
            input_schema="test.relative-intensity-f32/1", require_acquisition_identity=true))
    outputs = (zeros(Float32, 2), trues(2), UInt8[0])
    domain = ntuple(UInt8, 16)
    metadata(sequence, kind) = SampleMetadata(sequence; terminal=true, chopped_image_kind=kind,
        exposure_duration_nanoseconds=17,
        acquisition_identity=SampleAcquisitionIdentity(domain, 1, sequence))
    first = metadata(1, ChoppedImageFringed)
    second = metadata(2, ChoppedImageUnfringed)
    @test process!(outputs..., prepared, Float32[3,4], trues(2), (image=first, input_validity=nothing)) === nothing
    @test prepared.workspace.committed_duration == 17
    @test FilterGraphAlgorithms._process_admitted!(outputs..., prepared, Float32[2,3], trues(2),
        FilterGraphAlgorithms._admitted_sample_metadata(second), FilterGraphAlgorithms._NO_SAMPLE_METADATA) === nothing
    FilterGraphAlgorithms.discard!(prepared.workspace)
    @test prepared.workspace.committed_duration == 17
    @test process!(outputs..., prepared, Float32[2,3], trues(2), (image=second, input_validity=nothing)) === nothing
    @test outputs[1] == ones(Float32, 2)
    @test all(outputs[2])
    @test only(outputs[3]) == UInt8(ChoppedImageComplete)
end

end
