module SccCorrectionPipeWireTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!, reset!
import FilterGraphPipeWire

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

function scc_borrowed_allocation_bytes(output, prepared, fringed, unfringed, correction)
    first = scc_metadata(1)
    second = scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)
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

function scc_borrowed_rolling_allocation_bytes(output, prepared, fringed, unfringed)
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

@testset "SCC rectangular row-major borrowed camera frames" begin
    shape = (2, 3)
    reference = Float32[0.01 0.02 0.03; 0.04 0.05 0.06]
    mask = Bool[false true false; true false true]
    matrix = zeros(Float32, 3, 3)
    matrix[1, 1] = 1.0f0
    matrix[2, 2] = 2.0f0
    matrix[3, 3] = 3.0f0
    calibration = SccCalibration(; frame_shape=shape, reference_difference=reference,
        correction_mask=mask, reconstructor=matrix, coordinate_order=Tuple(1:3),
        fringed_response=zeros(Float32, 3, 3), unfringed_response=zeros(Float32, 3, 3),
        command_unit="normalized-bax307", reconstructor_sign=:command_error,
        exposure_duration_nanoseconds=SCC_DURATION_NS, image_schema=SCC_IMAGE_SCHEMA,
        bias_command=zeros(Float32, 3), lower_limits=fill(-1.0f0, 3), upper_limits=fill(1.0f0, 3))
    prepared = prepared_scc(; calibration)
    @test algorithm_formats(prepared)[1].shape == shape
    @test calibration.residual.selected_indices == [2, 3, 6]
    unfringed = Float32[0.2 0.3 0.4; 0.5 0.6 0.7]
    error = Float32[0.0 0.02 0.0; 0.01 0.0 0.03]
    fringed = unfringed + reference + error
    fringe_storage = vec(permutedims(fringed))
    unfringe_storage = vec(permutedims(unfringed))
    command_storage = zeros(Float32, 3)
    GC.@preserve fringe_storage unfringe_storage command_storage begin
        borrowed_fringe = FilterGraphPipeWire.RowMajorBorrowedArray{Float32,2,false}(
            pointer(fringe_storage), shape, length(fringe_storage))
        borrowed_unfringe = FilterGraphPipeWire.RowMajorBorrowedArray{Float32,2,false}(
            pointer(unfringe_storage), shape, length(unfringe_storage))
        command = FilterGraphPipeWire.BorrowedArray(pointer(command_storage), (3,), Val(true))
        @test borrowed_fringe == fringed
        @test scc_step!(command, prepared, borrowed_fringe, scc_metadata(1)) === nothing
        @test all(iszero, command_storage)
        @test scc_step!(command, prepared, borrowed_unfringe,
            scc_metadata(2; trigger=SAMPLE_TRIGGER_LOW)) === nothing
        @test command_storage[1] ≈ -0.005f0 atol=4.0f-8
        @test command_storage[2] ≈ -0.02f0 atol=4.0f-8
        @test command_storage[3] ≈ -0.045f0 atol=4.0f-8
        @test count(!iszero, command_storage) == 3
        @test scc_borrowed_allocation_bytes(command, prepared, borrowed_fringe, borrowed_unfringe, false) == 0
        @test scc_borrowed_allocation_bytes(command, prepared, borrowed_fringe, borrowed_unfringe, true) == 0
        @test scc_borrowed_rolling_allocation_bytes(command, prepared, borrowed_fringe, borrowed_unfringe) == 0
    end
end

function prepared_scc_alias_fixture(shape)
    calibration = SccCalibration(; frame_shape=shape,
        reference_difference=zeros(Float32, shape), correction_mask=fill(true, shape),
        reconstructor=zeros(Float32, 468, prod(shape)), coordinate_order=Tuple(1:468),
        fringed_response=zeros(Float32, 468, 468), unfringed_response=zeros(Float32, 468, 468),
        command_unit="normalized-bax307", reconstructor_sign=:command_error,
        exposure_duration_nanoseconds=SCC_DURATION_NS, image_schema=SCC_IMAGE_SCHEMA,
        bias_command=fill(0.01f0, 468), lower_limits=fill(-1.0f0, 468), upper_limits=fill(1.0f0, 468))
    return prepared_scc(; calibration)
end

@testset "SCC protects indices and byte-sized storage from output aliasing" begin
    # 234 Int64 indices occupy exactly the bytes of 468 Float32 commands.
    prepared = prepared_scc_alias_fixture((13, 18))
    indices = prepared.plan.calibration.residual.selected_indices
    original = copy(indices)
    aliased_command = reinterpret(Float32, indices)
    @test length(aliased_command) == 468
    @test Base.mightalias(aliased_command, indices)
    @test_throws ArgumentError scc_step!(aliased_command, prepared,
        zeros(Float32, 13, 18), scc_metadata(1))
    @test indices == original
    @test prepared.workspace.state.committed_identity === nothing
    GC.@preserve indices begin
        for command in (
                FilterGraphPipeWire.BorrowedArray(Ptr{Float32}(pointer(indices)), (468,), Val(true)),
                FilterGraphPipeWire.RowMajorBorrowedArray{Float32,1,true}(
                    Ptr{Float32}(pointer(indices)), (468,), 468))
            @test Base.mightalias(command, indices)
            @test_throws ArgumentError scc_step!(command, prepared,
                zeros(Float32, 13, 18), scc_metadata(1))
            @test indices == original
            @test prepared.workspace.state.committed_identity === nothing
        end
    end

    # A Float32 output can also share bytes with Bool masks and validity arrays.
    prepared = prepared_scc_alias_fixture((36, 52))
    frame = zeros(Float32, 36, 52)
    for flags in (prepared.plan.calibration.residual.mask,
            prepared.workspace.workspace.difference_validity,
            prepared.workspace.workspace.selected_validity, prepared.workspace.workspace.image_validity,
            prepared.workspace.state.difference.committed_validity,
            prepared.workspace.state.difference.candidate_validity,
            prepared.workspace.state.difference.staged_validity)
        original_bytes = copy(reinterpret(UInt8, flags))
        aliased_command = reinterpret(Float32, flags)
        @test length(aliased_command) == 468
        @test Base.mightalias(aliased_command, flags)
        @test_throws ArgumentError scc_step!(aliased_command, prepared, frame, scc_metadata(1))
        @test reinterpret(UInt8, flags) == original_bytes
    end

    # A valid one-byte status array can borrow the first byte of a complete
    # command allocation. The byte overlap is independent of element type.
    execution = prepared.workspace
    scratch = execution.workspace
    command_storage = fill(0.01f0, 468)
    GC.@preserve command_storage begin
        status_bytes = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(pointer(command_storage)), 1; own=false)
        status_bytes[1] = UInt8(ChoppedImageComplete)
        scratch_with_status = FilterGraphAlgorithms.SccCorrectionWorkspace(
            scratch.image, scratch.image_validity, scratch.difference, scratch.difference_validity, status_bytes,
            scratch.selected_residuals, scratch.selected_validity, scratch.valid_fraction,
            scratch.reconstructed, scratch.correction, scratch.command_difference, scratch.phase_correction)
        prepared.workspace = FilterGraphAlgorithms.SccCorrectionExecution(
            execution.state, scratch_with_status, execution.initial_command)
        original_command = copy(command_storage)
        @test Base.mightalias(command_storage, status_bytes)
        @test_throws ArgumentError scc_step!(command_storage, prepared, frame, scc_metadata(1))
        @test status_bytes[1] == UInt8(ChoppedImageComplete)
        @test command_storage == original_command
        @test prepared.workspace.state.committed_identity === nothing
        for command in (
                FilterGraphPipeWire.BorrowedArray(pointer(command_storage), (468,), Val(true)),
                FilterGraphPipeWire.RowMajorBorrowedArray{Float32,1,true}(
                    pointer(command_storage), (468,), 468))
            @test Base.mightalias(command, status_bytes)
            @test_throws ArgumentError scc_step!(command, prepared, frame, scc_metadata(1))
            @test status_bytes[1] == UInt8(ChoppedImageComplete)
            @test command_storage == original_command
            @test prepared.workspace.state.committed_identity === nothing
        end
    end
end

end
