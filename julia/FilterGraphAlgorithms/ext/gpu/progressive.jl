import FilterGraphAlgorithms as FGA
import FilterGraphAlgorithms: PreparedAlgorithm, _process_admitted!, _replace_prepared_parameter, prepare_algorithm

struct ProgressiveDevicePlan{P,D,S} <: FGA.AbstractAlgorithmPlan
    source::P
    device::D
    storage::S
end

const ProgressiveDeviceDeclaration = Union{
    PixelCalibrationRowU16F32, PixelCalibrationReadoutU16F32,
    DetectorFrameAssemblerF32, IncrementalDenseReconstructorF32,
    PyramidPixelRowF32, PyramidMeasurementBlockF32, PyramidReadoutMeasurementBlockF32,
    PyramidRowReconstructorF32, PyramidReadoutReconstructorF32,
    ShackHartmannMeasurementBlockF32, ShackHartmannWeightedMeasurementBlockF32,
    ShackHartmannReadoutMeasurementBlockF32, ShackHartmannRowReconstructorF32,
    ShackHartmannReadoutReconstructorF32,
}

function prepare_algorithm(prepared::PreparedAlgorithm{D}, device::DeviceType) where {D<:ProgressiveDeviceDeclaration}
    source = _progressive_source(prepared.plan)
    return _with_eager_device(device) do
        plan = ProgressiveDevicePlan(source, device, _progressive_plan_storage(D, device, source))
        workspace = _progressive_workspace(D, plan)
        PreparedAlgorithm(D, plan, workspace, prepared.formats)
    end
end

_progressive_source(plan) = plan
_progressive_source(plan::ProgressiveDevicePlan) = plan.source

function _replace_progressive_parameter(declaration, replacement, plan::ProgressiveDevicePlan, values)
    source = replacement(plan.source, values)
    return _with_eager_device(plan.device) do
        storage = _progressive_plan_storage(declaration, plan.device, source)
        ProgressiveDevicePlan(source, plan.device, storage)
    end
end

function _require_progressive_arrays(device, outputs, inputs)
    _require_current_device(device)
    for (name, arrays) in (("output", outputs), ("input", inputs))
        for array in arrays
            applicable(_require_array_device, name, array, device) ||
                throw(ArgumentError("progressive $name must be a resident array on the prepared device"))
            _require_array_device(name, array, device)
        end
    end
    return nothing
end

function _require_progressive_output_aliases(outputs, inputs)
    for (index, output) in pairs(outputs)
        for earlier in 1:index-1
            _capture_arrays_overlap(output, outputs[earlier]) &&
                throw(ArgumentError("progressive outputs must not alias one another"))
        end
        for input in inputs
            _capture_arrays_overlap(output, input) &&
                throw(ArgumentError("progressive outputs must not alias inputs"))
        end
    end
    return nothing
end

function _require_progressive_workspace_disjoint(arrays, workspace_buffers)
    for buffer in workspace_buffers
        isnothing(buffer) && continue
        for array in arrays
            _capture_arrays_overlap(array, buffer) &&
                throw(ArgumentError("progressive inputs and outputs must not alias Workspace storage"))
        end
    end
    return nothing
end

function _progressive_admitted(prepared, outputs, inputs)
    _require_progressive_arrays(prepared.plan.device, outputs, inputs)
    _require_progressive_output_aliases(outputs, inputs)
    return prepared.plan, prepared.workspace
end

function _progressive_status!(
    completion, status, ::Type{E}
) where {E<:Enum}
    copyto!(completion, status)
    return iszero(completion[1]) ? nothing : E(completion[1])
end

# A normalized completion resets the logical sample's first published offset.
function _progressive_complete(sample, discontinuity)
    return AlgorithmProcessComplete(
        SampleMetadata(sample; offset=0, terminal=true, discontinuity, corrupted=false),
    )
end

struct MeasurementEntry
    value::Float32
    column::UInt32
end

@kernel function _publish_measurement_packet!(measurements, columns, count, packet)
    index = @index(Global, Linear)
    @inbounds begin
        entries = packet[1].column
        index == 1 && (count[1] = entries)
        if index <= entries
            entry = packet[index + 1]
            measurements[index] = entry.value
            columns[index] = entry.column
        end
    end
end

function _enqueue_measurement_publication!(measurements, columns, count, packet, entries)
    backend = KernelAbstractions.get_backend(measurements)
    _publish_measurement_packet!(backend, 256)(
        measurements, columns, count, packet; ndrange=max(entries, 1),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _accumulate_measurement_packet!(candidate, matrix, packet, status)
    row = @index(Global, Linear)
    if @inbounds iszero(status[1])
        @inbounds begin
            result = candidate[row]
            for index in 1:Int(packet[1].column)
                entry = packet[index + 1]
                result = fma(_projection_coefficient(matrix, Int(entry.column) + 1, row), entry.value, result)
            end
            candidate[row] = result
        end
    end
end

function _enqueue_measurement_accumulation!(backend, candidate, matrix, packet, status, failure)
    _accumulate_measurement_packet!(backend, 256)(
        candidate, matrix, packet, status; ndrange=length(candidate),
    )
    _validate_projection_values_portable!(backend, VALIDATION_WORKGROUP_SIZE)(
        status, candidate, Int32(failure); ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    return nothing
end

_abandon_progressive_frame!(frame::FGA.DetectorReadoutFrame) = FGA._abandon_detector_readout!(frame)
_abandon_progressive_frame!(frame::FGA.RowBlockFrame) = FGA._abandon_row_block_frame!(frame)
_complete_progressive_frame!(frame::FGA.DetectorReadoutFrame) = FGA._complete_detector_readout!(frame)
_complete_progressive_frame!(frame::FGA.RowBlockFrame) = FGA._complete_row_block_frame!(frame)

_progressive_frame(::Integer) = FGA.RowBlockFrame()
_progressive_frame(::AbstractVector) = FGA.DetectorReadoutFrame()

function _progressive_block_ordinal!(frame::FGA.RowBlockFrame, sample, input, detector, block_rows)
    first_row = Int(sample.offset)
    accepted, _ = FGA._accept_row_block!(frame, sample, first_row, block_rows, size(detector, 1))
    accepted || return nothing
    _enqueue_scatter_row_block!(detector, input, first_row)
    return div(first_row, block_rows) + 1
end

function _progressive_block_ordinal!(frame::FGA.DetectorReadoutFrame, sample, input, detector, indices)
    FGA._accept_detector_readout!(frame, sample, length(indices)) || return nothing
    ordinal = Int(sample.offset) + 1
    _enqueue_scatter_readout!(detector, input, indices[ordinal])
    return ordinal
end

_progressive_measurement_metadata(sample, frame) = SampleMetadata(
    sample; discontinuity=sample.discontinuity || (iszero(sample.offset) && frame.current_discontinuity),
)

const ProgressiveReconstructionPlan = Union{
    FGA.PyramidRowReconstructionPlan, FGA.PyramidReadoutReconstructionPlan,
    FGA.ShackHartmannRowReconstructionPlan, FGA.ShackHartmannReadoutReconstructionPlan,
}

_progressive_reconstructor_storage(device, source) = nothing
function _progressive_reconstructor_storage(device, source::ProgressiveReconstructionPlan)
    matrix = source.reconstructor.matrix
    return _capture_transposed_coefficients(device, matrix, axes(matrix, 2))
end

_progressive_reconstructed(device, source) = nothing
_progressive_reconstructed(device, source::ProgressiveReconstructionPlan) =
    _capture_zeros(device, Float32, size(source.reconstructor.matrix, 1))
