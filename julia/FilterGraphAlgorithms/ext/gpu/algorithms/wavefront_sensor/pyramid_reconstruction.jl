import FilterGraphAlgorithms:
    DenseReconstructionNonFiniteInput,
    DenseReconstructionNonFiniteOutput,
    DenseReconstructorPlan,
    DenseReconstructorWorkspace,
    PreparedAlgorithm,
    PyramidReconstructorF32,
    complete!,
    enqueue!,
    preflight,
    reset!

const PYRAMID_RECONSTRUCTION_DEFAULT_LANES = 256

@kernel function _pyramid_reconstruction!(
    output,
    frame_status,
    transposed_reconstructor,
    input,
    ::Val{Lanes},
) where {Lanes}
    local_index = @index(Local, Linear)
    _, row = @index(Group, NTuple)
    local_sums = @localmem eltype(output) (Lanes,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    value = zero(eltype(output))
    if frame_valid
        selected_coordinate_count = size(input, 1)
        for component in 1:2
            for coordinate in Int(local_index):Lanes:selected_coordinate_count
                matrix_column = (component - 1) * selected_coordinate_count + coordinate
                @inbounds value = muladd(
                    transposed_reconstructor[matrix_column, row],
                    input[coordinate, component],
                    value,
                )
            end
        end
    end
    @inbounds local_sums[local_index] = value
    @synchronize

    if Lanes == 256
        _reduce_sum!(local_sums, Int(local_index), 128)
        @synchronize
    end
    if Lanes >= 128
        _reduce_sum!(local_sums, Int(local_index), 64)
        @synchronize
    end
    if Lanes >= 64
        _reduce_sum!(local_sums, Int(local_index), 32)
        @synchronize
    end
    _reduce_sum!(local_sums, Int(local_index), 16)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 8)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 4)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 2)
    @synchronize
    _reduce_sum!(local_sums, Int(local_index), 1)
    @synchronize

    if frame_valid && local_index == 1
        @inbounds output[row] = local_sums[1]
    end
end

struct PyramidReconstruction{
    ValidateInput,
    Reconstruct,
    ValidateOutput,
    O,
    S,
    R,
    I,
    L,
}
    validate_input!::ValidateInput
    reconstruct!::Reconstruct
    validate_output!::ValidateOutput
    output::O
    frame_status::S
    transposed_reconstructor::R
    input::I
    lanes::L
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        PyramidReconstructorF32,
        <:DenseReconstructorPlan,
        <:DenseReconstructorWorkspace,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
    workgroups=nothing,
)
    length(outputs) == 1 || throw(
        ArgumentError("PyramidReconstructorF32 capture requires one output"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("PyramidReconstructorF32 capture requires one input"),
    )
    output = only(outputs)
    input = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :signal),
        input,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :reconstructed),
        output,
    )
    _require_capture_array("pyramid I4Q signal input", input, device)
    _require_capture_array("reconstructed controller output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    transposed_reconstructor =
        _capture_array(device, permutedims(prepared.plan.matrix))
    lanes = isnothing(workgroups) ?
            PYRAMID_RECONSTRUCTION_DEFAULT_LANES :
            workgroups.pyramid_reconstruction_lanes
    _require_reduction_lanes(lanes)
    backend = KernelAbstractions.get_backend(output)
    return PyramidReconstruction(
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        _pyramid_reconstruction!(
            backend,
            (lanes, 1),
        ),
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        output,
        frame_status,
        transposed_reconstructor,
        input,
        Val(lanes),
        _failure_code(node_index, DenseReconstructionNonFiniteInput),
        _failure_code(node_index, DenseReconstructionNonFiniteOutput),
    )
end

preflight(::PyramidReconstruction) = nothing

function enqueue!(captured::PyramidReconstruction)
    captured.validate_input!(
        captured.frame_status,
        captured.input,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    captured.reconstruct!(
        captured.output,
        captured.frame_status,
        captured.transposed_reconstructor,
        captured.input,
        captured.lanes;
        ndrange=(
            _lane_count(captured.lanes),
            length(captured.output),
        ),
    )
    captured.validate_output!(
        captured.frame_status,
        captured.output,
        captured.output_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    return nothing
end

function complete!(
    captured::PyramidReconstruction,
    status::Int32,
)
    status == captured.input_failure && return DenseReconstructionNonFiniteInput
    status == captured.output_failure && return DenseReconstructionNonFiniteOutput
    return nothing
end

reset!(captured::PyramidReconstruction) = captured
