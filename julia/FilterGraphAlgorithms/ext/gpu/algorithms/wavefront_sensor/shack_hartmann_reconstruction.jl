import FilterGraphAlgorithms:
    DenseReconstructionNonFiniteInput,
    DenseReconstructionNonFiniteOutput,
    DenseReconstructorPlan,
    DenseReconstructorWorkspace,
    PreparedAlgorithm,
    ShackHartmannReconstructorF32,
    complete!,
    enqueue!,
    preflight,
    reset!

const SHACK_HARTMANN_RECONSTRUCTION_DEFAULT_LANES = 128

@kernel function _shack_hartmann_reconstruction!(
    output,
    frame_status,
    x_reconstructor,
    y_reconstructor,
    slopes,
    ::Val{Lanes},
) where {Lanes}
    local_index = @index(Local, Linear)
    _, row = @index(Group, NTuple)
    local_sums = @localmem eltype(output) (Lanes,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    value = zero(eltype(output))
    if frame_valid
        for subaperture in Int(local_index):Lanes:size(slopes, 1)
            @inbounds begin
                value = muladd(
                    _projection_coefficient(x_reconstructor, subaperture, row),
                    slopes[subaperture, 1],
                    value,
                )
                value = muladd(
                    _projection_coefficient(y_reconstructor, subaperture, row),
                    slopes[subaperture, 2],
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

struct ShackHartmannReconstruction{
    ValidateInput,
    Reconstruct,
    ValidateOutput,
    Output,
    Status,
    XReconstructor,
    YReconstructor,
    Slopes,
    L,
}
    validate_input!::ValidateInput
    reconstruct!::Reconstruct
    validate_output!::ValidateOutput
    output::Output
    frame_status::Status
    x_reconstructor::XReconstructor
    y_reconstructor::YReconstructor
    slopes::Slopes
    lanes::L
    input_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        ShackHartmannReconstructorF32,
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
        ArgumentError(
            "ShackHartmannReconstructorF32 capture requires one output",
        ),
    )
    length(inputs) == 1 || throw(
        ArgumentError(
            "ShackHartmannReconstructorF32 capture requires one input",
        ),
    )
    output = only(outputs)
    slopes = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :slopes),
        slopes,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :reconstructed),
        output,
    )
    _require_capture_array("Shack-Hartmann slope input", slopes, device)
    _require_capture_array("reconstructed controller output", output, device)
    _require_capture_array("captured frame status", frame_status, device)

    matrix = prepared.plan.matrix
    x_reconstructor = _capture_transposed_coefficients(device, matrix, 1:2:size(matrix, 2))
    y_reconstructor = _capture_transposed_coefficients(device, matrix, 2:2:size(matrix, 2))
    lanes = isnothing(workgroups) ?
            SHACK_HARTMANN_RECONSTRUCTION_DEFAULT_LANES :
            workgroups.shack_hartmann_reconstruction_lanes
    _require_reduction_lanes(lanes)
    backend = KernelAbstractions.get_backend(output)
    return ShackHartmannReconstruction(
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        _shack_hartmann_reconstruction!(
            backend,
            (lanes, 1),
        ),
        _validate_projection_values!(
            backend,
            VALIDATION_WORKGROUP_SIZE,
        ),
        output,
        frame_status,
        x_reconstructor,
        y_reconstructor,
        slopes,
        Val(lanes),
        _failure_code(node_index, DenseReconstructionNonFiniteInput),
        _failure_code(node_index, DenseReconstructionNonFiniteOutput),
    )
end

preflight(::ShackHartmannReconstruction) = nothing

function enqueue!(
    captured::ShackHartmannReconstruction,
)
    captured.validate_input!(
        captured.frame_status,
        captured.slopes,
        captured.input_failure;
        ndrange=VALIDATION_WORKGROUP_SIZE,
    )
    captured.reconstruct!(
        captured.output,
        captured.frame_status,
        captured.x_reconstructor,
        captured.y_reconstructor,
        captured.slopes,
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
    captured::ShackHartmannReconstruction,
    status::Int32,
)
    status == captured.input_failure && return DenseReconstructionNonFiniteInput
    status == captured.output_failure && return DenseReconstructionNonFiniteOutput
    return nothing
end

reset!(captured::ShackHartmannReconstruction) =
    captured
