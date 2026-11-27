import FilterGraphAlgorithms: VectorOffsetF32, VectorOffsetPlan

function _prepare_capture(::CompleteTransition, device,
        prepared::PreparedAlgorithm{VectorOffsetF32, <:VectorOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == 2 && length(inputs) == 1 ||
        throw(ArgumentError("vector offset capture requires one input and two outputs"))
    for (name, array) in zip((:output, Symbol("current-offset"), :requested), (outputs..., inputs...))
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, name), array)
    end
    offset = copy(prepared.plan.offset)
    return _prepare_vector_combination(device, outputs, only(inputs),
        _capture_array(device, offset), prepared.plan.combination, frame_status, node_index;
        right_finite=all(isfinite, offset))
end

# Only the offset adapter has a current-offset array and one external input.
# Right validation retains the finite proof established during the private copy.
prepare_validation(c::VectorCombination{<:AbstractArray}, proofs::Tuple{Bool}) =
    prepare_validation(c, (only(proofs), false))
