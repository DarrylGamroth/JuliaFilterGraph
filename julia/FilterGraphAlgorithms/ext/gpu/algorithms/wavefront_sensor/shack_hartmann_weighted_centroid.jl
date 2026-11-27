import FilterGraphAlgorithms:
    ShackHartmannWeightedCentroidF32,
    ShackHartmannWeightedCentroidPlan

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{
        ShackHartmannWeightedCentroidF32,
        <:ShackHartmannWeightedCentroidPlan,
    },
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    return _prepare_shack_hartmann_image(
        device,
        prepared,
        prepared.plan.sensing,
        prepared.plan.weights,
        outputs,
        inputs,
        frame_status,
        node_index,
    )
end
