# The canonical plan remains the parameter/control authority. Its immutable
# replacement carries the matching transposed planes into the same adoption.
struct RowMajorPixelCalibrationPlan
    canonical::FilterGraphAlgorithms.PixelCalibrationPlan
    transposed::FilterGraphAlgorithms.PixelCalibrationPlan

    function RowMajorPixelCalibrationPlan(plan::FilterGraphAlgorithms.PixelCalibrationPlan)
        transposed = FilterGraphAlgorithms.PixelCalibrationPlan(
            permutedims(plan.flat), permutedims(plan.background),
        )
        return new(plan, transposed)
    end
end

function FilterGraphAlgorithms._replace_pixel_flat(plan::RowMajorPixelCalibrationPlan, values)
    return RowMajorPixelCalibrationPlan(
        FilterGraphAlgorithms._replace_pixel_flat(plan.canonical, values),
    )
end

function FilterGraphAlgorithms._replace_pixel_background(plan::RowMajorPixelCalibrationPlan, values)
    return RowMajorPixelCalibrationPlan(
        FilterGraphAlgorithms._replace_pixel_background(plan.canonical, values),
    )
end

# Dense startup inputs and other ordinary arrays retain the canonical helper.
function FilterGraphAlgorithms._process_pixel_calibration!(
    calibrated::AbstractMatrix{Float32}, plan::RowMajorPixelCalibrationPlan,
    raw::AbstractMatrix{UInt16},
)
    (Base.mightalias(calibrated, plan.transposed.flat) ||
     Base.mightalias(calibrated, plan.transposed.background)) && throw(
        ArgumentError("calibrated output must not alias calibration"),
    )
    return FilterGraphAlgorithms._process_pixel_calibration!(calibrated, plan.canonical, raw)
end

function FilterGraphAlgorithms._process_pixel_calibration!(
    calibrated::Transpose{Float32,Matrix{Float32}}, plan::RowMajorPixelCalibrationPlan,
    raw::RowMajorBorrowedArray{UInt16,2,false},
)
    Base.require_one_based_indexing(calibrated, raw)
    size(raw) == size(plan.canonical.flat) ||
        throw(DimensionMismatch("raw detector shape is incompatible"))
    size(calibrated) == size(plan.canonical.flat) ||
        throw(DimensionMismatch("calibrated detector shape is incompatible"))
    output = parent(calibrated)
    for coefficients in (plan.canonical.flat, plan.canonical.background,
                         plan.transposed.flat, plan.transposed.background)
        Base.mightalias(output, coefficients) && throw(
            ArgumentError("calibrated output must not alias calibration"),
        )
    end
    # Reordering an overlapping raw/output pair would change the legacy result.
    Base.mightalias(output, raw) && return FilterGraphAlgorithms._process_pixel_calibration!(
        calibrated, plan.canonical, raw,
    )
    raw_transposed = BorrowedArray{UInt16,2,false}(
        raw.data, reverse(size(raw)), length(raw),
    )
    rows, columns = size(output)
    @inbounds for column in 1:columns
        @simd ivdep for row in 1:rows
            output[row, column] = FilterGraphAlgorithms._calibrated_pixel(
                raw_transposed[row, column],
                plan.transposed.flat[row, column],
                plan.transposed.background[row, column],
            )
        end
    end
    return nothing
end

_row_major_calibration_external(::Any) = false
_row_major_calibration_external(::FilterGraphs.ExternalInput) = true

_row_major_calibration_output(::Any, index) = false
_row_major_calibration_output(::FilterGraphs.NodeOutput{Node,Port}, index) where {Node,Port} =
    Node == index && Port == 1

_row_major_calibration_declaration(::FilterGraphs.PreparedNode{D}) where {D} = D

function _validate_row_major_calibration!(consumers, graph, index)
    node = graph.nodes[index]
    node.prepared.plan isa FilterGraphAlgorithms.PixelCalibrationPlan || throw(
        ArgumentError("Node '$(node.name)' is not an ordinary pixel calibration"),
    )
    length(graph.input_routes[index]) == 1 &&
        _row_major_calibration_external(only(graph.input_routes[index])) || throw(
            ArgumentError("Node '$(node.name)' requires an external full-image raw input"),
        )
    for (consumer_index, (consumer, routes)) in enumerate(zip(graph.nodes, graph.input_routes))
        for (input, route) in zip(consumer.input_names, routes)
            _row_major_calibration_output(route, index) || continue
            _row_major_calibration_declaration(consumer) === FilterGraphAlgorithms.PyramidPupilImageF32 &&
                input === :image || throw(ArgumentError(
                    "Node '$(node.name)' has an unsupported calibrated-image consumer '$(consumer.name)'",
                ))
            FilterGraphAlgorithms.pyramid_geometry_requires_ordered_sum(consumer.prepared.plan) || throw(
                ArgumentError(
                    "Node '$(consumer.name)' requires the dense small-pupil normalization path",
                ),
            )
            consumers[consumer_index] = true
        end
    end
    return nothing
end

"""
    prepare_row_major_calibration(graph::PreparedGraph)

Prepare a detached ordinary CPU Graph with row-major storage for external
full-image `PixelCalibrationU16F32` Nodes. The source must be quiescent and its
operations serialized with preparation. Its Plans, Workspaces, metadata and
buffers remain unchanged and independent of the returned Graph.

Logical image dimensions, coordinates, Port formats and declaration identities
remain canonical. Selected internal calibrated images are logical transpose
views of owned dense transposed storage. Native row-major raw inputs use a
checked contiguous SIMD loop; other raw arrays use the original checked helper.
Use `PipeWireNode(result; boundary_layout=RowMajorLayout(), ...)` explicitly to
negotiate a row-major native boundary. Existing preparation and Node defaults
do not select this storage.

Only ordinary built-in CPU declarations and dense internal buffers are admitted.
Internal image consumers must be `PyramidPupilImageF32` Nodes whose geometry
requires ordered normalization on dense CPU images.
Other consumers and specialized executors are rejected. External publication
of a calibrated image retains the framework's logical output copy. Parameter
replacement prepares both coefficient orientations before normal adoption.
Detached consumers retain the canonical Float32 summation order.
"""
function prepare_row_major_calibration(graph::PreparedGraph)
    all(node -> node.prepared isa FilterGraphAlgorithms.PreparedAlgorithm &&
                _row_major_calibration_declaration(node) in FilterGraphAlgorithms.algorithms(),
        graph.nodes) || throw(ArgumentError(
        "row-major calibration requires ordinary built-in CPU Algorithm Nodes",
    ))
    all(outputs -> all(output -> output isa Array, outputs), graph.node_outputs) || throw(
        ArgumentError("row-major calibration requires dense CPU Graph buffers"),
    )
    any(inflight -> inflight[], graph.node_inflight) && throw(
        ArgumentError("row-major calibration preparation requires a quiescent Graph"),
    )
    indices = findall(node -> _row_major_calibration_declaration(node) ===
                             FilterGraphAlgorithms.PixelCalibrationU16F32, graph.nodes)
    isempty(indices) && throw(ArgumentError("Graph has no full-image UInt16 pixel calibration"))
    consumers = falses(length(graph.nodes))
    for index in indices
        _validate_row_major_calibration!(consumers, graph, index)
    end

    detached = deepcopy(graph)
    for index in findall(consumers)
        prepared = detached.nodes[index].prepared
        prepared.plan = FilterGraphAlgorithms.PyramidPixelImagePlan(prepared.plan, true)
    end
    nodes = detached.nodes
    outputs = detached.node_outputs
    for index in indices
        node = nodes[index]
        source = node.prepared
        prepared = FilterGraphAlgorithms.PreparedAlgorithm(
            FilterGraphAlgorithms.PixelCalibrationU16F32,
            RowMajorPixelCalibrationPlan(source.plan), source.workspace, source.formats,
        )
        replacement = FilterGraphs.PreparedNode(
            FilterGraphAlgorithms.PixelCalibrationU16F32, node.name, prepared,
            node.input_names, node.output_names, node.parameter_names,
            node.input_formats, node.output_formats, node.parameter_formats,
            node.output_conditional, node.output_metadata_sources,
        )
        nodes = Base.setindex(nodes, replacement, index)
        output = transpose(permutedims(only(outputs[index])))
        outputs = Base.setindex(outputs, (output,), index)
    end
    return PreparedGraph(
        nodes, detached.node_failure_types, outputs, detached.node_output_available,
        detached.node_output_metadata, detached.node_inflight, detached.input_routes,
        detached.parameter_routes, detached.output_routes, detached.input_formats,
        detached.parameter_formats, detached.output_formats, detached.output_conditional,
        detached.published_metadata, detached.default_input_metadata, detached.result_types,
        detached.rejection,
    )
end
