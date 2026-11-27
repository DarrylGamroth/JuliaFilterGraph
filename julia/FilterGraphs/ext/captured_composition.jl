const FGA = FilterGraphs.FilterGraphAlgorithms

struct CapturedCameraMetadata{A,H,F}
    adapter::A
    host_flags::H
    admit!::F
    source::Int
    node::Int
end

function FG._prepare_captured_camera_metadata(context::CapturedContext, graph, nodes, ports)
    index = findfirst(node -> node isa FG.CapturedNode{FGA.ChoppedImageDifferenceF32}, nodes)
    isnothing(index) && return nothing
    any(node -> node isa FG.CapturedNode{FGA.ChoppedImageDifferenceF32}, nodes[(index+1):end]) &&
        throw(FG.GraphConfigurationError("capture supports one chopped-image transition"))
    all(node -> node isa FG.CapturedNode{<:Union{FGA.PixelSelectionF32,FGA.PixelValidityF32}},
        nodes[1:(index-1)]) ||
        throw(FG.GraphConfigurationError("captured chopped metadata requires a non-rejecting pixel prefix"))
    source = FG._metadata_source_input(graph, first(graph.input_routes[index]))
    source = source > 0 ? findfirst(==(keys(graph.input_formats)[source]), ports.inputs) : nothing
    isnothing(source) && throw(FG.GraphConfigurationError("chopped metadata requires an external camera input"))
    extension = Base.get_extension(FGA, algorithm_extension)
    flags = _allocate_captured_host_buffer(context, (element_type=Int32, shape=(2,)))
    return CapturedCameraMetadata(nodes[index].adapter, flags, extension._admit_chopped_capture!, source, index)
end

FG._admit_captured_camera_metadata!(camera::CapturedCameraMetadata, metadata, rejection) =
    _admit_camera_sample!(camera, FGA._admitted_sample_metadata(getfield(metadata, camera.source)), rejection)
FG._copy_captured_camera_metadata!(context::CapturedContext, camera::CapturedCameraMetadata) =
    _copy_camera_metadata!(context, camera)

struct CapturedCompositionControl{N,A,H,F}
    node::N
    adapter::A
    host::H
    settings::F
end

struct CapturedCompositionStage{C,E,N,H,D,O,S,B,M,P}
    context::C
    executable::E
    nodes::N
    input_staging::H
    inputs::D
    outputs::O
    output_staging::S
    buffers::B
    camera_metadata::M
    controls::P
end

function _prepare_composition_stage(context, graph, stage, input_names, inputs, outputs, device, extension, prepare_node)
    status = _prepare_captured_frame_status(context)
    camera_metadata = nothing
    nodes = map(stage.indices) do index
        node = graph.nodes[index]
        routes = graph.input_routes[index]
        if node.prepared isa FGA.PreparedAlgorithm{FGA.ChoppedImageDifferenceF32}
            isnothing(camera_metadata) || throw(FG.GraphConfigurationError("composition stage has multiple chopped-image transitions"))
            source = FG._metadata_source_input(graph, first(routes))
            source > 0 && keys(graph.input_formats)[source] === first(input_names) ||
                throw(FG.GraphConfigurationError("chopped-image metadata must come from the stage camera"))
            for prior in stage.indices
                prior == index && break
                graph.nodes[prior].prepared isa FGA.PreparedAlgorithm{<:Union{FGA.PixelSelectionF32,FGA.PixelValidityF32}} ||
                    throw(FG.GraphConfigurationError("captured chopped-image metadata requires a non-rejecting pixel prefix"))
            end
            adapter = extension._prepare_chopped_capture(device, node.prepared,
                outputs[index], FG._route_inputs(routes, inputs, outputs), status.value)
            flags = _allocate_captured_host_buffer(context, (element_type=Int32, shape=(2,)))
            camera_metadata = CapturedCameraMetadata(adapter, flags, extension._admit_chopped_capture!, source, index)
            return FG.CapturedNode{FGA.ChoppedImageDifferenceF32,typeof(adapter)}(adapter)
        end
        if node.prepared isa FGA.PreparedAlgorithm{FGA.ClosedLoopIntegratorF32}
            adapter = extension._prepare_composition_integrator(device, node.prepared,
                outputs[index], FG._route_inputs(routes, inputs, outputs), status.value, Int32(index))
            return FG.CapturedNode{FGA.ClosedLoopIntegratorF32,typeof(adapter)}(adapter)
        end
        any(name -> !isempty(FGA.required_sample_metadata(node.prepared, name)), node.input_names) &&
            throw(FG.GraphConfigurationError("Captured composition cannot supply sample metadata to Node '$(node.name)'"))
        FG._require_captured_support(node, device)
        return prepare_node(node, outputs[index], routes, inputs, outputs,
            device, status.value, Int32(index), FG.CapturedWorkgroupProfile())
    end
    stage_outputs = FG._composition_select(outputs, stage.indices)
    nodes = _reuse_composition_validation(nodes, stage, stage_outputs, inputs, outputs, extension)
    if isempty(input_names)
        nodes = _reuse_shared_dm_validation(nodes, stage, stage_outputs, inputs, outputs, extension)
    end
    if camera_metadata !== nothing
        nodes = _fuse_captured_scc_pixels(nodes, stage.routes, stage.indices)
        position = findfirst(==(camera_metadata.node), stage.indices)
        camera_metadata = CapturedCameraMetadata(nodes[position].adapter,
            camera_metadata.host_flags, camera_metadata.admit!, camera_metadata.source,
            camera_metadata.node)
    end
    FG._require_captured_methods(stage.nodes, device, nodes)
    controls = map(stage.nodes, nodes) do node, captured
        adapter = captured.adapter
        settings = extension._composition_settings
        if applicable(settings, adapter, node.prepared.plan)
            values = settings(adapter, node.prepared.plan)
            host = _allocate_captured_host_buffer(context, (element_type=Float32, shape=(length(values),)))
            copyto!(host, values)
            return CapturedCompositionControl(node, adapter, host, settings)
        end
        node.prepared.plan isa FGA.AbstractAlgorithmPlan &&
            any(info -> info.update isa FGA.RuntimeProperty, FGA.property_infos(node.prepared.plan)) &&
            throw(FG.GraphConfigurationError("Captured composition has no runtime property adapter for Node '$(node.name)'"))
        return nothing
    end
    executable = FG._capture_warm_and_record(context, status, nodes)
    device_inputs = map(name -> getproperty(inputs, name), input_names)
    host_inputs = map(device_inputs) do buffer
        _allocate_captured_host_buffer(context, (element_type=eltype(buffer), shape=size(buffer)))
    end
    indices = Tuple(i for i in eachindex(graph.output_routes) if
        FG._composition_shared_output(graph.output_routes[i], stage.indices))
    names = map(i -> keys(graph.output_formats)[i], indices)
    routes = map(i -> graph.output_routes[i], indices)
    copied = map(route -> FG._route_value(route, (), outputs), routes)
    staging = map(copied) do buffer
        _allocate_captured_host_buffer(context, (element_type=eltype(buffer), shape=size(buffer)))
    end
    destinations = map(route -> FG._route_value(route, (), graph.node_outputs), routes)
    return CapturedCompositionStage(context, executable, nodes, host_inputs, device_inputs,
        (device=NamedTuple{names}(copied), destination=destinations), NamedTuple{names}(staging),
        stage_outputs, camera_metadata, filter(!isnothing, controls))
end

function _reuse_shared_dm_validation(nodes, stage, stage_outputs, inputs, outputs, extension)
    isdefined(extension, :_dense_output_is_finite) || return nodes
    # Dense projection preserves its input and still checks its full output.
    # Use that guarantee only in the shared stage, with no incoming facts from
    # camera corrections or delayed feedback.
    dense(adapter) = adapter isa extension.DenseVectorReconstruction
    preserves_inputs(adapter) = dense(adapter) ?
        extension._dense_output_is_finite(adapter) : FGA.preserves_inputs(adapter)
    finite_output(adapter, port) = dense(adapter) ?
        port isa Val{1} && extension._dense_output_is_finite(adapter) : FGA.finite_output(adapter, port)
    function prepare_validation(adapter, proofs)
        if dense(adapter) && only(proofs)
            return extension._without_dense_input_scan(adapter)
        elseif adapter isa extension.VdmToPdm
            return FGA.prepare_validation(adapter, proofs)
        end
        return adapter
    end
    validation = (; preserves_inputs, finite_output, prepare_validation)
    return FG._reuse_captured_validation(nodes, stage.routes, stage_outputs, inputs, outputs; validation)
end

function _reuse_composition_validation(nodes, stage, stage_outputs, inputs, outputs, extension)
    isdefined(extension, :_without_dense_input_scan) || return nodes
    return _reuse_adjacent_validation(nodes, stage.indices, stage.routes, stage_outputs,
        inputs, outputs) do producer, consumer, input
        refined = _reuse_scc_residual_validation(producer, consumer, input, extension)
        return _reuse_modal_reference_validation(producer, refined, extension)
    end
end

function _reuse_adjacent_validation(refine, nodes, indices, routes, stage_outputs, inputs, outputs)
    return ntuple(length(nodes)) do i
        consumer = nodes[i]
        i > 1 || return consumer
        node_routes = routes[i]
        length(node_routes) == 1 && only(node_routes) isa FG.NodeOutput{indices[i-1],1} ||
            return consumer
        routed = FG._route_inputs(node_routes, inputs, outputs)
        FG._captured_safe_overlap(consumer.adapter, stage_outputs[i], routed) || return consumer
        return refine(nodes[i-1], consumer, only(routed))
    end
end

function _reuse_scc_residual_validation(nodes, indices, routes, stage_outputs, inputs, outputs)
    extension = Base.get_extension(FGA, algorithm_extension)
    extension !== nothing && isdefined(extension, :_without_dense_input_scan) || return nodes
    return _reuse_adjacent_validation(nodes, indices, routes, stage_outputs,
        inputs, outputs) do producer, consumer, input
        return _reuse_scc_residual_validation(producer, consumer, input, extension)
    end
end

function _reuse_scc_residual_validation(producer, consumer, input, extension)
    producer isa FG.CapturedNode{FGA.SccPixelResidualF32} || return consumer
    residual, reconstruction = producer.adapter, consumer.adapter
    # This residual gather writes either a checked subtraction or zero at every
    # selected index. The shared status retains pair and valid-count failures.
    residual isa extension.SccPixelResidual && reconstruction isa extension.DenseVectorReconstruction &&
        residual.selected_residuals === input === reconstruction.input &&
        residual.frame_status === reconstruction.frame_status || return consumer
    return FG._captured_node_with_adapter(consumer, extension._without_dense_input_scan(reconstruction))
end

function _reuse_modal_reference_validation(producer, consumer, extension)
    consumer isa FG.CapturedNode{FGA.VectorOffsetF32} &&
        producer.adapter isa extension.DenseVectorReconstruction &&
        isdefined(extension, :_dense_output_is_finite) &&
        extension._dense_output_is_finite(producer.adapter) &&
        FGA.preserves_inputs(consumer.adapter) || return consumer
    adapter = FGA.prepare_validation(consumer.adapter, (true,))
    return FG._captured_node_with_adapter(consumer, adapter)
end

function _prepare_captured_composition(composition, device, extension, prepare_node)
    FG._captured_composition(composition) && throw(ArgumentError("composition is already captured"))
    extension !== nothing && isdefined(extension, :_composition_settings) ||
        throw(ArgumentError("composition requires matching FilterGraphAlgorithms device adapters"))
    # Do not let a capture share scientific workspaces or publication storage
    # with the CPU owner. Capture starts a new acquisition boundary.
    source = deepcopy(composition)
    FG.reset!(source)
    graph = source.graph
    all(node -> all(!, node.output_conditional), graph.nodes) ||
        throw(FG.GraphConfigurationError("Captured composition stages require unconditional node outputs"))
    context = _prepare_captured_context(device)
    return FG._with_captured_context(context) do
        outputs = FG._captured_node_output_buffers(context, graph.nodes)
        inputs = FG._captured_external_buffers(context, graph.input_formats)
        # The DM stage writes the next feedback in the very buffers the
        # branches read. No feedback changes until both admitted stages finish.
        for branch in source.branches
            inputs = merge(inputs, NamedTuple{(branch.feedback,)}((
                FG._route_value(branch.returned_route, (), outputs),)))
        end
        branches = map(source.branches) do branch
            execution = _prepare_composition_stage(context, graph, branch.stage,
                (branch.camera,), inputs, outputs, device, extension, prepare_node)
            return merge(branch, (stage=FG.CompositionStage(graph, branch.nodes, execution),
                camera_buffer=only(execution.input_staging)))
        end
        execution = _prepare_composition_stage(context, graph, source.shared_stage,
            (), inputs, outputs, device, extension, prepare_node)
        stage = FG.CompositionStage(graph, source.shared_nodes, execution)
        for buffers in outputs, buffer in buffers
            fill!(buffer, zero(eltype(buffer)))
        end
        _synchronize_captured_graph!(context)
        host_inputs = source.inputs
        for branch in branches
            host_inputs = merge(host_inputs, NamedTuple{(branch.camera,)}((branch.camera_buffer,)))
        end
        return FG.PreparedComposition(graph, branches, source.shared_nodes, stage,
            source.public_routes, source.input_formats, source.output_formats,
            host_inputs, source.metadata, source.contributors, source.last_inputs,
            source.accepted, FG._composition_domain(), source.generation, source.sequence, source.trigger, source.failed)
    end
end

@inline _admit_camera_metadata!(::Nothing, metadata, rejection) = nothing
@inline function _admit_camera_metadata!(camera::CapturedCameraMetadata, metadata, rejection)
    return _admit_camera_sample!(camera, getfield(metadata, camera.source), rejection)
end

@inline function _admit_camera_sample!(camera, sample, rejection)
    result = camera.admit!(camera.adapter, sample.present ? sample.value : nothing)
    if result isa Enum
        FG._record_rejection!(rejection, camera.node, result)
        return result
    end
    camera.host_flags[1], camera.host_flags[2] = result
    return nothing
end

_copy_camera_metadata!(context, ::Nothing) = nothing
function _copy_camera_metadata!(context, camera::CapturedCameraMetadata)
    _copy_to_captured!(context, camera.adapter.flags, camera.host_flags)
    return nothing
end

function _run_composition_stage!(execution, rejection)
    context = execution.context
    previous = _activate_captured_context(context)
    try
        GC.@preserve execution try
            _copy_composition_controls!(context, execution.controls)
            _copy_camera_metadata!(context, execution.camera_metadata)
            FG._copy_to_captured_inputs!(context, execution.inputs, execution.input_staging)
            status = FG._launch_and_complete_captured_frame!(context, execution.executable,
                execution.output_staging, execution.outputs.device)
            result = FG._complete_captured_nodes!(execution.nodes, status, rejection)
            result === nothing || return result
            FG._copy_captured_outputs!(execution.outputs.destination, values(execution.output_staging))
            return nothing
        catch
            FG._drain_captured_storage!(context, execution)
            rethrow()
        end
    finally
        _restore_captured_context!(context, previous)
    end
end

_copy_composition_controls!(context, ::Tuple{}) = nothing
function _copy_composition_controls!(context, controls::Tuple)
    control = first(controls)
    settings = control.settings(control.adapter, control.node.prepared.plan)
    if any(i -> !isequal(settings[i], control.host[i]), eachindex(settings))
        copyto!(control.host, settings)
        _copy_to_captured!(context, control.adapter.settings, control.host)
    end
    return _copy_composition_controls!(context, Base.tail(controls))
end

function _publish_stage_metadata!(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, metadata, all_metadata)
    return nothing
end
function _publish_stage_metadata!(nodes, routes, availability, slots, metadata, all_metadata)
    routed = FG._route_metadata(first(routes), metadata, all_metadata)
    FG._set_available!(first(availability))
    FG._publish_node_metadata!(first(slots), first(nodes).output_metadata_sources, routed,
        false, FGA._NO_SAMPLE_METADATA)
    return _publish_stage_metadata!(Base.tail(nodes), Base.tail(routes), Base.tail(availability),
        Base.tail(slots), metadata, all_metadata)
end

function FG._composition_stage!(composition, stage::FG.CompositionStage, metadata, execution::CapturedCompositionStage)
    graph = composition.graph
    result = FG._preflight_captured_nodes(execution.nodes, graph.rejection)
    if result !== nothing
        graph.rejection.node = stage.indices[graph.rejection.node]
        return true
    end
    result = _admit_camera_metadata!(execution.camera_metadata, metadata, graph.rejection)
    result === nothing || return true
    result = _run_composition_stage!(execution, graph.rejection)
    if result !== nothing
        graph.rejection.node = stage.indices[graph.rejection.node]
        return true
    end
    _publish_stage_metadata!(stage.nodes, stage.routes, stage.available, stage.metadata,
        metadata, graph.node_output_metadata)
    return false
end

function FG._reset_composition_stage!(execution::CapturedCompositionStage)
    context = execution.context
    FG._with_captured_context(context) do
        _synchronize_captured_graph!(context)
        try
            FG._reset_captured_nodes!(execution.nodes)
            for control in execution.controls
                fill!(control.host, NaN32)
            end
            for buffers in execution.buffers, buffer in buffers
                fill!(buffer, zero(eltype(buffer)))
            end
            _synchronize_captured_graph!(context)
        catch
            FG._drain_captured_storage!(context, execution)
            rethrow()
        end
    end
    FG._release_captured_storage!(execution)
    return nothing
end
