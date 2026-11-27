# One ordered two-branch region per capture. Choose contiguous regions so the
# original order defines which failure wins and which state may commit.
const FG = FilterGraphs

_parent_node(::Any) = 0
_parent_node(::FG.NodeOutput{I}) where I = I

function _branch_ranges(routes)
    parents = map(routes) do inputs
        unique(filter(!iszero, map(_parent_node, inputs)))
    end
    any(p -> length(p) == 2, parents) || return ()
    ancestors = BitSet[]
    for (i, dependencies) in pairs(parents)
        closure = BitSet((i,))
        for parent in dependencies
            union!(closure, ancestors[parent])
        end
        push!(ancestors, closure)
    end
    regions = Tuple{UnitRange{Int},UnitRange{Int}}[]
    for join in reverse(eachindex(parents))
        length(parents[join]) == 2 || continue
        a, b = map(i -> ancestors[i], parents[join])
        left, right = setdiff(a, b), setdiff(b, a)
        # Two extra status launches have a cost. Skip single-node branches.
        min(length(left), length(right)) >= 2 || continue
        first(left) > first(right) && ((left, right) = (right, left))
        early, late = first(left):last(left), first(right):last(right)
        length(early) == length(left) && length(late) == length(right) &&
            last(early)+1 == first(late) && last(late)+1 == join || continue
        push!(regions, (early, late))
    end
    return regions
end

function _branches_disjoint(early, late, inputs, outputs)
    earlier = Iterators.flatten(outputs[early])
    later = Iterators.flatten(outputs[late])
    prefix = Iterators.flatten(outputs[1:first(early)-1])
    shared = Iterators.flatten((values(inputs), prefix))
    aliases(a, b) = any(x -> any(y -> FG._captured_mightalias(x, y), b), a)
    return !aliases(earlier, later) && !aliases(earlier, shared) && !aliases(later, shared)
end

_qualified_fork_node(node::FG.CapturedNode{D}, adapters) where D =
    parentmodule(D) === FG.FilterGraphAlgorithms && node.adapter isa adapters

function _cuda_fork_regions(nodes, routes, inputs, outputs)
    ext = Base.get_extension(FG.FilterGraphAlgorithms, :FilterGraphAlgorithmsCUDAExt)
    ext === nothing && return nothing
    required = (:ControllerToVdm, :VdmShapeOffset, :VdmToPdm, :PdmOffset,
        :VectorCombination, :PdmFeedbackToVdm, :VdmFeedbackToController,
        :DenseVectorReconstruction, :PdmCommand, :BiquadCascade, :_PdmStages)
    all(name -> isdefined(ext, name), required) || return nothing
    # These adapters own their private scratch and only expose routed arrays.
    # The later branch may commit only a staged PDM, whose adoption can move
    # to the join. Earlier branch transitions keep their usual ordering.
    stateless = Union{ext.ControllerToVdm,ext.VdmShapeOffset,ext.VdmToPdm,
        ext.PdmOffset,ext.VectorCombination,ext.PdmFeedbackToVdm,
        ext.VdmFeedbackToController,ext.DenseVectorReconstruction}
    earlier = Union{stateless,ext.PdmCommand,ext.BiquadCascade}
    later = Union{stateless,ext.PdmCommand}
    for (early, late) in _branch_ranges(routes)
        all(i -> _qualified_fork_node(nodes[i], earlier), early) || continue
        all(i -> _qualified_fork_node(nodes[i], later), late) || continue
        commits = filter(i -> nodes[i].adapter isa ext.PdmCommand, late)
        length(commits) <= 1 || continue
        commit = isempty(commits) ? nothing : only(commits)
        commit === nothing || nodes[commit].adapter.execution isa ext._PdmStages || continue
        _branches_disjoint(early, late, inputs, outputs) || continue
        return (; early, late, commit, pdm_type=ext.PdmCommand)
    end
    return nothing
end

@kernel function _copy_branch_status!(branch, status)
    @inbounds branch[1] = status[1]
end

@kernel function _join_branch_status!(status, branch, adoption, last_pdm_failure)
    @inbounds begin
        early = status[1]
        late = branch[1]
        if adoption !== nothing
            # Failure after PDM still preserves its ordinary state adoption.
            adoption[1] = iszero(early) && (iszero(late) || late > last_pdm_failure) ?
                Int32(0) : Int32(1)
        end
        if iszero(early)
            status[1] = late
        end
    end
end

struct _DeferPdmAdoption end
(::_DeferPdmAdoption)(args...; ndrange) = nothing

function _branch_node(node::FG.CapturedNode{D}, status, constructor, defer) where D
    adapter = node.adapter
    # Cold reconstruction of a qualified immutable adapter. All state and
    # buffer fields are shared with the ordinary owner used by reset,
    # completion, and controller recapture.
    args = map(fieldnames(typeof(adapter))) do field
        field === :frame_status && return status
        field === :adopt! && defer && return _DeferPdmAdoption()
        return getfield(adapter, field)
    end
    branch = constructor(args...)
    return FG.CapturedNode{D,typeof(branch)}(branch)
end

_enqueue_branch_adoption!(::Nothing, status) = nothing
function _enqueue_branch_adoption!(command, status)
    command.adopt!(command.committed_current, command.candidate_current,
        status; ndrange=length(command.demanded))
    return nothing
end

function _capture_segment(f, graph, after, context, workspace=context.workspace)
    return FG._with_captured_workspace(context, workspace) do
        frontier = CUDA.capture!(graph; after, stream=context.stream) do
            # The raw workspace pointer used by cuBLAS needs a graph lease.
            UInt(pointer(workspace)) % 256 == 0 ||
                error("captured cuBLAS workspace must be 256-byte aligned")
            f()
        end
        isempty(frontier) && error("CUDA capture lost its dependency frontier")
        return frontier
    end
end

function _record_captured_fork(context, status, nodes, feedback, region)
    branch_status = similar(status.value)
    adoption = region.commit === nothing ? nothing : similar(status.value)
    workspace = similar(context.workspace)
    command = region.commit === nothing ? nothing : nodes[region.commit].adapter
    pdm_failure = command === nothing ? Int32(0) :
        max(command.input_failure, command.output_failure)
    roots = (context, status, nodes, feedback, branch_status, adoption, workspace)
    try
        prefix = nodes[1:first(region.early)-1]
        suffix = nodes[last(region.late)+1:end]
        earlier = nodes[region.early]
        later = map(Tuple(region.late)) do i
            defer = i == region.commit
            constructor = defer ? region.pdm_type : typeof(nodes[i].adapter)
            return _branch_node(nodes[i], branch_status, constructor, defer)
        end
        backend = KernelAbstractions.get_backend(branch_status)
        copy! = _copy_branch_status!(backend, 1)
        join! = _join_branch_status!(backend, 1)
        FG._with_captured_workspace(context) do
            status.clear!(status.value; ndrange=1)
            copy!(branch_status, status.value; ndrange=1)
            join!(status.value, branch_status, adoption, pdm_failure; ndrange=1)
            FG._warm_captured!(context, status, nodes, feedback)
        end

        graph = CUDA.CuGraph()
        head = _capture_segment(graph, CUDA.CuGraphNode[], context) do
            status.clear!(status.value; ndrange=1)
            FG._enqueue_captured_nodes!(prefix)
            copy!(branch_status, status.value; ndrange=1)
        end
        first_branch = _capture_segment(graph, head, context) do
            FG._enqueue_captured_nodes!(earlier)
        end
        second_branch = _capture_segment(graph, head, context, workspace) do
            FG._enqueue_captured_nodes!(later)
        end
        _capture_segment(graph, vcat(first_branch, second_branch), context) do
            join!(status.value, branch_status, adoption, pdm_failure; ndrange=1)
            _enqueue_branch_adoption!(command, adoption)
            FG._enqueue_captured_nodes!(suffix)
            FG._enqueue_captured_feedback!(feedback)
        end
        _require_cuda_graph_fixed_storage(graph)
        executable = _instantiate_cuda_graph(context, graph)
        outcome = CapturedFrameOutcome(status.value, status.host)
        return CUDACapturedExecutable(graph, executable, outcome, nothing, feedback)
    catch
        FG._drain_captured_storage!(context, roots)
        rethrow()
    end
end

const _CUDA_CAN_CAPTURE_DEPENDENCIES = pkgversion(CUDA) >= v"6.5.1" &&
    isdefined(CUDA, :capture!) && isdefined(CUDA, :CuGraphNode) && _CAPTURE_SUPPORTS_STREAM &&
    hasmethod(CUDA.capture!, Tuple{Function,CUDA.CuGraph}, (:after, :stream))

function FG._prepare_captured_executable(context::CUDACapturedContext, status,
        nodes, feedback, routes, inputs, outputs)
    if _CUDA_CAN_CAPTURE_DEPENDENCIES && CUDA.driver_version() >= v"12.3" &&
            !iszero(CUDA.attribute(context.device, CUDA.CUDACore.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))
        region = _cuda_fork_regions(nodes, routes, inputs, outputs)
        region === nothing ||
            return _record_captured_fork(context, status, nodes, feedback, region)
    end
    return FG._capture_warm_and_record(context, status, nodes, feedback)
end
