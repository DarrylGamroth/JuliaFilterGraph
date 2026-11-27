module CapturedAliasesTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

struct AliasIdentity end
struct AliasProducer end
struct AliasReader{P}
    proofs::P
end
struct AliasOpaque end
struct InvalidAlias end
struct AliasView end

FilterGraphAlgorithms.preserves_inputs(
    ::Union{AliasIdentity,AliasProducer,AliasReader,InvalidAlias}
) = true
FilterGraphAlgorithms.alias_input(::AliasIdentity, ::Val{1}) = 1
FilterGraphAlgorithms.alias_input(::InvalidAlias, ::Val{1}) = 0
FilterGraphAlgorithms.alias_input(::AliasView, ::Val{1}) = 1
FilterGraphAlgorithms.finite_output(::AliasProducer, ::Val{1}) = true
FilterGraphAlgorithms.prepare_validation(adapter::AliasReader, proofs::Tuple) =
    AliasReader(proofs)

alias_node(adapter) = FilterGraphs.CapturedNode{Nothing,typeof(adapter)}(adapter)

@testset "captured alias preparation and validation proofs" begin
    external = (input = Float32[1, 2],)
    root, branch = Float32[0, 0], Float32[0, 0]
    nodes = (alias_node(AliasIdentity()), alias_node(AliasIdentity()))
    routes = ((FilterGraphs.ExternalInput{1}(),),
              (FilterGraphs.NodeOutput{1,1}(),))
    outputs = ((root,), (branch,))

    # Identity outputs follow the exact routed object, including on a branch.
    finalized = FilterGraphs._captured_alias_outputs(
        (nodes..., alias_node(AliasIdentity())),
        (routes..., (FilterGraphs.NodeOutput{1,1}(),)), external,
        (outputs..., (Float32[0, 0],)))
    @test finalized[1][1] === external.input
    @test finalized[2][1] === external.input
    @test finalized[3][1] === external.input
    @test FilterGraphs._captured_alias_outputs(
        (nodes..., alias_node(AliasOpaque())),
        (routes..., (FilterGraphs.NodeOutput{1,1}(),)), external,
        (outputs..., (Float32[0, 0],)))[1][1] === root

    # Protected external buffers retain separate snapshots through identity chains.
    protected = FilterGraphs._captured_alias_outputs(
        nodes, routes, external, outputs, (external.input,))
    @test protected[1][1] === root
    @test protected[2][1] === root

    @test_throws GraphConfigurationError FilterGraphs._captured_alias_outputs(
        (alias_node(AliasIdentity()),),
        ((FilterGraphs.ExternalInput{1}(),),), external,
        ((Float32[0],),))
    @test_throws GraphConfigurationError FilterGraphs._captured_alias_outputs(
        (alias_node(AliasIdentity()),),
        ((FilterGraphs.ExternalInput{1}(),),), external,
        ((Float64[0, 0],),))
    @test_throws GraphConfigurationError FilterGraphs._captured_alias_outputs(
        (alias_node(InvalidAlias()),),
        ((FilterGraphs.ExternalInput{1}(),),), external,
        ((Float32[0, 0],),))

    # Other backends keep their original output buffers.
    @test FilterGraphs._prepare_captured_aliases(
        :fake, nodes, routes, external, outputs, ()) === outputs

    # Only the declared exact input/output identity is a safe overlap. An
    # unqualified overlap and an overlapping view remain barriers.
    same = external.input
    @test FilterGraphs._captured_safe_overlap(
        AliasIdentity(), (same,), (same,))
    @test !FilterGraphs._captured_safe_overlap(
        AliasOpaque(), (same,), (same,))
    overlapping = @view same[:]
    @test overlapping !== same
    @test Base.mightalias(overlapping, same)
    @test !FilterGraphs._captured_safe_overlap(
        AliasView(), (overlapping,), (same,))

    # A finite producer's proof survives an identity alias; an external input
    # remains unproven even when an adapter returns that exact object.
    produced = Float32[1, 2]
    source = (input = Float32[1, 2],)
    reader1 = AliasReader(())
    proof_nodes = (alias_node(AliasProducer()), alias_node(AliasIdentity()),
                   alias_node(reader1))
    proof_routes = ((FilterGraphs.ExternalInput{1}(),),
                    (FilterGraphs.NodeOutput{1,1}(),),
                    (FilterGraphs.NodeOutput{2,1}(),))
    proof_outputs = ((produced,), (produced,), (Float32[0, 0],))
    reused = FilterGraphs._reuse_captured_validation(
        proof_nodes, proof_routes, proof_outputs, source, proof_outputs)
    @test reused[3].adapter.proofs == (true,)

    external_reader = AliasReader(())
    external_nodes = (alias_node(AliasIdentity()), alias_node(external_reader))
    external_routes = ((FilterGraphs.ExternalInput{1}(),),
                       (FilterGraphs.NodeOutput{1,1}(),))
    external_outputs = ((source.input,), (Float32[0, 0],))
    external_reused = FilterGraphs._reuse_captured_validation(
        external_nodes, external_routes, external_outputs, source, external_outputs)
    @test external_reused[2].adapter.proofs == (false,)

    opaque_nodes = (alias_node(AliasProducer()), alias_node(AliasOpaque()),
                    alias_node(AliasReader(())))
    opaque_routes = ((FilterGraphs.ExternalInput{1}(),),
                     (FilterGraphs.NodeOutput{1,1}(),),
                     (FilterGraphs.NodeOutput{2,1}(),))
    opaque_outputs = ((produced,), (Float32[0, 0],), (Float32[0, 0],))
    opaque_reused = FilterGraphs._reuse_captured_validation(
        opaque_nodes, opaque_routes, opaque_outputs, source, opaque_outputs)
    @test opaque_reused[3].adapter.proofs == (false,)
end

end # module CapturedAliasesTests

module CapturedBindingLifetimeTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

struct LifetimeAdapter{S}
    scratch::S
end
struct LifetimeNode end
mutable struct LifetimeContext
    scratch_refs::Vector{WeakRef}
    sync_count::Int
    pending::Bool
    fail_sync::Bool
end
struct LifetimeDevice
    context::LifetimeContext
end

FilterGraphAlgorithms.preserves_inputs(::LifetimeAdapter) = true
FilterGraphAlgorithms.alias_input(::LifetimeAdapter, ::Val{1}) = 1
FilterGraphAlgorithms.enqueue!(::LifetimeAdapter) = nothing
FilterGraphAlgorithms.preflight(::LifetimeAdapter) = nothing
FilterGraphAlgorithms.complete!(::LifetimeAdapter, ::Int32) = nothing
FilterGraphAlgorithms.reset!(::LifetimeAdapter) = nothing

function FilterGraphs._prepare_captured_node(
        ::LifetimeNode, outputs::Tuple, routes::Tuple, device_inputs,
        node_outputs, device::LifetimeDevice, frame_status, node_index::Int32,
        workgroups,
)
    scratch = Ref(Int(node_index))
    push!(device.context.scratch_refs, WeakRef(scratch))
    device.context.pending = true
    adapter = LifetimeAdapter(scratch)
    return FilterGraphs.CapturedNode{Nothing,typeof(adapter)}(adapter)
end

function FilterGraphs._prepare_captured_aliases(
        device::LifetimeDevice, nodes, routes, inputs, outputs, protected,
)
    return FilterGraphs._captured_alias_outputs(nodes, routes, inputs, outputs, protected)
end

function FilterGraphs._synchronize_captured_graph!(context::LifetimeContext)
    context.sync_count += 1
    GC.gc(true)
    @test all(reference -> reference.value !== nothing, context.scratch_refs)
    context.fail_sync && error("injected fake drain failure")
    context.pending = false
    return nothing
end

function lifetime_fixture(; fail_sync=false)
    context = LifetimeContext(WeakRef[], 0, false, fail_sync)
    device = LifetimeDevice(context)
    inputs = (input = Float32[1, 2],)
    outputs = ((Float32[0, 0],), (Float32[0, 0],))
    graph = (
        nodes = (LifetimeNode(), LifetimeNode()),
        input_routes = ((FilterGraphs.ExternalInput{1}(),),
                        (FilterGraphs.NodeOutput{1,1}(),)),
        output_routes = (),
    )
    status = (value = Ref(Int32(0)),)
    return context, graph, device, status, inputs, outputs
end

function quarantined_lifetime_key(context)
    return lock(FilterGraphs._captured_storage_quarantine_lock) do
        for (key, record) in FilterGraphs._captured_storage_quarantine
            record.roots isa AbstractVector || continue
            any(root -> root === context, record.roots) && return key
        end
        return nothing
    end
end

@testset "captured binding preparation drains discarded adapters" begin
    context, graph, device, status, inputs, outputs = lifetime_fixture()
    nodes, finalized = FilterGraphs._prepare_captured_bindings(
        context, graph, device, status, inputs, outputs,
        FilterGraphs.CapturedWorkgroupProfile(), NamedTuple())
    @test context.sync_count == 1
    @test !context.pending
    @test length(context.scratch_refs) == 4
    @test all(output -> output[1] === inputs.input, finalized)
    @test all(node -> node.adapter isa LifetimeAdapter, nodes)

    failed, graph, device, status, inputs, outputs = lifetime_fixture(fail_sync=true)
    @test_throws ErrorException FilterGraphs._prepare_captured_bindings(
        failed, graph, device, status, inputs, outputs,
        FilterGraphs.CapturedWorkgroupProfile(), NamedTuple())
    @test failed.sync_count == 2
    @test failed.pending
    @test length(failed.scratch_refs) == 4
    key = quarantined_lifetime_key(failed)
    @test key !== nothing
    record = lock(FilterGraphs._captured_storage_quarantine_lock) do
        FilterGraphs._captured_storage_quarantine[key]
    end
    @test all(reference -> reference.value !== nothing, failed.scratch_refs)
    @test record.failure isa CapturedException

    # Recover only this fake context's quarantine entry; unrelated quarantines
    # created by other tests remain untouched.
    failed.fail_sync = false
    @test FilterGraphs._drain_captured_storage!(failed, key, record.roots)
    @test !failed.pending
    @test lock(FilterGraphs._captured_storage_quarantine_lock) do
        !haskey(FilterGraphs._captured_storage_quarantine, key)
    end
end

end # module CapturedBindingLifetimeTests
