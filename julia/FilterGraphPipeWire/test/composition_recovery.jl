module CompositionRecoveryTests
using Test, FilterGraphAlgorithms, FilterGraphs, FilterGraphPipeWire, PipeWireAO
include(joinpath(@__DIR__, "data", "support", "composition_fixture.jl"))
using .CompositionFixture: CompositionGateF32, composition_fixture
const FGA = FilterGraphAlgorithms
const JFG = FilterGraphs
const FGP = FilterGraphPipeWire
const native_callback_buffer = parentmodule(@__MODULE__).native_callback_buffer
const native_callback_flags = parentmodule(@__MODULE__).native_callback_flags

@algorithm UnknownCommandF32 begin
    label = "composition-unknown-command-f32"
    @configuration UnknownCommandF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), FGA.CONTROLLER_COMMAND_V1),
        output(:output, Float32, (configuration.extent,), FGA.CONTROLLER_COMMAND_V1; metadata_from=:input))
    prepare(configuration) = nothing
    process!(plan) = (copyto!(output, input); nothing)
end

function recovery_composition(directory; unknown=false, causal=false, bias=false)
    path = composition_fixture(directory)
    input_node = causal ? """{name=a-input type=ndarray label=leaky-integrator-f32 config={extent=1 initial_state=0
        input_schema="$(FGA.CALIBRATED_PIXELS_V1)" output_schema="$(FGA.CALIBRATED_PIXELS_V1)"}}""" :
        """{name=a-input type=ndarray label=vector-offset-f32 config={extent=1 schema="$(FGA.CALIBRATED_PIXELS_V1)"}}"""
    projection = unknown ? "{name=a-projection type=ndarray label=composition-unknown-command-f32 config={extent=1}}" :
        bias ? """{name=a-projection type=ndarray label=vector-offset-f32 config={extent=1 schema="$(FGA.CONTROLLER_COMMAND_V1)"}}""" :
        """{name=a-projection type=ndarray label=dense-vector-reconstructor-f32 config={input_count=1 output_count=1 initial_reconstructor=[1]
          input_schema="$(FGA.CONTROLLER_COMMAND_V1)" output_schema="$(FGA.CONTROLLER_COMMAND_V1)" reconstructor_schema=test.projection/1}}"""
    input_port = causal ? "input" : "requested"
    projection_port = unknown ? "input" : bias ? "requested" : "ordered-input"
    write(joinpath(directory, "a.conf"), """
    # Generated test graph; source/regeneration: test/composition_recovery.jl recovery_composition.
    filter.graph = {
      nodes = [
        $input_node
        {name=a-valid type=ndarray label=pixel-validity-f32 config={extent=1 input_schema="$(FGA.CALIBRATED_PIXELS_V1)"}}
        {name=a-pair type=ndarray label=chopped-image-difference-f32 config={pixel_count=1 trigger_high_is_fringed=true require_acquisition_identity=true}}
        {name=a-residual type=ndarray label=scc-pixel-residual-f32 config={pixel_count=1 mask=[true] initial_reference_image=[0] minimum_valid_fraction=1}}
        {name=a-map type=ndarray label=dense-vector-reconstructor-f32 config={input_count=1 output_count=1 initial_reconstructor=[1]
          input_schema="$(FGA.SCC_SELECTED_PIXEL_RESIDUALS_V1)" output_schema="$(FGA.CONTROLLER_RESIDUAL_ERROR_V1)" reconstructor_schema=test.scc-map/1}}
        {name=a-integrator type=ndarray label=closed-loop-integrator-f32 config={state_length=1} props={gain=1 pole=1 anti-windup-gain=1}}
        $projection
      ]
      links = [
        {output=a-input:output input=a-valid:input} {output=a-input:output input=a-pair:image}
        {output=a-valid:validity input=a-pair:input-validity}
        {output=a-pair:difference input=a-residual:difference}
        {output=a-pair:output-validity input=a-residual:difference-validity}
        {output=a-pair:pair-status input=a-residual:pair-status}
        {output=a-residual:selected-residuals input=a-map:ordered-input}
        {output=a-map:output input=a-integrator:residual-error}
        {output=a-integrator:correction input=a-projection:$projection_port}
      ]
      inputs = [{name=a-image endpoint=a-input:$input_port}
                {name=a-feedback endpoint=a-integrator:constraint-feedback}]
      outputs = [{name=a-correction endpoint=a-projection:output}]
    }
    """)
    shared = read(joinpath(directory, "shared.conf"), String)
    shared = replace(shared, "nodes = [" => """nodes = [
      {name=flat type=ndarray label=vector-offset-f32 config={extent=1 schema="$(FGA.REQUESTED_PDM_COMMAND_V1)"}}
      """, "{output = pdm-project:output input = limit:requested}" =>
        "{output=pdm-project:output input=flat:requested} {output=flat:output input=limit:requested}",
        "slew_limit = null initial_current = null" => "slew_limit = 0.1 initial_current = [0.2]",
        "inputs = [" => "inputs = [{name=flat endpoint=flat:offset}")
    write(joinpath(directory, "shared.conf"), "# Generated test graph; source/regeneration: test/composition_recovery.jl recovery_composition.\n" * shared)
    composition = JFG.prepare_composition(path; algorithms=(FGA.algorithms()..., CompositionGateF32, UnknownCommandF32))
    JFG.replace_parameters!(composition, :flat => Float32[.1])
    return composition
end

function prepared(directory)
    composition = recovery_composition(directory)
    node = FGP.PipeWireNode(composition; connect=false,
        recovery=(controllers=(Symbol("a-integrator"), Symbol("b-integrator")),
            participants=(Symbol("a-pair"), :limit), reference_timeout_ms=60000))
    FGP._prepare_owner(node.owner)
    outputs = map(format -> zeros(format.element_type, format.shape), JFG.output_formats(composition))
    native = vcat((native_callback_buffer(array; header_available=true) for array in values(outputs))...)
    return (; composition, node, outputs, native, a=Float32[0], b=Float32[0],
        scope=node.filter.callbacks.on_get_properties.recovery.scope)
end
function input(state, camera, sequence; value=0f0, kind=nothing)
    state.a[1] = camera === :a ? value : 0
    state.b[1] = camera === :b ? value : 0
    a = native_callback_buffer(state.a; present=camera === :a,
        header=BufferHeader(0, 0, sequence, 0, UInt64(100 + 7sequence)),
        acquisition_sequence=UInt64(sequence), exposure_duration_nanoseconds=UInt64(100),
        optical_classification=kind === nothing ? (isodd(sequence) ? OPTICAL_FRINGED : OPTICAL_UNFRINGED) : kind)
    b = native_callback_buffer(state.b; present=camera === :b,
        header=BufferHeader(0, 0, sequence, 0, UInt64(200 + 7sequence)), acquisition_sequence=UInt64(sequence))
    return vcat(a, b)
end
function step!(state, storage)
    GC.@preserve state storage begin
        inputs = NdArrayFilterBuffers{false}(Ptr{Cvoid}(pointer(storage)), UInt32(2))
        outputs = NdArrayFilterBuffers{true}(Ptr{Cvoid}(pointer(state.native)), UInt32(length(state.native)))
        state.node.filter.callbacks.on_process(state.node.filter, inputs, outputs)
    end
    index = findfirst(==(:command), keys(state.outputs))
    return iszero(native_callback_flags(state.native, index) & PipeWireAO.LibPipeWire.PW_NDARRAY_FILTER_BUFFER_FLAG_OUTPUT_UNAVAILABLE)
end
step!(state, camera, sequence; kwargs...) = step!(state, input(state, camera, sequence; kwargs...))
function props(state)
    return Dict(name => PipeWireAO.pod_value(value) for (name, value) in
        state.node.filter.callbacks.on_get_properties(state.node.filter).values)
end
node(state, name) = only(filter(node -> node.name === name, state.composition.graph.nodes)).prepared
status(state) = FGP.recovery_status(state.scope)
request!(state, token=1) = FGP.stage_properties!(state.node, SPA.Props("pipewireao-recovery:request-token" => Int64(token)))
apply!(state) = state.node.filter.callbacks.on_apply_properties(state.node.filter)
function release!(state)
    FGP.stage_properties!(state.node, SPA.Props("pipewireao-recovery:release-token" => Int64(status(state).active_generation)))
    apply!(state)
end
close_branch!(state, branch) = FGP.stage_properties!(state.node, SPA.Props("$branch-integrator:closed" => true))

@testset "Native release with a stale reference leaves both controllers held" begin
    mktempdir() do directory
        state = prepared(directory)
        try
            request!(state)
            apply!(state)
            for sequence in 1:3; step!(state, :b, sequence); end
            @test status(state).reference_reached
            generation = status(state).active_generation
            native_generation = props(state)["pipewireao-recovery:active-generation"]
            state.node.owner.recovery_observation_timestamp_ns[] = time_ns() - UInt64(60_000_000_001)
            @test_throws InvalidStateException FGP.stage_recovery_release!(state.scope, generation)
            release = SPA.Props("pipewireao-recovery:release-token" => Int64(generation))
            @test_throws InvalidStateException FGP.stage_properties!(state.node, release)

            filter = state.node.filter
            callback = filter.callbacks.on_properties
            callback(filter, release)
            observed = props(state)
            @test observed["pipewireao-recovery:state"] == "release_refused"
            @test observed["pipewireao-recovery:closure-inhibited"]
            @test observed["pipewireao-recovery:release-token"] == generation
            @test observed["pipewireao-recovery:recovery-generation"] == generation
            @test observed["pipewireao-recovery:active-generation"] ==
                observed["pipewireao-recovery:requested-generation"] == native_generation + 1
            @test !node(state, Symbol("a-integrator")).plan.closed
            @test !node(state, Symbol("b-integrator")).plan.closed

            callback(filter, SPA.Props("pipewireao-recovery:release-token" => Int64(generation + 1)))
            @test props(state)["pipewireao-recovery:active-generation"] == native_generation + 1
            @test step!(state, :b, 4)
            callback(filter, release)
            pending = props(state)
            @test pending["pipewireao-recovery:state"] == "release_pending"
            callback(filter, release)
            @test props(state)["pipewireao-recovery:requested-generation"] ==
                pending["pipewireao-recovery:requested-generation"]
            apply!(state)
            @test props(state)["pipewireao-recovery:state"] == "released"
            @test !status(state).closure_inhibited
        finally
            close(state.node)
        end
    end
end

@testset "Coupled scope admission and atomic no-camera reconciliation" begin
    for options in ((;unknown=true), (;causal=true), (;bias=true))
        mktempdir() do directory
            composition = recovery_composition(directory; options...)
            owner = FGP.PreparedGraphOwner(composition.graph, composition)
            @test_throws ArgumentError FGP.prepare_recovery(owner;
                controllers=(Symbol("a-integrator"), Symbol("b-integrator")), participants=(Symbol("a-pair"), :limit))
            @test all(node -> node.prepared.plan.closed,
                filter(node -> node.name in (Symbol("a-integrator"), Symbol("b-integrator")), composition.graph.nodes))
        end
    end
    mktempdir() do directory
        state = prepared(directory)
        try
            @test props(state)["pipewireao-recovery:version"] === Int32(2)
            @test props(state)["pipewireao-recovery:controller-count"] == 2
            @test !haskey(props(state), "pipewireao-recovery:controller")
            @test_throws ArgumentError FGP.stage_properties!(state.node,
                SPA.Props("pipewireao-recovery:controller-1-ready" => true))
            @test_throws ArgumentError FGP.prepare_recovery(state.node; controllers=(Symbol("a-integrator"), Symbol("b-integrator")), participants=(:limit,))
            @test_throws ArgumentError FGP.prepare_recovery(state.node; controllers=(Symbol("a-integrator"), Symbol("b-integrator")), participants=(Symbol("a-pair"),))
            @test_throws ArgumentError FGP.prepare_recovery(state.node; controller=Symbol("a-integrator"), controllers=(Symbol("a-integrator"), Symbol("b-integrator")))
            @test_throws ArgumentError FGP.prepare_recovery(state.node; controllers=(Symbol("a-integrator"),), participants=(Symbol("a-pair"), :limit))
            FGP.stage_properties!(state.node, SPA.Props("b-integrator:gain" => 1f0)); apply!(state)
            @test !step!(state, :a, 1; value=2f0)
            @test step!(state, :a, 2; value=0f0)
            @test step!(state, :b, 1; value=2f0)
            old_current = copy(node(state, :limit).workspace.committed_current)
            identity = node(state, Symbol("a-pair")).workspace.committed_identity
            observation = JFG.composition_observation(state.composition)
            watermark = copy(state.composition.last_inputs)
            @test any(!iszero, state.composition.branches[1].feedback_buffer)
            request!(state)
            @test_throws FGA.PropertyUpdateBusy request!(state, 2)
            @test status(state).closure_inhibited
            apply!(state) # No camera arrival and no command publication.
            @test status(state).status === :applied
            @test status(state).controller_requested_generations == (UInt64(1), UInt64(2))
            @test status(state).controller_active_generations == (UInt64(1), UInt64(2))
            @test !status(state).reference_reached
            @test JFG.composition_observation(state.composition).identity == observation.identity
            @test state.composition.last_inputs == watermark
            @test all(isnothing, state.composition.contributors)
            @test node(state, :limit).workspace.committed_current == old_current
            @test node(state, :flat).plan.offset == [.1f0]
            @test node(state, Symbol("a-pair")).workspace.committed_identity == identity
            @test !node(state, Symbol("a-pair")).workspace.committed_primed
            for branch in state.composition.branches
                @test all(iszero, branch.feedback_buffer)
                @test all(iszero, JFG._route_value(branch.route, (), state.composition.graph.node_outputs))
            end
            @test_throws InvalidStateException FGP.stage_recovery_release!(state.scope, UInt64(1))
            @test_throws InvalidStateException close_branch!(state, :b)
            @test_throws InvalidStateException FGP.replace_parameters!(state.node, :flat => Float32[.2])
            # LLOWFS alone drives the retained SCC reference and conditioner slew.
            for sequence in 2:7; @test step!(state, :b, sequence); end
            @test state.outputs.command ≈ [.1f0]
            @test status(state).reference_reached
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test props(state)["pipewireao-recovery:controller-2-ready"]
            release!(state)
            @test !status(state).closure_inhibited
            @test !node(state, Symbol("a-integrator")).plan.closed
            @test !node(state, Symbol("b-integrator")).plan.closed
            property_bytes(owner) = @allocated FGP._adopt_pending_properties!(owner)
            property_bytes(state.node.owner)
            close_branch!(state, :b)
            @test property_bytes(state.node.owner) == 0
            @test node(state, Symbol("b-integrator")).plan.closed
            @test props(state)["pipewireao-recovery:controller-1-close-guarded"]
            @test !props(state)["pipewireao-recovery:controller-2-close-guarded"]
            @test_throws InvalidStateException close_branch!(state, :a)
            # Legitimate active LLOWFS contribution must not forbid later SCC close.
            @test step!(state, :b, 8; value=.05f0)
            @test !status(state).reference_reached
            @test !step!(state, :a, 3; value=0f0) # Must not pair with pre-recovery exposure 2.
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test step!(state, :a, 4; value=0f0)
            @test props(state)["pipewireao-recovery:controller-1-ready"]
            close_branch!(state, :a); apply!(state)
            @test node(state, Symbol("a-integrator")).plan.closed
            @test !status(state).close_guarded
            @test step!(state, :a, 5; value=0f0)
            @test state.outputs.command ≈ [.15f0]
            @test node(state, Symbol("a-integrator")).workspace.committed_output == [0f0]
        finally
            close(state.node)
        end
    end
end

@testset "Coupled readiness loss, guarded adoption, reset and allocation" begin
    mktempdir() do directory
        state = prepared(directory)
        try
            request!(state); apply!(state)
            for sequence in 1:3; step!(state, :b, sequence); end
            @test status(state).reference_reached
            @test props(state)["pipewireao-recovery:controller-1-close-guarded"]
            @test props(state)["pipewireao-recovery:controller-2-close-guarded"]
            release!(state)
            close_branch!(state, :b)
            state.node.owner.recovery_observation_timestamp_ns[] = time_ns() - UInt64(60_000_000_001)
            apply!(state)
            @test !node(state, Symbol("b-integrator")).plan.closed
            @test !status(state).closure_inhibited
            @test props(state)["pipewireao-recovery:controller-2-close-guarded"]
            step!(state, :b, 4; value=-1f0) # Successful invalid measurement hold cannot establish readiness.
            @test !props(state)["pipewireao-recovery:controller-2-ready"]
            @test_throws InvalidStateException close_branch!(state, :b)
            step!(state, :b, 5)
            @test props(state)["pipewireao-recovery:controller-2-ready"]
            FGP._reset_owner!(state.node.owner)
            @test !props(state)["pipewireao-recovery:controller-2-ready"]
            @test_throws InvalidStateException close_branch!(state, :b)
            step!(state, :b, 6)
            @test props(state)["pipewireao-recovery:controller-2-ready"]
            @test !step!(state, :a, 1)
            @test step!(state, :a, 2)
            @test props(state)["pipewireao-recovery:controller-1-ready"]
            @test !step!(state, :a, 4; kind=OPTICAL_MIXED)
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test !step!(state, :a, 5)
            @test step!(state, :a, 6)
            @test props(state)["pipewireao-recovery:controller-1-ready"]
            # Actual warmed native callback and seqlocked observation path.
            buffers = input(state, :b, 100)
            previous = FGA.SampleMetadata(1)
            function bytes!()
                state.composition.last_inputs[2] = previous
                return @allocated step!(state, buffers)
            end
            bytes!()
            @test bytes!() == 0
            adoption_bytes() = @allocated FGP._adopt_pending_recovery!(state.node.owner)
            adoption_bytes()
            request!(state, 2)
            @test adoption_bytes() == 0
            @test !status(state).reference_reached
        finally
            close(state.node)
        end
    end
end

@testset "Branch freshness is independent and recovery failures never replay" begin
    mktempdir() do directory
        state = prepared(directory)
        try
            request!(state); apply!(state)
            step!(state, :b, 1)
            release!(state)
            state.node.owner.recovery_branch_timestamps[2][] = time_ns() - UInt64(60_000_000_001)
            @test !step!(state, :a, 1)
            @test step!(state, :a, 2)
            @test status(state).reference_reached
            @test props(state)["pipewireao-recovery:controller-1-ready"]
            @test !props(state)["pipewireao-recovery:controller-2-ready"]
            @test_throws InvalidStateException close_branch!(state, :b)
            FGP.replace_parameters!(state.node, :flat => Float32[.2])
            @test step!(state, :b, 2)
            @test status(state).reference_reached
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test props(state)["pipewireao-recovery:controller-2-ready"]
            witness = props(state)["pipewireao-recovery:observation-sequence"]
            FGP._reset_owner!(state.node.owner)
            step!(state, :b, 1)
            @test props(state)["pipewireao-recovery:observation-sequence"] > witness
            @test_throws InvalidStateException step!(state, :b, 2; value=Inf32)
            @test state.composition.failed
            @test !status(state).reference_reached
            @test !props(state)["pipewireao-recovery:controller-2-ready"]
            @test_throws InvalidStateException request!(state, 2)
        finally
            close(state.node)
        end
    end
    mktempdir() do directory
        state = prepared(directory)
        try
            request!(state)
            owner = state.node.owner
            transaction = owner.recovery_transaction
            index = findfirst(node -> node.name === Symbol("a-input"), owner.graph.nodes)
            owner.recovery_transaction = FGP.PreparedPropertyTransaction(transaction.plans,
                Base.setindex(transaction.touched, true, index)) # Inject unexpected lifecycle failure after cold admission.
            @test_throws MethodError apply!(state)
            @test status(state).status === :unknown
            @test status(state).closure_inhibited && state.composition.failed
            @test !FGP._adopt_pending_recovery!(owner)
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test_throws InvalidStateException FGP.stage_recovery_release!(state.scope, UInt64(1))
        finally
            close(state.node)
        end
    end
end

@testset "Guarded close stages from published evidence and rechecks live state" begin
    mktempdir() do directory
        state = prepared(directory)
        try
            request!(state); apply!(state)
            step!(state, :b, 1)
            release!(state)
            workspace = node(state, :limit).workspace
            current = copy(workspace.committed_current)
            # Model scientific mutation concurrent with control preparation.
            # Preparation may inspect only published atomic evidence; adoption
            # owns the scientific state and must detect the changed command.
            fill!(workspace.committed_current, .3f0)
            @test close_branch!(state, :b) === nothing
            apply!(state)
            @test !node(state, Symbol("b-integrator")).plan.closed
            @test props(state)["pipewireao-recovery:controller-2-close-guarded"]
            @test !status(state).closure_inhibited
            copyto!(workspace.committed_current, current)
            step!(state, :b, 2)
            close_branch!(state, :b); apply!(state)
            @test node(state, Symbol("b-integrator")).plan.closed
        finally
            close(state.node)
        end
    end
end

@testset "Foreign coupled scopes cannot publish matching native evidence" begin
    mktempdir() do directory
        state = prepared(directory)
        try
            foreign = FGP.prepare_recovery(state.node; controllers=(Symbol("b-integrator"), Symbol("a-integrator")),
                participants=(Symbol("a-pair"), :limit), reference_timeout_ms=60000)
            generation = FGP.stage_recovery!(foreign)
            apply!(state)
            step!(state, :b, 1)
            @test FGP.recovery_status(foreign).reference_reached
            @test !status(state).scope_matches
            @test !props(state)["pipewireao-recovery:reference-reached"]
            @test !props(state)["pipewireao-recovery:controller-1-ready"]
            @test !props(state)["pipewireao-recovery:controller-2-ready"]
            @test props(state)["pipewireao-recovery:controller-1-observation-sequence"] == 0
            @test_throws InvalidStateException FGP.stage_recovery_release!(state.scope, generation)
        finally
            close(state.node)
        end
    end
end
end
