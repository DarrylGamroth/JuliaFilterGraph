module CapturedValidationTests

using FilterGraphAlgorithms
using FilterGraphs
using Test

struct ValidationProducer{O}
    output::O
end

struct ValidationReader{F, H}
    finite_inputs::F
    hook_called::H
end

struct ValidationOpaque end

struct ValidationWriter{O}
    output::O
end

struct TwoPortValidationProducer end

FilterGraphAlgorithms.preserves_inputs(::ValidationProducer) = true
FilterGraphAlgorithms.finite_output(::ValidationProducer, ::Val{1}) = true

FilterGraphAlgorithms.preserves_inputs(::ValidationReader) = true
function FilterGraphAlgorithms.prepare_validation(
        adapter::ValidationReader,
        proofs::Tuple
)
    adapter.hook_called[] = true
    return ValidationReader(proofs, adapter.hook_called)
end

FilterGraphAlgorithms.preserves_inputs(::ValidationWriter) = true
FilterGraphAlgorithms.preserves_inputs(::TwoPortValidationProducer) = true
FilterGraphAlgorithms.finite_output(::TwoPortValidationProducer, ::Val{1}) = true
FilterGraphAlgorithms.finite_output(::TwoPortValidationProducer, ::Val{2}) = false

validation_node(adapter) = FilterGraphs.CapturedNode{Nothing, typeof(adapter)}(adapter)
validation_outputs(buffers...) = map(buffer -> (buffer,), buffers)

@testset "captured finite-input validation reuse" begin
    external = (input = Float32[1, 2],)
    produced = Float32[1, 2]
    reader_output = Float32[0, 0]

    # External inputs remain unproven, while an exact output from a qualified
    # producer proves the corresponding downstream input.
    external_reader = ValidationReader((), Ref(false))
    nodes = (validation_node(external_reader),)
    result = FilterGraphs._reuse_captured_validation(
        nodes, ((FilterGraphs.ExternalInput{1}(),),), ((reader_output,),),
        external, ((produced,),)
    )
    @test result[1].adapter.finite_inputs == (false,)
    @test result[1].adapter.hook_called[]

    first_reader = ValidationReader((), Ref(false))
    second_reader = ValidationReader((), Ref(false))
    nodes = (validation_node(ValidationProducer(produced)),
        validation_node(first_reader), validation_node(second_reader))
    routes = ((FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.NodeOutput{1, 1}(),),
        (FilterGraphs.NodeOutput{1, 1}(),))
    outputs = validation_outputs(produced, Float32[0, 0], Float32[0, 0])
    result = FilterGraphs._reuse_captured_validation(
        nodes, routes, outputs, external, outputs
    )
    @test result[2].adapter.finite_inputs == (true,)
    @test result[3].adapter.finite_inputs == (true,)

    # Default traits make producers unqualified, and an opaque adapter clears
    # earlier proofs before a later reader can consume them.
    opaque_producer = ValidationOpaque()
    reader = ValidationReader((), Ref(false))
    nodes = (validation_node(ValidationProducer(produced)),
        validation_node(opaque_producer), validation_node(reader))
    outputs = validation_outputs(produced, Float32[0, 0], Float32[0, 0])
    result = FilterGraphs._reuse_captured_validation(
        nodes,
        ((FilterGraphs.ExternalInput{1}(),),
            (FilterGraphs.NodeOutput{1, 1}(),),
            (FilterGraphs.NodeOutput{1, 1}(),)),
        outputs, external, outputs
    )
    @test result[3].adapter.finite_inputs == (false,)

    # Per-input flags preserve external-input uncertainty alongside a proven
    # internal edge. Output guarantees are indexed independently by port.
    mixed_reader = ValidationReader((), Ref(false))
    nodes = (validation_node(ValidationProducer(produced)),
        validation_node(mixed_reader))
    mixed_outputs = ((produced,), (Float32[0, 0], Float32[0, 0]))
    routes = ((FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.NodeOutput{1, 1}(), FilterGraphs.ExternalInput{1}()))
    result = FilterGraphs._reuse_captured_validation(
        nodes, routes, mixed_outputs, external, mixed_outputs
    )
    @test result[2].adapter.finite_inputs == (true, false)

    two1, two2 = Float32[1], Float32[2]
    two_reader_1 = ValidationReader((), Ref(false))
    two_reader_2 = ValidationReader((), Ref(false))
    nodes = (validation_node(TwoPortValidationProducer()),
        validation_node(two_reader_1), validation_node(two_reader_2))
    outputs = ((two1, two2), (Float32[0],), (Float32[0],))
    routes = ((FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.NodeOutput{1, 1}(),),
        (FilterGraphs.NodeOutput{1, 2}(),))
    result = FilterGraphs._reuse_captured_validation(
        nodes, routes, outputs, external, outputs
    )
    @test result[2].adapter.finite_inputs == (true,)
    @test result[3].adapter.finite_inputs == (false,)

    # A later output that aliases a proven buffer expires that proof for all
    # subsequent readers, even when its own input is disjoint.
    writer_output = Float32[0, 0]
    trailing_reader = ValidationReader((), Ref(false))
    nodes = (validation_node(ValidationProducer(produced)),
        validation_node(ValidationWriter(writer_output)),
        validation_node(trailing_reader))
    outputs = ((produced,), (produced,), (Float32[0, 0],))
    routes = ((FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.NodeOutput{1, 1}(),))
    result = FilterGraphs._reuse_captured_validation(
        nodes, routes, outputs, external, outputs
    )
    @test result[3].adapter.finite_inputs == (false,)

    # Input/output overlap prevents the refinement hook from being called.
    alias_reader = ValidationReader((), Ref(false))
    nodes = (validation_node(ValidationProducer(produced)), validation_node(alias_reader))
    outputs = ((produced,), (produced,))
    routes = ((FilterGraphs.ExternalInput{1}(),),
        (FilterGraphs.NodeOutput{1, 1}(),))
    result = FilterGraphs._reuse_captured_validation(
        nodes, routes, outputs, external, outputs
    )
    @test !result[2].adapter.hook_called[]
    @test result[2].adapter.finite_inputs == ()

    # Proofs attach to exact objects, not overlapping views.
    view_reader = ValidationReader((), Ref(false))
    aliased_view = @view produced[:]
    @test aliased_view !== produced
    @test Base.mightalias(aliased_view, produced)
    @test !FilterGraphs._captured_input_finite(
        FilterGraphs.NodeOutput{1, 1}(), external, ((aliased_view,),), [produced]
    )

    # Non-CUDA backends retain their original adapters by default.
    original = (validation_node(view_reader),)
    @test FilterGraphs._prepare_captured_validation(
        :fake, original, (), (), external, ()
    ) === original
end

end # module CapturedValidationTests
