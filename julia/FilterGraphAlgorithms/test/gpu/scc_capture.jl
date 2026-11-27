using LinearAlgebra

import FilterGraphAlgorithms:
    complete!,
    enqueue!,
    prepare_capture,
    process!

function _check_scc_capture(device, upload, allocate, device_fill, synchronize)
    roi = Bool[
        true,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
    ]
    phase = Float32[0, 0, 0, 0, pi / 2, 0, 0, 0, 0, 0, 0, 0]
    spatial = prepare_algorithm(
        SccSpatialCarrierDemodulationF32,
        SccSpatialCarrierDemodulationF32Configuration(
            image_size=(3, 4),
            roi=roi,
            fir_tap_count=3,
            initial_reference_difference=ones(Float32, 12),
            initial_static_validity=nothing,
            initial_carrier_phase=phase,
            initial_fir=Float32[0, 1, 0],
            minimum_support=0.8f0,
            minimum_valid_fraction=2.0f0 / 3.0f0,
        ),
    )
    difference_values = Float32.(1:12)
    difference = upload(difference_values)
    difference_validity = upload(fill(false, 12))
    pair_status = upload(UInt8[UInt8(ChoppedImagePriming)])
    spatial_outputs = (
        allocate(Float32, 6),
        allocate(Float32, 6),
        allocate(Bool, 6),
        allocate(Float32, 1),
    )
    frame_status = allocate(Int32, 1)
    captured_spatial = prepare_capture(
        CompleteTransition(),
        device,
        spatial,
        spatial_outputs,
        (difference, difference_validity, pair_status),
        frame_status,
        Int32(1),
    )
    @test Array(difference) == difference_values
    @test all(!, Array(difference_validity))
    @test only(Array(pair_status)) == UInt8(ChoppedImagePriming)
    @test_throws ArgumentError prepare_capture(
        CompleteTransition(), device, spatial,
        (first(spatial_outputs), first(spatial_outputs), spatial_outputs[3], spatial_outputs[4]),
        (difference, difference_validity, pair_status), frame_status, Int32(1),
    )
    fill!(difference_validity, true)
    fill!(pair_status, UInt8(ChoppedImageComplete))
    enqueue!(captured_spatial)
    synchronize()
    @test only(Array(frame_status)) == 0
    @test complete!(captured_spatial, Int32(0)) === nothing

    cpu_spatial_outputs = (
        zeros(Float32, 6),
        zeros(Float32, 6),
        fill(false, 6),
        zeros(Float32, 1),
    )
    @test process!(
        cpu_spatial_outputs...,
        spatial,
        difference_values,
        fill(true, 12),
        UInt8[UInt8(ChoppedImageComplete)],
    ) === nothing
    @test Array.(spatial_outputs) == cpu_spatial_outputs

    interaction = Float32[1 0; 0 1; 1 1; 2 -1]
    regularization_operator = Float32[1 0; 0 1]
    weighted = prepare_algorithm(
        SccWeightedRegularizedReconstructorF32,
        SccWeightedRegularizedReconstructorF32Configuration(
            measurement_count=4,
            controller_coordinate_count=2,
            regularization_row_count=2,
            measurement_schema=SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
            initial_interaction_matrix=
                collect(vec(transpose(interaction))),
            initial_regularization_operator=
                collect(vec(transpose(regularization_operator))),
            regularization=0.1f0,
            iterations=8,
            relative_tolerance=1.0f-5,
            minimum_diagonal=1.0f-6,
        ),
    )
    measurement_values = interaction * Float32[2, -1]
    weight_values = Float32[1, 0.5, 1.5, 2]
    measurements = upload(measurement_values)
    weights = upload(weight_values)
    weighted_output = allocate(Float32, 2)
    fill!(frame_status, Int32(0))
    captured_weighted = prepare_capture(
        CompleteTransition(),
        device,
        weighted,
        (weighted_output,),
        (measurements, weights),
        frame_status,
        Int32(2),
    )
    copyto!(measurements, measurement_values)
    copyto!(weights, weight_values)
    enqueue!(captured_weighted)
    synchronize()
    @test only(Array(frame_status)) == 0
    cpu_weighted_output = zeros(Float32, 2)
    @test process!(
        cpu_weighted_output,
        weighted,
        measurement_values,
        weight_values,
    ) === nothing
    @test Array(weighted_output) ≈ cpu_weighted_output rtol = 2.0f-5

    fill!(frame_status, Int32(0))
    invalid_weights = copy(weight_values)
    invalid_weights[2] = -1
    copyto!(weights, invalid_weights)
    enqueue!(captured_weighted)
    synchronize()
    status = only(Array(frame_status))
    @test complete!(captured_weighted, status) ===
          SccWeightedRegularizedReconstructionInvalidWeight

    singular = prepare_algorithm(
        SccWeightedRegularizedReconstructorF32,
        SccWeightedRegularizedReconstructorF32Configuration(
            measurement_count=4,
            controller_coordinate_count=2,
            regularization_row_count=2,
            measurement_schema=SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
            initial_interaction_matrix=zeros(Float32, 8),
            initial_regularization_operator=zeros(Float32, 4),
            regularization=0.0f0,
            iterations=1,
            relative_tolerance=1.0f-5,
            minimum_diagonal=1.0f-6,
        ),
    )
    fill!(frame_status, Int32(0))
    singular_output = device_fill(71.0f0, 2)
    captured_singular = prepare_capture(
        CompleteTransition(),
        device,
        singular,
        (singular_output,),
        (measurements, weights),
        frame_status,
        Int32(3),
    )
    copyto!(measurements, measurement_values)
    copyto!(weights, weight_values)
    enqueue!(captured_singular)
    synchronize()
    status = only(Array(frame_status))
    @test complete!(captured_singular, status) ===
          SccWeightedRegularizedReconstructionSingularSystem
    @test Array(singular_output) == fill(71.0f0, 2)
end

function _check_scc_residual(device, upload, allocate)
    configuration = (
        pixel_count=3, mask=Bool[true, false, true],
        initial_reference_image=Float32[1, 10, -1], minimum_valid_fraction=0.5f0,
    )
    prepared = prepare_algorithm(SccPixelResidualF32,
        SccPixelResidualF32Configuration(; configuration...))
    difference = Float32[5, 6, 7]
    validity = Bool[true, false, false]
    status = UInt8[UInt8(ChoppedImagePriming)]
    input = upload.((difference, validity, status))
    outputs = (zeros(Float32, 2), fill(false, 2), zeros(Float32, 1))
    capture_outputs = upload.(outputs)
    capture_status = allocate(Int32, 1)
    prepare_capture(CompleteTransition(), device, prepared, capture_outputs,
        input, capture_status, Int32(1))
    @test Array.(input) == (difference, validity, status)

    document = (;
        nodes=((name=:residual, label="scc-pixel-residual-f32", config=configuration),),
        links=(),
        inputs=(difference="residual:difference", validity="residual:difference-validity",
            status="residual:pair-status"),
        outputs=(residuals="residual:selected-residuals", validity="residual:selected-validity",
            fraction="residual:valid-fraction"),
    )
    graph = prepare_captured_graph(
        prepare_graph(document; algorithms=(SccPixelResidualF32,)), device)
    graph_outputs = (residuals=similar(outputs[1]), validity=similar(outputs[2]),
        fraction=similar(outputs[3]))
    status[1] = UInt8(ChoppedImageComplete)
    @test process!(outputs..., prepared, difference, validity, status) === nothing
    @test FilterGraphs.process!(graph_outputs, graph, (; difference, validity, status)) ==
        (residuals=true, validity=true, fraction=true)
    @test values(graph_outputs) == outputs
end

function _scc_reconstruction_graph(; linked=true)
    limit = floatmax(Float32)
    residual = (name=:residual, label="scc-pixel-residual-f32", config=(
        pixel_count=5, mask=Bool[true, false, true, true, true],
        initial_reference_image=Float32[1, 0, 0, -limit, 0], minimum_valid_fraction=0.5f0,
    ))
    reconstruction = (name=:reconstruction, label="dense-vector-reconstructor-f32", config=(
        input_count=4, output_count=2, initial_reconstructor=Float32[limit, 1, 0, 0, 0, 0, 1, 2],
        input_schema=SCC_SELECTED_PIXEL_RESIDUALS_V1, output_schema="test.modal-residual/1",
        reconstructor_schema="test.modal-reconstructor/1",
    ))
    inputs = (difference="residual:difference", validity="residual:difference-validity",
        status="residual:pair-status")
    document = (
        nodes=(residual, reconstruction),
        links=linked ? ((output="residual:selected-residuals", input="reconstruction:ordered-input"),) : (),
        inputs=linked ? inputs : merge(inputs, (ordered="reconstruction:ordered-input",)),
        outputs=(modal="reconstruction:output", residuals="residual:selected-residuals",
            valid="residual:selected-validity", fraction="residual:valid-fraction"),
    )
    return prepare_graph(document; algorithms=(SccPixelResidualF32, DenseVectorReconstructorF32))
end

function _check_scc_reconstruction_guards(device, allocate, checked, captured)
    residual, reconstruction = captured.nodes
    adapter = reconstruction.adapter
    algorithms = parentmodule(typeof(adapter))
    backend = parentmodule(typeof(captured.context))
    output = (adapter.output,)
    input = (adapter.input,)
    function dense_node(status)
        full = prepare_capture(CompleteTransition(), device, checked.nodes[2].prepared,
            output, input, status, Int32(2))
        return FilterGraphs._captured_node_with_adapter(reconstruction, full)
    end
    original = dense_node(adapter.frame_status)
    @test !(original.adapter.validate_input! isa algorithms._FiniteInput)
    buffers = ((residual.adapter.selected_residuals, residual.adapter.selected_validity,
        residual.adapter.valid_fraction), output)
    routes = checked.input_routes
    refine(nodes, routes, stage_outputs, outputs) = backend._reuse_scc_residual_validation(
        nodes, eachindex(nodes), routes, stage_outputs, captured.device_inputs, outputs)
    nodes = (residual, original)
    @test refine(nodes, routes, buffers, buffers)[2].adapter.validate_input! isa algorithms._FiniteInput

    shifted_routes = (routes[1], (FilterGraphs.NodeOutput{3, 1}(),))
    global_outputs = ((), (), buffers...)
    shifted = if backend === Base.get_extension(FilterGraphs, :FilterGraphsCUDAExt)
        FilterGraphs._prepare_captured_validation(device, nodes, shifted_routes,
            buffers, captured.device_inputs, global_outputs; indices=3:4)
    else
        backend._reuse_scc_residual_validation(nodes, 3:4, shifted_routes,
            buffers, captured.device_inputs, global_outputs)
    end
    @test shifted[2].adapter.validate_input! isa algorithms._FiniteInput

    alias_output = @view adapter.input[1:length(adapter.output)]
    @test FilterGraphs._captured_mightalias(alias_output, adapter.input)
    @test refine(nodes, routes, (buffers[1], (alias_output,)), buffers)[2] === original

    # Even an equal-sized, overlapping view is a different routed object.
    view_buffer = @view adapter.input[:]
    @test view_buffer !== adapter.input
    different_buffers = ((view_buffer, Base.tail(buffers[1])...), output)
    @test refine(nodes, routes, buffers, different_buffers)[2] === original
    wrong_port = (routes[1], (FilterGraphs.NodeOutput{1, 2}(),))
    @test refine(nodes, wrong_port, buffers, buffers)[2] === original

    separate_status = dense_node(allocate(Int32, 1))
    @test refine((residual, separate_status), routes, buffers, buffers)[2] === separate_status
end

function _check_scc_reconstruction_capture(device, allocate)
    cpu = _scc_reconstruction_graph()
    captured = prepare_captured_graph(_scc_reconstruction_graph(), device)
    residual, reconstruction = map(node -> node.adapter, captured.nodes)
    algorithms = parentmodule(typeof(reconstruction))
    @test reconstruction.validate_input! isa algorithms._FiniteInput
    @test !(reconstruction.validate_output! isa algorithms._FiniteInput)
    @test reconstruction.input === residual.selected_residuals
    @test reconstruction.frame_status === residual.frame_status
    _check_scc_reconstruction_guards(device, allocate, cpu, captured)

    actual = (modal=zeros(Float32, 2), residuals=zeros(Float32, 4),
        valid=fill(false, 4), fraction=zeros(Float32, 1))
    expected = map(copy, actual)
    limit = floatmax(Float32)
    partial = Float32[1.25, 999, NaN, limit, 2]
    complete = UInt8(ChoppedImageComplete)
    inputs(difference, status) = (; difference, validity=fill(true, 5), status=UInt8[status])
    metadata(sequence) = (difference=SampleMetadata(sequence), validity=nothing, status=nothing)
    success = (modal=true, residuals=true, valid=true, fraction=true)
    @test process!(actual, captured, inputs(partial, complete), metadata(1)) == success
    @test process!(expected, cpu, inputs(partial, complete), metadata(1)) == success
    @test actual == expected
    @test actual.residuals == Float32[0.25, 0, 0, 2]
    @test actual.valid == Bool[true, false, false, true]
    @test actual.fraction == Float32[0.5]
    published = map(copy, actual)
    published_metadata = FilterGraphs.output_metadata(captured)
    @test published_metadata == FilterGraphs.output_metadata(cpu)

    overflow = Float32[3, 999, NaN, limit, 2]
    insufficient = Float32[3, 999, NaN, limit, -Inf]
    failures = (
        (overflow, UInt8(ChoppedImagePriming), SccPixelResidualIncompletePair, :residual),
        (overflow, UInt8(0xff), SccPixelResidualInvalidPairStatus, :residual),
        (insufficient, complete, SccPixelResidualInsufficientValidPixels, :residual),
        (overflow, complete, DenseReconstructionNonFiniteOutput, :reconstruction),
    )
    for (sequence, (difference, status, failure, node)) in enumerate(failures)
        frame, sample = inputs(difference, status), metadata(sequence + 1)
        @test process!(actual, captured, frame, sample) === failure
        @test process!(expected, cpu, frame, sample) === failure
        @test actual == expected == published
        @test FilterGraphs.output_metadata(captured) == published_metadata
        diagnostic = FilterGraphs.rejection_context(captured)
        @test diagnostic.node === node
        @test diagnostic.reason === failure
        @test diagnostic.inputs == sample
    end
    @test process!(actual, captured, inputs(partial, complete), metadata(6)) == success
    @test actual == published
    @test FilterGraphs.output_metadata(captured).modal == metadata(6).difference
    @test FilterGraphs.rejection_context(captured) === nothing

    external = prepare_captured_graph(_scc_reconstruction_graph(linked=false), device)
    @test !(external.nodes[2].adapter.validate_input! isa algorithms._FiniteInput)
    invalid = merge(inputs(partial, complete), (ordered=Float32[NaN, 0, 0, 0],))
    @test process!(actual, external, invalid) === DenseReconstructionNonFiniteInput
    @test actual == published
end

function run_scc_capture_tests(device, upload, allocate, device_fill, synchronize)
    @testset "Shared SCC capture adapters match CPU on $device" begin
        _check_scc_capture(device, upload, allocate, device_fill, synchronize)
    end
    @testset "SCC pixel residual begins with caller-owned input storage" begin
        _check_scc_residual(device, upload, allocate)
    end
    @testset "SCC residual proves the adjacent reconstruction input on $device" begin
        _check_scc_reconstruction_capture(device, allocate)
    end
end
