using LinearAlgebra
import FilterGraphAlgorithms: process!, reset!

function run_structured_projection_tests(device, ext, to_device, device_zeros, synchronize)
    function selection(columns, count)
        matrix = zeros(Float32, length(columns), count)
        for row in eachindex(columns)
            matrix[row, columns[row]] = 1
        end
        return matrix
    end
    @testset "shared GPU structured projections preserve bits and all-column rejection" begin
        count = 523
        columns = [mod(17row, count) + 1 for row in 1:277]
        columns[2] = columns[1]
        columns[end] = count
        matrices = (Matrix{Float32}(I, count, count), selection(columns, count))
        values = collect(reinterpret(Float32, UInt32[
            mod(index, 4) == 0 ? 0x80000001 : mod(index, 4) == 1 ? 0x80000000 :
            mod(index, 4) == 2 ? 0x00000001 : 0x00000000 for index in 1:count]))
        input = to_device(values)
        status = device_zeros(Int32, 1)
        for (matrix, indices, type) in ((matrices[1], collect(1:count), ext.IdentityProjection),
                (matrices[2], columns, ext.SelectionProjection))
            output = device_zeros(Float32, length(indices))
            projection = ext._prepare_projection(device, matrix, output, input, 128)
            @test projection isa type
            @test isconcretetype(typeof(projection))
            @test ext._projection_matrix(projection) === nothing
            fill!(matrix, 7) # Classification and columns are a cold snapshot.
            copyto!(input, values)
            fill!(status, 0)
            ext._enqueue_projection!(projection, output, status, input)
            synchronize()
            @test reinterpret(UInt32, Array(output)) == reinterpret(UInt32, values[indices])
            input_pointer, output_pointer = pointer(input), pointer(output)
            fill!(output, 71)
            fill!(status, Int32(0x103))
            fill!(input, NaN32)
            ext._enqueue_projection!(projection, output, status, input)
            synchronize()
            @test only(Array(status)) == 0x103
            @test all(==(71), Array(output))
            if projection isa ext.SelectionProjection
                unselected = findlast(index -> index ∉ indices, eachindex(values))
                @test !isnothing(unselected)
                for nonfinite in (Inf32, -Inf32, NaN32)
                    bad = copy(values)
                    bad[unselected] = nonfinite
                    copyto!(input, bad)
                    fill!(status, 0)
                    ext._enqueue_projection!(projection, output, status, input)
                    synchronize()
                    @test all(isnan, Array(output))
                    @test only(Array(status)) == 0 # Final validator owns attribution.
                end
            end
            @test pointer(input) == input_pointer
            @test pointer(output) == output_pointer
        end
    end

    @testset "shared structured feedback keeps conservative alias boundaries" begin
        identity = Matrix{Float32}(I, 3, 3)
        selected = selection([3, 1], 3)
        dense = copy(identity)
        dense[1, 2] = 0.25f0
        for (first, last) in ((identity, identity), (identity, selected),
                (selected, Matrix{Float32}(I, 2, 2)), (identity, dense), (dense, identity))
            for overlapping in (false, true)
                prepared = prepare_algorithm(PdmFeedbackToVdmF32,
                    PdmFeedbackToVdmF32Configuration(active_vdm_count=size(last, 1),
                        full_vdm_count=size(first, 1), physical_actuator_count=size(first, 2)))
                replace_parameter!(prepared, "pdm-to-vdm", first)
                replace_parameter!(prepared, "full-to-active", last)
                storage = device_zeros(Float32, size(first, 2) + size(last, 1) - 1)
                input = overlapping ? view(storage, 1:size(first, 2)) :
                    device_zeros(Float32, size(first, 2))
                output = overlapping ? view(storage, size(first, 2):length(storage)) :
                    device_zeros(Float32, size(last, 1))
                status = device_zeros(Int32, 1)
                captured = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), device,
                    prepared, (output,), (input,), status, Int32(2))
                structured = first !== dense && last !== dense
                @test (captured.full_vdm_feedback === input ||
                    captured.full_vdm_feedback === output) == (structured && !overlapping)
                values = Float32[-0.0, 0.125, -0.25]
                expected = zeros(Float32, length(output))
                @test process!(expected, prepared, values) === nothing
                copyto!(input, values)
                FilterGraphAlgorithms.enqueue!(captured)
                synchronize()
                @test only(Array(status)) == 0
                @test Array(output) ≈ expected
                before_input, before_output = Array(input), Array(output)
                FilterGraphAlgorithms.reset!(captured)
                synchronize()
                @test isequal(Array(input), before_input)
                @test isequal(Array(output), before_output)
                if structured
                    fill!(status, Int32(0x103))
                    FilterGraphAlgorithms.enqueue!(captured)
                    synchronize()
                    @test only(Array(status)) == 0x103
                    @test isequal(Array(output), before_output)
                end
                bad = copy(values)
                bad[2] = NaN32
                copyto!(input, bad)
                before_output = Array(output)
                fill!(status, 0)
                FilterGraphAlgorithms.enqueue!(captured)
                synchronize()
                @test FilterGraphAlgorithms.complete!(captured, only(Array(status))) ===
                    PdmFeedbackToVdmNonFiniteInput
                # Dense CUDA candidates remain deliberately unpredicated.
                if structured
                    @test isequal(Array(output), before_output)
                end
            end
        end
    end

    @testset "shared reverse feedback embeds finite values and gates publication" begin
        rows, columns = 301, 259
        selected = [mod(37 * column, rows) + 1 for column in 1:columns]
        matrix = zeros(Float32, rows, columns)
        for (column, row) in pairs(selected)
            matrix[row, column] = 1f0
        end
        prepared = mktemp() do path, io
            write(io, """
            filter.graph = {
              nodes = [ { type = ndarray name = feedback label = vdm-feedback-to-controller-f32
                config = { controller_command_count = $rows active_vdm_count = $columns } }
                { type = ndarray name = loop label = closed-loop-correction-f32
                  config = { state_length = $rows hidden_mode_count = 1 }
                  props = { gain = 0.5 pole = 0.25 anti-windup-gain = 0.5 hidden-mode-gain = 0.0 } } ]
              inputs = [ "feedback:vdm-constraint-feedback" "feedback:vdm-to-controller"
                "loop:residual-error" "loop:constraint-feedback" "loop:shape-to-hidden" "loop:hidden-to-shape" ]
              outputs = [ "feedback:controller-constraint-feedback" "loop:correction" ]
            }
            """)
            flush(io)
            prepare_graph(path; algorithms=(VdmFeedbackToControllerF32, ClosedLoopCorrectionF32))
        end
        parameter = Symbol("vdm-to-controller")
        replace_parameters!(prepared, parameter => matrix,
            "shape-to-hidden" => zeros(Float32, 1, rows), "hidden-to-shape" => zeros(Float32, rows, 1))
        output_name = Symbol("controller-constraint-feedback")
        capture(source) = prepare_captured_graph(source, device;
            feedback=(var"constraint-feedback"=output_name,), outputs=(output_name,))
        graph = capture(prepared)
        binding = only(graph.executable.feedback)
        captured = only(node.adapter for node in graph.nodes if node.adapter isa ext.VdmFeedbackToController)
        @test binding.source === captured.output
        @test nameof(typeof(captured.project)) == :EmbeddingProjection
        @test ext._projection_matrix(captured.project) === nothing
        @test captured.validate_output! isa ext._FiniteInput
        @test FilterGraphAlgorithms.alias_input(captured, Val(1)) === nothing
        @test captured.output !== captured.input

        subnormal = nextfloat(0f0)
        values = zeros(Float32, columns)
        values[1:7] .= (-0f0, subnormal, -subnormal, 0.25f0, -0.5f0,
            floatmax(Float32), -floatmax(Float32))
        expected = zeros(Float32, rows)
        prepared_algorithm = only(node.prepared for node in prepared.nodes if node.name === :feedback)
        @test process!(expected, prepared_algorithm, values) === nothing
        @test reinterpret(UInt32, expected[selected[1:3]]) == UInt32[0, 1, 0x80000001]

        input = to_device(values)
        inputs = (var"vdm-constraint-feedback"=input, var"residual-error"=device_zeros(Float32, rows))
        seen = Ref(0)
        function consume(outputs)
            output = only(outputs)
            @test output === captured.output
            @test reinterpret(UInt32, Array(output)) == reinterpret(UInt32, expected)
            seen[] += 1
        end
        @test FilterGraphs.process_device!(consume, graph, inputs) isa NamedTuple
        @test isequal(Array(binding.destination), expected)
        for nonfinite in (NaN32, Inf32, -Inf32)
            bad = copy(values)
            bad[end] = nonfinite
            copyto!(input, bad)
            @test FilterGraphs.process_device!(consume, graph, inputs) ===
                VdmFeedbackToControllerNonFiniteInput
            @test seen[] == 1
            @test isequal(Array(binding.destination), expected)
        end
        copyto!(input, values)
        @test reset!(graph) === graph
        @test all(iszero, Array(binding.destination))
        @test FilterGraphs.process_device!(consume, graph, inputs) isa NamedTuple
        @test seen[] == 2
        @test reinterpret(UInt32, Array(input)) == reinterpret(UInt32, values)

        output, status = device_zeros(Float32, rows), device_zeros(Int32, 1)
        direct = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), device,
            prepared_algorithm, (output,), (input,), status, Int32(2))
        fill!(status, Int32(0x103))
        fill!(output, 71f0)
        FilterGraphAlgorithms.enqueue!(direct)
        synchronize()
        @test only(Array(status)) == 0x103
        @test all(==(71f0), Array(output))

        storage = device_zeros(Float32, rows + 1)
        overlapping = FilterGraphAlgorithms.prepare_capture(CompleteTransition(), device,
            prepared_algorithm, (view(storage, 2:rows+1),), (view(storage, 1:columns),),
            status, Int32(2))
        @test ext._projection_matrix(overlapping.project) !== nothing

        mixed = FilterGraphAlgorithms.PreparedAlgorithm(VdmFeedbackToControllerF32,
            VdmFeedbackToControllerPlan(Float64.(matrix)), prepared_algorithm.workspace,
            prepared_algorithm.formats)
        if ext._projection_backend(device) === Val(:cublas)
            @test_throws ArgumentError FilterGraphAlgorithms.prepare_capture(
                CompleteTransition(), device, mixed, (output,), (input,), status, Int32(2))
        else
            mixed_capture = FilterGraphAlgorithms.prepare_capture(
                CompleteTransition(), device, mixed, (output,), (input,), status, Int32(2))
            @test eltype(ext._projection_matrix(mixed_capture.project)) === Float64
        end

        dense = copy(matrix)
        dense[selected[1], 1] = floatmax(Float32)
        replace_parameters!(prepared, parameter => dense)
        fallback = capture(prepared)
        fallback_adapter = only(node.adapter for node in fallback.nodes if node.adapter isa ext.VdmFeedbackToController)
        @test ext._projection_matrix(fallback_adapter.project) !== nothing
        overflow = zeros(Float32, columns)
        overflow[1] = 2f0
        copyto!(input, overflow)
        @test FilterGraphs.process_device!(consume, fallback, inputs) ===
            VdmFeedbackToControllerNonFiniteOutput
        @test seen[] == 2
        copyto!(input, values)
        @test FilterGraphs.process_device!(consume, graph, inputs) isa NamedTuple
        @test seen[] == 3
    end
end

function run_structured_projection_graph_tests(device, synchronize, used_memory)
    function graph(matrix)
        return mktemp() do path, io
            write(io, """
            filter.graph = {
              nodes = [ { type = ndarray name = projection label = controller-to-vdm-f32
                config = { controller_command_count = $(size(matrix, 2))
                           active_vdm_count = $(size(matrix, 1)) } } ]
              inputs = [ "projection:controller-command" "projection:controller-to-vdm" ]
              outputs = [ "projection:vdm-command" ]
            }
            """)
            flush(io)
            prepared = prepare_graph(path; algorithms=(ControllerToVdmF32,))
            replace_parameters!(prepared, Symbol("controller-to-vdm") => matrix)
            return prepared
        end
    end
    frame_bytes(output, graph, input) = @allocated process!(output, graph, input)
    @testset "shared structured graphs retain fixed storage and replay" begin
        for matrix in (Matrix{Float32}(I, 3, 3), Float32[0 0 1; 1 0 0])
            cpu, gpu = graph(matrix), prepare_captured_graph(graph(matrix), device)
            adapter = only(gpu.nodes).adapter
            reuse = FilterGraphAlgorithms._supports_validation_reuse(
                KernelAbstractions.get_backend(adapter.frame_status))
            @test FilterGraphAlgorithms.preserves_inputs(adapter) == reuse
            @test FilterGraphAlgorithms.finite_output(adapter, Val(1)) == reuse
            if !reuse
                @test FilterGraphAlgorithms.alias_input(adapter, Val(1)) === nothing
                @test FilterGraphAlgorithms.prepare_validation(adapter, (true,)) === adapter
            end
            output = (var"vdm-command"=zeros(Float32, size(matrix, 1)),)
            expected = deepcopy(output)
            input = (var"controller-command"=Float32[-0.0, 0.125, -0.25],)
            addresses = (pointer(adapter.output), pointer(adapter.input), pointer(adapter.frame_status))
            for _ in 1:3
                @test process!(output, gpu, input) == process!(expected, cpu, input)
                @test reinterpret(UInt32, output.var"vdm-command") ==
                    reinterpret(UInt32, expected.var"vdm-command")
            end
            frame_bytes(output, gpu, input)
            synchronize()
            before_memory = used_memory()
            @test frame_bytes(output, gpu, input) == 0
            synchronize()
            @test used_memory() == before_memory
            for nonfinite in (NaN32, Inf32, -Inf32)
                bad = (var"controller-command"=Float32[0, nonfinite, 1],)
                fill!(output.var"vdm-command", 71)
                fill!(expected.var"vdm-command", 71)
                result = process!(output, gpu, bad)
                @test result == process!(expected, cpu, bad)
                @test !(result isa NamedTuple)
                @test all(==(71), output.var"vdm-command")
                @test process!(output, gpu, input) == process!(expected, cpu, input)
            end
            reset!(cpu)
            reset!(gpu)
            @test process!(output, gpu, input) == process!(expected, cpu, input)
            @test (pointer(adapter.output), pointer(adapter.input), pointer(adapter.frame_status)) == addresses
        end
    end
end
