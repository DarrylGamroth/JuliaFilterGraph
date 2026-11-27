module CalibrationCaptureTests

using FilterGraphAlgorithms, Test
import FilterGraphs
import FilterGraphAlgorithms: FrameData, OutputPort, complete!, enqueue!, prepare_capture,
    prepare_properties, process!, reset!

const SCHEMA = "example/calibration-test/1"

same_calibration_result(left, right) = eltype(left) === Float32 ? isapprox(left, right) : left == right

function run_calibration_capture_tests(device, upload, synchronize, record)
    function check(declaration, configuration, samples; compare=isequal)
        prepared = prepare_algorithm(declaration, configuration)
        if declaration === DocrimeClosedLoopInteractionMatrixF32
            prepared.plan = prepare_properties(prepared.plan, (PropertyAssignment(0, 0.4f0),))
        elseif declaration === InteractionMatrixStabilityF32
            prepared.plan = prepare_properties(prepared.plan, (PropertyAssignment(1, 0.1f0),))
        end
        formats = algorithm_formats(prepared)
        ports = algorithm_ports(declaration)
        expected = Tuple(fill(one(format.element_type), format.shape)
            for (port, format) in zip(ports, formats)
            if port.direction isa OutputPort && port.role isa FrameData)
        outputs = map(upload, expected)
        inputs = map(upload, first(samples))
        status = upload(zeros(Int32, 1))
        adapter = prepare_capture(
            CompleteTransition(), device, prepared, outputs, inputs, status, Int32(3),
        )
        @test FilterGraphAlgorithms.capture_support(declaration, device) isa SupportsCapture
        enqueue!(adapter)
        synchronize()
        reset!(adapter)
        replay! = record(() -> enqueue!(adapter))
        for (index, values) in enumerate(samples)
            if index == 2
                # A prior-node rejection must suppress both writes and state commitment.
                fill!(status, Int32(1))
                ordinary = map(Array, outputs)
                replay!()
                synchronize()
                @test only(Array(status)) == 1
                @test isequal(map(Array, outputs), ordinary)
            end
            if index == 1
                # Initialize the published caller state equally after warmup.
                foreach(copyto!, outputs, expected)
            end
            foreach(copyto!, inputs, values)
            fill!(status, Int32(0))
            expected_failure = process!(expected..., prepared, values...)
            previous = expected_failure === nothing ? nothing : map(Array, outputs)
            replay!()
            synchronize()
            @test complete!(adapter, only(Array(status))) === expected_failure
            actual = map(Array, outputs)
            if previous !== nothing
                @test isequal(actual, previous)
            end
            @test all(compare(result, target) for (result, target) in zip(actual, expected))
        end
        reset!(adapter)
        reset!(prepared.workspace)
        values = first(samples)
        foreach(copyto!, inputs, values)
        fill!(status, Int32(0))
        @test process!(expected..., prepared, values...) === nothing
        replay!()
        synchronize()
        @test iszero(only(Array(status)))
        @test all(compare(actual, target) for (actual, target) in zip(map(Array, outputs), expected))
    end

    @testset "Resident calibration transitions" begin
        @testset "Deterministic excitation and structured probe cycles" begin
            amplitudes = Float32[0.25, 0.125, 0.5]
            for (declaration, cfg) in (
                (DocrimeExcitationF32, DocrimeExcitationF32Configuration(amplitudes=amplitudes, seed=UInt64(9))),
                (DocrimeBinaryExcitationF32, DocrimeBinaryExcitationF32Configuration(amplitudes=amplitudes, seed=UInt64(9))),
                (ZonalPushPullF32, ZonalPushPullF32Configuration(amplitudes=amplitudes, excitation_schema=SCHEMA)),
                (HadamardPushPullF32, HadamardPushPullF32Configuration(amplitudes=amplitudes, excitation_schema=SCHEMA)),
                (ModalPushPullF32, ModalPushPullF32Configuration(mode_count=2,
                    corrector_coordinate_count=3, initial_positive_commands=Float32[0, 2, 0, 3, 1, 0],
                    excitation_schema=SCHEMA)),
            )
                check(declaration, cfg, fill((), 11))
            end
        end
        @testset "Matched differencing and DO-CRIME covariance history" begin
            extents = (measurement_count=3, corrector_coordinate_count=2)
            schemas = (measurements_schema=SCHEMA, corrector_sample_schema=SCHEMA)
            samples = [
                (Float32[1, 2, 3], Float32[2, 3]),
                (Float32[0, 1, 3], Float32[Inf, 3]),
                (Float32[NaN, 2, 3], Float32[Inf, 3]),
                (Float32[2, 4, 1], Float32[5, 2]),
                (Float32[3, 6, 2], Float32[6, 2]),
                (Float32[NaN, 8, 4], Float32[7, 3]),
            ]
            check(MatchedCalibrationDifferencingF32, MatchedCalibrationDifferencingF32Configuration(
                ; extents..., schemas..., filtered_measurements_schema=SCHEMA,
                filtered_corrector_sample_schema=SCHEMA), samples)
            check(DocrimeCovarianceF32, DocrimeCovarianceF32Configuration(
                ; extents..., schemas..., sample_limit=UInt64(3),
                cross_covariance_schema=SCHEMA, corrector_covariance_schema=SCHEMA), samples)
        end
        @testset "SCC and structured interaction matrix estimates" begin
            finite_response = (Float32[4, -1, 3], Float32[1, 1, 3],
                Float32[2, 3, 5], Float32[2, 0, 2])
            bad_response = (finite_response[1], Float32[1, NaN, 3],
                finite_response[3], Float32[Inf, 0, 2])
            overflow_response = (Float32[floatmax(Float32), 0, 0],
                Float32[-floatmax(Float32), 0, 0], finite_response[3], finite_response[4])
            check(SccPushPullResponseF32, SccPushPullResponseF32Configuration(pixel_count=3),
                [finite_response, bad_response, overflow_response, finite_response])
            for (correctors, zonal_rows, hadamard_rows) in ((2, 4, 8), (4, 8, 16))
                options = (measurement_count=3, amplitudes=Float32.(1:correctors),
                    response_schema=SCHEMA, interaction_matrix_schema=SCHEMA)
                for (declaration, cfg, rows) in (
                    (ZonalInteractionMatrixF32, ZonalInteractionMatrixF32Configuration(; options...), zonal_rows),
                    (HadamardInteractionMatrixF32, HadamardInteractionMatrixF32Configuration(; options...), hadamard_rows),
                )
                    values = [Float32(mod(index^2 + 3 * index, 29))
                        for index in 1:(rows * options.measurement_count)]
                    responses = reshape(values, rows, :)
                    nonfinite = copy(responses)
                    nonfinite[3, 2] = NaN
                    overflow = copy(responses)
                    overflow[1:2, 1] .= (floatmax(Float32), -floatmax(Float32))
                    check(declaration, cfg, [(responses,), (nonfinite,), (overflow,), (-responses,)])
                end
            end
        end
        @testset "Stability and DO-CRIME conditioning" begin
            extents = (measurement_count=3, corrector_coordinate_count=3, interaction_matrix_schema=SCHEMA)
            matrix = Float32[2 5 1; 4 3 2; 6 1 2]
            nonfinite = copy(matrix)
            nonfinite[1] = NaN
            tiny = copy(matrix)
            tiny[:, 2] .= 0
            check(InteractionMatrixStabilityF32, InteractionMatrixStabilityF32Configuration(
                ; extents..., minimum_column_norm=0.25f0),
                [(matrix, copy(matrix)), (nonfinite, matrix), (matrix, nonfinite),
                    (matrix, tiny), (matrix, -matrix), (matrix, matrix .* floatmax(Float32)),
                    (matrix, fill(floatmax(Float32), size(matrix))), (matrix, 1.4f0 .* matrix),
                    (matrix, 1.05f0 .* matrix)]; compare=same_calibration_result)
            options = (; extents..., minimum_corrector_variance=0.25f0,
                cross_covariance_schema=SCHEMA, corrector_covariance_schema=SCHEMA)
            corrector = Float32[8 2 1; 2 12 3; 1 3 9]
            asymmetric = Float32[8 3 1; 2 12 3; 1 3 9]
            variance = Float32[8 0 0; 0 0 0; 0 0 9]
            correlated = Float32[4 4 0; 4 4 0; 0 0 9]
            indefinite = Float32[10 7.5 7.5; 7.5 10 -7.5; 7.5 -7.5 10]
            small_pivot = Float32[100 0 0; 0 75 0; 0 0 0.5]
            solve_overflow = fill(floatmax(Float32), size(matrix))
            small_variance = Float32[0.5 0 0; 0 0.5 0; 0 0 0.5]
            for (declaration, cfg) in (
                (DocrimeOpenLoopInteractionMatrixF32, DocrimeOpenLoopInteractionMatrixF32Configuration(; options...)),
                (DocrimeClosedLoopInteractionMatrixF32, DocrimeClosedLoopInteractionMatrixF32Configuration(; options...)),
            )
                check(declaration, cfg, [(matrix, corrector, UInt64[3]),
                    (matrix, corrector, UInt64[1]), (matrix, asymmetric, UInt64[2]),
                    (nonfinite, corrector, UInt64[3]), (matrix, variance, UInt64[3]),
                    (matrix, correlated, UInt64[3]), (matrix, indefinite, UInt64[3]),
                    (matrix, small_pivot, UInt64[3]), (solve_overflow, small_variance, UInt64[3]),
                    (matrix, corrector, UInt64[2^24 + 2]),
                    (matrix, corrector, UInt64[typemax(UInt64)]), (matrix, corrector, UInt64[4])];
                    compare=same_calibration_result)
            end
        end
    end
end

const DOCRIME_CHAIN = """
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = covariance
            label = docrime-covariance-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                sample_limit = null
                measurements_schema = "example/test-measurements/1"
                corrector_sample_schema = "example/test-correctors/1"
                cross_covariance_schema = "example/test-cross-covariance/1"
                corrector_covariance_schema = "example/test-corrector-covariance/1"
            }
        }
        {
            type = ndarray
            name = solve
            label = docrime-open-loop-interaction-matrix-f32
            config = {
                measurement_count = 2
                corrector_coordinate_count = 2
                minimum_corrector_variance = 0.1
                cross_covariance_schema = "example/test-cross-covariance/1"
                corrector_covariance_schema = "example/test-corrector-covariance/1"
                interaction_matrix_schema = "example/test-interaction-matrix/1"
            }
        }
    ]
    links = [
        { output = "covariance:measurement-corrector-cross-covariance-sum" input = "solve:measurement-corrector-cross-covariance-sum" }
        { output = "covariance:corrector-covariance-sum" input = "solve:corrector-covariance-sum" }
        { output = "covariance:sample-count" input = "solve:sample-count" }
    ]
    inputs = [ "covariance:measurements" "covariance:corrector-sample" ]
    outputs = [ "solve:interaction-matrix" "solve:status" "solve:diagnostic-sample-count" ]
}
"""

function run_docrime_captured_graph_tests(device)
    @testset "Captured covariance and open-loop solve" begin
        checked = mktemp() do path, io
            write(io, DOCRIME_CHAIN)
            flush(io)
            FilterGraphs.prepare_graph(path; algorithms=(DocrimeCovarianceF32, DocrimeOpenLoopInteractionMatrixF32))
        end
        captured = FilterGraphs.prepare_captured_graph(checked, device)
        outputs = (var"interaction-matrix"=zeros(Float32, 2, 2), status=zeros(UInt8, 1),
            var"diagnostic-sample-count"=zeros(UInt64, 1))
        cpu_outputs = map(copy, outputs)
        model = Float32[2 -1; 0.5 3]
        for (frame, corrector) in enumerate((Float32[1, 0], Float32[-1, 0],
                Float32[0, 1], Float32[0, -1], Float32[1, 2]))
            input = (measurements=model * corrector .+ Float32[0.5, 4], var"corrector-sample"=corrector)
            if frame == 3
                before = map(copy, outputs)
                rejected = merge(input, (measurements=Float32[NaN, 1],))
                @test FilterGraphs.process!(outputs, captured, rejected) === DocrimeCovarianceNonFiniteMeasurements
                @test outputs == before
            end
            @test FilterGraphs.process!(outputs, captured, input) ==
                  FilterGraphs.process!(cpu_outputs, checked, input)
            @test outputs.var"interaction-matrix" ≈ cpu_outputs.var"interaction-matrix"
            @test outputs.status == cpu_outputs.status
            @test outputs.var"diagnostic-sample-count" == UInt64[frame]
        end
        @test outputs.var"interaction-matrix" ≈ model
        FilterGraphs.reset!(captured)
        FilterGraphs.process!(outputs, captured, (measurements=Float32[2, 3], var"corrector-sample"=Float32[1, 0]))
        @test outputs.var"diagnostic-sample-count" == UInt64[1]
    end
end

end
