using LinearAlgebra
import FilterGraphAlgorithms: complete!, enqueue!, prepare_capture, process!, reset!

# The same captured-solver experiments exercise both native GPU backends.
function run_weighted_scc_capture_tests(device, ext, device_zeros, device_ones,
    synchronize, capture_graph, instantiate, launch, node_count, used_memory;
    prepare_memory_measurement=Returns(nothing))
    SENTINEL = 71f0
function prepared_solver(H, L; regularization=0.125f0, iterations=12,
    relative_tolerance=1f-5, minimum_diagonal=1f-6)
    return prepare_algorithm(SccWeightedRegularizedReconstructorF32,
        SccWeightedRegularizedReconstructorF32Configuration(
            measurement_count=size(H, 1), controller_coordinate_count=size(H, 2),
            regularization_row_count=size(L, 1),
            measurement_schema=SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,
            initial_interaction_matrix=collect(vec(transpose(H))),
            initial_regularization_operator=collect(vec(transpose(L))),
            regularization=regularization, iterations=iterations,
            relative_tolerance=relative_tolerance, minimum_diagonal=minimum_diagonal))
end

submit!(owner) = submit!(owner, owner.mode)
submit!(owner, ::Val{:auto}) = enqueue!(owner.captured)
submit!(owner, ::Val{:original}) = ext._enqueue_scc!(owner.captured, nothing)
function submit!(owner, ::Val{:layout})
    execution = owner.captured.execution
    return ext._enqueue_scc!(owner.captured,
        ext._SccStages(execution.interaction, execution.regularization))
end

function owner(prepared; mode=Val(:auto), alias_input=false)
    plan = prepared.plan
    measurements = device_zeros(Float32, size(plan.interaction_matrix, 1))
    weights = device_ones(Float32, length(measurements))
    output = alias_input ? measurements : device_zeros(Float32, size(plan.interaction_matrix, 2))
    status = device_zeros(Int32, 1)
    fill!(measurements, 0.125f0)
    fill!(weights, 0.75f0)
    captured = prepare_capture(CompleteTransition(), device, prepared,
        (output,), (measurements, weights), status, Int32(3))
    @test all(==(0.125f0), Array(measurements))
    @test all(==(0.75f0), Array(weights))
    base = (; prepared, captured, measurements, weights, output, status, mode)
    submit!(base)
    synchronize()
    graph = capture_graph(() -> submit!(base))
    return merge(base, (; graph, executable=instantiate(graph)))
end

bits(values) = collect(reinterpret(UInt32, vec(values)))
launch_bytes(executable) = @allocated launch(executable)


function replay!(owner, measurements, weights; incoming=Int32(0))
    fill!(owner.output, SENTINEL)
    copyto!(owner.measurements, measurements)
    copyto!(owner.weights, weights)
    fill!(owner.status, incoming)
    retained = Array(owner.output)
    launch(owner.executable)
    synchronize()
    status = only(Array(owner.status))
    return (; status, failure=complete!(owner.captured, status),
        output=Array(owner.output), retained)
end

function compare_owners!(owners, measurements, weights; expected=nothing)
    cpu = fill(SENTINEL, length(first(owners).output))
    failure = process!(cpu, first(owners).prepared, measurements, weights)
    @test failure === expected
    results = map(o -> replay!(o, measurements, weights), owners)
    @test all(r -> r.failure === expected, results)
    @test all(r -> r.status == first(results).status, results)
    @test all(r -> bits(r.output) == bits(first(results).output), results)
    if expected === nothing
        @test first(results).status == 0
        @test first(results).output ≈ cpu rtol=2f-5 atol=0
    else
        @test all(r -> bits(r.output) == bits(r.retained), results)
        @test cpu == fill(SENTINEL, length(cpu))
    end
    return results
end

function shape_fixture(m, c, r)
    H, L = zeros(Float32, m, c), zeros(Float32, r, c)
    for i in 1:min(m, c)
        H[i, i] = 1f0
    end
    H[end, end] = 0.5f0
    for i in 1:r
        L[i, mod1(i, c)] = 0.5f0
    end
    signal = Float32[mod(i, 7) - 3 for i in 1:c] ./ 8f0
    return H, L, H * signal, ones(Float32, m)
end

@testset "bounded SCC capture covers lane tails and staged eligibility" begin
    for (m, c, r) in ((1, 1, 1), (31, 2, 3), (33, 31, 32),
        (65, 33, 35), (255, 32, 31), (256, 33, 32),
        (32, 255, 256), (33, 256, 256),
        (257, 32, 31), (32, 257, 257), (31, 32, 257))
        @testset "M=$m C=$c R=$r" begin
            H, L, measurements, weights = shape_fixture(m, c, r)
            prepared = prepared_solver(H, L)
            selected = owner(prepared)
            original = owner(prepared; mode=Val(:original))
            bounded = max(m, c, r) <= 256
            @test (selected.captured.execution isa ext._SccBlock) == bounded
            @test node_count(selected.graph) == (bounded ? 1 : 9 * prepared.plan.iterations + 9)
            @test node_count(original.graph) == 9 * prepared.plan.iterations + 9
            compare_owners!((selected, original), measurements, weights)
        end
    end
end

@testset "SCC block and prepared adjoint layout retain original arithmetic" begin
    H = Float32[1 0; 0 2; 1 1]
    L = Matrix{Float32}(I, 2, 2)
    prepared = prepared_solver(H, L)
    owners = (owner(prepared), owner(prepared; mode=Val(:original)),
        owner(prepared; mode=Val(:layout)))
    weights = Float32[1, 0.5, 2]
    base = H * Float32[0.5, -0.25]
    for exponent in (-120, -90, 0, 90, 120)
        compare_owners!(owners, ldexp.(base, exponent), weights)
    end
    compare_owners!(owners, zeros(Float32, 3), weights)
    compare_owners!(owners, Float32[NaN, Inf, -Inf], zeros(Float32, 3))
    compare_owners!(owners, Float32[NaN, base[2], base[3]], Float32[0, 0.5, 2])

    cases = (
        (base, Float32[-1, 0.5, 2], SccWeightedRegularizedReconstructionInvalidWeight),
        (base, Float32[NaN, 0.5, 2], SccWeightedRegularizedReconstructionInvalidWeight),
        (base, Float32[Inf, 0.5, 2], SccWeightedRegularizedReconstructionInvalidWeight),
        (Float32[NaN, base[2], base[3]], weights, SccWeightedRegularizedReconstructionInvalidMeasurement),
        (Float32[NaN, base[2], base[3]], Float32[1, 0.5, -1], SccWeightedRegularizedReconstructionInvalidMeasurement),
        (Float32[base[1], base[2], NaN], Float32[-1, 0.5, 2], SccWeightedRegularizedReconstructionInvalidWeight),
    )
    for (measurements, trial_weights, expected) in cases
        compare_owners!(owners, measurements, trial_weights; expected)
    end
    # Recovery must reset per-solve flags and disregard rejected private scratch.
    compare_owners!(owners, base, weights)
    for o in owners
        @test reset!(o.captured) === o.captured
        result = replay!(o, base, weights; incoming=Int32(42))
        @test result.status == 42
        @test result.failure === nothing
        @test bits(result.output) == bits(result.retained)
        @test only(Array(o.captured.solve_state)) == 0
    end
    compare_owners!(owners, base, weights)
end

@testset "SCC zero RHS cannot bypass diagonal or convergence checks" begin
    H = Float32[1 0; 0 2; 1 1]
    L = Matrix{Float32}(I, 2, 2)
    for (regularization, expected) in ((0f0, SccWeightedRegularizedReconstructionSingularSystem),
        (0.125f0, nothing))
        prepared = prepared_solver(H, L; regularization)
        owners = (owner(prepared), owner(prepared; mode=Val(:original)))
        compare_owners!(owners, zeros(Float32, 3), zeros(Float32, 3); expected)
    end
    prepared = prepared_solver(H, L; iterations=1, relative_tolerance=eps(Float32))
    owners = (owner(prepared), owner(prepared; mode=Val(:original)))
    compare_owners!(owners, H * Float32[0.5, -0.25], Float32[1, 0.5, 2];
        expected=SccWeightedRegularizedReconstructionNotConverged)
end

@testset "SCC rechecks the published candidate with the actual operator" begin
    H = Float32[0.5587443 -0.44319695; -0.5483226 0.4367202]
    prepared = prepared_solver(H, zeros(Float32, 1, 2); regularization=0f0,
        iterations=100, relative_tolerance=1f-6, minimum_diagonal=1f-12)
    owners = (owner(prepared), owner(prepared; mode=Val(:original)))
    compare_owners!(owners, Float32[-0.725702, -0.739205], ones(Float32, 2);
        expected=SccWeightedRegularizedReconstructionNotConverged)
end

@testset "SCC automatic capture excludes zero-weight rows and zero regularization" begin
    for m in (3, 257)
        H = zeros(Float32, m, 1)
        H[1, 1] = H[2, 1] = 1f0
        H[end, 1] = floatmax(Float32)
        prepared = prepared_solver(H, ones(Float32, 1, 1); regularization=1f0, iterations=2)
        selected = owner(prepared)
        @test (selected.captured.execution isa ext._SccBlock) == (m == 3)
        @test node_count(selected.graph) == (m == 3 ? 1 : 9 * (prepared.plan.iterations + 1))
        measurements = zeros(Float32, m)
        weights = zeros(Float32, m)
        measurements[1], weights[1] = 4f0, 1f0
        # The second row contributes in this frame and is masked in the next.
        measurements[2], weights[2] = 6f0, 2f0
        measurements[end] = NaN32
        result = replay!(selected, measurements, weights)
        @test result.status == 0
        @test result.output == Float32[4]
        measurements[2], weights[2] = NaN32, 0f0
        for (measurement, weight) in ((floatmax(Float32), 0f0), (NaN32, -0f0))
            measurements[end], weights[end] = measurement, weight
            result = replay!(selected, measurements, weights)
            @test result.status == 0
            @test result.output == Float32[2]
        end
        for (measurement, weight, failure) in (
            (1f0, 1f0, SccWeightedRegularizedReconstructionSingularSystem),
            (NaN32, 1f0, SccWeightedRegularizedReconstructionInvalidMeasurement),
            (NaN32, -1f0, SccWeightedRegularizedReconstructionInvalidWeight),
        )
            measurements[end], weights[end] = measurement, weight
            result = replay!(selected, measurements, weights)
            @test result.failure === failure
            @test bits(result.output) == bits(result.retained)
        end
        measurements[end], weights[end] = NaN32, 0f0
        @test replay!(selected, measurements, weights).output == Float32[2]

        H[end, 1] = 0f0
        prepared = prepared_solver(H, fill(floatmax(Float32), 1, 1);
            regularization=-0f0, iterations=2)
        selected = owner(prepared)
        @test (selected.captured.execution isa ext._SccBlock) == (m == 3)
        @test node_count(selected.graph) == (m == 3 ? 1 : 8 * (prepared.plan.iterations + 1))
        measurements[1] = 2f0
        result = replay!(selected, measurements, weights)
        @test result.status == 0
        @test result.output == Float32[2]
    end
end

@testset "SCC automatic capture retains active product order during residual verification" begin
    for m in (2, 257), regularization in (0f0, ldexp(1f0, -123))
        H = zeros(Float32, m, 1)
        H[1, 1] = 1f0
        measurements = zeros(Float32, m)
        weights = ones(Float32, m)
        L = ones(Float32, 1, 1)
        if iszero(regularization)
            # The forward product overflows before its active weight can scale it.
            H[end, 1] = 2f0
            weights[end] = 0.0625f0
            measurements[1] = floatmax(Float32) * 0.875f0
        else
            # The regularization adjoint overflows before lambda can scale it.
            L[1, 1] = ldexp(1f0, 65)
            measurements[1] = 1024f0
        end
        prepared = prepared_solver(H, L; regularization, iterations=2)
        selected = owner(prepared)
        @test (selected.captured.execution isa ext._SccBlock) == (m == 2)
        result = replay!(selected, measurements, weights)
        @test result.failure === SccWeightedRegularizedReconstructionSingularSystem
        candidate = only(Array(selected.captured.candidate))
        @test isfinite(candidate) && candidate > 1f0
        @test !isfinite(only(Array(selected.captured.residual_vector)))
        @test bits(result.output) == bits(result.retained)
    end
end

@testset "SCC representability checks precede output publication" begin
    prepared = prepared_solver(ones(Float32, 1, 1), ones(Float32, 1, 1);
        regularization=0f0, iterations=2)
    owners = (owner(prepared), owner(prepared; mode=Val(:original)))
    for value in (nextfloat(0f0), floatmin(Float32), floatmax(Float32))
        results = compare_owners!(owners, Float32[value], Float32[1])
        @test all(r -> r.output == Float32[value], results)
    end
    for (coefficient, measurement, minimum_diagonal) in (
        (2f0, floatmax(Float32), 1f-6), # Unrepresentable assembled RHS.
        (floatmax(Float32), 1f0, 1f-6), # Unrepresentable diagonal.
        (1f-20, 1f0, nextfloat(0f0)), # Preconditioner/rho breakdown.
        (1f-10, 1f30, 1f-21), # Finite recurrence, overflowing final rescaling.
    )
        prepared = prepared_solver(fill(coefficient, 1, 1), ones(Float32, 1, 1);
            regularization=0f0, iterations=2, minimum_diagonal)
        owners = (owner(prepared), owner(prepared; mode=Val(:original)))
        compare_owners!(owners, Float32[measurement], Float32[1];
            expected=SccWeightedRegularizedReconstructionSingularSystem)
        if coefficient == 1f-10
            @test all(o -> only(Array(o.captured.solve_state)) == 1, owners)
        end
    end
end

@testset "SCC capture reuses owned storage and supports admitted input aliasing" begin
    prepared = prepared_solver(Matrix{Float32}(I, 2, 2), Matrix{Float32}(I, 2, 2))
    owners = (owner(prepared), owner(prepared; mode=Val(:original)))
    for o in owners
        c = o.captured
        buffers = (c.candidate, c.right_hand_side, c.diagonal, c.residual_vector,
            c.direction_vector, c.measurement_product_vector, c.regularization_product_vector)
        addresses = map(pointer, buffers)
        for values in (Float32[0.5, -0.25], Float32[-2, 3], zeros(Float32, 2))
            replay!(o, values, ones(Float32, 2))
        end
        @test map(pointer, buffers) == addresses
        launch_bytes(o.executable)
        # Collect earlier fixtures before measuring global device-pool usage.
        GC.@preserve owners buffers begin
            GC.gc(true)
            prepare_memory_measurement()
            synchronize()
            before_memory = used_memory()
            @test launch_bytes(o.executable) == 0
            synchronize()
            @test used_memory() == before_memory
        end
    end
    aliases = (owner(prepared; alias_input=true),
        owner(prepared; alias_input=true, mode=Val(:original)))
    compare_owners!(aliases, Float32[0.5, -0.25], ones(Float32, 2))
    compare_owners!(aliases, Float32[0.5, -0.25], Float32[-1, 1];
        expected=SccWeightedRegularizedReconstructionInvalidWeight)
end

end
