module BiquadCascadeTests

using FilterGraphAlgorithms
using Test
import FilterGraphAlgorithms: commit!, discard!, process!, reset!

function prepare_cascade(coefficients; extent=3)
    return prepare_algorithm(
        BiquadCascadeF32,
        BiquadCascadeF32Configuration(
            extent=extent,
            section_count=size(coefficients, 2),
            per_mode=size(coefficients, 3) != 1,
            initial_coefficients=vec(Float32.(coefficients)),
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        ),
    )
end

# Independent direct form I oracle: explicit input/output histories in BigFloat,
# with no transposed state or implementation helper shared with the Algorithm.
function reference_cascade(coefficients, samples)
    result = BigFloat.(samples)
    for mode in axes(samples, 1), section in axes(coefficients, 2)
        set = size(coefficients, 3) == 1 ? 1 : mode
        b0, b1, b2, a1, a2 = BigFloat.(coefficients[:, section, set])
        x1 = x2 = y1 = y2 = BigFloat(0)
        for frame in axes(samples, 2)
            x = result[mode, frame]
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            result[mode, frame] = y
            x2, x1 = x1, x
            y2, y1 = y1, y
        end
    end
    return result
end

@testset "biquad declaration and identity defaults" begin
    @test algorithm_label(BiquadCascadeF32) == "biquad-cascade-f32"
    @test BiquadCascadeF32 in algorithms()
    @test FilterGraphAlgorithms.algorithm_failure_type(BiquadCascadeF32) === BiquadCascadeFailure
    ports = algorithm_ports(BiquadCascadeF32)
    @test getproperty.(ports, :name) == (:input, :output, :coefficients)
    @test ports[2].metadata_source === :input
    for per_mode in (false, true)
        configuration = BiquadCascadeF32Configuration(
            extent=3, section_count=2, per_mode=per_mode,
            initial_coefficients=nothing,
            input_schema=CONTROLLER_RESIDUAL_ERROR_V1,
            output_schema=CONTROLLER_RESIDUAL_ERROR_V1,
        )
        prepared = prepare_algorithm(BiquadCascadeF32, configuration)
        @test size(prepared.plan.coefficients) == (5, 2, per_mode ? 3 : 1)
        @test prepared.formats[3].schema == BIQUAD_COEFFICIENTS_V1
        input = Float32[1, -2, 3]
        output = similar(input)
        @test process!(output, prepared, input) === nothing
        @test output == input
    end
end

@testset "biquad preparation copies coefficients and requires stable sections" begin
    for T in (Float32, Float64)
        coefficients = reshape(T[0.5, 0.25, 0.125, -0.4, 0.12], 5, 1, 1)
        plan = BiquadCascadePlan(coefficients, 3)
        coefficients[1] = 99
        @test plan.coefficients[1] == T(0.5)
        @test_throws ArgumentError BiquadCascadePlan(plan.coefficients, 0)
        @test_throws DimensionMismatch BiquadCascadePlan(zeros(T, 6, 1, 1), 3)
        @test_throws ArgumentError BiquadCascadePlan(zeros(T, 5, 0, 1), 3)
        @test_throws DimensionMismatch BiquadCascadePlan(zeros(T, 5, 1, 2), 3)
        for (a1, a2) in ((-1, 0), (1, 0), (0, 1), (0, -1),
                         (0, 1.01), (0, -1.01), (1.8, 0.5), (-1.8, 0.5))
            unstable = reshape(T[1, 0, 0, a1, a2], 5, 1, 1)
            @test_throws ArgumentError BiquadCascadePlan(unstable, 3)
        end
        for bad in (T(NaN), T(Inf), T(-Inf)), coefficient in 1:5
            invalid = reshape(T[1, 0, 0, 0, 0], 5, 1, 1)
            invalid[coefficient] = bad
            @test_throws ArgumentError BiquadCascadePlan(invalid, 3)
        end
        stable = reshape(T[1, 0, 0, -prevfloat(one(T)), 0], 5, 1, 1)
        @test BiquadCascadePlan(stable, 3) isa BiquadCascadePlan{T}
    end
    # Coefficients that round onto the unit circle in the operating precision
    # must be rejected even if the original design was strictly stable.
    rounded = reshape(Float32.([1, 0, 0, -2 * (1 - 1e-10), (1 - 1e-10)^2]), 5, 1, 1)
    @test_throws ArgumentError BiquadCascadePlan(rounded, 1)
end

@testset "biquad mode-region preparation" begin
    short = reshape(Float64[0.5, 0, 0, -0.25, 0], 5, 1)
    long = Float64[
        0.75 0.25
        0.1  0
        0.05 0
        -0.4 -0.2
        0.12 0.03
    ]
    regions = (3:3 => short, 1:2 => long)
    plan = BiquadCascadePlan(regions, 3)
    expected = zeros(Float64, 5, 2, 3)
    expected[1, :, :] .= 1
    expected[:, :, 1] .= long
    expected[:, :, 2] .= long
    expected[:, 1, 3] .= short[:, 1]
    explicit = BiquadCascadePlan(expected, 3)
    @test plan.coefficients == explicit.coefficients
    @test plan.coefficients[:, 2, 3] == [1, 0, 0, 0, 0]
    @test BiquadCascadePlan(collect(regions), 3).coefficients == expected

    original = copy(short)
    copied_plan = BiquadCascadePlan((1:3 => short,), 3)
    short[1] = 42
    @test copied_plan.coefficients[1, 1, 1] == original[1]

    @test_throws ArgumentError BiquadCascadePlan((1:1 => short, 3:3 => short), 3) # gap
    @test_throws ArgumentError BiquadCascadePlan((1:2 => short, 2:3 => short), 3) # overlap
    @test_throws ArgumentError BiquadCascadePlan((0:1 => short, 2:3 => short), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:0 => short,), 3)
    @test_throws ArgumentError BiquadCascadePlan((), 3)
    @test_throws DimensionMismatch BiquadCascadePlan((1:3 => zeros(4, 1),), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:3 => zeros(Float64, 5, 0),), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:1 => long, 2:3 => Float32.(short)), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:3 => reshape(Int[1, 0, 0, 0, 0], 5, 1),), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:3 => reshape(Float64[1, 0, 0, NaN, 0], 5, 1),), 3)
    @test_throws ArgumentError BiquadCascadePlan((1:3 => reshape(Float64[1, 0, 0, 1, 0], 5, 1),), 3)
end

@testset "shared and per-mode cascades match direct form I histories" begin
    for T in (Float32, Float64), sets in (1, 3)
        coefficients = Array{T}(undef, 5, 2, sets)
        for set in 1:sets
            coefficients[:, 1, set] = T[0.2 * set, 0.1, 0.05, -0.4, 0.12]
            coefficients[:, 2, set] = T[0.8, -1.2, 0.8, -1.1, 0.7]
        end
        samples = T[frame == 1 ? mode : sin(frame * 0.13 * mode)
                    for mode in 1:3, frame in 1:512]
        reference = T.(reference_cascade(coefficients, samples))
        plan = BiquadCascadePlan(coefficients, 3)
        workspace = BiquadCascadeWorkspace(plan)
        result = similar(samples)
        for frame in axes(samples, 2)
            @test process!(view(result, :, frame), workspace, plan,
                           view(samples, :, frame)) === view(result, :, frame)
            commit!(workspace)
        end
        @test result ≈ reference rtol=(T === Float32 ? 2e-5 : 2e-13) atol=(T === Float32 ? 2e-6 : 2e-14)
    end
end

@testset "high-Q section impulse and steady-state response" begin
    for T in (Float32, Float64)
        r = T(0.995)
        angle = T(0.17)
        coefficients = reshape(T[1, -2cos(angle), 1, -2r*cos(angle), r*r], 5, 1, 1)
        samples = zeros(T, 1, 4096)
        samples[1] = 1
        reference = T.(reference_cascade(coefficients, samples))
        plan = BiquadCascadePlan(coefficients, 1)
        workspace = BiquadCascadeWorkspace(plan)
        result = similar(samples)
        for frame in axes(samples, 2)
            process!(view(result, :, frame), workspace, plan, view(samples, :, frame))
            commit!(workspace)
        end
        @test all(isfinite, result)
        @test result ≈ reference rtol=(T === Float32 ? 5e-4 : 2e-12) atol=(T === Float32 ? 2e-6 : 2e-14)
        reset!(workspace)
        input = ones(T, 1)
        output = similar(input)
        for _ in 1:8192
            process!(output, workspace, plan, input)
            commit!(workspace)
        end
        b0, b1, b2, a1, a2 = coefficients
        expected_dc = (BigFloat(b0) + b1 + b2) / (1 + BigFloat(a1) + a2)
        @test output[1] ≈ T(expected_dc) rtol=(T === Float32 ? 5e-5 : 2e-12)
    end
end

@testset "late-section overflow preserves previously committed cascade history" begin
    coefficients = reshape(Float32[
        0.5 2
        0   0
        0   0
        -0.5 -0.25
        0   0
    ], 5, 2, 1)
    prepared = prepare_cascade(coefficients; extent=2)
    output = zeros(Float32, 2)
    process!(output, prepared, Float32[2, 4])
    state_before = copy(prepared.workspace.committed_state)
    fill!(output, 71)
    # First section is finite; the second overflows only in the second mode.
    coefficients[1, 2, 1] = 4
    # Use a separate Plan with the same dimensions and existing history to
    # exercise rejection, rather than replacement (which intentionally resets).
    overflow_plan = BiquadCascadePlan(coefficients, 2)
    @test process!(output, prepared.workspace, overflow_plan,
                   Float32[0, floatmax(Float32)]) === BiquadCascadeNonFiniteOutput
    @test output == fill(71.0f0, 2)
    @test prepared.workspace.committed_state == state_before
    @test prepared.workspace.candidate_state == state_before
    process!(output, prepared, zeros(Float32, 2))
    reference = Float32.(reference_cascade(prepared.plan.coefficients, Float32[2 0; 4 0]))
    @test output == reference[:, 2]
end

@testset "staging, discard, reset, views, and overlapping input/output" begin
    coefficients = reshape(Float32[0.5, 0, 0, -0.5, 0], 5, 1, 1)
    plan = BiquadCascadePlan(coefficients, 2)
    workspace = BiquadCascadeWorkspace(plan)
    output = fill(71.0f0, 2)
    @test process!(output, workspace, plan, Float32[2, 4]) === output
    @test output == Float32[1, 2]
    @test all(iszero, workspace.committed_state)
    discard!(workspace)
    process!(output, workspace, plan, Float32[4, 8])
    @test output == Float32[2, 4]
    commit!(workspace)
    process!(output, workspace, plan, zeros(Float32, 2))
    @test output == Float32[1, 2]
    reset!(workspace)
    @test all(iszero, workspace.committed_state)
    @test all(iszero, workspace.candidate_state)
    @test all(iszero, workspace.candidate_output)
    parent = Float32[2, 4, 71]
    process!(view(parent, 2:3), workspace, plan, view(parent, 1:2))
    @test parent == Float32[2, 1, 2]
    reset!(workspace)
    input_output = Float32[2, 4]
    process!(input_output, workspace, plan, input_output)
    @test input_output == Float32[1, 2]
    @test_throws DimensionMismatch process!(zeros(Float32, 3), workspace, plan, zeros(Float32, 3))
    @test_throws ArgumentError process!(workspace.candidate_output, workspace, plan, zeros(Float32, 2))
    @test_throws ArgumentError process!(output, workspace, plan, view(vec(workspace.committed_state), 1:2))
    @test_throws ArgumentError process!(view(vec(plan.coefficients), 1:2), workspace, plan, zeros(Float32, 2))
end

@testset "nonfinite input, output, and internal state reject without publication" begin
    for coefficients in (
        reshape(Float32[2, 0, 0, 0, 0], 5, 1, 1),
        # The current output stays finite, but the next DF2T state overflows.
        reshape(Float32[0, floatmax(Float32), 0, 0, 0], 5, 1, 1),
    )
        prepared = prepare_cascade(coefficients; extent=2)
        output = fill(71.0f0, 2)
        state_before = copy(prepared.workspace.committed_state)
        @test process!(output, prepared, Float32[0, floatmax(Float32)]) === BiquadCascadeNonFiniteOutput
        @test output == fill(71.0f0, 2)
        @test prepared.workspace.committed_state == state_before
        @test prepared.workspace.candidate_state == state_before
    end
    prepared = prepare_cascade(reshape(Float32[0.5, 0, 0, -0.5, 0], 5, 1, 1); extent=2)
    output = zeros(Float32, 2)
    process!(output, prepared, Float32[2, 4])
    state_before = copy(prepared.workspace.committed_state)
    for bad in (NaN32, Inf32, -Inf32)
        fill!(output, 71)
        @test process!(output, prepared, Float32[3, bad]) === BiquadCascadeNonFiniteInput
        @test output == fill(71.0f0, 2)
        @test prepared.workspace.committed_state == state_before
        @test prepared.workspace.candidate_state == state_before
        @test all(iszero, prepared.workspace.candidate_output)
    end
    process!(output, prepared, zeros(Float32, 2))
    @test output == Float32[0.5, 1]
end

@testset "coefficient adoption resets history only after successful validation" begin
    coefficients = reshape(Float32[0.5, 0, 0, -0.5, 0], 5, 1, 1)
    prepared = prepare_cascade(coefficients; extent=2)
    output = zeros(Float32, 2)
    process!(output, prepared, Float32[2, 4])
    old_plan = prepared.plan
    old_state = copy(prepared.workspace.committed_state)
    invalid = copy(coefficients)
    invalid[4] = -1
    @test_throws ArgumentError replace_parameter!(prepared, :coefficients, invalid)
    @test prepared.plan === old_plan
    @test prepared.workspace.committed_state == old_state
    @test_throws DimensionMismatch replace_parameter!(prepared, :coefficients, zeros(Float32, 5, 2, 1))
    @test prepared.workspace.committed_state == old_state
    @test replace_parameter!(prepared, :coefficients, coefficients) === prepared
    @test all(iszero, prepared.workspace.committed_state)
    coefficients[1] = 99
    process!(output, prepared, Float32[2, 4])
    @test output == Float32[1, 2]
end

process_bytes(output, prepared, input) = @allocated process!(output, prepared, input)
stage_bytes(output, workspace, plan, input) = @allocated begin
    process!(output, workspace, plan, input)
    nothing
end
commit_bytes(workspace) = @allocated commit!(workspace)
discard_bytes(workspace) = @allocated discard!(workspace)
reset_bytes(workspace) = @allocated reset!(workspace)

@testset "warmed checked processing and lifecycle allocate no heap" begin
    prepared = prepare_cascade(reshape(Float32[0.5, 0, 0, -0.5, 0], 5, 1, 1))
    input, output = ones(Float32, 3), zeros(Float32, 3)
    process_bytes(output, prepared, input)
    @test process_bytes(output, prepared, input) == 0
    stage_bytes(output, prepared.workspace, prepared.plan, input)
    @test stage_bytes(output, prepared.workspace, prepared.plan, input) == 0
    for measure in (commit_bytes, discard_bytes, reset_bytes)
        measure(prepared.workspace)
        @test measure(prepared.workspace) == 0
    end
    input[end] = NaN32
    process_bytes(output, prepared, input)
    @test process_bytes(output, prepared, input) == 0
end

# Scalar mode-major DF2T preserves the qualified FMA grouping. The
# independent BigFloat direct-form-I histories above remain the numerical oracle.
function scalar_stage!(output, workspace, plan, input)
    for element in eachindex(input)
        value = input[element]
        set = size(plan.coefficients, 3) == 1 ? 1 : element
        for section in axes(plan.coefficients, 2)
            b0, b1, b2, a1, a2 = plan.coefficients[:, section, set]
            s1 = workspace.committed_state[1, element, section]
            s2 = workspace.committed_state[2, element, section]
            y = fma(b0, value, s1)
            workspace.candidate_state[1, element, section] = fma(-a1, y, b1 * value) + s2
            workspace.candidate_state[2, element, section] = fma(-a2, y, b2 * value)
            value = y
        end
        workspace.candidate_output[element] = value
    end
    copyto!(output, workspace.candidate_output)
end

bits(values::AbstractArray{Float32}) = reinterpret(UInt32, vec(values))
bits(values::AbstractArray{Float64}) = reinterpret(UInt64, vec(values))

@testset "mode blocks preserve scalar DF2T bits and lifecycle" begin
    for T in (Float32, Float64), extent in (1, 7, 16, 63, 64, 65, 129, 1025, 4095, 4096, 4097),
        sections in (1, 3, 8), shared in (false, true)
        coefficients = Array{T}(undef, 5, sections, shared ? 1 : extent)
        for set in axes(coefficients, 3), section in 1:sections
            coefficients[:, section, set] .= T[
                0.7 + 0.01sin(set + section), 0.1, 0.04, -0.4, 0.12,
            ]
        end
        plan = BiquadCascadePlan(coefficients, extent)
        workspace, reference = BiquadCascadeWorkspace(plan), BiquadCascadeWorkspace(plan)
        output, expected, input = zeros(T, extent), zeros(T, extent), zeros(T, extent)
        for frame in 1:24
            input .= T[sin(0.13 * frame * mode) for mode in 1:extent]
            frame == 1 && (input[1] = -zero(T))
            scalar_stage!(expected, reference, plan, input)
            @test process!(output, workspace, plan, input) === output
            @test bits(output) == bits(expected)
            @test bits(workspace.candidate_state) == bits(reference.candidate_state)
            if frame % 5 == 0
                discard!(workspace)
                discard!(reference)
            else
                commit!(workspace)
                commit!(reference)
            end
        end
        stage_bytes(output, workspace, plan, input)
        @test stage_bytes(output, workspace, plan, input) == 0
        before = copy(workspace.committed_state)
        input[end] = T(NaN)
        fill!(output, 71)
        @test process!(output, workspace, plan, input) === BiquadCascadeNonFiniteInput
        @test output == fill(T(71), extent)
        @test workspace.committed_state == workspace.candidate_state == before
    end
end

@testset "block tails reject intermediate overflow and support strided overlap" begin
    for T in (Float32, Float64), extent in (63, 64, 65, 129), shared in (false, true)
        coefficients = zeros(T, 5, 3, shared ? 1 : extent)
        coefficients[1, :, :] .= 1
        coefficients[1, 2, :] .= 2
        # Last section is zero gain: intermediate overflow must still reject.
        coefficients[1, 3, :] .= 0
        plan = BiquadCascadePlan(coefficients, extent)
        workspace = BiquadCascadeWorkspace(plan)
        input, output = zeros(T, extent), fill(T(71), extent)
        input[end] = floatmax(T)
        @test process!(output, workspace, plan, input) === BiquadCascadeNonFiniteOutput
        @test output == fill(T(71), extent)
        @test all(iszero, workspace.committed_state)
        @test workspace.candidate_state == workspace.committed_state
        @test all(iszero, workspace.candidate_output)

        identity = zeros(T, 5, 3, shared ? 1 : extent)
        identity[1, :, :] .= 1
        identity_plan = BiquadCascadePlan(identity, extent)
        parent = T.(1:(2extent + 2))
        input_view = view(parent, 1:2:(2extent - 1))
        output_view = view(parent, 3:2:(2extent + 1))
        expected = copy(input_view)
        @test process!(output_view, workspace, identity_plan, input_view) === output_view
        @test output_view == expected
        stage_bytes(output_view, workspace, identity_plan, input_view)
        @test stage_bytes(output_view, workspace, identity_plan, input_view) == 0
    end
end

@testset "dense SIMD preconditions are checked before staging" begin
    plan = BiquadCascadePlan(reshape(Float32[1, 0, 0, 0, 0], 5, 1, 1), 65)
    input, output = ones(Float32, 65), fill(71.0f0, 65)
    for buffer in (:committed_state, :candidate_state)
        workspace = BiquadCascadeWorkspace(plan)
        setproperty!(workspace, buffer, zeros(Float32, 1, 65, 1))
        @test_throws DimensionMismatch process!(output, workspace, plan, input)
        @test output == fill(71.0f0, 65)
    end
    workspace = BiquadCascadeWorkspace(plan)
    workspace.candidate_state = workspace.committed_state
    @test_throws ArgumentError process!(output, workspace, plan, input)
    @test output == fill(71.0f0, 65)
end


@testset "FMA long high-Q trajectories agree with independent histories" begin
    for T in (Float32, Float64), shared in (false, true)
        extent, frames = 3, 32768
        coefficients = Array{T}(undef, 5, 2, shared ? 1 : extent)
        for set in axes(coefficients, 3), section in 1:2
            r, angle = T(section == 1 ? 0.999 : 0.997), T(section == 1 ? 0.23 : 0.67)
            coefficients[:, section, set] .= T[0.01 + 0.001set, 0.003, 0.002,
                                               -2r*cos(angle), r*r]
        end
        samples = T[sin((mode == 1 ? 0.05 : mode == 2 ? 0.23 : 1.1) * frame) +
                    (frame == 1 ? 0.25 : 0) for mode in 1:extent, frame in 1:frames]
        expected = reference_cascade(coefficients, samples)
        plan = BiquadCascadePlan(coefficients, extent)
        workspace = BiquadCascadeWorkspace(plan)
        result = similar(samples)
        for frame in axes(samples, 2)
            process!(view(result, :, frame), workspace, plan, view(samples, :, frame))
            commit!(workspace)
        end
        error = maximum(abs, BigFloat.(result) .- expected)
        scale = maximum(abs, expected)
        @test all(isfinite, result)
        @test error / scale < (T === Float32 ? 3e-4 : 3e-12)
    end
end

@testset "FMA sinusoidal response matches coefficient transfer function" begin
    for T in (Float32, Float64), bin in (1, 75, 311)
        r, angle = T(0.995), T(0.23)
        coefficients = reshape(T[0.02, 0.005, 0.003, -2r*cos(angle), r*r], 5, 1, 1)
        plan = BiquadCascadePlan(coefficients, 1)
        workspace = BiquadCascadeWorkspace(plan)
        input, output = zeros(T, 1), zeros(T, 1)
        omega = 2BigFloat(pi) * bin / 2048
        input_component = Complex{BigFloat}(0)
        output_component = Complex{BigFloat}(0)
        for frame in 1:12288
            input[1] = T(sin(omega * frame))
            process!(output, workspace, plan, input)
            commit!(workspace)
            if frame > 8192
                phase = cis(-omega * frame)
                input_component += BigFloat(input[1]) * phase
                output_component += BigFloat(output[1]) * phase
            end
        end
        b0, b1, b2, a1, a2 = BigFloat.(vec(coefficients))
        z = cis(-omega)
        expected = (b0 + b1*z + b2*z*z) / (1 + a1*z + a2*z*z)
        measured = output_component / input_component
        @test abs(measured - expected) / abs(expected) < (T === Float32 ? 3e-5 : 1e-12)
    end
end

@testset "FMA overflow cancellation and true nonfinite rejection" begin
    for T in (Float32, Float64)
        largest = floatmax(T)
        # The product alone overflows, but the exact sum is representable.
        plan = BiquadCascadePlan(reshape(T[2, 0, 0, 0, 0], 5, 1, 1), 65)
        workspace = BiquadCascadeWorkspace(plan)
        workspace.committed_state[1, :, :] .= -largest
        output, input = zeros(T, 65), fill(largest, 65)
        @test !isfinite(T(2) * largest - largest)
        @test process!(output, workspace, plan, input) === output
        @test output == input
        @test all(iszero, workspace.candidate_state)
        @test all(==(-largest), workspace.committed_state[1, :, :])
        discard!(workspace)
        @test workspace.candidate_state == workspace.committed_state
        @test all(iszero, workspace.candidate_output)
        process!(output, workspace, plan, input)
        commit!(workspace)
        @test all(iszero, workspace.committed_state)
        # No cancellation remains: the complete sample must still reject.
        fill!(output, T(71))
        @test process!(output, workspace, plan, input) === BiquadCascadeNonFiniteOutput
        @test output == fill(T(71), 65)
        @test workspace.candidate_state == workspace.committed_state
        reset!(workspace)
        @test all(iszero, workspace.committed_state)

        # Feedback FMA also retains a finite cancellation at the original
        # grouping boundary, while b1*x and the final +s2 remain separately rounded.
        feedback = BiquadCascadePlan(reshape(T[1, -1, 0, -1.5, 0.75], 5, 1, 1), 1)
        history = BiquadCascadeWorkspace(feedback)
        @test process!(zeros(T, 1), history, feedback, T[largest]) isa Vector{T}
        expected = T(BigFloat(1.5) * BigFloat(largest) - BigFloat(largest))
        @test history.candidate_state[1] == expected
        @test isfinite(history.candidate_state[2])
    end
end

@testset "prepared coefficient ownership and alias protection" begin
    for T in (Float32, Float64), sets in (1, 65)
        coefficients = zeros(T, 5, 2, sets)
        coefficients[1, :, :] .= 1
        plan = BiquadCascadePlan(coefficients, 65)
        @test plan.mode_coefficients == permutedims(plan.coefficients, (3, 1, 2))
        @test !Base.mightalias(plan.coefficients, plan.mode_coefficients)
        coefficients .= 99
        @test all(==(one(T)), plan.mode_coefficients[:, 1, :])
        workspace = BiquadCascadeWorkspace(plan)
        input, output = ones(T, 65), fill(T(71), 65)
        if sets == 65
            @test_throws ArgumentError process!(output, workspace, plan, view(vec(plan.mode_coefficients), 1:65))
            @test_throws ArgumentError process!(view(vec(plan.mode_coefficients), 1:65), workspace, plan, input)
            @test output == fill(T(71), 65)
        end
    end
end

end
