"""
    InterActuatorStrokeLimitPlan(actuator_count, edges, max_differences)

Prepare radial projection onto physical inter-actuator stroke-difference
limits. Each edge is a two-element vector of zero-based actuator indices, as
authored in the FGN configuration. `max_differences` supplies one finite,
nonnegative absolute-difference limit per edge.

The graph must contain at least one edge, all endpoints must be distinct and
in range, and an undirected edge may occur only once. Preparation copies the
topology and limits into fixed CPU storage.
"""
struct InterActuatorStrokeLimitPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    actuator_count::Int
    first_indices::Vector{Int}
    second_indices::Vector{Int}
    max_differences::Vector{T}
end

_valid_inter_actuator_limit(value::Float64) = isfinite(value) && value >= 0.0
_valid_inter_actuator_limit(value::Float32) =
    isfinite(value) && (!signbit(value) || reinterpret(UInt32, value) == 0x8000_0000)

function InterActuatorStrokeLimitPlan(
    actuator_count::Int,
    edges::AbstractVector,
    max_differences::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(edges, max_differences)
    actuator_count > 0 || throw(
        ArgumentError("inter-actuator stroke extent must be nonempty"),
    )
    isempty(edges) && throw(
        ArgumentError("inter-actuator stroke graph must contain an edge"),
    )
    length(max_differences) == length(edges) || throw(
        DimensionMismatch(
            "inter-actuator limit extent does not match the edge count",
        ),
    )

    first_indices = Vector{Int}(undef, length(edges))
    second_indices = Vector{Int}(undef, length(edges))
    prepared_differences = Vector{T}(undef, length(edges))
    for index in eachindex(edges, max_differences)
        edge = edges[index]
        edge isa AbstractVector{<:Integer} || throw(
            ArgumentError("each inter-actuator edge must contain integer indices"),
        )
        Base.require_one_based_indexing(edge)
        length(edge) == 2 || throw(
            DimensionMismatch("each inter-actuator edge must contain two indices"),
        )
        first_zero_based = edge[1]
        second_zero_based = edge[2]
        0 <= first_zero_based < actuator_count || throw(
            ArgumentError("inter-actuator stroke edge index is out of range"),
        )
        0 <= second_zero_based < actuator_count || throw(
            ArgumentError("inter-actuator stroke edge index is out of range"),
        )
        first_zero_based != second_zero_based || throw(
            ArgumentError("inter-actuator stroke graph contains a self-edge"),
        )
        max_difference = max_differences[index]
        _valid_inter_actuator_limit(max_difference) || throw(
            ArgumentError(
                "inter-actuator stroke limits must be finite and nonnegative",
            ),
        )

        first = Int(first_zero_based) + 1
        second = Int(second_zero_based) + 1
        low = min(first, second)
        high = max(first, second)
        for prior in 1:(index - 1)
            prior_low = min(first_indices[prior], second_indices[prior])
            prior_high = max(first_indices[prior], second_indices[prior])
            prior_low == low && prior_high == high && throw(
                ArgumentError(
                    "inter-actuator stroke graph contains a duplicate edge",
                ),
            )
        end
        first_indices[index] = first
        second_indices[index] = second
        prepared_differences[index] = max_difference
    end
    return InterActuatorStrokeLimitPlan(
        actuator_count,
        first_indices,
        second_indices,
        prepared_differences,
    )
end

function _replace_inter_actuator_max_differences(
    plan::InterActuatorStrokeLimitPlan{T},
    max_differences::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(max_differences)
    length(max_differences) == length(plan.max_differences) || throw(
        DimensionMismatch(
            "inter-actuator limit extent does not match the edge count",
        ),
    )
    prepared = Vector{T}(undef, length(max_differences))
    for index in eachindex(max_differences)
        value = max_differences[index]
        _valid_inter_actuator_limit(value) || throw(
            ArgumentError(
                "inter-actuator stroke limits must be finite and nonnegative",
            ),
        )
        prepared[index] = value
    end
    return InterActuatorStrokeLimitPlan(
        plan.actuator_count,
        plan.first_indices,
        plan.second_indices,
        prepared,
    )
end

"""Expected numerical rejection from inter-actuator stroke limiting."""
@enum InterActuatorStrokeLimitFailure::UInt8 begin
    InterActuatorStrokeLimitNonFiniteInput = 1
end

@doc "The requested PDM command contains a non-finite actuator value."
InterActuatorStrokeLimitNonFiniteInput

@inline function _inter_actuator_difference_gain(first::T, second::T, limit::T) where {T}
    if signbit(first) == signbit(second)
        difference = abs(first - second)
        return difference <= limit ? one(T) : limit / difference
    end

    first_magnitude = abs(first)
    second_magnitude = abs(second)
    scale = max(first_magnitude, second_magnitude)
    iszero(scale) && return one(T)
    normalized_difference =
        one(T) + min(first_magnitude, second_magnitude) / scale
    if scale <= limit / normalized_difference
        return one(T)
    end
    return (limit / scale) / normalized_difference
end

@noinline function _project_small_stroke_gain!(
    stroke_limited::AbstractVector{T},
    plan::InterActuatorStrokeLimitPlan{T},
    requested::AbstractVector{T},
    gain::T,
) where {T}
    if !iszero(gain) && all(eachindex(plan.max_differences)) do edge
        abs(requested[plan.first_indices[edge]] * gain -
            requested[plan.second_indices[edge]] * gain) <= plan.max_differences[edge]
    end
        @inbounds for index in eachindex(stroke_limited, requested)
            stroke_limited[index] = requested[index] * gain
        end
        return nothing
    end

    # Represent the gain with a normalized significand and a separate
    # exponent so multiplication retains its precision below floatmin(T).
    fraction, exponent = one(T), 0
    @inbounds for edge in eachindex(plan.max_differences)
        first = requested[plan.first_indices[edge]]
        second = requested[plan.second_indices[edge]]
        limit = plan.max_differences[edge]
        edge_gain = _inter_actuator_difference_gain(first, second, limit)
        if edge_gain < floatmin(T)
            if iszero(limit)
                @inbounds for index in eachindex(stroke_limited, requested)
                    stroke_limited[index] = requested[index] * zero(T)
                end
                return nothing
            end
            scale = max(abs(first), abs(second))
            # Preserve close same-sign differences by subtracting first.
            if signbit(first) == signbit(second)
                scale = abs(first - second)
                normalized = one(T)
            else
                normalized = one(T) + min(abs(first), abs(second)) / scale
            end
            lim, le = frexp(limit)
            denom, de = frexp(scale)
            ef, shift = frexp((lim / denom) / normalized)
            ee = le - de + shift
            if ee < exponent || (ee == exponent && ef < fraction)
                fraction, exponent = ef, ee
            end
        end
    end
    # Bias the exceptional result inward to allow for output rounding.
    fraction = prevfloat(fraction)
    @inbounds for index in eachindex(stroke_limited, requested)
        value, shift = frexp(requested[index])
        stroke_limited[index] = ldexp(value * fraction, shift + exponent)
    end
    return nothing
end

function _process_inter_actuator_stroke_limit!(
    stroke_limited::AbstractVector{T},
    plan::InterActuatorStrokeLimitPlan{T},
    requested::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(stroke_limited, requested)
    length(requested) == plan.actuator_count || throw(
        DimensionMismatch("inter-actuator stroke input extent is invalid"),
    )
    length(stroke_limited) == plan.actuator_count || throw(
        DimensionMismatch("inter-actuator stroke output extent is invalid"),
    )
    stroke_limited === requested || !Base.mightalias(stroke_limited, requested) ||
        throw(
            ArgumentError(
                "partially overlapping requested and stroke-limited " *
                "commands are unsupported",
            ),
        )
    for value in requested
        isfinite(value) || return InterActuatorStrokeLimitNonFiniteInput
    end

    return _project_inter_actuator_stroke!(stroke_limited, plan, requested)
end

function _project_inter_actuator_stroke!(
    stroke_limited::AbstractVector{T},
    plan::InterActuatorStrokeLimitPlan{T},
    requested::AbstractVector{T},
) where {T}
    gain = one(T)
    @inbounds for edge in eachindex(plan.max_differences)
        edge_gain = _inter_actuator_difference_gain(
            requested[plan.first_indices[edge]],
            requested[plan.second_indices[edge]],
            plan.max_differences[edge],
        )
        gain = min(gain, edge_gain)
    end
    gain < floatmin(T) && return _project_small_stroke_gain!(stroke_limited, plan, requested, gain)
    @inbounds for index in eachindex(stroke_limited, requested)
        stroke_limited[index] = requested[index] * gain
    end
    return nothing
end

@inline function _inter_actuator_product(value::Float64, gain::Float64, mode::Val)
    product = value * gain
    rounded = _dm_narrow(product, mode)
    # Normal Float32 midpoints have a single bit set among the 29 fraction
    # bits Float64 discards. Subnormal midpoints can have fewer than 25
    # significant bits, so also check those products on the slow path.
    low_bits = reinterpret(UInt64, product) & 0x0000_0000_1fff_ffff
    if low_bits != 0x0000_0000_1000_0000 &&
       (!iszero(low_bits) || iszero(product) || abs(product) >= 0x1p-126)
        return rounded
    end
    return _inter_actuator_round_midpoint(value, gain, product, rounded, mode)
end

@noinline function _inter_actuator_round_midpoint(
    value::Float64,
    gain::Float64,
    product::Float64,
    rounded::Float32,
    mode::Val,
)
    rounded_wide = _dm_widen(rounded, mode)
    product == rounded_wide && return rounded

    # Correct double rounding only when the wide product is on a Float32 midpoint.
    neighbor = product > rounded_wide ? nextfloat(rounded) : prevfloat(rounded)
    if product == (rounded_wide + _dm_widen(neighbor, mode)) * 0.5
        residual = fma(value, gain, -product)
        if product > rounded_wide ? residual > 0.0 : residual < 0.0
            return neighbor
        end
    end
    return rounded
end

@inline function _inter_actuator_difference_exceeds(
    first::Float64,
    second::Float64,
    limit::Float64,
)
    difference = abs(first - second)
    difference == limit || return difference > limit

    first_magnitude = abs(first)
    second_magnitude = abs(second)
    larger = max(first_magnitude, second_magnitude)
    smaller = min(first_magnitude, second_magnitude)
    term = signbit(first) == signbit(second) ? -smaller : smaller
    # FastTwoSum supplies the exact residual when the rounded difference equals the limit.
    return term > difference - larger
end

function _inter_actuator_rounding_gain(requested::Float64, rounded::Float32, mode::Val)
    magnitude = _dm_widen(abs(rounded), mode)
    iszero(magnitude) && return 0.0
    midpoint = (magnitude + _dm_widen(prevfloat(abs(rounded)), mode)) * 0.5
    gain = midpoint / abs(requested)
    # The fused residual places the exact product on the correct side of
    # the midpoint. At equality, ties-to-even selects the lower magnitude
    # exactly when the upper Float32 has an odd significand.
    residual = fma(abs(requested), gain, -midpoint)
    rounds_lower = residual < 0.0 || (iszero(residual) && isodd(reinterpret(UInt32, rounded)))
    return rounds_lower ? gain : prevfloat(gain)
end

function _inter_actuator_opposite_sign_gain(
    first::Float64,
    second::Float64,
    first_output::Float32,
    second_output::Float32,
    limit::Float64,
    mode::Val,
)
    gain = max(
        _inter_actuator_rounding_gain(first, first_output, mode),
        _inter_actuator_rounding_gain(second, second_output, mode),
    )
    if !_inter_actuator_difference_exceeds(
        _dm_widen(_inter_actuator_product(first, gain, mode), mode),
        _dm_widen(_inter_actuator_product(second, gain, mode), mode),
        limit,
    )
        return gain
    end

    # Opposite-sign differences are monotone in the common gain. Search the
    # ordered nonnegative Float64 encodings for the largest feasible gain.
    lower = UInt64(0)
    upper = reinterpret(UInt64, gain)
    while upper - lower > 1
        middle = lower + ((upper - lower) >> 1)
        candidate = reinterpret(Float64, middle)
        if _inter_actuator_difference_exceeds(
            _dm_widen(_inter_actuator_product(first, candidate, mode), mode),
            _dm_widen(_inter_actuator_product(second, candidate, mode), mode),
            limit,
        )
            upper = middle
        else
            lower = middle
        end
    end
    return reinterpret(Float64, lower)
end

@inline function _stroke_needs_rounded_check(first, second, limit, gain)
    # Twice the maximum endpoint rounding error covers rounding of the bound.
    margin = (abs(first) + abs(second)) * eps(Float32)
    return gain * (abs(first - second) + margin) + 0x1p-148 > limit
end

const _STROKE_BLOCK_SIZE = 32

@inline function _stroke_block_may_bind(plan, requested, start, gain, mode, projected)
    # Test a full group with one fixed gain so the comparisons can vectorize.
    # Every later gain is smaller or equal. Excluded edges stay admissible.
    may_bind = false
    @inbounds @simd for offset in 0:(_STROKE_BLOCK_SIZE - 1)
        edge = start + offset
        first = _dm_widen(requested[plan.first_indices[edge]], mode)
        second = _dm_widen(requested[plan.second_indices[edge]], mode)
        limit = _dm_widen(plan.max_differences[edge], mode)
        may_bind |= if projected
            _stroke_needs_rounded_check(first, second, limit, gain)
        else
            abs(first - second) * gain >= limit
        end
    end
    return may_bind
end

function _refine_inter_actuator_stroke_gain(
    plan,
    requested::AbstractVector{Float32},
    gain::Float64,
    passes::Int,
    mode::Val,
)
    # Each correction excludes only gains that violate an edge. A later
    # correction can reintroduce a violation on a same-sign edge.
    for _ in 1:passes
        previous_gain = gain
        @inbounds for start in 1:_STROKE_BLOCK_SIZE:length(plan.max_differences)
            count = min(_STROKE_BLOCK_SIZE, length(plan.max_differences) - start + 1)
            if count == _STROKE_BLOCK_SIZE && !_stroke_block_may_bind(plan, requested, start, gain, mode, true)
                continue
            end
            for edge in start:(start + count - 1)
                first = _dm_widen(requested[plan.first_indices[edge]], mode)
                second = _dm_widen(requested[plan.second_indices[edge]], mode)
                limit = _dm_widen(plan.max_differences[edge], mode)
                _stroke_needs_rounded_check(first, second, limit, gain) || continue
                first_output = _inter_actuator_product(first, gain, mode)
                second_output = _inter_actuator_product(second, gain, mode)
                _inter_actuator_difference_exceeds(
                    _dm_widen(first_output, mode), _dm_widen(second_output, mode), limit,
                ) ||
                    continue
                if signbit(first) == signbit(second)
                    # Reducing the smaller magnitude cannot help until the
                    # larger output drops. Jump to that first rounding event.
                    gain = if abs(first) > abs(second)
                        _inter_actuator_rounding_gain(first, first_output, mode)
                    else
                        _inter_actuator_rounding_gain(second, second_output, mode)
                    end
                else
                    gain = _inter_actuator_opposite_sign_gain(
                        first, second, first_output, second_output, limit, mode,
                    )
                end
            end
        end
        gain == previous_gain && return gain, true
    end
    return gain, false
end

function _inter_actuator_stroke_gain(
    plan,
    requested::AbstractVector{Float32},
    mode::Val,
)
    # Float64 covers the full ratio of finite Float32 commands and limits,
    # including gains smaller than the least Float32 subnormal.
    gain = 1.0
    needs_correction = false
    @inbounds for start in 1:_STROKE_BLOCK_SIZE:length(plan.max_differences)
        count = min(_STROKE_BLOCK_SIZE, length(plan.max_differences) - start + 1)
        if count == _STROKE_BLOCK_SIZE && !_stroke_block_may_bind(plan, requested, start, gain, mode, false)
            continue
        end
        for edge in start:(start + count - 1)
            first = _dm_widen(requested[plan.first_indices[edge]], mode)
            second = _dm_widen(requested[plan.second_indices[edge]], mode)
            limit = _dm_widen(plan.max_differences[edge], mode)
            difference = abs(first - second)
            if difference * gain > limit
                iszero(limit) && return 0.0
                gain = limit / difference
            elseif gain == 1.0 && difference == limit
                needs_correction |= _inter_actuator_difference_exceeds(first, second, limit)
            end
        end
    end
    gain == 1.0 && !needs_correction && return gain

    gain, feasible = _refine_inter_actuator_stroke_gain(plan, requested, gain, 8, mode)
    feasible && return gain

    # Eight strictly decreasing Float64 steps more than cover the rounding
    # of the analytical ratio. Scale by the largest power of two at most this
    # gain so only subnormal outputs round. The difference then has an error
    # of at most one least Float32 subnormal. Because differences and limits
    # are multiples of that unit, a strictly feasible exact difference still
    # satisfies the limit after rounding.
    _, exponent = frexp(gain)
    safe_gain = ldexp(1.0, exponent - 1)

    # Bound both endpoints' rounding error by a relative half-ULP plus half
    # of the least subnormal each. The small Float64 margin covers the
    # arithmetic in this bound. Opposite-sign edges stay feasible as the
    # gain decreases.
    envelope_gain = gain
    @inbounds for edge in eachindex(plan.max_differences)
        first = _dm_widen(requested[plan.first_indices[edge]], mode)
        second = _dm_widen(requested[plan.second_indices[edge]], mode)
        (first == second || signbit(first) != signbit(second)) && continue
        available = _dm_widen(plan.max_differences[edge], mode) - 0x1p-149
        error = (abs(first) + abs(second)) * (0.5 * eps(Float32))
        envelope = (abs(first - second) + error) * (1.0 + 4eps(Float64))
        envelope_gain = min(envelope_gain, max(0.0, available) / envelope)
    end
    safe_gain = max(safe_gain, envelope_gain)

    # For nearly equal commands the rounding error can be comparable to
    # the edge's requested difference. Try interior gains before accepting
    # that conservative bound.
    if safe_gain < 0.99 * gain
        upper = gain
        for fraction in (1 / 64, 1 / 16, 1 / 4, 1 / 2, 3 / 4)
            candidate = min(upper, gain - fraction * (gain - safe_gain))
            upper, feasible = _refine_inter_actuator_stroke_gain(plan, requested, candidate, 4, mode)
            feasible && return max(safe_gain, upper)
        end
    end
    return safe_gain
end

function _project_inter_actuator_stroke!(
    stroke_limited::AbstractVector{Float32},
    plan::InterActuatorStrokeLimitPlan{Float32},
    requested::AbstractVector{Float32},
)
    if get_zero_subnormals()
        return _project_inter_actuator_stroke!(stroke_limited, plan, requested, Val(true))
    end
    return _project_inter_actuator_stroke!(stroke_limited, plan, requested, Val(false))
end

function _project_inter_actuator_stroke!(
    stroke_limited::AbstractVector{Float32},
    plan::InterActuatorStrokeLimitPlan{Float32},
    requested::AbstractVector{Float32},
    mode::Val,
)
    gain = _inter_actuator_stroke_gain(plan, requested, mode)
    if gain == 1.0
        stroke_limited === requested || copyto!(stroke_limited, requested)
        return nothing
    end
    @inbounds for index in eachindex(stroke_limited, requested)
        stroke_limited[index] = _inter_actuator_product(_dm_widen(requested[index], mode), gain, mode)
    end
    return nothing
end

"""
Float32 radial projection of a requested physical-DM command onto the
configured inter-actuator stroke-difference limits. The stored Float32
outputs satisfy every limit and use a single common gain. Commands already
within all limits pass through bit for bit. When rounding couples several
active edges, the bounded search can select a conservative gain. The output
retains the requested-command schema for subsequent conditioning.
"""
@algorithm InterActuatorStrokeLimitF32 begin
    label = "inter-actuator-stroke-limit-f32"
    failure = InterActuatorStrokeLimitFailure

    """Configuration for [`InterActuatorStrokeLimitF32`](@ref)."""
    @configuration InterActuatorStrokeLimitF32Configuration begin
        actuator_count::Int => graph_rebuild
        edges::Vector{Vector{Int}} => graph_rebuild
        max_differences::Vector{Float32} => construction
    end

    ports(configuration) = (
        input(
            :requested,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1,
        ),
        output(
            :stroke_limited,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            name="stroke-limited",
            metadata_from=:requested,
        ),
        parameter(
            :max_differences,
            schema_element_type(INTER_ACTUATOR_STROKE_LIMIT_V1),
            (length(configuration.edges),),
            INTER_ACTUATOR_STROKE_LIMIT_V1;
            name="max-differences",
            replace=(plan, values) ->
                _replace_inter_actuator_max_differences(plan, values),
        ),
    )

    prepare(configuration) = InterActuatorStrokeLimitPlan(
        configuration.actuator_count,
        configuration.edges,
        configuration.max_differences,
    )

    process!(plan) = begin
        _process_inter_actuator_stroke_limit!(stroke_limited, plan, requested)
    end
end
