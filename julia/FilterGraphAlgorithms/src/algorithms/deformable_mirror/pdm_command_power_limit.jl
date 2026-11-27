_pdm_power_bound(value::Float64) = value
_pdm_power_bound(value::Float32) =
    get_zero_subnormals() ? _dm_widen(value, Val(true)) : value

"""
    PdmCommandPowerLimitPlan(actuator_count, max_command_power)

Prepare projection of one physical-deformable-mirror command onto a total
command-power ball. Command power is the sum of squared actuator values. A
command outside the prepared bound receives one common gain, so its direction
in physical-actuator coordinates is preserved.

`actuator_count` must be positive. `max_command_power` must be finite and
nonnegative.
"""
struct PdmCommandPowerLimitPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    actuator_count::Int
    max_command_power::T
    max_norm::T

    function PdmCommandPowerLimitPlan(
        actuator_count::Int,
        max_command_power::T,
    ) where {T<:Union{Float32,Float64}}
        actuator_count > 0 || throw(
            ArgumentError("PDM command-power extent must be nonempty"),
        )
        power = _pdm_power_bound(max_command_power)
        isfinite(power) && power >= zero(power) || throw(
            ArgumentError(
                "maximum PDM command power must be finite and nonnegative",
            ),
        )
        return new{T}(
            actuator_count,
            max_command_power,
            T(sqrt(power)),
        )
    end
end

"""Expected numerical rejection from PDM command-power projection."""
@enum PdmCommandPowerLimitFailure::UInt8 begin
    PdmCommandPowerLimitNonFiniteInput = 1
end

@doc "The requested PDM command contains a non-finite actuator value."
PdmCommandPowerLimitNonFiniteInput

function _process_pdm_command_power_limit!(
    power_limited::AbstractVector{T},
    plan::PdmCommandPowerLimitPlan{T},
    requested::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(power_limited, requested)
    length(requested) == plan.actuator_count || throw(
        DimensionMismatch("PDM command-power input extent is invalid"),
    )
    length(power_limited) == plan.actuator_count || throw(
        DimensionMismatch("PDM command-power output extent is invalid"),
    )
    power_limited === requested || !Base.mightalias(power_limited, requested) ||
        throw(
            ArgumentError(
                "partially overlapping requested and power-limited " *
                "commands are unsupported",
            ),
        )

    return _project_pdm_command_power!(power_limited, plan, requested)
end

function _project_pdm_command_power!(
    power_limited::AbstractVector{Float64},
    plan::PdmCommandPowerLimitPlan{Float64},
    requested::AbstractVector{Float64},
)
    scale = 0.0
    sum_squares = 1.0
    for value in requested
        isfinite(value) || return PdmCommandPowerLimitNonFiniteInput
        magnitude = abs(value)
        iszero(magnitude) && continue
        if scale < magnitude
            ratio = scale / magnitude
            sum_squares = 1.0 + sum_squares * ratio * ratio
            scale = magnitude
        else
            ratio = magnitude / scale
            sum_squares += ratio * ratio
        end
    end

    gain = if iszero(scale)
        1.0
    elseif iszero(plan.max_norm)
        0.0
    else
        normalized_norm = sqrt(sum_squares)
        if scale <= plan.max_norm / normalized_norm
            1.0
        else
            limited_norm = plan.max_norm / normalized_norm
            gain = (plan.max_norm / scale) / normalized_norm
            if gain < floatmin(Float64) && (iszero(gain) || scale * gain > limited_norm)
                # Divide out the input scale before multiplying by the bound.
                # Keeping the subnormal common factor out of storage also
                # preserves commands whose projected samples are representable.
                target_norm = max(0.0, prevfloat(limited_norm))
                @inbounds for index in eachindex(power_limited, requested)
                    power_limited[index] = (requested[index] / scale) * target_norm
                end
                return nothing
            end
            gain
        end
    end

    @inbounds for index in eachindex(power_limited, requested)
        power_limited[index] = requested[index] * gain
    end
    return nothing
end

@inline _pdm_projected_value(value::Float32, gain::Float64, mode::Val) =
    _dm_narrow(_dm_widen(value, mode) * gain, mode)

@inline _pdm_projected_square(value::Float32, gain::Float64, mode::Val) =
    abs2(_dm_widen(_pdm_projected_value(value, gain, mode), mode))

@inline function _pdm_power_words(power::Float64)
    bits = reinterpret(UInt64, power)
    significand = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
    # Every squared Float32 value is an integer multiple of 2^-298.
    offset = Int(bits >> 52) - 1023 - 52 + 298
    if offset < 0
        significand >>= -offset
        offset = 0
    end
    word, shift = divrem(offset, 64)
    return word + 1, UInt128(significand) << shift
end

function _pdm_power_fits_words(requested, gain::Float64, power::Float64, first_index, mode::Val)
    # Seven words cover bounds below 2^128 in units of 2^-298.
    # A borrow past the highest word means the sum has exceeded the bound.
    first_word, bound_digits = _pdm_power_words(power)
    remaining = ntuple(Val(7)) do word
        word == first_word && return bound_digits % UInt64
        word == first_word + 1 && return UInt64(bound_digits >> 64)
        return UInt64(0)
    end
    for index in first_index:lastindex(requested)
        @inbounds square = _pdm_projected_square(requested[index], gain, mode)
        iszero(square) && continue
        word, digits = _pdm_power_words(square)
        while !iszero(digits)
            word <= length(remaining) || return false
            @inbounds available = remaining[word]
            subtrahend = digits % UInt64
            remaining = Base.setindex(remaining, available - subtrahend, word)
            digits = (digits >> 64) + (available < subtrahend)
            word += 1
        end
    end
    return true
end

function _pdm_exact_power_fits(requested, gain::Float64, power::Float64, mode::Val)
    remaining = power
    for index in firstindex(requested):lastindex(requested)
        @inbounds square = _pdm_projected_square(requested[index], gain, mode)
        square > remaining && return false
        updated = remaining - square
        # By Sterbenz's lemma at least one subtraction is exact. The reverse
        # subtraction recovers the square exactly if the first one was exact.
        if remaining - updated != square
            return _pdm_power_fits_words(requested, gain, remaining, index, mode)
        end
        remaining = updated
    end
    return true
end

function _pdm_power_fits(
    requested,
    gain::Float64,
    total::Float64,
    power::Float64,
    guard::Float64,
    mode::Val,
)
    # Products of Float32 values are exact in Float64. When the guard is below
    # 1/2, four unit roundoffs per term cover the sum and threshold rounding.
    if guard < 0.5
        total < power * (1.0 - guard) && return true
        total > power * (1.0 + guard) && return false
    end
    return _pdm_exact_power_fits(requested, gain, power, mode)
end

function _pdm_projected_power_fits(requested, gain::Float64, power::Float64, guard::Float64, mode::Val)
    total = 0.0
    @simd for index in eachindex(requested)
        @inbounds total += _pdm_projected_square(requested[index], gain, mode)
    end
    return _pdm_power_fits(requested, gain, total, power, guard, mode)
end

function _pdm_checked_gain(requested, gain::Float64, power::Float64, guard::Float64, mode::Val)
    _pdm_projected_power_fits(requested, gain, power, guard, mode) && return gain

    # A small reduction compensates for rounding the stored actuator values.
    gain *= 1 - eps(Float32) / 2
    _pdm_projected_power_fits(requested, gain, power, guard, mode) && return gain
    gain *= 1 - eps(Float32)
    _pdm_projected_power_fits(requested, gain, power, guard, mode) && return gain

    # Positive Float64 values and their bit patterns have the same order.
    # A zero gain always fits and each actuator magnitude is monotone in gain.
    low = UInt64(0)
    high = reinterpret(UInt64, gain)
    while high - low > 1
        middle = low + (high - low) ÷ 2
        if _pdm_projected_power_fits(requested, reinterpret(Float64, middle), power, guard, mode)
            low = middle
        else
            high = middle
        end
    end
    return reinterpret(Float64, low)
end

function _project_pdm_command_power!(
    power_limited::AbstractVector{Float32},
    plan::PdmCommandPowerLimitPlan{Float32},
    requested::AbstractVector{Float32},
)
    if get_zero_subnormals()
        return _project_pdm_command_power!(power_limited, plan, requested, Val(true))
    end
    return _project_pdm_command_power!(power_limited, plan, requested, Val(false))
end

function _project_pdm_command_power!(
    power_limited::AbstractVector{Float32},
    plan::PdmCommandPowerLimitPlan{Float32},
    requested::AbstractVector{Float32},
    mode::Val,
)
    gain = _pdm_command_power_gain(requested, plan.max_command_power, mode)
    isfinite(gain) || return PdmCommandPowerLimitNonFiniteInput
    if gain == 1.0
        power_limited === requested || copyto!(power_limited, requested)
        return nothing
    end
    @inbounds for index in eachindex(power_limited, requested)
        power_limited[index] = _pdm_projected_value(requested[index], gain, mode)
    end
    return nothing
end

function _pdm_command_power_gain(requested, max_command_power::Float32, mode::Val)
    total = 0.0
    @simd for index in eachindex(requested)
        @inbounds value = requested[index]
        total += abs2(_dm_widen(value, mode))
    end
    # Even length(requested) copies of floatmax(Float32)^2 fit in Float64.
    isfinite(total) || return NaN

    power = _dm_widen(max_command_power, mode)
    guard = 2eps(Float64) * length(requested)
    if iszero(total) || (!iszero(power) && _pdm_power_fits(requested, 1.0, total, power, guard, mode))
        return 1.0
    end
    return iszero(power) ? 0.0 :
        _pdm_checked_gain(requested, min(1.0, sqrt(power / total)), power, guard, mode)
end

"""
Float32 projection of a requested physical-DM command onto a configured total
command-power ball. The sum of squared stored actuator values is at most the
configured bound. The output remains a requested command for subsequent
conditioning.
"""
@algorithm PdmCommandPowerLimitF32 begin
    label = "pdm-command-power-limit-f32"
    failure = PdmCommandPowerLimitFailure

    """Configuration for [`PdmCommandPowerLimitF32`](@ref)."""
    @configuration PdmCommandPowerLimitF32Configuration begin
        actuator_count::Int => graph_rebuild
        max_command_power::Float32 => construction
    end

    ports(configuration) = (
        input(
            :requested,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1,
        ),
        output(
            :power_limited,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            name="power-limited",
            metadata_from=:requested,
        ),
    )

    prepare(configuration) = PdmCommandPowerLimitPlan(
        configuration.actuator_count,
        configuration.max_command_power,
    )

    process!(plan) = begin
        _process_pdm_command_power_limit!(power_limited, plan, requested)
    end
end
