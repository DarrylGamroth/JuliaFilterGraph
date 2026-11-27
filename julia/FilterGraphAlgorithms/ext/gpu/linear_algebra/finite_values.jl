# A shared workgroup finite scan. Each Algorithm supplies its own failure code
# and decides whether a proven input makes this scan redundant. VDM role
# publication shares this validated workgroup to avoid another kernel launch.
# KA CPU lowering carries direct @index bindings and @uniform group snapshots
# across barriers. Keep Int conversions at helper calls rather than around @index.
const VALIDATION_WORKGROUP_SIZE = 256

@inline function _reduce_validation!(
    local_failures,
    local_index::Int,
    stride::Int,
)
    if local_index <= stride
        @inbounds local_failures[local_index] = max(
            local_failures[local_index],
            local_failures[local_index + stride],
        )
    end
    return nothing
end

@kernel function _validate_projection_values_portable!(
    frame_status,
    values,
    failure,
    role_output=nothing,
    roles=nothing,
)
    local_index = @index(Local, Linear)
    local_failures = @localmem Int32 (
        VALIDATION_WORKGROUP_SIZE,
    )
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    local_failure = Int32(0)
    value_index = Int(local_index)
    while frame_valid && value_index <= length(values)
        @inbounds if !isfinite(values[value_index])
            local_failure = Int32(1)
            break
        end
        value_index += VALIDATION_WORKGROUP_SIZE
    end
    @inbounds local_failures[local_index] = local_failure
    @synchronize

    _reduce_validation!(local_failures, Int(local_index), 128)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 64)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 32)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 16)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 8)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 4)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 2)
    @synchronize
    _reduce_validation!(local_failures, Int(local_index), 1)
    @synchronize

    if frame_valid && local_index == 1 && @inbounds(!iszero(local_failures[1]))
        @inbounds frame_status[1] = failure
    end
    # VDM projection has a constant role output. Publish it only after validation,
    # in the same workgroup, without another launch or host-to-device transfer.
    if role_output !== nothing && frame_valid && @inbounds(iszero(local_failures[1]))
        role_index = Int(local_index)
        while role_index <= length(role_output)
            @inbounds role_output[role_index] = roles[role_index]
            role_index += VALIDATION_WORKGROUP_SIZE
        end
    end
end

# Backends may select a native collective without changing scan callers. The
# portable tree remains the default, including the CPU and AMD GPU paths.
_validate_projection_values!(backend, geometry...) =
    _validate_projection_values_portable!(backend, geometry...)

struct _FiniteInput end

# Recorded only after cold Graph preparation proves the exact input unchanged
# and finite while status is zero. A failed frame already skips the old check.
(::_FiniteInput)(status, values, failure; ndrange) = nothing
